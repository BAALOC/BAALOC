#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "python-dateutil", "pyyaml"]
# ///
"""Regenerate the all-time GitHub stats blocks in README.md.

Commits are the union of two sources:

* every commit that exists today on any ref of any repository the token can
  read, deduplicated by SHA (GitHub itself decides which author emails belong
  to the account);
* GitHub's contribution record, which keeps per-day commit counts even after
  the commits were rewritten away or the repository was deleted. For each
  repository and day, whatever the record holds beyond the commits that still
  exist is added on top.

Lines and languages need the content of a commit, so they come from existing
commits only. Languages are weighted by the lines the user added, using
GitHub Linguist's extension table and, like GitHub's language bar, only its
programming and markup languages.
"""

import logging
import re
from base64 import b64encode
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from os import environ
from pathlib import Path, PurePosixPath
from subprocess import run
from sys import exit as sys_exit
from tempfile import TemporaryDirectory
from typing import Any

from dateutil.relativedelta import relativedelta
from requests import HTTPError, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from yaml import safe_load

API_URL = "https://api.github.com"
LINGUIST_PATH = "/repos/github-linguist/linguist/contents/lib/linguist/languages.yml"
COUNTED_LANGUAGE_TYPES = {"programming", "markup"}
# Third-party code checked into a repository is not the user's writing.
VENDORED_DIRS = {"node_modules", "vendor", "third_party", "dist", "build"}
START_MARKER = "<!-- STATS:START -->"
END_MARKER = "<!-- STATS:END -->"
LABEL_WIDTH = 16
MAX_LANGUAGES_SHOWN = 4
BUSIEST_DAYS_SHOWN = 2
PEAK_WINDOW_HOURS = 3

USER_QUERY = """
query($login: String!) {
  user(login: $login) {
    createdAt
    pullRequests { totalCount }
    contributionsCollection { contributionYears }
  }
}
"""

# Queried one month at a time so that no repository has more than 100 days of
# commit contributions in a window, which keeps the nested list unpaginated.
CONTRIBUTIONS_QUERY = """
query($login: String!, $from: DateTime!, $to: DateTime!) {
  user(login: $login) {
    contributionsCollection(from: $from, to: $to) {
      restrictedContributionsCount
      commitContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner }
        contributions(first: 100) { nodes { occurredAt commitCount } }
      }
      pullRequestContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner }
        contributions { totalCount }
      }
      issueContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner }
      }
      pullRequestReviewContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner }
      }
    }
  }
}
"""

# numstat prints renames as "dir/{old => new}/file" or "old => new".
RENAME_BRACES = re.compile(r"\{[^{}]* => ([^{}]*)\}")

logger = logging.getLogger(__name__)


class GitHub:
    def __init__(self, token: str) -> None:
        self.token = token
        self.session = Session()
        # Dropped connections and 5xx gateway errors from GitHub are transient.
        retry = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=(502, 503, 504),
            allowed_methods=None,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            f"{API_URL}/graphql",
            json={"query": query, "variables": variables},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if "errors" in payload:
            raise RuntimeError(f"GraphQL error: {payload['errors']}")
        return payload["data"]

    def get(self, path: str) -> Any:
        response = self.session.get(f"{API_URL}{path}", timeout=30)
        response.raise_for_status()
        return response.json()

    def get_raw(self, path: str) -> str:
        response = self.session.get(
            f"{API_URL}{path}",
            headers={"Accept": "application/vnd.github.raw+json"},
            timeout=30,
        )
        response.raise_for_status()
        return response.text

    def get_all(self, path: str, params: dict[str, str] | None = None) -> list[Any]:
        items: list[Any] = []
        url: str | None = f"{API_URL}{path}"
        query: dict[str, str] | None = {"per_page": "100", **(params or {})}
        while url:
            response = self.session.get(url, params=query, timeout=30)
            response.raise_for_status()
            items.extend(response.json())
            url = response.links.get("next", {}).get("url")
            query = None  # the "next" link already carries the query string
        return items


@dataclass
class ContributionRecord:
    # (repo, author's local date) -> commits GitHub recorded that day
    commit_days: Counter[tuple[str, date]] = field(default_factory=Counter)
    pull_requests: Counter[str] = field(default_factory=Counter)
    repos: set[str] = field(default_factory=set)
    restricted: int = 0


@dataclass
class Commit:
    authored: datetime  # in the author's own timezone offset
    changes: list[tuple[int, int, str]] = field(default_factory=list)  # +, -, path


@dataclass
class Linguist:
    by_filename: dict[str, list[str]]
    by_extension: dict[str, list[str]]
    types: dict[str, str]

    def language(self, path: str, repo_language: str | None) -> str | None:
        """Counted Linguist language of a path, or None.

        Ambiguity goes to the repo's language, then to the language whose
        primary extension it is. Types are filtered only afterwards, so that
        e.g. ".md" resolves to Markdown (prose) instead of a rarer match.
        """
        name = PurePosixPath(path).name
        candidates = self.by_filename.get(name) or self.by_extension.get(
            PurePosixPath(name).suffix.lower(), []
        )
        if not candidates:
            return None
        language = repo_language if repo_language in candidates else candidates[0]
        return language if self.types[language] in COUNTED_LANGUAGE_TYPES else None


def load_linguist(github: GitHub) -> Linguist:
    linguist = Linguist(by_filename={}, by_extension={}, types={})
    for name, language in safe_load(github.get_raw(LINGUIST_PATH)).items():
        linguist.types[name] = language["type"]
        for filename in language.get("filenames", []):
            linguist.by_filename.setdefault(filename, []).append(name)
        for position, extension in enumerate(language.get("extensions", [])):
            candidates = linguist.by_extension.setdefault(extension.lower(), [])
            # A language's first extension is its primary one: rank it first.
            if position == 0:
                candidates.insert(0, name)
            else:
                candidates.append(name)
    return linguist


def is_owned(repo: str, login: str) -> bool:
    return repo.split("/")[0].lower() == login.lower()


def month_windows(
    years: list[int], now: datetime
) -> Iterator[tuple[datetime, datetime]]:
    for year in sorted(years):
        for month in range(1, 13):
            start = datetime(year, month, 1, tzinfo=UTC)
            if start > now:
                return
            yield start, start + relativedelta(months=1) - timedelta(seconds=1)


def collect_record(
    github: GitHub, login: str, years: list[int], now: datetime
) -> ContributionRecord:
    record = ContributionRecord()
    for start, end in month_windows(years, now):
        variables = {"login": login, "from": start.isoformat(), "to": end.isoformat()}
        collection = github.graphql(CONTRIBUTIONS_QUERY, variables)["user"][
            "contributionsCollection"
        ]
        record.restricted += collection["restrictedContributionsCount"]

        for entry in collection["commitContributionsByRepository"]:
            repo = entry["repository"]["nameWithOwner"]
            for day in entry["contributions"]["nodes"]:
                # occurredAt is local midnight of the viewer's timezone (UTC-12..+12),
                # so the nearest UTC midnight carries the contribution date.
                occurred = datetime.fromisoformat(day["occurredAt"]) + timedelta(
                    hours=12
                )
                record.commit_days[repo, occurred.date()] += day["commitCount"]

        for entry in collection["pullRequestContributionsByRepository"]:
            repo = entry["repository"]["nameWithOwner"]
            record.pull_requests[repo] += entry["contributions"]["totalCount"]

        for key in (
            "commitContributionsByRepository",
            "pullRequestContributionsByRepository",
            "issueContributionsByRepository",
            "pullRequestReviewContributionsByRepository",
        ):
            record.repos.update(
                entry["repository"]["nameWithOwner"] for entry in collection[key]
            )
    return record


def repo_info(github: GitHub, repo: str) -> dict[str, Any] | None:
    """Repository metadata, or None if it was deleted or can't be read."""
    try:
        return github.get(f"/repos/{repo}")
    except HTTPError as error:
        if error.response is not None and error.response.status_code == 404:
            return None
        raise


def existing_commits(
    github: GitHub, repo: str, login: str, workdir: Path, attribution: dict[str, bool]
) -> dict[str, Commit]:
    """The user's commits on every ref, keyed by SHA.

    Dates come from the git objects because REST and GraphQL report them in
    UTC, while GitHub buckets contributions by the author's own offset.
    `attribution` caches whether GitHub links an author email to `login`.
    """
    clone = workdir / repo.replace("/", "__")
    credentials = b64encode(f"x-access-token:{github.token}".encode()).decode()
    git_env = {
        **environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.extraHeader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {credentials}",
    }
    run(
        ["git", "clone", "--quiet", "--bare",
         f"https://github.com/{repo}.git", str(clone)],
        check=True, capture_output=True, env=git_env,
    )  # fmt: skip
    # Merge commits print no numstat, so merged work isn't counted twice.
    log = run(
        ["git", "--git-dir", str(clone), "log", "--all", "--numstat",
         "--format=@%H %aI %ae"],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip

    commits: dict[str, Commit] = {}
    current: Commit | None = None
    for line in log.splitlines():
        if line.startswith("@"):
            sha, authored, email = line[1:].split(" ", 2)
            email = email.lower()
            if email not in attribution:
                author = github.get(f"/repos/{repo}/commits/{sha}")["author"]
                attribution[email] = (
                    bool(author) and author["login"].lower() == login.lower()
                )
            current = (
                Commit(datetime.fromisoformat(authored)) if attribution[email] else None
            )
            if current is not None:
                commits[sha] = current
        elif line and current is not None:
            added, deleted, path = line.split("\t", 2)
            if added == "-":  # binary file
                continue
            path = RENAME_BRACES.sub(r"\1", path).split(" => ")[-1]
            current.changes.append((int(added), int(deleted), path))
    return commits


def commits_per_day(
    record: ContributionRecord, existing: dict[str, dict[str, Commit]]
) -> Counter[date]:
    """Every commit that ever existed, bucketed by the author's local date.

    Existing commits are counted once per SHA; on top of that comes whatever
    GitHub recorded for a repository and day beyond what still exists there.
    """
    unique = {sha: c for commits in existing.values() for sha, c in commits.items()}
    days: Counter[date] = Counter(c.authored.date() for c in unique.values())
    present: Counter[tuple[str, date]] = Counter(
        (repo, c.authored.date())
        for repo, commits in existing.items()
        for c in commits.values()
    )
    for (repo, day), recorded in record.commit_days.items():
        days[day] += max(0, recorded - present[repo, day])
    return days


def count_lines(
    existing: dict[str, dict[str, Commit]],
    infos: dict[str, dict[str, Any]],
    linguist: Linguist,
) -> tuple[Counter[str], int]:
    """Lines added per language and lines deleted, once per SHA."""
    added: Counter[str] = Counter()
    deleted = 0
    seen: set[str] = set()
    for repo, commits in existing.items():
        for sha, commit in commits.items():
            if sha in seen:
                continue
            seen.add(sha)
            for plus, minus, path in commit.changes:
                if VENDORED_DIRS.intersection(PurePosixPath(path).parts):
                    continue
                language = linguist.language(path, infos[repo]["language"])
                if language is None:
                    continue
                added[language] += plus
                deleted += minus
    return added, deleted


def format_count(count: int) -> str:
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def format_top_languages(added: Counter[str]) -> str:
    total = sum(added.values())
    if total == 0:
        return "N/A"
    shares = [
        (name, round(lines / total * 100))
        for name, lines in added.most_common(MAX_LANGUAGES_SHOWN)
    ]
    return " · ".join(f"{name} {share}%" for name, share in shares if share > 0)


def format_peak_hours(moments: list[datetime]) -> str:
    """The busiest PEAK_WINDOW_HOURS-long window of the day, in local time."""
    hours = Counter(moment.hour for moment in moments)
    start = max(
        range(24),
        key=lambda hour: sum(hours[(hour + i) % 24] for i in range(PEAK_WINDOW_HOURS)),
    )
    return f"{start:02}:00–{(start + PEAK_WINDOW_HOURS) % 24:02}:00"


def humanize_since(joined: datetime, now: datetime) -> str:
    delta = relativedelta(now, joined)

    def unit(count: int, word: str) -> str:
        return f"{count} {word}{'' if count == 1 else 's'}"

    parts = []
    if delta.years:
        parts.append(unit(delta.years, "year"))
    if delta.months:
        parts.append(unit(delta.months, "month"))
    if delta.days or not parts:
        parts.append(unit(delta.days, "day"))
    return ", ".join(parts) + " ago"


def terminal_block(command: str, rows: list[tuple[str, str]]) -> str:
    body = "\n".join(f"{label:<{LABEL_WIDTH}}{value}" for label, value in rows)
    return f"```\n$ {command}\n{body}\n```"


def build_stats(github: GitHub, login: str) -> str:
    now = datetime.now(UTC)
    user = github.graphql(USER_QUERY, {"login": login})["user"]
    record = collect_record(
        github, login, user["contributionsCollection"]["contributionYears"], now
    )
    if record.restricted:
        logger.warning(
            "%d contributions are hidden from this token (it needs the repo and "
            "read:user scopes); the stats are incomplete.",
            record.restricted,
        )

    accessible = github.get_all(
        "/user/repos", {"affiliation": "owner,collaborator,organization_member"}
    )
    owned = [repo for repo in accessible if is_owned(repo["full_name"], login)]

    infos: dict[str, dict[str, Any]] = {}
    existing: dict[str, dict[str, Commit]] = {}
    attribution: dict[str, bool] = {}
    with TemporaryDirectory() as workdir:
        for repo in sorted({r["full_name"] for r in accessible} | record.repos):
            info = repo_info(github, repo)
            if info is None:
                continue
            infos[repo] = info
            existing[repo] = existing_commits(
                github, repo, login, Path(workdir), attribution
            )

    days = commits_per_day(record, existing)
    added, deleted = count_lines(existing, infos, load_linguist(github))

    # The pullRequests connection only sees repositories that still exist.
    total_prs = user["pullRequests"]["totalCount"] + sum(
        count for repo, count in record.pull_requests.items() if repo not in infos
    )

    committed_to = {repo for repo, commits in existing.items() if commits}
    committed_to |= {repo for repo, _ in record.commit_days}
    contributed = {
        repo for repo in record.repos | committed_to if not is_owned(repo, login)
    }

    weekdays: Counter[str] = Counter()
    for day, count in days.items():
        weekdays[day.strftime("%A")] += count
    moments = list(
        {
            sha: c.authored
            for commits in existing.values()
            for sha, c in commits.items()
        }.values()
    )

    summary = terminal_block(
        "stats --summary",
        [
            ("Joined", humanize_since(datetime.fromisoformat(user["createdAt"]), now)),
            ("Repos", f"{len(owned)} ({sum(r['private'] for r in owned)} private)"),
            ("Commits", str(sum(days.values()))),
            (
                "Lines",
                f"+{format_count(sum(added.values()))} / −{format_count(deleted)}",
            ),
            ("PRs opened", str(total_prs)),
            ("Contributed to", f"{len(contributed)} repos"),
            ("Top languages", format_top_languages(added)),
        ],
    )
    activity = terminal_block(
        "stats --activity",
        [
            ("Active days", str(sum(1 for count in days.values() if count))),
            (
                "Busiest days",
                " · ".join(
                    name for name, _ in weekdays.most_common(BUSIEST_DAYS_SHOWN)
                ),
            ),
            ("Peak hours", format_peak_hours(moments)),
        ],
    )
    return f"{summary}\n\n{activity}"


def splice_readme(readme: str, block: str) -> str:
    start = readme.find(START_MARKER)
    end = readme.find(END_MARKER)
    if start == -1 or end == -1:
        raise RuntimeError(f"{START_MARKER} / {END_MARKER} markers not found in README")
    return f"{readme[: start + len(START_MARKER)]}\n{block}\n{readme[end:]}"


def main() -> None:
    logging.basicConfig(format="%(levelname)s: %(message)s")
    try:
        token = environ["GH_PAT"]
        login = environ["GITHUB_USERNAME"]
    except KeyError as missing:
        sys_exit(f"Missing required environment variable: {missing}")

    block = build_stats(GitHub(token), login)

    if environ.get("DRY_RUN", "").lower() in ("1", "true", "yes"):
        print(block)
        return

    readme_path = Path(environ.get("README_PATH", "README.md"))
    readme_path.write_text(
        splice_readme(readme_path.read_text(encoding="utf-8"), block), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

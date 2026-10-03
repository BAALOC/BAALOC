#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "python-dateutil"]
# ///

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dateutil.relativedelta import relativedelta

API_ROOT = "https://api.github.com"
GRAPHQL_URL = f"{API_ROOT}/graphql"

START_MARKER = "<!-- STATS:START -->"
END_MARKER = "<!-- STATS:END -->"

LABEL_WIDTH = 16
MAX_LANGUAGES_SHOWN = 4
YEAR_WINDOW = timedelta(days=365)

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def _rest_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _request(method, url: str, **kwargs) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = method(url, timeout=30, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise
        if resp.status_code >= 500 and attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue
        return resp
    raise last_exc


def _raise_with_hint(resp: requests.Response) -> None:
    if resp.status_code == 401:
        raise RuntimeError(
            "GitHub API returned 401 Unauthorized. Check that GH_PAT is set "
            "and hasn't expired (Settings -> Developer settings -> "
            "Personal access tokens)."
        )
    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        reset = resp.headers.get("X-RateLimit-Reset")
        raise RuntimeError(f"GitHub API rate limit hit. Resets at epoch {reset}.")
    resp.raise_for_status()


def rest_get(path: str, token: str) -> dict:
    resp = _request(requests.get, f"{API_ROOT}{path}", headers=_rest_headers(token))
    _raise_with_hint(resp)
    return resp.json()


def rest_get_all(path: str, token: str, params: Optional[dict] = None) -> list[dict]:
    items: list[dict] = []
    url = f"{API_ROOT}{path}"
    query = dict(params or {})
    query.setdefault("per_page", 100)

    while url:
        resp = _request(requests.get, url, headers=_rest_headers(token), params=query)
        _raise_with_hint(resp)
        items.extend(resp.json())
        url = resp.links.get("next", {}).get("url")
        query = None
    return items


def get_account_emails(token: str) -> set[str]:
    try:
        emails = rest_get_all("/user/emails", token)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise RuntimeError(
                "GET /user/emails returned 404 -- GH_PAT is missing the "
                "user:email scope (repo + read:user alone aren't enough for "
                "this endpoint). Add it at https://github.com/settings/tokens."
            ) from exc
        raise

    result = {e["email"].strip().lower() for e in emails if e.get("email")}
    extra = os.environ.get("EXTRA_AUTHOR_EMAILS", "")
    result |= {e.strip().lower() for e in extra.split(",") if e.strip()}
    return result


def graphql(query: str, variables: dict, token: str) -> dict:
    resp = _request(
        requests.post,
        GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    _raise_with_hint(resp)
    payload = resp.json()
    if "errors" in payload:
        raise RuntimeError(f"GraphQL error: {payload['errors']}")
    return payload["data"]


def get_join_date(username: str, token: str) -> datetime:
    data = rest_get(f"/users/{username}", token)
    return datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))


def get_owned_repos(username: str, token: str) -> list[dict]:
    return rest_get_all(
        "/user/repos", token, params={"affiliation": "owner", "visibility": "all"}
    )


def get_scannable_repos(token: str, extra_repo_names: Optional[set[str]] = None) -> list[dict]:
    repos = rest_get_all(
        "/user/repos",
        token,
        params={"affiliation": "owner,collaborator,organization_member", "visibility": "all"},
    )
    repos = [r for r in repos if not r.get("fork")]

    known_names = {r["full_name"] for r in repos}
    for name in sorted((extra_repo_names or set()) - known_names):
        try:
            extra = rest_get(f"/repos/{name}", token)
        except requests.exceptions.HTTPError as exc:
            print(f"  [warn] couldn't fetch metadata for {name}, skipping ({exc})", file=sys.stderr)
            continue
        if not extra.get("fork"):
            repos.append(extra)

    return repos


def to_github_datetime(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def year_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    windows = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + YEAR_WINDOW, end)
        windows.append((cursor, chunk_end))
        cursor = chunk_end
    return windows


def count_commits_all_branches(
        repo_full_name: str, token: str, emails: set[str], workdir: str
) -> tuple[int, Counter]:
    clone_path = os.path.join(workdir, repo_full_name.replace("/", "__"))
    clone = subprocess.run(
        [
            "git",
            "-c", "credential.helper=",
            "-c", "credential.helper=!f() { echo username=x-access-token; "
                  "echo password=$GIT_CLONE_TOKEN; }; f",
            "clone", "--quiet", "--bare",
            f"https://github.com/{repo_full_name}.git", clone_path,
        ],
        capture_output=True, text=True,
        env={**os.environ, "GIT_CLONE_TOKEN": token},
    )
    if clone.returncode != 0:
        print(
            f"  [warn] clone failed for {repo_full_name}, skipping "
            f"({clone.stderr.strip()[-200:]})",
            file=sys.stderr,
        )
        return 0, Counter()

    try:
        log = subprocess.run(
            ["git", "--git-dir", clone_path, "log", "--all", "--pretty=format:%ae"],
            capture_output=True, text=True,
        )
    finally:
        shutil.rmtree(clone_path, ignore_errors=True)

    if log.returncode != 0:
        print(f"  [warn] git log failed for {repo_full_name}, skipping", file=sys.stderr)
        return 0, Counter()

    author_emails = [
        line.strip().lower() for line in log.stdout.splitlines() if line.strip()
    ]
    matched = sum(1 for e in author_emails if e in emails)
    unmatched = Counter(e for e in author_emails if e not in emails)
    return matched, unmatched


def scan_all_branches(
        username: str, token: str, extra_repo_names: Optional[set[str]] = None
) -> dict:
    emails = get_account_emails(token)
    repos = get_scannable_repos(token, extra_repo_names)

    total_commits = 0
    repo_languages: dict[str, Optional[str]] = {}
    external_repos: set[str] = set()
    unmatched_totals: Counter = Counter()

    with tempfile.TemporaryDirectory(prefix="commit-scan-") as workdir:
        print(
            f"Commits per repo, all branches "
            f"({len(repos)} repos, {len(emails)} known emails):",
            file=sys.stderr,
        )
        for repo in repos:
            n, unmatched = count_commits_all_branches(repo["full_name"], token, emails, workdir)
            total_commits += n
            unmatched_totals.update(unmatched)
            print(f"  {repo['full_name']}: {n}", file=sys.stderr)

            if n > 0:
                repo_languages[repo["full_name"]] = repo.get("language")
                owner_login = (repo.get("owner") or {}).get("login", "")
                if owner_login.lower() != username.lower():
                    external_repos.add(repo["full_name"])

    print(f"  total across all repos: {total_commits} commits", file=sys.stderr)

    if unmatched_totals:
        print(
            "Commits found on some branch but NOT counted above (author email "
            "isn't in your registered/extra list):",
            file=sys.stderr,
        )
        for email, count in unmatched_totals.most_common(10):
            print(f"  {email}: {count}", file=sys.stderr)
        print(
            "If any of these are really you, either add the email at "
            "https://github.com/settings/emails (also fixes your public "
            "contribution graph) or list it in EXTRA_AUTHOR_EMAILS.",
            file=sys.stderr,
        )

    return {
        "total_commits": total_commits,
        "repo_languages": repo_languages,
        "external_repos": external_repos,
        "scanned_repo_names": {r["full_name"] for r in repos},
    }


CONTRIBUTIONS_QUERY = """
query($login: String!, $from: DateTime!, $to: DateTime!) {
  user(login: $login) {
    contributionsCollection(from: $from, to: $to) {
      totalCommitContributions
      restrictedContributionsCount
      commitContributionsByRepository(maxRepositories: 100) {
        repository {
          nameWithOwner
          owner { login }
          primaryLanguage { name }
        }
      }
      issueContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner owner { login } }
      }
      pullRequestContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner owner { login } }
      }
      pullRequestReviewContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner owner { login } }
      }
    }
  }
}
"""


def get_alltime_contribution_data(username: str, token: str, created_at: datetime, now: datetime) -> dict:
    total_commits = 0
    restricted_total = 0
    committed_repo_languages: dict[str, Optional[str]] = {}
    external_repos: set[str] = set()

    for start, end in year_windows(created_at, now):
        data = graphql(
            CONTRIBUTIONS_QUERY,
            {"login": username, "from": to_github_datetime(start), "to": to_github_datetime(end)},
            token,
        )
        coll = data["user"]["contributionsCollection"]
        total_commits += coll["totalCommitContributions"] + coll["restrictedContributionsCount"]
        restricted_total += coll["restrictedContributionsCount"]

        for entry in coll["commitContributionsByRepository"]:
            repo = entry["repository"]
            name = repo["nameWithOwner"]
            lang = (repo.get("primaryLanguage") or {}).get("name")
            committed_repo_languages[name] = lang
            owner_login = (repo.get("owner") or {}).get("login", "")
            if owner_login.lower() != username.lower():
                external_repos.add(name)

        for key in (
                "issueContributionsByRepository",
                "pullRequestContributionsByRepository",
                "pullRequestReviewContributionsByRepository",
        ):
            for entry in coll[key]:
                repo = entry["repository"]
                owner_login = (repo.get("owner") or {}).get("login", "")
                if owner_login.lower() != username.lower():
                    external_repos.add(repo["nameWithOwner"])

    return {
        "total_commits": total_commits,
        "restricted_total": restricted_total,
        "committed_repo_languages": committed_repo_languages,
        "external_repos": external_repos,
    }


PR_TOTAL_QUERY = """
query($login: String!) {
  user(login: $login) {
    pullRequests { totalCount }
  }
}
"""


def get_pr_total(username: str, token: str) -> int:
    data = graphql(PR_TOTAL_QUERY, {"login": username}, token)
    return data["user"]["pullRequests"]["totalCount"]


def compute_top_languages(
        owned_repos: list[dict],
        committed_repo_languages: dict[str, Optional[str]],
        max_shown: int = MAX_LANGUAGES_SHOWN,
) -> str:
    languages: dict[str, Optional[str]] = dict(committed_repo_languages)
    for repo in owned_repos:
        languages.setdefault(repo["full_name"], repo.get("language"))

    counts: dict[str, int] = {}
    for lang in languages.values():
        if not lang:
            continue
        counts[lang] = counts.get(lang, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return "N/A"

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_shown]
    parts = [f"{name} {round(count / total * 100)}%" for name, count in ranked]
    return " · ".join(parts)


def humanize_join_duration(joined: datetime, now: datetime) -> str:
    rd = relativedelta(now, joined)

    def unit(n: int, word: str) -> str:
        return f"{n} {word}{'s' if n != 1 else ''}"

    parts = []
    if rd.years:
        parts.append(unit(rd.years, "year"))
    if rd.months:
        parts.append(unit(rd.months, "month"))
    if rd.days or not parts:
        parts.append(unit(rd.days, "day"))

    return ", ".join(parts) + " ago"


def format_stats_block(
        joined: datetime,
        now: datetime,
        total_repos: int,
        private_repos: int,
        total_commits: int,
        pr_total: int,
        contributed_to: int,
        top_languages: str,
) -> str:
    def line(label: str, value: str) -> str:
        return f"{label:<{LABEL_WIDTH}}{value}"

    rows = [
        line("Joined", humanize_join_duration(joined, now)),
        line("Repos", f"{total_repos} ({private_repos} private)"),
        line("Commits", str(total_commits)),
        line("PRs opened", str(pr_total)),
        line("Contributed to", f"{contributed_to} repos"),
        line("Top languages", top_languages),
    ]
    body = "\n".join(rows)
    return f"```\n$ stats --summary\n{body}\n```"


def splice_readme(readme_text: str, new_block: str) -> str:
    start_idx = readme_text.find(START_MARKER)
    end_idx = readme_text.find(END_MARKER)
    if start_idx == -1 or end_idx == -1:
        raise RuntimeError(
            f"Could not find {START_MARKER} / {END_MARKER} markers in the README."
        )
    before = readme_text[: start_idx + len(START_MARKER)]
    after = readme_text[end_idx:]
    return f"{before}\n{new_block}\n{after}"


def main() -> None:
    try:
        token = os.environ["GH_PAT"]
        username = os.environ["GITHUB_USERNAME"]
    except KeyError as exc:
        sys.exit(f"Missing required environment variable: {exc}")

    readme_path = os.environ.get("README_PATH", "README.md")
    dry_run = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

    joined = get_join_date(username, token)
    now = datetime.now(timezone.utc)
    owned_repos = get_owned_repos(username, token)
    private_repos = sum(1 for r in owned_repos if r.get("private"))
    contrib = get_alltime_contribution_data(username, token, joined, now)
    pr_total = get_pr_total(username, token)
    scan = scan_all_branches(username, token, extra_repo_names=contrib["external_repos"])

    total_commits = contrib["total_commits"]
    top_languages = compute_top_languages(
        owned_repos, {**contrib["committed_repo_languages"], **scan["repo_languages"]}
    )
    contributed_to = contrib["external_repos"] | scan["external_repos"]

    print(
        f"Note: Commits below uses GraphQL's default-branch/verified-email "
        f"count ({total_commits}) -- same as your public profile. The "
        f"all-branches scan above (author-email match, {scan['total_commits']} "
        f"commits found across {len(scan['scanned_repo_names'])} repos) isn't "
        f"used for Commits, only for Top languages/Contributed to, where it "
        f"added data GraphQL's narrower view missed.",
        file=sys.stderr,
    )

    unreachable = set(contrib["committed_repo_languages"]) - scan["scanned_repo_names"]
    if unreachable:
        print(
            f"Note: GraphQL attributes default-branch commits to {len(unreachable)} "
            f"repo(s) the all-branches scan never even attempted to clone -- likely "
            f"deleted, renamed, or transferred, since GraphQL's contribution count "
            f"is a permanent historical record while a clone can only see what "
            f"currently exists: {sorted(unreachable)}",
            file=sys.stderr,
        )

    if contrib["restricted_total"]:
        print(
            f"Note: {contrib['restricted_total']} contribution(s) across all years were "
            "reported as restricted (invisible to this token even with read:user). "
            "This is usually an SSO-protected organization -- see "
            "https://github.com/settings/tokens for authorizing the PAT against it.",
            file=sys.stderr,
        )

    block = format_stats_block(
        joined=joined,
        now=now,
        total_repos=len(owned_repos),
        private_repos=private_repos,
        total_commits=total_commits,
        pr_total=pr_total,
        contributed_to=len(contributed_to),
        top_languages=top_languages,
    )

    if dry_run:
        print(block)
        return

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()

    updated = splice_readme(readme_text, block)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated)

    print("Updated stats block:\n")
    print(block)


if __name__ == "__main__":
    main()

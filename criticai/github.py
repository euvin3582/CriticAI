"""GitHub API client for CriticAI.

Handles all GitHub REST API interactions: fetching PR diffs, resolving
bot identity, finding/creating/updating review comments.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import requests

from criticai.config import Config

# Hidden marker embedded in every posted comment so we can find and
# update our own comment on subsequent runs (update-in-place behavior).
COMMENT_MARKER = "<!-- criticai:review -->"


class GitHubClient:
    """Thin wrapper around the GitHub REST API for PR review operations."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {config.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self._bot_login: Optional[str] = None

    @property
    def _base_url(self) -> str:
        return f"https://api.github.com/repos/{self._config.repository}"

    # ------------------------------------------------------------------
    # PR data
    # ------------------------------------------------------------------

    def get_pr_diff(self, base_sha: Optional[str] = None) -> str:
        """Fetch the PR diff. Raises on failure.

        If base_sha is provided, fetches the diff between that commit and
        the PR head (incremental review of only new commits). Otherwise
        fetches the full PR diff against the base branch.
        """
        if base_sha:
            # Incremental: compare specific commits
            head_sha = self.get_pr_head_sha()
            if head_sha and base_sha != head_sha:
                url = f"{self._base_url}/compare/{base_sha}...{head_sha}"
                response = self._session.get(
                    url, headers={"Accept": "application/vnd.github.diff"}
                )
                if response.status_code == 200:
                    print(f"Incremental diff fetched ({base_sha[:7]}..{head_sha[:7]})")
                    return response.text
                # Fall through to full diff on failure
                print(f"Incremental diff failed (HTTP {response.status_code}), using full diff.")

        # Full PR diff
        url = f"{self._base_url}/pulls/{self._config.pr_number}"
        response = self._session.get(
            url, headers={"Accept": "application/vnd.github.diff"}
        )
        if response.status_code == 403:
            print(
                "Error: 403 fetching PR diff. The workflow's GITHUB_TOKEN "
                "is likely missing `contents: read`. Add it to the calling "
                "workflow's `permissions:` block."
            )
        response.raise_for_status()
        print(f"PR diff fetched (HTTP {response.status_code})")
        return response.text

    def get_pr_head_sha(self) -> Optional[str]:
        """Fetch the PR head commit SHA. Best-effort, returns None on failure."""
        url = f"{self._base_url}/pulls/{self._config.pr_number}"
        try:
            response = self._session.get(url)
            response.raise_for_status()
            return response.json().get("head", {}).get("sha")
        except requests.RequestException as e:
            print(f"Warning: could not fetch PR head SHA: {e}")
            return None

    def get_file_content(self, path: str, ref: Optional[str] = None) -> Optional[str]:
        """Fetch the content of a single file from the repository.

        Args:
            path: File path relative to repo root (e.g. "src/types.ts")
            ref: Git ref (branch, tag, SHA). Defaults to PR head if None.

        Returns the decoded file content, or None if the file doesn't exist
        or can't be fetched (non-critical, just means less context).
        """
        url = f"{self._base_url}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref

        try:
            response = self._session.get(url, params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            # GitHub returns base64-encoded content for files under 1MB
            if data.get("encoding") == "base64":
                import base64
                return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
            # For larger files or different encodings, skip
            return None
        except requests.RequestException:
            return None

    # ------------------------------------------------------------------
    # Comment management (update-in-place)
    # ------------------------------------------------------------------

    def find_existing_comment(self) -> Tuple[Optional[int], Optional[str]]:
        """Find this bot's existing review comment on the PR.

        Returns (comment_id, comment_body) or (None, None).
        Paginates through all comments, newest-first, and verifies
        authorship to avoid spoofed markers.
        """
        url: Optional[str] = (
            f"{self._base_url}/issues/{self._config.pr_number}/comments"
        )
        params = {"per_page": 100, "sort": "created", "direction": "desc"}

        try:
            while url:
                response = self._session.get(url, params=params)
                response.raise_for_status()
                for comment in response.json():
                    body = comment.get("body") or ""
                    author = (comment.get("user") or {}).get("login") or ""
                    if COMMENT_MARKER in body and author == self._resolve_bot_login():
                        return comment["id"], body
                params = None  # only needed for first request
                url = response.links.get("next", {}).get("url")
        except requests.RequestException as e:
            print(f"Warning: could not list comments, will post new: {e}")

        return None, None

    def post_comment(self, body: str, existing_comment_id: Optional[int] = None) -> None:
        """Post or update the review comment on the PR."""
        if existing_comment_id:
            url = f"{self._base_url}/issues/comments/{existing_comment_id}"
            response = self._session.patch(url, json={"body": body})
            action = "updated"
        else:
            url = f"{self._base_url}/issues/{self._config.pr_number}/comments"
            response = self._session.post(url, json={"body": body})
            action = "posted"

        response.raise_for_status()
        print(f"Comment {action} successfully!")

    def post_inline_review(
        self,
        comments: list[dict],
        head_sha: Optional[str] = None,
    ) -> None:
        """Submit a PR review with inline comments on specific diff lines.

        Each item in `comments` should have: path, position, body.
        Optionally start_line/line for multi-line comments.
        Uses event=COMMENT so it doesn't block merges.
        """
        if not comments:
            print("No inline comments to post.")
            return

        url = f"{self._base_url}/pulls/{self._config.pr_number}/reviews"
        payload: dict = {
            "event": "COMMENT",
            "comments": comments,
        }
        if head_sha:
            payload["commit_id"] = head_sha

        try:
            response = self._session.post(url, json=payload)
            response.raise_for_status()
            print(f"Inline review posted with {len(comments)} comment(s).")
        except requests.RequestException as e:
            # Don't fail the whole action if inline comments fail —
            # the summary comment is already posted.
            print(f"Warning: could not post inline review: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Response: {e.response.text[:500]}")

    # ------------------------------------------------------------------
    # Existing thread lookup (deduplication)
    # ------------------------------------------------------------------

    def get_existing_bot_threads(self) -> list[dict]:
        """Fetch all unresolved review threads posted by this bot.

        Returns a list of dicts with keys: path, body (first comment body).
        Used to deduplicate findings so the bot doesn't re-post the same
        inline comment on every push.
        """
        owner, repo = self._config.repository.split("/", 1)
        pr_number = int(self._config.pr_number)
        bot_login = self._resolve_bot_login()

        # Also match without [bot] suffix (GraphQL inconsistency)
        bot_logins = {bot_login, bot_login.replace("[bot]", "")}
        if bot_login == "github-actions[bot]":
            bot_logins.add("github-actions")
        elif bot_login == "github-actions":
            bot_logins.add("github-actions[bot]")

        query = """
        query($owner: String!, $repo: String!, $pr: Int!) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $pr) {
              reviewThreads(first: 100) {
                nodes {
                  isResolved
                  path
                  comments(first: 1) {
                    nodes {
                      author { login }
                      body
                    }
                  }
                }
              }
            }
          }
        }
        """

        try:
            response = self._session.post(
                "https://api.github.com/graphql",
                json={"query": query, "variables": {"owner": owner, "repo": repo, "pr": pr_number}},
            )
            if response.status_code != 200:
                return []

            data = response.json()
            if data.get("errors"):
                return []

            threads = (
                data.get("data", {})
                .get("repository", {})
                .get("pullRequest", {})
                .get("reviewThreads", {})
                .get("nodes", [])
            )

            existing = []
            for thread in threads:
                if thread.get("isResolved"):
                    continue
                comments = thread.get("comments", {}).get("nodes", [])
                if not comments:
                    continue
                author = (comments[0].get("author") or {}).get("login", "")
                if author not in bot_logins:
                    continue
                existing.append({
                    "path": thread.get("path", ""),
                    "body": comments[0].get("body", ""),
                })

            return existing

        except requests.RequestException:
            return []

    # ------------------------------------------------------------------
    # Identity resolution
    # ------------------------------------------------------------------

    def _resolve_bot_login(self) -> str:
        """Determine the GitHub login this token is posting as.

        For the default GITHUB_TOKEN (an installation token), GET /user
        returns 404 — the identity is always 'github-actions[bot]'.
        For a PAT, it returns the token owner's login.
        """
        if self._bot_login is not None:
            return self._bot_login

        try:
            response = self._session.get("https://api.github.com/user")
            if response.status_code == 404:
                self._bot_login = "github-actions[bot]"
            else:
                response.raise_for_status()
                self._bot_login = response.json()["login"]
        except requests.RequestException as e:
            print(f"Warning: could not resolve bot identity: {e}")
            self._bot_login = "github-actions[bot]"

        return self._bot_login


def extract_previous_findings(comment_body: Optional[str]) -> Optional[str]:
    """Extract the Findings section from a previous review comment.

    Returns the findings text, or None if no Findings section exists.
    """
    if not comment_body:
        return None
    match = re.search(
        r"## Findings\s*\n(.*?)(?=\n## |\n---)",
        comment_body,
        re.DOTALL,
    )
    if match:
        findings = match.group(1).strip()
        return findings if findings else None
    return None


def extract_reviewed_sha(comment_body: Optional[str]) -> Optional[str]:
    """Extract the commit SHA from a previous review comment's header.

    The comment header contains: '<sub>CriticAI reviewed at commit `abc1234`</sub>'
    Returns the full short SHA, or None if not found.
    """
    if not comment_body:
        return None
    match = re.search(r"reviewed at commit `([a-f0-9]+)`", comment_body)
    return match.group(1) if match else None

"""CriticAI entrypoint — wires modules together and runs the review.

This is intentionally thin. All logic lives in the criticai/ package.
"""

import os
import sys

from criticai.auto_approve import should_auto_approve, post_auto_approval
from criticai.config import Config
from criticai.context import gather_context
from criticai.diagrams import generate_diagram
from criticai.diff import filter_diff, build_position_map, find_position, extract_changed_files
from criticai.github import GitHubClient, extract_previous_findings, extract_reviewed_sha
from criticai.inline import parse_model_output, filter_by_confidence
from criticai.learnings import load_learnings, build_suppression_prompt
from criticai.linters import run_linters, deduplicate_findings
from criticai.pr_description import maybe_generate_description
from criticai.renderer import render_comment
from criticai.resolve import resolve_outdated_threads
from criticai.review import ReviewEngine
from criticai.rules import load_rules


def main() -> None:
    # Load configuration from environment
    config = Config.from_env()

    # Initialize clients
    github = GitHubClient(config)
    engine = ReviewEngine(config)

    # Fetch PR diff — always use the FULL PR diff for review so all
    # findings are surfaced at once. The incremental SHA is only used
    # for resolution tracking (knowing what's new since last review).
    existing_id, existing_body = github.find_existing_comment()
    last_reviewed_sha = extract_reviewed_sha(existing_body)

    if last_reviewed_sha:
        print(f"Previous review found (reviewed at {last_reviewed_sha}).")

    # Always review the full PR diff so every issue is caught in one pass
    raw_diff = github.get_pr_diff()
    diff = filter_diff(raw_diff, config.home_directory)

    # Fetch incremental diff separately for resolution tracking — this
    # tells the LLM what changed SINCE the last review so it can
    # accurately determine which prior findings were addressed.
    incremental_diff = ""
    if last_reviewed_sha:
        raw_incremental = github.get_pr_diff(base_sha=last_reviewed_sha)
        if raw_incremental:
            incremental_diff = filter_diff(raw_incremental, config.home_directory)

    # Fetch head SHA for labeling the comment
    head_sha = github.get_pr_head_sha()

    # Extract changed file list (needed for auto-approval, linters, context)
    changed_files = extract_changed_files(diff)

    # Load custom rules from .criticai.yml (if present)
    repo_rules = load_rules(github, config, head_sha)
    rules_prompt = ""
    if repo_rules:
        print(f"Custom rules loaded ({len(repo_rules.rules)} rules, {len(repo_rules.focus)} focus areas)")
        rules_prompt = repo_rules.build_prompt_addition()

        # Check auto-approval BEFORE running the full review
        approve, reason = should_auto_approve(diff, changed_files, repo_rules)
        if approve:
            post_auto_approval(config, github, head_sha, reason)
            # Still generate PR description if needed
            if not existing_id:
                maybe_generate_description(github, config, diff)
            return  # Skip the full review — PR is safe

        # Filter out ignored files from the diff
        if repo_rules.ignore:
            import re as _re
            lines = diff.split("\n")
            kept_lines = []
            current_file = None
            for line in lines:
                if line.startswith("diff --git"):
                    match = _re.search(r"diff --git a/(.*?) b/", line)
                    if match:
                        current_file = match.group(1)
                if current_file and not repo_rules.should_ignore_file(current_file):
                    kept_lines.append(line)
                elif current_file is None:
                    kept_lines.append(line)
            diff = "\n".join(kept_lines)
            changed_files = extract_changed_files(diff)
    else:
        # No rules file — still check auto-approval with defaults
        approve, reason = should_auto_approve(diff, changed_files)
        if approve:
            post_auto_approval(config, github, head_sha, reason)
            if not existing_id:
                maybe_generate_description(github, config, diff)
            return

    # Load team learnings (dismissed findings)
    dismissed = load_learnings(github, head_sha)
    learnings_prompt = build_suppression_prompt(dismissed)

    # Gather codebase context (referenced files for cross-file awareness)
    changed_files = extract_changed_files(diff)
    context = gather_context(diff, changed_files, github, head_sha)

    # Extract prior findings for resolution tracking
    previous_findings = extract_previous_findings(existing_body)
    if previous_findings:
        print("Found previous findings — will track resolution across pushes.")

    # Generate PR description if missing (first review only)
    if not existing_id:
        maybe_generate_description(github, config, diff)

    # Run the AI review (with context, rules, learnings, previous findings)
    raw_output = engine.run(
        diff,
        previous_findings,
        context=context,
        rules_prompt=rules_prompt,
        learnings_prompt=learnings_prompt,
        incremental_diff=incremental_diff,
    )
    if raw_output is None:
        print("No review to post (all models failed).")
        sys.exit(1)

    # Parse into summary + structured inline findings
    review_output = parse_model_output(raw_output)

    # Run static analysis linters on changed files and merge findings
    linters_config = os.environ.get("INPUT_LINTERS", "auto")
    linter_findings = run_linters(changed_files, linters_config)
    if linter_findings:
        review_output.findings = deduplicate_findings(
            review_output.findings, linter_findings
        )

    # Generate sequence diagram for the walkthrough
    diagram = generate_diagram(config, diff)
    summary_with_diagram = review_output.summary
    if diagram:
        # Insert diagram after the Walkthrough section
        if "## Findings" in summary_with_diagram:
            summary_with_diagram = summary_with_diagram.replace(
                "## Findings",
                f"\n<details><summary>📊 Sequence Diagram</summary>\n\n{diagram}\n\n</details>\n\n## Findings"
            )
        elif "## Suggested follow-ups" in summary_with_diagram:
            summary_with_diagram = summary_with_diagram.replace(
                "## Suggested follow-ups",
                f"\n<details><summary>📊 Sequence Diagram</summary>\n\n{diagram}\n\n</details>\n\n## Suggested follow-ups"
            )
        else:
            summary_with_diagram += f"\n\n<details><summary>📊 Sequence Diagram</summary>\n\n{diagram}\n\n</details>"

    # Post/update the summary comment (issue comment, update-in-place)
    summary_body = render_comment(summary_with_diagram, head_sha, config)
    github.post_comment(summary_body, existing_id)

    # Post inline review comments on the diff (if any findings have positions)
    if review_output.findings:
        # Apply noise control — only post high-confidence findings
        min_confidence = "medium"  # default
        if repo_rules and repo_rules.min_severity in ("critical", "major"):
            min_confidence = "high"  # stricter noise control when severity threshold is high
        filtered_findings = filter_by_confidence(review_output.findings, min_confidence)

        # Deduplicate: skip findings that match an existing unresolved bot thread
        # on the same file (prevents re-posting the same comment on every push).
        existing_threads = github.get_existing_bot_threads()
        if existing_threads:
            deduplicated = []
            for finding in filtered_findings:
                if _finding_already_posted(finding, existing_threads):
                    print(
                        f"  Skipping duplicate: {finding.path}:{finding.line} "
                        f"(already posted in a previous review)"
                    )
                else:
                    deduplicated.append(finding)
            skipped = len(filtered_findings) - len(deduplicated)
            if skipped:
                print(f"  Deduplication: skipped {skipped} finding(s) already posted.")
            filtered_findings = deduplicated

        position_map = build_position_map(diff)
        inline_comments = []

        for finding in filtered_findings:
            position = find_position(position_map, finding.path, finding.line)
            if position is None:
                print(
                    f"  Skipping inline comment for {finding.path}:{finding.line} "
                    f"(not in diff)"
                )
                continue

            comment: dict = {
                "path": finding.path,
                "position": position,
                "body": finding.format_body(),
            }
            inline_comments.append(comment)

        if inline_comments:
            github.post_inline_review(inline_comments, head_sha)
        else:
            print("No findings mapped to diff positions — inline review skipped.")
    else:
        print("No structured findings — inline review skipped.")

    # Auto-resolve threads from previous reviews (user pushed a fix)
    # Uses three passes: 1) match resolved findings from the summary,
    # 2) GitHub's isOutdated flag, 3) semantic LLM check for the rest.
    resolve_outdated_threads(
        github, config, diff, head_sha=head_sha,
        review_summary=review_output.summary,
    )


def _finding_already_posted(finding, existing_threads: list[dict]) -> bool:
    """Check if a finding matches an existing unresolved bot thread.

    Matches by file path and keyword overlap in the comment body.
    Requires BOTH high keyword overlap AND at least 3 shared keywords
    to avoid false positives where two different concerns on the same
    file share domain vocabulary (e.g., "token", "auth", "validate").
    """
    import re as _re

    for thread in existing_threads:
        # Must be on the same file
        if thread["path"] != finding.path:
            continue

        # Check keyword overlap between the finding and the existing comment
        existing_body = thread["body"].lower()
        finding_body = finding.format_body().lower()

        # Extract meaningful words (4+ chars, skip common review boilerplate)
        stop_words = {
            "this", "that", "with", "from", "have", "been", "will",
            "would", "could", "should", "about", "their", "which",
            "when", "where", "what", "than", "them", "then", "into",
            "some", "other", "more", "very", "just", "also", "your",
            "each", "only", "major", "minor", "critical", "finding",
            "suggestion", "suggested", "change", "code", "file",
            "line", "function", "method", "class", "return", "value",
        }

        def extract_words(text):
            cleaned = _re.sub(r"[`*~#\[\](){}|>]", " ", text)
            words = set(_re.findall(r"\b[a-z]{4,}\b", cleaned))
            return words - stop_words

        existing_words = extract_words(existing_body)
        finding_words = extract_words(finding_body)

        if not existing_words or not finding_words:
            continue

        overlap = existing_words & finding_words

        # Require BOTH conditions to consider it a duplicate:
        # 1. At least 3 shared meaningful keywords (absolute threshold)
        # 2. At least 50% of the finding's keywords overlap (relative threshold)
        overlap_ratio = len(overlap) / max(len(finding_words), 1)
        if len(overlap) >= 3 and overlap_ratio >= 0.5:
            return True

    return False


if __name__ == "__main__":
    main()

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

    # Fetch PR diff (incremental if this is a re-review)
    existing_id, existing_body = github.find_existing_comment()
    last_reviewed_sha = extract_reviewed_sha(existing_body)

    if last_reviewed_sha:
        print(f"Previous review found (reviewed at {last_reviewed_sha}) — trying incremental diff.")
        raw_diff = github.get_pr_diff(base_sha=last_reviewed_sha)
    else:
        raw_diff = github.get_pr_diff()

    diff = filter_diff(raw_diff, config.home_directory)

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
    # Passes the diff so the LLM can semantically evaluate whether each
    # concern was addressed — not just whether the exact line changed.
    resolve_outdated_threads(github, config, diff, head_sha=head_sha)


if __name__ == "__main__":
    main()

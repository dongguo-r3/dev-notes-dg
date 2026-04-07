# Skill: Blockers Report

## Trigger
Run this when the user asks for a blockers report, or wants to see what is blocked across all projects.

## Steps
1. Open each active worker file in `projects/`.
2. For each file, read the `## Blockers` section.
3. Collect all entries that are not marked as resolved.
4. Compile the report using the output format below.
5. Present the report to the user.
6. Ask if the user wants to take action on any blocker (update, resolve, escalate).

## Output Format

## Blockers Report — YYYY-MM-DD

| Project | Blocker | Since | Owner |
|---|---|---|---|
| project-alpha | Waiting on design review | 2026-04-05 | @designlead |
| feature-xyz | Dependency not released yet | 2026-04-03 | @devteam |

_(If no blockers found, report "No active blockers across all projects.")_

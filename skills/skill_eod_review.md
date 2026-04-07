# Skill: EOD Review

## Trigger
Run this at end of day, or when the user asks for an EOD review or daily update.

## Steps
1. Open `MASTER.md` and read the Project Index.
2. For each project with status other than `Archived`, open its worker file.
3. For each worker file:
   a. Read the most recent entry in `## Notes / Running Log`.
   b. Check `## Blockers` for any new or unresolved blockers.
   c. Note the current `Status` from Meta.
4. Compose a new dated summary block for MASTER.md using the output format below.
5. Append the block to the `## Daily Logs` section in MASTER.md (newest on top).
6. Refresh the Project Index table in MASTER.md:
   - Update Status and Last Updated columns where changed.
7. Report back: list which files were reviewed and confirm what was written to MASTER.

## Output Format (Daily Log Entry)

### YYYY-MM-DD
- **[project-name]**: [1-line summary of progress]. [🔴 Blocked: reason — if applicable]
- **[project-name]**: [1-line summary]. [🟢 Done — if just completed]

## Rules
- Do not copy full notes into MASTER — summarize only.
- If nothing changed for a project today, skip it from the log entry.
- If a project was marked Done today, note it explicitly.

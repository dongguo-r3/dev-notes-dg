# Skill: Status Update

## Trigger
Run this when the user wants to update the status or progress of a specific project.

## Inputs Required
- Project name or filename
- New status (Draft / In Progress / Blocked / Done / Archived)
- Optional: new blocker description, or blocker resolution note
- Optional: a progress note to append to the running log

## Steps
1. Open the worker file in `projects/`.
2. Update `Status` in the Meta block to the new value (with emoji).
3. Update `Last Updated` in Meta to today's date.
4. If status is `Blocked`:
   a. Add the blocker to the `## Blockers` section if not already present.
   b. Append a note to `## Notes / Running Log`: `YYYY-MM-DD: Status set to Blocked — [reason].`
5. If a blocker is being resolved:
   a. Remove or strike through the relevant entry in `## Blockers`.
   b. Append a note to `## Notes / Running Log`: `YYYY-MM-DD: Blocker resolved — [resolution].`
6. If a progress note was provided, append it to `## Notes / Running Log`.
7. Update the Project Index row in MASTER.md:
   - Refresh Status and Last Updated columns.
8. Report back: confirm what was changed.

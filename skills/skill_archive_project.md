# Skill: Archive Project

## Trigger
Run this when the user wants to close and archive a completed project.

## Inputs Required
- Project name or filename

## Steps
1. Open the worker file in `projects/`.
2. Confirm with the user that this project is ready to archive.
3. In the worker file:
   a. Set `Status` to `⚪ Archived` in Meta.
   b. Update `Last Updated` to today's date.
   c. Append a final entry to `## Notes / Running Log`:
      `YYYY-MM-DD: Project archived.`
4. Move the file from `projects/` to `archive/`.
   - Rename with a date prefix: `YYYY-MM-[original-slug].md`
5. Update the Project Index row in MASTER.md:
   - Set Status to `⚪ Archived`.
   - Update Last Updated.
   - (Optional) Convert the link to point to the new archive path.
6. Append a note in today's EOD log in MASTER.md (or create a new entry if none exists):
   `- **[project-name]**: ⚪ Archived.`
7. Report back: confirm the file was moved and MASTER was updated.

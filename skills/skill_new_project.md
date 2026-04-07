# Skill: New Project

## Trigger
Run this when the user wants to create a new project or worker file.

## Inputs Required
- Project name
- Owner
- Priority (Low / Medium / High / Critical)
- Brief goal description
- Initial tasks (if known)

## Steps
1. Copy `_template.md` content as the base.
2. Fill in all Meta fields using the provided inputs.
3. Set `Created` and `Last Updated` to today's date (YYYY-MM-DD).
4. Set Status to `🔵 Draft`.
5. Write the Goal section using the provided description.
6. Populate the Tasks checklist with any provided initial tasks.
7. Save the file as `projects/[project-slug].md`.
   - Use lowercase, hyphen-separated slugs (e.g. `feature-user-auth.md`).
8. Add a new row to the Project Index table in `MASTER.md`:
   - Project name (linked to file), Owner, Status, Priority, Last Updated.
9. Report back: confirm the file was created and the MASTER index was updated.

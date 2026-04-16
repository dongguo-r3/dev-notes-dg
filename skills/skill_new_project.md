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
1. Choose a filename following the hierarchical naming convention (see `claude.md`):
   - Format: `level1_level2_..._levelK.md` (general → specific, underscores, no generic suffixes).
   - Example: a note about the T2A model → `omni_model_t2a.md`; a specific sub-approach → `video_captioning_director_style.md`.
2. Copy `_template.md` content as the base.
3. Add an **intro block** at the very top of the file (before the Meta section):
   - One sentence explaining what this file is for.
   - A summary of current content in the most appropriate format (status table, catalog table, or bullet list).
   - Format as a Markdown blockquote (`>`). See `claude.md` for the required format.
4. Fill in all Meta fields using the provided inputs.
5. Set `Created` and `Last Updated` to today's date (YYYY-MM-DD).
6. Set Status to `🔵 Draft`.
7. Write the Goal section using the provided description.
8. Populate the Tasks checklist with any provided initial tasks.
9. Save the file to `projects/[chosen-filename].md`.
10. Add a new row to the Project Index table in `MASTER.md`:
    - Project name (linked to file), Owner, Status, Priority, Last Updated.
11. Report back: confirm the file was created and the MASTER index was updated.

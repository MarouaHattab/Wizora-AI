# Wizora AI README Rewrite Design

## Goal

Reorganize the repository README so it follows the clear, project-oriented structure
of the provided CourseTA example while remaining accurate to the current Wizora AI
implementation.

## Scope

- Modify only `README.md` for the user-facing documentation change.
- Keep the README detailed, including architecture, workflow diagrams, API routes,
  configuration, testing, and limitations.
- Use the project name `Wizora AI` consistently.
- Remove the Demo section and do not add screenshots, videos, hosted demo URLs, or
  other demo claims.
- Keep the existing generated architecture diagrams and their repository-relative links.
- Document the current Python 3.11 and `uv` setup, provider choices, FFmpeg behavior,
  backend startup, Gradio startup, API prefixes, SSE behavior, and validation commands.
- Preserve the documented API contracts and describe the current route paths rather
  than copying outdated CourseTA endpoints.

## Validation

Because this is a documentation-only change, validation will consist of checking the
resulting Markdown for stale CourseTA/demo references, broken local diagram references,
and consistency with the current entry points and API route definitions. No application
runtime behavior will be changed.

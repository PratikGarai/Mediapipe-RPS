---
name: generate-docstrings
description: 'Generate Python module, class, function, and method docstrings using this repository''s documentation rules. Use when adding or fixing Google-style docstrings, checking Args/Returns/Raises sections, and aligning lightweight type annotations with the existing codebase conventions.'
argument-hint: 'Target files, folders, or scope for the docstring pass'
user-invocable: true
---

# Generate Docstrings

Use this skill to perform a focused documentation pass on Python code in this repository.

This workflow is for cases where the goal is to add or normalize docstrings, not to redesign APIs or refactor unrelated behavior.

## When to Use

- Add missing docstrings across one file, a folder, or the whole repository.
- Normalize existing Python docstrings to Google style.
- Ensure docstrings match the repository rules in [.github/copilot-instructions.md](../../copilot-instructions.md).
- Add small missing type annotations when they are required to keep documentation updates consistent with local rules.

## Inputs

- Requested scope, such as a file, folder, or the full Python workspace.
- Any special exclusions the user mentions.

## Procedure

1. Start from the requested scope.
   - If the user names files, use those files.
   - If the user asks for a broad pass, enumerate Python files under the target scope.

2. Read the local documentation rules before editing.
   - Follow [.github/copilot-instructions.md](../../copilot-instructions.md) for docstrings and type annotations.
   - Treat those rules as the source of truth over generic style defaults.

3. Inspect only enough code to identify missing or non-compliant documentation.
   - Cover module docstrings when absent.
   - Cover every class, function, and method.
   - Preserve existing behavior and avoid unrelated cleanup.

4. Add or update docstrings using Google style.
   - Use a one-line imperative summary.
   - Add `Args:` for functions or methods with parameters.
   - Add `Returns:` when the return value is not `None`.
   - Add `Raises:` only when the code explicitly raises an exception.
   - Keep wording concrete and tied to actual behavior.

5. Align signatures only when required by repository rules.
   - Add missing parameter annotations or return annotations when they are straightforward.
   - Prefer built-in generic types like `list[str]` and `dict[str, int]`.
   - Do not introduce suppressions or broad refactors.

6. Validate immediately after the edit pass.
   - Run diagnostics on the touched files or the workspace.
   - Fix any syntax or annotation issues introduced by the documentation update.

7. Report completion concisely.
   - Summarize the scope covered.
   - Note any files that were skipped or any ambiguities that remain.

## Decision Points

- If the repo has explicit documentation rules, follow them instead of inventing a new standard.
- If a function already has a docstring, update it only when it is clearly incomplete or inconsistent with the rules.
- If adding a required annotation would need a larger design decision, stop at the smallest safe change and call out the gap.
- If validation reveals a defect unrelated to the docstring edits, do not expand scope to fix it unless the user asks.

## Completion Checks

- Every targeted Python module has a module docstring when appropriate.
- Every targeted class, function, and method has a Google-style docstring.
- `Args:`, `Returns:`, and `Raises:` sections are present only when warranted.
- Any added type annotations match repository conventions.
- Diagnostics are clean after the edit.

## Example Prompts

- `/generate-docstrings src/commons`
- `/generate-docstrings add docstrings to demos/*.py`
- `/generate-docstrings run a full repo docstring pass`
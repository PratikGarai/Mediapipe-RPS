## Code Quality

- NEVER use lint suppressions. No `pylint: disable`, no `# type: ignore`, no `# noqa`. Fix the actual issue instead of suppressing it.
- Do not add suppression rules to `pyproject.toml`, `.pylintrc`, or any other config file to hide warnings globally either. Let them get flagged; fix them properly.

## Docstrings

- Every class, function, and method **must** have a Google-style docstring.
- Use a one-line summary in the imperative mood (e.g. "Return …", "Fetch …").
- Add an `Args:` section when the function has parameters (skip for zero-arg functions).
- Add a `Returns:` section when the function returns something other than `None`.
- Add a `Raises:` section only when the function explicitly raises an exception.

**Reference example:**

```python
def get_tasks_by_category(
    self,
    tasks: list[dict],
    category: TaskCategory,
) -> list[dict]:
    """Return tasks classified under the given category.

    Args:
        tasks: Full list of task nodes.
        category: The desired category to filter by.

    Returns:
        List of tasks matching the category.
    """
```

## Type Annotations

- Every function **must** have type annotations on all parameters and on the return value.
- Use raw Python built-in types first: `int`, `str`, `float`, `bool`, `list`, `dict`, `tuple`, `set`, `None`.
- Use generic syntax directly: `list[str]`, `dict[str, int]`, `tuple[str, ...]`.
- Use `X | None` (PEP 604 union syntax) instead of `Optional[X]`.
- Use the `typing` module **only** as a last resort for types not expressible with builtins (e.g. `Callable`, `Any`, `Iterator`).
- Do **not** use pydantic for type annotations.

## String Formatting

- **Always** use f-strings for string interpolation: `f"value is {x}"`.
- **Never** use `str.format()` — prefer f-strings in all cases.
- **Exception**: `logging` calls (`logger.info`, `logger.warning`, etc.) **must** use `%s` / `%d` style placeholders with positional arguments — this is the standard Python logging convention and enables lazy formatting and log aggregation grouping.

## Imports

- **Never** use lazy/deferred imports (imports inside functions) to work around circular imports. They hide cyclic dependency bugs until runtime.
- Fix circular imports structurally: move the offending module to the correct architectural layer.
- All imports must be at module top level.

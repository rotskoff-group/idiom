# Docstrings

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings),
with brevity as the default:

- Start with a short sentence describing the behavior.
- Use one sentence for simple helpers; omit sections that repeat the summary or signature.
- Use `Args:`, `Returns:`, `Yields:`, and `Raises:` when callers need more detail.
- Keep types in annotations; include them in docstrings only for unannotated parameters or results.
- Document shapes, units, coordinate conventions, meaningful defaults, and side effects.
- Describe the contract, not each implementation step. Keep tutorials in the cookbook.
- Avoid repeating constructor arguments in class attributes or documenting obvious test helpers.

Ruff checks basic docstring formatting without requiring a docstring on every function.
Run `ruff check src cookbook tests` before submitting changes.

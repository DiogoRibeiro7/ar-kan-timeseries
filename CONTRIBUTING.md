# Contributing Guide

Thanks for helping improve **ar-kan-timeseries**!

## Development Setup

- Python: 3.10–3.12
- Package manager: Poetry (recommended)

```bash
poetry install
poetry run ruff check . --fix
poetry run mypy src
poetry run pytest -q
```

## Branching & Commits

- Branch from `main` using short prefixes:

  - `feat/…`, `fix/…`, `docs/…`, `refactor/…`, `perf/…`, `test/…`, `build/…`, `ci/…`

- Use **Conventional Commits** examples:

  - `feat: add bspline basis`
  - `fix: handle STL failure on short series`

## Code Style & Quality

- Type hints are **required**.
- Keep functions small and documented (Google-style docstrings).
- Lint & format: `ruff` (line length ≤ 100).
- Add/extend tests for any behavior change.

## Tests

- Place tests in `tests/`.
- For new modules, add at least:

  - shape/typing smoke tests,
  - failure path (bad input) tests,
  - simple determinism tests (seeded).

## PR Checklist

- [ ] Code is typed and documented
- [ ] `ruff`, `mypy`, `pytest` all pass
- [ ] Added/updated tests
- [ ] Updated README/docs if needed
- [ ] Linked related issues

## Reporting Bugs

Open a **Bug report** issue (template available) with:

- Repro steps, minimal code snippet
- Expected vs. actual behavior
- Environment (OS, Python, package versions)

## Security

Do not post sensitive logs or credentials in issues/PRs.

## Contact

Maintainer: **Diogo Ribeiro** -- dfr@esmad.ipp.pt

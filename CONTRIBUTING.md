# Contributing

This document defines the development conventions for this project.
It exists primarily as a disciplined engineering practice and portfolio signal.

---

## Branch Naming

| Branch type          | Pattern                              | Example                                  |
|----------------------|--------------------------------------|------------------------------------------|
| Phase development    | `phase/<N>-<short-description>`      | `phase/1-theoretical-foundation`         |
| Bug fix              | `fix/<short-description>`            | `fix/precision-at-recall-interpolation`  |
| Documentation        | `docs/<short-description>`           | `docs/update-readme-reproduction`        |
| Post-publication fix | `fix/post-publication-<description>` | `fix/post-publication-broken-link`       |

**Rules:**
- Always branch from `main`.
- Never commit directly to `main`.
- One branch per phase or logical unit of work.
- Delete branches after merging.

---

## Commit Message Convention (Conventional Commits)

Format: `<type>(<scope>): <short description>`

| Type       | When to use                                        |
|------------|----------------------------------------------------|
| `feat`     | Adding new content, code, or section               |
| `fix`      | Correcting an error (code, math, text)             |
| `docs`     | Documentation-only changes                         |
| `refactor` | Code restructuring without behavior change         |
| `test`     | Adding or modifying experiment scripts             |
| `chore`    | Tooling, dependencies, configuration               |
| `style`    | Formatting, whitespace (no logic change)           |

**Examples:**

```
feat(theory): add Bayes derivation connecting precision and base rate
fix(experiment-c): correct threshold interpolation in precision@recall
docs(readme): add step-by-step reproduction instructions
chore(deps): pin scikit-learn to 1.4.0 for reproducibility
```

**Rules:**
- Subject line: imperative mood, lowercase after type tag, no period at end.
- 72 characters max per line.
- Body (optional): explain *why*, not *what*. Separate from subject with blank line.

---

## Pull Request Protocol

- Title format: `[Phase N] Short description`
- Squash and merge strategy — one clean commit per phase in `main`.
- Fill out the PR template completely before requesting review.
- All checklist items must be resolved before merge.

---

## Code Standards

- Python 3.11+
- All scripts importable from `src/` — no logic duplication in `scripts/`.
- All random seeds controlled via `config.yaml` (never hardcoded).
- All figures saved to `figures/` at 300dpi, PNG format, consistent naming.
- No hardcoded file paths — use `config.yaml` or `pathlib.Path` relative to project root.

---

## Reproducibility Contract

Running the following in a clean environment must regenerate all figures:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
```

Any deviation from this is a bug.

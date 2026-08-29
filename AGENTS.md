  # Project instructions

  Before working on this repository, read `CLAUDE.md` and follow its project guidance.

  Run tests with:

  ```bash
  pytest -q
  ```

  Use the project virtualenv for all Python commands: `.venv/bin/python`.

## Scratch notebook imports

  Scratch notebooks may be launched with either the repository root or `scratch/` as the
  working directory. Before importing repository modules, find the repository root by walking
  upward until the project module is present, then prepend that directory to `sys.path`:

  ```python
  import sys
  from pathlib import Path

  repo_root = Path.cwd().resolve()
  while not (repo_root / "data_generation.py").exists():
      repo_root = repo_root.parent
  sys.path.insert(0, str(repo_root))
  ```

  Do this before imports such as `import data_generation` or `from scratch import ...`.

## API verification and smoke tests

  Before calling project APIs from new orchestration code, inspect or verify their exact
  signatures, especially keyword-only arguments. After adding a workflow, run a minimal
  end-to-end smoke test with tiny sizes before relying only on unit tests; this catches wiring,
  argument-shape, and integration errors that isolated tests may miss.

  Keep changes focused and preserve unrelated user work.

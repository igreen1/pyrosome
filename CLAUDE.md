# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pyrosome` is versioned, append-only on-disk storage for polars data frames (parquet via `ParquetDal`, CSV via `CsvDal`), safe for several processes sharing one data directory. Python 3.15+ (uses `lazy import`, PEP 695 generics, deferred annotations). Managed with uv; the package lives in `src/pyrosome/`. Not published to PyPI.

## Platform support

The library must work on Windows, Linux and macOS, for data directories on local disks and NFS shares. Filesystem operations must behave correctly on all of these: e.g. on Windows a read-only file cannot be deleted and the read-only attribute is shared by every hard link to a file.

SMB shares are second-class: support them where it's practical, but they are lower priority than local disks and NFS. Don't compromise correctness, simplicity or performance on the first-class targets to accommodate SMB; document SMB-specific limitations instead (e.g. many SMB shares don't support hard links, and some ignore or reject `chmod`).

## Commands

```console
uv sync --all-groups                     # dev + docs dependencies
uv run pytest                            # unit tests AND every doc/README/docstring example
uv run pytest tests/dal/test_dal.py::TestWrite::test_write_then_read
uv run pytest docs/recipes.rst           # one docs page's doctests (also README.md, src/pyrosome/dal.py)
uv run ruff check src docs conftest.py && uv run ruff format --check src docs conftest.py
uv run ty check src docs conftest.py
uv run mypy src docs conftest.py
uv run pylint src docs conftest.py
uv run sphinx-build -W -b html docs docs/_build/html
```

If uv warns about hard-linking, set `UV_LINK_MODE=copy`.

## Lint and type-check strictness

All four tools are configured at their strictest in `pyproject.toml` (ruff `select = ["ALL"]` with preview, ty `all = "error"`, mypy `strict` plus `disallow_any_expr`/`disallow_any_explicit`, pylint with every extension and message enabled). `src/`, `docs/` and `conftest.py` pass cleanly; keep them that way rather than adding ignores. `tests/` does **not** yet pass (missing annotations/docstrings), so `uv run ruff check` on the whole repo fails — that's known, not a regression.

Consequences that shape the code:
- No `while` loops and at most one statement per `try` (pylint). Retry loops go through `_claim_first_free` in `dal.py` or a bounded `for`.
- No `Any` expressions: e.g. `json.loads` output is narrowed with `cast("object", ...)` plus `match`.
- Every function/method docstring is Google style with `Args`, `Returns` (including `None.`) and `Raises` sections, and `Raises` lists exceptions propagated from callees too. The ruff/pylint rules that would forbid this (`docstring-extraneous-returns`, `docstring-extraneous-exception`, `redundant-returns-doc`) are deliberately disabled. Omit `Args`/`Raises` only when they'd be empty (napoleon renders placeholders badly).
- Ruff config references rules by name, not code (ruff 0.17 flags codes). No copyright headers.

## Architecture

`src/pyrosome/api.py` holds `SimpleIo`, a thin parquet-only wrapper whose `read` takes equality filters as keyword arguments (`None` matches nulls; floats are rejected), and whose `write`/`append` also take pandas frames (named index levels become columns, an unnamed index is dropped). pandas is never imported by pyrosome itself: it's an optional extra (`pyrosome[pandas]`, with pyarrow, which polars needs for pandas' string columns), and a dev dependency with `pandas-stubs`. `src/pyrosome/datasets.py` holds `Dataset`, which reads every profile below some keys through a `SimpleIo` with the same filters; subclasses define `_calc_<profile>` methods returning a frame, which `get_cache_profile` calls (unbound) on the dataset `without_filters` and writes, recalculating only with `refresh=True`. `get_subset`/`without_filters` construct `type(self)`, so subclasses must keep `Dataset.__init__`'s signature. The storage itself is two modules in `src/pyrosome/`:

- `dal.py`: the `Dal` protocol (public API: `read`, `write`, `append`), `_FileDal` (abstract `Dal` implementation holding all the file operations), `ParquetDal` and `CsvDal` (subclass `_FileDal`, defining only `_sink`/`_scan` and `extension`), `get_accessor` (file ending → class), and `DataManifestFile`.
- `simple_file_lock.py`: `SimpleWriteLock`, an advisory cross-process lock implemented as a `<file>.write_lock` sibling file, with timeout/poll retry.

Public names (`SimpleIo`, `Dataset`, `Dal`, `ParquetDal`, `CsvDal`, `get_accessor`) are re-exported from `pyrosome/__init__.py`. Docs, examples and docstrings teach `from pyrosome import ParquetDal`, and `docs/api.rst` documents them as `pyrosome.<name>` (cross-reference them that way); rarely needed internals such as `DataManifestFile` and the lock are documented in their submodules. Library code imports from the submodules, never from `pyrosome`, to avoid circular imports.

On-disk layout: `data_directory/<keys...>/` is a *profile* containing `manifest.json` (`{"latest_data_dir": "vN"}`) and one `vN/` dir per version holding `0.<ext>`, `1.<ext>`, … Keys are path components and must resolve inside `data_directory`.

The concurrency design is the non-obvious part, spread across both modules:
- `write` claims `vN` with atomic `mkdir(exist_ok=False)` (`reserve_new_data_dir`), sinks to a hidden `.<uuid>.<ext>.tmp`, renames to `0.<ext>`, and only then calls `commit_data_dir`, which updates the manifest under the lock and never moves it backwards.
- `append` sinks to a tmp file, claims the next free index by exclusively creating a hidden `.<idx>.<ext>.claim` marker (`touch(exist_ok=False)`), then `replace`s the tmp file to `<idx>.<ext>`. Markers are never deleted: deleting one would let a later append re-claim the index and silently replace published data (a plain rename overwrites on POSIX). Hard links aren't used because many SMB shares don't support them.
- Data files are made read only before being moved into place; `_remove_tmp_file` restores write permission before deleting a leftover tmp file because Windows can't delete read-only files.
- The manifest is replaced atomically (tmp + `replace`), so readers take no lock.
- Readers only see files matching `_DATA_FILE_PATTERN[extension]` (`^\d+\.<ext>$`). Adding a new format means adding a pattern there as well as subclassing `_FileDal`.

Known limitation: `ParquetDal` and `CsvDal` share the manifest format, so mixing formats in one profile hides the other format's data.

## Tests and doctests

`uv run pytest` collects `tests/`, doctests in `src/` modules, `docs/*.rst` and `README.md` (configured in `[tool.pytest]`). `conftest.py`'s `run_in_tmp_path` fixture (applied via `usefixtures`, not autouse) chdirs every test into a fresh tmp dir, so examples use relative paths like `Path("data")` and must be self-contained (explicit imports, no injected globals).

When changing behaviour, update the matching doc examples: they are tests. In the README, `pycon` blocks need a blank line before the closing fence or doctest treats the fence as expected output. Examples that can't be doctests (e.g. a process pool needs importable workers) live in `docs/examples/*.py`, are pulled in with `literalinclude`, and are run by `tests/docs/test_examples.py`, which also checks their output matches what the docs show. Show paths with `.as_posix()` so doctests are platform-independent.

`tests/dal/` includes multi-process tests (`ProcessPoolExecutor`); worker functions must be module-level.

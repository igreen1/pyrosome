# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pyrosome` is versioned, append-only on-disk storage for polars data frames (parquet via `ParquetDal`, CSV via `CsvDal`), safe for several processes sharing one data directory. Python 3.14+ (uses PEP 695 generics, deferred annotations). Managed with uv; the package lives in `src/pyrosome/`. Published to PyPI as `pyrosome` by the `publish-pypi` job in `.github/workflows/ci.yml` (trusted publishing). MIT licensed.

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

- `dal.py`: the `Dal` protocol (public API: `read`, `write`, `append`, `available_profiles`), `_FileDal` (abstract `Dal` implementation holding the read/write/append logic), `ParquetDal` and `CsvDal` (subclass `_FileDal`, defining only `_sink`/`_scan` and `extension`), `get_accessor` (file ending → class), and the errors `ProfileNotFoundError` and `SupersededAppendError`. It re-exports `DataManifestFile` and `ProfileChangedError` (listed in its `__all__`), which are documented as `pyrosome.dal.<name>`.
- `_maintenance.py`: the `Maintenance` mixin that `_FileDal` inherits, holding `compact`, `prune` and `vacuum`. It reaches `_FileDal` only through the abstract `get_profile_dir`, `_require_latest_data` and `_rewrite_version`. It exists to keep `dal.py` under pylint's 1000-line module limit.
- `_manifest.py`: `DataManifestFile` (read/commit the manifest, `reserve_new_data_dir`, `lock()`) and `discard_version`/`discarded_on_failure`, which delete a reserved version unless the manifest already points at it.
- `_files.py`: plain file operations: Windows in-use retries (`retry_while_in_use`), tmp files (`temporary_file`, `remove_tmp_file`), read-only handling (`make_read_only`, `remove_tree`), claiming (`claim_first_free`, `claim_marker`, `publish_append_file`) and listing (`data_files`, `version_dirs`).
- `_keys.py`: key validation (`check_key`), the `vN`/manifest names, and `find_profile_dirs`, the walk behind `available_profiles` (which then keeps only profiles whose latest version has files of the DAL's format).
- `_schema.py`: `check_appendable`, which `append` uses to reject column sets or types that don't cast losslessly to the existing data, and `promote_null_columns`, which every new version (write, first append, compact) applies so all-null `Null` columns are stored as `String`. A `Null` column can't be fixed at append time instead: polars can't scan a version whose first file has `Null` and a later file has `String`.
- `simple_file_lock.py`: `SimpleWriteLock`, an advisory cross-process lock implemented as a `<file>.write_lock` sibling file, with timeout/poll retry. The lock file holds JSON naming host, pid, `created` (UTC ISO time) and a unique token. With `stale_after` (the manifest uses `LOCK_STALE_AFTER` = 120 s; the wait `LOCK_TIMEOUT` stays 10 s), an older lock is broken: it is renamed to a hidden `.<name>.<uuid>.stale` file, and if the renamed file isn't the one judged stale (another process broke and retook it meanwhile) it is put back. Locks without a readable `created` fall back to the file's mtime. A waiter reads the lock file only when the lock it last saw could have become stale (normally once per acquire), because on Windows an open reader stops the holder deleting it; the lock file's read, rename and delete also go through `retry_while_in_use`. Release only deletes the lock file if it still holds this lock's token. On Windows (`_PERMISSION_ERROR_MEANS_HELD`), a `PermissionError` on create means "held" until the timeout, then raises `FileLockPermissionError`.

Public names (`SimpleIo`, `Dataset`, `Dal`, `ParquetDal`, `CsvDal`, `get_accessor`) are re-exported from `pyrosome/__init__.py`. Docs, examples and docstrings teach `from pyrosome import ParquetDal`, and `docs/api.rst` documents them as `pyrosome.<name>` (cross-reference them that way); rarely needed internals such as `DataManifestFile` and the lock are documented in their submodules. Library code imports from the submodules, never from `pyrosome`, to avoid circular imports.

On-disk layout: `data_directory/<keys...>/` is a *profile* containing `manifest.json` (`{"latest_data_dir": "vN"}`) and one `vN/` dir per version holding `0.<ext>`, `1.<ext>`, … Keys are path components and must resolve inside `data_directory`. Keys are case sensitive on every platform: on case-insensitive file systems (Windows, macOS) `_keys.matches_case_on_disk` makes a key differing only in case from an existing directory read as missing and refuse to write (it lists a parent directory only when the swapped-case name also exists, so case-sensitive file systems pay one extra `stat` per component).

The concurrency design is the non-obvious part, spread across these modules:
- `write` claims `vN` with atomic `mkdir(exist_ok=False)` (`reserve_new_data_dir`), sinks to a hidden `.<uuid>.<ext>.tmp`, renames to `0.<ext>`, and only then calls `commit_data_dir`, which updates the manifest under the lock and never moves it backwards (it returns whether the manifest now points at the version). `commit_data_dir` refuses a version directory that no longer exists, so `vacuum` (which deletes uncommitted versions while holding the lock) can't leave the manifest pointing at nothing.
- `append` sinks to a tmp file, claims the next free index by exclusively creating a hidden `.<idx>.<ext>.claim` marker (`touch(exist_ok=False)`), then `replace`s the tmp file to `<idx>.<ext>`. Markers are never deleted: deleting one would let a later append re-claim the index and silently replace published data (a plain rename overwrites on POSIX). Hard links aren't used because many SMB shares don't support them.
- After publishing, `append` re-reads the manifest. If a newer version was committed meanwhile, it raises `SupersededAppendError` (logged via `logged_failure`) rather than reporting success for data readers can't see. This was the user's chosen "minimal fix": appends take no lock.
- An append to a profile with no manifest claims `v0` itself with `mkdir(exist_ok=False)` and writes it. Concurrent first appends that lose that race poll (up to `FIRST_COMMIT_TIMEOUT`, 120 s) for the manifest and then append to `v0`, raising `TimeoutError` if it never appears. If the profile has a manifest but no files of this format, `append` still falls back to `write`.
- `compact` reserves a new version and commits it with `only_if`: under the manifest lock it checks the source version is still latest and lists exactly the files it read. Otherwise it discards the new version and retries, up to `COMPACT_ATTEMPTS` times, then raises `ProfileChangedError`. A narrow race remains: an append that publishes and checks the manifest between that listing and the manifest replace returns normally but stays in the old version (documented in concepts.rst).
- Cleanup after a raise: tmp files are removed in `temporary_file`'s `finally`, and a failed write/first append deletes its reserved version (`discarded_on_failure`) unless the manifest already points at it. Cleanup errors are logged, not raised, so they never replace the original exception. Killed processes are out of scope; `vacuum` handles what they leave, using an age threshold (`older_than`) because it can't tell abandoned files from in-progress ones.
- Data files are made read only before being moved into place; `remove_tmp_file` and `remove_tree` restore write permission before deleting because Windows can't delete read-only files.
- The manifest is replaced atomically (tmp + `replace`), so readers take no lock. Windows refuses to replace (and sometimes to open) a file another process has open, so both the replace and the read go through `_files.retry_while_in_use`, which retries a `PermissionError` for up to `IN_USE_TIMEOUT` (5 s) on Windows only.
- Readers only see files whose name is `<digits>.<extension>` (`_files.data_files`, built from the DAL's `extension`), so adding a format only means subclassing `_FileDal`.

Logging: each module has `_logger = logging.getLogger(__name__)` and uses lazy `%`-style arguments (no f-strings). The exceptions are `_files.py`, `_manifest.py` and `_maintenance.py`, which log as `pyrosome.dal` because the docs tell users that logger covers storage. Levels: `DEBUG` for steps, `INFO` for completed writes/appends/compactions/prunes/vacuums/cached profiles and superseded commits, `WARNING` for a broken stale lock, `ERROR` for problems with stored data (corrupt manifest, missing version dir, lock timeout). Failures that leave something behind (write, append, `_calc_<profile>`, lock release) are logged with traceback via `pyrosome._logging.logged_failure` (a context manager, which also sidesteps pylint's one-statement `try`) and re-raised. Caller errors (bad keys, mismatched columns) only raise, never log. `pyrosome/_logging.py` adds only a `NullHandler` to the `pyrosome` logger, so nothing reaches stderr via `logging.lastResort`; users enable output via the `pyrosome` logger (see the "Debug with logging" recipe).

Known limitation: `ParquetDal` and `CsvDal` share the manifest format, so mixing formats in one profile hides the other format's data.

## Tests and doctests

`uv run pytest` collects `tests/`, doctests in `src/` modules, `docs/*.rst` and `README.md` (configured in `[tool.pytest]`). `conftest.py`'s `run_in_tmp_path` fixture (applied via `usefixtures`, not autouse) chdirs every test into a fresh tmp dir, so examples use relative paths like `Path("data")` and must be self-contained (explicit imports, no injected globals).

When changing behaviour, update the matching doc examples: they are tests. In the README, `pycon` blocks need a blank line before the closing fence or doctest treats the fence as expected output. Examples that can't be doctests (e.g. a process pool needs importable workers) live in `docs/examples/*.py`, are pulled in with `literalinclude`, and are run by `tests/docs/test_examples.py`, which also checks their output matches what the docs show. Show paths with `.as_posix()` so doctests are platform-independent.

`tests/dal/` includes multi-process tests (`ProcessPoolExecutor`); worker functions must be module-level.

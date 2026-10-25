# salp

Versioned, append-only storage for [polars](https://pola.rs) data frames.

salp stores data frames on disk as parquet (or CSV) files, keeps every version you write, and reads them back as lazy frames. Several processes can read, write and append to the same data at once without overwriting each other or seeing half-written data.

## Purpose

Saving data frames to files is easy. Keeping those files trustworthy is harder:

- overwriting a file loses the previous data, and a crash halfway through can leave it corrupt;
- two processes writing at once can clobber each other;
- a reader can pick up a file that is still being written.

salp handles this with a small, predictable layout on the filesystem. There is no database or server to run.

## Project summary

- **Versioned**: every `write` creates a new version (`v0`, `v1`, …). Older versions stay on disk until you remove them, so rollbacks and audits are simple.
- **Append-only**: `append` adds a new file to the latest version and never rewrites existing files. Columns are reordered and cast to match the stored data.
- **Safe with several processes**: versions are claimed with an atomic `mkdir`, appends claim their index by exclusively creating a marker file so they never overwrite each other, and the manifest naming the latest version is replaced in one step.
- **Lazy**: `read` returns a `polars.LazyFrame`, so filters and column selections are pushed down into the file scan.
- **Parquet or CSV**: `Dal` stores parquet; `CsvDal` has the same API and stores CSV for people and tools that need to open the files directly.

Data is organised into *profiles*, named by one or more keys:

```text
data/
└── sales/
    └── orders/                # profile for keys ("sales", "orders")
        ├── manifest.json      # {"latest_data_dir": "v1"}
        ├── v0/
        │   ├── 0.parquet      # written by write()
        │   └── 1.parquet      # added by append()
        └── v1/
            └── 0.parquet
```

## Platform support

salp supports Windows, Linux and macOS, with the data directory on a local disk or an NFS share (NFSv3 or later).

SMB shares are supported as a lower priority. salp aims to work on them, but they get less attention and testing, and some SMB setups have limitations. For example, they may ignore the read-only permission salp sets on finished data files.

## Install

salp requires Python 3.15 or newer. It is not published to PyPI, so install it from a local checkout.

Add it to another uv project:

```console
uv add /path/to/salp
```

Or install it into the current environment with pip:

```console
pip install /path/to/salp
```

### Development setup

```console
uv sync --all-groups
```

This installs the development tools (pytest, ruff, ty, mypy, pylint) and the documentation tools (Sphinx, furo).

| Task | Command |
| --- | --- |
| Run tests, including every example in the docs and this README | `uv run pytest` |
| Lint and format | `uv run ruff check` and `uv run ruff format` |
| Type check | `uv run ty check` and `uv run mypy src` |
| Pylint | `uv run pylint src` |
| Build docs | `uv run sphinx-build -W -b html docs docs/_build/html` |

## Quickstart

`write` creates a new version and returns its directory:

```pycon
>>> from pathlib import Path
>>> import polars as pl
>>> from salp.dal.dal import Dal
>>> dal = Dal(Path("data"))
>>> orders = pl.DataFrame(
...     {"order_id": [1, 2, 3], "region": ["eu", "eu", "us"], "total": [9.5, 20.0, 7.25]}
... )
>>> dal.write(orders, "sales", "orders").as_posix()
'data/sales/orders/v0'

```

`append` adds a file to the latest version. Column order doesn't matter:

```pycon
>>> more = pl.DataFrame({"total": [12.0], "order_id": [4], "region": ["us"]})
>>> dal.append(more, "sales", "orders").as_posix()
'data/sales/orders/v0/1.parquet'

```

`read` returns a `LazyFrame` over the latest version:

```pycon
>>> (
...     dal.read("sales", "orders")
...     .filter(pl.col("region") == "eu")
...     .select(pl.col("total").sum())
...     .collect()
...     .item()
... )
29.5

```

Writing again creates `v1`, and `read` now returns only the new data. The earlier version stays on disk:

```pycon
>>> _ = dal.write(pl.DataFrame({"order_id": [10], "status": ["open"]}), "sales", "orders")
>>> dal.read("sales", "orders").collect().columns
['order_id', 'status']
>>> sorted(path.as_posix() for path in Path("data").rglob("[!.]*.*"))
['data/sales/orders/manifest.json', 'data/sales/orders/v0/0.parquet', 'data/sales/orders/v0/1.parquet', 'data/sales/orders/v1/0.parquet']

```

To store CSV instead, use `CsvDal` with the same methods. Keep each format in its own data directory, because a CSV write to a parquet profile hides the parquet data from `Dal`:

```pycon
>>> from salp.dal.dal import CsvDal
>>> CsvDal(Path("exports")).write(orders, "sales", "orders").as_posix()
'exports/sales/orders/v0'

```

`read` raises `FileNotFoundError` for a profile with no data, and `append` raises `ValueError` if the columns don't match the stored data:

```pycon
>>> dal.read("sales", "returns")
Traceback (most recent call last):
...
FileNotFoundError: Cannot find files for ('sales', 'returns')

```

## Documentation

The full documentation is in `docs/` and covers concepts, 14 recipes (compaction, rollbacks, pruning old versions, multi-process ingest and more), and the API reference. Build it with:

```console
uv run sphinx-build -W -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`.

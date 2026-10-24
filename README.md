# salp

Versioned, append-only storage for [polars](https://pola.rs) data frames.

salp stores data frames on disk as parquet (or CSV) files, keeps every version you write, and reads them back as lazy frames. Several processes can read, write and append to the same data at once without overwriting each other or seeing half-written data.

## Purpose

Saving data frames to files is easy. Keeping those files trustworthy is harder:

- overwriting a file loses the previous data, and a crash halfway through can leave it corrupt;
- two processes writing at once can clobber each other;
- a reader can pick up a file that is still being written.

salp handles this with a small, predictable layout on the local filesystem. There is no database or server to run.

## Project summary

- **Versioned**: every `write` creates a new version (`v0`, `v1`, …). Older versions stay on disk until you remove them, so rollbacks and audits are simple.
- **Append-only**: `append` adds a new file to the latest version and never rewrites existing files. Columns are reordered and cast to match the stored data.
- **Safe with several processes**: versions are claimed with an atomic `mkdir`, appends use hard links that fail rather than overwrite, and the manifest naming the latest version is replaced in one step.
- **Lazy**: `read` returns a `polars.LazyFrame`, so filters and column selections are pushed down into the file scan.
- **Parquet or CSV**: `DAL` stores parquet; `CsvDal` has the same API and stores CSV for people and tools that need to open the files directly.

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
| Run tests | `uv run pytest` |
| Lint and format | `uv run ruff check` and `uv run ruff format` |
| Type check | `uv run ty check` and `uv run mypy src` |
| Pylint | `uv run pylint src` |
| Build docs | `uv run sphinx-build -W -b html docs docs/_build/html` |
| Test docs examples | `uv run sphinx-build -W -b doctest docs docs/_build/doctest` |

## Quickstart

```python
from pathlib import Path

import polars as pl

from salp.dal.dal import DAL

dal = DAL(Path("data"))

# write creates a new version and returns its directory
orders = pl.DataFrame(
    {"order_id": [1, 2, 3], "region": ["eu", "eu", "us"], "total": [9.5, 20.0, 7.25]}
)
dal.write(orders, "sales", "orders")  # data/sales/orders/v0

# append adds a file to the latest version; column order doesn't matter
more = pl.DataFrame({"total": [12.0], "order_id": [4], "region": ["us"]})
dal.append(more, "sales", "orders")  # data/sales/orders/v0/1.parquet

# read returns a LazyFrame over the latest version
eu_total = (
    dal.read("sales", "orders")
    .filter(pl.col("region") == "eu")
    .select(pl.col("total").sum())
    .collect()
    .item()
)
print(eu_total)  # 29.5

# writing again creates v1; read now returns only the new data
dal.write(pl.DataFrame({"order_id": [10], "status": ["open"]}), "sales", "orders")
print(dal.read("sales", "orders").collect().columns)  # ['order_id', 'status']
```

To store CSV instead, use `CsvDal` with the same methods. Keep each format in its own data directory, because a CSV write to a parquet profile hides the parquet data from `DAL`.

```python
from salp.dal.dal import CsvDal

CsvDal(Path("exports")).write(orders, "sales", "orders")  # exports/sales/orders/v0/0.csv
```

`read` raises `FileNotFoundError` for a profile with no data, and `append` raises `ValueError` if the columns don't match the stored data.

## Documentation

The full documentation is in `docs/` and covers concepts, 14 recipes (compaction, rollbacks, pruning old versions, multi-process ingest and more), and the API reference. Build it with:

```console
uv run sphinx-build -W -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`.

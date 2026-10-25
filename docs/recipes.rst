Recipes
=======

Short solutions to common tasks. The recipes build on each other and start from:

.. doctest::

   >>> from pathlib import Path
   >>> import polars as pl
   >>> from salp import ParquetDal
   >>> data_dir = Path("data")
   >>> dal = ParquetDal(data_dir)

Keep a snapshot of a full refresh
---------------------------------

When a source is reloaded in full, write each load as a new version. Readers
always get the newest complete snapshot, and earlier ones stay on disk.

.. doctest::

   >>> for day, rows in [("2026-10-22", 3), ("2026-10-23", 5)]:
   ...     snapshot = pl.DataFrame({"day": [day] * rows, "n": range(rows)})
   ...     _ = dal.write(snapshot, "inventory")
   >>> dal.read("inventory").collect().height
   5

Ingest data in batches
----------------------

When new rows arrive over time, append each batch. Each batch becomes its own
file, so a failed batch never damages earlier ones.

.. doctest::

   >>> _ = dal.write(pl.DataFrame({"event": ["start"], "ms": [0]}), "events")
   >>> for batch in ([("click", 120)], [("click", 340), ("stop", 900)]):
   ...     _ = dal.append(pl.DataFrame(batch, schema=["event", "ms"], orient="row"), "events")
   >>> dal.read("events").collect()["event"].to_list()
   ['start', 'click', 'click', 'stop']

To start a profile that may not exist yet, `~salp.ParquetDal.append` falls back
to a write and warns. Write the first batch explicitly to avoid the warning.

Query without loading everything
--------------------------------

`~salp.ParquetDal.read` is lazy. Filter and select before ``collect`` so polars
only reads the rows and columns it needs.

.. doctest::

   >>> (
   ...     dal.read("events")
   ...     .filter(pl.col("event") == "click")
   ...     .select(pl.col("ms").mean())
   ...     .collect()
   ...     .item()
   ... )
   230.0

Filter by column values
-----------------------

For the common case of keeping rows where columns equal given values,
`~salp.SimpleIo` takes the filters as keyword arguments to
`~salp.SimpleIo.read`. It stores parquet, like `~salp.ParquetDal`,
so it reads the same data directory. ``None`` matches nulls, and a misspelt column
name raises straight away rather than at ``collect``.

.. doctest::

   >>> from salp import SimpleIo
   >>> io = SimpleIo(data_dir)
   >>> io.read("events", event="click").collect()["ms"].to_list()
   [120, 340]
   >>> io.read("events", evnt="click")
   Traceback (most recent call last):
   ...
   polars.exceptions.ColumnNotFoundError: No column named evnt in events

Floats can't be filtered on, because float equality is unreliable. Use
``filter`` with a tolerance for those.

`~salp.SimpleIo.write` and `~salp.SimpleIo.append` also take pandas
data frames. Named index levels, such as the keys of a ``groupby``, are stored as
columns, and an unnamed index is dropped. Converting pandas data needs the
``pandas`` extra, which installs pandas and pyarrow.

.. doctest::

   >>> import pandas as pd
   >>> views = pd.DataFrame({"page": ["home", "cart", "home"], "ms": [50, 70, 90]})
   >>> io.write(views[views["ms"] > 60], "views").name
   'v0'
   >>> io.read("views", page="home").collect().rows()
   [('home', 90)]

Group profiles into a dataset
-----------------------------

A `~salp.Dataset` reads the profiles below the same keys with the same filters,
so every profile in it must have the filtered columns.
`~salp.Dataset.get_subset` narrows it further.

.. doctest::

   >>> from salp import Dataset
   >>> orders = pl.DataFrame({"region": ["eu", "us", "eu"], "amount": [1, 2, 3]})
   >>> _ = io.write(orders, "shop", "orders")
   >>> eu = Dataset(io, "shop", region="eu")
   >>> eu.get_profile("orders").collect()["amount"].to_list()
   [1, 3]
   >>> eu.get_subset(amount=3).get_profile("orders").collect()["amount"].to_list()
   [3]

To derive a profile from others, subclass `~salp.Dataset` and add a
``_calc_<profile>`` method returning its data. The first
`~salp.Dataset.get_cache_profile` stores it, calculated from every row rather
than just the dataset's subset, and later calls read the stored copy. It isn't
updated when the profiles it's calculated from change: pass ``refresh=True`` to
recalculate it.

.. doctest::

   >>> class Shop(Dataset):
   ...     def _calc_totals(self):
   ...         orders = self.get_profile("orders")
   ...         return orders.group_by("region").agg(pl.col("amount").sum())
   >>> eu = Shop(io, "shop", region="eu")
   >>> eu.get_cache_profile("totals").collect().rows()
   [('eu', 4)]
   >>> _ = io.append(pl.DataFrame({"region": ["eu"], "amount": [10]}), "shop", "orders")
   >>> eu.get_cache_profile("totals").collect().rows()
   [('eu', 4)]
   >>> eu.get_cache_profile("totals", refresh=True).collect().rows()
   [('eu', 14)]

Read data that may not exist yet
--------------------------------

`~salp.ParquetDal.read` raises `FileNotFoundError` when a profile has no data.

.. doctest::

   >>> def read_or_empty(dal, schema, *keys):
   ...     try:
   ...         return dal.read(*keys)
   ...     except FileNotFoundError:
   ...         return pl.LazyFrame(schema=schema)
   >>> read_or_empty(dal, {"user": pl.String}, "users").collect().shape
   (0, 1)

.. _recipe-combine-profiles:

Combine several profiles
------------------------

Use keys to split data, for example by region, then concatenate the lazy frames
to query them together.

.. doctest::

   >>> for region, total in [("eu", 10.0), ("us", 4.5)]:
   ...     _ = dal.write(pl.DataFrame({"total": [total]}), "sales", region)
   >>> combined = pl.concat(
   ...     dal.read("sales", region).with_columns(region=pl.lit(region))
   ...     for region in ["eu", "us"]
   ... )
   >>> combined.collect().sort("region")["region"].to_list()
   ['eu', 'us']

Compact many small appends
--------------------------

Many small appended files make reads slower. Rewrite the profile as a new
version with one file. The lazy read streams straight into the new version.

.. doctest::

   >>> compacted = dal.write(dal.read("events"), "events")
   >>> sorted(path.name for path in compacted.iterdir())
   ['0.parquet']
   >>> dal.read("events").collect().height
   4

Change the schema
-----------------

`~salp.ParquetDal.append` rejects data with different columns. To add a column,
write a new version that includes it.

.. doctest::

   >>> try:
   ...     dal.append(pl.DataFrame({"event": ["start"], "ms": [5], "user": ["ana"]}), "events")
   ... except ValueError as error:
   ...     print("rejected")
   rejected
   >>> widened = dal.read("events").with_columns(user=pl.lit(None, dtype=pl.String))
   >>> _ = dal.write(widened, "events")
   >>> _ = dal.append(pl.DataFrame({"event": ["start"], "ms": [5], "user": ["ana"]}), "events")
   >>> dal.read("events").collect().columns
   ['event', 'ms', 'user']

Read an older version
---------------------

Old versions are plain directories, so scan them directly.

.. doctest::

   >>> profile_dir = data_dir / "inventory"
   >>> pl.scan_parquet(profile_dir / "v0" / "*.parquet").collect().height
   3

Roll back to an older version
-----------------------------

Write the old version again as a new version. History is kept, and the rollback
can itself be undone.

.. doctest::

   >>> _ = dal.write(pl.scan_parquet(profile_dir / "v0" / "*.parquet"), "inventory")
   >>> dal.read("inventory").collect().height
   3

.. _recipe-prune:

Remove old versions
-------------------

Delete version directories older than the last few. Only do this when no reader
can still be using them, since a lazy frame from an earlier
`~salp.ParquetDal.read` fails if its files disappear.

.. doctest::

   >>> import shutil
   >>> import stat
   >>> from salp.dal import DataManifestFile
   >>> def remove_read_only(remove, path, _error):
   ...     # data files are read only, and Windows can't delete read-only files
   ...     Path(path).chmod(stat.S_IWRITE)
   ...     remove(path)
   >>> def prune(dal, *keys, keep=2):
   ...     profile_dir = dal.data_directory.joinpath(*keys)
   ...     latest = DataManifestFile(profile_dir).get_latest_data_dir()
   ...     oldest_kept = int(latest.name.removeprefix("v")) - keep + 1
   ...     for version in profile_dir.glob("v*"):
   ...         index = version.name.removeprefix("v")
   ...         if version.is_dir() and index.isdigit() and int(index) < oldest_kept:
   ...             shutil.rmtree(version, onexc=remove_read_only)
   >>> prune(dal, "inventory", keep=2)
   >>> sorted(path.name for path in profile_dir.iterdir())
   ['manifest.json', 'v1', 'v2']

Versions newer than the manifest's latest are kept, because another process may
still be writing them.

.. _recipe-parallel:

Write from several processes
----------------------------

Writes and appends are safe from several processes at once, so a process pool
can append batches to the same profile directly.

Worker functions must be importable by the pool, so this recipe is a script
rather than an interactive session:

.. literalinclude:: examples/parallel_ingest.py
   :caption: docs/examples/parallel_ingest.py

Running it prints:

.. code-block:: text

   99 batches appended as 99 files, 100 rows

Exchange data as CSV
--------------------

Convert between formats by reading with one class and writing with the other.
Keep each format in its own data directory; see the limitations in
:doc:`concepts`.

.. doctest::

   >>> from salp import CsvDal
   >>> csv_dal = CsvDal(data_dir / "export")
   >>> export_dir = csv_dal.write(dal.read("sales", "eu"), "sales", "eu")
   >>> (export_dir / "0.csv").read_text().splitlines()
   ['total', '10.0']
   >>> _ = dal.write(csv_dal.read("sales", "eu"), "sales", "eu_from_csv")

.. _recipe-stale-lock:

Recover from a crashed writer
-----------------------------

If a process dies while updating a manifest, its lock file stays behind and
later writes raise `~salp.simple_file_lock.FileAlreadyLockedError` after
`~salp.dal.DataManifestFile.LOCK_TIMEOUT` seconds. Once you are sure no
writer is running, remove the lock file.

.. doctest::

   >>> from salp.simple_file_lock import file_is_locked
   >>> manifest_path = data_dir / "events" / "manifest.json"
   >>> lock_file = manifest_path.with_name("manifest.json.write_lock")
   >>> lock_file.touch()  # simulate a crashed writer
   >>> file_is_locked(manifest_path)
   True
   >>> lock_file.unlink(missing_ok=True)
   >>> file_is_locked(manifest_path)
   False

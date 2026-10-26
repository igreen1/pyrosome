Concepts
========

The examples on this page build on each other and start from:

.. doctest::

   >>> import json
   >>> from pathlib import Path
   >>> import polars as pl
   >>> from salpa import ParquetDal, CsvDal
   >>> data_dir = Path("data")
   >>> dal = ParquetDal(data_dir)

On-disk layout
--------------

Each profile is a directory under the data directory. It holds one directory per
version and a manifest that names the latest one:

.. code-block:: text

   data_directory/
   └── sales/
       └── orders/                  # profile for keys ("sales", "orders")
           ├── manifest.json        # {"latest_data_dir": "v1"}
           ├── v0/
           │   ├── 0.parquet        # written by write()
           │   ├── 1.parquet        # added by append()
           │   └── .1.parquet.claim # marks index 1 as taken
           └── v1/
               └── 0.parquet

* `~salpa.ParquetDal.write` creates the next ``vN`` directory, writes
  ``0.parquet`` into it, then points the manifest at it.
* `~salpa.ParquetDal.append` adds ``1.parquet``, ``2.parquet``, and so on to the
  version the manifest points at, plus a hidden ``.claim`` marker for each.
* `~salpa.ParquetDal.read` scans every numbered data file in that version, in
  index order. Other files, such as hidden ``.tmp`` and ``.claim`` files, are
  ignored.

Data files are read only once written, so existing data cannot be changed by
accident.

This write, append, write sequence produces exactly that layout:

.. doctest::

   >>> _ = dal.write(pl.DataFrame({"id": [1]}), "sales", "orders")
   >>> _ = dal.append(pl.DataFrame({"id": [2]}), "sales", "orders")
   >>> _ = dal.write(pl.DataFrame({"id": [3]}), "sales", "orders")
   >>> profile_dir = data_dir / "sales" / "orders"
   >>> sorted(path.relative_to(profile_dir).as_posix() for path in profile_dir.rglob("*"))
   ['manifest.json', 'v0', 'v0/.1.parquet.claim', 'v0/0.parquet', 'v0/1.parquet', 'v1', 'v1/0.parquet']
   >>> json.loads((profile_dir / "manifest.json").read_text())
   {'latest_data_dir': 'v1'}
   >>> _ = (profile_dir / "v1" / ".half-written.parquet.tmp").write_bytes(b"")
   >>> dal.read("sales", "orders").collect()["id"].to_list()
   [3]

Keys
----

Keys are path components, so ``dal.write(df, "sales", "orders")`` and
``dal.write(df, "sales/orders")`` name the same profile. At least one key is
required, and keys that resolve outside the data directory, such as ``".."``,
raise `ValueError`.

.. doctest::

   >>> dal.read("sales/orders").collect()["id"].to_list()
   [3]
   >>> dal.read()
   Traceback (most recent call last):
   ...
   ValueError: At least one key is required
   >>> dal.read("..")  # doctest: +IGNORE_EXCEPTION_DETAIL
   Traceback (most recent call last):
   ...
   ValueError: Profile directory data/.. is not within data directory data

A profile is only the directory that holds a manifest. ``dal.read("sales")`` does
not combine the profiles below ``sales/``; see :ref:`recipe-combine-profiles`.

.. doctest::

   >>> dal.read("sales")
   Traceback (most recent call last):
   ...
   FileNotFoundError: Cannot find files for ('sales',)

Versions
--------

Versions are never modified by `~salpa.ParquetDal.write`. That makes rollbacks and
audits simple, but the data directory grows with every write until old versions
are removed. See :ref:`recipe-prune`.

Each version can have its own schema. Within a version, every appended file must
have the same column names as the existing data.

.. doctest::

   >>> dal.append(pl.DataFrame({"name": ["x"]}), "sales", "orders")
   Traceback (most recent call last):
   ...
   ValueError: Columns ['name'] do not match existing columns ['id'] for ('sales', 'orders')

Safety with several processes
-----------------------------

salpa is designed for several processes sharing one data directory, on a local
disk or an NFS share, from Windows, Linux or macOS. SMB shares are supported as
a lower priority:

* **Writers never share a version.** Each write claims its ``vN`` directory with
  an atomic ``mkdir``.
* **Appends never overwrite each other.** Each append writes to a temporary
  file, claims the next free index by exclusively creating a hidden
  ``.<index>.<ext>.claim`` marker, and only then moves the file into place.
  Markers are never deleted, so an index is only ever claimed once. A marker
  without a data file, left by an append that failed, just leaves a gap in the
  indices.
* **Readers never see partial data.** Data files and the manifest are written to
  temporary files and moved into place in one step. The manifest only moves to a
  new version once its data is complete.
* **The manifest never moves backwards.** If two writes finish out of order, the
  manifest keeps pointing at the higher version.

Manifest updates take a short lock, the ``manifest.json.write_lock`` file. A
writer waits up to `~salpa.dal.DataManifestFile.LOCK_TIMEOUT` seconds for it
before raising `~salpa.simple_file_lock.FileAlreadyLockedError`.

These guarantees are covered by the multi-process tests in ``tests/``, and
:ref:`recipe-parallel` shows them in use.

Limitations
-----------

* **Stale locks are not cleaned up.** If a process is killed while holding the
  manifest lock, writes to that profile fail until the lock file is deleted. See
  :ref:`recipe-stale-lock`.
* **Exclusive file creation is required** for `~salpa.ParquetDal.append`. Local
  disks, SMB and NFSv3 or later support it; NFSv2 does not.
* **Don't mix formats in one profile.** `~salpa.ParquetDal` and
  `~salpa.CsvDal` share the manifest format, so a CSV write to a profile
  hides its parquet data from `~salpa.ParquetDal` and the other way round. Use a
  separate data directory, or separate keys, for each format.

  .. doctest::

     >>> _ = CsvDal(data_dir).write(pl.DataFrame({"id": [4]}), "sales", "orders")
     >>> dal.read("sales", "orders")
     Traceback (most recent call last):
     ...
     FileNotFoundError: Cannot find files for ('sales', 'orders')

* **CSV types are inferred.** `~salpa.CsvDal` stores no schema, so types
  such as dates or small integers come back as strings or ``Int64``.

  .. doctest::

     >>> import datetime
     >>> typed = pl.DataFrame(
     ...     {"day": [datetime.date(2026, 10, 24)], "n": pl.Series([1], dtype=pl.Int8)}
     ... )
     >>> csv_dal = CsvDal(Path("exports"))
     >>> _ = csv_dal.write(typed, "typed")
     >>> csv_dal.read("typed").collect_schema()
     Schema({'day': String, 'n': Int64})

* **Appends into an old version are not possible.** `~salpa.ParquetDal.append`
  always targets the latest version.

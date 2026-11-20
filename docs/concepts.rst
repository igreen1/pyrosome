Concepts
========

The examples on this page build on each other and start from:

.. doctest::

   >>> import json
   >>> from pathlib import Path
   >>> import polars as pl
   >>> from pyrosome import ParquetDal, CsvDal
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

* `~pyrosome.ParquetDal.write` creates the next ``vN`` directory, writes
  ``0.parquet`` into it, then points the manifest at it.
* `~pyrosome.ParquetDal.append` adds ``1.parquet``, ``2.parquet``, and so on to the
  version the manifest points at, plus a hidden ``.claim`` marker for each.
* `~pyrosome.ParquetDal.read` scans every numbered data file in that version, in
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
required. Keys that are empty, ``"."`` or ``".."``, hidden (starting with
``"."``), contain ``":"`` or NUL, or name a version (``"v0"``) or
``"manifest.json"`` would collide with the files pyrosome stores or leave the data
directory, so they raise `ValueError`.

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
   ValueError: Invalid key '..': empty, '.' and '..' path components are not allowed

A profile is only the directory that holds a manifest. ``dal.read("sales")`` does
not combine the profiles below ``sales/``; see :ref:`recipe-combine-profiles`.

.. doctest::

   >>> dal.read("sales")
   Traceback (most recent call last):
   ...
   pyrosome.dal.ProfileNotFoundError: Cannot find files for ('sales',)

Versions
--------

Versions are never modified by `~pyrosome.ParquetDal.write`. That makes rollbacks and
audits simple, but the data directory grows with every write until old versions
are removed with `~pyrosome.ParquetDal.prune`. See :ref:`recipe-prune`.

Each version can have its own schema. Within a version, every appended file must
have the same column names as the existing data, and column types that cast to the
existing types without losing values (an integer to a wider integer, say). A float
appended to an integer column raises `ValueError` instead of being truncated.

.. doctest::

   >>> dal.append(pl.DataFrame({"name": ["x"]}), "sales", "orders")
   Traceback (most recent call last):
   ...
   ValueError: Columns ['name'] do not match existing columns ['id'] for ('sales', 'orders')

A column whose values are all null has type ``Null``, which can't hold anything
else. Writes store such columns as ``String`` instead, so later appends can fill
them in:

.. doctest::

   >>> _ = dal.write(pl.DataFrame({"id": [5], "note": [None]}), "sales", "notes")
   >>> dal.read("sales", "notes").collect_schema()
   Schema({'id': Int64, 'note': String})
   >>> _ = dal.append(pl.DataFrame({"id": [6], "note": ["late"]}), "sales", "notes")
   >>> dal.read("sales", "notes").collect()["note"].to_list()
   [None, 'late']

Safety with several processes
-----------------------------

pyrosome is designed for several processes sharing one data directory, on a local
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
* **Appends never vanish silently.** An append adds its file to the version that
  was latest when it started. If another process commits a newer version before
  the file is in place, readers won't see it, so
  `~pyrosome.ParquetDal.append` raises `~pyrosome.dal.SupersededAppendError`
  instead of returning. Append the same data again to add it to the new version.
* **The first appends to a new profile all land in** ``v0``. One of them writes
  ``v0``, and the others wait up to
  `~pyrosome.ParquetDal.FIRST_COMMIT_TIMEOUT` seconds for it, then append to it.
* **Failures clean up after themselves.** If a write or append raises, its
  temporary file and, for a write, its uncommitted version are deleted. A process
  that is killed can't clean up; `~pyrosome.ParquetDal.vacuum` removes what it
  left behind.

Manifest updates take a short lock, the ``manifest.json.write_lock`` file, which
names the host and process holding it and when it was taken. A writer waits up
to `~pyrosome.dal.DataManifestFile.LOCK_TIMEOUT` seconds for it before raising
`~pyrosome.simple_file_lock.FileAlreadyLockedError`. Updating the manifest takes
milliseconds, so a lock older than
`~pyrosome.dal.DataManifestFile.LOCK_STALE_AFTER` seconds was left by a process
that died, and the next writer deletes it.

Compaction and rollback
~~~~~~~~~~~~~~~~~~~~~~~

Compacting a profile, or rolling it back to an older version, reads one version
and writes the result as a new version. Appends to the old version that finish
while that runs would be left behind in it:

* `~pyrosome.ParquetDal.compact` checks, while committing, that nothing was
  appended since it read the data. If something was, it discards its new version
  and starts again, so appends made while compacting are never lost.
* A rollback written with `~pyrosome.ParquetDal.write` (see
  :ref:`recipe-rollback`) discards on purpose the appends that finish before it
  commits. Appends that finish after it commits raise
  `~pyrosome.dal.SupersededAppendError`, so they can be retried against the
  rolled-back data.

This only applies to writes that rebuild a profile from its own data. A write of
new data, such as a full refresh, replaces the profile whatever was appended.

These guarantees are covered by the multi-process tests in ``tests/``, and
:ref:`recipe-parallel` shows them in use.

Limitations
-----------

* **Stale locks block writes for two minutes.** If a process is killed while
  holding the manifest lock, writes to that profile fail until the lock is
  `~pyrosome.dal.DataManifestFile.LOCK_STALE_AFTER` seconds old and is broken. A
  lock's age is measured with the clock of the host that took it, so hosts
  sharing a data directory need roughly synchronised clocks. See
  :ref:`recipe-stale-lock`.
* **A narrow race remains when compacting.** An append that publishes its file
  and checks the manifest within the few milliseconds between
  `~pyrosome.ParquetDal.compact` checking for appends and committing its new
  version returns normally, but its file stays in the old version.
* **Exclusive file creation is required** for `~pyrosome.ParquetDal.append`. Local
  disks, SMB and NFSv3 or later support it; NFSv2 does not.
* **Don't mix formats in one profile.** `~pyrosome.ParquetDal` and
  `~pyrosome.CsvDal` share the manifest format, so a CSV write to a profile
  hides its parquet data from `~pyrosome.ParquetDal` and the other way round. Use a
  separate data directory, or separate keys, for each format.

  .. doctest::

     >>> _ = CsvDal(data_dir).write(pl.DataFrame({"id": [4]}), "sales", "orders")
     >>> dal.read("sales", "orders")
     Traceback (most recent call last):
     ...
     pyrosome.dal.ProfileNotFoundError: Cannot find files for ('sales', 'orders')

* **CSV types are inferred.** `~pyrosome.CsvDal` stores no schema, so types
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

* **Appends into an old version are not possible.** `~pyrosome.ParquetDal.append`
  always targets the latest version.

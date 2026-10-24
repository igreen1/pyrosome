Concepts
========

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
           │   └── 1.parquet        # added by append()
           └── v1/
               └── 0.parquet

* `~salp.dal.dal.DAL.write` creates the next ``vN`` directory, writes
  ``0.parquet`` into it, then points the manifest at it.
* `~salp.dal.dal.DAL.append` adds ``1.parquet``, ``2.parquet``, and so on to the
  version the manifest points at.
* `~salp.dal.dal.DAL.read` scans every numbered data file in that version, in
  index order. Other files, such as hidden ``.tmp`` files, are ignored.

Keys
----

Keys are path components, so ``dal.write(df, "sales", "orders")`` and
``dal.write(df, "sales/orders")`` name the same profile. At least one key is
required, and keys that resolve outside the data directory, such as ``".."``,
raise `ValueError`.

A profile is only the directory that holds a manifest. ``dal.read("sales")`` does
not combine the profiles below ``sales/``; see :ref:`recipe-combine-profiles`.

Versions
--------

Versions are never modified by `~salp.dal.dal.DAL.write`. That makes rollbacks and
audits simple, but the data directory grows with every write until old versions
are removed. See :ref:`recipe-prune`.

Each version can have its own schema. Within a version, every appended file must
have the same column names as the existing data.

Safety with several processes
-----------------------------

salp is designed for several processes sharing one data directory on a local
filesystem:

* **Writers never share a version.** Each write claims its ``vN`` directory with
  an atomic ``mkdir``.
* **Appends never overwrite each other.** Each append writes to a temporary file
  and then hard-links it to the next free index, which fails rather than
  replacing an existing file.
* **Readers never see partial data.** Data files and the manifest are written to
  temporary files and moved into place in one step. The manifest only moves to a
  new version once its data is complete.
* **The manifest never moves backwards.** If two writes finish out of order, the
  manifest keeps pointing at the higher version.

Manifest updates take a short lock, the ``manifest.json.write_lock`` file. A
writer waits up to `~salp.dal.dal.DataManifestFile.LOCK_TIMEOUT` seconds for it
before raising `~salp.dal.simple_file_lock.FileAlreadyLockedError`.

Limitations
-----------

* **Stale locks are not cleaned up.** If a process is killed while holding the
  manifest lock, writes to that profile fail until the lock file is deleted. See
  :ref:`recipe-stale-lock`.
* **Hard links are required** for `~salp.dal.dal.DAL.append`. Most local
  filesystems support them; some network and FUSE filesystems do not.
* **Don't mix formats in one profile.** `~salp.dal.dal.DAL` and
  `~salp.dal.dal.CsvDal` share the manifest format, so a CSV write to a profile
  hides its parquet data from `~salp.dal.dal.DAL` and the other way round. Use a
  separate data directory, or separate keys, for each format.
* **CSV types are inferred.** `~salp.dal.dal.CsvDal` stores no schema, so types
  such as dates or small integers may come back as strings or ``Int64``.
* **Appends into an old version are not possible.** `~salp.dal.dal.DAL.append`
  always targets the latest version.

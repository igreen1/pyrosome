salp
====

salp stores `polars <https://pola.rs>`_ data frames on disk as versioned,
append-only datasets, and reads them back as lazy frames.

.. doctest::

   >>> dal = DAL(data_dir)
   >>> dal.write(pl.DataFrame({"city": ["Oslo", "Lima"], "temp": [4, 19]}), "weather")
   PosixPath('.../weather/v0')
   >>> dal.append(pl.DataFrame({"city": ["Pune"], "temp": [31]}), "weather")
   PosixPath('.../weather/v0/1.parquet')
   >>> dal.read("weather").filter(pl.col("temp") > 10).collect()["city"].to_list()
   ['Lima', 'Pune']

* **Versioned**: every `~salp.dal.dal.DAL.write` creates a new version. Old
  versions stay on disk until you remove them.
* **Append-only**: `~salp.dal.dal.DAL.append` adds new files and never rewrites
  existing ones.
* **Safe with several processes**: concurrent writes and appends never overwrite
  each other, and readers never see half-written data.
* **Lazy**: `~salp.dal.dal.DAL.read` returns a `polars.LazyFrame`, so filters
  and column selections are pushed down into the file scan.

.. toctree::
   :maxdepth: 2

   quickstart
   concepts
   recipes
   api

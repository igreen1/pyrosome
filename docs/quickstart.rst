Quickstart
==========

Install
-------

salp needs Python 3.15 or newer and polars. It is not published to PyPI, so
install it from a local checkout:

.. code-block:: console

   $ uv add /path/to/salp

Create a data access layer
--------------------------

A `~salp.dal.dal.Dal` is a handle on a directory. Creating one does not touch the
disk; directories are created on the first write.

.. doctest::

   >>> from pathlib import Path
   >>> import polars as pl
   >>> from salp.dal.dal import Dal
   >>> dal = Dal(Path("data"))

Write data
----------

Data is stored under a *profile*, named by one or more keys. Each key becomes a
directory level, so ``"sales", "eu"`` is stored in ``sales/eu/``.

.. doctest::

   >>> orders = pl.DataFrame(
   ...     {"order_id": [1, 2, 3], "region": ["eu", "eu", "us"], "total": [9.5, 20.0, 7.25]}
   ... )
   >>> dal.write(orders, "sales", "orders").name
   'v0'

`~salp.dal.dal.Dal.write` accepts a `polars.DataFrame` or a `polars.LazyFrame`
and returns the new version directory.

Read data
---------

`~salp.dal.dal.Dal.read` returns a `polars.LazyFrame` over the latest version.
Nothing is loaded until you call ``collect``.

.. doctest::

   >>> lazy = dal.read("sales", "orders")
   >>> lazy.filter(pl.col("region") == "eu").select("total").collect()["total"].sum()
   29.5

Append data
-----------

`~salp.dal.dal.Dal.append` adds a new file to the latest version. Columns may be
in any order and are cast to the stored types.

.. doctest::

   >>> more = pl.DataFrame({"total": [12.0], "order_id": [4], "region": ["us"]})
   >>> dal.append(more, "sales", "orders").name
   '1.parquet'
   >>> dal.read("sales", "orders").collect()["order_id"].to_list()
   [1, 2, 3, 4]

Write a new version
-------------------

Writing again replaces what `~salp.dal.dal.Dal.read` returns. The schema of a new
version does not have to match the old one.

.. doctest::

   >>> dal.write(pl.DataFrame({"order_id": [10], "status": ["open"]}), "sales", "orders").name
   'v1'
   >>> dal.read("sales", "orders").collect().columns
   ['order_id', 'status']

Use CSV instead of parquet
--------------------------

`~salp.dal.dal.CsvDal` has the same API but stores CSV files. Use it when people
or other tools need to open the files directly; prefer parquet otherwise.

.. doctest::

   >>> from salp.dal.dal import CsvDal
   >>> csv_dal = CsvDal(Path("exports"))
   >>> csv_dal.write(orders, "sales", "orders").as_posix()
   'exports/sales/orders/v0'
   >>> csv_dal.read("sales", "orders").collect().height
   3

Next, :doc:`concepts` explains how the data is laid out on disk, and
:doc:`recipes` shows common tasks.

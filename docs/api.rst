API reference
=============

Everything most code needs is importable from ``salpa`` itself, e.g.
``from salpa import ParquetDal``.

Data access
-----------

.. autoclass:: salpa.Dal

.. autoclass:: salpa.ParquetDal
   :inherited-members:

.. autoclass:: salpa.CsvDal
   :inherited-members:

.. autofunction:: salpa.get_accessor

Filtering reads
---------------

.. autoclass:: salpa.SimpleIo

.. autodata:: salpa.api.FilterValue

Datasets
--------

.. autoclass:: salpa.Dataset

Internals
---------

These live in submodules and are rarely needed directly.

.. automodule:: salpa.dal
   :no-members:

.. autoclass:: salpa.dal.DataManifestFile

.. automodule:: salpa.simple_file_lock

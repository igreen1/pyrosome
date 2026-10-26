API reference
=============

Everything most code needs is importable from ``pyrosome`` itself, e.g.
``from pyrosome import ParquetDal``.

Data access
-----------

.. autoclass:: pyrosome.Dal

.. autoclass:: pyrosome.ParquetDal
   :inherited-members:

.. autoclass:: pyrosome.CsvDal
   :inherited-members:

.. autofunction:: pyrosome.get_accessor

Filtering reads
---------------

.. autoclass:: pyrosome.SimpleIo

.. autodata:: pyrosome.api.FilterValue

Datasets
--------

.. autoclass:: pyrosome.Dataset

Internals
---------

These live in submodules and are rarely needed directly.

.. automodule:: pyrosome.dal
   :no-members:

.. autoclass:: pyrosome.dal.DataManifestFile

.. automodule:: pyrosome.simple_file_lock

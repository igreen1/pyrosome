API reference
=============

Everything most code needs is importable from ``salp`` itself, e.g.
``from salp import ParquetDal``.

Data access
-----------

.. autoclass:: salp.Dal

.. autoclass:: salp.ParquetDal
   :inherited-members:

.. autoclass:: salp.CsvDal
   :inherited-members:

.. autofunction:: salp.get_accessor

Filtering reads
---------------

.. autoclass:: salp.SimpleIo

.. autodata:: salp.api.FilterValue

Internals
---------

These live in submodules and are rarely needed directly.

.. automodule:: salp.dal
   :no-members:

.. autoclass:: salp.dal.DataManifestFile

.. automodule:: salp.simple_file_lock

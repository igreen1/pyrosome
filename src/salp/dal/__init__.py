"""Handles file io.

Doesn't actually read files - just gets a concrete list of handles.
"""

from salp.dal.dal import CsvDal, Dal, ParquetDal, get_accessor

__all__ = ["CsvDal", "Dal", "ParquetDal", "get_accessor"]

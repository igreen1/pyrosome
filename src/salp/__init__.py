"""File IO layer."""

from salp.api import SimpleIo
from salp.dal import CsvDal, Dal, ParquetDal, get_accessor

__all__ = [
    "CsvDal",
    "Dal",
    "ParquetDal",
    "SimpleIo",
    "get_accessor",
]

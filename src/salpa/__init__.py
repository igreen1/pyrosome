"""File IO layer."""

from salpa.api import SimpleIo
from salpa.dal import CsvDal, Dal, ParquetDal, get_accessor
from salpa.datasets import Dataset

__all__ = [
    "CsvDal",
    "Dal",
    "Dataset",
    "ParquetDal",
    "SimpleIo",
    "get_accessor",
]

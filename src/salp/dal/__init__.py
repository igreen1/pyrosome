"""Handles file io.

Doesn't actually read files - just gets a concrete list of handles.
"""

from salp.dal.dal import CsvDal, Dal, get_accessor

__all__ = ["CsvDal", "Dal", "get_accessor"]

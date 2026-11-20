"""User facing api to get data."""

import datetime as dt
import itertools
import logging
from typing import TYPE_CHECKING, cast

import polars as pl

from pyrosome.dal import get_accessor

if TYPE_CHECKING:
    import os
    from pathlib import Path

    import pandas as pd

    from pyrosome.dal import ParquetDal


_logger = logging.getLogger(__name__)

type FilterValue = str | int | bool | dt.date | None
"""A value `pyrosome.SimpleIo.read` can filter a column by.

`datetime.datetime` is a `datetime.date`, so both are accepted. Floats are not:
equality on floats is unreliable, since ``0.1 + 0.2 != 0.3``.
"""


def _equals(column: str, value: object) -> pl.Expr:
    """Make an expression that is true where column equals value.

    Args:
        column: The column to compare.
        value: The value the column must equal. None matches nulls.

    Returns:
        The filter expression.

    Raises:
        TypeError: If value is a float, or any other type `FilterValue` excludes.
    """
    match value:
        case None:
            return pl.col(column).is_null()
        case float():
            msg = (
                f"Cannot filter {column!r} by the float {value!r}, since float "
                "equality is unreliable. Filter the frame with a tolerance instead."
            )
            raise TypeError(msg)
        case str() | int() | dt.date():
            return pl.col(column) == value
        case _:
            msg = f"Cannot filter {column!r} by a {type(value).__name__}"
            raise TypeError(msg)


def _to_polars(
    data: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Convert a pandas data frame to polars, leaving polars frames as they are.

    Named index levels, such as the keys of a ``groupby``, become columns. An
    unnamed index, such as the row numbers left after filtering, is dropped.

    Args:
        data: The data to convert.

    Returns:
        The data as a polars frame.

    Raises:
        ImportError: If pyarrow is not installed and data is a pandas frame with a
            column that is not a plain numpy type, such as strings.
    """
    if isinstance(data, pl.DataFrame | pl.LazyFrame):
        return data
    # only the level names are read, so the element type is irrelevant
    index_names = cast("pd.Index[str]", data.index).names
    if any(name is not None for name in index_names):
        _logger.debug("Moving pandas index levels %s into columns", list(index_names))
        data = data.reset_index()
    _logger.debug("Converting a pandas frame of shape %s to polars", data.shape)
    return pl.from_pandas(data)


class SimpleIo:
    """Parquet storage whose `read` can also filter rows by column values.

    Example:
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from pyrosome import SimpleIo
        >>> io = SimpleIo(Path("data"))
        >>> df = pl.DataFrame({"x": [1, 2, 2], "y": ["a", "b", None]})
        >>> io.write(df, "example").name
        'v0'
        >>> io.read("example", x=2, y="b").collect().rows()
        [(2, 'b')]
        >>> io.read("example", y=None).collect().rows()
        [(2, None)]

        `write` and `append` also take pandas data frames:

        >>> import pandas as pd
        >>> io.append(pd.DataFrame({"x": [3], "y": ["d"]}), "example").name
        '1.parquet'
        >>> io.read("example", x=3).collect().rows()
        [(3, 'd')]
    """

    dal: ParquetDal
    """The data access layer reads and writes go through."""

    def __init__(self, data_directory: Path | str | os.PathLike[str]) -> None:
        """Create the inner dal. Nothing is read or written until used.

        Args:
            data_directory: The directory where data files are stored.

        Returns:
            None.
        """
        self.dal = get_accessor("parquet")(data_directory)

    def available_profiles(self, *keys: str) -> tuple[tuple[str, ...], ...]:
        """List the full keys of every readable profile at or below keys.

        Profiles at any depth are listed, but only those with parquet data. Keys
        are case sensitive on every platform. See
        `pyrosome.ParquetDal.available_profiles` for the details.

        Example:
            >>> from pathlib import Path
            >>> import polars as pl
            >>> from pyrosome import SimpleIo
            >>> io = SimpleIo(Path("data"))
            >>> df = pl.DataFrame({"x": [1]})
            >>> _ = io.write(df, "shop", "orders")
            >>> _ = io.write(df, "shop", "eu", "totals")
            >>> io.available_profiles("shop")
            (('shop', 'eu', 'totals'), ('shop', 'orders'))

        Args:
            *keys: Path components to look below. With no keys, every profile in
                the data directory. If keys name a profile, it is included.

        Returns:
            The full keys of each profile, as path components, sorted.

        Raises:
            ValueError: If a key is invalid or the keys lead outside the data
                directory.
            OSError: If a directory cannot be listed, for any reason other than
                not existing.
        """
        return self.dal.available_profiles(*keys)

    def read(self, /, *keys: str, **filters: FilterValue) -> pl.LazyFrame:
        """Read a lazy frame, keeping only rows where each column equals a value.

        Filters are equality only. Pass a column whose name isn't a valid keyword
        argument by unpacking a dict, e.g. ``io.read("sales", **{"unit price": 3})``.

        Args:
            *keys: Path components naming the profile.
            **filters: Column names and the value each column must equal. None
                matches nulls.

        Returns:
            A lazy scan over the latest version, keeping only rows that match every
            filter.

        Raises:
            polars.exceptions.ColumnNotFoundError: If a filter names a column the
                data doesn't have.
            TypeError: If a filter value is a float or another unsupported type.
            FileNotFoundError: If there is no data for keys.
            ValueError: If there are no keys, a key is invalid or leads outside the
                data directory, or the profile's manifest file is corrupt.
            polars.exceptions.PolarsError: If the data files cannot be scanned.
        """
        data = self.dal.read(*keys)
        if not filters:
            return data
        schema = data.collect_schema()
        if missing := [column for column in filters if column not in schema]:
            msg = f"No column named {', '.join(missing)} in {'/'.join(keys)}"
            raise pl.exceptions.ColumnNotFoundError(msg)
        _logger.debug("Filtering %s by %s", keys, filters)
        return data.filter(*itertools.starmap(_equals, filters.items()))

    def write(
        self, data: pl.DataFrame | pl.LazyFrame | pd.DataFrame, *keys: str
    ) -> Path:
        """Write the data as a new version.

        A pandas data frame's named index levels are stored as columns; an unnamed
        index is dropped.

        Args:
            data: The data to write, as a polars or pandas frame.
            *keys: Path components naming the profile.

        Returns:
            The versioned directory the data was written to.

        Raises:
            ImportError: If data is a pandas frame needing pyarrow to convert, and
                pyarrow is not installed.
            ValueError: If there are no keys, a key is invalid or leads outside the
                data directory, or the profile's manifest file is corrupt or names an
                invalid version.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `pyrosome.dal.DataManifestFile.LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the profile directory cannot be created or written to.
        """
        return self.dal.write(_to_polars(data), *keys)

    def append(
        self, data: pl.DataFrame | pl.LazyFrame | pd.DataFrame, *keys: str
    ) -> Path:
        """Add a new file to the latest version without touching existing files.

        A pandas data frame's named index levels are stored as columns; an unnamed
        index is dropped.

        Args:
            data: The data to append, as a polars or pandas frame. Its columns must
                match the existing data.
            *keys: Path components naming the profile.

        Returns:
            The path of the new file, or the new versioned directory if there was no
            existing data.

        Raises:
            ImportError: If data is a pandas frame needing pyarrow to convert, and
                pyarrow is not installed.
            pyrosome.dal.SupersededAppendError: If another process committed a newer
                version while appending, so readers don't see the appended data.
            TimeoutError: If the profile has no data and another process's first
                version is not committed within
                `pyrosome.ParquetDal.FIRST_COMMIT_TIMEOUT` seconds.
            ValueError: If the columns or their types do not match the existing data
                (a type must cast to the existing one without losing values), or for
                any reason `write` raises it.
            FileNotFoundError: If the latest version directory is missing.
            RuntimeError: If the latest data spans more than one directory.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If there was no data
                and another process holds the manifest lock for longer than
                `pyrosome.dal.DataManifestFile.LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If there was no data
                and the manifest cannot be locked.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the data directory cannot be written to.
        """
        return self.dal.append(_to_polars(data), *keys)

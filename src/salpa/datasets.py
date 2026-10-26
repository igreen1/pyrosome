"""Datasets are simplified apis to allow a user to group related data together."""

import contextlib
from typing import TYPE_CHECKING, Self, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pandas as pd
    import polars as pl

    from salpa.api import FilterValue, SimpleIo

    type Frame = pl.DataFrame | pl.LazyFrame | pd.DataFrame


class Dataset:
    """Represents a dataset.

    A dataset is defined by a set of keys, which point to
    specific folders in the data directory. This usually will
    be a system that contains profiles, but a user could add more
    nesting if they so desired.

    Every profile read through a dataset is filtered by the dataset's filters, so
    every profile is expected to have the filtered columns.

    Subclasses can derive profiles from other profiles: a method named
    ``_calc_<profile>`` (keys joined by underscores) returns the profile's data,
    and `get_cache_profile` writes it when the profile has no data yet. The method
    is called on the dataset `without_filters`, so it computes the profile for
    every row and the cache is shared by all subsets.

    Example:
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from salpa import Dataset, SimpleIo
        >>> class Shop(Dataset):
        ...     def _calc_totals(self) -> pl.LazyFrame:
        ...         orders = self.get_profile("orders")
        ...         return orders.group_by("region").agg(pl.col("amount").sum())
        >>> io = SimpleIo(Path("data"))
        >>> orders = pl.DataFrame({"region": ["eu", "us", "eu"], "amount": [1, 2, 3]})
        >>> io.write(orders, "shop", "orders").name
        'v0'
        >>> eu = Shop(io, "shop", region="eu")
        >>> eu.get_profile("orders").collect()["amount"].to_list()
        [1, 3]
        >>> eu.get_cache_profile("totals").collect().rows()
        [('eu', 4)]
        >>> io.read("shop", "totals").collect().height
        2
    """

    keys: tuple[str, ...]
    """Keys which define this dataset."""

    filters: Mapping[str, FilterValue]
    """Row-wise filters to apply when loading the data."""

    _dal: SimpleIo
    """Accessor to grab the data."""

    def __init__(self, dal: SimpleIo, *keys: str, **filters: FilterValue) -> None:
        """Set up the dataset. Nothing is read until a profile is requested.

        Subclasses should keep this signature: `get_subset` and `without_filters`
        call it to make datasets of the same class.

        Args:
            dal: The storage the dataset's profiles are read from.
            *keys: Path components naming the dataset's directory.
            **filters: Column names and the value each column must equal in every
                profile read through this dataset. None matches nulls.

        Returns:
            None.
        """
        self._dal = dal
        self.keys = tuple(keys)
        self.filters = filters

    def get_profile(self, *keys: str) -> pl.LazyFrame:
        """Get a profile defined by keys, filtered by the dataset's filters.

        Args:
            *keys: Path components naming the profile, below the dataset's keys.

        Returns:
            A lazy scan over the profile's latest version, keeping only rows that
            match every filter.

        Raises:
            polars.exceptions.ColumnNotFoundError: If a filter names a column the
                profile doesn't have.
            TypeError: If a filter value is a float or another unsupported type, or
                the profile's manifest file is corrupt.
            FileNotFoundError: If the profile has no data.
            ValueError: If there are no keys at all, or they lead outside the data
                directory.
            polars.exceptions.PolarsError: If the data files cannot be scanned.
        """
        return self._dal.read(*(*self.keys, *keys), **self.filters)

    def get_cache_profile(self, *keys: str, refresh: bool = False) -> pl.LazyFrame:
        """Get a profile, or calculate and cache it if not found.

        The profile is calculated by the ``_calc_<profile>`` method on the dataset
        `without_filters`, and written as a new version, so the cached profile
        holds every row. The dataset's filters are applied when it is read.

        Args:
            *keys: Path components naming the profile, below the dataset's keys.
            refresh: Recalculate the profile even if it has data, e.g. after the
                profiles it is calculated from have changed.

        Returns:
            A lazy scan over the profile's latest version, keeping only rows that
            match every filter.

        Raises:
            AttributeError: If the profile has to be calculated and there is no
                ``_calc_<profile>`` method.
            polars.exceptions.ColumnNotFoundError: If a filter names a column the
                profile doesn't have.
            TypeError: If a filter value is a float or another unsupported type, or
                the profile's manifest file is corrupt.
            FileNotFoundError: If a profile the calculation reads has no data.
            ValueError: If there are no keys at all, or they lead outside the data
                directory.
            ImportError: If the calculation returns a pandas frame needing pyarrow
                to convert, and pyarrow is not installed.
            salpa.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `salpa.dal.DataManifestFile.LOCK_TIMEOUT`.
            salpa.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            polars.exceptions.PolarsError: If the data cannot be computed, written
                or scanned.
            OSError: If the profile directory cannot be created or written to.
            Exception: Anything the ``_calc_<profile>`` method raises.
        """
        if not refresh:
            with contextlib.suppress(FileNotFoundError):
                return self.get_profile(*keys)
        self._calculate_profile(*keys)
        return self.get_profile(*keys)

    def _calculate_profile(self, *keys: str) -> None:
        """Calculate the profile defined by keys and write it as a new version.

        Args:
            *keys: Path components naming the profile, below the dataset's keys.

        Returns:
            None.

        Raises:
            AttributeError: If there is no ``_calc_<profile>`` method for keys.
            FileNotFoundError: If a profile the calculation reads has no data.
            ValueError: If there are no keys at all, they lead outside the data
                directory, or the profile's manifest names an invalid version.
            TypeError: If the profile's manifest file is corrupt.
            ImportError: If the calculation returns a pandas frame needing pyarrow
                to convert, and pyarrow is not installed.
            salpa.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `salpa.dal.DataManifestFile.LOCK_TIMEOUT`.
            salpa.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the profile directory cannot be created or written to.
            Exception: Anything the ``_calc_<profile>`` method raises.
        """
        calc_fn = self._get_calc_function(*keys)
        # calculate from every row, not just this dataset's subset
        self._dal.write(calc_fn(self.without_filters()), *self.keys, *keys)

    def without_filters(self) -> Self:
        """Get this dataset without any filters.

        Returns:
            A dataset of the same class, with the same keys and no filters.
        """
        return type(self)(self._dal, *self.keys)

    def get_subset(self, **new_filters: FilterValue) -> Self:
        """Apply additional filters to further reduce this dataset.

        Args:
            **new_filters: Column names and the value each column must equal, on
                top of this dataset's filters.

        Returns:
            A dataset of the same class, with the same keys and the combined
            filters.

        Raises:
            ValueError: If this dataset already filters one of the columns.
        """
        if already_filtered := sorted(self.filters.keys() & new_filters.keys()):
            msg = f"Dataset is already filtered by {', '.join(already_filtered)}"
            raise ValueError(msg)
        return type(self)(self._dal, *self.keys, **self.filters, **new_filters)

    def _get_calc_function(self, *keys: str) -> Callable[[Self], Frame]:
        """Find the method that calculates the profile defined by keys.

        Args:
            *keys: Path components naming the profile, below the dataset's keys.

        Returns:
            The ``_calc_<keys joined by underscores>`` method, unbound, so it can be
            called on another dataset of this class.

        Raises:
            AttributeError: If the dataset has no such method.
        """
        profile_name = keys
        fn_name = f"_calc_{'_'.join(profile_name)}"
        if hasattr(type(self), fn_name):
            return cast("Callable[[Self], Frame]", getattr(type(self), fn_name))
        msg = f"{type(self).__name__} has no {fn_name} method to calculate {keys}"
        raise AttributeError(msg)

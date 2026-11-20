"""Checks that appended data fits the schema of the data already stored."""

import logging
from typing import Final

import polars as pl
import polars.selectors as cs

_logger = logging.getLogger(__name__)

# the range of values each numeric type holds exactly; a float holds every integer
# up to the size of its significand
_EXACT_RANGES: Final[dict[pl.DataType, tuple[int, int]]] = {
    **{
        dtype: (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        for dtype, bits in (
            (pl.Int8(), 8),
            (pl.Int16(), 16),
            (pl.Int32(), 32),
            (pl.Int64(), 64),
            (pl.Int128(), 128),
        )
    },
    **{
        dtype: (0, (1 << bits) - 1)
        for dtype, bits in (
            (pl.UInt8(), 8),
            (pl.UInt16(), 16),
            (pl.UInt32(), 32),
            (pl.UInt64(), 64),
        )
    },
    pl.Float32(): (-(1 << 24), 1 << 24),
    pl.Float64(): (-(1 << 53), 1 << 53),
}
_FLOATS: Final[frozenset[pl.DataType]] = frozenset({pl.Float32(), pl.Float64()})


def _is_lossless_cast(source: pl.DataType, target: pl.DataType) -> bool:
    """Check that casting from source to target keeps every value exactly.

    Only identical types and widenings are lossless: an integer to a wider integer
    (or to a float that represents it exactly), a float to a wider float, or an
    all-null column to anything. Floats never cast losslessly to integers.

    Args:
        source: The type of the new data.
        target: The type of the existing data.

    Returns:
        Whether every value of source is the same value in target.
    """
    if source in {target, pl.Null()}:
        return True
    source_range = _EXACT_RANGES.get(source)
    target_range = _EXACT_RANGES.get(target)
    if source_range is None or target_range is None:
        return False
    if source in _FLOATS and target not in _FLOATS:
        return False
    return target_range[0] <= source_range[0] and source_range[1] <= target_range[1]


def check_appendable(
    new: pl.Schema, existing: pl.Schema, keys: tuple[str, ...]
) -> None:
    """Check that data with the new schema can be appended to the existing schema.

    The columns must be the same, in any order, and every column's type must be the
    existing type or one that casts to it without losing values.

    Args:
        new: The schema of the data to append.
        existing: The schema of the data already stored.
        keys: Path components naming the profile, for the error message.

    Returns:
        None.

    Raises:
        ValueError: If the columns differ, or a column's type cannot be cast to the
            existing type without losing values.
    """
    if set(new.names()) != set(existing.names()):
        msg = (
            f"Columns {sorted(new.names())} do not match existing columns "
            f"{existing.names()} for {keys}"
        )
        raise ValueError(msg)
    if casts := {
        name: f"{new[name]} -> {dtype}"
        for name, dtype in existing.items()
        if new[name] != dtype
    }:
        _logger.debug("Appending to %s needs casts %s", keys, casts)
    if mismatched := {
        name: f"{new[name]} into {dtype}"
        for name, dtype in existing.items()
        if not _is_lossless_cast(new[name], dtype)
    }:
        msg = (
            f"Column types do not match the existing types for {keys}: {mismatched}. "
            "Cast them explicitly before appending."
        )
        raise ValueError(msg)


def promote_null_columns(data: pl.LazyFrame) -> pl.LazyFrame:
    """Store all-null columns as strings, so that later appends can fill them.

    A column whose type is ``Null`` can only ever hold nulls, so appending any
    value to it would be rejected.

    Args:
        data: The data to write.

    Returns:
        The data with every ``Null`` column cast to ``String``.
    """
    return data.with_columns(cs.by_dtype(pl.Null).cast(pl.String))

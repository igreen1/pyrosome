"""Tests for SimpleIo, the filtering wrapper around ParquetDal."""

import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from salp import SimpleIo
from salp.dal import ParquetDal

KEYS = ("profile", "sub")


@pytest.fixture
def io(tmp_path):
    return SimpleIo(tmp_path)


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "name": ["a", "b", "c", None],
            "count": [1, 2, 2, 3],
            "flag": [True, False, True, False],
            "day": [
                dt.date(2026, 1, 1),
                dt.date(2026, 1, 2),
                dt.date(2026, 1, 2),
                None,
            ],
            "at": [
                dt.datetime(2026, 1, 1, 9),
                dt.datetime(2026, 1, 1, 10),
                None,
                dt.datetime(2026, 1, 1, 9),
            ],
            "price": [0.1 + 0.2, 1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def stored(io, df):
    io.write(df, *KEYS)
    return df


class TestInit:
    def test_uses_parquet(self, tmp_path):
        assert isinstance(SimpleIo(tmp_path).dal, ParquetDal)

    def test_accepts_str_path(self, tmp_path):
        assert SimpleIo(str(tmp_path)).dal.data_directory == tmp_path


class TestRead:
    def test_no_filters_returns_everything(self, io, stored):
        assert_frame_equal(io.read(*KEYS).collect(), stored)

    @pytest.mark.parametrize(
        ("filters", "expected_counts"),
        [
            ({"name": "b"}, [2]),
            ({"count": 2}, [2, 2]),
            ({"flag": True}, [1, 2]),
            ({"day": dt.date(2026, 1, 2)}, [2, 2]),
            ({"at": dt.datetime(2026, 1, 1, 9)}, [1, 3]),
        ],
    )
    def test_filter_by_value(self, io, stored, filters, expected_counts):
        result = io.read(*KEYS, **filters).collect()
        assert result["count"].to_list() == expected_counts

    def test_filters_are_combined_with_and(self, io, stored):
        result = io.read(*KEYS, count=2, flag=True).collect()
        assert result["name"].to_list() == ["c"]

    def test_none_matches_nulls(self, io, stored):
        result = io.read(*KEYS, name=None).collect()
        assert result["count"].to_list() == [3]

    def test_no_matches_is_empty(self, io, stored):
        result = io.read(*KEYS, name="missing").collect()
        assert result.is_empty()
        assert result.schema == stored.schema

    def test_int_matches_float_column(self, io, stored):
        result = io.read(*KEYS, price=2).collect()
        assert result["name"].to_list() == ["c"]

    def test_column_named_self(self, io):
        io.write(pl.DataFrame({"self": [1, 2], "keys": ["a", "b"]}), *KEYS)
        assert io.read(*KEYS, self=2).collect()["keys"].to_list() == ["b"]
        assert io.read(*KEYS, keys="a").collect()["self"].to_list() == [1]

    def test_column_name_that_is_not_an_identifier(self, io):
        io.write(pl.DataFrame({"unit price": [1, 2]}), *KEYS)
        result = io.read(*KEYS, **{"unit price": 2}).collect()
        assert result["unit price"].to_list() == [2]

    def test_filter_reads_appended_data(self, io, stored):
        io.append(stored, *KEYS)
        assert io.read(*KEYS, name="b").collect().height == 2

    def test_missing_column_raises_before_collect(self, io, stored):
        with pytest.raises(pl.exceptions.ColumnNotFoundError, match="nmae, typo"):
            io.read(*KEYS, nmae="a", typo=1, count=1)

    def test_float_raises(self, io, stored):
        with pytest.raises(TypeError, match="float"):
            io.read(*KEYS, price=0.3)

    def test_unsupported_type_raises(self, io, stored):
        with pytest.raises(TypeError, match="list"):
            io.read(*KEYS, count=[1, 2])

    def test_missing_profile_raises(self, io):
        with pytest.raises(FileNotFoundError):
            io.read(*KEYS, name="a")

    def test_no_keys_raises(self, io, stored):
        with pytest.raises(ValueError, match="At least one key"):
            io.read(name="a")


class TestWriteAndAppend:
    def test_write_returns_version_dir(self, io, df):
        assert io.write(df, *KEYS).name == "v0"
        assert io.write(df, *KEYS).name == "v1"

    def test_append_adds_a_file(self, io, df):
        io.write(df, *KEYS)
        assert io.append(df, *KEYS).name == "1.parquet"
        assert_frame_equal(io.read(*KEYS).collect(), pl.concat([df, df]))

    def test_append_rejects_different_columns(self, io, df):
        io.write(df, *KEYS)
        with pytest.raises(ValueError):
            io.append(df.drop("price"), *KEYS)

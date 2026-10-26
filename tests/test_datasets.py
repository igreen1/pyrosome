"""Tests for Dataset, a group of profiles sharing keys and filters."""

import pandas as pd
import polars as pl
import pytest

from pyrosome import Dataset, SimpleIo
from pyrosome.dal import DataManifestFile

ORDERS = pl.DataFrame({"region": ["eu", "us", "eu"], "amount": [1, 2, 3]})


class Shop(Dataset):
    def _calc_totals(self):
        return self.get_profile("orders").group_by("region").agg(pl.col("amount").sum())

    def _calc_by_region_eu(self):
        return pd.DataFrame({"region": ["eu"], "n": [len(self.filters)]})

    def _calc_double(self):
        totals = self.get_cache_profile("totals")
        return totals.with_columns(pl.col("amount") * 2)


@pytest.fixture
def io(tmp_path):
    io = SimpleIo(tmp_path)
    io.write(ORDERS, "shop", "orders")
    return io


@pytest.fixture
def eu(io):
    return Shop(io, "shop", region="eu")


def totals(dataset, **kwargs):
    return sorted(dataset.get_cache_profile("totals", **kwargs).collect().rows())


class TestGetProfile:
    def test_applies_filters(self, eu):
        assert eu.get_profile("orders").collect()["amount"].to_list() == [1, 3]

    def test_no_filters(self, io):
        assert Dataset(io, "shop").get_profile("orders").collect().height == 3

    def test_missing_profile_raises(self, eu):
        with pytest.raises(FileNotFoundError):
            eu.get_profile("missing")

    def test_profile_without_filter_column_raises(self, io, eu):
        io.write(pl.DataFrame({"name": ["ana"]}), "shop", "customers")
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            eu.get_profile("customers")


class TestWithoutFilters:
    def test_drops_filters_and_keeps_keys(self, eu):
        unfiltered = eu.without_filters()
        assert unfiltered.keys == ("shop",)
        assert unfiltered.filters == {}

    def test_keeps_class(self, eu):
        assert type(eu.without_filters()) is Shop

    def test_leaves_original_filtered(self, eu):
        eu.without_filters()
        assert eu.filters == {"region": "eu"}


class TestGetSubset:
    def test_combines_filters(self, eu):
        subset = eu.get_subset(amount=3)
        assert subset.filters == {"region": "eu", "amount": 3}
        assert subset.get_profile("orders").collect()["amount"].to_list() == [3]

    def test_keeps_class_and_keys(self, eu):
        subset = eu.get_subset(amount=3)
        assert type(subset) is Shop
        assert subset.keys == eu.keys

    def test_subset_can_calculate(self, eu):
        assert totals(eu.get_subset(amount=4)) == [("eu", 4)]

    @pytest.mark.parametrize("value", ["eu", "us"])
    def test_refiltering_a_column_raises(self, eu, value):
        with pytest.raises(ValueError, match="already filtered by region"):
            eu.get_subset(region=value, amount=1)


class TestGetCacheProfile:
    def test_calculates_missing_profile(self, eu):
        assert totals(eu) == [("eu", 4)]

    def test_caches_every_row(self, io, eu):
        eu.get_cache_profile("totals")
        assert sorted(io.read("shop", "totals").collect().rows()) == [
            ("eu", 4),
            ("us", 2),
        ]

    def test_calculates_without_filters(self, io, monkeypatch):
        seen = []
        monkeypatch.setattr(
            Shop,
            "_calc_totals",
            lambda self: seen.append(dict(self.filters)) or ORDERS,
        )
        Shop(io, "shop", region="eu").get_cache_profile("totals")
        assert seen == [{}]

    def test_uses_cache(self, io, eu, monkeypatch):
        eu.get_cache_profile("totals")
        monkeypatch.setattr(
            Shop, "_calc_totals", lambda self: pytest.fail("recalculated")
        )
        assert totals(eu) == [("eu", 4)]

    def test_existing_profile_needs_no_calc_method(self, eu):
        assert eu.get_cache_profile("orders").collect().height == 2

    def test_stale_without_refresh(self, io, eu):
        eu.get_cache_profile("totals")
        io.append(pl.DataFrame({"region": ["eu"], "amount": [10]}), "shop", "orders")
        assert totals(eu) == [("eu", 4)]

    def test_refresh_recalculates(self, io, eu, tmp_path):
        eu.get_cache_profile("totals")
        io.append(pl.DataFrame({"region": ["eu"], "amount": [10]}), "shop", "orders")
        assert totals(eu, refresh=True) == [("eu", 14)]
        latest = DataManifestFile(tmp_path / "shop" / "totals").get_latest_data_dir()
        assert latest.name == "v1"

    def test_refresh_calculates_missing_profile(self, eu):
        assert totals(eu, refresh=True) == [("eu", 4)]

    def test_multi_key_profile(self, io, eu):
        result = eu.get_cache_profile("by", "region", "eu").collect()
        assert result.rows() == [("eu", 0)]
        assert io.read("shop", "by", "region", "eu").collect().height == 1

    def test_calculation_can_use_other_cached_profiles(self, io, eu):
        assert eu.get_cache_profile("double").collect().rows() == [("eu", 8)]
        assert io.read("shop", "totals").collect().height == 2

    def test_missing_calc_method_raises(self, eu):
        with pytest.raises(AttributeError, match="Shop has no _calc_missing method"):
            eu.get_cache_profile("missing")

    def test_refresh_without_calc_method_raises(self, eu):
        with pytest.raises(AttributeError, match="_calc_orders"):
            eu.get_cache_profile("orders", refresh=True)

    def test_calculation_error_propagates(self, io):
        with pytest.raises(FileNotFoundError):
            Shop(io, "other").get_cache_profile("totals")

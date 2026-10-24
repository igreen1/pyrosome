"""Tests for the CSV data access layer."""

from concurrent.futures import ProcessPoolExecutor

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from salp.dal.dal import DAL, CsvDal

KEYS = ("profile", "sub")


@pytest.fixture
def dal(tmp_path):
    return CsvDal(tmp_path)


@pytest.fixture
def df():
    return pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [0.5, 1.5]})


def visible_files(directory) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


class TestWrite:
    def test_write_then_read(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        assert output_dir == dal.data_directory / "profile" / "sub" / "v0"
        assert visible_files(output_dir) == ["0.csv"]
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_writes_plain_csv(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        assert (output_dir / "0.csv").read_text() == "a,b,c\n1,x,0.5\n2,y,1.5\n"

    def test_write_lazyframe(self, dal, df):
        dal.write(df.lazy(), *KEYS)
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_second_write_creates_new_version(self, dal, df):
        dal.write(df, *KEYS)
        new_df = pl.DataFrame({"d": ["new"]})
        output_dir = dal.write(new_df, *KEYS)
        assert output_dir.name == "v1"
        assert_frame_equal(dal.read(*KEYS).collect(), new_df)
        assert (output_dir.parent / "v0" / "0.csv").exists()


class TestRead:
    def test_missing_profile(self, dal):
        with pytest.raises(FileNotFoundError):
            dal.read(*KEYS)

    def test_ignores_non_csv_files(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        (output_dir / ".abc.csv.tmp").write_text("partial")
        (output_dir / "1.parquet").write_text("other format")
        (output_dir / "x.csv").write_text("not,ours\n")
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_does_not_read_parquet_data(self, tmp_path, df):
        # the manifest is shared, but a CSV reader must not pick up parquet files
        DAL(tmp_path).write(df, *KEYS)
        with pytest.raises(FileNotFoundError):
            CsvDal(tmp_path).read(*KEYS)

    def test_reads_files_in_index_order(self, dal, df):
        output_dir = dal.write(df.head(1), *KEYS)
        for _ in range(10):
            dal.append(df.tail(1), *KEYS)
        assert "10.csv" in visible_files(output_dir)
        expected = pl.concat([df.head(1), *[df.tail(1)] * 10])
        assert_frame_equal(dal.read(*KEYS).collect(), expected)


class TestAppend:
    def test_append_adds_file(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        output_file = dal.append(df, *KEYS)
        assert output_file == output_dir / "1.csv"
        assert visible_files(output_dir) == ["0.csv", "1.csv"]
        assert_frame_equal(dal.read(*KEYS).collect(), pl.concat([df, df]))

    def test_append_to_empty_writes(self, dal, df):
        with pytest.warns(UserWarning, match="Writing instead"):
            output_dir = dal.append(df, *KEYS)
        assert visible_files(output_dir) == ["0.csv"]
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_append_reorders_and_casts_to_inferred_schema(self, dal, df):
        dal.write(df, *KEYS)
        other = pl.DataFrame(
            {"c": [2.5], "b": ["z"], "a": pl.Series([3], dtype=pl.Int32)}
        )
        output_file = dal.append(other, *KEYS)
        assert output_file.read_text().splitlines()[0] == "a,b,c"
        result = dal.read(*KEYS).collect()
        assert result.schema == df.schema
        assert result["a"].to_list() == [1, 2, 3]

    @pytest.mark.parametrize(
        "columns",
        [{"a": [1]}, {"a": [1], "b": ["x"], "c": [0.0], "d": [0]}],
        ids=["missing", "extra"],
    )
    def test_append_column_mismatch(self, dal, df, columns):
        output_dir = dal.write(df, *KEYS)
        with pytest.raises(ValueError, match="do not match"):
            dal.append(pl.DataFrame(columns), *KEYS)
        assert visible_files(output_dir) == ["0.csv"]

    def test_append_cleans_up_tmp_file(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        assert not [p for p in output_dir.iterdir() if p.name.startswith(".")]


def _append(data_directory, keys):
    return CsvDal(data_directory).append(pl.DataFrame({"a": [1, 2]}), *keys)


def test_concurrent_appends(tmp_path):
    n = 16
    dal = CsvDal(tmp_path)
    output_dir = dal.write(pl.DataFrame({"a": [1, 2]}), *KEYS)
    with ProcessPoolExecutor(8) as executor:
        files = list(executor.map(_append, [tmp_path] * n, [KEYS] * n))
    assert len(set(files)) == n
    assert visible_files(output_dir) == sorted(f"{i}.csv" for i in range(n + 1))
    assert dal.read(*KEYS).collect().height == 2 * (n + 1)

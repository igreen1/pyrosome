"""Tests for the data access layer."""

import json
import stat
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from salp.dal.dal import DAL, DataManifestFile
from salp.dal.simple_file_lock import FileAlreadyLockedError, SimpleWriteLock

WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH

KEYS = ("profile", "sub")


@pytest.fixture
def dal(tmp_path):
    return DAL(tmp_path)


@pytest.fixture
def df():
    return pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})


def manifest_contents(dal: DAL, *keys: str) -> dict:
    return json.loads(
        (dal.data_directory.joinpath(*keys) / "manifest.json").read_text()
    )


def visible_files(directory) -> list[str]:
    """Files in directory, apart from the claim markers left by appends."""
    return sorted(
        path.name for path in directory.iterdir() if path.suffix != ".claim"
    )


class TestWrite:
    def test_write_then_read(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        assert output_dir == dal.data_directory / "profile" / "sub" / "v0"
        assert visible_files(output_dir) == ["0.parquet"]
        assert manifest_contents(dal, *KEYS) == {"latest_data_dir": "v0"}
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_written_file_is_read_only(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        assert not (output_dir / "0.parquet").stat().st_mode & WRITE_BITS

    def test_write_lazyframe(self, dal, df):
        dal.write(df.lazy(), *KEYS)
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_second_write_creates_new_version(self, dal, df):
        dal.write(df, *KEYS)
        new_df = pl.DataFrame({"c": [1.5]})
        output_dir = dal.write(new_df, *KEYS)
        assert output_dir.name == "v1"
        assert manifest_contents(dal, *KEYS) == {"latest_data_dir": "v1"}
        assert_frame_equal(dal.read(*KEYS).collect(), new_df)
        # the previous version is kept untouched
        assert (output_dir.parent / "v0" / "0.parquet").exists()

    def test_profiles_are_independent(self, dal, df):
        dal.write(df, "one")
        dal.write(df.head(1), "two")
        assert dal.read("one").collect().height == 2
        assert dal.read("two").collect().height == 1

    def test_failed_write_keeps_previous_version(self, dal, df, monkeypatch):
        dal.write(df, *KEYS)

        def failing_sink(_data, _output_file):
            raise RuntimeError

        monkeypatch.setattr(DAL, "_sink", staticmethod(failing_sink))
        with pytest.raises(RuntimeError):
            dal.write(df, *KEYS)
        assert manifest_contents(dal, *KEYS) == {"latest_data_dir": "v0"}
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_skips_versions_not_yet_committed(self, dal, df):
        # an in-progress write has reserved v1 but not committed it
        (dal.data_directory / "p" / "v1").mkdir(parents=True)
        assert dal.write(df, "p").name == "v2"


class TestRead:
    def test_missing_profile(self, dal):
        with pytest.raises(FileNotFoundError):
            dal.read(*KEYS)

    def test_ignores_non_data_files(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        (output_dir / ".abc.parquet.tmp").write_text("partial")
        (output_dir / "notes.txt").write_text("hello")
        (output_dir / "x.parquet").write_text("not ours")
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_empty_version_dir(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        (output_dir / "0.parquet").unlink()
        with pytest.raises(FileNotFoundError):
            dal.read(*KEYS)

    def test_manifest_points_to_missing_dir(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        (output_dir / "0.parquet").unlink()
        output_dir.rmdir()
        with pytest.raises(FileNotFoundError):
            dal.read(*KEYS)

    def test_corrupt_manifest(self, dal, df):
        dal.write(df, *KEYS)
        (dal.data_directory.joinpath(*KEYS) / "manifest.json").write_text("{}")
        with pytest.raises(TypeError, match="corrupt"):
            dal.read(*KEYS)


class TestAppend:
    def test_appended_file_is_read_only(self, dal, df):
        dal.write(df, *KEYS)
        appended = dal.append(df, *KEYS)
        assert not appended.stat().st_mode & WRITE_BITS

    def test_append_adds_file(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        output_file = dal.append(df, *KEYS)
        assert output_file == output_dir / "1.parquet"
        assert visible_files(output_dir) == ["0.parquet", "1.parquet"]
        assert_frame_equal(dal.read(*KEYS).collect(), pl.concat([df, df]))

    def test_append_to_empty_writes(self, dal, df):
        with pytest.warns(UserWarning, match="Writing instead"):
            output_dir = dal.append(df, *KEYS)
        assert output_dir.name == "v0"
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_append_reorders_and_casts_columns(self, dal, df):
        dal.write(df, *KEYS)
        other = pl.DataFrame({"b": ["z"], "a": pl.Series([3], dtype=pl.Int32)})
        dal.append(other, *KEYS)
        result = dal.read(*KEYS).collect()
        assert result.schema == df.schema
        assert result["a"].to_list() == [1, 2, 3]

    @pytest.mark.parametrize(
        "columns",
        [{"a": [1]}, {"a": [1], "b": ["x"], "c": [0]}],
        ids=["missing", "extra"],
    )
    def test_append_column_mismatch(self, dal, df, columns):
        output_dir = dal.write(df, *KEYS)
        with pytest.raises(ValueError, match="do not match"):
            dal.append(pl.DataFrame(columns), *KEYS)
        assert visible_files(output_dir) == ["0.parquet"]

    def test_append_skips_taken_indices(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        # another process published 1.parquet after we listed the files
        latest_data = dal._get_latest_profile_data_paths(*KEYS)
        dal.append(df, *KEYS)
        tmp_file = output_dir / ".tmp.parquet.tmp"
        df.write_parquet(tmp_file)
        assert (
            dal._publish_append_file(tmp_file, latest_data) == output_dir / "2.parquet"
        )

    def test_append_cleans_up_tmp_file(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        assert not list(output_dir.glob("*.tmp"))

    def test_append_leaves_claim_marker(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        assert (output_dir / ".1.parquet.claim").exists()

    def test_append_skips_claimed_indices(self, dal, df):
        # a claim whose append never finished, e.g. the process crashed
        output_dir = dal.write(df, *KEYS)
        (output_dir / ".1.parquet.claim").touch()
        assert dal.append(df, *KEYS) == output_dir / "2.parquet"
        assert visible_files(output_dir) == ["0.parquet", "2.parquet"]
        assert_frame_equal(dal.read(*KEYS).collect(), pl.concat([df, df]))

    def test_append_removes_read_only_tmp_file_on_failure(
        self, dal, df, monkeypatch
    ):
        output_dir = dal.write(df, *KEYS)
        real_unlink = Path.unlink

        def windows_unlink(path, missing_ok=False):
            # Windows cannot delete read-only files
            if path.exists() and not path.stat().st_mode & stat.S_IWUSR:
                raise PermissionError(path)
            real_unlink(path, missing_ok=missing_ok)

        def failing_replace(path, target):
            raise OSError("replace failed")

        monkeypatch.setattr(Path, "unlink", windows_unlink)
        monkeypatch.setattr(Path, "replace", failing_replace)
        with pytest.raises(OSError, match="replace failed"):
            dal.append(df, *KEYS)
        assert not list(output_dir.glob("*.tmp"))

    def test_append_goes_to_latest_version(self, dal, df):
        dal.write(df, *KEYS)
        latest_dir = dal.write(df, *KEYS)
        assert dal.append(df, *KEYS).parent == latest_dir


class TestKeys:
    def test_no_keys(self, dal, df):
        with pytest.raises(ValueError, match="At least one key"):
            dal.write(df)

    @pytest.mark.parametrize("keys", [("..",), ("a", "..", ".."), ("/etc",)])
    def test_escaping_data_directory(self, dal, df, keys):
        with pytest.raises(ValueError, match="not within data directory"):
            dal.write(df, *keys)


class TestManifest:
    def test_commit_does_not_go_backwards(self, tmp_path):
        manifest = DataManifestFile(tmp_path)
        v0 = manifest.reserve_new_data_dir()
        v1 = manifest.reserve_new_data_dir()
        manifest.commit_data_dir(v1)
        manifest.commit_data_dir(v0)
        assert manifest.get_latest_data_dir() == v1

    def test_commit_waits_for_lock(self, tmp_path, monkeypatch):
        monkeypatch.setattr(DataManifestFile, "LOCK_TIMEOUT", 0.1)
        manifest = DataManifestFile(tmp_path)
        v0 = manifest.reserve_new_data_dir()
        with (
            SimpleWriteLock(manifest.manifest_path),
            pytest.raises(FileAlreadyLockedError),
        ):
            manifest.commit_data_dir(v0)
        assert not manifest.exists()

    def test_invalid_data_dir_name(self, tmp_path):
        manifest = DataManifestFile(tmp_path)
        bad_dir = tmp_path / "latest"
        bad_dir.mkdir()
        with pytest.raises(ValueError, match="Invalid data directory name"):
            manifest.commit_data_dir(bad_dir)

    def test_write_leaves_no_tmp_files(self, tmp_path):
        manifest = DataManifestFile(tmp_path)
        manifest.commit_data_dir(manifest.reserve_new_data_dir())
        assert visible_files(tmp_path) == ["manifest.json", "v0"]


def _write(data_directory, keys):
    return DAL(data_directory).write(pl.DataFrame({"a": [1, 2]}), *keys)


def _append(data_directory, keys):
    return DAL(data_directory).append(pl.DataFrame({"a": [1, 2]}), *keys)


class TestConcurrency:
    N = 16

    def test_concurrent_writes(self, tmp_path):
        with ProcessPoolExecutor(8) as executor:
            dirs = list(executor.map(_write, [tmp_path] * self.N, [KEYS] * self.N))
        assert len(set(dirs)) == self.N
        assert manifest_contents(DAL(tmp_path), *KEYS) == {
            "latest_data_dir": f"v{self.N - 1}"
        }

    def test_concurrent_appends(self, tmp_path):
        dal = DAL(tmp_path)
        output_dir = _write(tmp_path, KEYS)
        with ProcessPoolExecutor(8) as executor:
            files = list(executor.map(_append, [tmp_path] * self.N, [KEYS] * self.N))
        assert len(set(files)) == self.N
        assert visible_files(output_dir) == sorted(
            f"{i}.parquet" for i in range(self.N + 1)
        )
        assert dal.read(*KEYS).collect().height == 2 * (self.N + 1)

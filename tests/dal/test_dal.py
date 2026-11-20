"""Tests for the data access layer."""

import json
import logging
import os
import stat
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyrosome import _files, _keys
from pyrosome.api import SimpleIo
from pyrosome.dal import CsvDal, Dal, DataManifestFile, ParquetDal, ProfileNotFoundError
from pyrosome.simple_file_lock import FileAlreadyLockedError, SimpleWriteLock

WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH

KEYS = ("profile", "sub")


@pytest.fixture
def dal(tmp_path):
    return ParquetDal(tmp_path)


@pytest.fixture
def df():
    return pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})


def manifest_contents(dal: ParquetDal, *keys: str) -> dict:
    return json.loads(
        (dal.data_directory.joinpath(*keys) / "manifest.json").read_text()
    )


def delete_data_file(path: Path) -> None:
    """Data files are read only, and Windows can't delete read-only files."""
    path.chmod(stat.S_IWRITE)
    path.unlink()


def visible_files(directory) -> list[str]:
    """Files in directory, apart from the claim markers left by appends."""
    return sorted(path.name for path in directory.iterdir() if path.suffix != ".claim")


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

        monkeypatch.setattr(ParquetDal, "_sink", staticmethod(failing_sink))
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
        delete_data_file(output_dir / "0.parquet")
        with pytest.raises(FileNotFoundError):
            dal.read(*KEYS)

    def test_manifest_points_to_missing_dir(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        delete_data_file(output_dir / "0.parquet")
        output_dir.rmdir()
        with pytest.raises(FileNotFoundError):
            dal.read(*KEYS)

    def test_corrupt_manifest(self, dal, df):
        dal.write(df, *KEYS)
        (dal.data_directory.joinpath(*KEYS) / "manifest.json").write_text("{}")
        with pytest.raises(ValueError, match="corrupt"):
            dal.read(*KEYS)

    @pytest.mark.parametrize("contents", ["{", '{"latest_data_dir": "../other/v0"}'])
    def test_unusable_manifest(self, dal, df, contents):
        dal.write(df, *KEYS)
        (dal.data_directory.joinpath(*KEYS) / "manifest.json").write_text(contents)
        with pytest.raises(ValueError):
            dal.read(*KEYS)

    def test_missing_profile_is_profile_not_found(self, dal):
        with pytest.raises(ProfileNotFoundError):
            dal.read(*KEYS)

    def test_missing_version_dir_is_not_profile_not_found(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        delete_data_file(output_dir / "0.parquet")
        output_dir.rmdir()
        with pytest.raises(FileNotFoundError) as error:
            dal.read(*KEYS)
        assert not isinstance(error.value, ProfileNotFoundError)


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

    def test_append_warning_points_at_caller(self, dal, df):
        with pytest.warns(UserWarning, match="Writing instead") as record:
            dal.append(df, *KEYS)
        assert record[0].filename == __file__

    def test_append_reorders_and_casts_columns(self, dal, df):
        dal.write(df, *KEYS)
        other = pl.DataFrame({"b": ["z"], "a": pl.Series([3], dtype=pl.Int32)})
        dal.append(other, *KEYS)
        result = dal.read(*KEYS).collect()
        assert result.schema == df.schema
        assert result["a"].to_list() == [1, 2, 3]

    @pytest.mark.parametrize(
        ("existing", "new"),
        [
            (pl.Int64, pl.Float64),  # would truncate
            (pl.Int8, pl.Int64),  # could overflow
            (pl.Int64, pl.UInt64),
            (pl.UInt8, pl.Int8),
            (pl.Float32, pl.Float64),  # would round
            (pl.Float64, pl.Int64),  # beyond 2**53
            (pl.String, pl.Int64),
            (pl.Int64, pl.String),
            (pl.Datetime("us"), pl.Datetime("ms")),
        ],
    )
    def test_append_rejects_lossy_types(self, dal, existing, new):
        output_dir = dal.write(pl.DataFrame({"a": pl.Series([], dtype=existing)}), *KEYS)
        with pytest.raises(ValueError, match="types do not match"):
            dal.append(pl.DataFrame({"a": pl.Series([], dtype=new)}), *KEYS)
        assert visible_files(output_dir) == ["0.parquet"]

    @pytest.mark.parametrize(
        ("existing", "new"),
        [
            (pl.Int64, pl.Int8),
            (pl.Int64, pl.UInt32),
            (pl.UInt64, pl.UInt8),
            (pl.Float64, pl.Float32),
            (pl.Float64, pl.Int32),
            (pl.Float32, pl.Int16),
            (pl.String, pl.Null),
            (pl.Datetime("us"), pl.Datetime("us")),
        ],
    )
    def test_append_allows_lossless_types(self, dal, existing, new):
        dal.write(pl.DataFrame({"a": pl.Series([], dtype=existing)}), *KEYS)
        dal.append(pl.DataFrame({"a": pl.Series([], dtype=new)}), *KEYS)
        assert dal.read(*KEYS).collect_schema()["a"] == existing

    def test_append_float_to_int_column_is_rejected_not_truncated(self, dal, df):
        output_dir = dal.write(df, *KEYS)
        with pytest.raises(ValueError, match="a': 'Float64 into Int64"):
            dal.append(pl.DataFrame({"a": [1.5], "b": ["z"]}), *KEYS)
        assert_frame_equal(dal.read(*KEYS).collect(), df)
        assert visible_files(output_dir) == ["0.parquet"]

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
            _files.publish_append_file(tmp_file, latest_data, "parquet") == output_dir / "2.parquet"
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

    def test_append_removes_read_only_tmp_file_on_failure(self, dal, df, monkeypatch):
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

    @pytest.mark.parametrize(
        "keys",
        [
            ("..",),
            ("a", "..", ".."),
            ("/etc",),
            ("a/../..",),
            ("a\\..\\..",),
            ("C:",),
            ("C:\\evil",),
        ],
    )
    def test_escaping_data_directory(self, dal, df, keys):
        with pytest.raises(ValueError, match="Invalid key"):
            dal.write(df, *keys)

    @pytest.mark.parametrize(
        "keys",
        [
            ("",),
            (".",),
            ("a", ""),
            ("a/",),
            ("a//b",),
            ("a/./b",),
            ("v0",),
            ("a/v12",),
            ("manifest.json",),
            (".hidden",),
            ("a", ".b"),
            ("a/.b/c",),
            ("a\\.b",),
            (".snapshot",),
            ("a\0b",),
            ("2024-01-01T10:00",),
        ],
    )
    def test_invalid_keys(self, dal, df, keys):
        with pytest.raises(ValueError, match="Invalid key"):
            dal.write(df, *keys)
        assert not list(dal.data_directory.iterdir())

    def test_hidden_key_message(self, dal, df):
        with pytest.raises(ValueError, match="hidden path components"):
            dal.write(df, "a", ".b")

    def test_hidden_data_directory_is_allowed(self, tmp_path, df):
        dal = ParquetDal(tmp_path / ".cache")
        dal.write(df, *KEYS)
        assert dal.available_profiles() == (KEYS,)
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_invalid_keys_are_rejected_on_read_and_append(self, dal, df):
        for method in (dal.read, lambda *keys: dal.append(df, *keys)):
            with pytest.raises(ValueError, match="Invalid key"):
                method("")

    @pytest.mark.parametrize("keys", [("a/b",), ("a\\b",), ("v", "vv", "v1a", "version")])
    def test_valid_keys(self, dal, df, keys):
        dal.write(df, *keys)
        assert_frame_equal(dal.read(*keys).collect(), df)


class TestAvailableProfiles:
    def test_empty_data_directory(self, dal):
        assert dal.available_profiles() == ()

    def test_missing_data_directory(self, tmp_path):
        assert ParquetDal(tmp_path / "missing").available_profiles() == ()

    def test_missing_keys(self, dal, df):
        dal.write(df, "a")
        assert dal.available_profiles("b") == ()
        assert dal.available_profiles("a", "b") == ()

    def test_lists_nested_profiles(self, dal, df):
        for keys in [("b",), ("a", "x"), ("a", "y", "z"), ("a", "y", "z", "deeper")]:
            dal.write(df, *keys)
        assert dal.available_profiles() == (
            ("a", "x"),
            ("a", "y", "z"),
            ("a", "y", "z", "deeper"),
            ("b",),
        )
        assert dal.available_profiles("a") == (
            ("a", "x"),
            ("a", "y", "z"),
            ("a", "y", "z", "deeper"),
        )
        assert dal.available_profiles("a", "y") == (
            ("a", "y", "z"),
            ("a", "y", "z", "deeper"),
        )

    def test_includes_profile_named_by_keys(self, dal, df):
        dal.write(df, *KEYS)
        assert dal.available_profiles(*KEYS) == (KEYS,)

    def test_keys_with_separators(self, dal, df):
        dal.write(df, *KEYS)
        assert dal.available_profiles("profile/sub") == (KEYS,)

    def test_results_can_be_read(self, dal, df):
        dal.write(df, *KEYS)
        dal.write(df, "other")
        for keys in dal.available_profiles():
            assert_frame_equal(dal.read(*keys).collect(), df)

    def test_uncommitted_write_is_not_listed(self, dal):
        (dal.data_directory / "pending").mkdir()
        DataManifestFile(dal.data_directory / "pending").reserve_new_data_dir()
        assert dal.available_profiles() == ()

    def test_ignores_files_and_version_dirs(self, dal, df):
        dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        (dal.data_directory / "stray.txt").write_text("")
        assert dal.available_profiles() == (KEYS,)

    def test_data_directory_manifest_is_not_a_profile(self, dal, df):
        (dal.data_directory / "manifest.json").write_text("{}")
        dal.write(df, "a")
        assert dal.available_profiles() == (("a",),)

    def test_key_that_is_a_file(self, dal):
        (dal.data_directory / "file").write_text("")
        assert dal.available_profiles("file") == ()

    def test_invalid_keys(self, dal):
        with pytest.raises(ValueError, match="Invalid key"):
            dal.available_profiles("..")

    def test_skips_hidden_directories(self, dal, df):
        dal.write(df, "a")
        ParquetDal(dal.data_directory / ".snapshot" / "hourly").write(df, "a")
        assert dal.available_profiles() == (("a",),)

    @pytest.mark.parametrize(
        "name",
        [
            "v3",
            pytest.param(
                "a:b",
                marks=pytest.mark.skipif(
                    sys.platform == "win32", reason="':' is not allowed in names"
                ),
            ),
        ],
    )
    def test_skips_directories_that_are_not_valid_keys(self, dal, df, name):
        dal.write(df, "a")
        ParquetDal(dal.data_directory / name).write(df, "q")
        assert dal.available_profiles() == (("a",),)

    @pytest.mark.skipif(sys.platform == "win32", reason="'\\' separates paths")
    def test_backslash_in_directory_name(self, dal, df):
        dal.write(df, "a\\b")
        assert dal.available_profiles() == (("a\\b",),)
        assert_frame_equal(dal.read(*dal.available_profiles()[0]).collect(), df)

    def test_skips_other_formats(self, dal, df):
        dal.write(df, "parquet")
        CsvDal(dal.data_directory).write(df, "csv")
        assert dal.available_profiles() == (("parquet",),)
        assert CsvDal(dal.data_directory).available_profiles() == (("csv",),)

    def test_skips_corrupt_manifest(self, dal, df, caplog):
        dal.write(df, "good")
        dal.write(df, "bad")
        (dal.data_directory / "bad" / "manifest.json").write_text("not json")
        assert dal.available_profiles() == (("good",),)
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_skips_manifest_naming_missing_version(self, dal, df, caplog):
        dal.write(df, "good")
        (dal.data_directory / "bad").mkdir()
        (dal.data_directory / "bad" / "manifest.json").write_text(
            json.dumps({"latest_data_dir": "v3"})
        )
        assert dal.available_profiles() == (("good",),)
        assert any("missing v3" in record.getMessage() for record in caplog.records)

    def test_does_not_follow_symlinks(self, dal, df):
        dal.write(df, "real", "p")
        try:
            (dal.data_directory / "link").symlink_to(
                dal.data_directory / "real", target_is_directory=True
            )
        except OSError:
            pytest.skip("cannot create symlinks")
        assert dal.available_profiles() == (("real", "p"),)

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0,
        reason="needs POSIX permissions and a non-root user",
    )
    def test_unreadable_directory_raises(self, dal, df):
        dal.write(df, "locked", "p")
        locked = dal.data_directory / "locked"
        locked.chmod(0)
        try:
            with pytest.raises(PermissionError):
                dal.available_profiles()
        finally:
            locked.chmod(stat.S_IRWXU)

    def test_simple_io_lists_nested_profiles(self, tmp_path, df):
        io = SimpleIo(tmp_path)
        io.write(df, "shop", "orders")
        io.write(df, "shop", "eu", "totals")
        assert io.available_profiles("shop") == (
            ("shop", "eu", "totals"),
            ("shop", "orders"),
        )

    def test_protocol_subclass_without_implementation_raises(self):
        class Partial(Dal):
            pass

        with pytest.raises(NotImplementedError, match="Partial"):
            Partial().available_profiles()


def is_case_insensitive(directory: Path) -> bool:
    probe = directory / "case_probe"
    probe.touch()
    try:
        return (directory / "CASE_PROBE").exists()
    finally:
        probe.unlink()


class TestCaseSensitivity:
    @pytest.mark.parametrize(
        "keys", [("SHOP", "orders"), ("shop", "Orders"), ("Shop", "orders")]
    )
    def test_read_with_other_case_is_not_found(self, dal, df, keys):
        dal.write(df, "shop", "orders")
        with pytest.raises(ProfileNotFoundError):
            dal.read(*keys)

    def test_available_profiles_with_other_case_is_empty(self, dal, df):
        dal.write(df, "shop", "orders")
        assert dal.available_profiles("SHOP") == ()
        assert dal.available_profiles("shop", "ORDERS") == ()
        assert dal.available_profiles() == (("shop", "orders"),)

    def test_listed_keys_use_case_on_disk(self, dal, df):
        dal.write(df, "Shop", "Orders")
        assert dal.available_profiles("Shop") == (("Shop", "Orders"),)

    def test_write_with_other_case(self, dal, df):
        dal.write(df, "shop", "orders")
        other = pl.DataFrame({"a": [9], "b": ["z"]})
        if is_case_insensitive(dal.data_directory):
            with pytest.raises(ValueError, match="differ only in case"):
                dal.write(other, "SHOP", "orders")
            with pytest.raises(ValueError, match="differ only in case"):
                dal.append(other, "shop", "ORDERS")
            assert not (dal.data_directory / "shop" / "new").exists()
            with pytest.raises(ValueError, match="differ only in case"):
                dal.write(other, "SHOP", "new")
            assert not (dal.data_directory / "shop" / "new").exists()
            assert dal.available_profiles() == (("shop", "orders"),)
        else:
            dal.write(other, "SHOP", "orders")
            assert dal.available_profiles() == (("SHOP", "orders"), ("shop", "orders"))
            assert_frame_equal(dal.read("SHOP", "orders").collect(), other)
        assert_frame_equal(dal.read("shop", "orders").collect(), df)

    def test_keys_without_letters(self, dal, df):
        dal.write(df, "2024", "01")
        assert dal.available_profiles("2024") == (("2024", "01"),)
        assert_frame_equal(dal.read("2024", "01").collect(), df)


class TestMatchesCaseOnDisk:
    """Emulate a case-insensitive file system: other-case names resolve through
    symlinks, but listing a directory only shows the real names."""

    @pytest.fixture
    def insensitive(self, tmp_path, monkeypatch):
        (tmp_path / "shop" / "orders").mkdir(parents=True)
        for alias, target in [("SHOP", "shop"), ("sHOP", "shop"), ("Shop", "shop")]:
            try:
                (tmp_path / alias).symlink_to(
                    tmp_path / target, target_is_directory=True
                )
            except OSError:
                pytest.skip("cannot create symlinks")
        (tmp_path / "shop" / "ORDERS").symlink_to("orders", target_is_directory=True)

        def has_real_entry(directory, name):
            entry = directory / name
            return entry.exists() and not entry.is_symlink()

        monkeypatch.setattr(_keys, "_has_entry", has_real_entry)
        return tmp_path

    @pytest.mark.parametrize(
        ("parts", "expected"),
        [
            (("shop", "orders"), True),
            (("SHOP", "orders"), False),
            (("Shop", "orders"), False),
            (("shop", "ORDERS"), False),
            (("shop", "missing"), True),
            (("missing", "orders"), True),
        ],
    )
    def test_matches(self, insensitive, parts, expected):
        assert (
            _keys.matches_case_on_disk(insensitive, insensitive.joinpath(*parts))
            is expected
        )


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

    def test_commit_while_reading(self, tmp_path):
        """Readers must not make a commit fail, which Windows can do by locking files."""
        manifest = DataManifestFile(tmp_path)
        manifest.commit_data_dir(manifest.reserve_new_data_dir())
        done = threading.Event()
        errors: list[BaseException] = []

        def read_forever():
            try:
                while not done.is_set():
                    manifest.get_latest_data_dir()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        readers = [threading.Thread(target=read_forever) for _ in range(4)]
        for reader in readers:
            reader.start()
        try:
            for _ in range(300):
                manifest.commit_data_dir(manifest.reserve_new_data_dir())
        finally:
            done.set()
            for reader in readers:
                reader.join()
        assert errors == []
        assert manifest.get_latest_data_dir() == tmp_path / "v300"


def _write(data_directory, keys):
    return ParquetDal(data_directory).write(pl.DataFrame({"a": [1, 2]}), *keys)


def _append(data_directory, keys):
    return ParquetDal(data_directory).append(pl.DataFrame({"a": [1, 2]}), *keys)


class TestConcurrency:
    N = 16

    def test_concurrent_writes(self, tmp_path):
        with ProcessPoolExecutor(8) as executor:
            dirs = list(executor.map(_write, [tmp_path] * self.N, [KEYS] * self.N))
        assert len(set(dirs)) == self.N
        assert manifest_contents(ParquetDal(tmp_path), *KEYS) == {
            "latest_data_dir": f"v{self.N - 1}"
        }

    def test_concurrent_appends(self, tmp_path):
        dal = ParquetDal(tmp_path)
        output_dir = _write(tmp_path, KEYS)
        with ProcessPoolExecutor(8) as executor:
            files = list(executor.map(_append, [tmp_path] * self.N, [KEYS] * self.N))
        assert len(set(files)) == self.N
        assert visible_files(output_dir) == sorted(
            f"{i}.parquet" for i in range(self.N + 1)
        )
        assert dal.read(*KEYS).collect().height == 2 * (self.N + 1)


class TestLogging:
    def test_write_append_read_log_their_steps(self, dal, df, caplog):
        caplog.set_level(logging.DEBUG, logger="pyrosome")
        dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        dal.read(*KEYS)
        messages = [record.getMessage() for record in caplog.records]
        assert any(m.startswith("Reserved ") for m in messages)
        assert any(m.startswith("Committed v0 to ") for m in messages)
        assert any(m.startswith("Acquired lock ") for m in messages)
        assert any(m.startswith("Released lock ") for m in messages)
        assert any(m.endswith("1.parquet") and "Appended" in m for m in messages)
        assert any(m.startswith("Reading ") and "2 file(s)" in m for m in messages)
        assert all(record.levelno <= logging.INFO for record in caplog.records)

    def test_write_and_append_log_at_info(self, dal, df, caplog):
        caplog.set_level(logging.INFO, logger="pyrosome")
        output_dir = dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        dal.read(*KEYS)
        assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
            (logging.INFO, f"Wrote {KEYS} as {output_dir}"),
            (logging.INFO, f"Appended to {KEYS} as {output_dir / '1.parquet'}"),
        ]

    def test_stale_commit_is_logged(self, dal, df, caplog):
        caplog.set_level(logging.INFO, logger="pyrosome")
        manifest = DataManifestFile(dal.get_profile_dir(*KEYS))
        dal.write(df, *KEYS)
        dal.write(df, *KEYS)
        manifest.commit_data_dir(manifest.profile_path / "v0")
        assert "already points at newer v1" in caplog.text

    def test_reads_are_silent_at_info(self, dal, df, caplog):
        dal.write(df, *KEYS)
        caplog.set_level(logging.INFO)
        dal.read(*KEYS)
        assert not caplog.records

    def test_package_logger_has_null_handler(self):
        handlers = logging.getLogger("pyrosome").handlers
        assert any(isinstance(h, logging.NullHandler) for h in handlers)

    def test_failed_write_logs_exception(self, dal, df, monkeypatch, caplog):
        def failing_sink(_data, _output_file):
            raise RuntimeError

        monkeypatch.setattr(ParquetDal, "_sink", staticmethod(failing_sink))
        with pytest.raises(RuntimeError):
            dal.write(df, *KEYS)
        [record] = caplog.records
        assert record.levelno == logging.ERROR
        assert record.exc_info is not None
        assert record.getMessage().startswith(f"Failed to write {KEYS}")
        assert "version is removed" in record.getMessage()

    def test_failed_append_logs_exception(self, dal, df, monkeypatch, caplog):
        dal.write(df, *KEYS)

        def failing_sink(_data, _output_file):
            raise RuntimeError

        monkeypatch.setattr(ParquetDal, "_sink", staticmethod(failing_sink))
        with pytest.raises(RuntimeError):
            dal.append(df, *KEYS)
        [record] = caplog.records
        assert record.levelno == logging.ERROR
        assert record.exc_info is not None
        assert record.getMessage().startswith(f"Failed to append to {KEYS}")

    @pytest.mark.parametrize("contents", ["{}", "{"])
    def test_corrupt_manifest_logs_error(self, dal, df, caplog, contents):
        dal.write(df, *KEYS)
        (dal.data_directory.joinpath(*KEYS) / "manifest.json").write_text(contents)
        with pytest.raises(ValueError):
            dal.read(*KEYS)
        [record] = caplog.records
        assert record.levelno == logging.ERROR
        assert "manifest.json is corrupt" in record.getMessage()

    def test_missing_version_dir_logs_exception(self, dal, df, caplog):
        output_dir = dal.write(df, *KEYS)
        delete_data_file(output_dir / "0.parquet")
        output_dir.rmdir()
        with pytest.raises(FileNotFoundError):
            dal.read(*KEYS)
        [record] = caplog.records
        assert record.levelno == logging.ERROR
        assert record.getMessage() == f"Cannot list version {output_dir} of {KEYS}"

    def test_invalid_input_is_not_logged(self, dal, df, caplog):
        dal.write(df, *KEYS)
        with pytest.raises(ValueError):
            dal.append(df.rename({"a": "c"}), *KEYS)
        with pytest.raises(ValueError):
            dal.read("..")
        with pytest.raises(ProfileNotFoundError):
            dal.read("missing")
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

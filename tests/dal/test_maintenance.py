"""Tests for superseded appends, first appends, cleanup after failures, null
columns, and the compact, prune and vacuum maintenance operations."""

import datetime
import logging
import os
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyrosome import _files
from pyrosome.dal import (
    CsvDal,
    DataManifestFile,
    ParquetDal,
    ProfileChangedError,
    ProfileNotFoundError,
    SupersededAppendError,
)
from pyrosome.simple_file_lock import FileAlreadyLockedError, SimpleWriteLock

KEYS = ("profile", "sub")
LONG_AGO = time.time() - 2 * 3600


@pytest.fixture
def dal(tmp_path):
    return ParquetDal(tmp_path)


@pytest.fixture
def df():
    return pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})


def entries(directory: Path) -> list[str]:
    """Every file and directory below directory, as sorted relative paths."""
    return sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*"))


def set_mtime(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


def run_during_sink(monkeypatch, action, *, times=1):
    """Patch ParquetDal._sink so that action runs after the next `times` sinks."""
    real_sink = ParquetDal._sink
    remaining = [times]

    def sink(data, output_file):
        real_sink(data, output_file)
        if remaining[0]:
            remaining[0] -= 1
            # the action's own writes and appends sink normally
            monkeypatch.setattr(ParquetDal, "_sink", staticmethod(real_sink))
            try:
                action()
            finally:
                monkeypatch.setattr(ParquetDal, "_sink", staticmethod(sink))

    monkeypatch.setattr(ParquetDal, "_sink", staticmethod(sink))


class TestSupersededAppend:
    def test_append_racing_write_raises(self, dal, df, monkeypatch):
        v0 = dal.write(df, *KEYS)
        run_during_sink(monkeypatch, lambda: dal.write(df.head(1), *KEYS))
        with pytest.raises(SupersededAppendError) as error:
            dal.append(df, *KEYS)
        assert error.value.published == v0 / "1.parquet"
        assert error.value.latest_data_dir == v0.parent / "v1"
        assert "Append it again" in str(error.value)
        assert_frame_equal(dal.read(*KEYS).collect(), df.head(1))

    def test_superseded_append_can_be_retried(self, dal, df, monkeypatch):
        dal.write(df, *KEYS)
        run_during_sink(monkeypatch, lambda: dal.write(df.head(1), *KEYS))
        with pytest.raises(SupersededAppendError):
            dal.append(df, *KEYS)
        monkeypatch.undo()
        dal.append(df, *KEYS)
        assert dal.read(*KEYS).collect().height == 3

    def test_superseded_append_is_logged(self, dal, df, monkeypatch, caplog):
        dal.write(df, *KEYS)
        run_during_sink(monkeypatch, lambda: dal.write(df, *KEYS))
        with pytest.raises(SupersededAppendError):
            dal.append(df, *KEYS)
        [record] = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert record.getMessage().startswith(f"Failed to append to {KEYS}")
        assert record.exc_info is not None


class TestFirstAppend:
    def test_first_append_writes_v0(self, dal, df):
        with pytest.warns(UserWarning, match="Writing instead"):
            output_dir = dal.append(df, *KEYS)
        assert output_dir == dal.get_profile_dir(*KEYS) / "v0"
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_waits_for_another_process_to_commit_v0(self, dal, df):
        # another process has claimed v0 and commits it a little later
        v0 = dal.get_profile_dir(*KEYS) / "v0"
        v0.mkdir(parents=True)

        def commit():
            df.write_parquet(v0 / "0.parquet")
            DataManifestFile(v0.parent).commit_data_dir(v0)

        timer = threading.Timer(0.2, commit)
        timer.start()
        try:
            published = dal.append(df, *KEYS)
        finally:
            timer.join()
        assert published == v0 / "1.parquet"
        assert dal.read(*KEYS).collect().height == 4

    def test_times_out_on_uncommitted_v0(self, dal, df, monkeypatch, caplog):
        monkeypatch.setattr(ParquetDal, "FIRST_COMMIT_TIMEOUT", 0.1)
        (dal.get_profile_dir(*KEYS) / "v0").mkdir(parents=True)
        with pytest.raises(TimeoutError, match="vacuum"):
            dal.append(df, *KEYS)
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_failed_first_append_removes_v0(self, dal, df, monkeypatch):
        def failing_sink(_data, output_file):
            output_file.write_bytes(b"partial")
            raise RuntimeError

        monkeypatch.setattr(ParquetDal, "_sink", staticmethod(failing_sink))
        with pytest.warns(UserWarning), pytest.raises(RuntimeError):
            dal.append(df, *KEYS)
        assert entries(dal.get_profile_dir(*KEYS)) == []
        monkeypatch.undo()
        with pytest.warns(UserWarning):
            assert dal.append(df, *KEYS).name == "v0"

    def test_first_append_superseded_by_write(self, dal, df, monkeypatch):
        # a write reserves v1 while the first append writes v0, and commits first
        run_during_sink(monkeypatch, lambda: dal.write(df.head(1), *KEYS))
        with pytest.warns(UserWarning), pytest.raises(SupersededAppendError) as error:
            dal.append(df, *KEYS)
        assert error.value.published == dal.get_profile_dir(*KEYS) / "v0" / "0.parquet"
        assert error.value.latest_data_dir.name == "v1"

    def test_profile_with_other_format_writes_new_version(self, dal, df):
        CsvDal(dal.data_directory).write(df, *KEYS)
        with pytest.warns(UserWarning, match="Writing instead"):
            assert dal.append(df, *KEYS).name == "v1"
        assert_frame_equal(dal.read(*KEYS).collect(), df)


def _append_new(data_directory, keys, value):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ParquetDal(data_directory).append(pl.DataFrame({"a": [value]}), *keys)


def test_concurrent_first_appends_keep_every_row(tmp_path):
    n = 16
    with ProcessPoolExecutor(8) as executor:
        files = list(executor.map(_append_new, [tmp_path] * n, [KEYS] * n, range(n)))
    dal = ParquetDal(tmp_path)
    assert sorted(dal.read(*KEYS).collect()["a"].to_list()) == list(range(n))
    profile_dir = dal.get_profile_dir(*KEYS)
    assert all(file.is_relative_to(profile_dir / "v0") for file in files)
    assert sorted(path.name for path in profile_dir.iterdir() if path.is_dir()) == ["v0"]


class TestCleanupAfterFailure:
    @staticmethod
    def failing_sink(_data, output_file):
        output_file.write_bytes(b"partial")
        raise RuntimeError

    def test_failed_write_removes_version(self, dal, df, monkeypatch):
        dal.write(df, *KEYS)
        monkeypatch.setattr(ParquetDal, "_sink", staticmethod(self.failing_sink))
        with pytest.raises(RuntimeError):
            dal.write(df, *KEYS)
        assert entries(dal.get_profile_dir(*KEYS)) == ["manifest.json", "v0", "v0/0.parquet"]

    def test_failed_append_removes_tmp_file(self, dal, df, monkeypatch):
        v0 = dal.write(df, *KEYS)
        monkeypatch.setattr(ParquetDal, "_sink", staticmethod(self.failing_sink))
        with pytest.raises(RuntimeError):
            dal.append(df, *KEYS)
        assert entries(v0) == ["0.parquet"]

    def test_failed_commit_removes_version(self, dal, df, monkeypatch):
        v0 = dal.write(df, *KEYS)
        monkeypatch.setattr(DataManifestFile, "LOCK_TIMEOUT", 0.1)
        with SimpleWriteLock(v0.parent / "manifest.json"):
            with pytest.raises(FileAlreadyLockedError):
                dal.write(df, *KEYS)
        assert not (v0.parent / "v1").exists()

    def test_version_committed_before_failure_is_kept(self, dal, df, monkeypatch):
        def failing_exit(self, *_args):
            self._release()
            raise OSError("release failed")

        monkeypatch.setattr(SimpleWriteLock, "__exit__", failing_exit)
        with pytest.raises(OSError, match="release failed"):
            dal.write(df, *KEYS)
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_failed_manifest_write_removes_tmp_file(self, tmp_path, monkeypatch):
        manifest = DataManifestFile(tmp_path)
        v0 = manifest.reserve_new_data_dir()

        def failing_replace(_self, _target):
            raise OSError("replace failed")

        monkeypatch.setattr(Path, "replace", failing_replace)
        with pytest.raises(OSError, match="replace failed"):
            manifest.commit_data_dir(v0)
        assert entries(tmp_path) == ["v0"]

    def test_cannot_commit_removed_version(self, tmp_path):
        manifest = DataManifestFile(tmp_path)
        v0 = manifest.reserve_new_data_dir()
        v0.rmdir()
        with pytest.raises(FileNotFoundError, match="no longer exists"):
            manifest.commit_data_dir(v0)
        assert not manifest.exists()


class TestNullColumns:
    def test_write_stores_null_columns_as_strings(self, dal):
        dal.write(pl.DataFrame({"a": [1], "s": [None]}), *KEYS)
        assert dal.read(*KEYS).collect_schema()["s"] == pl.String

    def test_later_appends_fill_null_columns(self, dal):
        dal.write(pl.DataFrame({"a": [1], "s": [None]}), *KEYS)
        dal.append(pl.DataFrame({"a": [2], "s": ["x"]}), *KEYS)
        dal.append(pl.DataFrame({"a": [3], "s": [None]}), *KEYS)
        assert dal.read(*KEYS).collect()["s"].to_list() == [None, "x", None]

    def test_first_append_stores_null_columns_as_strings(self, dal):
        with pytest.warns(UserWarning):
            dal.append(pl.DataFrame({"a": [1], "s": [None]}), *KEYS)
        dal.append(pl.DataFrame({"a": [2], "s": ["x"]}), *KEYS)
        assert dal.read(*KEYS).collect()["s"].to_list() == [None, "x"]

    def test_other_types_are_unchanged(self, dal, df):
        dal.write(df, *KEYS)
        assert dal.read(*KEYS).collect_schema() == df.schema


class TestCompact:
    def test_compacts_into_one_file(self, dal, df):
        dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        dal.append(df.head(1), *KEYS)
        expected = dal.read(*KEYS).collect()
        compacted = dal.compact(*KEYS)
        assert compacted.name == "v1"
        assert entries(compacted) == ["0.parquet"]
        assert_frame_equal(dal.read(*KEYS).collect(), expected)

    def test_single_file_is_left_alone(self, dal, df):
        v0 = dal.write(df, *KEYS)
        assert dal.compact(*KEYS) == v0
        assert not (v0.parent / "v1").exists()

    def test_missing_profile(self, dal):
        with pytest.raises(ProfileNotFoundError):
            dal.compact(*KEYS)

    def test_starts_again_when_appended_to_meanwhile(self, dal, df, monkeypatch):
        dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        run_during_sink(monkeypatch, lambda: dal.append(df.head(1), *KEYS))
        compacted = dal.compact(*KEYS)
        assert entries(compacted) == ["0.parquet"]
        assert dal.read(*KEYS).collect().height == 5
        # the discarded attempt left nothing behind
        assert sorted(p.name for p in compacted.parent.iterdir() if p.is_dir()) == ["v0", "v1"]

    def test_starts_again_when_written_meanwhile(self, dal, df, monkeypatch):
        dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        newer = pl.DataFrame({"a": [9], "b": ["z"]})
        run_during_sink(monkeypatch, lambda: dal.write(newer, *KEYS))
        compacted = dal.compact(*KEYS)
        # the newer version has one file, so there is nothing more to compact
        assert compacted.name == "v2"
        assert_frame_equal(dal.read(*KEYS).collect(), newer)

    def test_gives_up_if_always_changing(self, dal, df, monkeypatch):
        dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        monkeypatch.setattr(ParquetDal, "COMPACT_ATTEMPTS", 2)
        run_during_sink(monkeypatch, lambda: dal.append(df, *KEYS), times=2)
        with pytest.raises(ProfileChangedError):
            dal.compact(*KEYS)
        profile_dir = dal.get_profile_dir(*KEYS)
        assert sorted(p.name for p in profile_dir.iterdir() if p.is_dir()) == ["v0"]
        assert dal.read(*KEYS).collect().height == 8

    def test_append_after_compaction_is_superseded(self, dal, df, monkeypatch):
        dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        run_during_sink(monkeypatch, lambda: dal.compact(*KEYS))
        with pytest.raises(SupersededAppendError):
            dal.append(df, *KEYS)
        assert dal.read(*KEYS).collect().height == 4


class TestPrune:
    def test_keeps_latest_versions(self, dal, df):
        for _ in range(4):
            dal.write(df, *KEYS)
        profile_dir = dal.get_profile_dir(*KEYS)
        assert dal.prune(*KEYS, keep=2) == (profile_dir / "v0", profile_dir / "v1")
        assert sorted(p.name for p in profile_dir.iterdir() if p.is_dir()) == ["v2", "v3"]
        assert_frame_equal(dal.read(*KEYS).collect(), df)

    def test_keep_one(self, dal, df):
        dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        v1 = dal.write(df, *KEYS)
        dal.prune(*KEYS, keep=1)
        assert entries(v1.parent) == ["manifest.json", "v1", "v1/0.parquet"]

    def test_nothing_to_prune(self, dal, df):
        dal.write(df, *KEYS)
        assert dal.prune(*KEYS, keep=3) == ()

    def test_keeps_versions_above_latest(self, dal, df):
        dal.write(df, *KEYS)
        dal.write(df, *KEYS)
        in_progress = dal.get_profile_dir(*KEYS) / "v2"
        in_progress.mkdir()
        dal.prune(*KEYS, keep=1)
        assert in_progress.exists()

    @pytest.mark.parametrize("keep", [0, -1])
    def test_keep_must_be_positive(self, dal, df, keep):
        dal.write(df, *KEYS)
        with pytest.raises(ValueError, match="keep must be at least 1"):
            dal.prune(*KEYS, keep=keep)

    def test_missing_profile(self, dal):
        with pytest.raises(ProfileNotFoundError):
            dal.prune(*KEYS, keep=1)


class TestVacuum:
    def test_removes_old_tmp_files(self, dal, df):
        v0 = dal.write(df, *KEYS)
        dal.append(df, *KEYS)
        old_tmp = v0 / ".abc.parquet.tmp"
        old_manifest_tmp = v0.parent / ".abc.json.tmp"
        for path in (old_tmp, old_manifest_tmp):
            path.write_bytes(b"")
            set_mtime(path, LONG_AGO)
        assert dal.vacuum(*KEYS) == (old_manifest_tmp, old_tmp)
        # data files and claim markers are kept, however old
        assert entries(v0) == [".1.parquet.claim", "0.parquet", "1.parquet"]

    def test_keeps_recent_files(self, dal, df):
        v0 = dal.write(df, *KEYS)
        (v0 / ".abc.parquet.tmp").write_bytes(b"")
        (v0.parent / "v1").mkdir()
        assert dal.vacuum(*KEYS) == ()
        assert (v0 / ".abc.parquet.tmp").exists()
        assert (v0.parent / "v1").exists()

    def test_removes_old_uncommitted_versions(self, dal, df):
        v0 = dal.write(df, *KEYS)
        v1 = dal.write(df, *KEYS)
        uncommitted = v0.parent / "v2"
        uncommitted.mkdir()
        (uncommitted / "0.parquet").write_bytes(b"")
        for path in (v0, v1, uncommitted, uncommitted / "0.parquet"):
            set_mtime(path, LONG_AGO)
        assert dal.vacuum(*KEYS) == (uncommitted,)
        # versions below the latest are left to prune
        assert v0.exists()
        assert v1.exists()

    def test_profile_without_manifest(self, dal):
        v0 = dal.get_profile_dir(*KEYS) / "v0"
        v0.mkdir(parents=True)
        set_mtime(v0, LONG_AGO)
        assert dal.vacuum(*KEYS) == (v0,)

    def test_older_than(self, dal, df):
        v0 = dal.write(df, *KEYS)
        (v0 / ".abc.parquet.tmp").write_bytes(b"")
        assert dal.vacuum(*KEYS, older_than=datetime.timedelta(0)) == (
            v0 / ".abc.parquet.tmp",
        )

    def test_missing_profile(self, dal):
        assert dal.vacuum(*KEYS) == ()

    def test_vacuumed_version_cannot_be_committed(self, dal, df):
        manifest = DataManifestFile(dal.get_profile_dir(*KEYS))
        dal.write(df, *KEYS)
        reserved = manifest.reserve_new_data_dir()
        set_mtime(reserved, LONG_AGO)
        dal.vacuum(*KEYS)
        with pytest.raises(FileNotFoundError):
            manifest.commit_data_dir(reserved)
        assert manifest.get_latest_data_dir().name == "v0"


def _read_manifest_until(profile_dir, ready_file, stop_file):
    manifest = DataManifestFile(profile_dir)
    reads = 0
    errors = []
    while not stop_file.exists():
        try:
            manifest.get_latest_data_dir()
            reads += 1
        except Exception as error:  # noqa: BLE001
            errors.append(repr(error))
        if reads == 1:
            ready_file.touch()
    return reads, errors


def test_commit_while_other_processes_read(tmp_path):
    """Readers in other processes must not make a commit fail (review finding SE M1).

    Windows can refuse to replace a file another process has open, so this is
    mainly a check for the Windows CI runner.
    """
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    manifest = DataManifestFile(profile_dir)
    manifest.commit_data_dir(manifest.reserve_new_data_dir())
    stop_file = tmp_path / "stop"
    ready_files = [tmp_path / f"ready{n}" for n in range(4)]
    with ProcessPoolExecutor(len(ready_files)) as executor:
        readers = [
            executor.submit(_read_manifest_until, profile_dir, ready, stop_file)
            for ready in ready_files
        ]
        try:
            deadline = time.monotonic() + 60
            while not all(ready.exists() for ready in ready_files):
                assert time.monotonic() < deadline, "readers did not start"
                time.sleep(0.01)
            for _ in range(300):
                manifest.commit_data_dir(manifest.reserve_new_data_dir())
        finally:
            stop_file.touch()
        results = [reader.result() for reader in readers]
    assert [errors for _, errors in results] == [[]] * len(readers)
    assert all(reads > 0 for reads, _ in results)
    assert manifest.get_latest_data_dir() == profile_dir / "v300"


class TestRetryWhileInUse:
    """Windows refuses to replace or open a file another process has open."""

    @pytest.fixture(autouse=True)
    def on_windows(self, monkeypatch):
        monkeypatch.setattr(_files, "IN_USE_MEANS_RETRY", True)
        monkeypatch.setattr(_files, "IN_USE_POLL_INTERVAL", 0.001)

    @staticmethod
    def fails(times):
        calls = []

        def action():
            calls.append(None)
            if len(calls) <= times:
                raise PermissionError
            return len(calls)

        return action

    def test_retries_until_the_file_is_free(self, tmp_path):
        assert _files.retry_while_in_use(self.fails(3), tmp_path) == 4

    def test_gives_up_after_the_timeout(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_files, "IN_USE_TIMEOUT", 0.01)
        with pytest.raises(PermissionError):
            _files.retry_while_in_use(self.fails(1000), tmp_path)

    def test_other_errors_are_not_retried(self, tmp_path):
        def action():
            raise FileNotFoundError

        with pytest.raises(FileNotFoundError):
            _files.retry_while_in_use(action, tmp_path)

    def test_not_retried_elsewhere(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_files, "IN_USE_MEANS_RETRY", False)
        with pytest.raises(PermissionError):
            _files.retry_while_in_use(self.fails(1), tmp_path)

    def test_commit_retries_a_refused_replace(self, tmp_path, monkeypatch):
        manifest = DataManifestFile(tmp_path)
        manifest.commit_data_dir(manifest.reserve_new_data_dir())
        replace = Path.replace
        refusals = []

        def refuse_twice(self, target):
            if Path(target).name == "manifest.json" and len(refusals) < 2:
                refusals.append(target)
                raise PermissionError
            return replace(self, target)

        monkeypatch.setattr(Path, "replace", refuse_twice)
        manifest.commit_data_dir(manifest.reserve_new_data_dir())
        assert len(refusals) == 2
        assert manifest.get_latest_data_dir() == tmp_path / "v1"

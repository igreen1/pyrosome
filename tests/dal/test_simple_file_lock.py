"""Tests for the simple file lock."""

import datetime
import json
import logging
import os
import socket
import time
from pathlib import Path

import pytest

from pyrosome import _files, simple_file_lock
from pyrosome.dal import DataManifestFile
from pyrosome.simple_file_lock import (
    FileAlreadyLockedError,
    FileLockPermissionError,
    LockError,
    SimpleWriteLock,
    _Holder,
    file_is_locked,
)


def test_lock_creates_and_removes_lock_file(tmp_path):
    path = tmp_path / "data.json"
    with SimpleWriteLock(path) as lock:
        assert isinstance(lock, SimpleWriteLock)
        assert file_is_locked(path)
    assert not file_is_locked(path)


def test_lock_released_on_exception(tmp_path):
    path = tmp_path / "data.json"
    with pytest.raises(RuntimeError), SimpleWriteLock(path):
        raise RuntimeError
    assert not file_is_locked(path)


def test_second_lock_fails_without_timeout(tmp_path):
    path = tmp_path / "data.json"
    with SimpleWriteLock(path):
        with pytest.raises(FileAlreadyLockedError), SimpleWriteLock(path):
            pass
        # the failed lock must not remove the lock held by the first
        assert file_is_locked(path)


def test_lock_waits_until_timeout(tmp_path):
    path = tmp_path / "data.json"
    with SimpleWriteLock(path):
        start = time.monotonic()
        with (
            pytest.raises(FileAlreadyLockedError),
            SimpleWriteLock(path, timeout=0.2, poll_interval=0.01),
        ):
            pass
        assert time.monotonic() - start >= 0.2


def test_lock_acquired_once_released(tmp_path):
    path = tmp_path / "data.json"
    lock = SimpleWriteLock(path)
    lock.__enter__()
    waiter = SimpleWriteLock(path, timeout=5, poll_interval=0.01)
    lock.__exit__(None, None, None)
    with waiter:
        assert file_is_locked(path)


def test_reentering_same_lock_fails(tmp_path):
    lock = SimpleWriteLock(tmp_path / "data.json")
    with lock, pytest.raises(LockError):
        lock.__enter__()


def test_lock_timeout_logs_error(tmp_path, caplog):
    path = tmp_path / "file.txt"
    with SimpleWriteLock(path):
        with pytest.raises(FileAlreadyLockedError), SimpleWriteLock(path):
            pass
    [record] = caplog.records
    assert record.levelno == logging.ERROR
    assert "crashed process" in record.getMessage()


def test_failed_release_logs_exception(tmp_path, monkeypatch, caplog):
    path = tmp_path / "file.txt"

    def failing_unlink(_self, missing_ok=False):
        raise PermissionError

    lock = SimpleWriteLock(path)
    with pytest.raises(PermissionError):
        with lock:
            monkeypatch.setattr(Path, "unlink", failing_unlink)
    monkeypatch.undo()
    [record] = caplog.records
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert record.getMessage().startswith("Cannot release lock ")


def lock_file(path):
    return path.with_name(path.name + ".write_lock")


def write_holder(path, seconds_ago, token="other"):
    created = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
        seconds=seconds_ago
    )
    lock_file(path).write_text(
        json.dumps({"host": "elsewhere", "pid": 1, "created": created.isoformat(), "token": token})
    )


def test_lock_file_names_holder(tmp_path):
    path = tmp_path / "data.json"
    with SimpleWriteLock(path):
        holder = json.loads(lock_file(path).read_text())
    assert holder["host"] == socket.gethostname()
    assert holder["pid"] == os.getpid()
    created = datetime.datetime.fromisoformat(holder["created"])
    assert abs(datetime.datetime.now(datetime.UTC) - created).total_seconds() < 60


def test_stale_lock_is_broken(tmp_path, caplog):
    path = tmp_path / "data.json"
    write_holder(path, seconds_ago=200)
    with SimpleWriteLock(path, stale_after=120):
        assert json.loads(lock_file(path).read_text())["pid"] == os.getpid()
    assert not file_is_locked(path)
    [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "Broke stale lock" in record.getMessage()
    assert "elsewhere" in record.getMessage()
    # the renamed lock file is deleted too
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_fresh_lock_is_not_broken(tmp_path):
    path = tmp_path / "data.json"
    write_holder(path, seconds_ago=5)
    with pytest.raises(FileAlreadyLockedError), SimpleWriteLock(path, stale_after=120):
        pass
    assert file_is_locked(path)


def test_lock_is_never_broken_without_stale_after(tmp_path):
    path = tmp_path / "data.json"
    write_holder(path, seconds_ago=10_000)
    with pytest.raises(FileAlreadyLockedError), SimpleWriteLock(path):
        pass


@pytest.mark.parametrize("contents", ["", "{", '{"created": "yesterday"}'])
def test_lock_without_time_uses_modification_time(tmp_path, contents):
    path = tmp_path / "data.json"
    lock_file(path).write_text(contents)
    with pytest.raises(FileAlreadyLockedError), SimpleWriteLock(path, stale_after=120):
        pass
    long_ago = time.time() - 200
    os.utime(lock_file(path), (long_ago, long_ago))
    with SimpleWriteLock(path, stale_after=120):
        pass


def test_breaking_lock_taken_meanwhile_puts_it_back(tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    # another process broke the stale lock and took it after we checked its age
    write_holder(path, seconds_ago=0, token="fresh")
    fresh = lock_file(path).read_text()
    monkeypatch.setattr(
        SimpleWriteLock, "_holder", lambda _self: _Holder("stale contents", 1000.0)
    )
    with pytest.raises(FileAlreadyLockedError), SimpleWriteLock(path, stale_after=120):
        pass
    assert lock_file(path).read_text() == fresh
    assert sorted(p.name for p in tmp_path.iterdir()) == [lock_file(path).name]


def test_release_leaves_lock_taken_by_another_process(tmp_path, caplog):
    path = tmp_path / "data.json"
    with SimpleWriteLock(path):
        # held so long that another process broke it as stale and took it
        write_holder(path, seconds_ago=0)
    assert file_is_locked(path)
    assert "broken as stale" in caplog.text


def test_manifest_lock_is_broken_after_two_minutes(tmp_path):
    lock = DataManifestFile(tmp_path).lock()
    assert lock.stale_after == 120
    assert lock.timeout == DataManifestFile.LOCK_TIMEOUT


def test_permission_error_raises_immediately(tmp_path, monkeypatch):
    monkeypatch.setattr(simple_file_lock, "_PERMISSION_ERROR_MEANS_HELD", False)

    def denied(*_args, **_kwargs):
        raise PermissionError

    monkeypatch.setattr(Path, "open", denied)
    start = time.monotonic()
    with pytest.raises(FileLockPermissionError), SimpleWriteLock(tmp_path / "f", timeout=5):
        pass
    assert time.monotonic() - start < 1


def test_permission_error_on_windows_means_held(tmp_path, monkeypatch):
    monkeypatch.setattr(simple_file_lock, "_PERMISSION_ERROR_MEANS_HELD", True)
    real_open = Path.open
    denials = [3]

    def denied_at_first(self, *args, **kwargs):
        if denials[0]:
            denials[0] -= 1
            raise PermissionError
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_at_first)
    path = tmp_path / "data.json"
    with SimpleWriteLock(path, timeout=5, poll_interval=0.01):
        assert file_is_locked(path)
    assert denials == [0]


def test_permission_error_on_windows_raises_after_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(simple_file_lock, "_PERMISSION_ERROR_MEANS_HELD", True)

    def denied(*_args, **_kwargs):
        raise PermissionError

    monkeypatch.setattr(Path, "open", denied)
    start = time.monotonic()
    with (
        pytest.raises(FileLockPermissionError),
        SimpleWriteLock(tmp_path / "f", timeout=0.2, poll_interval=0.01),
    ):
        pass
    assert time.monotonic() - start >= 0.2


def test_waiting_reads_a_fresh_lock_file_once(tmp_path, monkeypatch):
    """Reading the lock file stops its holder deleting it on Windows."""
    path = tmp_path / "data.json"
    write_holder(path, seconds_ago=0)
    reads = []
    read_text = simple_file_lock._read_text

    def counted(file):
        reads.append(file)
        return read_text(file)

    monkeypatch.setattr(simple_file_lock, "_read_text", counted)
    lock = SimpleWriteLock(path, timeout=0.2, poll_interval=0.01, stale_after=120)
    with pytest.raises(FileAlreadyLockedError):
        lock.__enter__()
    assert reads == [lock_file(path)]


def test_release_retries_while_windows_has_the_file_open(tmp_path, monkeypatch):
    monkeypatch.setattr(_files, "IN_USE_MEANS_RETRY", True)
    monkeypatch.setattr(_files, "IN_USE_POLL_INTERVAL", 0.001)
    path = tmp_path / "data.json"
    unlink = Path.unlink
    refusals = []

    def refuse_twice(self, *args, **kwargs):
        if self == lock_file(path) and len(refusals) < 2:
            refusals.append(self)
            raise PermissionError
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_twice)
    with SimpleWriteLock(path):
        pass
    assert len(refusals) == 2
    assert not file_is_locked(path)

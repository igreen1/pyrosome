"""Tests for the simple file lock."""

import time

import pytest

from salp.dal.simple_file_lock import (
    FileAlreadyLockedError,
    LockError,
    SimpleWriteLock,
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

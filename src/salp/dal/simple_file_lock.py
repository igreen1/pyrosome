"""Create a simple lock on a file.

Nothing can read or write to a file while it is locked.
This prevents multiple processes from reading or writing to the same file at the same time,
 which could cause data corruption.
"""

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


_WRITE_LOCK_SUFFIX = ".write_lock"


class LockError(Exception):
    """An error that occurs during file locking."""


class FileAlreadyLockedError(LockError):
    """An error that occurs when a file is already locked and a new lock cannot be made."""


class FileLockPermissionError(LockError):
    """An error that occurs when a file cannot be locked due to permission issues."""


class FileLockedError(LockError):
    """An error that occurs when a file is locked and cannot be accessed."""


class SimpleWriteLock:
    """Simple lock that prevents other processes from accessing a file being written to."""

    def __init__(self, path: Path, timeout: float = 0.0, poll_interval: float = 0.05):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + _WRITE_LOCK_SUFFIX)
        self.locked = False
        self.timeout = timeout
        """Seconds to keep retrying if another process holds the lock."""
        self.poll_interval = poll_interval

    def __enter__(self):
        if self.locked:
            msg = f"File {self.path} is already locked."
            raise LockError(msg)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.lock_path.touch(exist_ok=False)
                self.locked = True
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise FileAlreadyLockedError(self.path) from None
                time.sleep(self.poll_interval)
            except OSError as e:
                raise FileLockPermissionError(self.path) from e

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        if self.locked:
            self.lock_path.unlink()
            self.locked = False


def file_is_locked(path: Path) -> bool:
    """Check if a file is locked."""
    lock_path = path.with_suffix(path.suffix + _WRITE_LOCK_SUFFIX)
    return lock_path.exists()

"""Create a simple lock on a file.

Nothing can read or write to a file while it is locked.
This prevents multiple processes from reading or writing to the same file at the
same time, which could cause data corruption.
"""

import math
import time
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


_WRITE_LOCK_SUFFIX = ".write_lock"


class LockError(Exception):
    """An error that occurs during file locking."""


class FileAlreadyLockedError(LockError):
    """An error that occurs when a file is already locked and cannot be locked again."""


class FileLockPermissionError(LockError):
    """An error that occurs when a file cannot be locked due to permission issues."""


class FileLockedError(LockError):
    """An error that occurs when a file is locked and cannot be accessed."""


class SimpleWriteLock:
    """Simple lock that prevents other processes accessing a file being written to."""

    def __init__(
        self, path: Path, timeout: float = 0.0, poll_interval: float = 0.05
    ) -> None:
        """Create a lock for path. The lock is only taken when entering the context.

        Args:
            path: The file to lock.
            timeout: Seconds to keep retrying if another process holds the lock.
            poll_interval: Seconds to wait between attempts to take the lock.
        """
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + _WRITE_LOCK_SUFFIX)
        self.locked = False
        self.timeout = timeout
        """Seconds to keep retrying if another process holds the lock."""
        self.poll_interval = poll_interval

    def __enter__(self) -> Self:
        """Take the lock, retrying until the timeout if another process holds it.

        Returns:
            This lock.

        Raises:
            LockError: If this lock is already held.
            FileAlreadyLockedError: If another process still holds the lock after the
                timeout.
        """
        if self.locked:
            msg = f"File {self.path} is already locked."
            raise LockError(msg)
        attempts = 1 + math.ceil(self.timeout / self.poll_interval)
        for attempt in range(attempts):
            if attempt:
                time.sleep(self.poll_interval)
            if self._try_acquire():
                return self
        raise FileAlreadyLockedError(self.path)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the lock."""
        del exc_type, exc_value, traceback
        if self.locked:
            self.lock_path.unlink()
            self.locked = False

    def _try_acquire(self) -> bool:
        """Try to take the lock once.

        Returns:
            Whether the lock was taken.

        Raises:
            FileLockPermissionError: If the lock file cannot be created.
        """
        try:
            self.lock_path.touch(exist_ok=False)
        except FileExistsError:
            return False
        except OSError as e:
            raise FileLockPermissionError(self.path) from e
        self.locked = True
        return True


def file_is_locked(path: Path) -> bool:
    """Check if a file is locked.

    Args:
        path: The file to check.

    Returns:
        Whether a lock is currently held on path.
    """
    lock_path = path.with_suffix(path.suffix + _WRITE_LOCK_SUFFIX)
    return lock_path.exists()

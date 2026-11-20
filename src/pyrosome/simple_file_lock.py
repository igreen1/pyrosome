"""Create a simple lock on a file.

Nothing can read or write to a file while it is locked.
This prevents multiple processes from reading or writing to the same file at the
same time, which could cause data corruption.
"""

import contextlib
import datetime as dt
import json
import logging
import math
import os
import socket
import sys
import time
import uuid
from typing import TYPE_CHECKING, Final, NamedTuple, Self, cast

from pyrosome._files import retry_while_in_use
from pyrosome._logging import logged_failure

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from types import TracebackType


_logger = logging.getLogger(__name__)

_WRITE_LOCK_SUFFIX = ".write_lock"

# Windows denies creating a file whose previous version is still being deleted,
# which happens while another process (or a virus scanner) has the old lock file
# open, so a denied create there usually means the lock is still held
_WINDOWS: Final[str] = "win32"
_PERMISSION_ERROR_MEANS_HELD: Final[bool] = sys.platform == _WINDOWS


class LockError(Exception):
    """An error that occurs during file locking."""


class FileAlreadyLockedError(LockError):
    """An error that occurs when a file is already locked and cannot be locked again."""


class FileLockPermissionError(LockError):
    """An error that occurs when a file cannot be locked due to permission issues."""


class _Holder(NamedTuple):
    """What a lock file says about the process holding it."""

    contents: str
    """The lock file's contents, which identify this particular hold."""

    age: float
    """Seconds since the lock was taken."""


def _lock_contents() -> str:
    """Describe this process, to be written into a lock file it takes.

    Returns:
        JSON naming the host, process id and time, plus a token unique to this hold
        of the lock.
    """
    holder: dict[str, str | int] = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "created": dt.datetime.now(dt.UTC).isoformat(),
        "token": uuid.uuid4().hex,
    }
    return json.dumps(holder)


def _created_time(contents: str) -> dt.datetime | None:
    """Get the time a lock was taken from its lock file's contents.

    Args:
        contents: The lock file's contents.

    Returns:
        When the lock was taken, or None if contents don't say, such as a lock file
        that is still being written or was created by an older version of
        pyrosome.
    """
    try:
        holder = cast("object", json.loads(contents))
    except ValueError:
        return None
    match holder:
        case {"created": str() as created}:
            return _parse_time(created)
        case _:
            return None


def _parse_time(text: str) -> dt.datetime | None:
    """Parse a time zone aware ISO 8601 time.

    Args:
        text: The time to parse.

    Returns:
        The time, or None if text is not a time zone aware ISO 8601 time.
    """
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _read_text(path: Path) -> str | None:
    """Read a lock file.

    Args:
        path: The lock file.

    Returns:
        Its contents, or None if it doesn't exist.

    Raises:
        OSError: If the file exists but cannot be read, on Windows only once it
            has been in use by other processes for
            `pyrosome._files.IN_USE_TIMEOUT`.
    """
    try:
        return retry_while_in_use(lambda: path.read_text(encoding="utf-8"), path)
    except FileNotFoundError:
        return None


def _modified_time(path: Path) -> float | None:
    """Get when a file was last modified.

    Args:
        path: The file.

    Returns:
        The modification time, in seconds since the epoch, or None if the file
        doesn't exist.

    Raises:
        OSError: If the file exists but cannot be inspected.
    """
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _rename_if_exists(source: Path, target: Path) -> bool:
    """Rename source to target, if source still exists.

    Args:
        source: The file to rename.
        target: The new name, which must not exist.

    Returns:
        Whether source was renamed, False if it no longer existed.

    Raises:
        OSError: If source exists but cannot be renamed.
    """
    try:
        retry_while_in_use(lambda: source.rename(target), source)
    except FileNotFoundError:
        return False
    return True


def _create_exclusive(path: Path, contents: str) -> bool:
    """Create path holding contents, unless it already exists.

    Args:
        path: The file to create.
        contents: What to write into it.

    Returns:
        Whether the file was created, False if it already existed.

    Raises:
        OSError: If the file cannot be created or written for any other reason.
    """
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError:
        return False
    with _removed_on_failure(path), handle:
        handle.write(contents)
    return True


@contextlib.contextmanager
def _removed_on_failure(path: Path) -> Generator[None]:
    """Remove path if the block raises, then re-raise.

    Args:
        path: The file to remove.

    Yields:
        None, once.

    Raises:
        Exception: Anything the block raises.
    """
    try:
        yield
    except Exception:
        path.unlink(missing_ok=True)
        raise


class SimpleWriteLock:
    """Simple lock that prevents other processes accessing a file being written to.

    The lock is a `<file>.write_lock` file next to the locked file, so it works
    across processes but is only advisory: code that ignores the lock is not
    blocked. The lock file names the host and process holding it, and when it was
    taken.

    Example:
        >>> from pathlib import Path
        >>> from pyrosome.simple_file_lock import SimpleWriteLock, file_is_locked
        >>> target = Path("notes.txt")
        >>> with SimpleWriteLock(target, timeout=1.0):
        ...     file_is_locked(target)
        True
        >>> file_is_locked(target)
        False
    """

    def __init__(
        self,
        path: Path,
        timeout: float = 0.0,
        poll_interval: float = 0.05,
        stale_after: float | None = None,
    ) -> None:
        """Create a lock for path. The lock is only taken when entering the context.

        Args:
            path: The file to lock.
            timeout: Seconds to keep retrying if another process holds the lock.
            poll_interval: Seconds to wait between attempts to take the lock.
            stale_after: Seconds after which a lock held by another process is
                assumed to have been left by a process that died, and is broken.
                None never breaks a lock. Ages are measured with the clock of the
                host that took the lock, so allow for clock differences between
                hosts sharing a file system.

        Returns:
            None.

        Raises:
            ValueError: If path has an empty name, such as `/`.
        """
        self.path = path
        # checks path has a name, so lock_path never raises
        path.with_suffix(path.suffix + _WRITE_LOCK_SUFFIX)
        self.locked = False
        self.timeout = timeout
        """Seconds to keep retrying if another process holds the lock."""
        self.poll_interval = poll_interval
        self.stale_after = stale_after
        """Seconds after which another process's lock is broken, or None for never."""
        self._contents = ""
        self._permission_error: PermissionError | None = None

    @property
    def lock_path(self) -> Path:
        """The lock file, next to the locked file.

        Returns:
            The path of the lock file.
        """
        return self.path.with_suffix(self.path.suffix + _WRITE_LOCK_SUFFIX)

    def __enter__(self) -> Self:
        """Take the lock, retrying until the timeout if another process holds it.

        Returns:
            This lock.

        Raises:
            LockError: If this lock is already held.
            FileAlreadyLockedError: If another process still holds the lock after the
                timeout.
            FileLockPermissionError: If the lock file cannot be created because
                permission is denied. On Windows, where that also happens while
                another process is releasing the lock, only once the timeout has
                passed.
            OSError: If the lock file cannot be created for any other reason, such
                as a missing directory, or a stale lock cannot be broken.
        """
        if self.locked:
            msg = f"File {self.path} is already locked."
            raise LockError(msg)
        attempts = 1 + math.ceil(self.timeout / self.poll_interval)
        # reading the lock file stops its holder deleting it on Windows, so it is
        # only read again once the lock last seen could have become stale
        check_stale_at = time.monotonic()
        for attempt in range(attempts):
            if attempt:
                time.sleep(self.poll_interval)
            if self._try_acquire():
                acquired = True
            elif time.monotonic() < check_stale_at:
                acquired = False
            else:
                stale_in = self._break_if_stale()
                check_stale_at = time.monotonic() + stale_in
                acquired = not stale_in and self._try_acquire()
            if acquired:
                _logger.debug(
                    "Acquired lock %s after %d attempt(s)", self.lock_path, attempt + 1
                )
                return self
            if not attempt:
                _logger.debug(
                    "Lock %s is held by another process, waiting up to %ss",
                    self.lock_path,
                    self.timeout,
                )
        if self._permission_error is not None:
            _logger.error(
                "No permission to create lock %s after %d attempt(s) over %ss",
                self.lock_path,
                attempts,
                self.timeout,
            )
            raise FileLockPermissionError(self.path) from self._permission_error
        _logger.error(
            "Gave up on lock %s after %d attempt(s) over %ss. If no other process is "
            "writing, it was left by a crashed process and can be deleted",
            self.lock_path,
            attempts,
            self.timeout,
        )
        raise FileAlreadyLockedError(self.path)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the lock.

        If the lock was held for so long that another process broke it as stale,
        the other process's lock file is left in place.

        Args:
            exc_type: Type of the exception raised in the context, if any. Unused.
            exc_value: The exception raised in the context, if any. Unused.
            traceback: Traceback of the exception raised in the context, if any.
                Unused.

        Returns:
            None, so exceptions raised in the context are never suppressed.

        Raises:
            OSError: If the lock file cannot be read or removed.
        """
        del exc_type, exc_value, traceback
        if self.locked:
            with logged_failure(
                _logger,
                "Cannot release lock %s: other processes wait for it until it is "
                "deleted",
                self.lock_path,
            ):
                self._release()
            self.locked = False

    def _release(self) -> None:
        """Delete the lock file, if it is still this lock's.

        Returns:
            None.

        Raises:
            OSError: If the lock file cannot be read or removed.
        """
        if _read_text(self.lock_path) != self._contents:
            _logger.error(
                "Lock %s was broken as stale while held, so it is not released",
                self.lock_path,
            )
            return
        retry_while_in_use(self.lock_path.unlink, self.lock_path)
        _logger.debug("Released lock %s", self.lock_path)

    def _try_acquire(self) -> bool:
        """Try to take the lock once.

        Returns:
            Whether the lock was taken.

        Raises:
            FileLockPermissionError: If the lock file cannot be created because
                permission is denied, except on Windows.
            OSError: If the lock file cannot be created for any other reason, such
                as a missing directory.
        """
        contents = _lock_contents()
        try:
            created = _create_exclusive(self.lock_path, contents)
        except PermissionError as e:
            if _PERMISSION_ERROR_MEANS_HELD:
                _logger.debug("Permission denied creating %s, retrying", self.lock_path)
                self._permission_error = e
                return False
            _logger.exception("No permission to create lock %s", self.lock_path)
            raise FileLockPermissionError(self.path) from e
        self._permission_error = None
        if created:
            self._contents = contents
            self.locked = True
        return created

    def _holder(self) -> _Holder | None:
        """Read who holds the lock, and for how long.

        Returns:
            The holder, or None if the lock is not held.

        Raises:
            OSError: If the lock file exists but cannot be read.
        """
        if (contents := _read_text(self.lock_path)) is None:
            return None
        if (created := _created_time(contents)) is not None:
            return _Holder(
                contents, (dt.datetime.now(dt.UTC) - created).total_seconds()
            )
        # a lock file being written, or written by an older version of pyrosome
        if (modified := _modified_time(self.lock_path)) is None:
            return None
        return _Holder(contents, time.time() - modified)

    def _break_if_stale(self) -> float:
        """Delete the lock file if it is older than `stale_after`.

        The lock file is renamed before it is deleted, so of several processes
        breaking the same stale lock only one succeeds. If that rename caught a
        fresh lock, taken after the stale one was checked, it is put back.

        Returns:
            Seconds until the lock could become stale, so it need not be read
            again before then: 0 if it was broken, is no longer held or was broken
            by another process, and infinity without `stale_after`. A lock taken
            after this one is only younger, so waiting that long is safe.

        Raises:
            OSError: If the lock file cannot be read, renamed or deleted.
        """
        if self.stale_after is None:
            return math.inf
        if (holder := self._holder()) is None:
            return 0.0
        if holder.age < self.stale_after:
            return self.stale_after - holder.age
        broken = self.lock_path.with_name(
            f".{self.lock_path.name}.{uuid.uuid4().hex}.stale"
        )
        if not _rename_if_exists(self.lock_path, broken):
            return 0.0
        if (contents := _read_text(broken)) != holder.contents:
            # another process broke the stale lock and took it in the meantime
            if not _create_exclusive(self.lock_path, contents or ""):
                _logger.error(
                    "Could not restore lock %s, renamed while breaking a stale lock",
                    self.lock_path,
                )
            broken.unlink()
            return 0.0
        broken.unlink()
        _logger.warning(
            "Broke stale lock %s, taken %.0fs ago by %s",
            self.lock_path,
            holder.age,
            holder.contents,
        )
        return 0.0


def file_is_locked(path: Path) -> bool:
    """Check if a file is locked.

    Args:
        path: The file to check.

    Returns:
        Whether a lock is currently held on path.

    Raises:
        ValueError: If path has an empty name, such as `/`.
    """
    lock_path = path.with_suffix(path.suffix + _WRITE_LOCK_SUFFIX)
    return lock_path.exists()

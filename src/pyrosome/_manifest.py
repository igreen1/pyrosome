"""The manifest naming the latest version of a profile."""

import contextlib
import json
import logging
from typing import TYPE_CHECKING, Final, cast

from pyrosome._files import (
    claim_first_free,
    remove_quietly,
    remove_tree,
    retry_while_in_use,
    temporary_file,
    version_dir_index,
    version_dirs,
)
from pyrosome._keys import DATA_DIR_PATTERN, MANIFEST_FILE_NAME
from pyrosome._logging import logged_failure
from pyrosome.simple_file_lock import SimpleWriteLock

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

# logged as part of pyrosome.dal, the logger users configure to see storage steps
_logger = logging.getLogger("pyrosome.dal")


def discard_version(manifest: DataManifestFile, version_dir: Path) -> None:
    """Delete a version this process reserved, unless the manifest points at it.

    Nothing else ever writes into a version before it is committed, so deleting
    an uncommitted one only affects the process that reserved it.

    Args:
        manifest: The profile's manifest.
        version_dir: The version to delete.

    Returns:
        None.
    """
    try:
        latest_name = manifest.latest_version_name()
    except ValueError, OSError:
        # can't tell whether version_dir was committed, so keep it
        _logger.debug("Keeping %s: the manifest cannot be read", version_dir)
        return
    if latest_name == version_dir.name:
        _logger.debug("Keeping %s: it is committed", version_dir)
        return
    remove_quietly(remove_tree, version_dir)


@contextlib.contextmanager
def discarded_on_failure(
    manifest: DataManifestFile, version_dir: Path
) -> Generator[None]:
    """Delete a reserved version if the block raises, then re-raise.

    Args:
        manifest: The profile's manifest.
        version_dir: The version being written.

    Yields:
        None, once.

    Raises:
        Exception: Anything the block raises.
    """
    try:
        yield
    except Exception:
        discard_version(manifest, version_dir)
        raise


class DataManifestFile:
    """A manifest file that records the latest version of a profile."""

    profile_path: Path
    """Path to the profile directory."""

    MANIFEST_FILE_NAME: Final[str] = MANIFEST_FILE_NAME
    """Name of the manifest file inside the profile directory."""

    LOCK_TIMEOUT: Final[float] = 10.0
    """Seconds to wait for another process to finish writing the manifest."""

    LOCK_STALE_AFTER: Final[float] = 120.0
    """Seconds after which a manifest lock is assumed to have been left by a process
    that died, and is broken. Updating the manifest takes milliseconds."""

    def __init__(self, profile_path: Path) -> None:
        """Create a handle on the manifest in profile_path. Nothing is read yet.

        Args:
            profile_path: Path to the profile directory.

        Returns:
            None.
        """
        self.profile_path = profile_path
        self.manifest_path = profile_path / self.MANIFEST_FILE_NAME

    def exists(self) -> bool:
        """Does the manifest file even exist yet?

        Returns:
            Whether the manifest file exists.
        """
        return self.manifest_path.exists()

    def get_latest_data_dir(self) -> Path:
        """Get the latest data directory described by the manifest file.

        Returns:
            The latest data directory.

        Raises:
            FileNotFoundError: If the manifest file does not exist. The directory it
                names is not checked, so listing it raises if it is missing.
            ValueError: If the manifest file is corrupt.
        """
        return self.profile_path / self._read_latest_data_dir_name()

    def latest_version_name(self) -> str | None:
        """Get the name of the latest version, if one has been committed.

        Returns:
            The latest version's name, such as ``v3``, or None if there is no
            manifest.

        Raises:
            ValueError: If the manifest file is corrupt.
            OSError: If the manifest file exists but cannot be read.
        """
        try:
            return self._read_latest_data_dir_name()
        except FileNotFoundError:
            return None

    def latest_version_index(self) -> int | None:
        """Get the index of the latest version, if one has been committed.

        Returns:
            The latest version's index, or None if there is no manifest.

        Raises:
            ValueError: If the manifest file is corrupt.
            OSError: If the manifest file exists but cannot be read.
        """
        if (name := self.latest_version_name()) is None:
            return None
        return self.version_index(name)

    def reserve_new_data_dir(self) -> Path:
        """Create and return the next version of the data directory.

        The manifest is not updated; call `commit_data_dir` once the data is written.

        Returns:
            The new, empty data directory.

        Raises:
            OSError: If the profile directory cannot be listed or written to.
        """
        existing = version_dirs(self.profile_path)
        next_dir_idx = max(existing, default=-1) + 1
        idx = claim_first_free(
            next_dir_idx,
            # mkdir is atomic, so concurrent writers never share a directory
            lambda idx: (self.profile_path / f"v{idx}").mkdir(exist_ok=False),
        )
        data_dir = self.profile_path / f"v{idx}"
        _logger.debug("Reserved %s (%d existing version(s))", data_dir, len(existing))
        return data_dir

    def commit_data_dir(
        self, data_dir: Path, only_if: Callable[[str | None], bool] | None = None
    ) -> bool:
        """Point the manifest at data_dir, unless it already points at a newer version.

        Args:
            data_dir: The finished data directory.
            only_if: Called with the name of the latest version (None if there is
                none) while holding the lock. The manifest is only updated if it
                returns True.

        Returns:
            Whether the manifest now points at data_dir.

        Raises:
            ValueError: If data_dir or the version in the manifest has an invalid
                name, or the manifest file is corrupt.
            FileNotFoundError: If data_dir no longer exists, such as when `vacuum`
                deleted it.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than `LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            OSError: If the manifest file cannot be written.
        """
        new_idx = self.version_index(data_dir.name)
        with self.lock():
            current_name = self.latest_version_name()
            if only_if is not None and not only_if(current_name):
                _logger.debug("Not committing %s: its condition failed", data_dir.name)
                return False
            if current_name is not None and self.version_index(current_name) >= new_idx:
                _logger.info(
                    "Not committing %s: manifest %s already points at newer %s",
                    data_dir.name,
                    self.manifest_path,
                    current_name,
                )
                return current_name == data_dir.name
            if not data_dir.is_dir():
                msg = f"Cannot commit {data_dir}: it no longer exists"
                raise FileNotFoundError(msg)
            self._write(data_dir.name)
            _logger.debug(
                "Committed %s to %s (was %s)",
                data_dir.name,
                self.manifest_path,
                current_name,
            )
            return True

    @staticmethod
    def version_index(name: str) -> int:
        """Get the version index from a data directory name like 'v3'.

        Args:
            name: The data directory name.

        Returns:
            The version index.

        Raises:
            ValueError: If name is not a valid data directory name.
        """
        if (idx := version_dir_index(name)) is None:
            msg = f"Invalid data directory name {name!r}"
            raise ValueError(msg)
        return idx

    def _read_latest_data_dir_name(self) -> str:
        """Read the latest data directory name from the manifest file.

        Returns:
            The latest data directory name.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
            PermissionError: If the manifest file cannot be read, on Windows only
                once it has been in use for `pyrosome._files.IN_USE_TIMEOUT`.
            json.JSONDecodeError: If the manifest file is not valid JSON.
            ValueError: If the manifest file has no valid latest data directory.
        """
        # writes replace the file atomically, so no lock is needed to read it
        text = retry_while_in_use(self.manifest_path.read_text, self.manifest_path)
        with logged_failure(_logger, "Manifest file %s is corrupt", self.manifest_path):
            manifest = cast("object", json.loads(text))
        match manifest:
            case {"latest_data_dir": str() as name} if DATA_DIR_PATTERN.fullmatch(name):
                return name
            case _:
                msg = (
                    f"Manifest file {self.manifest_path} is corrupt: "
                    "missing or invalid 'latest_data_dir'."
                )
                _logger.error(msg)
                raise ValueError(msg)

    def _write(self, latest_data_dir: str) -> None:
        """Atomically replace the manifest file. The caller must hold the lock.

        Args:
            latest_data_dir: Name of the latest data directory.

        Returns:
            None.

        Raises:
            OSError: If the manifest file cannot be written, including when, on
                Windows, readers keep it open for longer than
                `pyrosome._files.IN_USE_TIMEOUT`.
        """
        manifest: dict[str, str] = {"latest_data_dir": latest_data_dir}
        with temporary_file(self.profile_path, ".json") as tmp_path:
            tmp_path.write_text(json.dumps(manifest, indent=4))
            retry_while_in_use(
                lambda: tmp_path.replace(self.manifest_path), self.manifest_path
            )

    def lock(self) -> SimpleWriteLock:
        """Lock the manifest file, as every update to it does.

        A lock left by a process that died is broken after `LOCK_STALE_AFTER`
        seconds.

        Returns:
            A lock on the manifest file, taken when entered.
        """
        return SimpleWriteLock(
            self.manifest_path,
            timeout=self.LOCK_TIMEOUT,
            stale_after=self.LOCK_STALE_AFTER,
        )

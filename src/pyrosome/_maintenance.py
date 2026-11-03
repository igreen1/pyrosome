"""Tidying up stored profiles: compacting, pruning and vacuuming."""

import datetime as dt
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final

from pyrosome._files import last_modified, remove_tmp_file, remove_tree, version_dirs
from pyrosome._keys import matches_case_on_disk
from pyrosome._manifest import DataManifestFile, discard_version

if TYPE_CHECKING:
    from pathlib import Path

# logged as part of pyrosome.dal, the logger users configure to see storage steps
_logger = logging.getLogger("pyrosome.dal")

# hidden tmp files left by writes, appends and manifest updates
_TMP_FILE_PATTERN = re.compile(r"\..+\.tmp")


class ProfileChangedError(RuntimeError):
    """A profile kept changing while it was being compacted."""


class Maintenance(ABC):
    """Operations that tidy up the profiles a file DAL stores."""

    data_directory: Path
    """The directory where data files are stored."""

    COMPACT_ATTEMPTS: Final[int] = 5
    """How many times `compact` starts again when the profile changes meanwhile."""

    def compact(self, *keys: str) -> Path:
        """Rewrite the latest version as a new version with a single data file.

        Many small appended files make reads slower. If another process appends
        to the latest version, or commits a newer one, while compacting, the new
        version is discarded and compacting starts again, up to
        `COMPACT_ATTEMPTS` times.

        Example:
            >>> from pathlib import Path
            >>> import polars as pl
            >>> from pyrosome import ParquetDal
            >>> dal = ParquetDal(Path("data"))
            >>> _ = dal.write(pl.DataFrame({"x": [1]}), "example")
            >>> _ = dal.append(pl.DataFrame({"x": [2]}), "example")
            >>> compacted = dal.compact("example")
            >>> compacted.name, sorted(path.name for path in compacted.iterdir())
            ('v1', ['0.parquet'])
            >>> dal.read("example").collect()["x"].to_list()
            [1, 2]

        Args:
            *keys: Path components naming the profile.

        Returns:
            The new version, or the latest version if it already has a single data
            file.

        Raises:
            pyrosome.dal.ProfileNotFoundError: If there is no data for keys.
            ProfileChangedError: If the profile changed during every attempt.
            ValueError: If there are no keys, a key is invalid or leads outside the
                data directory, or the profile's manifest file is corrupt.
            FileNotFoundError: If the manifest names a missing version directory.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `pyrosome.dal.DataManifestFile.LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            polars.exceptions.PolarsError: If the data cannot be read or written.
            OSError: If the profile directory cannot be written to.
        """
        for attempt in range(self.COMPACT_ATTEMPTS):
            source = self._require_latest_data(*keys)
            source_dir = source[0].parent
            if len(source) == 1:
                _logger.debug("Not compacting %s: it has one data file", keys)
                return source_dir
            manifest = DataManifestFile(source_dir.parent)
            output_dir = manifest.reserve_new_data_dir()
            if self._rewrite_version(source, keys, manifest, output_dir):
                _logger.info(
                    "Compacted %s: %d file(s) in %s into %s",
                    keys,
                    len(source),
                    source_dir.name,
                    output_dir,
                )
                return output_dir
            discard_version(manifest, output_dir)
            _logger.debug(
                "%s changed while compacting it (attempt %d)", keys, attempt + 1
            )
        msg = (
            f"{keys} changed during each of {self.COMPACT_ATTEMPTS} attempts to "
            "compact it"
        )
        raise ProfileChangedError(msg)

    def prune(self, *keys: str, keep: int) -> tuple[Path, ...]:
        """Delete all but the newest versions of a profile.

        Only do this when no reader can still be using the deleted versions, since
        a lazy frame from an earlier `read` fails if its files disappear.

        Example:
            >>> from pathlib import Path
            >>> import polars as pl
            >>> from pyrosome import ParquetDal
            >>> dal = ParquetDal(Path("data"))
            >>> for x in range(4):
            ...     _ = dal.write(pl.DataFrame({"x": [x]}), "example")
            >>> [path.name for path in dal.prune("example", keep=2)]
            ['v0', 'v1']

        Args:
            *keys: Path components naming the profile.
            keep: How many versions to keep: the latest and those numbered just
                below it. Versions above the latest are always kept, because
                another process may still be writing them.

        Returns:
            The deleted version directories, oldest first.

        Raises:
            ValueError: If keep is less than 1, there are no keys, a key is invalid
                or leads outside the data directory, or the profile's manifest file
                is corrupt.
            pyrosome.dal.ProfileNotFoundError: If there is no data for keys.
            FileNotFoundError: If the manifest names a missing version directory.
            OSError: If a version cannot be deleted.
        """
        if keep < 1:
            msg = f"keep must be at least 1, not {keep}"
            raise ValueError(msg)
        latest_dir = self._require_latest_data(*keys)[0].parent
        oldest_kept = DataManifestFile.version_index(latest_dir.name) - keep + 1
        removed = tuple(
            path
            for idx, path in sorted(version_dirs(latest_dir.parent).items())
            if idx < oldest_kept
        )
        for path in removed:
            remove_tree(path)
        _logger.info(
            "Pruned %d version(s) of %s below v%d", len(removed), keys, oldest_kept
        )
        return removed

    def vacuum(
        self, *keys: str, older_than: dt.timedelta = dt.timedelta(hours=1)
    ) -> tuple[Path, ...]:
        """Delete files that failed or interrupted writes and appends left behind.

        That is hidden tmp files, and versions that were never committed: those
        above the latest version, or every version if the profile has no
        manifest. Versions below the latest are left to `prune`, and the claim
        markers of appends are always kept.

        Only files older than older_than are deleted, since newer ones may belong
        to writes and appends still in progress. Set it longer than your slowest
        write.

        Example:
            >>> import datetime
            >>> from pathlib import Path
            >>> import polars as pl
            >>> from pyrosome import ParquetDal
            >>> dal = ParquetDal(Path("data"))
            >>> output_dir = dal.write(pl.DataFrame({"x": [1]}), "example")
            >>> _ = (output_dir / ".abandoned.parquet.tmp").write_bytes(b"")
            >>> (output_dir.parent / "v1").mkdir()  # an interrupted write
            >>> removed = dal.vacuum("example", older_than=datetime.timedelta(0))
            >>> [path.relative_to(output_dir.parent).as_posix() for path in removed]
            ['v0/.abandoned.parquet.tmp', 'v1']

        Args:
            *keys: Path components naming the profile.
            older_than: How long since a file or version was last modified before
                it is deleted.

        Returns:
            The deleted files and versions, sorted. Empty if the profile doesn't
            exist.

        Raises:
            ValueError: If there are no keys, a key is invalid or leads outside the
                data directory, or the profile's manifest file is corrupt.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `pyrosome.dal.DataManifestFile.LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            OSError: If a file or version cannot be deleted.
        """
        profile_dir = self.get_profile_dir(*keys)
        if not (
            matches_case_on_disk(self.data_directory, profile_dir)
            and profile_dir.is_dir()
        ):
            _logger.debug("Nothing to vacuum for %s", keys)
            return ()
        cutoff = time.time() - older_than.total_seconds()
        manifest = DataManifestFile(profile_dir)
        # a commit only happens under the lock, so no version is committed while
        # it is being deleted
        with manifest.lock():
            latest_idx = manifest.latest_version_index()
            uncommitted = [
                path
                for idx, path in sorted(version_dirs(profile_dir).items())
                if (latest_idx is None or idx > latest_idx)
                and last_modified(path) < cutoff
            ]
            for path in uncommitted:
                remove_tree(path)
        tmp_files = [
            file
            for directory in (profile_dir, *version_dirs(profile_dir).values())
            for file in directory.iterdir()
            if _TMP_FILE_PATTERN.fullmatch(file.name) and file.stat().st_mtime < cutoff
        ]
        for file in tmp_files:
            remove_tmp_file(file)
        removed = tuple(sorted([*uncommitted, *tmp_files]))
        _logger.info("Vacuumed %d file(s) and version(s) of %s", len(removed), keys)
        return removed

    @abstractmethod
    def get_profile_dir(self, *keys: str) -> Path:
        """Get path to the profile directory described by keys.

        Args:
            *keys: Path components naming the profile.

        Returns:
            The profile directory.

        Raises:
            ValueError: If there are no keys, a key is invalid, or the keys lead
                outside the data directory.
        """

    @abstractmethod
    def _require_latest_data(self, *keys: str) -> tuple[Path, ...]:
        """Get the data files of the latest version, which must have some.

        Args:
            *keys: Path components naming the profile.

        Returns:
            The latest data files, sorted by index.

        Raises:
            pyrosome.dal.ProfileNotFoundError: If there is no data for keys.
            ValueError: If the keys are invalid or the manifest file is corrupt.
            FileNotFoundError: If the manifest names a missing version directory.
        """

    @abstractmethod
    def _rewrite_version(
        self,
        source: tuple[Path, ...],
        keys: tuple[str, ...],
        manifest: DataManifestFile,
        output_dir: Path,
    ) -> bool:
        """Write the data in source files as a reserved version, then commit it.

        It is only committed if the version holding source is still the latest and
        has exactly the same data files.

        Args:
            source: The data files of the latest version.
            keys: Path components naming the profile, for logging.
            manifest: The profile's manifest.
            output_dir: The reserved, empty version directory.

        Returns:
            Whether the manifest now points at output_dir.

        Raises:
            ValueError: If the manifest file is corrupt.
            OSError: If the data or the manifest cannot be written.
        """

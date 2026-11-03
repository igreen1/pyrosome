"""Primary API for writing/reading/appending."""

import functools
import logging
import math
import os
import time
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, overload, override

import polars as pl

from pyrosome._files import (
    data_files,
    make_read_only,
    publish_append_file,
    temporary_file,
    try_claim,
)
from pyrosome._keys import (
    check_key,
    find_profile_dirs,
    matches_case_on_disk,
)
from pyrosome._logging import logged_failure
from pyrosome._maintenance import Maintenance, ProfileChangedError
from pyrosome._manifest import DataManifestFile, discarded_on_failure
from pyrosome._schema import check_appendable, promote_null_columns

if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    "CsvDal",
    "Dal",
    "DataManifestFile",
    "ParquetDal",
    "ProfileChangedError",
    "ProfileNotFoundError",
    "SupersededAppendError",
    "get_accessor",
]

_logger = logging.getLogger(__name__)

# warnings point at the caller of pyrosome, not at the module that raised them
_PACKAGE_DIR = str(Path(__file__).parent) + os.sep


@overload
def get_accessor(file_ending: Literal["parquet"]) -> type[ParquetDal]: ...
@overload
def get_accessor(file_ending: Literal["csv"]) -> type[CsvDal]: ...
def get_accessor(file_ending: str) -> type[ParquetDal | CsvDal]:
    """Get the accessor class that stores files with the given file ending.

    Args:
        file_ending: The file format to store, without a leading dot.

    Returns:
        `ParquetDal` for ``"parquet"`` or `CsvDal` for ``"csv"``.

    Raises:
        ValueError: If file_ending is not a supported format.
    """
    match file_ending:
        case "csv":
            return CsvDal
        case "parquet":
            return ParquetDal
        case _:
            msg = f"Invalid file ending for accessor: {file_ending!r}"
            raise ValueError(msg)


def _warn_writing_instead(keys: tuple[str, ...]) -> None:
    """Warn that an append is writing the first version of a profile.

    Args:
        keys: Path components naming the profile.

    Returns:
        None.
    """
    msg = f"Appending to empty data for {keys}. Writing instead"
    warnings.warn(msg, stacklevel=2, skip_file_prefixes=(_PACKAGE_DIR,))


class ProfileNotFoundError(FileNotFoundError):
    """There is no data for a profile, as opposed to data that cannot be found."""


class SupersededAppendError(RuntimeError):
    """An append finished after another process committed a newer version.

    The appended file was published into a version that is no longer the latest,
    so readers don't see it. Append the data again to add it to the latest version.
    """

    published: Path
    """The file the data was appended as, in the superseded version."""

    latest_data_dir: Path
    """The version that was committed while appending."""

    def __init__(
        self, keys: tuple[str, ...], published: Path, latest_data_dir: Path
    ) -> None:
        """Describe an append that went into a superseded version.

        Args:
            keys: Path components naming the profile.
            published: The file the data was appended as.
            latest_data_dir: The version that was committed while appending.

        Returns:
            None.
        """
        super().__init__(
            f"Appended to {keys} as {published}, but {latest_data_dir.name} was "
            f"committed meanwhile, so readers don't see it. Append it again to add "
            "it to the latest version."
        )
        self.published = published
        self.latest_data_dir = latest_data_dir


class Dal(Protocol):
    """Versioned, append-only storage for data frames, whatever the file format.

    Example:
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from pyrosome import CsvDal, Dal, ParquetDal
        >>> def save(dal: Dal) -> str:
        ...     return dal.write(pl.DataFrame({"x": [1]}), "example").name
        >>> save(ParquetDal(Path("parquet"))), save(CsvDal(Path("csv")))
        ('v0', 'v0')
    """

    def read(self, *keys: str) -> pl.LazyFrame:
        """Read data from the latest version of the data described by keys.

        Args:
            *keys: Path components naming the profile.

        Returns:
            A lazy scan over every data file in the latest version.

        Raises:
            FileNotFoundError: If there is no data for keys.
        """
        ...

    def write(self, data: pl.DataFrame | pl.LazyFrame, *keys: str) -> Path:
        """Write the data as a new version.

        Args:
            data: The data to write.
            *keys: Path components naming the profile.

        Returns:
            The versioned directory the data was written to.
        """
        ...

    def append(self, data: pl.DataFrame | pl.LazyFrame, *keys: str) -> Path:
        """Add a new file to the latest version without touching existing files.

        Args:
            data: The data to append. Its columns must match the existing data.
            *keys: Path components naming the profile.

        Returns:
            The path of the new file, or the new versioned directory if there was no
            existing data.
        """
        ...

    def available_profiles(self, *keys: str) -> tuple[tuple[str, ...], ...]:
        """List the keys of every profile at or below keys that `read` can read.

        Args:
            *keys: Path components to look below, case sensitive on every
                platform. With no keys, every profile. If keys name a profile, it
                is included.

        Returns:
            The full keys of each profile, sorted, including those nested in
            directories that group profiles and in other profiles.

        Raises:
            NotImplementedError: If the class does not implement listing.
        """
        msg = f"{type(self).__name__} does not implement available_profiles"
        raise NotImplementedError(msg)


class _FileDal(Maintenance, Dal):
    """Abstract file DAL with read/write/append for files."""

    data_directory: Path
    """The directory where data files are stored."""

    extension: str
    """File extension of the data files, without the leading dot."""

    FIRST_COMMIT_TIMEOUT: Final[float] = 120.0
    """Seconds `append` waits for another process to commit a profile's first
    version."""

    _POLL_INTERVAL: Final[float] = 0.05

    @staticmethod
    @abstractmethod
    def _sink(data: pl.LazyFrame, output_file: Path) -> None:
        """Wrapper so we can replace our file type easily.

        Args:
            data: The data to write.
            output_file: The file to write to.

        Returns:
            None.

        Raises:
            polars.exceptions.PolarsError: If the data cannot be computed or written.
        """

    @staticmethod
    @abstractmethod
    def _scan(output_paths: Path | list[Path]) -> pl.LazyFrame:
        """Wrapper so we can replace our file type easily.

        Args:
            output_paths: The data files to scan.

        Returns:
            A lazy scan over output_paths.

        Raises:
            polars.exceptions.PolarsError: If the files cannot be scanned.
        """

    def __init__(self, data_directory: Path | str | os.PathLike[str]) -> None:
        """Create a data access layer. Nothing is read or written until used.

        Args:
            data_directory: The directory where data files are stored.

        Returns:
            None.
        """
        self.data_directory = Path(os.fspath(data_directory))
        _logger.debug(
            "Created %s for data directory %s",
            type(self).__name__,
            self.data_directory,
        )

    @override
    def read(self, *keys: str) -> pl.LazyFrame:
        """Read data from the latest version of the data described by keys.

        Args:
            *keys: Path components naming the profile.

        Returns:
            A lazy scan over every data file in the latest version.

        Raises:
            ProfileNotFoundError: If there is no data for keys, including when a
                directory on disk differs from keys only in case.
            FileNotFoundError: If the manifest names a missing version directory.
            ValueError: If there are no keys, a key is invalid or leads outside the
                data directory, or the profile's manifest file is corrupt.
            polars.exceptions.PolarsError: If the data files cannot be scanned.
        """
        latest_data = self._require_latest_data(*keys)
        _logger.debug(
            "Reading %s from %d file(s) in %s",
            keys,
            len(latest_data),
            latest_data[0].parent,
        )
        return self._scan(list(latest_data))

    @override
    def write(self, data: pl.DataFrame | pl.LazyFrame, *keys: str) -> Path:
        """Write the data as a new version.

        Readers keep seeing the previous version until the write is complete.
        Columns whose type is ``Null``, because every value is null, are stored as
        ``String`` so that later appends can fill them. If the write fails, the
        new version is removed again.

        Args:
            data: The data to write.
            *keys: Path components naming the profile.

        Returns:
            The versioned directory the data was written to.

        Raises:
            ValueError: If there are no keys, a key is invalid or leads outside the
                data directory, the keys differ only in case from an existing
                directory on a case-insensitive file system, or the profile's
                manifest file is corrupt or names an invalid version.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `pyrosome.dal.DataManifestFile.LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the profile directory cannot be created or written to.
        """
        _logger.debug("Writing new version of %s", keys)
        manifest = DataManifestFile(self._make_profile_dir(keys))
        output_dir = manifest.reserve_new_data_dir()
        self._write_version(data, keys, manifest, output_dir)
        _logger.info("Wrote %s as %s", keys, output_dir)
        return output_dir

    @override
    def append(self, data: pl.DataFrame | pl.LazyFrame, *keys: str) -> Path:
        """Add a new file to the latest data directory without touching existing files.

        Columns are reordered to match the existing data, and cast to its types when
        that loses no values, such as ``Int32`` to ``Int64``.

        If the profile has no data yet, this warns and writes the first version
        instead. When several processes do that at once, one writes ``v0`` and
        the others wait up to `FIRST_COMMIT_TIMEOUT` seconds for it to be
        committed, then append to it.

        Args:
            data: The data to append. Its columns must match the existing data.
            *keys: Path components naming the profile.

        Returns:
            The path of the new file, or the new versioned directory if there was no
            existing data.

        Raises:
            SupersededAppendError: If another process committed a newer version
                while appending, so readers don't see the appended data.
            TimeoutError: If the profile has no data and another process's first
                version is not committed within `FIRST_COMMIT_TIMEOUT` seconds.
            ValueError: If the columns or their types do not match the existing data
                (a type must cast to the existing one without losing values), or for
                any reason `write` raises it.
            FileNotFoundError: If the latest version directory is missing.
            RuntimeError: If the latest data spans more than one directory.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If there was no data
                and another process holds the manifest lock for longer than
                `pyrosome.dal.DataManifestFile.LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If there was no data
                and the manifest cannot be locked.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the data directory cannot be written to.
        """
        _logger.debug("Appending to %s", keys)
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        if latest_data := self._get_latest_profile_data_paths(*keys):
            return self._append_to_version(data, keys, latest_data)
        return self._append_to_new_profile(data, keys)

    def _make_profile_dir(self, keys: tuple[str, ...]) -> Path:
        """Create the profile directory named by keys, if it doesn't exist.

        Args:
            keys: Path components naming the profile.

        Returns:
            The profile directory.

        Raises:
            ValueError: If there are no keys, a key is invalid or leads outside the
                data directory, or the keys differ only in case from an existing
                directory on a case-insensitive file system.
            OSError: If the directory cannot be created.
        """
        profile_dir = self.get_profile_dir(*keys)
        self._check_case_for_write(profile_dir, keys)
        profile_dir.mkdir(parents=True, exist_ok=True)
        # another writer may have created a directory with different case meanwhile
        self._check_case_for_write(profile_dir, keys)
        return profile_dir

    def _write_version(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        keys: tuple[str, ...],
        manifest: DataManifestFile,
        output_dir: Path,
        only_if: Callable[[str | None], bool] | None = None,
    ) -> bool:
        """Write data as the first file of a reserved version, then commit it.

        If anything fails, the version is removed again, unless the manifest
        already points at it.

        Args:
            data: The data to write.
            keys: Path components naming the profile, for logging.
            manifest: The profile's manifest.
            output_dir: The reserved, empty version directory.
            only_if: Passed on to `DataManifestFile.commit_data_dir`.

        Returns:
            Whether the manifest now points at output_dir.

        Raises:
            ValueError: If the manifest file is corrupt or names an invalid version.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `pyrosome.dal.DataManifestFile.LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the data or the manifest cannot be written.
        """
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        data = promote_null_columns(data)
        with (
            discarded_on_failure(manifest, output_dir),
            logged_failure(
                _logger,
                "Failed to write %s as %s, so the version is removed",
                keys,
                output_dir,
            ),
        ):
            with temporary_file(output_dir, f".{self.extension}") as tmp_file:
                _logger.debug("Sinking data for %s to %s", keys, tmp_file)
                self._sink(data, tmp_file)
                make_read_only(tmp_file)
                published = output_dir / f"0.{self.extension}"
                tmp_file.replace(published)
                _logger.debug("Moved %s to %s", tmp_file.name, published)
            # only point the manifest at the new version once its data is complete
            return manifest.commit_data_dir(output_dir, only_if=only_if)

    @override
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
            ValueError: If the manifest file is corrupt or names an invalid version.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `pyrosome.dal.DataManifestFile.LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            polars.exceptions.PolarsError: If the data cannot be read or written.
            OSError: If the data or the manifest cannot be written.
        """
        unchanged = functools.partial(self._is_unchanged, source)
        return self._write_version(
            self._scan(list(source)), keys, manifest, output_dir, only_if=unchanged
        )

    def _is_unchanged(self, source: tuple[Path, ...], latest_name: str | None) -> bool:
        """Check that a version is still the latest and has the same data files.

        Args:
            source: The data files of the version, as they were.
            latest_name: The name of the latest version now.

        Returns:
            Whether the version is still the latest and has exactly source.

        Raises:
            OSError: If the version cannot be listed.
        """
        source_dir = source[0].parent
        return latest_name == source_dir.name and (
            data_files(source_dir, self.extension) == source
        )

    def _append_to_version(
        self,
        data: pl.LazyFrame,
        keys: tuple[str, ...],
        latest_data: tuple[Path, ...],
    ) -> Path:
        """Append data as a new file in the version holding latest_data.

        Args:
            data: The data to append.
            keys: Path components naming the profile.
            latest_data: The data files in the latest version.

        Returns:
            The path of the new file.

        Raises:
            SupersededAppendError: If another process committed a newer version
                while appending.
            ValueError: If the columns or their types do not match the existing
                data, or the manifest is corrupt.
            RuntimeError: If latest_data spans more than one directory.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the data directory cannot be written to.
        """
        existing_schema = self._scan(list(latest_data)).collect_schema()
        check_appendable(data.collect_schema(), existing_schema, keys)
        data = data.select(existing_schema.names()).cast(existing_schema)

        data_dir = latest_data[0].parent
        with logged_failure(_logger, "Failed to append to %s in %s", keys, data_dir):
            with temporary_file(data_dir, f".{self.extension}") as tmp_file:
                _logger.debug("Sinking appended data for %s to %s", keys, tmp_file)
                self._sink(data, tmp_file)
                make_read_only(tmp_file)
                published = publish_append_file(tmp_file, latest_data, self.extension)
            # a version committed before the file was published hides it from
            # readers, so the caller must know to append again
            manifest = DataManifestFile(data_dir.parent)
            if (latest_dir := manifest.get_latest_data_dir()) != data_dir:
                raise SupersededAppendError(keys, published, latest_dir)
        _logger.info("Appended to %s as %s", keys, published)
        return published

    def _append_to_new_profile(self, data: pl.LazyFrame, keys: tuple[str, ...]) -> Path:
        """Append to a profile with no data, by writing its first version.

        If another process is already writing the first version, wait for it to
        be committed and append to it instead.

        Args:
            data: The data to append.
            keys: Path components naming the profile.

        Returns:
            The new version directory, or the appended file if another process
            wrote the first version.

        Raises:
            SupersededAppendError: If another process committed a newer version
                while writing the first one.
            TimeoutError: If another process's first version is not committed within
                `FIRST_COMMIT_TIMEOUT` seconds.
            ValueError: For any reason `write` or `append` raises it.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `pyrosome.dal.DataManifestFile.LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the profile directory cannot be written to.
        """
        manifest = DataManifestFile(self._make_profile_dir(keys))
        first_version = manifest.profile_path / "v0"
        attempts = 1 + math.ceil(self.FIRST_COMMIT_TIMEOUT / self._POLL_INTERVAL)
        for attempt in range(attempts):
            if attempt:
                time.sleep(self._POLL_INTERVAL)
            if latest_data := self._get_latest_profile_data_paths(*keys):
                _logger.debug("Another process wrote the first version of %s", keys)
                return self._append_to_version(data, keys, latest_data)
            if manifest.exists():
                # the latest version only holds another format's data
                _warn_writing_instead(keys)
                return self.write(data, *keys)
            if try_claim(lambda _: first_version.mkdir(), 0):
                _warn_writing_instead(keys)
                return self._write_first_version(data, keys, manifest, first_version)
            if not attempt:
                _logger.debug("Waiting for another process to commit %s", first_version)
        msg = (
            f"{first_version} was not committed within {self.FIRST_COMMIT_TIMEOUT}s. "
            "Another process may still be writing it, or a failed write left it "
            "behind: vacuum removes it once no write is running."
        )
        _logger.error(msg)
        raise TimeoutError(msg)

    def _write_first_version(
        self,
        data: pl.LazyFrame,
        keys: tuple[str, ...],
        manifest: DataManifestFile,
        first_version: Path,
    ) -> Path:
        """Write the data of an append to a profile with no data.

        Args:
            data: The data to write.
            keys: Path components naming the profile.
            manifest: The profile's manifest.
            first_version: The ``v0`` directory, which this process claimed.

        Returns:
            The new version directory.

        Raises:
            SupersededAppendError: If another process committed a newer version
                while writing first_version.
            ValueError: For any reason `write` raises it.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `pyrosome.dal.DataManifestFile.LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the profile directory cannot be written to.
        """
        if not self._write_version(data, keys, manifest, first_version):
            raise SupersededAppendError(
                keys,
                first_version / f"0.{self.extension}",
                manifest.get_latest_data_dir(),
            )
        _logger.info("Wrote %s as %s", keys, first_version)
        return first_version

    @override
    def available_profiles(self, *keys: str) -> tuple[tuple[str, ...], ...]:
        """List the keys of every profile at or below keys that `read` can read.

        A profile is listed when its manifest names a version holding files of
        this format, so each result can be passed straight to `read`, profiles
        written by a DAL with another format are not listed, and a write that is
        still in progress is only listed once it has committed. Profiles with a
        corrupt manifest, or one naming a missing version, are logged and skipped.

        Hidden directories (such as NFS ``.snapshot`` directories), directories
        whose names are not valid keys, and symlinked directories below keys are
        not searched. Keys holding several path components, like ``"a/b"``, are
        listed as separate components. Keys are case sensitive on every platform,
        so a key differing only in case from a directory on disk matches nothing.

        Example:
            >>> from pathlib import Path
            >>> import polars as pl
            >>> from pyrosome import ParquetDal
            >>> dal = ParquetDal(Path("data"))
            >>> df = pl.DataFrame({"x": [1]})
            >>> for keys in [("shop", "orders"), ("shop", "eu", "totals"), ("misc",)]:
            ...     _ = dal.write(df, *keys)
            >>> dal.available_profiles("shop")
            (('shop', 'eu', 'totals'), ('shop', 'orders'))
            >>> dal.available_profiles()
            (('misc',), ('shop', 'eu', 'totals'), ('shop', 'orders'))
            >>> dal.available_profiles("missing")
            ()

        Args:
            *keys: Path components to look below. With no keys, every profile in
                the data directory. If keys name a profile, it is included.

        Returns:
            The full keys of each profile, as path components, sorted. Empty if
            nothing exists below keys.

        Raises:
            ValueError: If a key is invalid or the keys lead outside the data
                directory.
            OSError: If a directory cannot be listed, for any reason other than
                not existing, such as missing permissions.
        """
        root = self.get_profile_dir(*keys) if keys else self.data_directory
        if not matches_case_on_disk(self.data_directory, root):
            _logger.debug("No profiles below %s: a directory differs in case", keys)
            return ()
        found = sorted(
            directory.relative_to(self.data_directory).parts
            for directory in find_profile_dirs(root)
            if directory != self.data_directory and self._has_data(directory)
        )
        _logger.debug("Found %d profile(s) at or below %s", len(found), root)
        return tuple(found)

    def _has_data(self, profile_dir: Path) -> bool:
        """Check whether the latest version of a profile has files of this format.

        Args:
            profile_dir: The profile directory, holding a manifest.

        Returns:
            Whether `read` would find data files, False if the manifest is corrupt
            or names a missing version.

        Raises:
            OSError: If the manifest or the version directory cannot be read.
        """
        try:
            latest_dir = DataManifestFile(profile_dir).get_latest_data_dir()
        except ValueError:
            # reading the manifest already logged why it is corrupt
            _logger.debug("Not listing %s: its manifest is corrupt", profile_dir)
            return False
        try:
            files = data_files(latest_dir, self.extension)
        except FileNotFoundError:
            _logger.exception(
                "Not listing %s: its manifest names missing %s",
                profile_dir,
                latest_dir.name,
            )
            return False
        if not files:
            _logger.debug(
                "Not listing %s: %s has no .%s files",
                profile_dir,
                latest_dir.name,
                self.extension,
            )
        return bool(files)

    def _check_case_for_write(self, profile_dir: Path, keys: tuple[str, ...]) -> None:
        """Refuse to write to a directory whose name differs from keys only in case.

        Args:
            profile_dir: The profile directory named by keys.
            keys: Path components naming the profile, for the error message.

        Returns:
            None.

        Raises:
            ValueError: If an existing directory differs from keys only in case.
            OSError: If a directory cannot be listed to check its case.
        """
        if not matches_case_on_disk(self.data_directory, profile_dir):
            msg = (
                f"Keys {keys} differ only in case from an existing directory in "
                f"{self.data_directory}. Keys are case sensitive, but this file "
                "system is not, so it cannot hold both."
            )
            raise ValueError(msg)

    @override
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
        if not keys:
            msg = "At least one key is required"
            raise ValueError(msg)
        for key in keys:
            check_key(key)
        profile_dir = self.data_directory.joinpath(*keys)
        # check that the profile directory is within the data directory
        if not profile_dir.resolve().is_relative_to(self.data_directory.resolve()):
            msg = (
                f"Profile directory {profile_dir} is not within data directory "
                f"{self.data_directory}"
            )
            raise ValueError(msg)
        return profile_dir

    def _get_latest_profile_version(self, *keys: str) -> Path | None:
        """Get the latest subdirectory from manifest.

        Args:
            *keys: Path components naming the profile.

        Returns:
            The latest version directory, or None if nothing exists yet, including
            when a directory on disk differs from keys only in case.

        Raises:
            ValueError: If there are no keys, a key is invalid, the keys lead outside
                the data directory, or the profile's manifest file is corrupt.
            OSError: If a directory cannot be listed to check its case.
        """
        profile_dir = self.get_profile_dir(*keys)
        if not matches_case_on_disk(self.data_directory, profile_dir):
            _logger.debug("No profile %s: a directory differs in case", keys)
            return None
        manifest_file = DataManifestFile(profile_dir)
        try:
            return manifest_file.get_latest_data_dir()
        except FileNotFoundError:
            _logger.debug("No manifest at %s", manifest_file.manifest_path)
            return None

    @override
    def _require_latest_data(self, *keys: str) -> tuple[Path, ...]:
        """Get the data files of the latest version, which must have some.

        Args:
            *keys: Path components naming the profile.

        Returns:
            The latest data files, sorted by index.

        Raises:
            ProfileNotFoundError: If there is no data for keys, including when a
                directory on disk differs from keys only in case.
            ValueError: If there are no keys, a key is invalid, the keys lead outside
                the data directory, or the profile's manifest file is corrupt.
            FileNotFoundError: If the manifest names a missing version directory.
            OSError: If the version directory cannot be listed.
        """
        if not (latest_data := self._get_latest_profile_data_paths(*keys)):
            _logger.debug("No data found for %s", keys)
            msg = f"Cannot find files for {keys}"
            raise ProfileNotFoundError(msg)
        return latest_data

    def _get_latest_profile_data_paths(self, *keys: str) -> tuple[Path, ...]:
        """Get a concrete list of the paths with the latest data.

        Args:
            *keys: Path components naming the profile.

        Returns:
            The latest data files, empty if nothing exists yet.

        Raises:
            ValueError: If there are no keys, a key is invalid, the keys lead outside
                the data directory, or the profile's manifest file is corrupt.
            FileNotFoundError: If the manifest names a missing version directory.
            OSError: If the version directory cannot be listed.
        """
        if (latest_dir := self._get_latest_profile_version(*keys)) is None:
            return ()
        with logged_failure(_logger, "Cannot list version %s of %s", latest_dir, keys):
            return data_files(latest_dir, extension=self.extension)


class ParquetDal(_FileDal):
    """Data Access Layer storing versioned data as parquet files.

    Each profile, named by one or more keys, is a directory under
    `data_directory`. Every `write` creates a new version and `append` adds files
    to the latest version, so existing data is never modified in place.

    Example:
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from pyrosome import ParquetDal
        >>> dal = ParquetDal(Path("data"))
        >>> dal.write(pl.DataFrame({"x": [1, 2]}), "example").name
        'v0'
        >>> dal.append(pl.DataFrame({"x": [3]}), "example").name
        '1.parquet'
        >>> dal.read("example").collect()["x"].to_list()
        [1, 2, 3]
    """

    extension: str = "parquet"
    """File extension of the data files, without the leading dot."""

    @override
    @staticmethod
    def _sink(data: pl.LazyFrame, output_file: Path) -> None:
        """Wrapper so we can replace our file type easily.

        Args:
            data: The data to write.
            output_file: The file to write to.

        Returns:
            None.

        Raises:
            polars.exceptions.PolarsError: If the data cannot be computed or written.
        """
        data.sink_parquet(output_file)

    @override
    @staticmethod
    def _scan(output_paths: Path | list[Path]) -> pl.LazyFrame:
        """Wrapper so we can replace our file type easily.

        Args:
            output_paths: The data files to scan.

        Returns:
            A lazy scan over output_paths.

        Raises:
            polars.exceptions.PolarsError: If the files cannot be scanned.
        """
        return pl.scan_parquet(output_paths)


class CsvDal(_FileDal):
    """Same as a ParquetDal but uses csv files instead of parquet files.

    Broadly recommend using the parquet version, but this exists for user convenience.
    Column types are inferred when the files are read back.

    Example:
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from pyrosome import CsvDal
        >>> csv_dal = CsvDal(Path("data"))
        >>> output_dir = csv_dal.write(pl.DataFrame({"x": [1, 2]}), "example_csv")
        >>> (output_dir / "0.csv").read_text().splitlines()
        ['x', '1', '2']
    """

    extension: str = "csv"
    """File extension of the data files, without the leading dot."""

    @override
    @staticmethod
    def _sink(data: pl.LazyFrame, output_file: Path) -> None:
        """Wrapper so we can replace our file type easily.

        Args:
            data: The data to write.
            output_file: The file to write to.

        Returns:
            None.

        Raises:
            polars.exceptions.PolarsError: If the data cannot be computed or written.
        """
        data.sink_csv(output_file)

    @override
    @staticmethod
    def _scan(output_paths: Path | list[Path]) -> pl.LazyFrame:
        """Wrapper so we can replace our file type easily.

        Args:
            output_paths: The data files to scan.

        Returns:
            A lazy scan over output_paths.

        Raises:
            polars.exceptions.PolarsError: If the files cannot be scanned.
        """
        return pl.scan_csv(output_paths)

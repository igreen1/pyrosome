"""Primary API for writing/reading/appending."""

import contextlib
import itertools
import json
import re
import stat
import uuid
import warnings
from typing import TYPE_CHECKING, Final, cast, override

lazy import polars as pl

from salp.dal.simple_file_lock import SimpleWriteLock

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path


_DATA_DIR_PATTERN = re.compile(r"v(\d+)")
_DATA_FILE_PATTERN = {
    "parquet": re.compile(r"(\d+)\.parquet"),
    "csv": re.compile(r"(\d+)\.csv"),
}


def _tmp_path(directory: Path, suffix: str) -> Path:
    """Get a hidden temporary path in directory that is never picked up as data.

    Args:
        directory: The directory the temporary file will be in.
        suffix: Suffix to put before the final `.tmp`.

    Returns:
        A new, unique temporary path.
    """
    return directory / f".{uuid.uuid4().hex}{suffix}.tmp"


def _make_read_only(file: Path) -> None:
    """Remove every write permission from file so finished data is never modified.

    Args:
        file: The file to make read only.

    Returns:
        None.

    Raises:
        OSError: If the file's permissions cannot be read or changed.
    """
    file.chmod(
        stat.S_IMODE(file.stat().st_mode)
        & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    )


def _remove_tmp_file(file: Path) -> None:
    """Delete a tmp file if it still exists, even if it has been made read only.

    Windows cannot delete read-only files, so write permission is restored first.
    This is only safe because tmp files are never linked anywhere else.

    Args:
        file: The tmp file to delete.

    Returns:
        None.

    Raises:
        OSError: If the file exists but cannot be deleted.
    """
    with contextlib.suppress(FileNotFoundError):
        file.chmod(stat.S_IREAD | stat.S_IWRITE)
    file.unlink(missing_ok=True)


def _claim_marker(data_dir: Path, idx: int, extension: str) -> Path:
    """Get the hidden marker file whose creation claims a data file index.

    Args:
        data_dir: The version directory.
        idx: The data file index.
        extension: The file extension of the data files.

    Returns:
        The marker path, e.g. `.3.parquet.claim`.
    """
    return data_dir / f".{idx}.{extension}.claim"


def _data_file_index(file: Path) -> int:
    """Get the index of a data file like `3.parquet`.

    Args:
        file: The data file.

    Returns:
        The index in the file name.

    Raises:
        ValueError: If the file name is not an integer index.
    """
    return int(file.stem)


def _data_files(data_dir: Path, extension: str) -> tuple[Path, ...]:
    """Get the data files in a version directory, sorted by their index.

    Args:
        data_dir: The version directory.
        extension: The file extension of the data files.

    Returns:
        The data files, sorted by index.

    Raises:
        KeyError: If there is no data file pattern for extension.
        OSError: If data_dir cannot be listed.
    """
    pat = _DATA_FILE_PATTERN[extension]
    files = (file for file in data_dir.iterdir() if pat.fullmatch(file.name))
    return tuple(sorted(files, key=_data_file_index))


def _claim_first_free[T](candidates: Iterable[T], claim: Callable[[T], None]) -> T:
    """Claim the first candidate that is not already taken.

    Args:
        candidates: Candidates to try, in order.
        claim: Atomically claims a candidate, raising FileExistsError if it is taken.

    Returns:
        The candidate that was claimed.

    Raises:
        RuntimeError: If every candidate is already taken.
        Exception: Any error from claim other than FileExistsError.
    """
    for candidate in candidates:
        try:
            claim(candidate)
        except FileExistsError:
            continue  # handles race conditions with other processes
        return candidate
    msg = "Every candidate is already taken"
    raise RuntimeError(msg)


class DAL:
    """Data Access Layer storing versioned data as parquet files.

    Each profile, named by one or more keys, is a directory under
    `data_directory`. Every `write` creates a new version and `append` adds files
    to the latest version, so existing data is never modified in place.

    Example:
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from salp.dal.dal import DAL
        >>> dal = DAL(Path("data"))
        >>> dal.write(pl.DataFrame({"x": [1, 2]}), "example").name
        'v0'
        >>> dal.append(pl.DataFrame({"x": [3]}), "example").name
        '1.parquet'
        >>> dal.read("example").collect()["x"].to_list()
        [1, 2, 3]
    """

    data_directory: Path
    """The directory where data files are stored."""

    extension: str = "parquet"
    """File extension of the data files, without the leading dot."""

    def __init__(self, data_directory: Path) -> None:
        """Create a data access layer. Nothing is read or written until used.

        Args:
            data_directory: The directory where data files are stored.

        Returns:
            None.
        """
        self.data_directory = data_directory

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

    def read(self, *keys: str) -> pl.LazyFrame:
        """Read data from the latest version of the data described by keys.

        Args:
            *keys: Path components naming the profile.

        Returns:
            A lazy scan over every data file in the latest version.

        Raises:
            FileNotFoundError: If there is no data for keys.
            ValueError: If there are no keys or they lead outside the data directory.
            TypeError: If the profile's manifest file is corrupt.
            polars.exceptions.PolarsError: If the data files cannot be scanned.
        """
        if not (latest_data := self._get_latest_profile_data_paths(*keys)):
            msg = f"Cannot find files for {keys}"
            raise FileNotFoundError(msg)
        return self._scan(list(latest_data))

    def write(self, data: pl.DataFrame | pl.LazyFrame, *keys: str) -> Path:
        """Write the data as a new version.

        Readers keep seeing the previous version until the write is complete.

        Args:
            data: The data to write.
            *keys: Path components naming the profile.

        Returns:
            The versioned directory the data was written to.

        Raises:
            ValueError: If there are no keys, they lead outside the data directory, or
                the profile's manifest names an invalid version.
            TypeError: If the profile's manifest file is corrupt.
            salp.dal.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `DataManifestFile.LOCK_TIMEOUT`.
            salp.dal.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the profile directory cannot be created or written to.
        """
        profile_dir = self._get_profile_dir(*keys)
        profile_dir.mkdir(parents=True, exist_ok=True)
        manifest = DataManifestFile(profile_dir)
        output_dir = manifest.reserve_new_data_dir()
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        tmp_file = _tmp_path(output_dir, f".{self.extension}")
        self._sink(data, tmp_file)
        _make_read_only(tmp_file)
        tmp_file.replace(output_dir / f"0.{self.extension}")
        # only point the manifest at the new version once its data is complete
        manifest.commit_data_dir(output_dir)
        return output_dir

    def append(self, data: pl.DataFrame | pl.LazyFrame, *keys: str) -> Path:
        """Add a new file to the latest data directory without touching existing files.

        Columns are reordered and cast to match the existing data. If there is no
        existing data this warns and writes instead.

        Args:
            data: The data to append. Its columns must match the existing data.
            *keys: Path components naming the profile.

        Returns:
            The path of the new file, or the new versioned directory if there was no
            existing data.

        Raises:
            ValueError: If the columns do not match the existing data, or for any
                reason `write` raises it.
            polars.exceptions.InvalidOperationError: If a column cannot be cast to
                the existing type.
            FileNotFoundError: If the latest version directory is missing.
            TypeError: If the profile's manifest file is corrupt.
            RuntimeError: If the latest data spans more than one directory.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the data directory cannot be written to.
        """
        if not (latest_data := self._get_latest_profile_data_paths(*keys)):
            msg = f"Appending to empty data for {keys}. Writing instead"
            warnings.warn(msg, stacklevel=2)
            return self.write(data, *keys)

        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        existing_schema = self._scan(list(latest_data)).collect_schema()
        if (new_names := set(data.collect_schema().names())) != set(
            existing_schema.names()
        ):
            msg = (
                f"Columns {sorted(new_names)} do not match existing columns "
                f"{existing_schema.names()} for {keys}"
            )
            raise ValueError(msg)
        data = data.select(existing_schema.names()).cast(existing_schema)

        data_dir = latest_data[0].parent
        tmp_file = _tmp_path(data_dir, f".{self.extension}")
        self._sink(data, tmp_file)
        _make_read_only(tmp_file)
        try:
            return self._publish_append_file(tmp_file, latest_data)
        finally:
            _remove_tmp_file(tmp_file)

    def _publish_append_file(
        self, tmp_file: Path, latest_data: tuple[Path, ...]
    ) -> Path:
        """Give the finished tmp file the next free index without overwriting others.

        Args:
            tmp_file: The finished data file to publish.
            latest_data: The data files in the latest version.

        Returns:
            The published path.

        Raises:
            RuntimeError: If latest_data spans more than one directory.
            OSError: If the index cannot be claimed or the file cannot be moved into
                the data directory.
        """
        parent_dirs = {file.parent for file in latest_data}
        if len(parent_dirs) > 1:
            msg = f"Latest data spans multiple directories: {sorted(parent_dirs)}"
            raise RuntimeError(msg)
        data_dir = parent_dirs.pop()
        next_idx = max(_data_file_index(file) for file in latest_data) + 1
        # exclusive create is atomic on local disks, SMB and NFSv3+, so exactly one
        # append owns each index. Markers are kept so an index is never re-claimed:
        # a plain rename would silently replace a file published by another append
        idx = _claim_first_free(
            itertools.count(next_idx),
            lambda idx: _claim_marker(data_dir, idx, self.extension).touch(
                exist_ok=False
            ),
        )
        published = data_dir / f"{idx}.{self.extension}"
        tmp_file.replace(published)
        return published

    def _get_profile_dir(self, *keys: str) -> Path:
        """Get path to the profile directory described by keys.

        Args:
            *keys: Path components naming the profile.

        Returns:
            The profile directory.

        Raises:
            ValueError: If there are no keys or they lead outside the data directory.
        """
        if not keys:
            msg = "At least one key is required"
            raise ValueError(msg)
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
            The latest version directory, or None if nothing exists yet.

        Raises:
            ValueError: If there are no keys or they lead outside the data directory.
            FileNotFoundError: If the manifest names a missing version directory.
            TypeError: If the profile's manifest file is corrupt.
        """
        profile_dir = self._get_profile_dir(*keys)
        manifest_file = DataManifestFile(profile_dir)
        if not manifest_file.exists():
            return None
        return manifest_file.get_latest_data_dir()

    def _get_latest_profile_data_paths(self, *keys: str) -> tuple[Path, ...]:
        """Get a concrete list of the paths with the latest data.

        Args:
            *keys: Path components naming the profile.

        Returns:
            The latest data files, empty if nothing exists yet.

        Raises:
            ValueError: If there are no keys or they lead outside the data directory.
            FileNotFoundError: If the manifest names a missing version directory.
            TypeError: If the profile's manifest file is corrupt.
        """
        if (latest_dir := self._get_latest_profile_version(*keys)) is None:
            return ()
        return _data_files(latest_dir, extension=self.extension)


class CsvDal(DAL):
    """Same as a DAL but uses csv files instead of parquet files.

    Broadly recommend using the parquet version, but this exists for user convenience.
    Column types are inferred when the files are read back.

    Example:
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from salp.dal.dal import CsvDal
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


class DataManifestFile:
    """A manifest file that records the latest version of a profile."""

    profile_path: Path
    """Path to the profile directory."""

    MANIFEST_FILE_NAME: Final[str] = "manifest.json"
    """Name of the manifest file inside the profile directory."""

    LOCK_TIMEOUT: Final[float] = 10.0
    """Seconds to wait for another process to finish writing the manifest."""

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
            FileNotFoundError: If the manifest file or the latest data directory does
                not exist.
            TypeError: If the manifest file is corrupt.
        """
        latest_data_dir = self.profile_path / self._read_latest_data_dir_name()
        if not latest_data_dir.exists():
            msg = f"Latest data directory {latest_data_dir} does not exist."
            raise FileNotFoundError(msg)
        return latest_data_dir

    def reserve_new_data_dir(self) -> Path:
        """Create and return the next version of the data directory.

        The manifest is not updated; call `commit_data_dir` once the data is written.

        Returns:
            The new, empty data directory.

        Raises:
            OSError: If the profile directory cannot be listed or written to.
        """
        existing = [
            int(match.group(1))
            for path in self.profile_path.iterdir()
            if path.is_dir() and (match := _DATA_DIR_PATTERN.fullmatch(path.name))
        ]
        next_dir_idx = max(existing, default=-1) + 1
        return _claim_first_free(
            (self.profile_path / f"v{idx}" for idx in itertools.count(next_dir_idx)),
            # mkdir is atomic, so concurrent writers never share a directory
            lambda path: path.mkdir(exist_ok=False),
        )

    def commit_data_dir(self, data_dir: Path) -> None:
        """Point the manifest at data_dir, unless it already points at a newer version.

        Args:
            data_dir: The finished data directory.

        Returns:
            None.

        Raises:
            ValueError: If data_dir or the version in the manifest has an invalid
                name.
            TypeError: If the manifest file is corrupt.
            salp.dal.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than `LOCK_TIMEOUT`.
            salp.dal.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            OSError: If the manifest file cannot be written.
        """
        new_idx = self._data_dir_index(data_dir.name)
        with self._lock():
            if self.exists():
                current_idx = self._data_dir_index(self._read_latest_data_dir_name())
                if current_idx >= new_idx:
                    return
            self._write(data_dir.name)

    def _data_dir_index(self, name: str) -> int:
        """Get the version index from a data directory name like 'v3'.

        Args:
            name: The data directory name.

        Returns:
            The version index.

        Raises:
            ValueError: If name is not a valid data directory name.
        """
        if (match := _DATA_DIR_PATTERN.fullmatch(name)) is None:
            msg = f"Invalid data directory name {name!r} for {self.manifest_path}"
            raise ValueError(msg)
        return int(match.group(1))

    def _read_latest_data_dir_name(self) -> str:
        """Read the latest data directory name from the manifest file.

        Returns:
            The latest data directory name.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
            TypeError: If the manifest file has no valid latest data directory.
            json.JSONDecodeError: If the manifest file is not valid JSON.
        """
        if not self.manifest_path.exists():
            msg = f"Manifest file {self.manifest_path} does not exist."
            raise FileNotFoundError(msg)
        # writes replace the file atomically, so no lock is needed to read it
        match cast("object", json.loads(self.manifest_path.read_text())):
            case {"latest_data_dir": str() as name}:
                return name
            case _:
                msg = (
                    f"Manifest file {self.manifest_path} is corrupt: "
                    "missing or invalid 'latest_data_dir'."
                )
                raise TypeError(msg)

    def _write(self, latest_data_dir: str) -> None:
        """Atomically replace the manifest file. The caller must hold the lock.

        Args:
            latest_data_dir: Name of the latest data directory.

        Returns:
            None.

        Raises:
            OSError: If the manifest file cannot be written.
        """
        manifest: dict[str, str] = {"latest_data_dir": latest_data_dir}
        tmp_path = _tmp_path(self.profile_path, ".json")
        tmp_path.write_text(json.dumps(manifest, indent=4))
        tmp_path.replace(self.manifest_path)

    def _lock(self) -> SimpleWriteLock:
        """Lock the manifest file.

        Returns:
            A lock on the manifest file, taken when entered.
        """
        return SimpleWriteLock(self.manifest_path, timeout=self.LOCK_TIMEOUT)

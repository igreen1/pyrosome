"""Primary API for writing/reading/appending."""

import contextlib
import itertools
import json
import os
import re
import stat
import uuid
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast, overload, override

import polars as pl

from pyrosome.simple_file_lock import SimpleWriteLock

if TYPE_CHECKING:
    from collections.abc import Callable


_DATA_DIR_PATTERN = re.compile(r"v(\d+)")
_MANIFEST_FILE_NAME = "manifest.json"
_KEY_SEPARATORS = re.compile(r"[\\/]")
_INVALID_KEY_PARTS = frozenset({"", ".", ".."})
_NON_PORTABLE_CHARS = frozenset(":\0")
# warnings point at the caller of pyrosome, not at the module that raised them
_PACKAGE_DIR = str(Path(__file__).parent) + os.sep
_DATA_FILE_PATTERN = {
    "parquet": re.compile(r"(\d+)\.parquet"),
    "csv": re.compile(r"(\d+)\.csv"),
}


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


# the range of values each numeric type holds exactly; a float holds every integer
# up to the size of its significand
_EXACT_RANGES: Final[dict[pl.DataType, tuple[int, int]]] = {
    **{
        dtype: (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        for dtype, bits in (
            (pl.Int8(), 8),
            (pl.Int16(), 16),
            (pl.Int32(), 32),
            (pl.Int64(), 64),
            (pl.Int128(), 128),
        )
    },
    **{
        dtype: (0, (1 << bits) - 1)
        for dtype, bits in (
            (pl.UInt8(), 8),
            (pl.UInt16(), 16),
            (pl.UInt32(), 32),
            (pl.UInt64(), 64),
        )
    },
    pl.Float32(): (-(1 << 24), 1 << 24),
    pl.Float64(): (-(1 << 53), 1 << 53),
}
_FLOATS: Final[frozenset[pl.DataType]] = frozenset({pl.Float32(), pl.Float64()})


def _is_lossless_cast(source: pl.DataType, target: pl.DataType) -> bool:
    """Check that casting from source to target keeps every value exactly.

    Only identical types and widenings are lossless: an integer to a wider integer
    (or to a float that represents it exactly), a float to a wider float, or an
    all-null column to anything. Floats never cast losslessly to integers.

    Args:
        source: The type of the new data.
        target: The type of the existing data.

    Returns:
        Whether every value of source is the same value in target.
    """
    if source in {target, pl.Null()}:
        return True
    source_range = _EXACT_RANGES.get(source)
    target_range = _EXACT_RANGES.get(target)
    if source_range is None or target_range is None:
        return False
    if source in _FLOATS and target not in _FLOATS:
        return False
    return target_range[0] <= source_range[0] and source_range[1] <= target_range[1]


def _check_appendable(
    new: pl.Schema, existing: pl.Schema, keys: tuple[str, ...]
) -> None:
    """Check that data with the new schema can be appended to the existing schema.

    The columns must be the same, in any order, and every column's type must be the
    existing type or one that casts to it without losing values.

    Args:
        new: The schema of the data to append.
        existing: The schema of the data already stored.
        keys: Path components naming the profile, for the error message.

    Returns:
        None.

    Raises:
        ValueError: If the columns differ, or a column's type cannot be cast to the
            existing type without losing values.
    """
    if set(new.names()) != set(existing.names()):
        msg = (
            f"Columns {sorted(new.names())} do not match existing columns "
            f"{existing.names()} for {keys}"
        )
        raise ValueError(msg)
    if mismatched := {
        name: f"{new[name]} into {dtype}"
        for name, dtype in existing.items()
        if not _is_lossless_cast(new[name], dtype)
    }:
        msg = (
            f"Column types do not match the existing types for {keys}: {mismatched}. "
            "Cast them explicitly before appending."
        )
        raise ValueError(msg)


def _check_key(key: str) -> None:
    """Check that a key is a safe path, made of components that name no other file.

    A key may hold several path components, like ``"sales/orders"``.

    Args:
        key: The key to check.

    Returns:
        None.

    Raises:
        ValueError: If a component of the key is empty, ``.`` or ``..``, would be
            mistaken for a version directory or the manifest, or contains a character
            that is not portable across file systems.
    """
    # re.split is typed as returning `str | Any`
    for part in cast("list[str]", _KEY_SEPARATORS.split(key)):
        if part in _INVALID_KEY_PARTS:
            reason = "empty, '.' and '..' path components are not allowed"
        elif (
            _DATA_DIR_PATTERN.fullmatch(part) is not None or part == _MANIFEST_FILE_NAME
        ):
            reason = f"{part!r} is reserved for the files pyrosome stores"
        elif not _NON_PORTABLE_CHARS.isdisjoint(part):
            reason = "':' and NUL are not allowed"
        else:
            continue
        msg = f"Invalid key {key!r}: {reason}"
        raise ValueError(msg)


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


def _try_claim(claim: Callable[[int], None], candidate: int) -> bool:
    """Try to claim a candidate.

    Args:
        claim: Atomically claims a candidate, raising FileExistsError if it is taken.
        candidate: The candidate to claim.

    Returns:
        Whether the candidate was claimed, False if another process already did.

    Raises:
        Exception: Any error from claim other than FileExistsError.
    """
    try:
        claim(candidate)
    except FileExistsError:
        return False  # handles race conditions with other processes
    return True


def _claim_first_free(start: int, claim: Callable[[int], None]) -> int:
    """Claim the first number from start that is not already taken.

    Args:
        start: The first number to try.
        claim: Atomically claims a number, raising FileExistsError if it is taken.

    Returns:
        The number that was claimed.

    Raises:
        Exception: Any error from claim other than FileExistsError.
    """
    return next(idx for idx in itertools.count(start) if _try_claim(claim, idx))


class ProfileNotFoundError(FileNotFoundError):
    """There is no data for a profile, as opposed to data that cannot be found."""


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


class _FileDal(Dal):
    """Abstract file DAL with read/write/append for files."""

    data_directory: Path
    """The directory where data files are stored."""

    extension: str
    """File extension of the data files, without the leading dot."""

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

    @override
    def read(self, *keys: str) -> pl.LazyFrame:
        """Read data from the latest version of the data described by keys.

        Args:
            *keys: Path components naming the profile.

        Returns:
            A lazy scan over every data file in the latest version.

        Raises:
            ProfileNotFoundError: If there is no data for keys.
            FileNotFoundError: If the manifest names a missing version directory.
            ValueError: If there are no keys, a key is invalid or leads outside the
                data directory, or the profile's manifest file is corrupt.
            polars.exceptions.PolarsError: If the data files cannot be scanned.
        """
        if not (latest_data := self._get_latest_profile_data_paths(*keys)):
            msg = f"Cannot find files for {keys}"
            raise ProfileNotFoundError(msg)
        return self._scan(list(latest_data))

    @override
    def write(self, data: pl.DataFrame | pl.LazyFrame, *keys: str) -> Path:
        """Write the data as a new version.

        Readers keep seeing the previous version until the write is complete.

        Args:
            data: The data to write.
            *keys: Path components naming the profile.

        Returns:
            The versioned directory the data was written to.

        Raises:
            ValueError: If there are no keys, a key is invalid or leads outside the
                data directory, or the profile's manifest file is corrupt or names an
                invalid version.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than
                `pyrosome.dal.DataManifestFile.LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the profile directory cannot be created or written to.
        """
        profile_dir = self.get_profile_dir(*keys)
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

    @override
    def append(self, data: pl.DataFrame | pl.LazyFrame, *keys: str) -> Path:
        """Add a new file to the latest data directory without touching existing files.

        Columns are reordered to match the existing data, and cast to its types when
        that loses no values, such as ``Int32`` to ``Int64``. If there is no existing
        data this warns and writes instead.

        Args:
            data: The data to append. Its columns must match the existing data.
            *keys: Path components naming the profile.

        Returns:
            The path of the new file, or the new versioned directory if there was no
            existing data.

        Raises:
            ValueError: If the columns or their types do not match the existing data
                (a type must cast to the existing one without losing values), or for
                any reason `write` raises it.
            FileNotFoundError: If the latest version directory is missing.
            RuntimeError: If the latest data spans more than one directory.
            polars.exceptions.PolarsError: If the data cannot be computed or written.
            OSError: If the data directory cannot be written to.
        """
        if not (latest_data := self._get_latest_profile_data_paths(*keys)):
            msg = f"Appending to empty data for {keys}. Writing instead"
            warnings.warn(msg, stacklevel=2, skip_file_prefixes=(_PACKAGE_DIR,))
            return self.write(data, *keys)

        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        existing_schema = self._scan(list(latest_data)).collect_schema()
        _check_appendable(data.collect_schema(), existing_schema, keys)
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
            next_idx,
            lambda idx: _claim_marker(data_dir, idx, self.extension).touch(
                exist_ok=False
            ),
        )
        published = data_dir / f"{idx}.{self.extension}"
        tmp_file.replace(published)
        return published

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
            _check_key(key)
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
            ValueError: If there are no keys, a key is invalid, the keys lead outside
                the data directory, or the profile's manifest file is corrupt.
        """
        manifest_file = DataManifestFile(self.get_profile_dir(*keys))
        try:
            return manifest_file.get_latest_data_dir()
        except FileNotFoundError:
            return None

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
        return _data_files(latest_dir, extension=self.extension)


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


class DataManifestFile:
    """A manifest file that records the latest version of a profile."""

    profile_path: Path
    """Path to the profile directory."""

    MANIFEST_FILE_NAME: Final[str] = _MANIFEST_FILE_NAME
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
            FileNotFoundError: If the manifest file does not exist. The directory it
                names is not checked, so listing it raises if it is missing.
            ValueError: If the manifest file is corrupt.
        """
        return self.profile_path / self._read_latest_data_dir_name()

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
        idx = _claim_first_free(
            next_dir_idx,
            # mkdir is atomic, so concurrent writers never share a directory
            lambda idx: (self.profile_path / f"v{idx}").mkdir(exist_ok=False),
        )
        return self.profile_path / f"v{idx}"

    def commit_data_dir(self, data_dir: Path) -> None:
        """Point the manifest at data_dir, unless it already points at a newer version.

        Args:
            data_dir: The finished data directory.

        Returns:
            None.

        Raises:
            ValueError: If data_dir or the version in the manifest has an invalid
                name, or the manifest file is corrupt.
            pyrosome.simple_file_lock.FileAlreadyLockedError: If another process holds
                the manifest lock for longer than `LOCK_TIMEOUT`.
            pyrosome.simple_file_lock.FileLockPermissionError: If the manifest cannot
                be locked.
            OSError: If the manifest file cannot be written.
        """
        new_idx = self._data_dir_index(data_dir.name)
        with self._lock():
            try:
                current_name = self._read_latest_data_dir_name()
            except FileNotFoundError:
                current_name = None
            if (
                current_name is not None
                and self._data_dir_index(current_name) >= new_idx
            ):
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
            ValueError: If the manifest file has no valid latest data directory, or is
                not valid JSON (`json.JSONDecodeError`).
        """
        # writes replace the file atomically, so no lock is needed to read it
        match cast("object", json.loads(self.manifest_path.read_text())):
            case {"latest_data_dir": str() as name} if _DATA_DIR_PATTERN.fullmatch(
                name
            ):
                return name
            case _:
                msg = (
                    f"Manifest file {self.manifest_path} is corrupt: "
                    "missing or invalid 'latest_data_dir'."
                )
                raise ValueError(msg)

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

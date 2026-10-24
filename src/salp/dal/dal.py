"""Primary API for writing/reading/appending."""

import json
import os
import re
import uuid
import warnings
from typing import TYPE_CHECKING, Final

lazy import polars as pl

from .simple_file_lock import SimpleWriteLock

if TYPE_CHECKING:
    from pathlib import Path


_DATA_DIR_PATTERN = re.compile(r"v(\d+)")
_DATA_FILE_PATTERN = {"parquet": re.compile(r"(\d+)\.parquet")}


def _tmp_path(directory: Path, suffix: str) -> Path:
    """Get a hidden temporary path in directory that is never picked up as data."""
    return directory / f".{uuid.uuid4().hex}{suffix}.tmp"


def _data_files(data_dir: Path, extension: str) -> tuple[Path, ...]:
    """Get the data files in a version directory, sorted by their index."""
    pat = _DATA_FILE_PATTERN[extension]
    files = (file for file in data_dir.iterdir() if pat.fullmatch(file.name))
    return tuple(sorted(files, key=lambda file: int(file.stem)))


class DAL:
    """Data Access Layer."""

    data_directory: Path
    """The directory where data files are stored."""

    extension: str = "parquet"

    def __init__(self, data_directory: Path):
        self.data_directory = data_directory

    @staticmethod
    def _sink(data: pl.LazyFrame, output_file: Path) -> None:
        """Wrapper so we can replace our file type easily."""
        data.sink_parquet(output_file)

    @staticmethod
    def _scan(output_paths: Path | list[Path]) -> pl.LazyFrame:
        """Wrapper so we can replace our file type easily."""
        return pl.scan_parquet(output_paths)

    def read(self, *keys: str) -> pl.LazyFrame:
        """Read data from the latest version of the data described by keys."""
        latest_data = self._get_latest_profile_data_paths(*keys)
        if not latest_data:
            msg = f"Cannot find files for {keys}"
            raise FileNotFoundError(msg)
        return self._scan(list(latest_data))

    def write(self, data: pl.DataFrame | pl.LazyFrame, *keys: str) -> Path:
        """Write the data and return the profile versioned directory used."""
        profile_dir = self._get_profile_dir(*keys)
        profile_dir.mkdir(parents=True, exist_ok=True)
        manifest = DataManifestFile(profile_dir)
        output_dir = manifest.reserve_new_data_dir()
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        tmp_file = _tmp_path(output_dir, f".{self.extension}")
        self._sink(data, tmp_file)
        os.replace(tmp_file, output_dir / f"0.{self.extension}")
        # only point the manifest at the new version once its data is complete
        manifest.commit_data_dir(output_dir)
        return output_dir

    def append(self, data: pl.DataFrame | pl.LazyFrame, *keys: str) -> Path:
        """Add a new file to the latest data directory without touching existing files.

        Returns the path of the new file.
        If there is no existing data this writes instead and returns the new versioned directory.
        """
        latest_data = self._get_latest_profile_data_paths(*keys)
        if not latest_data:
            msg = f"Appending to empty data for {keys}. Writing instead"
            warnings.warn(msg, stacklevel=2)
            return self.write(data, *keys)

        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        existing_schema = self._scan(list(latest_data)).collect_schema()
        new_names = set(data.collect_schema().names())
        if new_names != set(existing_schema.names()):
            msg = (
                f"Columns {sorted(new_names)} do not match existing columns "
                f"{existing_schema.names()} for {keys}"
            )
            raise ValueError(msg)
        data = data.select(existing_schema.names()).cast(existing_schema)

        data_dir = latest_data[0].parent
        tmp_file = _tmp_path(data_dir, f".{self.extension}")
        self._sink(data, tmp_file)
        try:
            return self._publish_append_file(tmp_file, latest_data)
        finally:
            tmp_file.unlink(missing_ok=True)

    def _publish_append_file(
        self, tmp_file: Path, latest_data: tuple[Path, ...]
    ) -> Path:
        """Give the finished tmp file the next free index, without overwriting other appends."""
        parent_dirs = {file.parent for file in latest_data}
        if len(parent_dirs) > 1:
            msg = f"Latest data spans multiple directories: {sorted(parent_dirs)}"
            raise RuntimeError(msg)
        data_dir = parent_dirs.pop()
        next_idx = max(int(file.stem) for file in latest_data) + 1
        while True:
            next_path = data_dir / f"{next_idx}.{self.extension}"
            try:
                # hard links fail if the target exists, so concurrent appends never clobber
                os.link(tmp_file, next_path)
                return next_path
            except FileExistsError:
                next_idx += 1  # handles race conditions with appending

    def _get_profile_dir(self, *keys: str) -> Path:
        """Get path to the profile directory described by keys."""
        if not keys:
            msg = "At least one key is required"
            raise ValueError(msg)
        profile_dir = self.data_directory.joinpath(*keys)
        # check that the profile directory is within the data directory
        if not profile_dir.resolve().is_relative_to(self.data_directory.resolve()):
            msg = f"Profile directory {profile_dir} is not within data directory {self.data_directory}"
            raise ValueError(msg)
        return profile_dir

    def _get_latest_profile_version(self, *keys: str) -> Path | None:
        """Get the latest subdirectory from manifest or None if nothing exists yet."""
        profile_dir = self._get_profile_dir(*keys)
        manifest_file = DataManifestFile(profile_dir)
        if not manifest_file.exists():
            return None
        return manifest_file.get_latest_data_dir()

    def _get_latest_profile_data_paths(self, *keys: str) -> tuple[Path, ...] | None:
        """Get a concrete list of the paths with the latest data."""
        profile_latest_version_dir = self._get_latest_profile_version(*keys)
        if profile_latest_version_dir is None:
            return None
        return _data_files(profile_latest_version_dir, extension=self.extension)


class CsvDal(DAL):
    """Same as a DAL but uses csv files instead of parquet files.

    Broadly recommend using the parquet version, but this exists for user convenience
    """

    extension: str = "csv"

    @staticmethod
    def _sink(data: pl.LazyFrame, output_file: Path) -> None:
        """Wrapper so we can replace our file type easily."""
        data.sink_csv(output_file)

    @staticmethod
    def _scan(output_paths: Path | list[Path]) -> pl.LazyFrame:
        """Wrapper so we can replace our file type easily."""
        return pl.scan_csv(output_paths)


class DataManifestFile:
    """A manifest file that describes the data in a directory."""

    profile_path: Path
    """Path to the profile directory."""

    MANIFEST_FILE_NAME: Final[str] = "manifest.json"
    LOCK_TIMEOUT: Final[float] = 10.0
    """Seconds to wait for another process to finish writing the manifest."""

    def __init__(self, profile_path: Path):
        self.profile_path = profile_path
        self.manifest_path = profile_path / self.MANIFEST_FILE_NAME

    def exists(self) -> bool:
        """Does the manifest file even exist yet?"""
        return self.manifest_path.exists()

    def get_latest_data_dir(self) -> Path:
        """Get the latest data directory described by the manifest file."""
        latest_data_dir = self.profile_path / self._latest_data_dir_name(self._read())
        if not latest_data_dir.exists():
            msg = f"Latest data directory {latest_data_dir} does not exist."
            raise FileNotFoundError(msg)
        return latest_data_dir

    def reserve_new_data_dir(self) -> Path:
        """Create and return the next version of the data directory.

        The manifest is not updated; call `commit_data_dir` once the data is written.
        """
        existing = [
            int(match.group(1))
            for path in self.profile_path.iterdir()
            if path.is_dir() and (match := _DATA_DIR_PATTERN.fullmatch(path.name))
        ]
        next_dir_idx = max(existing, default=-1) + 1
        while True:
            next_dir = self.profile_path / f"v{next_dir_idx}"
            try:
                # mkdir is atomic, so concurrent writers never share a directory
                next_dir.mkdir(exist_ok=False)
                return next_dir
            except FileExistsError:
                next_dir_idx += 1

    def commit_data_dir(self, data_dir: Path) -> None:
        """Point the manifest at data_dir, unless it already points at a newer version."""
        new_idx = self._data_dir_index(data_dir.name)
        with self._lock():
            if self.exists():
                current_idx = self._data_dir_index(
                    self._latest_data_dir_name(self._read())
                )
                if current_idx >= new_idx:
                    return
            self._write({"latest_data_dir": data_dir.name})

    def _latest_data_dir_name(self, manifest: dict[str, str | float | bool]) -> str:
        """Get the latest data directory name out of a manifest."""
        name = manifest.get("latest_data_dir")
        if not isinstance(name, str):
            msg = f"Manifest file {self.manifest_path} is corrupt: missing or invalid 'latest_data_dir'."
            raise TypeError(msg)
        return name

    def _data_dir_index(self, name: str) -> int:
        """Get the version index from a data directory name like 'v3'."""
        match = _DATA_DIR_PATTERN.fullmatch(name)
        if match is None:
            msg = f"Invalid data directory name {name!r} for {self.manifest_path}"
            raise ValueError(msg)
        return int(match.group(1))

    def _read(self) -> dict[str, str | float | bool]:
        """Read the manifest file."""
        if not self.manifest_path.exists():
            msg = f"Manifest file {self.manifest_path} does not exist."
            raise FileNotFoundError(msg)
        # writes replace the file atomically, so no lock is needed to read it
        return json.loads(self.manifest_path.read_text())

    def _write(self, manifest: dict[str, str | float | bool]) -> None:
        """Atomically replace the manifest file. The caller must hold the lock."""
        tmp_path = _tmp_path(self.profile_path, ".json")
        tmp_path.write_text(json.dumps(manifest, indent=4))
        os.replace(tmp_path, self.manifest_path)

    def _lock(self) -> SimpleWriteLock:
        """Lock the manifest file."""
        return SimpleWriteLock(self.manifest_path, timeout=self.LOCK_TIMEOUT)

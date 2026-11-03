"""File system operations shared by the data access layer and the manifest."""

import contextlib
import itertools
import logging
import re
import shutil
import stat
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pyrosome._keys import DATA_DIR_PATTERN

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

# logged as part of pyrosome.dal, the logger users configure to see storage steps
_logger = logging.getLogger("pyrosome.dal")


def new_tmp_path(directory: Path, suffix: str) -> Path:
    """Get a hidden temporary path in directory that is never picked up as data.

    Args:
        directory: The directory the temporary file will be in.
        suffix: Suffix to put before the final `.tmp`.

    Returns:
        A new, unique temporary path.
    """
    return directory / f".{uuid.uuid4().hex}{suffix}.tmp"


def make_read_only(file: Path) -> None:
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


def remove_tmp_file(file: Path) -> None:
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
        _logger.debug("Removing leftover tmp file %s", file)
    file.unlink(missing_ok=True)


def _remove_read_only(
    remove: Callable[[str], object], path: str, _error: object
) -> None:
    """Retry removing a path after making it writable, for `shutil.rmtree`.

    Data files are read only, and Windows can't delete read-only files.

    Args:
        remove: The function that failed to remove path.
        path: The path to remove.
        _error: The exception remove raised. Unused.

    Returns:
        None.

    Raises:
        OSError: If the path still cannot be removed.
    """
    Path(path).chmod(stat.S_IREAD | stat.S_IWRITE)
    remove(path)


def remove_tree(directory: Path) -> None:
    """Delete a directory and everything in it, including read-only data files.

    Args:
        directory: The directory to delete.

    Returns:
        None.

    Raises:
        OSError: If anything in the directory cannot be deleted.
    """
    shutil.rmtree(directory, onexc=_remove_read_only)
    _logger.debug("Removed %s", directory)


def remove_quietly(remove: Callable[[Path], None], path: Path) -> None:
    """Remove something left behind by a failure, logging instead of raising.

    Used while another exception is propagating, which must not be replaced by a
    failure to clean up.

    Args:
        remove: The function that removes path.
        path: The file or directory to remove.

    Returns:
        None.
    """
    try:
        remove(path)
    except OSError:
        _logger.exception("Cannot remove %s, left behind by a failure", path)


@contextlib.contextmanager
def temporary_file(directory: Path, suffix: str) -> Generator[Path]:
    """Provide a hidden tmp file path in directory, removed when the block ends.

    A block that succeeds has usually moved the file into place already.

    Args:
        directory: The directory the tmp file will be in.
        suffix: Suffix to put before the final `.tmp`.

    Yields:
        The tmp file path, which doesn't exist yet.

    Raises:
        Exception: Anything the block raises.
    """
    tmp_file = new_tmp_path(directory, suffix)
    try:
        yield tmp_file
    finally:
        remove_quietly(remove_tmp_file, tmp_file)


def version_dir_index(name: str) -> int | None:
    """Get the index of a version directory name like ``v3``.

    Args:
        name: The directory name.

    Returns:
        The index, or None if name is not a version directory name.
    """
    if (match := DATA_DIR_PATTERN.fullmatch(name)) is None:
        return None
    return int(match.group(1))


def version_dirs(profile_dir: Path) -> dict[int, Path]:
    """Get every version directory in a profile, by index.

    Args:
        profile_dir: The profile directory.

    Returns:
        Each version directory, keyed by its index.

    Raises:
        OSError: If profile_dir cannot be listed.
    """
    return {
        idx: path
        for path in profile_dir.iterdir()
        if (idx := version_dir_index(path.name)) is not None and path.is_dir()
    }


def last_modified(path: Path) -> float:
    """Get the latest modification time of path and, for a directory, its entries.

    Args:
        path: The file or directory.

    Returns:
        The latest modification time, in seconds since the epoch.

    Raises:
        OSError: If path or an entry cannot be inspected.
    """
    entries: list[Path] = list(path.iterdir()) if path.is_dir() else []
    return max(entry.stat().st_mtime for entry in (path, *entries))


def try_claim(claim: Callable[[int], None], candidate: int) -> bool:
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
        # handles race conditions with other processes
        _logger.debug("Candidate %d is already claimed, trying the next", candidate)
        return False
    return True


def claim_first_free(start: int, claim: Callable[[int], None]) -> int:
    """Claim the first number from start that is not already taken.

    Args:
        start: The first number to try.
        claim: Atomically claims a number, raising FileExistsError if it is taken.

    Returns:
        The number that was claimed.

    Raises:
        Exception: Any error from claim other than FileExistsError.
    """
    return next(idx for idx in itertools.count(start) if try_claim(claim, idx))


def claim_marker(data_dir: Path, idx: int, extension: str) -> Path:
    """Get the hidden marker file whose creation claims a data file index.

    Args:
        data_dir: The version directory.
        idx: The data file index.
        extension: The file extension of the data files.

    Returns:
        The marker path, e.g. `.3.parquet.claim`.
    """
    return data_dir / f".{idx}.{extension}.claim"


def data_file_index(file: Path) -> int:
    """Get the index of a data file like `3.parquet`.

    Args:
        file: The data file.

    Returns:
        The index in the file name.

    Raises:
        ValueError: If the file name is not an integer index.
    """
    return int(file.stem)


def data_files(data_dir: Path, extension: str) -> tuple[Path, ...]:
    """Get the data files in a version directory, sorted by their index.

    Args:
        data_dir: The version directory.
        extension: The file extension of the data files.

    Returns:
        The data files, sorted by index.

    Raises:
        OSError: If data_dir cannot be listed.
    """
    # only numbered files are data, never hidden tmp and claim files
    pattern = re.compile(rf"\d+\.{re.escape(extension)}")
    files = (file for file in data_dir.iterdir() if pattern.fullmatch(file.name))
    return tuple(sorted(files, key=data_file_index))


def publish_append_file(
    tmp_file: Path, latest_data: tuple[Path, ...], extension: str
) -> Path:
    """Give the finished tmp file the next free index without overwriting others.

    Args:
        tmp_file: The finished data file to publish.
        latest_data: The data files in the latest version.
        extension: The file extension of the data files.

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
        _logger.error(msg)
        raise RuntimeError(msg)
    data_dir = parent_dirs.pop()
    next_idx = max(data_file_index(file) for file in latest_data) + 1
    # exclusive create is atomic on local disks, SMB and NFSv3+, so exactly one
    # append owns each index. Markers are kept so an index is never re-claimed:
    # a plain rename would silently replace a file published by another append
    idx = claim_first_free(
        next_idx,
        lambda idx: claim_marker(data_dir, idx, extension).touch(exist_ok=False),
    )
    if idx != next_idx:
        _logger.debug(
            "Indices %d to %d of %s were already claimed by other appends",
            next_idx,
            idx - 1,
            data_dir,
        )
    published = data_dir / f"{idx}.{extension}"
    tmp_file.replace(published)
    _logger.debug("Moved %s to %s", tmp_file.name, published)
    return published

"""Keys, the path components naming profiles, and finding the profiles on disk."""

import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

DATA_DIR_PATTERN = re.compile(r"v(\d+)")
MANIFEST_FILE_NAME = "manifest.json"
_KEY_SEPARATORS = re.compile(r"[\\/]")
_INVALID_KEY_PARTS = frozenset({"", ".", ".."})
_NON_PORTABLE_CHARS = frozenset(":\0")


def check_key(key: str) -> None:
    """Check that a key is a safe path, made of components that name no other file.

    A key may hold several path components, like ``"sales/orders"``.

    Args:
        key: The key to check.

    Returns:
        None.

    Raises:
        ValueError: If a component of the key is empty, ``.`` or ``..``, is hidden
            (starts with ``.``), would be mistaken for a version directory or the
            manifest, or contains a character that is not portable across file
            systems.
    """
    # re.split is typed as returning `str | Any`
    for part in cast("list[str]", _KEY_SEPARATORS.split(key)):
        if part in _INVALID_KEY_PARTS:
            reason = "empty, '.' and '..' path components are not allowed"
        elif part.startswith("."):
            # pyrosome's own tmp and claim files are hidden, as are directories
            # like NFS .snapshot that profiles must never be found in
            reason = "hidden path components, starting with '.', are not allowed"
        elif DATA_DIR_PATTERN.fullmatch(part) is not None or part == MANIFEST_FILE_NAME:
            reason = f"{part!r} is reserved for the files pyrosome stores"
        elif not _NON_PORTABLE_CHARS.isdisjoint(part):
            reason = "':' and NUL are not allowed"
        else:
            continue
        msg = f"Invalid key {key!r}: {reason}"
        raise ValueError(msg)


def _is_listed_key(name: str) -> bool:
    """Check whether a directory name could hold profiles that should be listed.

    Only valid keys are searched, which leaves out version directories and hidden
    directories such as NFS ``.snapshot`` directories.

    Args:
        name: The directory name.

    Returns:
        Whether the directory should be searched for profiles.
    """
    try:
        check_key(name)
    except ValueError:
        return False
    return True


def _raise_unless_missing(error: OSError) -> None:
    """Re-raise an error from listing a directory, unless the directory is gone.

    A directory below the root can only go missing if something other than
    pyrosome deletes it during the search, which is not an error.

    Args:
        error: The error from listing the directory.

    Returns:
        None, if the directory doesn't exist.

    Raises:
        OSError: The error, if it isn't about a missing directory.
    """
    if not isinstance(error, FileNotFoundError | NotADirectoryError):
        raise error


def find_profile_dirs(root: Path) -> Iterator[Path]:
    """Find every directory at or below root that holds a manifest.

    Version directories, hidden directories and directories whose names are not
    valid keys are not searched. Symlinked directories below root are not followed.

    Args:
        root: The directory to search.

    Yields:
        Each directory holding a manifest, root first, in no particular order.

    Raises:
        OSError: If a directory cannot be listed, for any reason other than not
            existing.
    """
    for directory, sub_dirs, files in root.walk(on_error=_raise_unless_missing):
        sub_dirs[:] = [name for name in sub_dirs if _is_listed_key(name)]
        if MANIFEST_FILE_NAME in files:
            yield directory


def _has_entry(directory: Path, name: str) -> bool:
    """Check whether listing directory shows name, with exactly that case.

    Args:
        directory: The directory to list.
        name: The entry name to look for.

    Returns:
        Whether directory holds an entry called exactly name.

    Raises:
        OSError: If directory cannot be listed.
    """
    return any(entry.name == name for entry in directory.iterdir())


def matches_case_on_disk(base: Path, path: Path) -> bool:
    """Check that every existing component of path below base has its case on disk.

    Keys are case sensitive on every platform, but on case-insensitive file systems
    (Windows, macOS by default) ``Shop`` would open the directory ``shop``. Where a
    component exists under both its own name and its name with the case swapped,
    the parent directory is listed to check the exact name is there, so this only
    costs listings on case-insensitive file systems.

    Args:
        base: The directory path starts in, whose own case is not checked.
        path: A path below base.

    Returns:
        Whether no existing component of path is a different-case name for a
        directory on disk. True if path doesn't exist at all.

    Raises:
        ValueError: If path is not below base.
        OSError: If a parent directory cannot be listed.
    """
    current = base
    for part in path.relative_to(base).parts:
        candidate = current / part
        if not candidate.exists():
            return True
        swapped = part.swapcase()
        if (
            swapped != part
            and (current / swapped).exists()
            and not _has_entry(current, part)
        ):
            return False
        current = candidate
    return True

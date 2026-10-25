"""Shared pytest configuration.

`pytest` also runs every example in the docstrings, the docs and the README as a
doctest, so they fail the build when they go stale.
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


# pytest types its fixture decorator with Any, which disallow_any_expr rejects
@pytest.fixture  # type: ignore[misc]
def run_in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test and doctest in its own empty working directory.

    Applied to every test by `usefixtures` in pyproject.toml.

    Doc examples use relative paths like `Path("data")`, so each example file
    starts from a clean directory and never writes into the repository.

    Args:
        tmp_path: A fresh temporary directory for this test.
        monkeypatch: Used to change the working directory for this test only.

    Returns:
        None.
    """
    monkeypatch.chdir(tmp_path)

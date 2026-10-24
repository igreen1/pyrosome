"""Sphinx configuration for the salp documentation."""

# Sphinx reads these exact lowercase names and its own config shapes
# pylint: disable=invalid-name,consider-using-namedtuple-or-dataclass

from importlib.metadata import version as package_version

project = "salp"
author = "ian"
release = package_version("salp")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build"]
# single backticks in docstrings link to Python objects
default_role = "py:obj"

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
}

# every document's examples run against a fresh, empty data directory
doctest_global_setup = """
import tempfile
from pathlib import Path

import polars as pl

from salp.dal.dal import DAL, CsvDal, DataManifestFile
from salp.dal.simple_file_lock import SimpleWriteLock, file_is_locked

data_dir = Path(tempfile.mkdtemp())
"""

html_theme = "furo"
html_title = f"salp {release}"

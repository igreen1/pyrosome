"""Sphinx configuration for the pyrosome documentation."""

# Sphinx reads these exact lowercase names and its own config shapes
# pylint: disable=invalid-name,consider-using-namedtuple-or-dataclass

from importlib.metadata import version as package_version

project = "pyrosome"
author = "ian"
release = package_version("pyrosome")

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

# examples are run by pytest (see conftest.py); `sphinx-build -b doctest` also
# works, and runs each document in a fresh directory so it never writes here
doctest_global_setup = "import os, tempfile; os.chdir(tempfile.mkdtemp())"

html_theme = "furo"
html_title = f"pyrosome {release}"

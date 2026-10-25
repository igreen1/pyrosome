"""Run the example scripts included in the docs.

Doctests cover the interactive examples. These scripts are included with
`literalinclude` instead, because they need real modules, such as for a process
pool.
"""

import subprocess
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).parents[2] / "docs"


def run_example(name, cwd):
    return subprocess.run(
        [sys.executable, DOCS_DIR / "examples" / name],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_parallel_ingest(tmp_path):
    output = run_example("parallel_ingest.py", tmp_path)
    # must match the output shown in docs/recipes.rst
    assert output == "99 batches appended as 99 files, 100 rows\n"
    assert (
        "99 batches appended as 99 files, 100 rows"
        in (DOCS_DIR / "recipes.rst").read_text()
    )

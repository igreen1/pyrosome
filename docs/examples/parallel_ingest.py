"""Append batches to one profile from a pool of processes."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from salp.dal.dal import ParquetDal

DATA_DIR = Path("data")


def ingest(batch_number: int) -> Path:
    """Append one batch of metrics.

    Args:
        batch_number: The batch to generate and append.

    Returns:
        The file the batch was appended as.
    """
    batch = pl.DataFrame({"batch": [batch_number], "value": [batch_number * 10]})
    return ParquetDal(DATA_DIR).append(batch, "metrics")


if __name__ == "__main__":
    ParquetDal(DATA_DIR).write(pl.DataFrame({"batch": [0], "value": [0]}), "metrics")
    with ProcessPoolExecutor() as pool:
        files = list(pool.map(ingest, range(1, 100)))
    rows = ParquetDal(DATA_DIR).read("metrics").collect().height
    print(f"{len(files)} batches appended as {len(set(files))} files, {rows} rows")

"""Logging helpers shared by pyrosome's modules."""

import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

# every pyrosome module imports this one, so the package logger stays quiet unless
# the application configures logging; otherwise logging.lastResort would print
# WARNING and above to stderr
logging.getLogger(__package__).addHandler(logging.NullHandler())


@contextlib.contextmanager
def logged_failure(logger: logging.Logger, msg: str, *args: object) -> Generator[None]:
    """Log any exception raised in the block, with its traceback, and re-raise it.

    Args:
        logger: The logger to log to, at ``ERROR`` level.
        msg: The `%`-style message to log.
        *args: Arguments for msg.

    Yields:
        None, once.

    Raises:
        Exception: Anything the block raises.
    """
    try:
        yield
    except Exception:
        logger.exception(msg, *args)
        raise

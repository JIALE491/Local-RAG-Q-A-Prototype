"""Logging setup shared by every command.

Log messages (progress, timings, warnings) go to stderr and command results go to stdout,
so results can be piped or redirected without the noise.
"""

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(*, verbose: bool = False) -> None:
    """Send this project's log messages to stderr: INFO by default, DEBUG when verbose.

    Only the `rag` logger is configured, and calling this again replaces its handler
    rather than adding a second one. Third-party libraries keep Python's default, which
    prints their warnings and errors to stderr and nothing below that.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    logger = logging.getLogger("rag")
    for old in logger.handlers[:]:
        logger.removeHandler(old)
        old.close()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

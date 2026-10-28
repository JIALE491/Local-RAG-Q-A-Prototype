"""Shared test setup."""

import logging
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def restore_rag_logger() -> Iterator[None]:
    """Undo configure_logging() calls, which the CLI makes, after every test."""
    logger = logging.getLogger("rag")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    yield
    for handler in logger.handlers:
        if handler not in saved[0]:
            handler.close()
    logger.handlers[:], logger.level, logger.propagate = saved

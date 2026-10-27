"""Logging goes to stderr so that command output on stdout stays clean."""

import logging
from collections.abc import Iterator

import pytest

from rag.log import configure_logging


@pytest.fixture(autouse=True)
def restore_rag_logger() -> Iterator[None]:
    logger = logging.getLogger("rag")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    yield
    for handler in logger.handlers:
        handler.close()
    logger.handlers[:], logger.level, logger.propagate = saved


def test_messages_go_to_stderr_not_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    logging.getLogger("rag.services.ingest").info("indexed 3 chunks")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "indexed 3 chunks" in captured.err


def test_debug_appears_only_when_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    logging.getLogger("rag.x").debug("hidden")
    configure_logging(verbose=True)
    logging.getLogger("rag.x").debug("shown")
    err = capsys.readouterr().err
    assert "hidden" not in err
    assert "shown" in err


def test_configuring_twice_does_not_duplicate_output(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    configure_logging()
    logging.getLogger("rag.x").info("once")
    assert capsys.readouterr().err.count("once") == 1

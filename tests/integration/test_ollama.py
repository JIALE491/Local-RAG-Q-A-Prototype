"""Against the real Ollama server: run with `make test-integration` while Ollama is running."""

import csv
import re
from collections.abc import Iterator
from pathlib import Path

import ollama
import pytest

from rag.adapters.ollama_chat import OllamaChat
from rag.adapters.ollama_embedder import OllamaEmbedder
from rag.cli import main
from rag.config import AppConfig, load_config
from rag.domain.errors import AnswerTooLongError, EmbeddingError, ModelUnavailableError
from rag.domain.ports import ChatModel, Embedder
from tests.contracts.chat_model import CONTEXT_TOKENS, TERSE_ANSWER_TOKENS, ChatModelContract
from tests.contracts.embedder import EmbedderContract
from tests.fakes import cosine_similarity

pytestmark = pytest.mark.integration

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.toml"
CONFIG: AppConfig = load_config(REPO_CONFIG)


def real_embedder(*, model: str | None = None, host: str | None = None) -> OllamaEmbedder:
    settings = CONFIG.ollama
    return OllamaEmbedder(
        host=host or settings.host,
        model=model or settings.embed_model,
        document_prefix=settings.embed_document_prefix,
        query_prefix=settings.embed_query_prefix,
        timeout_s=settings.timeout_s,
        keep_alive=settings.keep_alive,
    )


class TestOllamaEmbedder(EmbedderContract):
    @pytest.fixture
    def embedder(self) -> Iterator[Embedder]:
        embedder = real_embedder()
        yield embedder
        embedder.close()


def test_the_document_prefix_is_applied() -> None:
    text = "Refunds take 5 business days."
    with ollama.Client(host=CONFIG.ollama.host) as client:
        raw = client.embed(model=CONFIG.ollama.embed_model, input=[f"search_document: {text}"])
    embedder = real_embedder()
    try:
        [ours] = embedder.embed_documents([text])
        [as_query] = embedder.embed_queries([text])
    finally:
        embedder.close()
    assert ours == pytest.approx(list(raw.embeddings[0]), abs=1e-4)
    assert cosine_similarity(ours, as_query) < 0.999


def test_text_longer_than_the_model_accepts_raises_instead_of_being_cut() -> None:
    embedder = real_embedder()
    try:
        with pytest.raises(EmbeddingError, match="too long"):
            embedder.embed_documents(["refund policy details " * 3000])
    finally:
        embedder.close()


def test_a_model_that_is_not_pulled_is_unavailable() -> None:
    embedder = real_embedder(model="no-such-model-for-tests")
    try:
        with pytest.raises(ModelUnavailableError) as caught:
            embedder.embed_documents(["text"])
    finally:
        embedder.close()
    assert caught.value.hint == "run `ollama pull no-such-model-for-tests`"


def test_an_unreachable_server_is_unavailable() -> None:
    embedder = real_embedder(host="http://127.0.0.1:9")
    try:
        with pytest.raises(ModelUnavailableError, match="cannot reach Ollama"):
            embedder.embed_documents(["text"])
    finally:
        embedder.close()


def test_ingest_then_search_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "data" / "docs"
    docs.mkdir(parents=True)
    (docs / "vacation.md").write_text(
        "# Vacation\n\nEmployees get 15 vacation days a year.\n", encoding="utf-8"
    )
    (docs / "refunds.md").write_text(
        "# Refunds\n\nRefunds are paid within 5 business days.\n", encoding="utf-8"
    )
    config = tmp_path / "config.toml"
    config.write_text(REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["--config", str(config), "ingest"]) == 0
    assert (
        main(["--config", str(config), "search", "How many vacation days do I get?", "-k", "1"])
        == 0
    )
    out = capsys.readouterr().out
    assert "1. vacation.md (Vacation)" in out


def test_eval_from_the_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    docs = tmp_path / "data" / "docs"
    docs.mkdir(parents=True)
    (docs / "vacation.md").write_text(
        "# Vacation\n\nEmployees get 15 vacation days a year.\n", encoding="utf-8"
    )
    (docs / "refunds.md").write_text(
        "# Refunds\n\nRefunds are paid within 5 business days.\n", encoding="utf-8"
    )
    questions = tmp_path / "data" / "eval" / "questions.csv"
    questions.parent.mkdir(parents=True)
    questions.write_text(
        "id,question,expected_doc,expected_answer\n"
        "q1,How much time off do I get each year?,vacation.md,15 days\n"
        "q2,When will I get my money back?,refunds.md,5 business days\n"
        "u1,What is the Wi-Fi password?,,\n",
        encoding="utf-8",
    )
    questions.with_name("facts.csv").write_text(
        "id,fact\nq1,15 vacation days a year\nq2,within 5 business days\n", encoding="utf-8"
    )
    config = tmp_path / "config.toml"
    config.write_text(REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["--config", str(config), "ingest"]) == 0
    assert main(["--config", str(config), "eval"]) == 0
    out = capsys.readouterr().out
    assert "2 answerable questions; 1 question with no answer left out" in out
    assert re.search(r"hit rate at 1 +2/2 \(100%\)", out)
    assert re.search(r"answer coverage at 3 +2/2 \(100%\)", out)
    assert (tmp_path / "results" / "retrieval-sections-4000.csv").is_file()


def test_ingest_and_compare_every_strategy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "data" / "docs"
    docs.mkdir(parents=True)
    (docs / "vacation.md").write_text(
        "# Vacation\n\nEmployees get 15 vacation days a year.\n", encoding="utf-8"
    )
    (docs / "refunds.md").write_text(
        "# Refunds\n\nRefunds are paid within 5 business days.\n", encoding="utf-8"
    )
    questions = tmp_path / "data" / "eval" / "questions.csv"
    questions.parent.mkdir(parents=True)
    questions.write_text(
        "id,question,expected_doc,expected_answer\n"
        "q1,How much time off do I get each year?,vacation.md,15 days\n",
        encoding="utf-8",
    )
    questions.with_name("facts.csv").write_text(
        "id,fact\nq1,15 vacation days a year\n", encoding="utf-8"
    )
    config = tmp_path / "config.toml"
    config.write_text(REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["--config", str(config), "ingest", "--strategy", "all"]) == 0
    assert main(["--config", str(config), "eval", "--strategy", "all"]) == 0
    out = capsys.readouterr().out
    assert "Order by the adoption rule (top-3, then top-1, then mean reciprocal rank):" in out
    assert "In use ([chunking] strategy): sections-4000" in out
    assert all(name in out for name in ("fixed-300-50", "sentences-300", "sections-4000"))
    assert (tmp_path / "results" / "comparison.csv").is_file()


def real_chat_model(*, max_answer_tokens: int | None = None) -> OllamaChat:
    settings = CONFIG.ollama
    assert settings.chat_context_tokens == CONTEXT_TOKENS
    return OllamaChat(
        host=settings.host,
        model=settings.chat_model,
        context_tokens=settings.chat_context_tokens,
        max_answer_tokens=max_answer_tokens or settings.chat_max_answer_tokens,
        temperature=settings.chat_temperature,
        timeout_s=settings.timeout_s,
        keep_alive=settings.keep_alive,
    )


class TestOllamaChat(ChatModelContract):
    @pytest.fixture
    def chat_model(self) -> Iterator[ChatModel]:
        chat_model = real_chat_model()
        yield chat_model
        chat_model.close()

    @pytest.fixture
    def terse_chat_model(self) -> Iterator[ChatModel]:
        chat_model = real_chat_model(max_answer_tokens=TERSE_ANSWER_TOKENS)
        yield chat_model
        chat_model.close()


def test_ingest_then_ask_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "data" / "docs"
    docs.mkdir(parents=True)
    (docs / "vacation.md").write_text(
        "# Vacation\n\nEmployees get 15 vacation days a year.\n", encoding="utf-8"
    )
    (docs / "refunds.md").write_text(
        "# Refunds\n\nRefunds are paid within 5 business days.\n", encoding="utf-8"
    )
    config = tmp_path / "config.toml"
    config.write_text(REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["--config", str(config), "ingest"]) == 0
    capsys.readouterr()
    assert main(["--config", str(config), "ask", "How many vacation days do I get?"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"\b15\b", out.split("\n\nSources")[0])
    assert "] vacation.md (Vacation)" in out
    assert re.search(r"with 2 chunks from sections-4000: prompt \d+ of 4,096 tokens", out)


def test_an_answer_that_reaches_the_end_of_the_context_raises_instead_of_dropping_the_prompt() -> (
    None
):
    """The prompt fits, but prompt plus answer would not: with `shift: false` Ollama stops at
    the end of the context instead of silently dropping part of the prompt."""
    filler = "The storeroom shelves are checked every morning before opening. " * 320
    chat_model = real_chat_model()
    try:
        with pytest.raises(AnswerTooLongError, match="filled the 4,096-token context"):
            chat_model.complete(
                system="Do exactly what you are asked.",
                user=f"{filler}\n\nCount from 1 to 400, with each number on its own line.",
            )
    finally:
        chat_model.close()


def test_answer_all_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "data" / "docs"
    docs.mkdir(parents=True)
    (docs / "vacation.md").write_text(
        "# Vacation\n\nEmployees get 15 vacation days a year.\n", encoding="utf-8"
    )
    (docs / "refunds.md").write_text(
        "# Refunds\n\nRefunds are paid within 5 business days.\n", encoding="utf-8"
    )
    questions = tmp_path / "data" / "eval" / "questions.csv"
    questions.parent.mkdir(parents=True)
    questions.write_text(
        "id,question,expected_doc,expected_answer\n"
        "q1,How much time off do I get each year?,vacation.md,15 days\n"
        "u1,What is the Wi-Fi password?,,\n",
        encoding="utf-8",
    )
    questions.with_name("facts.csv").write_text(
        "id,fact\nq1,15 vacation days a year\n", encoding="utf-8"
    )
    config = tmp_path / "config.toml"
    config.write_text(REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["--config", str(config), "ingest"]) == 0
    assert main(["--config", str(config), "answer-all"]) == 0
    assert "Answering 2 questions with llama3.2:3b" in capsys.readouterr().out
    with (tmp_path / "results" / "answers.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["id"] for row in rows] == ["q1", "u1"]
    assert all(row["answer"] and not row["supported"] and not row["correct"] for row in rows)

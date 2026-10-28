"""The Ollama embedder's request and error handling, with a stub client (no Ollama needed)."""

from collections.abc import Sequence

import httpx
import ollama
import pytest

from rag.adapters.ollama_embedder import OllamaEmbedder
from rag.domain.errors import EmbeddingError, ModelUnavailableError


class StubClient:
    """Records requests; returns fixed vectors or raises a given error."""

    def __init__(
        self, *, error: Exception | None = None, vectors: list[list[float]] | None = None
    ) -> None:
        self.error = error
        self.vectors = vectors
        self.requests: list[tuple[str, list[str], bool, str]] = []
        self.closed = False

    def embed(
        self, *, model: str, input: Sequence[str], truncate: bool, keep_alive: str
    ) -> ollama.EmbedResponse:
        self.requests.append((model, list(input), truncate, keep_alive))
        if self.error is not None:
            raise self.error
        vectors = self.vectors if self.vectors is not None else [[1.0, 0.0]] * len(input)
        return ollama.EmbedResponse(model=model, embeddings=vectors)

    def close(self) -> None:
        self.closed = True


def embedder(client: StubClient) -> OllamaEmbedder:
    return OllamaEmbedder(
        host="http://127.0.0.1:11434",
        model="nomic-embed-text",
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        timeout_s=30,
        keep_alive="10m",
        client=client,
    )


def test_adds_the_task_prefixes_and_never_truncates() -> None:
    client = StubClient()
    embedder(client).embed_documents(["Refunds take 5 days."])
    embedder(client).embed_queries(["How long do refunds take?"])
    assert client.requests == [
        ("nomic-embed-text", ["search_document: Refunds take 5 days."], False, "10m"),
        ("nomic-embed-text", ["search_query: How long do refunds take?"], False, "10m"),
    ]


def test_no_texts_means_no_request() -> None:
    client = StubClient()
    assert embedder(client).embed_documents([]) == []
    assert client.requests == []


def test_blank_text_is_rejected_before_any_request() -> None:
    client = StubClient()
    with pytest.raises(EmbeddingError, match="blank"):
        embedder(client).embed_documents(["fine", " "])
    assert client.requests == []


def test_a_missing_model_says_how_to_pull_it() -> None:
    client = StubClient(error=ollama.ResponseError('model "nomic-embed-text" not found', 404))
    with pytest.raises(ModelUnavailableError) as caught:
        embedder(client).embed_documents(["text"])
    assert caught.value.hint == "run `ollama pull nomic-embed-text`"


def test_text_longer_than_the_model_accepts_is_an_error() -> None:
    error = ollama.ResponseError("the input length exceeds the context length", 400)
    with pytest.raises(EmbeddingError, match="too long") as caught:
        embedder(StubClient(error=error)).embed_documents(["text"])
    assert caught.value.hint is not None
    assert "chunk size" in caught.value.hint


def test_other_ollama_errors_are_embedding_errors() -> None:
    error = ollama.ResponseError("out of memory", 500)
    with pytest.raises(EmbeddingError, match="out of memory"):
        embedder(StubClient(error=error)).embed_documents(["text"])


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ConnectionError("refused"), "cannot reach Ollama"),
        (httpx.ReadTimeout("timed out"), "did not answer within 30 s"),
        (httpx.RemoteProtocolError("server disconnected"), "lost the connection"),
    ],
)
def test_connection_problems_mean_the_model_is_unavailable(error: Exception, message: str) -> None:
    with pytest.raises(ModelUnavailableError, match=message):
        embedder(StubClient(error=error)).embed_documents(["text"])


def test_an_unexpected_number_of_vectors_is_an_error() -> None:
    client = StubClient(vectors=[[1.0, 0.0]])
    with pytest.raises(EmbeddingError, match="returned 1 embeddings"):
        embedder(client).embed_documents(["one", "two"])


def test_close_closes_the_client() -> None:
    client = StubClient()
    embedder(client).close()
    assert client.closed

"""Embedder backed by a local Ollama model.

Every text gets the model's task prefix ("search_document: " or "search_query: " for
nomic-embed-text). Ollama's own default is to cut off text that is too long for the model
without saying so; this adapter turns that off, so over-long text raises an error instead.
"""

from collections.abc import Sequence
from typing import Protocol

import httpx
import ollama

from rag.domain.errors import EmbeddingError, ModelUnavailableError
from rag.domain.models import Embedding


class EmbedClient(Protocol):
    """The part of ollama.Client this adapter uses; tests substitute a stub."""

    def embed(
        self, *, model: str, input: Sequence[str], truncate: bool, keep_alive: str
    ) -> ollama.EmbedResponse: ...

    def close(self) -> None: ...


class OllamaEmbedder:
    """Turns text into vectors with an Ollama embedding model. Call close() when done."""

    def __init__(
        self,
        *,
        host: str,
        model: str,
        document_prefix: str,
        query_prefix: str,
        timeout_s: float,
        keep_alive: str,
        client: EmbedClient | None = None,
    ) -> None:
        self._host = host
        self._model = model
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._timeout_s = timeout_s
        self._keep_alive = keep_alive
        self._client: EmbedClient = client or ollama.Client(host=host, timeout=timeout_s)

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return self._embed(texts, prefix=self._document_prefix)

    def embed_queries(self, texts: Sequence[str]) -> list[Embedding]:
        return self._embed(texts, prefix=self._query_prefix)

    def close(self) -> None:
        self._client.close()

    def _embed(self, texts: Sequence[str], *, prefix: str) -> list[Embedding]:
        blank = [position for position, text in enumerate(texts) if not text.strip()]
        if blank:
            raise EmbeddingError(f"cannot embed blank text (input positions {blank})")
        if not texts:
            return []
        try:
            response = self._client.embed(
                model=self._model,
                input=[prefix + text for text in texts],
                truncate=False,
                keep_alive=self._keep_alive,
            )
        except ollama.ResponseError as exc:
            raise self._response_error(exc) from exc
        except ConnectionError as exc:
            raise ModelUnavailableError(
                f"cannot reach Ollama at {self._host}",
                hint="start the Ollama app, or check [ollama] host in config.toml",
            ) from exc
        except httpx.TimeoutException as exc:
            raise ModelUnavailableError(
                f"Ollama did not answer within {self._timeout_s:g} s",
                hint="the first request also loads the model; raise [ollama] timeout_s "
                "if this keeps happening",
            ) from exc
        except httpx.TransportError as exc:
            raise ModelUnavailableError(f"lost the connection to Ollama: {exc}") from exc
        return self._checked([list(vector) for vector in response.embeddings], len(texts))

    def _response_error(self, exc: ollama.ResponseError) -> Exception:
        if exc.status_code == 404:
            return ModelUnavailableError(
                f"the model {self._model!r} is not available in Ollama",
                hint=f"run `ollama pull {self._model}`",
            )
        if "context length" in exc.error:
            return EmbeddingError(
                f"a text is too long for {self._model} ({exc.error})",
                hint="use a smaller chunk size",
            )
        return EmbeddingError(f"Ollama could not embed the texts: {exc.error} ({exc.status_code})")

    def _checked(self, vectors: list[Embedding], expected: int) -> list[Embedding]:
        sizes = {len(vector) for vector in vectors}
        if len(vectors) != expected or len(sizes) != 1 or 0 in sizes:
            raise EmbeddingError(
                f"Ollama returned {len(vectors)} embeddings of sizes {sorted(sizes)} "
                f"for {expected} texts"
            )
        return vectors

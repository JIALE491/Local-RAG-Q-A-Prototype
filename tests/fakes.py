"""In-memory stand-ins for the ports, so unit tests run without Ollama or chromadb.

They pass the same contract tests as the real adapters (see tests/contracts/), so code
that works against the fakes works against the real thing.
"""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag.domain.errors import (
    EmbeddingError,
    GenerationError,
    IndexNotBuiltError,
    InvalidDataError,
    VectorStoreError,
)
from rag.domain.models import (
    Chunk,
    Document,
    Embedding,
    IndexManifest,
    RetrievedChunk,
    SkippedFile,
    SourceScan,
)

_WORD = re.compile(r"[a-z0-9]+")


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors: 1.0 means they point the same way."""
    norms = math.hypot(*a) * math.hypot(*b)
    if norms == 0:
        raise VectorStoreError("cannot compare a zero vector")
    return math.sumprod(a, b) / norms


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts that share words get similar vectors.

    Deterministic across runs (blake2b, not Python's per-process randomised hash()), and
    records every call so tests can check batching.
    """

    def __init__(self, *, dim: int = 64, model_name: str = "fake-embed") -> None:
        self.dim = dim
        self._model_name = model_name
        self.calls: list[tuple[str, int]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        self.calls.append(("documents", len(texts)))
        return self._embed_all(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[Embedding]:
        self.calls.append(("queries", len(texts)))
        return self._embed_all(texts)

    def _embed_all(self, texts: Sequence[str]) -> list[Embedding]:
        blank = [i for i, text in enumerate(texts) if not text.strip()]
        if blank:
            raise EmbeddingError(f"cannot embed blank text (input positions {blank})")
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> Embedding:
        vector = [0.0] * self.dim
        for word in _WORD.findall(text.lower()) or [text.strip()]:
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest) % self.dim] += 1.0
        norm = math.hypot(*vector)
        return [x / norm for x in vector]


class InMemoryIndex:
    """Brute-force cosine-similarity search over a list. Ties keep insertion order."""

    def __init__(self) -> None:
        self._manifest: IndexManifest | None = None
        self._rows: list[tuple[Chunk, Embedding]] = []
        self._ids: set[str] = set()
        self._dim: int | None = None

    def reset(self, manifest: IndexManifest) -> None:
        self._manifest = manifest
        self._rows = []
        self._ids = set()
        self._dim = None

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        self._require_built()
        if len(chunks) != len(embeddings):
            raise VectorStoreError(f"got {len(chunks)} chunks but {len(embeddings)} embeddings")
        ids = [c.chunk_id for c in chunks]
        repeated = {i for i, n in Counter(ids).items() if n > 1} | (self._ids & set(ids))
        if repeated:
            raise VectorStoreError(f"duplicate chunk ids: {', '.join(sorted(repeated)[:5])}")
        sizes = {len(e) for e in embeddings} | ({self._dim} if self._dim is not None else set())
        if len(sizes) > 1 or 0 in sizes:
            raise VectorStoreError(
                f"embedding sizes must match and be non-zero, got {sorted(sizes)}"
            )
        if embeddings:
            self._dim = len(embeddings[0])
        self._rows.extend((c, list(e)) for c, e in zip(chunks, embeddings, strict=True))
        self._ids.update(ids)

    def query(self, embeddings: Sequence[Embedding], k: int) -> list[list[RetrievedChunk]]:
        self._require_built()
        if k < 1:
            raise InvalidDataError(f"k must be 1 or more, got {k}")
        return [self._search(embedding, k) for embedding in embeddings]

    def count(self) -> int:
        return len(self._rows)

    def manifest(self) -> IndexManifest | None:
        return self._manifest

    def _search(self, query: Embedding, k: int) -> list[RetrievedChunk]:
        if self._dim is not None and len(query) != self._dim:
            raise VectorStoreError(f"query has {len(query)} dimensions, the index has {self._dim}")
        scored = sorted(
            ((cosine_similarity(query, e), i) for i, (_, e) in enumerate(self._rows)),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [
            RetrievedChunk(chunk=self._rows[i][0], score=score, rank=rank)
            for rank, (score, i) in enumerate(scored[:k], start=1)
        ]

    def _require_built(self) -> None:
        if self._manifest is None:
            raise IndexNotBuiltError("the index has not been built", hint="run `rag ingest`")


class FakeChatModel:
    """Returns a fixed reply, or one computed from the prompt, and records every prompt."""

    def __init__(
        self,
        reply: str | Callable[[str, str], str] = "I don't know.",
        *,
        model_name: str = "fake-chat",
    ) -> None:
        self._reply = reply
        self._model_name = model_name
        self.prompts: list[tuple[str, str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, *, system: str, user: str) -> str:
        self.prompts.append((system, user))
        text = self._reply(system, user) if callable(self._reply) else self._reply
        if not text.strip():
            raise GenerationError("the model returned an empty reply")
        return text


@dataclass
class FakeDocumentSource:
    """Hands back documents that a test has already built."""

    documents: Sequence[Document]
    skipped: Sequence[SkippedFile] = ()

    def scan(self) -> SourceScan:
        return SourceScan(documents=tuple(self.documents), skipped=tuple(self.skipped))

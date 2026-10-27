"""Interfaces between the pipeline and the outside world.

Services depend only on these Protocols. Real implementations live in rag.adapters
(file system, pypdf, Ollama, chromadb) and in-memory fakes live in tests/fakes.py.
Each port's docstrings are its contract, and tests/contracts/ checks implementations
against them, the fakes included.
"""

from collections.abc import Sequence
from typing import Protocol

from rag.domain.models import Chunk, Document, Embedding, IndexManifest, RetrievedChunk, SourceScan


class DocumentSource(Protocol):
    """Finds and parses the documents to index."""

    def scan(self) -> SourceScan:
        """Load every supported document in the collection.

        Files that are not PDF or markdown come back in `SourceScan.skipped` with a reason;
        they are never dropped silently. Raises DocumentLoadError if a supported file cannot
        be parsed (including a PDF with no extractable text) or if two documents share a
        file name, because the test set names documents by file name.
        """


class ChunkingStrategy(Protocol):
    """Splits a document into chunks. Pure: no I/O, and the same input gives the same output."""

    @property
    def name(self) -> str:
        """Stable id that includes the parameters, e.g. 'fixed-800-100'. Names the index."""

    def split(self, document: Document) -> list[Chunk]:
        """Chunks in reading order, with ordinals 0, 1, 2, ...

        A chunk never crosses a segment boundary, so it keeps that segment's page and section.
        """


class Embedder(Protocol):
    """Turns text into vectors.

    Documents and queries are separate calls because some models, nomic-embed-text among
    them, expect a different task prefix for each.
    """

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        """One vector per text, in input order, all the same length; [] for [].

        Raises EmbeddingError for blank text or for text longer than the model accepts
        (it never truncates silently), and ModelUnavailableError if the model can't be
        reached.
        """

    def embed_queries(self, texts: Sequence[str]) -> list[Embedding]:
        """Same contract as embed_documents, for search queries."""


class VectorIndex(Protocol):
    """One searchable collection of chunk embeddings.

    Built with one chunking strategy and one embedding model, and stored with an
    IndexManifest that records which.
    """

    def reset(self, manifest: IndexManifest) -> None:
        """Delete everything and start an empty index described by `manifest`."""

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """Store chunks with their embeddings. All or nothing: a failed add stores nothing.

        Raises IndexNotBuiltError before the first reset(), and VectorStoreError when the
        two lengths differ, a vector has the wrong size, or a chunk id is already stored.
        """

    def query(self, embeddings: Sequence[Embedding], k: int) -> list[list[RetrievedChunk]]:
        """The k most similar chunks for each query embedding, best first (rank 1).

        Takes a batch, so evaluating 50 questions is one call instead of 50. Returns fewer
        than k results when the index holds fewer chunks. Raises IndexNotBuiltError before
        the first reset(), VectorStoreError for a wrong vector size, and InvalidDataError
        if k is below 1.
        """

    def count(self) -> int:
        """Number of stored chunks; 0 if the index has never been built."""

    def manifest(self) -> IndexManifest | None:
        """What the index was built with, or None if it has never been built."""


class ChatModel(Protocol):
    """Generates a reply from a prompt."""

    @property
    def model_name(self) -> str: ...

    def complete(self, *, system: str, user: str) -> str:
        """The model's reply to one system and user message.

        Raises GenerationError for an error or an empty reply, and ModelUnavailableError
        if the model can't be reached.
        """

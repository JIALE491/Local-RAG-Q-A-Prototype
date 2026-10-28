"""Ingest: rebuild one chunking strategy's index from every document, from scratch."""

import itertools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from rag.domain.errors import InvalidDataError, NoDocumentsError
from rag.domain.models import Chunk, IndexManifest, SkippedFile
from rag.domain.ports import ChunkingStrategy, DocumentSource, Embedder, VectorIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class IngestReport:
    """What one ingest run did, and how long embedding took."""

    strategy: str
    embed_model: str
    chunks_per_document: tuple[tuple[str, int], ...]
    """(file name, number of chunks) for each indexed document, in scan order."""
    skipped: tuple[SkippedFile, ...]
    batch_sizes: tuple[int, ...]
    batch_seconds: tuple[float, ...]
    """Embedding time per batch. The first batch also includes loading the model, if needed."""
    total_seconds: float

    @property
    def chunks(self) -> int:
        return sum(self.batch_sizes)

    @property
    def embed_seconds(self) -> float:
        return sum(self.batch_seconds)

    @property
    def steady_chunks_per_second(self) -> float | None:
        """Embedding speed after the first batch; None when there is only one batch."""
        later_seconds = sum(self.batch_seconds[1:])
        if len(self.batch_sizes) < 2 or later_seconds <= 0:
            return None
        return sum(self.batch_sizes[1:]) / later_seconds


def ingest(
    *,
    source: DocumentSource,
    strategy: ChunkingStrategy,
    embedder: Embedder,
    index: VectorIndex,
    batch_size: int,
    clock: Callable[[], float] = time.perf_counter,
) -> IngestReport:
    """Load, chunk, embed and store every document, replacing the strategy's previous index.

    Chunking happens before the index is touched, and the index only becomes searchable once
    the last batch is stored. A failure part-way, such as Ollama stopping, never leaves a
    partial index that looks complete.
    """
    if batch_size < 1:
        raise InvalidDataError(f"batch_size must be 1 or more, got {batch_size}")
    started = clock()
    scan = source.scan()
    for skipped in scan.skipped:
        logger.warning("skipped %s: %s", skipped.path, skipped.reason)
    if not scan.documents:
        raise NoDocumentsError(
            "found no PDF or markdown documents to index",
            hint="check [paths] docs_dir in config.toml",
        )

    per_document: list[tuple[str, int]] = []
    chunks: list[Chunk] = []
    for document in scan.documents:
        document_chunks = strategy.split(document)
        per_document.append((document.info.name, len(document_chunks)))
        chunks.extend(document_chunks)
    if not chunks:
        raise NoDocumentsError(f"{strategy.name} produced no chunks from the documents")
    logger.info(
        "chunked %d documents into %d chunks with %s",
        len(scan.documents),
        len(chunks),
        strategy.name,
    )

    index.begin_rebuild()
    sizes: list[int] = []
    seconds: list[float] = []
    for batch in itertools.batched(chunks, batch_size):
        batch_started = clock()
        vectors = embedder.embed_documents([chunk.text for chunk in batch])
        seconds.append(clock() - batch_started)
        index.add(batch, vectors)
        sizes.append(len(batch))
        logger.info("embedded and stored %d/%d chunks", sum(sizes), len(chunks))
    index.finish_rebuild(IndexManifest(strategy=strategy.name, embed_model=embedder.model_name))

    return IngestReport(
        strategy=strategy.name,
        embed_model=embedder.model_name,
        chunks_per_document=tuple(per_document),
        skipped=scan.skipped,
        batch_sizes=tuple(sizes),
        batch_seconds=tuple(seconds),
        total_seconds=clock() - started,
    )

"""Ingest: chunk, embed in batches, and rebuild the index, using the fakes."""

from collections.abc import Sequence
from itertools import count

import pytest

from rag.chunking.fixed import FixedSizeChunker
from rag.domain.errors import EmbeddingError, IndexNotBuiltError, InvalidDataError, NoDocumentsError
from rag.domain.models import Document, Embedding, IndexManifest, Segment, SkippedFile
from rag.services.ingest import ingest
from tests.builders import doc_info
from tests.fakes import FakeDocumentSource, FakeEmbedder, InMemoryIndex

STRATEGY = FixedSizeChunker(size=300, overlap=50)


def documents(how_many: int) -> list[Document]:
    """Documents of one short segment each, so each becomes exactly one chunk."""
    return [
        Document(
            info=doc_info(f"doc{i:02d}.md"),
            segments=(Segment(text=f"Topic {i} explained.", page=None, section=None),),
        )
        for i in range(how_many)
    ]


class FailingEmbedder(FakeEmbedder):
    """Fails on the second batch, the way Ollama would if it stopped mid-run."""

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        if len(self.calls) == 1:
            raise EmbeddingError("Ollama stopped")
        return super().embed_documents(texts)


def run(docs: list[Document], index: InMemoryIndex, embedder: FakeEmbedder | None = None) -> None:
    ingest(
        source=FakeDocumentSource(docs),
        strategy=STRATEGY,
        embedder=embedder or FakeEmbedder(),
        index=index,
        batch_size=32,
    )


def test_builds_a_searchable_index_of_every_chunk() -> None:
    index = InMemoryIndex()
    report = ingest(
        source=FakeDocumentSource(documents(3)),
        strategy=STRATEGY,
        embedder=FakeEmbedder(),
        index=index,
        batch_size=32,
    )
    assert index.count() == report.chunks == 3
    assert index.manifest() == IndexManifest(strategy="fixed-300-50", embed_model="fake-embed")
    assert report.chunks_per_document == (("doc00.md", 1), ("doc01.md", 1), ("doc02.md", 1))


def test_embeds_in_batches_rather_than_one_call_per_chunk() -> None:
    embedder = FakeEmbedder()
    run(documents(70), InMemoryIndex(), embedder)
    assert embedder.calls == [("documents", 32), ("documents", 32), ("documents", 6)]


def test_reports_skipped_files() -> None:
    skipped = (SkippedFile(path="notes.txt", reason="unsupported file type .txt"),)
    report = ingest(
        source=FakeDocumentSource(documents(1), skipped),
        strategy=STRATEGY,
        embedder=FakeEmbedder(),
        index=InMemoryIndex(),
        batch_size=32,
    )
    assert report.skipped == skipped


def test_reingesting_drops_documents_that_are_gone() -> None:
    index = InMemoryIndex()
    run(documents(3), index)
    run(documents(3)[1:], index)
    assert sorted({chunk.doc.name for chunk in index.chunks()}) == ["doc01.md", "doc02.md"]


def test_a_failure_part_way_leaves_the_index_unsearchable() -> None:
    index = InMemoryIndex()
    run(documents(3), index)
    with pytest.raises(EmbeddingError):
        run(documents(40), index, FailingEmbedder())
    assert index.manifest() is None
    with pytest.raises(IndexNotBuiltError):
        index.query([[1.0] * 64], k=1)


def test_no_documents_is_an_error_with_a_hint() -> None:
    with pytest.raises(NoDocumentsError) as caught:
        run([], InMemoryIndex())
    assert caught.value.hint is not None


def test_timing_comes_from_the_clock() -> None:
    ticks = count()
    report = ingest(
        source=FakeDocumentSource(documents(70)),
        strategy=STRATEGY,
        embedder=FakeEmbedder(),
        index=InMemoryIndex(),
        batch_size=32,
        clock=lambda: float(next(ticks)),
    )
    assert report.batch_seconds == (1.0, 1.0, 1.0)
    assert report.embed_seconds == 3.0
    assert report.steady_chunks_per_second == 19.0


def test_a_single_batch_has_no_steady_speed() -> None:
    ticks = count()
    report = ingest(
        source=FakeDocumentSource(documents(2)),
        strategy=STRATEGY,
        embedder=FakeEmbedder(),
        index=InMemoryIndex(),
        batch_size=32,
        clock=lambda: float(next(ticks)),
    )
    assert report.steady_chunks_per_second is None


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(InvalidDataError, match="batch_size"):
        ingest(
            source=FakeDocumentSource(documents(1)),
            strategy=STRATEGY,
            embedder=FakeEmbedder(),
            index=InMemoryIndex(),
            batch_size=0,
        )

"""Contract for rag.domain.ports.VectorIndex."""

import datetime as dt
from collections.abc import Sequence

import pytest

from rag.domain.errors import IndexNotBuiltError, InvalidDataError, VectorStoreError
from rag.domain.models import Chunk, Embedding, IndexManifest
from rag.domain.ports import VectorIndex
from tests.builders import chunk

MANIFEST = IndexManifest(strategy="test-strategy", embed_model="test-embed")


def build(index: VectorIndex, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
    """One complete rebuild, the way ingest runs it."""
    index.begin_rebuild()
    index.add(chunks, embeddings)
    index.finish_rebuild(MANIFEST)


class VectorIndexContract:
    """Subclass and provide an `index` fixture that returns a fresh, never-built index."""

    # Rebuild lifecycle

    def test_new_index_is_empty_and_not_searchable(self, index: VectorIndex) -> None:
        assert index.manifest() is None
        assert index.count() == 0
        with pytest.raises(IndexNotBuiltError):
            index.query([[1.0, 0.0]], k=1)

    def test_finished_rebuild_stores_the_manifest(self, index: VectorIndex) -> None:
        build(index, [chunk()], [[1.0, 0.0]])
        assert index.manifest() == MANIFEST
        assert index.count() == 1

    def test_unfinished_rebuild_is_not_searchable(self, index: VectorIndex) -> None:
        index.begin_rebuild()
        index.add([chunk()], [[1.0, 0.0]])
        assert index.manifest() is None
        assert index.count() == 1
        with pytest.raises(IndexNotBuiltError):
            index.query([[1.0, 0.0]], k=1)

    def test_starting_a_rebuild_hides_the_previous_build(self, index: VectorIndex) -> None:
        build(index, [chunk()], [[1.0, 0.0]])
        index.begin_rebuild()
        assert index.manifest() is None
        assert index.count() == 0
        with pytest.raises(IndexNotBuiltError):
            index.query([[1.0, 0.0]], k=1)

    def test_rebuild_drops_chunks_of_documents_no_longer_ingested(self, index: VectorIndex) -> None:
        deleted = chunk("Refunds take 60 days.", doc="old-refund-policy.md")
        current = chunk("Refunds take 30 days.", doc="refund-policy.md")
        build(index, [deleted, current], [[1.0, 0.0], [0.9, 0.1]])
        build(index, [current], [[0.9, 0.1]])
        [results] = index.query([[1.0, 0.0]], k=5)
        assert [r.chunk for r in results] == [current]

    def test_rebuild_accepts_ids_from_the_previous_build(self, index: VectorIndex) -> None:
        build(index, [chunk("Old text.")], [[1.0, 0.0]])
        build(index, [chunk("Edited text.")], [[0.0, 1.0]])
        [results] = index.query([[0.0, 1.0]], k=5)
        assert [r.chunk.text for r in results] == ["Edited text."]

    def test_rebuild_allows_a_new_vector_size(self, index: VectorIndex) -> None:
        build(index, [chunk()], [[1.0, 0.0]])
        build(index, [chunk()], [[1.0, 0.0, 0.0]])
        [results] = index.query([[1.0, 0.0, 0.0]], k=1)
        assert len(results) == 1

    def test_add_without_a_rebuild_raises(self, index: VectorIndex) -> None:
        with pytest.raises(VectorStoreError, match="no rebuild"):
            index.add([chunk()], [[1.0, 0.0]])

    def test_add_after_a_finished_rebuild_raises(self, index: VectorIndex) -> None:
        build(index, [chunk(ordinal=0)], [[1.0, 0.0]])
        with pytest.raises(VectorStoreError, match="no rebuild"):
            index.add([chunk(ordinal=1)], [[0.0, 1.0]])

    def test_finish_without_a_rebuild_raises(self, index: VectorIndex) -> None:
        with pytest.raises(VectorStoreError, match="no rebuild"):
            index.finish_rebuild(MANIFEST)

    # Adding chunks within one rebuild

    def test_add_rejects_mismatched_lengths(self, index: VectorIndex) -> None:
        index.begin_rebuild()
        with pytest.raises(VectorStoreError):
            index.add([chunk()], [])

    def test_add_rejects_duplicate_ids_in_one_batch(self, index: VectorIndex) -> None:
        index.begin_rebuild()
        with pytest.raises(VectorStoreError, match="#0000"):
            index.add([chunk(), chunk()], [[1.0, 0.0], [0.0, 1.0]])
        assert index.count() == 0

    def test_add_rejects_an_id_already_added_in_this_rebuild(self, index: VectorIndex) -> None:
        index.begin_rebuild()
        index.add([chunk()], [[1.0, 0.0]])
        with pytest.raises(VectorStoreError, match="#0000"):
            index.add([chunk()], [[0.0, 1.0]])
        assert index.count() == 1

    def test_add_rejects_mixed_vector_sizes(self, index: VectorIndex) -> None:
        index.begin_rebuild()
        with pytest.raises(VectorStoreError):
            index.add([chunk(ordinal=0), chunk(ordinal=1)], [[1.0, 0.0], [1.0, 0.0, 0.0]])
        assert index.count() == 0

    def test_add_rejects_a_vector_size_change(self, index: VectorIndex) -> None:
        index.begin_rebuild()
        index.add([chunk(ordinal=0)], [[1.0, 0.0]])
        with pytest.raises(VectorStoreError):
            index.add([chunk(ordinal=1)], [[1.0, 0.0, 0.0]])

    # Searching a finished index

    def test_query_returns_best_match_first(self, index: VectorIndex) -> None:
        a, b, c = (chunk(f"text {i}", ordinal=i) for i in range(3))
        build(index, [a, b, c], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.6, 0.0]])
        [results] = index.query([[1.0, 0.0, 0.0]], k=2)
        assert [r.chunk for r in results] == [a, c]
        assert [r.rank for r in results] == [1, 2]
        assert [r.score for r in results] == pytest.approx([1.0, 0.8], abs=1e-5)

    def test_one_result_list_per_query(self, index: VectorIndex) -> None:
        a, b = chunk("a", ordinal=0), chunk("b", ordinal=1)
        build(index, [a, b], [[1.0, 0.0], [0.0, 1.0]])
        results = index.query([[0.0, 1.0], [1.0, 0.0]], k=1)
        assert [[r.chunk for r in hits] for hits in results] == [[b], [a]]

    def test_no_queries_give_no_results(self, index: VectorIndex) -> None:
        build(index, [chunk()], [[1.0, 0.0]])
        assert index.query([], k=3) == []

    def test_k_larger_than_index_returns_everything(self, index: VectorIndex) -> None:
        build(index, [chunk()], [[1.0, 0.0]])
        [results] = index.query([[1.0, 0.0]], k=5)
        assert len(results) == 1

    def test_empty_finished_index_returns_empty_results(self, index: VectorIndex) -> None:
        build(index, [], [])
        assert index.query([[1.0, 0.0]], k=3) == [[]]

    def test_query_rejects_a_wrong_vector_size(self, index: VectorIndex) -> None:
        build(index, [chunk()], [[1.0, 0.0]])
        with pytest.raises(VectorStoreError):
            index.query([[1.0, 0.0, 0.0]], k=1)

    def test_chunks_lists_every_chunk_by_document_then_position(self, index: VectorIndex) -> None:
        b1, a0, b0 = (
            chunk("b1", doc="b.md", ordinal=1),
            chunk("a0", doc="a.md"),
            chunk("b0", doc="b.md"),
        )
        build(index, [b1, a0, b0], [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
        assert index.chunks() == [a0, b0, b1]

    def test_chunks_needs_a_finished_rebuild(self, index: VectorIndex) -> None:
        index.begin_rebuild()
        index.add([chunk()], [[1.0, 0.0]])
        with pytest.raises(IndexNotBuiltError):
            index.chunks()

    def test_query_rejects_k_below_one(self, index: VectorIndex) -> None:
        build(index, [chunk()], [[1.0, 0.0]])
        with pytest.raises(InvalidDataError):
            index.query([[1.0, 0.0]], k=0)

    def test_chunks_round_trip_with_all_metadata(self, index: VectorIndex) -> None:
        pdf = chunk(
            "Refunds take 30 days.",
            doc="refund-policy.pdf",
            ordinal=3,
            page=2,
            section="Refund window",
            created=dt.date(2026, 3, 1),
        )
        markdown = chunk(
            "Carry over up to 5 days.", doc="handbook.md", section="Time off > Vacation"
        )
        bare = chunk("General questions.", doc="faq.md")
        vectors = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]
        build(index, [pdf, markdown, bare], vectors)
        returned = {
            hit.chunk.chunk_id: hit.chunk for hits in index.query(vectors, k=3) for hit in hits
        }
        assert returned == {c.chunk_id: c for c in (pdf, markdown, bare)}

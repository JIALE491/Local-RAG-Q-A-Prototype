"""Contract for rag.domain.ports.VectorIndex."""

import datetime as dt

import pytest

from rag.domain.errors import IndexNotBuiltError, InvalidDataError, VectorStoreError
from rag.domain.models import IndexManifest
from rag.domain.ports import VectorIndex
from tests.builders import chunk

MANIFEST = IndexManifest(strategy="test-strategy", embed_model="test-embed")


class VectorIndexContract:
    """Subclass and override the `index` fixture to return a fresh, never-built index."""

    @pytest.fixture
    def index(self) -> VectorIndex:
        raise NotImplementedError

    @pytest.fixture
    def built(self, index: VectorIndex) -> VectorIndex:
        index.reset(MANIFEST)
        return index

    def test_new_index_is_unbuilt_and_empty(self, index: VectorIndex) -> None:
        assert index.manifest() is None
        assert index.count() == 0

    def test_reset_stores_the_manifest(self, built: VectorIndex) -> None:
        assert built.manifest() == MANIFEST
        assert built.count() == 0

    def test_add_before_reset_raises(self, index: VectorIndex) -> None:
        with pytest.raises(IndexNotBuiltError):
            index.add([chunk()], [[1.0, 0.0]])

    def test_query_before_reset_raises(self, index: VectorIndex) -> None:
        with pytest.raises(IndexNotBuiltError):
            index.query([[1.0, 0.0]], k=1)

    def test_query_returns_best_match_first(self, built: VectorIndex) -> None:
        a, b, c = (chunk(f"text {i}", ordinal=i) for i in range(3))
        built.add([a, b, c], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.6, 0.0]])
        [results] = built.query([[1.0, 0.0, 0.0]], k=2)
        assert [r.chunk for r in results] == [a, c]
        assert [r.rank for r in results] == [1, 2]
        assert [r.score for r in results] == pytest.approx([1.0, 0.8], abs=1e-5)

    def test_one_result_list_per_query(self, built: VectorIndex) -> None:
        a, b = chunk("a", ordinal=0), chunk("b", ordinal=1)
        built.add([a, b], [[1.0, 0.0], [0.0, 1.0]])
        results = built.query([[0.0, 1.0], [1.0, 0.0]], k=1)
        assert [[r.chunk for r in hits] for hits in results] == [[b], [a]]

    def test_no_queries_give_no_results(self, built: VectorIndex) -> None:
        assert built.query([], k=3) == []

    def test_k_larger_than_index_returns_everything(self, built: VectorIndex) -> None:
        built.add([chunk()], [[1.0, 0.0]])
        [results] = built.query([[1.0, 0.0]], k=5)
        assert len(results) == 1

    def test_empty_index_returns_empty_results(self, built: VectorIndex) -> None:
        assert built.query([[1.0, 0.0]], k=3) == [[]]

    def test_count_reports_stored_chunks(self, built: VectorIndex) -> None:
        built.add([chunk(ordinal=0), chunk(ordinal=1)], [[1.0, 0.0], [0.0, 1.0]])
        assert built.count() == 2

    def test_reset_removes_previous_contents(self, built: VectorIndex) -> None:
        built.add([chunk()], [[1.0, 0.0]])
        built.reset(MANIFEST)
        assert built.count() == 0

    def test_reset_allows_a_new_vector_size(self, built: VectorIndex) -> None:
        built.add([chunk()], [[1.0, 0.0]])
        built.reset(MANIFEST)
        built.add([chunk()], [[1.0, 0.0, 0.0]])
        assert built.count() == 1

    def test_add_rejects_mismatched_lengths(self, built: VectorIndex) -> None:
        with pytest.raises(VectorStoreError):
            built.add([chunk()], [])

    def test_add_rejects_duplicate_ids_in_one_batch(self, built: VectorIndex) -> None:
        with pytest.raises(VectorStoreError, match="#0000"):
            built.add([chunk(), chunk()], [[1.0, 0.0], [0.0, 1.0]])
        assert built.count() == 0

    def test_add_rejects_an_id_already_stored(self, built: VectorIndex) -> None:
        built.add([chunk()], [[1.0, 0.0]])
        with pytest.raises(VectorStoreError, match="#0000"):
            built.add([chunk()], [[0.0, 1.0]])
        assert built.count() == 1

    def test_add_rejects_mixed_vector_sizes(self, built: VectorIndex) -> None:
        with pytest.raises(VectorStoreError):
            built.add([chunk(ordinal=0), chunk(ordinal=1)], [[1.0, 0.0], [1.0, 0.0, 0.0]])
        assert built.count() == 0

    def test_add_rejects_a_vector_size_change(self, built: VectorIndex) -> None:
        built.add([chunk(ordinal=0)], [[1.0, 0.0]])
        with pytest.raises(VectorStoreError):
            built.add([chunk(ordinal=1)], [[1.0, 0.0, 0.0]])

    def test_query_rejects_a_wrong_vector_size(self, built: VectorIndex) -> None:
        built.add([chunk()], [[1.0, 0.0]])
        with pytest.raises(VectorStoreError):
            built.query([[1.0, 0.0, 0.0]], k=1)

    def test_query_rejects_k_below_one(self, built: VectorIndex) -> None:
        with pytest.raises(InvalidDataError):
            built.query([[1.0, 0.0]], k=0)

    def test_chunks_round_trip_with_all_metadata(self, built: VectorIndex) -> None:
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
        built.add([pdf, markdown, bare], vectors)
        returned = {
            hit.chunk.chunk_id: hit.chunk for hits in built.query(vectors, k=3) for hit in hits
        }
        assert returned == {c.chunk_id: c for c in (pdf, markdown, bare)}

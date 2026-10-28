"""Retrieval with the fakes: relevance, batching, and refusing the wrong index."""

import pytest

from rag.domain.errors import IndexMismatchError, IndexNotBuiltError, InvalidDataError
from rag.domain.models import Chunk, IndexManifest
from rag.services.retrieve import Retriever
from tests.builders import chunk
from tests.fakes import FakeEmbedder, InMemoryIndex

STRATEGY = "fixed-300-50"
CHUNKS = [
    chunk("Refunds are issued within 30 days.", doc="refunds.md"),
    chunk("Employees get 25 vacation days per year.", doc="vacation.md"),
]


def built(chunks: list[Chunk], *, embed_model: str = "fake-embed") -> InMemoryIndex:
    index = InMemoryIndex()
    index.begin_rebuild()
    index.add(chunks, FakeEmbedder().embed_documents([c.text for c in chunks]))
    index.finish_rebuild(IndexManifest(strategy=STRATEGY, embed_model=embed_model))
    return index


def test_the_most_relevant_chunk_comes_first() -> None:
    retriever = Retriever(embedder=FakeEmbedder(), index=built(CHUNKS), strategy=STRATEGY)
    [results] = retriever.search(["How many vacation days do employees get?"], k=2)
    assert [result.chunk.doc.name for result in results] == ["vacation.md", "refunds.md"]


def test_all_questions_are_embedded_in_one_call() -> None:
    embedder = FakeEmbedder()
    retriever = Retriever(embedder=embedder, index=built(CHUNKS), strategy=STRATEGY)
    results = retriever.search(["refunds?", "vacation?", "holidays?"], k=1)
    assert len(results) == 3
    assert embedder.calls == [("queries", 3)]


def test_an_index_built_with_another_model_is_refused_before_embedding() -> None:
    embedder = FakeEmbedder()
    index = built(CHUNKS, embed_model="other-model")
    with pytest.raises(IndexMismatchError):
        Retriever(embedder=embedder, index=index, strategy=STRATEGY).search(["refunds?"], k=1)
    assert embedder.calls == []


def test_an_unbuilt_index_is_refused_with_a_hint() -> None:
    retriever = Retriever(embedder=FakeEmbedder(), index=InMemoryIndex(), strategy=STRATEGY)
    with pytest.raises(IndexNotBuiltError) as caught:
        retriever.search(["refunds?"], k=1)
    assert caught.value.hint is not None


@pytest.mark.parametrize(("question", "k"), [("  ", 3), ("refunds?", 0)])
def test_blank_questions_and_k_below_one_are_refused(question: str, k: int) -> None:
    retriever = Retriever(embedder=FakeEmbedder(), index=built(CHUNKS), strategy=STRATEGY)
    with pytest.raises(InvalidDataError):
        retriever.search([question], k=k)

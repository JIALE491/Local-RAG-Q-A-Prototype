"""Strategy 1: fixed-size windows of whole words."""

import itertools

import pytest

from rag.chunking.fixed import FixedSizeChunker
from rag.domain.errors import InvalidDataError
from rag.domain.models import Document, Segment
from rag.domain.ports import ChunkingStrategy
from tests.builders import doc_info

WORDS = [f"word{i:03d}" for i in range(200)]


def document(*segments: Segment, name: str = "guide.md") -> Document:
    return Document(info=doc_info(name), segments=segments)


def test_satisfies_the_port_and_names_its_parameters() -> None:
    strategy: ChunkingStrategy = FixedSizeChunker(size=300, overlap=50)
    assert strategy.name == "fixed-300-50"


def test_a_short_segment_is_one_chunk_with_whitespace_collapsed() -> None:
    doc = document(Segment(text="Refunds\n\ntake   5\ndays.", page=None, section="Refunds"))
    [only] = FixedSizeChunker(size=300, overlap=50).split(doc)
    assert (only.text, only.section, only.page, only.ordinal) == (
        "Refunds take 5 days.",
        "Refunds",
        None,
        0,
    )


def test_a_long_segment_becomes_overlapping_windows_of_whole_words() -> None:
    doc = document(Segment(text=" ".join(WORDS), page=None, section=None))
    chunks = [chunk.text.split() for chunk in FixedSizeChunker(size=100, overlap=30).split(doc)]
    assert len(chunks) > 1
    assert all(len(" ".join(words)) <= 100 for words in chunks)
    rebuilt = list(chunks[0])
    for previous, current in itertools.pairwise(chunks):
        shared = next(n for n in range(len(current), 0, -1) if current[:n] == previous[-n:])
        assert len(" ".join(current[:shared])) <= 30
        rebuilt.extend(current[shared:])
    assert rebuilt == WORDS


def test_chunks_never_cross_segments_and_keep_their_location() -> None:
    doc = document(
        Segment(text=" ".join(WORDS[:30]), page=1, section="Scope"),
        Segment(text=" ".join(WORDS[30:60]), page=2, section="Refunds"),
        name="policy.pdf",
    )
    chunks = FixedSizeChunker(size=100, overlap=30).split(doc)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        words = set(chunk.text.split())
        expected = (1, "Scope") if words <= set(WORDS[:30]) else (2, "Refunds")
        assert words <= set(WORDS[:30]) or words <= set(WORDS[30:60])
        assert (chunk.page, chunk.section) == expected


def test_a_word_longer_than_the_size_is_a_chunk_of_its_own() -> None:
    long_word = "x" * 50
    doc = document(Segment(text=f"short {long_word} end", page=None, section=None))
    texts = [chunk.text for chunk in FixedSizeChunker(size=20, overlap=5).split(doc)]
    assert texts == ["short", long_word, "end"]


def test_the_same_document_always_gives_the_same_chunks() -> None:
    doc = document(Segment(text=" ".join(WORDS), page=None, section=None))
    strategy = FixedSizeChunker(size=100, overlap=30)
    assert strategy.split(doc) == strategy.split(doc)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(InvalidDataError):
        FixedSizeChunker(size=size, overlap=overlap)

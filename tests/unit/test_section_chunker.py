"""Strategy 3: one chunk per segment, split only when a segment is too long."""

import pytest

from rag.chunking.sections import SectionChunker
from rag.domain.errors import InvalidDataError
from rag.domain.models import Document, Segment
from rag.domain.ports import ChunkingStrategy
from tests.builders import doc_info


def test_satisfies_the_port_and_names_its_limit() -> None:
    strategy: ChunkingStrategy = SectionChunker(max_size=4000)
    assert strategy.name == "sections-4000"


def test_each_segment_becomes_one_chunk_with_whitespace_collapsed() -> None:
    segments = (
        Segment(text="Scope\nWho this\ncovers.", page=1, section="Scope"),
        Segment(text="Refunds take\n5 days.", page=2, section="Refunds"),
    )
    chunks = SectionChunker(max_size=4000).split(
        Document(info=doc_info("policy.pdf"), segments=segments)
    )
    assert [(c.text, c.page, c.section, c.ordinal) for c in chunks] == [
        ("Scope Who this covers.", 1, "Scope", 0),
        ("Refunds take 5 days.", 2, "Refunds", 1),
    ]


def test_a_section_longer_than_the_limit_is_split_at_sentence_boundaries() -> None:
    text = " ".join(f"Sentence number {i} has a few words in it." for i in range(10))
    segment = Segment(text=text, page=None, section="Long")
    chunks = SectionChunker(max_size=200).split(
        Document(info=doc_info("guide.md"), segments=(segment,))
    )
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 and chunk.text.endswith(".") for chunk in chunks)
    assert " ".join(chunk.text for chunk in chunks) == text
    assert {chunk.section for chunk in chunks} == {"Long"}


def test_max_size_must_be_positive() -> None:
    with pytest.raises(InvalidDataError, match="max_size"):
        SectionChunker(max_size=0)

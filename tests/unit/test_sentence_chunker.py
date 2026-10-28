"""Strategy 2, and the sentence splitting it shares with strategy 3."""

import pytest

from rag.chunking.sentences import SentenceChunker, pack, split_sentences
from rag.domain.errors import InvalidDataError
from rag.domain.models import Document, Segment
from rag.domain.ports import ChunkingStrategy
from tests.builders import doc_info

SENTENCES = [f"Sentence number {i} has a few words in it." for i in range(20)]


def test_sentences_end_at_end_punctuation() -> None:
    assert split_sentences("Refunds take 5 days. Exchanges are free! Questions?  Ask us.") == [
        "Refunds take 5 days.",
        "Exchanges are free!",
        "Questions?",
        "Ask us.",
    ]


def test_abbreviations_and_decimals_do_not_end_a_sentence() -> None:
    text = (
        "Stores open at 9 a.m. Monday through Saturday. Mileage is $0.67 per mile. "
        "Martin Luther King Jr. Day is a holiday."
    )
    assert split_sentences(text) == [
        "Stores open at 9 a.m. Monday through Saturday.",
        "Mileage is $0.67 per mile.",
        "Martin Luther King Jr. Day is a holiday.",
    ]


def test_pdf_line_wraps_do_not_break_sentences() -> None:
    text = "Refunds go back to the original\npayment method. Card issuers\ncan take longer."
    assert split_sentences(text) == [
        "Refunds go back to the original payment method.",
        "Card issuers can take longer.",
    ]


def test_list_items_and_table_rows_are_sentences_of_their_own() -> None:
    text = "Not reimbursable:\n- Traffic fines\n- Pet care\n\n| Item | Price |\n| Tent | $25 |"
    assert split_sentences(text) == [
        "Not reimbursable:",
        "- Traffic fines",
        "- Pet care",
        "| Item | Price |",
        "| Tent | $25 |",
    ]


def test_a_list_item_that_wraps_stays_one_sentence() -> None:
    text = "- Hygiene items once their packaging is\nopened\n- Gift cards"
    assert split_sentences(text) == [
        "- Hygiene items once their packaging is opened",
        "- Gift cards",
    ]


def test_pack_never_splits_a_sentence() -> None:
    assert list(pack(["a" * 10, "b" * 10, "c" * 10], size=21)) == [
        f"{'a' * 10} {'b' * 10}",
        "c" * 10,
    ]


def test_pack_keeps_a_sentence_longer_than_the_size_whole() -> None:
    assert list(pack(["short", "x" * 50, "end"], size=20)) == ["short", "x" * 50, "end"]


def test_satisfies_the_port_and_names_its_size() -> None:
    strategy: ChunkingStrategy = SentenceChunker(size=300)
    assert strategy.name == "sentences-300"


def test_chunks_are_whole_sentences_within_the_size_with_nothing_lost() -> None:
    segment = Segment(text=" ".join(SENTENCES), page=None, section="Guide")
    chunks = SentenceChunker(size=100).split(
        Document(info=doc_info("guide.md"), segments=(segment,))
    )
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert all(chunk.text.endswith(".") for chunk in chunks)
    assert " ".join(chunk.text for chunk in chunks) == " ".join(SENTENCES)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.section for chunk in chunks} == {"Guide"}


def test_chunks_never_cross_segments() -> None:
    segments = (
        Segment(text=" ".join(SENTENCES[:3]), page=1, section="Scope"),
        Segment(text=" ".join(SENTENCES[3:6]), page=2, section="Refunds"),
    )
    chunks = SentenceChunker(size=100).split(
        Document(info=doc_info("policy.pdf"), segments=segments)
    )
    for chunk in chunks:
        first = chunk.text.split(".")[0] + "."
        expected = (1, "Scope") if first in SENTENCES[:3] else (2, "Refunds")
        assert (chunk.page, chunk.section) == expected


def test_size_must_be_positive() -> None:
    with pytest.raises(InvalidDataError, match="size"):
        SentenceChunker(size=0)

"""The frozen fact list against the real corpus: every fact must survive text extraction."""

from rag.adapters.filesystem import DirectorySource
from rag.adapters.questions_csv import load_test_set
from rag.domain.models import contains_fact
from scripts.build_pdfs import OUTPUT_DIR

EVAL_DIR = OUTPUT_DIR.parent / "eval"
TEST_SET = load_test_set(EVAL_DIR / "questions.csv", EVAL_DIR / "facts.csv")
SEGMENTS = {
    document.info.name: document.segments
    for document in DirectorySource(OUTPUT_DIR).scan().documents
}
KNOWN_PAGE_BREAKS = {
    ("q44", "Tell your manager about any change that keeps you away longer than planned"),
}
"""Facts that a page break splits, with the page footer in between (decisions 10 and 23)."""


def test_every_fact_is_whole_in_one_segment_of_its_expected_document() -> None:
    """A fact split across segments could never be inside one chunk, whatever the strategy."""
    broken = {
        (question.qid, fact)
        for question in TEST_SET
        if question.expected_doc is not None
        for fact in question.facts
        if not any(contains_fact(s.text, fact) for s in SEGMENTS[question.expected_doc])
    }
    assert broken == KNOWN_PAGE_BREAKS


def test_the_known_exception_is_split_by_the_page_break() -> None:
    segments = SEGMENTS["travel-and-expense-policy.pdf"]
    end_of_page_one = [s.text for s in segments if s.page == 1][-1]
    start_of_page_two = next(s.text for s in segments if s.page == 2)
    assert contains_fact(end_of_page_one, "Tell your manager about any change that")
    assert end_of_page_one.endswith("Page 1 of 2")
    assert start_of_page_two.startswith("keeps you away longer than planned.")

"""Domain rules: every type rejects bad data when it is created."""

import datetime as dt
import math

import pytest

from rag.domain.errors import IndexMismatchError, InvalidDataError
from rag.domain.models import (
    Answer,
    Chunk,
    DateSource,
    Document,
    DocumentInfo,
    DocumentKind,
    EvalQuestion,
    EvalReport,
    IndexManifest,
    QuestionResult,
    RetrievedChunk,
    Segment,
    Source,
)
from tests.builders import chunk, doc_info


class TestDocumentInfo:
    def test_accepts_valid_info(self) -> None:
        info = DocumentInfo(
            doc_id="policies/refund-policy.pdf",
            name="refund-policy.pdf",
            kind=DocumentKind.PDF,
            created=dt.date(2026, 3, 1),
            created_from=DateSource.PDF_METADATA,
        )
        assert info.name == "refund-policy.pdf"

    @pytest.mark.parametrize("name", ["", "   "])
    def test_rejects_a_blank_name(self, name: str) -> None:
        with pytest.raises(InvalidDataError, match="name"):
            DocumentInfo(
                doc_id="x.md",
                name=name,
                kind=DocumentKind.MARKDOWN,
                created=None,
                created_from=None,
            )

    def test_created_date_needs_its_source(self) -> None:
        with pytest.raises(InvalidDataError, match="together"):
            DocumentInfo(
                doc_id="x.md",
                name="x.md",
                kind=DocumentKind.MARKDOWN,
                created=dt.date(2026, 1, 1),
                created_from=None,
            )


class TestSegmentAndDocument:
    def test_segment_rejects_blank_text(self) -> None:
        with pytest.raises(InvalidDataError, match="blank"):
            Segment(text=" \n ", page=1, section=None)

    def test_segment_rejects_page_zero(self) -> None:
        with pytest.raises(InvalidDataError, match="page"):
            Segment(text="text", page=0, section=None)

    def test_segment_rejects_a_blank_section(self) -> None:
        with pytest.raises(InvalidDataError, match="section"):
            Segment(text="text", page=1, section="  ")

    def test_accepts_a_valid_pdf_document(self) -> None:
        segments = (
            Segment(text="Our refund policy.", page=1, section=None),
            Segment(text="Refunds take 30 days.", page=2, section="Refund window"),
        )
        document = Document(info=doc_info("refund-policy.pdf"), segments=segments)
        assert [s.page for s in document.segments] == [1, 2]

    def test_document_needs_at_least_one_segment(self) -> None:
        with pytest.raises(InvalidDataError, match="no text"):
            Document(info=doc_info("scan.pdf"), segments=())

    def test_pdf_segments_need_a_page(self) -> None:
        with pytest.raises(InvalidDataError, match="page number"):
            Document(info=doc_info("a.pdf"), segments=(Segment(text="t", page=None, section=None),))

    def test_markdown_segments_have_no_page(self) -> None:
        with pytest.raises(InvalidDataError, match="no pages"):
            Document(info=doc_info("a.md"), segments=(Segment(text="t", page=2, section=None),))


class TestChunk:
    def test_id_is_readable_and_deterministic(self) -> None:
        first = chunk(doc="refund-policy.pdf", ordinal=3)
        again = chunk(doc="refund-policy.pdf", ordinal=3)
        assert first.chunk_id == again.chunk_id == "refund-policy.pdf#0003"

    def test_rejects_a_negative_ordinal(self) -> None:
        with pytest.raises(InvalidDataError, match="ordinal"):
            chunk(ordinal=-1)

    def test_rejects_blank_text(self) -> None:
        with pytest.raises(InvalidDataError, match="blank"):
            chunk("   ")

    def test_pdf_chunk_needs_a_page(self) -> None:
        with pytest.raises(InvalidDataError, match="page number"):
            Chunk(doc=doc_info("a.pdf"), ordinal=0, text="t", page=None, section=None)

    def test_markdown_chunk_has_no_page(self) -> None:
        with pytest.raises(InvalidDataError, match="no pages"):
            Chunk(doc=doc_info("a.md"), ordinal=0, text="t", page=1, section=None)

    def test_source_carries_the_location(self) -> None:
        located = chunk(doc="refund-policy.pdf", page=2, section="Refund window")
        assert located.source == Source(
            doc_name="refund-policy.pdf", page=2, section="Refund window"
        )


class TestSourceLabel:
    """Success criterion 1: document name plus page (PDF) or heading (markdown)."""

    @pytest.mark.parametrize(
        ("page", "section", "label"),
        [
            (3, "Refund window", "refund-policy.pdf, p. 3 (Refund window)"),
            (3, None, "refund-policy.pdf, p. 3"),
        ],
    )
    def test_pdf(self, page: int, section: str | None, label: str) -> None:
        assert Source(doc_name="refund-policy.pdf", page=page, section=section).label() == label

    @pytest.mark.parametrize(
        ("section", "label"),
        [
            ("Time off > Vacation", "handbook.md (Time off > Vacation)"),
            (None, "handbook.md"),
        ],
    )
    def test_markdown(self, section: str | None, label: str) -> None:
        assert Source(doc_name="handbook.md", page=None, section=section).label() == label


class TestRetrievedChunk:
    def test_rejects_rank_zero(self) -> None:
        with pytest.raises(InvalidDataError, match="rank"):
            RetrievedChunk(chunk=chunk(), score=0.9, rank=0)

    @pytest.mark.parametrize("score", [math.nan, math.inf])
    def test_rejects_a_non_finite_score(self, score: float) -> None:
        with pytest.raises(InvalidDataError, match="score"):
            RetrievedChunk(chunk=chunk(), score=score, rank=1)


class TestAnswer:
    def test_keeps_its_sources_in_rank_order(self) -> None:
        sources = (
            RetrievedChunk(chunk=chunk(ordinal=0), score=0.9, rank=1),
            RetrievedChunk(chunk=chunk(ordinal=1), score=0.7, rank=2),
        )
        answer = Answer(
            question="How long?", text="30 days [1].", sources=sources, model="llama3.2:3b"
        )
        assert [s.rank for s in answer.sources] == [1, 2]

    def test_rejects_blank_text(self) -> None:
        with pytest.raises(InvalidDataError, match="answer text"):
            Answer(question="How long?", text=" ", sources=(), model="llama3.2:3b")


class TestIndexManifest:
    MANIFEST = IndexManifest(strategy="fixed-800-100", embed_model="nomic-embed-text")

    def test_same_settings_are_compatible(self) -> None:
        self.MANIFEST.ensure_compatible(strategy="fixed-800-100", embed_model="nomic-embed-text")

    @pytest.mark.parametrize(
        ("strategy", "embed_model"),
        [("fixed-800-100", "other-model"), ("other-strategy", "nomic-embed-text")],
    )
    def test_different_settings_raise_with_a_hint(self, strategy: str, embed_model: str) -> None:
        with pytest.raises(IndexMismatchError) as caught:
            self.MANIFEST.ensure_compatible(strategy=strategy, embed_model=embed_model)
        assert caught.value.hint is not None
        assert "rag ingest" in caught.value.hint


def result(expected: str, *retrieved_docs: str) -> QuestionResult:
    """A QuestionResult whose retrieved chunks come from `retrieved_docs`, in rank order."""
    retrieved = tuple(
        RetrievedChunk(chunk=chunk(doc=doc, ordinal=i), score=1.0 - i / 10, rank=i + 1)
        for i, doc in enumerate(retrieved_docs)
    )
    question = EvalQuestion(qid="q01", question="How long do refunds take?", expected_doc=expected)
    return QuestionResult(question=question, retrieved=retrieved)


class TestEvaluation:
    def test_hit_rank_is_the_first_chunk_from_the_expected_doc(self) -> None:
        assert result("b.md", "a.md", "b.md", "b.md").hit_rank == 2

    def test_a_miss_has_no_rank(self) -> None:
        assert result("c.md", "a.md", "b.md").hit_rank is None

    @pytest.mark.parametrize(("k", "hit"), [(3, True), (2, False)])
    def test_hit_depends_on_the_cut_off(self, k: int, hit: bool) -> None:
        assert result("c.md", "a.md", "b.md", "c.md").is_hit(k) is hit

    def test_report_counts_hits_and_misses(self) -> None:
        hit_1, hit_2 = result("a.md", "a.md"), result("b.md", "x.md", "b.md")
        miss_1, miss_2 = result("c.md", "x.md"), result("d.md", "x.md", "y.md", "z.md", "d.md")
        report = EvalReport(strategy="s", k=3, results=(hit_1, miss_1, hit_2, miss_2))
        assert report.hits == 2
        assert report.hit_rate == 0.5
        assert report.misses == (miss_1, miss_2)

    def test_report_needs_at_least_one_question(self) -> None:
        with pytest.raises(InvalidDataError, match="at least one"):
            EvalReport(strategy="s", k=3, results=())

    def test_report_rejects_k_below_one(self) -> None:
        with pytest.raises(InvalidDataError, match="k must"):
            EvalReport(strategy="s", k=0, results=(result("a.md", "a.md"),))

    @pytest.mark.parametrize("blank_field", ["qid", "question", "expected_doc"])
    def test_question_rejects_blank_fields(self, blank_field: str) -> None:
        fields = {"qid": "q01", "question": "How long?", "expected_doc": "a.md", blank_field: " "}
        with pytest.raises(InvalidDataError, match="blank"):
            EvalQuestion(**fields)

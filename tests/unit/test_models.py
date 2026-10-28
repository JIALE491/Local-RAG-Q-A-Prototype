"""Domain rules: every type rejects bad data when it is created."""

import datetime as dt
import math

import pytest

from rag.domain.errors import IndexMismatchError, InvalidDataError
from rag.domain.models import (
    Answer,
    AnsweredQuestion,
    Chunk,
    Completion,
    Correctness,
    DateSource,
    Document,
    DocumentInfo,
    DocumentKind,
    EvalQuestion,
    EvalReport,
    Grade,
    IndexManifest,
    QuestionResult,
    RetrievedChunk,
    Segment,
    Source,
    adoption_order,
    contains_fact,
    count_grades,
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


def answer(text: str, *, sources: int = 2) -> Answer:
    """An answer whose prompt held `sources` chunks."""
    retrieved = tuple(
        RetrievedChunk(chunk=chunk(ordinal=i), score=0.9 - i / 10, rank=i + 1)
        for i in range(sources)
    )
    return Answer(
        question="How long?",
        text=text,
        sources=retrieved,
        prompt="Sources: ... Question: How long?",
        model="llama3.2:3b",
        prompt_tokens=900,
        answer_tokens=12,
    )


class TestCompletion:
    def test_rejects_a_blank_reply(self) -> None:
        with pytest.raises(InvalidDataError, match="reply"):
            Completion(text=" ", prompt_tokens=10, answer_tokens=1)

    def test_rejects_token_counts_below_one(self) -> None:
        with pytest.raises(InvalidDataError, match="token counts"):
            Completion(text="Yes.", prompt_tokens=10, answer_tokens=0)


class TestAnswer:
    def test_keeps_its_sources_in_rank_order(self) -> None:
        assert [s.rank for s in answer("30 days [1].").sources] == [1, 2]

    def test_rejects_blank_text(self) -> None:
        with pytest.raises(InvalidDataError, match="answer text"):
            answer(" ")

    def test_rejects_token_counts_below_one(self) -> None:
        with pytest.raises(InvalidDataError, match="token counts"):
            Answer(
                question="How long?",
                text="30 days.",
                sources=(),
                prompt="Sources: ... Question: How long?",
                model="llama3.2:3b",
                prompt_tokens=0,
                answer_tokens=3,
            )

    @pytest.mark.parametrize(
        "text", ["I don't know.", "I don't know. [1]", "  i DON\u2019T know [2]\n", "I don't know"]
    )
    def test_a_reply_of_just_i_dont_know_says_it_doesnt_know(self, text: str) -> None:
        assert answer(text).says_dont_know

    @pytest.mark.parametrize(
        "text",
        ["I don't know the exact date, but it is 30 days [1].", "Refunds take 30 days [1].", "No."],
    )
    def test_any_other_reply_is_an_answer(self, text: str) -> None:
        assert not answer(text).says_dont_know

    def test_citations_are_read_in_order_of_first_mention_each_once(self) -> None:
        assert answer("Yes [2]. It takes 30 days [1][2], or 45 [1, 2].").citations == (2, 1)

    def test_an_answer_without_numbers_in_brackets_cites_nothing(self) -> None:
        reply = answer("I don't know. [Note: nothing matched.]")
        assert (reply.citations, reply.cited, reply.invalid_citations) == ((), (), ())

    def test_numbers_that_match_no_source_are_invalid_citations(self) -> None:
        reply = answer("30 days [1], or see [3] and [0].", sources=2)
        assert reply.cited == (1,)
        assert reply.invalid_citations == (3, 0)

    def test_source_numbers_count_from_one(self) -> None:
        reply = answer("30 days [2].")
        assert reply.source(2) is reply.sources[1]
        with pytest.raises(InvalidDataError, match=r"no source \[3\]"):
            reply.source(3)


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


def result(expected: str, *retrieved_docs: str, qid: str = "q01") -> QuestionResult:
    """A QuestionResult whose retrieved chunks come from `retrieved_docs`, in rank order.

    Every chunk has the builders' default text, which contains the question's one fact.
    """
    retrieved = tuple(
        RetrievedChunk(chunk=chunk(doc=doc, ordinal=i), score=1.0 - i / 10, rank=i + 1)
        for i, doc in enumerate(retrieved_docs)
    )
    question = EvalQuestion(
        qid=qid, question="How long do refunds take?", expected_doc=expected, facts=("30 days",)
    )
    return QuestionResult(question=question, retrieved=retrieved)


def covering(facts: tuple[str, ...], *retrieved: tuple[str, str]) -> QuestionResult:
    """A result for a question about refunds.md whose chunks are (document, text), best first."""
    chunks = tuple(
        RetrievedChunk(chunk=chunk(text, doc=doc, ordinal=i), score=1.0 - i / 10, rank=i + 1)
        for i, (doc, text) in enumerate(retrieved)
    )
    question = EvalQuestion(
        qid="q01", question="How do refunds work?", expected_doc="refunds.md", facts=facts
    )
    return QuestionResult(question=question, retrieved=chunks)


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

    def test_hits_can_be_counted_at_any_cut_off(self) -> None:
        report = EvalReport(
            strategy="s",
            k=3,
            results=(result("a.md", "a.md"), result("b.md", "x.md", "y.md", "b.md")),
        )
        assert (report.hits_at(1), report.hits_at(3), report.hits) == (1, 2, 2)
        with pytest.raises(InvalidDataError, match="n must"):
            report.hits_at(0)

    def test_report_needs_at_least_one_question(self) -> None:
        with pytest.raises(InvalidDataError, match="at least one"):
            EvalReport(strategy="s", k=3, results=())

    def test_report_rejects_k_below_one(self) -> None:
        with pytest.raises(InvalidDataError, match="k must"):
            EvalReport(strategy="s", k=0, results=(result("a.md", "a.md"),))

    @pytest.mark.parametrize("blank_field", ["qid", "question", "expected_doc", "expected_answer"])
    def test_question_rejects_blank_fields(self, blank_field: str) -> None:
        fields = {"qid": "q01", "question": "How long?", "expected_doc": "a.md", blank_field: " "}
        with pytest.raises(InvalidDataError, match="blank"):
            EvalQuestion(
                qid=fields["qid"],
                question=fields["question"],
                expected_doc=fields["expected_doc"],
                expected_answer=fields.get("expected_answer"),
            )

    def test_question_can_carry_a_reference_answer(self) -> None:
        question = EvalQuestion(
            qid="q01", question="How long?", expected_doc="a.md", expected_answer="45 days"
        )
        assert question.answerable
        assert question.expected_answer == "45 days"

    def test_question_without_a_document_is_unanswerable(self) -> None:
        question = EvalQuestion(qid="u01", question="Who painted the mural?", expected_doc=None)
        assert not question.answerable

    def test_unanswerable_questions_are_not_scored(self) -> None:
        question = EvalQuestion(qid="u01", question="Who painted the mural?", expected_doc=None)
        with pytest.raises(InvalidDataError, match="unanswerable"):
            QuestionResult(question=question, retrieved=())

    def test_a_scored_question_needs_its_facts(self) -> None:
        question = EvalQuestion(qid="q01", question="How long?", expected_doc="a.md")
        with pytest.raises(InvalidDataError, match="needs the facts"):
            QuestionResult(question=question, retrieved=())

    @pytest.mark.parametrize(
        ("facts", "problem"),
        [(("30 days", " "), "a fact must not be blank"), (("30 days", "30 days"), "twice")],
    )
    def test_facts_must_be_filled_in_and_distinct(
        self, facts: tuple[str, ...], problem: str
    ) -> None:
        with pytest.raises(InvalidDataError, match=problem):
            EvalQuestion(qid="q01", question="How long?", expected_doc="a.md", facts=facts)

    def test_a_question_that_no_document_answers_has_no_facts(self) -> None:
        with pytest.raises(InvalidDataError, match="no facts"):
            EvalQuestion(qid="u01", question="Wi-Fi?", expected_doc=None, facts=("hunter2",))


class TestAnswerCoverage:
    def test_facts_match_ignoring_case_and_spacing(self) -> None:
        assert contains_fact("Refunds  within\n30 DAYS of purchase.", "refunds within 30 days")

    def test_a_fact_broken_by_other_text_does_not_match(self) -> None:
        text = "Refunds within 30 - Policy, page 1 of 2 - days of purchase."
        assert not contains_fact(text, "within 30 days of purchase")

    def test_covered_once_every_fact_is_in_the_top_chunks_of_the_expected_document(self) -> None:
        result = covering(
            ("30 days", "store credit"),
            ("refunds.md", "Refunds are paid within 30 days."),
            ("faq.md", "Without a receipt you get store credit."),
            ("refunds.md", "After that, you get store credit."),
        )
        assert result.missing_facts(2) == ("store credit",)
        assert not result.is_covered(2)
        assert result.is_covered(3)
        assert result.coverage_rank == 3

    def test_each_fact_must_be_whole_inside_one_chunk(self) -> None:
        result = covering(
            ("within 30 days of purchase",),
            ("refunds.md", "Refunds are paid within 30"),
            ("refunds.md", "days of purchase."),
        )
        assert result.missing_facts(2) == ("within 30 days of purchase",)
        assert result.coverage_rank is None

    def test_report_counts_coverage_at_any_cut_off(self) -> None:
        first = covering(("30 days",), ("refunds.md", "Paid within 30 days."))
        second = covering(
            ("store credit",), ("faq.md", "Ask us."), ("refunds.md", "You get store credit.")
        )
        report = EvalReport(strategy="s", k=1, results=(first, second))
        assert (report.covered_at(1), report.covered_at(2)) == (1, 2)
        assert report.uncovered == (second,)
        with pytest.raises(InvalidDataError, match="n must"):
            report.covered_at(0)


class TestComparingStrategies:
    def test_mean_reciprocal_rank_counts_a_miss_as_zero(self) -> None:
        report = EvalReport(
            strategy="s",
            k=3,
            results=(
                result("a.md", "a.md", qid="q1"),
                result("b.md", "x.md", "b.md", qid="q2"),
                result("c.md", "x.md", qid="q3"),
            ),
        )
        assert report.mean_reciprocal_rank == pytest.approx((1 + 0.5 + 0) / 3)

    def test_strategies_are_ordered_by_hits_at_k_then_at_one_then_mrr(self) -> None:
        def report(strategy: str, first: tuple[str, ...], second: tuple[str, ...]) -> EvalReport:
            results = (result("x.md", *first, qid="q1"), result("y.md", *second, qid="q2"))
            return EvalReport(strategy=strategy, k=3, results=results)

        fewest_hits = report("fewest-hits", ("x.md",), ("z.md", "z.md", "z.md", "y.md"))
        best_top_one = report("best-top-one", ("x.md",), ("y.md",))
        lower_mrr = report("lower-mrr", ("x.md",), ("z.md", "z.md", "y.md"))
        higher_mrr = report("higher-mrr", ("x.md",), ("z.md", "y.md"))
        ordered = adoption_order([fewest_hits, lower_mrr, higher_mrr, best_top_one])
        assert [r.strategy for r in ordered] == [
            "best-top-one",
            "higher-mrr",
            "lower-mrr",
            "fewest-hits",
        ]

    def test_ties_keep_the_order_given(self) -> None:
        first = EvalReport(strategy="first", k=3, results=(result("x.md", "x.md", qid="q1"),))
        second = EvalReport(strategy="second", k=3, results=(result("x.md", "x.md", qid="q1"),))
        assert [r.strategy for r in adoption_order([first, second])] == ["first", "second"]

    def test_strategies_must_share_questions_and_cut_off(self) -> None:
        base = EvalReport(strategy="a", k=3, results=(result("x.md", "x.md", qid="q1"),))
        other_question = EvalReport(strategy="b", k=3, results=(result("x.md", "x.md", qid="q2"),))
        other_k = EvalReport(strategy="c", k=1, results=(result("x.md", "x.md", qid="q1"),))
        with pytest.raises(InvalidDataError, match="same questions"):
            adoption_order([base, other_question])
        with pytest.raises(InvalidDataError, match="same cut-off"):
            adoption_order([base, other_k])


class TestAnsweredQuestion:
    QUESTION = EvalQuestion(
        qid="q01", question="How long?", expected_doc="refund-policy.md", facts=("30 days",)
    )

    def test_needs_an_answer_or_an_error_but_not_both(self) -> None:
        with pytest.raises(InvalidDataError, match="an answer or an error"):
            AnsweredQuestion(question=self.QUESTION, answer=None, error=None, seconds=1.0)
        with pytest.raises(InvalidDataError, match="an answer or an error"):
            AnsweredQuestion(
                question=self.QUESTION, answer=answer("30 days [1]."), error="cut off", seconds=1.0
            )

    def test_facts_given_checks_every_chunk_in_the_prompt(self) -> None:
        given = AnsweredQuestion(
            question=self.QUESTION, answer=answer("30 days [1]."), error=None, seconds=2.5
        )
        other = EvalQuestion(
            qid="q02", question="Credit?", expected_doc="refund-policy.md", facts=("store credit",)
        )
        missing = AnsweredQuestion(question=other, answer=answer("Yes [1]."), error=None, seconds=1)
        assert (given.facts_given, missing.facts_given) == (True, False)

    def test_facts_given_is_unknown_without_an_answer_or_an_expected_document(self) -> None:
        failed = AnsweredQuestion(question=self.QUESTION, answer=None, error="cut off", seconds=0.5)
        unanswerable = EvalQuestion(qid="u01", question="Wi-Fi?", expected_doc=None)
        refused = AnsweredQuestion(
            question=unanswerable, answer=answer("I don't know."), error=None, seconds=1.0
        )
        assert (failed.facts_given, refused.facts_given) == (None, None)

    def test_seconds_must_be_zero_or_more(self) -> None:
        with pytest.raises(InvalidDataError, match="seconds"):
            AnsweredQuestion(question=self.QUESTION, answer=None, error="cut off", seconds=-1.0)


class TestGrades:
    def test_counts_every_level_even_when_none_has_it(self) -> None:
        grades = [
            Grade(
                qid="q1",
                answerable=True,
                supported=True,
                correct=Correctness.YES,
                all_facts_given=True,
            ),
            Grade(
                qid="q2",
                answerable=True,
                supported=False,
                correct=Correctness.YES,
                all_facts_given=True,
            ),
            Grade(
                qid="q3",
                answerable=True,
                supported=True,
                correct=Correctness.NO,
                all_facts_given=False,
            ),
        ]
        counts = count_grades(grades)
        assert (counts.answers, counts.supported) == (3, 2)
        assert counts.correct == {Correctness.YES: 2, Correctness.PARTIAL: 0, Correctness.NO: 1}

    def test_a_grade_needs_its_question_id(self) -> None:
        with pytest.raises(InvalidDataError, match="question id"):
            Grade(
                qid=" ",
                answerable=True,
                supported=True,
                correct=Correctness.YES,
                all_facts_given=None,
            )

"""Writing results: retrieval rows, the strategy comparison, and answers for grading."""

import csv
import re
from pathlib import Path

import pytest

from rag.adapters.results_csv import (
    new_answers_path,
    read_grades,
    write_answers,
    write_comparison,
    write_retrieval_results,
)
from rag.domain.errors import GradesError, ResultsWriteError
from rag.domain.models import (
    Answer,
    AnsweredQuestion,
    Correctness,
    EvalQuestion,
    EvalReport,
    Grade,
    QuestionResult,
    RetrievedChunk,
)
from tests.builders import chunk


def report() -> EvalReport:
    retrieved = (
        RetrievedChunk(chunk=chunk(doc="faq.md", section="Refunds"), score=0.9, rank=1),
        RetrievedChunk(chunk=chunk(doc="refunds.pdf", page=2), score=0.81234, rank=2),
    )
    question = EvalQuestion(
        qid="q1", question="How long, exactly?", expected_doc="refunds.pdf", facts=("30 days",)
    )
    result = QuestionResult(question=question, retrieved=retrieved)
    return EvalReport(strategy="fixed-300-50", k=1, results=(result,))


def test_writes_one_row_per_retrieved_chunk(tmp_path: Path) -> None:
    path = write_retrieval_results(report(), tmp_path / "results")
    assert path == tmp_path / "results" / "retrieval-fixed-300-50.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    question = {
        "id": "q1",
        "question": "How long, exactly?",
        "expected_doc": "refunds.pdf",
        "hit_at_1": "no",
        "expected_rank": "2",
        "covered_at_1": "no",
        "covered_rank": "2",
    }
    assert rows == [
        {
            **question,
            "rank": "1",
            "retrieved_doc": "faq.md",
            "retrieved_source": "faq.md (Refunds)",
            "score": "0.9000",
        },
        {
            **question,
            "rank": "2",
            "retrieved_doc": "refunds.pdf",
            "retrieved_source": "refunds.pdf, p. 2",
            "score": "0.8123",
        },
    ]


def test_a_second_run_replaces_the_file_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    write_retrieval_results(report(), tmp_path)
    path = write_retrieval_results(report(), tmp_path)
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]


def test_a_folder_that_cannot_be_created_is_an_error(tmp_path: Path) -> None:
    blocker = tmp_path / "results"
    blocker.write_text("not a folder", encoding="utf-8")
    with pytest.raises(ResultsWriteError, match="cannot write the results"):
        write_retrieval_results(report(), blocker)


def test_the_comparison_follows_the_rule_and_marks_the_strategy_in_use(tmp_path: Path) -> None:
    def report_for(strategy: str, retrieved_doc: str) -> EvalReport:
        retrieved = (RetrievedChunk(chunk=chunk(doc=retrieved_doc), score=0.9, rank=1),)
        question = EvalQuestion(
            qid="q1", question="How long?", expected_doc="refunds.md", facts=("30 days",)
        )
        result = QuestionResult(question=question, retrieved=retrieved)
        return EvalReport(strategy=strategy, k=1, results=(result,))

    reports = [report_for("fixed-300-50", "faq.md"), report_for("sections-4000", "refunds.md")]
    counts = {"fixed-300-50": 241, "sections-4000": 168}
    path = write_comparison(
        reports, counts, tmp_path, hit_cutoffs=[1, 3], coverage_cutoffs=[1], in_use="fixed-300-50"
    )
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert path.name == "comparison.csv"
    assert [
        (r["rule_rank"], r["strategy"], r["in_use"], r["chunks"], r["hit_at_1"], r["covered_at_1"])
        for r in rows
    ] == [
        ("1", "sections-4000", "no", "168", "1", "1"),
        ("2", "fixed-300-50", "yes", "241", "0", "0"),
    ]
    assert rows[0]["mean_reciprocal_rank"] == "1.0000"


def answered(text: str | None, *, error: str | None = None) -> AnsweredQuestion:
    """A test question with the given reply (two chunks in the prompt), or with an error."""
    question = EvalQuestion(
        qid="q1",
        question="How long, exactly?",
        expected_doc="refunds.pdf",
        expected_answer="30 days",
        facts=("30 days",),
    )
    sources = (
        RetrievedChunk(
            chunk=chunk(doc="refunds.pdf", page=2, section="Refunds"), score=0.9, rank=1
        ),
        RetrievedChunk(chunk=chunk(doc="faq.md"), score=0.8, rank=2),
    )
    answer = None
    if text is not None:
        answer = Answer(
            question=question.question,
            text=text,
            sources=sources,
            prompt="Sources:\n\n[1] refunds.pdf, p. 2 (Refunds)\n...",
            model="llama3.2:3b",
            prompt_tokens=600,
            answer_tokens=9,
        )
    return AnsweredQuestion(question=question, answer=answer, error=error, seconds=2.345)


def write(items: list[AnsweredQuestion], folder: Path) -> list[dict[str, str]]:
    path = write_answers(items, folder, model="llama3.2:3b", strategy="sections-4000", top_k=5)
    assert path == folder / "answers.csv"
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_answers_keep_the_reply_unchanged_and_leave_the_grades_empty(tmp_path: Path) -> None:
    [row] = write([answered("  Within 30 days [1].\n")], tmp_path)
    assert row["answer"] == "  Within 30 days [1].\n"
    assert (row["supported"], row["correct"], row["error"]) == ("", "", "")
    assert (row["id"], row["expected_doc"], row["expected_answer"]) == (
        "q1",
        "refunds.pdf",
        "30 days",
    )
    assert row["prompt"].startswith("Sources:\n\n[1] refunds.pdf")
    assert row["all_facts_given"] == "yes"
    assert (row["prompt_tokens"], row["answer_tokens"], row["seconds"]) == ("600", "9", "2.3")
    assert (row["model"], row["strategy"], row["top_k"]) == ("llama3.2:3b", "sections-4000", "5")


@pytest.mark.parametrize(
    ("text", "cited"),
    [
        ("30 days [1].", "[1] refunds.pdf, p. 2 (Refunds)"),
        (
            "30 days [2][1], see [7].",
            "[2] faq.md\n[1] refunds.pdf, p. 2 (Refunds)\n[7] (no such source)",
        ),
        ("I don't know. [1]", "none (the model says it doesn't know)"),
        ("Thirty days.", "none"),
    ],
)
def test_the_cited_column_lists_what_the_answer_cites(
    tmp_path: Path, text: str, cited: str
) -> None:
    [row] = write([answered(text)], tmp_path)
    assert row["cited"] == cited


def test_a_question_without_an_answer_shows_why(tmp_path: Path) -> None:
    [row] = write([answered(None, error="the answer reached the limit of 512 tokens")], tmp_path)
    assert (row["answer"], row["cited"], row["all_facts_given"]) == ("", "", "")
    assert row["error"] == "the answer reached the limit of 512 tokens"


def test_answers_never_replace_a_file_that_may_hold_grades(tmp_path: Path) -> None:
    graded = tmp_path / "answers.csv"
    graded.write_text("id,supported\nq1,yes\n", encoding="utf-8")
    with pytest.raises(ResultsWriteError, match="may hold your grades") as caught:
        new_answers_path(tmp_path)
    assert caught.value.hint == "rename or move it to answer the test set again"
    with pytest.raises(ResultsWriteError, match="may hold your grades"):
        write([answered("30 days [1].")], tmp_path)
    assert graded.read_text(encoding="utf-8") == "id,supported\nq1,yes\n"
    assert [entry.name for entry in tmp_path.iterdir()] == ["answers.csv"]


def test_a_file_that_appears_while_answering_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check before answering passed, then a file appeared before the answers were written."""
    monkeypatch.setattr(
        "rag.adapters.results_csv.new_answers_path", lambda folder: folder / "answers.csv"
    )
    graded = tmp_path / "answers.csv"
    graded.write_text("graded\n", encoding="utf-8")
    with pytest.raises(ResultsWriteError, match="may hold your grades"):
        write([answered("30 days [1].")], tmp_path)
    assert graded.read_text(encoding="utf-8") == "graded\n"
    assert [entry.name for entry in tmp_path.iterdir()] == ["answers.csv"]


GRADED = (
    "id,question,expected_doc,answer,supported,correct,all_facts_given\n"
    "q1,How long?,refunds.pdf,30 days [1].,yes,yes,yes\n"
    'q2,How much?,prices.md,"$5, or $7 [2].",No,Partial,no\n'
    "u1,Wi-Fi?,,I don't know.,yes,yes,\n"
)


def graded(tmp_path: Path, text: str = GRADED, *, encoding: str = "utf-8") -> Path:
    (tmp_path / "answers.csv").write_text(text, encoding=encoding)
    return tmp_path


def test_grades_are_read_in_file_order_ignoring_case(tmp_path: Path) -> None:
    first, second, third = read_grades(graded(tmp_path, encoding="utf-8-sig"))
    assert first == Grade(
        qid="q1", answerable=True, supported=True, correct=Correctness.YES, all_facts_given=True
    )
    assert (second.supported, second.correct, second.all_facts_given) == (
        False,
        Correctness.PARTIAL,
        False,
    )
    assert (third.answerable, third.all_facts_given) == (False, None)


@pytest.mark.parametrize(
    ("old", "new", "problem"),
    [
        ("yes,yes,yes", ",yes,yes", "q1 is not graded yet (supported is empty)"),
        ("yes,yes,yes", "yes,maybe,yes", "q1 has correct 'maybe'; it must be yes or partial or no"),
        ("yes,yes,yes", "sure,yes,yes", "q1 has supported 'sure'; it must be yes or no"),
        ("yes,yes,yes", "yes,yes,perhaps", "q1 has all_facts_given 'perhaps'"),
        ("u1,", "q1,", "repeats the ids q1"),
        ("yes,yes,yes", "yes,yes,yes,extra", "line 2: the row has more fields than the header"),
        (",all_facts_given", "", "has no all_facts_given column"),
    ],
)
def test_grades_are_checked_strictly(tmp_path: Path, old: str, new: str, problem: str) -> None:
    with pytest.raises(GradesError, match=re.escape(problem)):
        read_grades(graded(tmp_path, GRADED.replace(old, new, 1)))


def test_grades_without_an_answers_file_say_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(GradesError, match="does not exist") as caught:
        read_grades(tmp_path)
    assert (
        caught.value.hint == "run `rag answer-all`, then fill in its supported and correct columns"
    )

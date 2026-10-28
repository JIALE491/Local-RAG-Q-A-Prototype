"""Reading the test set strictly (questions and their facts), with errors that name the line."""

import re
from pathlib import Path

import pytest

from rag.adapters.questions_csv import load_questions, load_test_set
from rag.domain.errors import QuestionSetError

REPO_QUESTIONS = Path(__file__).resolve().parents[2] / "data" / "eval" / "questions.csv"
REPO_FACTS = REPO_QUESTIONS.with_name("facts.csv")
HEADER = "id,question,expected_doc,expected_answer\n"
QUESTIONS = HEADER + "q1,How long?,refunds.pdf,\nq2,How much?,prices.md,\nu1,Wi-Fi?,,\n"


def write(tmp_path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    path = tmp_path / "questions.csv"
    path.write_text(text, encoding=encoding)
    return path


def test_the_repository_test_set_loads() -> None:
    questions = load_questions(REPO_QUESTIONS)
    assert sum(question.answerable for question in questions) == 50
    assert sum(not question.answerable for question in questions) == 5


def test_rows_become_questions_in_file_order(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        HEADER + 'q1,"How long, exactly?",refunds.pdf,45 days\nu1,What is the Wi-Fi password?,,\n',
    )
    first, second = load_questions(path)
    assert (first.qid, first.question, first.expected_doc, first.expected_answer) == (
        "q1",
        "How long, exactly?",
        "refunds.pdf",
        "45 days",
    )
    assert (second.qid, second.expected_doc, second.expected_answer) == ("u1", None, None)
    assert not second.answerable


def test_the_expected_answer_column_is_optional(tmp_path: Path) -> None:
    [only] = load_questions(write(tmp_path, "id,question,expected_doc\nq1,How long?,refunds.pdf\n"))
    assert only.expected_answer is None


def test_a_byte_order_mark_is_accepted(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + "q1,How long?,refunds.pdf,\n", encoding="utf-8-sig")
    assert [question.qid for question in load_questions(path)] == ["q1"]


def test_empty_rows_are_skipped(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + "q1,How long?,refunds.pdf,\n,,,\n\n")
    assert len(load_questions(path)) == 1


def test_a_missing_file_is_an_error_with_a_hint(tmp_path: Path) -> None:
    with pytest.raises(QuestionSetError, match="not found") as caught:
        load_questions(tmp_path / "missing.csv")
    assert caught.value.hint == "check [paths] questions_csv in config.toml"


def test_an_empty_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(QuestionSetError, match="is empty"):
        load_questions(write(tmp_path, ""))


@pytest.mark.parametrize(
    ("header", "problem"),
    [
        ("id,question,expected_answer\n", "missing expected_doc"),
        ("id,question,expected_doc,expected_answer,notes\n", "unknown 'notes'"),
        ("id,question,expected doc\n", "missing expected_doc; unknown 'expected doc'"),
    ],
)
def test_the_header_must_name_the_known_columns(tmp_path: Path, header: str, problem: str) -> None:
    with pytest.raises(QuestionSetError, match=problem) as caught:
        load_questions(write(tmp_path, header))
    assert caught.value.hint == "its first line must be: id,question,expected_doc,expected_answer"


def test_a_row_with_too_many_fields_names_its_line(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + "q1,How long?,refunds.pdf,45 days,extra\n")
    with pytest.raises(QuestionSetError, match="line 2: the row has more fields"):
        load_questions(path)


def test_a_row_without_a_question_names_its_line(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + "q1,How long?,refunds.pdf,\nq2,,refunds.pdf,\n")
    with pytest.raises(QuestionSetError, match="line 3: question must not be blank"):
        load_questions(path)


def test_repeated_ids_are_an_error(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + "q1,How long?,refunds.pdf,\nq1,How much?,refunds.pdf,\n")
    with pytest.raises(QuestionSetError, match="repeats the ids q1"):
        load_questions(path)


def test_a_set_without_answerable_questions_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(QuestionSetError, match="no answerable questions"):
        load_questions(write(tmp_path, HEADER + "u1,What is the Wi-Fi password?,,\n"))


def write_facts(tmp_path: Path, rows: str, *, header: str = "id,fact\n") -> Path:
    path = tmp_path / "facts.csv"
    path.write_text(header + rows, encoding="utf-8")
    return path


def test_the_repository_fact_list_covers_every_answerable_question() -> None:
    questions = load_test_set(REPO_QUESTIONS, REPO_FACTS)
    assert all(question.facts for question in questions if question.answerable)
    assert sum(len(question.facts) for question in questions) == 78


def test_facts_are_attached_to_their_questions_in_file_order(tmp_path: Path) -> None:
    facts = write_facts(
        tmp_path, 'q1,within 30 days\nq2,$5\nq1,"store credit, without a receipt"\n,\n'
    )
    first, second, unanswerable = load_test_set(write(tmp_path, QUESTIONS), facts)
    assert first.facts == ("within 30 days", "store credit, without a receipt")
    assert second.facts == ("$5",)
    assert unanswerable.facts == ()


@pytest.mark.parametrize(
    ("rows", "problem"),
    [
        ("q1,30 days\nq2,$5\nq9,$7\n", "facts for ids that are not in questions.csv: q9"),
        ("q1,30 days\nq2,$5\nu1,hunter2\n", "facts for questions that no document answers: u1"),
        ("q1,30 days\nq2,$5\nq1,30 days\n", "line 4: q1 already has this fact"),
        ("q1,30 days\nq2,\n", "line 3: id and fact must both be filled in"),
        ("q1,30 days\n", "has no facts for q2"),
    ],
)
def test_the_fact_list_must_match_the_questions(tmp_path: Path, rows: str, problem: str) -> None:
    with pytest.raises(QuestionSetError, match=re.escape(problem)):
        load_test_set(write(tmp_path, QUESTIONS), write_facts(tmp_path, rows))


def test_a_missing_fact_list_is_an_error_with_a_hint(tmp_path: Path) -> None:
    with pytest.raises(QuestionSetError, match="the fact list was not found") as caught:
        load_test_set(write(tmp_path, QUESTIONS), tmp_path / "facts.csv")
    assert caught.value.hint == "check [paths] facts_csv in config.toml"


def test_the_fact_list_header_must_be_id_and_fact(tmp_path: Path) -> None:
    facts = write_facts(tmp_path, "q1,30 days\n", header="id,facts\n")
    with pytest.raises(QuestionSetError, match="missing fact; unknown 'facts'") as caught:
        load_test_set(write(tmp_path, QUESTIONS), facts)
    assert caught.value.hint == "its first line must be: id,fact"

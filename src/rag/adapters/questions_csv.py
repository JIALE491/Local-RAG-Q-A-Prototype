"""Reading the test set: data/eval/questions.csv and its fact list, data/eval/facts.csv.

questions.csv must name exactly the columns id, question and expected_doc, plus optionally
expected_answer. A row with an empty expected_doc is a question that no document answers.
facts.csv has the columns id and fact, one row per fact; every answerable question needs
at least one, and the others none. Fully empty rows are skipped. Both files are checked
strictly, so a typo in a header or a repeated id stops the evaluation instead of skewing
it. A byte-order mark, as some spreadsheet programs write, is allowed.
"""

import csv
import dataclasses
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from rag.domain.errors import InvalidDataError, QuestionSetError
from rag.domain.models import EvalQuestion

REQUIRED = ("id", "question", "expected_doc")
OPTIONAL = ("expected_answer",)
FACT_COLUMNS = ("id", "fact")


def load_questions(path: Path) -> tuple[EvalQuestion, ...]:
    """Every question in the file, in file order. Raises QuestionSetError naming the line."""
    questions = [
        _question(values, line=line, path=path)
        for line, values in _rows(path, REQUIRED, OPTIONAL, setting="questions_csv")
    ]
    repeated = sorted(qid for qid, count in Counter(q.qid for q in questions).items() if count > 1)
    if repeated:
        raise QuestionSetError(f"{path} repeats the ids {', '.join(repeated)}")
    if not any(question.answerable for question in questions):
        raise QuestionSetError(
            f"{path} has no answerable questions",
            hint="fill in expected_doc for the questions a document answers",
        )
    return tuple(questions)


def load_test_set(questions_path: Path, facts_path: Path) -> tuple[EvalQuestion, ...]:
    """The questions, in file order, with the facts of their expected answers attached.

    Raises QuestionSetError for a fact listed twice, a fact whose id is not an answerable
    question, or an answerable question without facts.
    """
    questions = load_questions(questions_path)
    facts = _load_facts(facts_path)
    by_id = {question.qid: question for question in questions}
    unknown = [qid for qid in facts if qid not in by_id]
    if unknown:
        raise QuestionSetError(
            f"{facts_path} has facts for ids that are not in {questions_path.name}: "
            f"{', '.join(unknown)}"
        )
    unanswerable = [qid for qid in facts if not by_id[qid].answerable]
    if unanswerable:
        raise QuestionSetError(
            f"{facts_path} has facts for questions that no document answers: "
            f"{', '.join(unanswerable)}"
        )
    without = [q.qid for q in questions if q.answerable and q.qid not in facts]
    if without:
        raise QuestionSetError(
            f"{facts_path} has no facts for {', '.join(without)}",
            hint="list the facts each expected answer relies on, worded as the document has them",
        )
    return tuple(dataclasses.replace(q, facts=facts.get(q.qid, ())) for q in questions)


def _load_facts(path: Path) -> dict[str, tuple[str, ...]]:
    facts: dict[str, list[str]] = {}
    for line, values in _rows(path, FACT_COLUMNS, (), setting="facts_csv"):
        qid, fact = values["id"], values["fact"]
        if not qid or not fact:
            raise QuestionSetError(f"{path}, line {line}: id and fact must both be filled in")
        if fact in facts.get(qid, []):
            raise QuestionSetError(f"{path}, line {line}: {qid} already has this fact")
        facts.setdefault(qid, []).append(fact)
    return {qid: tuple(items) for qid, items in facts.items()}


def _rows(
    path: Path, required: Sequence[str], optional: Sequence[str], *, setting: str
) -> Iterator[tuple[int, dict[str, str]]]:
    """(line number, stripped values) for every row that is not fully empty."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            _check_header(reader.fieldnames, path, required, optional)
            for row in reader:
                values = _values(row, line=reader.line_num, path=path)
                if any(values.values()):
                    yield reader.line_num, values
    except FileNotFoundError as exc:
        what = "the test set" if setting == "questions_csv" else "the fact list"
        raise QuestionSetError(
            f"{what} was not found: {path}", hint=f"check [paths] {setting} in config.toml"
        ) from exc
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise QuestionSetError(f"cannot read {path}: {exc}") from exc


def _check_header(
    fieldnames: Sequence[str] | None,
    path: Path,
    required: Sequence[str],
    optional: Sequence[str],
) -> None:
    expected = ",".join((*required, *optional))
    if not fieldnames:
        raise QuestionSetError(f"{path} is empty", hint=f"its first line must be: {expected}")
    missing = [column for column in required if column not in fieldnames]
    unknown = [column for column in fieldnames if column not in (*required, *optional)]
    if missing or unknown:
        problems = [f"missing {', '.join(missing)}"] if missing else []
        problems += [f"unknown {', '.join(repr(column) for column in unknown)}"] if unknown else []
        raise QuestionSetError(
            f"{path} has the wrong columns ({'; '.join(problems)})",
            hint=f"its first line must be: {expected}",
        )


def _values(row: Mapping[str | None, object], *, line: int, path: Path) -> dict[str, str]:
    if None in row:
        raise QuestionSetError(f"{path}, line {line}: the row has more fields than the header")
    return {str(key): value.strip() if isinstance(value, str) else "" for key, value in row.items()}


def _question(values: Mapping[str, str], *, line: int, path: Path) -> EvalQuestion:
    try:
        return EvalQuestion(
            qid=values["id"],
            question=values["question"],
            expected_doc=values["expected_doc"] or None,
            expected_answer=values.get("expected_answer") or None,
        )
    except InvalidDataError as exc:
        raise QuestionSetError(f"{path}, line {line}: {exc.message}") from exc

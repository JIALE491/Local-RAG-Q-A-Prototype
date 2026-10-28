"""Evaluation results as CSV files in the results folder.

- retrieval-<strategy>.csv: one row per retrieved chunk of every question, for one strategy.
- comparison.csv: one row per strategy, in the order of the adoption rule, marking the
  strategy config.toml uses, which a recorded decision may have chosen over the rule.
- answers.csv: one row per test question with the model's answer, and empty `supported`
  and `correct` columns for grading. Once graded, its grades are read back for the results.

The two retrieval files are replaced by the next run. answers.csv never is, because it may
hold grades: a run refuses to start while it exists. Every file is written to a temporary
file first and then moved into place, so an interrupted run never leaves a half-written
file behind.
"""

import csv
import secrets
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path

from rag.domain.errors import GradesError, ResultsWriteError
from rag.domain.models import (
    Answer,
    AnsweredQuestion,
    Correctness,
    EvalReport,
    Grade,
    adoption_order,
)

ANSWERS_FILE = "answers.csv"
GRADE_COLUMNS = ("id", "expected_doc", "supported", "correct", "all_facts_given")
"""The columns of answers.csv that reading the grades needs; the others are ignored."""


def write_retrieval_results(report: EvalReport, folder: Path) -> Path:
    """Write the report's per-question results into `folder` and return the file's path."""
    header = [
        "id",
        "question",
        "expected_doc",
        f"hit_at_{report.k}",
        "expected_rank",
        f"covered_at_{report.k}",
        "covered_rank",
        "rank",
        "retrieved_doc",
        "retrieved_source",
        "score",
    ]
    rows = (
        [
            result.question.qid,
            result.question.question,
            result.question.expected_doc,
            "yes" if result.is_hit(report.k) else "no",
            result.hit_rank or "",
            "yes" if result.is_covered(report.k) else "no",
            result.coverage_rank or "",
            retrieved.rank,
            retrieved.chunk.doc.name,
            retrieved.chunk.source.label(),
            f"{retrieved.score:.4f}",
        ]
        for result in report.results
        for retrieved in result.retrieved
    )
    return _write(folder / f"retrieval-{report.strategy}.csv", header, rows)


def write_comparison(
    reports: Sequence[EvalReport],
    chunk_counts: dict[str, int],
    folder: Path,
    *,
    hit_cutoffs: Sequence[int],
    coverage_cutoffs: Sequence[int],
    in_use: str,
) -> Path:
    """Write one row per strategy, in the order of the adoption rule, and return the path.

    `in_use` names the strategy that config.toml uses.
    """
    header = ["rule_rank", "strategy", "in_use", "chunks", "questions"]
    header += [f"hit_at_{n}" for n in hit_cutoffs] + ["mean_reciprocal_rank"]
    header += [f"covered_at_{n}" for n in coverage_cutoffs]
    rows = (
        [
            position,
            report.strategy,
            "yes" if report.strategy == in_use else "no",
            chunk_counts[report.strategy],
            len(report.results),
            *(report.hits_at(n) for n in hit_cutoffs),
            f"{report.mean_reciprocal_rank:.4f}",
            *(report.covered_at(n) for n in coverage_cutoffs),
        ]
        for position, report in enumerate(adoption_order(reports), start=1)
    )
    return _write(folder / "comparison.csv", header, rows)


def new_answers_path(folder: Path) -> Path:
    """Where answers.csv will go. Raises ResultsWriteError if it is already there."""
    path = folder / ANSWERS_FILE
    if path.exists():
        raise _answers_exist(path)
    return path


def write_answers(
    answered: Sequence[AnsweredQuestion],
    folder: Path,
    *,
    model: str,
    strategy: str,
    top_k: int,
) -> Path:
    """Write one row per question, with empty `supported` and `correct` columns for grading,
    and return the path. The answer is the model's reply exactly as it came back.
    Never replaces an existing answers.csv."""
    header = [
        "id",
        "question",
        "expected_doc",
        "expected_answer",
        "answer",
        "supported",
        "correct",
        "cited",
        "error",
        "prompt",
        "all_facts_given",
        "prompt_tokens",
        "answer_tokens",
        "seconds",
        "model",
        "strategy",
        "top_k",
    ]
    rows = (
        [
            item.question.qid,
            item.question.question,
            item.question.expected_doc or "",
            item.question.expected_answer or "",
            item.answer.text if item.answer else "",
            "",
            "",
            _cited(item.answer) if item.answer else "",
            item.error or "",
            item.answer.prompt if item.answer else "",
            _yes_no(item.facts_given),
            item.answer.prompt_tokens if item.answer else "",
            item.answer.answer_tokens if item.answer else "",
            f"{item.seconds:.1f}",
            model,
            strategy,
            top_k,
        ]
        for item in answered
    )
    return _write_new(new_answers_path(folder), header, rows)


def _cited(answer: Answer) -> str:
    """The cited sources with their labels, one per line."""
    if answer.says_dont_know:
        return "none (the model says it doesn't know)"
    cited = [f"[{number}] {answer.source(number).chunk.source.label()}" for number in answer.cited]
    made_up = [f"[{number}] (no such source)" for number in answer.invalid_citations]
    return "\n".join(cited + made_up) or "none"


def _yes_no(value: bool | None) -> str:
    return "" if value is None else ("yes" if value else "no")


def _answers_exist(path: Path) -> ResultsWriteError:
    return ResultsWriteError(
        f"{path} already exists and may hold your grades",
        hint="rename or move it to answer the test set again",
    )


def _write(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> Path:
    """Write the file, replacing an earlier one."""
    temporary = _temporary_file(path, header, rows)
    try:
        temporary.replace(path)
    except OSError as exc:
        _remove(temporary)
        raise ResultsWriteError(f"cannot write the results to {path.parent}: {exc}") from exc
    return path


def _write_new(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> Path:
    """Write the file only if there is none, because an existing one may hold grades."""
    temporary = _temporary_file(path, header, rows)
    try:
        path.hardlink_to(temporary)  # unlike a rename, fails if the file appeared meanwhile
    except FileExistsError as exc:
        raise _answers_exist(path) from exc
    except OSError as exc:
        raise ResultsWriteError(f"cannot write the results to {path.parent}: {exc}") from exc
    finally:
        _remove(temporary)  # once linked, this second name for the file is spare
    return path


def _temporary_file(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> Path:
    """Write the rows to a new temporary file in the folder of `path`, and return its path."""
    folder = path.parent
    temporary = folder / f".{path.name}.{secrets.token_hex(4)}.tmp"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
    except OSError as exc:
        _remove(temporary)
        raise ResultsWriteError(f"cannot write the results to {folder}: {exc}") from exc
    return temporary


def _remove(path: Path) -> None:
    with suppress(OSError):  # it may not exist, or its folder may not
        path.unlink()


def read_grades(folder: Path) -> tuple[Grade, ...]:
    """The grades in answers.csv, in file order. Raises GradesError naming the first problem.

    Grades are case-insensitive: `supported` must be yes or no, and `correct` yes, partial
    or no. A byte-order mark, as some spreadsheet programs write, is allowed.
    """
    path = folder / ANSWERS_FILE
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = [
                column for column in GRADE_COLUMNS if column not in (reader.fieldnames or ())
            ]
            if missing:
                raise GradesError(
                    f"{path} has no {', '.join(missing)} column",
                    hint="grade the file that `rag answer-all` wrote, keeping its columns",
                )
            grades = [_grade(row, line=reader.line_num, path=path) for row in reader]
    except FileNotFoundError as exc:
        raise GradesError(
            f"there are no answers to read grades from: {path} does not exist",
            hint="run `rag answer-all`, then fill in its supported and correct columns",
        ) from exc
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise GradesError(f"cannot read {path}: {exc}") from exc
    repeated = sorted(qid for qid, n in Counter(grade.qid for grade in grades).items() if n > 1)
    if repeated:
        raise GradesError(f"{path} repeats the ids {', '.join(repeated)}")
    return tuple(grades)


def _grade(row: Mapping[str | None, str | None], *, line: int, path: Path) -> Grade:
    if None in row:
        raise GradesError(f"{path}, line {line}: the row has more fields than the header")
    qid = (row["id"] or "").strip()
    if not qid:
        raise GradesError(f"{path}, line {line}: the id is empty")
    supported = _graded(row, "supported", ("yes", "no"), qid=qid, path=path)
    correct = _graded(row, "correct", tuple(Correctness), qid=qid, path=path)
    facts = (row["all_facts_given"] or "").strip()
    if facts not in ("yes", "no", ""):
        raise GradesError(
            f"{path}: {qid} has all_facts_given {facts!r}; it must be yes, no or empty"
        )
    return Grade(
        qid=qid,
        answerable=bool((row["expected_doc"] or "").strip()),
        supported=supported == "yes",
        correct=Correctness(correct),
        all_facts_given={"yes": True, "no": False}.get(facts),
    )


def _graded(
    row: Mapping[str | None, str | None],
    column: str,
    options: Sequence[str],
    *,
    qid: str,
    path: Path,
) -> str:
    value = (row[column] or "").strip().lower()
    if not value:
        raise GradesError(
            f"{path}: {qid} is not graded yet ({column} is empty)",
            hint="fill in supported (yes/no) and correct (yes/partial/no) for every answer",
        )
    if value not in options:
        raise GradesError(
            f"{path}: {qid} has {column} {value!r}; it must be {' or '.join(options)}"
        )
    return value

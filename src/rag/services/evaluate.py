"""Retrieval evaluation: does a chunk from the right document come back for each question,
and do the top chunks contain every fact the answer needs?"""

from collections.abc import Collection, Sequence

from rag.domain.errors import InvalidDataError, QuestionSetError
from rag.domain.models import EvalQuestion, EvalReport, QuestionResult
from rag.services.retrieve import Retriever


def evaluate_retrieval(
    *,
    questions: Sequence[EvalQuestion],
    retriever: Retriever,
    indexed_documents: Collection[str],
    strategy: str,
    k: int,
    depth: int,
) -> EvalReport:
    """Search every answerable question once, all in one batch, and score the results.

    `k` is the cut-off for a hit; `depth` chunks are kept per question (at least k), so hit
    rates and answer coverage at other cut-offs can be reported too. Questions no document
    answers are left out. Before anything is searched, every answerable question must have
    facts, and every expected document must be in `indexed_documents`; otherwise
    QuestionSetError names the questions, because a misspelled file name would count as a
    retrieval miss.
    """
    if depth < k:
        raise InvalidDataError(f"depth ({depth}) must be at least k ({k})")
    answerable = [question for question in questions if question.answerable]
    if not answerable:
        raise QuestionSetError("there are no answerable questions to evaluate")
    without_facts = [question.qid for question in answerable if not question.facts]
    if without_facts:
        raise QuestionSetError(
            f"questions without facts for answer coverage: {', '.join(without_facts)}",
            hint="list the facts of each expected answer in the fact list ([paths] facts_csv)",
        )
    unknown = [
        f"{question.qid} ({question.expected_doc})"
        for question in answerable
        if question.expected_doc not in indexed_documents
    ]
    if unknown:
        raise QuestionSetError(
            f"expected documents that are not in the index: {', '.join(unknown)}",
            hint="check the file names in the test set, or re-run `rag ingest`",
        )
    retrieved = retriever.search([question.question for question in answerable], k=depth)
    results = tuple(
        QuestionResult(question=question, retrieved=tuple(chunks))
        for question, chunks in zip(answerable, retrieved, strict=True)
    )
    return EvalReport(strategy=strategy, k=k, results=results)

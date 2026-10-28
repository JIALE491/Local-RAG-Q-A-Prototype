"""Retrieval evaluation with the fakes: scoring, batching and refusing unknown documents."""

import pytest

from rag.domain.errors import InvalidDataError, QuestionSetError
from rag.domain.models import EvalQuestion, EvalReport, IndexManifest
from rag.services.evaluate import evaluate_retrieval
from rag.services.retrieve import Retriever
from tests.builders import chunk
from tests.fakes import FakeEmbedder, InMemoryIndex

STRATEGY = "fixed-300-50"
CHUNKS = [
    chunk("Refunds are issued within 30 days.", doc="refunds.md"),
    chunk("Employees get 25 vacation days per year.", doc="vacation.md"),
    chunk("Laptops are replaced every 4 years.", doc="laptops.md"),
]
INDEXED = {"refunds.md", "vacation.md", "laptops.md"}
VACATION = "How many vacation days do employees get?"


def retriever_and_embedder() -> tuple[Retriever, FakeEmbedder]:
    index = InMemoryIndex()
    index.begin_rebuild()
    index.add(CHUNKS, FakeEmbedder().embed_documents([c.text for c in CHUNKS]))
    index.finish_rebuild(IndexManifest(strategy=STRATEGY, embed_model="fake-embed"))
    embedder = FakeEmbedder()
    return Retriever(embedder=embedder, index=index, strategy=STRATEGY), embedder


def question(
    qid: str, text: str, expected_doc: str | None, facts: tuple[str, ...] = ("days",)
) -> EvalQuestion:
    """A test question. An answerable one needs facts, so it gets a placeholder by default."""
    return EvalQuestion(
        qid=qid, question=text, expected_doc=expected_doc, facts=facts if expected_doc else ()
    )


def evaluate(questions: list[EvalQuestion], *, k: int = 1, depth: int = 3) -> EvalReport:
    retriever, _ = retriever_and_embedder()
    return evaluate_retrieval(
        questions=questions,
        retriever=retriever,
        indexed_documents=INDEXED,
        strategy=STRATEGY,
        k=k,
        depth=depth,
    )


def test_hits_and_misses_are_scored_at_k() -> None:
    report = evaluate(
        [question("q1", VACATION, "vacation.md"), question("q2", VACATION, "laptops.md")]
    )
    assert (report.strategy, report.k, report.hits) == (STRATEGY, 1, 1)
    [miss] = report.misses
    assert miss.question.qid == "q2"
    assert miss.hit_rank == 3
    assert report.hits_at(3) == 2


def test_each_question_keeps_depth_chunks() -> None:
    [result] = evaluate([question("q1", VACATION, "vacation.md")], depth=2).results
    assert [retrieved.rank for retrieved in result.retrieved] == [1, 2]


def test_questions_without_an_answer_are_left_out() -> None:
    report = evaluate([question("q1", VACATION, "vacation.md"), question("u1", "Wi-Fi?", None)])
    assert [result.question.qid for result in report.results] == ["q1"]


def test_all_questions_are_searched_in_one_batch() -> None:
    retriever, embedder = retriever_and_embedder()
    evaluate_retrieval(
        questions=[
            question("q1", VACATION, "vacation.md"),
            question("q2", "Refunds?", "refunds.md"),
        ],
        retriever=retriever,
        indexed_documents=INDEXED,
        strategy=STRATEGY,
        k=1,
        depth=3,
    )
    assert embedder.calls == [("queries", 2)]


def test_an_expected_document_missing_from_the_index_is_refused_before_searching() -> None:
    retriever, embedder = retriever_and_embedder()
    questions = [
        question("q1", VACATION, "vacation.md"),
        question("q2", "Refunds?", "refund-policy.md"),
    ]
    with pytest.raises(QuestionSetError, match=r"q2 \(refund-policy\.md\)") as caught:
        evaluate_retrieval(
            questions=questions,
            retriever=retriever,
            indexed_documents=INDEXED,
            strategy=STRATEGY,
            k=1,
            depth=3,
        )
    assert caught.value.hint is not None
    assert embedder.calls == []


def test_answer_coverage_checks_the_facts_in_the_retrieved_chunks() -> None:
    report = evaluate(
        [
            question("q1", VACATION, "vacation.md", facts=("25 vacation days per year",)),
            question("q2", VACATION, "vacation.md", facts=("25 vacation days", "carry over")),
        ]
    )
    assert report.covered_at(1) == 1
    [uncovered] = report.uncovered
    assert uncovered.missing_facts(1) == ("carry over",)


def test_questions_without_facts_are_refused_before_searching() -> None:
    retriever, embedder = retriever_and_embedder()
    with pytest.raises(QuestionSetError, match="without facts for answer coverage: q2"):
        evaluate_retrieval(
            questions=[
                question("q1", VACATION, "vacation.md"),
                question("q2", "Refunds?", "refunds.md", facts=()),
            ],
            retriever=retriever,
            indexed_documents=INDEXED,
            strategy=STRATEGY,
            k=1,
            depth=3,
        )
    assert embedder.calls == []


def test_depth_must_be_at_least_k() -> None:
    with pytest.raises(InvalidDataError, match="depth"):
        evaluate([question("q1", VACATION, "vacation.md")], k=3, depth=2)


def test_a_set_without_answerable_questions_is_refused() -> None:
    with pytest.raises(QuestionSetError, match="no answerable questions"):
        evaluate([question("u1", "Wi-Fi?", None)])

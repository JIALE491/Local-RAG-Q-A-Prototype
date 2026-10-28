"""Answering with the fakes: what the chat model is given, and the answer that comes back."""

import pytest

from rag.domain.errors import (
    AnswerTooLongError,
    IndexNotBuiltError,
    InvalidDataError,
    ModelUnavailableError,
    PromptTooLongError,
)
from rag.domain.models import DONT_KNOW, Completion, EvalQuestion, IndexManifest, RetrievedChunk
from rag.services.answer import SYSTEM_PROMPT, Answerer, answer_questions, build_prompt
from rag.services.retrieve import Retriever
from tests.builders import chunk
from tests.fakes import FakeChatModel, FakeEmbedder, InMemoryIndex

STRATEGY = "sections-4000"
CHUNKS = [
    chunk("Refunds are issued within 30 days.", doc="refunds.pdf", page=2, section="Refund window"),
    chunk("Employees get 25 vacation days per year.", doc="handbook.md", section="Vacation"),
    chunk("Laptops are replaced every 4 years.", doc="it-faq.md"),
]
REFUNDS = "How long do refunds take?"


def answerer(chat_model: FakeChatModel, *, top_k: int = 2, built: bool = True) -> Answerer:
    index = InMemoryIndex()
    if built:
        index.begin_rebuild()
        index.add(CHUNKS, FakeEmbedder().embed_documents([c.text for c in CHUNKS]))
        index.finish_rebuild(IndexManifest(strategy=STRATEGY, embed_model="fake-embed"))
    retriever = Retriever(embedder=FakeEmbedder(), index=index, strategy=STRATEGY)
    return Answerer(retriever=retriever, chat_model=chat_model, top_k=top_k)


def test_the_prompt_numbers_each_source_under_its_label_and_ends_with_the_question() -> None:
    retrieved = [
        RetrievedChunk(chunk=CHUNKS[0], score=0.9, rank=1),
        RetrievedChunk(chunk=CHUNKS[1], score=0.5, rank=2),
    ]
    assert build_prompt(f"  {REFUNDS} ", retrieved) == (
        "Sources:\n\n"
        "[1] refunds.pdf, p. 2 (Refund window)\nRefunds are issued within 30 days.\n\n"
        "[2] handbook.md (Vacation)\nEmployees get 25 vacation days per year.\n\n"
        f"Question: {REFUNDS}"
    )


def test_the_instructions_ask_for_citations_and_for_i_dont_know() -> None:
    assert "square brackets, like [1] or [2][3]" in SYSTEM_PROMPT
    assert f"do not answer the question, reply with exactly: {DONT_KNOW}\n" in SYSTEM_PROMPT


def test_the_model_gets_the_top_k_chunks_and_the_answer_keeps_them() -> None:
    chat_model = FakeChatModel("Within 30 days [1].")
    answer = answerer(chat_model, top_k=2).answer(REFUNDS)
    [(system, user)] = chat_model.prompts
    assert system == SYSTEM_PROMPT
    assert user.startswith("Sources:\n\n[1] refunds.pdf, p. 2 (Refund window)\n")
    assert "[2] " in user
    assert "[3] " not in user
    assert user.endswith(f"Question: {REFUNDS}")
    assert answer.prompt == user
    assert [source.rank for source in answer.sources] == [1, 2]
    assert answer.source(1).chunk is CHUNKS[0]


def test_the_answer_keeps_the_reply_exactly_as_it_came_back() -> None:
    answer = answerer(FakeChatModel("  Within 30 days [1].\n")).answer(REFUNDS)
    assert (answer.question, answer.text, answer.model) == (
        REFUNDS,
        "  Within 30 days [1].\n",
        "fake-chat",
    )
    assert answer.answer_tokens == 4
    assert answer.prompt_tokens > len(SYSTEM_PROMPT.split())
    assert answer.cited == (1,)


class FailsOnLaptops(FakeChatModel):
    """Answers every question except the one about laptops, the way `error` says."""

    def __init__(self, error: Exception) -> None:
        super().__init__("Within 30 days [1].")
        self.error = error

    def complete(self, *, system: str, user: str) -> Completion:
        if "laptops" in user.rsplit("Question:", 1)[1].lower():
            raise self.error
        return super().complete(system=system, user=user)


QUESTIONS = [
    EvalQuestion(qid="q1", question=REFUNDS, expected_doc="refunds.pdf", facts=("30 days",)),
    EvalQuestion(qid="q2", question="When are laptops replaced?", expected_doc="it-faq.md"),
    EvalQuestion(qid="u1", question="What is the Wi-Fi password?", expected_doc=None),
]


def test_every_question_is_answered_in_turn_with_its_time() -> None:
    answered = list(answer_questions(QUESTIONS, answerer(FakeChatModel("Within 30 days [1]."))))
    assert [item.question.qid for item in answered] == ["q1", "q2", "u1"]
    assert all(item.answer is not None and item.error is None for item in answered)
    assert all(item.seconds >= 0 for item in answered)


def test_a_question_the_model_cannot_answer_is_recorded_and_the_run_goes_on() -> None:
    chat_model = FailsOnLaptops(AnswerTooLongError("the answer was cut off"))
    first, second, third = answer_questions(QUESTIONS, answerer(chat_model))
    assert (second.answer, second.error) == (None, "the answer was cut off")
    assert first.answer is not None
    assert third.answer is not None


def test_a_model_that_cannot_be_reached_stops_the_run() -> None:
    chat_model = FailsOnLaptops(ModelUnavailableError("cannot reach Ollama"))
    answered = answer_questions(QUESTIONS, answerer(chat_model))
    assert next(answered).answer is not None
    with pytest.raises(ModelUnavailableError):
        next(answered)


def test_top_k_must_be_at_least_one() -> None:
    with pytest.raises(InvalidDataError, match="top_k"):
        answerer(FakeChatModel(), top_k=0)


def test_a_blank_question_is_refused() -> None:
    with pytest.raises(InvalidDataError, match="blank"):
        answerer(FakeChatModel()).answer("   ")


def test_an_index_that_is_not_built_is_refused_before_the_model_is_asked() -> None:
    chat_model = FakeChatModel()
    with pytest.raises(IndexNotBuiltError):
        answerer(chat_model, built=False).answer(REFUNDS)
    assert chat_model.prompts == []


def test_a_prompt_too_long_for_the_model_is_an_error_not_a_shorter_prompt() -> None:
    chat_model = FakeChatModel(context_tokens=30)
    with pytest.raises(PromptTooLongError):
        answerer(chat_model, top_k=3).answer(REFUNDS)

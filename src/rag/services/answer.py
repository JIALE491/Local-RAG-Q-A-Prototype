"""Question answering: the closest chunks from the index, then the chat model's reply from them."""

import time
from collections.abc import Iterator, Sequence

from rag.domain.errors import GenerationError, InvalidDataError
from rag.domain.models import DONT_KNOW, Answer, AnsweredQuestion, EvalQuestion, RetrievedChunk
from rag.domain.ports import ChatModel
from rag.services.retrieve import Retriever

SYSTEM_PROMPT = f"""\
You answer questions using only the numbered sources you are given.
Rules:
1. Use only facts stated in the sources. Do not add anything from general knowledge.
2. After each sentence, cite the sources it uses by their numbers in square brackets, \
like [1] or [2][3].
3. If the sources do not answer the question, reply with exactly: {DONT_KNOW}
4. Answer in one to three sentences."""


def build_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """The user message: each chunk numbered from 1 under its source label, then the question."""
    sources = "\n\n".join(
        f"[{number}] {retrieved.chunk.source.label()}\n{retrieved.chunk.text}"
        for number, retrieved in enumerate(chunks, start=1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question.strip()}"


class Answerer:
    """Answers questions from one strategy's index, giving the chat model the top k chunks."""

    def __init__(self, *, retriever: Retriever, chat_model: ChatModel, top_k: int) -> None:
        if top_k < 1:
            raise InvalidDataError(f"top_k must be 1 or more, got {top_k}")
        self._retriever = retriever
        self._chat_model = chat_model
        self._top_k = top_k

    def answer(self, question: str) -> Answer:
        """The model's answer, exactly as it replied, and what it was given.

        Raises whatever retrieval raises (an index that is not built or was built with other
        settings) and whatever the chat model raises (a prompt or answer that does not fit,
        a model that cannot be reached).
        """
        [retrieved] = self._retriever.search([question], k=self._top_k)
        prompt = build_prompt(question, retrieved)
        completion = self._chat_model.complete(system=SYSTEM_PROMPT, user=prompt)
        return Answer(
            question=question.strip(),
            text=completion.text,
            sources=tuple(retrieved),
            prompt=prompt,
            model=self._chat_model.model_name,
            prompt_tokens=completion.prompt_tokens,
            answer_tokens=completion.answer_tokens,
        )


def answer_questions(
    questions: Sequence[EvalQuestion], answerer: Answerer
) -> Iterator[AnsweredQuestion]:
    """Answer each question in turn, exactly as `rag ask` would, yielding each as it is done.

    A question the chat model cannot answer within its limits (GenerationError: a prompt or
    answer that does not fit, or an empty reply) is yielded with the error, and the run goes
    on. Any other error, such as Ollama becoming unreachable, stops the run.
    """
    for question in questions:
        started = time.perf_counter()
        try:
            answer = answerer.answer(question.question)
        except GenerationError as exc:
            yield AnsweredQuestion(
                question=question,
                answer=None,
                error=exc.message,
                seconds=time.perf_counter() - started,
            )
        else:
            yield AnsweredQuestion(
                question=question,
                answer=answer,
                error=None,
                seconds=time.perf_counter() - started,
            )

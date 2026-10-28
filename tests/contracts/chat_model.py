"""Contract for rag.domain.ports.ChatModel."""

import pytest

from rag.domain.errors import AnswerTooLongError, PromptTooLongError
from rag.domain.ports import ChatModel

CONTEXT_TOKENS = 4096
"""The context window the `chat_model` fixture must have."""
TERSE_ANSWER_TOKENS = 8
"""The answer limit of the `terse_chat_model` fixture: too few to count from 1 to 30."""


class ChatModelContract:
    """Subclass and provide two fixtures (yield them if they need closing afterwards):
    `chat_model`, with a context window of CONTEXT_TOKENS, and `terse_chat_model`, the same
    but with an answer limit of TERSE_ANSWER_TOKENS."""

    def test_has_a_model_name(self, chat_model: ChatModel) -> None:
        assert chat_model.model_name.strip()

    def test_replies_with_the_tokens_it_used(self, chat_model: ChatModel) -> None:
        completion = chat_model.complete(system="Reply with one word.", user="Is water wet?")
        assert completion.text.strip()
        assert completion.prompt_tokens > 0
        assert completion.answer_tokens > 0
        assert completion.prompt_tokens + completion.answer_tokens <= CONTEXT_TOKENS

    def test_a_prompt_longer_than_the_context_raises_instead_of_being_cut(
        self, chat_model: ChatModel
    ) -> None:
        with pytest.raises(PromptTooLongError):
            chat_model.complete(
                system="Reply with one word.", user="The shelf is full. " * CONTEXT_TOKENS
            )

    def test_an_answer_longer_than_the_limit_raises_instead_of_being_cut(
        self, terse_chat_model: ChatModel
    ) -> None:
        with pytest.raises(AnswerTooLongError):
            terse_chat_model.complete(
                system="Do exactly what you are asked.",
                user="Count from 1 to 30, with each number on its own line.",
            )

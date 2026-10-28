"""The fakes obey the same contracts as the real adapters will."""

import pytest

from rag.domain.errors import AnswerTooLongError, GenerationError
from rag.domain.models import Completion
from rag.domain.ports import ChatModel, DocumentSource, Embedder, VectorIndex
from tests.contracts.chat_model import CONTEXT_TOKENS, TERSE_ANSWER_TOKENS, ChatModelContract
from tests.contracts.embedder import EmbedderContract
from tests.contracts.vector_index import VectorIndexContract
from tests.fakes import FakeChatModel, FakeDocumentSource, FakeEmbedder, InMemoryIndex


class TestFakeEmbedder(EmbedderContract):
    @pytest.fixture
    def embedder(self) -> Embedder:
        return FakeEmbedder()

    def test_records_each_batch(self) -> None:
        fake = FakeEmbedder()
        fake.embed_documents(["a", "b"])
        fake.embed_queries(["c"])
        assert fake.calls == [("documents", 2), ("queries", 1)]

    def test_is_deterministic_across_instances(self) -> None:
        texts = ["Refunds are issued within 30 days."]
        assert FakeEmbedder().embed_documents(texts) == FakeEmbedder().embed_documents(texts)


class TestInMemoryIndex(VectorIndexContract):
    @pytest.fixture
    def index(self) -> VectorIndex:
        return InMemoryIndex()


class TestFakeChatModel(ChatModelContract):
    @pytest.fixture
    def chat_model(self) -> ChatModel:
        return FakeChatModel("Yes.", context_tokens=CONTEXT_TOKENS)

    @pytest.fixture
    def terse_chat_model(self) -> ChatModel:
        counting = "\n".join(str(number) for number in range(1, 31))
        return FakeChatModel(
            counting, context_tokens=CONTEXT_TOKENS, max_answer_tokens=TERSE_ANSWER_TOKENS
        )

    def test_returns_the_reply_and_records_the_prompt(self) -> None:
        fake = FakeChatModel("Refunds take 30 days [1].")
        completion = fake.complete(system="Answer from the sources.", user="How long?")
        assert completion == Completion(
            text="Refunds take 30 days [1].", prompt_tokens=6, answer_tokens=5
        )
        assert fake.prompts == [("Answer from the sources.", "How long?")]

    def test_reply_can_depend_on_the_prompt(self) -> None:
        chat = FakeChatModel(lambda system, user: f"echo: {user}")
        assert chat.complete(system="s", user="hello").text == "echo: hello"

    def test_empty_reply_raises(self) -> None:
        with pytest.raises(GenerationError):
            FakeChatModel("   ").complete(system="s", user="u")

    def test_a_reply_longer_than_the_room_the_prompt_leaves_raises(self) -> None:
        chat = FakeChatModel("one two three", context_tokens=10)
        assert chat.complete(system="s", user="six words of prompt in here").answer_tokens == 3
        with pytest.raises(AnswerTooLongError):
            chat.complete(system="s", user="seven words of prompt in here now")


def test_fake_document_source_returns_what_it_was_given() -> None:
    source: DocumentSource = FakeDocumentSource(documents=[])
    scan = source.scan()
    assert scan.documents == ()
    assert scan.skipped == ()

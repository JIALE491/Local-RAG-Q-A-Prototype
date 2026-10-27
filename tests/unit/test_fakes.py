"""The fakes obey the same contracts as the real adapters will."""

import pytest

from rag.domain.errors import GenerationError
from rag.domain.ports import ChatModel, DocumentSource, Embedder, VectorIndex
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


class TestFakeChatModel:
    def test_returns_the_reply_and_records_the_prompt(self) -> None:
        fake = FakeChatModel("Refunds take 30 days [1].")
        chat: ChatModel = fake  # mypy checks that the fake satisfies the port
        reply = chat.complete(system="Answer from the sources.", user="How long?")
        assert reply == "Refunds take 30 days [1]."
        assert fake.prompts == [("Answer from the sources.", "How long?")]

    def test_reply_can_depend_on_the_prompt(self) -> None:
        chat = FakeChatModel(lambda system, user: f"echo: {user}")
        assert chat.complete(system="s", user="hello") == "echo: hello"

    def test_empty_reply_raises(self) -> None:
        with pytest.raises(GenerationError):
            FakeChatModel("   ").complete(system="s", user="u")


def test_fake_document_source_returns_what_it_was_given() -> None:
    source: DocumentSource = FakeDocumentSource(documents=[])
    scan = source.scan()
    assert scan.documents == ()
    assert scan.skipped == ()

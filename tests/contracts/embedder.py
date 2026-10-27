"""Contract for rag.domain.ports.Embedder."""

import pytest

from rag.domain.errors import EmbeddingError
from rag.domain.ports import Embedder
from tests.fakes import cosine_similarity


class EmbedderContract:
    """Subclass and override the `embedder` fixture."""

    @pytest.fixture
    def embedder(self) -> Embedder:
        raise NotImplementedError

    def test_has_a_model_name(self, embedder: Embedder) -> None:
        assert embedder.model_name.strip()

    def test_returns_one_vector_per_text_in_order(self, embedder: Embedder) -> None:
        refunds, vacation, refunds_again = embedder.embed_documents(
            [
                "Refunds are issued within 30 days.",
                "Employees get 25 vacation days.",
                "Refunds are issued within 30 days.",
            ]
        )
        assert refunds == pytest.approx(refunds_again, abs=1e-4)
        assert refunds != pytest.approx(vacation, abs=1e-4)

    def test_documents_and_queries_have_the_same_size(self, embedder: Embedder) -> None:
        [document] = embedder.embed_documents(["Refunds are issued within 30 days."])
        [query] = embedder.embed_queries(["How long do refunds take?"])
        assert len(document) == len(query) > 0

    def test_empty_input_gives_empty_output(self, embedder: Embedder) -> None:
        assert embedder.embed_documents([]) == []
        assert embedder.embed_queries([]) == []

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    def test_blank_document_raises(self, embedder: Embedder, blank: str) -> None:
        with pytest.raises(EmbeddingError, match="blank"):
            embedder.embed_documents(["Valid text.", blank])

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_query_raises(self, embedder: Embedder, blank: str) -> None:
        with pytest.raises(EmbeddingError, match="blank"):
            embedder.embed_queries([blank])

    def test_related_texts_are_closer(self, embedder: Embedder) -> None:
        [query] = embedder.embed_queries(["How many vacation days do employees get?"])
        vacation, refunds = embedder.embed_documents(
            ["Employees get 25 vacation days per year.", "Refunds are issued within 30 days."]
        )
        assert cosine_similarity(query, vacation) > cosine_similarity(query, refunds)

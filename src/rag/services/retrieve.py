"""Retrieval: the chunks most similar to a question, from an index built for this setup."""

from collections.abc import Sequence

from rag.domain.errors import IndexNotBuiltError, InvalidDataError
from rag.domain.models import RetrievedChunk
from rag.domain.ports import Embedder, VectorIndex


class Retriever:
    """Searches one strategy's index with the configured embedding model."""

    def __init__(self, *, embedder: Embedder, index: VectorIndex, strategy: str) -> None:
        self._embedder = embedder
        self._index = index
        self._strategy = strategy

    def search(self, questions: Sequence[str], *, k: int) -> list[list[RetrievedChunk]]:
        """The top k chunks for each question, best first. Questions are embedded together.

        Checks before embedding anything that the index is finished (IndexNotBuiltError) and
        was built with this strategy and embedding model (IndexMismatchError).
        """
        if k < 1:
            raise InvalidDataError(f"k must be 1 or more, got {k}")
        if any(not question.strip() for question in questions):
            raise InvalidDataError("questions must not be blank")
        manifest = self._index.manifest()
        if manifest is None:
            raise IndexNotBuiltError(
                f"there is no finished index for {self._strategy!r}: it was never built, "
                "or its last rebuild did not finish",
                hint="run `rag ingest`",
            )
        manifest.ensure_compatible(strategy=self._strategy, embed_model=self._embedder.model_name)
        return self._index.query(self._embedder.embed_queries(questions), k)

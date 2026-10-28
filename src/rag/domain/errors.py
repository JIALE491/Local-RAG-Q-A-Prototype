"""Errors the pipeline raises on purpose.

Every one inherits from RagError, so the CLI can show it as a one-line message plus a
hint instead of a stack trace. Any other exception is a bug and is left to crash loudly.
"""


class RagError(Exception):
    """Base class for expected failures: bad input, missing setup, unreachable services.

    `hint` says what the user can do about it, e.g. "run `ollama pull nomic-embed-text`".
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class InvalidDataError(RagError, ValueError):
    """Data broke a domain rule, for example an empty chunk or page 0."""


class ConfigError(RagError):
    """config.toml is missing, malformed or has an invalid value."""


class DocumentLoadError(RagError):
    """A document could not be read or parsed."""

    def __init__(self, path: str, reason: str, *, hint: str | None = None) -> None:
        super().__init__(f"{path}: {reason}", hint=hint)
        self.path = path


class NoDocumentsError(RagError):
    """There is nothing to index: no supported documents, or no text in them."""


class ModelUnavailableError(RagError):
    """Ollama is not reachable or too old, or the model has not been pulled."""


class EmbeddingError(RagError):
    """The embedding model rejected the input or returned an unusable result."""


class GenerationError(RagError):
    """The chat model returned an error or an empty reply."""


class PromptTooLongError(GenerationError):
    """The prompt does not fit in the chat model's context window, so it was not answered."""


class AnswerTooLongError(GenerationError):
    """The answer ran out of room, either at the answer limit or at the end of the context
    window, so it would have been cut off."""


class VectorStoreError(RagError):
    """The vector index rejected an operation, e.g. a duplicate id or a wrong vector size."""


class IndexNotBuiltError(VectorStoreError):
    """The index has not been built yet."""


class IndexMismatchError(VectorStoreError):
    """The index was built with a different embedding model or chunking strategy."""


class QuestionSetError(RagError):
    """The test set (questions and their facts) is missing or malformed, or names a document
    that is not in the index."""


class ResultsWriteError(RagError):
    """An evaluation results file could not be written."""


class GradesError(RagError):
    """The graded answers file is missing, not fully graded, or does not match the test set."""

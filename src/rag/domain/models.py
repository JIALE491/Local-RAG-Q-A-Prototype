"""Data types that flow through the pipeline.

    Document --chunking--> Chunk --embed + index--> RetrievedChunk --chat model--> Answer

Plain frozen dataclasses with no I/O. Each one validates itself when it is created, so bad
data fails where it enters the pipeline instead of three stages later.
"""

import datetime as dt
import math
from dataclasses import dataclass
from enum import StrEnum

from rag.domain.errors import IndexMismatchError, InvalidDataError

type Embedding = list[float]
"""A vector from the embedding model (768 numbers for nomic-embed-text)."""


class DocumentKind(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"


class DateSource(StrEnum):
    """Where a document's created date came from, most trustworthy first."""

    PDF_METADATA = "pdf_metadata"
    """The /CreationDate field written into the PDF."""
    FRONT_MATTER = "front_matter"
    """A `created:` line in the markdown file's front matter."""
    FILE_SYSTEM = "file_system"
    """The file's creation time on disk, which changes when the file is copied."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentInfo:
    """Facts about a whole document, shared by all of its chunks."""

    doc_id: str
    """Path relative to the documents folder, with '/' separators. Unique."""
    name: str
    """File name. Shown in sources, and how the test set names the right document."""
    kind: DocumentKind
    created: dt.date | None
    created_from: DateSource | None

    def __post_init__(self) -> None:
        _require_text("doc_id", self.doc_id)
        _require_text("name", self.name)
        if (self.created is None) != (self.created_from is None):
            raise InvalidDataError(f"{self.name}: created and created_from must be set together")


@dataclass(frozen=True, slots=True, kw_only=True)
class Segment:
    """A run of text with a single location.

    For a PDF: one page, or the part of a page under one bookmark. For markdown: the text
    under one heading. Chunks never cross segment boundaries, so every chunk has exactly
    one page and one section.
    """

    text: str
    page: int | None
    """1-based page number: always set for PDFs, always None for markdown."""
    section: str | None
    """Bookmark title (PDF) or heading path such as 'Time off > Vacation' (markdown).
    None when it cannot be determined reliably; it is never guessed."""

    def __post_init__(self) -> None:
        _require_text("segment text", self.text)
        _check_page(self.page)
        _check_section(self.section)


@dataclass(frozen=True, slots=True, kw_only=True)
class Document:
    """A parsed document, ready for chunking."""

    info: DocumentInfo
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise InvalidDataError(f"{self.info.name}: document has no text")
        for segment in self.segments:
            _check_page_matches_kind(self.info, segment.page)


@dataclass(frozen=True, slots=True, kw_only=True)
class Source:
    """Where a piece of evidence came from, as shown next to an answer."""

    doc_name: str
    page: int | None
    section: str | None

    def label(self) -> str:
        """Document name plus page (PDF) or heading (markdown).

        'refund-policy.pdf, p. 3 (Refund window)', 'handbook.md (Time off > Vacation)',
        or just the document name when neither is known.
        """
        where = self.doc_name if self.page is None else f"{self.doc_name}, p. {self.page}"
        return f"{where} ({self.section})" if self.section else where


@dataclass(frozen=True, slots=True, kw_only=True)
class Chunk:
    """A piece of one segment, embedded and retrieved on its own."""

    doc: DocumentInfo
    ordinal: int
    """Position within the document in reading order, counting from 0."""
    text: str
    page: int | None
    section: str | None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise InvalidDataError(
                f"{self.doc.name}: ordinal must be 0 or more, got {self.ordinal}"
            )
        _require_text("chunk text", self.text)
        _check_page(self.page)
        _check_section(self.section)
        _check_page_matches_kind(self.doc, self.page)

    @property
    def chunk_id(self) -> str:
        """Readable and deterministic, e.g. 'policies/refund-policy.pdf#0003'."""
        return f"{self.doc.doc_id}#{self.ordinal:04d}"

    @property
    def source(self) -> Source:
        return Source(doc_name=self.doc.name, page=self.page, section=self.section)


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievedChunk:
    """A chunk returned by a search, with how well it matched."""

    chunk: Chunk
    score: float
    """Cosine similarity between query and chunk: 1.0 is identical, higher is better."""
    rank: int
    """1 for the best match."""

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise InvalidDataError(f"rank must be 1 or more, got {self.rank}")
        if not math.isfinite(self.score):
            raise InvalidDataError(f"score must be a finite number, got {self.score}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Answer:
    """The chat model's reply to one question, with the chunks it was given."""

    question: str
    text: str
    sources: tuple[RetrievedChunk, ...]
    """The chunks that were put in the prompt, best match first."""
    model: str

    def __post_init__(self) -> None:
        _require_text("question", self.question)
        _require_text("answer text", self.text)
        _require_text("model", self.model)


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexManifest:
    """What an index was built with.

    Stored with the index and checked before every search, because vectors from two
    different embedding models (or chunks from another strategy) cannot be compared.
    """

    strategy: str
    """Chunking strategy name, including its parameters."""
    embed_model: str

    def __post_init__(self) -> None:
        _require_text("strategy", self.strategy)
        _require_text("embed_model", self.embed_model)

    def ensure_compatible(self, *, strategy: str, embed_model: str) -> None:
        """Raise IndexMismatchError unless the index was built with these settings."""
        if (self.strategy, self.embed_model) != (strategy, embed_model):
            raise IndexMismatchError(
                f"index was built with strategy {self.strategy!r} and model "
                f"{self.embed_model!r}, but {strategy!r} and {embed_model!r} were requested",
                hint="re-run `rag ingest` to rebuild the index",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkippedFile:
    """A file in the documents folder that was not indexed, and why."""

    path: str
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceScan:
    """Everything found in the documents folder."""

    documents: tuple[Document, ...]
    skipped: tuple[SkippedFile, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalQuestion:
    """One row of the test set: a question and the document that answers it."""

    qid: str
    question: str
    expected_doc: str
    """File name of the document that contains the answer, e.g. 'refund-policy.pdf'."""

    def __post_init__(self) -> None:
        _require_text("question id", self.qid)
        _require_text("question", self.question)
        _require_text("expected_doc", self.expected_doc)


@dataclass(frozen=True, slots=True, kw_only=True)
class QuestionResult:
    """What retrieval returned for one test question."""

    question: EvalQuestion
    retrieved: tuple[RetrievedChunk, ...]
    """Best match first."""

    @property
    def hit_rank(self) -> int | None:
        """Rank of the first chunk from the expected document, or None if none came back."""
        expected = self.question.expected_doc
        return next((r.rank for r in self.retrieved if r.chunk.doc.name == expected), None)

    def is_hit(self, k: int) -> bool:
        """True if a chunk from the expected document is among the top k."""
        rank = self.hit_rank
        return rank is not None and rank <= k


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalReport:
    """Retrieval results for the whole test set with one chunking strategy."""

    strategy: str
    k: int
    """Cut-off for a hit: the charter's metric is top-3."""
    results: tuple[QuestionResult, ...]

    def __post_init__(self) -> None:
        if self.k < 1:
            raise InvalidDataError(f"k must be 1 or more, got {self.k}")
        if not self.results:
            raise InvalidDataError("an evaluation needs at least one question")

    @property
    def hits(self) -> int:
        return sum(result.is_hit(self.k) for result in self.results)

    @property
    def hit_rate(self) -> float:
        return self.hits / len(self.results)

    @property
    def misses(self) -> tuple[QuestionResult, ...]:
        return tuple(result for result in self.results if not result.is_hit(self.k))


def _require_text(what: str, value: str) -> None:
    if not value.strip():
        raise InvalidDataError(f"{what} must not be blank")


def _check_page(page: int | None) -> None:
    if page is not None and page < 1:
        raise InvalidDataError(f"page must be 1 or more, got {page}")


def _check_section(section: str | None) -> None:
    if section is not None and not section.strip():
        raise InvalidDataError("section must be None or non-blank text")


def _check_page_matches_kind(doc: DocumentInfo, page: int | None) -> None:
    if doc.kind is DocumentKind.PDF and page is None:
        raise InvalidDataError(f"{doc.name}: PDF text needs a page number")
    if doc.kind is DocumentKind.MARKDOWN and page is not None:
        raise InvalidDataError(f"{doc.name}: markdown has no pages, got page {page}")

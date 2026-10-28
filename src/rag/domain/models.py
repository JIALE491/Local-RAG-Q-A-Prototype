"""Data types that flow through the pipeline.

    Document --chunking--> Chunk --embed + index--> RetrievedChunk --chat model--> Answer

Plain frozen dataclasses with no I/O. Each one validates itself when it is created, so bad
data fails where it enters the pipeline instead of three stages later.
"""

import datetime as dt
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from rag.domain.errors import IndexMismatchError, InvalidDataError

type Embedding = list[float]
"""A vector from the embedding model (768 numbers for nomic-embed-text)."""

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
"""A citation in an answer: [2], or several sources at once as [1, 3]."""

DONT_KNOW = "I don't know."
"""The whole reply the chat model is told to give when its sources do not answer."""


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
        _check_optional_text("section", self.section)


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
        _check_optional_text("section", self.section)
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
class Completion:
    """A chat model's reply, with the tokens it took: prompt plus answer must fit the context."""

    text: str
    prompt_tokens: int
    """The prompt as the model read it, chat template included."""
    answer_tokens: int

    def __post_init__(self) -> None:
        _require_text("reply", self.text)
        _check_tokens(self.prompt_tokens, self.answer_tokens)


@dataclass(frozen=True, slots=True, kw_only=True)
class Answer:
    """The chat model's reply to one question, with the chunks it was given."""

    question: str
    text: str
    """The model's reply exactly as it came back."""
    sources: tuple[RetrievedChunk, ...]
    """The chunks that were put in the prompt, best match first. The prompt numbers them
    from 1, and the answer cites them by those numbers, such as [2]."""
    prompt: str
    """The user message the model was given: the numbered sources, then the question."""
    model: str
    prompt_tokens: int
    answer_tokens: int

    def __post_init__(self) -> None:
        _require_text("question", self.question)
        _require_text("answer text", self.text)
        _require_text("prompt", self.prompt)
        _require_text("model", self.model)
        _check_tokens(self.prompt_tokens, self.answer_tokens)

    @property
    def says_dont_know(self) -> bool:
        """True if the whole answer is "I don't know", ignoring citations, case, spacing,
        the kind of apostrophe and a final period."""
        text = _CITATION.sub(" ", self.text).replace("\u2019", "'")
        return _for_matching(text).rstrip(" .") == _for_matching(DONT_KNOW).rstrip(" .")

    @property
    def citations(self) -> tuple[int, ...]:
        """The source numbers the answer cites, in order of first mention, each once."""
        groups = _CITATION.findall(self.text)
        return tuple(dict.fromkeys(int(n) for group in groups for n in group.split(",")))

    @property
    def cited(self) -> tuple[int, ...]:
        """The cited numbers that match a source."""
        return tuple(n for n in self.citations if 1 <= n <= len(self.sources))

    @property
    def invalid_citations(self) -> tuple[int, ...]:
        """The cited numbers that match no source: the model made them up."""
        return tuple(n for n in self.citations if not 1 <= n <= len(self.sources))

    def source(self, number: int) -> RetrievedChunk:
        """The chunk the prompt numbered `number`, counting from 1."""
        if not 1 <= number <= len(self.sources):
            raise InvalidDataError(
                f"there is no source [{number}]; the prompt had {len(self.sources)}"
            )
        return self.sources[number - 1]


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
    """One row of the test set.

    An answerable question names the document that contains its answer, and only these
    count toward the retrieval hit rate. A question without one has no answer anywhere in
    the collection: it tests whether the chat model admits that it doesn't know.
    """

    qid: str
    question: str
    expected_doc: str | None
    """File name of the document with the answer, e.g. 'refund-policy.pdf'; None if none."""
    expected_answer: str | None = None
    """Optional reference answer, used only when grading answers."""
    facts: tuple[str, ...] = ()
    """What the expected answer relies on, each worded exactly as the expected document
    states it. Answer coverage checks the retrieved chunks for them."""

    def __post_init__(self) -> None:
        _require_text("question id", self.qid)
        _require_text("question", self.question)
        _check_optional_text("expected_doc", self.expected_doc)
        _check_optional_text("expected_answer", self.expected_answer)
        for fact in self.facts:
            _require_text(f"{self.qid}: a fact", fact)
        if len(set(self.facts)) < len(self.facts):
            raise InvalidDataError(f"{self.qid}: a fact is listed twice")
        if self.facts and not self.answerable:
            raise InvalidDataError(f"{self.qid}: a question that no document answers has no facts")

    @property
    def answerable(self) -> bool:
        return self.expected_doc is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class QuestionResult:
    """What retrieval returned for one answerable test question."""

    question: EvalQuestion
    retrieved: tuple[RetrievedChunk, ...]
    """Best match first."""

    def __post_init__(self) -> None:
        if not self.question.answerable:
            raise InvalidDataError(
                f"{self.question.qid}: unanswerable questions are not part of the retrieval "
                "evaluation"
            )
        if not self.question.facts:
            raise InvalidDataError(
                f"{self.question.qid}: answer coverage needs the facts of the expected answer"
            )

    @property
    def hit_rank(self) -> int | None:
        """Rank of the first chunk from the expected document, or None if none came back."""
        expected = self.question.expected_doc
        return next((r.rank for r in self.retrieved if r.chunk.doc.name == expected), None)

    def is_hit(self, k: int) -> bool:
        """True if a chunk from the expected document is among the top k."""
        rank = self.hit_rank
        return rank is not None and rank <= k

    def missing_facts(self, n: int) -> tuple[str, ...]:
        """The facts that no chunk from the expected document among the top n contains whole."""
        texts = [
            r.chunk.text
            for r in self.retrieved
            if r.rank <= n and r.chunk.doc.name == self.question.expected_doc
        ]
        return tuple(
            fact
            for fact in self.question.facts
            if not any(contains_fact(text, fact) for text in texts)
        )

    def is_covered(self, n: int) -> bool:
        """True if the top n chunks contain every fact, each one whole inside one chunk."""
        return not self.missing_facts(n)

    @property
    def coverage_rank(self) -> int | None:
        """The smallest n at which the top n chunks contain every fact; None if they never do."""
        return next((r.rank for r in self.retrieved if self.is_covered(r.rank)), None)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalReport:
    """Retrieval results for every answerable test question with one chunking strategy."""

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
        return self.hits_at(self.k)

    def hits_at(self, n: int) -> int:
        """Questions with a chunk from the expected document among the top n."""
        if n < 1:
            raise InvalidDataError(f"n must be 1 or more, got {n}")
        return sum(result.is_hit(n) for result in self.results)

    @property
    def hit_rate(self) -> float:
        return self.hits / len(self.results)

    @property
    def misses(self) -> tuple[QuestionResult, ...]:
        return tuple(result for result in self.results if not result.is_hit(self.k))

    @property
    def mean_reciprocal_rank(self) -> float:
        """The average of 1/rank of the first chunk from the expected document, counting 0
        for a question where none of the retrieved chunks is from it."""
        return sum(1 / r.hit_rank for r in self.results if r.hit_rank) / len(self.results)

    def covered_at(self, n: int) -> int:
        """Answer coverage: questions whose top n chunks contain every fact of the answer."""
        if n < 1:
            raise InvalidDataError(f"n must be 1 or more, got {n}")
        return sum(result.is_covered(n) for result in self.results)

    @property
    def uncovered(self) -> tuple[QuestionResult, ...]:
        """Questions whose top k chunks miss at least one fact."""
        return tuple(result for result in self.results if not result.is_covered(self.k))


@dataclass(frozen=True, slots=True, kw_only=True)
class AnsweredQuestion:
    """One test question put through the whole pipeline: its answer, or why there is none."""

    question: EvalQuestion
    answer: Answer | None
    error: str | None
    """Why the model gave no answer, such as a prompt too long for its context."""
    seconds: float

    def __post_init__(self) -> None:
        if (self.answer is None) == (self.error is None):
            raise InvalidDataError(f"{self.question.qid}: needs an answer or an error, not both")
        if self.error is not None:
            _require_text("error", self.error)
        if not (math.isfinite(self.seconds) and self.seconds >= 0):
            raise InvalidDataError(f"seconds must be 0 or more, got {self.seconds}")

    @property
    def facts_given(self) -> bool | None:
        """Whether the prompt held every fact of the expected answer, each whole in one chunk.

        None for a question that no document answers, and when there is no answer.
        """
        if self.answer is None or not self.question.answerable:
            return None
        result = QuestionResult(question=self.question, retrieved=self.answer.sources)
        return result.is_covered(len(self.answer.sources))


class Correctness(StrEnum):
    """How an answer compares with the expected answer."""

    YES = "yes"
    PARTIAL = "partial"
    NO = "no"


@dataclass(frozen=True, slots=True, kw_only=True)
class Grade:
    """The two grades one answer was given, by the rubric in docs/decisions.md."""

    qid: str
    answerable: bool
    supported: bool
    """Every statement in the answer, conclusions included, follows from its sources."""
    correct: Correctness
    """For a question that no document answers, yes means the model said it doesn't know."""
    all_facts_given: bool | None
    """Whether the prompt held every fact of the expected answer; None if not known."""

    def __post_init__(self) -> None:
        _require_text("question id", self.qid)


@dataclass(frozen=True, slots=True, kw_only=True)
class GradeCounts:
    """How many answers in a group got each grade."""

    answers: int
    supported: int
    correct: dict[Correctness, int]
    """Answers per level, with every level present."""


def count_grades(grades: Sequence[Grade]) -> GradeCounts:
    return GradeCounts(
        answers=len(grades),
        supported=sum(grade.supported for grade in grades),
        correct={level: sum(grade.correct is level for grade in grades) for level in Correctness},
    )


def adoption_order(reports: Sequence[EvalReport]) -> list[EvalReport]:
    """The strategies best first: by hit rate at k, then at 1, then mean reciprocal rank.

    Strategies that tie on all three keep the order they were given in. Raises
    InvalidDataError unless every report used the same cut-off and the same questions.
    """
    if len({report.k for report in reports}) > 1:
        raise InvalidDataError("strategies must be compared at the same cut-off")
    if len({tuple(r.question.qid for r in report.results) for report in reports}) > 1:
        raise InvalidDataError("strategies must be compared on the same questions")
    return sorted(
        reports,
        key=lambda report: (report.hits, report.hits_at(1), report.mean_reciprocal_rank),
        reverse=True,
    )


def contains_fact(text: str, fact: str) -> bool:
    """True if `fact` appears in `text` as one unbroken run, ignoring case and spacing.

    A fact that a line break, page break or page footer splits does not count.
    """
    return _for_matching(fact) in _for_matching(text)


def _for_matching(text: str) -> str:
    return " ".join(text.split()).lower()


def _require_text(what: str, value: str) -> None:
    if not value.strip():
        raise InvalidDataError(f"{what} must not be blank")


def _check_tokens(prompt_tokens: int, answer_tokens: int) -> None:
    if prompt_tokens < 1 or answer_tokens < 1:
        raise InvalidDataError(
            f"token counts must be 1 or more, got {prompt_tokens} and {answer_tokens}"
        )


def _check_page(page: int | None) -> None:
    if page is not None and page < 1:
        raise InvalidDataError(f"page must be 1 or more, got {page}")


def _check_optional_text(what: str, value: str | None) -> None:
    if value is not None and not value.strip():
        raise InvalidDataError(f"{what} must be None or non-blank text")


def _check_page_matches_kind(doc: DocumentInfo, page: int | None) -> None:
    if doc.kind is DocumentKind.PDF and page is None:
        raise InvalidDataError(f"{doc.name}: PDF text needs a page number")
    if doc.kind is DocumentKind.MARKDOWN and page is not None:
        raise InvalidDataError(f"{doc.name}: markdown has no pages, got page {page}")

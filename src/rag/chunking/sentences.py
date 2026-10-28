"""Strategy 2: whole sentences packed into chunks, plus the sentence splitting it needs.

Sentence splitting is also used by strategy 3 for the rare section that is too long to be a
single chunk.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from rag.domain.errors import InvalidDataError
from rag.domain.models import Chunk, Document

_ITEM_START = re.compile(r"(?:[-*] |\d+\. |\|)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_ABBREVIATIONS = frozenset(
    {
        "a.m.",
        "p.m.",
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "no.",
        "st.",
        "jr.",
        "sr.",
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "inc.",
        "ltd.",
        "co.",
        "u.s.",
    }
)


@dataclass(frozen=True, slots=True)
class SentenceChunker:
    """Packs whole sentences into chunks of at most `size` characters, never cutting one.

    It has the same size budget as the fixed windows of strategy 1, but needs no overlap:
    overlap exists to keep a fact that straddles a cut whole somewhere, and these cuts only
    fall between sentences. A sentence longer than `size` becomes a chunk of its own.
    """

    size: int

    def __post_init__(self) -> None:
        if self.size < 1:
            raise InvalidDataError(f"size must be 1 or more, got {self.size}")

    @property
    def name(self) -> str:
        return f"sentences-{self.size}"

    def split(self, document: Document) -> list[Chunk]:
        pieces = (
            (segment, text)
            for segment in document.segments
            for text in pack(split_sentences(segment.text), self.size)
        )
        return [
            Chunk(
                doc=document.info,
                ordinal=ordinal,
                text=text,
                page=segment.page,
                section=segment.section,
            )
            for ordinal, (segment, text) in enumerate(pieces)
        ]


def split_sentences(text: str) -> list[str]:
    """The sentences of a segment, in order, with whitespace collapsed.

    Blank lines, list items ("- ", "* ", "1. ") and table rows ("|") always start a new
    block, so items and rows are never glued together. Other line breaks count as spaces,
    because PDF text breaks lines in the middle of sentences. Within a block, a sentence
    ends at ".", "!" or "?" followed by a space, except after common abbreviations such as
    "a.m." or "e.g.". Text without end punctuation, such as a heading, a list item or a
    table row, is one sentence.
    """
    return [sentence for block in _blocks(text) for sentence in _sentences(block)]


def pack(sentences: Iterable[str], size: int) -> Iterator[str]:
    """Join consecutive sentences into pieces of at most `size` characters.

    A piece never ends inside a sentence; a sentence longer than `size` is a piece of its own.
    """
    current: list[str] = []
    length = 0
    for sentence in sentences:
        if current and length + 1 + len(sentence) > size:
            yield " ".join(current)
            current, length = [], 0
        length += len(sentence) + (1 if current else 0)
        current.append(sentence)
    if current:
        yield " ".join(current)


def _blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = " ".join(line.split())
        if not stripped or _ITEM_START.match(stripped):
            if current:
                blocks.append(" ".join(current))
            current = [stripped] if stripped else []
        else:
            current.append(stripped)
    if current:
        blocks.append(" ".join(current))
    return blocks


def _sentences(block: str) -> Iterator[str]:
    start = 0
    for boundary in _SENTENCE_END.finditer(block):
        last_word = block[start : boundary.start()].rsplit(" ", 1)[-1].lower()
        if last_word in _ABBREVIATIONS:
            continue
        yield block[start : boundary.start()]
        start = boundary.end()
    if block[start:]:
        yield block[start:]

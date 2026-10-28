"""Strategy 1: fixed-size windows of whole words, overlapping, inside each segment."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from rag.domain.errors import InvalidDataError
from rag.domain.models import Chunk, Document


@dataclass(frozen=True, slots=True)
class FixedSizeChunker:
    """Cuts every segment into windows of at most `size` characters, never splitting a word.

    Whitespace is collapsed first, so PDF line breaks do not count. Consecutive windows share
    up to `overlap` characters of whole words, so a short fact that straddles a cut still
    appears whole in one chunk. A single word longer than `size` becomes a chunk on its own.
    """

    size: int
    overlap: int

    def __post_init__(self) -> None:
        if self.size < 1:
            raise InvalidDataError(f"size must be 1 or more, got {self.size}")
        if not 0 <= self.overlap < self.size:
            raise InvalidDataError(
                f"overlap must be 0 or more and less than size ({self.size}), got {self.overlap}"
            )

    @property
    def name(self) -> str:
        return f"fixed-{self.size}-{self.overlap}"

    def split(self, document: Document) -> list[Chunk]:
        pieces = (
            (segment, text)
            for segment in document.segments
            for text in _windows(segment.text.split(), self.size, self.overlap)
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


def _windows(words: Sequence[str], size: int, overlap: int) -> Iterator[str]:
    start = 0
    while start < len(words):
        end, length = start + 1, len(words[start])
        while end < len(words) and length + 1 + len(words[end]) <= size:
            length += 1 + len(words[end])
            end += 1
        yield " ".join(words[start:end])
        if end == len(words):
            return
        back, shared = end, 0
        while back - 1 > start:
            candidate = len(words[back - 1]) + (shared + 1 if shared else 0)
            if candidate > overlap:
                break
            back, shared = back - 1, candidate
        start = back

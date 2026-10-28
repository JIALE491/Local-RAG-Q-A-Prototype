"""Strategy 3: one chunk per segment, the text under one heading or bookmark."""

from dataclasses import dataclass

from rag.chunking.sentences import pack, split_sentences
from rag.domain.errors import InvalidDataError
from rag.domain.models import Chunk, Document, Segment


@dataclass(frozen=True, slots=True)
class SectionChunker:
    """Makes each segment one chunk: a markdown section, or the part of a PDF page under one
    bookmark (a whole page when the PDF has no bookmarks).

    A segment longer than `max_size` characters is split into pieces of whole sentences. The
    limit is a safety net for the embedding model's input size, not a tuning knob.
    """

    max_size: int

    def __post_init__(self) -> None:
        if self.max_size < 1:
            raise InvalidDataError(f"max_size must be 1 or more, got {self.max_size}")

    @property
    def name(self) -> str:
        return f"sections-{self.max_size}"

    def split(self, document: Document) -> list[Chunk]:
        pieces = ((segment, text) for segment in document.segments for text in self._texts(segment))
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

    def _texts(self, segment: Segment) -> list[str]:
        whole = " ".join(segment.text.split())
        if len(whole) <= self.max_size:
            return [whole]
        return list(pack(split_sentences(segment.text), self.max_size))

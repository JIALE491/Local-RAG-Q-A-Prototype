"""PDF files: text per page from pypdf, sections from bookmarks, the created date from metadata.

Sections follow one rule and are never guessed from fonts or layout:

- A page on which no bookmark starts belongs to the last bookmark before it, or to no section
  if no bookmark comes before it.
- A page on which bookmarks start is split at each line whose whole text equals a bookmark
  title (ignoring spacing). The part above the first split continues the previous section.
- When a bookmark's title is not on the page as a line of its own, the part of the page in
  which its section begins cannot be told apart from the section before it, so that part
  gets no section.

Page headers and footers stay in the text.
"""

import datetime as dt
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError
from pypdf.generic import Destination

from rag.adapters.files import file_created
from rag.domain.errors import DocumentLoadError
from rag.domain.models import DateSource, Document, DocumentInfo, DocumentKind, Segment

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Bookmark:
    title: str
    """Looked for as a line of its own on the page where the section starts."""
    section: str
    """The section name chunks get: parent bookmarks and this one, joined by ' > '."""
    page: int
    """1-based page the bookmark points to."""


def load_pdf(path: Path, *, doc_id: str) -> Document:
    """Parse one PDF. Raises DocumentLoadError if it cannot be used."""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise DocumentLoadError(
                doc_id, "the PDF is encrypted", hint="remove the password, or leave the file out"
            )
        pages = [page.extract_text() for page in reader.pages]
    except (OSError, PyPdfError) as exc:
        raise DocumentLoadError(doc_id, f"cannot read the PDF: {exc}") from exc
    if not any(text.strip() for text in pages):
        raise DocumentLoadError(
            doc_id,
            "the PDF has no extractable text",
            hint="scanned PDFs need OCR, which this prototype does not do",
        )
    segments = tuple(_segments(pages, _bookmarks(reader, doc_id=doc_id)))
    created, created_from = _created(reader, path, doc_id=doc_id)
    info = DocumentInfo(
        doc_id=doc_id,
        name=path.name,
        kind=DocumentKind.PDF,
        created=created,
        created_from=created_from,
    )
    return Document(info=info, segments=segments)


def _bookmarks(reader: PdfReader, *, doc_id: str) -> list[_Bookmark]:
    """Every bookmark, sorted by page; [] if the outline cannot be used as a whole.

    One unusable bookmark makes the whole outline unusable: dropping it alone would label
    its section's text with the section before it.
    """
    try:
        outline = reader.outline
    except PyPdfError as exc:
        logger.warning("%s: ignoring bookmarks that cannot be read (%s)", doc_id, exc)
        return []
    found: list[_Bookmark] = []
    unusable: list[str] = []

    def walk(entries: Sequence[object], parents: tuple[str, ...]) -> None:
        current = parents
        for entry in entries:
            if isinstance(entry, list):
                walk(entry, current)
                continue
            if not isinstance(entry, Destination):
                unusable.append(f"an entry of type {type(entry).__name__}")
                continue
            title = " ".join((entry.title or "").split())
            page = reader.get_destination_page_number(entry)
            if not title or page is None or page < 0:
                unusable.append(repr(entry.title))
                continue
            current = (*parents, title)
            found.append(_Bookmark(title=title, section=" > ".join(current), page=page + 1))

    walk(outline, ())
    if unusable:
        logger.warning(
            "%s: ignoring all bookmarks, because some have no title, no page or an "
            "unexpected form: %s",
            doc_id,
            ", ".join(unusable),
        )
        return []
    return sorted(found, key=lambda bookmark: bookmark.page)


def _segments(pages: Sequence[str], bookmarks: Sequence[_Bookmark]) -> Iterator[Segment]:
    for number, text in enumerate(pages, start=1):
        lines = text.splitlines()
        earlier = [bookmark for bookmark in bookmarks if bookmark.page < number]
        pieces: list[tuple[int, str | None]] = [(0, earlier[-1].section if earlier else None)]
        cursor = 0
        for bookmark in (bookmark for bookmark in bookmarks if bookmark.page == number):
            index = _find_line(lines, bookmark.title, start=cursor)
            if index is None:
                pieces[-1] = (pieces[-1][0], None)
            else:
                pieces.append((index, bookmark.section))
                cursor = index + 1
        for position, (start, section) in enumerate(pieces):
            end = pieces[position + 1][0] if position + 1 < len(pieces) else len(lines)
            piece = "\n".join(lines[start:end]).strip()
            if piece:
                yield Segment(text=piece, page=number, section=section)


def _find_line(lines: Sequence[str], title: str, *, start: int) -> int | None:
    """Index of the first line, from `start` on, whose whole text is `title`."""
    return next(
        (index for index in range(start, len(lines)) if " ".join(lines[index].split()) == title),
        None,
    )


def _created(
    reader: PdfReader, path: Path, *, doc_id: str
) -> tuple[dt.date, DateSource] | tuple[None, None]:
    try:
        created = reader.metadata.creation_date if reader.metadata is not None else None
    except (ValueError, PyPdfError) as exc:
        logger.warning("%s: ignoring an unreadable /CreationDate (%s)", doc_id, exc)
        created = None
    if created is not None:
        return created.date(), DateSource.PDF_METADATA
    return file_created(path)

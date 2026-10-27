"""Small factories for domain objects, so each test states only the fields it cares about."""

import datetime as dt

from rag.domain.models import Chunk, DateSource, DocumentInfo, DocumentKind


def doc_info(name: str = "refund-policy.md", *, created: dt.date | None = None) -> DocumentInfo:
    """DocumentInfo whose kind follows the file extension: .pdf, otherwise markdown."""
    kind = DocumentKind.PDF if name.endswith(".pdf") else DocumentKind.MARKDOWN
    created_from = None
    if created is not None:
        is_pdf = kind is DocumentKind.PDF
        created_from = DateSource.PDF_METADATA if is_pdf else DateSource.FRONT_MATTER
    return DocumentInfo(
        doc_id=name, name=name, kind=kind, created=created, created_from=created_from
    )


def chunk(
    text: str = "Refunds are issued within 30 days of purchase.",
    *,
    doc: str = "refund-policy.md",
    ordinal: int = 0,
    page: int | None = None,
    section: str | None = None,
    created: dt.date | None = None,
) -> Chunk:
    """A valid chunk. PDF chunks get page 1 unless a page is given."""
    info = doc_info(doc, created=created)
    if info.kind is DocumentKind.PDF and page is None:
        page = 1
    return Chunk(doc=info, ordinal=ordinal, text=text, page=page, section=section)

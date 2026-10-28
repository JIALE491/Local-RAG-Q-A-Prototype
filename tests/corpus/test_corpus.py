"""The sample corpus: 20 fictional documents with the metadata the loaders rely on."""

import datetime as dt
import re
from pathlib import Path

import pytest
from pypdf import PdfReader
from pypdf.generic import Destination

from scripts.build_pdfs import OUTPUT_DIR, SOURCE_DIR, PdfSource, parse_source, render

MARKDOWN = sorted(OUTPUT_DIR.glob("*.md"))
PDFS = sorted(OUTPUT_DIR.glob("*.pdf"))
SOURCES: dict[str, PdfSource] = {
    path.stem: parse_source(path.read_text(encoding="utf-8"), name=path.name)
    for path in sorted(SOURCE_DIR.glob("*.md"))
}
FRONT_MATTER = re.compile(
    r"---\ntitle: .+\nowner: .+\ncreated: (?P<created>\d{4}-\d{2}-\d{2})\n---\n"
)


def markdown_created(path: Path) -> dt.date:
    match = FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    assert match is not None, f"{path.name}: front matter must be title, owner and created"
    return dt.date.fromisoformat(match["created"])


def test_ten_markdown_files_and_ten_pdfs() -> None:
    assert len(MARKDOWN) == 10
    assert len(PDFS) == 10


def test_every_pdf_comes_from_a_source() -> None:
    assert {path.stem for path in PDFS} == set(SOURCES)


def test_file_names_are_unique_even_without_the_extension() -> None:
    stems = [path.stem for path in MARKDOWN + PDFS]
    assert len(stems) == len(set(stems))


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda path: path.name)
def test_markdown_has_front_matter_with_a_created_date(path: Path) -> None:
    markdown_created(path)


def test_created_dates_are_distinct_spread_out_and_in_the_past() -> None:
    dates = [markdown_created(path) for path in MARKDOWN] + [s.created for s in SOURCES.values()]
    assert len(set(dates)) == len(dates) == 20
    assert len({date.year for date in dates}) >= 5
    assert max(dates) < dt.datetime.now(dt.UTC).date()


@pytest.mark.parametrize("stem", sorted(SOURCES))
def test_pdf_is_up_to_date_with_its_source(stem: str) -> None:
    assert (OUTPUT_DIR / f"{stem}.pdf").read_bytes() == render(SOURCES[stem]), "run `make pdfs`"


@pytest.mark.parametrize("stem", sorted(SOURCES))
def test_pdf_metadata_matches_its_source(stem: str) -> None:
    meta = PdfReader(OUTPUT_DIR / f"{stem}.pdf").metadata
    assert meta is not None
    assert meta.title == SOURCES[stem].title
    assert meta.creation_date is not None
    assert meta.creation_date.date() == SOURCES[stem].created


@pytest.mark.parametrize("stem", sorted(SOURCES))
def test_bookmarks_follow_the_source(stem: str) -> None:
    source = SOURCES[stem]
    outline = PdfReader(OUTPUT_DIR / f"{stem}.pdf").outline
    titles = [entry.title for entry in outline if isinstance(entry, Destination)]
    expected = [block.bookmark or block.text for block in source.blocks if block.kind == "heading"]
    assert titles == (expected if source.bookmarks else [])
    assert len(titles) == len(outline), "bookmarks must not be nested"


@pytest.mark.parametrize("path", PDFS, ids=lambda path: path.name)
def test_every_pdf_page_has_extractable_text(path: Path) -> None:
    pages = PdfReader(path).pages
    assert len(pages) >= 2
    assert all(page.extract_text().strip() for page in pages)


def test_corpus_keeps_its_section_edge_cases() -> None:
    """Cases the PDF section logic must handle; editing a source must not remove them."""
    bookmarked = [stem for stem, source in SOURCES.items() if source.bookmarks]
    assert len(bookmarked) == 7
    assert any(block.bookmark for stem in bookmarked for block in SOURCES[stem].blocks)

    def continues_across_a_page_break(stem: str) -> bool:
        headings = {block.text for block in SOURCES[stem].blocks if block.kind == "heading"}
        later_pages = PdfReader(OUTPUT_DIR / f"{stem}.pdf").pages[1:]
        return any(page.extract_text().splitlines()[0] not in headings for page in later_pages)

    assert any(continues_across_a_page_break(stem) for stem in bookmarked)

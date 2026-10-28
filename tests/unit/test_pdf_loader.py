"""PDF loading: the whole-line section rule, created dates, and PDFs that cannot be used.

Test PDFs are generated with the corpus builder, so their bookmarks and layout are known;
the malformed ones are then altered with pypdf's writer.
"""

import datetime as dt
import logging
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

from rag.adapters.pdf_loader import load_pdf
from rag.domain.errors import DocumentLoadError
from rag.domain.models import DateSource, DocumentKind
from scripts.build_pdfs import parse_source, render

TWO_SECTIONS = "## Policy\n\nWho this covers.\n\n## Refunds\n\nRefunds take 5 days.\n"

LONG_PARAGRAPH = " ".join(["Staff must follow the travel rules on every trip."] * 220)


def make_pdf(tmp_path: Path, body: str, *, bookmarks: bool = True) -> Path:
    source = (
        "---\ntitle: Test Policy\nowner: People Team\ncreated: 2024-05-06\n"
        f"bookmarks: {'true' if bookmarks else 'false'}\n---\n{body}"
    )
    path = tmp_path / "policy.pdf"
    path.write_bytes(render(parse_source(source, name="policy.md")))
    return path


def segments(path: Path) -> list[tuple[int | None, str | None, str]]:
    """(page, section, first line) of every segment."""
    document = load_pdf(path, doc_id=path.name)
    return [(s.page, s.section, s.text.splitlines()[0]) for s in document.segments]


def test_a_page_is_split_where_a_bookmark_title_is_a_line_of_its_own(tmp_path: Path) -> None:
    path = make_pdf(
        tmp_path, "## Scope\n\nWho this covers.\n\n## Refunds\n\nRefunds take 5 days.\n"
    )
    assert segments(path) == [
        (1, None, "Test Policy"),
        (1, "Scope", "Scope"),
        (1, "Refunds", "Refunds"),
    ]


def test_a_title_found_only_inside_a_sentence_leaves_that_part_without_a_section(
    tmp_path: Path,
) -> None:
    body = (
        "## Eligibility\n\nYou can join the 401(k) plan after 60 days.\n\n"
        "## Saving for retirement {bookmark: 401(k) plan}\n\nThe match is 4%.\n\n"
        "## Insurance\n\nLife cover is free.\n"
    )
    path = make_pdf(tmp_path, body)
    assert segments(path) == [
        (1, None, "Test Policy"),
        (1, None, "Eligibility"),
        (1, "Insurance", "Insurance"),
    ]
    unlabelled = load_pdf(path, doc_id="policy.pdf").segments[1]
    assert "The match is 4%." in unlabelled.text


def test_pages_inside_a_long_section_keep_its_name(tmp_path: Path) -> None:
    path = make_pdf(tmp_path, f"## Travel\n\n{LONG_PARAGRAPH}\n\n## Receipts\n\nKeep them.\n")
    result = segments(path)
    pages = sorted({page for page, _, _ in result if page is not None})
    assert len(pages) >= 3, "the long section must fill at least one whole page"
    middle = [entry for entry in result if entry[0] == 2]
    assert [section for _, section, _ in middle] == ["Travel"]
    last_page = [entry for entry in result if entry[0] == pages[-1]]
    assert [section for _, section, _ in last_page] == ["Travel", "Receipts"]


def test_without_bookmarks_each_page_is_one_segment_without_a_section(tmp_path: Path) -> None:
    path = make_pdf(tmp_path, "## Scope\n\nText.\n\n## Refunds\n\nMore.\n", bookmarks=False)
    assert segments(path) == [(1, None, "Test Policy")]


def test_footers_stay_in_the_text(tmp_path: Path) -> None:
    path = make_pdf(tmp_path, "## Scope\n\nText.\n")
    last = load_pdf(path, doc_id="policy.pdf").segments[-1]
    assert last.text.endswith("Tallowmere Outfitters - Test Policy - Page 1 of 1")


def test_document_info_and_created_date_come_from_the_pdf(tmp_path: Path) -> None:
    info = load_pdf(make_pdf(tmp_path, "## Scope\n\nText.\n"), doc_id="policies/policy.pdf").info
    assert (info.doc_id, info.name, info.kind) == (
        "policies/policy.pdf",
        "policy.pdf",
        DocumentKind.PDF,
    )
    assert (info.created, info.created_from) == (dt.date(2024, 5, 6), DateSource.PDF_METADATA)


def test_without_a_creation_date_the_date_comes_from_the_disk(tmp_path: Path) -> None:
    writer = PdfWriter(clone_from=make_pdf(tmp_path, "## Scope\n\nText.\n"))
    writer.metadata = {"/Title": "Test Policy"}
    undated = tmp_path / "undated.pdf"
    writer.write(undated)
    info = load_pdf(undated, doc_id="undated.pdf").info
    has_birth_time = hasattr(undated.stat(), "st_birthtime")
    assert info.created_from is (DateSource.FILE_SYSTEM if has_birth_time else None)


def test_a_pdf_without_text_is_an_error_that_mentions_ocr(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    scan = tmp_path / "scan.pdf"
    writer.write(scan)
    with pytest.raises(DocumentLoadError, match="no extractable text") as caught:
        load_pdf(scan, doc_id="scan.pdf")
    assert caught.value.hint is not None
    assert "OCR" in caught.value.hint


def test_an_encrypted_pdf_is_an_error(tmp_path: Path) -> None:
    writer = PdfWriter(clone_from=make_pdf(tmp_path, "## Scope\n\nText.\n"))
    writer.encrypt(user_password="secret", algorithm="RC4-128")
    locked = tmp_path / "locked.pdf"
    writer.write(locked)
    with pytest.raises(DocumentLoadError, match="encrypted"):
        load_pdf(locked, doc_id="locked.pdf")


def test_a_corrupt_file_is_an_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7 this is not a real PDF")
    with pytest.raises(DocumentLoadError, match="cannot read the PDF"):
        load_pdf(broken, doc_id="broken.pdf")


# Malformed bookmarks and metadata. Each fixture starts from a PDF without bookmarks and
# adds its own outline or metadata, so the damage is the only thing that differs.


def with_outline(tmp_path: Path, add: str) -> Path:
    """The two-section PDF, re-saved with a hand-made outline described by `add`."""
    writer = PdfWriter(clone_from=make_pdf(tmp_path, TWO_SECTIONS, bookmarks=False))
    if add == "nested":
        parent = writer.add_outline_item("Policy", 0)
        writer.add_outline_item("Refunds", 0, parent=parent)
    elif add == "empty title":
        writer.add_outline_item("Policy", 0)
        writer.add_outline_item("", 0)
    elif add == "missing page":
        writer.add_blank_page()
        writer.add_outline_item("Policy", 0)
        writer.add_outline_item("Refunds", 1)
        writer.remove_page(1, clean=False)
    path = tmp_path / f"{add.replace(' ', '-')}.pdf"
    writer.write(path)
    return path


def test_nested_bookmarks_give_a_parent_path(tmp_path: Path) -> None:
    assert segments(with_outline(tmp_path, "nested")) == [
        (1, None, "Test Policy"),
        (1, "Policy", "Policy"),
        (1, "Policy > Refunds", "Refunds"),
    ]


@pytest.mark.parametrize("damage", ["empty title", "missing page"])
def test_one_unusable_bookmark_makes_every_section_empty_with_a_warning(
    tmp_path: Path, damage: str, caplog: pytest.LogCaptureFixture
) -> None:
    path = with_outline(tmp_path, damage)
    with caplog.at_level(logging.WARNING, logger="rag.adapters.pdf_loader"):
        result = segments(path)
    assert result == [(1, None, "Test Policy")]
    assert "ignoring all bookmarks" in caplog.text


def test_an_unreadable_creation_date_falls_back_to_the_disk_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    writer = PdfWriter(clone_from=make_pdf(tmp_path, TWO_SECTIONS))
    writer.add_metadata({"/CreationDate": "not a date"})
    path = tmp_path / "bad-date.pdf"
    writer.write(path)
    with caplog.at_level(logging.WARNING, logger="rag.adapters.pdf_loader"):
        info = load_pdf(path, doc_id="bad-date.pdf").info
    has_birth_time = hasattr(path.stat(), "st_birthtime")
    assert info.created_from is (DateSource.FILE_SYSTEM if has_birth_time else None)
    assert "unreadable /CreationDate" in caplog.text


# pypdf 6 absorbs every outline corruption we could build into a file (loops, dangling
# references, an /Outlines that is not a dictionary), so these two guards are exercised
# by making pypdf itself misbehave.


def test_an_outline_pypdf_cannot_read_is_ignored_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def unreadable(_reader: PdfReader) -> list[object]:
        raise PdfReadError("broken outline")

    path = make_pdf(tmp_path, TWO_SECTIONS)
    monkeypatch.setattr(PdfReader, "outline", property(unreadable))
    with caplog.at_level(logging.WARNING, logger="rag.adapters.pdf_loader"):
        result = segments(path)
    assert result == [(1, None, "Test Policy")]
    assert "bookmarks that cannot be read" in caplog.text


def test_an_outline_entry_of_an_unexpected_type_makes_every_section_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = make_pdf(tmp_path, TWO_SECTIONS)
    monkeypatch.setattr(PdfReader, "outline", property(lambda _reader: ["not a bookmark"]))
    with caplog.at_level(logging.WARNING, logger="rag.adapters.pdf_loader"):
        result = segments(path)
    assert result == [(1, None, "Test Policy")]
    assert "an entry of type str" in caplog.text

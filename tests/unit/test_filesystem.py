"""Scanning the documents folder."""

from pathlib import Path

import pytest

from rag.adapters.filesystem import DirectorySource
from rag.domain.errors import DocumentLoadError
from rag.domain.models import SkippedFile
from scripts.build_pdfs import parse_source, render

MARKDOWN = "---\ncreated: 2023-06-19\n---\n\n# FAQ\n\nAnswers.\n"
PDF_SOURCE = (
    "---\ntitle: Refunds\nowner: Finance\ncreated: 2024-02-12\nbookmarks: false\n---\n"
    "## Window\n\n45 days.\n"
)


def write(root: Path, relative: str, content: str | bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def test_finds_markdown_and_pdf_files_in_subfolders_in_path_order(tmp_path: Path) -> None:
    write(tmp_path, "policies/refunds.pdf", render(parse_source(PDF_SOURCE, name="refunds")))
    write(tmp_path, "faq.md", MARKDOWN)
    write(tmp_path, "Guide.MD", MARKDOWN)
    scan = DirectorySource(tmp_path).scan()
    assert [d.info.doc_id for d in scan.documents] == ["Guide.MD", "faq.md", "policies/refunds.pdf"]
    assert [d.info.name for d in scan.documents] == ["Guide.MD", "faq.md", "refunds.pdf"]
    assert scan.skipped == ()


def test_unsupported_files_are_reported_not_dropped(tmp_path: Path) -> None:
    write(tmp_path, "faq.md", MARKDOWN)
    write(tmp_path, "notes.txt", "x")
    write(tmp_path, "LICENSE", "x")
    scan = DirectorySource(tmp_path).scan()
    assert scan.skipped == (
        SkippedFile(path="LICENSE", reason="no file extension; only PDF and markdown"),
        SkippedFile(path="notes.txt", reason="unsupported file type .txt; only PDF and markdown"),
    )


def test_hidden_files_and_folders_are_ignored(tmp_path: Path) -> None:
    write(tmp_path, "faq.md", MARKDOWN)
    write(tmp_path, ".DS_Store", b"\x00")
    write(tmp_path, ".drafts/old.md", MARKDOWN)
    scan = DirectorySource(tmp_path).scan()
    assert [d.info.doc_id for d in scan.documents] == ["faq.md"]
    assert scan.skipped == ()


def test_two_files_with_the_same_name_are_an_error(tmp_path: Path) -> None:
    write(tmp_path, "a/faq.md", MARKDOWN)
    write(tmp_path, "b/faq.md", MARKDOWN)
    with pytest.raises(DocumentLoadError, match=r"same file name as a/faq\.md") as caught:
        DirectorySource(tmp_path).scan()
    assert caught.value.path == "b/faq.md"


def test_a_missing_folder_is_an_error_with_a_hint(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="does not exist") as caught:
        DirectorySource(tmp_path / "missing").scan()
    assert caught.value.hint is not None
    assert "docs_dir" in caught.value.hint

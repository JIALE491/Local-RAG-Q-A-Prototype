"""Markdown loading: front matter dates, heading paths, and what becomes a segment."""

import datetime as dt
from pathlib import Path

import pytest

from rag.adapters.markdown_loader import load_markdown
from rag.domain.errors import DocumentLoadError
from rag.domain.models import DateSource, Document, DocumentKind

HANDBOOK = """\
---
title: Handbook
created: 2022-01-10
---

# Employee Handbook

Applies to everyone.

## Vacation

### How much vacation do I earn?

Fifteen days a year.

### Can I carry vacation over?

Up to five days.

## Sick leave

Ten days a year.
"""


def load(tmp_path: Path, text: str, name: str = "handbook.md") -> Document:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return load_markdown(path, doc_id=name)


def sections(document: Document) -> list[tuple[str | None, str]]:
    return [(segment.section, segment.text) for segment in document.segments]


def test_sections_are_the_heading_path_below_the_title(tmp_path: Path) -> None:
    assert sections(load(tmp_path, HANDBOOK)) == [
        ("Employee Handbook", "Employee Handbook\n\nApplies to everyone."),
        (
            "Vacation > How much vacation do I earn?",
            "How much vacation do I earn?\n\nFifteen days a year.",
        ),
        ("Vacation > Can I carry vacation over?", "Can I carry vacation over?\n\nUp to five days."),
        ("Sick leave", "Sick leave\n\nTen days a year."),
    ]


def test_document_info_comes_from_the_file_and_front_matter(tmp_path: Path) -> None:
    document = load(tmp_path, HANDBOOK)
    info = document.info
    assert (info.doc_id, info.name, info.kind) == (
        "handbook.md",
        "handbook.md",
        DocumentKind.MARKDOWN,
    )
    assert (info.created, info.created_from) == (dt.date(2022, 1, 10), DateSource.FRONT_MATTER)
    assert all(segment.page is None for segment in document.segments)


def test_a_quoted_created_date_is_accepted(tmp_path: Path) -> None:
    info = load(tmp_path, HANDBOOK.replace("2022-01-10", '"2022-01-10"')).info
    assert info.created == dt.date(2022, 1, 10)


def test_without_front_matter_the_date_comes_from_the_disk(tmp_path: Path) -> None:
    document = load(tmp_path, "# Title\n\nText.\n")
    has_birth_time = hasattr((tmp_path / "handbook.md").stat(), "st_birthtime")
    expected = DateSource.FILE_SYSTEM if has_birth_time else None
    assert document.info.created_from is expected


def test_an_invalid_created_date_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="YYYY-MM-DD"):
        load(tmp_path, HANDBOOK.replace("2022-01-10", "soon"))


def test_unclosed_front_matter_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="never closed"):
        load(tmp_path, "---\ncreated: 2022-01-10\n\n# Title\n\nText.\n")


def test_text_before_any_heading_has_no_section(tmp_path: Path) -> None:
    assert sections(load(tmp_path, "Intro.\n\n## Part\n\nBody.\n")) == [
        (None, "Intro."),
        ("Part", "Part\n\nBody."),
    ]


def test_several_level_one_headings_all_stay_in_the_path(tmp_path: Path) -> None:
    text = "# First\n\nOne.\n\n# Second\n\n## Detail\n\nTwo.\n"
    assert [section for section, _ in sections(load(tmp_path, text))] == [
        "First",
        "Second > Detail",
    ]


def test_hashes_inside_a_code_fence_are_not_headings(tmp_path: Path) -> None:
    text = "## Setup\n\n```\n# not a heading\n```\n"
    [(section, body)] = sections(load(tmp_path, text))
    assert section == "Setup"
    assert "# not a heading" in body


def test_closing_hashes_are_not_part_of_the_heading(tmp_path: Path) -> None:
    [(section, _)] = sections(load(tmp_path, "## Returns ##\n\nText.\n"))
    assert section == "Returns"


def test_a_file_with_only_headings_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="no text"):
        load(tmp_path, "---\ncreated: 2022-01-10\n---\n\n## Empty\n")


def test_a_file_that_is_not_utf8_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_bytes(b"\xff\xfe# Title")
    with pytest.raises(DocumentLoadError, match="cannot read"):
        load_markdown(path, doc_id="broken.md")

"""The PDF builder: a strict source format, and deterministic output with metadata."""

import datetime as dt
import io

import pytest
from pypdf import PdfReader
from pypdf.generic import Destination

from scripts.build_pdfs import Block, SourceError, parse_source, render

SOURCE = """\
---
title: Sample Policy
owner: People Team
created: 2024-05-06
bookmarks: true
---
## First heading
Some text that
continues here.

- One
- Two

## Printed heading {bookmark: Outline title}
More text.
"""


def read(source_text: str) -> PdfReader:
    return PdfReader(io.BytesIO(render(parse_source(source_text, name="sample.md"))))


def test_parses_front_matter_and_blocks() -> None:
    source = parse_source(SOURCE, name="sample.md")
    assert (source.title, source.owner, source.created, source.bookmarks) == (
        "Sample Policy",
        "People Team",
        dt.date(2024, 5, 6),
        True,
    )
    assert source.blocks == (
        Block("heading", "First heading"),
        Block("paragraph", "Some text that continues here."),
        Block("bullet", "One"),
        Block("bullet", "Two"),
        Block("heading", "Printed heading", "Outline title"),
        Block("paragraph", "More text."),
    )


@pytest.mark.parametrize(
    ("broken", "problem"),
    [
        (SOURCE.replace("owner: People Team\n", ""), "missing ['owner']"),
        (
            SOURCE.replace("bookmarks: true", "bookmarks: true\ncolor: red"),
            "unknown keys ['color']",
        ),
        (SOURCE.replace("2024-05-06", "May 6"), "YYYY-MM-DD"),
        (SOURCE.replace("bookmarks: true", "bookmarks: yes"), "true or false"),
        (SOURCE.replace("Some text", "Some café text"), "sample.md:8"),
        (SOURCE.replace("## First", "### First"), "sample.md:7: only '## ' headings"),
        (SOURCE.replace("bookmarks: true", "bookmarks: false"), "needs bookmarks: true"),
        ("no front matter", "front matter"),
    ],
)
def test_rejects_malformed_sources(broken: str, problem: str) -> None:
    with pytest.raises(SourceError) as caught:
        parse_source(broken, name="sample.md")
    assert problem in str(caught.value)


def test_writes_metadata_and_bookmarks() -> None:
    reader = read(SOURCE)
    meta = reader.metadata
    assert meta is not None
    assert meta.title == "Sample Policy"
    assert meta.author == "Tallowmere Outfitters People Team"
    assert meta.creation_date is not None
    assert meta.creation_date.date() == dt.date(2024, 5, 6)
    titles = [entry.title for entry in reader.outline if isinstance(entry, Destination)]
    assert titles == ["First heading", "Outline title"]


def test_prints_the_heading_not_the_bookmark_override() -> None:
    text = read(SOURCE).pages[0].extract_text()
    assert "Printed heading" in text
    assert "Outline title" not in text


def test_writes_no_bookmarks_when_disabled() -> None:
    plain = SOURCE.replace(" {bookmark: Outline title}", "").replace(
        "bookmarks: true", "bookmarks: false"
    )
    assert read(plain).outline == []


def test_output_is_deterministic() -> None:
    source = parse_source(SOURCE, name="sample.md")
    assert render(source) == render(source)

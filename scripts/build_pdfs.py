"""Build the sample corpus PDFs from their text sources.

    python scripts/build_pdfs.py          # write data/docs/<name>.pdf for every source
    python scripts/build_pdfs.py --check  # exit 1 if any PDF is missing or out of date

Each source in data/pdf-sources/ uses a small markdown subset:

    ---
    title: Returns and Refunds Policy
    owner: Customer Experience
    created: 2024-02-12
    bookmarks: true
    ---
    ## Heading                     a section heading; also a bookmark when bookmarks is true
    ## Heading {bookmark: Other}   the bookmark text differs from the printed heading
    - item                         a bullet point
    any other text                 a paragraph; consecutive lines are joined

The output is deterministic: the same source always gives a byte-identical PDF, so the
committed PDFs can be checked against their sources.
"""

import argparse
import datetime as dt
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fpdf
from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "pdf-sources"
OUTPUT_DIR = ROOT / "data" / "docs"
COMPANY = "Tallowmere Outfitters"

_HEADING = re.compile(r"## (?P<text>.+?)(?: \{bookmark: (?P<bookmark>[^}]+)\})?")
_FRONT_MATTER_KEYS = frozenset({"title", "owner", "created", "bookmarks"})


class SourceError(ValueError):
    """A PDF source file is malformed."""


@dataclass(frozen=True, slots=True)
class Block:
    kind: Literal["heading", "paragraph", "bullet"]
    text: str
    bookmark: str | None = None
    """Outline entry for a heading, when it differs from the printed heading."""


@dataclass(frozen=True, slots=True)
class PdfSource:
    title: str
    owner: str
    created: dt.date
    bookmarks: bool
    blocks: tuple[Block, ...]


def parse_source(text: str, *, name: str) -> PdfSource:
    """Parse one source file. Raises SourceError naming the file and, where possible, line."""
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        bad = next((char for char in line if not " " <= char <= "~"), None)
        if bad is not None:
            raise SourceError(
                f"{name}:{number}: {bad!r} is not printable ASCII; the built-in PDF fonts "
                "only cover Latin-1, so sources stay ASCII"
            )
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        raise SourceError(f"{name}: must start with front matter between two '---' lines")
    end = lines.index("---", 1)
    meta = _parse_front_matter(lines[1:end], name=name)
    try:
        created = dt.date.fromisoformat(meta["created"])
    except ValueError as exc:
        raise SourceError(f"{name}: created must be YYYY-MM-DD, got {meta['created']!r}") from exc
    if meta["bookmarks"] not in {"true", "false"}:
        raise SourceError(f"{name}: bookmarks must be true or false, got {meta['bookmarks']!r}")
    bookmarks = meta["bookmarks"] == "true"
    blocks = _parse_body(lines[end + 1 :], name=name, first_line=end + 2)
    if not bookmarks and any(block.bookmark for block in blocks):
        raise SourceError(f"{name}: a {{bookmark: ...}} override needs bookmarks: true")
    return PdfSource(
        title=meta["title"],
        owner=meta["owner"],
        created=created,
        bookmarks=bookmarks,
        blocks=blocks,
    )


def render(source: PdfSource) -> bytes:
    """Lay out a source as a letter-size PDF with metadata and, if asked, bookmarks."""
    pdf = _Document(source)
    pdf.set_title(source.title)
    pdf.set_author(f"{COMPANY} {source.owner}")
    pdf.set_creator("scripts/build_pdfs.py")
    pdf.set_producer(f"fpdf2 {fpdf.__version__}")
    pdf.set_lang("en-US")
    pdf.set_creation_date(dt.datetime.combine(source.created, dt.time(12, 0), tzinfo=dt.UTC))
    pdf.set_margins(left=22, top=20, right=22)
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()

    pdf.set_font("Helvetica", style="B", size=18)
    _write(pdf, source.title, height=9)
    pdf.set_font("Helvetica", size=9.5)
    pdf.set_text_color(90)
    effective = f"{source.created:%B} {source.created.day}, {source.created.year}"
    _write(pdf, f"{COMPANY} | {source.owner} | Effective {effective}", height=5.5)
    pdf.set_text_color(0)
    pdf.ln(5)

    for block in source.blocks:
        if block.kind == "heading":
            if pdf.will_page_break(20):
                pdf.add_page()
            pdf.ln(2.5)
            if source.bookmarks:
                pdf.start_section(block.bookmark or block.text)
            pdf.set_font("Helvetica", style="B", size=12.5)
            _write(pdf, block.text, height=6.5)
            pdf.ln(1)
        elif block.kind == "paragraph":
            pdf.set_font("Helvetica", size=10.5)
            _write(pdf, block.text, height=5.4)
            pdf.ln(2)
        else:
            pdf.set_font("Helvetica", size=10.5)
            pdf.set_x(pdf.l_margin + 4)
            _write(pdf, f"- {block.text}", height=5.4, width=pdf.epw - 4)
            pdf.ln(0.8)
    return bytes(pdf.output())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/docs/*.pdf from data/pdf-sources/.")
    parser.add_argument(
        "--check", action="store_true", help="report missing or out-of-date PDFs, write nothing"
    )
    check: bool = parser.parse_args(argv).check
    sources = sorted(SOURCE_DIR.glob("*.md"))
    if not sources:
        print(f"error: no sources in {SOURCE_DIR}", file=sys.stderr)
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for path in sources:
        try:
            pdf_bytes = render(parse_source(path.read_text(encoding="utf-8"), name=path.name))
        except SourceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        target = OUTPUT_DIR / f"{path.stem}.pdf"
        if check:
            if not target.is_file() or target.read_bytes() != pdf_bytes:
                stale.append(target.name)
        else:
            target.write_bytes(pdf_bytes)
            print(f"wrote {target.relative_to(ROOT)} ({len(pdf_bytes):,} bytes)")
    if stale:
        print(f"out of date: {', '.join(stale)} (run `make pdfs`)", file=sys.stderr)
        return 1
    return 0


class _Document(FPDF):
    """Letter-size page with a footer naming the company, document and page."""

    def __init__(self, source: PdfSource) -> None:
        super().__init__(orientation="portrait", unit="mm", format="letter")
        self._title = source.title

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", style="I", size=8)
        self.set_text_color(110)
        footer = f"{COMPANY} - {self._title} - Page {self.page_no()} of {{nb}}"
        self.cell(w=0, h=8, text=footer, align="C")


def _write(pdf: FPDF, text: str, *, height: float, width: float = 0) -> None:
    pdf.multi_cell(w=width, h=height, text=text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _parse_front_matter(lines: Sequence[str], *, name: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition(":")
        if not separator or not value.strip():
            raise SourceError(f"{name}: front matter line {line!r} is not 'key: value'")
        meta[key.strip()] = value.strip()
    unknown = sorted(set(meta) - _FRONT_MATTER_KEYS)
    missing = sorted(_FRONT_MATTER_KEYS - set(meta))
    if unknown or missing:
        raise SourceError(f"{name}: front matter has unknown keys {unknown}, missing {missing}")
    return meta


def _parse_body(lines: Sequence[str], *, name: str, first_line: int) -> tuple[Block, ...]:
    blocks: list[Block] = []
    paragraph: list[str] = []

    def end_paragraph() -> None:
        if paragraph:
            blocks.append(Block("paragraph", " ".join(paragraph)))
            paragraph.clear()

    for offset, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            end_paragraph()
        elif line.startswith("#"):
            end_paragraph()
            match = _HEADING.fullmatch(line)
            if match is None:
                raise SourceError(f"{name}:{first_line + offset}: only '## ' headings are allowed")
            blocks.append(Block("heading", match["text"], match["bookmark"]))
        elif line.startswith("- "):
            end_paragraph()
            blocks.append(Block("bullet", line.removeprefix("- ").strip()))
        else:
            paragraph.append(line)
    end_paragraph()
    if not blocks:
        raise SourceError(f"{name}: has no content after the front matter")
    return tuple(blocks)


if __name__ == "__main__":
    raise SystemExit(main())

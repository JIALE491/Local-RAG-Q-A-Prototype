"""Markdown files: the created date from front matter, sections from headings.

Front matter is optional; only its `created: YYYY-MM-DD` line is read. Every heading starts a
new segment, labelled with the path of headings above it, such as
"Vacation > How much vacation do I earn?", and the segment text begins with the heading
itself. A heading with no text of its own (only subheadings) produces no segment.

When a file has exactly one level-1 heading and it comes before all other headings, it is
the document's title: it is left out of the paths below it, and only the text directly under
it is labelled with it. Lines inside code fences are never treated as headings.
"""

import datetime as dt
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

from rag.adapters.files import file_created
from rag.domain.errors import DocumentLoadError
from rag.domain.models import DateSource, Document, DocumentInfo, DocumentKind, Segment

_HEADING = re.compile(r"(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?")
_FENCE = re.compile(r"[ \t]{0,3}(?:```|~~~)")
_CREATED = re.compile(r"created:[ \t]*(?P<value>.*)")


def load_markdown(path: Path, *, doc_id: str) -> Document:
    """Parse one markdown file. Raises DocumentLoadError if it cannot be used."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DocumentLoadError(doc_id, f"cannot read the file: {exc}") from exc
    front_matter, body = _split_front_matter(text.splitlines(), doc_id=doc_id)
    created, created_from = _created(front_matter, path, doc_id=doc_id)
    segments = tuple(_segments(body))
    if not segments:
        raise DocumentLoadError(doc_id, "the file has no text to index")
    info = DocumentInfo(
        doc_id=doc_id,
        name=path.name,
        kind=DocumentKind.MARKDOWN,
        created=created,
        created_from=created_from,
    )
    return Document(info=info, segments=segments)


def _split_front_matter(lines: list[str], *, doc_id: str) -> tuple[list[str], list[str]]:
    if not lines or lines[0].strip() != "---":
        return [], lines
    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            return lines[1:index], lines[index + 1 :]
    raise DocumentLoadError(doc_id, "the front matter starts with '---' but is never closed")


def _created(
    front_matter: Sequence[str], path: Path, *, doc_id: str
) -> tuple[dt.date, DateSource] | tuple[None, None]:
    for line in front_matter:
        match = _CREATED.fullmatch(line.strip())
        if match is None:
            continue
        value = match["value"].strip().strip("'\"")
        try:
            return dt.date.fromisoformat(value), DateSource.FRONT_MATTER
        except ValueError as exc:
            raise DocumentLoadError(
                doc_id, f"created in the front matter must be a YYYY-MM-DD date, got {value!r}"
            ) from exc
    return file_created(path)


def _segments(lines: Sequence[str]) -> Iterator[Segment]:
    headings = _headings(lines)
    level_one = [heading for heading in headings if heading[1] == 1]
    has_title = bool(headings) and headings[0][1] == 1 and len(level_one) == 1

    first = headings[0][0] if headings else len(lines)
    yield from _segment(lines[:first], heading=None, section=None)

    path: list[tuple[int, str]] = []
    for position, (index, level, text) in enumerate(headings):
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, text))
        names = [name for _, name in path]
        if has_title and len(names) > 1:
            names = names[1:]
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        yield from _segment(lines[index + 1 : end], heading=text, section=" > ".join(names))


def _headings(lines: Sequence[str]) -> list[tuple[int, int, str]]:
    """(line index, level, text) of every heading outside code fences."""
    found: list[tuple[int, int, str]] = []
    in_fence = False
    for index, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and (match := _HEADING.fullmatch(line.rstrip())):
            found.append((index, len(match[1]), match[2].strip()))
    return found


def _segment(body: Sequence[str], *, heading: str | None, section: str | None) -> Iterator[Segment]:
    content = "\n".join(body).strip()
    if content:
        text = f"{heading}\n\n{content}" if heading else content
        yield Segment(text=text, page=None, section=section)

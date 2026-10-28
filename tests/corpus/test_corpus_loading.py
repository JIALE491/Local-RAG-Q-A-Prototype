"""The real sample corpus through the loaders: the section rule on its built-in edge cases."""

from rag.adapters.filesystem import DirectorySource
from rag.domain.models import DateSource, Segment
from scripts.build_pdfs import OUTPUT_DIR

SCAN = DirectorySource(OUTPUT_DIR).scan()
DOCUMENTS = {document.info.name: document for document in SCAN.documents}


def page(name: str, number: int) -> list[Segment]:
    return [segment for segment in DOCUMENTS[name].segments if segment.page == number]


def test_every_document_loads_and_nothing_is_skipped() -> None:
    assert len(SCAN.documents) == 20
    assert SCAN.skipped == ()


def test_created_dates_come_from_the_documents_not_the_disk() -> None:
    sources = {document.info.created_from for document in SCAN.documents}
    assert sources == {DateSource.PDF_METADATA, DateSource.FRONT_MATTER}


def test_benefits_page_one_leaves_the_part_with_the_unfound_bookmark_unlabelled() -> None:
    segments = page("benefits-overview.pdf", 1)
    assert [(s.section, s.text.splitlines()[0]) for s in segments] == [
        (None, "Benefits Overview 2026"),
        ("Who is eligible", "Who is eligible"),
        ("Medical plans", "Medical plans"),
        (None, "Dental and vision"),
        ("Insurance", "Insurance"),
        ("Employee discount", "Employee discount"),
        ("Wellness stipend", "Wellness stipend"),
    ]
    assert "You can join the 401(k) plan" in segments[3].text


def test_travel_page_two_starts_by_continuing_travel_safety() -> None:
    segments = page("travel-and-expense-policy.pdf", 2)
    assert segments[0].section == "Travel safety"
    assert segments[0].text.startswith("keeps you away longer than planned.")
    assert segments[1].section == "Ground transportation"


def test_footers_stay_in_the_page_text() -> None:
    last = page("benefits-overview.pdf", 1)[-1]
    assert last.text.endswith("Tallowmere Outfitters - Benefits Overview 2026 - Page 1 of 2")


def test_pdfs_without_bookmarks_are_one_unlabelled_segment_per_page() -> None:
    for name in ("code-of-conduct.pdf", "data-retention-policy.pdf", "warranty-and-repairs.pdf"):
        segments = DOCUMENTS[name].segments
        assert [s.page for s in segments] == [1, 2]
        assert all(s.section is None for s in segments)


def test_markdown_sections_are_heading_paths() -> None:
    sections = {segment.section for segment in DOCUMENTS["employee-handbook-time-off.md"].segments}
    assert "Employee Handbook: Time Off" in sections
    assert "Vacation > How much vacation do I earn?" in sections
    assert "Vacation" not in sections, "a heading with only subheadings has no segment"

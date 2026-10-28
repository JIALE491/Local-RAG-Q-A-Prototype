"""DocumentSource for a folder of PDF and markdown files."""

from pathlib import Path

from rag.adapters.markdown_loader import load_markdown
from rag.adapters.pdf_loader import load_pdf
from rag.domain.errors import DocumentLoadError
from rag.domain.models import Document, SkippedFile, SourceScan


class DirectorySource:
    """Every PDF and markdown file under a folder, including subfolders, sorted by path.

    Hidden files and folders (names starting with '.') are ignored. Other unsupported files
    are reported as skipped, with the reason.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def scan(self) -> SourceScan:
        if not self._root.is_dir():
            raise DocumentLoadError(
                str(self._root),
                "the documents folder does not exist",
                hint="check [paths] docs_dir in config.toml",
            )
        documents: list[Document] = []
        skipped: list[SkippedFile] = []
        names: dict[str, str] = {}
        for path in sorted(self._root.rglob("*")):
            relative = path.relative_to(self._root)
            if not path.is_file() or any(part.startswith(".") for part in relative.parts):
                continue
            doc_id = relative.as_posix()
            suffix = path.suffix.lower()
            if suffix not in {".md", ".pdf"}:
                reason = f"unsupported file type {suffix}" if suffix else "no file extension"
                skipped.append(SkippedFile(path=doc_id, reason=f"{reason}; only PDF and markdown"))
                continue
            if path.name in names:
                raise DocumentLoadError(
                    doc_id,
                    f"has the same file name as {names[path.name]}",
                    hint="file names must be unique, because the test set names documents by file",
                )
            names[path.name] = doc_id
            load = load_pdf if suffix == ".pdf" else load_markdown
            documents.append(load(path, doc_id=doc_id))
        return SourceScan(documents=tuple(documents), skipped=tuple(skipped))

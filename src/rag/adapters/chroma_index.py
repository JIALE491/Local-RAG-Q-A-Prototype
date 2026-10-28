"""VectorIndex stored on disk with chromadb: one folder per chunking strategy.

    index/fixed-300-50/
        current.json                          which build is live, and what built it
        build-20261028T104512123456-a3f9c2/   one chromadb database per build

A rebuild first deletes current.json, so the index stops answering searches, then deletes
the earlier builds and fills a new build folder. When it finishes, current.json is written to
a temporary file in the same folder, flushed to disk and renamed into place, so it is never
half-written. Every build gets a folder chromadb has never opened, because chromadb keeps each
database it opens cached for the life of the process: a client opened on a deleted and
recreated folder would be handed the stale copy. Only chromadb's public API is used, and
telemetry is off.

chromadb silently ignores an add() for an id it already stores, so ids are checked here first.
It also rejects None in metadata, so fields without a value are left out.
"""

import datetime as dt
import json
import logging
import os
import re
import secrets
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata
from chromadb.config import Settings
from chromadb.errors import ChromaError

from rag.domain.errors import IndexNotBuiltError, InvalidDataError, VectorStoreError
from rag.domain.models import (
    Chunk,
    DateSource,
    DocumentInfo,
    DocumentKind,
    Embedding,
    IndexManifest,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

SETTINGS = Settings(anonymized_telemetry=False)
POINTER = "current.json"
BUILD_PREFIX = "build-"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]")
"""chromadb's rule for collection names, which also makes a safe folder name."""


class ChromaIndex:
    """One chunking strategy's index, in a folder under `path` named after the strategy."""

    def __init__(self, *, path: Path, collection: str) -> None:
        if not _NAME.fullmatch(collection):
            raise InvalidDataError(
                f"{collection!r} cannot name an index: use 3 to 512 letters, digits, '.', '_' "
                "or '-', starting and ending with a letter or digit"
            )
        self._name = collection
        self._folder = path / collection
        self._clients: dict[Path, ClientAPI] = {}
        self._building: tuple[Path, Collection] | None = None
        self._dim: int | None = None

    def begin_rebuild(self) -> None:
        try:
            self._folder.mkdir(parents=True, exist_ok=True)
            (self._folder / POINTER).unlink(missing_ok=True)
        except OSError as exc:
            raise VectorStoreError(f"cannot prepare {self._folder} for a rebuild: {exc}") from exc
        self._remove_earlier_builds()
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%f")
        build = self._folder / f"{BUILD_PREFIX}{stamp}-{secrets.token_hex(3)}"
        with _chroma_errors(f"create a new build in {self._folder}"):
            collection = self._client(build).create_collection(
                self._name,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,
            )
        self._building = (build, collection)
        self._dim = None

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        _, collection = self._require_rebuild()
        if len(chunks) != len(embeddings):
            raise VectorStoreError(f"got {len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        with _chroma_errors("check for chunk ids that are already stored"):
            stored = set(collection.get(ids=ids, include=[])["ids"])
        repeated = {i for i, n in Counter(ids).items() if n > 1} | stored
        if repeated:
            raise VectorStoreError(f"duplicate chunk ids: {', '.join(sorted(repeated)[:5])}")
        sizes = {len(e) for e in embeddings} | ({self._dim} if self._dim is not None else set())
        if len(sizes) > 1 or 0 in sizes:
            raise VectorStoreError(
                f"embedding sizes must match and be non-zero, got {sorted(sizes)}"
            )
        metadatas: list[Metadata] = [_to_metadata(chunk) for chunk in chunks]
        with _chroma_errors("store chunks"):
            collection.add(
                ids=ids,
                embeddings=_vectors(embeddings),
                documents=[chunk.text for chunk in chunks],
                metadatas=metadatas,
            )
        self._dim = len(embeddings[0])

    def finish_rebuild(self, manifest: IndexManifest) -> None:
        build, _ = self._require_rebuild()
        self._write_pointer(
            {
                "build": build.name,
                "strategy": manifest.strategy,
                "embed_model": manifest.embed_model,
            }
        )
        self._building = None

    def query(self, embeddings: Sequence[Embedding], k: int) -> list[list[RetrievedChunk]]:
        collection = self._finished_collection()
        if k < 1:
            raise InvalidDataError(f"k must be 1 or more, got {k}")
        if not embeddings:
            return []
        with _chroma_errors("search"):
            result = collection.query(
                query_embeddings=_vectors(embeddings),
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        documents, metadatas, distances = (
            result["documents"],
            result["metadatas"],
            result["distances"],
        )
        if documents is None or metadatas is None or distances is None:
            raise VectorStoreError("chromadb returned a search result without its fields")
        return [
            [
                RetrievedChunk(chunk=_to_chunk(text, meta), score=1.0 - distance, rank=rank)
                for rank, (text, meta, distance) in enumerate(
                    zip(texts, metas, dists, strict=True), start=1
                )
            ]
            for texts, metas, dists in zip(documents, metadatas, distances, strict=True)
        ]

    def chunks(self) -> list[Chunk]:
        collection = self._finished_collection()
        with _chroma_errors("read the stored chunks"):
            result = collection.get(include=["documents", "metadatas"])
        documents, metadatas = result["documents"], result["metadatas"]
        if documents is None or metadatas is None:
            raise VectorStoreError("chromadb returned stored chunks without their fields")
        chunks = [_to_chunk(text, meta) for text, meta in zip(documents, metadatas, strict=True)]
        return sorted(chunks, key=lambda chunk: (chunk.doc.doc_id, chunk.ordinal))

    def count(self) -> int:
        if self._building is not None:
            collection = self._building[1]
        else:
            current = self._current()
            if current is None:
                return 0
            collection = self._open(current[0])
        with _chroma_errors("count chunks"):
            return collection.count()

    def manifest(self) -> IndexManifest | None:
        current = self._current()
        return None if current is None else current[1]

    def _current(self) -> tuple[Path, IndexManifest] | None:
        """The finished build named in current.json, with its manifest; None if there is none."""
        pointer = self._folder / POINTER
        try:
            text = pointer.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise VectorStoreError(f"cannot read {pointer}: {exc}") from exc
        damaged = f"{pointer} is damaged"
        try:
            content = json.loads(text)
            build = self._folder / _text(content, "build")
            manifest = IndexManifest(
                strategy=_text(content, "strategy"), embed_model=_text(content, "embed_model")
            )
        except ValueError as exc:
            raise VectorStoreError(f"{damaged}: {exc}", hint="run `rag ingest` to rebuild") from exc
        if build.parent != self._folder or not build.name.startswith(BUILD_PREFIX):
            raise VectorStoreError(
                f"{damaged}: it names {build.name!r}, which is not a build folder",
                hint="run `rag ingest` to rebuild",
            )
        if not build.is_dir():
            raise VectorStoreError(
                f"{damaged}: the build it names, {build.name}, does not exist",
                hint="run `rag ingest` to rebuild",
            )
        return build, manifest

    def _finished_collection(self) -> Collection:
        current = self._current()
        if current is None:
            if self._has_builds():
                raise IndexNotBuiltError(
                    f"the last rebuild of {self._name!r} did not finish",
                    hint="run `rag ingest` again",
                )
            raise IndexNotBuiltError(
                f"there is no index for {self._name!r} yet", hint="run `rag ingest`"
            )
        return self._open(current[0])

    def _open(self, build: Path) -> Collection:
        with _chroma_errors(f"open the build {build.name}"):
            return self._client(build).get_collection(self._name, embedding_function=None)

    def _client(self, build: Path) -> ClientAPI:
        if build not in self._clients:
            with _chroma_errors(f"open {build}"):
                self._clients[build] = chromadb.PersistentClient(path=str(build), settings=SETTINGS)
        return self._clients[build]

    def _has_builds(self) -> bool:
        return self._folder.is_dir() and any(_is_build(entry) for entry in self._folder.iterdir())

    def _remove_earlier_builds(self) -> None:
        """Delete every earlier build, finished or not, and leftover temporary pointer files.

        macOS and Linux allow deleting files that chromadb still has open. Where that fails,
        the leftover is reported and the next rebuild tries again.
        """
        try:
            entries = list(self._folder.iterdir())
        except OSError as exc:
            raise VectorStoreError(f"cannot list {self._folder}: {exc}") from exc
        for entry in entries:
            try:
                if _is_build(entry):
                    self._clients.pop(entry, None)
                    shutil.rmtree(entry)
                elif entry.is_file() and _is_temporary_pointer(entry):
                    entry.unlink()
            except OSError as exc:
                logger.warning(
                    "could not remove the old index files %s (%s); the next rebuild tries again",
                    entry,
                    exc,
                )

    def _write_pointer(self, content: Mapping[str, str]) -> None:
        temporary = self._folder / f".{POINTER}.{secrets.token_hex(4)}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(content, file, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(self._folder / POINTER)
            _sync_folder(self._folder)
        except OSError as exc:
            with suppress(OSError):  # the temporary file may never have been created
                temporary.unlink()
            raise VectorStoreError(
                f"cannot record the finished rebuild in {self._folder}: {exc}"
            ) from exc

    def _require_rebuild(self) -> tuple[Path, Collection]:
        if self._building is None:
            raise VectorStoreError("no rebuild in progress", hint="call begin_rebuild() first")
        return self._building


@contextmanager
def _chroma_errors(action: str) -> Iterator[None]:
    try:
        yield
    except ChromaError as exc:
        raise VectorStoreError(f"chromadb could not {action}: {exc}") from exc


def _is_build(entry: Path) -> bool:
    return entry.name.startswith(BUILD_PREFIX) and entry.is_dir()


def _is_temporary_pointer(entry: Path) -> bool:
    return entry.name.startswith(f".{POINTER}.") and entry.name.endswith(".tmp")


def _sync_folder(folder: Path) -> None:
    """Make a rename inside `folder` durable. Windows cannot open folders, and needs no sync."""
    if os.name == "nt":
        return
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _text(content: object, key: str) -> str:
    if not isinstance(content, dict):
        raise ValueError("it is not a JSON object")
    value = content.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key!r} should be a non-empty string, got {value!r}")
    return value


def _vectors(embeddings: Sequence[Embedding]) -> list[Sequence[float] | Sequence[int]]:
    """The embeddings in the exact list type chromadb's signatures declare."""
    return [list(embedding) for embedding in embeddings]


def _to_metadata(chunk: Chunk) -> Metadata:
    info = chunk.doc
    metadata: dict[str, str | int] = {
        "doc_id": info.doc_id,
        "doc_name": info.name,
        "kind": info.kind.value,
        "ordinal": chunk.ordinal,
    }
    if chunk.page is not None:
        metadata["page"] = chunk.page
    if chunk.section is not None:
        metadata["section"] = chunk.section
    if info.created is not None and info.created_from is not None:
        metadata["created"] = info.created.isoformat()
        metadata["created_from"] = info.created_from.value
    return metadata


def _to_chunk(text: str | None, metadata: Mapping[str, object] | None) -> Chunk:
    if text is None or metadata is None:
        raise VectorStoreError(
            "a stored chunk has no text or no metadata", hint="rebuild with `rag ingest`"
        )
    try:
        created = metadata.get("created")
        created_from = metadata.get("created_from")
        info = DocumentInfo(
            doc_id=_get(metadata, "doc_id", str),
            name=_get(metadata, "doc_name", str),
            kind=DocumentKind(_get(metadata, "kind", str)),
            created=dt.date.fromisoformat(created) if isinstance(created, str) else None,
            created_from=DateSource(created_from) if isinstance(created_from, str) else None,
        )
        page = metadata.get("page")
        section = metadata.get("section")
        return Chunk(
            doc=info,
            ordinal=_get(metadata, "ordinal", int),
            text=text,
            page=page if isinstance(page, int) else None,
            section=section if isinstance(section, str) else None,
        )
    except ValueError as exc:
        raise VectorStoreError(
            f"a stored chunk has unreadable metadata: {exc}", hint="rebuild with `rag ingest`"
        ) from exc


def _get[T](metadata: Mapping[str, object], key: str, kind: type[T]) -> T:
    value = metadata.get(key)
    if not isinstance(value, kind):
        raise ValueError(f"{key} should be {kind.__name__}, got {value!r}")
    return value

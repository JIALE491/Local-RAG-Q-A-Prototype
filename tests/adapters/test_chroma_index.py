"""ChromaIndex: the VectorIndex contract on disk, plus what only a disk-backed index must do.

The restart tests build or break the index in a separate Python process, so this process
opens the folder fresh, exactly as the next `rag` command would.
"""

import json
import logging
import os
import re
import stat
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import chromadb
import pytest
from chromadb.api.types import Metadata

from rag.adapters.chroma_index import BUILD_PREFIX, POINTER, SETTINGS, ChromaIndex
from rag.domain.errors import IndexNotBuiltError, InvalidDataError, VectorStoreError
from rag.domain.ports import VectorIndex
from tests.builders import chunk
from tests.contracts.vector_index import MANIFEST, VectorIndexContract, build

REPO = Path(__file__).resolve().parents[2]
STRATEGY = "fixed-300-50"
VALID_METADATA: Metadata = {"doc_id": "x.md", "doc_name": "x.md", "kind": "markdown", "ordinal": 0}
ONE_VECTOR: list[Sequence[float] | Sequence[int]] = [[0.0, 1.0]]

needs_permissions = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX file permissions and a non-root user",
)


class TestChromaIndex(VectorIndexContract):
    @pytest.fixture
    def index(self, tmp_path: Path) -> VectorIndex:
        return ChromaIndex(path=tmp_path / "index", collection="test-index")


def in_another_process(script: str, root: Path) -> subprocess.CompletedProcess[str]:
    """Run `script` in a fresh Python process, with `index` opened on `root`."""
    prelude = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from rag.adapters.chroma_index import ChromaIndex\n"
        "from tests.builders import chunk\n"
        "from tests.contracts.vector_index import build\n"
        f"index = ChromaIndex(path=Path(sys.argv[1]), collection={STRATEGY!r})\n"
    )
    return subprocess.run(  # noqa: S603 - this interpreter, running a fixed script
        [sys.executable, "-c", prelude + script, str(root)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def builds(root: Path, strategy: str = STRATEGY) -> list[str]:
    folder = root / strategy
    return sorted(
        p.name for p in folder.iterdir() if p.is_dir() and p.name.startswith(BUILD_PREFIX)
    )


def built(tmp_path: Path) -> ChromaIndex:
    index = ChromaIndex(path=tmp_path, collection=STRATEGY)
    build(index, [chunk()], [[1.0, 0.0]])
    return index


# Restarts and crashes


def test_the_index_survives_a_restart(tmp_path: Path) -> None:
    script = (
        "build(index, [chunk('Refunds take 5 days.', doc='refunds.pdf', page=2, section='Refunds'),"
        " chunk('Vacation is 15 days.', doc='handbook.md', section='Vacation')],"
        " [[1.0, 0.0], [0.0, 1.0]])\n"
    )
    result = in_another_process(script, tmp_path)
    assert result.returncode == 0, result.stderr
    restarted = ChromaIndex(path=tmp_path, collection=STRATEGY)
    assert restarted.manifest() == MANIFEST
    assert restarted.count() == 2
    [results] = restarted.query([[0.0, 1.0]], k=1)
    assert results[0].chunk == chunk("Vacation is 15 days.", doc="handbook.md", section="Vacation")


def test_a_crash_part_way_through_a_rebuild_leaves_the_index_refusing_searches(
    tmp_path: Path,
) -> None:
    script = (
        "build(index, [chunk(ordinal=0)], [[1.0, 0.0]])\n"
        "index.begin_rebuild()\n"
        "index.add([chunk(ordinal=0), chunk(ordinal=1)], [[1.0, 0.0], [0.0, 1.0]])\n"
        "os._exit(3)\n"
    )
    result = in_another_process(script, tmp_path)
    assert result.returncode == 3, result.stderr
    restarted = ChromaIndex(path=tmp_path, collection=STRATEGY)
    assert restarted.manifest() is None
    with pytest.raises(IndexNotBuiltError, match="did not finish"):
        restarted.query([[1.0, 0.0]], k=1)
    with pytest.raises(IndexNotBuiltError, match="did not finish"):
        restarted.chunks()
    build(restarted, [chunk()], [[1.0, 0.0]])
    assert restarted.count() == 1
    assert len(builds(tmp_path)) == 1, "the next rebuild removes the crashed build"


# The folder layout


def test_a_finished_rebuild_names_its_build_in_current_json(tmp_path: Path) -> None:
    built(tmp_path)
    folder = tmp_path / STRATEGY
    assert json.loads((folder / POINTER).read_text(encoding="utf-8")) == {
        "build": builds(tmp_path)[0],
        "strategy": MANIFEST.strategy,
        "embed_model": MANIFEST.embed_model,
    }
    assert [p.name for p in folder.iterdir() if p.is_file()] == [POINTER], "no temporary file"


def test_rebuilding_keeps_only_the_newest_build_of_that_strategy(tmp_path: Path) -> None:
    other = ChromaIndex(path=tmp_path, collection="fixed-800-100")
    build(other, [chunk()], [[1.0, 0.0]])
    index = ChromaIndex(path=tmp_path, collection=STRATEGY)
    for _ in range(3):
        build(index, [chunk(ordinal=0), chunk(ordinal=1)], [[1.0, 0.0], [0.0, 1.0]])
    assert len(builds(tmp_path)) == 1
    assert len(builds(tmp_path, "fixed-800-100")) == 1
    assert (index.count(), other.count()) == (2, 1)


def test_a_rebuild_deletes_only_its_own_build_folders(tmp_path: Path) -> None:
    index = built(tmp_path)
    folder = tmp_path / STRATEGY
    (folder / "notes.txt").write_text("keep me", encoding="utf-8")
    (folder / "backup").mkdir()
    build(index, [chunk()], [[1.0, 0.0]])
    assert (folder / "notes.txt").exists()
    assert (folder / "backup").is_dir()


def test_leftover_temporary_pointer_files_are_ignored_then_removed(tmp_path: Path) -> None:
    index = built(tmp_path)
    leftover = tmp_path / STRATEGY / f".{POINTER}.dead.tmp"
    leftover.write_text("{", encoding="utf-8")
    assert index.manifest() == MANIFEST
    build(index, [chunk()], [[1.0, 0.0]])
    assert not leftover.exists()


def test_reading_an_index_that_does_not_exist_creates_nothing(tmp_path: Path) -> None:
    index = ChromaIndex(path=tmp_path / "index", collection=STRATEGY)
    assert (index.count(), index.manifest()) == (0, None)
    with pytest.raises(IndexNotBuiltError, match="no index for 'fixed-300-50' yet"):
        index.query([[1.0, 0.0]], k=1)
    assert not (tmp_path / "index").exists()


def test_telemetry_is_off() -> None:
    assert SETTINGS.anonymized_telemetry is False


# Error paths


@pytest.mark.parametrize("name", ["ab", "../escape", "has space", "-leading"])
def test_a_strategy_name_that_cannot_be_a_folder_is_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(InvalidDataError, match="cannot name an index"):
        ChromaIndex(path=tmp_path, collection=name)


def test_an_index_path_that_is_a_file_is_an_error(tmp_path: Path) -> None:
    blocker = tmp_path / "index"
    blocker.write_text("not a folder", encoding="utf-8")
    with pytest.raises(VectorStoreError, match="cannot prepare"):
        ChromaIndex(path=blocker, collection=STRATEGY).begin_rebuild()


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ("{not json", "is damaged"),
        ('["a", "list"]', "it is not a JSON object"),
        ('{"build": "", "strategy": "s", "embed_model": "m"}', "'build' should be a non-empty"),
        ('{"build": "build-x", "strategy": "s"}', "'embed_model' should be a non-empty"),
        ('{"build": "../elsewhere", "strategy": "s", "embed_model": "m"}', "not a build folder"),
        ('{"build": "build-gone", "strategy": "s", "embed_model": "m"}', "does not exist"),
    ],
)
def test_a_damaged_current_json_is_reported_with_a_hint(
    tmp_path: Path, content: str, problem: str
) -> None:
    index = built(tmp_path)
    (tmp_path / STRATEGY / POINTER).write_text(content, encoding="utf-8")
    with pytest.raises(VectorStoreError, match=re.escape(problem)) as caught:
        index.query([[1.0, 0.0]], k=1)
    assert caught.value.hint == "run `rag ingest` to rebuild"


def test_a_build_folder_without_a_database_is_an_error(tmp_path: Path) -> None:
    index = built(tmp_path)
    (tmp_path / STRATEGY / "build-empty").mkdir()
    pointer = {"build": "build-empty", "strategy": "s", "embed_model": "m"}
    (tmp_path / STRATEGY / POINTER).write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(VectorStoreError, match="could not open the build build-empty"):
        index.query([[1.0, 0.0]], k=1)


def stored_collection(tmp_path: Path) -> chromadb.Collection:
    """The live build's chromadb collection, opened directly to damage what it stores."""
    client = chromadb.PersistentClient(
        path=str(tmp_path / STRATEGY / builds(tmp_path)[0]), settings=SETTINGS
    )
    return client.get_collection(STRATEGY, embedding_function=None)


@pytest.mark.parametrize(
    ("metadata", "problem"),
    [
        ({"doc_id": "x.md", "kind": "markdown", "ordinal": 0}, "doc_name should be str"),
        ({**VALID_METADATA, "kind": "docx"}, "'docx' is not a valid DocumentKind"),
        ({**VALID_METADATA, "ordinal": "first"}, "ordinal should be int"),
        ({**VALID_METADATA, "created": "soon", "created_from": "front_matter"}, "soon"),
    ],
)
def test_a_stored_chunk_with_unreadable_metadata_is_an_error(
    tmp_path: Path, metadata: Metadata, problem: str
) -> None:
    index = built(tmp_path)
    stored_collection(tmp_path).add(
        ids=["damaged"], embeddings=ONE_VECTOR, documents=["Damaged."], metadatas=[metadata]
    )
    with pytest.raises(VectorStoreError, match="unreadable metadata") as caught:
        index.chunks()
    assert problem in caught.value.message
    assert caught.value.hint == "rebuild with `rag ingest`"


def test_a_stored_chunk_without_text_is_an_error(tmp_path: Path) -> None:
    index = built(tmp_path)
    stored_collection(tmp_path).add(
        ids=["no-text"], embeddings=ONE_VECTOR, metadatas=[VALID_METADATA]
    )
    with pytest.raises(VectorStoreError, match="no text"):
        index.query([[0.0, 1.0]], k=1)


@needs_permissions
def test_an_old_build_that_cannot_be_deleted_is_reported_and_removed_later(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    index = built(tmp_path)
    [old] = builds(tmp_path)
    old_folder = tmp_path / STRATEGY / old
    old_folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with caplog.at_level(logging.WARNING, logger="rag.adapters.chroma_index"):
            build(index, [chunk()], [[0.0, 1.0]])
        assert "could not remove the old index files" in caplog.text
        assert old in builds(tmp_path)
        [results] = index.query([[0.0, 1.0]], k=1)
        assert len(results) == 1
    finally:
        old_folder.chmod(stat.S_IRWXU)
    build(index, [chunk()], [[1.0, 0.0]])
    assert old not in builds(tmp_path)
    assert len(builds(tmp_path)) == 1


@needs_permissions
def test_failing_to_record_a_finished_rebuild_leaves_the_index_unsearchable(
    tmp_path: Path,
) -> None:
    index = ChromaIndex(path=tmp_path, collection=STRATEGY)
    index.begin_rebuild()
    index.add([chunk()], [[1.0, 0.0]])
    folder = tmp_path / STRATEGY
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(VectorStoreError, match="cannot record the finished rebuild"):
            index.finish_rebuild(MANIFEST)
    finally:
        folder.chmod(stat.S_IRWXU)
    assert index.manifest() is None
    assert [p.name for p in folder.iterdir() if p.is_file()] == []


@needs_permissions
def test_an_unreadable_current_json_is_an_error(tmp_path: Path) -> None:
    index = built(tmp_path)
    pointer = tmp_path / STRATEGY / POINTER
    pointer.chmod(0)
    try:
        with pytest.raises(VectorStoreError, match="cannot read"):
            index.manifest()
    finally:
        pointer.chmod(stat.S_IRUSR | stat.S_IWUSR)


@needs_permissions
def test_a_strategy_folder_that_cannot_be_listed_stops_the_rebuild(tmp_path: Path) -> None:
    index = built(tmp_path)
    folder = tmp_path / STRATEGY
    folder.chmod(stat.S_IWUSR | stat.S_IXUSR)
    try:
        with pytest.raises(VectorStoreError, match="cannot list"):
            index.begin_rebuild()
    finally:
        folder.chmod(stat.S_IRWXU)

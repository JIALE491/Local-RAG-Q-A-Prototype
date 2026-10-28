"""config.toml loading: strict, with errors that name the bad key."""

import copy
from pathlib import Path

import pytest

from rag.config import load_config, parse_config
from rag.domain.errors import ConfigError

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.toml"

VALID: dict[str, dict[str, object]] = {
    "paths": {
        "docs_dir": "data/docs",
        "index_dir": "index",
        "questions_csv": "data/eval/questions.csv",
        "facts_csv": "data/eval/facts.csv",
        "results_dir": "results",
    },
    "ollama": {
        "host": "http://127.0.0.1:11434",
        "embed_model": "nomic-embed-text",
        "embed_document_prefix": "search_document: ",
        "embed_query_prefix": "search_query: ",
        "chat_model": "llama3.2:3b",
        "chat_context_tokens": 4096,
        "chat_max_answer_tokens": 512,
        "chat_temperature": 0,
        "timeout_s": 120,
        "keep_alive": "10m",
    },
    "ingest": {"batch_size": 32},
    "chunking": {
        "strategy": "sections",
        "fixed": {"size": 300, "overlap": 50},
        "sentences": {"size": 300},
        "sections": {"max_size": 4000},
    },
    "retrieval": {"top_k": 4, "eval_k": 3},
}


def with_fixed(key: str, value: object) -> dict[str, dict[str, object]]:
    raw = copy.deepcopy(VALID)
    fixed = raw["chunking"]["fixed"]
    assert isinstance(fixed, dict)
    fixed[key] = value
    return raw


def with_value(table: str, key: str, value: object) -> dict[str, dict[str, object]]:
    raw = copy.deepcopy(VALID)
    raw[table][key] = value
    return raw


def test_the_repository_config_is_valid() -> None:
    config = load_config(REPO_CONFIG)
    assert config.ollama.embed_model == "nomic-embed-text"
    assert config.retrieval.eval_k == 3
    assert config.paths.docs_dir == REPO_CONFIG.parent / "data" / "docs"


def test_relative_paths_resolve_against_the_config_folder(tmp_path: Path) -> None:
    config = parse_config(VALID, base_dir=tmp_path)
    assert config.paths.index_dir == tmp_path.resolve() / "index"


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    config = parse_config(with_value("paths", "index_dir", str(elsewhere)), base_dir=Path("/x"))
    assert config.paths.index_dir == elsewhere.resolve()


def test_missing_file_raises_with_a_hint(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found") as caught:
        load_config(tmp_path / "missing.toml")
    assert caught.value.hint is not None


def test_invalid_toml_raises(tmp_path: Path) -> None:
    broken = tmp_path / "config.toml"
    broken.write_text("[paths\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(broken)


def test_unreadable_file_raises(tmp_path: Path) -> None:
    not_utf8 = tmp_path / "config.toml"
    not_utf8.write_bytes(b"\xff\xfe[paths]")
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(not_utf8)


def test_missing_table_is_named() -> None:
    raw = copy.deepcopy(VALID)
    del raw["ingest"]
    with pytest.raises(ConfigError, match=r"missing setting \[ingest\]"):
        parse_config(raw, base_dir=Path())


def test_missing_key_is_named() -> None:
    raw = copy.deepcopy(VALID)
    del raw["ollama"]["chat_model"]
    with pytest.raises(ConfigError, match=r"missing setting \[ollama\] chat_model"):
        parse_config(raw, base_dir=Path())


def test_misspelled_key_suggests_the_right_one() -> None:
    raw = copy.deepcopy(VALID)
    raw["retrieval"]["top-k"] = raw["retrieval"].pop("top_k")
    with pytest.raises(ConfigError, match=r"unknown setting \[retrieval\] top-k") as caught:
        parse_config(raw, base_dir=Path())
    assert caught.value.hint == "did you mean 'top_k'?"


def test_misspelled_table_suggests_the_right_one() -> None:
    raw = copy.deepcopy(VALID)
    raw["retreival"] = raw.pop("retrieval")
    with pytest.raises(ConfigError, match=r"unknown setting \[retreival\]") as caught:
        parse_config(raw, base_dir=Path())
    assert caught.value.hint == "did you mean 'retrieval'?"


def test_a_table_must_be_a_table() -> None:
    raw: dict[str, object] = {**VALID, "ingest": 32}
    with pytest.raises(ConfigError, match=r"\[ingest\] must be a table"):
        parse_config(raw, base_dir=Path())


@pytest.mark.parametrize(
    ("table", "key", "value", "problem"),
    [
        ("retrieval", "top_k", 0, "between 1 and 20"),
        ("retrieval", "top_k", "4", "whole number"),
        ("retrieval", "eval_k", True, "whole number"),
        ("ingest", "batch_size", 2.5, "whole number"),
        ("ollama", "timeout_s", 0, "more than 0"),
        ("ollama", "timeout_s", "fast", "must be a number"),
        ("ollama", "host", "127.0.0.1:11434", "must look like"),
        ("ollama", "keep_alive", "ten minutes", "must look like"),
        ("ollama", "embed_model", "  ", "non-empty string"),
        ("ollama", "chat_context_tokens", 256, "between 512 and 131072"),
        ("ollama", "chat_max_answer_tokens", 8, "between 16 and 8192"),
        ("ollama", "chat_temperature", 2.5, "between 0 and 2"),
        ("ollama", "chat_temperature", "hot", "must be a number"),
        ("paths", "docs_dir", 42, "must be a string"),
        ("paths", "facts_csv", "", "non-empty string"),
    ],
)
def test_invalid_values_name_the_key(table: str, key: str, value: object, problem: str) -> None:
    with pytest.raises(ConfigError) as caught:
        parse_config(with_value(table, key, value), base_dir=Path())
    assert f"[{table}] {key}" in caught.value.message
    assert problem in caught.value.message


def test_the_answer_limit_must_leave_room_for_the_prompt() -> None:
    raw = with_value("ollama", "chat_max_answer_tokens", 4096)
    with pytest.raises(ConfigError, match="must be smaller than chat_context_tokens"):
        parse_config(raw, base_dir=Path())


def test_the_repository_config_uses_strategy_three_and_a_4096_token_context() -> None:
    config = load_config(REPO_CONFIG)
    assert (config.chunking.strategy, config.chunking.sections.max_size) == ("sections", 4000)
    assert (config.ollama.chat_context_tokens, config.ollama.chat_temperature) == (4096, 0.0)


def test_nested_table_errors_name_the_full_path() -> None:
    with pytest.raises(ConfigError, match=r"\[chunking\.fixed\] size must be between 50 and 4000"):
        parse_config(with_fixed("size", 10), base_dir=Path())


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ConfigError, match=r"\[chunking\.fixed\] overlap must be smaller than size"):
        parse_config(with_fixed("overlap", 300), base_dir=Path())


def test_strategy_must_be_a_known_one() -> None:
    with pytest.raises(ConfigError, match=r"\[chunking\] strategy must be one of fixed"):
        parse_config(with_value("chunking", "strategy", "semantic"), base_dir=Path())


def test_embedding_prefixes_may_be_empty_but_must_be_strings() -> None:
    raw = with_value("ollama", "embed_document_prefix", "")
    raw["ollama"]["embed_query_prefix"] = ""
    config = parse_config(raw, base_dir=Path())
    assert (config.ollama.embed_document_prefix, config.ollama.embed_query_prefix) == ("", "")
    with pytest.raises(ConfigError, match=r"\[ollama\] embed_query_prefix must be a string"):
        parse_config(with_value("ollama", "embed_query_prefix", 5), base_dir=Path())


@pytest.mark.parametrize("keep_alive", ["10m", "1h30m", "45s", "500ms", "0", "-1"])
def test_ollama_keep_alive_formats_are_accepted(keep_alive: str) -> None:
    config = parse_config(with_value("ollama", "keep_alive", keep_alive), base_dir=Path())
    assert config.ollama.keep_alive == keep_alive


def test_every_strategy_has_its_settings() -> None:
    chunking = load_config(REPO_CONFIG).chunking
    assert (chunking.sentences.size, chunking.sections.max_size) == (300, 4000)


@pytest.mark.parametrize(
    ("table", "key", "value", "problem"),
    [
        ("sentences", "size", 10, r"\[chunking\.sentences\] size must be between 50 and 4000"),
        ("sections", "max_size", 9000, r"\[chunking\.sections\] max_size must be between 200"),
    ],
)
def test_strategy_sizes_are_range_checked(table: str, key: str, value: int, problem: str) -> None:
    raw = copy.deepcopy(VALID)
    settings = raw["chunking"][table]
    assert isinstance(settings, dict)
    settings[key] = value
    with pytest.raises(ConfigError, match=problem):
        parse_config(raw, base_dir=Path())

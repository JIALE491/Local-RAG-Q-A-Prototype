"""Settings from config.toml.

Strict on purpose: every key is required, unknown keys are rejected so that a typo cannot
silently fall back to a default, and every value is type- and range-checked on load.
Relative paths are resolved against the folder that contains the config file.
"""

import difflib
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from rag.domain.errors import ConfigError

_HOST = re.compile(r"https?://\S+")
_KEEP_ALIVE = re.compile(r"-1|0|(?:\d+(?:ms|s|m|h))+")


@dataclass(frozen=True, slots=True, kw_only=True)
class PathsConfig:
    docs_dir: Path
    index_dir: Path
    questions_csv: Path
    results_dir: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class OllamaConfig:
    host: str
    embed_model: str
    chat_model: str
    timeout_s: float
    keep_alive: str


@dataclass(frozen=True, slots=True, kw_only=True)
class IngestConfig:
    batch_size: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalConfig:
    top_k: int
    """Chunks given to the chat model."""
    eval_k: int
    """Cut-off for the retrieval hit rate."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AppConfig:
    paths: PathsConfig
    ollama: OllamaConfig
    ingest: IngestConfig
    retrieval: RetrievalConfig


def load_config(path: Path) -> AppConfig:
    """Read and validate a config file. Raises ConfigError that names the bad key."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(
            f"config file not found: {path}",
            hint="run from the project folder, or pass --config PATH",
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    return parse_config(raw, base_dir=path.resolve().parent)


def parse_config(raw: Mapping[str, object], *, base_dir: Path) -> AppConfig:
    """Validate an already-parsed config. Relative paths resolve against base_dir."""
    root = _Table(raw, name="", keys=("paths", "ollama", "ingest", "retrieval"))

    paths = root.table("paths", keys=("docs_dir", "index_dir", "questions_csv", "results_dir"))
    ollama = root.table(
        "ollama", keys=("host", "embed_model", "chat_model", "timeout_s", "keep_alive")
    )
    ingest = root.table("ingest", keys=("batch_size",))
    retrieval = root.table("retrieval", keys=("top_k", "eval_k"))

    return AppConfig(
        paths=PathsConfig(
            docs_dir=paths.path("docs_dir", base_dir=base_dir),
            index_dir=paths.path("index_dir", base_dir=base_dir),
            questions_csv=paths.path("questions_csv", base_dir=base_dir),
            results_dir=paths.path("results_dir", base_dir=base_dir),
        ),
        ollama=OllamaConfig(
            host=ollama.matching("host", _HOST, example="http://127.0.0.1:11434"),
            embed_model=ollama.text("embed_model"),
            chat_model=ollama.text("chat_model"),
            timeout_s=ollama.number("timeout_s", above=0, at_most=3600),
            keep_alive=ollama.matching("keep_alive", _KEEP_ALIVE, example='"10m", "0" or "-1"'),
        ),
        ingest=IngestConfig(batch_size=ingest.integer("batch_size", low=1, high=1024)),
        retrieval=RetrievalConfig(
            top_k=retrieval.integer("top_k", low=1, high=20),
            eval_k=retrieval.integer("eval_k", low=1, high=20),
        ),
    )


class _Table:
    """One [table] of the config, checked for unknown and missing keys up front.

    The typed getters raise ConfigError naming the table and key, e.g. "[retrieval] top_k".
    """

    def __init__(self, raw: Mapping[str, object], *, name: str, keys: Iterable[str]) -> None:
        self._raw = raw
        self._name = name
        expected = set(keys)
        unknown = sorted(set(raw) - expected)
        if unknown:
            close = difflib.get_close_matches(unknown[0], expected, n=1)
            hint = (
                f"did you mean {close[0]!r}?"
                if close
                else f"expected: {', '.join(sorted(expected))}"
            )
            raise ConfigError(f"unknown setting {self._where(unknown[0])}", hint=hint)
        missing = sorted(expected - set(raw))
        if missing:
            raise ConfigError(f"missing setting {self._where(missing[0])}")

    def table(self, key: str, *, keys: Iterable[str]) -> "_Table":
        value = self._raw[key]
        if not isinstance(value, dict):
            raise self._error(key, f"must be a table like [{key}], got {value!r}")
        return _Table(value, name=key, keys=keys)

    def text(self, key: str) -> str:
        value = self._raw[key]
        if not isinstance(value, str) or not value.strip():
            raise self._error(key, f"must be a non-empty string, got {value!r}")
        return value

    def matching(self, key: str, pattern: re.Pattern[str], *, example: str) -> str:
        value = self.text(key)
        if not pattern.fullmatch(value):
            raise self._error(key, f"must look like {example}, got {value!r}")
        return value

    def integer(self, key: str, *, low: int, high: int) -> int:
        value = self._raw[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._error(key, f"must be a whole number, got {value!r}")
        if not low <= value <= high:
            raise self._error(key, f"must be between {low} and {high}, got {value}")
        return value

    def number(self, key: str, *, above: float, at_most: float) -> float:
        value = self._raw[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._error(key, f"must be a number, got {value!r}")
        if not above < value <= at_most:
            raise self._error(key, f"must be more than {above} and at most {at_most}, got {value}")
        return float(value)

    def path(self, key: str, *, base_dir: Path) -> Path:
        return (base_dir / Path(self.text(key)).expanduser()).resolve()

    def _where(self, key: str) -> str:
        return f"[{self._name}] {key}" if self._name else f"[{key}]"

    def _error(self, key: str, problem: str) -> ConfigError:
        return ConfigError(f"{self._where(key)} {problem}")

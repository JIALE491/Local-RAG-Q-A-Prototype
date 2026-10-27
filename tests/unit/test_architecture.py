"""The dependency rule: business logic never imports infrastructure.

    domain  <-  chunking, services  <-  adapters  <-  cli (wires everything together)

Inner layers may import only the standard library and the layers inside them, so they can
be tested with fakes and never need Ollama, chromadb or pypdf. A new top-level module fails
test_every_module_has_a_rule until it is given one here.
"""

import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rag"


@dataclass(frozen=True)
class Rule:
    rag_modules: tuple[str, ...]
    """Project modules this layer may import (prefix match)."""
    third_party: bool
    """Whether non-standard-library imports are allowed."""


RULES = {
    "domain": Rule(("rag.domain",), third_party=False),
    "chunking": Rule(("rag.domain", "rag.chunking"), third_party=False),
    "services": Rule(("rag.domain", "rag.services"), third_party=False),
    "adapters": Rule(("rag.domain", "rag.adapters"), third_party=True),
    "config.py": Rule(("rag.domain",), third_party=False),
    "log.py": Rule((), third_party=False),
    "__init__.py": Rule((), third_party=False),
    "cli.py": Rule(("rag",), third_party=True),
}

MODULES = sorted(PACKAGE.rglob("*.py"))


def layer(path: Path) -> str:
    return path.relative_to(PACKAGE).parts[0]


def imports(path: Path) -> Iterator[tuple[int, str]]:
    """(line, module) for every import in the file; relative imports come back as '.'."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from ((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.lineno, "." if node.level or node.module is None else node.module


def allowed(module: str, rule: Rule) -> bool:
    top = module.split(".")[0]
    if top == "rag":
        return any(module == m or module.startswith(f"{m}.") for m in rule.rag_modules)
    if top in sys.stdlib_module_names:
        return True
    return rule.third_party and top != ""


def test_every_module_has_a_rule() -> None:
    assert MODULES, f"no modules found under {PACKAGE}"
    assert {layer(path) for path in MODULES} <= RULES.keys()


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.relative_to(PACKAGE).as_posix())
def test_module_respects_the_dependency_rule(path: Path) -> None:
    rule = RULES[layer(path)]
    violations = [
        f"line {line}: {module}" for line, module in imports(path) if not allowed(module, rule)
    ]
    assert not violations, f"{layer(path)} may not import: {violations}"

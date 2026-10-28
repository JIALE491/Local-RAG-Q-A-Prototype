# Common tasks. `make check` runs everything a commit must pass.
PY := .venv/bin/python

.PHONY: install lint format typecheck test coverage test-integration pdfs check

install:  # package plus dev tools, editable, into .venv
	$(PY) -m pip install -e ".[dev]"

lint:  # lint rules and formatting, without changing files
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  # apply formatting and safe lint fixes
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck:
	$(PY) -m mypy

test:  # unit tests; no Ollama needed
	$(PY) -m pytest

coverage:  # unit tests with branch coverage (slower, about 15 s)
	$(PY) -m pytest --cov --cov-report=term-missing

test-integration:  # tests against the real Ollama and chromadb
	$(PY) -m pytest -m integration

check: lint typecheck test

pdfs:  # rebuild data/docs/*.pdf from data/pdf-sources/
	$(PY) scripts/build_pdfs.py

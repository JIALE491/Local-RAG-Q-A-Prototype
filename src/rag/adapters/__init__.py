"""Implementations of the ports on real infrastructure: files, pypdf, Ollama and chromadb.

The only package allowed to import third-party libraries. Each adapter translates its
library's exceptions into the errors in rag.domain.errors, and must pass the same contract
tests (tests/contracts/) as the in-memory fakes.
"""

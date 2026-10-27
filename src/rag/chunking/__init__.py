"""Chunking strategies: pure functions from a Document to Chunks.

Each strategy implements rag.domain.ports.ChunkingStrategy, and all of them are compared on
the same test set. May import only rag.domain and the standard library; this is enforced by
tests/unit/test_architecture.py.
"""

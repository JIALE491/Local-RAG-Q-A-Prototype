"""Use cases: ingest, retrieve, evaluate and answer.

They orchestrate the ports in rag.domain.ports and do no I/O of their own, so they are
unit-tested with the fakes in tests/fakes.py. May import only rag.domain and the standard
library; this is enforced by tests/unit/test_architecture.py.
"""

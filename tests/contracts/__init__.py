"""Behaviour every implementation of a port must have.

Each module defines a *Contract class. A test module subclasses it and supplies the
implementation through a fixture: the fakes in tests/unit/test_fakes.py, and later the real
adapters as integration tests. Running the same tests against both keeps the fakes honest.
"""

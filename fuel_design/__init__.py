"""SCULPTOR entry package: sets up distributed code paths and the CLI entry."""
from .bootstrap import add_code_paths

add_code_paths()

from graph import HalfGraph, NodeKind  # noqa: E402,F401
from symmetry import mirror_complete  # noqa: E402,F401

__all__ = ["HalfGraph", "NodeKind", "mirror_complete"]

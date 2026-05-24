"""Read-only access to external research dossiers under knowledge/."""

from lbg.knowledge.factors import get_dossier, load_index, search

__all__ = ["load_index", "search", "get_dossier"]

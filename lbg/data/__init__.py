"""Data layer: fetch, cross-check, persist, and serve SPY daily OHLCV.

LLM agents must NEVER import from this package directly. The Orchestrator
pulls the splits it needs and feeds derived features (never raw OHLCV) into
agent-facing context. See PROPOSAL.html §5, §6.
"""

"""Stable ABI export; evolved strategies are injected dynamically."""
try:
    from runtime_bases import ThinkingBase
except ImportError:  # pragma: no cover - package import
    from evolution.runtime_bases import ThinkingBase

__all__ = ["ThinkingBase"]

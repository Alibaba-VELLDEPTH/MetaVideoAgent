"""Stable ABI export; evolved strategies are injected dynamically."""
try:
    from runtime_bases import VideoStructuringBase
except ImportError:  # pragma: no cover - package import
    from evolution.runtime_bases import VideoStructuringBase

__all__ = ["VideoStructuringBase"]

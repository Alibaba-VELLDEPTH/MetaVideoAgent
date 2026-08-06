"""Stable ABI export; evolved strategies are injected dynamically."""
try:
    from runtime_bases import WorkMemoryBase
except ImportError:  # pragma: no cover - package import
    from evolution.runtime_bases import WorkMemoryBase

__all__ = ["WorkMemoryBase"]

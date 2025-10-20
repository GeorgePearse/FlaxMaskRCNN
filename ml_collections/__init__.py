"""Minimal stub of the ``ml_collections`` package for local testing.

This provides a lightweight drop-in replacement for :class:`ConfigDict`
covering the subset of behaviour required by the Detectax training loop. It
offers attribute-style access in addition to dictionary semantics and
recursively wraps nested dictionaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["ConfigDict"]


class ConfigDict(dict):
    """Simplified ConfigDict supporting both dict and attribute access."""

    def __init__(
        self,
        initial: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        data: dict[str, Any] = {}
        if initial is not None:
            if isinstance(initial, Mapping):
                data.update(initial.items())
            else:
                for key, value in initial:
                    data[key] = value
        data.update(kwargs)
        super().__init__()
        for key, value in data.items():
            self[key] = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._to_configdict(value))

    def setdefault(self, key: str, default: Any | None = None) -> Any:
        return super().setdefault(key, self._to_configdict(default))

    @staticmethod
    def _to_configdict(value: Any) -> Any:
        if isinstance(value, ConfigDict):
            return value
        if isinstance(value, dict):
            return ConfigDict(value)
        return value

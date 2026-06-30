"""Bounded LRU dict: a fixed-capacity OrderedDict that evicts the oldest entry."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Generic, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")


class BoundedDict(Generic[_K, _V]):
    """Thread-unsafe LRU dict capped at *maxsize* entries.

    On insertion when full, the least-recently-used (oldest-inserted) key is
    dropped. Reads do NOT promote entries (access-order LRU requires a lock
    or explicit ``touch``; this is write-order eviction, suitable for cooldown
    caches where insertion time is what matters).
    """

    __slots__ = ("_maxsize", "_data")

    def __init__(self, maxsize: int = 1024) -> None:
        self._maxsize = max(1, maxsize)
        self._data: OrderedDict[_K, _V] = OrderedDict()

    def __setitem__(self, key: _K, value: _V) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __getitem__(self, key: _K) -> _V:
        return self._data[key]

    def get(self, key: _K, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"BoundedDict(maxsize={self._maxsize}, len={len(self._data)})"

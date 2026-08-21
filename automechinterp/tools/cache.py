"""Content-addressed result cache (Sec. 3.6). Stands in for the proposal's
Redis-backed cache: same interface (get/set keyed by a content hash), swap
the backing dict for a Redis client to scale to real parallel Layer Agents.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable


class ContentCache:
    def __init__(self):
        self._store: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(tool_name: str, args: dict) -> str:
        payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get_or_compute(self, tool_name: str, args: dict, compute: Callable[[], Any]) -> Any:
        key = self._key(tool_name, args)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        result = compute()
        self._store[key] = result
        return result

    def stats(self) -> dict:
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": rate}


CACHE = ContentCache()

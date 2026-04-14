"""
Rain ⛈️ — MemPalace Integration

Wraps MemPalace (pip install mempalace) to give Rain a ChromaDB-backed
semantic store alongside its SQLite Tier 3.  MemPalace's spatial hierarchy
(Wings → Rooms) organises Rain conversations by agent type, delivering
metadata-filtered retrieval that outperforms flat vector search.

Wing/Room structure
-------------------
  wing = "rain"
    room = "dev"       — DEV agent exchanges (code, debugging)
    room = "logic"     — LOGIC / DOMAIN agent exchanges (reasoning)
    room = "general"   — GENERAL / SYNTHESIZER exchanges
    room = "search"    — SEARCH agent exchanges (live data)
    room = "facts"     — persistent user profile facts

Graceful degradation
--------------------
If mempalace is not installed or ChromaDB can't initialise, _MP_AVAILABLE is
False and every method returns an empty result silently — identical behaviour
to other optional Rain components (_KG_AVAILABLE, _VISION_AVAILABLE).
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

_MP_AVAILABLE = False
try:
    import chromadb
    from mempalace.config import MempalaceConfig
    _MP_AVAILABLE = True
except ImportError:
    pass

RAIN_WING = "rain"

_AGENT_ROOM: Dict[str, str] = {
    "DEV":         "dev",
    "LOGIC":       "logic",
    "DOMAIN":      "logic",
    "GENERAL":     "general",
    "SEARCH":      "search",
    "SYNTHESIZER": "general",
    "REFLECTION":  "general",
}


def _room_for(agent_type: str) -> str:
    return _AGENT_ROOM.get(agent_type.upper() if agent_type else "", "general")


class MemPalaceAdapter:
    """
    Thin adapter — stores Rain Q&A exchanges as verbatim drawers in a MemPalace
    palace and provides semantic search over them.

    Thread-safe: all writes happen in daemon threads; reads are synchronous.
    """

    def __init__(self):
        self._available = False
        self._col = None
        self._palace_path: str = ""

        if not _MP_AVAILABLE:
            return

        try:
            cfg = MempalaceConfig()
            self._palace_path = cfg.palace_path
            Path(self._palace_path).mkdir(parents=True, exist_ok=True)

            client = chromadb.PersistentClient(path=self._palace_path)
            self._col = client.get_or_create_collection(cfg.collection_name)
            self._available = True
        except Exception:
            pass

    # ── Public interface ───────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    def store_exchange(self, query: str, response: str, agent_type: str = "GENERAL") -> None:
        """
        Persist a Q&A pair as a verbatim drawer.

        Runs in a background daemon thread — never blocks the response path.
        Silently skips duplicates (same content within a 0.95 distance threshold).
        """
        if not self._available:
            return
        threading.Thread(
            target=self._store_exchange_sync,
            args=(query, response, agent_type),
            daemon=True,
        ).start()

    def search(
        self,
        query: str,
        wing: str = RAIN_WING,
        room: str = None,
        n_results: int = 5,
        min_similarity: float = 0.35,
    ) -> List[Dict]:
        """
        Semantic search over palace drawers.

        Returns a list of dicts:
          {"text": str, "wing": str, "room": str, "similarity": float}

        Only results with similarity >= min_similarity are returned.
        Returns [] on any failure.
        """
        if not self._available or not self._col:
            return []

        try:
            count = self._col.count()
            if count == 0:
                return []

            where = self._build_where(wing, room)
            kwargs: Dict = {
                "query_texts": [query],
                "n_results": min(n_results, count),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            results = self._col.query(**kwargs)
        except Exception:
            return []

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sim = round(1.0 - dist, 3)
            if sim >= min_similarity:
                hits.append({
                    "text":       doc,
                    "wing":       meta.get("wing", "unknown"),
                    "room":       meta.get("room", "unknown"),
                    "similarity": sim,
                })

        return hits

    def store_fact(self, key: str, value: str) -> None:
        """
        Persist a user profile fact in the facts room.
        Fire-and-forget (daemon thread).
        """
        if not self._available:
            return
        threading.Thread(
            target=self._upsert_fact_sync,
            args=(key, value),
            daemon=True,
        ).start()

    def status(self) -> Dict:
        """Return a brief status dict for the startup roster / --agents output."""
        if not self._available:
            return {"available": False}
        try:
            count = self._col.count() if self._col else 0
        except Exception:
            count = 0
        return {
            "available":    True,
            "palace_path":  self._palace_path,
            "drawer_count": count,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _store_exchange_sync(self, query: str, response: str, agent_type: str) -> None:
        """Write a Q&A exchange to ChromaDB. Called inside a daemon thread."""
        if not self._col:
            return
        try:
            content = f"Q: {query}\n\nA: {response}"
            content = content[:3000]  # cap to keep drawers manageable

            # Skip near-duplicate drawers (same query, same room)
            room = _room_for(agent_type)
            dup_check = self._col.query(
                query_texts=[content[:200]],
                n_results=1,
                where={"$and": [{"wing": RAIN_WING}, {"room": room}]},
                include=["distances"],
            )
            if dup_check["distances"] and dup_check["distances"][0]:
                if dup_check["distances"][0][0] < 0.05:   # ~95% similar → skip
                    return

            drawer_id = (
                f"rain_{room}_"
                + hashlib.md5(
                    (content[:120] + datetime.now().isoformat()).encode()
                ).hexdigest()[:16]
            )

            self._col.add(
                ids=[drawer_id],
                documents=[content],
                metadatas=[{
                    "wing":        RAIN_WING,
                    "room":        room,
                    "source_file": "rain_conversation",
                    "added_by":    "rain",
                    "filed_at":    datetime.now().isoformat(),
                }],
            )
        except Exception:
            pass

    def _upsert_fact_sync(self, key: str, value: str) -> None:
        """Write a user fact drawer. Idempotent on (key, value) pairs."""
        if not self._col:
            return
        try:
            fact_id = "fact_" + hashlib.md5(f"{key}:{value}".encode()).hexdigest()[:16]
            content = f"{key}: {value}"

            # Upsert — get() returns empty if not present
            existing = self._col.get(ids=[fact_id])
            if existing["ids"]:
                return  # already stored

            self._col.add(
                ids=[fact_id],
                documents=[content],
                metadatas=[{
                    "wing":        RAIN_WING,
                    "room":        "facts",
                    "source_file": "user_profile",
                    "added_by":    "rain",
                    "filed_at":    datetime.now().isoformat(),
                }],
            )
        except Exception:
            pass

    @staticmethod
    def _build_where(wing: Optional[str], room: Optional[str]) -> Dict:
        if wing and room:
            return {"$and": [{"wing": wing}, {"room": room}]}
        if wing:
            return {"wing": wing}
        if room:
            return {"room": room}
        return {}

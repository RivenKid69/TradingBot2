# -*- coding: utf-8 -*-
"""
services/core/durable_store.py
==============================

Durable persistence for alerts and audit events (P2 #21).

Previously alerts lived only in an in-memory dict (lost on restart) and the
hash-chained audit writer wasn't fed live events. This provides:

  * ``DurableAlertStore`` — SQLite-backed alert store that survives restarts
    (save / update_status / load_all), so the AlertingService can rehydrate.
  * ``AuditChain`` — append-only, tamper-evident hash chain (each row binds
    sha256(payload + prev_hash)); ``verify()`` detects any tampering. Use it to
    durably record live order/risk/kill-switch events with WORM-style integrity.

Pure stdlib (sqlite3 + hashlib + json). Thread-safe.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class DurableAlertStore:
    """SQLite-backed alert store (survives restart)."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = str(db_path or (Path.home() / ".ccea" / "alerts.db"))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS alerts ("
            "alert_id TEXT PRIMARY KEY, triggered_at TEXT, severity TEXT, status TEXT, json TEXT)"
        )
        self._conn.commit()

    @staticmethod
    def _to_dict(alert: Any) -> Dict[str, Any]:
        if isinstance(alert, dict):
            return alert
        import dataclasses

        d = dataclasses.asdict(alert) if dataclasses.is_dataclass(alert) else dict(vars(alert))
        # enums -> value
        for k, v in list(d.items()):
            if hasattr(v, "value"):
                d[k] = v.value
        return d

    def save(self, alert: Any) -> None:
        d = self._to_dict(alert)
        aid = str(d.get("alert_id", ""))
        if not aid:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO alerts (alert_id, triggered_at, severity, status, json) "
                "VALUES (?,?,?,?,?)",
                (
                    aid,
                    str(d.get("triggered_at", "")),
                    str(d.get("severity", "")),
                    str(d.get("status", "")),
                    json.dumps(d, default=str),
                ),
            )
            self._conn.commit()

    def update_status(self, alert_id: str, status: str, **fields: Any) -> None:
        with self._lock:
            cur = self._conn.execute("SELECT json FROM alerts WHERE alert_id=?", (alert_id,))
            row = cur.fetchone()
            if not row:
                return
            d = json.loads(row[0])
            d["status"] = status
            d.update(fields)
            self._conn.execute(
                "UPDATE alerts SET status=?, json=? WHERE alert_id=?",
                (status, json.dumps(d, default=str), alert_id),
            )
            self._conn.commit()

    def load_all(self, *, limit: int = 10000) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT json FROM alerts ORDER BY triggered_at DESC LIMIT ?", (int(limit),)
            )
            return [json.loads(r[0]) for r in cur.fetchall()]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])


@dataclass
class AuditEntry:
    seq: int
    ts: str
    event_type: str
    payload: Dict[str, Any]
    prev_hash: str
    entry_hash: str


class AuditChain:
    """Append-only, tamper-evident hash-chained audit log (WORM-style integrity).

    Each entry hash = sha256(seq | ts | event_type | payload_json | prev_hash). Any
    edit/deletion of a past row breaks the chain, detectable by ``verify()``.
    """

    _GENESIS = "0" * 64

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = str(db_path or (Path.home() / ".ccea" / "audit_chain.db"))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS audit ("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, event_type TEXT, "
            "payload TEXT, prev_hash TEXT, entry_hash TEXT)"
        )
        self._conn.commit()

    @staticmethod
    def _hash(seq: int, ts: str, event_type: str, payload_json: str, prev_hash: str) -> str:
        raw = f"{seq}|{ts}|{event_type}|{payload_json}|{prev_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _last_hash(self) -> str:
        cur = self._conn.execute("SELECT entry_hash FROM audit ORDER BY seq DESC LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else self._GENESIS

    def append(self, event_type: str, payload: Dict[str, Any]) -> AuditEntry:
        with self._lock:
            ts = _now_iso()
            prev = self._last_hash()
            cur = self._conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM audit")
            seq = int(cur.fetchone()[0])
            payload_json = json.dumps(payload, sort_keys=True, default=str)
            h = self._hash(seq, ts, event_type, payload_json, prev)
            self._conn.execute(
                "INSERT INTO audit (seq, ts, event_type, payload, prev_hash, entry_hash) "
                "VALUES (?,?,?,?,?,?)",
                (seq, ts, event_type, payload_json, prev, h),
            )
            self._conn.commit()
            return AuditEntry(seq, ts, event_type, payload, prev, h)

    def verify(self) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT seq, ts, event_type, payload, prev_hash, entry_hash FROM audit ORDER BY seq ASC"
            )
            prev = self._GENESIS
            for seq, ts, et, payload_json, prev_hash, entry_hash in cur.fetchall():
                if prev_hash != prev:
                    return False
                if self._hash(seq, ts, et, payload_json, prev_hash) != entry_hash:
                    return False
                prev = entry_hash
            return True

    def tail(self, n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT seq, ts, event_type, payload, entry_hash FROM audit ORDER BY seq DESC LIMIT ?",
                (int(n),),
            )
            return [
                {"seq": s, "ts": t, "event_type": e, "payload": json.loads(p), "entry_hash": h}
                for s, t, e, p, h in cur.fetchall()
            ]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0])


# Process-wide default audit chain for live events (orders/risk/kill-switch).
_GLOBAL_AUDIT: Optional[AuditChain] = None
_AUDIT_LOCK = threading.Lock()


def get_audit_chain(db_path: Optional[str] = None) -> AuditChain:
    global _GLOBAL_AUDIT
    with _AUDIT_LOCK:
        if _GLOBAL_AUDIT is None:
            _GLOBAL_AUDIT = AuditChain(db_path)
        return _GLOBAL_AUDIT


def record_audit_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Convenience: append a live event to the global tamper-evident audit chain."""
    try:
        get_audit_chain().append(event_type, payload)
    except Exception:  # pragma: no cover - audit must never break the trading path
        pass


__all__ = ["DurableAlertStore", "AuditChain", "AuditEntry", "get_audit_chain", "record_audit_event"]

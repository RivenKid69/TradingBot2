# -*- coding: utf-8 -*-
"""
Tamper-evident hash chain - AGENT ZONE.

A reusable keyed (HMAC-SHA256) hash chain for append-only books-and-records:
order journal audit log, trade blotter, cash ledger. Each record commits to the
previous record's hash, so any insertion, deletion, reordering or mutation of a
past record breaks verification from that point onward.

Design
------
* ``entry_hash_i = HMAC_SHA256(key, prev_hash_{i-1} || canonical(payload_i) || seq_i)``
  rendered as hex. The genesis ``prev_hash`` is a fixed sentinel.
* When no key is supplied (no vault master key available), the chain falls back to
  **unkeyed SHA-256** — still tamper-EVIDENT (detects edits) though not tamper-PROOF
  against an attacker who can recompute the whole chain. With a key (sourced from
  the Agent vault), forging requires the secret too.
* ``canonical(payload)`` is JSON with sorted keys and no whitespace, so the hash is
  stable regardless of dict ordering.

References: Schneier & Kelsey, "Secure Audit Logs to Support Computer Forensics"
(1999); RFC 2104 (HMAC). This is the same construction used by blockchains' block
linking, scoped to a single local append-only file.

PROHIBITED in Cloud zone (Agent owns its own records).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

GENESIS_HASH = "0" * 64


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def chain_hash(prev_hash: str, payload: Any, seq: int, *, key: Optional[bytes] = None) -> str:
    """Compute the chained hash for a record given the previous hash."""
    msg = prev_hash.encode("ascii") + _canonical(payload) + str(int(seq)).encode("ascii")
    if key:
        return hmac.new(key, msg, hashlib.sha256).hexdigest()
    return hashlib.sha256(msg).hexdigest()


@dataclass
class ChainRecord:
    seq: int
    payload: Dict[str, Any]
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {"seq": self.seq, "payload": self.payload,
                "prev_hash": self.prev_hash, "entry_hash": self.entry_hash}


@dataclass
class HashChain:
    """In-memory append-only keyed hash chain (mirror of the persisted chain)."""

    key: Optional[bytes] = None
    records: List[ChainRecord] = field(default_factory=list)

    @property
    def head_hash(self) -> str:
        return self.records[-1].entry_hash if self.records else GENESIS_HASH

    @property
    def keyed(self) -> bool:
        return self.key is not None

    def append(self, payload: Dict[str, Any]) -> ChainRecord:
        seq = len(self.records) + 1
        prev = self.head_hash
        h = chain_hash(prev, payload, seq, key=self.key)
        rec = ChainRecord(seq=seq, payload=payload, prev_hash=prev, entry_hash=h)
        self.records.append(rec)
        return rec

    def verify(self) -> Dict[str, Any]:
        return verify_chain(self.records, key=self.key)


def verify_chain(records: List[ChainRecord], *, key: Optional[bytes] = None) -> Dict[str, Any]:
    """Recompute the chain and check linkage + per-record hashes.

    Returns ``{"valid": bool, "n": int, "broken_at": Optional[int], "reason": str}``.
    ``broken_at`` is the 1-based ``seq`` of the first tampered/forged record.
    """
    prev = GENESIS_HASH
    for i, rec in enumerate(records, start=1):
        if rec.seq != i:
            return {"valid": False, "n": len(records), "broken_at": rec.seq,
                    "reason": f"sequence gap: expected {i}, got {rec.seq}"}
        if rec.prev_hash != prev:
            return {"valid": False, "n": len(records), "broken_at": rec.seq,
                    "reason": "prev_hash linkage broken (insertion/deletion/reorder)"}
        expect = chain_hash(prev, rec.payload, rec.seq, key=key)
        if expect != rec.entry_hash:
            return {"valid": False, "n": len(records), "broken_at": rec.seq,
                    "reason": "entry_hash mismatch (record payload was mutated)"}
        prev = rec.entry_hash
    return {"valid": True, "n": len(records), "broken_at": None, "reason": "ok"}


__all__ = ["HashChain", "ChainRecord", "chain_hash", "verify_chain", "GENESIS_HASH"]

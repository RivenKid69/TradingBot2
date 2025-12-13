# -*- coding: utf-8 -*-
"""
Evidence Recording - Audit trail for approvals.

AGENT ZONE ONLY.

Creates tamper-evident records of approval decisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


def compute_evidence_hash(data: Dict[str, Any]) -> str:
    """
    Compute hash of evidence data.

    Creates a deterministic hash for audit purposes.

    Args:
        data: Data to hash

    Returns:
        SHA256 hash as hex string
    """
    # Sort keys for deterministic output
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()


@dataclass
class EvidenceRecord:
    """
    Evidence record for an approval decision.

    Provides audit trail that can be exported for compliance.
    """

    record_id: str
    request_id: str
    command_type: str
    decision: str  # approved, denied
    decided_by: str
    decided_at: datetime

    # Hashes for integrity
    evidence_hash: str
    config_hash: Optional[str] = None
    artifact_hash: Optional[str] = None

    # Context
    agent_id: str = ""
    strategy_id: str = ""
    description: str = ""
    reason: str = ""

    # Change details
    change_class: str = "trading_impacting"
    diff_summary: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "request_id": self.request_id,
            "command_type": self.command_type,
            "decision": self.decision,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at.isoformat(),
            "evidence_hash": self.evidence_hash,
            "config_hash": self.config_hash,
            "artifact_hash": self.artifact_hash,
            "agent_id": self.agent_id,
            "strategy_id": self.strategy_id,
            "description": self.description,
            "reason": self.reason,
            "change_class": self.change_class,
            "diff_summary": self.diff_summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceRecord:
        """Create from dictionary."""
        return cls(
            record_id=data["record_id"],
            request_id=data["request_id"],
            command_type=data["command_type"],
            decision=data["decision"],
            decided_by=data["decided_by"],
            decided_at=datetime.fromisoformat(data["decided_at"]),
            evidence_hash=data["evidence_hash"],
            config_hash=data.get("config_hash"),
            artifact_hash=data.get("artifact_hash"),
            agent_id=data.get("agent_id", ""),
            strategy_id=data.get("strategy_id", ""),
            description=data.get("description", ""),
            reason=data.get("reason", ""),
            change_class=data.get("change_class", "trading_impacting"),
            diff_summary=data.get("diff_summary"),
            metadata=data.get("metadata", {}),
        )

    def verify_hash(self) -> bool:
        """Verify evidence hash matches content."""
        content = {
            "request_id": self.request_id,
            "command_type": self.command_type,
            "decision": self.decision,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at.isoformat(),
            "config_hash": self.config_hash,
            "artifact_hash": self.artifact_hash,
        }
        computed = compute_evidence_hash(content)
        return computed == self.evidence_hash

    def to_audit_line(self) -> str:
        """Format as single audit log line."""
        return (
            f"{self.decided_at.isoformat()} | "
            f"{self.decision.upper()} | "
            f"{self.command_type} | "
            f"{self.decided_by} | "
            f"{self.evidence_hash[:16]}... | "
            f"{self.reason or 'no reason'}"
        )


class EvidenceStore:
    """
    Store for evidence records.

    Provides durable storage and query capabilities.
    """

    def __init__(self):
        """Initialize store."""
        self._records: Dict[str, EvidenceRecord] = {}

    def add(self, record: EvidenceRecord) -> None:
        """Add evidence record."""
        self._records[record.record_id] = record

    def get(self, record_id: str) -> Optional[EvidenceRecord]:
        """Get record by ID."""
        return self._records.get(record_id)

    def get_by_request(self, request_id: str) -> Optional[EvidenceRecord]:
        """Get record by request ID."""
        for record in self._records.values():
            if record.request_id == request_id:
                return record
        return None

    def get_all(self) -> list[EvidenceRecord]:
        """Get all records."""
        return list(self._records.values())

    def export_audit_log(self) -> str:
        """Export all records as audit log."""
        lines = []
        for record in sorted(self._records.values(), key=lambda r: r.decided_at):
            lines.append(record.to_audit_line())
        return "\n".join(lines)

    def export_json(self) -> str:
        """Export all records as JSON."""
        records = [r.to_dict() for r in self._records.values()]
        return json.dumps(records, indent=2)

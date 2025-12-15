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
        # Include previous_hash if this is part of a chain
        if "previous_hash" in self.metadata:
            content["previous_hash"] = self.metadata["previous_hash"]
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

    def verify_all(self) -> tuple[bool, list[str]]:
        """
        Verify all records' integrity.

        Returns:
            Tuple of (all_valid, list of invalid record IDs)
        """
        invalid = []
        for record in self._records.values():
            if not record.verify_hash():
                invalid.append(record.record_id)
        return len(invalid) == 0, invalid


class EvidenceChain:
    """
    Chain of evidence records with cryptographic linking.

    Each record references the hash of the previous record,
    creating an immutable audit trail.
    """

    def __init__(self):
        """Initialize chain."""
        self._chain: list[EvidenceRecord] = []
        self._previous_hash: Optional[str] = None

    def add(self, record: EvidenceRecord) -> str:
        """
        Add record to chain.

        The record's metadata is updated to include the previous
        record's hash, creating the chain link.

        Args:
            record: Evidence record to add

        Returns:
            Record's evidence hash
        """
        # Add chain link
        if self._previous_hash:
            record.metadata["previous_hash"] = self._previous_hash
            # Recompute hash with chain link
            content = {
                "request_id": record.request_id,
                "command_type": record.command_type,
                "decision": record.decision,
                "decided_by": record.decided_by,
                "decided_at": record.decided_at.isoformat(),
                "config_hash": record.config_hash,
                "artifact_hash": record.artifact_hash,
                "previous_hash": self._previous_hash,
            }
            record.evidence_hash = compute_evidence_hash(content)

        self._chain.append(record)
        self._previous_hash = record.evidence_hash

        return record.evidence_hash

    def verify_chain(self) -> tuple[bool, list[str]]:
        """
        Verify entire chain integrity.

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors: list[str] = []
        prev_hash: Optional[str] = None

        for i, record in enumerate(self._chain):
            # Verify record hash
            if not record.verify_hash():
                errors.append(f"Record {i}: Hash mismatch")

            # Verify chain link
            if i > 0:
                expected_prev = record.metadata.get("previous_hash")
                if expected_prev != prev_hash:
                    errors.append(f"Record {i}: Chain link broken")

            prev_hash = record.evidence_hash

        return len(errors) == 0, errors

    def get_latest(self) -> Optional[EvidenceRecord]:
        """Get most recent record."""
        if self._chain:
            return self._chain[-1]
        return None

    def get_all(self) -> list[EvidenceRecord]:
        """Get all records in chain order."""
        return self._chain.copy()

    def export(self) -> list[Dict[str, Any]]:
        """Export chain as list of dicts."""
        return [r.to_dict() for r in self._chain]

    def import_chain(self, data: list[Dict[str, Any]]) -> bool:
        """
        Import and verify chain from data.

        Args:
            data: List of record dictionaries

        Returns:
            True if imported and verified successfully
        """
        records = [EvidenceRecord.from_dict(d) for d in data]

        # Verify imported chain
        temp = EvidenceChain()
        temp._chain = records
        if records:
            temp._previous_hash = records[-1].evidence_hash

        is_valid, _ = temp.verify_chain()
        if not is_valid:
            return False

        self._chain = records
        if records:
            self._previous_hash = records[-1].evidence_hash

        return True

    def __len__(self) -> int:
        """Get chain length."""
        return len(self._chain)


def create_evidence_record(
    request_id: str,
    command_type: str,
    approved: bool,
    decided_by: str,
    config_hash: Optional[str] = None,
    artifact_hash: Optional[str] = None,
    reason: str = "",
    change_class: str = "trading_impacting",
    description: str = "",
    strategy_id: str = "",
    agent_id: str = "",
) -> EvidenceRecord:
    """
    Create a new evidence record with computed hash.

    Args:
        request_id: Original request ID
        command_type: Type of command
        approved: Whether approved
        decided_by: Who decided
        config_hash: Config digest
        artifact_hash: Artifact digest
        reason: Decision reason
        change_class: Change classification
        description: Human-readable description
        strategy_id: Strategy identifier
        agent_id: Agent identifier

    Returns:
        EvidenceRecord with computed hash
    """
    from uuid import uuid4

    decided_at = datetime.utcnow()
    decision = "approved" if approved else "denied"

    # Compute evidence hash
    content = {
        "request_id": request_id,
        "command_type": command_type,
        "decision": decision,
        "decided_by": decided_by,
        "decided_at": decided_at.isoformat(),
        "config_hash": config_hash,
        "artifact_hash": artifact_hash,
    }
    evidence_hash = compute_evidence_hash(content)

    return EvidenceRecord(
        record_id=str(uuid4()),
        request_id=request_id,
        command_type=command_type,
        decision=decision,
        decided_by=decided_by,
        decided_at=decided_at,
        evidence_hash=evidence_hash,
        config_hash=config_hash,
        artifact_hash=artifact_hash,
        agent_id=agent_id,
        strategy_id=strategy_id,
        description=description,
        reason=reason,
        change_class=change_class,
    )

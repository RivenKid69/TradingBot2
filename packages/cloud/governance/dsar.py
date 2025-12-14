# -*- coding: utf-8 -*-
"""
GDPR Data Subject Access Request (DSAR) Service.

CCEA Phase 8 Implementation.

This service handles GDPR compliance requirements:
    - Data export (Article 20 - Right to data portability)
    - Data deletion (Article 17 - Right to erasure)
    - Data access (Article 15 - Right of access)
    - Processing status tracking

Design Doc Reference:
    - Phase 8 (13.3): "Retention per tenant, auto-purge, export/delete (DSAR)"
    - GDPR compliance for EU tenants

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set
from uuid import UUID, uuid4


# ============================================================================
# Constants
# ============================================================================

# GDPR requires response within 30 days
DSAR_RESPONSE_DEADLINE_DAYS: Final[int] = 30

# Extension allowed once for complex requests
DSAR_EXTENSION_DAYS: Final[int] = 60

# Data categories subject to DSAR
DSAR_DATA_CATEGORIES: Final[Set[str]] = {
    "telemetry_events",
    "alerts",
    "commands",
    "approval_records",
    "access_audits",
    "user_settings",
    "agent_data",
    "run_data",
    "deployment_data",
}


class DSARRequestType(Enum):
    """Types of DSAR requests."""
    ACCESS = auto()      # Right of access (Article 15)
    PORTABILITY = auto()  # Right to data portability (Article 20)
    ERASURE = auto()     # Right to erasure (Article 17)
    RECTIFICATION = auto()  # Right to rectification (Article 16)


class DSARStatus(Enum):
    """DSAR request processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_VERIFICATION = "awaiting_verification"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    REJECTED = "rejected"
    EXTENDED = "extended"


@dataclass
class DSARRequest:
    """
    DSAR request record.

    Attributes:
        id: Request identifier
        request_type: Type of DSAR request
        user_id: Subject user ID
        workspace_id: Workspace scope
        status: Current status
        data_categories: Categories to include
        reason: Reason for request (especially for erasure)
        verification_method: How identity was verified
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    request_type: DSARRequestType = DSARRequestType.ACCESS
    user_id: str = ""
    workspace_id: str = ""
    status: DSARStatus = DSARStatus.PENDING
    data_categories: Set[str] = field(default_factory=lambda: set(DSAR_DATA_CATEGORIES))
    reason: str = ""
    verification_method: str = ""
    verified_at: Optional[datetime] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=DSAR_RESPONSE_DEADLINE_DAYS))
    extended_deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Processing
    processed_by: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    # Result
    export_path: Optional[str] = None
    export_checksum: Optional[str] = None
    records_processed: int = 0
    records_deleted: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "request_type": self.request_type.name,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "status": self.status.value,
            "data_categories": list(self.data_categories),
            "reason": self.reason,
            "verification_method": self.verification_method,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat(),
            "extended_deadline": self.extended_deadline.isoformat() if self.extended_deadline else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processed_by": self.processed_by,
            "notes": self.notes,
            "export_path": self.export_path,
            "export_checksum": self.export_checksum,
            "records_processed": self.records_processed,
            "records_deleted": self.records_deleted,
        }

    @property
    def is_overdue(self) -> bool:
        """Check if request is overdue."""
        effective_deadline = self.extended_deadline or self.deadline
        return datetime.utcnow() > effective_deadline and self.status not in (
            DSARStatus.COMPLETED, DSARStatus.REJECTED
        )


@dataclass
class DSARResult:
    """Result of DSAR processing."""
    success: bool = True
    request_id: str = ""
    request_type: DSARRequestType = DSARRequestType.ACCESS
    status: DSARStatus = DSARStatus.COMPLETED
    records_processed: int = 0
    records_deleted: int = 0
    export_path: Optional[str] = None
    export_checksum: Optional[str] = None
    error: Optional[str] = None
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "request_id": self.request_id,
            "request_type": self.request_type.name,
            "status": self.status.value,
            "records_processed": self.records_processed,
            "records_deleted": self.records_deleted,
            "export_path": self.export_path,
            "export_checksum": self.export_checksum,
            "error": self.error,
            "processing_time_seconds": self.processing_time_seconds,
        }


class DSARService:
    """
    GDPR Data Subject Access Request service.

    Handles GDPR compliance for data export, deletion, and access.

    Usage:
        dsar = DSARService(data_fetcher=my_data_fetcher)

        # Create access request
        request = dsar.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # Process request
        result = await dsar.process_request(request.id)

        # For erasure
        request = dsar.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
            reason="User requested account deletion",
        )

    COMPLIANCE:
        - Must respond within 30 days (extendable to 60)
        - Must verify identity before processing
        - Must log all DSAR activities
    """

    def __init__(
        self,
        data_fetcher: Optional[Callable[[str, str, Set[str]], List[Dict]]] = None,
        data_deleter: Optional[Callable[[str, str, Set[str]], int]] = None,
        export_dir: Optional[Path] = None,
    ):
        """
        Initialize DSAR service.

        Args:
            data_fetcher: Function to fetch user data (user_id, workspace_id, categories) -> records
            data_deleter: Function to delete user data (user_id, workspace_id, categories) -> count
            export_dir: Directory for export files
        """
        self._data_fetcher = data_fetcher
        self._data_deleter = data_deleter
        self._export_dir = export_dir or Path("/tmp/dsar_exports")
        self._requests: Dict[str, DSARRequest] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def create_request(
        self,
        user_id: str,
        workspace_id: str,
        request_type: DSARRequestType,
        data_categories: Optional[Set[str]] = None,
        reason: str = "",
    ) -> DSARRequest:
        """
        Create a new DSAR request.

        Args:
            user_id: User making the request
            workspace_id: Workspace scope
            request_type: Type of request
            data_categories: Categories to include (defaults to all)
            reason: Reason for request

        Returns:
            Created DSARRequest
        """
        request = DSARRequest(
            request_type=request_type,
            user_id=user_id,
            workspace_id=workspace_id,
            data_categories=data_categories or set(DSAR_DATA_CATEGORIES),
            reason=reason,
        )

        with self._lock:
            self._requests[request.id] = request
            self._log_audit("request_created", request)

        return request

    def get_request(self, request_id: str) -> Optional[DSARRequest]:
        """Get request by ID."""
        with self._lock:
            return self._requests.get(request_id)

    def get_user_requests(self, user_id: str) -> List[DSARRequest]:
        """Get all requests for a user."""
        with self._lock:
            return [r for r in self._requests.values() if r.user_id == user_id]

    def verify_identity(
        self,
        request_id: str,
        verification_method: str,
        verified_by: str,
    ) -> bool:
        """
        Mark request as identity verified.

        Args:
            request_id: Request to verify
            verification_method: How identity was verified
            verified_by: Who verified

        Returns:
            True if verification recorded
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            request.verification_method = verification_method
            request.verified_at = datetime.utcnow()
            request.status = DSARStatus.IN_PROGRESS
            request.notes.append(f"Identity verified by {verified_by} via {verification_method}")

            self._log_audit("identity_verified", request)
            return True

    def extend_deadline(
        self,
        request_id: str,
        reason: str,
        extended_by: str,
    ) -> bool:
        """
        Extend request deadline (allowed once per GDPR).

        Args:
            request_id: Request to extend
            reason: Reason for extension
            extended_by: Who extended

        Returns:
            True if extension granted
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            # Can only extend once
            if request.extended_deadline:
                return False

            request.extended_deadline = datetime.utcnow() + timedelta(days=DSAR_EXTENSION_DAYS)
            request.status = DSARStatus.EXTENDED
            request.notes.append(f"Deadline extended by {extended_by}: {reason}")

            self._log_audit("deadline_extended", request)
            return True

    async def process_request(
        self,
        request_id: str,
        processed_by: str = "system",
    ) -> DSARResult:
        """
        Process a DSAR request.

        Args:
            request_id: Request to process
            processed_by: Who is processing

        Returns:
            DSARResult
        """
        start_time = datetime.utcnow()

        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return DSARResult(
                    success=False,
                    request_id=request_id,
                    error="Request not found",
                )

            # Must be verified for erasure
            if request.request_type == DSARRequestType.ERASURE and not request.verified_at:
                return DSARResult(
                    success=False,
                    request_id=request_id,
                    request_type=request.request_type,
                    status=DSARStatus.AWAITING_VERIFICATION,
                    error="Identity verification required for erasure",
                )

            request.processed_by = processed_by
            request.status = DSARStatus.IN_PROGRESS

        try:
            if request.request_type in (DSARRequestType.ACCESS, DSARRequestType.PORTABILITY):
                result = await self._process_export(request)
            elif request.request_type == DSARRequestType.ERASURE:
                result = await self._process_erasure(request)
            else:
                result = DSARResult(
                    success=False,
                    request_id=request_id,
                    request_type=request.request_type,
                    error=f"Unsupported request type: {request.request_type}",
                )

            # Update request
            with self._lock:
                request.status = result.status
                request.completed_at = datetime.utcnow()
                request.records_processed = result.records_processed
                request.records_deleted = result.records_deleted
                request.export_path = result.export_path
                request.export_checksum = result.export_checksum

                self._log_audit("request_completed", request)

            result.processing_time_seconds = (datetime.utcnow() - start_time).total_seconds()
            return result

        except Exception as e:
            with self._lock:
                request.status = DSARStatus.PARTIALLY_COMPLETED
                request.notes.append(f"Error during processing: {str(e)}")
                self._log_audit("request_error", request, {"error": str(e)})

            return DSARResult(
                success=False,
                request_id=request_id,
                request_type=request.request_type,
                status=DSARStatus.PARTIALLY_COMPLETED,
                error=str(e),
            )

    async def _process_export(self, request: DSARRequest) -> DSARResult:
        """Process data export request."""
        if not self._data_fetcher:
            return DSARResult(
                success=False,
                request_id=request.id,
                request_type=request.request_type,
                error="Data fetcher not configured",
            )

        # Fetch data
        records = self._data_fetcher(
            request.user_id,
            request.workspace_id,
            request.data_categories,
        )

        # Ensure export directory exists
        self._export_dir.mkdir(parents=True, exist_ok=True)

        # Create export file
        export_filename = f"dsar_export_{request.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        export_path = self._export_dir / export_filename

        # Write export
        export_data = {
            "metadata": {
                "request_id": request.id,
                "request_type": request.request_type.name,
                "user_id": request.user_id,
                "exported_at": datetime.utcnow().isoformat(),
                "data_categories": list(request.data_categories),
                "record_count": len(records),
            },
            "data": records,
        }

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        # Calculate checksum
        with open(export_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        return DSARResult(
            success=True,
            request_id=request.id,
            request_type=request.request_type,
            status=DSARStatus.COMPLETED,
            records_processed=len(records),
            export_path=str(export_path),
            export_checksum=checksum,
        )

    async def _process_erasure(self, request: DSARRequest) -> DSARResult:
        """Process data erasure request."""
        if not self._data_deleter:
            return DSARResult(
                success=False,
                request_id=request.id,
                request_type=request.request_type,
                error="Data deleter not configured",
            )

        # First export for audit purposes
        if self._data_fetcher:
            records = self._data_fetcher(
                request.user_id,
                request.workspace_id,
                request.data_categories,
            )
            records_count = len(records)

            # Create pre-deletion snapshot (for compliance audit)
            self._export_dir.mkdir(parents=True, exist_ok=True)
            snapshot_filename = f"erasure_snapshot_{request.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            snapshot_path = self._export_dir / snapshot_filename

            snapshot_data = {
                "metadata": {
                    "request_id": request.id,
                    "type": "pre_erasure_snapshot",
                    "snapshot_at": datetime.utcnow().isoformat(),
                },
                "record_count": records_count,
                # Don't include actual data in snapshot for privacy
            }

            with open(snapshot_path, "w") as f:
                json.dump(snapshot_data, f, indent=2)
        else:
            records_count = 0

        # Perform deletion
        deleted_count = self._data_deleter(
            request.user_id,
            request.workspace_id,
            request.data_categories,
        )

        return DSARResult(
            success=True,
            request_id=request.id,
            request_type=request.request_type,
            status=DSARStatus.COMPLETED,
            records_processed=records_count,
            records_deleted=deleted_count,
        )

    def reject_request(
        self,
        request_id: str,
        reason: str,
        rejected_by: str,
    ) -> bool:
        """
        Reject a DSAR request.

        Args:
            request_id: Request to reject
            reason: Rejection reason
            rejected_by: Who rejected

        Returns:
            True if rejection recorded
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            request.status = DSARStatus.REJECTED
            request.completed_at = datetime.utcnow()
            request.notes.append(f"Rejected by {rejected_by}: {reason}")

            self._log_audit("request_rejected", request, {"reason": reason})
            return True

    def get_overdue_requests(self) -> List[DSARRequest]:
        """Get all overdue requests."""
        with self._lock:
            return [r for r in self._requests.values() if r.is_overdue]

    def get_pending_requests(self) -> List[DSARRequest]:
        """Get all pending requests."""
        with self._lock:
            return [
                r for r in self._requests.values()
                if r.status in (DSARStatus.PENDING, DSARStatus.IN_PROGRESS, DSARStatus.AWAITING_VERIFICATION)
            ]

    def _log_audit(
        self,
        action: str,
        request: DSARRequest,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log audit event."""
        entry = {
            "action": action,
            "request_id": request.id,
            "request_type": request.request_type.name,
            "user_id": request.user_id,
            "workspace_id": request.workspace_id,
            "status": request.status.value,
            "timestamp": datetime.utcnow().isoformat(),
            **(extra or {}),
        }
        self._audit_log.append(entry)

    def get_audit_log(self, request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit log, optionally filtered by request ID."""
        with self._lock:
            if request_id:
                return [e for e in self._audit_log if e.get("request_id") == request_id]
            return list(self._audit_log)

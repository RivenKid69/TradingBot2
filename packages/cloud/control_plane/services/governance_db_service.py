# -*- coding: utf-8 -*-
"""
Governance DB Service - Connects governance services to database persistence.

CCEA Phase 10: Design Doc compliance for data governance.

This service bridges the in-memory governance services (residency, retention, DSAR)
with real database persistence, making them the source of truth for:
- Data residency policies
- Retention policies
- DSAR requests

Design Doc Requirements:
- Data residency: EU region default for EU tenants
- Retention per tenant, auto-purge
- DSAR export/delete workflows
- Audit trail for all governance operations

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from packages.cloud.governance.residency import (
    DataResidencyManager,
    ResidencyPolicy,
    ResidencyConfig,
    DataRegion,
    ResidencyMode,
)
from packages.cloud.governance.retention import (
    RetentionService,
    RetentionPolicy,
    RetentionConfig,
    RetentionAction,
)
from packages.cloud.governance.dsar import (
    DSARService,
    DSARRequest,
    DSARRequestType,
    DSARStatus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Database Models for Governance
# ============================================================================

# Note: These would typically be in models.py, but we define schemas here
# to show what fields are needed.


class GovernancePolicyType(str, Enum):
    """Types of governance policies stored in DB."""

    RESIDENCY = "residency"
    RETENTION = "retention"
    DSAR = "dsar"


@dataclass
class GovernancePolicyRecord:
    """Database record for governance policies."""

    id: UUID
    workspace_id: UUID
    policy_type: GovernancePolicyType
    data_type: str  # e.g., "telemetry_events", "alerts", etc.
    config: Dict[str, Any]  # JSON field with policy configuration
    enabled: bool
    created_at: datetime
    updated_at: datetime
    created_by: Optional[UUID] = None


@dataclass
class DSARRequestRecord:
    """Database record for DSAR requests."""

    id: UUID
    workspace_id: UUID
    user_id: UUID
    request_type: str  # DSARRequestType
    status: str  # DSARStatus
    data_categories: List[str]
    reason: str
    verification_method: Optional[str]
    verified_at: Optional[datetime]
    deadline: datetime
    extended_deadline: Optional[datetime]
    completed_at: Optional[datetime]
    processed_by: Optional[str]
    export_path: Optional[str]
    export_checksum: Optional[str]
    records_processed: int
    records_deleted: int
    notes: List[str]
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Governance DB Service
# ============================================================================


class GovernanceDBService:
    """
    Service that connects governance services to database persistence.

    This service acts as the integration layer between:
    - In-memory governance services (residency, retention, DSAR)
    - Database persistence
    - Purge scheduler

    It ensures that governance policies are:
    1. Persisted to the database
    2. Loaded on startup
    3. Used as the source of truth for all governance decisions

    Usage:
        service = GovernanceDBService(session)

        # Create residency policy (persisted to DB)
        await service.create_residency_policy(
            workspace_id=workspace_id,
            country_code="DE",  # Germany -> EU region
        )

        # Create retention policy (persisted to DB)
        await service.create_retention_policy(
            workspace_id=workspace_id,
            data_type="telemetry_events",
            retention_days=90,
        )

        # Create DSAR request (persisted to DB)
        await service.create_dsar_request(
            workspace_id=workspace_id,
            user_id=user_id,
            request_type=DSARRequestType.ERASURE,
        )
    """

    def __init__(
        self,
        session: AsyncSession,
        residency_config: Optional[ResidencyConfig] = None,
        retention_config: Optional[RetentionConfig] = None,
    ):
        """
        Initialize governance DB service.

        Args:
            session: Database session
            residency_config: Optional residency configuration
            retention_config: Optional retention configuration
        """
        self._session = session

        # Initialize in-memory services
        self._residency_manager = DataResidencyManager(config=residency_config)
        self._retention_service = RetentionService(config=retention_config)
        self._dsar_service = DSARService()

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

    # ===== Residency Policy Methods =====

    async def create_residency_policy(
        self,
        workspace_id: UUID,
        country_code: str,
        mode: ResidencyMode = ResidencyMode.CLOUD,
        telemetry_local: bool = False,
        created_by: Optional[UUID] = None,
    ) -> ResidencyPolicy:
        """
        Create and persist a residency policy.

        Args:
            workspace_id: Workspace identifier
            country_code: ISO country code
            mode: Residency mode
            telemetry_local: Keep telemetry local (enterprise)
            created_by: User who created the policy

        Returns:
            Created ResidencyPolicy
        """
        # Create policy using in-memory service
        policy = self._residency_manager.create_policy_for_country(
            workspace_id=str(workspace_id),
            country_code=country_code,
        )

        # Override mode if specified
        if mode != ResidencyMode.CLOUD:
            policy.mode = mode
        if telemetry_local:
            policy.telemetry_local = True

        # Persist to database
        await self._persist_residency_policy(workspace_id, policy, created_by)

        # Log audit
        self._log_audit(
            "residency_policy_created",
            workspace_id,
            {
                "country_code": country_code,
                "primary_region": policy.primary_region.value,
                "mode": policy.mode.name,
            },
        )

        return policy

    async def create_enterprise_local_policy(
        self,
        workspace_id: UUID,
        primary_region: DataRegion = DataRegion.US_EAST,
        created_by: Optional[UUID] = None,
    ) -> ResidencyPolicy:
        """
        Create enterprise local-only policy.

        Telemetry stays on agent, not sent to cloud.
        """
        policy = self._residency_manager.create_enterprise_local_policy(
            workspace_id=str(workspace_id),
            primary_region=primary_region,
        )

        await self._persist_residency_policy(workspace_id, policy, created_by)

        self._log_audit(
            "enterprise_local_policy_created",
            workspace_id,
            {"primary_region": policy.primary_region.value},
        )

        return policy

    async def get_residency_policy(self, workspace_id: UUID) -> Optional[ResidencyPolicy]:
        """
        Get residency policy for workspace.

        Loads from DB and updates in-memory service.
        """
        # First check in-memory
        policy = self._residency_manager.get_policy(str(workspace_id))
        if policy:
            return policy

        # Load from DB
        policy = await self._load_residency_policy(workspace_id)
        return policy

    async def update_residency_policy(
        self,
        workspace_id: UUID,
        updates: Dict[str, Any],
    ) -> Optional[ResidencyPolicy]:
        """Update residency policy."""
        policy = self._residency_manager.update_policy(str(workspace_id), updates)
        if policy:
            await self._persist_residency_policy(workspace_id, policy)
            self._log_audit(
                "residency_policy_updated",
                workspace_id,
                {"updates": list(updates.keys())},
            )
        return policy

    async def _persist_residency_policy(
        self,
        workspace_id: UUID,
        policy: ResidencyPolicy,
        created_by: Optional[UUID] = None,
    ) -> None:
        """Persist residency policy to database."""
        from ..models import GovernancePolicy

        # Check if policy already exists
        existing = await self._session.execute(
            select(GovernancePolicy).where(
                GovernancePolicy.workspace_id == workspace_id,
                GovernancePolicy.policy_type == "residency",
            )
        )
        db_policy = existing.scalar_one_or_none()

        policy_config = {
            "workspace_id": str(policy.workspace_id),
            "primary_region": policy.primary_region.value,
            "failover_region": policy.failover_region.value if policy.failover_region else None,
            "mode": policy.mode.name,
            "gdpr_compliant": policy.gdpr_compliant,
            "telemetry_local": policy.telemetry_local,
            "allowed_regions": [r.value for r in policy.allowed_regions],
        }

        if db_policy:
            # Update existing
            db_policy.config = policy_config
            db_policy.enabled = True
        else:
            # Create new
            db_policy = GovernancePolicy(
                workspace_id=workspace_id,
                policy_type="residency",
                config=policy_config,
                enabled=True,
                created_by_user_id=created_by,
            )
            self._session.add(db_policy)

        await self._session.commit()
        logger.info(f"Persisted residency policy for workspace {workspace_id}")

    async def _load_residency_policy(self, workspace_id: UUID) -> Optional[ResidencyPolicy]:
        """Load residency policy from database."""
        from ..models import GovernancePolicy

        result = await self._session.execute(
            select(GovernancePolicy).where(
                GovernancePolicy.workspace_id == workspace_id,
                GovernancePolicy.policy_type == "residency",
                GovernancePolicy.enabled == True,
            )
        )
        db_policy = result.scalar_one_or_none()

        if not db_policy:
            return None

        config = db_policy.config
        policy = ResidencyPolicy(
            workspace_id=config.get("workspace_id", str(workspace_id)),
            primary_region=DataRegion(config.get("primary_region", "us_east")),
            failover_region=(
                DataRegion(config["failover_region"]) if config.get("failover_region") else None
            ),
            mode=ResidencyMode[config.get("mode", "CLOUD")],
            gdpr_compliant=config.get("gdpr_compliant", False),
            telemetry_local=config.get("telemetry_local", False),
        )

        # Register in memory manager
        self._residency_manager._policies[str(workspace_id)] = policy

        return policy

    # ===== Retention Policy Methods =====

    async def create_retention_policy(
        self,
        workspace_id: UUID,
        data_type: str,
        retention_days: int,
        action: RetentionAction = RetentionAction.DELETE,
        enabled: bool = True,
        created_by: Optional[UUID] = None,
    ) -> RetentionPolicy:
        """
        Create and persist a retention policy.

        Args:
            workspace_id: Workspace identifier
            data_type: Type of data (e.g., "telemetry_events")
            retention_days: Days to retain data
            action: Action after retention (DELETE, ARCHIVE, etc.)
            enabled: Whether policy is active
            created_by: User who created the policy

        Returns:
            Created RetentionPolicy
        """
        # Create policy using in-memory service
        policy = self._retention_service.create_policy(
            workspace_id=str(workspace_id),
            data_type=data_type,
            retention_days=retention_days,
            action=action,
        )

        # Persist to database
        await self._persist_retention_policy(workspace_id, policy, created_by)

        # Log audit
        self._log_audit(
            "retention_policy_created",
            workspace_id,
            {
                "data_type": data_type,
                "retention_days": retention_days,
                "action": action.name,
            },
        )

        return policy

    async def get_retention_policies(self, workspace_id: UUID) -> List[RetentionPolicy]:
        """Get all retention policies for workspace."""
        return self._retention_service.get_workspace_policies(str(workspace_id))

    async def update_retention_policy(
        self,
        workspace_id: UUID,
        data_type: str,
        retention_days: Optional[int] = None,
        action: Optional[RetentionAction] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[RetentionPolicy]:
        """Update retention policy."""
        policy = self._retention_service.update_policy(
            workspace_id=str(workspace_id),
            data_type=data_type,
            retention_days=retention_days,
            action=action,
            enabled=enabled,
        )

        if policy:
            await self._persist_retention_policy(workspace_id, policy)
            self._log_audit(
                "retention_policy_updated",
                workspace_id,
                {"data_type": data_type},
            )

        return policy

    async def set_legal_hold(
        self,
        workspace_id: UUID,
        data_type: str,
        reason: str,
        until: Optional[datetime] = None,
        set_by: Optional[UUID] = None,
    ) -> bool:
        """Set legal hold on data type."""
        result = self._retention_service.set_legal_hold(
            workspace_id=str(workspace_id),
            data_type=data_type,
            reason=reason,
            until=until,
        )

        if result:
            self._log_audit(
                "legal_hold_set",
                workspace_id,
                {
                    "data_type": data_type,
                    "reason": reason,
                    "until": until.isoformat() if until else "indefinite",
                },
            )

        return result

    async def release_legal_hold(
        self,
        workspace_id: UUID,
        data_type: str,
        released_by: UUID,
    ) -> bool:
        """Release legal hold."""
        result = self._retention_service.release_legal_hold(
            workspace_id=str(workspace_id),
            data_type=data_type,
            released_by=str(released_by),
        )

        if result:
            self._log_audit(
                "legal_hold_released",
                workspace_id,
                {"data_type": data_type, "released_by": str(released_by)},
            )

        return result

    async def _persist_retention_policy(
        self,
        workspace_id: UUID,
        policy: RetentionPolicy,
        created_by: Optional[UUID] = None,
    ) -> None:
        """Persist retention policy to database."""
        from ..models import GovernancePolicy, DataRetentionPolicy

        # Try GovernancePolicy model first (unified governance)
        try:
            existing = await self._session.execute(
                select(GovernancePolicy).where(
                    GovernancePolicy.workspace_id == workspace_id,
                    GovernancePolicy.policy_type == "retention",
                    GovernancePolicy.data_type == policy.data_type,
                )
            )
            db_policy = existing.scalar_one_or_none()

            policy_config = {
                "data_type": policy.data_type,
                "retention_days": policy.retention_days,
                "action": policy.action.name,
                "enabled": policy.enabled,
                "last_applied": policy.last_applied.isoformat() if policy.last_applied else None,
            }

            if db_policy:
                db_policy.config = policy_config
                db_policy.enabled = policy.enabled
            else:
                db_policy = GovernancePolicy(
                    workspace_id=workspace_id,
                    policy_type="retention",
                    data_type=policy.data_type,
                    config=policy_config,
                    enabled=policy.enabled,
                    created_by_user_id=created_by,
                )
                self._session.add(db_policy)

            # Also update legacy DataRetentionPolicy for compatibility
            try:
                legacy_result = await self._session.execute(
                    select(DataRetentionPolicy).where(
                        DataRetentionPolicy.workspace_id == workspace_id,
                        DataRetentionPolicy.data_type == policy.data_type,
                    )
                )
                legacy_policy = legacy_result.scalar_one_or_none()

                if legacy_policy:
                    legacy_policy.retention_days = policy.retention_days
                    legacy_policy.auto_purge_enabled = policy.enabled
                else:
                    legacy_policy = DataRetentionPolicy(
                        workspace_id=workspace_id,
                        data_type=policy.data_type,
                        retention_days=policy.retention_days,
                        auto_purge_enabled=policy.enabled,
                    )
                    self._session.add(legacy_policy)
            except Exception:
                pass  # Legacy model may not exist

            await self._session.commit()
            logger.info(
                f"Persisted retention policy for workspace {workspace_id}, data_type={policy.data_type}"
            )

        except Exception as e:
            logger.warning(f"Failed to persist retention policy: {e}")
            await self._session.rollback()

    # ===== DSAR Methods =====

    async def create_dsar_request(
        self,
        workspace_id: UUID,
        user_id: UUID,
        request_type: DSARRequestType,
        data_categories: Optional[Set[str]] = None,
        reason: str = "",
    ) -> DSARRequest:
        """
        Create and persist a DSAR request.

        Args:
            workspace_id: Workspace identifier
            user_id: User making the request
            request_type: Type of request (ACCESS, ERASURE, etc.)
            data_categories: Data categories to include
            reason: Reason for request

        Returns:
            Created DSARRequest
        """
        # Create request using in-memory service
        request = self._dsar_service.create_request(
            user_id=str(user_id),
            workspace_id=str(workspace_id),
            request_type=request_type,
            data_categories=data_categories,
            reason=reason,
        )

        # Persist to database
        await self._persist_dsar_request(workspace_id, request)

        # Log audit
        self._log_audit(
            "dsar_request_created",
            workspace_id,
            {
                "request_id": request.id,
                "request_type": request_type.name,
                "user_id": str(user_id),
            },
        )

        return request

    async def get_dsar_request(self, request_id: str) -> Optional[DSARRequest]:
        """Get DSAR request by ID."""
        return self._dsar_service.get_request(request_id)

    async def verify_dsar_identity(
        self,
        request_id: str,
        verification_method: str,
        verified_by: UUID,
    ) -> bool:
        """Verify identity for DSAR request."""
        result = self._dsar_service.verify_identity(
            request_id=request_id,
            verification_method=verification_method,
            verified_by=str(verified_by),
        )

        if result:
            request = self._dsar_service.get_request(request_id)
            if request:
                await self._persist_dsar_request(
                    UUID(request.workspace_id),
                    request,
                )
                self._log_audit(
                    "dsar_identity_verified",
                    UUID(request.workspace_id),
                    {"request_id": request_id, "method": verification_method},
                )

        return result

    async def process_dsar_request(
        self,
        request_id: str,
        processed_by: UUID,
    ):
        """
        Process a DSAR request.

        This integrates with the purge scheduler for actual data operations.
        """
        from .purge_scheduler import DSARDeletionWorkflow

        request = self._dsar_service.get_request(request_id)
        if not request:
            raise ValueError(f"DSAR request {request_id} not found")

        # Process using DSAR service
        result = await self._dsar_service.process_request(
            request_id=request_id,
            processed_by=str(processed_by),
        )

        # For erasure requests, use the DB workflow
        if request.request_type == DSARRequestType.ERASURE:
            workflow = DSARDeletionWorkflow(self._session)
            deletion_results = await workflow.execute_erasure(
                user_id=UUID(request.user_id),
                workspace_id=UUID(request.workspace_id),
                data_categories=request.data_categories,
            )
            result.records_deleted = sum(deletion_results.values())

        # Update request in DB
        await self._persist_dsar_request(UUID(request.workspace_id), request)

        self._log_audit(
            "dsar_request_processed",
            UUID(request.workspace_id),
            {
                "request_id": request_id,
                "request_type": request.request_type.name,
                "records_processed": result.records_processed,
                "records_deleted": result.records_deleted,
            },
        )

        return result

    async def get_pending_dsar_requests(
        self, workspace_id: Optional[UUID] = None
    ) -> List[DSARRequest]:
        """Get pending DSAR requests."""
        requests = self._dsar_service.get_pending_requests()
        if workspace_id:
            requests = [r for r in requests if r.workspace_id == str(workspace_id)]
        return requests

    async def get_overdue_dsar_requests(self) -> List[DSARRequest]:
        """Get overdue DSAR requests."""
        return self._dsar_service.get_overdue_requests()

    async def _persist_dsar_request(
        self,
        workspace_id: UUID,
        request: DSARRequest,
    ) -> None:
        """Persist DSAR request to database."""
        from ..models import DSARRequest as DBDSARRequest

        try:
            # Check if request exists
            existing = await self._session.execute(
                select(DBDSARRequest).where(
                    DBDSARRequest.id == UUID(request.id),
                )
            )
            db_request = existing.scalar_one_or_none()

            if db_request:
                # Update existing
                db_request.status = request.status.value
                db_request.verification_method = request.verification_method
                db_request.verified_at = request.verified_at
                db_request.completed_at = request.completed_at
                db_request.export_path = request.export_path
                db_request.export_checksum = request.export_checksum
                db_request.records_processed = request.records_processed
                db_request.records_deleted = request.records_deleted
                db_request.notes = {"notes": request.notes} if request.notes else None
            else:
                # Create new
                db_request = DBDSARRequest(
                    id=UUID(request.id),
                    workspace_id=workspace_id,
                    user_id=UUID(request.user_id),
                    request_type=request.request_type.value,
                    status=request.status.value,
                    data_categories=(
                        list(request.data_categories) if request.data_categories else None
                    ),
                    reason=request.reason,
                    deadline=request.deadline,
                    extended_deadline=request.extended_deadline,
                )
                self._session.add(db_request)

            await self._session.commit()
            logger.info(f"Persisted DSAR request {request.id} for workspace {workspace_id}")

        except Exception as e:
            logger.warning(f"Failed to persist DSAR request: {e}")
            await self._session.rollback()

    # ===== Compliance Reports =====

    async def get_compliance_report(self, workspace_id: UUID) -> Dict[str, Any]:
        """
        Generate compliance report for workspace.

        Includes:
        - Residency policy status
        - Retention policy status
        - Pending/overdue DSAR requests
        - Legal holds
        """
        residency = self._residency_manager.get_policy(str(workspace_id))
        retention_policies = self._retention_service.get_workspace_policies(str(workspace_id))
        pending_dsars = await self.get_pending_dsar_requests(workspace_id)
        overdue_dsars = [r for r in pending_dsars if r.is_overdue]

        # Get retention report
        retention_report = self._retention_service.get_retention_report(str(workspace_id))

        return {
            "workspace_id": str(workspace_id),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "residency": {
                "configured": residency is not None,
                "primary_region": residency.primary_region.value if residency else None,
                "gdpr_compliant": residency.gdpr_compliant if residency else False,
                "telemetry_local": residency.telemetry_local if residency else False,
            },
            "retention": {
                "policies_count": len(retention_policies),
                "legal_holds": retention_report.get("legal_holds", []),
                "upcoming_purges": retention_report.get("upcoming_purges", []),
            },
            "dsar": {
                "pending_count": len(pending_dsars),
                "overdue_count": len(overdue_dsars),
                "overdue_requests": [r.id for r in overdue_dsars],
            },
        }

    # ===== Audit Methods =====

    def _log_audit(
        self,
        action: str,
        workspace_id: UUID,
        details: Dict[str, Any],
    ) -> None:
        """Log governance audit event."""
        entry = {
            "action": action,
            "workspace_id": str(workspace_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
        }
        self._audit_log.append(entry)

        # Trim audit log
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]

        logger.info(f"GOVERNANCE AUDIT: {action} workspace={workspace_id}")

    def get_audit_log(
        self,
        workspace_id: Optional[UUID] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get governance audit log."""
        entries = self._audit_log

        if workspace_id:
            entries = [e for e in entries if e.get("workspace_id") == str(workspace_id)]
        if action:
            entries = [e for e in entries if e.get("action") == action]

        return entries[-limit:]


# ============================================================================
# Factory
# ============================================================================


def create_governance_service(
    session: AsyncSession,
    residency_config: Optional[ResidencyConfig] = None,
    retention_config: Optional[RetentionConfig] = None,
) -> GovernanceDBService:
    """
    Create a governance DB service.

    Args:
        session: Database session
        residency_config: Optional residency configuration
        retention_config: Optional retention configuration

    Returns:
        Configured GovernanceDBService
    """
    return GovernanceDBService(
        session=session,
        residency_config=residency_config,
        retention_config=retention_config,
    )

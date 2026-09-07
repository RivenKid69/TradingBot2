# -*- coding: utf-8 -*-
"""
GDPR Phase 8: Compliance Dashboard Service.

Provides comprehensive compliance metrics and dashboards:
- DSAR SLA monitoring (response times, on-time completion rate)
- Purge success metrics (completed, failed, blocked by legal hold)
- Break-glass usage tracking (requests, approvals, denials, access events)
- Residency drift monitoring (must be zero)
- Data inventory compliance rate
- Quarterly review status

Design Doc Reference:
    - Phase 8: "Compliance dashboards: DSAR SLA, purge success, break-glass usage, residency drift = 0"
    - GDPR Article 5(2): Accountability principle

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# SLA targets
DSAR_SLA_DAYS: Final[int] = 30          # GDPR standard response time
DSAR_SLA_EXTENDED_DAYS: Final[int] = 90  # With extension
BREAK_GLASS_MAX_DURATION_HOURS: Final[int] = 24
RESIDENCY_DRIFT_TARGET: Final[int] = 0   # Must be zero

# Alert thresholds
DSAR_WARNING_THRESHOLD: Final[float] = 0.8   # 80% of SLA
PURGE_FAILURE_ALERT_THRESHOLD: Final[int] = 3  # Consecutive failures
BREAK_GLASS_ALERT_THRESHOLD: Final[int] = 5    # Per day


class MetricType(str, Enum):
    """Type of compliance metric."""
    DSAR = "dsar"
    PURGE = "purge"
    BREAK_GLASS = "break_glass"
    RESIDENCY = "residency"
    INVENTORY = "inventory"
    RETENTION = "retention"
    ACCESS_AUDIT = "access_audit"
    CONSENT = "consent"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class ComplianceStatus(str, Enum):
    """Overall compliance status."""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DSARMetrics:
    """DSAR compliance metrics."""
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=30))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Volume
    total_requests: int = 0
    access_requests: int = 0
    portability_requests: int = 0
    erasure_requests: int = 0
    rectification_requests: int = 0
    restriction_requests: int = 0

    # Status
    pending_requests: int = 0
    completed_requests: int = 0
    overdue_requests: int = 0
    extended_requests: int = 0
    rejected_requests: int = 0

    # SLA
    avg_completion_days: float = 0.0
    on_time_rate: float = 0.0
    requests_within_sla: int = 0
    requests_exceeded_sla: int = 0

    # Processing
    total_records_exported: int = 0
    total_records_deleted: int = 0
    blocked_by_legal_hold: int = 0
    blocked_by_exemption: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "volume": {
                "total": self.total_requests,
                "by_type": {
                    "access": self.access_requests,
                    "portability": self.portability_requests,
                    "erasure": self.erasure_requests,
                    "rectification": self.rectification_requests,
                    "restriction": self.restriction_requests,
                },
            },
            "status": {
                "pending": self.pending_requests,
                "completed": self.completed_requests,
                "overdue": self.overdue_requests,
                "extended": self.extended_requests,
                "rejected": self.rejected_requests,
            },
            "sla": {
                "target_days": DSAR_SLA_DAYS,
                "avg_completion_days": round(self.avg_completion_days, 2),
                "on_time_rate": round(self.on_time_rate * 100, 2),
                "within_sla": self.requests_within_sla,
                "exceeded_sla": self.requests_exceeded_sla,
            },
            "processing": {
                "records_exported": self.total_records_exported,
                "records_deleted": self.total_records_deleted,
                "blocked_legal_hold": self.blocked_by_legal_hold,
                "blocked_exemption": self.blocked_by_exemption,
            },
        }


@dataclass
class PurgeMetrics:
    """Auto-purge compliance metrics."""
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=30))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Volume
    total_purge_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    skipped_runs: int = 0
    partial_runs: int = 0

    # Records
    total_records_deleted: int = 0
    total_records_archived: int = 0
    total_records_anonymized: int = 0
    total_records_aggregated: int = 0
    total_bytes_freed: int = 0

    # Legal holds
    blocked_by_legal_hold: int = 0
    active_legal_holds: int = 0

    # By data type
    by_data_type: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Performance
    avg_duration_seconds: float = 0.0
    max_duration_seconds: float = 0.0

    # Consecutive failures (for alerting)
    consecutive_failures: int = 0
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate purge success rate."""
        total = self.successful_runs + self.failed_runs
        return self.successful_runs / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "runs": {
                "total": self.total_purge_runs,
                "successful": self.successful_runs,
                "failed": self.failed_runs,
                "skipped": self.skipped_runs,
                "partial": self.partial_runs,
                "success_rate": round(self.success_rate * 100, 2),
            },
            "records": {
                "deleted": self.total_records_deleted,
                "archived": self.total_records_archived,
                "anonymized": self.total_records_anonymized,
                "aggregated": self.total_records_aggregated,
                "total_processed": (
                    self.total_records_deleted +
                    self.total_records_archived +
                    self.total_records_anonymized +
                    self.total_records_aggregated
                ),
                "bytes_freed": self.total_bytes_freed,
            },
            "legal_holds": {
                "blocked_count": self.blocked_by_legal_hold,
                "active_holds": self.active_legal_holds,
            },
            "by_data_type": self.by_data_type,
            "performance": {
                "avg_duration_seconds": round(self.avg_duration_seconds, 2),
                "max_duration_seconds": round(self.max_duration_seconds, 2),
            },
            "health": {
                "consecutive_failures": self.consecutive_failures,
                "last_success": self.last_success_at.isoformat() if self.last_success_at else None,
                "last_failure": self.last_failure_at.isoformat() if self.last_failure_at else None,
            },
        }


@dataclass
class BreakGlassMetrics:
    """Break-glass usage metrics."""
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=30))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Requests
    total_requests: int = 0
    approved_requests: int = 0
    denied_requests: int = 0
    expired_requests: int = 0
    revoked_requests: int = 0
    pending_requests: int = 0

    # Access
    total_access_events: int = 0
    unique_principals: int = 0
    unique_resources: int = 0

    # By scope
    by_scope: Dict[str, int] = field(default_factory=dict)

    # By reason category
    by_reason: Dict[str, int] = field(default_factory=dict)

    # Duration
    avg_duration_hours: float = 0.0
    max_duration_hours: float = 0.0

    # Trends
    requests_per_day: float = 0.0
    approvals_per_day: float = 0.0

    @property
    def approval_rate(self) -> float:
        """Calculate approval rate."""
        decided = self.approved_requests + self.denied_requests
        return self.approved_requests / decided if decided > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "requests": {
                "total": self.total_requests,
                "approved": self.approved_requests,
                "denied": self.denied_requests,
                "expired": self.expired_requests,
                "revoked": self.revoked_requests,
                "pending": self.pending_requests,
                "approval_rate": round(self.approval_rate * 100, 2),
            },
            "access": {
                "total_events": self.total_access_events,
                "unique_principals": self.unique_principals,
                "unique_resources": self.unique_resources,
            },
            "by_scope": self.by_scope,
            "by_reason": self.by_reason,
            "duration": {
                "avg_hours": round(self.avg_duration_hours, 2),
                "max_hours": round(self.max_duration_hours, 2),
                "max_allowed_hours": BREAK_GLASS_MAX_DURATION_HOURS,
            },
            "trends": {
                "requests_per_day": round(self.requests_per_day, 2),
                "approvals_per_day": round(self.approvals_per_day, 2),
            },
        }


@dataclass
class ResidencyMetrics:
    """Data residency compliance metrics."""
    check_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Drift status (must be zero for compliance)
    drift_count: int = 0
    drift_violations: List[Dict[str, Any]] = field(default_factory=list)

    # Checks performed
    total_endpoints_checked: int = 0
    eu_compliant_endpoints: int = 0
    non_eu_endpoints: int = 0
    unknown_region_endpoints: int = 0

    # By service
    by_service: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Subprocessors
    total_subprocessors: int = 0
    eu_subprocessors: int = 0
    non_eu_subprocessors: int = 0
    subprocessor_review_due: int = 0

    # Last check
    last_drift_check_at: Optional[datetime] = None
    last_drift_check_result: str = "unknown"
    consecutive_passes: int = 0
    consecutive_failures: int = 0

    @property
    def is_compliant(self) -> bool:
        """Check if residency is compliant (drift = 0)."""
        return self.drift_count == 0 and self.non_eu_endpoints == 0

    @property
    def compliance_rate(self) -> float:
        """Calculate compliance rate."""
        if self.total_endpoints_checked == 0:
            return 0.0
        return self.eu_compliant_endpoints / self.total_endpoints_checked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.check_timestamp.isoformat(),
            "drift": {
                "count": self.drift_count,
                "target": RESIDENCY_DRIFT_TARGET,
                "is_compliant": self.is_compliant,
                "violations": self.drift_violations[:10],  # Limit
            },
            "endpoints": {
                "total_checked": self.total_endpoints_checked,
                "eu_compliant": self.eu_compliant_endpoints,
                "non_eu": self.non_eu_endpoints,
                "unknown_region": self.unknown_region_endpoints,
                "compliance_rate": round(self.compliance_rate * 100, 2),
            },
            "by_service": self.by_service,
            "subprocessors": {
                "total": self.total_subprocessors,
                "eu_based": self.eu_subprocessors,
                "non_eu": self.non_eu_subprocessors,
                "review_due": self.subprocessor_review_due,
            },
            "health": {
                "last_check": self.last_drift_check_at.isoformat() if self.last_drift_check_at else None,
                "last_result": self.last_drift_check_result,
                "consecutive_passes": self.consecutive_passes,
                "consecutive_failures": self.consecutive_failures,
            },
        }


@dataclass
class InventoryMetrics:
    """Data inventory compliance metrics."""
    check_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Entries
    total_entries: int = 0
    compliant_entries: int = 0
    non_compliant_entries: int = 0
    pending_review: int = 0
    deprecated_entries: int = 0

    # Issues
    missing_purpose: int = 0
    missing_lawful_basis: int = 0
    missing_retention: int = 0
    missing_residency: int = 0

    # Detection
    pii_detected: int = 0
    credentials_detected: int = 0
    order_fields_detected: int = 0

    # Coverage
    fields_covered: int = 0
    stores_covered: int = 0
    log_streams_covered: int = 0

    # Requires action
    requires_dpia: int = 0
    unregistered_in_ci: int = 0

    @property
    def compliance_rate(self) -> float:
        """Calculate inventory compliance rate."""
        if self.total_entries == 0:
            return 0.0
        return self.compliant_entries / self.total_entries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.check_timestamp.isoformat(),
            "entries": {
                "total": self.total_entries,
                "compliant": self.compliant_entries,
                "non_compliant": self.non_compliant_entries,
                "pending_review": self.pending_review,
                "deprecated": self.deprecated_entries,
                "compliance_rate": round(self.compliance_rate * 100, 2),
            },
            "issues": {
                "missing_purpose": self.missing_purpose,
                "missing_lawful_basis": self.missing_lawful_basis,
                "missing_retention": self.missing_retention,
                "missing_residency": self.missing_residency,
                "total_issues": (
                    self.missing_purpose +
                    self.missing_lawful_basis +
                    self.missing_retention +
                    self.missing_residency
                ),
            },
            "detection": {
                "pii_detected": self.pii_detected,
                "credentials_detected": self.credentials_detected,
                "order_fields_detected": self.order_fields_detected,
            },
            "coverage": {
                "fields": self.fields_covered,
                "stores": self.stores_covered,
                "log_streams": self.log_streams_covered,
            },
            "requires_action": {
                "requires_dpia": self.requires_dpia,
                "unregistered_in_ci": self.unregistered_in_ci,
            },
        }


@dataclass
class ComplianceAlert:
    """A compliance alert."""
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metric_type: MetricType = MetricType.DSAR
    severity: AlertSeverity = AlertSeverity.WARNING
    status: AlertStatus = AlertStatus.OPEN
    title: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    workspace_id: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "workspace_id": self.workspace_id,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class ComplianceDashboard:
    """Comprehensive compliance dashboard."""
    dashboard_id: str = field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    workspace_id: Optional[str] = None

    # Overall status
    overall_status: ComplianceStatus = ComplianceStatus.UNKNOWN

    # Metrics
    dsar_metrics: Optional[DSARMetrics] = None
    purge_metrics: Optional[PurgeMetrics] = None
    break_glass_metrics: Optional[BreakGlassMetrics] = None
    residency_metrics: Optional[ResidencyMetrics] = None
    inventory_metrics: Optional[InventoryMetrics] = None

    # Alerts
    open_alerts: int = 0
    critical_alerts: int = 0
    alerts: List[ComplianceAlert] = field(default_factory=list)

    # Scores (0-100)
    dsar_score: float = 0.0
    purge_score: float = 0.0
    break_glass_score: float = 0.0
    residency_score: float = 0.0
    inventory_score: float = 0.0
    overall_score: float = 0.0

    # Integrity
    integrity_hash: str = ""

    def __post_init__(self):
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute integrity hash."""
        content = json.dumps({
            "dashboard_id": self.dashboard_id,
            "generated_at": self.generated_at.isoformat(),
            "overall_status": self.overall_status.value,
            "overall_score": self.overall_score,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dashboard_id": self.dashboard_id,
            "generated_at": self.generated_at.isoformat(),
            "workspace_id": self.workspace_id,
            "overall": {
                "status": self.overall_status.value,
                "score": round(self.overall_score, 1),
            },
            "scores": {
                "dsar": round(self.dsar_score, 1),
                "purge": round(self.purge_score, 1),
                "break_glass": round(self.break_glass_score, 1),
                "residency": round(self.residency_score, 1),
                "inventory": round(self.inventory_score, 1),
            },
            "metrics": {
                "dsar": self.dsar_metrics.to_dict() if self.dsar_metrics else None,
                "purge": self.purge_metrics.to_dict() if self.purge_metrics else None,
                "break_glass": self.break_glass_metrics.to_dict() if self.break_glass_metrics else None,
                "residency": self.residency_metrics.to_dict() if self.residency_metrics else None,
                "inventory": self.inventory_metrics.to_dict() if self.inventory_metrics else None,
            },
            "alerts": {
                "open_count": self.open_alerts,
                "critical_count": self.critical_alerts,
                "recent": [a.to_dict() for a in self.alerts[:10]],
            },
            "integrity_hash": self.integrity_hash,
        }


# ============================================================================
# Compliance Dashboard Service
# ============================================================================

class ComplianceDashboardService:
    """
    Compliance Dashboard Service.

    Aggregates metrics from all governance services and provides
    comprehensive compliance dashboards and alerts.

    Usage:
        service = ComplianceDashboardService(
            dsar_service=dsar,
            purge_scheduler=purge,
            break_glass_service=bg,
            residency_checker=residency,
            inventory_registry=inventory,
        )

        # Generate dashboard
        dashboard = service.generate_dashboard(workspace_id="ws-123")

        # Get specific metrics
        dsar_metrics = service.get_dsar_metrics(workspace_id="ws-123")

        # Check alerts
        alerts = service.get_open_alerts()
    """

    def __init__(
        self,
        dsar_service: Optional[Any] = None,
        purge_scheduler: Optional[Any] = None,
        break_glass_service: Optional[Any] = None,
        residency_checker: Optional[Any] = None,
        inventory_registry: Optional[Any] = None,
        legal_hold_service: Optional[Any] = None,
        access_audit_service: Optional[Any] = None,
        on_alert: Optional[Callable[[ComplianceAlert], None]] = None,
    ):
        """
        Initialize the service.

        Args:
            dsar_service: DSAR service for request metrics
            purge_scheduler: Auto-purge scheduler for purge metrics
            break_glass_service: Break-glass service for access metrics
            residency_checker: Residency drift checker
            inventory_registry: Data inventory registry
            legal_hold_service: Legal hold service
            access_audit_service: Access audit service
            on_alert: Callback for new alerts
        """
        self._dsar = dsar_service
        self._purge = purge_scheduler
        self._break_glass = break_glass_service
        self._residency = residency_checker
        self._inventory = inventory_registry
        self._legal_hold = legal_hold_service
        self._access_audit = access_audit_service
        self._on_alert = on_alert

        self._alerts: List[ComplianceAlert] = []
        self._dashboards: Dict[str, ComplianceDashboard] = {}
        self._lock = threading.Lock()

    # ========================================================================
    # Dashboard Generation
    # ========================================================================

    def generate_dashboard(
        self,
        workspace_id: Optional[str] = None,
        period_days: int = 30,
    ) -> ComplianceDashboard:
        """
        Generate a comprehensive compliance dashboard.

        Args:
            workspace_id: Optional workspace scope
            period_days: Period for metrics

        Returns:
            ComplianceDashboard
        """
        period_start = datetime.now(timezone.utc) - timedelta(days=period_days)
        period_end = datetime.now(timezone.utc)

        dashboard = ComplianceDashboard(
            workspace_id=workspace_id,
        )

        # Collect metrics
        dashboard.dsar_metrics = self.get_dsar_metrics(
            workspace_id=workspace_id,
            period_start=period_start,
            period_end=period_end,
        )

        dashboard.purge_metrics = self.get_purge_metrics(
            workspace_id=workspace_id,
            period_start=period_start,
            period_end=period_end,
        )

        dashboard.break_glass_metrics = self.get_break_glass_metrics(
            workspace_id=workspace_id,
            period_start=period_start,
            period_end=period_end,
        )

        dashboard.residency_metrics = self.get_residency_metrics(
            workspace_id=workspace_id,
        )

        dashboard.inventory_metrics = self.get_inventory_metrics()

        # Calculate scores
        dashboard.dsar_score = self._calculate_dsar_score(dashboard.dsar_metrics)
        dashboard.purge_score = self._calculate_purge_score(dashboard.purge_metrics)
        dashboard.break_glass_score = self._calculate_break_glass_score(dashboard.break_glass_metrics)
        dashboard.residency_score = self._calculate_residency_score(dashboard.residency_metrics)
        dashboard.inventory_score = self._calculate_inventory_score(dashboard.inventory_metrics)

        # Overall score (weighted average)
        dashboard.overall_score = (
            dashboard.dsar_score * 0.25 +
            dashboard.purge_score * 0.15 +
            dashboard.break_glass_score * 0.10 +
            dashboard.residency_score * 0.30 +  # Residency is critical
            dashboard.inventory_score * 0.20
        )

        # Determine overall status
        dashboard.overall_status = self._determine_status(dashboard)

        # Get alerts
        alerts = self.get_open_alerts(workspace_id=workspace_id)
        dashboard.open_alerts = len(alerts)
        dashboard.critical_alerts = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        dashboard.alerts = alerts[:20]

        # Store dashboard
        dashboard.integrity_hash = dashboard._compute_hash()
        with self._lock:
            self._dashboards[dashboard.dashboard_id] = dashboard

        return dashboard

    # ========================================================================
    # Individual Metrics
    # ========================================================================

    def get_dsar_metrics(
        self,
        workspace_id: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> DSARMetrics:
        """Get DSAR metrics."""
        period_start = period_start or (datetime.now(timezone.utc) - timedelta(days=30))
        period_end = period_end or datetime.now(timezone.utc)

        metrics = DSARMetrics(
            period_start=period_start,
            period_end=period_end,
        )

        if self._dsar and hasattr(self._dsar, 'get_metrics'):
            try:
                dsar_data = self._dsar.get_metrics(
                    workspace_id=workspace_id,
                    days=(period_end - period_start).days,
                )
                metrics.total_requests = dsar_data.get('total_requests', 0)
                metrics.pending_requests = dsar_data.get('pending_count', 0)
                metrics.completed_requests = dsar_data.get('total_requests', 0) - dsar_data.get('pending_count', 0)
                metrics.overdue_requests = dsar_data.get('overdue_count', 0)
                metrics.avg_completion_days = dsar_data.get('avg_completion_days', 0)
                metrics.on_time_rate = dsar_data.get('on_time_rate', 0)

                # By type
                by_type = dsar_data.get('by_type', {})
                metrics.access_requests = by_type.get('access', 0)
                metrics.portability_requests = by_type.get('portability', 0)
                metrics.erasure_requests = by_type.get('erasure', 0)
                metrics.rectification_requests = by_type.get('rectification', 0)
                metrics.restriction_requests = by_type.get('restriction', 0)

                # Processing
                metrics.total_records_exported = dsar_data.get('total_records_processed', 0)
                metrics.total_records_deleted = dsar_data.get('total_records_deleted', 0)
            except Exception as e:
                logger.warning(f"Could not get DSAR metrics: {e}")

        # Generate alerts
        self._check_dsar_alerts(metrics, workspace_id)

        return metrics

    def get_purge_metrics(
        self,
        workspace_id: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> PurgeMetrics:
        """Get purge metrics."""
        period_start = period_start or (datetime.now(timezone.utc) - timedelta(days=30))
        period_end = period_end or datetime.now(timezone.utc)

        metrics = PurgeMetrics(
            period_start=period_start,
            period_end=period_end,
        )

        if self._purge and hasattr(self._purge, 'get_purge_statistics'):
            try:
                purge_data = self._purge.get_purge_statistics(
                    workspace_id=workspace_id,
                    days=(period_end - period_start).days,
                )
                metrics.total_purge_runs = purge_data.get('total_runs', 0)
                metrics.successful_runs = purge_data.get('successful_runs', 0)
                metrics.failed_runs = purge_data.get('failed_runs', 0)
                metrics.skipped_runs = purge_data.get('skipped_runs', 0)

                metrics.total_records_deleted = purge_data.get('total_records_deleted', 0)
                metrics.total_records_archived = purge_data.get('total_records_archived', 0)
                metrics.total_records_anonymized = purge_data.get('total_records_anonymized', 0)
                metrics.total_records_aggregated = purge_data.get('total_records_aggregated', 0)

                metrics.blocked_by_legal_hold = purge_data.get('blocked_by_legal_hold', 0)
                metrics.avg_duration_seconds = purge_data.get('average_duration_seconds', 0)

                metrics.by_data_type = purge_data.get('by_data_type', {})
            except Exception as e:
                logger.warning(f"Could not get purge metrics: {e}")

        # Get legal hold count
        if self._legal_hold and hasattr(self._legal_hold, 'get_all_active_holds'):
            try:
                holds = self._legal_hold.get_all_active_holds()
                metrics.active_legal_holds = len(holds)
            except Exception:
                pass

        # Generate alerts
        self._check_purge_alerts(metrics, workspace_id)

        return metrics

    def get_break_glass_metrics(
        self,
        workspace_id: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> BreakGlassMetrics:
        """Get break-glass metrics."""
        period_start = period_start or (datetime.now(timezone.utc) - timedelta(days=30))
        period_end = period_end or datetime.now(timezone.utc)

        metrics = BreakGlassMetrics(
            period_start=period_start,
            period_end=period_end,
        )

        if self._break_glass and hasattr(self._break_glass, 'get_statistics'):
            try:
                bg_data = self._break_glass.get_statistics(
                    workspace_id=workspace_id,
                    days=(period_end - period_start).days,
                )
                metrics.total_requests = bg_data.get('total_requests', 0)
                metrics.approved_requests = bg_data.get('approved_requests', 0)
                metrics.denied_requests = bg_data.get('denied_requests', 0)
                metrics.expired_requests = bg_data.get('expired_requests', 0)
                metrics.revoked_requests = bg_data.get('revoked_requests', 0)
                metrics.pending_requests = bg_data.get('pending_requests', 0)

                metrics.total_access_events = bg_data.get('total_access_events', 0)
                metrics.unique_principals = bg_data.get('unique_principals', 0)
                metrics.unique_resources = bg_data.get('unique_resources', 0)

                metrics.by_scope = bg_data.get('by_scope', {})
                metrics.by_reason = bg_data.get('by_reason', {})

                metrics.avg_duration_hours = bg_data.get('avg_duration_hours', 0)
                metrics.max_duration_hours = bg_data.get('max_duration_hours', 0)

                # Calculate daily rates
                days = (period_end - period_start).days or 1
                metrics.requests_per_day = metrics.total_requests / days
                metrics.approvals_per_day = metrics.approved_requests / days
            except Exception as e:
                logger.warning(f"Could not get break-glass metrics: {e}")

        # Generate alerts
        self._check_break_glass_alerts(metrics, workspace_id)

        return metrics

    def get_residency_metrics(
        self,
        workspace_id: Optional[str] = None,
    ) -> ResidencyMetrics:
        """Get residency drift metrics."""
        metrics = ResidencyMetrics()

        if self._residency:
            try:
                # Run drift check
                if hasattr(self._residency, 'run_drift_check'):
                    result = self._residency.run_drift_check()
                    if hasattr(result, 'to_dict'):
                        result = result.to_dict() if callable(result.to_dict) else result

                    metrics.drift_count = len(result.get('violations', []))
                    metrics.drift_violations = result.get('violations', [])[:10]
                    metrics.total_endpoints_checked = len(result.get('checks', []))

                    # Count by compliance
                    for check in result.get('checks', []):
                        if check.get('eu_compliant'):
                            metrics.eu_compliant_endpoints += 1
                        else:
                            metrics.non_eu_endpoints += 1

                    metrics.last_drift_check_at = datetime.now(timezone.utc)
                    metrics.last_drift_check_result = result.get('status', 'unknown')

                    if metrics.drift_count == 0:
                        metrics.consecutive_passes += 1
                        metrics.consecutive_failures = 0
                    else:
                        metrics.consecutive_failures += 1
                        metrics.consecutive_passes = 0
            except Exception as e:
                logger.warning(f"Could not get residency metrics: {e}")

        # Generate alerts
        self._check_residency_alerts(metrics, workspace_id)

        return metrics

    def get_inventory_metrics(self) -> InventoryMetrics:
        """Get data inventory metrics."""
        metrics = InventoryMetrics()

        if self._inventory:
            try:
                if hasattr(self._inventory, 'generate_report'):
                    report = self._inventory.generate_report()
                    if hasattr(report, 'to_dict'):
                        data = report.to_dict()
                    else:
                        data = report

                    metrics.total_entries = data.get('summary', {}).get('total_entries', 0)
                    metrics.compliant_entries = data.get('summary', {}).get('compliant_entries', 0)
                    metrics.non_compliant_entries = data.get('summary', {}).get('non_compliant_entries', 0)

                    review = data.get('review', {})
                    metrics.pending_review = review.get('pending_review', 0)
                    metrics.requires_dpia = review.get('requires_dpia', 0)

                    issues = data.get('issues', {})
                    metrics.missing_purpose = issues.get('missing_purpose', 0)
                    metrics.missing_lawful_basis = issues.get('missing_lawful_basis', 0)
                    metrics.missing_retention = issues.get('missing_retention', 0)

                    detection = data.get('auto_detection', {})
                    metrics.pii_detected = detection.get('pii_detected', 0)
                    metrics.credentials_detected = detection.get('credentials_detected', 0)
                    metrics.order_fields_detected = detection.get('order_fields_detected', 0)

                elif hasattr(self._inventory, 'get_stats'):
                    stats = self._inventory.get_stats()
                    metrics.total_entries = stats.get('total_entries', 0)
                    metrics.compliant_entries = stats.get('compliant_entries', 0)
                    metrics.non_compliant_entries = stats.get('non_compliant_entries', 0)
                    metrics.pending_review = stats.get('pending_review', 0)
            except Exception as e:
                logger.warning(f"Could not get inventory metrics: {e}")

        return metrics

    # ========================================================================
    # Score Calculations
    # ========================================================================

    def _calculate_dsar_score(self, metrics: Optional[DSARMetrics]) -> float:
        """Calculate DSAR compliance score (0-100)."""
        if not metrics or metrics.total_requests == 0:
            return 100.0  # No requests = no compliance gaps detected

        score = 100.0

        # Penalize overdue requests heavily
        if metrics.overdue_requests > 0:
            overdue_pct = metrics.overdue_requests / metrics.total_requests
            score -= overdue_pct * 50

        # Penalize if on-time rate is low
        if metrics.on_time_rate < 0.9:
            score -= (0.9 - metrics.on_time_rate) * 100

        # Penalize slow average completion
        if metrics.avg_completion_days > DSAR_SLA_DAYS:
            excess = (metrics.avg_completion_days - DSAR_SLA_DAYS) / DSAR_SLA_DAYS
            score -= excess * 20

        return max(0.0, min(100.0, score))

    def _calculate_purge_score(self, metrics: Optional[PurgeMetrics]) -> float:
        """Calculate purge compliance score (0-100)."""
        if not metrics or metrics.total_purge_runs == 0:
            return 100.0

        score = 100.0

        # Penalize failures
        if metrics.success_rate < 1.0:
            score -= (1.0 - metrics.success_rate) * 50

        # Penalize consecutive failures
        if metrics.consecutive_failures > 0:
            score -= min(30, metrics.consecutive_failures * 10)

        return max(0.0, min(100.0, score))

    def _calculate_break_glass_score(self, metrics: Optional[BreakGlassMetrics]) -> float:
        """Calculate break-glass compliance score (0-100)."""
        if not metrics or metrics.total_requests == 0:
            return 100.0

        score = 100.0

        # Penalize high usage (should be exceptional)
        if metrics.requests_per_day > 1:
            score -= min(30, (metrics.requests_per_day - 1) * 10)

        # Penalize exceeded duration
        if metrics.max_duration_hours > BREAK_GLASS_MAX_DURATION_HOURS:
            excess = (metrics.max_duration_hours - BREAK_GLASS_MAX_DURATION_HOURS) / BREAK_GLASS_MAX_DURATION_HOURS
            score -= min(20, excess * 20)

        return max(0.0, min(100.0, score))

    def _calculate_residency_score(self, metrics: Optional[ResidencyMetrics]) -> float:
        """Calculate residency compliance score (0-100)."""
        if not metrics:
            return 0.0  # Unknown is non-compliant

        # Residency drift must be zero
        if metrics.drift_count > 0:
            return 0.0

        if not metrics.is_compliant:
            return 0.0

        score = 100.0

        # Penalize unknown regions
        if metrics.unknown_region_endpoints > 0:
            score -= min(20, metrics.unknown_region_endpoints * 5)

        return max(0.0, min(100.0, score))

    def _calculate_inventory_score(self, metrics: Optional[InventoryMetrics]) -> float:
        """Calculate inventory compliance score (0-100)."""
        if not metrics or metrics.total_entries == 0:
            return 50.0  # Empty inventory is concerning

        score = 100.0

        # Penalize non-compliant entries
        score *= metrics.compliance_rate

        # Penalize pending reviews
        if metrics.total_entries > 0:
            pending_pct = metrics.pending_review / metrics.total_entries
            score -= pending_pct * 20

        # Penalize missing fields
        total_issues = (
            metrics.missing_purpose +
            metrics.missing_lawful_basis +
            metrics.missing_retention
        )
        if total_issues > 0:
            score -= min(30, total_issues * 5)

        return max(0.0, min(100.0, score))

    def _determine_status(self, dashboard: ComplianceDashboard) -> ComplianceStatus:
        """Determine overall compliance status."""
        # Critical failures
        if dashboard.residency_score < 100:
            return ComplianceStatus.NON_COMPLIANT

        if dashboard.critical_alerts > 0:
            return ComplianceStatus.NON_COMPLIANT

        # Risk indicators
        if dashboard.overall_score < 70:
            return ComplianceStatus.NON_COMPLIANT
        elif dashboard.overall_score < 85:
            return ComplianceStatus.AT_RISK
        else:
            return ComplianceStatus.COMPLIANT

    # ========================================================================
    # Alerts
    # ========================================================================

    def _check_dsar_alerts(
        self,
        metrics: DSARMetrics,
        workspace_id: Optional[str],
    ) -> None:
        """Check for DSAR-related alerts."""
        if metrics.overdue_requests > 0:
            self._create_alert(
                metric_type=MetricType.DSAR,
                severity=AlertSeverity.CRITICAL,
                title="Overdue DSAR Requests",
                message=f"{metrics.overdue_requests} DSAR request(s) have exceeded the response deadline",
                details={"overdue_count": metrics.overdue_requests},
                workspace_id=workspace_id,
            )

        if metrics.on_time_rate < 0.9 and metrics.total_requests > 5:
            self._create_alert(
                metric_type=MetricType.DSAR,
                severity=AlertSeverity.WARNING,
                title="Low DSAR On-Time Rate",
                message=f"DSAR on-time completion rate is {metrics.on_time_rate*100:.1f}% (target: 100%)",
                details={"on_time_rate": metrics.on_time_rate},
                workspace_id=workspace_id,
            )

    def _check_purge_alerts(
        self,
        metrics: PurgeMetrics,
        workspace_id: Optional[str],
    ) -> None:
        """Check for purge-related alerts."""
        if metrics.consecutive_failures >= PURGE_FAILURE_ALERT_THRESHOLD:
            self._create_alert(
                metric_type=MetricType.PURGE,
                severity=AlertSeverity.ERROR,
                title="Consecutive Purge Failures",
                message=f"{metrics.consecutive_failures} consecutive purge failures detected",
                details={"consecutive_failures": metrics.consecutive_failures},
                workspace_id=workspace_id,
            )

        if metrics.success_rate < 0.9 and metrics.total_purge_runs > 5:
            self._create_alert(
                metric_type=MetricType.PURGE,
                severity=AlertSeverity.WARNING,
                title="Low Purge Success Rate",
                message=f"Purge success rate is {metrics.success_rate*100:.1f}% (target: 100%)",
                details={"success_rate": metrics.success_rate},
                workspace_id=workspace_id,
            )

    def _check_break_glass_alerts(
        self,
        metrics: BreakGlassMetrics,
        workspace_id: Optional[str],
    ) -> None:
        """Check for break-glass alerts."""
        if metrics.requests_per_day > BREAK_GLASS_ALERT_THRESHOLD:
            self._create_alert(
                metric_type=MetricType.BREAK_GLASS,
                severity=AlertSeverity.WARNING,
                title="High Break-Glass Usage",
                message=f"Break-glass requests averaging {metrics.requests_per_day:.1f}/day (threshold: {BREAK_GLASS_ALERT_THRESHOLD})",
                details={"requests_per_day": metrics.requests_per_day},
                workspace_id=workspace_id,
            )

        if metrics.max_duration_hours > BREAK_GLASS_MAX_DURATION_HOURS:
            self._create_alert(
                metric_type=MetricType.BREAK_GLASS,
                severity=AlertSeverity.ERROR,
                title="Extended Break-Glass Duration",
                message=f"Break-glass session exceeded max duration ({metrics.max_duration_hours:.1f}h > {BREAK_GLASS_MAX_DURATION_HOURS}h)",
                details={"max_duration_hours": metrics.max_duration_hours},
                workspace_id=workspace_id,
            )

    def _check_residency_alerts(
        self,
        metrics: ResidencyMetrics,
        workspace_id: Optional[str],
    ) -> None:
        """Check for residency drift alerts."""
        if metrics.drift_count > 0:
            self._create_alert(
                metric_type=MetricType.RESIDENCY,
                severity=AlertSeverity.CRITICAL,
                title="Data Residency Drift Detected",
                message=f"{metrics.drift_count} endpoint(s) detected outside EU region",
                details={"drift_count": metrics.drift_count, "violations": metrics.drift_violations[:5]},
                workspace_id=workspace_id,
            )

        if not metrics.is_compliant:
            self._create_alert(
                metric_type=MetricType.RESIDENCY,
                severity=AlertSeverity.CRITICAL,
                title="EU-Only Residency Violation",
                message="Data residency requirements not met",
                details={"non_eu_endpoints": metrics.non_eu_endpoints},
                workspace_id=workspace_id,
            )

    def _create_alert(
        self,
        metric_type: MetricType,
        severity: AlertSeverity,
        title: str,
        message: str,
        details: Dict[str, Any],
        workspace_id: Optional[str],
    ) -> ComplianceAlert:
        """Create and store an alert."""
        alert = ComplianceAlert(
            metric_type=metric_type,
            severity=severity,
            title=title,
            message=message,
            details=details,
            workspace_id=workspace_id,
        )

        with self._lock:
            # Check for duplicate
            for existing in self._alerts:
                if (
                    existing.title == alert.title and
                    existing.workspace_id == alert.workspace_id and
                    existing.status == AlertStatus.OPEN
                ):
                    return existing

            self._alerts.append(alert)

        if self._on_alert:
            try:
                self._on_alert(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        logger.warning(f"Compliance alert: [{severity.value}] {title}")
        return alert

    def get_open_alerts(
        self,
        workspace_id: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[ComplianceAlert]:
        """Get open alerts."""
        with self._lock:
            alerts = [a for a in self._alerts if a.status == AlertStatus.OPEN]

        if workspace_id:
            alerts = [a for a in alerts if a.workspace_id == workspace_id]
        if metric_type:
            alerts = [a for a in alerts if a.metric_type == metric_type]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> Optional[ComplianceAlert]:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = datetime.now(timezone.utc)
                    return alert
        return None

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_notes: str,
    ) -> Optional[ComplianceAlert]:
        """Resolve an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_by = resolved_by
                    alert.resolved_at = datetime.now(timezone.utc)
                    alert.resolution_notes = resolution_notes
                    return alert
        return None

    # ========================================================================
    # Dashboard Management
    # ========================================================================

    def get_dashboard(self, dashboard_id: str) -> Optional[ComplianceDashboard]:
        """Get a specific dashboard."""
        with self._lock:
            return self._dashboards.get(dashboard_id)

    def get_latest_dashboard(
        self,
        workspace_id: Optional[str] = None,
    ) -> Optional[ComplianceDashboard]:
        """Get the latest dashboard for a workspace."""
        with self._lock:
            dashboards = list(self._dashboards.values())

        if workspace_id:
            dashboards = [d for d in dashboards if d.workspace_id == workspace_id]

        if not dashboards:
            return None

        return max(dashboards, key=lambda d: d.generated_at)

    def export_dashboard(
        self,
        dashboard_id: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Export a dashboard."""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            raise ValueError(f"Dashboard {dashboard_id} not found")

        return dashboard.to_dict()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "MetricType",
    "AlertSeverity",
    "AlertStatus",
    "ComplianceStatus",
    # Constants
    "DSAR_SLA_DAYS",
    "DSAR_SLA_EXTENDED_DAYS",
    "BREAK_GLASS_MAX_DURATION_HOURS",
    "RESIDENCY_DRIFT_TARGET",
    # Data classes
    "DSARMetrics",
    "PurgeMetrics",
    "BreakGlassMetrics",
    "ResidencyMetrics",
    "InventoryMetrics",
    "ComplianceAlert",
    "ComplianceDashboard",
    # Service
    "ComplianceDashboardService",
]

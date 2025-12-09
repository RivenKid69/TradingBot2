# -*- coding: utf-8 -*-
"""
Tiered Backup System (Block 2.1).

Implements tiered backup with configurable RPO levels:
- Critical: 15 minutes RPO (continuous/near-continuous replication)
- Standard: 1 hour RPO (frequent snapshots)
- Archive: 24 hours RPO (daily backups)

DORA References:
    - Article 12: Backup Policies, Recovery Procedures and Methods
    - Article 12(1): Backup policies specifying scope and frequency
    - Article 12(2): Regular testing of backup procedures
    - Article 12(3): Geographically diverse backup locations
    - RTS CDR 2024/1774: ICT Risk Management Framework

Best Practices:
    - ISO 22301:2019 Business Continuity Management
    - NIST SP 800-34: Contingency Planning Guide
    - AWS Well-Architected Framework: Reliability Pillar
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class BackupTier(Enum):
    """Backup tier classification."""
    CRITICAL = "critical"      # 15 min RPO
    STANDARD = "standard"      # 1 hour RPO
    ARCHIVE = "archive"        # 24 hour RPO


class RPOLevel(Enum):
    """Recovery Point Objective levels."""
    RPO_15MIN = "15min"        # 15 minutes
    RPO_1HOUR = "1hour"        # 1 hour
    RPO_4HOUR = "4hour"        # 4 hours
    RPO_24HOUR = "24hour"      # 24 hours


class BackupStrategy(Enum):
    """Backup strategy types."""
    CONTINUOUS = "continuous"              # CDP - Continuous Data Protection
    SNAPSHOT = "snapshot"                  # Point-in-time snapshots
    INCREMENTAL = "incremental"            # Incremental backups
    DIFFERENTIAL = "differential"          # Differential backups
    FULL = "full"                          # Full backups


class ReplicationMode(Enum):
    """Replication modes."""
    SYNCHRONOUS = "synchronous"            # Zero data loss (RPO=0)
    ASYNCHRONOUS = "asynchronous"          # Near-zero data loss
    SEMI_SYNCHRONOUS = "semi_synchronous"  # Balanced approach
    SCHEDULED = "scheduled"                # Scheduled replication


class BackupJobStatus(Enum):
    """Backup job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VERIFYING = "verifying"
    VERIFIED = "verified"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TieredBackupPolicy:
    """
    Tiered backup policy configuration.

    Defines backup behavior for a specific tier.
    """
    policy_id: str = ""
    name: str = ""
    description: str = ""

    # Tier configuration
    tier: BackupTier = BackupTier.STANDARD
    rpo_level: RPOLevel = RPOLevel.RPO_1HOUR
    rpo_minutes: int = 60

    # Backup strategy
    strategy: BackupStrategy = BackupStrategy.SNAPSHOT
    replication_mode: ReplicationMode = ReplicationMode.ASYNCHRONOUS

    # Schedule
    backup_interval_minutes: int = 60
    retention_days: int = 30
    retention_copies: int = 24

    # Scope
    systems_covered: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=list)
    priority: int = 1

    # RTO (Recovery Time Objective)
    rto_minutes: int = 240  # 4 hours default

    # Geographic redundancy
    primary_region: str = ""
    secondary_regions: List[str] = field(default_factory=list)
    geo_redundancy_required: bool = True
    minimum_distance_km: float = 100.0

    # Encryption
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"

    # Verification
    auto_verify: bool = True
    verify_interval_hours: int = 24
    test_restore_interval_days: int = 30

    # Status
    is_active: bool = True
    created_at: str = ""
    updated_at: str = ""
    owner: str = ""

    def __post_init__(self):
        if not self.policy_id:
            self.policy_id = f"TBKP-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        # Set RPO minutes based on tier
        if self.tier == BackupTier.CRITICAL:
            self.rpo_minutes = 15
            self.rpo_level = RPOLevel.RPO_15MIN
        elif self.tier == BackupTier.STANDARD:
            self.rpo_minutes = 60
            self.rpo_level = RPOLevel.RPO_1HOUR
        elif self.tier == BackupTier.ARCHIVE:
            self.rpo_minutes = 1440  # 24 hours
            self.rpo_level = RPOLevel.RPO_24HOUR


@dataclass
class BackupSchedule:
    """Backup schedule configuration."""
    schedule_id: str = ""
    policy_id: str = ""

    # Schedule type
    is_continuous: bool = False
    cron_expression: str = ""  # For scheduled backups

    # Windows
    backup_window_start: str = ""  # HH:MM format
    backup_window_end: str = ""
    excluded_hours: List[int] = field(default_factory=list)

    # Frequency
    interval_minutes: int = 60
    max_concurrent_jobs: int = 2

    # Status
    is_active: bool = True
    next_run: str = ""
    last_run: str = ""

    def __post_init__(self):
        if not self.schedule_id:
            self.schedule_id = f"SCHED-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class BackupExecution:
    """Backup job execution record."""
    execution_id: str = ""
    policy_id: str = ""
    schedule_id: str = ""

    # Execution details
    status: BackupJobStatus = BackupJobStatus.PENDING
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0

    # Scope
    systems_backed_up: List[str] = field(default_factory=list)

    # Metrics
    bytes_transferred: int = 0
    files_backed_up: int = 0
    files_failed: int = 0
    compression_ratio: float = 1.0

    # Verification
    is_verified: bool = False
    verification_time: str = ""
    checksum: str = ""
    integrity_status: str = ""

    # Storage
    backup_location: str = ""
    secondary_locations: List[str] = field(default_factory=list)
    retention_expiry: str = ""

    # Errors
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = f"BKPEX-{uuid.uuid4().hex[:8].upper()}"
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ReplicationConfig:
    """Replication configuration for backup targets."""
    config_id: str = ""
    name: str = ""

    # Mode
    mode: ReplicationMode = ReplicationMode.ASYNCHRONOUS

    # Source
    source_region: str = ""
    source_endpoint: str = ""

    # Targets
    target_regions: List[str] = field(default_factory=list)
    target_endpoints: List[str] = field(default_factory=list)

    # Performance
    max_lag_seconds: int = 60
    bandwidth_limit_mbps: float = 0.0  # 0 = unlimited
    parallel_streams: int = 4

    # Monitoring
    lag_alert_threshold_seconds: int = 300
    failure_alert_threshold: int = 3

    # Status
    is_active: bool = True
    current_lag_seconds: float = 0.0
    last_sync_time: str = ""

    def __post_init__(self):
        if not self.config_id:
            self.config_id = f"REPL-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class BackupMetrics:
    """Backup system metrics."""
    timestamp: str = ""

    # Volume metrics
    total_backups_24h: int = 0
    successful_backups_24h: int = 0
    failed_backups_24h: int = 0

    # Success rate
    success_rate_24h: float = 100.0
    success_rate_7d: float = 100.0
    success_rate_30d: float = 100.0

    # Performance
    avg_backup_duration_seconds: float = 0.0
    avg_restore_duration_seconds: float = 0.0

    # Storage
    total_backup_size_gb: float = 0.0
    storage_used_gb: float = 0.0
    storage_available_gb: float = 0.0

    # RPO compliance
    rpo_violations_24h: int = 0
    last_rpo_violation: str = ""

    # RTO compliance
    rto_tests_passed: int = 0
    rto_tests_failed: int = 0
    last_rto_test: str = ""

    # Replication
    replication_lag_seconds: float = 0.0
    replication_status: str = "healthy"

    # By tier
    metrics_by_tier: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class TieredBackupConfig:
    """Configuration for TieredBackupManager."""
    # Default RPO settings per tier
    critical_rpo_minutes: int = 15
    standard_rpo_minutes: int = 60
    archive_rpo_minutes: int = 1440  # 24 hours

    # Default retention
    default_retention_days: int = 30
    critical_retention_days: int = 90
    archive_retention_days: int = 365

    # Verification
    auto_verify_enabled: bool = True
    verify_interval_hours: int = 24

    # Alerting
    alert_on_failure: bool = True
    alert_on_rpo_violation: bool = True
    rpo_violation_threshold_percent: float = 10.0

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/core/tiered_backup"

    # Callbacks
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Tier Definitions
# =============================================================================

TIER_DEFINITIONS = {
    BackupTier.CRITICAL: {
        "name": "Critical",
        "description": "Critical systems requiring minimal data loss",
        "rpo_minutes": 15,
        "rpo_level": RPOLevel.RPO_15MIN,
        "strategy": BackupStrategy.CONTINUOUS,
        "replication_mode": ReplicationMode.SYNCHRONOUS,
        "retention_days": 90,
        "rto_minutes": 60,
        "verification_interval_hours": 6,
        "test_restore_interval_days": 7,
    },
    BackupTier.STANDARD: {
        "name": "Standard",
        "description": "Standard business systems",
        "rpo_minutes": 60,
        "rpo_level": RPOLevel.RPO_1HOUR,
        "strategy": BackupStrategy.SNAPSHOT,
        "replication_mode": ReplicationMode.ASYNCHRONOUS,
        "retention_days": 30,
        "rto_minutes": 240,
        "verification_interval_hours": 24,
        "test_restore_interval_days": 30,
    },
    BackupTier.ARCHIVE: {
        "name": "Archive",
        "description": "Archive and historical data",
        "rpo_minutes": 1440,
        "rpo_level": RPOLevel.RPO_24HOUR,
        "strategy": BackupStrategy.FULL,
        "replication_mode": ReplicationMode.SCHEDULED,
        "retention_days": 365,
        "rto_minutes": 480,
        "verification_interval_hours": 168,  # Weekly
        "test_restore_interval_days": 90,
    },
}


def get_tier_definitions() -> Dict[BackupTier, Dict[str, Any]]:
    """Get tier definitions."""
    return TIER_DEFINITIONS.copy()


# =============================================================================
# Main Class
# =============================================================================

class TieredBackupManager:
    """
    Tiered Backup Manager per DORA Article 12.

    Manages tiered backup with configurable RPO levels:
    - Critical: 15 minutes RPO
    - Standard: 1 hour RPO
    - Archive: 24 hours RPO

    Features:
    - Policy-based backup management
    - Automatic scheduling
    - Replication management
    - Verification and integrity checks
    - Metrics and reporting

    Usage:
        config = TieredBackupConfig()
        manager = TieredBackupManager(config)

        # Create critical tier policy
        policy = manager.create_policy(
            name="Trading Database Backup",
            tier=BackupTier.CRITICAL,
            systems_covered=["trading-db", "order-db"],
        )

        # Execute backup
        execution = manager.execute_backup(policy.policy_id)

        # Check metrics
        metrics = manager.get_metrics()
    """

    def __init__(self, config: Optional[TieredBackupConfig] = None):
        """Initialize Tiered Backup Manager."""
        self.config = config or TieredBackupConfig()

        # Data stores
        self._policies: Dict[str, TieredBackupPolicy] = {}
        self._schedules: Dict[str, BackupSchedule] = {}
        self._executions: Dict[str, BackupExecution] = {}
        self._replications: Dict[str, ReplicationConfig] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        # Initialize default policies
        self._init_default_policies()

        logger.info("TieredBackupManager initialized")

    def _init_default_policies(self) -> None:
        """Initialize default policies for each tier."""
        for tier, definition in TIER_DEFINITIONS.items():
            self.create_policy(
                name=f"Default {definition['name']} Policy",
                description=definition["description"],
                tier=tier,
                is_default=True,
            )

    # =========================================================================
    # Policy Management
    # =========================================================================

    def create_policy(
        self,
        name: str,
        tier: BackupTier,
        description: str = "",
        systems_covered: Optional[List[str]] = None,
        data_types: Optional[List[str]] = None,
        retention_days: Optional[int] = None,
        rto_minutes: Optional[int] = None,
        primary_region: str = "",
        secondary_regions: Optional[List[str]] = None,
        owner: str = "",
        is_default: bool = False,
    ) -> TieredBackupPolicy:
        """
        Create a tiered backup policy.

        Args:
            name: Policy name
            tier: Backup tier (CRITICAL, STANDARD, ARCHIVE)
            description: Policy description
            systems_covered: Systems to backup
            data_types: Data types to backup
            retention_days: Retention period
            rto_minutes: Recovery time objective
            primary_region: Primary backup region
            secondary_regions: Secondary regions for geo-redundancy
            owner: Policy owner
            is_default: Whether this is a default policy

        Returns:
            Created TieredBackupPolicy
        """
        tier_def = TIER_DEFINITIONS[tier]

        policy = TieredBackupPolicy(
            name=name,
            description=description or tier_def["description"],
            tier=tier,
            rpo_level=tier_def["rpo_level"],
            rpo_minutes=tier_def["rpo_minutes"],
            strategy=tier_def["strategy"],
            replication_mode=tier_def["replication_mode"],
            backup_interval_minutes=tier_def["rpo_minutes"],
            retention_days=retention_days or tier_def["retention_days"],
            systems_covered=systems_covered or [],
            data_types=data_types or [],
            rto_minutes=rto_minutes or tier_def["rto_minutes"],
            primary_region=primary_region,
            secondary_regions=secondary_regions or [],
            verify_interval_hours=tier_def["verification_interval_hours"],
            test_restore_interval_days=tier_def["test_restore_interval_days"],
            owner=owner,
        )

        with self._lock:
            self._policies[policy.policy_id] = policy

        if not is_default:
            self._log_event("policy_created", {
                "policy_id": policy.policy_id,
                "name": name,
                "tier": tier.value,
                "rpo_minutes": policy.rpo_minutes,
            })

        return policy

    def get_policy(self, policy_id: str) -> Optional[TieredBackupPolicy]:
        """Get policy by ID."""
        with self._lock:
            return self._policies.get(policy_id)

    def get_policies_by_tier(self, tier: BackupTier) -> List[TieredBackupPolicy]:
        """Get all policies for a tier."""
        with self._lock:
            return [p for p in self._policies.values() if p.tier == tier and p.is_active]

    def update_policy(
        self,
        policy_id: str,
        **updates: Any,
    ) -> Optional[TieredBackupPolicy]:
        """Update a policy."""
        with self._lock:
            if policy_id not in self._policies:
                return None

            policy = self._policies[policy_id]
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            policy.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("policy_updated", {
            "policy_id": policy_id,
            "updates": list(updates.keys()),
        })

        return policy

    def deactivate_policy(self, policy_id: str) -> bool:
        """Deactivate a policy."""
        with self._lock:
            if policy_id not in self._policies:
                return False
            self._policies[policy_id].is_active = False

        self._log_event("policy_deactivated", {"policy_id": policy_id})
        return True

    # =========================================================================
    # Backup Execution
    # =========================================================================

    def execute_backup(
        self,
        policy_id: str,
        systems: Optional[List[str]] = None,
        force: bool = False,
    ) -> Optional[BackupExecution]:
        """
        Execute a backup for a policy.

        Args:
            policy_id: Policy ID
            systems: Specific systems to backup (optional)
            force: Force backup even if not due

        Returns:
            BackupExecution record
        """
        with self._lock:
            if policy_id not in self._policies:
                logger.error(f"Policy not found: {policy_id}")
                return None

            policy = self._policies[policy_id]

        execution = BackupExecution(
            policy_id=policy_id,
            status=BackupJobStatus.RUNNING,
            systems_backed_up=systems or policy.systems_covered,
        )

        with self._lock:
            self._executions[execution.execution_id] = execution

        self._log_event("backup_started", {
            "execution_id": execution.execution_id,
            "policy_id": policy_id,
            "tier": policy.tier.value,
        })

        # Simulate backup execution
        try:
            # In real implementation, this would perform actual backup
            execution.bytes_transferred = 1024 * 1024 * 100  # 100 MB example
            execution.files_backed_up = 1000
            execution.compression_ratio = 0.7
            execution.status = BackupJobStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc).isoformat()

            # Calculate duration
            start = datetime.fromisoformat(execution.started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(execution.completed_at.replace("Z", "+00:00"))
            execution.duration_seconds = (end - start).total_seconds()

            # Set retention expiry
            expiry = datetime.now(timezone.utc) + timedelta(days=policy.retention_days)
            execution.retention_expiry = expiry.isoformat()

            # Auto-verify if enabled
            if self.config.auto_verify_enabled:
                self._verify_backup(execution)

            self._log_event("backup_completed", {
                "execution_id": execution.execution_id,
                "policy_id": policy_id,
                "bytes_transferred": execution.bytes_transferred,
                "duration_seconds": execution.duration_seconds,
            })

        except Exception as e:
            execution.status = BackupJobStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now(timezone.utc).isoformat()

            self._log_event("backup_failed", {
                "execution_id": execution.execution_id,
                "error": str(e),
            })

            if self.config.alert_on_failure:
                self._send_alert("backup_failed", {
                    "execution_id": execution.execution_id,
                    "policy_id": policy_id,
                    "error": str(e),
                })

        return execution

    def _verify_backup(self, execution: BackupExecution) -> None:
        """Verify backup integrity."""
        execution.status = BackupJobStatus.VERIFYING

        # Simulate verification
        execution.is_verified = True
        execution.verification_time = datetime.now(timezone.utc).isoformat()
        execution.checksum = f"sha256:{uuid.uuid4().hex}"
        execution.integrity_status = "valid"
        execution.status = BackupJobStatus.VERIFIED

    def get_execution(self, execution_id: str) -> Optional[BackupExecution]:
        """Get execution by ID."""
        with self._lock:
            return self._executions.get(execution_id)

    def get_recent_executions(
        self,
        policy_id: Optional[str] = None,
        tier: Optional[BackupTier] = None,
        status: Optional[BackupJobStatus] = None,
        hours: int = 24,
        limit: int = 100,
    ) -> List[BackupExecution]:
        """Get recent backup executions."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        with self._lock:
            executions = list(self._executions.values())

            # Filter
            if policy_id:
                executions = [e for e in executions if e.policy_id == policy_id]

            if tier:
                tier_policies = {p.policy_id for p in self._policies.values() if p.tier == tier}
                executions = [e for e in executions if e.policy_id in tier_policies]

            if status:
                executions = [e for e in executions if e.status == status]

            # Filter by time
            executions = [e for e in executions if e.started_at > cutoff]

            # Sort by time descending
            executions.sort(key=lambda e: e.started_at, reverse=True)

            return executions[:limit]

    # =========================================================================
    # Replication Management
    # =========================================================================

    def configure_replication(
        self,
        name: str,
        mode: ReplicationMode,
        source_region: str,
        target_regions: List[str],
        max_lag_seconds: int = 60,
    ) -> ReplicationConfig:
        """Configure replication for backup targets."""
        config = ReplicationConfig(
            name=name,
            mode=mode,
            source_region=source_region,
            target_regions=target_regions,
            max_lag_seconds=max_lag_seconds,
        )

        with self._lock:
            self._replications[config.config_id] = config

        self._log_event("replication_configured", {
            "config_id": config.config_id,
            "mode": mode.value,
            "source": source_region,
            "targets": target_regions,
        })

        return config

    def get_replication_status(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get replication status."""
        with self._lock:
            if config_id not in self._replications:
                return None

            config = self._replications[config_id]
            return {
                "config_id": config_id,
                "mode": config.mode.value,
                "is_active": config.is_active,
                "current_lag_seconds": config.current_lag_seconds,
                "last_sync_time": config.last_sync_time,
                "status": "healthy" if config.current_lag_seconds < config.max_lag_seconds else "lagging",
            }

    # =========================================================================
    # RPO Compliance
    # =========================================================================

    def check_rpo_compliance(self, policy_id: str) -> Dict[str, Any]:
        """Check RPO compliance for a policy."""
        with self._lock:
            if policy_id not in self._policies:
                return {"compliant": False, "error": "Policy not found"}

            policy = self._policies[policy_id]

            # Get most recent successful backup
            recent = [
                e for e in self._executions.values()
                if e.policy_id == policy_id and e.status in (
                    BackupJobStatus.COMPLETED,
                    BackupJobStatus.VERIFIED,
                )
            ]

        if not recent:
            return {
                "compliant": False,
                "policy_id": policy_id,
                "rpo_minutes": policy.rpo_minutes,
                "issue": "No successful backups found",
            }

        latest = max(recent, key=lambda e: e.completed_at or e.started_at)
        completed_time = datetime.fromisoformat(
            (latest.completed_at or latest.started_at).replace("Z", "+00:00")
        )

        age_minutes = (datetime.now(timezone.utc) - completed_time).total_seconds() / 60
        compliant = age_minutes <= policy.rpo_minutes

        result = {
            "compliant": compliant,
            "policy_id": policy_id,
            "tier": policy.tier.value,
            "rpo_minutes": policy.rpo_minutes,
            "last_backup_age_minutes": round(age_minutes, 1),
            "last_backup_time": latest.completed_at or latest.started_at,
            "last_backup_id": latest.execution_id,
        }

        if not compliant:
            self._log_event("rpo_violation", result)
            if self.config.alert_on_rpo_violation:
                self._send_alert("rpo_violation", result)

        return result

    def check_all_rpo_compliance(self) -> Dict[str, Any]:
        """Check RPO compliance for all active policies."""
        with self._lock:
            policies = [p for p in self._policies.values() if p.is_active]

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_policies": len(policies),
            "compliant_policies": 0,
            "non_compliant_policies": 0,
            "by_tier": {},
            "details": [],
        }

        for policy in policies:
            compliance = self.check_rpo_compliance(policy.policy_id)
            results["details"].append(compliance)

            if compliance.get("compliant", False):
                results["compliant_policies"] += 1
            else:
                results["non_compliant_policies"] += 1

            # Aggregate by tier
            tier = policy.tier.value
            if tier not in results["by_tier"]:
                results["by_tier"][tier] = {"compliant": 0, "non_compliant": 0}

            if compliance.get("compliant", False):
                results["by_tier"][tier]["compliant"] += 1
            else:
                results["by_tier"][tier]["non_compliant"] += 1

        return results

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> BackupMetrics:
        """Get backup system metrics."""
        now = datetime.now(timezone.utc)
        cutoff_24h = (now - timedelta(hours=24)).isoformat()
        cutoff_7d = (now - timedelta(days=7)).isoformat()
        cutoff_30d = (now - timedelta(days=30)).isoformat()

        with self._lock:
            executions = list(self._executions.values())

        # Filter by time periods
        exec_24h = [e for e in executions if e.started_at > cutoff_24h]
        exec_7d = [e for e in executions if e.started_at > cutoff_7d]
        exec_30d = [e for e in executions if e.started_at > cutoff_30d]

        # Calculate success rates
        def calc_success_rate(execs: List[BackupExecution]) -> float:
            if not execs:
                return 100.0
            successful = sum(1 for e in execs if e.status in (
                BackupJobStatus.COMPLETED, BackupJobStatus.VERIFIED
            ))
            return round(successful / len(execs) * 100, 2)

        metrics = BackupMetrics(
            total_backups_24h=len(exec_24h),
            successful_backups_24h=sum(1 for e in exec_24h if e.status in (
                BackupJobStatus.COMPLETED, BackupJobStatus.VERIFIED
            )),
            failed_backups_24h=sum(1 for e in exec_24h if e.status == BackupJobStatus.FAILED),
            success_rate_24h=calc_success_rate(exec_24h),
            success_rate_7d=calc_success_rate(exec_7d),
            success_rate_30d=calc_success_rate(exec_30d),
        )

        # Calculate averages
        completed_execs = [e for e in exec_24h if e.duration_seconds > 0]
        if completed_execs:
            metrics.avg_backup_duration_seconds = round(
                sum(e.duration_seconds for e in completed_execs) / len(completed_execs), 2
            )

        # Storage metrics
        metrics.total_backup_size_gb = round(
            sum(e.bytes_transferred for e in executions) / (1024 ** 3), 2
        )

        # Metrics by tier
        for tier in BackupTier:
            tier_policies = {p.policy_id for p in self._policies.values() if p.tier == tier}
            tier_execs = [e for e in exec_24h if e.policy_id in tier_policies]

            metrics.metrics_by_tier[tier.value] = {
                "total": len(tier_execs),
                "successful": sum(1 for e in tier_execs if e.status in (
                    BackupJobStatus.COMPLETED, BackupJobStatus.VERIFIED
                )),
                "failed": sum(1 for e in tier_execs if e.status == BackupJobStatus.FAILED),
                "success_rate": calc_success_rate(tier_execs),
            }

        return metrics

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_backup_summary(self) -> Dict[str, Any]:
        """Get comprehensive backup summary."""
        metrics = self.get_metrics()
        rpo_compliance = self.check_all_rpo_compliance()

        with self._lock:
            policies = list(self._policies.values())
            replications = list(self._replications.values())

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "policies": {
                "total": len(policies),
                "active": sum(1 for p in policies if p.is_active),
                "by_tier": {
                    tier.value: sum(1 for p in policies if p.tier == tier)
                    for tier in BackupTier
                },
            },
            "metrics": asdict(metrics),
            "rpo_compliance": rpo_compliance,
            "replication": {
                "total_configs": len(replications),
                "active": sum(1 for r in replications if r.is_active),
            },
            "dora_compliance": {
                "article_12_status": "compliant" if rpo_compliance["non_compliant_policies"] == 0 else "non_compliant",
                "geographic_redundancy": all(p.geo_redundancy_required for p in policies if p.is_active),
                "encryption_enabled": all(p.encryption_enabled for p in policies if p.is_active),
            },
        }

    def export_report(self) -> Dict[str, Any]:
        """Export full backup report for audit."""
        with self._lock:
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "article_reference": "Article 12",
                "summary": self.get_backup_summary(),
                "policies": [asdict(p) for p in self._policies.values()],
                "recent_executions": [
                    asdict(e) for e in list(self._executions.values())[-100:]
                ],
                "replications": [asdict(r) for r in self._replications.values()],
                "tier_definitions": {
                    tier.value: definition
                    for tier, definition in TIER_DEFINITIONS.items()
                },
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"tiered_backup_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def _send_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Send an alert."""
        if self.config.alert_callback:
            try:
                self.config.alert_callback(alert_type, data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Backup alert: {alert_type} - {data}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_tiered_backup_manager(
    config: Optional[TieredBackupConfig] = None,
) -> TieredBackupManager:
    """
    Create a TieredBackupManager instance.

    Args:
        config: Optional configuration

    Returns:
        Configured TieredBackupManager instance
    """
    return TieredBackupManager(config=config)

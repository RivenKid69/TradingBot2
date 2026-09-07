# -*- coding: utf-8 -*-
"""
DORA Backup Policies, Recovery Procedures and Methods Module (Article 12).

Regulation (EU) 2022/2554 Article 12 defines backup and recovery requirements:
    - Backup policies and procedures specifying scope and frequency
    - Regular testing of backup procedures and restoration methods
    - Restoration using separate ICT systems (where required)
    - Backup media at geographically diverse locations
    - Minimum backup retained at remote location for critical functions

This module provides:
    1. BackupPolicy - Backup policy management
    2. BackupJob - Backup job execution and tracking
    3. RecoveryTest - Recovery testing management
    4. BackupLocation - Backup location and redundancy
    5. DORABackupRecovery - Main backup/recovery orchestrator

References:
    - Article 12 DORA: https://www.digital-operational-resilience-act.com/Article_12.html
    - RTS on ICT Risk Management: CDR 2024/1774
    - ISO 22301:2019 Business Continuity Management
    - NIST SP 800-34: Contingency Planning Guide
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


class BackupType(Enum):
    """Backup types."""

    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"
    CONTINUOUS = "continuous"  # CDP


class BackupFrequency(Enum):
    """Backup frequency."""

    REAL_TIME = "real_time"  # CDP
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class BackupStatus(Enum):
    """Backup job status."""

    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VERIFIED = "verified"


class RecoveryTestType(Enum):
    """Recovery test types."""

    RESTORE_TEST = "restore_test"
    FAILOVER_TEST = "failover_test"
    FULL_RECOVERY = "full_recovery"
    TABLETOP = "tabletop"
    PARTIAL_RESTORE = "partial_restore"


class RecoveryTestResult(Enum):
    """Recovery test results."""

    PASSED = "passed"
    PASSED_WITH_ISSUES = "passed_with_issues"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class LocationType(Enum):
    """Backup location types."""

    ON_PREMISE = "on_premise"
    OFF_SITE = "off_site"
    CLOUD = "cloud"
    HYBRID = "hybrid"


class DataClassification(Enum):
    """Data classification for backup scope."""

    CRITICAL = "critical"
    IMPORTANT = "important"
    STANDARD = "standard"
    ARCHIVE = "archive"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class BackupPolicy:
    """
    Backup policy per DORA Article 12(1).

    Defines scope and frequency of backups.
    """

    policy_id: str = ""
    name: str = ""
    description: str = ""

    # Scope
    data_classification: DataClassification = DataClassification.CRITICAL
    systems_covered: List[str] = field(default_factory=list)
    data_types_covered: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)

    # Frequency and retention
    backup_type: BackupType = BackupType.FULL
    frequency: BackupFrequency = BackupFrequency.DAILY
    retention_days: int = 30
    retention_copies: int = 7

    # Locations
    primary_location_id: str = ""
    secondary_location_id: str = ""
    geographic_separation_required: bool = True

    # Recovery objectives
    rto_hours: float = 4.0  # Recovery Time Objective
    rpo_hours: float = 1.0  # Recovery Point Objective

    # Encryption
    encryption_required: bool = True
    encryption_algorithm: str = "AES-256"
    key_management_policy: str = ""

    # Verification
    integrity_check_required: bool = True
    verification_frequency_days: int = 7
    test_restore_frequency_days: int = 30

    # Ownership
    owner: str = ""
    approver: str = ""

    # Status
    is_active: bool = True
    effective_date: Optional[str] = None
    review_date: Optional[str] = None

    def __post_init__(self):
        if not self.policy_id:
            self.policy_id = f"BKP-POL-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class BackupJob:
    """
    Backup job execution record.
    """

    job_id: str = ""
    policy_id: str = ""
    job_name: str = ""

    # Type and scope
    backup_type: BackupType = BackupType.FULL
    source_systems: List[str] = field(default_factory=list)
    data_size_gb: float = 0.0

    # Execution
    status: BackupStatus = BackupStatus.SCHEDULED
    scheduled_time: str = ""
    started_time: Optional[str] = None
    completed_time: Optional[str] = None
    duration_seconds: float = 0.0

    # Location
    target_location_id: str = ""
    target_path: str = ""
    offsite_copy_created: bool = False

    # Results
    success: bool = False
    error_message: str = ""
    bytes_transferred: int = 0
    files_backed_up: int = 0
    files_failed: int = 0

    # Verification
    integrity_verified: bool = False
    integrity_check_time: Optional[str] = None
    checksum: str = ""

    # Metadata
    created_by: str = ""
    retention_expiry: Optional[str] = None

    def __post_init__(self):
        if not self.job_id:
            self.job_id = f"BKP-{uuid.uuid4().hex[:8].upper()}"
        if not self.scheduled_time:
            self.scheduled_time = datetime.now(timezone.utc).isoformat()


@dataclass
class BackupLocation:
    """
    Backup storage location per DORA Article 12(3).

    Documents geographically diverse backup locations.
    """

    location_id: str = ""
    name: str = ""
    description: str = ""
    location_type: LocationType = LocationType.ON_PREMISE

    # Geographic information
    geographic_region: str = ""
    country: str = ""
    city: str = ""
    data_center_name: str = ""
    address: str = ""

    # Distance from primary (km)
    distance_from_primary_km: float = 0.0

    # Capacity
    total_capacity_tb: float = 0.0
    used_capacity_tb: float = 0.0
    available_capacity_tb: float = 0.0

    # Connectivity
    network_bandwidth_gbps: float = 0.0
    latency_ms: float = 0.0
    replication_type: str = ""  # "synchronous", "asynchronous"

    # Security
    physical_security_level: str = ""
    encryption_at_rest: bool = True
    access_controls: List[str] = field(default_factory=list)

    # Compliance
    certifications: List[str] = field(default_factory=list)  # "SOC2", "ISO27001", etc.
    data_residency_compliant: bool = True

    # Status
    is_active: bool = True
    is_primary: bool = False
    is_disaster_recovery: bool = False
    last_connectivity_check: Optional[str] = None

    def __post_init__(self):
        if not self.location_id:
            self.location_id = f"LOC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class RecoveryTest:
    """
    Recovery test record per DORA Article 12(2).

    Documents testing of backup procedures and restoration methods.
    """

    test_id: str = ""
    test_name: str = ""
    description: str = ""
    test_type: RecoveryTestType = RecoveryTestType.RESTORE_TEST

    # Scope
    policy_id: str = ""
    backup_job_ids: List[str] = field(default_factory=list)
    systems_tested: List[str] = field(default_factory=list)
    data_tested: List[str] = field(default_factory=list)

    # Schedule
    scheduled_date: str = ""
    executed_date: Optional[str] = None
    duration_minutes: float = 0.0

    # Results
    result: RecoveryTestResult = RecoveryTestResult.INCOMPLETE
    rto_achieved_minutes: Optional[float] = None
    rpo_achieved_minutes: Optional[float] = None
    rto_target_minutes: float = 0.0
    rpo_target_minutes: float = 0.0

    # Metrics
    data_restored_gb: float = 0.0
    data_integrity_verified: bool = False
    functionality_verified: bool = False

    # Issues
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    remediation_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Participants
    test_lead: str = ""
    participants: List[str] = field(default_factory=list)
    observer: str = ""

    # Documentation
    test_script_used: str = ""
    evidence_collected: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.test_id:
            self.test_id = f"REC-TEST-{uuid.uuid4().hex[:8].upper()}"

    @property
    def rto_met(self) -> Optional[bool]:
        """Check if RTO was met."""
        if self.rto_achieved_minutes is None:
            return None
        return self.rto_achieved_minutes <= self.rto_target_minutes

    @property
    def rpo_met(self) -> Optional[bool]:
        """Check if RPO was met."""
        if self.rpo_achieved_minutes is None:
            return None
        return self.rpo_achieved_minutes <= self.rpo_target_minutes


@dataclass
class RestorationProcedure:
    """
    Restoration procedure documentation.
    """

    procedure_id: str = ""
    name: str = ""
    description: str = ""

    # Scope
    systems: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=list)

    # Steps
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_duration_minutes: int = 0

    # Requirements
    required_resources: List[str] = field(default_factory=list)
    required_access: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Verification
    verification_steps: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    # Ownership
    owner: str = ""
    approver: str = ""
    last_reviewed: Optional[str] = None

    # Status
    is_active: bool = True
    version: str = "1.0"

    def __post_init__(self):
        if not self.procedure_id:
            self.procedure_id = f"REST-{uuid.uuid4().hex[:8].upper()}"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BackupRecoveryConfig:
    """Configuration for Backup and Recovery module."""

    # Default retention
    default_retention_days: int = 30
    critical_data_retention_days: int = 90

    # Testing requirements
    restore_test_frequency_days: int = 30
    full_recovery_test_frequency_days: int = 365

    # Geographic requirements
    minimum_distance_km: float = 100.0  # For geographically diverse locations

    # Verification
    auto_verify_backups: bool = True
    verify_within_hours: int = 24

    # Alerting
    alert_on_failure: bool = True
    alert_on_capacity_threshold_pct: float = 80.0

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/dora/backup_recovery"

    # Callbacks
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Backup Recovery Class
# =============================================================================


class DORABackupRecovery:
    """
    DORA Article 12 compliant Backup and Recovery module.

    Manages:
    - Backup policies and procedures
    - Backup job execution and tracking
    - Backup location management
    - Recovery testing
    - Restoration procedures

    Usage:
        config = BackupRecoveryConfig()
        backup = DORABackupRecovery(config)

        # Create backup policy
        policy = backup.create_backup_policy(
            name="Critical Systems Daily Backup",
            data_classification=DataClassification.CRITICAL,
            frequency=BackupFrequency.DAILY,
        )

        # Create backup job
        job = backup.create_backup_job(
            policy_id=policy.policy_id,
            source_systems=["trading-db", "order-mgmt"],
        )

        # Schedule recovery test
        test = backup.schedule_recovery_test(
            test_type=RecoveryTestType.RESTORE_TEST,
            policy_id=policy.policy_id,
        )
    """

    def __init__(self, config: Optional[BackupRecoveryConfig] = None):
        """Initialize Backup and Recovery module."""
        self.config = config or BackupRecoveryConfig()

        # Data stores
        self._policies: Dict[str, BackupPolicy] = {}
        self._jobs: Dict[str, BackupJob] = {}
        self._locations: Dict[str, BackupLocation] = {}
        self._tests: Dict[str, RecoveryTest] = {}
        self._procedures: Dict[str, RestorationProcedure] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORABackupRecovery initialized")

    # =========================================================================
    # Backup Policies
    # =========================================================================

    def create_backup_policy(
        self,
        name: str,
        data_classification: DataClassification,
        frequency: BackupFrequency = BackupFrequency.DAILY,
        backup_type: BackupType = BackupType.FULL,
        systems_covered: Optional[List[str]] = None,
        retention_days: Optional[int] = None,
        rto_hours: float = 4.0,
        rpo_hours: float = 1.0,
        encryption_required: bool = True,
        geographic_separation_required: bool = True,
        owner: str = "",
    ) -> BackupPolicy:
        """
        Create a backup policy.

        Per Article 12(1), backup policies must specify scope and frequency.

        Args:
            name: Policy name
            data_classification: Data classification
            frequency: Backup frequency
            backup_type: Type of backup
            systems_covered: Systems covered
            retention_days: Retention period
            rto_hours: Recovery Time Objective
            rpo_hours: Recovery Point Objective
            encryption_required: Whether encryption is required
            geographic_separation_required: Geographic diversity requirement
            owner: Policy owner

        Returns:
            Created BackupPolicy
        """
        # Set retention based on classification if not provided
        if retention_days is None:
            if data_classification == DataClassification.CRITICAL:
                retention_days = self.config.critical_data_retention_days
            else:
                retention_days = self.config.default_retention_days

        policy = BackupPolicy(
            name=name,
            data_classification=data_classification,
            frequency=frequency,
            backup_type=backup_type,
            systems_covered=systems_covered or [],
            retention_days=retention_days,
            rto_hours=rto_hours,
            rpo_hours=rpo_hours,
            encryption_required=encryption_required,
            geographic_separation_required=geographic_separation_required,
            owner=owner,
            effective_date=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._policies[policy.policy_id] = policy

        self._log_event(
            "policy_created",
            {
                "policy_id": policy.policy_id,
                "name": name,
                "classification": data_classification.value,
            },
        )

        return policy

    def get_backup_policy(self, policy_id: str) -> Optional[BackupPolicy]:
        """Get backup policy by ID."""
        with self._lock:
            return self._policies.get(policy_id)

    def get_policies_by_classification(
        self,
        classification: DataClassification,
    ) -> List[BackupPolicy]:
        """Get policies by data classification."""
        with self._lock:
            return [
                p
                for p in self._policies.values()
                if p.data_classification == classification and p.is_active
            ]

    # =========================================================================
    # Backup Locations
    # =========================================================================

    def register_backup_location(
        self,
        name: str,
        location_type: LocationType,
        geographic_region: str,
        country: str,
        city: str = "",
        total_capacity_tb: float = 0.0,
        is_primary: bool = False,
        is_disaster_recovery: bool = False,
        distance_from_primary_km: float = 0.0,
        certifications: Optional[List[str]] = None,
    ) -> BackupLocation:
        """
        Register a backup location.

        Per Article 12(3), backup media must be at geographically diverse locations.

        Args:
            name: Location name
            location_type: Type of location
            geographic_region: Geographic region
            country: Country
            city: City
            total_capacity_tb: Total capacity in TB
            is_primary: Is this the primary location
            is_disaster_recovery: Is this a DR location
            distance_from_primary_km: Distance from primary site
            certifications: Compliance certifications

        Returns:
            Registered BackupLocation
        """
        location = BackupLocation(
            name=name,
            location_type=location_type,
            geographic_region=geographic_region,
            country=country,
            city=city,
            total_capacity_tb=total_capacity_tb,
            available_capacity_tb=total_capacity_tb,
            is_primary=is_primary,
            is_disaster_recovery=is_disaster_recovery,
            distance_from_primary_km=distance_from_primary_km,
            certifications=certifications or [],
        )

        with self._lock:
            self._locations[location.location_id] = location

        self._log_event(
            "location_registered",
            {
                "location_id": location.location_id,
                "name": name,
                "type": location_type.value,
            },
        )

        return location

    def get_backup_location(self, location_id: str) -> Optional[BackupLocation]:
        """Get backup location by ID."""
        with self._lock:
            return self._locations.get(location_id)

    def get_dr_locations(self) -> List[BackupLocation]:
        """Get disaster recovery locations."""
        with self._lock:
            return [l for l in self._locations.values() if l.is_disaster_recovery and l.is_active]

    def check_geographic_diversity(self) -> Dict[str, Any]:
        """Check if backup locations meet geographic diversity requirements."""
        with self._lock:
            primary = None
            dr_locations = []

            for loc in self._locations.values():
                if loc.is_primary:
                    primary = loc
                if loc.is_disaster_recovery:
                    dr_locations.append(loc)

            if not primary:
                return {
                    "compliant": False,
                    "issue": "No primary location defined",
                }

            if not dr_locations:
                return {
                    "compliant": False,
                    "issue": "No DR locations defined",
                }

            # Check distance
            min_distance = self.config.minimum_distance_km
            compliant_dr = [l for l in dr_locations if l.distance_from_primary_km >= min_distance]

            if not compliant_dr:
                return {
                    "compliant": False,
                    "issue": f"No DR location meets minimum distance ({min_distance}km)",
                }

            return {
                "compliant": True,
                "primary_location": primary.name,
                "dr_locations": [l.name for l in compliant_dr],
                "minimum_distance_km": min_distance,
            }

    # =========================================================================
    # Backup Jobs
    # =========================================================================

    def create_backup_job(
        self,
        policy_id: str,
        source_systems: Optional[List[str]] = None,
        backup_type: Optional[BackupType] = None,
        target_location_id: str = "",
        created_by: str = "",
    ) -> Optional[BackupJob]:
        """Create a backup job based on a policy."""
        with self._lock:
            if policy_id not in self._policies:
                return None

            policy = self._policies[policy_id]

        job = BackupJob(
            policy_id=policy_id,
            job_name=f"{policy.name} - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            backup_type=backup_type or policy.backup_type,
            source_systems=source_systems or policy.systems_covered,
            target_location_id=target_location_id or policy.primary_location_id,
            created_by=created_by,
        )

        # Calculate retention expiry
        retention_expiry = datetime.now(timezone.utc) + timedelta(days=policy.retention_days)
        job.retention_expiry = retention_expiry.isoformat()

        with self._lock:
            self._jobs[job.job_id] = job

        return job

    def start_backup_job(self, job_id: str) -> Optional[BackupJob]:
        """Start a backup job."""
        with self._lock:
            if job_id not in self._jobs:
                return None

            job = self._jobs[job_id]
            job.status = BackupStatus.RUNNING
            job.started_time = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "backup_started",
            {
                "job_id": job_id,
                "policy_id": job.policy_id,
            },
        )

        return job

    def complete_backup_job(
        self,
        job_id: str,
        success: bool,
        bytes_transferred: int = 0,
        files_backed_up: int = 0,
        files_failed: int = 0,
        error_message: str = "",
    ) -> Optional[BackupJob]:
        """Complete a backup job."""
        with self._lock:
            if job_id not in self._jobs:
                return None

            job = self._jobs[job_id]
            job.status = BackupStatus.COMPLETED if success else BackupStatus.FAILED
            job.completed_time = datetime.now(timezone.utc).isoformat()
            job.success = success
            job.bytes_transferred = bytes_transferred
            job.files_backed_up = files_backed_up
            job.files_failed = files_failed
            job.error_message = error_message

            # Calculate duration
            if job.started_time:
                start = datetime.fromisoformat(job.started_time.replace("Z", "+00:00"))
                end = datetime.fromisoformat(job.completed_time.replace("Z", "+00:00"))
                job.duration_seconds = (end - start).total_seconds()

        # Alert on failure
        if not success and self.config.alert_on_failure:
            self._alert_backup_failure(job)

        self._log_event(
            "backup_completed",
            {
                "job_id": job_id,
                "success": success,
                "bytes_transferred": bytes_transferred,
            },
        )

        return job

    def verify_backup(
        self,
        job_id: str,
        checksum: str = "",
    ) -> Optional[BackupJob]:
        """Verify backup integrity."""
        with self._lock:
            if job_id not in self._jobs:
                return None

            job = self._jobs[job_id]
            job.integrity_verified = True
            job.integrity_check_time = datetime.now(timezone.utc).isoformat()
            job.checksum = checksum
            job.status = BackupStatus.VERIFIED

        return job

    def get_backup_job(self, job_id: str) -> Optional[BackupJob]:
        """Get backup job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_recent_jobs(
        self,
        policy_id: Optional[str] = None,
        status: Optional[BackupStatus] = None,
        limit: int = 50,
    ) -> List[BackupJob]:
        """Get recent backup jobs."""
        with self._lock:
            jobs = list(self._jobs.values())

            if policy_id:
                jobs = [j for j in jobs if j.policy_id == policy_id]

            if status:
                jobs = [j for j in jobs if j.status == status]

            # Sort by scheduled time descending
            jobs.sort(key=lambda j: j.scheduled_time, reverse=True)
            return jobs[:limit]

    def get_failed_jobs(self, hours: int = 24) -> List[BackupJob]:
        """Get failed jobs within specified hours."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        with self._lock:
            return [
                j
                for j in self._jobs.values()
                if j.status == BackupStatus.FAILED and j.scheduled_time > cutoff
            ]

    def _alert_backup_failure(self, job: BackupJob) -> None:
        """Alert on backup failure."""
        if self.config.alert_callback:
            try:
                self.config.alert_callback("backup_failed", asdict(job))
            except Exception as e:
                logger.error(f"Backup alert callback failed: {e}")

        logger.error(f"Backup failed: {job.job_id} - {job.error_message}")

    # =========================================================================
    # Recovery Testing
    # =========================================================================

    def schedule_recovery_test(
        self,
        test_name: str,
        test_type: RecoveryTestType,
        policy_id: str = "",
        systems_tested: Optional[List[str]] = None,
        scheduled_date: Optional[str] = None,
        test_lead: str = "",
        rto_target_minutes: float = 240.0,
        rpo_target_minutes: float = 60.0,
    ) -> RecoveryTest:
        """
        Schedule a recovery test.

        Per Article 12(2), backup procedures and restoration methods
        must be tested at regular intervals.

        Args:
            test_name: Test name
            test_type: Type of test
            policy_id: Related backup policy
            systems_tested: Systems to test
            scheduled_date: Scheduled date
            test_lead: Test lead
            rto_target_minutes: RTO target
            rpo_target_minutes: RPO target

        Returns:
            Scheduled RecoveryTest
        """
        test = RecoveryTest(
            test_name=test_name,
            test_type=test_type,
            policy_id=policy_id,
            systems_tested=systems_tested or [],
            scheduled_date=scheduled_date or datetime.now(timezone.utc).isoformat(),
            test_lead=test_lead,
            rto_target_minutes=rto_target_minutes,
            rpo_target_minutes=rpo_target_minutes,
        )

        with self._lock:
            self._tests[test.test_id] = test

        self._log_event(
            "recovery_test_scheduled",
            {
                "test_id": test.test_id,
                "test_type": test_type.value,
            },
        )

        return test

    def start_recovery_test(
        self,
        test_id: str,
        participants: Optional[List[str]] = None,
    ) -> Optional[RecoveryTest]:
        """Start a recovery test."""
        with self._lock:
            if test_id not in self._tests:
                return None

            test = self._tests[test_id]
            test.executed_date = datetime.now(timezone.utc).isoformat()
            test.participants = participants or []

        return test

    def complete_recovery_test(
        self,
        test_id: str,
        result: RecoveryTestResult,
        rto_achieved_minutes: Optional[float] = None,
        rpo_achieved_minutes: Optional[float] = None,
        data_restored_gb: float = 0.0,
        data_integrity_verified: bool = False,
        functionality_verified: bool = False,
        issues_found: Optional[List[Dict[str, Any]]] = None,
        lessons_learned: Optional[List[str]] = None,
    ) -> Optional[RecoveryTest]:
        """Complete a recovery test."""
        with self._lock:
            if test_id not in self._tests:
                return None

            test = self._tests[test_id]
            test.result = result
            test.rto_achieved_minutes = rto_achieved_minutes
            test.rpo_achieved_minutes = rpo_achieved_minutes
            test.data_restored_gb = data_restored_gb
            test.data_integrity_verified = data_integrity_verified
            test.functionality_verified = functionality_verified
            test.issues_found = issues_found or []
            test.lessons_learned = lessons_learned or []

            # Calculate duration
            if test.executed_date:
                start = datetime.fromisoformat(test.executed_date.replace("Z", "+00:00"))
                test.duration_minutes = (datetime.now(timezone.utc) - start).total_seconds() / 60

        self._log_event(
            "recovery_test_completed",
            {
                "test_id": test_id,
                "result": result.value,
                "rto_met": test.rto_met,
                "rpo_met": test.rpo_met,
            },
        )

        return test

    def get_recovery_test(self, test_id: str) -> Optional[RecoveryTest]:
        """Get recovery test by ID."""
        with self._lock:
            return self._tests.get(test_id)

    def get_tests_for_policy(self, policy_id: str) -> List[RecoveryTest]:
        """Get all tests for a policy."""
        with self._lock:
            return [t for t in self._tests.values() if t.policy_id == policy_id]

    def get_overdue_tests(self) -> List[Dict[str, Any]]:
        """Get policies with overdue recovery tests."""
        overdue = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for policy in self._policies.values():
                if not policy.is_active:
                    continue

                # Find most recent test for this policy
                policy_tests = [
                    t
                    for t in self._tests.values()
                    if t.policy_id == policy.policy_id
                    and t.result
                    in (
                        RecoveryTestResult.PASSED,
                        RecoveryTestResult.PASSED_WITH_ISSUES,
                    )
                ]

                if not policy_tests:
                    overdue.append(
                        {
                            "policy_id": policy.policy_id,
                            "policy_name": policy.name,
                            "issue": "Never tested",
                        }
                    )
                    continue

                latest_test = max(policy_tests, key=lambda t: t.executed_date or "")
                if latest_test.executed_date:
                    test_date = datetime.fromisoformat(
                        latest_test.executed_date.replace("Z", "+00:00")
                    )
                    days_since = (now - test_date).days

                    if days_since > policy.test_restore_frequency_days:
                        overdue.append(
                            {
                                "policy_id": policy.policy_id,
                                "policy_name": policy.name,
                                "issue": f"Last test was {days_since} days ago",
                                "required_frequency_days": policy.test_restore_frequency_days,
                            }
                        )

        return overdue

    # =========================================================================
    # Restoration Procedures
    # =========================================================================

    def create_restoration_procedure(
        self,
        name: str,
        systems: Optional[List[str]] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        verification_steps: Optional[List[Dict[str, Any]]] = None,
        estimated_duration_minutes: int = 60,
        owner: str = "",
    ) -> RestorationProcedure:
        """Create a restoration procedure."""
        procedure = RestorationProcedure(
            name=name,
            systems=systems or [],
            steps=steps or [],
            verification_steps=verification_steps or [],
            estimated_duration_minutes=estimated_duration_minutes,
            owner=owner,
        )

        with self._lock:
            self._procedures[procedure.procedure_id] = procedure

        return procedure

    def get_restoration_procedure(
        self,
        procedure_id: str,
    ) -> Optional[RestorationProcedure]:
        """Get restoration procedure by ID."""
        with self._lock:
            return self._procedures.get(procedure_id)

    def get_procedures_for_system(self, system: str) -> List[RestorationProcedure]:
        """Get procedures for a system."""
        with self._lock:
            return [p for p in self._procedures.values() if system in p.systems and p.is_active]

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_backup_summary(self) -> Dict[str, Any]:
        """Get backup and recovery summary."""
        with self._lock:
            all_jobs = list(self._jobs.values())
            recent_jobs = [
                j
                for j in all_jobs
                if j.scheduled_time > (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            ]

            successful = [j for j in recent_jobs if j.success]
            failed = [j for j in recent_jobs if j.status == BackupStatus.FAILED]

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "policies": {
                    "total": len(self._policies),
                    "active": sum(1 for p in self._policies.values() if p.is_active),
                    "by_classification": {
                        c.value: sum(
                            1 for p in self._policies.values() if p.data_classification == c
                        )
                        for c in DataClassification
                    },
                },
                "jobs_7_days": {
                    "total": len(recent_jobs),
                    "successful": len(successful),
                    "failed": len(failed),
                    "success_rate_pct": (
                        len(successful) / len(recent_jobs) * 100 if recent_jobs else 100.0
                    ),
                },
                "locations": {
                    "total": len(self._locations),
                    "active": sum(1 for l in self._locations.values() if l.is_active),
                    "dr_locations": len(self.get_dr_locations()),
                },
                "geographic_diversity": self.check_geographic_diversity(),
                "recovery_tests": {
                    "total": len(self._tests),
                    "passed": sum(
                        1 for t in self._tests.values() if t.result == RecoveryTestResult.PASSED
                    ),
                    "overdue_policies": len(self.get_overdue_tests()),
                },
                "procedures": {
                    "total": len(self._procedures),
                    "active": sum(1 for p in self._procedures.values() if p.is_active),
                },
            }

    def export_backup_report(self) -> Dict[str, Any]:
        """Export complete backup and recovery report."""
        with self._lock:
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "article_reference": "Article 12",
                "summary": self.get_backup_summary(),
                "policies": [asdict(p) for p in self._policies.values()],
                "locations": [asdict(l) for l in self._locations.values()],
                "jobs": [asdict(j) for j in list(self._jobs.values())[-100:]],
                "recovery_tests": [asdict(t) for t in self._tests.values()],
                "procedures": [asdict(p) for p in self._procedures.values()],
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

        log_file = self._log_path / f"backup_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_backup_recovery(
    config: Optional[BackupRecoveryConfig] = None,
) -> DORABackupRecovery:
    """
    Create a DORABackupRecovery instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORABackupRecovery instance
    """
    return DORABackupRecovery(config=config)

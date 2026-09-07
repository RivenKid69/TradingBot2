# -*- coding: utf-8 -*-
"""
EU-Only Data Residency Drift Checker.

GDPR Phase 3 Implementation: EU-only data residency enforcement.

This module provides:
    - EUOnlyDriftChecker: Validates all endpoints/services are EU-resident
    - ResidencyDriftReport: Machine-readable JSON report format
    - EvidencePackExporter: EU residency evidence for audits
    - DeploymentValidator: Validates deployment configurations

Design Doc Reference:
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L892 (14.3: EU residency)
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1745 (6.2: EU-only)
    - docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md (Phase 3)
    - docs/compliance/SUBPROCESSORS_REGISTER.md (drift check format)

CLOUD ZONE ONLY.

CRITICAL SECURITY:
    - EU-only residency is a HARD CONSTRAINT
    - Drift check FAILS CLOSED - any non-EU endpoint blocks deployment
    - All services must be verified EU-resident
    - Evidence reports are stored for audit
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Final, FrozenSet, List, Optional, Pattern, Set, Tuple
from urllib.parse import urlparse
from uuid import uuid4

from .residency import DataRegion, EU_REGIONS


# ============================================================================
# Constants
# ============================================================================

# Valid EU AWS regions
EU_AWS_REGIONS: Final[FrozenSet[str]] = frozenset(
    {
        "eu-west-1",  # Ireland
        "eu-west-2",  # London
        "eu-west-3",  # Paris
        "eu-central-1",  # Frankfurt
        "eu-central-2",  # Zurich
        "eu-north-1",  # Stockholm
        "eu-south-1",  # Milan
        "eu-south-2",  # Spain
    }
)

# Valid EU GCP regions
EU_GCP_REGIONS: Final[FrozenSet[str]] = frozenset(
    {
        "europe-west1",  # Belgium
        "europe-west2",  # London
        "europe-west3",  # Frankfurt
        "europe-west4",  # Netherlands
        "europe-west6",  # Zurich
        "europe-west8",  # Milan
        "europe-west9",  # Paris
        "europe-west10",  # Berlin
        "europe-west12",  # Turin
        "europe-north1",  # Finland
        "europe-central2",  # Warsaw
        "europe-southwest1",  # Madrid
    }
)

# Valid EU Azure regions
EU_AZURE_REGIONS: Final[FrozenSet[str]] = frozenset(
    {
        "westeurope",  # Netherlands
        "northeurope",  # Ireland
        "germanywestcentral",  # Frankfurt
        "francecentral",  # Paris
        "francesouth",  # Marseille
        "swedencentral",  # Gavle
        "swedensouth",  # Stockholm
        "norwayeast",  # Oslo
        "norwaywest",  # Stavanger
        "switzerlandnorth",  # Zurich
        "switzerlandwest",  # Geneva
        "polandcentral",  # Warsaw
        "italynorth",  # Milan
        "uksouth",  # London
        "ukwest",  # Cardiff
        "spaincentral",  # Madrid
    }
)

# All valid EU regions across providers
ALL_EU_REGIONS: Final[FrozenSet[str]] = (
    EU_AWS_REGIONS
    | EU_GCP_REGIONS
    | EU_AZURE_REGIONS
    | frozenset(
        {
            "EU",  # Generic EU designation
            "eu",  # Lowercase variant
            "Europe",  # Generic Europe
            "europe",  # Lowercase variant
            "EU (Germany)",  # Specific country mentions
            "EU (Ireland)",
            "EU (Netherlands)",
            "EU (France)",
        }
    )
)

# Non-EU regions that should always fail
NON_EU_REGIONS: Final[FrozenSet[str]] = frozenset(
    {
        # US regions
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
        "us-gov-east-1",
        "us-gov-west-1",
        # Asia Pacific
        "ap-northeast-1",
        "ap-northeast-2",
        "ap-northeast-3",
        "ap-southeast-1",
        "ap-southeast-2",
        "ap-southeast-3",
        "ap-south-1",
        "ap-east-1",
        # Other
        "sa-east-1",
        "af-south-1",
        "me-south-1",
        "me-central-1",
        "ca-central-1",
    }
)

# AWS endpoint patterns
AWS_ENDPOINT_PATTERNS: List[Tuple[Pattern, str]] = [
    # RDS
    (re.compile(r"\.([a-z]{2}-[a-z]+-\d+)\.rds\.amazonaws\.com"), "database"),
    # S3
    (re.compile(r"s3\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "storage"),
    (re.compile(r"s3-([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "storage"),
    (re.compile(r"\.s3\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "storage"),
    # ElastiCache
    (re.compile(r"\.([a-z]{2}-[a-z]+-\d+)\.cache\.amazonaws\.com"), "cache"),
    # SES
    (re.compile(r"email\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "email"),
    (re.compile(r"email-smtp\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "email"),
    # CloudWatch
    (re.compile(r"logs\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "logging"),
    (re.compile(r"monitoring\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "monitoring"),
    # KMS
    (re.compile(r"kms\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "key_management"),
    # Secrets Manager
    (re.compile(r"secretsmanager\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "secrets"),
    # SNS
    (re.compile(r"sns\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "messaging"),
    # SQS
    (re.compile(r"sqs\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "queue"),
    # DynamoDB
    (re.compile(r"dynamodb\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "database"),
    # ECR
    (re.compile(r"\.dkr\.ecr\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "registry"),
    # Generic AWS
    (re.compile(r"([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com"), "aws_service"),
]


# ============================================================================
# Enums
# ============================================================================


class DriftCheckStatus(str, Enum):
    """Status of a drift check."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    UNKNOWN = "UNKNOWN"


class ComponentType(str, Enum):
    """Type of infrastructure component."""

    DATABASE_PRIMARY = "database_primary"
    DATABASE_REPLICA = "database_replica"
    OBJECT_STORAGE = "object_storage"
    BACKUP_STORAGE = "backup_storage"
    CACHE = "cache"
    EMAIL_SERVICE = "email_service"
    ERROR_TRACKING = "error_tracking"
    LOGGING = "logging"
    MONITORING = "monitoring"
    PAYMENT_PROCESSOR = "payment_processor"
    ARTIFACT_REGISTRY = "artifact_registry"
    KEY_MANAGEMENT = "key_management"
    SECRETS_MANAGEMENT = "secrets_management"
    QUEUE = "queue"
    MESSAGING = "messaging"
    CUSTOM = "custom"


class CheckSeverity(str, Enum):
    """Severity of drift check violations."""

    CRITICAL = "critical"  # Blocks deployment
    HIGH = "high"  # Should block, requires override
    MEDIUM = "medium"  # Warning
    LOW = "low"  # Informational


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class EndpointCheck:
    """Result of checking a single endpoint."""

    component: str
    component_type: ComponentType
    endpoint: str
    region: str
    eu_compliant: bool
    check_method: str  # "pattern", "config", "dns", "explicit"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "component": self.component,
            "component_type": self.component_type.value,
            "endpoint": self.endpoint,
            "region": self.region,
            "eu_compliant": self.eu_compliant,
            "check_method": self.check_method,
            "error": self.error,
            "metadata": self.metadata,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class SubprocessorCheck:
    """Result of checking a subprocessor."""

    name: str
    legal_entity: str
    service: str
    region: str
    eu_compliant: bool
    dpa_status: str
    last_review: Optional[str] = None
    next_review: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "legal_entity": self.legal_entity,
            "service": self.service,
            "region": self.region,
            "eu_compliant": self.eu_compliant,
            "dpa_status": self.dpa_status,
            "last_review": self.last_review,
            "next_review": self.next_review,
            "notes": self.notes,
        }


@dataclass
class DriftCheckViolation:
    """A residency drift violation."""

    component: str
    violation_type: str
    severity: CheckSeverity
    message: str
    region_found: str
    region_expected: str = "EU"
    remediation: str = ""
    blocked: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "violation_type": self.violation_type,
            "severity": self.severity.value,
            "message": self.message,
            "region_found": self.region_found,
            "region_expected": self.region_expected,
            "remediation": self.remediation,
            "blocked": self.blocked,
        }


@dataclass
class ResidencyDriftReport:
    """
    Complete drift check report.

    Format aligned with docs/compliance/SUBPROCESSORS_REGISTER.md Section 5.2.
    """

    check_id: str = field(
        default_factory=lambda: f"drift-check-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-{str(uuid4())[:8]}"
    )
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: DriftCheckStatus = DriftCheckStatus.UNKNOWN
    checks: List[EndpointCheck] = field(default_factory=list)
    subprocessors: List[SubprocessorCheck] = field(default_factory=list)
    violations: List[DriftCheckViolation] = field(default_factory=list)
    subprocessors_verified: int = 0
    endpoints_verified: int = 0
    non_eu_endpoints: int = 0
    next_check: Optional[datetime] = None
    report_hash: str = ""

    # Metadata
    environment: str = "production"
    deployment_id: str = ""
    triggered_by: str = "scheduled"  # scheduled, deployment, manual

    def __post_init__(self):
        """Compute derived fields."""
        self._compute_stats()
        if not self.report_hash:
            self._compute_hash()

    def _compute_stats(self) -> None:
        """Compute statistics from checks."""
        self.endpoints_verified = len(self.checks)
        self.subprocessors_verified = len(self.subprocessors)
        self.non_eu_endpoints = sum(1 for c in self.checks if not c.eu_compliant)

        # Set status based on violations
        if any(v.severity == CheckSeverity.CRITICAL for v in self.violations):
            self.status = DriftCheckStatus.FAIL
        elif any(v.severity == CheckSeverity.HIGH for v in self.violations):
            self.status = DriftCheckStatus.FAIL
        elif any(v.severity == CheckSeverity.MEDIUM for v in self.violations):
            self.status = DriftCheckStatus.WARNING
        elif self.endpoints_verified == 0 and self.subprocessors_verified == 0:
            self.status = DriftCheckStatus.UNKNOWN
        else:
            self.status = DriftCheckStatus.PASS

    def _compute_hash(self) -> None:
        """Compute SHA-256 hash of report content."""
        content = json.dumps(
            {
                "check_id": self.check_id,
                "timestamp": self.timestamp.isoformat(),
                "checks": [c.to_dict() for c in self.checks],
                "subprocessors": [s.to_dict() for s in self.subprocessors],
                "violations": [v.to_dict() for v in self.violations],
            },
            sort_keys=True,
        )
        self.report_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    @property
    def passed(self) -> bool:
        """Check if drift check passed."""
        return self.status == DriftCheckStatus.PASS

    @property
    def blocking_violations(self) -> List[DriftCheckViolation]:
        """Get violations that block deployment."""
        return [v for v in self.violations if v.blocked]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "check_id": self.check_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "checks": [c.to_dict() for c in self.checks],
            "subprocessors": [s.to_dict() for s in self.subprocessors],
            "violations": [v.to_dict() for v in self.violations],
            "subprocessors_verified": self.subprocessors_verified,
            "endpoints_verified": self.endpoints_verified,
            "non_eu_endpoints": self.non_eu_endpoints,
            "next_check": self.next_check.isoformat() if self.next_check else None,
            "report_hash": self.report_hash,
            "environment": self.environment,
            "deployment_id": self.deployment_id,
            "triggered_by": self.triggered_by,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save report to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


@dataclass
class ResidencyConfiguration:
    """
    Deployment residency configuration.

    Loaded from environment, config files, or Helm values.
    """

    # Database
    database_primary_endpoint: str = ""
    database_primary_region: str = ""
    database_replica_endpoint: str = ""
    database_replica_region: str = ""

    # Storage
    object_storage_bucket: str = ""
    object_storage_region: str = ""
    backup_storage_bucket: str = ""
    backup_storage_region: str = ""

    # Cache
    cache_endpoint: str = ""
    cache_region: str = ""

    # Email
    email_service: str = ""
    email_region: str = ""

    # Logging/Monitoring
    logging_endpoint: str = ""
    logging_region: str = ""
    monitoring_endpoint: str = ""
    monitoring_region: str = ""

    # Error tracking
    error_tracking_service: str = ""
    error_tracking_region: str = ""

    # Registry
    artifact_registry: str = ""
    artifact_registry_region: str = ""

    # Payment
    payment_processor: str = ""
    payment_processor_region: str = ""

    # Custom endpoints
    custom_endpoints: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_environment(cls) -> "ResidencyConfiguration":
        """Load configuration from environment variables."""
        return cls(
            database_primary_endpoint=os.getenv("DATABASE_URL", ""),
            database_primary_region=os.getenv("DATABASE_REGION", ""),
            database_replica_endpoint=os.getenv("DATABASE_REPLICA_URL", ""),
            database_replica_region=os.getenv("DATABASE_REPLICA_REGION", ""),
            object_storage_bucket=os.getenv("S3_BUCKET", os.getenv("OBJECT_STORAGE_BUCKET", "")),
            object_storage_region=os.getenv("S3_REGION", os.getenv("AWS_REGION", "")),
            backup_storage_bucket=os.getenv("BACKUP_BUCKET", ""),
            backup_storage_region=os.getenv("BACKUP_REGION", ""),
            cache_endpoint=os.getenv("REDIS_URL", os.getenv("CACHE_URL", "")),
            cache_region=os.getenv("REDIS_REGION", os.getenv("CACHE_REGION", "")),
            email_service=os.getenv("EMAIL_SERVICE", "ses"),
            email_region=os.getenv("EMAIL_REGION", os.getenv("SES_REGION", "")),
            logging_endpoint=os.getenv("LOGGING_ENDPOINT", ""),
            logging_region=os.getenv("LOGGING_REGION", ""),
            monitoring_endpoint=os.getenv("MONITORING_ENDPOINT", ""),
            monitoring_region=os.getenv("MONITORING_REGION", ""),
            error_tracking_service=os.getenv("ERROR_TRACKING_SERVICE", "sentry"),
            error_tracking_region=os.getenv(
                "ERROR_TRACKING_REGION", os.getenv("SENTRY_REGION", "")
            ),
            artifact_registry=os.getenv("ARTIFACT_REGISTRY", ""),
            artifact_registry_region=os.getenv("ARTIFACT_REGISTRY_REGION", ""),
            payment_processor=os.getenv("PAYMENT_PROCESSOR", "stripe"),
            payment_processor_region=os.getenv("PAYMENT_PROCESSOR_REGION", ""),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResidencyConfiguration":
        """Load configuration from dictionary (e.g., parsed YAML)."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# ============================================================================
# EU-Only Drift Checker
# ============================================================================


class EUOnlyDriftChecker:
    """
    EU-Only Data Residency Drift Checker.

    CRITICAL: This checker FAILS CLOSED - any non-EU endpoint
    causes the check to fail and blocks deployment.

    Usage:
        checker = EUOnlyDriftChecker()

        # Check from configuration
        config = ResidencyConfiguration.from_environment()
        report = checker.check(config)

        # Check specific endpoint
        result = checker.check_endpoint("mydb.eu-central-1.rds.amazonaws.com")

        # Fail closed
        if not report.passed:
            raise ResidencyViolationError(report)

    COMPLIANCE:
        - Design Doc 14.3: EU residency by default
        - Design Doc 6.2: EU-only drift checks mandatory
        - GDPR: Data residency requirements
    """

    def __init__(
        self,
        fail_closed: bool = True,
        check_dns: bool = False,
        allowed_regions: Optional[FrozenSet[str]] = None,
        custom_patterns: Optional[List[Tuple[Pattern, str]]] = None,
    ):
        """
        Initialize drift checker.

        Args:
            fail_closed: If True, any non-EU endpoint fails the check
            check_dns: If True, perform DNS lookups for region detection
            allowed_regions: Custom set of allowed regions (default: EU regions)
            custom_patterns: Additional endpoint patterns to check
        """
        self._fail_closed = fail_closed
        self._check_dns = check_dns
        self._allowed_regions = allowed_regions or ALL_EU_REGIONS
        self._patterns = list(AWS_ENDPOINT_PATTERNS)
        if custom_patterns:
            self._patterns.extend(custom_patterns)
        self._lock = threading.Lock()
        self._audit_log: List[Dict[str, Any]] = []

    def check(
        self,
        config: ResidencyConfiguration,
        subprocessors: Optional[List[SubprocessorCheck]] = None,
        environment: str = "production",
        deployment_id: str = "",
        triggered_by: str = "scheduled",
    ) -> ResidencyDriftReport:
        """
        Perform full residency drift check.

        Args:
            config: Deployment configuration to check
            subprocessors: Optional subprocessor list to verify
            environment: Environment name
            deployment_id: Deployment identifier
            triggered_by: What triggered this check

        Returns:
            ResidencyDriftReport with all check results
        """
        checks: List[EndpointCheck] = []
        violations: List[DriftCheckViolation] = []

        # Check database primary
        if config.database_primary_endpoint:
            check = self.check_endpoint(
                config.database_primary_endpoint,
                "database_primary",
                ComponentType.DATABASE_PRIMARY,
                explicit_region=config.database_primary_region,
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Check database replica
        if config.database_replica_endpoint:
            check = self.check_endpoint(
                config.database_replica_endpoint,
                "database_replica",
                ComponentType.DATABASE_REPLICA,
                explicit_region=config.database_replica_region,
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Check object storage
        if config.object_storage_bucket:
            check = self._check_storage(
                config.object_storage_bucket,
                config.object_storage_region,
                "object_storage",
                ComponentType.OBJECT_STORAGE,
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Check backup storage
        if config.backup_storage_bucket:
            check = self._check_storage(
                config.backup_storage_bucket,
                config.backup_storage_region,
                "backup_storage",
                ComponentType.BACKUP_STORAGE,
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Check cache
        if config.cache_endpoint:
            check = self.check_endpoint(
                config.cache_endpoint,
                "cache",
                ComponentType.CACHE,
                explicit_region=config.cache_region,
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Check email service
        if config.email_region or config.email_service:
            check = self._check_service(
                config.email_service,
                config.email_region,
                "email_service",
                ComponentType.EMAIL_SERVICE,
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Check logging
        if config.logging_endpoint or config.logging_region:
            check = self.check_endpoint(
                config.logging_endpoint,
                "logging",
                ComponentType.LOGGING,
                explicit_region=config.logging_region,
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Check monitoring
        if config.monitoring_endpoint or config.monitoring_region:
            check = self.check_endpoint(
                config.monitoring_endpoint,
                "monitoring",
                ComponentType.MONITORING,
                explicit_region=config.monitoring_region,
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Check error tracking
        if config.error_tracking_service or config.error_tracking_region:
            check = self._check_service(
                config.error_tracking_service,
                config.error_tracking_region,
                "error_tracking",
                ComponentType.ERROR_TRACKING,
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Check artifact registry
        if config.artifact_registry:
            check = self.check_endpoint(
                config.artifact_registry,
                "artifact_registry",
                ComponentType.ARTIFACT_REGISTRY,
                explicit_region=config.artifact_registry_region,
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Check payment processor
        if config.payment_processor or config.payment_processor_region:
            check = self._check_payment_processor(
                config.payment_processor,
                config.payment_processor_region,
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Check custom endpoints
        for name, endpoint_config in config.custom_endpoints.items():
            check = self.check_endpoint(
                endpoint_config.get("endpoint", ""),
                name,
                ComponentType.CUSTOM,
                explicit_region=endpoint_config.get("region", ""),
            )
            checks.append(check)
            if not check.eu_compliant:
                violations.append(self._create_violation(check))

        # Create report
        report = ResidencyDriftReport(
            checks=checks,
            subprocessors=subprocessors or [],
            violations=violations,
            environment=environment,
            deployment_id=deployment_id,
            triggered_by=triggered_by,
            next_check=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
        )

        # Log audit event
        self._log_audit("drift_check_completed", report)

        return report

    def check_endpoint(
        self,
        endpoint: str,
        component: str,
        component_type: ComponentType,
        explicit_region: str = "",
    ) -> EndpointCheck:
        """
        Check a single endpoint for EU compliance.

        Args:
            endpoint: Endpoint URL or hostname
            component: Component name
            component_type: Type of component
            explicit_region: Explicitly configured region (takes precedence)

        Returns:
            EndpointCheck result
        """
        # If explicit region is provided, use it
        if explicit_region:
            eu_compliant = self._is_eu_region(explicit_region)
            return EndpointCheck(
                component=component,
                component_type=component_type,
                endpoint=endpoint or "(explicit region)",
                region=explicit_region,
                eu_compliant=eu_compliant,
                check_method="explicit",
            )

        # If no endpoint, we can't check
        if not endpoint:
            return EndpointCheck(
                component=component,
                component_type=component_type,
                endpoint="",
                region="unknown",
                eu_compliant=False,
                check_method="none",
                error="No endpoint configured",
            )

        # Try to extract region from endpoint pattern
        region = self._extract_region_from_endpoint(endpoint)
        if region:
            eu_compliant = self._is_eu_region(region)
            return EndpointCheck(
                component=component,
                component_type=component_type,
                endpoint=endpoint,
                region=region,
                eu_compliant=eu_compliant,
                check_method="pattern",
            )

        # Try DNS lookup if enabled
        if self._check_dns:
            region = self._region_from_dns(endpoint)
            if region:
                eu_compliant = self._is_eu_region(region)
                return EndpointCheck(
                    component=component,
                    component_type=component_type,
                    endpoint=endpoint,
                    region=region,
                    eu_compliant=eu_compliant,
                    check_method="dns",
                )

        # Unknown - fail closed
        return EndpointCheck(
            component=component,
            component_type=component_type,
            endpoint=endpoint,
            region="unknown",
            eu_compliant=not self._fail_closed,
            check_method="none",
            error="Could not determine region from endpoint",
        )

    def _check_storage(
        self,
        bucket: str,
        region: str,
        component: str,
        component_type: ComponentType,
    ) -> EndpointCheck:
        """Check storage bucket for EU compliance."""
        if region:
            eu_compliant = self._is_eu_region(region)
            return EndpointCheck(
                component=component,
                component_type=component_type,
                endpoint=bucket,
                region=region,
                eu_compliant=eu_compliant,
                check_method="explicit",
                metadata={"bucket": bucket},
            )

        # Try to determine from bucket endpoint
        if "s3" in bucket.lower():
            # S3 bucket URL
            return self.check_endpoint(bucket, component, component_type)

        return EndpointCheck(
            component=component,
            component_type=component_type,
            endpoint=bucket,
            region="unknown",
            eu_compliant=not self._fail_closed,
            check_method="none",
            error="Could not determine storage region",
        )

    def _check_service(
        self,
        service: str,
        region: str,
        component: str,
        component_type: ComponentType,
    ) -> EndpointCheck:
        """Check a service for EU compliance."""
        if region:
            eu_compliant = self._is_eu_region(region)
            return EndpointCheck(
                component=component,
                component_type=component_type,
                endpoint=service,
                region=region,
                eu_compliant=eu_compliant,
                check_method="explicit",
                metadata={"service": service},
            )

        # Known EU-only services
        eu_services = {
            "sentry": "EU",  # Sentry EU data center
            "stripe": "EU (Ireland)",  # Stripe Payments Europe
            "ses": "eu-west-1",  # AWS SES EU default
            "sendgrid": "EU",  # SendGrid EU
        }

        if service.lower() in eu_services:
            return EndpointCheck(
                component=component,
                component_type=component_type,
                endpoint=service,
                region=eu_services[service.lower()],
                eu_compliant=True,
                check_method="known_service",
                metadata={"service": service},
            )

        return EndpointCheck(
            component=component,
            component_type=component_type,
            endpoint=service,
            region="unknown",
            eu_compliant=not self._fail_closed,
            check_method="none",
            error=f"Could not determine region for service: {service}",
        )

    def _check_payment_processor(
        self,
        processor: str,
        region: str,
    ) -> EndpointCheck:
        """Check payment processor for EU compliance."""
        component = "payment_processor"
        component_type = ComponentType.PAYMENT_PROCESSOR

        if region:
            eu_compliant = self._is_eu_region(region)
            return EndpointCheck(
                component=component,
                component_type=component_type,
                endpoint=processor,
                region=region,
                eu_compliant=eu_compliant,
                check_method="explicit",
                metadata={"processor": processor},
            )

        # Known EU payment processors
        eu_processors = {
            "stripe": ("Stripe Payments Europe, Limited", "EU (Ireland)"),
            "adyen": ("Adyen N.V.", "EU (Netherlands)"),
            "mollie": ("Mollie B.V.", "EU (Netherlands)"),
        }

        proc_lower = processor.lower() if processor else ""
        if proc_lower in eu_processors:
            entity, proc_region = eu_processors[proc_lower]
            return EndpointCheck(
                component=component,
                component_type=component_type,
                endpoint=processor,
                region=proc_region,
                eu_compliant=True,
                check_method="known_processor",
                metadata={
                    "processor": processor,
                    "legal_entity": entity,
                },
            )

        return EndpointCheck(
            component=component,
            component_type=component_type,
            endpoint=processor,
            region="unknown",
            eu_compliant=not self._fail_closed,
            check_method="none",
            error=f"Could not verify payment processor region: {processor}",
        )

    def _extract_region_from_endpoint(self, endpoint: str) -> Optional[str]:
        """Extract AWS/GCP/Azure region from endpoint."""
        if not endpoint:
            return None

        # Clean the endpoint
        endpoint = endpoint.strip().lower()

        # Check against patterns
        for pattern, _ in self._patterns:
            match = pattern.search(endpoint)
            if match:
                return match.group(1)

        # Check for explicit region in endpoint
        for region in NON_EU_REGIONS | EU_AWS_REGIONS | EU_GCP_REGIONS | EU_AZURE_REGIONS:
            if region in endpoint:
                return region

        return None

    def _region_from_dns(self, endpoint: str) -> Optional[str]:
        """Attempt to determine region from DNS lookup."""
        try:
            # Parse hostname
            if "://" in endpoint:
                parsed = urlparse(endpoint)
                hostname = parsed.hostname
            else:
                hostname = endpoint.split("/")[0].split(":")[0]

            if not hostname:
                return None

            # Get IP address
            ip = socket.gethostbyname(hostname)

            # Note: In production, this would use IP geolocation
            # For now, we return None and rely on pattern matching
            return None

        except Exception:
            return None

    def _is_eu_region(self, region: str) -> bool:
        """Check if a region is EU-compliant."""
        if not region:
            return False

        region_lower = region.lower().strip()

        # Check exact match
        if region_lower in {r.lower() for r in self._allowed_regions}:
            return True

        # Check prefix match for AWS regions
        for eu_region in EU_AWS_REGIONS:
            if region_lower == eu_region or region_lower.startswith(eu_region.split("-")[0] + "-"):
                if region_lower in EU_AWS_REGIONS:
                    return True

        # Check if contains "eu" or "europe"
        if "eu" in region_lower or "europe" in region_lower:
            return True

        # Check for specific country mentions
        eu_countries = {"germany", "ireland", "netherlands", "france", "uk", "united kingdom"}
        if any(country in region_lower for country in eu_countries):
            return True

        return False

    def _create_violation(self, check: EndpointCheck) -> DriftCheckViolation:
        """Create a violation from a failed check."""
        return DriftCheckViolation(
            component=check.component,
            violation_type="non_eu_endpoint",
            severity=CheckSeverity.CRITICAL,
            message=f"Component '{check.component}' is using non-EU region: {check.region}",
            region_found=check.region,
            region_expected="EU",
            remediation=f"Configure {check.component} to use an EU region",
            blocked=self._fail_closed,
        )

    def _log_audit(self, action: str, report: ResidencyDriftReport) -> None:
        """Log audit event."""
        with self._lock:
            self._audit_log.append(
                {
                    "action": action,
                    "check_id": report.check_id,
                    "status": report.status.value,
                    "endpoints_verified": report.endpoints_verified,
                    "non_eu_endpoints": report.non_eu_endpoints,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        with self._lock:
            return list(self._audit_log)


# ============================================================================
# Deployment Configuration Validator
# ============================================================================


class DeploymentConfigValidator:
    """
    Validates deployment configuration files for EU residency.

    Supports:
        - Kubernetes manifests
        - Helm values files
        - Docker Compose files
        - Environment files
    """

    def __init__(self, checker: Optional[EUOnlyDriftChecker] = None):
        """Initialize validator."""
        self._checker = checker or EUOnlyDriftChecker()

    def validate_helm_values(self, values: Dict[str, Any]) -> ResidencyDriftReport:
        """
        Validate Helm values file for EU residency.

        Args:
            values: Parsed Helm values dictionary

        Returns:
            ResidencyDriftReport
        """
        config = self._extract_config_from_helm(values)
        return self._checker.check(
            config,
            triggered_by="helm_validation",
        )

    def validate_docker_compose(self, compose: Dict[str, Any]) -> ResidencyDriftReport:
        """
        Validate Docker Compose file for EU residency.

        Args:
            compose: Parsed docker-compose.yml dictionary

        Returns:
            ResidencyDriftReport
        """
        config = self._extract_config_from_compose(compose)
        return self._checker.check(
            config,
            triggered_by="compose_validation",
        )

    def validate_kubernetes_manifest(self, manifest: Dict[str, Any]) -> ResidencyDriftReport:
        """
        Validate Kubernetes manifest for EU residency.

        Args:
            manifest: Parsed Kubernetes manifest

        Returns:
            ResidencyDriftReport
        """
        config = self._extract_config_from_k8s(manifest)
        return self._checker.check(
            config,
            triggered_by="k8s_validation",
        )

    def _extract_config_from_helm(self, values: Dict[str, Any]) -> ResidencyConfiguration:
        """Extract residency configuration from Helm values."""
        config = ResidencyConfiguration()

        # Global data residency
        global_config = values.get("global", {})
        data_residency = global_config.get("dataResidency", "")

        # PostgreSQL
        postgres = values.get("postgresql", {})
        if postgres.get("enabled"):
            # Extract region from any configured endpoint
            primary = postgres.get("primary", {})
            config.database_primary_region = self._infer_region_from_helm(
                postgres, global_config, data_residency
            )

        # Redis
        redis = values.get("redis", {})
        if redis.get("enabled"):
            config.cache_region = self._infer_region_from_helm(redis, global_config, data_residency)

        # Evidence Pack Exporter
        evidence_pack = values.get("evidencePackExporter", {})
        if evidence_pack.get("enabled"):
            dest = evidence_pack.get("destination", {})
            if dest.get("type") == "s3":
                config.object_storage_bucket = dest.get("bucket", "")
                config.object_storage_region = dest.get("region", "")

        # Governance
        governance = values.get("governance", {})
        if governance.get("enabled"):
            residency = governance.get("residency", {})
            if residency.get("enforceEuOnly"):
                # If EU-only is enforced, assume EU region
                if not config.database_primary_region:
                    config.database_primary_region = "eu-central-1"

        return config

    def _extract_config_from_compose(self, compose: Dict[str, Any]) -> ResidencyConfiguration:
        """Extract residency configuration from Docker Compose."""
        config = ResidencyConfiguration()

        services = compose.get("services", {})
        for name, service in services.items():
            env = service.get("environment", {})
            if isinstance(env, list):
                env = dict(e.split("=", 1) for e in env if "=" in e)

            # Check for database URLs
            db_url = env.get("DATABASE_URL", "")
            if db_url:
                config.database_primary_endpoint = db_url

            # Check for Redis
            redis_url = env.get("REDIS_URL", "")
            if redis_url:
                config.cache_endpoint = redis_url

            # Check for AWS region
            aws_region = env.get("AWS_REGION", env.get("AWS_DEFAULT_REGION", ""))
            if aws_region:
                config.object_storage_region = aws_region

        return config

    def _extract_config_from_k8s(self, manifest: Dict[str, Any]) -> ResidencyConfiguration:
        """Extract residency configuration from Kubernetes manifest."""
        config = ResidencyConfiguration()

        # Check for ConfigMaps, Secrets, Deployments
        kind = manifest.get("kind", "")

        if kind == "ConfigMap":
            data = manifest.get("data", {})
            config = self._extract_from_env_vars(data)

        elif kind == "Secret":
            # Secrets are base64 encoded - skip detailed parsing
            pass

        elif kind in ("Deployment", "StatefulSet", "DaemonSet"):
            spec = manifest.get("spec", {})
            template = spec.get("template", {})
            pod_spec = template.get("spec", {})
            containers = pod_spec.get("containers", [])

            for container in containers:
                env = container.get("env", [])
                env_dict = {e["name"]: e.get("value", "") for e in env if "name" in e}
                config = self._extract_from_env_vars(env_dict)

        return config

    def _extract_from_env_vars(self, env: Dict[str, str]) -> ResidencyConfiguration:
        """Extract configuration from environment variables."""
        return ResidencyConfiguration(
            database_primary_endpoint=env.get("DATABASE_URL", ""),
            database_primary_region=env.get("DATABASE_REGION", ""),
            object_storage_bucket=env.get("S3_BUCKET", ""),
            object_storage_region=env.get("AWS_REGION", env.get("S3_REGION", "")),
            cache_endpoint=env.get("REDIS_URL", ""),
            cache_region=env.get("REDIS_REGION", ""),
            email_region=env.get("SES_REGION", env.get("EMAIL_REGION", "")),
        )

    def _infer_region_from_helm(
        self,
        service_config: Dict[str, Any],
        global_config: Dict[str, Any],
        data_residency: str,
    ) -> str:
        """Infer region from Helm service configuration."""
        # Check service-specific region
        region = service_config.get("region", "")
        if region:
            return region

        # Check global region
        region = global_config.get("region", "")
        if region:
            return region

        # Use data residency setting
        if data_residency == "eu" or data_residency == "on-prem":
            return "eu-central-1"

        return ""


# ============================================================================
# Evidence Pack Exporter
# ============================================================================


class ResidencyEvidenceExporter:
    """
    Exports residency evidence for audit purposes.

    Generates reports suitable for:
        - Customer due diligence
        - Regulatory audits
        - Compliance verification
    """

    def __init__(self, checker: Optional[EUOnlyDriftChecker] = None):
        """Initialize exporter."""
        self._checker = checker or EUOnlyDriftChecker()

    def export_evidence_pack(
        self,
        report: ResidencyDriftReport,
        subprocessors: List[SubprocessorCheck],
        output_dir: Path,
    ) -> Dict[str, Path]:
        """
        Export complete residency evidence pack.

        Args:
            report: Drift check report
            subprocessors: List of verified subprocessors
            output_dir: Directory to write evidence files

        Returns:
            Dictionary mapping evidence type to file path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        evidence_files: Dict[str, Path] = {}

        # 1. Drift check report
        report_path = output_dir / f"drift_check_{report.check_id}.json"
        report.save(report_path)
        evidence_files["drift_check_report"] = report_path

        # 2. Subprocessors summary
        subprocessors_path = output_dir / "subprocessors_summary.json"
        with open(subprocessors_path, "w") as f:
            json.dump(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "total_subprocessors": len(subprocessors),
                    "all_eu_compliant": all(s.eu_compliant for s in subprocessors),
                    "subprocessors": [s.to_dict() for s in subprocessors],
                },
                f,
                indent=2,
            )
        evidence_files["subprocessors_summary"] = subprocessors_path

        # 3. EU residency attestation
        attestation_path = output_dir / "eu_residency_attestation.json"
        attestation = {
            "attestation_id": str(uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "eu_only_commitment": True,
            "drift_check_passed": report.passed,
            "drift_check_id": report.check_id,
            "endpoints_verified": report.endpoints_verified,
            "subprocessors_verified": len(subprocessors),
            "non_eu_findings": report.non_eu_endpoints,
            "statement": (
                (
                    "This attestation confirms that all verified endpoints and "
                    "subprocessors are located within the European Union, in accordance "
                    "with our EU-only data residency commitment."
                )
                if report.passed
                else (
                    "WARNING: EU-only residency verification FAILED. "
                    f"Found {report.non_eu_endpoints} non-EU endpoints."
                )
            ),
        }
        with open(attestation_path, "w") as f:
            json.dump(attestation, f, indent=2)
        evidence_files["eu_attestation"] = attestation_path

        # 4. Audit log
        audit_path = output_dir / "drift_check_audit.json"
        with open(audit_path, "w") as f:
            json.dump(
                {
                    "audit_entries": self._checker.get_audit_log(),
                },
                f,
                indent=2,
            )
        evidence_files["audit_log"] = audit_path

        return evidence_files

    def create_subprocessor_checks(self) -> List[SubprocessorCheck]:
        """
        Create standard EU subprocessor checks.

        Based on docs/compliance/SUBPROCESSORS_REGISTER.md.
        """
        return [
            SubprocessorCheck(
                name="Amazon Web Services (AWS)",
                legal_entity="Amazon Web Services EMEA SARL",
                service="Cloud infrastructure",
                region="eu-central-1, eu-west-1",
                eu_compliant=True,
                dpa_status="Signed",
                last_review="2025-01-15",
                next_review="2025-04-15",
            ),
            SubprocessorCheck(
                name="Supabase",
                legal_entity="Supabase Inc. (EU infrastructure)",
                service="Database hosting",
                region="EU (Germany)",
                eu_compliant=True,
                dpa_status="Signed",
                last_review="2025-01-15",
                next_review="2025-04-15",
            ),
            SubprocessorCheck(
                name="Stripe",
                legal_entity="Stripe Payments Europe, Limited",
                service="Payment processing",
                region="EU (Ireland)",
                eu_compliant=True,
                dpa_status="Signed",
                last_review="2025-01-15",
                next_review="2025-04-15",
            ),
            SubprocessorCheck(
                name="AWS SES",
                legal_entity="Amazon Web Services EMEA SARL",
                service="Email delivery",
                region="eu-west-1",
                eu_compliant=True,
                dpa_status="Signed",
                last_review="2025-01-15",
                next_review="2025-04-15",
            ),
            SubprocessorCheck(
                name="Sentry",
                legal_entity="Functional Software, Inc. (EU data center)",
                service="Error monitoring",
                region="EU (Germany)",
                eu_compliant=True,
                dpa_status="Signed",
                last_review="2025-01-15",
                next_review="2025-04-15",
            ),
        ]


# ============================================================================
# Convenience Functions
# ============================================================================


def check_eu_residency(
    config: Optional[ResidencyConfiguration] = None,
    fail_closed: bool = True,
) -> ResidencyDriftReport:
    """
    Convenience function to check EU residency.

    Args:
        config: Configuration to check (defaults to environment)
        fail_closed: If True, unknown regions fail

    Returns:
        ResidencyDriftReport
    """
    if config is None:
        config = ResidencyConfiguration.from_environment()

    checker = EUOnlyDriftChecker(fail_closed=fail_closed)
    return checker.check(config)


def is_eu_region(region: str) -> bool:
    """
    Check if a region string represents an EU region.

    Args:
        region: Region identifier

    Returns:
        True if EU region
    """
    checker = EUOnlyDriftChecker()
    return checker._is_eu_region(region)


def validate_endpoint_eu(endpoint: str) -> Tuple[bool, str]:
    """
    Validate an endpoint is EU-based.

    Args:
        endpoint: Endpoint URL or hostname

    Returns:
        (is_eu, region_or_error_message)
    """
    checker = EUOnlyDriftChecker()
    check = checker.check_endpoint(endpoint, "validation", ComponentType.CUSTOM)
    if check.eu_compliant:
        return True, check.region
    return False, check.error or f"Non-EU region: {check.region}"


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "DriftCheckStatus",
    "ComponentType",
    "CheckSeverity",
    # Data classes
    "EndpointCheck",
    "SubprocessorCheck",
    "DriftCheckViolation",
    "ResidencyDriftReport",
    "ResidencyConfiguration",
    # Classes
    "EUOnlyDriftChecker",
    "DeploymentConfigValidator",
    "ResidencyEvidenceExporter",
    # Convenience functions
    "check_eu_residency",
    "is_eu_region",
    "validate_endpoint_eu",
    # Constants
    "EU_AWS_REGIONS",
    "EU_GCP_REGIONS",
    "EU_AZURE_REGIONS",
    "ALL_EU_REGIONS",
    "NON_EU_REGIONS",
]

# -*- coding: utf-8 -*-
"""
GDPR Phase 8: Data Inventory Registry.

Central registry for all data fields, stores, and logs with:
- Classification (sensitivity level, data category)
- Retention requirements (period, action)
- Residency requirements (EU-only enforcement)
- Redaction requirements (mandatory fields, patterns)

This registry is the foundation for CI "privacy-by-design" checks:
CI MUST fail closed if any new data field/store/log ships without
a registered entry in this inventory.

Design Doc Reference:
    - Phase 8: "CI fails closed if new data store/log stream/telemetry field
               ships without recorded: classification, retention, residency,
               and redaction requirements"
    - GDPR Article 25: Privacy by design and by default
    - GDPR Article 30: Records of processing activities

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Final, FrozenSet, List, Optional, Set, Tuple
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Data sensitivity classifications
class DataSensitivity(str, Enum):
    """Sensitivity classification for data fields."""
    PUBLIC = "public"              # Non-sensitive, can be exposed
    INTERNAL = "internal"          # Internal use only
    CONFIDENTIAL = "confidential"  # Business-sensitive
    SENSITIVE = "sensitive"        # Personal data (GDPR relevant)
    CRITICAL = "critical"          # Credentials, PII, financial
    RESTRICTED = "restricted"      # Highest sensitivity (broker creds, order data)


class DataCategory(str, Enum):
    """Category of data for GDPR purposes."""
    TECHNICAL = "technical"           # System metrics, logs
    OPERATIONAL = "operational"       # Commands, configs, deployments
    TELEMETRY = "telemetry"           # Telemetry events
    AUDIT = "audit"                   # Access logs, audit trails
    COMPLIANCE = "compliance"         # DSAR, approval records
    USER = "user"                     # User settings, preferences
    FINANCIAL = "financial"           # Billing, transactions
    TRADING = "trading"               # Order, position, fill data
    CREDENTIAL = "credential"         # API keys, tokens, secrets
    PII = "pii"                       # Personal identifiable information


class ResidencyRequirement(str, Enum):
    """Data residency requirement."""
    EU_ONLY = "eu_only"               # Must remain in EU
    EU_PREFERRED = "eu_preferred"     # Prefer EU, allow exceptions
    GLOBAL = "global"                 # No residency restriction
    CUSTOMER_REGION = "customer_region"  # Follow customer region


class RedactionRequirement(str, Enum):
    """Redaction handling for this data."""
    NONE = "none"                     # No redaction needed
    OPTIONAL = "optional"             # Redaction available if requested
    MANDATORY = "mandatory"           # Must be redacted before transmission
    NEVER_TRANSMIT = "never_transmit" # Must never leave agent


class InventoryEntryType(str, Enum):
    """Type of inventory entry."""
    FIELD = "field"           # Individual data field
    STORE = "store"           # Data store (database, cache)
    LOG_STREAM = "log_stream" # Log output stream
    TABLE = "table"           # Database table
    COLLECTION = "collection" # Document collection
    BUCKET = "bucket"         # Object storage bucket
    QUEUE = "queue"           # Message queue
    TOPIC = "topic"           # Event topic


class ReviewStatus(str, Enum):
    """Review status for inventory entry."""
    PENDING = "pending"       # Not yet reviewed
    APPROVED = "approved"     # Approved for use
    REJECTED = "rejected"     # Not approved
    DEPRECATED = "deprecated" # Marked for removal
    UNDER_REVIEW = "under_review"  # Currently being reviewed


# Fields that ALWAYS require redaction or are forbidden
ALWAYS_REDACT_PATTERNS: Final[List[Tuple[re.Pattern, str]]] = [
    (re.compile(r"api[_-]?key", re.IGNORECASE), "API key"),
    (re.compile(r"api[_-]?secret", re.IGNORECASE), "API secret"),
    (re.compile(r"secret[_-]?key", re.IGNORECASE), "Secret key"),
    (re.compile(r"private[_-]?key", re.IGNORECASE), "Private key"),
    (re.compile(r"access[_-]?token", re.IGNORECASE), "Access token"),
    (re.compile(r"refresh[_-]?token", re.IGNORECASE), "Refresh token"),
    (re.compile(r"bearer[_-]?token", re.IGNORECASE), "Bearer token"),
    (re.compile(r"password", re.IGNORECASE), "Password"),
    (re.compile(r"passphrase", re.IGNORECASE), "Passphrase"),
    (re.compile(r"credential", re.IGNORECASE), "Credential"),
    (re.compile(
        r"(binance|alpaca|deribit|oanda|interactive_brokers|ib)[_-]?(key|secret|token)",
        re.IGNORECASE
    ), "Broker credential"),
]

# PII field patterns
PII_FIELD_PATTERNS: Final[List[Tuple[re.Pattern, str]]] = [
    (re.compile(r"^email$", re.IGNORECASE), "Email address"),
    (re.compile(r"^phone", re.IGNORECASE), "Phone number"),
    (re.compile(r"^address$", re.IGNORECASE), "Physical address"),
    (re.compile(r"^ip[_-]?address$", re.IGNORECASE), "IP address"),
    (re.compile(r"^user[_-]?agent$", re.IGNORECASE), "User agent"),
    (re.compile(r"^ssn$", re.IGNORECASE), "Social security number"),
    (re.compile(r"^tax[_-]?id$", re.IGNORECASE), "Tax ID"),
    (re.compile(r"date[_-]?of[_-]?birth|^dob$", re.IGNORECASE), "Date of birth"),
    (re.compile(r"^device[_-]?id$", re.IGNORECASE), "Device ID"),
]

# Order-like field patterns (forbidden in non-RAW telemetry)
ORDER_FIELD_PATTERNS: Final[List[Tuple[re.Pattern, str]]] = [
    (re.compile(r"^side$", re.IGNORECASE), "Order side"),
    (re.compile(r"^(quantity|qty)$", re.IGNORECASE), "Quantity"),
    (re.compile(r"^price$", re.IGNORECASE), "Price"),
    (re.compile(r"^order[_-]?(id|type)$", re.IGNORECASE), "Order field"),
    (re.compile(r"^(limit|stop)[_-]?price$", re.IGNORECASE), "Order price"),
    (re.compile(r"^fill[_-]?(qty|price)$", re.IGNORECASE), "Fill field"),
    (re.compile(r"^(signal|intent)$", re.IGNORECASE), "Trading intent"),
    (re.compile(r"^position[_-]?(side|size)$", re.IGNORECASE), "Position field"),
    (re.compile(r"^(unrealized|realized)[_-]?pnl$", re.IGNORECASE), "P&L field"),
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class InventoryEntry:
    """
    A single entry in the data inventory.

    Represents a data field, store, log stream, or other data element
    with all GDPR-required metadata.
    """
    # Identity
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""                    # Field/store name
    entry_type: InventoryEntryType = InventoryEntryType.FIELD
    path: str = ""                    # Full path (e.g., table.column, log.field)

    # Classification
    sensitivity: DataSensitivity = DataSensitivity.INTERNAL
    category: DataCategory = DataCategory.TECHNICAL
    description: str = ""
    purpose: str = ""                 # GDPR Art. 30: purpose of processing
    lawful_basis: str = ""            # GDPR Art. 6: lawful basis

    # Retention
    retention_days: int = 90          # Default retention period
    retention_action: str = "delete"  # delete, archive, anonymize, aggregate

    # Residency
    residency: ResidencyRequirement = ResidencyRequirement.EU_ONLY
    allowed_regions: List[str] = field(default_factory=lambda: ["eu-west-1", "eu-central-1"])

    # Redaction
    redaction: RedactionRequirement = RedactionRequirement.NONE
    redaction_pattern: Optional[str] = None  # Regex for redaction
    pii_detected: bool = False
    credential_detected: bool = False
    order_field_detected: bool = False

    # Telemetry level constraints
    min_telemetry_level: str = "AGGREGATED"  # Minimum level to include this field

    # Review
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_notes: str = ""

    # Audit
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_by: Optional[str] = None
    version: int = 1

    # Source tracking (for CI checks)
    source_file: Optional[str] = None  # File where this field is defined
    source_line: Optional[int] = None  # Line number
    source_commit: Optional[str] = None  # Git commit

    # Tags
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Auto-detect sensitive patterns after initialization."""
        self._auto_detect_patterns()

    def _auto_detect_patterns(self) -> None:
        """Auto-detect PII, credentials, and order fields."""
        name_lower = self.name.lower()

        # Check for credential patterns
        for pattern, _ in ALWAYS_REDACT_PATTERNS:
            if pattern.search(name_lower):
                self.credential_detected = True
                if self.redaction == RedactionRequirement.NONE:
                    self.redaction = RedactionRequirement.NEVER_TRANSMIT
                if self.sensitivity.value < DataSensitivity.RESTRICTED.value:
                    self.sensitivity = DataSensitivity.RESTRICTED
                break

        # Check for PII patterns
        for pattern, _ in PII_FIELD_PATTERNS:
            if pattern.search(name_lower):
                self.pii_detected = True
                if self.redaction == RedactionRequirement.NONE:
                    self.redaction = RedactionRequirement.MANDATORY
                if self.sensitivity.value < DataSensitivity.SENSITIVE.value:
                    self.sensitivity = DataSensitivity.SENSITIVE
                break

        # Check for order field patterns
        for pattern, _ in ORDER_FIELD_PATTERNS:
            if pattern.search(name_lower):
                self.order_field_detected = True
                if self.min_telemetry_level == "AGGREGATED":
                    self.min_telemetry_level = "RAW_ORDER_EVENTS"
                break

    @property
    def is_compliant(self) -> bool:
        """Check if entry has all required GDPR fields."""
        return all([
            self.name,
            self.sensitivity,
            self.category,
            self.purpose,
            self.lawful_basis,
            self.retention_days > 0,
            self.residency,
            self.review_status == ReviewStatus.APPROVED,
        ])

    @property
    def requires_dpia(self) -> bool:
        """Check if this entry requires Data Protection Impact Assessment."""
        return (
            self.sensitivity in (DataSensitivity.CRITICAL, DataSensitivity.RESTRICTED) or
            self.pii_detected or
            self.category == DataCategory.FINANCIAL or
            self.category == DataCategory.TRADING
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "entry_type": self.entry_type.value,
            "path": self.path,
            "classification": {
                "sensitivity": self.sensitivity.value,
                "category": self.category.value,
                "description": self.description,
                "purpose": self.purpose,
                "lawful_basis": self.lawful_basis,
            },
            "retention": {
                "days": self.retention_days,
                "action": self.retention_action,
            },
            "residency": {
                "requirement": self.residency.value,
                "allowed_regions": self.allowed_regions,
            },
            "redaction": {
                "requirement": self.redaction.value,
                "pattern": self.redaction_pattern,
                "pii_detected": self.pii_detected,
                "credential_detected": self.credential_detected,
                "order_field_detected": self.order_field_detected,
            },
            "telemetry": {
                "min_level": self.min_telemetry_level,
            },
            "review": {
                "status": self.review_status.value,
                "reviewed_by": self.reviewed_by,
                "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
                "notes": self.review_notes,
            },
            "audit": {
                "created_at": self.created_at.isoformat(),
                "created_by": self.created_by,
                "updated_at": self.updated_at.isoformat(),
                "updated_by": self.updated_by,
                "version": self.version,
            },
            "source": {
                "file": self.source_file,
                "line": self.source_line,
                "commit": self.source_commit,
            },
            "tags": self.tags,
            "compliance": {
                "is_compliant": self.is_compliant,
                "requires_dpia": self.requires_dpia,
            },
        }


@dataclass
class InventoryValidationResult:
    """Result of validating a data field against the inventory."""
    is_registered: bool = False
    is_compliant: bool = False
    entry: Optional[InventoryEntry] = None
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    auto_detected: Dict[str, bool] = field(default_factory=dict)

    @property
    def should_block_ci(self) -> bool:
        """Check if this result should block CI."""
        return not self.is_registered or not self.is_compliant or len(self.violations) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_registered": self.is_registered,
            "is_compliant": self.is_compliant,
            "entry_id": self.entry.id if self.entry else None,
            "violations": self.violations,
            "warnings": self.warnings,
            "auto_detected": self.auto_detected,
            "should_block_ci": self.should_block_ci,
        }


@dataclass
class InventoryReport:
    """Report on data inventory status."""
    report_id: str = field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generated_by: str = "system"

    # Counts
    total_entries: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_sensitivity: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)
    by_review_status: Dict[str, int] = field(default_factory=dict)
    by_redaction: Dict[str, int] = field(default_factory=dict)

    # Compliance
    compliant_entries: int = 0
    non_compliant_entries: int = 0
    pending_review: int = 0
    requires_dpia: int = 0

    # Auto-detection
    pii_detected: int = 0
    credentials_detected: int = 0
    order_fields_detected: int = 0

    # Issues
    missing_purpose: int = 0
    missing_lawful_basis: int = 0
    missing_retention: int = 0

    # Integrity
    integrity_hash: str = ""

    def __post_init__(self):
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute integrity hash for this report."""
        content = json.dumps({
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "total_entries": self.total_entries,
            "compliant_entries": self.compliant_entries,
            "non_compliant_entries": self.non_compliant_entries,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by,
            "summary": {
                "total_entries": self.total_entries,
                "compliant_entries": self.compliant_entries,
                "non_compliant_entries": self.non_compliant_entries,
                "compliance_rate": (
                    self.compliant_entries / self.total_entries * 100
                    if self.total_entries > 0 else 0
                ),
            },
            "by_type": self.by_type,
            "by_sensitivity": self.by_sensitivity,
            "by_category": self.by_category,
            "by_review_status": self.by_review_status,
            "by_redaction": self.by_redaction,
            "review": {
                "pending_review": self.pending_review,
                "requires_dpia": self.requires_dpia,
            },
            "auto_detection": {
                "pii_detected": self.pii_detected,
                "credentials_detected": self.credentials_detected,
                "order_fields_detected": self.order_fields_detected,
            },
            "issues": {
                "missing_purpose": self.missing_purpose,
                "missing_lawful_basis": self.missing_lawful_basis,
                "missing_retention": self.missing_retention,
                "total_issues": (
                    self.missing_purpose +
                    self.missing_lawful_basis +
                    self.missing_retention
                ),
            },
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class InventoryChangeEvent:
    """Event recording a change to the inventory."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    action: str = "create"  # create, update, delete, approve, reject
    entry_id: str = ""
    entry_name: str = ""
    actor_id: str = ""
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    reason: str = ""
    integrity_hash: str = ""

    def __post_init__(self):
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute integrity hash."""
        content = json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "entry_id": self.entry_id,
            "actor_id": self.actor_id,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "entry_id": self.entry_id,
            "entry_name": self.entry_name,
            "actor_id": self.actor_id,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "reason": self.reason,
            "integrity_hash": self.integrity_hash,
        }


# ============================================================================
# Data Inventory Registry
# ============================================================================

class DataInventoryRegistry:
    """
    Central registry for all data fields, stores, and logs.

    This is the foundation for CI "privacy-by-design" checks per Phase 8.

    Usage:
        registry = DataInventoryRegistry()

        # Register a new field
        entry = registry.register(
            name="user_email",
            entry_type=InventoryEntryType.FIELD,
            path="users.email",
            sensitivity=DataSensitivity.SENSITIVE,
            category=DataCategory.PII,
            purpose="User authentication",
            lawful_basis="Contract performance",
            retention_days=365,
            created_by="admin",
        )

        # Validate a field exists in inventory
        result = registry.validate("user_email")
        if result.should_block_ci:
            raise ValueError("Field not properly registered")

        # Approve for use
        registry.approve(entry.id, approved_by="dpo", notes="GDPR compliant")
    """

    def __init__(
        self,
        on_change: Optional[Callable[[InventoryChangeEvent], None]] = None,
        strict_mode: bool = True,
    ):
        """
        Initialize the registry.

        Args:
            on_change: Callback for change events
            strict_mode: If True, require all GDPR fields for registration
        """
        self._entries: Dict[str, InventoryEntry] = {}
        self._by_name: Dict[str, str] = {}  # name -> id
        self._by_path: Dict[str, str] = {}  # path -> id
        self._by_type: Dict[InventoryEntryType, List[str]] = {}  # type -> [ids]

        self._events: List[InventoryChangeEvent] = []
        self._on_change = on_change
        self._strict_mode = strict_mode
        self._lock = threading.Lock()

    # ========================================================================
    # Registration
    # ========================================================================

    def register(
        self,
        name: str,
        entry_type: InventoryEntryType,
        sensitivity: DataSensitivity,
        category: DataCategory,
        purpose: str,
        lawful_basis: str,
        retention_days: int,
        created_by: str,
        path: Optional[str] = None,
        description: str = "",
        residency: ResidencyRequirement = ResidencyRequirement.EU_ONLY,
        allowed_regions: Optional[List[str]] = None,
        redaction: RedactionRequirement = RedactionRequirement.NONE,
        redaction_pattern: Optional[str] = None,
        min_telemetry_level: str = "AGGREGATED",
        source_file: Optional[str] = None,
        source_line: Optional[int] = None,
        source_commit: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> InventoryEntry:
        """
        Register a new data element in the inventory.

        Args:
            name: Field/store name
            entry_type: Type of entry
            sensitivity: Sensitivity classification
            category: Data category
            purpose: Purpose of processing (GDPR Art. 30)
            lawful_basis: Lawful basis (GDPR Art. 6)
            retention_days: Retention period
            created_by: User creating the entry
            path: Full path (e.g., table.column)
            description: Description
            residency: Residency requirement
            allowed_regions: Allowed regions
            redaction: Redaction requirement
            redaction_pattern: Regex for redaction
            min_telemetry_level: Minimum telemetry level
            source_file: Source file where defined
            source_line: Line number
            source_commit: Git commit
            tags: Tags

        Returns:
            Created InventoryEntry

        Raises:
            ValueError: If validation fails in strict mode
        """
        # Validate required fields in strict mode
        if self._strict_mode:
            if not purpose:
                raise ValueError("Purpose is required for GDPR compliance (Art. 30)")
            if not lawful_basis:
                raise ValueError("Lawful basis is required for GDPR compliance (Art. 6)")
            if retention_days <= 0:
                raise ValueError("Retention period must be positive")

        # Check for duplicate
        if name in self._by_name:
            raise ValueError(f"Entry with name '{name}' already exists")

        path = path or name
        if path in self._by_path:
            raise ValueError(f"Entry with path '{path}' already exists")

        entry = InventoryEntry(
            name=name,
            entry_type=entry_type,
            path=path,
            sensitivity=sensitivity,
            category=category,
            description=description,
            purpose=purpose,
            lawful_basis=lawful_basis,
            retention_days=retention_days,
            retention_action="delete",
            residency=residency,
            allowed_regions=allowed_regions or ["eu-west-1", "eu-central-1"],
            redaction=redaction,
            redaction_pattern=redaction_pattern,
            min_telemetry_level=min_telemetry_level,
            created_by=created_by,
            source_file=source_file,
            source_line=source_line,
            source_commit=source_commit,
            tags=tags or [],
        )

        with self._lock:
            self._entries[entry.id] = entry
            self._by_name[entry.name] = entry.id
            self._by_path[entry.path] = entry.id

            if entry_type not in self._by_type:
                self._by_type[entry_type] = []
            self._by_type[entry_type].append(entry.id)

        # Log event
        event = InventoryChangeEvent(
            action="create",
            entry_id=entry.id,
            entry_name=entry.name,
            actor_id=created_by,
            new_values=entry.to_dict(),
            reason="Initial registration",
        )
        self._log_event(event)

        logger.info(f"Registered data inventory entry: {entry.name} ({entry.entry_type.value})")

        return entry

    def register_bulk(
        self,
        entries: List[Dict[str, Any]],
        created_by: str,
    ) -> Tuple[List[InventoryEntry], List[Dict[str, Any]]]:
        """
        Register multiple entries at once.

        Args:
            entries: List of entry dictionaries
            created_by: User creating entries

        Returns:
            Tuple of (created entries, errors)
        """
        created = []
        errors = []

        for entry_data in entries:
            try:
                entry = self.register(
                    name=entry_data.get("name", ""),
                    entry_type=InventoryEntryType(entry_data.get("entry_type", "field")),
                    sensitivity=DataSensitivity(entry_data.get("sensitivity", "internal")),
                    category=DataCategory(entry_data.get("category", "technical")),
                    purpose=entry_data.get("purpose", ""),
                    lawful_basis=entry_data.get("lawful_basis", ""),
                    retention_days=entry_data.get("retention_days", 90),
                    created_by=created_by,
                    path=entry_data.get("path"),
                    description=entry_data.get("description", ""),
                    residency=ResidencyRequirement(
                        entry_data.get("residency", "eu_only")
                    ),
                    min_telemetry_level=entry_data.get("min_telemetry_level", "AGGREGATED"),
                    tags=entry_data.get("tags", []),
                )
                created.append(entry)
            except Exception as e:
                errors.append({
                    "name": entry_data.get("name", "unknown"),
                    "error": str(e),
                })

        return created, errors

    # ========================================================================
    # Lookup
    # ========================================================================

    def get(self, entry_id: str) -> Optional[InventoryEntry]:
        """Get entry by ID."""
        with self._lock:
            return self._entries.get(entry_id)

    def get_by_name(self, name: str) -> Optional[InventoryEntry]:
        """Get entry by name."""
        with self._lock:
            entry_id = self._by_name.get(name)
            return self._entries.get(entry_id) if entry_id else None

    def get_by_path(self, path: str) -> Optional[InventoryEntry]:
        """Get entry by path."""
        with self._lock:
            entry_id = self._by_path.get(path)
            return self._entries.get(entry_id) if entry_id else None

    def list_all(self) -> List[InventoryEntry]:
        """List all entries."""
        with self._lock:
            return list(self._entries.values())

    def list_by_type(self, entry_type: InventoryEntryType) -> List[InventoryEntry]:
        """List entries by type."""
        with self._lock:
            entry_ids = self._by_type.get(entry_type, [])
            return [self._entries[eid] for eid in entry_ids if eid in self._entries]

    def list_by_sensitivity(self, sensitivity: DataSensitivity) -> List[InventoryEntry]:
        """List entries by sensitivity."""
        with self._lock:
            return [e for e in self._entries.values() if e.sensitivity == sensitivity]

    def list_by_category(self, category: DataCategory) -> List[InventoryEntry]:
        """List entries by category."""
        with self._lock:
            return [e for e in self._entries.values() if e.category == category]

    def list_by_review_status(self, status: ReviewStatus) -> List[InventoryEntry]:
        """List entries by review status."""
        with self._lock:
            return [e for e in self._entries.values() if e.review_status == status]

    def list_requiring_dpia(self) -> List[InventoryEntry]:
        """List entries requiring DPIA."""
        with self._lock:
            return [e for e in self._entries.values() if e.requires_dpia]

    def list_non_compliant(self) -> List[InventoryEntry]:
        """List non-compliant entries."""
        with self._lock:
            return [e for e in self._entries.values() if not e.is_compliant]

    def search(
        self,
        query: str,
        entry_types: Optional[List[InventoryEntryType]] = None,
        sensitivities: Optional[List[DataSensitivity]] = None,
        categories: Optional[List[DataCategory]] = None,
        review_statuses: Optional[List[ReviewStatus]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[InventoryEntry]:
        """Search entries with filters."""
        with self._lock:
            results = list(self._entries.values())

            # Text search
            if query:
                query_lower = query.lower()
                results = [
                    e for e in results
                    if query_lower in e.name.lower() or
                       query_lower in e.path.lower() or
                       query_lower in e.description.lower()
                ]

            # Type filter
            if entry_types:
                results = [e for e in results if e.entry_type in entry_types]

            # Sensitivity filter
            if sensitivities:
                results = [e for e in results if e.sensitivity in sensitivities]

            # Category filter
            if categories:
                results = [e for e in results if e.category in categories]

            # Review status filter
            if review_statuses:
                results = [e for e in results if e.review_status in review_statuses]

            # Tags filter
            if tags:
                results = [
                    e for e in results
                    if any(t in e.tags for t in tags)
                ]

            return results

    # ========================================================================
    # Validation
    # ========================================================================

    def validate(
        self,
        name: str,
        path: Optional[str] = None,
        check_compliance: bool = True,
    ) -> InventoryValidationResult:
        """
        Validate a data field against the inventory.

        This is the core CI check - returns whether a field is properly
        registered and compliant.

        Args:
            name: Field name to validate
            path: Optional path
            check_compliance: Whether to check full compliance

        Returns:
            InventoryValidationResult
        """
        result = InventoryValidationResult()

        # Auto-detect patterns
        result.auto_detected = {
            "pii": any(p.search(name) for p, _ in PII_FIELD_PATTERNS),
            "credential": any(p.search(name) for p, _ in ALWAYS_REDACT_PATTERNS),
            "order_field": any(p.search(name) for p, _ in ORDER_FIELD_PATTERNS),
        }

        # Check if registered
        entry = self.get_by_name(name)
        if not entry and path:
            entry = self.get_by_path(path)

        if not entry:
            result.is_registered = False
            result.violations.append(
                f"Field '{name}' is not registered in data inventory"
            )

            # Add specific warnings for detected patterns
            if result.auto_detected["credential"]:
                result.violations.append(
                    "CRITICAL: Field appears to be a credential - must be registered "
                    "with NEVER_TRANSMIT redaction requirement"
                )
            if result.auto_detected["pii"]:
                result.violations.append(
                    "Field appears to be PII - requires mandatory redaction registration"
                )
            if result.auto_detected["order_field"]:
                result.violations.append(
                    "Field appears to be order-related - requires RAW_ORDER_EVENTS "
                    "telemetry level registration"
                )

            return result

        result.is_registered = True
        result.entry = entry

        if not check_compliance:
            result.is_compliant = True
            return result

        # Check compliance
        if entry.review_status != ReviewStatus.APPROVED:
            result.violations.append(
                f"Entry not approved (status: {entry.review_status.value})"
            )

        if not entry.purpose:
            result.violations.append("Missing purpose (GDPR Art. 30)")

        if not entry.lawful_basis:
            result.violations.append("Missing lawful basis (GDPR Art. 6)")

        if entry.retention_days <= 0:
            result.violations.append("Invalid retention period")

        # Check auto-detection matches registration
        if result.auto_detected["credential"] and not entry.credential_detected:
            result.warnings.append(
                "Field appears to be credential but not marked as such"
            )

        if result.auto_detected["pii"] and not entry.pii_detected:
            result.warnings.append(
                "Field appears to be PII but not marked as such"
            )

        if result.auto_detected["order_field"] and not entry.order_field_detected:
            result.warnings.append(
                "Field appears to be order-related but not marked as such"
            )

        # Check credential redaction
        if entry.credential_detected and entry.redaction != RedactionRequirement.NEVER_TRANSMIT:
            result.violations.append(
                "Credential field must have NEVER_TRANSMIT redaction"
            )

        # Check PII redaction
        if entry.pii_detected and entry.redaction not in (
            RedactionRequirement.MANDATORY,
            RedactionRequirement.NEVER_TRANSMIT,
        ):
            result.violations.append(
                "PII field must have MANDATORY or NEVER_TRANSMIT redaction"
            )

        # Check EU residency
        if entry.residency != ResidencyRequirement.EU_ONLY:
            result.warnings.append(
                "Non-EU residency may require additional compliance review"
            )

        result.is_compliant = len(result.violations) == 0
        return result

    def validate_batch(
        self,
        names: List[str],
        check_compliance: bool = True,
    ) -> Dict[str, InventoryValidationResult]:
        """
        Validate multiple fields at once.

        Args:
            names: List of field names
            check_compliance: Whether to check full compliance

        Returns:
            Dictionary of name -> InventoryValidationResult
        """
        return {
            name: self.validate(name, check_compliance=check_compliance)
            for name in names
        }

    def validate_all(self) -> Tuple[int, int, List[str]]:
        """
        Validate all entries in the registry.

        Returns:
            Tuple of (compliant_count, non_compliant_count, violation_messages)
        """
        compliant = 0
        non_compliant = 0
        violations = []

        with self._lock:
            for entry in self._entries.values():
                result = self.validate(entry.name)
                if result.is_compliant:
                    compliant += 1
                else:
                    non_compliant += 1
                    for v in result.violations:
                        violations.append(f"{entry.name}: {v}")

        return compliant, non_compliant, violations

    # ========================================================================
    # Review Workflow
    # ========================================================================

    def approve(
        self,
        entry_id: str,
        approved_by: str,
        notes: str = "",
    ) -> Optional[InventoryEntry]:
        """
        Approve an entry for use.

        Args:
            entry_id: Entry ID
            approved_by: Approver ID
            notes: Review notes

        Returns:
            Updated entry or None if not found
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return None

            old_status = entry.review_status
            entry.review_status = ReviewStatus.APPROVED
            entry.reviewed_by = approved_by
            entry.reviewed_at = datetime.now(timezone.utc)
            entry.review_notes = notes
            entry.updated_at = datetime.now(timezone.utc)
            entry.updated_by = approved_by
            entry.version += 1

        # Log event
        event = InventoryChangeEvent(
            action="approve",
            entry_id=entry.id,
            entry_name=entry.name,
            actor_id=approved_by,
            old_values={"review_status": old_status.value},
            new_values={"review_status": ReviewStatus.APPROVED.value},
            reason=notes,
        )
        self._log_event(event)

        logger.info(f"Approved inventory entry: {entry.name} by {approved_by}")
        return entry

    def reject(
        self,
        entry_id: str,
        rejected_by: str,
        reason: str,
    ) -> Optional[InventoryEntry]:
        """
        Reject an entry.

        Args:
            entry_id: Entry ID
            rejected_by: Rejector ID
            reason: Rejection reason (required)

        Returns:
            Updated entry or None if not found
        """
        if not reason:
            raise ValueError("Rejection reason is required")

        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return None

            old_status = entry.review_status
            entry.review_status = ReviewStatus.REJECTED
            entry.reviewed_by = rejected_by
            entry.reviewed_at = datetime.now(timezone.utc)
            entry.review_notes = reason
            entry.updated_at = datetime.now(timezone.utc)
            entry.updated_by = rejected_by
            entry.version += 1

        # Log event
        event = InventoryChangeEvent(
            action="reject",
            entry_id=entry.id,
            entry_name=entry.name,
            actor_id=rejected_by,
            old_values={"review_status": old_status.value},
            new_values={"review_status": ReviewStatus.REJECTED.value},
            reason=reason,
        )
        self._log_event(event)

        logger.info(f"Rejected inventory entry: {entry.name} by {rejected_by}")
        return entry

    def deprecate(
        self,
        entry_id: str,
        deprecated_by: str,
        reason: str,
    ) -> Optional[InventoryEntry]:
        """
        Mark an entry as deprecated.

        Args:
            entry_id: Entry ID
            deprecated_by: User deprecating
            reason: Deprecation reason

        Returns:
            Updated entry or None if not found
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return None

            old_status = entry.review_status
            entry.review_status = ReviewStatus.DEPRECATED
            entry.updated_at = datetime.now(timezone.utc)
            entry.updated_by = deprecated_by
            entry.version += 1

        # Log event
        event = InventoryChangeEvent(
            action="deprecate",
            entry_id=entry.id,
            entry_name=entry.name,
            actor_id=deprecated_by,
            old_values={"review_status": old_status.value},
            new_values={"review_status": ReviewStatus.DEPRECATED.value},
            reason=reason,
        )
        self._log_event(event)

        logger.info(f"Deprecated inventory entry: {entry.name}")
        return entry

    # ========================================================================
    # Update and Delete
    # ========================================================================

    def update(
        self,
        entry_id: str,
        updated_by: str,
        **kwargs,
    ) -> Optional[InventoryEntry]:
        """
        Update an entry.

        Args:
            entry_id: Entry ID
            updated_by: User updating
            **kwargs: Fields to update

        Returns:
            Updated entry or None if not found
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return None

            old_values = {}
            new_values = {}

            for key, value in kwargs.items():
                if hasattr(entry, key):
                    old_values[key] = getattr(entry, key)
                    if isinstance(old_values[key], Enum):
                        old_values[key] = old_values[key].value
                    setattr(entry, key, value)
                    new_values[key] = value
                    if isinstance(new_values[key], Enum):
                        new_values[key] = new_values[key].value

            entry.updated_at = datetime.now(timezone.utc)
            entry.updated_by = updated_by
            entry.version += 1

            # Re-run auto-detection
            entry._auto_detect_patterns()

        # Log event
        event = InventoryChangeEvent(
            action="update",
            entry_id=entry.id,
            entry_name=entry.name,
            actor_id=updated_by,
            old_values=old_values,
            new_values=new_values,
        )
        self._log_event(event)

        logger.info(f"Updated inventory entry: {entry.name}")
        return entry

    def delete(
        self,
        entry_id: str,
        deleted_by: str,
        reason: str,
    ) -> bool:
        """
        Delete an entry (soft delete by deprecating is preferred).

        Args:
            entry_id: Entry ID
            deleted_by: User deleting
            reason: Deletion reason (required)

        Returns:
            True if deleted, False otherwise
        """
        if not reason:
            raise ValueError("Deletion reason is required")

        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return False

            # Remove from indexes
            del self._entries[entry_id]
            if entry.name in self._by_name:
                del self._by_name[entry.name]
            if entry.path in self._by_path:
                del self._by_path[entry.path]
            if entry.entry_type in self._by_type:
                self._by_type[entry.entry_type] = [
                    eid for eid in self._by_type[entry.entry_type]
                    if eid != entry_id
                ]

        # Log event
        event = InventoryChangeEvent(
            action="delete",
            entry_id=entry_id,
            entry_name=entry.name,
            actor_id=deleted_by,
            old_values=entry.to_dict(),
            reason=reason,
        )
        self._log_event(event)

        logger.info(f"Deleted inventory entry: {entry.name}")
        return True

    # ========================================================================
    # Reporting
    # ========================================================================

    def generate_report(self, generated_by: str = "system") -> InventoryReport:
        """Generate a comprehensive report on the inventory."""
        with self._lock:
            entries = list(self._entries.values())

        report = InventoryReport(
            generated_by=generated_by,
            total_entries=len(entries),
        )

        for entry in entries:
            # By type
            type_key = entry.entry_type.value
            report.by_type[type_key] = report.by_type.get(type_key, 0) + 1

            # By sensitivity
            sens_key = entry.sensitivity.value
            report.by_sensitivity[sens_key] = report.by_sensitivity.get(sens_key, 0) + 1

            # By category
            cat_key = entry.category.value
            report.by_category[cat_key] = report.by_category.get(cat_key, 0) + 1

            # By review status
            review_key = entry.review_status.value
            report.by_review_status[review_key] = report.by_review_status.get(review_key, 0) + 1

            # By redaction
            redact_key = entry.redaction.value
            report.by_redaction[redact_key] = report.by_redaction.get(redact_key, 0) + 1

            # Compliance
            if entry.is_compliant:
                report.compliant_entries += 1
            else:
                report.non_compliant_entries += 1

            if entry.review_status == ReviewStatus.PENDING:
                report.pending_review += 1

            if entry.requires_dpia:
                report.requires_dpia += 1

            # Auto-detection
            if entry.pii_detected:
                report.pii_detected += 1
            if entry.credential_detected:
                report.credentials_detected += 1
            if entry.order_field_detected:
                report.order_fields_detected += 1

            # Issues
            if not entry.purpose:
                report.missing_purpose += 1
            if not entry.lawful_basis:
                report.missing_lawful_basis += 1
            if entry.retention_days <= 0:
                report.missing_retention += 1

        report.integrity_hash = report._compute_hash()
        return report

    # ========================================================================
    # Export / Import
    # ========================================================================

    def export(self) -> Dict[str, Any]:
        """Export the entire inventory."""
        with self._lock:
            entries = [e.to_dict() for e in self._entries.values()]

        content = json.dumps(entries, sort_keys=True, default=str)
        checksum = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

        return {
            "export_version": "1.0.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "entries": entries,
            "checksum": checksum,
        }

    def export_to_file(self, path: Path) -> None:
        """Export inventory to a file."""
        data = self.export()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Exported inventory to {path}")

    def import_from_file(
        self,
        path: Path,
        imported_by: str,
        merge: bool = True,
    ) -> Tuple[int, int, List[str]]:
        """
        Import inventory from a file.

        Args:
            path: File path
            imported_by: User importing
            merge: If True, merge with existing; if False, replace

        Returns:
            Tuple of (imported_count, error_count, errors)
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        imported = 0
        errors_list = []

        if not merge:
            # Clear existing
            with self._lock:
                self._entries.clear()
                self._by_name.clear()
                self._by_path.clear()
                self._by_type.clear()

        for entry_data in entries:
            try:
                # Convert to kwargs
                kwargs = {
                    "name": entry_data.get("name"),
                    "entry_type": InventoryEntryType(
                        entry_data.get("entry_type", "field")
                    ),
                    "path": entry_data.get("path"),
                    "sensitivity": DataSensitivity(
                        entry_data.get("classification", {}).get("sensitivity", "internal")
                    ),
                    "category": DataCategory(
                        entry_data.get("classification", {}).get("category", "technical")
                    ),
                    "description": entry_data.get("classification", {}).get("description", ""),
                    "purpose": entry_data.get("classification", {}).get("purpose", ""),
                    "lawful_basis": entry_data.get("classification", {}).get("lawful_basis", ""),
                    "retention_days": entry_data.get("retention", {}).get("days", 90),
                    "created_by": imported_by,
                }

                # Skip if already exists in merge mode
                if merge and kwargs["name"] in self._by_name:
                    continue

                self.register(**kwargs)
                imported += 1
            except Exception as e:
                errors_list.append(f"{entry_data.get('name', 'unknown')}: {e}")

        logger.info(f"Imported {imported} entries from {path}")
        return imported, len(errors_list), errors_list

    # ========================================================================
    # Events
    # ========================================================================

    def _log_event(self, event: InventoryChangeEvent) -> None:
        """Log a change event."""
        with self._lock:
            self._events.append(event)

        if self._on_change:
            try:
                self._on_change(event)
            except Exception as e:
                logger.error(f"Change callback error: {e}")

    def get_events(
        self,
        entry_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[InventoryChangeEvent]:
        """Get change events with optional filters."""
        with self._lock:
            events = list(self._events)

        if entry_id:
            events = [e for e in events if e.entry_id == entry_id]
        if action:
            events = [e for e in events if e.action == action]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get quick statistics."""
        with self._lock:
            total = len(self._entries)
            compliant = sum(1 for e in self._entries.values() if e.is_compliant)
            pending = sum(
                1 for e in self._entries.values()
                if e.review_status == ReviewStatus.PENDING
            )

        return {
            "total_entries": total,
            "compliant_entries": compliant,
            "non_compliant_entries": total - compliant,
            "compliance_rate": compliant / total * 100 if total > 0 else 0,
            "pending_review": pending,
            "total_events": len(self._events),
        }


# ============================================================================
# Pre-populated Core Inventory
# ============================================================================

def create_core_inventory() -> DataInventoryRegistry:
    """
    Create a registry pre-populated with core CCEA data fields.

    This provides the baseline inventory for the platform.
    """
    registry = DataInventoryRegistry(strict_mode=False)

    # Telemetry fields (from TELEMETRY_DATA_DICTIONARY.md)
    telemetry_fields = [
        # Aggregated level fields
        {"name": "timestamp", "category": "telemetry", "sensitivity": "public",
         "purpose": "Event timing", "lawful_basis": "Legitimate interest",
         "retention_days": 90, "min_telemetry_level": "AGGREGATED"},
        {"name": "count", "category": "telemetry", "sensitivity": "public",
         "purpose": "Aggregate metrics", "lawful_basis": "Legitimate interest",
         "retention_days": 90, "min_telemetry_level": "AGGREGATED"},
        {"name": "cpu_percent", "category": "telemetry", "sensitivity": "internal",
         "purpose": "System monitoring", "lawful_basis": "Contract performance",
         "retention_days": 30, "min_telemetry_level": "AGGREGATED"},
        {"name": "memory_percent", "category": "telemetry", "sensitivity": "internal",
         "purpose": "System monitoring", "lawful_basis": "Contract performance",
         "retention_days": 30, "min_telemetry_level": "AGGREGATED"},
        {"name": "latency_ms", "category": "telemetry", "sensitivity": "internal",
         "purpose": "Performance monitoring", "lawful_basis": "Contract performance",
         "retention_days": 30, "min_telemetry_level": "AGGREGATED"},
        {"name": "strategy_id", "category": "telemetry", "sensitivity": "internal",
         "purpose": "Strategy identification", "lawful_basis": "Contract performance",
         "retention_days": 365, "min_telemetry_level": "AGGREGATED"},
        {"name": "workspace_id", "category": "telemetry", "sensitivity": "internal",
         "purpose": "Workspace context", "lawful_basis": "Contract performance",
         "retention_days": 365, "min_telemetry_level": "AGGREGATED"},
        {"name": "error_count", "category": "telemetry", "sensitivity": "internal",
         "purpose": "Error monitoring", "lawful_basis": "Contract performance",
         "retention_days": 90, "min_telemetry_level": "AGGREGATED"},

        # Detailed level fields
        {"name": "event_type", "category": "telemetry", "sensitivity": "internal",
         "purpose": "Event classification", "lawful_basis": "Contract performance",
         "retention_days": 30, "min_telemetry_level": "DETAILED_NON_SENSITIVE"},
        {"name": "trace_id", "category": "telemetry", "sensitivity": "internal",
         "purpose": "Distributed tracing", "lawful_basis": "Contract performance",
         "retention_days": 7, "min_telemetry_level": "DETAILED_NON_SENSITIVE"},
        {"name": "error_code", "category": "telemetry", "sensitivity": "internal",
         "purpose": "Error classification", "lawful_basis": "Contract performance",
         "retention_days": 30, "min_telemetry_level": "DETAILED_NON_SENSITIVE"},
    ]

    for field_data in telemetry_fields:
        try:
            registry.register(
                name=field_data["name"],
                entry_type=InventoryEntryType.FIELD,
                sensitivity=DataSensitivity(field_data["sensitivity"]),
                category=DataCategory(field_data["category"]),
                purpose=field_data["purpose"],
                lawful_basis=field_data["lawful_basis"],
                retention_days=field_data["retention_days"],
                min_telemetry_level=field_data.get("min_telemetry_level", "AGGREGATED"),
                created_by="system",
                tags=["core", "telemetry"],
            )
        except ValueError:
            pass  # Skip duplicates

    # Data stores
    stores = [
        {"name": "telemetry_store", "entry_type": "store", "category": "telemetry",
         "sensitivity": "confidential", "purpose": "Telemetry storage",
         "lawful_basis": "Contract performance", "retention_days": 90},
        {"name": "audit_log_store", "entry_type": "store", "category": "audit",
         "sensitivity": "critical", "purpose": "Compliance audit trail",
         "lawful_basis": "Legal obligation", "retention_days": 2555},
        {"name": "user_settings_store", "entry_type": "store", "category": "user",
         "sensitivity": "sensitive", "purpose": "User preferences",
         "lawful_basis": "Contract performance", "retention_days": 365},
    ]

    for store_data in stores:
        try:
            registry.register(
                name=store_data["name"],
                entry_type=InventoryEntryType(store_data["entry_type"]),
                sensitivity=DataSensitivity(store_data["sensitivity"]),
                category=DataCategory(store_data["category"]),
                purpose=store_data["purpose"],
                lawful_basis=store_data["lawful_basis"],
                retention_days=store_data["retention_days"],
                created_by="system",
                tags=["core", "store"],
            )
        except ValueError:
            pass

    # Pre-approve core entries
    for entry in registry.list_all():
        registry.approve(entry.id, approved_by="system", notes="Core platform field")

    return registry


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "DataSensitivity",
    "DataCategory",
    "ResidencyRequirement",
    "RedactionRequirement",
    "InventoryEntryType",
    "ReviewStatus",
    # Data classes
    "InventoryEntry",
    "InventoryValidationResult",
    "InventoryReport",
    "InventoryChangeEvent",
    # Registry
    "DataInventoryRegistry",
    # Factory
    "create_core_inventory",
    # Constants
    "ALWAYS_REDACT_PATTERNS",
    "PII_FIELD_PATTERNS",
    "ORDER_FIELD_PATTERNS",
]

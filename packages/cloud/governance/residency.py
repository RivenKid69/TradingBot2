# -*- coding: utf-8 -*-
"""
Data Residency Manager.

CCEA Phase 8 Implementation.

This module manages data residency requirements:
    - EU region default for EU tenants
    - Enterprise "telemetry stays local" mode
    - Cross-region data transfer controls
    - Compliance tracking per region

Design Doc Reference:
    - Phase 8 (13.4): "Data residency: EU region default for EU tenants"
    - "Enterprise: telemetry stays local mode or selective export"

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Final, FrozenSet, List, Optional, Set
from uuid import uuid4


# ============================================================================
# Constants
# ============================================================================

class DataRegion(Enum):
    """Supported data regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    UK = "eu-west-2"
    AP_NORTHEAST = "ap-northeast-1"
    AP_SOUTHEAST = "ap-southeast-1"


# EU regions for GDPR compliance
EU_REGIONS: Final[FrozenSet[DataRegion]] = frozenset({
    DataRegion.EU_WEST,
    DataRegion.EU_CENTRAL,
    DataRegion.UK,
})

# Region country mappings
COUNTRY_TO_REGION: Final[Dict[str, DataRegion]] = {
    # EU countries -> EU regions
    "DE": DataRegion.EU_CENTRAL,
    "FR": DataRegion.EU_WEST,
    "IT": DataRegion.EU_WEST,
    "ES": DataRegion.EU_WEST,
    "NL": DataRegion.EU_WEST,
    "BE": DataRegion.EU_WEST,
    "AT": DataRegion.EU_CENTRAL,
    "PL": DataRegion.EU_CENTRAL,
    "SE": DataRegion.EU_WEST,
    "DK": DataRegion.EU_WEST,
    "FI": DataRegion.EU_WEST,
    "IE": DataRegion.EU_WEST,
    "PT": DataRegion.EU_WEST,
    "GR": DataRegion.EU_WEST,
    "CZ": DataRegion.EU_CENTRAL,
    "RO": DataRegion.EU_CENTRAL,
    "HU": DataRegion.EU_CENTRAL,
    "SK": DataRegion.EU_CENTRAL,
    "BG": DataRegion.EU_CENTRAL,
    "HR": DataRegion.EU_CENTRAL,
    "SI": DataRegion.EU_CENTRAL,
    "LT": DataRegion.EU_CENTRAL,
    "LV": DataRegion.EU_CENTRAL,
    "EE": DataRegion.EU_CENTRAL,
    "LU": DataRegion.EU_WEST,
    "MT": DataRegion.EU_WEST,
    "CY": DataRegion.EU_WEST,
    # UK (post-Brexit, but similar requirements)
    "GB": DataRegion.UK,
    # US
    "US": DataRegion.US_EAST,
    # Asia Pacific
    "JP": DataRegion.AP_NORTHEAST,
    "SG": DataRegion.AP_SOUTHEAST,
    "AU": DataRegion.AP_SOUTHEAST,
}

# Default region
DEFAULT_REGION: Final[DataRegion] = DataRegion.US_EAST


class ResidencyMode(Enum):
    """Data residency mode."""
    CLOUD = auto()           # Data stored in cloud (default)
    LOCAL_ONLY = auto()      # Data stays on agent (enterprise)
    SELECTIVE = auto()       # Selective export (enterprise)
    HYBRID = auto()          # Some local, some cloud


@dataclass
class ResidencyPolicy:
    """
    Data residency policy for a workspace.

    Attributes:
        id: Policy identifier
        workspace_id: Workspace this policy applies to
        primary_region: Primary data region
        allowed_regions: Regions data can be stored in
        mode: Residency mode
        telemetry_local: Keep telemetry local
        export_allowed_to: Regions export is allowed to
        retention_region_specific: Region-specific retention
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    primary_region: DataRegion = DataRegion.US_EAST
    allowed_regions: Set[DataRegion] = field(default_factory=lambda: {DataRegion.US_EAST})
    mode: ResidencyMode = ResidencyMode.CLOUD
    telemetry_local: bool = False
    export_allowed_to: Set[DataRegion] = field(default_factory=set)
    retention_region_specific: Dict[DataRegion, int] = field(default_factory=dict)

    # Compliance flags
    gdpr_compliant: bool = False
    requires_eu_only: bool = False
    requires_local_only: bool = False

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate and set compliance flags."""
        # EU regions are used as a signal for EU/GDPR residency expectations
        if self.primary_region in EU_REGIONS:
            self.gdpr_compliant = True

        # If requires EU only, enforce it
        if self.requires_eu_only:
            self.allowed_regions = self.allowed_regions.intersection(EU_REGIONS) or {DataRegion.EU_WEST}
            if self.primary_region not in EU_REGIONS:
                self.primary_region = DataRegion.EU_WEST

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "primary_region": self.primary_region.value,
            "allowed_regions": [r.value for r in self.allowed_regions],
            "mode": self.mode.name,
            "telemetry_local": self.telemetry_local,
            "export_allowed_to": [r.value for r in self.export_allowed_to],
            "retention_region_specific": {r.value: d for r, d in self.retention_region_specific.items()},
            "gdpr_compliant": self.gdpr_compliant,
            "requires_eu_only": self.requires_eu_only,
            "requires_local_only": self.requires_local_only,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ResidencyConfig:
    """Global residency configuration."""
    default_region: DataRegion = DataRegion.US_EAST
    eu_default_region: DataRegion = DataRegion.EU_WEST
    enforce_gdpr_for_eu: bool = True
    allow_cross_region_transfer: bool = True
    require_explicit_consent: bool = True


@dataclass
class DataLocationInfo:
    """Information about where data is stored."""
    data_type: str
    current_region: DataRegion
    is_compliant: bool
    compliance_issues: List[str] = field(default_factory=list)


class DataResidencyManager:
    """
    Manages data residency for tenants.

    Ensures data is stored in appropriate regions based on
    tenant location and compliance requirements.

    Usage:
        manager = DataResidencyManager()

        # Create policy for EU tenant
        policy = manager.create_policy_for_country("workspace-123", "DE")

        # Check if data transfer is allowed
        if manager.can_transfer_to(policy, DataRegion.US_EAST):
            transfer_data()

        # Get enterprise local-only policy
        policy = manager.create_enterprise_local_policy("workspace-456")

    COMPLIANCE:
        - EU data stays in EU by default
        - Cross-region transfers require explicit configuration
        - Audit trail for data location changes
    """

    def __init__(self, config: Optional[ResidencyConfig] = None):
        """
        Initialize residency manager.

        Args:
            config: Global configuration
        """
        self.config = config or ResidencyConfig()
        self._policies: Dict[str, ResidencyPolicy] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def create_policy_for_country(
        self,
        workspace_id: str,
        country_code: str,
    ) -> ResidencyPolicy:
        """
        Create residency policy based on country.

        Args:
            workspace_id: Workspace identifier
            country_code: ISO country code

        Returns:
            Created policy
        """
        # Determine region from country
        region = COUNTRY_TO_REGION.get(country_code.upper(), self.config.default_region)

        # Check if EU country
        is_eu = country_code.upper() in COUNTRY_TO_REGION and COUNTRY_TO_REGION[country_code.upper()] in EU_REGIONS

        policy = ResidencyPolicy(
            workspace_id=workspace_id,
            primary_region=region,
            allowed_regions={region},
            mode=ResidencyMode.CLOUD,
            gdpr_compliant=is_eu,
            requires_eu_only=is_eu and self.config.enforce_gdpr_for_eu,
        )

        # Set allowed regions based on requirements
        if is_eu and self.config.enforce_gdpr_for_eu:
            policy.allowed_regions = EU_REGIONS.copy()
        else:
            policy.allowed_regions = {region}
            if self.config.allow_cross_region_transfer:
                # Allow transfer to same-jurisdiction regions
                if region in EU_REGIONS:
                    policy.allowed_regions = EU_REGIONS.copy()

        with self._lock:
            self._policies[workspace_id] = policy
            self._log_audit("policy_created", workspace_id, policy)

        return policy

    def create_enterprise_local_policy(
        self,
        workspace_id: str,
        primary_region: DataRegion = DataRegion.US_EAST,
    ) -> ResidencyPolicy:
        """
        Create enterprise local-only policy.

        Telemetry stays on agent, not sent to cloud.

        Args:
            workspace_id: Workspace identifier
            primary_region: Primary region for metadata

        Returns:
            Created policy
        """
        policy = ResidencyPolicy(
            workspace_id=workspace_id,
            primary_region=primary_region,
            allowed_regions={primary_region},
            mode=ResidencyMode.LOCAL_ONLY,
            telemetry_local=True,
            requires_local_only=True,
        )

        with self._lock:
            self._policies[workspace_id] = policy
            self._log_audit("policy_created", workspace_id, policy, {"type": "enterprise_local"})

        return policy

    def create_selective_export_policy(
        self,
        workspace_id: str,
        primary_region: DataRegion,
        export_to: Set[DataRegion],
    ) -> ResidencyPolicy:
        """
        Create selective export policy.

        Data stays local but can be exported to specific regions.

        Args:
            workspace_id: Workspace identifier
            primary_region: Primary region
            export_to: Regions export is allowed to

        Returns:
            Created policy
        """
        policy = ResidencyPolicy(
            workspace_id=workspace_id,
            primary_region=primary_region,
            allowed_regions={primary_region},
            mode=ResidencyMode.SELECTIVE,
            telemetry_local=True,
            export_allowed_to=export_to,
        )

        with self._lock:
            self._policies[workspace_id] = policy
            self._log_audit("policy_created", workspace_id, policy, {"type": "selective_export"})

        return policy

    def get_policy(self, workspace_id: str) -> Optional[ResidencyPolicy]:
        """Get policy for workspace."""
        with self._lock:
            return self._policies.get(workspace_id)

    def get_or_create_policy(
        self,
        workspace_id: str,
        country_code: str = "US",
    ) -> ResidencyPolicy:
        """Get existing policy or create new one."""
        policy = self.get_policy(workspace_id)
        if not policy:
            policy = self.create_policy_for_country(workspace_id, country_code)
        return policy

    def update_policy(
        self,
        workspace_id: str,
        updates: Dict[str, Any],
    ) -> Optional[ResidencyPolicy]:
        """
        Update residency policy.

        Args:
            workspace_id: Workspace identifier
            updates: Fields to update

        Returns:
            Updated policy or None
        """
        with self._lock:
            policy = self._policies.get(workspace_id)
            if not policy:
                return None

            # Apply updates
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)

            policy.updated_at = datetime.utcnow()

            # Re-validate
            policy.__post_init__()

            self._log_audit("policy_updated", workspace_id, policy, {"updates": list(updates.keys())})

            return policy

    def can_store_in(
        self,
        workspace_id: str,
        region: DataRegion,
    ) -> tuple[bool, str]:
        """
        Check if data can be stored in a region.

        Args:
            workspace_id: Workspace identifier
            region: Target region

        Returns:
            Tuple of (allowed, reason)
        """
        policy = self.get_policy(workspace_id)
        if not policy:
            return True, "No policy defined"

        if policy.requires_local_only:
            return False, "Policy requires local-only storage"

        if region not in policy.allowed_regions:
            return False, f"Region {region.value} not in allowed regions"

        if policy.requires_eu_only and region not in EU_REGIONS:
            return False, "Policy requires EU-only storage"

        return True, "Allowed"

    def can_transfer_to(
        self,
        workspace_id: str,
        target_region: DataRegion,
    ) -> tuple[bool, str]:
        """
        Check if data can be transferred to a region.

        Args:
            workspace_id: Workspace identifier
            target_region: Target region

        Returns:
            Tuple of (allowed, reason)
        """
        policy = self.get_policy(workspace_id)
        if not policy:
            if not self.config.allow_cross_region_transfer:
                return False, "Cross-region transfer disabled"
            return True, "No policy defined"

        # Check if local-only
        if policy.requires_local_only and policy.mode != ResidencyMode.SELECTIVE:
            return False, "Policy requires local-only storage"

        # Check selective export
        if policy.mode == ResidencyMode.SELECTIVE:
            if target_region in policy.export_allowed_to:
                return True, "Allowed by selective export policy"
            return False, "Region not in export allowlist"

        # Check allowed regions
        if target_region not in policy.allowed_regions:
            return False, f"Transfer to {target_region.value} not allowed"

        # EU restriction
        if policy.requires_eu_only and target_region not in EU_REGIONS:
            return False, "EU data cannot be transferred outside EU"

        return True, "Allowed"

    def should_keep_telemetry_local(self, workspace_id: str) -> bool:
        """Check if telemetry should be kept local."""
        policy = self.get_policy(workspace_id)
        if not policy:
            return False
        return policy.telemetry_local

    def get_storage_region(self, workspace_id: str) -> DataRegion:
        """Get the storage region for a workspace."""
        policy = self.get_policy(workspace_id)
        if not policy:
            return self.config.default_region
        return policy.primary_region

    def validate_compliance(
        self,
        workspace_id: str,
        data_locations: List[DataLocationInfo],
    ) -> tuple[bool, List[str]]:
        """
        Validate data locations against policy.

        Args:
            workspace_id: Workspace identifier
            data_locations: Current data locations

        Returns:
            Tuple of (compliant, list of issues)
        """
        policy = self.get_policy(workspace_id)
        if not policy:
            return True, []

        issues = []

        for location in data_locations:
            allowed, reason = self.can_store_in(workspace_id, location.current_region)
            if not allowed:
                issues.append(f"{location.data_type}: {reason}")
                location.is_compliant = False
                location.compliance_issues.append(reason)
            else:
                location.is_compliant = True

        return len(issues) == 0, issues

    def get_recommended_region(self, country_code: str) -> DataRegion:
        """Get recommended region for a country."""
        return COUNTRY_TO_REGION.get(country_code.upper(), self.config.default_region)

    def is_eu_country(self, country_code: str) -> bool:
        """Check if country is in EU."""
        region = COUNTRY_TO_REGION.get(country_code.upper())
        return region in EU_REGIONS if region else False

    def _log_audit(
        self,
        action: str,
        workspace_id: str,
        policy: ResidencyPolicy,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log audit event."""
        entry = {
            "action": action,
            "workspace_id": workspace_id,
            "policy_id": policy.id,
            "primary_region": policy.primary_region.value,
            "mode": policy.mode.name,
            "timestamp": datetime.utcnow().isoformat(),
            **(extra or {}),
        }
        self._audit_log.append(entry)

    def get_audit_log(self, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit log."""
        with self._lock:
            if workspace_id:
                return [e for e in self._audit_log if e.get("workspace_id") == workspace_id]
            return list(self._audit_log)

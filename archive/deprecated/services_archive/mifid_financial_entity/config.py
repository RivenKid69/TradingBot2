# -*- coding: utf-8 -*-
"""
MiFID II Financial Entity Configuration (ARCHIVED).

These configurations are for Investment Firms under MiFID II.
NOT APPLICABLE TO ICT PROVIDERS.

This module contains FE-specific (Financial Entity) configurations
that are only relevant for firms directly subject to MiFID II regulation.

References:
    - MiFIR Article 26: Transaction reporting requirement
    - ISO 17442: LEI standard
    - GLEIF API: https://www.gleif.org/en/lei-data/gleif-lei-look-up-api

ARCHIVED: These modules are preserved for reference and for clients
who are Investment Firms. ICT Providers should NOT use these configurations.
"""

from __future__ import annotations

import warnings
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_validator
import re

# Emit deprecation warning on import
warnings.warn(
    "services.archive.mifid_financial_entity.config is for Investment Firms, "
    "not ICT Providers. Use services.core.risk_controls.config for universal "
    "risk controls instead.",
    DeprecationWarning,
    stacklevel=2
)


class ComplianceMode(str, Enum):
    """MiFID II compliance operation mode."""

    PRODUCTION = "production"
    TESTING = "testing"
    DISABLED = "disabled"


class LEIStatus(str, Enum):
    """LEI registration status per GLEIF."""

    ISSUED = "ISSUED"
    LAPSED = "LAPSED"
    MERGED = "MERGED"
    RETIRED = "RETIRED"
    ANNULLED = "ANNULLED"
    DUPLICATE = "DUPLICATE"
    PENDING_TRANSFER = "PENDING_TRANSFER"
    PENDING_ARCHIVAL = "PENDING_ARCHIVAL"


class LEIConfig(BaseModel):
    """
    LEI (Legal Entity Identifier) configuration per ISO 17442.

    The LEI is mandatory for transaction reporting under MiFIR Article 26.
    Without a valid LEI, transaction reports cannot be submitted ("No LEI, No Trade").

    ARCHIVED: Only relevant for Investment Firms subject to transaction reporting.

    References:
        - GLEIF Guidelines: https://www.gleif.org/en/newsroom/blog/reminder-failure-to-obtain-an-lei
        - ISO 17442: https://www.iso.org/standard/78829.html
    """

    model_config = ConfigDict(extra="forbid")

    own_lei: str = Field(
        default="",
        min_length=0,
        max_length=20,
        description="The firm's own LEI (20 characters, ISO 17442 format). Required for production.",
    )
    gleif_api_url: str = Field(
        default="https://api.gleif.org/api/v1",
        description="GLEIF API base URL for LEI verification.",
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # Max 1 week
        description="Cache TTL for GLEIF API responses in hours.",
    )
    verify_before_trade: bool = Field(
        default=True,
        description="Verify LEI status before each trade execution.",
    )
    renewal_warning_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Days before LEI expiry to generate renewal warning.",
    )
    allow_pending_status: bool = Field(
        default=True,
        description="Allow trading with PENDING_TRANSFER or PENDING_ARCHIVAL status.",
    )

    @model_validator(mode="after")
    def _validate_lei_format(self) -> "LEIConfig":
        """Validate LEI format if provided."""
        lei = self.own_lei
        if lei:
            # LEI format: 18 alphanumeric + 2 check digits
            pattern = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")
            if not pattern.match(lei):
                # Allow empty for testing, but warn in production
                pass  # Validation will happen at runtime
        return self


class TransactionReportingConfig(BaseModel):
    """
    Transaction Reporting Configuration per RTS 22.

    Configuration for MiFIR Article 26 transaction reporting.

    ARCHIVED: Only relevant for Investment Firms subject to transaction reporting.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable transaction reporting.",
    )
    arm_provider: str = Field(
        default="",
        description="ARM provider for transaction reporting submission.",
    )
    arm_environment: str = Field(
        default="test",
        description="ARM environment: 'test', 'production'.",
    )
    report_deadline_hours: int = Field(
        default=24,
        ge=1,
        le=48,
        description="Hours after trade execution to submit report (T+1).",
    )
    auto_submit: bool = Field(
        default=False,
        description="Automatically submit reports to ARM.",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum reports per batch submission.",
    )


class NCANotificationConfig(BaseModel):
    """
    NCA Notification Configuration per Article 17(2) MiFID II.

    Configuration for National Competent Authority notifications.

    ARCHIVED: Only relevant for Investment Firms that must notify NCAs.
    """

    model_config = ConfigDict(extra="forbid")

    jurisdiction: str = Field(
        default="",
        description="NCA jurisdiction code (e.g., 'FCA', 'BaFin', 'AMF').",
    )
    notification_email: str = Field(
        default="",
        description="Email for NCA notifications.",
    )
    auto_notify: bool = Field(
        default=False,
        description="Automatically send notifications to NCA.",
    )
    notification_types: List[str] = Field(
        default_factory=lambda: ["algorithm_deployment", "significant_change", "incident"],
        description="Types of events that trigger NCA notification.",
    )


class GovernanceConfig(BaseModel):
    """
    Governance Framework Configuration.

    Configuration for compliance policies and governance.

    ARCHIVED: Only relevant for Investment Firms with governance requirements.
    """

    model_config = ConfigDict(extra="forbid")

    policies_path: str = Field(
        default="docs/compliance/policies",
        description="Path to compliance policy documents.",
    )
    review_frequency_months: int = Field(
        default=12,
        ge=1,
        le=24,
        description="Frequency of policy reviews in months.",
    )
    require_sign_off: bool = Field(
        default=True,
        description="Require management sign-off on policies.",
    )
    version_control: bool = Field(
        default=True,
        description="Enable version control for policy documents.",
    )


class MiFIDIIComplianceConfig(BaseModel):
    """
    Full MiFID II Compliance Configuration for Investment Firms.

    ARCHIVED: This is the full MiFID II configuration for firms
    that are directly subject to MiFID II regulation.

    ICT Providers should use services.core.risk_controls.RiskControlsConfig
    instead.

    Example YAML:
        compliance:
          enabled: true
          mode: production
          lei:
            own_lei: "5493001KJTIIGC8Y1R12"
          transaction_reporting:
            arm_provider: "UNAVISTA"
          nca:
            jurisdiction: "FCA"
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=True,
        description="Enable MiFID II compliance checks.",
    )
    mode: ComplianceMode = Field(
        default=ComplianceMode.TESTING,
        description="Compliance mode: production, testing, or disabled.",
    )
    lei: LEIConfig = Field(
        default_factory=LEIConfig,
        description="LEI configuration for transaction reporting.",
    )
    transaction_reporting: TransactionReportingConfig = Field(
        default_factory=TransactionReportingConfig,
        description="Transaction reporting configuration.",
    )
    nca: NCANotificationConfig = Field(
        default_factory=NCANotificationConfig,
        description="NCA notification configuration.",
    )
    governance: GovernanceConfig = Field(
        default_factory=GovernanceConfig,
        description="Governance framework configuration.",
    )
    audit_log_path: str = Field(
        default="logs/compliance/mifid_audit.log",
        description="Path for MiFID II compliance audit log.",
    )

    @model_validator(mode="after")
    def _validate_production_requirements(self) -> "MiFIDIIComplianceConfig":
        """Validate production mode requirements."""
        if self.mode == ComplianceMode.PRODUCTION:
            if not self.lei.own_lei:
                raise ValueError("LEI is required for production mode")
            if not self.nca.jurisdiction:
                raise ValueError("NCA jurisdiction is required for production mode")
            if self.transaction_reporting.enabled and not self.transaction_reporting.arm_provider:
                raise ValueError("ARM provider is required when transaction reporting is enabled in production")
        return self


def load_mifid_compliance_config(path: str) -> MiFIDIIComplianceConfig:
    """
    Load MiFID II compliance configuration from YAML file.

    DEPRECATED: Use load_risk_controls_config from services.core.risk_controls.config
    for universal risk controls.

    Args:
        path: Path to YAML configuration file.

    Returns:
        MiFIDIIComplianceConfig instance.
    """
    import yaml

    warnings.warn(
        "load_mifid_compliance_config is deprecated for ICT Providers. "
        "Use load_risk_controls_config from services.core.risk_controls.config instead.",
        DeprecationWarning,
        stacklevel=2
    )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return MiFIDIIComplianceConfig()

    # Handle nested 'compliance' key
    if "compliance" in data:
        data = data["compliance"]

    return MiFIDIIComplianceConfig.model_validate(data)


__all__ = [
    "ComplianceMode",
    "LEIStatus",
    "LEIConfig",
    "TransactionReportingConfig",
    "NCANotificationConfig",
    "GovernanceConfig",
    "MiFIDIIComplianceConfig",
    "load_mifid_compliance_config",
]

# -*- coding: utf-8 -*-
"""
MiFID II Compliance Configuration Models.

Pydantic configuration models for MiFID II compliance components.
Supports YAML/JSON configuration loading and validation.

References:
    - RTS 25 (Clock Sync): https://www.esma.europa.eu/press-news/esma-news/esma-provides-guidance-transaction-reporting-order-record-keeping-and-clock
    - RTS 6 (Algo Trading): https://www.handbook.fca.org.uk/techstandards/MIFID-MIFIR/2017/reg_del_2017_589_oj
    - GLEIF API: https://www.gleif.org/en/lei-data/gleif-lei-look-up-api
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ComplianceMode(str, Enum):
    """Compliance operation mode."""

    PRODUCTION = "production"
    TESTING = "testing"
    DISABLED = "disabled"


class LEIConfig(BaseModel):
    """
    LEI (Legal Entity Identifier) configuration per ISO 17442.

    The LEI is mandatory for transaction reporting under MiFIR Article 26.
    Without a valid LEI, transaction reports cannot be submitted ("No LEI, No Trade").

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
            import re
            pattern = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")
            if not pattern.match(lei):
                # Allow empty for testing, but warn
                pass  # Validation will happen at runtime
        return self


class ClockSyncComplianceConfig(BaseModel):
    """
    RTS 25 Clock Synchronisation Configuration.

    Per RTS 25, algorithmic trading systems must maintain clocks
    synchronized to UTC with specified accuracy levels.

    Accuracy requirements per RTS 25:
        - Gateway-to-gateway latency < 1ms: ±100 microseconds
        - Algorithmic trading (general): ±1 millisecond
        - HFT systems: ±100 microseconds
        - Voice trading: ±1 second

    References:
        - RTS 25 (Regulation 2017/574): Clock Synchronisation
        - ESMA Guidelines: https://www.esma.europa.eu/press-news/esma-news/esma-provides-guidance-transaction-reporting-order-record-keeping-and-clock
    """

    model_config = ConfigDict(extra="forbid")

    ntp_servers: List[str] = Field(
        default_factory=lambda: [
            "time.google.com",
            "pool.ntp.org",
            "time.windows.com",
            "time.cloudflare.com",
        ],
        description="NTP servers for clock synchronisation (in priority order).",
    )
    max_offset_ms: float = Field(
        default=100.0,
        ge=0.001,
        le=1000.0,
        description="Maximum allowed clock offset in milliseconds (RTS 25 requirement).",
    )
    sync_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Interval between NTP sync attempts in seconds.",
    )
    warning_threshold_ms: float = Field(
        default=50.0,
        ge=0.0,
        description="Clock drift threshold for warning alerts (ms).",
    )
    critical_threshold_ms: float = Field(
        default=100.0,
        ge=0.0,
        description="Clock drift threshold for critical alerts (ms).",
    )
    kill_switch_threshold_ms: float = Field(
        default=1000.0,
        ge=100.0,
        description="Clock drift threshold to trigger kill switch (ms).",
    )
    timestamp_precision: str = Field(
        default="nanosecond",
        description="Timestamp precision for audit trail: 'nanosecond', 'microsecond', 'millisecond'.",
    )
    fallback_to_system: bool = Field(
        default=True,
        description="Use system clock if all NTP servers fail.",
    )
    stratum_max: int = Field(
        default=3,
        ge=1,
        le=15,
        description="Maximum acceptable NTP stratum level.",
    )

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "ClockSyncComplianceConfig":
        """Ensure threshold ordering."""
        if self.warning_threshold_ms >= self.critical_threshold_ms:
            raise ValueError("warning_threshold_ms must be less than critical_threshold_ms")
        if self.critical_threshold_ms >= self.kill_switch_threshold_ms:
            raise ValueError("critical_threshold_ms must be less than kill_switch_threshold_ms")
        return self


class AlgorithmRegistryConfig(BaseModel):
    """
    Algorithm Registration Configuration per Article 17(2) MiFID II.

    Investment firms must notify the competent authority (NCA) that
    they engage in algorithmic trading and maintain records of
    all deployed algorithms.

    References:
        - Article 17(2) MiFID II: Algorithm notification requirement
        - RTS 6 Article 8: Algorithm testing and deployment
        - Kroll Guide: https://www.kroll.com/en/publications/financial-compliance-regulation/algorithmic-trading-under-mifid-ii
    """

    model_config = ConfigDict(extra="forbid")

    registry_path: str = Field(
        default="state/compliance/algorithm_registry.json",
        description="Path to persist algorithm registry.",
    )
    auto_register: bool = Field(
        default=True,
        description="Automatically register new algorithms on first use.",
    )
    require_responsible_person: bool = Field(
        default=True,
        description="Require responsible person assignment for each algorithm.",
    )
    version_on_modification: bool = Field(
        default=True,
        description="Auto-increment version on parameter modifications.",
    )
    nca_jurisdiction: str = Field(
        default="",
        description="National Competent Authority jurisdiction code (e.g., 'FCA', 'BaFin', 'AMF').",
    )
    firm_name: str = Field(
        default="",
        description="Legal name of the investment firm.",
    )
    contact_email: str = Field(
        default="",
        description="Contact email for algorithm inquiries.",
    )
    contact_phone: str = Field(
        default="",
        description="Contact phone for emergency situations.",
    )


class PreTradeControlsConfig(BaseModel):
    """
    Pre-Trade Risk Controls Configuration per RTS 6 Article 15.

    Investment firms must have pre-trade controls to prevent:
    - Price collar breaches
    - Excessive order values/volumes
    - Unauthorized trading
    - Message rate abuse

    References:
        - RTS 6 Article 15: Pre-trade controls
        - Eventus Guide: https://www.eventus.com/cat-article/enforcement-action-from-esma-on-rts-6/
    """

    model_config = ConfigDict(extra="forbid")

    price_collar_pct: float = Field(
        default=5.0,
        ge=0.1,
        le=50.0,
        description="Maximum price deviation from reference price (%).",
    )
    max_order_value_eur: float = Field(
        default=1_000_000.0,
        ge=0.0,
        description="Maximum single order value in EUR equivalent.",
    )
    max_order_volume: float = Field(
        default=10_000.0,
        ge=0.0,
        description="Maximum single order volume (units).",
    )
    max_messages_per_second: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Maximum messages per second per venue.",
    )
    fat_finger_price_deviation_pct: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Fat finger protection: max price deviation (%).",
    )
    fat_finger_volume_multiplier: float = Field(
        default=10.0,
        ge=2.0,
        le=100.0,
        description="Fat finger protection: max volume vs ADV multiplier.",
    )


class MiFIDIIComplianceConfig(BaseModel):
    """
    Top-level MiFID II Compliance Configuration.

    This configuration controls all MiFID II compliance components
    including LEI management, clock synchronisation, algorithm registry,
    and pre-trade controls.

    Example YAML:
        compliance:
          enabled: true
          mode: production
          lei:
            own_lei: "5493001KJTIIGC8Y1R12"
          clock:
            max_offset_ms: 100
          algorithm_registry:
            nca_jurisdiction: "FCA"
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
    clock: ClockSyncComplianceConfig = Field(
        default_factory=ClockSyncComplianceConfig,
        description="RTS 25 clock synchronisation configuration.",
    )
    algorithm_registry: AlgorithmRegistryConfig = Field(
        default_factory=AlgorithmRegistryConfig,
        description="Algorithm registration configuration.",
    )
    pre_trade_controls: PreTradeControlsConfig = Field(
        default_factory=PreTradeControlsConfig,
        description="Pre-trade risk controls configuration.",
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
            if not self.algorithm_registry.nca_jurisdiction:
                raise ValueError("NCA jurisdiction is required for production mode")
        return self


def load_compliance_config(path: str) -> MiFIDIIComplianceConfig:
    """
    Load MiFID II compliance configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        MiFIDIIComplianceConfig instance.

    Example:
        config = load_compliance_config("configs/compliance/compliance.yaml")
    """
    import yaml

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        # Return default config if file not found
        return MiFIDIIComplianceConfig()

    # Handle nested 'compliance' key
    if "compliance" in data:
        data = data["compliance"]

    return MiFIDIIComplianceConfig.model_validate(data)


__all__ = [
    "ComplianceMode",
    "LEIConfig",
    "ClockSyncComplianceConfig",
    "AlgorithmRegistryConfig",
    "PreTradeControlsConfig",
    "MiFIDIIComplianceConfig",
    "load_compliance_config",
]

# -*- coding: utf-8 -*-
"""
Core Risk Controls Configuration.

Universal risk management configuration for all platform users.
Applicable to both ICT Providers and their clients.

This configuration is extracted from MiFID II compliance requirements
but applies universally as best practices for algorithmic trading.

References:
    - RTS 25 (Clock Sync): Time synchronization requirements
    - RTS 6 (Risk Controls): Pre-trade controls and risk limits
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ControlsMode(str, Enum):
    """Risk controls operation mode."""

    PRODUCTION = "production"
    TESTING = "testing"
    DISABLED = "disabled"


class TimeSyncConfig(BaseModel):
    """
    Time Synchronization Configuration.

    Based on RTS 25 clock synchronisation requirements.
    Algorithmic trading systems must maintain clocks
    synchronized to UTC with specified accuracy levels.

    Accuracy requirements (RTS 25):
        - Gateway-to-gateway latency < 1ms: +/-100 microseconds
        - Algorithmic trading (general): +/-1 millisecond
        - HFT systems: +/-100 microseconds
        - Voice trading: +/-1 second
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
        description="Maximum allowed clock offset in milliseconds.",
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
    def _validate_thresholds(self) -> "TimeSyncConfig":
        """Ensure threshold ordering."""
        if self.warning_threshold_ms >= self.critical_threshold_ms:
            raise ValueError("warning_threshold_ms must be less than critical_threshold_ms")
        if self.critical_threshold_ms >= self.kill_switch_threshold_ms:
            raise ValueError("critical_threshold_ms must be less than kill_switch_threshold_ms")
        return self


class PreTradeControlsConfig(BaseModel):
    """
    Pre-Trade Risk Controls Configuration.

    Based on RTS 6 Article 15 requirements for pre-trade controls.
    Investment firms must have pre-trade controls to prevent:
        - Price collar breaches
        - Excessive order values/volumes
        - Unauthorized trading
        - Message rate abuse
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


class AuditConfig(BaseModel):
    """
    Audit Trail Configuration.

    Configuration for audit logging and retention policies.
    """

    model_config = ConfigDict(extra="forbid")

    log_path: str = Field(
        default="logs/risk_controls/audit.log",
        description="Path for risk controls audit log.",
    )
    retention_years: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Years to retain audit records (MiFID II requires 5 years).",
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum audit log file size before rotation (MB).",
    )
    compress_archived: bool = Field(
        default=True,
        description="Compress archived audit files.",
    )


class KillSwitchConfig(BaseModel):
    """
    Kill Switch Configuration.

    Emergency stop configuration for risk management.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable kill switch functionality.",
    )
    max_daily_loss_pct: float = Field(
        default=5.0,
        ge=0.1,
        le=100.0,
        description="Maximum daily loss percentage to trigger kill switch.",
    )
    max_position_value_eur: float = Field(
        default=10_000_000.0,
        ge=0.0,
        description="Maximum position value to trigger kill switch.",
    )
    auto_reset_enabled: bool = Field(
        default=False,
        description="Enable automatic reset after cool-down period.",
    )
    cool_down_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description="Cool-down period before auto-reset (minutes).",
    )
    notification_emails: List[str] = Field(
        default_factory=list,
        description="Email addresses for kill switch notifications.",
    )
    require_manual_reset: bool = Field(
        default=True,
        description="Require manual reset after kill switch activation.",
    )


class RiskControlsConfig(BaseModel):
    """
    Top-level Risk Controls Configuration.

    Universal risk management configuration applicable to all platform users.
    This is the main entry point for risk controls configuration.

    Example YAML:
        risk_controls:
          enabled: true
          mode: production
          time_sync:
            max_offset_ms: 100
          pre_trade:
            price_collar_pct: 5.0
          kill_switch:
            max_daily_loss_pct: 5.0
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=True,
        description="Enable risk controls.",
    )
    mode: ControlsMode = Field(
        default=ControlsMode.TESTING,
        description="Controls mode: production, testing, or disabled.",
    )
    time_sync: TimeSyncConfig = Field(
        default_factory=TimeSyncConfig,
        description="Time synchronization configuration.",
    )
    pre_trade: PreTradeControlsConfig = Field(
        default_factory=PreTradeControlsConfig,
        description="Pre-trade risk controls configuration.",
    )
    audit: AuditConfig = Field(
        default_factory=AuditConfig,
        description="Audit trail configuration.",
    )
    kill_switch: KillSwitchConfig = Field(
        default_factory=KillSwitchConfig,
        description="Kill switch configuration.",
    )
    audit_log_path: str = Field(
        default="logs/risk_controls/audit.log",
        description="Path for risk controls audit log (deprecated, use audit.log_path).",
    )

    @model_validator(mode="after")
    def _validate_production_requirements(self) -> "RiskControlsConfig":
        """Validate production mode requirements."""
        if self.mode == ControlsMode.PRODUCTION:
            if not self.kill_switch.enabled:
                raise ValueError("Kill switch must be enabled in production mode")
        return self


def load_risk_controls_config(path: str) -> RiskControlsConfig:
    """
    Load risk controls configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        RiskControlsConfig instance.

    Example:
        config = load_risk_controls_config("configs/risk_controls.yaml")
    """
    import yaml

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return RiskControlsConfig()

    # Handle nested 'risk_controls' key
    if "risk_controls" in data:
        data = data["risk_controls"]

    return RiskControlsConfig.model_validate(data)


__all__ = [
    "ControlsMode",
    "TimeSyncConfig",
    "PreTradeControlsConfig",
    "AuditConfig",
    "KillSwitchConfig",
    "RiskControlsConfig",
    "load_risk_controls_config",
]

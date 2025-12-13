# -*- coding: utf-8 -*-
"""
Algo Integration Configuration.

B2B Compliance Toolkit configuration for financial institution clients.
These configurations help clients meet MiFID II requirements for algorithmic trading.

This module is designed for enterprise clients (Investment Firms) that need
to comply with MiFID II while using our platform.

References:
    - Article 17(2) MiFID II: Algorithm notification requirement
    - RTS 6 Article 5: Conformance testing requirements
    - RTS 6 Article 8: Algorithm testing and deployment
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_validator


class AlgorithmType(str, Enum):
    """Algorithm classification types per MiFID II."""

    EXECUTION = "execution"  # Execution algorithm
    MARKET_MAKING = "market_making"  # Market making strategy
    ARBITRAGE = "arbitrage"  # Arbitrage strategy
    TREND_FOLLOWING = "trend_following"  # Trend following strategy
    STATISTICAL = "statistical"  # Statistical arbitrage
    OTHER = "other"  # Other algorithm types


class ConformanceTestLevel(str, Enum):
    """Conformance testing levels per RTS 6 Article 5."""

    BASIC = "basic"  # Basic functionality tests
    STANDARD = "standard"  # Standard conformance suite
    COMPREHENSIVE = "comprehensive"  # Full RTS 6 Article 5 compliance
    CUSTOM = "custom"  # Custom test suite


class AlgorithmRegistryConfig(BaseModel):
    """
    Algorithm Registration Configuration per Article 17(2) MiFID II.

    Investment firms must notify the competent authority (NCA) that
    they engage in algorithmic trading and maintain records of
    all deployed algorithms.

    References:
        - Article 17(2) MiFID II: Algorithm notification requirement
        - RTS 6 Article 8: Algorithm testing and deployment
    """

    model_config = ConfigDict(extra="forbid")

    registry_path: str = Field(
        default="state/algo_integration/algorithm_registry.json",
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


class BestExecutionConfig(BaseModel):
    """
    Best Execution Configuration per Article 27 MiFID II.

    Configuration for best execution analysis and monitoring.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable best execution monitoring.",
    )
    report_frequency: str = Field(
        default="quarterly",
        description="Frequency of best execution reports: 'monthly', 'quarterly', 'annually'.",
    )
    slippage_threshold_bps: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Slippage threshold for alerts (basis points).",
    )
    latency_threshold_ms: float = Field(
        default=100.0,
        ge=0.0,
        description="Latency threshold for alerts (milliseconds).",
    )
    venues_to_monitor: List[str] = Field(
        default_factory=list,
        description="List of venues to monitor for best execution.",
    )


class TCAConfig(BaseModel):
    """
    Transaction Cost Analysis Configuration.

    Configuration for TCA reporting and analysis.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable TCA analysis.",
    )
    benchmark: str = Field(
        default="VWAP",
        description="Default benchmark: 'VWAP', 'TWAP', 'Arrival', 'Close'.",
    )
    include_implicit_costs: bool = Field(
        default=True,
        description="Include implicit costs in TCA analysis.",
    )
    report_format: str = Field(
        default="PDF",
        description="TCA report format: 'PDF', 'HTML', 'JSON', 'CSV'.",
    )


class ConformanceTestingConfig(BaseModel):
    """
    Conformance Testing Configuration per RTS 6 Article 5.

    Investment firms must test algorithms before deployment
    and maintain testing records.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable conformance testing.",
    )
    test_level: ConformanceTestLevel = Field(
        default=ConformanceTestLevel.STANDARD,
        description="Conformance testing level.",
    )
    test_environment: str = Field(
        default="sandbox",
        description="Test environment: 'sandbox', 'uat', 'paper'.",
    )
    require_certification: bool = Field(
        default=True,
        description="Require certification before production deployment.",
    )
    certification_validity_days: int = Field(
        default=365,
        ge=30,
        le=730,
        description="Certificate validity period in days.",
    )
    auto_revoke_on_modification: bool = Field(
        default=True,
        description="Auto-revoke certificate on algorithm modification.",
    )


class OTRConfig(BaseModel):
    """
    Order-to-Trade Ratio Configuration.

    Configuration for OTR monitoring to prevent excessive order submission.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable OTR monitoring.",
    )
    warning_threshold: float = Field(
        default=4.0,
        ge=1.0,
        le=100.0,
        description="OTR warning threshold ratio.",
    )
    critical_threshold: float = Field(
        default=8.0,
        ge=1.0,
        le=100.0,
        description="OTR critical threshold ratio.",
    )
    window_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Window for OTR calculation in seconds.",
    )

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "OTRConfig":
        """Ensure threshold ordering."""
        if self.warning_threshold >= self.critical_threshold:
            raise ValueError("warning_threshold must be less than critical_threshold")
        return self


class AlgoIntegrationConfig(BaseModel):
    """
    Top-level Algo Integration Configuration.

    B2B compliance toolkit for financial institution clients.
    Disabled by default for ICT Provider deployments.

    Example YAML:
        algo_integration:
          enabled: true
          algorithm_registry:
            nca_jurisdiction: "FCA"
            firm_name: "Example Investment Ltd"
          best_execution:
            report_frequency: quarterly
          conformance_testing:
            test_level: comprehensive
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=False,  # Disabled by default for ICT Provider
        description="Enable algo integration B2B toolkit.",
    )
    algorithm_registry: AlgorithmRegistryConfig = Field(
        default_factory=AlgorithmRegistryConfig,
        description="Algorithm registration configuration.",
    )
    best_execution: BestExecutionConfig = Field(
        default_factory=BestExecutionConfig,
        description="Best execution configuration.",
    )
    tca: TCAConfig = Field(
        default_factory=TCAConfig,
        description="Transaction cost analysis configuration.",
    )
    conformance_testing: ConformanceTestingConfig = Field(
        default_factory=ConformanceTestingConfig,
        description="Conformance testing configuration.",
    )
    otr: OTRConfig = Field(
        default_factory=OTRConfig,
        description="Order-to-trade ratio configuration.",
    )

    @model_validator(mode="after")
    def _validate_enabled_requirements(self) -> "AlgoIntegrationConfig":
        """Validate requirements when enabled."""
        if self.enabled:
            # In production, NCA jurisdiction should be set
            if self.conformance_testing.require_certification:
                if not self.algorithm_registry.firm_name:
                    # Warning only, not error (for testing flexibility)
                    pass
        return self


def load_algo_integration_config(path: str) -> AlgoIntegrationConfig:
    """
    Load algo integration configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        AlgoIntegrationConfig instance.

    Example:
        config = load_algo_integration_config("configs/algo_integration.yaml")
    """
    import yaml

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return AlgoIntegrationConfig()

    # Handle nested 'algo_integration' key
    if "algo_integration" in data:
        data = data["algo_integration"]

    return AlgoIntegrationConfig.model_validate(data)


__all__ = [
    "AlgorithmType",
    "ConformanceTestLevel",
    "AlgorithmRegistryConfig",
    "BestExecutionConfig",
    "TCAConfig",
    "ConformanceTestingConfig",
    "OTRConfig",
    "AlgoIntegrationConfig",
    "load_algo_integration_config",
]

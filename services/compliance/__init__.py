# -*- coding: utf-8 -*-
"""
MiFID II Compliance Module for AI-Powered Quantitative Research Platform.

MiFID II (Directive 2014/65/EU) Compliance Implementation.

This package provides compliance tools for algorithmic trading per MiFID II/MiFIR requirements:

Phase 1 - Foundation (LEI, Clock Sync, Algorithm Registry):
    - lei_manager: LEI Management and Validation (ISO 17442)
    - gleif_client: GLEIF API Integration for LEI verification
    - compliance_clock: RTS 25 Compliant Clock Synchronisation
    - algorithm_registry: Algorithm Registration per Article 17(2)

Phase 2 - Transaction Reporting (RTS 22):
    - transaction_report: Transaction Report Data Model
    - arm_client: ARM (Approved Reporting Mechanism) Integration
    - reporting_pipeline: End-to-end Reporting Pipeline

Phase 3 - Algorithmic Trading Controls (RTS 6):
    - enhanced_kill_switch: RTS 6 Article 12 Kill Switch
    - pre_trade_controls: RTS 6 Article 15 Pre-Trade Controls
    - realtime_monitor: RTS 6 Article 17 Real-Time Monitoring
    - otr_monitor: Order-to-Trade Ratio Monitoring

Classification:
    This system is classified as an ALGORITHMIC TRADING SYSTEM per
    Article 17 MiFID II and subject to RTS 6 requirements.

Applicable Regulations:
    - MiFID II (Directive 2014/65/EU) Article 17
    - MiFIR (Regulation 600/2014) Articles 25-26
    - RTS 6 (Regulation 2017/589) - Algorithmic Trading
    - RTS 22 (Regulation 2017/590) - Transaction Reporting
    - RTS 25 (Regulation 2017/574) - Clock Synchronisation

References:
    - ESMA MiFID II Rulebook: https://www.esma.europa.eu/publications-and-data/interactive-single-rulebook/mifid-ii
    - GLEIF LEI Lookup: https://www.gleif.org/en/lei-data/gleif-lei-look-up-api
    - Article 17 MiFID II: https://www.esma.europa.eu/publications-and-data/interactive-single-rulebook/mifid-ii/article-17-algorithmic-trading
"""

from __future__ import annotations

__version__ = "1.0.0"
__mifid_ii_compliance_phase__ = 1  # Current implementation phase

# Phase 1 exports (Foundation)
from services.compliance.lei_manager import (
    LEIRecord,
    LEIStatus,
    LEIValidationResult,
    LEIManager,
    create_lei_manager,
)

from services.compliance.gleif_client import (
    GLEIFClient,
    GLEIFResponse,
    GLEIFError,
    create_gleif_client,
)

from services.compliance.compliance_clock import (
    ClockSyncStatus,
    ClockDriftSeverity,
    ComplianceClock,
    create_compliance_clock,
)

from services.compliance.algorithm_registry import (
    AlgorithmType,
    AlgorithmStatus,
    AlgorithmRecord,
    AlgorithmRegistry,
    create_algorithm_registry,
    get_default_algorithm_types,
)

from services.compliance.config import (
    MiFIDIIComplianceConfig,
    LEIConfig,
    ClockSyncComplianceConfig,
    AlgorithmRegistryConfig,
)

__all__ = [
    # Version info
    "__version__",
    "__mifid_ii_compliance_phase__",
    # LEI Management
    "LEIRecord",
    "LEIStatus",
    "LEIValidationResult",
    "LEIManager",
    "create_lei_manager",
    # GLEIF Client
    "GLEIFClient",
    "GLEIFResponse",
    "GLEIFError",
    "create_gleif_client",
    # Compliance Clock (RTS 25)
    "ClockSyncStatus",
    "ClockDriftSeverity",
    "ComplianceClock",
    "create_compliance_clock",
    # Algorithm Registry (Article 17)
    "AlgorithmType",
    "AlgorithmStatus",
    "AlgorithmRecord",
    "AlgorithmRegistry",
    "create_algorithm_registry",
    "get_default_algorithm_types",
    # Configuration
    "MiFIDIIComplianceConfig",
    "LEIConfig",
    "ClockSyncComplianceConfig",
    "AlgorithmRegistryConfig",
]

# -*- coding: utf-8 -*-
"""
DORA Compliance Module for AI-Powered Quantitative Research Platform.

Digital Operational Resilience Act (DORA) - Regulation (EU) 2022/2554.

This package provides compliance tools for financial entities per DORA requirements:

Phase 0 - Proportionality Assessment (Articles 2, 3(22), 4, 16):
    - scope_verification: DORA Scope Verification (Article 2)
    - function_classification: Critical/Important Function Classification (Article 3(22))
    - proportionality: Entity Classification and Regime Determination (Articles 4, 16)

Phase 1 - ICT Risk Management Framework (Articles 5-16):
    - governance: Management Body Oversight and ICT Governance (Article 5)
    - ict_risk_framework: ICT Risk Management Framework (Article 6)
    - ict_systems: ICT Systems, Protocols and Tools (Article 7)
    - identification: ICT Risk Identification (Article 8)
    - protection: Protection and Prevention (Article 9)
    - detection: Anomaly Detection (Article 10)
    - response_recovery: Response and Recovery (Article 11)
    - backup: Backup and Recovery (Article 12)
    - learning: Learning and Evolving (Article 13)
    - communication: Crisis Communication (Article 14)

Phase 2 - ICT Incident Management & Reporting (Articles 17-23):
    - incident_classification: Incident Classification (CDR 2024/1772)
    - incident_management: ICT Incident Management (Article 17)
    - incident_reporting: Major Incident Reporting (Article 19)
    - cyber_threat_notification: Cyber Threat Notification (Article 19a)

Phase 3 - Digital Resilience Testing (Articles 24-27):
    - vulnerability_assessment: Vulnerability Assessments (Article 24)
    - penetration_testing: Penetration Testing (Article 24)
    - tlpt: Threat-Led Penetration Testing (Article 26) - if designated

Phase 4 - Third-Party ICT Risk Management (Articles 28-30):
    - register_of_information: Register of Information (Article 28)
    - third_party_risk: Third-Party Risk Assessment (Article 28)
    - contract_requirements: Contractual Arrangements (Article 30)
    - its_export: ITS Export (CIR 2024/2956)

Phase 5 - Information Sharing & Integration (Article 45):
    - threat_intelligence: Threat Intelligence Sharing
    - cross_regulation: Cross-Regulation Integration (AI Act, MiFID II)

Scope of Application:
    Financial entities subject to DORA per Article 2(1):
    - Investment firms (Article 2(1)(e))
    - Crypto-asset service providers (Article 2(1)(f))
    - Other 19 types of financial entities

Key Compliance Dates:
    - Application Date: 17 January 2025
    - Register of Information: 30 April 2025 (via NCAs to ESAs)
    - Reference Date for ROI: 31 March 2025

Technical Standards:
    - RTS on ICT Risk Management Framework (CDR 2024/1774)
    - RTS on Incident Classification (CDR 2024/1772)
    - RTS on Incident Reporting (CDR 2025/301)
    - ITS on Register of Information (CIR 2024/2956)

References:
    - DORA Full Text: https://eur-lex.europa.eu/eli/reg/2022/2554/oj
    - DORA Article 2: https://www.digital-operational-resilience-act.com/Article_2.html
    - ESAs Technical Standards: https://www.esma.europa.eu/publications-and-data/dora
    - CTPP Designations: https://www.esma.europa.eu/press-news/esma-news/european-supervisory-authorities-designate-critical-ict-third-party-providers
"""

from __future__ import annotations

__version__ = "0.1.0"
__dora_compliance_phase__ = 0  # Current implementation phase

# =============================================================================
# Phase 0 exports (Proportionality Assessment)
# =============================================================================

from services.dora.scope_verification import (
    # Enums
    DORAEntityType,
    DORAScopeResult,
    # Data structures
    ScopeVerification,
    EntityAuthorization,
    # Main class
    DORAScope,
    # Factory functions
    create_scope_verifier,
    get_entity_type_description,
)

from services.dora.function_classification import (
    # Enums
    FunctionCriticality,
    ImpairmentType,
    # Data structures
    FunctionClassification,
    ICTService,
    ThirdPartyProvider,
    # Main class
    FunctionClassifier,
    # Factory functions
    create_function_classifier,
    get_platform_functions,
    get_ict_providers,
)

from services.dora.proportionality import (
    # Enums
    DORARegime,
    ExemptionType,
    # Data structures
    EntityClassification,
    ProportionalityAssessment,
    RegimeExemption,
    # Main class
    ProportionalityAssessor,
    # Factory functions
    create_proportionality_assessor,
    assess_entity_proportionality,
)

# =============================================================================
# __all__ exports
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__dora_compliance_phase__",

    # =========================================================================
    # Phase 0: Proportionality Assessment
    # =========================================================================

    # Scope Verification (Article 2)
    "DORAEntityType",
    "DORAScopeResult",
    "ScopeVerification",
    "EntityAuthorization",
    "DORAScope",
    "create_scope_verifier",
    "get_entity_type_description",

    # Function Classification (Article 3(22))
    "FunctionCriticality",
    "ImpairmentType",
    "FunctionClassification",
    "ICTService",
    "ThirdPartyProvider",
    "FunctionClassifier",
    "create_function_classifier",
    "get_platform_functions",
    "get_ict_providers",

    # Proportionality (Articles 4, 16)
    "DORARegime",
    "ExemptionType",
    "EntityClassification",
    "ProportionalityAssessment",
    "RegimeExemption",
    "ProportionalityAssessor",
    "create_proportionality_assessor",
    "assess_entity_proportionality",
]

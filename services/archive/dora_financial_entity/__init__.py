# -*- coding: utf-8 -*-
"""
Archived DORA Financial Entity Modules.

These modules implement DORA requirements for FINANCIAL ENTITIES (Art. 2),
NOT for ICT Third-Party Service Providers (Art. 30).

IMPORTANT:
    As an ICT service provider, we:
    - Comply with Art. 30 (contractual requirements)
    - Support client due diligence (Art. 28)
    - DO NOT implement full FE DORA framework ourselves

    These modules are archived for reference and potential product development
    for financial entity customers.

Active DORA code lives in:
    - services/core/           - Operational resilience
    - services/dora_integration/ - Client-facing interfaces

When to Use These Modules:
    If building a product FOR financial entities to manage their own
    DORA compliance, these modules provide a reference implementation.

Archived Modules (23 total):
    - scope_verification.py: Art. 2 - DORA scope determination
    - function_classification.py: Art. 3(22) - Critical function classification
    - proportionality.py: Art. 4, 16 - Proportionality regime
    - governance.py: Art. 5 - ICT governance framework
    - ict_risk_framework.py: Art. 6 - ICT risk management
    - ict_systems.py: Art. 7 - ICT systems management
    - ict_identification.py: Art. 8 - ICT asset identification
    - protection.py: Art. 9 - Protection controls
    - detection.py: Art. 10 - Anomaly detection
    - response_recovery.py: Art. 11 - Incident response
    - backup_recovery.py: Art. 12 - Backup policies
    - learning.py: Art. 13 - Learning & evolving
    - ict_business_continuity.py: Art. 15 - Business continuity
    - simplified_framework.py: Art. 16 - Simplified ICT framework
    - incident_management.py: Art. 17 - Incident management
    - supervisory_feedback.py: Art. 22 - NCA feedback handling
    - resilience_testing.py: Art. 24 - Testing programme
    - ict_testing.py: Art. 25 - ICT tools testing
    - tlpt.py: Art. 26 - Threat-led penetration testing
    - tester_management.py: Art. 27 - Tester requirements
    - pooled_testing.py: Art. 26(3) - Pooled TLPT
    - cross_regulation.py: Cross-regulation integration
    - training_participation.py: Art. 30(2)(i) - FE training requests

Archived Configs (in configs/):
    - entity_classification.yaml: Entity type classification
    - nca_identification.yaml: NCA contact mapping

Migration Status: Phase 0 - Archive directory created
                  Phase 7 will move FE modules here
"""

from __future__ import annotations

# No exports - this is an archive
__all__: list[str] = []

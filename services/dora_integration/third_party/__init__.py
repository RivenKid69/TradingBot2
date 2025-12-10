# -*- coding: utf-8 -*-
"""
Third-Party Risk Interface.

Provides:
- Concentration risk assessment (Art. 29)
- CTPP (Critical Third-Party Provider) oversight preparation (Art. 31-44)
- Subcontractor management with client consent (Art. 30(2)(b))
- Third-party incident coordination

DORA Context:
    As an ICT provider, we may have our own subcontractors.
    We must:
    - Disclose our subcontractor chain to clients
    - Obtain prior consent for material changes
    - Monitor our own concentration risk
    - Prepare for potential CTPP designation

Modules (to be migrated in Phase 3):
    - concentration_risk.py: CTPP designation risk assessment
    - ctpp_oversight.py: Art. 31-44 oversight framework
    - third_party_risk.py: Risk assessment models
    - third_party_incidents.py: Subcontractor incident handling
    - subcontractor_management.py: Art. 30(2)(b) consent workflows

Target Exports (Phase 3):
    - DORAConcentrationRisk: Concentration assessment
    - DORACtppOversight: CTPP preparation
    - DORAThirdPartyRiskManagement: Risk models
    - DORAThirdPartyIncidents: Incident coordination
    - DORASubcontractorManagement: Consent and disclosure

Links with Core:
    services/core/subcontractor_monitoring.py provides operational monitoring.
    This module adds DORA-specific compliance layer.

References:
    - DORA Article 29: ICT concentration risk
    - DORA Article 30(2)(b): Prior consent for subcontracting
    - DORA Articles 31-44: CTPP oversight framework
    - CDR 2024/1773: RTS on CTPP designation criteria

Migration Status: Phase 0 - Structure only, awaiting Phase 3 migration
"""

from __future__ import annotations

__all__: list[str] = []  # Will be populated in Phase 3

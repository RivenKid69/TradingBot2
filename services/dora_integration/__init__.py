# -*- coding: utf-8 -*-
"""
DORA Integration Layer.

This package provides interfaces for interacting with financial entity clients
in a DORA-compliant manner. It implements ICT provider obligations under Art. 30.

Architecture Context:
    services/core/              - Operational resilience (14 modules) - NOT TOUCHED
    services/dora_integration/  - Client-facing interfaces (21 modules)
    services/dora/              - Legacy facade for backward compatibility
    services/archive/dora_financial_entity/  - Archived FE modules (23 modules)

Subpackages:
    - due_diligence: Audit readiness, provider info packages (Art. 30(3)(e), Art. 28(3))
    - incident_interface: Client notifications, incident data export (Art. 30(2)(d))
    - third_party: Subcontractor management, concentration risk (Art. 30(2)(b))
    - contracts: Contractual requirements, SLA guardrails, exit strategies (Art. 30)
    - reporting: Unified reporting, ROI data generation (Art. 28(3))
    - sharing: Information sharing arrangements (Art. 45)

Key Principle:
    We are an ICT Third-Party Provider (Art. 30), NOT a Financial Entity (Art. 2).
    This integration layer is what we provide to clients for THEIR compliance.

References:
    - DORA Article 30: https://www.digital-operational-resilience-act.com/Article_30.html
    - CIR 2024/2956: ITS on Register of Information
    - CDR 2024/1772: RTS on Incident Classification

Migration Status:
    Phase 0: Directory structure created (CURRENT)
    Phase 1-8: Module migration pending
"""

from __future__ import annotations

__version__ = "1.0.0"
__migration_phase__ = 0  # Current migration phase

# Subpackages will be populated during Phase 1-8 migration
# For now, this is a placeholder to establish the directory structure

__all__ = [
    "__version__",
    "__migration_phase__",
]

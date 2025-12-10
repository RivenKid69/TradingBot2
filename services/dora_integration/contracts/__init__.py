# -*- coding: utf-8 -*-
"""
Contractual Interface Layer.

Provides:
- Contract compliance checking (Art. 30)
- SLA guardrails and capacity validation (Art. 30(2)(e))
- Exit strategy templates and data migration plans (Art. 28(8))

DORA Context:
    ICT provider contracts must include specific provisions per Art. 30.
    We provide tools to:
    - Validate our contracts meet DORA requirements
    - Define SLA guardrails that clients can rely on
    - Prepare exit strategies for orderly termination

Modules (to be migrated in Phase 4):
    - contractual_requirements.py: Art. 30 contract validation
    - sla_guardrails.py: Art. 30(2)(e) SLA framework
    - exit_strategies.py: Art. 28(8) exit planning

Target Exports (Phase 4):
    - DORAContractualRequirements: Contract checker
    - SLAGuardrails: SLA validation and monitoring
    - DORAExitStrategies: Exit plan generator

Key Contract Provisions (Art. 30):
    - Service location disclosure
    - Audit and access rights
    - Incident notification commitments
    - Data protection standards
    - Termination and exit procedures

References:
    - DORA Article 30: Key contractual provisions
    - DORA Article 28(8): Exit strategies
    - DORA Article 30(2)(e): Service level descriptions
    - CDR 2024/1774: RTS on ICT risk management (contractual aspects)

Migration Status: Phase 0 - Structure only, awaiting Phase 4 migration
"""

from __future__ import annotations

__all__: list[str] = []  # Will be populated in Phase 4

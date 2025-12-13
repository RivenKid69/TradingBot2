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

Modules:
    - contractual_requirements.py: Art. 30 contract validation
    - sla_guardrails.py: Art. 30(2)(e) SLA framework
    - exit_strategies.py: Art. 28(8) exit planning

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

Migration Status: Phase 4 - Complete
"""

from __future__ import annotations

# =============================================================================
# Contractual Requirements (Art. 30)
# =============================================================================
from services.dora_integration.contracts.contractual_requirements import (
    # Main class
    DORAContractualRequirements,
    # Configuration
    ContractualRequirementsConfig,
    # Enumerations
    RequirementCategory,
    RequirementType,
    ComplianceStatus,
    GapSeverity,
    RemediationStatus,
    ContractStatus,
    # Data structures
    ContractualRequirement,
    ContractProvision,
    ContractAssessment,
    ContractGap,
    ContractAmendment,
    SLADefinition,
    ICTContract,
    TerminationClause,
    # Factory and utility functions
    create_contractual_requirements,
    get_article_30_requirements,
    get_requirement_types,
    get_basic_requirement_count,
    get_critical_requirement_count,
    get_termination_clause_templates,
)

# =============================================================================
# SLA Guardrails (Art. 30(2)(e))
# =============================================================================
from services.dora_integration.contracts.sla_guardrails import (
    # Main class
    SLAGuardrails,
    # Configuration
    SLAGuardrailsConfig,
    # Enumerations
    SLATier,
    CapacityStatus,
    ApprovalStatus,
    InfrastructureRequirement,
    OnCallRequirement,
    # Data structures
    SLATierDefinition,
    CapacityValidation,
    SLACommitmentRequest,
    CurrentCapacityState,
    # Factory and utility functions
    create_sla_guardrails,
    get_sla_tier_definitions,
    get_sla_tiers,
)

# =============================================================================
# Exit Strategies (Art. 28(8))
# =============================================================================
from services.dora_integration.contracts.exit_strategies import (
    # Main class
    DORAExitStrategies,
    # Configuration
    ExitStrategiesConfig,
    # Enumerations
    ExitTrigger,
    ExitPhase,
    ExitPlanStatus,
    TransitionType,
    ReadinessLevel,
    AlternativeProviderStatus,
    RiskLevel,
    # Data structures
    AlternativeProvider,
    DataMigrationPlan,
    TransitionTask,
    ExitRisk,
    ExitCostEstimate,
    ExitPlan,
    ExitExecution,
    ExitReadinessAssessment,
    # Factory and utility functions
    create_exit_strategies,
    get_exit_triggers,
    get_exit_phases,
    get_transition_types,
)

__all__ = [
    # ==========================================================================
    # Contractual Requirements (Art. 30)
    # ==========================================================================
    # Main class
    "DORAContractualRequirements",
    # Configuration
    "ContractualRequirementsConfig",
    # Enumerations
    "RequirementCategory",
    "RequirementType",
    "ComplianceStatus",
    "GapSeverity",
    "RemediationStatus",
    "ContractStatus",
    # Data structures
    "ContractualRequirement",
    "ContractProvision",
    "ContractAssessment",
    "ContractGap",
    "ContractAmendment",
    "SLADefinition",
    "ICTContract",
    "TerminationClause",
    # Factory functions
    "create_contractual_requirements",
    "get_article_30_requirements",
    "get_requirement_types",
    "get_basic_requirement_count",
    "get_critical_requirement_count",
    "get_termination_clause_templates",
    # ==========================================================================
    # SLA Guardrails (Art. 30(2)(e))
    # ==========================================================================
    # Main class
    "SLAGuardrails",
    # Configuration
    "SLAGuardrailsConfig",
    # Enumerations
    "SLATier",
    "CapacityStatus",
    "ApprovalStatus",
    "InfrastructureRequirement",
    "OnCallRequirement",
    # Data structures
    "SLATierDefinition",
    "CapacityValidation",
    "SLACommitmentRequest",
    "CurrentCapacityState",
    # Factory functions
    "create_sla_guardrails",
    "get_sla_tier_definitions",
    "get_sla_tiers",
    # ==========================================================================
    # Exit Strategies (Art. 28(8))
    # ==========================================================================
    # Main class
    "DORAExitStrategies",
    # Configuration
    "ExitStrategiesConfig",
    # Enumerations
    "ExitTrigger",
    "ExitPhase",
    "ExitPlanStatus",
    "TransitionType",
    "ReadinessLevel",
    "AlternativeProviderStatus",
    "RiskLevel",
    # Data structures
    "AlternativeProvider",
    "DataMigrationPlan",
    "TransitionTask",
    "ExitRisk",
    "ExitCostEstimate",
    "ExitPlan",
    "ExitExecution",
    "ExitReadinessAssessment",
    # Factory functions
    "create_exit_strategies",
    "get_exit_triggers",
    "get_exit_phases",
    "get_transition_types",
]

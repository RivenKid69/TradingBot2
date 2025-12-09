# -*- coding: utf-8 -*-
"""
On-Premises Deployment Support Package.

DORA Phase 3 Block 3.6: On-prem deployment guide

Provides on-premises deployment capabilities for enterprise clients:
- Deployment procedures and automation
- Infrastructure requirements specification
- Compliance validation
- Air-gapped environment support

DORA References:
    - Art. 30(2)(b): Data location provisions
    - Art. 28(8): Exit strategies
"""

from services.enterprise.onprem.deployment import (
    DeploymentType,
    DeploymentStatus,
    ComponentType,
    DeploymentRequirement,
    DeploymentComponent,
    DeploymentChecklist,
    OnPremDeployment,
    DeploymentConfig,
    OnPremDeploymentService,
    create_onprem_deployment,
)

from services.enterprise.onprem.requirements import (
    RequirementCategory,
    RequirementPriority,
    ComplianceLevel,
    HardwareRequirement,
    SoftwareRequirement,
    NetworkRequirement,
    SecurityRequirement,
    OnPremRequirements,
    OnPremRequirementsService,
    create_onprem_requirements,
    get_minimum_requirements,
)

__all__ = [
    # Deployment
    "DeploymentType",
    "DeploymentStatus",
    "ComponentType",
    "DeploymentRequirement",
    "DeploymentComponent",
    "DeploymentChecklist",
    "OnPremDeployment",
    "DeploymentConfig",
    "OnPremDeploymentService",
    "create_onprem_deployment",
    # Requirements
    "RequirementCategory",
    "RequirementPriority",
    "ComplianceLevel",
    "HardwareRequirement",
    "SoftwareRequirement",
    "NetworkRequirement",
    "SecurityRequirement",
    "OnPremRequirements",
    "OnPremRequirementsService",
    "create_onprem_requirements",
    "get_minimum_requirements",
]

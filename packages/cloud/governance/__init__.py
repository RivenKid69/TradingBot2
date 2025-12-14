# -*- coding: utf-8 -*-
"""
Cloud Governance Module.

CCEA Phase 8 Implementation: Privacy/GDPR + Residency + Access Controls

This module provides:
    - DSARService: GDPR Data Subject Access Requests (export/delete)
    - DataResidencyManager: EU/US region handling
    - RetentionService: Auto-purge per tenant
    - BreakGlassController: Emergency access with audit
    - HealthMonitorService: Agent health dashboards
    - AlertRulesEngine: Event-based alerting
    - CustomerManagedKeysService: CMK support for enterprise

Design Doc Reference:
    - Phase 8: "Telemetry + Privacy/GDPR + Residency + Access Controls"
    - Cloud governance (13.3): retention, export/delete, RBAC, audit
    - Data residency (13.4): EU region default for EU tenants
    - Monitoring/alerts (4.1/3.1): health dashboards, alerts

CLOUD ZONE ONLY.
"""

from typing import Final

ZONE: Final[str] = "cloud"

from .dsar import (
    DSARService,
    DSARRequest,
    DSARRequestType,
    DSARStatus,
    DSARResult,
)
from .residency import (
    DataResidencyManager,
    DataRegion,
    ResidencyPolicy,
    ResidencyConfig,
)
from .retention import (
    RetentionService,
    RetentionPolicy,
    RetentionConfig,
    PurgeResult,
)
from .break_glass import (
    BreakGlassController,
    BreakGlassRequest,
    BreakGlassReason,
    BreakGlassResult,
)
from .health_monitor import (
    HealthMonitorService,
    AgentHealth,
    HealthStatus,
    HealthDashboard,
)
from .alert_rules import (
    AlertRulesEngine,
    AlertRule,
    AlertCondition,
    AlertAction,
    AlertTrigger,
)

# CMK requires cryptography, make it optional
try:
    from .cmk import (
        CustomerManagedKeysService,
        CMKConfig,
        KeyInfo,
        EncryptionResult,
    )
    _HAS_CMK = True
except ImportError:
    _HAS_CMK = False
    CustomerManagedKeysService = None
    CMKConfig = None
    KeyInfo = None
    EncryptionResult = None

__all__ = [
    "ZONE",
    # DSAR
    "DSARService",
    "DSARRequest",
    "DSARRequestType",
    "DSARStatus",
    "DSARResult",
    # Residency
    "DataResidencyManager",
    "DataRegion",
    "ResidencyPolicy",
    "ResidencyConfig",
    # Retention
    "RetentionService",
    "RetentionPolicy",
    "RetentionConfig",
    "PurgeResult",
    # Break Glass
    "BreakGlassController",
    "BreakGlassRequest",
    "BreakGlassReason",
    "BreakGlassResult",
    # Health Monitor
    "HealthMonitorService",
    "AgentHealth",
    "HealthStatus",
    "HealthDashboard",
    # Alert Rules
    "AlertRulesEngine",
    "AlertRule",
    "AlertCondition",
    "AlertAction",
    "AlertTrigger",
    # CMK
    "CustomerManagedKeysService",
    "CMKConfig",
    "KeyInfo",
    "EncryptionResult",
]

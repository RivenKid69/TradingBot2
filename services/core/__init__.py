# -*- coding: utf-8 -*-
"""
Core Operational Resilience Package (Phase 2).

This package provides core operational resilience services per DORA requirements:
- Tiered backup with configurable RPO (15min/1h/24h)
- Enhanced healthcheck endpoints (/health, /ready, /live)
- Structured logging with correlation IDs
- Comprehensive alerting system
- DR testing framework
- Multi-AZ deployment support
- On-call rotation management
- Trust center platform
- CTPP risk monitoring

DORA References:
    - Article 11: Response and Recovery
    - Article 12: Backup Policies, Recovery Procedures and Methods
    - Article 15: ICT Business Continuity Management
    - RTS on ICT Risk Management Framework (CDR 2024/1774)

Phase 2 Work Blocks:
    - 2.1: Tiered backup (15min/1h/24h RPO)
    - 2.2: Enhanced healthcheck
    - 2.3: Structured logging with correlation IDs
    - 2.4: Comprehensive alerting
    - 2.5: Quarterly DR testing
    - 2.6: CI/CD security gates
    - 2.7: SOC2-DORA control mapping
    - 2.8: services/core/ package (this package)
    - 2.9: Multi-AZ deployment
    - 2.10: On-call rotation
    - 2.11: Subcontractor status monitoring
    - 2.12: Trust center platform
    - 2.13: DR test execution
    - 2.14: CTPP risk monitoring
"""

from __future__ import annotations

__version__ = "1.0.0"
__phase__ = 2

# Block 2.1 - Tiered Backup
from services.core.tiered_backup import (
    # Enums
    BackupTier,
    RPOLevel,
    BackupStrategy,
    ReplicationMode,
    BackupJobStatus,
    # Data structures
    TieredBackupPolicy,
    BackupSchedule,
    BackupExecution,
    ReplicationConfig,
    BackupMetrics,
    TieredBackupConfig,
    # Main class
    TieredBackupManager,
    # Factory
    create_tiered_backup_manager,
    get_tier_definitions,
)

# Block 2.2 - Enhanced Healthcheck
from services.core.enhanced_healthcheck import (
    # Enums
    ProbeType,
    DependencyType,
    DependencyStatus,
    ReadinessCondition,
    # Data structures
    HealthProbe,
    DependencyCheck,
    LivenessResult,
    ReadinessResult,
    HealthResult,
    EnhancedHealthcheckConfig,
    # Main class
    EnhancedHealthcheck,
    # Factory
    create_enhanced_healthcheck,
)

# Block 2.3 - Structured Logging
from services.core.structured_logging import (
    # Enums
    LogLevel,
    LogCategory,
    # Data structures
    CorrelationContext,
    StructuredLogEntry,
    LoggingConfig,
    # Main class
    StructuredLogger,
    # Factory and utilities
    create_structured_logger,
    get_correlation_id,
    set_correlation_id,
    correlation_context,
)

# Block 2.4 - Comprehensive Alerting
from services.core.alerting import (
    # Enums
    AlertSeverity,
    AlertChannel,
    AlertStatus,
    EscalationLevel,
    # Data structures
    AlertRule,
    Alert,
    EscalationPolicy,
    AlertingConfig,
    NotificationResult,
    # Main class
    AlertingService,
    # Factory
    create_alerting_service,
)

# Block 2.5 - DR Testing Framework
from services.core.dr_testing import (
    # Enums
    DRTestType,
    DRTestStatus,
    DRTestResult,
    RecoveryPhase,
    # Data structures
    DRTestScenario,
    DRTestExecution,
    DRTestReport,
    RecoveryMetrics,
    DRTestingConfig,
    # Main class
    DRTestingFramework,
    # Factory
    create_dr_testing_framework,
)

# Block 2.6 - CI/CD Security Gates
from services.core.security_gates import (
    # Enums
    ScanType,
    ScanStatus,
    VulnerabilitySeverity,
    GateDecision,
    # Data structures
    SecurityScanResult,
    SecurityGate,
    GatePolicy,
    SecurityGatesConfig,
    # Main class
    SecurityGatesManager,
    # Factory
    create_security_gates_manager,
)

# Block 2.7 - SOC2-DORA Control Mapping
from services.core.soc2_dora_mapping import (
    # Enums
    SOC2Category,
    DORAArticle,
    ControlStatus,
    EvidenceStatus,
    # Data structures
    ControlMapping,
    SharedControl,
    EvidenceRequirement,
    ComplianceGap,
    SOC2DORAMappingConfig,
    # Main class
    SOC2DORAMapper,
    # Factory
    create_soc2_dora_mapper,
    get_control_mappings,
)

# Block 2.9 - Multi-AZ Deployment
from services.core.multi_az import (
    # Enums
    AvailabilityZone,
    DeploymentStrategy,
    FailoverMode,
    ZoneStatus,
    # Data structures
    ZoneConfig,
    DeploymentConfig,
    FailoverConfig,
    ZoneHealthStatus,
    MultiAZConfig,
    # Main class
    MultiAZManager,
    # Factory
    create_multi_az_manager,
)

# Block 2.10 - On-Call Rotation
from services.core.oncall_rotation import (
    # Enums
    OnCallTier,
    RotationSchedule,
    EscalationPath,
    IncidentPriority,
    # Data structures
    OnCallEngineer,
    OnCallShift,
    EscalationRule,
    OnCallIncident,
    OnCallRotationConfig,
    # Main class
    OnCallRotationManager,
    # Factory
    create_oncall_rotation_manager,
)

# Block 2.11 - Subcontractor Status Monitoring
from services.core.subcontractor_monitoring import (
    # Enums
    SubcontractorHealthStatus,
    MonitoringFrequency,
    AlertThreshold,
    # Data structures
    SubcontractorStatus,
    HealthCheckResult,
    StatusReport,
    SubcontractorMonitoringConfig,
    # Main class
    SubcontractorMonitor,
    # Factory
    create_subcontractor_monitor,
)

# Block 2.12 - Trust Center Platform
from services.core.trust_center import (
    # Enums
    DocumentType,
    AccessLevel,
    CertificationType,
    ComplianceStatus,
    # Data structures
    TrustDocument,
    CertificationRecord,
    SecurityPosture,
    TrustCenterConfig,
    # Main class
    TrustCenterPlatform,
    # Factory
    create_trust_center,
)

# Block 2.13 - DR Test Execution
from services.core.dr_execution import (
    # Enums
    ExecutionPhase,
    ExecutionStatus,
    ValidationResult,
    # Data structures
    ExecutionStep,
    ExecutionResult,
    ValidationCheck,
    DRExecutionConfig,
    # Main class
    DRExecutionManager,
    # Factory
    create_dr_execution_manager,
)

# Block 2.14 - CTPP Risk Monitoring
from services.core.ctpp_monitoring import (
    # Enums
    CTPPRiskLevel,
    MonitoringStatus,
    RiskIndicator,
    # Data structures
    CTPPRiskAssessment,
    RiskMetric,
    RiskAlert,
    CTPPMonitoringConfig,
    # Main class
    CTPPRiskMonitor,
    # Factory
    create_ctpp_risk_monitor,
)

__all__ = [
    # Version info
    "__version__",
    "__phase__",

    # Block 2.1 - Tiered Backup
    "BackupTier",
    "RPOLevel",
    "BackupStrategy",
    "ReplicationMode",
    "BackupJobStatus",
    "TieredBackupPolicy",
    "BackupSchedule",
    "BackupExecution",
    "ReplicationConfig",
    "BackupMetrics",
    "TieredBackupConfig",
    "TieredBackupManager",
    "create_tiered_backup_manager",
    "get_tier_definitions",

    # Block 2.2 - Enhanced Healthcheck
    "ProbeType",
    "DependencyType",
    "DependencyStatus",
    "ReadinessCondition",
    "HealthProbe",
    "DependencyCheck",
    "LivenessResult",
    "ReadinessResult",
    "HealthResult",
    "EnhancedHealthcheckConfig",
    "EnhancedHealthcheck",
    "create_enhanced_healthcheck",

    # Block 2.3 - Structured Logging
    "LogLevel",
    "LogCategory",
    "CorrelationContext",
    "StructuredLogEntry",
    "LoggingConfig",
    "StructuredLogger",
    "create_structured_logger",
    "get_correlation_id",
    "set_correlation_id",
    "correlation_context",

    # Block 2.4 - Comprehensive Alerting
    "AlertSeverity",
    "AlertChannel",
    "AlertStatus",
    "EscalationLevel",
    "AlertRule",
    "Alert",
    "EscalationPolicy",
    "AlertingConfig",
    "NotificationResult",
    "AlertingService",
    "create_alerting_service",

    # Block 2.5 - DR Testing Framework
    "DRTestType",
    "DRTestStatus",
    "DRTestResult",
    "RecoveryPhase",
    "DRTestScenario",
    "DRTestExecution",
    "DRTestReport",
    "RecoveryMetrics",
    "DRTestingConfig",
    "DRTestingFramework",
    "create_dr_testing_framework",

    # Block 2.6 - CI/CD Security Gates
    "ScanType",
    "ScanStatus",
    "VulnerabilitySeverity",
    "GateDecision",
    "SecurityScanResult",
    "SecurityGate",
    "GatePolicy",
    "SecurityGatesConfig",
    "SecurityGatesManager",
    "create_security_gates_manager",

    # Block 2.7 - SOC2-DORA Control Mapping
    "SOC2Category",
    "DORAArticle",
    "ControlStatus",
    "EvidenceStatus",
    "ControlMapping",
    "SharedControl",
    "EvidenceRequirement",
    "ComplianceGap",
    "SOC2DORAMappingConfig",
    "SOC2DORAMapper",
    "create_soc2_dora_mapper",
    "get_control_mappings",

    # Block 2.9 - Multi-AZ Deployment
    "AvailabilityZone",
    "DeploymentStrategy",
    "FailoverMode",
    "ZoneStatus",
    "ZoneConfig",
    "DeploymentConfig",
    "FailoverConfig",
    "ZoneHealthStatus",
    "MultiAZConfig",
    "MultiAZManager",
    "create_multi_az_manager",

    # Block 2.10 - On-Call Rotation
    "OnCallTier",
    "RotationSchedule",
    "EscalationPath",
    "IncidentPriority",
    "OnCallEngineer",
    "OnCallShift",
    "EscalationRule",
    "OnCallIncident",
    "OnCallRotationConfig",
    "OnCallRotationManager",
    "create_oncall_rotation_manager",

    # Block 2.11 - Subcontractor Monitoring
    "SubcontractorHealthStatus",
    "MonitoringFrequency",
    "AlertThreshold",
    "SubcontractorStatus",
    "HealthCheckResult",
    "StatusReport",
    "SubcontractorMonitoringConfig",
    "SubcontractorMonitor",
    "create_subcontractor_monitor",

    # Block 2.12 - Trust Center Platform
    "DocumentType",
    "AccessLevel",
    "CertificationType",
    "ComplianceStatus",
    "TrustDocument",
    "CertificationRecord",
    "SecurityPosture",
    "TrustCenterConfig",
    "TrustCenterPlatform",
    "create_trust_center",

    # Block 2.13 - DR Execution
    "ExecutionPhase",
    "ExecutionStatus",
    "ValidationResult",
    "ExecutionStep",
    "ExecutionResult",
    "ValidationCheck",
    "DRExecutionConfig",
    "DRExecutionManager",
    "create_dr_execution_manager",

    # Block 2.14 - CTPP Risk Monitoring
    "CTPPRiskLevel",
    "MonitoringStatus",
    "RiskIndicator",
    "CTPPRiskAssessment",
    "RiskMetric",
    "RiskAlert",
    "CTPPMonitoringConfig",
    "CTPPRiskMonitor",
    "create_ctpp_risk_monitor",
]

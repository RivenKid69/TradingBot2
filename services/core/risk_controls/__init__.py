# -*- coding: utf-8 -*-
"""
Core Risk Controls - Universal trading risk management.

This package provides essential risk controls for ALL platform users.
These are ICT Provider baseline functionality components that implement
industry best practices for algorithmic trading safety.

Modules:
    config: Risk controls configuration (TimeSyncConfig, PreTradeControlsConfig, etc.)
    audit_models: Audit record data models for regulatory compliance
    audit_storage: Storage backends (SQLite, File, Memory) for audit trails
    retention_policy: Data retention management per regulatory requirements
    audit_trail_writer: Write-once audit trail interface
    time_sync: Clock synchronization (RTS 25 compatible)
    kill_switch: Emergency stop functionality (RTS 6 Article 12)
    pre_trade_controls: Order validation, fat finger protection
    realtime_monitor: P&L and risk monitoring
    bcp: Business continuity planning

Note:
    These controls are regulatory-neutral and apply universally as
    best practices, not specific to any particular regulatory regime.
"""

__version__ = "1.0.0"

# =============================================================================
# Configuration
# =============================================================================
from services.core.risk_controls.config import (
    ControlsMode,
    TimeSyncConfig,
    PreTradeControlsConfig,
    AuditConfig,
    KillSwitchConfig,
    RiskControlsConfig,
    load_risk_controls_config,
)

# =============================================================================
# Audit Models
# =============================================================================
from services.core.risk_controls.audit_models import (
    # Enums
    AuditEventType,
    AuditRecordPriority,
    AuditRecordStatus,
    OrderSide,
    # Data classes
    AuditRecord,
    AuditRecordBuilder,
    AuditChainStatus,
    AuditExportRequest,
    AuditExportResult,
    # Factory functions
    create_order_submitted_record,
    create_order_filled_record,
    create_risk_event_record,
    create_system_event_record,
)

# =============================================================================
# Audit Storage
# =============================================================================
from services.core.risk_controls.audit_storage import (
    # Enums
    StorageBackendType,
    StorageState,
    # Config
    AuditStorageConfig,
    StorageMetrics,
    # Base class
    AuditStorageBackend,
    # Implementations
    MemoryAuditStorage,
    SQLiteAuditStorage,
    FileAuditStorage,
    # Factory
    create_audit_storage,
)

# =============================================================================
# Retention Policy
# =============================================================================
from services.core.risk_controls.retention_policy import (
    # Enums
    RetentionPeriod,
    ArchiveStatus,
    NCARequestType,
    # Config
    RetentionPolicyConfig,
    # Data classes
    NCARequest,
    RetentionRecord,
    RetentionMetrics,
    ArchiveOperation,
    # Main class
    RetentionManager,
    # Factory
    create_retention_manager,
)

# =============================================================================
# Audit Trail Writer
# =============================================================================
from services.core.risk_controls.audit_trail_writer import (
    # Enums
    WriterMode,
    WriterState,
    # Config
    AuditTrailWriterConfig,
    # Metrics
    WriterMetrics,
    # Main class
    AuditTrailWriter,
    # Factory
    create_audit_trail_writer,
)

# =============================================================================
# Time Synchronization (RTS 25)
# =============================================================================
from services.core.risk_controls.time_sync import (
    ClockDriftSeverity,
    ClockSyncStatus,
    ClockSyncEvent,
    ComplianceClock,
    create_compliance_clock,
)

# =============================================================================
# Kill Switch (RTS 6 Article 12)
# =============================================================================
from services.core.risk_controls.kill_switch import (
    KillSwitchScope,
    KillSwitchTriggerReason,
    KillSwitchState,
    KillSwitchEvent,
    KillSwitchConfig as KillSwitchDetailedConfig,  # Alias to avoid conflict
    EmergencyContact,
    EnhancedKillSwitch,
    create_enhanced_kill_switch,
)

# =============================================================================
# Pre-Trade Controls (RTS 6 Article 15)
# =============================================================================
from services.core.risk_controls.pre_trade_controls import (
    RejectionReason,
    ControlSeverity,
    PreTradeCheckResult,
    PreTradeControlsConfig as PreTradeDetailedConfig,  # Alias to avoid conflict
    TraderAuthorization,
    MessageRateWindow,
    PreTradeControls,
    create_pre_trade_controls,
)

# =============================================================================
# Real-Time Monitoring
# =============================================================================
from services.core.risk_controls.realtime_monitor import (
    AlertSeverity,
    AlertCategory,
    ComplianceAlert,
    MonitoringThreshold,
    RealTimeMonitorConfig,
    MonitoringMetrics,
    RealTimeMonitor,
    create_realtime_monitor,
)

# =============================================================================
# Business Continuity Planning
# =============================================================================
from services.core.risk_controls.bcp import (
    # Enums
    ScenarioCategory,
    ImpactLevel,
    LikelihoodLevel,
    RecoveryStatus,
    AlertLevel,
    # Data classes
    EmergencyContact as BCPEmergencyContact,  # Alias to avoid conflict
    RecoveryStep,
    RecoveryProcedure,
    BCPScenario,
    BCPIncident,
    BusinessContinuityPlan,
    # Factory functions
    create_business_continuity_plan,
    load_bcp_from_file,
    save_bcp_to_file,
    # Templates
    get_standard_bcp_scenarios,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # --- Config ---
    "ControlsMode",
    "TimeSyncConfig",
    "PreTradeControlsConfig",
    "AuditConfig",
    "KillSwitchConfig",
    "RiskControlsConfig",
    "load_risk_controls_config",
    # --- Audit Models ---
    "AuditEventType",
    "AuditRecordPriority",
    "AuditRecordStatus",
    "OrderSide",
    "AuditRecord",
    "AuditRecordBuilder",
    "AuditChainStatus",
    "AuditExportRequest",
    "AuditExportResult",
    "create_order_submitted_record",
    "create_order_filled_record",
    "create_risk_event_record",
    "create_system_event_record",
    # --- Audit Storage ---
    "StorageBackendType",
    "StorageState",
    "AuditStorageConfig",
    "StorageMetrics",
    "AuditStorageBackend",
    "MemoryAuditStorage",
    "SQLiteAuditStorage",
    "FileAuditStorage",
    "create_audit_storage",
    # --- Retention Policy ---
    "RetentionPeriod",
    "ArchiveStatus",
    "NCARequestType",
    "RetentionPolicyConfig",
    "NCARequest",
    "RetentionRecord",
    "RetentionMetrics",
    "ArchiveOperation",
    "RetentionManager",
    "create_retention_manager",
    # --- Audit Trail Writer ---
    "WriterMode",
    "WriterState",
    "AuditTrailWriterConfig",
    "WriterMetrics",
    "AuditTrailWriter",
    "create_audit_trail_writer",
    # --- Time Sync ---
    "ClockDriftSeverity",
    "ClockSyncStatus",
    "ClockSyncEvent",
    "ComplianceClock",
    "create_compliance_clock",
    # --- Kill Switch ---
    "KillSwitchScope",
    "KillSwitchTriggerReason",
    "KillSwitchState",
    "KillSwitchEvent",
    "KillSwitchDetailedConfig",
    "EmergencyContact",
    "EnhancedKillSwitch",
    "create_enhanced_kill_switch",
    # --- Pre-Trade Controls ---
    "RejectionReason",
    "ControlSeverity",
    "PreTradeCheckResult",
    "PreTradeDetailedConfig",
    "TraderAuthorization",
    "MessageRateWindow",
    "PreTradeControls",
    "create_pre_trade_controls",
    # --- Real-Time Monitoring ---
    "AlertSeverity",
    "AlertCategory",
    "ComplianceAlert",
    "MonitoringThreshold",
    "RealTimeMonitorConfig",
    "MonitoringMetrics",
    "RealTimeMonitor",
    "create_realtime_monitor",
    # --- BCP ---
    "ScenarioCategory",
    "ImpactLevel",
    "LikelihoodLevel",
    "RecoveryStatus",
    "AlertLevel",
    "BCPEmergencyContact",
    "RecoveryStep",
    "RecoveryProcedure",
    "BCPScenario",
    "BCPIncident",
    "BusinessContinuityPlan",
    "create_business_continuity_plan",
    "load_bcp_from_file",
    "save_bcp_to_file",
    "get_standard_bcp_scenarios",
]

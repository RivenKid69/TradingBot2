# -*- coding: utf-8 -*-
"""
MiFID II Financial Entity Modules (ARCHIVED).

This package contains modules implementing MiFID II requirements specifically
for INVESTMENT FIRMS (Financial Entities). These are NOT applicable to
ICT Providers / Software Vendors.

IMPORTANT:
    Importing from this package emits a DeprecationWarning because these
    modules are archived and not intended for use in ICT Provider deployments.

Modules:
    config: Financial Entity compliance configuration
    lei_manager: LEI validation (ISO 17442)
    gleif_client: GLEIF API integration
    transaction_report: Transaction reporting (RTS 22)
    arm_client: ARM submission client
    reporting_pipeline: T+1 reporting pipeline
    self_assessment: Annual self-assessment (RTS 6)
    governance: Policy document management
    compliance_policies: Policy templates
    nca_notification: NCA notification (MiFID II Article 17(2))

Why Archived:
    Per MiFID II scope, ICT Providers who supply trading software but do not:
    - Execute trades on behalf of clients
    - Hold client assets
    - Provide investment advice
    - Operate a trading venue
    Are NOT Investment Firms and these requirements do not apply.

References:
    - MiFIR Article 26: Transaction Reporting (FE only)
    - MiFID II Article 17(2): NCA Notification (FE only)
    - ISO 17442: LEI Standard
"""

import warnings

__version__ = "1.0.0"
__archived__ = True
__archive_reason__ = "Not applicable to ICT Providers per MiFID II scope"

# Emit deprecation warning on import
warnings.warn(
    "services.archive.mifid_financial_entity is archived. "
    "These modules are for Investment Firms under MiFID II, not ICT Providers. "
    "If you are an Investment Firm, you may ignore this warning.",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# Configuration
# =============================================================================
from services.archive.mifid_financial_entity.config import (
    ComplianceMode,
    LEIStatus,
    LEIConfig,
    TransactionReportingConfig,
    NCANotificationConfig,
    GovernanceConfig,
    MiFIDIIComplianceConfig,
    load_mifid_compliance_config,
)

# =============================================================================
# LEI Management (ISO 17442)
# =============================================================================
from services.archive.mifid_financial_entity.lei_manager import (
    LEIStatus as LEIStatusEnum,  # Alias to avoid conflict
    LEIRecord,
    LEIValidationResult,
    LEIManager,
    create_lei_manager,
)

# =============================================================================
# GLEIF Client
# =============================================================================
from services.archive.mifid_financial_entity.gleif_client import (
    GLEIFErrorCode,
    GLEIFError,
    GLEIFEntity,
    GLEIFRegistration,
    GLEIFResponse,
    GLEIFClient,
    create_gleif_client,
)

# =============================================================================
# Transaction Reporting (MiFIR Article 26, RTS 22)
# =============================================================================
from services.archive.mifid_financial_entity.transaction_report import (
    # Enums
    BuySellIndicator,
    TradingCapacity,
    IdentifierType,
    InstrumentIdentifierType,
    PriceType,
    QuantityType,
    TransactionType,
    ReportStatus,
    # Validators
    ISINValidator,
    MICValidator,
    CFIValidator,
    # Data classes
    TransactionReportParty,
    TransactionReport,
    # Builder
    TransactionReportBuilder,
)

# =============================================================================
# ARM Client
# =============================================================================
from services.archive.mifid_financial_entity.arm_client import (
    # Enums
    ARMProvider,
    ARMEnvironment,
    SubmissionStatus,
    ErrorCode,
    # Data classes
    ARMError,
    SubmissionResult,
    BatchSubmissionResult,
    ARMClientConfig,
    # Clients
    ARMClient,
    MockARMClient,
    BloombergBTRLClient,
    FileARMClient,
    # Factory
    create_arm_client,
)

# =============================================================================
# Reporting Pipeline
# =============================================================================
from services.archive.mifid_financial_entity.reporting_pipeline import (
    # Enums
    PipelineStatus,
    ReportQueuePriority,
    # Data classes
    PipelineConfig,
    QueuedReport,
    PipelineMetrics,
    # Main class
    TransactionReportingPipeline,
    # Factory
    create_reporting_pipeline,
)

# =============================================================================
# Self Assessment (RTS 6)
# =============================================================================
from services.archive.mifid_financial_entity.self_assessment import (
    # Enums
    AssessmentCategory,
    ComplianceStatus,
    RemediationPriority,
    AssessmentStatus,
    # Data classes
    Evidence,
    RemediationAction,
    SelfAssessmentQuestion,
    AnnualSelfAssessment,
    # Factory functions
    create_annual_assessment,
    load_assessment_from_file,
    save_assessment_to_file,
    # Template
    get_rts6_assessment_template,
)

# =============================================================================
# Governance
# =============================================================================
from services.archive.mifid_financial_entity.governance import (
    # Enums
    PolicyType,
    PolicyStatus,
    ApprovalLevel,
    ReviewFrequency,
    # Data classes
    PolicyVersion,
    PolicySection,
    PolicyDocument,
    GovernanceFramework,
    # Factory functions
    create_governance_framework,
    create_algorithmic_trading_policy,
    create_risk_management_policy,
    create_record_keeping_policy,
    load_framework_from_file,
    save_framework_to_file,
)

# =============================================================================
# Compliance Policies
# =============================================================================
from services.archive.mifid_financial_entity.compliance_policies import (
    create_best_execution_policy,
    create_order_handling_policy,
    create_conflicts_of_interest_policy,
    create_kill_switch_policy,
    create_transaction_reporting_policy,
    create_market_abuse_prevention_policy,
    create_business_continuity_policy,
    create_all_standard_policies,
)

# =============================================================================
# NCA Notification (MiFID II Article 17(2))
# =============================================================================
from services.archive.mifid_financial_entity.nca_notification import (
    # Enums
    NCAJurisdiction,
    NotificationType,
    NotificationStatus,
    AlgorithmCategory,
    # Data classes
    NCAContact,
    AlgorithmDescription,
    NCANotification,
    # Manager
    NCANotificationManager,
    # Factory functions
    create_algorithm_description,
    create_nca_notification_manager,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version & Status
    "__version__",
    "__archived__",
    "__archive_reason__",
    # --- Config ---
    "ComplianceMode",
    "LEIStatus",
    "LEIConfig",
    "TransactionReportingConfig",
    "NCANotificationConfig",
    "GovernanceConfig",
    "MiFIDIIComplianceConfig",
    "load_mifid_compliance_config",
    # --- LEI Manager ---
    "LEIStatusEnum",
    "LEIRecord",
    "LEIValidationResult",
    "LEIManager",
    "create_lei_manager",
    # --- GLEIF Client ---
    "GLEIFErrorCode",
    "GLEIFError",
    "GLEIFEntity",
    "GLEIFRegistration",
    "GLEIFResponse",
    "GLEIFClient",
    "create_gleif_client",
    # --- Transaction Report ---
    "BuySellIndicator",
    "TradingCapacity",
    "IdentifierType",
    "InstrumentIdentifierType",
    "PriceType",
    "QuantityType",
    "TransactionType",
    "ReportStatus",
    "ISINValidator",
    "MICValidator",
    "CFIValidator",
    "TransactionReportParty",
    "TransactionReport",
    "TransactionReportBuilder",
    # --- ARM Client ---
    "ARMProvider",
    "ARMEnvironment",
    "SubmissionStatus",
    "ErrorCode",
    "ARMError",
    "SubmissionResult",
    "BatchSubmissionResult",
    "ARMClientConfig",
    "ARMClient",
    "MockARMClient",
    "BloombergBTRLClient",
    "FileARMClient",
    "create_arm_client",
    # --- Reporting Pipeline ---
    "PipelineStatus",
    "ReportQueuePriority",
    "PipelineConfig",
    "QueuedReport",
    "PipelineMetrics",
    "TransactionReportingPipeline",
    "create_reporting_pipeline",
    # --- Self Assessment ---
    "AssessmentCategory",
    "ComplianceStatus",
    "RemediationPriority",
    "AssessmentStatus",
    "Evidence",
    "RemediationAction",
    "SelfAssessmentQuestion",
    "AnnualSelfAssessment",
    "create_annual_assessment",
    "load_assessment_from_file",
    "save_assessment_to_file",
    "get_rts6_assessment_template",
    # --- Governance ---
    "PolicyType",
    "PolicyStatus",
    "ApprovalLevel",
    "ReviewFrequency",
    "PolicyVersion",
    "PolicySection",
    "PolicyDocument",
    "GovernanceFramework",
    "create_governance_framework",
    "create_algorithmic_trading_policy",
    "create_risk_management_policy",
    "create_record_keeping_policy",
    "load_framework_from_file",
    "save_framework_to_file",
    # --- Compliance Policies ---
    "create_best_execution_policy",
    "create_order_handling_policy",
    "create_conflicts_of_interest_policy",
    "create_kill_switch_policy",
    "create_transaction_reporting_policy",
    "create_market_abuse_prevention_policy",
    "create_business_continuity_policy",
    "create_all_standard_policies",
    # --- NCA Notification ---
    "NCAJurisdiction",
    "NotificationType",
    "NotificationStatus",
    "AlgorithmCategory",
    "NCAContact",
    "AlgorithmDescription",
    "NCANotification",
    "NCANotificationManager",
    "create_algorithm_description",
    "create_nca_notification_manager",
]

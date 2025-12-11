# EU AI Act Optional Modules Refactoring Plan
## Task: Make High-Risk Modules Optional While Keeping GPAI Mandatory
### Version: 1.0.0 | Target: AI Agent Execution
---
## 1. EXECUTIVE SUMMARY
**Objective**: Restructure `services/ai_act/` to separate mandatory GPAI compliance (Articles 50, 53) from optional High-Risk compliance (Articles 9, 14, 15, 17, 43, 72).
**Legal Basis**: EU AI Act Regulation 2024/1689 - algorithmic trading NOT in Annex III (High-Risk). Platform = GPAI Provider per Article 53.
**Architecture Pattern**: Follow `services/dora/` facade pattern with integration layers.
**Estimated Scope**: 21 modules → 2 packages (core + enterprise) + facade + config.
---
## 2. LEGAL CLASSIFICATION REFERENCE
### 2.1 Mandatory (GPAI Provider - Article 53)
| Module | Article | Requirement |
|--------|---------|-------------|
| transparency_disclosure.py | 50 | AI interaction disclosure |
| gpai_model_card.py | 53(1)(b) | Model card for downstream providers |
| copyright_compliance.py | 53(1)(c) | Copyright policy for training data |
| training_data_summary.py | 53(1)(d) | Public training data summary |
| user_acknowledgment.py | 50 | User acknowledgment of AI |
| technical_documentation.py | 53(1)(a) | Technical documentation (partial) |
### 2.2 Optional (High-Risk - Enterprise Feature)
| Module | Article | Use Case |
|--------|---------|----------|
| risk_management.py | 9 | Enterprise clients requiring formal RMS |
| risk_registry.py | 9 | Risk tracking for regulated entities |
| human_oversight.py | 14 | Autonomous system oversight |
| accuracy_metrics.py | 15 | Declared accuracy thresholds |
| robustness_testing.py | 15 | Adversarial/distribution testing |
| explainability.py | 13 | Decision explainability |
| data_governance.py | 10 | Training data governance |
| data_lineage.py | 10 | Data provenance tracking |
| logging_system.py | 12 | 6-month retention logging |
| qms.py | 17 | Quality management system |
| testing_framework.py | 9 | Pre-deployment testing |
| cybersecurity.py | 15(5) | AI-specific cybersecurity |
| post_market_monitoring.py | 72 | Post-market surveillance |
| conformity_assessment.py | 43 | Self-assessment framework |
---
## 3. TARGET ARCHITECTURE
### 3.1 Directory Structure
```
services/ai_act/
├── __init__.py                    # Facade with lazy loading + deprecation warnings
├── config.py                      # NEW: AIActComplianceConfig with feature flags
├── core/                          # NEW: Mandatory GPAI modules
│   ├── __init__.py
│   ├── transparency_disclosure.py # Move from root
│   ├── gpai_model_card.py         # Move from root
│   ├── copyright_compliance.py    # Move from root
│   ├── training_data_summary.py   # Move from root
│   ├── user_acknowledgment.py     # Move from root
│   └── technical_documentation.py # Move from root (GPAI subset)
├── enterprise/                    # NEW: Optional High-Risk modules
│   ├── __init__.py               # Conditional imports based on config
│   ├── risk_management.py        # Move from root
│   ├── risk_registry.py          # Move from root
│   ├── human_oversight.py        # Move from root
│   ├── accuracy_metrics.py       # Move from root
│   ├── robustness_testing.py     # Move from root
│   ├── explainability.py         # Move from root
│   ├── data_governance.py        # Move from root
│   ├── data_lineage.py           # Move from root
│   ├── logging_system.py         # Move from root
│   ├── qms.py                    # Move from root
│   ├── testing_framework.py      # Move from root
│   ├── cybersecurity.py          # Move from root
│   ├── post_market_monitoring.py # Move from root
│   └── conformity_assessment.py  # Move from root
└── _compat.py                    # NEW: Backward compatibility aliases
```
### 3.2 Configuration Model
```python
# services/ai_act/config.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Set
class AIActComplianceLevel(str, Enum):
    """EU AI Act compliance levels based on business model."""
    GPAI_ONLY = "gpai"           # Art. 50 + 53 only (SaaS default)
    ENTERPRISE = "enterprise"    # Full High-Risk (B2B financial institutions)
    CUSTOM = "custom"            # Cherry-pick specific modules
class AIActComplianceConfig(BaseModel):
    """Configuration for EU AI Act compliance modules."""
    level: AIActComplianceLevel = Field(
        default=AIActComplianceLevel.GPAI_ONLY,
        description="Compliance level: gpai (mandatory only), enterprise (full), custom"
    )
    # Enterprise module toggles (only used when level=CUSTOM)
    enable_risk_management: bool = Field(default=False, description="Art. 9 RMS")
    enable_human_oversight: bool = Field(default=False, description="Art. 14 oversight")
    enable_accuracy_metrics: bool = Field(default=False, description="Art. 15 accuracy")
    enable_robustness_testing: bool = Field(default=False, description="Art. 15 robustness")
    enable_explainability: bool = Field(default=False, description="Art. 13 XAI")
    enable_data_governance: bool = Field(default=False, description="Art. 10 data gov")
    enable_logging_system: bool = Field(default=False, description="Art. 12 logging")
    enable_qms: bool = Field(default=False, description="Art. 17 QMS")
    enable_testing_framework: bool = Field(default=False, description="Art. 9 testing")
    enable_cybersecurity: bool = Field(default=False, description="Art. 15(5) security")
    enable_post_market: bool = Field(default=False, description="Art. 72 monitoring")
    enable_conformity: bool = Field(default=False, description="Art. 43 assessment")
    # Logging config (applicable to both levels)
    log_retention_months: int = Field(default=6, ge=1, le=120)
    audit_trail_enabled: bool = Field(default=True)
    def get_enabled_enterprise_modules(self) -> Set[str]:
        """Return set of enabled enterprise module names."""
        if self.level == AIActComplianceLevel.GPAI_ONLY:
            return set()
        if self.level == AIActComplianceLevel.ENTERPRISE:
            return {
                "risk_management", "risk_registry", "human_oversight",
                "accuracy_metrics", "robustness_testing", "explainability",
                "data_governance", "data_lineage", "logging_system",
                "qms", "testing_framework", "cybersecurity",
                "post_market_monitoring", "conformity_assessment"
            }
        # CUSTOM level - check individual flags
        enabled = set()
        if self.enable_risk_management:
            enabled.add("risk_management")
            enabled.add("risk_registry")
        if self.enable_human_oversight:
            enabled.add("human_oversight")
        if self.enable_accuracy_metrics:
            enabled.add("accuracy_metrics")
        if self.enable_robustness_testing:
            enabled.add("robustness_testing")
        if self.enable_explainability:
            enabled.add("explainability")
        if self.enable_data_governance:
            enabled.add("data_governance")
            enabled.add("data_lineage")
        if self.enable_logging_system:
            enabled.add("logging_system")
        if self.enable_qms:
            enabled.add("qms")
        if self.enable_testing_framework:
            enabled.add("testing_framework")
        if self.enable_cybersecurity:
            enabled.add("cybersecurity")
        if self.enable_post_market:
            enabled.add("post_market_monitoring")
        if self.enable_conformity:
            enabled.add("conformity_assessment")
        return enabled
# Global config instance (can be overridden at runtime)
_config: Optional[AIActComplianceConfig] = None
def get_ai_act_config() -> AIActComplianceConfig:
    global _config
    if _config is None:
        _config = AIActComplianceConfig()
    return _config
def set_ai_act_config(config: AIActComplianceConfig) -> None:
    global _config
    _config = config
def is_enterprise_enabled() -> bool:
    return get_ai_act_config().level in (
        AIActComplianceLevel.ENTERPRISE,
        AIActComplianceLevel.CUSTOM
    )
```
---
## 4. IMPLEMENTATION PHASES
### Phase 1: Create Configuration & Directory Structure
**Files to create:**
1. `services/ai_act/config.py` - Config model (see 3.2)
2. `services/ai_act/core/__init__.py` - Core package init
3. `services/ai_act/enterprise/__init__.py` - Enterprise package init
4. `services/ai_act/_compat.py` - Backward compat aliases
**Actions:**
```python
# Step 1.1: Create config.py with AIActComplianceConfig
# Step 1.2: Create core/__init__.py with GPAI exports
# Step 1.3: Create enterprise/__init__.py with conditional exports
# Step 1.4: Create _compat.py with import aliases
```
### Phase 2: Move Core (GPAI) Modules
**Source → Destination:**
| From | To |
|------|-----|
| services/ai_act/transparency_disclosure.py | services/ai_act/core/transparency_disclosure.py |
| services/ai_act/gpai_model_card.py | services/ai_act/core/gpai_model_card.py |
| services/ai_act/copyright_compliance.py | services/ai_act/core/copyright_compliance.py |
| services/ai_act/training_data_summary.py | services/ai_act/core/training_data_summary.py |
| services/ai_act/user_acknowledgment.py | services/ai_act/core/user_acknowledgment.py |
| services/ai_act/technical_documentation.py | services/ai_act/core/technical_documentation.py |
**Actions:**
```bash
# Step 2.1: git mv each file
# Step 2.2: Update internal imports in moved files
# Step 2.3: Update core/__init__.py exports
```
### Phase 3: Move Enterprise (High-Risk) Modules
**Source → Destination:**
| From | To |
|------|-----|
| services/ai_act/risk_management.py | services/ai_act/enterprise/risk_management.py |
| services/ai_act/risk_registry.py | services/ai_act/enterprise/risk_registry.py |
| services/ai_act/human_oversight.py | services/ai_act/enterprise/human_oversight.py |
| services/ai_act/accuracy_metrics.py | services/ai_act/enterprise/accuracy_metrics.py |
| services/ai_act/robustness_testing.py | services/ai_act/enterprise/robustness_testing.py |
| services/ai_act/explainability.py | services/ai_act/enterprise/explainability.py |
| services/ai_act/data_governance.py | services/ai_act/enterprise/data_governance.py |
| services/ai_act/data_lineage.py | services/ai_act/enterprise/data_lineage.py |
| services/ai_act/logging_system.py | services/ai_act/enterprise/logging_system.py |
| services/ai_act/qms.py | services/ai_act/enterprise/qms.py |
| services/ai_act/testing_framework.py | services/ai_act/enterprise/testing_framework.py |
| services/ai_act/cybersecurity.py | services/ai_act/enterprise/cybersecurity.py |
| services/ai_act/post_market_monitoring.py | services/ai_act/enterprise/post_market_monitoring.py |
| services/ai_act/conformity_assessment.py | services/ai_act/enterprise/conformity_assessment.py |
**Actions:**
```bash
# Step 3.1: git mv each file
# Step 3.2: Update internal imports in moved files
# Step 3.3: Update enterprise/__init__.py with conditional exports
```
### Phase 4: Implement Facade with Lazy Loading
**File: services/ai_act/__init__.py**
```python
# -*- coding: utf-8 -*-
"""
EU AI Act Compliance Module - Tiered Architecture.
Classification:
    - GPAI Provider (Article 53) - MANDATORY for all deployments
    - High-Risk (Articles 9-17, 43, 72) - OPTIONAL enterprise feature
Legal Reference: EU AI Act Regulation 2024/1689
    - Annex III does NOT include algorithmic trading
    - Platform = GPAI Provider, not High-Risk AI System
Usage:
    # Default (GPAI only):
    from services.ai_act import TransparencyDisclosureManager
    from services.ai_act.core import create_transparency_manager
    # Enterprise (explicit):
    from services.ai_act.config import set_ai_act_config, AIActComplianceConfig, AIActComplianceLevel
    set_ai_act_config(AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE))
    from services.ai_act.enterprise import AIActRiskManager
    # Legacy (deprecated, emits warning):
    from services.ai_act import AIActRiskManager  # Works but warns
"""
from __future__ import annotations
import warnings
from typing import Any, TYPE_CHECKING
__version__ = "5.0.0"
__ai_act_compliance_phase__ = 5
__gpai_compliance_version__ = "3.0.0"
# =============================================================================
# Always-available: Core GPAI Compliance (Articles 50, 53)
# =============================================================================
from services.ai_act.core import (
    # Transparency Disclosure (Article 50)
    DisclosureType,
    DisclosureContext,
    DisclosureLanguage,
    AIDisclosure,
    DisclosureRequirement,
    DisclosureAuditRecord,
    DISCLOSURE_REQUIREMENTS,
    TransparencyDisclosureManager,
    SyntheticContentMarker,
    create_transparency_manager,
    get_disclosure_requirements,
    validate_disclosure_text,
    # Copyright Compliance (Article 53(1)(c))
    DataSourceType,
    CopyrightStatus,
    OptOutMechanism,
    DataSourceRecord,
    OptOutCheck,
    RightsHolderRequest,
    DEFAULT_DATA_SOURCES,
    CopyrightComplianceManager,
    create_copyright_manager,
    get_default_data_sources,
    validate_source_record,
    # Training Data Summary (Article 53(1)(d))
    DataCategory,
    DataQualityLevel,
    DatasetInfo,
    TrainingDataSummary,
    TrainingDataSummaryManager,
    create_default_summary,
    create_summary_manager,
    get_data_categories,
    validate_dataset_info,
    # GPAI Model Card (Article 53(1)(b))
    IntendedUse,
    LimitationType,
    RiskLevel,
    EvaluationDataset,
    ModelLimitation,
    PerformanceMetric,
    BiasAssessment,
    EthicalConsideration,
    DownstreamRequirement,
    GPAIModelCard,
    ModelCardManager,
    create_default_model_card,
    create_model_card_manager,
    get_default_limitations,
    get_default_biases,
    get_default_downstream_requirements,
    validate_model_card,
    # User Acknowledgment (Article 50)
    AcknowledgmentType,
    AcknowledgmentStatus,
    FeatureCategory,
    UserAcknowledgment,
    AcknowledgmentAuditRecord,
    ACKNOWLEDGMENT_TEXTS,
    FEATURE_REQUIREMENTS,
    UserAcknowledgmentManager,
    create_acknowledgment_manager,
    get_acknowledgment_texts,
    get_acknowledgment_feature_requirements,
    validate_acknowledgment,
    get_acknowledgment_summary,
    # Technical Documentation (Article 53(1)(a)) - GPAI subset
    DocumentationSectionType,
    ComplianceStatus,
    ExportFormat,
    DocumentationMetadata,
    ComplianceEvidence,
    DocumentationSection,
    ChangeRecord,
    TechnicalDocumentationConfig,
    TechnicalDocumentationGenerator,
    create_technical_documentation_generator,
)
# =============================================================================
# Configuration
# =============================================================================
from services.ai_act.config import (
    AIActComplianceLevel,
    AIActComplianceConfig,
    get_ai_act_config,
    set_ai_act_config,
    is_enterprise_enabled,
)
# =============================================================================
# Enterprise modules - Lazy loaded based on config
# =============================================================================
_ENTERPRISE_EXPORTS = {
    # Risk Management (Article 9)
    "AIActRiskCategory", "AIActRiskSeverity", "AIActRiskLikelihood",
    "RiskIdentification", "RiskAssessment", "RiskMitigation",
    "AIActRiskManager", "AIActRiskConfig", "create_risk_manager",
    "RiskEntry", "RiskStatus", "RiskRegistry",
    "create_risk_registry", "get_default_trading_risks",
    # Human Oversight (Article 14)
    "OversightLevel", "OversightCapability", "HumanOversightConfig",
    "HumanOversightSystem", "AnomalyDetector", "ManualOverrideController",
    "AutomationBiasMonitor", "create_human_oversight_system",
    # Explainability (Article 13)
    "ExplanationType", "FeatureContribution", "DecisionExplanation",
    "CounterfactualExplanation", "DecisionExplainer", "create_decision_explainer",
    # Accuracy Metrics (Article 15)
    "MetricType", "AccuracyMetric", "DeclaredAccuracyMetrics",
    "AccuracyMonitor", "create_accuracy_monitor", "get_default_trading_metrics",
    # Robustness Testing (Article 15)
    "RobustnessTestType", "RobustnessTestResult", "RobustnessTestSuite",
    "AdversarialTester", "DistributionShiftTester", "FailsafeTester",
    "create_robustness_test_suite",
    # Logging System (Article 12)
    "AIActLogEventType", "AIActLogCategory", "AIActLogEvent", "LogSession",
    "AIActLoggingConfig", "AIActLogger", "LogRetentionManager",
    "LogIntegrityVerifier", "AIActEventBus",
    "create_ai_act_logger", "create_ai_act_event_bus",
    # Data Governance (Article 10)
    "DataQualityDimension", "BiasType", "DataGapType", "DatasetRole",
    "QualityCheckResult", "BiasCheckResult", "DataGap", "DataQualityReport",
    "DatasetMetadata", "DataGovernanceConfig", "DataQualityAssessor",
    "BiasDetector", "DataGapAnalyzer", "DataValidator",
    "DataGovernanceFramework", "create_data_governance_framework", "create_bias_detector",
    # Data Lineage
    "DataNodeType", "TransformationType", "DataNode", "DataTransformation",
    "LineageEdge", "DataLineageConfig", "LineageGraph", "DataLineageTracker",
    "create_data_lineage_tracker",
    # QMS (Article 17)
    "QMSElementType", "ProcedureStatus", "AuditType", "AuditStatus",
    "FindingSeverity", "ChangeType", "ChangeImpact", "CAPAType", "CAPAStatus",
    "QMSProcedure", "AuditFinding", "QMSAudit", "DesignReview",
    "ChangeRequest", "CAPARecord", "ResourceRecord", "AccountabilityRecord",
    "QMSConfig", "QualityManagementSystem", "create_qms", "get_default_qms_procedures",
    # Testing Framework (Article 9)
    "TestCategory", "TestPriority", "TestStatus", "TestMetricType",
    "ComparisonOperator", "VulnerableGroup", "TestMetric", "TestScenario",
    "TestExecution", "TestSuite", "RealWorldTestPlan", "TestingConfig",
    "AIActTestingFramework", "create_testing_framework", "get_default_test_metrics",
    # Cybersecurity (Article 15(5))
    "ThreatType", "ThreatSeverity", "ValidationStatus", "AccessLevel",
    "SecurityEventType", "SecurityThreat", "ValidationResult", "IntegrityRecord",
    "AccessRecord", "SecurityPolicy", "SecurityEvent", "CybersecurityConfig",
    "InputValidator", "ModelIntegrityVerifier", "AdversarialDetector",
    "DataPoisoningDetector", "AccessControlManager", "AIActCybersecurity",
    "create_cybersecurity", "get_default_security_policies",
    # Post-Market Monitoring (Article 72)
    "MonitoringMetricType", "DriftType", "DriftSeverity", "IncidentSeverity",
    "IncidentStatus", "FeedbackType", "AlertPriority", "MonitoringMetric",
    "DriftDetectionResult", "Incident", "Feedback", "MonitoringAlert",
    "PeriodicReport", "PostMarketConfig", "PerformanceMonitor",
    "IncidentTracker", "FeedbackCollector", "PostMarketMonitoringSystem",
    "create_post_market_monitoring", "get_default_monitoring_metrics",
    # Conformity Assessment (Article 43)
    "ConformityStatus", "ChecklistItemStatus", "RequirementCategory",
    "AssessmentPhase", "GapSeverity", "ChecklistItem", "ComplianceGap",
    "AssessmentReport", "EUDeclaration", "InstructionsForUse", "RegistrationInfo",
    "ConformityAssessmentConfig", "ConformitySelfAssessment",
    "create_conformity_assessment", "get_default_checklist",
}
def __getattr__(name: str) -> Any:
    """
    Lazy loading for enterprise modules with deprecation warning.
    Triggered when accessing attributes not directly imported.
    """
    if name in _ENTERPRISE_EXPORTS:
        config = get_ai_act_config()
        if config.level == AIActComplianceLevel.GPAI_ONLY:
            warnings.warn(
                f"Accessing enterprise module '{name}' with GPAI_ONLY config. "
                f"Set AIActComplianceLevel.ENTERPRISE or CUSTOM to enable. "
                f"This will raise ImportError in future versions.",
                DeprecationWarning,
                stacklevel=2
            )
        # Dynamic import from enterprise package
        from services.ai_act import enterprise
        try:
            return getattr(enterprise, name)
        except AttributeError:
            raise AttributeError(
                f"Enterprise module '{name}' not found. "
                f"Ensure services.ai_act.enterprise is properly configured."
            )
    raise AttributeError(f"module 'services.ai_act' has no attribute '{name}'")
def __dir__():
    """Return available exports based on config."""
    base = list(globals().keys())
    if is_enterprise_enabled():
        base.extend(_ENTERPRISE_EXPORTS)
    return base
# =============================================================================
# __all__ - Core exports only by default
# =============================================================================
__all__ = [
    # Version
    "__version__", "__ai_act_compliance_phase__", "__gpai_compliance_version__",
    # Config
    "AIActComplianceLevel", "AIActComplianceConfig",
    "get_ai_act_config", "set_ai_act_config", "is_enterprise_enabled",
    # Core GPAI (always available)
    # ... (all core exports listed above)
]
```
### Phase 5: Update Enterprise Package Init
**File: services/ai_act/enterprise/__init__.py**
```python
# -*- coding: utf-8 -*-
"""
EU AI Act Enterprise Compliance Modules.
Optional High-Risk AI System compliance for enterprise clients.
NOT required for SaaS platforms under Software Provider model.
Legal Basis:
    - Articles 9, 14, 15, 17, 43, 72 apply to High-Risk AI Systems (Annex III)
    - Algorithmic trading NOT in Annex III
    - These modules provide VOLUNTARY overcompliance for B2B clients
Usage:
    # Explicit import (recommended):
    from services.ai_act.enterprise import AIActRiskManager
    # Via facade with config:
    from services.ai_act.config import set_ai_act_config, AIActComplianceConfig, AIActComplianceLevel
    set_ai_act_config(AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE))
    from services.ai_act import AIActRiskManager  # Now available
"""
from __future__ import annotations
from services.ai_act.config import get_ai_act_config, AIActComplianceLevel
import warnings
# Conditional imports based on what's enabled
def _warn_if_gpai_only(module_name: str) -> None:
    config = get_ai_act_config()
    if config.level == AIActComplianceLevel.GPAI_ONLY:
        warnings.warn(
            f"Importing enterprise module '{module_name}' with GPAI_ONLY config. "
            f"Consider setting AIActComplianceLevel.ENTERPRISE.",
            UserWarning,
            stacklevel=3
        )
# All enterprise exports (always importable directly from enterprise package)
from services.ai_act.enterprise.risk_management import (
    AIActRiskCategory, AIActRiskSeverity, AIActRiskLikelihood,
    RiskIdentification, RiskAssessment, RiskMitigation,
    AIActRiskManager, AIActRiskConfig, create_risk_manager,
)
from services.ai_act.enterprise.risk_registry import (
    RiskEntry, RiskStatus, RiskRegistry,
    create_risk_registry, get_default_trading_risks,
)
from services.ai_act.enterprise.human_oversight import (
    OversightLevel, OversightCapability, HumanOversightConfig,
    HumanOversightSystem, AnomalyDetector, ManualOverrideController,
    AutomationBiasMonitor, create_human_oversight_system,
)
from services.ai_act.enterprise.explainability import (
    ExplanationType, FeatureContribution, DecisionExplanation,
    CounterfactualExplanation, DecisionExplainer, create_decision_explainer,
)
from services.ai_act.enterprise.accuracy_metrics import (
    MetricType, AccuracyMetric, DeclaredAccuracyMetrics,
    AccuracyMonitor, create_accuracy_monitor, get_default_trading_metrics,
)
from services.ai_act.enterprise.robustness_testing import (
    RobustnessTestType, RobustnessTestResult, RobustnessTestSuite,
    AdversarialTester, DistributionShiftTester, FailsafeTester,
    create_robustness_test_suite,
)
from services.ai_act.enterprise.logging_system import (
    AIActLogEventType, AIActLogCategory, AIActLogEvent, LogSession,
    AIActLoggingConfig, AIActLogger, LogRetentionManager,
    LogIntegrityVerifier, AIActEventBus,
    create_ai_act_logger, create_ai_act_event_bus,
)
from services.ai_act.enterprise.data_governance import (
    DataQualityDimension, BiasType, DataGapType, DatasetRole,
    QualityCheckResult, BiasCheckResult, DataGap, DataQualityReport,
    DatasetMetadata, DataGovernanceConfig, DataQualityAssessor,
    BiasDetector, DataGapAnalyzer, DataValidator,
    DataGovernanceFramework, create_data_governance_framework, create_bias_detector,
)
from services.ai_act.enterprise.data_lineage import (
    DataNodeType, TransformationType, DataNode, DataTransformation,
    LineageEdge, DataLineageConfig, LineageGraph, DataLineageTracker,
    create_data_lineage_tracker,
)
from services.ai_act.enterprise.qms import (
    QMSElementType, ProcedureStatus, AuditType, AuditStatus,
    FindingSeverity, ChangeType, ChangeImpact, CAPAType, CAPAStatus,
    QMSProcedure, AuditFinding, QMSAudit, DesignReview,
    ChangeRequest, CAPARecord, ResourceRecord, AccountabilityRecord,
    QMSConfig, QualityManagementSystem, create_qms, get_default_qms_procedures,
)
from services.ai_act.enterprise.testing_framework import (
    TestCategory, TestPriority, TestStatus, TestMetricType,
    ComparisonOperator, VulnerableGroup, TestMetric, TestScenario,
    TestExecution, TestSuite, RealWorldTestPlan, TestingConfig,
    AIActTestingFramework, create_testing_framework, get_default_test_metrics,
)
from services.ai_act.enterprise.cybersecurity import (
    ThreatType, ThreatSeverity, ValidationStatus, AccessLevel,
    SecurityEventType, SecurityThreat, ValidationResult, IntegrityRecord,
    AccessRecord, SecurityPolicy, SecurityEvent, CybersecurityConfig,
    InputValidator, ModelIntegrityVerifier, AdversarialDetector,
    DataPoisoningDetector, AccessControlManager, AIActCybersecurity,
    create_cybersecurity, get_default_security_policies,
)
from services.ai_act.enterprise.post_market_monitoring import (
    MonitoringMetricType, DriftType, DriftSeverity, IncidentSeverity,
    IncidentStatus, FeedbackType, AlertPriority, MonitoringMetric,
    DriftDetectionResult, Incident, Feedback, MonitoringAlert,
    PeriodicReport, PostMarketConfig, PerformanceMonitor,
    IncidentTracker, FeedbackCollector, PostMarketMonitoringSystem,
    create_post_market_monitoring, get_default_monitoring_metrics,
)
from services.ai_act.enterprise.conformity_assessment import (
    ConformityStatus, ChecklistItemStatus, RequirementCategory,
    AssessmentPhase, GapSeverity, ChecklistItem, ComplianceGap,
    AssessmentReport, EUDeclaration, InstructionsForUse, RegistrationInfo,
    ConformityAssessmentConfig, ConformitySelfAssessment,
    create_conformity_assessment, get_default_checklist,
)
__all__ = [
    # Risk Management
    "AIActRiskCategory", "AIActRiskSeverity", "AIActRiskLikelihood",
    "RiskIdentification", "RiskAssessment", "RiskMitigation",
    "AIActRiskManager", "AIActRiskConfig", "create_risk_manager",
    "RiskEntry", "RiskStatus", "RiskRegistry",
    "create_risk_registry", "get_default_trading_risks",
    # ... (all exports)
]
```
### Phase 6: Update Core Package Init
**File: services/ai_act/core/__init__.py**
```python
# -*- coding: utf-8 -*-
"""
EU AI Act Core Compliance Modules (GPAI Provider).
MANDATORY for all deployments under Software Provider model.
Legal Basis:
    - Article 50: AI Interaction Disclosure
    - Article 53: GPAI Provider Obligations
        - 53(1)(a): Technical Documentation
        - 53(1)(b): Model Card for Downstream Providers
        - 53(1)(c): Copyright Policy
        - 53(1)(d): Training Data Summary
References:
    - https://artificialintelligenceact.eu/article/50/
    - https://artificialintelligenceact.eu/article/53/
"""
from __future__ import annotations
# Transparency Disclosure (Article 50)
from services.ai_act.core.transparency_disclosure import (
    DisclosureType, DisclosureContext, DisclosureLanguage,
    AIDisclosure, DisclosureRequirement, DisclosureAuditRecord,
    DISCLOSURE_REQUIREMENTS, TransparencyDisclosureManager,
    SyntheticContentMarker, create_transparency_manager,
    get_disclosure_requirements, validate_disclosure_text,
)
# Copyright Compliance (Article 53(1)(c))
from services.ai_act.core.copyright_compliance import (
    DataSourceType, CopyrightStatus, OptOutMechanism,
    DataSourceRecord, OptOutCheck, RightsHolderRequest,
    DEFAULT_DATA_SOURCES, CopyrightComplianceManager,
    create_copyright_manager, get_default_data_sources, validate_source_record,
)
# Training Data Summary (Article 53(1)(d))
from services.ai_act.core.training_data_summary import (
    DataCategory, DataQualityLevel, DatasetInfo, TrainingDataSummary,
    TrainingDataSummaryManager, create_default_summary, create_summary_manager,
    get_data_categories, validate_dataset_info,
)
# GPAI Model Card (Article 53(1)(b))
from services.ai_act.core.gpai_model_card import (
    IntendedUse, LimitationType, RiskLevel, EvaluationDataset,
    ModelLimitation, PerformanceMetric, BiasAssessment, EthicalConsideration,
    DownstreamRequirement, GPAIModelCard, ModelCardManager,
    create_default_model_card, create_model_card_manager,
    get_default_limitations, get_default_biases,
    get_default_downstream_requirements, validate_model_card,
)
# User Acknowledgment (Article 50)
from services.ai_act.core.user_acknowledgment import (
    AcknowledgmentType, AcknowledgmentStatus, FeatureCategory,
    UserAcknowledgment, AcknowledgmentAuditRecord,
    ACKNOWLEDGMENT_TEXTS, FEATURE_REQUIREMENTS,
    UserAcknowledgmentManager, create_acknowledgment_manager,
    get_acknowledgment_texts, get_acknowledgment_feature_requirements,
    validate_acknowledgment, get_acknowledgment_summary,
)
# Technical Documentation (Article 53(1)(a))
from services.ai_act.core.technical_documentation import (
    DocumentationSectionType, ComplianceStatus, ExportFormat,
    DocumentationMetadata, ComplianceEvidence, DocumentationSection,
    ChangeRecord, TechnicalDocumentationConfig,
    TechnicalDocumentationGenerator, create_technical_documentation_generator,
)
__all__ = [
    # Transparency
    "DisclosureType", "DisclosureContext", "DisclosureLanguage",
    "AIDisclosure", "DisclosureRequirement", "DisclosureAuditRecord",
    "DISCLOSURE_REQUIREMENTS", "TransparencyDisclosureManager",
    "SyntheticContentMarker", "create_transparency_manager",
    "get_disclosure_requirements", "validate_disclosure_text",
    # Copyright
    "DataSourceType", "CopyrightStatus", "OptOutMechanism",
    "DataSourceRecord", "OptOutCheck", "RightsHolderRequest",
    "DEFAULT_DATA_SOURCES", "CopyrightComplianceManager",
    "create_copyright_manager", "get_default_data_sources", "validate_source_record",
    # Training Data
    "DataCategory", "DataQualityLevel", "DatasetInfo", "TrainingDataSummary",
    "TrainingDataSummaryManager", "create_default_summary", "create_summary_manager",
    "get_data_categories", "validate_dataset_info",
    # Model Card
    "IntendedUse", "LimitationType", "RiskLevel", "EvaluationDataset",
    "ModelLimitation", "PerformanceMetric", "BiasAssessment", "EthicalConsideration",
    "DownstreamRequirement", "GPAIModelCard", "ModelCardManager",
    "create_default_model_card", "create_model_card_manager",
    "get_default_limitations", "get_default_biases",
    "get_default_downstream_requirements", "validate_model_card",
    # User Acknowledgment
    "AcknowledgmentType", "AcknowledgmentStatus", "FeatureCategory",
    "UserAcknowledgment", "AcknowledgmentAuditRecord",
    "ACKNOWLEDGMENT_TEXTS", "FEATURE_REQUIREMENTS",
    "UserAcknowledgmentManager", "create_acknowledgment_manager",
    "get_acknowledgment_texts", "get_acknowledgment_feature_requirements",
    "validate_acknowledgment", "get_acknowledgment_summary",
    # Technical Documentation
    "DocumentationSectionType", "ComplianceStatus", "ExportFormat",
    "DocumentationMetadata", "ComplianceEvidence", "DocumentationSection",
    "ChangeRecord", "TechnicalDocumentationConfig",
    "TechnicalDocumentationGenerator", "create_technical_documentation_generator",
]
```
### Phase 7: Update Internal Imports in Moved Files
For each moved file, update relative imports:
**Pattern:**
```python
# Before (in root):
from services.ai_act.risk_registry import RiskEntry
# After (in enterprise/):
from services.ai_act.enterprise.risk_registry import RiskEntry
# Or use relative:
from .risk_registry import RiskEntry
```
**Files requiring import updates:**
1. `enterprise/risk_management.py` → imports from risk_registry
2. `enterprise/conformity_assessment.py` → imports from multiple modules
3. `enterprise/qms.py` → imports from logging_system
4. `enterprise/post_market_monitoring.py` → imports from logging_system
5. `enterprise/testing_framework.py` → imports from accuracy_metrics
6. `core/technical_documentation.py` → may import from data_governance (check)
### Phase 8: Update Tests
**Test file updates:**
```python
# tests/test_ai_act_*.py
# Before:
from services.ai_act import AIActRiskManager
# After (explicit enterprise):
from services.ai_act.enterprise import AIActRiskManager
# Or with config:
from services.ai_act.config import set_ai_act_config, AIActComplianceConfig, AIActComplianceLevel
set_ai_act_config(AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE))
from services.ai_act import AIActRiskManager
```
**conftest.py fixture:**
```python
# tests/conftest.py
import pytest
from services.ai_act.config import set_ai_act_config, AIActComplianceConfig, AIActComplianceLevel
@pytest.fixture(scope="session", autouse=False)
def ai_act_enterprise_config():
    """Enable enterprise AI Act modules for tests."""
    config = AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE)
    set_ai_act_config(config)
    yield config
@pytest.fixture
def ai_act_gpai_config():
    """GPAI-only config for testing minimal compliance."""
    config = AIActComplianceConfig(level=AIActComplianceLevel.GPAI_ONLY)
    set_ai_act_config(config)
    yield config
```
### Phase 9: Integration with core_config.py
**Add to core_config.py:**
```python
# core_config.py
from services.ai_act.config import AIActComplianceConfig, AIActComplianceLevel
class CommonRunConfig(BaseModel):
    # ... existing fields ...
    ai_act: AIActComplianceConfig = Field(
        default_factory=lambda: AIActComplianceConfig(level=AIActComplianceLevel.GPAI_ONLY),
        description="EU AI Act compliance configuration"
    )
```
**YAML config example:**
```yaml
# configs/examples/example_train_crypto.yaml
ai_act:
  level: gpai  # Options: gpai, enterprise, custom
  # For custom level:
  # enable_risk_management: true
  # enable_human_oversight: false
```
### Phase 10: Update Documentation
**Files to update:**
1. `docs/compliance/EU_AI_ACT_INTEGRATION_PLAN.md` - Add tiered architecture section
2. `services/ai_act/__init__.py` docstring - Already done in Phase 4
3. `README.md` - Add compliance level configuration example
**New doc section:**
```markdown
## EU AI Act Compliance Levels
### GPAI Only (Default)
Minimum legal requirement for Software Provider model.
- Article 50: AI disclosure
- Article 53: GPAI provider obligations
```yaml
ai_act:
  level: gpai
```
### Enterprise
Full High-Risk compliance for B2B financial institution clients.
```yaml
ai_act:
  level: enterprise
```
### Custom
Cherry-pick specific modules.
```yaml
ai_act:
  level: custom
  enable_risk_management: true
  enable_human_oversight: true
  enable_logging_system: true
```
```
---
## 5. MIGRATION CHECKLIST
### Pre-Migration
- [ ] Backup current services/ai_act/
- [ ] Run full test suite: `pytest tests/test_ai_act_*.py -v`
- [ ] Document current import patterns in codebase
### Phase 1: Config & Structure
- [ ] Create services/ai_act/config.py
- [ ] Create services/ai_act/core/__init__.py (empty)
- [ ] Create services/ai_act/enterprise/__init__.py (empty)
- [ ] Create services/ai_act/_compat.py
### Phase 2: Move Core Modules
- [ ] git mv transparency_disclosure.py core/
- [ ] git mv gpai_model_card.py core/
- [ ] git mv copyright_compliance.py core/
- [ ] git mv training_data_summary.py core/
- [ ] git mv user_acknowledgment.py core/
- [ ] git mv technical_documentation.py core/
- [ ] Update core/__init__.py exports
- [ ] Verify: `python -c "from services.ai_act.core import *"`
### Phase 3: Move Enterprise Modules
- [ ] git mv risk_management.py enterprise/
- [ ] git mv risk_registry.py enterprise/
- [ ] git mv human_oversight.py enterprise/
- [ ] git mv accuracy_metrics.py enterprise/
- [ ] git mv robustness_testing.py enterprise/
- [ ] git mv explainability.py enterprise/
- [ ] git mv data_governance.py enterprise/
- [ ] git mv data_lineage.py enterprise/
- [ ] git mv logging_system.py enterprise/
- [ ] git mv qms.py enterprise/
- [ ] git mv testing_framework.py enterprise/
- [ ] git mv cybersecurity.py enterprise/
- [ ] git mv post_market_monitoring.py enterprise/
- [ ] git mv conformity_assessment.py enterprise/
- [ ] Update enterprise/__init__.py exports
- [ ] Verify: `python -c "from services.ai_act.enterprise import *"`
### Phase 4-6: Update Inits
- [ ] Rewrite services/ai_act/__init__.py with facade
- [ ] Test lazy loading: `from services.ai_act import AIActRiskManager`
- [ ] Test deprecation warning appears
### Phase 7: Fix Internal Imports
- [ ] Update imports in enterprise/risk_management.py
- [ ] Update imports in enterprise/conformity_assessment.py
- [ ] Update imports in enterprise/qms.py
- [ ] Update imports in enterprise/post_market_monitoring.py
- [ ] Update imports in enterprise/testing_framework.py
- [ ] Update imports in core/technical_documentation.py
- [ ] Run: `python -m py_compile services/ai_act/**/*.py`
### Phase 8: Update Tests
- [ ] Add ai_act_enterprise_config fixture to conftest.py
- [ ] Update test imports to use explicit enterprise imports
- [ ] Run: `pytest tests/test_ai_act_*.py -v`
- [ ] Verify all 1007+ tests pass
### Phase 9: Integration
- [ ] Add ai_act field to CommonRunConfig
- [ ] Update YAML config examples
- [ ] Test config loading: `python -c "from core_config import CommonRunConfig; print(CommonRunConfig().ai_act)"`
### Phase 10: Documentation
- [ ] Update EU_AI_ACT_INTEGRATION_PLAN.md
- [ ] Update README.md with compliance levels
- [ ] Create migration guide for existing users
### Post-Migration
- [ ] Run full test suite
- [ ] Run type checking: `mypy services/ai_act/`
- [ ] Verify backward compatibility with old imports
- [ ] Create git tag: v5.0.0-ai-act-tiered
---
## 6. ROLLBACK PLAN
If issues arise:
```bash
# Revert all changes
git checkout HEAD~1 -- services/ai_act/
# Or restore from backup
cp -r services/ai_act.backup/ services/ai_act/
```
---
## 7. TESTING STRATEGY
### Unit Tests
```python
# test_ai_act_config.py
def test_gpai_only_level():
    config = AIActComplianceConfig(level=AIActComplianceLevel.GPAI_ONLY)
    assert config.get_enabled_enterprise_modules() == set()
def test_enterprise_level():
    config = AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE)
    modules = config.get_enabled_enterprise_modules()
    assert "risk_management" in modules
    assert "human_oversight" in modules
    assert len(modules) == 14
def test_custom_level():
    config = AIActComplianceConfig(
        level=AIActComplianceLevel.CUSTOM,
        enable_risk_management=True,
        enable_logging_system=True
    )
    modules = config.get_enabled_enterprise_modules()
    assert modules == {"risk_management", "risk_registry", "logging_system"}
def test_lazy_import_warning():
    set_ai_act_config(AIActComplianceConfig(level=AIActComplianceLevel.GPAI_ONLY))
    with pytest.warns(DeprecationWarning, match="enterprise module"):
        from services.ai_act import AIActRiskManager
```
### Integration Tests
```python
# test_ai_act_integration.py
def test_core_always_available():
    from services.ai_act.core import TransparencyDisclosureManager
    manager = create_transparency_manager()
    assert manager is not None
def test_enterprise_with_config():
    set_ai_act_config(AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE))
    from services.ai_act import AIActRiskManager
    manager = create_risk_manager()
    assert manager is not None
```
---
## 8. REFERENCES
### Legal
- EU AI Act (Regulation 2024/1689): https://artificialintelligenceact.eu/
- Annex III (High-Risk): https://artificialintelligenceact.eu/annex/3/
- Article 53 (GPAI): https://artificialintelligenceact.eu/article/53/
- Article 50 (Transparency): https://artificialintelligenceact.eu/article/50/
### Architecture Patterns
- DORA facade pattern: services/dora/__init__.py
- Lazy loading: Python __getattr__ module-level
- Feature flags: Pydantic BaseModel with Field defaults
### Best Practices
- Google Python Style: https://google.github.io/styleguide/pyguide.html
- PEP 562 (Module __getattr__): https://peps.python.org/pep-0562/
- Pydantic Settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
---
## 9. EXPECTED OUTCOMES
### For SaaS Users (Default)
- Minimal import footprint
- No enterprise module overhead
- Clear compliance boundary
- ~3,500 LOC loaded vs ~25,000 LOC
### For Enterprise Clients
- Full compliance framework available
- Explicit opt-in via config
- Premium feature positioning
- Same code, different config
### For Development
- Clean separation of concerns
- Easier testing (mock enterprise modules)
- Clear legal mapping
- Backward compatible imports
---
## 10. EXECUTION COMMAND SEQUENCE
```bash
# Phase 1
mkdir -p services/ai_act/core services/ai_act/enterprise
touch services/ai_act/config.py services/ai_act/_compat.py
touch services/ai_act/core/__init__.py services/ai_act/enterprise/__init__.py
# Phase 2 (Core)
git mv services/ai_act/transparency_disclosure.py services/ai_act/core/
git mv services/ai_act/gpai_model_card.py services/ai_act/core/
git mv services/ai_act/copyright_compliance.py services/ai_act/core/
git mv services/ai_act/training_data_summary.py services/ai_act/core/
git mv services/ai_act/user_acknowledgment.py services/ai_act/core/
git mv services/ai_act/technical_documentation.py services/ai_act/core/
# Phase 3 (Enterprise)
git mv services/ai_act/risk_management.py services/ai_act/enterprise/
git mv services/ai_act/risk_registry.py services/ai_act/enterprise/
git mv services/ai_act/human_oversight.py services/ai_act/enterprise/
git mv services/ai_act/accuracy_metrics.py services/ai_act/enterprise/
git mv services/ai_act/robustness_testing.py services/ai_act/enterprise/
git mv services/ai_act/explainability.py services/ai_act/enterprise/
git mv services/ai_act/data_governance.py services/ai_act/enterprise/
git mv services/ai_act/data_lineage.py services/ai_act/enterprise/
git mv services/ai_act/logging_system.py services/ai_act/enterprise/
git mv services/ai_act/qms.py services/ai_act/enterprise/
git mv services/ai_act/testing_framework.py services/ai_act/enterprise/
git mv services/ai_act/cybersecurity.py services/ai_act/enterprise/
git mv services/ai_act/post_market_monitoring.py services/ai_act/enterprise/
git mv services/ai_act/conformity_assessment.py services/ai_act/enterprise/
# Verify
python -c "from services.ai_act.core import TransparencyDisclosureManager; print('Core OK')"
python -c "from services.ai_act.enterprise import AIActRiskManager; print('Enterprise OK')"
pytest tests/test_ai_act_*.py -v --tb=short
```
---
END OF PLAN

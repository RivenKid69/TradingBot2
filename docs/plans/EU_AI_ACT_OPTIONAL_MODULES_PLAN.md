# EU AI Act Optional Modules Refactoring Plan v2.0
## Task: Make High-Risk Modules Optional While Keeping GPAI Mandatory
### Version: 2.0.0 | Target: AI Agent Execution | Reviewed: Critical Issues Fixed
---
## 1. EXECUTIVE SUMMARY
**Objective**: Restructure `services/ai_act/` to separate mandatory GPAI compliance (Articles 50, 53) from optional High-Risk compliance (Articles 9, 14, 15, 17, 43, 72).
**Legal Basis**: EU AI Act Regulation 2024/1689 - algorithmic trading NOT in Annex III (High-Risk). Platform = GPAI Provider per Article 53.
**Architecture Pattern**: Follow `services/dora/` facade pattern with integration layers.
**Key Changes from v1.0:**
- Thread-safe config via `contextvars` (not global state)
- Proper Article separation (Art.53/Annex XI for GPAI vs Art.11/Annex IV for High-Risk)
- Lazy loading in enterprise package
- Full `.pyi` stub support for IDE/mypy
- Proper test isolation with autouse fixtures
---
## 2. LEGAL CLASSIFICATION REFERENCE
### 2.1 Mandatory (GPAI Provider - Article 53, Annex XI)
| Module | Article | Requirement |
|--------|---------|-------------|
| transparency_disclosure.py | 50 | AI interaction disclosure |
| gpai_model_card.py | 53(1)(b) | Model card for downstream providers |
| copyright_compliance.py | 53(1)(c) | Copyright policy for training data |
| training_data_summary.py | 53(1)(d) | Public training data summary |
| user_acknowledgment.py | 50 | User acknowledgment of AI |
| gpai_technical_docs.py | 53(1)(a), Annex XI | GPAI technical documentation (NEW - split from technical_documentation.py) |
### 2.2 Optional (High-Risk - Enterprise Feature, Article 11, Annex IV)
| Module | Article | Use Case |
|--------|---------|----------|
| risk_management.py | 9 | Enterprise clients requiring formal RMS |
| risk_registry.py | 9 | Risk tracking (depends on risk_management) |
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
| high_risk_technical_docs.py | 11, Annex IV | High-Risk documentation (NEW - split) |
---
## 3. TARGET ARCHITECTURE
### 3.1 Directory Structure
```
services/ai_act/
├── __init__.py                    # Facade with lazy loading + TYPE_CHECKING
├── __init__.pyi                   # NEW: Stub file for IDE/mypy
├── config.py                      # Thread-safe config via contextvars
├── _compat.py                     # Backward compatibility with full aliases
├── _version.py                    # NEW: Version and deprecation constants
├── core/                          # Mandatory GPAI modules (Art. 50, 53)
│   ├── __init__.py
│   ├── __init__.pyi               # Stub file
│   ├── transparency_disclosure.py
│   ├── gpai_model_card.py
│   ├── copyright_compliance.py
│   ├── training_data_summary.py
│   ├── user_acknowledgment.py
│   └── gpai_technical_docs.py     # NEW: Art.53(1)(a), Annex XI only
└── enterprise/                    # Optional High-Risk modules
    ├── __init__.py                # Lazy loading via __getattr__
    ├── __init__.pyi               # Stub file
    ├── risk_management.py
    ├── risk_registry.py           # Imports from .risk_management (relative)
    ├── human_oversight.py
    ├── accuracy_metrics.py
    ├── robustness_testing.py
    ├── explainability.py
    ├── data_governance.py
    ├── data_lineage.py
    ├── logging_system.py
    ├── qms.py
    ├── testing_framework.py
    ├── cybersecurity.py
    ├── post_market_monitoring.py
    ├── conformity_assessment.py
    └── high_risk_technical_docs.py # NEW: Art.11, Annex IV
```
### 3.2 Configuration Model (Thread-Safe)
```python
# services/ai_act/config.py
"""
Thread-safe AI Act compliance configuration using contextvars.
References:
    - PEP 567: Context Variables
    - https://docs.python.org/3/library/contextvars.html
"""
from __future__ import annotations
import contextvars
from enum import Enum
from typing import Set, Optional, Any
from pydantic import BaseModel, Field, field_validator
class AIActComplianceLevel(str, Enum):
    """EU AI Act compliance levels based on business model."""
    GPAI_ONLY = "gpai"           # Art. 50 + 53 only (SaaS default)
    ENTERPRISE = "enterprise"    # Full High-Risk (B2B financial institutions)
    CUSTOM = "custom"            # Cherry-pick specific modules
class AIActComplianceConfig(BaseModel):
    """
    Configuration for EU AI Act compliance modules.
    Thread-safe via contextvars - each async context/thread gets its own config.
    """
    model_config = {"frozen": False, "validate_assignment": True}
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
    @field_validator("level", mode="before")
    @classmethod
    def normalize_level(cls, v: Any) -> Any:
        """Normalize level string: lowercase, strip whitespace."""
        if isinstance(v, str):
            normalized = v.lower().strip()
            # Handle common variations
            if normalized in ("gpai", "gpai_only", "gpai-only"):
                return "gpai"
            if normalized in ("enterprise", "ent", "full"):
                return "enterprise"
            if normalized in ("custom", "cherry-pick"):
                return "custom"
            return normalized
        return v
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
                "post_market_monitoring", "conformity_assessment",
                "high_risk_technical_docs",
            }
        # CUSTOM level - check individual flags
        enabled: Set[str] = set()
        if self.enable_risk_management:
            enabled.add("risk_management")
            enabled.add("risk_registry")  # Always paired
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
            enabled.add("data_lineage")  # Always paired
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
            enabled.add("high_risk_technical_docs")  # Required for conformity
        return enabled
    def is_module_enabled(self, module_name: str) -> bool:
        """Check if specific module is enabled."""
        if self.level == AIActComplianceLevel.GPAI_ONLY:
            return False
        if self.level == AIActComplianceLevel.ENTERPRISE:
            return True
        return module_name in self.get_enabled_enterprise_modules()
# =============================================================================
# Thread-Safe Context Variable
# =============================================================================
_ai_act_config_var: contextvars.ContextVar[AIActComplianceConfig] = contextvars.ContextVar(
    "ai_act_config",
    default=AIActComplianceConfig()
)
def get_ai_act_config() -> AIActComplianceConfig:
    """
    Get current AI Act config for this context.
    Thread-safe: each thread/async context has its own config.
    """
    return _ai_act_config_var.get()
def set_ai_act_config(config: AIActComplianceConfig) -> contextvars.Token[AIActComplianceConfig]:
    """
    Set AI Act config for this context.
    Returns token for resetting to previous value.
    Thread-safe: only affects current thread/async context.
    Usage:
        token = set_ai_act_config(new_config)
        try:
            # work with new_config
        finally:
            reset_ai_act_config(token)
    """
    return _ai_act_config_var.set(config)
def reset_ai_act_config(token: contextvars.Token[AIActComplianceConfig]) -> None:
    """Reset config to previous value using token from set_ai_act_config."""
    _ai_act_config_var.reset(token)
def is_enterprise_enabled() -> bool:
    """Check if any enterprise modules are enabled."""
    config = get_ai_act_config()
    return config.level in (AIActComplianceLevel.ENTERPRISE, AIActComplianceLevel.CUSTOM)
# =============================================================================
# Context Manager for Temporary Config
# =============================================================================
from contextlib import contextmanager
from typing import Generator
@contextmanager
def ai_act_config_context(
    config: AIActComplianceConfig
) -> Generator[AIActComplianceConfig, None, None]:
    """
    Context manager for temporarily setting AI Act config.
    Usage:
        with ai_act_config_context(AIActComplianceConfig(level=ENTERPRISE)):
            # enterprise modules available here
        # back to previous config
    """
    token = set_ai_act_config(config)
    try:
        yield config
    finally:
        reset_ai_act_config(token)
```
### 3.3 Version and Deprecation Constants
```python
# services/ai_act/_version.py
"""Version constants and deprecation timeline."""
__version__ = "5.0.0"
__ai_act_compliance_phase__ = 5
__gpai_compliance_version__ = "3.0.0"
# Deprecation timeline
DEPRECATION_VERSION = "5.0.0"  # When deprecation warnings started
REMOVAL_VERSION = "6.0.0"      # When legacy imports will be removed
DEPRECATION_DATE = "2025-01-15"
REMOVAL_DATE = "2025-07-15"    # 6 months deprecation period
DEPRECATION_MESSAGE_TEMPLATE = (
    "{name} imported from services.ai_act is deprecated since v{dep_version}. "
    "Use 'from services.ai_act.enterprise import {name}' instead. "
    "Legacy import will be removed in v{rem_version} ({rem_date}). "
    "Set AIActComplianceLevel.ENTERPRISE to suppress this warning."
)
```
### 3.4 Backward Compatibility Module
```python
# services/ai_act/_compat.py
"""
Backward compatibility aliases for legacy imports.
This module provides aliases for code that uses old import paths.
All aliases emit DeprecationWarning when accessed.
Migration Guide:
    Old: from services.ai_act import AIActRiskManager
    New: from services.ai_act.enterprise import AIActRiskManager
    Or:  set config level to ENTERPRISE, then import from services.ai_act
Timeline:
    - v5.0.0: Deprecation warnings added
    - v6.0.0: Legacy imports removed (ETA: 2025-07-15)
"""
from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Dict, Tuple, Any
from services.ai_act._version import (
    DEPRECATION_VERSION,
    REMOVAL_VERSION,
    REMOVAL_DATE,
    DEPRECATION_MESSAGE_TEMPLATE,
)
if TYPE_CHECKING:
    from services.ai_act.enterprise import *
# =============================================================================
# Module Mapping: old name -> (new_module, new_name)
# =============================================================================
ENTERPRISE_MODULE_ALIASES: Dict[str, Tuple[str, str]] = {
    # Risk Management (Article 9)
    "AIActRiskCategory": ("risk_management", "AIActRiskCategory"),
    "AIActRiskSeverity": ("risk_management", "AIActRiskSeverity"),
    "AIActRiskLikelihood": ("risk_management", "AIActRiskLikelihood"),
    "RiskIdentification": ("risk_management", "RiskIdentification"),
    "RiskAssessment": ("risk_management", "RiskAssessment"),
    "RiskMitigation": ("risk_management", "RiskMitigation"),
    "AIActRiskManager": ("risk_management", "AIActRiskManager"),
    "AIActRiskConfig": ("risk_management", "AIActRiskConfig"),
    "create_risk_manager": ("risk_management", "create_risk_manager"),
    # Risk Registry
    "RiskEntry": ("risk_registry", "RiskEntry"),
    "RiskStatus": ("risk_registry", "RiskStatus"),
    "RiskRegistry": ("risk_registry", "RiskRegistry"),
    "create_risk_registry": ("risk_registry", "create_risk_registry"),
    "get_default_trading_risks": ("risk_registry", "get_default_trading_risks"),
    # Human Oversight (Article 14)
    "OversightLevel": ("human_oversight", "OversightLevel"),
    "OversightCapability": ("human_oversight", "OversightCapability"),
    "HumanOversightConfig": ("human_oversight", "HumanOversightConfig"),
    "HumanOversightSystem": ("human_oversight", "HumanOversightSystem"),
    "AnomalyDetector": ("human_oversight", "AnomalyDetector"),
    "ManualOverrideController": ("human_oversight", "ManualOverrideController"),
    "AutomationBiasMonitor": ("human_oversight", "AutomationBiasMonitor"),
    "create_human_oversight_system": ("human_oversight", "create_human_oversight_system"),
    # Explainability (Article 13)
    "ExplanationType": ("explainability", "ExplanationType"),
    "FeatureContribution": ("explainability", "FeatureContribution"),
    "DecisionExplanation": ("explainability", "DecisionExplanation"),
    "CounterfactualExplanation": ("explainability", "CounterfactualExplanation"),
    "DecisionExplainer": ("explainability", "DecisionExplainer"),
    "create_decision_explainer": ("explainability", "create_decision_explainer"),
    # Accuracy Metrics (Article 15)
    "MetricType": ("accuracy_metrics", "MetricType"),
    "AccuracyMetric": ("accuracy_metrics", "AccuracyMetric"),
    "DeclaredAccuracyMetrics": ("accuracy_metrics", "DeclaredAccuracyMetrics"),
    "AccuracyMonitor": ("accuracy_metrics", "AccuracyMonitor"),
    "create_accuracy_monitor": ("accuracy_metrics", "create_accuracy_monitor"),
    "get_default_trading_metrics": ("accuracy_metrics", "get_default_trading_metrics"),
    # Robustness Testing (Article 15)
    "RobustnessTestType": ("robustness_testing", "RobustnessTestType"),
    "RobustnessTestResult": ("robustness_testing", "RobustnessTestResult"),
    "RobustnessTestSuite": ("robustness_testing", "RobustnessTestSuite"),
    "AdversarialTester": ("robustness_testing", "AdversarialTester"),
    "DistributionShiftTester": ("robustness_testing", "DistributionShiftTester"),
    "FailsafeTester": ("robustness_testing", "FailsafeTester"),
    "create_robustness_test_suite": ("robustness_testing", "create_robustness_test_suite"),
    # ... (continue for all enterprise exports)
}
def get_enterprise_attr(name: str) -> Any:
    """
    Get attribute from enterprise module with deprecation warning.
    Used by facade __getattr__ for lazy loading.
    """
    if name not in ENTERPRISE_MODULE_ALIASES:
        raise AttributeError(f"Unknown enterprise attribute: {name}")
    module_name, attr_name = ENTERPRISE_MODULE_ALIASES[name]
    # Check config before warning
    from services.ai_act.config import get_ai_act_config, AIActComplianceLevel
    config = get_ai_act_config()
    if config.level == AIActComplianceLevel.GPAI_ONLY:
        warnings.warn(
            DEPRECATION_MESSAGE_TEMPLATE.format(
                name=name,
                dep_version=DEPRECATION_VERSION,
                rem_version=REMOVAL_VERSION,
                rem_date=REMOVAL_DATE,
            ),
            DeprecationWarning,
            stacklevel=3,
        )
    # Dynamic import
    import importlib
    module = importlib.import_module(f"services.ai_act.enterprise.{module_name}")
    return getattr(module, attr_name)
def get_all_enterprise_names() -> list[str]:
    """Get list of all enterprise export names."""
    return list(ENTERPRISE_MODULE_ALIASES.keys())
```
### 3.5 Enterprise Package with Lazy Loading
```python
# services/ai_act/enterprise/__init__.py
"""
EU AI Act Enterprise Compliance Modules (High-Risk).
Implements lazy loading to avoid importing all 15 modules at once.
Legal Basis:
    - Articles 9, 14, 15, 17, 43, 72 apply to High-Risk AI Systems (Annex III)
    - Algorithmic trading NOT in Annex III
    - These modules provide VOLUNTARY overcompliance for B2B clients
Usage:
    # Direct import (recommended - only loads requested module):
    from services.ai_act.enterprise import AIActRiskManager
    # Via facade with config:
    from services.ai_act.config import ai_act_config_context, AIActComplianceConfig, AIActComplianceLevel
    with ai_act_config_context(AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE)):
        from services.ai_act import AIActRiskManager
"""
from __future__ import annotations
from typing import Any, List, TYPE_CHECKING
import importlib
# =============================================================================
# Lazy Loading via __getattr__ (PEP 562)
# =============================================================================
# Module -> list of exported names
_MODULE_EXPORTS: dict[str, list[str]] = {
    "risk_management": [
        "AIActRiskCategory", "AIActRiskSeverity", "AIActRiskLikelihood",
        "RiskIdentification", "RiskAssessment", "RiskMitigation",
        "AIActRiskManager", "AIActRiskConfig", "create_risk_manager",
    ],
    "risk_registry": [
        "RiskEntry", "RiskStatus", "RiskRegistry",
        "create_risk_registry", "get_default_trading_risks",
    ],
    "human_oversight": [
        "OversightLevel", "OversightCapability", "HumanOversightConfig",
        "HumanOversightSystem", "AnomalyDetector", "ManualOverrideController",
        "AutomationBiasMonitor", "create_human_oversight_system",
    ],
    "explainability": [
        "ExplanationType", "FeatureContribution", "DecisionExplanation",
        "CounterfactualExplanation", "DecisionExplainer", "create_decision_explainer",
    ],
    "accuracy_metrics": [
        "MetricType", "AccuracyMetric", "DeclaredAccuracyMetrics",
        "AccuracyMonitor", "create_accuracy_monitor", "get_default_trading_metrics",
    ],
    "robustness_testing": [
        "RobustnessTestType", "RobustnessTestResult", "RobustnessTestSuite",
        "AdversarialTester", "DistributionShiftTester", "FailsafeTester",
        "create_robustness_test_suite",
    ],
    "logging_system": [
        "AIActLogEventType", "AIActLogCategory", "AIActLogEvent", "LogSession",
        "AIActLoggingConfig", "AIActLogger", "LogRetentionManager",
        "LogIntegrityVerifier", "AIActEventBus",
        "create_ai_act_logger", "create_ai_act_event_bus",
    ],
    "data_governance": [
        "DataQualityDimension", "BiasType", "DataGapType", "DatasetRole",
        "QualityCheckResult", "BiasCheckResult", "DataGap", "DataQualityReport",
        "DatasetMetadata", "DataGovernanceConfig", "DataQualityAssessor",
        "BiasDetector", "DataGapAnalyzer", "DataValidator",
        "DataGovernanceFramework", "create_data_governance_framework", "create_bias_detector",
    ],
    "data_lineage": [
        "DataNodeType", "TransformationType", "DataNode", "DataTransformation",
        "LineageEdge", "DataLineageConfig", "LineageGraph", "DataLineageTracker",
        "create_data_lineage_tracker",
    ],
    "qms": [
        "QMSElementType", "ProcedureStatus", "AuditType", "AuditStatus",
        "FindingSeverity", "ChangeType", "ChangeImpact", "CAPAType", "CAPAStatus",
        "QMSProcedure", "AuditFinding", "QMSAudit", "DesignReview",
        "ChangeRequest", "CAPARecord", "ResourceRecord", "AccountabilityRecord",
        "QMSConfig", "QualityManagementSystem", "create_qms", "get_default_qms_procedures",
    ],
    "testing_framework": [
        "TestCategory", "TestPriority", "TestStatus", "TestMetricType",
        "ComparisonOperator", "VulnerableGroup", "TestMetric", "TestScenario",
        "TestExecution", "TestSuite", "RealWorldTestPlan", "TestingConfig",
        "AIActTestingFramework", "create_testing_framework", "get_default_test_metrics",
    ],
    "cybersecurity": [
        "ThreatType", "ThreatSeverity", "ValidationStatus", "AccessLevel",
        "SecurityEventType", "SecurityThreat", "ValidationResult", "IntegrityRecord",
        "AccessRecord", "SecurityPolicy", "SecurityEvent", "CybersecurityConfig",
        "InputValidator", "ModelIntegrityVerifier", "AdversarialDetector",
        "DataPoisoningDetector", "AccessControlManager", "AIActCybersecurity",
        "create_cybersecurity", "get_default_security_policies",
    ],
    "post_market_monitoring": [
        "MonitoringMetricType", "DriftType", "DriftSeverity", "IncidentSeverity",
        "IncidentStatus", "FeedbackType", "AlertPriority", "MonitoringMetric",
        "DriftDetectionResult", "Incident", "Feedback", "MonitoringAlert",
        "PeriodicReport", "PostMarketConfig", "PerformanceMonitor",
        "IncidentTracker", "FeedbackCollector", "PostMarketMonitoringSystem",
        "create_post_market_monitoring", "get_default_monitoring_metrics",
    ],
    "conformity_assessment": [
        "ConformityStatus", "ChecklistItemStatus", "RequirementCategory",
        "AssessmentPhase", "GapSeverity", "ChecklistItem", "ComplianceGap",
        "AssessmentReport", "EUDeclaration", "InstructionsForUse", "RegistrationInfo",
        "ConformityAssessmentConfig", "ConformitySelfAssessment",
        "create_conformity_assessment", "get_default_checklist",
    ],
    "high_risk_technical_docs": [
        "HighRiskDocumentationGenerator", "AnnexIVSection",
        "create_high_risk_documentation_generator",
    ],
}
# Build reverse lookup: name -> module
_NAME_TO_MODULE: dict[str, str] = {}
for module, names in _MODULE_EXPORTS.items():
    for name in names:
        _NAME_TO_MODULE[name] = module
# Cache for loaded modules
_loaded_modules: dict[str, Any] = {}
def __getattr__(name: str) -> Any:
    """
    Lazy load enterprise modules on first access.
    Only imports the specific module needed, not all 15.
    """
    if name not in _NAME_TO_MODULE:
        raise AttributeError(f"module 'services.ai_act.enterprise' has no attribute '{name}'")
    module_name = _NAME_TO_MODULE[name]
    # Check cache first
    if module_name not in _loaded_modules:
        _loaded_modules[module_name] = importlib.import_module(
            f"services.ai_act.enterprise.{module_name}"
        )
    return getattr(_loaded_modules[module_name], name)
def __dir__() -> List[str]:
    """Return all available exports for IDE autocompletion."""
    return list(_NAME_TO_MODULE.keys())
# For TYPE_CHECKING only - allows IDE to see types without runtime import
if TYPE_CHECKING:
    from services.ai_act.enterprise.risk_management import *
    from services.ai_act.enterprise.risk_registry import *
    from services.ai_act.enterprise.human_oversight import *
    from services.ai_act.enterprise.explainability import *
    from services.ai_act.enterprise.accuracy_metrics import *
    from services.ai_act.enterprise.robustness_testing import *
    from services.ai_act.enterprise.logging_system import *
    from services.ai_act.enterprise.data_governance import *
    from services.ai_act.enterprise.data_lineage import *
    from services.ai_act.enterprise.qms import *
    from services.ai_act.enterprise.testing_framework import *
    from services.ai_act.enterprise.cybersecurity import *
    from services.ai_act.enterprise.post_market_monitoring import *
    from services.ai_act.enterprise.conformity_assessment import *
    from services.ai_act.enterprise.high_risk_technical_docs import *
__all__ = list(_NAME_TO_MODULE.keys())
```
### 3.6 Stub Files for IDE/Mypy Support
```python
# services/ai_act/enterprise/__init__.pyi
"""
Type stubs for enterprise package.
Provides full type information for IDE autocompletion and mypy.
"""
from services.ai_act.enterprise.risk_management import (
    AIActRiskCategory as AIActRiskCategory,
    AIActRiskSeverity as AIActRiskSeverity,
    AIActRiskLikelihood as AIActRiskLikelihood,
    RiskIdentification as RiskIdentification,
    RiskAssessment as RiskAssessment,
    RiskMitigation as RiskMitigation,
    AIActRiskManager as AIActRiskManager,
    AIActRiskConfig as AIActRiskConfig,
    create_risk_manager as create_risk_manager,
)
from services.ai_act.enterprise.risk_registry import (
    RiskEntry as RiskEntry,
    RiskStatus as RiskStatus,
    RiskRegistry as RiskRegistry,
    create_risk_registry as create_risk_registry,
    get_default_trading_risks as get_default_trading_risks,
)
# ... (continue for all modules - full explicit re-exports)
__all__: list[str]
```
### 3.7 Main Facade Module
```python
# services/ai_act/__init__.py
"""
EU AI Act Compliance Module - Tiered Architecture.
Classification:
    - GPAI Provider (Article 53) - MANDATORY for all deployments
    - High-Risk (Articles 9-17, 43, 72) - OPTIONAL enterprise feature
Legal Reference: EU AI Act Regulation 2024/1689
    - Annex III does NOT include algorithmic trading
    - Platform = GPAI Provider, not High-Risk AI System
Usage:
    # Default (GPAI only - always available):
    from services.ai_act import TransparencyDisclosureManager
    from services.ai_act.core import create_transparency_manager
    # Enterprise (explicit import - recommended):
    from services.ai_act.enterprise import AIActRiskManager
    # Enterprise (via config - for legacy code):
    from services.ai_act.config import ai_act_config_context, AIActComplianceConfig, AIActComplianceLevel
    with ai_act_config_context(AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE)):
        from services.ai_act import AIActRiskManager  # No warning in ENTERPRISE mode
"""
from __future__ import annotations
from typing import Any, List, TYPE_CHECKING
# =============================================================================
# Version Information
# =============================================================================
from services.ai_act._version import (
    __version__,
    __ai_act_compliance_phase__,
    __gpai_compliance_version__,
    DEPRECATION_VERSION,
    REMOVAL_VERSION,
)
# =============================================================================
# Configuration (Thread-Safe)
# =============================================================================
from services.ai_act.config import (
    AIActComplianceLevel,
    AIActComplianceConfig,
    get_ai_act_config,
    set_ai_act_config,
    reset_ai_act_config,
    is_enterprise_enabled,
    ai_act_config_context,
)
# =============================================================================
# Core GPAI Compliance (Articles 50, 53) - Always Available
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
    # GPAI Technical Documentation (Article 53(1)(a), Annex XI)
    GPAIDocumentationGenerator,
    AnnexXISection,
    create_gpai_documentation_generator,
)
# =============================================================================
# Enterprise Lazy Loading (via __getattr__)
# =============================================================================
from services.ai_act._compat import get_enterprise_attr, get_all_enterprise_names
def __getattr__(name: str) -> Any:
    """
    Lazy loading for enterprise modules.
    - In GPAI_ONLY mode: emits DeprecationWarning
    - In ENTERPRISE/CUSTOM mode: loads silently
    """
    try:
        return get_enterprise_attr(name)
    except AttributeError:
        raise AttributeError(f"module 'services.ai_act' has no attribute '{name}'")
def __dir__() -> List[str]:
    """Return all available exports based on current config."""
    base = [
        # Version
        "__version__", "__ai_act_compliance_phase__", "__gpai_compliance_version__",
        "DEPRECATION_VERSION", "REMOVAL_VERSION",
        # Config
        "AIActComplianceLevel", "AIActComplianceConfig",
        "get_ai_act_config", "set_ai_act_config", "reset_ai_act_config",
        "is_enterprise_enabled", "ai_act_config_context",
        # Core exports (always available)
        "DisclosureType", "DisclosureContext", "DisclosureLanguage",
        "AIDisclosure", "DisclosureRequirement", "DisclosureAuditRecord",
        "DISCLOSURE_REQUIREMENTS", "TransparencyDisclosureManager",
        "SyntheticContentMarker", "create_transparency_manager",
        "get_disclosure_requirements", "validate_disclosure_text",
        "DataSourceType", "CopyrightStatus", "OptOutMechanism",
        "DataSourceRecord", "OptOutCheck", "RightsHolderRequest",
        "DEFAULT_DATA_SOURCES", "CopyrightComplianceManager",
        "create_copyright_manager", "get_default_data_sources", "validate_source_record",
        "DataCategory", "DataQualityLevel", "DatasetInfo", "TrainingDataSummary",
        "TrainingDataSummaryManager", "create_default_summary", "create_summary_manager",
        "get_data_categories", "validate_dataset_info",
        "IntendedUse", "LimitationType", "RiskLevel", "EvaluationDataset",
        "ModelLimitation", "PerformanceMetric", "BiasAssessment", "EthicalConsideration",
        "DownstreamRequirement", "GPAIModelCard", "ModelCardManager",
        "create_default_model_card", "create_model_card_manager",
        "get_default_limitations", "get_default_biases",
        "get_default_downstream_requirements", "validate_model_card",
        "AcknowledgmentType", "AcknowledgmentStatus", "FeatureCategory",
        "UserAcknowledgment", "AcknowledgmentAuditRecord",
        "ACKNOWLEDGMENT_TEXTS", "FEATURE_REQUIREMENTS",
        "UserAcknowledgmentManager", "create_acknowledgment_manager",
        "get_acknowledgment_texts", "get_acknowledgment_feature_requirements",
        "validate_acknowledgment", "get_acknowledgment_summary",
        "GPAIDocumentationGenerator", "AnnexXISection", "create_gpai_documentation_generator",
    ]
    # Add enterprise exports if enabled
    if is_enterprise_enabled():
        base.extend(get_all_enterprise_names())
    return base
# =============================================================================
# __all__ - Explicit exports
# =============================================================================
__all__ = [
    # Version
    "__version__",
    "__ai_act_compliance_phase__",
    "__gpai_compliance_version__",
    "DEPRECATION_VERSION",
    "REMOVAL_VERSION",
    # Config
    "AIActComplianceLevel",
    "AIActComplianceConfig",
    "get_ai_act_config",
    "set_ai_act_config",
    "reset_ai_act_config",
    "is_enterprise_enabled",
    "ai_act_config_context",
    # Transparency Disclosure (Article 50)
    "DisclosureType",
    "DisclosureContext",
    "DisclosureLanguage",
    "AIDisclosure",
    "DisclosureRequirement",
    "DisclosureAuditRecord",
    "DISCLOSURE_REQUIREMENTS",
    "TransparencyDisclosureManager",
    "SyntheticContentMarker",
    "create_transparency_manager",
    "get_disclosure_requirements",
    "validate_disclosure_text",
    # Copyright Compliance (Article 53(1)(c))
    "DataSourceType",
    "CopyrightStatus",
    "OptOutMechanism",
    "DataSourceRecord",
    "OptOutCheck",
    "RightsHolderRequest",
    "DEFAULT_DATA_SOURCES",
    "CopyrightComplianceManager",
    "create_copyright_manager",
    "get_default_data_sources",
    "validate_source_record",
    # Training Data Summary (Article 53(1)(d))
    "DataCategory",
    "DataQualityLevel",
    "DatasetInfo",
    "TrainingDataSummary",
    "TrainingDataSummaryManager",
    "create_default_summary",
    "create_summary_manager",
    "get_data_categories",
    "validate_dataset_info",
    # GPAI Model Card (Article 53(1)(b))
    "IntendedUse",
    "LimitationType",
    "RiskLevel",
    "EvaluationDataset",
    "ModelLimitation",
    "PerformanceMetric",
    "BiasAssessment",
    "EthicalConsideration",
    "DownstreamRequirement",
    "GPAIModelCard",
    "ModelCardManager",
    "create_default_model_card",
    "create_model_card_manager",
    "get_default_limitations",
    "get_default_biases",
    "get_default_downstream_requirements",
    "validate_model_card",
    # User Acknowledgment (Article 50)
    "AcknowledgmentType",
    "AcknowledgmentStatus",
    "FeatureCategory",
    "UserAcknowledgment",
    "AcknowledgmentAuditRecord",
    "ACKNOWLEDGMENT_TEXTS",
    "FEATURE_REQUIREMENTS",
    "UserAcknowledgmentManager",
    "create_acknowledgment_manager",
    "get_acknowledgment_texts",
    "get_acknowledgment_feature_requirements",
    "validate_acknowledgment",
    "get_acknowledgment_summary",
    # GPAI Technical Documentation (Article 53(1)(a), Annex XI)
    "GPAIDocumentationGenerator",
    "AnnexXISection",
    "create_gpai_documentation_generator",
]
```
---
## 4. CRITICAL: Split technical_documentation.py
### 4.1 Problem
Original `technical_documentation.py` mixes:
- Article 11 / Annex IV (High-Risk documentation)
- Article 53(1)(a) / Annex XI (GPAI documentation)
### 4.2 Solution
Split into two files:
**core/gpai_technical_docs.py** (Article 53(1)(a), Annex XI):
```python
"""
GPAI Technical Documentation (Article 53(1)(a), Annex XI).
Required for all GPAI providers. Documents model capabilities and limitations.
Annex XI Required Elements:
    1. Model identity and version
    2. Intended tasks and modalities
    3. Upstream provider identification
    4. Interaction with hardware/software
    5. Acceptable use policy
    6. Computational resources used
    7. Training data summary (cross-ref to training_data_summary.py)
    8. Evaluation results and known limitations
References:
    - Article 53: https://artificialintelligenceact.eu/article/53/
    - Annex XI: https://artificialintelligenceact.eu/annex/11/
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
# ... GPAI-specific documentation implementation
class AnnexXISection(Enum):
    MODEL_IDENTITY = "model_identity"
    INTENDED_TASKS = "intended_tasks"
    UPSTREAM_PROVIDER = "upstream_provider"
    HARDWARE_SOFTWARE = "hardware_software"
    ACCEPTABLE_USE = "acceptable_use"
    COMPUTATIONAL_RESOURCES = "computational_resources"
    TRAINING_DATA = "training_data"
    EVALUATION_RESULTS = "evaluation_results"
    KNOWN_LIMITATIONS = "known_limitations"
class GPAIDocumentationGenerator:
    """Generate Annex XI compliant documentation for GPAI models."""
    # ... implementation
```
**enterprise/high_risk_technical_docs.py** (Article 11, Annex IV):
```python
"""
High-Risk AI Technical Documentation (Article 11, Annex IV).
Required ONLY for High-Risk AI systems per Annex III.
Annex IV Required Elements:
    1. General description of the AI system
    2. Detailed description of elements and development process
    3. Monitoring, functioning, and control
    4. Performance metrics appropriateness
    5. Risk management system description
    6. Changes made to the system
References:
    - Article 11: https://artificialintelligenceact.eu/article/11/
    - Annex IV: https://artificialintelligenceact.eu/annex/4/
"""
from __future__ import annotations
from enum import Enum
# Import from enterprise siblings (relative imports)
from .risk_management import AIActRiskManager
from .human_oversight import HumanOversightSystem
from .data_governance import DataGovernanceFramework
class AnnexIVSection(Enum):
    GENERAL_DESCRIPTION = "general_description"
    ALGORITHM_AND_DATA = "algorithm_and_data"
    MONITORING_AND_CONTROL = "monitoring_and_control"
    PERFORMANCE_METRICS = "performance_metrics"
    RISK_MANAGEMENT = "risk_management"  # References enterprise module
    CHANGE_LOG = "change_log"
    HUMAN_OVERSIGHT = "human_oversight"  # References enterprise module
    DATA_GOVERNANCE = "data_governance"  # References enterprise module
class HighRiskDocumentationGenerator:
    """Generate Annex IV compliant documentation for High-Risk AI systems."""
    # ... implementation with enterprise dependencies
```
---
## 5. TEST FIXTURES (Thread-Safe)
```python
# tests/conftest_ai_act.py
"""
AI Act test fixtures with proper isolation.
Each test gets clean config state via contextvars.
"""
import pytest
from contextvars import copy_context
from services.ai_act.config import (
    AIActComplianceConfig,
    AIActComplianceLevel,
    set_ai_act_config,
    reset_ai_act_config,
    ai_act_config_context,
)
@pytest.fixture(autouse=True)
def reset_ai_act_config_after_test():
    """
    Reset AI Act config after each test.
    Uses contextvars token for proper cleanup.
    """
    # Store default config
    default_config = AIActComplianceConfig()
    token = set_ai_act_config(default_config)
    yield
    # Reset to default after test
    reset_ai_act_config(token)
@pytest.fixture
def ai_act_gpai_config():
    """GPAI-only config fixture."""
    config = AIActComplianceConfig(level=AIActComplianceLevel.GPAI_ONLY)
    with ai_act_config_context(config):
        yield config
@pytest.fixture
def ai_act_enterprise_config():
    """Enterprise config fixture."""
    config = AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE)
    with ai_act_config_context(config):
        yield config
@pytest.fixture
def ai_act_custom_config():
    """Custom config fixture factory."""
    def _make_config(**kwargs):
        config = AIActComplianceConfig(
            level=AIActComplianceLevel.CUSTOM,
            **kwargs
        )
        return ai_act_config_context(config)
    return _make_config
# =============================================================================
# Test Isolation for Parallel Execution (pytest-xdist)
# =============================================================================
@pytest.fixture(scope="function")
def isolated_context():
    """
    Run test in isolated context for pytest-xdist compatibility.
    Each test gets its own contextvars context.
    """
    ctx = copy_context()
    def run_in_context(func, *args, **kwargs):
        return ctx.run(func, *args, **kwargs)
    return run_in_context
```
---
## 6. INTERNAL IMPORT UPDATES
### 6.1 enterprise/risk_registry.py
```python
# Before:
from services.ai_act.risk_management import (
    AIActRiskCategory,
    AIActRiskSeverity,
    ...
)
# After (relative import within enterprise package):
from .risk_management import (
    AIActRiskCategory,
    AIActRiskSeverity,
    ...
)
```
### 6.2 All enterprise modules use relative imports
```python
# enterprise/conformity_assessment.py
from .risk_management import AIActRiskManager
from .human_oversight import HumanOversightSystem
from .data_governance import DataGovernanceFramework
from .logging_system import AIActLogger
# enterprise/qms.py
from .logging_system import AIActLogger, AIActLogEventType
# enterprise/post_market_monitoring.py
from .logging_system import AIActLogger
# enterprise/testing_framework.py
from .accuracy_metrics import AccuracyMetric, MetricType
```
---
## 7. ROLLBACK PLAN (Improved)
### 7.1 Pre-Migration
```bash
# Create feature branch
git checkout -b feature/ai-act-tiered-architecture
# Create backup tag
git tag backup/ai-act-before-tiered-$(date +%Y%m%d)
# Run baseline tests
pytest tests/test_ai_act_*.py -v > baseline_test_results.txt
```
### 7.2 Incremental Commits
Each phase should be a separate commit:
```bash
git commit -m "feat(ai-act): add thread-safe config with contextvars"
git commit -m "feat(ai-act): create core package with GPAI modules"
git commit -m "feat(ai-act): create enterprise package with lazy loading"
git commit -m "feat(ai-act): split technical_documentation into GPAI and High-Risk"
git commit -m "feat(ai-act): add facade with deprecation warnings"
git commit -m "feat(ai-act): add .pyi stub files for IDE support"
git commit -m "test(ai-act): update test fixtures for thread safety"
git commit -m "docs(ai-act): update documentation for tiered architecture"
```
### 7.3 Rollback Commands
```bash
# Rollback single phase (last commit)
git revert HEAD
# Rollback entire feature
git checkout main
git branch -D feature/ai-act-tiered-architecture
# Restore from backup tag
git checkout backup/ai-act-before-tiered-YYYYMMDD -- services/ai_act/
# Full reset to backup
git reset --hard backup/ai-act-before-tiered-YYYYMMDD
```
### 7.4 Validation Before Merge
```bash
# Run all AI Act tests
pytest tests/test_ai_act_*.py -v --tb=short
# Run with parallel workers (tests thread safety)
pytest tests/test_ai_act_*.py -v -n auto
# Type checking
mypy services/ai_act/ --strict
# Import verification
python -c "from services.ai_act import *; print('Facade OK')"
python -c "from services.ai_act.core import *; print('Core OK')"
python -c "from services.ai_act.enterprise import AIActRiskManager; print('Enterprise OK')"
# Deprecation warning test
python -c "
import warnings
warnings.filterwarnings('error', category=DeprecationWarning)
try:
    from services.ai_act import AIActRiskManager
    print('ERROR: Should have raised DeprecationWarning')
except DeprecationWarning as e:
    print(f'OK: DeprecationWarning raised: {e}')
"
```
---
## 8. MIGRATION CHECKLIST v2.0
### Pre-Migration
- [ ] Create feature branch: `feature/ai-act-tiered-architecture`
- [ ] Create backup tag: `backup/ai-act-before-tiered-YYYYMMDD`
- [ ] Run baseline tests and save results
- [ ] Document current import patterns with grep
### Phase 1: Core Infrastructure
- [ ] Create `services/ai_act/_version.py`
- [ ] Create `services/ai_act/config.py` with contextvars
- [ ] Create `services/ai_act/_compat.py` with full mapping
- [ ] Verify: `python -c "from services.ai_act.config import ai_act_config_context"`
- [ ] Commit: "feat(ai-act): add thread-safe config with contextvars"
### Phase 2: Create Core Package
- [ ] `mkdir -p services/ai_act/core`
- [ ] Create `services/ai_act/core/__init__.py`
- [ ] Create `services/ai_act/core/__init__.pyi`
- [ ] `git mv transparency_disclosure.py core/`
- [ ] `git mv gpai_model_card.py core/`
- [ ] `git mv copyright_compliance.py core/`
- [ ] `git mv training_data_summary.py core/`
- [ ] `git mv user_acknowledgment.py core/`
- [ ] Create `core/gpai_technical_docs.py` (NEW - extract from technical_documentation.py)
- [ ] Verify: `python -c "from services.ai_act.core import TransparencyDisclosureManager"`
- [ ] Commit: "feat(ai-act): create core package with GPAI modules"
### Phase 3: Create Enterprise Package
- [ ] `mkdir -p services/ai_act/enterprise`
- [ ] Create `services/ai_act/enterprise/__init__.py` with lazy loading
- [ ] Create `services/ai_act/enterprise/__init__.pyi`
- [ ] `git mv risk_management.py enterprise/`
- [ ] `git mv risk_registry.py enterprise/`
- [ ] `git mv human_oversight.py enterprise/`
- [ ] `git mv accuracy_metrics.py enterprise/`
- [ ] `git mv robustness_testing.py enterprise/`
- [ ] `git mv explainability.py enterprise/`
- [ ] `git mv data_governance.py enterprise/`
- [ ] `git mv data_lineage.py enterprise/`
- [ ] `git mv logging_system.py enterprise/`
- [ ] `git mv qms.py enterprise/`
- [ ] `git mv testing_framework.py enterprise/`
- [ ] `git mv cybersecurity.py enterprise/`
- [ ] `git mv post_market_monitoring.py enterprise/`
- [ ] `git mv conformity_assessment.py enterprise/`
- [ ] Create `enterprise/high_risk_technical_docs.py` (NEW - extract from technical_documentation.py)
- [ ] Delete original `technical_documentation.py`
- [ ] Verify: `python -c "from services.ai_act.enterprise import AIActRiskManager"`
- [ ] Commit: "feat(ai-act): create enterprise package with lazy loading"
### Phase 4: Update Internal Imports
- [ ] Update `enterprise/risk_registry.py`: `from .risk_management import ...`
- [ ] Update `enterprise/conformity_assessment.py`: relative imports
- [ ] Update `enterprise/qms.py`: relative imports
- [ ] Update `enterprise/post_market_monitoring.py`: relative imports
- [ ] Update `enterprise/testing_framework.py`: relative imports
- [ ] Update `enterprise/high_risk_technical_docs.py`: relative imports
- [ ] Run: `python -m py_compile services/ai_act/enterprise/*.py`
- [ ] Commit: "refactor(ai-act): update internal imports to relative"
### Phase 5: Update Facade
- [ ] Rewrite `services/ai_act/__init__.py`
- [ ] Create `services/ai_act/__init__.pyi`
- [ ] Verify facade imports
- [ ] Test deprecation warning
- [ ] Commit: "feat(ai-act): add facade with deprecation warnings"
### Phase 6: Update Tests
- [ ] Create `tests/conftest_ai_act.py` with fixtures
- [ ] Update test imports to use explicit paths
- [ ] Add import to main `conftest.py`: `pytest_plugins = ["tests.conftest_ai_act"]`
- [ ] Run: `pytest tests/test_ai_act_*.py -v`
- [ ] Run parallel: `pytest tests/test_ai_act_*.py -v -n auto`
- [ ] Commit: "test(ai-act): update fixtures for thread safety"
### Phase 7: Integration
- [ ] Add `ai_act` field to `CommonRunConfig` in `core_config.py`
- [ ] Update YAML config examples
- [ ] Verify: `python -c "from core_config import CommonRunConfig; print(CommonRunConfig().ai_act)"`
- [ ] Commit: "feat(config): integrate AI Act config with CommonRunConfig"
### Phase 8: Documentation
- [ ] Update `EU_AI_ACT_INTEGRATION_PLAN.md`
- [ ] Update module docstrings
- [ ] Create `MIGRATION_GUIDE.md` for users
- [ ] Commit: "docs(ai-act): update documentation for tiered architecture"
### Post-Migration
- [ ] Run full test suite: `pytest tests/test_ai_act_*.py -v`
- [ ] Run type checking: `mypy services/ai_act/ --strict`
- [ ] Run parallel tests: `pytest tests/test_ai_act_*.py -n auto`
- [ ] Verify all 1007+ tests pass
- [ ] Create PR for review
- [ ] After merge: `git tag v5.0.0-ai-act-tiered`
---
## 9. RESOLVED ISSUES SUMMARY
| # | Issue | Resolution |
|---|-------|------------|
| 1 | Thread safety | `contextvars` instead of global state |
| 2 | core→enterprise dependency | Split technical_documentation.py into GPAI (core) and High-Risk (enterprise) |
| 3 | Wrong dependency direction | Fixed: risk_registry imports FROM risk_management |
| 4 | IDE/mypy support | Added `.pyi` stub files + TYPE_CHECKING blocks |
| 5 | Import order dependency | Context manager `ai_act_config_context` for scoped config |
| 6 | Incomplete __all__ | Fully explicit __all__ with all exports |
| 7 | Test fixture state leak | `autouse=True` fixture with token-based reset |
| 8 | Empty _compat.py | Full implementation with module mapping |
| 9 | YAML enum validation | `@field_validator` with normalization |
| 10 | Primitive rollback | Feature branch + incremental commits + backup tags |
| 11 | Article 11 vs 53 | gpai_technical_docs.py (Art.53) vs high_risk_technical_docs.py (Art.11) |
| 12 | Enterprise loads all | Lazy loading via `__getattr__` + module cache |
| 13 | No deprecation versioning | `_version.py` with DEPRECATION_VERSION and REMOVAL_DATE |
---
## 10. EXECUTION COMMAND SEQUENCE
```bash
# Setup
git checkout -b feature/ai-act-tiered-architecture
git tag backup/ai-act-before-tiered-$(date +%Y%m%d)
# Phase 1: Infrastructure
mkdir -p services/ai_act/core services/ai_act/enterprise
cat > services/ai_act/_version.py << 'EOF'
__version__ = "5.0.0"
__ai_act_compliance_phase__ = 5
__gpai_compliance_version__ = "3.0.0"
DEPRECATION_VERSION = "5.0.0"
REMOVAL_VERSION = "6.0.0"
DEPRECATION_DATE = "2025-01-15"
REMOVAL_DATE = "2025-07-15"
EOF
# Create config.py, _compat.py (from plan sections 3.2, 3.4)
# Phase 2: Core modules
git mv services/ai_act/transparency_disclosure.py services/ai_act/core/
git mv services/ai_act/gpai_model_card.py services/ai_act/core/
git mv services/ai_act/copyright_compliance.py services/ai_act/core/
git mv services/ai_act/training_data_summary.py services/ai_act/core/
git mv services/ai_act/user_acknowledgment.py services/ai_act/core/
# Create gpai_technical_docs.py from technical_documentation.py (GPAI parts only)
# Phase 3: Enterprise modules
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
# Create high_risk_technical_docs.py from technical_documentation.py (High-Risk parts)
rm services/ai_act/technical_documentation.py
# Phase 4: Update imports in enterprise modules
sed -i 's/from services\.ai_act\.risk_management/from .risk_management/g' services/ai_act/enterprise/risk_registry.py
# ... (repeat for other files)
# Phase 5-8: Create __init__.py files, stubs, tests, docs
# Verify
python -c "from services.ai_act.core import TransparencyDisclosureManager; print('Core OK')"
python -c "from services.ai_act.enterprise import AIActRiskManager; print('Enterprise OK')"
pytest tests/test_ai_act_*.py -v --tb=short
pytest tests/test_ai_act_*.py -v -n auto  # Parallel
mypy services/ai_act/ --strict
# Commit and push
git add -A
git commit -m "feat(ai-act): implement tiered architecture with thread-safe config"
git push -u origin feature/ai-act-tiered-architecture
```
---
END OF PLAN v2.0

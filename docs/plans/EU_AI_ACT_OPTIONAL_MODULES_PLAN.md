# EU AI Act Optional Modules Refactoring Plan v2.2

## Task: Make Optional EU AI Act Modules Explicit

### Version: 2.2.0 | Target: AI Agent Execution | Reviewed: All Critical Issues Resolved + Phases Restructured

---

## 1. EXECUTIVE SUMMARY

**Objective**: Restructure `services/ai_act/` to separate transparency/documentation tooling (Articles 50/53; deployment-dependent) from optional conservative coverage of Articles 9–17 and related workflows (enterprise feature).
**Legal Basis**: EU AI Act Regulation 2024/1689. Applicability/classification depends on deployment context; do not self-classify in docs without counsel review.
**Architecture Pattern**: Follow `services/dora/` facade pattern with integration layers.
**Key Changes from v2.1:**

- **RESTRUCTURED**: 10 explicit phases (0-9) with clear Goals and Exit Criteria
- **ADDED**: Phase 0 (Pre-Migration) and Phase 9 (Post-Migration Validation) as formal phases
- **IMPROVED**: Each phase has Verification scripts and explicit Commit message

**Key Changes from v2.0:**

- **CRITICAL**: Enterprise access guard with `RuntimeError` (not just DeprecationWarning)
- **CRITICAL**: Config check in enterprise `__getattr__` before module loading
- **CRITICAL**: TYPE_CHECKING guards for cross-imports (prevent circular imports)
- Environment variable override (`AIACT_COMPLIANCE_LEVEL`)
- Config serialization (`to_dict()`/`from_dict()`) for persistence
- Module cache invalidation mechanism
- Graceful error handling with `EnterpriseNotAvailableError`
- Explicit Art.52/53(2) out-of-scope documentation
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

### 2.3 Explicitly Out-of-Scope (with Legal Justification)

| Article | Title | Why Not Applicable |
|---------|-------|-------------------|
| **Article 52** | Transparency for certain AI systems | Applies to: emotion recognition, biometric categorization, deepfakes. Platform does NOT use these capabilities — algo-trading models process market data, not biometric/emotional data. |
| **Article 53(2)** | GPAI with systemic risk | Applies to: GPAI models with >10^25 FLOPs training compute OR designated by EU Commission. Platform uses downstream GPAI models (e.g., LLMs for analysis), does NOT train foundation models at systemic risk scale. |
| **Article 6** | High-Risk classification rules | Classification depends on deployment context and roles; we do not self-classify in docs and recommend counsel review for applicability. |
| **Annex I** | Harmonised legislation | CE marking applicability depends on concrete product scope; verify with qualified counsel before any external statement. |

**Design Intent Summary (Non-legal):**
> The platform is designed as a **B2B software/ICT provider**. Users connect their own broker accounts; live execution occurs only via the customer-controlled Agent, and the Cloud does not hold assets/credentials or provide execution-as-a-service. EU AI Act applicability and classification are deployment-dependent; we do not self-classify as “high-risk AI” in documentation. Optional modules may support Articles 9–17 as a conservative engineering posture.

---

## 3. TARGET ARCHITECTURE

### 3.1 Directory Structure

```
services/ai_act/
├── __init__.py                    # Facade with lazy loading + TYPE_CHECKING
├── __init__.pyi                   # Stub file for IDE/mypy
├── config.py                      # Thread-safe config via contextvars + ENV override
├── exceptions.py                  # NEW: Custom exceptions (EnterpriseNotAvailableError)
├── _compat.py                     # Backward compatibility with full aliases
├── _version.py                    # Version and deprecation constants
├── _cache.py                      # NEW: Module cache with invalidation
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
    ├── __init__.py                # Lazy loading via __getattr__ + config check
    ├── __init__.pyi               # Stub file
    ├── _guard.py                  # NEW: Enterprise access guard (import-time check)
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

### 3.2 Custom Exceptions

```python
# services/ai_act/exceptions.py
"""
Custom exceptions for AI Act compliance module.
"""
from __future__ import annotations


class AIActComplianceError(Exception):
    """Base exception for AI Act compliance errors."""
    pass


class EnterpriseNotAvailableError(AIActComplianceError):
    """
    Raised when enterprise module is accessed without proper license/config.

    This is a BLOCKING error, not a warning. Enterprise modules are
    optional features for B2B clients and require explicit enablement.
    """

    def __init__(self, module_name: str, current_level: str):
        self.module_name = module_name
        self.current_level = current_level
        super().__init__(
            f"Enterprise module '{module_name}' is not available. "
            f"Current compliance level: {current_level}. "
            f"To use enterprise features, set AIACT_COMPLIANCE_LEVEL=enterprise "
            f"or configure AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE). "
            f"Enterprise modules are optional High-Risk compliance features for B2B clients."
        )


class ConfigurationError(AIActComplianceError):
    """Raised when configuration is invalid."""
    pass


class ModuleLoadError(AIActComplianceError):
    """Raised when enterprise module fails to load."""

    def __init__(self, module_name: str, original_error: Exception):
        self.module_name = module_name
        self.original_error = original_error
        super().__init__(
            f"Failed to load enterprise module '{module_name}': {original_error}. "
            f"This may indicate a missing dependency or code error."
        )
```

### 3.3 Module Cache with Invalidation

```python
# services/ai_act/_cache.py
"""
Thread-safe module cache with invalidation support.
"""
from __future__ import annotations
import threading
from typing import Any, Optional
import importlib


class ModuleCache:
    """
    Thread-safe cache for lazily loaded enterprise modules.

    Supports invalidation for testing and hot-reload scenarios.
    """

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, module_name: str) -> Optional[Any]:
        """Get cached module or None."""
        with self._lock:
            return self._cache.get(module_name)

    def set(self, module_name: str, module: Any) -> None:
        """Cache a loaded module."""
        with self._lock:
            self._cache[module_name] = module

    def load_or_get(self, full_module_path: str) -> Any:
        """
        Load module if not cached, otherwise return cached.

        Args:
            full_module_path: e.g., "services.ai_act.enterprise.risk_management"

        Returns:
            Loaded module object
        """
        module_name = full_module_path.split(".")[-1]

        with self._lock:
            if module_name not in self._cache:
                self._cache[module_name] = importlib.import_module(full_module_path)
            return self._cache[module_name]

    def invalidate(self, module_name: Optional[str] = None) -> None:
        """
        Invalidate cached module(s).

        Args:
            module_name: Specific module to invalidate, or None for all
        """
        with self._lock:
            if module_name is None:
                self._cache.clear()
            elif module_name in self._cache:
                del self._cache[module_name]

    def is_loaded(self, module_name: str) -> bool:
        """Check if module is in cache."""
        with self._lock:
            return module_name in self._cache

    @property
    def loaded_modules(self) -> list[str]:
        """List of currently loaded module names."""
        with self._lock:
            return list(self._cache.keys())


# Global cache instance
_enterprise_cache = ModuleCache()


def get_enterprise_cache() -> ModuleCache:
    """Get the global enterprise module cache."""
    return _enterprise_cache


def invalidate_enterprise_cache() -> None:
    """Invalidate all cached enterprise modules (for testing)."""
    _enterprise_cache.invalidate()
```

### 3.4 Configuration Model (Thread-Safe with ENV Override)

```python
# services/ai_act/config.py
"""
Thread-safe AI Act compliance configuration using contextvars.

Features:
    - Thread/async-safe via contextvars (PEP 567)
    - Environment variable override (AIACT_COMPLIANCE_LEVEL)
    - Serialization support (to_dict/from_dict) for persistence
    - Validation with normalization

References:
    - PEP 567: Context Variables
    - https://docs.python.org/3/library/contextvars.html
"""
from __future__ import annotations
import contextvars
import os
from enum import Enum
from typing import Set, Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator

from services.ai_act.exceptions import ConfigurationError
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

    # =========================================================================
    # Serialization Methods
    # =========================================================================
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize config to dictionary for persistence (YAML, JSON, DB).

        Returns:
            Dictionary with all config values
        """
        return {
            "level": self.level.value,
            "enable_risk_management": self.enable_risk_management,
            "enable_human_oversight": self.enable_human_oversight,
            "enable_accuracy_metrics": self.enable_accuracy_metrics,
            "enable_robustness_testing": self.enable_robustness_testing,
            "enable_explainability": self.enable_explainability,
            "enable_data_governance": self.enable_data_governance,
            "enable_logging_system": self.enable_logging_system,
            "enable_qms": self.enable_qms,
            "enable_testing_framework": self.enable_testing_framework,
            "enable_cybersecurity": self.enable_cybersecurity,
            "enable_post_market": self.enable_post_market,
            "enable_conformity": self.enable_conformity,
            "log_retention_months": self.log_retention_months,
            "audit_trail_enabled": self.audit_trail_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIActComplianceConfig":
        """
        Deserialize config from dictionary.

        Args:
            data: Dictionary with config values

        Returns:
            AIActComplianceConfig instance

        Raises:
            ConfigurationError: If data is invalid
        """
        try:
            return cls(**data)
        except Exception as e:
            raise ConfigurationError(f"Invalid config data: {e}") from e

    @classmethod
    def from_yaml_section(cls, yaml_data: Dict[str, Any]) -> "AIActComplianceConfig":
        """
        Create config from YAML section (e.g., from CommonRunConfig).

        Handles nested structure like:
            ai_act:
              level: enterprise
              modules:
                risk_management: true
                human_oversight: false
        """
        if not yaml_data:
            return cls()

        # Flatten nested modules if present
        flat_data = {"level": yaml_data.get("level", "gpai")}
        modules = yaml_data.get("modules", {})

        for key, value in modules.items():
            flat_key = f"enable_{key}"
            if flat_key in cls.model_fields:
                flat_data[flat_key] = value

        # Copy non-nested fields
        for key in ["log_retention_months", "audit_trail_enabled"]:
            if key in yaml_data:
                flat_data[key] = yaml_data[key]

        return cls.from_dict(flat_data)
# =============================================================================
# Thread-Safe Context Variable
# =============================================================================
_ai_act_config_var: contextvars.ContextVar[AIActComplianceConfig] = contextvars.ContextVar(
    "ai_act_config",
    default=AIActComplianceConfig()
)
# =============================================================================
# Environment Variable Override
# =============================================================================
_ENV_VAR_NAME = "AIACT_COMPLIANCE_LEVEL"
_ENV_LEVEL_MAPPING = {
    "gpai": AIActComplianceLevel.GPAI_ONLY,
    "gpai_only": AIActComplianceLevel.GPAI_ONLY,
    "enterprise": AIActComplianceLevel.ENTERPRISE,
    "custom": AIActComplianceLevel.CUSTOM,
}


def _get_env_override() -> Optional[AIActComplianceLevel]:
    """
    Get compliance level override from environment variable.

    Returns:
        AIActComplianceLevel if valid env var set, None otherwise
    """
    env_value = os.environ.get(_ENV_VAR_NAME, "").lower().strip()
    if not env_value:
        return None
    if env_value not in _ENV_LEVEL_MAPPING:
        import warnings
        warnings.warn(
            f"Invalid {_ENV_VAR_NAME}='{env_value}'. "
            f"Valid values: {list(_ENV_LEVEL_MAPPING.keys())}. "
            f"Ignoring environment override.",
            UserWarning,
            stacklevel=3,
        )
        return None
    return _ENV_LEVEL_MAPPING[env_value]


def get_ai_act_config() -> AIActComplianceConfig:
    """
    Get current AI Act config for this context.

    Resolution order:
        1. Environment variable (AIACT_COMPLIANCE_LEVEL) - highest priority
        2. Context variable (set via set_ai_act_config or context manager)
        3. Default config (GPAI_ONLY)

    Thread-safe: each thread/async context has its own config.
    """
    config = _ai_act_config_var.get()

    # Check for environment override
    env_level = _get_env_override()
    if env_level is not None and env_level != config.level:
        # Create new config with overridden level
        # Note: This doesn't modify the context var, just returns override
        return config.model_copy(update={"level": env_level})

    return config
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

### 3.5 Enterprise Access Guard

```python
# services/ai_act/enterprise/_guard.py
"""
Enterprise module access guard.

This module is imported at the top of every enterprise module to enforce
access control. It raises EnterpriseNotAvailableError if the current
config level is GPAI_ONLY.

Usage in enterprise modules:
    from services.ai_act.enterprise._guard import require_enterprise_access
    require_enterprise_access(__name__)  # Call at module level

Design Decision:
    We use RuntimeError (via EnterpriseNotAvailableError) instead of
    DeprecationWarning because enterprise modules are a B2B feature
    that requires explicit enablement. Silent warnings could lead to
    unintended usage and compliance issues.
"""
from __future__ import annotations
import os
from typing import Optional

# Avoid circular import - only import what we need
from services.ai_act.exceptions import EnterpriseNotAvailableError


def require_enterprise_access(module_name: str) -> None:
    """
    Enforce enterprise access at module import time.

    Args:
        module_name: The __name__ of the calling module

    Raises:
        EnterpriseNotAvailableError: If enterprise features are not enabled

    Note:
        This check happens at IMPORT TIME, not at runtime.
        Once a module is imported, subsequent usage doesn't re-check.
        This is intentional for performance.
    """
    # Defer import to avoid circular dependency
    from services.ai_act.config import get_ai_act_config, AIActComplianceLevel

    config = get_ai_act_config()

    if config.level == AIActComplianceLevel.GPAI_ONLY:
        # Extract short module name for cleaner error message
        short_name = module_name.split(".")[-1]
        raise EnterpriseNotAvailableError(
            module_name=short_name,
            current_level=config.level.value,
        )


def is_enterprise_enabled_for_module(module_name: str) -> bool:
    """
    Check if enterprise access is enabled for a specific module.

    Useful for conditional logic without raising exceptions.

    Args:
        module_name: Module name to check (e.g., "risk_management")

    Returns:
        True if module is enabled, False otherwise
    """
    from services.ai_act.config import get_ai_act_config

    config = get_ai_act_config()
    return config.is_module_enabled(module_name)


# =============================================================================
# Testing Support
# =============================================================================
_GUARD_DISABLED_FOR_TESTING: bool = False


def disable_guard_for_testing() -> None:
    """
    Disable the enterprise guard for testing purposes.

    WARNING: Only use in test fixtures, never in production code.
    """
    global _GUARD_DISABLED_FOR_TESTING
    _GUARD_DISABLED_FOR_TESTING = True


def enable_guard_after_testing() -> None:
    """Re-enable the enterprise guard after testing."""
    global _GUARD_DISABLED_FOR_TESTING
    _GUARD_DISABLED_FOR_TESTING = False


def _is_guard_active() -> bool:
    """Check if guard is currently active (not disabled for testing)."""
    return not _GUARD_DISABLED_FOR_TESTING
```

### 3.6 Enterprise Package with Lazy Loading and Config Check

```python
# services/ai_act/enterprise/__init__.py
"""
EU AI Act Enterprise Compliance Modules (High-Risk).

IMPORTANT: This package is OPTIONAL and requires explicit enablement.
Attempting to import from this package in GPAI_ONLY mode will raise
EnterpriseNotAvailableError.

Implements lazy loading to avoid importing all 15 modules at once.

Legal Basis:
    - Articles 9, 14, 15, 17, 43, 72 apply to AI systems that are classified as High-Risk (Annex III)
    - EU AI Act applicability and risk classification are deployment-dependent; do not assume out-of-scope without legal review
    - These modules provide OPTIONAL evidence/controls for B2B clients (when applicable)

Usage:
    # First, enable enterprise mode:
    export AIACT_COMPLIANCE_LEVEL=enterprise

    # Or programmatically:
    from services.ai_act.config import ai_act_config_context, AIActComplianceConfig, AIActComplianceLevel
    with ai_act_config_context(AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE)):
        from services.ai_act.enterprise import AIActRiskManager

    # Direct import ONLY works if enterprise is enabled:
    from services.ai_act.enterprise import AIActRiskManager  # Raises if GPAI_ONLY
"""
from __future__ import annotations
from typing import Any, List, TYPE_CHECKING
import importlib

from services.ai_act.exceptions import EnterpriseNotAvailableError, ModuleLoadError
from services.ai_act._cache import get_enterprise_cache
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
# Use thread-safe cache from _cache.py instead of module-level dict
# _loaded_modules is replaced by get_enterprise_cache()


def _check_enterprise_access(name: str) -> None:
    """
    Check if enterprise access is allowed before loading module.

    Raises:
        EnterpriseNotAvailableError: If in GPAI_ONLY mode
    """
    from services.ai_act.config import get_ai_act_config, AIActComplianceLevel
    from services.ai_act.enterprise._guard import _is_guard_active

    # Skip check if guard is disabled for testing
    if not _is_guard_active():
        return

    config = get_ai_act_config()

    if config.level == AIActComplianceLevel.GPAI_ONLY:
        raise EnterpriseNotAvailableError(
            module_name=name,
            current_level=config.level.value,
        )


def __getattr__(name: str) -> Any:
    """
    Lazy load enterprise modules on first access.

    IMPORTANT: This function enforces enterprise access control.
    Attempting to access any attribute in GPAI_ONLY mode will raise
    EnterpriseNotAvailableError.

    Only imports the specific module needed, not all 15.

    Args:
        name: Attribute name (e.g., "AIActRiskManager")

    Returns:
        The requested attribute from the appropriate module

    Raises:
        EnterpriseNotAvailableError: If enterprise features are not enabled
        AttributeError: If name is not a valid enterprise export
        ModuleLoadError: If module fails to load (wraps ImportError)
    """
    if name not in _NAME_TO_MODULE:
        raise AttributeError(f"module 'services.ai_act.enterprise' has no attribute '{name}'")

    # CRITICAL: Check enterprise access BEFORE loading module
    _check_enterprise_access(name)

    module_name = _NAME_TO_MODULE[name]
    cache = get_enterprise_cache()

    # Try to load from cache or import
    try:
        full_path = f"services.ai_act.enterprise.{module_name}"
        module = cache.load_or_get(full_path)
        return getattr(module, name)
    except ImportError as e:
        raise ModuleLoadError(module_name=module_name, original_error=e) from e
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
EU AI Act Alignment/Evidence Module - Tiered Architecture.
Classification:
    - Deployment-dependent applicability (counsel review required)
    - Optional modules may cover Articles 9–17, 43, 72 as a conservative engineering posture
Legal Reference: EU AI Act Regulation 2024/1689
    - Avoid self-classification in docs; validate applicability for concrete deployments
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
    """Generate Annex XI-aligned documentation for GPAI models."""
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
from typing import TYPE_CHECKING, Optional

# CRITICAL: Use TYPE_CHECKING to avoid circular imports
# Runtime imports are deferred to methods that need them
if TYPE_CHECKING:
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
    """Generate Annex IV-aligned documentation for High-Risk AI systems."""

    def __init__(
        self,
        risk_manager: Optional["AIActRiskManager"] = None,
        oversight_system: Optional["HumanOversightSystem"] = None,
        data_framework: Optional["DataGovernanceFramework"] = None,
    ):
        """
        Initialize with optional enterprise components.

        Args:
            risk_manager: Risk management system (Art. 9)
            oversight_system: Human oversight system (Art. 14)
            data_framework: Data governance framework (Art. 10)

        Note:
            Dependencies are injected, not imported directly.
            This avoids circular imports and allows for testing.
        """
        self._risk_manager = risk_manager
        self._oversight_system = oversight_system
        self._data_framework = data_framework

    def _get_risk_manager(self) -> "AIActRiskManager":
        """Lazy import of risk_management if not injected."""
        if self._risk_manager is None:
            from .risk_management import create_risk_manager
            self._risk_manager = create_risk_manager()
        return self._risk_manager

    # ... rest of implementation
```

---

## 5. TEST FIXTURES (Thread-Safe with Enterprise Guard Support)

```python
# tests/conftest_ai_act.py
"""
AI Act test fixtures with proper isolation.

Features:
    - Thread-safe config via contextvars
    - Enterprise guard control for testing
    - Module cache invalidation between tests
    - pytest-xdist compatibility
"""
import pytest
import os
from contextvars import copy_context
from typing import Generator, Callable

from services.ai_act.config import (
    AIActComplianceConfig,
    AIActComplianceLevel,
    set_ai_act_config,
    reset_ai_act_config,
    ai_act_config_context,
)
from services.ai_act._cache import invalidate_enterprise_cache
from services.ai_act.enterprise._guard import (
    disable_guard_for_testing,
    enable_guard_after_testing,
)


# =============================================================================
# Core Fixtures (autouse)
# =============================================================================


@pytest.fixture(autouse=True)
def reset_ai_act_state_after_test() -> Generator[None, None, None]:
    """
    Reset ALL AI Act state after each test.

    This fixture:
    1. Sets default config
    2. Clears environment variable override
    3. Invalidates module cache
    4. Re-enables enterprise guard (in case test disabled it)

    Uses contextvars token for proper cleanup.
    """
    # Clear any environment override
    env_backup = os.environ.pop("AIACT_COMPLIANCE_LEVEL", None)

    # Store default config
    default_config = AIActComplianceConfig()
    token = set_ai_act_config(default_config)

    yield

    # Cleanup in reverse order
    reset_ai_act_config(token)
    invalidate_enterprise_cache()  # Clear cached modules
    enable_guard_after_testing()   # Ensure guard is active

    # Restore environment if it was set
    if env_backup is not None:
        os.environ["AIACT_COMPLIANCE_LEVEL"] = env_backup


# =============================================================================
# Config Level Fixtures
# =============================================================================


@pytest.fixture
def ai_act_gpai_config() -> Generator[AIActComplianceConfig, None, None]:
    """GPAI-only config fixture (default mode)."""
    config = AIActComplianceConfig(level=AIActComplianceLevel.GPAI_ONLY)
    with ai_act_config_context(config):
        yield config


@pytest.fixture
def ai_act_enterprise_config() -> Generator[AIActComplianceConfig, None, None]:
    """
    Enterprise config fixture.

    Enables all enterprise modules. Use this for testing enterprise functionality.
    """
    config = AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE)
    with ai_act_config_context(config):
        yield config


@pytest.fixture
def ai_act_custom_config() -> Callable[..., "ai_act_config_context"]:
    """
    Custom config fixture factory.

    Usage:
        def test_custom_modules(ai_act_custom_config):
            with ai_act_custom_config(enable_risk_management=True):
                from services.ai_act.enterprise import AIActRiskManager
                # Test with only risk_management enabled
    """
    def _make_config(**kwargs) -> ai_act_config_context:
        config = AIActComplianceConfig(
            level=AIActComplianceLevel.CUSTOM,
            **kwargs
        )
        return ai_act_config_context(config)
    return _make_config


# =============================================================================
# Enterprise Guard Control Fixtures
# =============================================================================


@pytest.fixture
def disable_enterprise_guard() -> Generator[None, None, None]:
    """
    Temporarily disable enterprise guard for testing.

    WARNING: Use only when testing enterprise modules without proper config.
    The guard will be re-enabled after the test.

    Usage:
        def test_enterprise_internals(disable_enterprise_guard):
            # Guard disabled - can import without EnterpriseNotAvailableError
            from services.ai_act.enterprise.risk_management import AIActRiskManager
    """
    disable_guard_for_testing()
    yield
    enable_guard_after_testing()


@pytest.fixture
def with_enterprise_env() -> Generator[None, None, None]:
    """
    Set AIACT_COMPLIANCE_LEVEL=enterprise via environment.

    Useful for testing environment variable override.
    """
    old_value = os.environ.get("AIACT_COMPLIANCE_LEVEL")
    os.environ["AIACT_COMPLIANCE_LEVEL"] = "enterprise"
    yield
    if old_value is None:
        os.environ.pop("AIACT_COMPLIANCE_LEVEL", None)
    else:
        os.environ["AIACT_COMPLIANCE_LEVEL"] = old_value


# =============================================================================
# Test Isolation for Parallel Execution (pytest-xdist)
# =============================================================================


@pytest.fixture(scope="function")
def isolated_context() -> Callable:
    """
    Run test in isolated context for pytest-xdist compatibility.

    Each test gets its own contextvars context, preventing
    cross-test contamination in parallel execution.

    Usage:
        def test_parallel_safe(isolated_context):
            def inner_test():
                config = get_ai_act_config()
                assert config.level == AIActComplianceLevel.GPAI_ONLY
            isolated_context(inner_test)
    """
    ctx = copy_context()

    def run_in_context(func: Callable, *args, **kwargs):
        return ctx.run(func, *args, **kwargs)

    return run_in_context


# =============================================================================
# Exception Testing Fixtures
# =============================================================================


@pytest.fixture
def expect_enterprise_error():
    """
    Context manager for testing EnterpriseNotAvailableError.

    Usage:
        def test_gpai_blocks_enterprise(expect_enterprise_error):
            with expect_enterprise_error("risk_management"):
                from services.ai_act.enterprise import AIActRiskManager
    """
    from contextlib import contextmanager
    from services.ai_act.exceptions import EnterpriseNotAvailableError

    @contextmanager
    def _expect_error(module_name: str):
        import pytest
        with pytest.raises(EnterpriseNotAvailableError) as exc_info:
            yield
        assert exc_info.value.module_name == module_name
        assert exc_info.value.current_level == "gpai"

    return _expect_error


# =============================================================================
# Cleanup Utilities
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "enterprise: mark test as requiring enterprise config"
    )
    config.addinivalue_line(
        "markers",
        "gpai_only: mark test as testing GPAI-only behavior"
    )
```

---

## 6. INTERNAL IMPORT UPDATES (TYPE_CHECKING Pattern)

### 6.1 Pattern: TYPE_CHECKING for Cross-Module Dependencies

All enterprise modules that import from other enterprise modules MUST use the TYPE_CHECKING pattern to prevent circular imports:

```python
# Pattern for enterprise modules with cross-dependencies
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

# Runtime-safe imports (no circular dependencies)
from .shared_types import SomeEnum, SomeDataclass

# TYPE_CHECKING block for type hints only (not executed at runtime)
if TYPE_CHECKING:
    from .risk_management import AIActRiskManager
    from .human_oversight import HumanOversightSystem

class MyClass:
    def __init__(self, risk_manager: Optional["AIActRiskManager"] = None):
        # String annotation allows forward reference
        self._risk_manager = risk_manager

    def _get_risk_manager(self) -> "AIActRiskManager":
        # Lazy import at runtime when actually needed
        if self._risk_manager is None:
            from .risk_management import create_risk_manager
            self._risk_manager = create_risk_manager()
        return self._risk_manager
```

### 6.2 enterprise/risk_registry.py

```python
"""Risk Registry - depends on risk_management types."""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from enum import Enum
from dataclasses import dataclass

# TYPE_CHECKING for type hints
if TYPE_CHECKING:
    from .risk_management import (
        AIActRiskCategory,
        AIActRiskSeverity,
        AIActRiskLikelihood,
    )

# Runtime imports - only what's needed for actual execution
# Defer heavy imports to methods


class RiskStatus(Enum):
    """Risk lifecycle status."""
    IDENTIFIED = "identified"
    ASSESSED = "assessed"
    MITIGATED = "mitigated"
    ACCEPTED = "accepted"
    CLOSED = "closed"


@dataclass
class RiskEntry:
    """Single risk entry in registry."""
    risk_id: str
    category: "AIActRiskCategory"  # Forward reference (string)
    severity: "AIActRiskSeverity"
    likelihood: "AIActRiskLikelihood"
    status: RiskStatus
    # ...


class RiskRegistry:
    """Registry for tracking risks."""

    def add_risk(
        self,
        category: "AIActRiskCategory",
        severity: "AIActRiskSeverity",
        likelihood: "AIActRiskLikelihood",
    ) -> RiskEntry:
        # Runtime import when method is called
        from .risk_management import AIActRiskCategory  # noqa: F811
        # Validation uses runtime import
        if not isinstance(category, AIActRiskCategory):
            raise TypeError(f"Expected AIActRiskCategory, got {type(category)}")
        # ...
```

### 6.3 All Enterprise Modules with Cross-Dependencies

| Module | Dependencies | Pattern |
|--------|-------------|---------|
| `risk_registry.py` | risk_management | TYPE_CHECKING + lazy import |
| `conformity_assessment.py` | risk_management, human_oversight, data_governance, logging_system | TYPE_CHECKING + DI |
| `qms.py` | logging_system | TYPE_CHECKING + lazy import |
| `post_market_monitoring.py` | logging_system | TYPE_CHECKING + lazy import |
| `testing_framework.py` | accuracy_metrics | TYPE_CHECKING + lazy import |
| `high_risk_technical_docs.py` | risk_management, human_oversight, data_governance | TYPE_CHECKING + DI |

**DI = Dependency Injection**: Dependencies passed to `__init__`, not imported at module level.

### 6.4 Refactored conformity_assessment.py Example

```python
"""Conformity Assessment - heavy cross-dependencies, uses DI pattern."""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Protocol
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from .risk_management import AIActRiskManager
    from .human_oversight import HumanOversightSystem
    from .data_governance import DataGovernanceFramework
    from .logging_system import AIActLogger


# Protocol for dependency injection (allows duck typing)
class RiskManagerProtocol(Protocol):
    """Protocol for risk manager dependency."""
    def get_risk_summary(self) -> dict: ...


class ConformitySelfAssessment:
    """
    Self-assessment framework for High-Risk AI conformity.

    Uses Dependency Injection to avoid circular imports and
    enable easier testing with mocks.
    """

    def __init__(
        self,
        risk_manager: Optional["AIActRiskManager"] = None,
        oversight: Optional["HumanOversightSystem"] = None,
        data_gov: Optional["DataGovernanceFramework"] = None,
        logger: Optional["AIActLogger"] = None,
    ):
        """
        Initialize with optional dependencies.

        If not provided, dependencies are lazily created when needed.
        This allows:
        1. Testing with mocks
        2. Avoiding circular imports
        3. Delayed initialization (performance)
        """
        self._risk_manager = risk_manager
        self._oversight = oversight
        self._data_gov = data_gov
        self._logger = logger

    @property
    def risk_manager(self) -> "AIActRiskManager":
        """Lazy-load risk manager."""
        if self._risk_manager is None:
            from .risk_management import create_risk_manager
            self._risk_manager = create_risk_manager()
        return self._risk_manager

    @property
    def oversight(self) -> "HumanOversightSystem":
        """Lazy-load oversight system."""
        if self._oversight is None:
            from .human_oversight import create_human_oversight_system
            self._oversight = create_human_oversight_system()
        return self._oversight

    # ... rest of implementation
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

## 8. MIGRATION PHASES (0-9)

| Phase | Name | Description |
|-------|------|-------------|
| **0** | Pre-Migration | Branch, backup, baseline tests |
| **1** | Core Infrastructure | `_version.py`, `exceptions.py`, `_cache.py`, `config.py`, `_compat.py` |
| **2** | Create Core Package | `core/` directory with GPAI modules |
| **3** | Create Enterprise Package | `enterprise/` directory with High-Risk modules |
| **4** | Update Internal Imports | TYPE_CHECKING pattern + Dependency Injection |
| **5** | Update Facade | `__init__.py` with enterprise guard |
| **6** | Update Tests | Fixtures, markers, isolation |
| **7** | Integration | `CommonRunConfig` integration |
| **8** | Documentation | Docs and migration guide |
| **9** | Post-Migration Validation | Full verification and PR |

---

### Phase 0: Pre-Migration

**Goal:** Prepare environment, create safety net

- [ ] Create feature branch: `git checkout -b feature/ai-act-tiered-architecture`
- [ ] Create backup tag: `git tag backup/ai-act-before-tiered-$(date +%Y%m%d)`
- [ ] Run baseline tests: `pytest tests/test_ai_act_*.py -v > baseline_test_results.txt`
- [ ] Document current import patterns: `grep -r "from services.ai_act import" . > current_imports.txt`
- [ ] Count baseline LOC: `wc -l services/ai_act/*.py`

**Exit Criteria:**

- [ ] Feature branch exists
- [ ] Backup tag created
- [ ] Baseline test results saved
- [ ] Current imports documented

---

### Phase 1: Core Infrastructure

**Goal:** Create foundational files (NEW FILES ONLY, no moves)

- [ ] Create `services/ai_act/_version.py`
- [ ] Create `services/ai_act/exceptions.py` with `EnterpriseNotAvailableError`, `ModuleLoadError`, `ConfigurationError`
- [ ] Create `services/ai_act/_cache.py` with `ModuleCache` (double-checked locking)
- [ ] Create `services/ai_act/config.py` with `contextvars` + ENV override
- [ ] Create `services/ai_act/_compat.py` with full module mapping

**Verification:**

```bash
python -c "from services.ai_act.config import ai_act_config_context; print('OK: config')"
python -c "from services.ai_act.exceptions import EnterpriseNotAvailableError; print('OK: exceptions')"
python -c "from services.ai_act._cache import ModuleCache; print('OK: cache')"
```

**Commit:** `git commit -m "feat(ai-act): add core infrastructure files"`

**Exit Criteria:**

- [ ] All 5 files created
- [ ] All verification commands pass
- [ ] Commit created

---

### Phase 2: Create Core Package

**Goal:** Move GPAI modules to `core/` subdirectory

- [ ] `mkdir -p services/ai_act/core`
- [ ] Create `services/ai_act/core/__init__.py`
- [ ] Create `services/ai_act/core/__init__.pyi`
- [ ] `git mv services/ai_act/transparency_disclosure.py services/ai_act/core/`
- [ ] `git mv services/ai_act/gpai_model_card.py services/ai_act/core/`
- [ ] `git mv services/ai_act/copyright_compliance.py services/ai_act/core/`
- [ ] `git mv services/ai_act/training_data_summary.py services/ai_act/core/`
- [ ] `git mv services/ai_act/user_acknowledgment.py services/ai_act/core/`
- [ ] Create `services/ai_act/core/gpai_technical_docs.py` (extract from `technical_documentation.py`)

**Verification:**

```bash
python -c "from services.ai_act.core import TransparencyDisclosureManager; print('OK: core')"
python -c "from services.ai_act.core import GPAIDocumentationGenerator; print('OK: gpai_docs')"
```

**Commit:** `git commit -m "feat(ai-act): create core package with GPAI modules"`

**Exit Criteria:**

- [ ] 6 modules in `core/`
- [ ] `__init__.py` exports all GPAI classes
- [ ] Verification commands pass

---

### Phase 3: Create Enterprise Package

**Goal:** Move High-Risk modules to `enterprise/` subdirectory with access guard

- [ ] `mkdir -p services/ai_act/enterprise`
- [ ] Create `services/ai_act/enterprise/_guard.py` with `require_enterprise_access()`
- [ ] Create `services/ai_act/enterprise/__init__.py` with lazy loading + config check
- [ ] Create `services/ai_act/enterprise/__init__.pyi`
- [ ] Move 14 enterprise modules:
  - [ ] `git mv services/ai_act/risk_management.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/risk_registry.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/human_oversight.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/accuracy_metrics.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/robustness_testing.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/explainability.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/data_governance.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/data_lineage.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/logging_system.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/qms.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/testing_framework.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/cybersecurity.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/post_market_monitoring.py services/ai_act/enterprise/`
  - [ ] `git mv services/ai_act/conformity_assessment.py services/ai_act/enterprise/`
- [ ] Create `services/ai_act/enterprise/high_risk_technical_docs.py` (extract from `technical_documentation.py`)
- [ ] Delete `services/ai_act/technical_documentation.py`
- [ ] Add `require_enterprise_access(__name__)` to each enterprise module

**Verification:**

```bash
# Should FAIL in GPAI mode (default)
python -c "
try:
    from services.ai_act.enterprise import AIActRiskManager
    print('ERROR: Should have raised EnterpriseNotAvailableError')
    exit(1)
except Exception as e:
    print(f'OK: {type(e).__name__}')
"

# Should WORK with ENV override
AIACT_COMPLIANCE_LEVEL=enterprise python -c "
from services.ai_act.enterprise import AIActRiskManager
print('OK: Enterprise loaded with ENV')
"
```

**Commit:** `git commit -m "feat(ai-act): create enterprise package with access guard"`

**Exit Criteria:**

- [ ] 15 modules in `enterprise/`
- [ ] Guard blocks access in GPAI mode
- [ ] ENV override enables access
- [ ] `technical_documentation.py` deleted

---

### Phase 4: Update Internal Imports

**Goal:** Fix cross-module dependencies using TYPE_CHECKING pattern

Modules with cross-dependencies:

| Module | Dependencies | Pattern |
|--------|-------------|---------|
| `risk_registry.py` | `risk_management` | TYPE_CHECKING + lazy |
| `conformity_assessment.py` | `risk_management`, `human_oversight`, `data_governance`, `logging_system` | TYPE_CHECKING + DI |
| `qms.py` | `logging_system` | TYPE_CHECKING + lazy |
| `post_market_monitoring.py` | `logging_system` | TYPE_CHECKING + lazy |
| `testing_framework.py` | `accuracy_metrics` | TYPE_CHECKING + lazy |
| `high_risk_technical_docs.py` | `risk_management`, `human_oversight`, `data_governance` | TYPE_CHECKING + DI |

- [ ] Update `enterprise/risk_registry.py`
- [ ] Update `enterprise/conformity_assessment.py`
- [ ] Update `enterprise/qms.py`
- [ ] Update `enterprise/post_market_monitoring.py`
- [ ] Update `enterprise/testing_framework.py`
- [ ] Update `enterprise/high_risk_technical_docs.py`

**Verification:**

```bash
python -m py_compile services/ai_act/enterprise/*.py
AIACT_COMPLIANCE_LEVEL=enterprise python -c "from services.ai_act.enterprise import *; print('OK: no circular imports')"
```

**Commit:** `git commit -m "refactor(ai-act): update imports to TYPE_CHECKING pattern"`

**Exit Criteria:**

- [ ] All 6 modules updated
- [ ] No circular import errors
- [ ] py_compile passes

---

### Phase 5: Update Facade

**Goal:** Rewrite main `__init__.py` with tiered architecture

- [ ] Rewrite `services/ai_act/__init__.py`:
  - Import config exports
  - Import core exports (always available)
  - `__getattr__` for enterprise lazy loading with guard
  - `__dir__` for IDE autocompletion
  - Explicit `__all__`
- [ ] Create `services/ai_act/__init__.pyi` for type hints

**Verification:**

```bash
# GPAI imports work
python -c "from services.ai_act import TransparencyDisclosureManager; print('OK: GPAI')"

# Enterprise blocked in GPAI mode
python -c "
from services.ai_act.config import get_ai_act_config
print(f'Level: {get_ai_act_config().level}')
try:
    from services.ai_act.enterprise import AIActRiskManager
    print('ERROR')
    exit(1)
except Exception as e:
    print(f'OK: {type(e).__name__}')
"

# Enterprise works with ENV
AIACT_COMPLIANCE_LEVEL=enterprise python -c "
from services.ai_act.enterprise import AIActRiskManager
print('OK: Enterprise via ENV')
"

# Legacy import emits warning
python -W error::DeprecationWarning -c "
from services.ai_act.config import set_ai_act_config, AIActComplianceConfig, AIActComplianceLevel
set_ai_act_config(AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE))
from services.ai_act import AIActRiskManager
print('OK: Legacy import works in ENTERPRISE mode')
"
```

**Commit:** `git commit -m "feat(ai-act): add facade with enterprise guard"`

**Exit Criteria:**

- [ ] Facade imports GPAI without config
- [ ] Facade blocks enterprise in GPAI mode
- [ ] ENV override works
- [ ] Legacy imports work with deprecation warning

---

### Phase 6: Update Tests

**Goal:** Create test fixtures for proper isolation

- [ ] Create `tests/conftest_ai_act.py` with:
  - `reset_ai_act_state_after_test()` (autouse)
  - `ai_act_gpai_config` fixture
  - `ai_act_enterprise_config` fixture
  - `ai_act_custom_config` factory fixture
  - `disable_enterprise_guard` fixture
  - `with_enterprise_env` fixture
  - `expect_enterprise_error` fixture
  - `isolated_context` fixture (pytest-xdist)
- [ ] Add to `conftest.py`: `pytest_plugins = ["tests.conftest_ai_act"]`
- [ ] Register markers: `@pytest.mark.enterprise`, `@pytest.mark.gpai_only`
- [ ] Update test imports to explicit paths
- [ ] Add tests for:
  - [ ] `EnterpriseNotAvailableError` behavior
  - [ ] ENV override priority
  - [ ] Cache invalidation
  - [ ] Thread safety (contextvars)

**Verification:**

```bash
pytest tests/test_ai_act_*.py -v --tb=short
pytest tests/test_ai_act_*.py -v -n auto  # Parallel execution
```

**Commit:** `git commit -m "test(ai-act): update fixtures with enterprise guard support"`

**Exit Criteria:**

- [ ] All fixtures created
- [ ] Tests pass sequentially
- [ ] Tests pass in parallel (pytest-xdist)
- [ ] No state leaks between tests

---

### Phase 7: Integration

**Goal:** Integrate AI Act config with main application config

- [ ] Add `ai_act: Optional[AIActComplianceConfig]` field to `CommonRunConfig` in `core_config.py`
- [ ] Implement `from_yaml_section()` support
- [ ] Update YAML config examples:

```yaml
ai_act:
  level: enterprise  # or gpai, custom
  modules:
    risk_management: true
    human_oversight: true
```

- [ ] Add config validation on startup

**Verification:**

```bash
python -c "
from core_config import CommonRunConfig
config = CommonRunConfig()
print(f'AI Act config: {config.ai_act}')
"
```

**Commit:** `git commit -m "feat(config): integrate AI Act config with CommonRunConfig"`

**Exit Criteria:**

- [ ] `CommonRunConfig.ai_act` field exists
- [ ] YAML parsing works
- [ ] Default is GPAI_ONLY

---

### Phase 8: Documentation

**Goal:** Update all documentation

- [ ] Update `docs/EU_AI_ACT_INTEGRATION_PLAN.md`
- [ ] Update module docstrings in `core/` and `enterprise/`
- [ ] Create `docs/MIGRATION_GUIDE.md` with:
  - Before/after import examples
  - ENV variable usage
  - Config context manager usage
  - B2B licensing guidance
- [ ] Update `README.md` if needed

**Commit:** `git commit -m "docs(ai-act): update documentation for tiered architecture"`

**Exit Criteria:**

- [ ] Integration plan updated
- [ ] Migration guide created
- [ ] All docstrings accurate

---

### Phase 9: Post-Migration Validation

**Goal:** Full verification before merge

**Test Suite:**

- [ ] `pytest tests/test_ai_act_*.py -v` (all tests pass)
- [ ] `pytest tests/test_ai_act_*.py -v -n auto` (parallel tests pass)
- [ ] `mypy services/ai_act/ --strict` (type checking passes)

**Behavior Verification:**

```bash
# 1. Enterprise blocked in GPAI mode
python -c "
from services.ai_act.exceptions import EnterpriseNotAvailableError
try:
    from services.ai_act.enterprise import AIActRiskManager
    exit(1)
except EnterpriseNotAvailableError:
    print('PASS: Enterprise blocked in GPAI mode')
"

# 2. ENV override works
AIACT_COMPLIANCE_LEVEL=enterprise python -c "
from services.ai_act.enterprise import AIActRiskManager
print('PASS: ENV override enables enterprise')
"

# 3. Cache invalidation works
python -c "
from services.ai_act._cache import get_enterprise_cache, invalidate_enterprise_cache
invalidate_enterprise_cache()
print(f'PASS: Cache cleared, loaded: {get_enterprise_cache().loaded_modules}')
"

# 4. Context manager works
python -c "
from services.ai_act.config import ai_act_config_context, AIActComplianceConfig, AIActComplianceLevel
with ai_act_config_context(AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE)):
    from services.ai_act.enterprise import AIActRiskManager
    print('PASS: Context manager works')
"

# 5. GPAI always available
python -c "
from services.ai_act.core import TransparencyDisclosureManager
print('PASS: GPAI always available')
"
```

**Final Steps:**

- [ ] Verify all 1007+ tests pass
- [ ] Create PR: `gh pr create --title "feat(ai-act): tiered architecture v2.1"`
- [ ] After merge: `git tag v5.1.0-ai-act-tiered`

**Exit Criteria:**

- [ ] All tests pass (sequential + parallel)
- [ ] Type checking passes
- [ ] All 5 behavior verifications pass
- [ ] PR created and reviewed
- [ ] Tag created after merge

---

## 9. RESOLVED ISSUES SUMMARY

### v2.0 Issues (from v1.0)

| # | Issue | Resolution |
|---|-------|------------|
| 1 | Thread safety | `contextvars` instead of global state |
| 2 | core→enterprise dependency | Split technical_documentation.py into GPAI (core) and High-Risk (enterprise) |
| 3 | Wrong dependency direction | Fixed: risk_registry imports FROM risk_management |
| 4 | IDE/mypy support | Added `.pyi` stub files + TYPE_CHECKING blocks |
| 5 | Import order dependency | Context manager `ai_act_config_context` for scoped config |
| 6 | Incomplete **all** | Fully explicit **all** with all exports |
| 7 | Test fixture state leak | `autouse=True` fixture with token-based reset |
| 8 | Empty _compat.py | Full implementation with module mapping |
| 9 | YAML enum validation | `@field_validator` with normalization |
| 10 | Primitive rollback | Feature branch + incremental commits + backup tags |
| 11 | Article 11 vs 53 | gpai_technical_docs.py (Art.53) vs high_risk_technical_docs.py (Art.11) |
| 12 | Enterprise loads all | Lazy loading via `__getattr__` + module cache |
| 13 | No deprecation versioning | `_version.py` with DEPRECATION_VERSION and REMOVAL_DATE |

### v2.1 Issues (from v2.0 review)

| # | Issue | Resolution |
|---|-------|------------|
| 14 | **Enterprise loads without config check** | `_check_enterprise_access()` in `__getattr__` BEFORE loading module |
| 15 | **Soft warning vs hard block** | `EnterpriseNotAvailableError` (RuntimeError) instead of DeprecationWarning |
| 16 | **Direct import bypasses guard** | `_guard.py` with `require_enterprise_access()` for module-level check |
| 17 | **Circular imports in enterprise** | TYPE_CHECKING pattern + Dependency Injection for cross-dependencies |
| 18 | **No ENV override** | `AIACT_COMPLIANCE_LEVEL` environment variable with priority over config |
| 19 | **No config serialization** | `to_dict()`, `from_dict()`, `from_yaml_section()` methods |
| 20 | **No cache invalidation** | `_cache.py` with `ModuleCache` class and `invalidate()` method |
| 21 | **Test fixture doesn't clean cache** | `reset_ai_act_state_after_test()` fixture clears config + cache + guard |
| 22 | **No guard control for tests** | `disable_enterprise_guard()` fixture + `_GUARD_DISABLED_FOR_TESTING` flag |
| 23 | **Art.52/53(2) not documented** | Section 2.3 with explicit out-of-scope justification |
| 24 | **No graceful error handling** | `ModuleLoadError` wraps ImportError with context |
| 25 | **No testing markers** | `@pytest.mark.enterprise` and `@pytest.mark.gpai_only` markers |

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **RuntimeError over Warning** | Enterprise is B2B feature requiring explicit enablement. Silent warnings could lead to compliance issues and accidental usage. |
| **Import-time guard check** | Performance: check once at import, not every method call. Security: fail fast before code execution. |
| **TYPE_CHECKING + DI pattern** | Prevents circular imports while maintaining full type safety. DI enables testing with mocks. |
| **ENV override priority** | Allows DevOps to control compliance level without code changes. Useful for staging/production differences. |
| **Module cache with invalidation** | Performance: avoid re-importing. Testing: clean state between tests. Hot-reload: support development workflow. |

---

## 10. EXECUTION COMMAND SEQUENCE v2.1

```bash
# Setup
git checkout -b feature/ai-act-tiered-architecture-v2.1
git tag backup/ai-act-before-tiered-$(date +%Y%m%d)

# Phase 1: Infrastructure (NEW FILES)
mkdir -p services/ai_act/core services/ai_act/enterprise

# Create _version.py
cat > services/ai_act/_version.py << 'EOF'
__version__ = "5.1.0"
__ai_act_compliance_phase__ = 5
__gpai_compliance_version__ = "3.0.0"
DEPRECATION_VERSION = "5.0.0"
REMOVAL_VERSION = "6.0.0"
DEPRECATION_DATE = "2025-01-15"
REMOVAL_DATE = "2025-07-15"
EOF

# Create exceptions.py (from plan section 3.2)
# Create _cache.py (from plan section 3.3)
# Create config.py with ENV override (from plan section 3.4)
# Create _compat.py (from plan section 3.x)

git add services/ai_act/_version.py services/ai_act/exceptions.py services/ai_act/_cache.py
git commit -m "feat(ai-act): add core infrastructure files"
# Phase 2: Core modules
git mv services/ai_act/transparency_disclosure.py services/ai_act/core/
git mv services/ai_act/gpai_model_card.py services/ai_act/core/
git mv services/ai_act/copyright_compliance.py services/ai_act/core/
git mv services/ai_act/training_data_summary.py services/ai_act/core/
git mv services/ai_act/user_acknowledgment.py services/ai_act/core/
# Create gpai_technical_docs.py from technical_documentation.py (GPAI parts only)
git commit -m "feat(ai-act): create core package with GPAI modules"

# Phase 3: Enterprise modules with guard
# Create _guard.py (from plan section 3.5)
cat > services/ai_act/enterprise/_guard.py << 'EOF'
# ... content from plan section 3.5
EOF

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
git commit -m "feat(ai-act): create enterprise package with access guard"

# Phase 4: Update imports to TYPE_CHECKING pattern
# Update each enterprise module with cross-dependencies
# See plan section 6 for detailed patterns
git commit -m "refactor(ai-act): update imports to TYPE_CHECKING pattern"

# Phase 5-8: Create __init__.py files, stubs, tests, docs

# Verify - GPAI mode (default)
python -c "from services.ai_act.core import TransparencyDisclosureManager; print('Core OK')"

# Verify - Enterprise should FAIL in default mode
python -c "
try:
    from services.ai_act.enterprise import AIActRiskManager
    print('ERROR: Should have raised EnterpriseNotAvailableError')
    exit(1)
except Exception as e:
    print(f'OK: {type(e).__name__}')
"

# Verify - Enterprise should WORK with ENV override
AIACT_COMPLIANCE_LEVEL=enterprise python -c "
from services.ai_act.enterprise import AIActRiskManager
print('Enterprise OK with ENV override')
"

# Run tests
pytest tests/test_ai_act_*.py -v --tb=short
pytest tests/test_ai_act_*.py -v -n auto  # Parallel
mypy services/ai_act/ --strict

# Commit and push
git add -A
git commit -m "feat(ai-act): implement tiered architecture v2.1 with enterprise guard"
git push -u origin feature/ai-act-tiered-architecture-v2.1
```

---

## 11. QUICK REFERENCE CARD

### Environment Variable

```bash
# Enable enterprise features
export AIACT_COMPLIANCE_LEVEL=enterprise

# Values: gpai (default), enterprise, custom
```

### Python Config

```python
from services.ai_act.config import (
    AIActComplianceConfig,
    AIActComplianceLevel,
    ai_act_config_context,
)

# Temporary enterprise mode
with ai_act_config_context(AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE)):
    from services.ai_act.enterprise import AIActRiskManager
```

### Import Patterns

```python
# GPAI (always available)
from services.ai_act.core import TransparencyDisclosureManager

# Enterprise (requires enablement)
# Option 1: ENV variable
# Option 2: Context manager (see above)
# Option 3: Global config
from services.ai_act.config import set_ai_act_config, AIActComplianceConfig, AIActComplianceLevel
set_ai_act_config(AIActComplianceConfig(level=AIActComplianceLevel.ENTERPRISE))
from services.ai_act.enterprise import AIActRiskManager
```

### Testing

```python
# Test that enterprise is blocked
def test_gpai_blocks_enterprise(expect_enterprise_error):
    with expect_enterprise_error("AIActRiskManager"):
        from services.ai_act.enterprise import AIActRiskManager

# Test with enterprise enabled
def test_with_enterprise(ai_act_enterprise_config):
    from services.ai_act.enterprise import AIActRiskManager
    assert AIActRiskManager is not None
```

---
END OF PLAN v2.1

# -*- coding: utf-8 -*-
"""
Feature flags for gradual futures integration rollout.

Enables:
1. Shadow mode testing (run parallel to production without affecting positions)
2. Canary deployment (small % of traffic)
3. Kill switch for rapid rollback
4. A/B testing of execution algorithms

Usage:
    flags = FuturesFeatureFlags.load("configs/feature_flags_futures.yaml")

    if flags.is_enabled(FuturesFeature.PERPETUAL_TRADING):
        # Execute perpetual trading logic
        pass

    if flags.should_execute(FuturesFeature.L3_EXECUTION, symbol="BTCUSDT"):
        # Use L3 execution for this symbol
        pass

References:
- Martin Fowler: Feature Toggles (Feature Flags)
- LaunchDarkly: Best Practices for Feature Flags
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable, List
import logging
import json
from functools import wraps
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class RolloutStage(str, Enum):
    """
    Deployment stage for futures features.

    Attributes:
        DISABLED: Feature completely off - no code path executed
        SHADOW: Run in parallel, don't affect positions (for comparison testing)
        CANARY: Small % of traffic (configurable) for gradual rollout
        PRODUCTION: Full rollout to all traffic
    """

    DISABLED = "disabled"  # Feature completely off
    SHADOW = "shadow"  # Run in parallel, don't affect positions
    CANARY = "canary"  # Small % of traffic (configurable)
    PRODUCTION = "production"  # Full rollout


class FuturesFeature(str, Enum):
    """
    Individual futures features that can be toggled.

    Organized by functional area for clarity.
    """

    # ─────────────────────────────────────────────────────────────────
    # CORE TRADING FEATURES
    # ─────────────────────────────────────────────────────────────────
    PERPETUAL_TRADING = "perpetual_trading"
    QUARTERLY_TRADING = "quarterly_trading"
    INDEX_FUTURES = "index_futures"
    COMMODITY_FUTURES = "commodity_futures"
    CURRENCY_FUTURES = "currency_futures"
    BOND_FUTURES = "bond_futures"

    # ─────────────────────────────────────────────────────────────────
    # MARGIN & LIQUIDATION
    # ─────────────────────────────────────────────────────────────────
    CROSS_MARGIN = "cross_margin"
    ISOLATED_MARGIN = "isolated_margin"
    LIQUIDATION_SIMULATION = "liquidation_simulation"
    ADL_SIMULATION = "adl_simulation"
    SPAN_MARGIN = "span_margin"

    # ─────────────────────────────────────────────────────────────────
    # FUNDING (Perpetuals only)
    # ─────────────────────────────────────────────────────────────────
    FUNDING_RATE_TRACKING = "funding_rate_tracking"
    FUNDING_IN_REWARD = "funding_in_reward"
    PRO_RATA_FUNDING = "pro_rata_funding"

    # ─────────────────────────────────────────────────────────────────
    # SETTLEMENT (CME/Quarterly)
    # ─────────────────────────────────────────────────────────────────
    DAILY_SETTLEMENT = "daily_settlement"
    AUTO_ROLLOVER = "auto_rollover"
    VARIATION_MARGIN = "variation_margin"

    # ─────────────────────────────────────────────────────────────────
    # EXECUTION
    # ─────────────────────────────────────────────────────────────────
    L2_EXECUTION = "l2_execution"
    L3_EXECUTION = "l3_execution"
    LIQUIDATION_CASCADE_SLIPPAGE = "liquidation_cascade_slippage"
    MARK_PRICE_EXECUTION = "mark_price_execution"

    # ─────────────────────────────────────────────────────────────────
    # RISK MANAGEMENT
    # ─────────────────────────────────────────────────────────────────
    FUTURES_RISK_GUARDS = "futures_risk_guards"
    LEVERAGE_GUARD = "leverage_guard"
    FUNDING_EXPOSURE_GUARD = "funding_exposure_guard"
    CONCENTRATION_GUARD = "concentration_guard"
    MARGIN_GUARD = "margin_guard"
    CIRCUIT_BREAKER_GUARD = "circuit_breaker_guard"

    # ─────────────────────────────────────────────────────────────────
    # DATA & FEATURES PIPELINE
    # ─────────────────────────────────────────────────────────────────
    FUTURES_FEATURES_PIPELINE = "futures_features_pipeline"
    TERM_STRUCTURE_FEATURES = "term_structure_features"
    BASIS_TRADING_FEATURES = "basis_trading_features"
    FUNDING_RATE_FEATURES = "funding_rate_features"
    OPEN_INTEREST_FEATURES = "open_interest_features"

    # ─────────────────────────────────────────────────────────────────
    # TRAINING ENVIRONMENT
    # ─────────────────────────────────────────────────────────────────
    FUTURES_ENV_WRAPPER = "futures_env_wrapper"
    LEVERAGE_ACTION_SPACE = "leverage_action_space"
    MARGIN_OBSERVATION = "margin_observation"


@dataclass
class FeatureConfig:
    """
    Configuration for a single feature flag.

    Attributes:
        stage: Current rollout stage
        canary_percentage: 0-100, used when stage=CANARY
        allowed_symbols: If set, only these symbols are affected
        allowed_accounts: If set, only these accounts are affected
        metadata: Additional feature-specific metadata
    """

    stage: RolloutStage = RolloutStage.DISABLED
    canary_percentage: float = 0.0  # 0-100, used when stage=CANARY
    allowed_symbols: Optional[List[str]] = None  # If set, only these symbols
    allowed_accounts: Optional[List[str]] = None  # If set, only these accounts
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.canary_percentage <= 100.0:
            raise ValueError(f"canary_percentage must be 0-100, got {self.canary_percentage}")
        if isinstance(self.stage, str):
            self.stage = RolloutStage(self.stage)


@dataclass
class FuturesFeatureFlags:
    """
    Centralized feature flag management for futures integration.

    Thread-safe implementation with file persistence and runtime updates.

    Usage:
        # Load from file
        flags = FuturesFeatureFlags.load("configs/feature_flags_futures.yaml")

        # Check if enabled
        if flags.is_enabled(FuturesFeature.PERPETUAL_TRADING):
            # Execute perpetual trading logic
            pass

        # Check with context
        if flags.should_execute(FuturesFeature.L3_EXECUTION, symbol="BTCUSDT"):
            # Use L3 execution for this symbol
            pass

        # Emergency disable
        flags.enable_kill_switch()
    """

    features: Dict[FuturesFeature, FeatureConfig] = field(default_factory=dict)
    global_kill_switch: bool = False
    environment: str = "development"  # development, staging, production
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self):
        """Initialize all features as disabled by default."""
        for feature in FuturesFeature:
            if feature not in self.features:
                self.features[feature] = FeatureConfig()

    @classmethod
    def load(cls, path: str) -> "FuturesFeatureFlags":
        """
        Load feature flags from YAML/JSON file.

        Args:
            path: Path to configuration file

        Returns:
            FuturesFeatureFlags instance
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Feature flags file not found: {path}, using defaults")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml

                    data = yaml.safe_load(f)
                except ImportError:
                    logger.error("PyYAML not installed, falling back to defaults")
                    return cls()
            else:
                data = json.load(f)

        return cls._from_dict(data or {})

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "FuturesFeatureFlags":
        """Parse feature flags from dictionary."""
        features_dict: Dict[FuturesFeature, FeatureConfig] = {}

        for feature_name, config in data.get("features", {}).items():
            try:
                feature = FuturesFeature(feature_name)
                features_dict[feature] = FeatureConfig(
                    stage=RolloutStage(config.get("stage", "disabled")),
                    canary_percentage=float(config.get("canary_percentage", 0.0)),
                    allowed_symbols=config.get("allowed_symbols"),
                    allowed_accounts=config.get("allowed_accounts"),
                    metadata=config.get("metadata", {}),
                )
            except ValueError:
                logger.warning(f"Unknown feature flag: {feature_name}")

        flags = cls(
            features=features_dict,
            global_kill_switch=data.get("global_kill_switch", False),
            environment=data.get("environment", "development"),
        )

        return flags

    def is_enabled(self, feature: FuturesFeature) -> bool:
        """
        Check if feature is enabled (any stage except DISABLED).

        Args:
            feature: Feature to check

        Returns:
            True if feature is enabled
        """
        with self._lock:
            if self.global_kill_switch:
                return False
            return self.features[feature].stage != RolloutStage.DISABLED

    def is_production(self, feature: FuturesFeature) -> bool:
        """
        Check if feature is in full production rollout.

        Args:
            feature: Feature to check

        Returns:
            True if feature is in production stage
        """
        with self._lock:
            if self.global_kill_switch:
                return False
            return self.features[feature].stage == RolloutStage.PRODUCTION

    def is_shadow_mode(self, feature: FuturesFeature) -> bool:
        """
        Check if feature is in shadow mode (run but don't affect positions).

        Args:
            feature: Feature to check

        Returns:
            True if feature is in shadow mode
        """
        with self._lock:
            return self.features[feature].stage == RolloutStage.SHADOW

    def is_canary(self, feature: FuturesFeature) -> bool:
        """
        Check if feature is in canary rollout stage.

        Args:
            feature: Feature to check

        Returns:
            True if feature is in canary stage
        """
        with self._lock:
            return self.features[feature].stage == RolloutStage.CANARY

    def should_execute(
        self,
        feature: FuturesFeature,
        symbol: Optional[str] = None,
        account_id: Optional[str] = None,
        random_value: Optional[float] = None,  # 0-100 for canary selection
    ) -> bool:
        """
        Determine if feature should execute for given context.

        Args:
            feature: Feature to check
            symbol: Trading symbol (for symbol-specific rollout)
            account_id: Account ID (for account-specific rollout)
            random_value: Random value 0-100 for canary percentage check

        Returns:
            True if feature should execute
        """
        with self._lock:
            if self.global_kill_switch:
                return False

            config = self.features[feature]

            if config.stage == RolloutStage.DISABLED:
                return False

            if config.stage == RolloutStage.SHADOW:
                # Shadow mode: execute but caller should not affect positions
                return True

            if config.stage == RolloutStage.CANARY:
                # Check canary criteria
                if config.allowed_symbols and symbol and symbol not in config.allowed_symbols:
                    return False
                if (
                    config.allowed_accounts
                    and account_id
                    and account_id not in config.allowed_accounts
                ):
                    return False
                if random_value is not None:
                    return random_value < config.canary_percentage
                return True

            if config.stage == RolloutStage.PRODUCTION:
                return True

            return False

    def get_stage(self, feature: FuturesFeature) -> RolloutStage:
        """
        Get current rollout stage for feature.

        Args:
            feature: Feature to check

        Returns:
            Current RolloutStage
        """
        with self._lock:
            return self.features[feature].stage

    def get_config(self, feature: FuturesFeature) -> FeatureConfig:
        """
        Get full configuration for feature.

        Args:
            feature: Feature to get config for

        Returns:
            FeatureConfig instance
        """
        with self._lock:
            return self.features[feature]

    def set_stage(self, feature: FuturesFeature, stage: RolloutStage) -> None:
        """
        Set rollout stage for feature (runtime update).

        Args:
            feature: Feature to update
            stage: New rollout stage
        """
        with self._lock:
            old_stage = self.features[feature].stage
            logger.info(
                f"Feature {feature.value} stage changed: " f"{old_stage.value} -> {stage.value}"
            )
            self.features[feature].stage = stage

    def set_canary_percentage(self, feature: FuturesFeature, percentage: float) -> None:
        """
        Set canary percentage for feature.

        Args:
            feature: Feature to update
            percentage: New canary percentage (0-100)
        """
        if not 0.0 <= percentage <= 100.0:
            raise ValueError(f"percentage must be 0-100, got {percentage}")
        with self._lock:
            self.features[feature].canary_percentage = percentage

    def enable_kill_switch(self) -> None:
        """Emergency kill switch - disable all features."""
        with self._lock:
            logger.critical("GLOBAL KILL SWITCH ACTIVATED - All futures features disabled")
            self.global_kill_switch = True

    def disable_kill_switch(self) -> None:
        """Re-enable features after kill switch."""
        with self._lock:
            logger.warning("Global kill switch disabled - Features restored to configured state")
            self.global_kill_switch = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Export current state as dictionary.

        Returns:
            Dictionary representation
        """
        with self._lock:
            return {
                "global_kill_switch": self.global_kill_switch,
                "environment": self.environment,
                "features": {
                    f.value: {
                        "stage": self.features[f].stage.value,
                        "canary_percentage": self.features[f].canary_percentage,
                        "allowed_symbols": self.features[f].allowed_symbols,
                        "allowed_accounts": self.features[f].allowed_accounts,
                        "metadata": self.features[f].metadata,
                    }
                    for f in FuturesFeature
                },
            }

    def save(self, path: str) -> None:
        """
        Save current state to file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        data = self.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml

                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                except ImportError:
                    logger.error("PyYAML not installed, saving as JSON")
                    path = path.with_suffix(".json")
                    json.dump(data, f, indent=2)
            else:
                json.dump(data, f, indent=2)

        logger.info(f"Feature flags saved to {path}")

    def get_enabled_features(self) -> List[FuturesFeature]:
        """
        Get list of all enabled features.

        Returns:
            List of enabled FuturesFeature enums
        """
        with self._lock:
            if self.global_kill_switch:
                return []
            return [f for f in FuturesFeature if self.features[f].stage != RolloutStage.DISABLED]

    def get_production_features(self) -> List[FuturesFeature]:
        """
        Get list of features in production stage.

        Returns:
            List of production FuturesFeature enums
        """
        with self._lock:
            if self.global_kill_switch:
                return []
            return [f for f in FuturesFeature if self.features[f].stage == RolloutStage.PRODUCTION]


# ═══════════════════════════════════════════════════════════════════════════
# HELPER DECORATORS
# ═══════════════════════════════════════════════════════════════════════════


def feature_flag(
    feature: FuturesFeature,
    fallback: Optional[Callable] = None,
    shadow_log: bool = True,
):
    """
    Decorator to gate function execution by feature flag.

    Args:
        feature: Feature to check
        fallback: Optional fallback function if feature is disabled
        shadow_log: Whether to log shadow mode execution

    Usage:
        @feature_flag(FuturesFeature.L3_EXECUTION)
        def execute_l3(order, market):
            # Only runs if L3_EXECUTION is enabled
            ...

        @feature_flag(FuturesFeature.ADL_SIMULATION, fallback=lambda: None)
        def simulate_adl(positions):
            # Falls back to no-op if disabled
            ...
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get flags from kwarg or global
            flags = kwargs.pop("_feature_flags", None) or get_global_flags()

            if flags.should_execute(feature):
                if flags.is_shadow_mode(feature) and shadow_log:
                    logger.debug(f"[SHADOW] Executing {func.__name__} for {feature.value}")
                return func(*args, **kwargs)
            else:
                if fallback is not None:
                    return fallback()
                return None

        return wrapper

    return decorator


def require_feature(feature: FuturesFeature):
    """
    Decorator that raises an error if feature is not enabled.

    Args:
        feature: Feature to require

    Raises:
        RuntimeError: If feature is not enabled

    Usage:
        @require_feature(FuturesFeature.PERPETUAL_TRADING)
        def open_perpetual_position(symbol, qty, leverage):
            ...
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            flags = kwargs.pop("_feature_flags", None) or get_global_flags()

            if not flags.is_enabled(feature):
                raise RuntimeError(
                    f"Feature {feature.value} is required but not enabled. "
                    f"Current stage: {flags.get_stage(feature).value}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL FLAGS MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

# Global flags instance (set during initialization)
_global_flags: Optional[FuturesFeatureFlags] = None
_global_flags_lock = threading.Lock()


def init_feature_flags(path: str) -> FuturesFeatureFlags:
    """
    Initialize global feature flags from file.

    Args:
        path: Path to configuration file

    Returns:
        Initialized FuturesFeatureFlags instance
    """
    global _global_flags
    with _global_flags_lock:
        _global_flags = FuturesFeatureFlags.load(path)
        logger.info(f"Initialized feature flags from {path}")
        enabled = _global_flags.get_enabled_features()
        if enabled:
            logger.info(f"Enabled features: {[f.value for f in enabled]}")
        return _global_flags


def get_global_flags() -> FuturesFeatureFlags:
    """
    Get global flags instance, creating default if not initialized.

    Returns:
        FuturesFeatureFlags instance
    """
    global _global_flags
    if _global_flags is None:
        with _global_flags_lock:
            if _global_flags is None:
                _global_flags = FuturesFeatureFlags()
    return _global_flags


def set_global_flags(flags: FuturesFeatureFlags) -> None:
    """
    Set global flags instance (for testing or programmatic setup).

    Args:
        flags: FuturesFeatureFlags instance to set as global
    """
    global _global_flags
    with _global_flags_lock:
        _global_flags = flags


def reset_global_flags() -> None:
    """
    Reset global flags to None (for testing).
    """
    global _global_flags
    with _global_flags_lock:
        _global_flags = None


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE GROUP HELPERS
# ═══════════════════════════════════════════════════════════════════════════

CRYPTO_FEATURES = frozenset(
    {
        FuturesFeature.PERPETUAL_TRADING,
        FuturesFeature.QUARTERLY_TRADING,
        FuturesFeature.CROSS_MARGIN,
        FuturesFeature.ISOLATED_MARGIN,
        FuturesFeature.FUNDING_RATE_TRACKING,
        FuturesFeature.FUNDING_IN_REWARD,
        FuturesFeature.PRO_RATA_FUNDING,
        FuturesFeature.ADL_SIMULATION,
        FuturesFeature.LIQUIDATION_CASCADE_SLIPPAGE,
    }
)

CME_FEATURES = frozenset(
    {
        FuturesFeature.INDEX_FUTURES,
        FuturesFeature.COMMODITY_FUTURES,
        FuturesFeature.CURRENCY_FUTURES,
        FuturesFeature.BOND_FUTURES,
        FuturesFeature.SPAN_MARGIN,
        FuturesFeature.DAILY_SETTLEMENT,
        FuturesFeature.AUTO_ROLLOVER,
        FuturesFeature.VARIATION_MARGIN,
        FuturesFeature.CIRCUIT_BREAKER_GUARD,
    }
)

RISK_FEATURES = frozenset(
    {
        FuturesFeature.FUTURES_RISK_GUARDS,
        FuturesFeature.LEVERAGE_GUARD,
        FuturesFeature.FUNDING_EXPOSURE_GUARD,
        FuturesFeature.CONCENTRATION_GUARD,
        FuturesFeature.MARGIN_GUARD,
        FuturesFeature.CIRCUIT_BREAKER_GUARD,
    }
)

EXECUTION_FEATURES = frozenset(
    {
        FuturesFeature.L2_EXECUTION,
        FuturesFeature.L3_EXECUTION,
        FuturesFeature.LIQUIDATION_CASCADE_SLIPPAGE,
        FuturesFeature.MARK_PRICE_EXECUTION,
    }
)

TRAINING_FEATURES = frozenset(
    {
        FuturesFeature.FUTURES_ENV_WRAPPER,
        FuturesFeature.LEVERAGE_ACTION_SPACE,
        FuturesFeature.MARGIN_OBSERVATION,
        FuturesFeature.FUTURES_FEATURES_PIPELINE,
    }
)


def are_crypto_features_enabled(flags: Optional[FuturesFeatureFlags] = None) -> bool:
    """Check if core crypto futures features are enabled."""
    flags = flags or get_global_flags()
    return flags.is_enabled(FuturesFeature.PERPETUAL_TRADING)


def are_cme_features_enabled(flags: Optional[FuturesFeatureFlags] = None) -> bool:
    """Check if any CME futures features are enabled."""
    flags = flags or get_global_flags()
    return any(flags.is_enabled(f) for f in CME_FEATURES)


def are_risk_features_enabled(flags: Optional[FuturesFeatureFlags] = None) -> bool:
    """Check if risk management features are enabled."""
    flags = flags or get_global_flags()
    return flags.is_enabled(FuturesFeature.FUTURES_RISK_GUARDS)


def enable_all_for_testing(flags: Optional[FuturesFeatureFlags] = None) -> FuturesFeatureFlags:
    """
    Enable all features for testing purposes.

    Args:
        flags: Optional existing flags instance to modify

    Returns:
        FuturesFeatureFlags with all features in production stage
    """
    flags = flags or FuturesFeatureFlags()
    for feature in FuturesFeature:
        flags.set_stage(feature, RolloutStage.PRODUCTION)
    return flags


def create_minimal_crypto_flags() -> FuturesFeatureFlags:
    """
    Create minimal feature flags for crypto perpetual trading.

    Returns:
        FuturesFeatureFlags with essential crypto features enabled
    """
    flags = FuturesFeatureFlags()
    essential = [
        FuturesFeature.PERPETUAL_TRADING,
        FuturesFeature.CROSS_MARGIN,
        FuturesFeature.FUNDING_RATE_TRACKING,
        FuturesFeature.L2_EXECUTION,
        FuturesFeature.FUTURES_RISK_GUARDS,
        FuturesFeature.LEVERAGE_GUARD,
        FuturesFeature.FUTURES_ENV_WRAPPER,
    ]
    for feature in essential:
        flags.set_stage(feature, RolloutStage.PRODUCTION)
    return flags


def create_minimal_cme_flags() -> FuturesFeatureFlags:
    """
    Create minimal feature flags for CME futures trading.

    Returns:
        FuturesFeatureFlags with essential CME features enabled
    """
    flags = FuturesFeatureFlags()
    essential = [
        FuturesFeature.INDEX_FUTURES,
        FuturesFeature.SPAN_MARGIN,
        FuturesFeature.DAILY_SETTLEMENT,
        FuturesFeature.VARIATION_MARGIN,
        FuturesFeature.L2_EXECUTION,
        FuturesFeature.FUTURES_RISK_GUARDS,
        FuturesFeature.CIRCUIT_BREAKER_GUARD,
        FuturesFeature.FUTURES_ENV_WRAPPER,
    ]
    for feature in essential:
        flags.set_stage(feature, RolloutStage.PRODUCTION)
    return flags


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS (Module-level API)
# ═══════════════════════════════════════════════════════════════════════════


def is_feature_enabled(feature: FuturesFeature) -> bool:
    """
    Check if a feature is enabled (module-level convenience function).

    This is a shorthand for `get_global_flags().is_enabled(feature)`.

    Args:
        feature: Feature to check

    Returns:
        True if feature is enabled (any stage except DISABLED)
    """
    return get_global_flags().is_enabled(feature)


def load_feature_flags(path: str) -> FuturesFeatureFlags:
    """
    Load feature flags from file and set as global (alias for init_feature_flags).

    Args:
        path: Path to configuration file

    Returns:
        Initialized FuturesFeatureFlags instance
    """
    return init_feature_flags(path)

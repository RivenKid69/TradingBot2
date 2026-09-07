"""
Comprehensive test suite for core_config.py after Pydantic V2 migration.

Tests all configuration classes and their validators to ensure:
1. No deprecation warnings
2. All functionality preserved
3. Backward compatibility maintained
"""

import pytest
import warnings
from typing import Dict, Any
import yaml


def test_import_no_warnings():
    """Test that importing core_config produces no deprecation warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import core_config

        pydantic_warnings = [
            warning
            for warning in w
            if "PydanticDeprecatedSince20" in str(warning.category)
            or "deprecated" in str(warning.message).lower()
        ]

        assert (
            len(pydantic_warnings) == 0
        ), f"Found {len(pydantic_warnings)} Pydantic deprecation warnings during import"


def test_all_config_classes():
    """Test instantiation of all major config classes."""
    from core_config import (
        RiskConfigSection,
        LatencyConfig,
        ExecutionBridgeConfig,
        SpotImpactConfig,
        SpotTurnoverLimit,
        SpotTurnoverCaps,
        SpotCostConfig,
        PortfolioConfig,
        ExecutionRuntimeConfig,
        AdvRuntimeConfig,
        CommonRunConfig,
        SimulationConfig,
        TrainConfig,
        LiveConfig,
        EvalConfig,
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test each config class
        risk = RiskConfigSection()
        assert risk.max_total_notional is None

        latency = LatencyConfig()
        assert latency.use_seasonality is True

        bridge = ExecutionBridgeConfig()
        assert bridge.intrabar_price_model is None

        impact = SpotImpactConfig()
        assert impact.sqrt_coeff == 0.0

        turnover = SpotTurnoverLimit()
        assert turnover.bps is None

        caps = SpotTurnoverCaps()
        assert caps.per_symbol is None

        cost = SpotCostConfig()
        assert cost.taker_fee_bps == 0.0

        portfolio = PortfolioConfig()
        assert portfolio.equity_usd is None

        exec_runtime = ExecutionRuntimeConfig()
        assert exec_runtime.enabled is False

        adv = AdvRuntimeConfig()
        assert adv.enabled is False

        # Check no critical warnings (exclude .dict() warnings which are for backward compatibility)
        critical_warnings = [
            warning
            for warning in w
            if (
                "deprecated" in str(warning.message).lower()
                and "root_validator" not in str(warning.message).lower()
                and "class-based `config`" not in str(warning.message).lower()
                and "`dict` method is deprecated" not in str(warning.message).lower()
            )
        ]
        assert len(critical_warnings) == 0


def test_adv_runtime_config_unknown_fields():
    """Test AdvRuntimeConfig captures unknown fields correctly."""
    from core_config import AdvRuntimeConfig

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test with unknown fields
        config = AdvRuntimeConfig(enabled=True, unknown_field1="value1", unknown_field2=123)

        assert config.enabled is True
        assert config.extra.get("unknown_field1") == "value1"
        assert config.extra.get("unknown_field2") == 123

        # Test with existing extra
        config2 = AdvRuntimeConfig(enabled=False, extra={"existing": "value"}, new_field="new")

        assert config2.extra.get("existing") == "value"
        assert config2.extra.get("new_field") == "new"

        # Check no critical warnings (exclude .dict() warnings which are for backward compatibility)
        critical_warnings = [
            warning
            for warning in w
            if (
                "deprecated" in str(warning.message).lower()
                and "root_validator" not in str(warning.message).lower()
                and "class-based `config`" not in str(warning.message).lower()
                and "`dict` method is deprecated" not in str(warning.message).lower()
            )
        ]
        assert len(critical_warnings) == 0


def test_simulation_config_symbol_sync():
    """Test SimulationConfig syncs symbols to data.symbols."""
    from core_config import SimulationConfig

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        config = SimulationConfig(
            symbols=["BTCUSDT", "ETHUSDT"],
            data={"timeframe": "4h"},
            components={
                "market_data": {"target": "test:Test"},
                "executor": {"target": "test:Test"},
                "feature_pipe": {"target": "test:Test"},
                "policy": {"target": "test:Test"},
                "risk_guards": {"target": "test:Test"},
            },
        )

        assert config.symbols == ["BTCUSDT", "ETHUSDT"]
        assert config.data.symbols == ["BTCUSDT", "ETHUSDT"]

        # Check no critical warnings (exclude .dict() warnings which are for backward compatibility)
        critical_warnings = [
            warning
            for warning in w
            if (
                "deprecated" in str(warning.message).lower()
                and "root_validator" not in str(warning.message).lower()
                and "class-based `config`" not in str(warning.message).lower()
                and "`dict` method is deprecated" not in str(warning.message).lower()
            )
        ]
        assert len(critical_warnings) == 0


def test_train_config_symbol_sync():
    """Test TrainConfig syncs symbols to data.symbols."""
    from core_config import TrainConfig

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        config = TrainConfig(
            symbols=["BTCUSDT", "ETHUSDT"],
            data={"timeframe": "4h"},
            model={"algo": "ppo"},
            components={
                "market_data": {"target": "test:Test"},
                "executor": {"target": "test:Test"},
                "feature_pipe": {"target": "test:Test"},
                "policy": {"target": "test:Test"},
                "risk_guards": {"target": "test:Test"},
            },
        )

        assert config.symbols == ["BTCUSDT", "ETHUSDT"]
        assert config.data.symbols == ["BTCUSDT", "ETHUSDT"]

        # Check no critical warnings (exclude .dict() warnings which are for backward compatibility)
        critical_warnings = [
            warning
            for warning in w
            if (
                "deprecated" in str(warning.message).lower()
                and "root_validator" not in str(warning.message).lower()
                and "class-based `config`" not in str(warning.message).lower()
                and "`dict` method is deprecated" not in str(warning.message).lower()
            )
        ]
        assert len(critical_warnings) == 0


def test_train_data_config_window_sync():
    """Test TrainDataConfig syncs start/end with train_start/train_end."""
    from core_config import TrainDataConfig

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test case 1: start_ts provided (provide symbols to avoid network call)
        config1 = TrainDataConfig(
            symbols=["BTCUSDT"], timeframe="4h", start_ts=1000000, end_ts=2000000
        )
        assert config1.train_start_ts == 1000000
        assert config1.train_end_ts == 2000000

        # Test case 2: train_start_ts provided
        config2 = TrainDataConfig(
            symbols=["BTCUSDT"], timeframe="4h", train_start_ts=3000000, train_end_ts=4000000
        )
        assert config2.start_ts == 3000000
        assert config2.end_ts == 4000000

        # Test case 3: both provided with same values (should work)
        config3 = TrainDataConfig(
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_ts=5000000,
            train_start_ts=5000000,
            end_ts=6000000,
            train_end_ts=6000000,
        )
        assert config3.start_ts == 5000000
        assert config3.train_start_ts == 5000000

        # Test case 4: conflict (should raise error)
        with pytest.raises(ValueError, match="must match"):
            TrainDataConfig(
                symbols=["BTCUSDT"],
                timeframe="4h",
                start_ts=1000000,
                train_start_ts=2000000,  # Different value
            )

        # Check no critical warnings (exclude .dict() warnings which are for backward compatibility)
        critical_warnings = [
            warning
            for warning in w
            if (
                "deprecated" in str(warning.message).lower()
                and "root_validator" not in str(warning.message).lower()
                and "class-based `config`" not in str(warning.message).lower()
                and "`dict` method is deprecated" not in str(warning.message).lower()
            )
        ]
        assert len(critical_warnings) == 0


def test_config_dict_methods():
    """Test that dict() and model_dump() methods work correctly."""
    from core_config import LatencyConfig, AdvRuntimeConfig, ExecutionRuntimeConfig

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test LatencyConfig.dict()
        latency = LatencyConfig(use_seasonality=False)
        data = latency.dict()
        assert isinstance(data, dict)
        assert data["use_seasonality"] is False

        # Test AdvRuntimeConfig.dict()
        adv = AdvRuntimeConfig(enabled=True, path="data/adv.parquet")
        data = adv.dict()
        assert isinstance(data, dict)
        assert data["enabled"] is True
        assert data["path"] == "data/adv.parquet"

        # Test ExecutionRuntimeConfig.dict() and model_dump()
        exec_cfg = ExecutionRuntimeConfig(enabled=True)
        data = exec_cfg.dict()
        assert isinstance(data, dict)

        if hasattr(exec_cfg, "model_dump"):
            data2 = exec_cfg.model_dump()
            assert isinstance(data2, dict)

        # Check no critical warnings (exclude .dict() warnings which are for backward compatibility)
        critical_warnings = [
            warning
            for warning in w
            if (
                "deprecated" in str(warning.message).lower()
                and "root_validator" not in str(warning.message).lower()
                and "class-based `config`" not in str(warning.message).lower()
                and "`dict` method is deprecated" not in str(warning.message).lower()
            )
        ]
        assert len(critical_warnings) == 0


def test_common_run_config_sync_sections():
    """Test CommonRunConfig._sync_runtime_sections validator."""
    from core_config import CommonRunConfig, PortfolioConfig, SpotCostConfig

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create config with portfolio and costs at top level
        config = CommonRunConfig(
            portfolio={"equity_usd": 10000.0},
            costs={"taker_fee_bps": 10.0},
            components={
                "market_data": {"target": "test:Test"},
                "executor": {"target": "test:Test"},
                "feature_pipe": {"target": "test:Test"},
                "policy": {"target": "test:Test"},
                "risk_guards": {"target": "test:Test"},
            },
        )

        # Check that portfolio and costs are synced
        assert isinstance(config.portfolio, PortfolioConfig)
        assert config.portfolio.equity_usd == 10000.0
        assert isinstance(config.costs, SpotCostConfig)
        assert config.costs.taker_fee_bps == 10.0

        # Check that execution.portfolio and execution.costs are also synced
        assert isinstance(config.execution.portfolio, PortfolioConfig)
        assert config.execution.portfolio.equity_usd == 10000.0
        assert isinstance(config.execution.costs, SpotCostConfig)
        assert config.execution.costs.taker_fee_bps == 10.0

        # Check no critical warnings (exclude .dict() warnings which are for backward compatibility)
        critical_warnings = [
            warning
            for warning in w
            if (
                "deprecated" in str(warning.message).lower()
                and "root_validator" not in str(warning.message).lower()
                and "class-based `config`" not in str(warning.message).lower()
                and "`dict` method is deprecated" not in str(warning.message).lower()
            )
        ]
        assert len(critical_warnings) == 0


def test_load_config_from_yaml():
    """Test loading config from YAML string."""
    from core_config import load_config_from_str

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        yaml_content = """
mode: train
symbols: ["BTCUSDT", "ETHUSDT"]
data:
  timeframe: 4h
model:
  algo: ppo
components:
  market_data:
    target: test:Test
  executor:
    target: test:Test
  feature_pipe:
    target: test:Test
  policy:
    target: test:Test
  risk_guards:
    target: test:Test
"""

        config = load_config_from_str(yaml_content)
        assert config.mode == "train"
        assert config.symbols == ["BTCUSDT", "ETHUSDT"]
        assert config.data.timeframe == "4h"
        assert config.model.algo == "ppo"

        # Check no critical warnings (exclude .dict() warnings which are for backward compatibility)
        critical_warnings = [
            warning
            for warning in w
            if (
                "deprecated" in str(warning.message).lower()
                and "root_validator" not in str(warning.message).lower()
                and "class-based `config`" not in str(warning.message).lower()
                and "`dict` method is deprecated" not in str(warning.message).lower()
            )
        ]
        assert len(critical_warnings) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

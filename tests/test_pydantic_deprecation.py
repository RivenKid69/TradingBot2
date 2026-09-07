"""
Test to verify Pydantic V1 deprecation warnings in core_config.py

This test checks if there are any deprecation warnings when using
root_validator which is deprecated in Pydantic V2.
"""

import warnings
import pytest
from typing import List, Dict, Any


def test_pydantic_deprecation_warnings_on_import():
    """Test that importing core_config raises no deprecation warnings."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # Import core_config - this should trigger deprecation warnings if present
        import core_config

        # Filter for Pydantic deprecation warnings
        pydantic_warnings = [
            w
            for w in warning_list
            if "PydanticDeprecatedSince20" in str(w.category)
            or "root_validator" in str(w.message).lower()
            or "deprecated" in str(w.message).lower()
        ]

        # Filter for critical warnings only (exclude .dict() which is for backward compat)
        critical_warnings = [
            w
            for w in pydantic_warnings
            if (
                "root_validator" in str(w.message).lower()
                or "class-based `config`" in str(w.message).lower()
                or "__fields__" in str(w.message).lower()
                or "__fields_set__" in str(w.message).lower()
            )
        ]

        # Print all critical warnings for debugging
        if critical_warnings:
            print("\n=== Critical Pydantic Deprecation Warnings Found ===")
            for w in critical_warnings:
                print(f"  File: {w.filename}:{w.lineno}")
                print(f"  Category: {w.category.__name__}")
                print(f"  Message: {w.message}")
                print()

        # Assertion: should have NO critical deprecation warnings
        assert len(critical_warnings) == 0, (
            f"Found {len(critical_warnings)} critical Pydantic deprecation warnings. "
            f"See output above for details."
        )


def test_adv_runtime_config_capture_unknown():
    """Test AdvRuntimeConfig._capture_unknown validator (line 755)."""
    from core_config import AdvRuntimeConfig

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # Create instance with unknown fields
        config = AdvRuntimeConfig(
            enabled=True,
            path="data/adv.parquet",
            unknown_field_1="value1",
            unknown_field_2="value2",
        )

        # Check warnings
        pydantic_warnings = [w for w in warning_list if "deprecated" in str(w.message).lower()]
        assert (
            len(pydantic_warnings) == 0
        ), "AdvRuntimeConfig should not produce deprecation warnings"

        # Check functionality: unknown fields should be captured in 'extra'
        assert config.extra.get("unknown_field_1") == "value1"
        assert config.extra.get("unknown_field_2") == "value2"


def test_simulation_config_sync_symbols():
    """Test SimulationConfig._sync_symbols validator (line 1066)."""
    from core_config import SimulationConfig

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # Create config with minimal required fields
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

        # Check warnings
        pydantic_warnings = [w for w in warning_list if "deprecated" in str(w.message).lower()]
        assert (
            len(pydantic_warnings) == 0
        ), "SimulationConfig should not produce deprecation warnings"

        # Check functionality: symbols should sync to data.symbols
        assert config.data.symbols == ["BTCUSDT", "ETHUSDT"]


def test_train_data_config_sync_train_window():
    """Test TrainDataConfig._sync_train_window_aliases validator (line 1124)."""
    from core_config import TrainDataConfig

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # Test case 1: start_ts provided, train_start_ts should sync
        config1 = TrainDataConfig(timeframe="4h", start_ts=1000000, end_ts=2000000)

        # Check warnings
        pydantic_warnings = [w for w in warning_list if "deprecated" in str(w.message).lower()]
        assert (
            len(pydantic_warnings) == 0
        ), "TrainDataConfig should not produce deprecation warnings"

        # Check functionality
        assert config1.train_start_ts == 1000000
        assert config1.train_end_ts == 2000000


def test_train_config_sync_symbols():
    """Test TrainConfig._sync_symbols validator (line 1195)."""
    from core_config import TrainConfig

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # Create config with minimal required fields
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

        # Check warnings
        pydantic_warnings = [w for w in warning_list if "deprecated" in str(w.message).lower()]
        assert len(pydantic_warnings) == 0, "TrainConfig should not produce deprecation warnings"

        # Check functionality: symbols should sync to data.symbols
        assert config.data.symbols == ["BTCUSDT", "ETHUSDT"]


def test_all_validators_edge_cases():
    """Test edge cases for all validators."""
    from core_config import AdvRuntimeConfig, SimulationConfig, TrainDataConfig, TrainConfig

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # Test AdvRuntimeConfig with empty extra
        adv1 = AdvRuntimeConfig(enabled=False)
        assert adv1.extra == {}

        # Test AdvRuntimeConfig with existing extra
        adv2 = AdvRuntimeConfig(enabled=False, extra={"existing": "value"}, new_field="new_value")
        assert adv2.extra.get("existing") == "value"
        assert adv2.extra.get("new_field") == "new_value"

        # Test TrainDataConfig: train_start_ts provided, start_ts should sync
        train_data = TrainDataConfig(timeframe="4h", train_start_ts=3000000, train_end_ts=4000000)
        assert train_data.start_ts == 3000000
        assert train_data.end_ts == 4000000

        # Test TrainDataConfig: conflict should raise error
        with pytest.raises(ValueError, match="must match"):
            TrainDataConfig(
                timeframe="4h", start_ts=1000000, train_start_ts=2000000  # Different value
            )

        # Check no warnings
        pydantic_warnings = [w for w in warning_list if "deprecated" in str(w.message).lower()]
        assert len(pydantic_warnings) == 0, "No deprecation warnings should be present"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

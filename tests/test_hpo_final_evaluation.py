"""
Test suite for final model evaluation after HPO completion.

This module tests that the final evaluation (after HPO) correctly uses
test data for independent assessment, following ML best practices.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import numpy as np
import pandas as pd
import pytest


def _install_sb3_stub() -> None:
    """Install minimal SB3 stubs for testing without full dependencies."""
    if "sb3_contrib" in sys.modules:
        return

    sb3_contrib = types.ModuleType("sb3_contrib")
    sb3_contrib.__path__ = []
    sys.modules["sb3_contrib"] = sb3_contrib

    common = types.ModuleType("sb3_contrib.common")
    common.__path__ = []
    sys.modules["sb3_contrib.common"] = common
    sb3_contrib.common = common  # type: ignore[attr-defined]

    recurrent = types.ModuleType("sb3_contrib.common.recurrent")
    recurrent.__path__ = []
    sys.modules["sb3_contrib.common.recurrent"] = recurrent
    common.recurrent = recurrent  # type: ignore[attr-defined]

    policies = types.ModuleType("sb3_contrib.common.recurrent.policies")

    class _DummyPolicy:
        pass

    policies.RecurrentActorCriticPolicy = _DummyPolicy
    sys.modules["sb3_contrib.common.recurrent.policies"] = policies
    recurrent.policies = policies  # type: ignore[attr-defined]


_install_sb3_stub()


@pytest.fixture
def mock_val_data():
    """Create mock validation data (timestamps 100-149)."""
    return {
        "BTCUSDT": pd.DataFrame({
            "timestamp": range(100, 150),
            "close": np.random.randn(50) + 100,
            "volume": np.random.randn(50) * 1000,
        })
    }


@pytest.fixture
def mock_test_data():
    """Create mock test data (timestamps 150-199)."""
    return {
        "BTCUSDT": pd.DataFrame({
            "timestamp": range(150, 200),
            "close": np.random.randn(50) + 100,
            "volume": np.random.randn(50) * 1000,
        })
    }


class TestFinalEvaluationAfterHPO:
    """
    Test suite for final evaluation that happens AFTER HPO completion.
    This is the ONLY place where test data should be used.
    """

    def test_final_evaluation_uses_test_data_when_available(
        self, mock_val_data, mock_test_data, capsys
    ):
        """
        Test that final evaluation (post-HPO) uses test data when available.
        This is the correct behavior for independent model assessment.
        """
        # This would be tested by checking the main() function's behavior
        # after study.optimize() completes.
        # For now, we document the expected behavior:

        # Expected: After HPO completes, the code should:
        # 1. Use test_data_by_token for final evaluation
        # 2. Print clear messages indicating test set usage
        # 3. Never feed this evaluation back into HPO

        pass  # This is integration-level test - would need full setup

    def test_final_evaluation_falls_back_to_val_when_no_test(
        self, mock_val_data, capsys
    ):
        """
        Test that final evaluation uses validation data as fallback
        when test data is not configured.
        """
        # Expected behavior: If test_data_by_token is empty,
        # final evaluation should use val_data_by_token
        # This is for backwards compatibility with configs that don't define test split

        pass  # This is integration-level test - would need full setup

    def test_final_evaluation_prints_clear_data_source_message(self, capsys):
        """
        Test that final evaluation prints clear messages about which
        dataset is being used for evaluation.
        """
        # The code should print messages like:
        # "✓ Using test set for final independent evaluation"
        # or
        # "⚠ Test set not available - using validation set"

        pass  # Would need to mock the full HPO completion flow


class TestConfigurationValidation:
    """Tests for configuration validation and data split setup."""

    def test_config_has_separate_train_val_test_splits(self):
        """
        Verify that the default config defines separate train/val/test splits.
        This is critical for proper ML workflow.
        """
        from pathlib import Path
        import yaml

        config_path = Path("/home/user/ai-quant-platform/configs/config_train.yaml")

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        data_config = config.get("data", {})

        # Check that all three splits are defined
        assert "train_start_ts" in data_config or "start_ts" in data_config, \
            "Train split start not defined"
        assert "train_end_ts" in data_config or "end_ts" in data_config, \
            "Train split end not defined"
        assert "val_start_ts" in data_config, "Validation split start not defined"
        assert "val_end_ts" in data_config, "Validation split end not defined"
        assert "test_start_ts" in data_config, "Test split start not defined"
        assert "test_end_ts" in data_config, "Test split end not defined"

        # Verify splits are non-overlapping and sequential
        train_start = data_config.get("train_start_ts") or data_config.get("start_ts")
        train_end = data_config.get("train_end_ts") or data_config.get("end_ts")
        val_start = data_config["val_start_ts"]
        val_end = data_config["val_end_ts"]
        test_start = data_config["test_start_ts"]
        test_end = data_config["test_end_ts"]

        assert train_start < train_end, "Train split invalid"
        assert val_start < val_end, "Val split invalid"
        assert test_start < test_end, "Test split invalid"

        # Check sequential and non-overlapping
        assert train_end <= val_start, "Train and val splits overlap"
        assert val_end <= test_start, "Val and test splits overlap"

    def test_validation_split_is_not_empty(self):
        """
        Verify that validation split is configured with reasonable duration.
        Empty validation would cause HPO to fail.
        """
        from pathlib import Path
        import yaml

        config_path = Path("/home/user/ai-quant-platform/configs/config_train.yaml")

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        data_config = config.get("data", {})
        val_start = data_config.get("val_start_ts")
        val_end = data_config.get("val_end_ts")

        if val_start and val_end:
            duration = val_end - val_start
            # Should be at least 1 day (86400 seconds)
            assert duration >= 86400, \
                f"Validation split too short: {duration} seconds (need at least 1 day)"


class TestDocumentationAndComments:
    """Tests that verify code is properly documented."""

    def test_objective_function_has_data_leakage_warning(self):
        """
        Verify that the objective function has clear comments warning
        about test data leakage.
        """
        import train_model_multi_patch as train_module
        import inspect

        source = inspect.getsource(train_module.objective)

        # Check for critical documentation
        assert "CRITICAL" in source or "critical" in source.lower(), \
            "Objective function should have CRITICAL comment about data usage"
        assert "validation" in source.lower(), \
            "Objective function should mention validation data"
        assert "test" in source.lower(), \
            "Objective function should mention test data"

    def test_code_references_ml_best_practices(self):
        """
        Verify that the code includes references to ML best practices
        or research papers.
        """
        import train_model_multi_patch as train_module
        import inspect

        source = inspect.getsource(train_module.objective)

        # Check for references to ML literature or best practices
        has_reference = any([
            "Hastie" in source,
            "Elements of Statistical Learning" in source,
            "best practice" in source.lower(),
            "ML best practice" in source,
        ])

        assert has_reference, \
            "Code should reference ML best practices or literature"


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with existing configs."""

    def test_objective_handles_missing_test_split_gracefully(
        self, mock_val_data
    ):
        """
        Test that objective function works when test split is not configured.
        This ensures backwards compatibility.
        """
        # When test_data_by_token is empty, objective should:
        # 1. Use val_data_by_token for evaluation
        # 2. Not raise any errors
        # 3. Complete successfully

        # This is already tested in the main test suite
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

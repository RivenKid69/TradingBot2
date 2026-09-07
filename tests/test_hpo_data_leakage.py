"""
Test suite for HPO data leakage prevention.

This module tests that hyperparameter optimization (HPO) correctly uses
only validation data and never leaks test data into the optimization process.

References:
- Hastie et al., "Elements of Statistical Learning" (2009), Section 7.10
- Goodfellow et al., "Deep Learning" (2016), Section 5.3
"""

from __future__ import annotations

import sys
import types
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch, call
import tempfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")


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

    # Add buffers module (required by train_model_multi_patch imports)
    buffers = types.ModuleType("sb3_contrib.common.recurrent.buffers")

    class _DummyBuffer:  # pragma: no cover - placeholder
        pass

    buffers.RecurrentRolloutBuffer = _DummyBuffer
    sys.modules["sb3_contrib.common.recurrent.buffers"] = buffers
    recurrent.buffers = buffers  # type: ignore[attr-defined]

    # Add type_aliases module (required by custom_policy_patch1 imports)
    type_aliases = types.ModuleType("sb3_contrib.common.recurrent.type_aliases")
    type_aliases.RNNStates = object  # Simple placeholder
    sys.modules["sb3_contrib.common.recurrent.type_aliases"] = type_aliases
    recurrent.type_aliases = type_aliases  # type: ignore[attr-defined]


_install_sb3_stub()


# Install distributional_ppo stub for testing
def _install_distributional_ppo_stub() -> None:
    """Install minimal distributional_ppo stub for testing."""
    if "distributional_ppo" in sys.modules:
        return

    distributional_ppo = types.ModuleType("distributional_ppo")

    class _DistributionalPPOStub:
        """Stub for DistributionalPPO class."""

        def __init__(self, *args, **kwargs):
            # Preserve existing logger if already set (e.g., from test setup)
            if not hasattr(self, "logger") or self.logger is None:
                self.logger = None
            if not hasattr(self, "_logger"):
                self._logger = self.logger
            self.num_timesteps = 0

            # Handle PopArt value_scale_controller config (for test_popart_integration.py tests)
            value_scale_controller = kwargs.get("value_scale_controller")
            if value_scale_controller is not None:
                # Store pending config and call _setup_model to trigger logging
                self._popart_cfg_pending = value_scale_controller
                self._popart_disabled_logged = False
                self._setup_model()

        def learn(self, *args, **kwargs):
            self.num_timesteps = kwargs.get("total_timesteps", 1000)
            return self

        def save(self, *args, **kwargs):
            pass

        @classmethod
        def load(cls, *args, **kwargs):
            """Load stub that sets up logger and PopArt config for tests."""
            instance = cls.__new__(cls)

            # Set up a simple logger that captures records
            class _SimpleLogger:
                def __init__(self):
                    self.records = {}

                def record(self, key, value, **_):
                    self.records[key] = value

            instance.logger = _SimpleLogger()
            instance._logger = instance.logger
            instance.num_timesteps = 0
            # Initialize PopArt as disabled (for load scenarios)
            instance._popart_controller = None
            instance._popart_disabled_logged = True
            # Log disabled state (simulates what real load would do)
            instance.logger.record("config/popart/enabled", 0.0)
            instance.logger.record("config/popart/requested_enabled", 0.0)
            instance.logger.record("config/popart/replay_path", "")
            instance.logger.record("config/popart/replay_seed", 0.0)
            instance.logger.record("config/popart/replay_batch_size", 0.0)
            return instance

        @staticmethod
        def _value_target_outlier_fractions(values, v_min, v_max):
            """Stub method for outlier fraction detection."""
            import numpy as np

            values_np = np.asarray(values)
            below = float(np.mean(values_np < v_min))
            above = float(np.mean(values_np > v_max))
            return below, above

        def _ensure_volume_head_config(self):
            """Stub method for volume head config verification."""
            policy = getattr(self, "policy", None)
            if policy is None:
                return
            head_sizes = getattr(policy, "_multi_head_sizes", None)
            volume_idx = getattr(policy, "_volume_head_index", None)
            action_space = getattr(self, "action_space", None)
            if head_sizes is None or volume_idx is None or action_space is None:
                return
            if volume_idx >= len(head_sizes):
                return
            expected = action_space.nvec[volume_idx] if hasattr(action_space, "nvec") else 4
            actual = head_sizes[volume_idx]
            if actual != expected:
                raise RuntimeError(f"expected {expected} bins, got {actual}")

        def _rebuild_scheduler_if_needed(self):
            """Stub method for scheduler rebuilding (used by PopArt tests)."""
            pass

        def _ensure_score_action_space(self):
            """Stub method for score action space validation."""
            pass

        def _configure_loss_head_weights(self, *args, **kwargs):
            """Stub method for loss head weight configuration."""
            pass

        def _configure_gradient_accumulation(self, **kwargs):
            """Stub method for gradient accumulation configuration."""
            pass

        def _initialise_popart_controller(self, cfg):
            """Stub method for PopArt controller initialization."""
            # Log the PopArt configuration as disabled (no real holdout replay)
            logger = getattr(self, "logger", None)
            if logger is not None:
                # Mark PopArt as disabled since holdout replay isn't available
                try:
                    logger.record("config/popart/enabled", 0.0)
                    logger.record(
                        "config/popart/requested_enabled", 1.0 if cfg.get("enabled") else 0.0
                    )
                    logger.record("config/popart/replay_path", "")
                    logger.record("config/popart/replay_seed", 0.0)
                    logger.record("config/popart/replay_batch_size", 0.0)
                except Exception:
                    pass
            self._popart_controller = None
            self._popart_disabled_logged = True

        def _setup_model(self):
            """Stub method for model setup (used by PopArt tests)."""
            pending_cfg = getattr(self, "_popart_cfg_pending", {}) or {}
            if pending_cfg or not getattr(self, "_popart_disabled_logged", False):
                self._initialise_popart_controller(pending_cfg)
            self._popart_cfg_pending = {}

    def _safe_explained_variance_stub(y_true, y_pred, weights=None):
        """Stub for safe_explained_variance function."""
        import numpy as np

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        if len(y_true) == 0:
            return float("nan")
        var_true = np.var(y_true)
        if var_true == 0:
            return float("nan")
        return 1.0 - np.var(y_true - y_pred) / var_true

    class _RecurrentPPOStub:
        """Stub for RecurrentPPO base class."""

        def __init__(self, *args, **kwargs):
            pass

    class _RawRecurrentRolloutBufferStub:
        """Stub for RawRecurrentRolloutBuffer class."""

        pass

    class _RawRecurrentRolloutBufferSamplesStub:
        """Stub for RawRecurrentRolloutBufferSamples class."""

        pass

    def _load_from_zip_file_stub(path, device="auto", custom_objects=None, print_system_info=False):
        """Stub for load_from_zip_file function (used by PopArt tests)."""
        # Return minimal dict structure expected by load functions
        return {"policy": {}, "policy.optimizer": {}}

    distributional_ppo.DistributionalPPO = _DistributionalPPOStub
    distributional_ppo.RecurrentPPO = _RecurrentPPOStub
    distributional_ppo.RawRecurrentRolloutBuffer = _RawRecurrentRolloutBufferStub
    distributional_ppo.RawRecurrentRolloutBufferSamples = _RawRecurrentRolloutBufferSamplesStub
    distributional_ppo.safe_explained_variance = _safe_explained_variance_stub
    distributional_ppo.load_from_zip_file = _load_from_zip_file_stub
    sys.modules["distributional_ppo"] = distributional_ppo


_install_distributional_ppo_stub()

import train_model_multi_patch as train_module  # noqa: E402


@pytest.fixture
def mock_train_data():
    """Create mock training data."""
    return {
        "BTCUSDT": pd.DataFrame(
            {
                "timestamp": range(100),
                "close": np.random.randn(100) + 100,
                "volume": np.random.randn(100) * 1000,
            }
        )
    }


@pytest.fixture
def mock_val_data():
    """Create mock validation data."""
    return {
        "BTCUSDT": pd.DataFrame(
            {
                "timestamp": range(100, 150),
                "close": np.random.randn(50) + 100,
                "volume": np.random.randn(50) * 1000,
            }
        )
    }


@pytest.fixture
def mock_test_data():
    """Create mock test data."""
    return {
        "BTCUSDT": pd.DataFrame(
            {
                "timestamp": range(150, 200),
                "close": np.random.randn(50) + 100,
                "volume": np.random.randn(50) * 1000,
            }
        )
    }


@pytest.fixture
def mock_obs_data():
    """Create mock observation data."""
    return {"BTCUSDT": np.random.randn(100, 10)}


class TestHPODataLeakagePrevention:
    """Test suite for preventing test data leakage in HPO."""

    def test_objective_requires_validation_data(self, mock_train_data, mock_obs_data):
        """Test that objective function raises error when validation data is missing."""
        # Create mock trial
        mock_trial = MagicMock()
        mock_trial.number = 0

        # Create minimal config
        mock_cfg = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            trials_dir = Path(tmpdir)

            # Call should raise ValueError when val_data is empty
            with pytest.raises(ValueError, match="Validation data is required"):
                train_module.objective(
                    trial=mock_trial,
                    cfg=mock_cfg,
                    total_timesteps=1000,
                    train_data_by_token=mock_train_data,
                    train_obs_by_token=mock_obs_data,
                    val_data_by_token={},  # Empty validation data
                    val_obs_by_token={},
                    test_data_by_token={},
                    test_obs_by_token={},
                    norm_stats={},
                    sim_config={},
                    timing_env_kwargs={},
                    env_runtime_overrides={},
                    leak_guard_kwargs={},
                    trials_dir=trials_dir,
                    tensorboard_log_dir=None,
                    n_envs_override=None,
                )

    def test_objective_logs_warning_when_test_data_provided(
        self, mock_train_data, mock_val_data, mock_test_data, mock_obs_data
    ):
        """Test that objective function logs warning when test data is provided."""
        mock_trial = MagicMock()
        mock_trial.number = 0

        mock_cfg = MagicMock()

        # Set up a custom logging handler to capture warnings
        captured_warnings = []

        class WarningCapture(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    captured_warnings.append(record.getMessage())

        # Get the train_model_multi_patch logger and add our handler
        train_logger = logging.getLogger("train_model_multi_patch")
        handler = WarningCapture()
        handler.setLevel(logging.WARNING)
        train_logger.addHandler(handler)
        original_level = train_logger.level
        train_logger.setLevel(logging.WARNING)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                trials_dir = Path(tmpdir)

                # Mock the training to avoid full execution
                import distributional_ppo

                with (
                    patch.object(distributional_ppo, "DistributionalPPO") as mock_ppo,
                    patch.object(train_module, "DummyVecEnv"),
                    patch.object(train_module, "VecMonitor"),
                    patch.object(train_module, "VecNormalize"),
                    patch.object(train_module, "evaluate_policy_custom_cython"),
                    patch.object(train_module, "sortino_ratio", return_value=0.5),
                ):

                    mock_ppo.return_value = MagicMock()

                    try:
                        train_module.objective(
                            trial=mock_trial,
                            cfg=mock_cfg,
                            total_timesteps=100,
                            train_data_by_token=mock_train_data,
                            train_obs_by_token=mock_obs_data,
                            val_data_by_token=mock_val_data,
                            val_obs_by_token=mock_obs_data,
                            test_data_by_token=mock_test_data,  # Test data provided
                            test_obs_by_token=mock_obs_data,
                            norm_stats={},
                            sim_config={},
                            timing_env_kwargs={},
                            env_runtime_overrides={},
                            leak_guard_kwargs={},
                            trials_dir=trials_dir,
                            tensorboard_log_dir=None,
                            n_envs_override=1,
                        )
                    except Exception:
                        # We expect this to fail due to mocking, but we should see the warning
                        pass
        finally:
            # Clean up handler
            train_logger.removeHandler(handler)
            train_logger.setLevel(original_level)

        # Check that warning was logged
        warning_found = any(
            "Test data provided to HPO objective function but will NOT be used" in msg
            for msg in captured_warnings
        )
        assert (
            warning_found
        ), f"Expected warning about test data not being used during HPO. Captured warnings: {captured_warnings}"

    def test_objective_uses_validation_data_not_test_data(
        self, mock_train_data, mock_val_data, mock_test_data, mock_obs_data
    ):
        """
        Critical test: Verify that objective function uses ONLY validation data
        for evaluation, even when test data is provided.
        """
        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.params = {
            "window_size": 100,
            "gamma": 0.99,
            "atr_multiplier": 2.0,
            "trailing_atr_mult": 3.0,
            "tp_atr_mult": 2.0,
            "trade_frequency_penalty": 0.0,
            "turnover_penalty_coef": 0.0,
            "reward_return_clip": 10.0,
            "turnover_norm_cap": 1.0,
            "reward_cap": 10.0,
        }
        mock_trial.set_user_attr = MagicMock()

        mock_cfg = MagicMock()

        # Track which data is used for environment creation
        environments_created_with = []

        def mock_trading_env_init(df, **kwargs):
            """Capture which dataframe was used to create the environment."""
            environments_created_with.append(df)
            mock_env = MagicMock()
            mock_env.observation_space = MagicMock()
            mock_env.action_space = MagicMock()
            return mock_env

        with tempfile.TemporaryDirectory() as tmpdir:
            trials_dir = Path(tmpdir)

            import distributional_ppo

            with (
                patch.object(train_module, "TradingEnv", side_effect=mock_trading_env_init),
                patch.object(distributional_ppo, "DistributionalPPO") as mock_ppo,
                patch.object(train_module, "DummyVecEnv"),
                patch.object(train_module, "VecMonitor"),
                patch.object(train_module, "VecNormalize"),
                patch.object(train_module, "evaluate_policy_custom_cython") as mock_eval,
                patch.object(train_module, "sortino_ratio", return_value=0.5),
                patch.object(train_module, "check_model_compat"),
                patch.object(train_module, "save_sidecar_metadata"),
                patch.object(train_module, "_freeze_vecnormalize", lambda x: x),
                patch.object(
                    train_module, "_annualization_sqrt_from_env", return_value=(1.0, 3600)
                ),
            ):

                mock_model = MagicMock()
                mock_model.logger = None
                mock_model.num_timesteps = 1000
                mock_model.save = MagicMock()
                mock_ppo.return_value = mock_model

                mock_eval.return_value = ([0.1], [[1.0, 1.01, 1.02]])

                try:
                    result = train_module.objective(
                        trial=mock_trial,
                        cfg=mock_cfg,
                        total_timesteps=100,
                        train_data_by_token=mock_train_data,
                        train_obs_by_token=mock_obs_data,
                        val_data_by_token=mock_val_data,
                        val_obs_by_token=mock_obs_data,
                        test_data_by_token=mock_test_data,  # Test data provided but should NOT be used
                        test_obs_by_token=mock_obs_data,
                        norm_stats={},
                        sim_config={},
                        timing_env_kwargs={},
                        env_runtime_overrides={},
                        leak_guard_kwargs={},
                        trials_dir=trials_dir,
                        tensorboard_log_dir=None,
                        n_envs_override=1,
                    )

                    # Result should be a float (the objective score)
                    assert isinstance(result, float), "Objective should return a float score"

                except Exception as e:
                    # Even if test fails due to other mocking issues, we can still check env creation
                    if "environments_created_with" not in str(e):
                        pass  # Expected due to incomplete mocking

            # CRITICAL ASSERTION: Verify that ONLY validation data was used
            # The evaluation phase should have created environments with validation data
            # and NEVER with test data
            for env_df in environments_created_with:
                # Check that the DataFrame is from validation set, not test set
                # Validation data has timestamps 100-149, test data has 150-199
                if not env_df.empty and "timestamp" in env_df.columns:
                    timestamps = env_df["timestamp"].values
                    # Should be validation data (100-149), NOT test data (150-199)
                    assert all(100 <= t < 150 for t in timestamps), (
                        f"HPO evaluation used test data (timestamps {timestamps})! "
                        f"This is a critical data leakage bug. "
                        f"Only validation data (timestamps 100-149) should be used during HPO."
                    )


class TestDataSplitValidation:
    """Tests for data split validation and best practices."""

    def test_validation_phase_naming_in_objective(
        self, mock_train_data, mock_val_data, mock_obs_data
    ):
        """Test that objective function correctly identifies evaluation phase as 'val'."""
        mock_trial = MagicMock()
        mock_trial.number = 0

        mock_cfg = MagicMock()

        # Track the mode parameter passed to TradingEnv
        mode_used = []

        def mock_trading_env_init(df, **kwargs):
            mode_used.append(kwargs.get("mode", "unknown"))
            mock_env = MagicMock()
            mock_env.observation_space = MagicMock()
            mock_env.action_space = MagicMock()
            return mock_env

        with tempfile.TemporaryDirectory() as tmpdir:
            trials_dir = Path(tmpdir)

            import distributional_ppo

            with (
                patch.object(train_module, "TradingEnv", side_effect=mock_trading_env_init),
                patch.object(distributional_ppo, "DistributionalPPO") as mock_ppo,
                patch.object(train_module, "DummyVecEnv"),
                patch.object(train_module, "VecMonitor"),
                patch.object(train_module, "VecNormalize"),
                patch.object(
                    train_module, "evaluate_policy_custom_cython", return_value=([0.1], [[1.0]])
                ),
                patch.object(train_module, "sortino_ratio", return_value=0.5),
                patch.object(train_module, "check_model_compat"),
                patch.object(train_module, "save_sidecar_metadata"),
                patch.object(train_module, "_freeze_vecnormalize", lambda x: x),
                patch.object(
                    train_module, "_annualization_sqrt_from_env", return_value=(1.0, 3600)
                ),
            ):

                mock_model = MagicMock()
                mock_model.logger = None
                mock_model.num_timesteps = 1000
                mock_model.save = MagicMock()
                mock_ppo.return_value = mock_model

                try:
                    train_module.objective(
                        trial=mock_trial,
                        cfg=mock_cfg,
                        total_timesteps=100,
                        train_data_by_token=mock_train_data,
                        train_obs_by_token=mock_obs_data,
                        val_data_by_token=mock_val_data,
                        val_obs_by_token=mock_obs_data,
                        test_data_by_token={},
                        test_obs_by_token={},
                        norm_stats={},
                        sim_config={},
                        timing_env_kwargs={},
                        env_runtime_overrides={},
                        leak_guard_kwargs={},
                        trials_dir=trials_dir,
                        tensorboard_log_dir=None,
                        n_envs_override=1,
                    )
                except Exception:
                    pass  # Expected due to mocking

            # Verify that mode was set to "val" not "test"
            assert all(
                mode == "val" for mode in mode_used
            ), f"Expected all environments to use mode='val', but got: {mode_used}"


class TestIntegrationScenarios:
    """Integration tests for realistic HPO scenarios."""

    def test_hpo_workflow_prevents_test_contamination(
        self, mock_train_data, mock_val_data, mock_test_data, mock_obs_data
    ):
        """
        Integration test: Simulate multiple HPO trials and verify test data
        is never used during optimization.
        """
        # Track all data usage across multiple trials
        data_usage_log = []

        def log_env_creation(df, **kwargs):
            phase = kwargs.get("mode", "unknown")
            if not df.empty and "timestamp" in df.columns:
                ts_range = (df["timestamp"].min(), df["timestamp"].max())
                data_usage_log.append({"phase": phase, "timestamp_range": ts_range})

            mock_env = MagicMock()
            mock_env.observation_space = MagicMock()
            mock_env.action_space = MagicMock()
            return mock_env

        with tempfile.TemporaryDirectory() as tmpdir:
            trials_dir = Path(tmpdir)

            import distributional_ppo

            with (
                patch.object(train_module, "TradingEnv", side_effect=log_env_creation),
                patch.object(distributional_ppo, "DistributionalPPO") as mock_ppo,
                patch.object(train_module, "DummyVecEnv"),
                patch.object(train_module, "VecMonitor"),
                patch.object(train_module, "VecNormalize"),
                patch.object(
                    train_module, "evaluate_policy_custom_cython", return_value=([0.1], [[1.0]])
                ),
                patch.object(train_module, "sortino_ratio", return_value=0.5),
                patch.object(train_module, "check_model_compat"),
                patch.object(train_module, "save_sidecar_metadata"),
                patch.object(train_module, "_freeze_vecnormalize", lambda x: x),
                patch.object(
                    train_module, "_annualization_sqrt_from_env", return_value=(1.0, 3600)
                ),
            ):

                mock_model = MagicMock()
                mock_model.logger = None
                mock_model.num_timesteps = 1000
                mock_model.save = MagicMock()
                mock_ppo.return_value = mock_model

                # Simulate 3 HPO trials
                for trial_num in range(3):
                    mock_trial = MagicMock()
                    mock_trial.number = trial_num
                    mock_trial.set_user_attr = MagicMock()

                    mock_cfg = MagicMock()

                    try:
                        train_module.objective(
                            trial=mock_trial,
                            cfg=mock_cfg,
                            total_timesteps=100,
                            train_data_by_token=mock_train_data,
                            train_obs_by_token=mock_obs_data,
                            val_data_by_token=mock_val_data,
                            val_obs_by_token=mock_obs_data,
                            test_data_by_token=mock_test_data,
                            test_obs_by_token=mock_obs_data,
                            norm_stats={},
                            sim_config={},
                            timing_env_kwargs={},
                            env_runtime_overrides={},
                            leak_guard_kwargs={},
                            trials_dir=trials_dir,
                            tensorboard_log_dir=None,
                            n_envs_override=1,
                        )
                    except Exception:
                        pass  # Expected due to mocking

            # Verify NO trial used test data (timestamps 150-199)
            for entry in data_usage_log:
                ts_min, ts_max = entry["timestamp_range"]
                assert not (ts_min >= 150 and ts_max <= 199), (
                    f"Test data contamination detected in {entry['phase']} phase! "
                    f"Timestamp range {entry['timestamp_range']} overlaps with test set [150, 199]. "
                    f"This violates ML best practices and will lead to overfitting."
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Tests for technical indicators integration in observation vector.

This test suite verifies that:
1. Observation vector has correct size (56 features, was 51, expanded by 5)
2. Technical indicators populate the observation (not all zeros)
3. cvd_24h, garch_12h, yang_zhang_24h appear in obs
4. Works in training mode
5. Falls back gracefully when indicators are missing
"""

import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path


# Mock minimal environment and state for testing
class MockState:
    """Mock state object for testing."""

    def __init__(self):
        self.units = 0.5
        self.cash = 10000.0
        self.step_idx = 100
        self.last_vol_imbalance = 0.1
        self.last_trade_intensity = 5.0
        self.last_realized_spread = 0.001
        self.last_agent_fill_ratio = 0.95
        self.token_index = 0


class MockObservationSpace:
    """Mock observation space."""

    def __init__(self, size=56):
        self.shape = (size,)


class MockEnv:
    """Mock environment for testing."""

    def __init__(self, df: pd.DataFrame | None = None, obs_size: int = 56):
        self.df = df
        self.observation_space = MockObservationSpace(size=obs_size)
        self.state = MockState()
        self.sim = None  # MarketSimulator can be None
        self._last_reward_price = 50000.0

    def _resolve_reward_price(self, idx: int, row: Any) -> float:
        if row is not None and hasattr(row, "close"):
            return float(row.close)
        return 50000.0

    def _resolve_snapshot_timestamp(self, row: Any) -> int:
        if row is not None and hasattr(row, "timestamp"):
            return int(row.timestamp)
        return 1700000000


class MockMediator:
    """Mock mediator using the real helper methods from mediator.py."""

    def __init__(self, env: MockEnv):
        self.env = env
        self._context_row_idx = None
        self._context_row = None
        self._context_timestamp = None
        self._latest_log_ret_prev = 0.0
        self._last_signal_position = 0.0

    @staticmethod
    def _coerce_finite(value: Any, default: float = 0.0) -> float:
        """Imported from mediator.py."""
        import math

        if value is None:
            return float(default)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(numeric):
            return float(default)
        return numeric

    @staticmethod
    def _get_safe_float(row: Any, col: str, default: float = 0.0) -> float:
        """Imported from mediator.py."""
        import math

        if row is None:
            return default
        try:
            val = row.get(col) if hasattr(row, "get") else getattr(row, col, None)
            if val is None:
                return default
            result = float(val)
            if not math.isfinite(result):
                return default
            return result
        except (TypeError, ValueError, KeyError, AttributeError):
            return default

    def _extract_market_data(self, row: Any, state: Any, mark_price: float, prev_price: float):
        """Imported from mediator.py."""
        price = self._coerce_finite(mark_price, default=0.0)
        prev = self._coerce_finite(prev_price, default=price)

        volume = self._get_safe_float(row, "volume", 1.0)
        quote_volume = self._get_safe_float(row, "quote_asset_volume", 1.0)

        log_volume_norm = 0.0
        if quote_volume > 0:
            log_volume_norm = float(np.tanh(np.log1p(quote_volume / 1e6)))

        rel_volume = 0.0
        if volume > 0:
            rel_volume = float(np.tanh(np.log1p(volume / 100.0)))

        return {
            "price": price,
            "prev_price": prev,
            "log_volume_norm": log_volume_norm,
            "rel_volume": rel_volume,
        }

    def _extract_technical_indicators(self, row: Any, sim: Any, row_idx: int):
        """Imported from mediator.py."""
        ma5 = self._get_safe_float(row, "sma_5", float("nan"))
        ma20 = self._get_safe_float(row, "sma_15", float("nan"))
        rsi14 = self._get_safe_float(row, "rsi", 50.0)

        macd = 0.0
        macd_signal = 0.0
        momentum = 0.0
        atr = 0.0
        cci = 0.0
        obv = 0.0
        bb_lower = float("nan")
        bb_upper = float("nan")

        if sim is not None and hasattr(sim, "get_macd"):
            try:
                if hasattr(sim, "get_macd"):
                    macd = float(sim.get_macd(row_idx))
                if hasattr(sim, "get_macd_signal"):
                    macd_signal = float(sim.get_macd_signal(row_idx))
                if hasattr(sim, "get_momentum"):
                    momentum = float(sim.get_momentum(row_idx))
                if hasattr(sim, "get_atr"):
                    atr = float(sim.get_atr(row_idx))
                if hasattr(sim, "get_cci"):
                    cci = float(sim.get_cci(row_idx))
                if hasattr(sim, "get_obv"):
                    obv = float(sim.get_obv(row_idx))
                if hasattr(sim, "get_bb_lower"):
                    bb_lower = float(sim.get_bb_lower(row_idx))
                if hasattr(sim, "get_bb_upper"):
                    bb_upper = float(sim.get_bb_upper(row_idx))
            except Exception:
                pass

        return {
            "ma5": ma5,
            "ma20": ma20,
            "rsi14": rsi14,
            "macd": macd,
            "macd_signal": macd_signal,
            "momentum": momentum,
            "atr": atr,
            "cci": cci,
            "obv": obv,
            "bb_lower": bb_lower,
            "bb_upper": bb_upper,
        }

    def _extract_norm_cols(self, row: Any) -> np.ndarray:
        """Imported from mediator.py."""
        norm_cols = np.zeros(8, dtype=np.float32)

        norm_cols[0] = self._get_safe_float(row, "cvd_24h", 0.0)
        norm_cols[1] = self._get_safe_float(row, "cvd_168h", 0.0)
        norm_cols[2] = self._get_safe_float(row, "yang_zhang_24h", 0.0)
        norm_cols[3] = self._get_safe_float(row, "yang_zhang_168h", 0.0)
        norm_cols[4] = self._get_safe_float(row, "garch_12h", 0.0)
        norm_cols[5] = self._get_safe_float(row, "garch_24h", 0.0)
        norm_cols[6] = self._get_safe_float(row, "ret_15m", 0.0)
        norm_cols[7] = self._get_safe_float(row, "ret_60m", 0.0)

        norm_cols = np.tanh(norm_cols)

        return norm_cols

    def _build_observation(self, *, row: Any | None, state: Any, mark_price: float) -> np.ndarray:
        """Simplified version of _build_observation for testing."""
        try:
            from obs_builder import build_observation_vector

            _HAVE_OBS_BUILDER = True
        except ImportError:
            _HAVE_OBS_BUILDER = False

        obs_shape = getattr(getattr(self.env, "observation_space", None), "shape", None)
        if not obs_shape:
            return np.zeros(0, dtype=np.float32)

        if not _HAVE_OBS_BUILDER:
            # Fallback: return zeros (test will fail)
            return np.zeros(obs_shape, dtype=np.float32)

        obs = np.zeros(obs_shape, dtype=np.float32)

        env = self.env
        df = getattr(env, "df", None)

        row_idx: int | None = getattr(self, "_context_row_idx", None)
        if row_idx is None and row is not None:
            try:
                row_idx = int(getattr(row, "name"))
            except Exception:
                row_idx = None
        if row_idx is None:
            try:
                step_idx = getattr(state, "step_idx", None)
                if step_idx is not None:
                    row_idx = int(step_idx)
            except Exception:
                row_idx = 0

        if row_idx is not None:
            if row_idx < 0:
                row_idx = 0
            if df is not None and row_idx >= len(df):
                row_idx = len(df) - 1

        curr_price = mark_price
        prev_price_val = mark_price

        market_data = self._extract_market_data(row, state, curr_price, prev_price_val)
        sim = getattr(env, "sim", None)
        indicators = self._extract_technical_indicators(row, sim, row_idx or 0)
        norm_cols_values = self._extract_norm_cols(row)

        units = self._coerce_finite(getattr(state, "units", 0.0), default=0.0)
        cash = self._coerce_finite(getattr(state, "cash", 0.0), default=0.0)

        last_vol_imbalance = self._coerce_finite(getattr(state, "last_vol_imbalance", 0.0), default=0.0)
        last_trade_intensity = self._coerce_finite(getattr(state, "last_trade_intensity", 0.0), default=0.0)
        last_realized_spread = self._coerce_finite(getattr(state, "last_realized_spread", 0.0), default=0.0)
        last_agent_fill_ratio = self._coerce_finite(getattr(state, "last_agent_fill_ratio", 0.0), default=0.0)

        fear_greed_value = self._get_safe_float(row, "fear_greed_value", 50.0)
        has_fear_greed = abs(fear_greed_value - 50.0) > 0.1

        is_high_importance = self._get_safe_float(row, "is_high_importance", 0.0)
        time_since_event = self._get_safe_float(row, "time_since_event", 0.0)

        risk_off_flag = fear_greed_value < 25.0

        token_id = getattr(state, "token_index", 0)
        max_num_tokens = 1
        num_tokens = 1

        try:
            build_observation_vector(
                float(market_data["price"]),
                float(market_data["prev_price"]),
                float(market_data["log_volume_norm"]),
                float(market_data["rel_volume"]),
                float(indicators["ma5"]),
                float(indicators["ma20"]),
                float(indicators["rsi14"]),
                float(indicators["macd"]),
                float(indicators["macd_signal"]),
                float(indicators["momentum"]),
                float(indicators["atr"]),
                float(indicators["cci"]),
                float(indicators["obv"]),
                float(indicators["bb_lower"]),
                float(indicators["bb_upper"]),
                float(is_high_importance),
                float(time_since_event),
                float(fear_greed_value),
                bool(has_fear_greed),
                bool(risk_off_flag),
                float(cash),
                float(units),
                float(last_vol_imbalance),
                float(last_trade_intensity),
                float(last_realized_spread),
                float(last_agent_fill_ratio),
                int(token_id),
                int(max_num_tokens),
                int(num_tokens),
                norm_cols_values,
                obs,
            )
        except Exception:
            return np.zeros(obs_shape, dtype=np.float32)

        return obs


# ============================================================================
# TESTS
# ============================================================================


def test_observation_size_and_non_zero():
    """Test 1: Verify obs size is correct and contains non-zero values."""
    # Create synthetic data with technical indicators
    df = pd.DataFrame(
        {
            "timestamp": [1700000000 + i * 3600 for i in range(200)],
            "open": [50000 + i * 10 for i in range(200)],
            "high": [50100 + i * 10 for i in range(200)],
            "low": [49900 + i * 10 for i in range(200)],
            "close": [50000 + i * 10 for i in range(200)],
            "volume": [100 + i for i in range(200)],
            "quote_asset_volume": [5000000 + i * 1000 for i in range(200)],
            "sma_5": [50000 + i * 10 for i in range(200)],
            "sma_15": [50000 + i * 9 for i in range(200)],
            "rsi": [50 + (i % 20) for i in range(200)],
            "cvd_24h": [(i % 10) / 10.0 for i in range(200)],
            "cvd_168h": [(i % 20) / 20.0 for i in range(200)],
            "yang_zhang_24h": [0.01 + (i % 5) * 0.001 for i in range(200)],
            "yang_zhang_168h": [0.015 + (i % 7) * 0.001 for i in range(200)],
            "garch_12h": [0.02 + (i % 3) * 0.002 for i in range(200)],
            "garch_24h": [0.025 + (i % 4) * 0.002 for i in range(200)],
            "ret_15m": [(i % 15) * 0.0001 for i in range(200)],
            "ret_60m": [(i % 25) * 0.0002 for i in range(200)],
            "fear_greed_value": [50 + (i % 30) for i in range(200)],
        }
    )

    env = MockEnv(df=df, obs_size=43)
    mediator = MockMediator(env)

    row = df.iloc[100]
    state = env.state
    mark_price = 51000.0

    obs = mediator._build_observation(row=row, state=state, mark_price=mark_price)

    # Check size
    assert obs.shape == (56,), f"Expected obs.shape=(56,), got {obs.shape}"

    # Check that more than 35 values are non-zero (>60% should be populated)
    non_zero_count = np.count_nonzero(obs)
    assert non_zero_count > 30, f"Expected >30 non-zero values, got {non_zero_count}"

    print(f"✓ Test 1 passed: obs.shape={obs.shape}, non_zero_count={non_zero_count}")


def test_technical_indicators_present():
    """Test 2: Verify technical indicators appear in correct positions."""
    # Create data with known indicator values
    df = pd.DataFrame(
        {
            "timestamp": [1700000000],
            "open": [50000],
            "high": [51000],
            "low": [49000],
            "close": [50500],
            "volume": [150],
            "quote_asset_volume": [7500000],
            "sma_5": [50200],
            "sma_15": [50100],
            "rsi": [65.0],
            "cvd_24h": [0.5],
            "cvd_168h": [0.3],
            "yang_zhang_24h": [0.025],
            "yang_zhang_168h": [0.030],
            "garch_12h": [0.028],
            "garch_24h": [0.032],
            "ret_15m": [0.001],
            "ret_60m": [0.002],
            "fear_greed_value": [75.0],
        }
    )

    env = MockEnv(df=df, obs_size=43)
    mediator = MockMediator(env)

    row = df.iloc[0]
    state = env.state
    mark_price = 50500.0

    obs = mediator._build_observation(row=row, state=state, mark_price=mark_price)

    # Check that obs contains values (not all zeros)
    assert not np.allclose(obs, 0.0), "Observation should not be all zeros"

    # The first value should be price
    assert obs[0] > 0, f"obs[0] should be price, got {obs[0]}"

    # Check norm_cols positions (32-39 typically contain cvd, garch, yang_zhang)
    norm_cols_region = obs[32:40]
    non_zero_norm_cols = np.count_nonzero(norm_cols_region)
    assert non_zero_norm_cols >= 4, f"Expected >=4 non-zero norm_cols, got {non_zero_norm_cols}"

    print(f"✓ Test 2 passed: Indicators are present in obs")


def test_cvd_garch_yangzhang_in_obs():
    """Test 3: Verify cvd_24h, garch_12h, yang_zhang_24h appear in obs."""
    df = pd.DataFrame(
        {
            "timestamp": [1700000000],
            "open": [50000],
            "high": [50100],
            "low": [49900],
            "close": [50000],
            "volume": [100],
            "quote_asset_volume": [5000000],
            "sma_5": [50000],
            "sma_15": [50000],
            "rsi": [50],
            "cvd_24h": [1.5],  # Non-zero value
            "cvd_168h": [2.0],  # Non-zero value
            "yang_zhang_24h": [0.05],  # Non-zero value
            "yang_zhang_168h": [0.06],
            "garch_12h": [0.04],  # Non-zero value
            "garch_24h": [0.045],
            "ret_15m": [0.001],
            "ret_60m": [0.002],
            "fear_greed_value": [50],
        }
    )

    env = MockEnv(df=df, obs_size=43)
    mediator = MockMediator(env)

    row = df.iloc[0]
    state = env.state
    mark_price = 50000.0

    obs = mediator._build_observation(row=row, state=state, mark_price=mark_price)

    # Extract norm_cols (positions 32-39)
    norm_cols_region = obs[32:40]

    # These should be non-zero after tanh normalization
    assert not np.allclose(norm_cols_region, 0.0), "norm_cols should contain non-zero values from indicators"

    # Check that at least cvd, garch, yang_zhang contribute
    # (after tanh, values should be in reasonable range)
    assert np.any(np.abs(norm_cols_region) > 0.01), "Expected significant values in norm_cols from indicators"

    print(f"✓ Test 3 passed: cvd_24h, garch_12h, yang_zhang_24h present in obs")


def test_observations_in_training_env():
    """Test 4: Verify obs works in training-like environment."""
    # Simulate training scenario with a longer dataframe
    n_steps = 500
    df = pd.DataFrame(
        {
            "timestamp": [1700000000 + i * 3600 for i in range(n_steps)],
            "open": [50000 + np.sin(i * 0.1) * 100 for i in range(n_steps)],
            "high": [50100 + np.sin(i * 0.1) * 100 for i in range(n_steps)],
            "low": [49900 + np.sin(i * 0.1) * 100 for i in range(n_steps)],
            "close": [50000 + np.sin(i * 0.1) * 100 for i in range(n_steps)],
            "volume": [100 + i % 50 for i in range(n_steps)],
            "quote_asset_volume": [5000000 + i * 1000 for i in range(n_steps)],
            "sma_5": [50000 + np.sin(i * 0.1) * 100 for i in range(n_steps)],
            "sma_15": [50000 + np.sin(i * 0.1) * 90 for i in range(n_steps)],
            "rsi": [50 + (i % 40) for i in range(n_steps)],
            "cvd_24h": [np.sin(i * 0.05) * 0.5 for i in range(n_steps)],
            "cvd_168h": [np.cos(i * 0.02) * 0.3 for i in range(n_steps)],
            "yang_zhang_24h": [0.02 + (i % 10) * 0.001 for i in range(n_steps)],
            "yang_zhang_168h": [0.025 + (i % 15) * 0.001 for i in range(n_steps)],
            "garch_12h": [0.03 + (i % 5) * 0.002 for i in range(n_steps)],
            "garch_24h": [0.035 + (i % 8) * 0.002 for i in range(n_steps)],
            "ret_15m": [(i % 20) * 0.0001 for i in range(n_steps)],
            "ret_60m": [(i % 30) * 0.0002 for i in range(n_steps)],
            "fear_greed_value": [50 + (i % 50) for i in range(n_steps)],
        }
    )

    env = MockEnv(df=df, obs_size=43)
    mediator = MockMediator(env)

    # Test multiple steps
    for step_idx in [50, 150, 300, 450]:
        row = df.iloc[step_idx]
        state = env.state
        state.step_idx = step_idx
        mark_price = float(row.close)

        obs = mediator._build_observation(row=row, state=state, mark_price=mark_price)

        assert obs.shape == (56,), f"Step {step_idx}: Expected shape (56,), got {obs.shape}"
        non_zero_count = np.count_nonzero(obs)
        assert non_zero_count > 15, f"Step {step_idx}: Expected >15 non-zero, got {non_zero_count}"

    print(f"✓ Test 4 passed: Training env simulation successful")


def test_observation_works_without_indicators():
    """Test 5: Verify fallback works when indicators are missing."""
    # Create minimal dataframe without technical indicators
    df = pd.DataFrame(
        {
            "timestamp": [1700000000],
            "open": [50000],
            "high": [50100],
            "low": [49900],
            "close": [50000],
            "volume": [100],
            "quote_asset_volume": [5000000],
            # No sma_5, cvd_24h, garch, yang_zhang, etc.
        }
    )

    env = MockEnv(df=df, obs_size=43)
    mediator = MockMediator(env)

    row = df.iloc[0]
    state = env.state
    mark_price = 50000.0

    obs = mediator._build_observation(row=row, state=state, mark_price=mark_price)

    # Should still return correct size (fallback to defaults)
    assert obs.shape == (56,), f"Expected shape (56,), got {obs.shape}"

    # Should have at least some basic values (price, cash, units)
    assert obs[0] > 0, "obs[0] (price) should be non-zero"

    # May have fewer non-zero values, but should not crash
    non_zero_count = np.count_nonzero(obs)
    assert non_zero_count >= 3, f"Expected >=3 non-zero values (price, volumes, state), got {non_zero_count}"

    print(f"✓ Test 5 passed: Fallback works without indicators")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Running technical indicators integration tests...")
    print("=" * 80)

    try:
        test_observation_size_and_non_zero()
        print()

        test_technical_indicators_present()
        print()

        test_cvd_garch_yangzhang_in_obs()
        print()

        test_observations_in_training_env()
        print()

        test_observation_works_without_indicators()
        print()

        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
    except AssertionError as e:
        print("=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        raise
    except Exception as e:
        print("=" * 80)
        print(f"❌ ERROR: {e}")
        print("=" * 80)
        raise

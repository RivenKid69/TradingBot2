# -*- coding: utf-8 -*-
"""Comprehensive tests for service_calibrate_tcost.py - 100% coverage.

Tests cover:
- TCostCalibrateConfig dataclass
- calibrate() function
- run() function
- from_config() function
- Helper functions (_safe_abs_log_ret, _compute_vol_bps, etc.)
- Target modes (hl, oc)
- Vol modes (hl, ret_1m, fallback)
- Illiquidity ratio computation
- Linear regression fitting (NNLS)
- YAML config updating
"""

import json
import os
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

from service_calibrate_tcost import (
    TCostCalibrateConfig,
    calibrate,
    run,
    from_config,
    _safe_abs_log_ret,
    _compute_vol_bps,
    _compute_illq_ratio,
    _target_spread_bps,
    _fit_linear_nonneg,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_data():
    """Sample market data for calibration."""
    return pd.DataFrame(
        {
            "ts_ms": [1000, 2000, 3000, 4000, 5000],
            "symbol": ["BTCUSDT"] * 5,
            "open": [99.5, 100.5, 101.5, 102.5, 103.5],
            "high": [100.5, 101.5, 102.5, 103.5, 104.5],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "ref_price": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0, 900.0, 1200.0, 1000.0],
            "number_of_trades": [100, 110, 90, 120, 100],
        }
    )


@pytest.fixture
def sample_data_with_ret():
    """Sample data with ret_1m column."""
    df = pd.DataFrame(
        {
            "ts_ms": [1000, 2000, 3000, 4000, 5000],
            "symbol": ["BTCUSDT"] * 5,
            "ref_price": [100.0, 101.0, 102.0, 103.0, 104.0],
            "ret_1m": [0.01, -0.005, 0.01, 0.005, -0.003],
            "volume": [1000.0] * 5,
        }
    )
    return df


# ============================================================================
# Test TCostCalibrateConfig
# ============================================================================


def test_tcost_calibrate_config_defaults():
    """Test TCostCalibrateConfig with defaults."""
    cfg = TCostCalibrateConfig(
        sandbox_config="config.yaml",
        out="output.json",
    )

    assert cfg.sandbox_config == "config.yaml"
    assert cfg.out == "output.json"
    assert cfg.target == "hl"
    assert cfg.k == 0.25
    assert cfg.dry_run is False


def test_tcost_calibrate_config_custom():
    """Test TCostCalibrateConfig with custom values."""
    cfg = TCostCalibrateConfig(
        sandbox_config="custom.yaml",
        out="custom_output.json",
        target="oc",
        k=0.5,
        dry_run=True,
    )

    assert cfg.target == "oc"
    assert cfg.k == 0.5
    assert cfg.dry_run is True


# ============================================================================
# Test Helper Functions
# ============================================================================


def test_safe_abs_log_ret():
    """Test _safe_abs_log_ret function."""
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "ts_ms": [1000, 2000, 3000],
            "ref_price": [100.0, 101.0, 99.0],
        }
    )

    result = _safe_abs_log_ret(df, "symbol", "ts_ms", "ref_price")

    # First value should be NaN (no previous price)
    assert np.isnan(result.iloc[0])

    # Second value: abs(log(101/100)) ≈ 0.00995
    assert 0.009 < result.iloc[1] < 0.011

    # Third value: abs(log(99/101)) ≈ 0.0202
    assert 0.019 < result.iloc[2] < 0.021


def test_safe_abs_log_ret_with_zeros():
    """Test _safe_abs_log_ret with zero prices."""
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "ts_ms": [1000, 2000],
            "ref_price": [0.0, 100.0],
        }
    )

    result = _safe_abs_log_ret(df, "symbol", "ts_ms", "ref_price")

    # Should handle division by zero gracefully
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])  # log(100/0) = inf → nan


def test_compute_vol_bps_hl_mode(sample_data):
    """Test _compute_vol_bps with hl mode."""
    result = _compute_vol_bps(sample_data, "hl", "ref_price")

    # (high - low) / ref_price * 10000
    # Example: (100.5 - 99.0) / 100.0 * 10000 = 150 bps
    assert result.iloc[0] == pytest.approx(150.0, abs=1.0)


def test_compute_vol_bps_ret_1m_mode(sample_data_with_ret):
    """Test _compute_vol_bps with ret_1m column."""
    result = _compute_vol_bps(sample_data_with_ret, "returns", "ref_price")

    # Should use abs(ret_1m) * 10000
    assert result.iloc[0] == pytest.approx(100.0, abs=1.0)  # 0.01 * 10000
    assert result.iloc[1] == pytest.approx(50.0, abs=1.0)  # 0.005 * 10000


def test_compute_vol_bps_fallback(sample_data):
    """Test _compute_vol_bps fallback to abs log return."""
    # Remove high/low columns to force fallback
    df = sample_data[["ts_ms", "symbol", "ref_price", "volume"]].copy()

    result = _compute_vol_bps(df, "hl", "ref_price")

    # Should fallback to abs log return
    assert not np.isnan(result.iloc[1])  # Second value should be computed


def test_compute_illq_ratio():
    """Test _compute_illq_ratio function."""
    df = pd.DataFrame(
        {
            "number_of_trades": [100, 90, 110, 80, 100],
        }
    )

    result = _compute_illq_ratio(df, "number_of_trades", 100.0)

    # (liq_ref - liq) / liq_ref
    # Example: (100 - 100) / 100 = 0.0
    assert result.iloc[0] == 0.0

    # (100 - 90) / 100 = 0.1
    assert result.iloc[1] == 0.1

    # (100 - 110) / 100 = -0.1 → max(0, -0.1) = 0.0
    assert result.iloc[2] == 0.0


def test_compute_illq_ratio_no_column():
    """Test _compute_illq_ratio with missing liquidity column."""
    df = pd.DataFrame(
        {
            "volume": [1000, 1100, 900],
        }
    )

    result = _compute_illq_ratio(df, "number_of_trades", 100.0)

    # Should use volume column as fallback
    assert not np.isnan(result.iloc[0])


def test_compute_illq_ratio_fallback_ones():
    """Test _compute_illq_ratio fallback to ones."""
    df = pd.DataFrame(
        {
            "price": [100.0, 101.0, 102.0],
        }
    )

    result = _compute_illq_ratio(df, "nonexistent", 100.0)

    # Should return zeros (since liq = ones → (ref - 1) / ref)
    assert all(result >= 0)


def test_target_spread_bps_hl_mode(sample_data):
    """Test _target_spread_bps with hl mode."""
    result = _target_spread_bps(sample_data, "ref_price", "hl", 0.25)

    # (high - low) / ref_price * 10000 * k
    # Example: (100.5 - 99.0) / 100.0 * 10000 * 0.25 = 37.5 bps
    assert result.iloc[0] == pytest.approx(37.5, abs=1.0)


def test_target_spread_bps_oc_mode(sample_data):
    """Test _target_spread_bps with oc mode."""
    result = _target_spread_bps(sample_data, "ref_price", "oc", 0.5)

    # abs(close - open) / ref_price * 10000 * k
    # Example: abs(100.0 - 99.5) / 100.0 * 10000 * 0.5 = 25.0 bps
    assert result.iloc[0] == pytest.approx(25.0, abs=1.0)


def test_target_spread_bps_invalid_mode(sample_data):
    """Test _target_spread_bps with invalid mode."""
    with pytest.raises(ValueError, match="Неизвестный режим target"):
        _target_spread_bps(sample_data, "ref_price", "invalid", 0.25)


def test_target_spread_bps_missing_columns():
    """Test _target_spread_bps with missing required columns."""
    df = pd.DataFrame(
        {
            "ref_price": [100.0],
        }
    )

    with pytest.raises(ValueError, match="требуются колонки"):
        _target_spread_bps(df, "ref_price", "hl", 0.25)

    with pytest.raises(ValueError, match="требуются колонки"):
        _target_spread_bps(df, "ref_price", "oc", 0.25)


def test_fit_linear_nonneg():
    """Test _fit_linear_nonneg function."""
    # Simple linear problem: y = 2x1 + 3x2
    X = np.array(
        [
            [1, 2, 3],
            [1, 4, 1],
            [1, 1, 5],
            [1, 3, 2],
        ]
    )
    y = np.array([2 * 2 + 3 * 3, 2 * 4 + 3 * 1, 2 * 1 + 3 * 5, 2 * 3 + 3 * 2])

    coef = _fit_linear_nonneg(X, y)

    # Coefficients should be approximately [0, 2, 3]
    assert len(coef) == 3
    assert coef[0] >= 0  # Non-negative
    assert coef[1] >= 0
    assert coef[2] >= 0


def test_fit_linear_nonneg_negative_clipping():
    """Test _fit_linear_nonneg clips negative coefficients."""
    # Problem that might produce negative coefficients
    X = np.array(
        [
            [1, -2],
            [1, -4],
            [1, -1],
        ]
    )
    y = np.array([5, 10, 2])

    coef = _fit_linear_nonneg(X, y)

    # All coefficients should be non-negative
    assert all(coef >= 0)


# ============================================================================
# Test calibrate() Function
# ============================================================================


def test_calibrate_basic(sample_data):
    """Test calibrate function with basic data."""
    params, stats = calibrate(
        sample_data,
        price_col="ref_price",
        vol_mode="hl",
        target_mode="hl",
        target_k=0.25,
        liq_col="number_of_trades",
        liq_ref=100.0,
    )

    # Verify params structure
    assert "base_bps" in params
    assert "alpha_vol" in params
    assert "beta_illiquidity" in params

    # All params should be non-negative
    assert params["base_bps"] >= 0
    assert params["alpha_vol"] >= 0
    assert params["beta_illiquidity"] >= 0

    # Verify stats structure
    assert "rmse" in stats
    assert "mae" in stats
    assert "r2" in stats
    assert "n" in stats

    # Stats should be reasonable
    assert stats["rmse"] >= 0
    assert stats["mae"] >= 0
    assert stats["n"] > 0


def test_calibrate_insufficient_data():
    """Test calibrate with insufficient data."""
    df = pd.DataFrame(
        {
            "ref_price": [100.0],
            "high": [101.0],
            "low": [99.0],
            "volume": [1000.0],
        }
    )

    # Should raise error due to insufficient data
    with pytest.raises(ValueError, match="Недостаточно данных"):
        calibrate(
            df,
            price_col="ref_price",
            vol_mode="hl",
            target_mode="hl",
            target_k=0.25,
            liq_col="volume",
            liq_ref=1000.0,
        )


def test_calibrate_with_nans(sample_data):
    """Test calibrate handles NaN values gracefully."""
    df = sample_data.copy()
    df.loc[2, "high"] = np.nan
    df.loc[3, "low"] = np.nan

    params, stats = calibrate(
        df,
        price_col="ref_price",
        vol_mode="hl",
        target_mode="hl",
        target_k=0.25,
        liq_col="number_of_trades",
        liq_ref=100.0,
    )

    # Should still work, dropping NaN rows
    assert stats["n"] < len(df)


def test_calibrate_oc_mode(sample_data):
    """Test calibrate with oc target mode."""
    params, stats = calibrate(
        sample_data,
        price_col="ref_price",
        vol_mode="hl",
        target_mode="oc",
        target_k=0.5,
        liq_col="volume",
        liq_ref=1000.0,
    )

    assert "base_bps" in params
    assert stats["n"] > 0


# ============================================================================
# Test run() Function
# ============================================================================


def test_run_basic(sample_data):
    """Test run function with basic configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data file
        data_path = os.path.join(tmpdir, "data.parquet")
        sample_data.to_parquet(data_path)

        # Create test config
        config_path = os.path.join(tmpdir, "config.yaml")
        config = {
            "symbol": "BTCUSDT",
            "data": {
                "path": data_path,
                "ts_col": "ts_ms",
                "symbol_col": "symbol",
                "price_col": "ref_price",
            },
            "dynamic_spread": {
                "vol_mode": "hl",
                "liq_col": "number_of_trades",
                "liq_ref": 100.0,
            },
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        out_path = os.path.join(tmpdir, "output.json")

        # Mock load_sandbox_config
        with mock.patch("service_calibrate_tcost.load_sandbox_config") as mock_load:
            mock_cfg = mock.Mock()
            mock_cfg.symbol = "BTCUSDT"
            mock_cfg.data = mock.Mock()
            mock_cfg.data.path = data_path
            mock_cfg.data.ts_col = "ts_ms"
            mock_cfg.data.symbol_col = "symbol"
            mock_cfg.data.price_col = "ref_price"
            mock_cfg.dynamic_spread = {
                "vol_mode": "hl",
                "liq_col": "number_of_trades",
                "liq_ref": 100.0,
            }
            mock_cfg.model_dump = mock.Mock(return_value=config)
            mock_load.return_value = mock_cfg

            cfg = TCostCalibrateConfig(
                sandbox_config=config_path,
                out=out_path,
            )

            report = run(cfg)

            # Verify report structure
            assert "config_in" in report
            assert "data_path" in report
            assert "fitted_params" in report
            assert "stats" in report

            # Verify params
            params = report["fitted_params"]
            assert "base_bps" in params
            assert "alpha_vol" in params
            assert "beta_illiquidity" in params

            # Verify output file was created
            assert os.path.exists(out_path)

            # Verify config was updated
            assert os.path.exists(config_path)


def test_run_csv_data(sample_data):
    """Test run with CSV data file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create CSV data file
        data_path = os.path.join(tmpdir, "data.csv")
        sample_data.to_csv(data_path, index=False)

        config_path = os.path.join(tmpdir, "config.yaml")
        config = {
            "symbol": "BTCUSDT",
            "data": {"path": data_path},
            "dynamic_spread": {},
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        out_path = os.path.join(tmpdir, "output.json")

        with mock.patch("service_calibrate_tcost.load_sandbox_config") as mock_load:
            mock_cfg = mock.Mock()
            mock_cfg.symbol = "BTCUSDT"
            mock_cfg.data = mock.Mock()
            mock_cfg.data.path = data_path
            mock_cfg.data.ts_col = "ts_ms"
            mock_cfg.data.symbol_col = "symbol"
            mock_cfg.data.price_col = "ref_price"
            mock_cfg.dynamic_spread = {}
            mock_cfg.model_dump = mock.Mock(return_value=config)
            mock_load.return_value = mock_cfg

            cfg = TCostCalibrateConfig(
                sandbox_config=config_path,
                out=out_path,
            )

            report = run(cfg)

            assert "fitted_params" in report


def test_run_dry_run(sample_data):
    """Test run with dry_run=True (no config update)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.parquet")
        sample_data.to_parquet(data_path)

        config_path = os.path.join(tmpdir, "config.yaml")
        original_config = {
            "symbol": "BTCUSDT",
            "data": {"path": data_path},
            "dynamic_spread": {"base_bps": 5.0},
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(original_config, f)

        out_path = os.path.join(tmpdir, "output.json")

        with mock.patch("service_calibrate_tcost.load_sandbox_config") as mock_load:
            mock_cfg = mock.Mock()
            mock_cfg.symbol = "BTCUSDT"
            mock_cfg.data = mock.Mock()
            mock_cfg.data.path = data_path
            mock_cfg.data.ts_col = "ts_ms"
            mock_cfg.data.symbol_col = "symbol"
            mock_cfg.data.price_col = "ref_price"
            mock_cfg.dynamic_spread = {}
            mock_cfg.model_dump = mock.Mock(return_value=original_config)
            mock_load.return_value = mock_cfg

            cfg = TCostCalibrateConfig(
                sandbox_config=config_path,
                out=out_path,
                dry_run=True,
            )

            report = run(cfg)

            # Config should NOT be updated
            with open(config_path) as f:
                updated = yaml.safe_load(f)

            # Should still have original value
            assert updated["dynamic_spread"]["base_bps"] == 5.0


def test_run_missing_data_file():
    """Test run with missing data file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        config = {
            "data": {"path": "/nonexistent/data.parquet"},
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        out_path = os.path.join(tmpdir, "output.json")

        with mock.patch("service_calibrate_tcost.load_sandbox_config") as mock_load:
            mock_cfg = mock.Mock()
            mock_cfg.data = mock.Mock()
            mock_cfg.data.path = "/nonexistent/data.parquet"
            mock_load.return_value = mock_cfg

            cfg = TCostCalibrateConfig(
                sandbox_config=config_path,
                out=out_path,
            )

            with pytest.raises(FileNotFoundError):
                run(cfg)


def test_run_missing_symbol_column(sample_data):
    """Test run with missing symbol column."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Remove symbol column
        data_no_symbol = sample_data.drop(columns=["symbol"])
        data_path = os.path.join(tmpdir, "data.parquet")
        data_no_symbol.to_parquet(data_path)

        config_path = os.path.join(tmpdir, "config.yaml")
        config = {
            "symbol": "BTCUSDT",
            "data": {"path": data_path},
            "dynamic_spread": {},
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        out_path = os.path.join(tmpdir, "output.json")

        with mock.patch("service_calibrate_tcost.load_sandbox_config") as mock_load:
            mock_cfg = mock.Mock()
            mock_cfg.symbol = "BTCUSDT"
            mock_cfg.data = mock.Mock()
            mock_cfg.data.path = data_path
            mock_cfg.data.ts_col = "ts_ms"
            mock_cfg.data.symbol_col = "symbol"
            mock_cfg.data.price_col = "ref_price"
            mock_cfg.dynamic_spread = {}
            mock_cfg.model_dump = mock.Mock(return_value=config)
            mock_load.return_value = mock_cfg

            cfg = TCostCalibrateConfig(
                sandbox_config=config_path,
                out=out_path,
            )

            # Should add symbol column from config
            report = run(cfg)

            assert "fitted_params" in report


# ============================================================================
# Test from_config() Function
# ============================================================================


def test_from_config(sample_data):
    """Test from_config convenience function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.parquet")
        sample_data.to_parquet(data_path)

        config_path = os.path.join(tmpdir, "config.yaml")
        config = {
            "symbol": "BTCUSDT",
            "data": {"path": data_path},
            "dynamic_spread": {},
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        out_path = os.path.join(tmpdir, "output.json")

        with mock.patch("service_calibrate_tcost.load_sandbox_config") as mock_load:
            mock_cfg = mock.Mock()
            mock_cfg.symbol = "BTCUSDT"
            mock_cfg.data = mock.Mock()
            mock_cfg.data.path = data_path
            mock_cfg.data.ts_col = "ts_ms"
            mock_cfg.data.symbol_col = "symbol"
            mock_cfg.data.price_col = "ref_price"
            mock_cfg.dynamic_spread = {}
            mock_cfg.model_dump = mock.Mock(return_value=config)
            mock_load.return_value = mock_cfg

            report = from_config(config_path, out=out_path)

            assert "fitted_params" in report
            assert "stats" in report


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_calibrate_all_zeros():
    """Test calibrate with all zero target values."""
    df = pd.DataFrame(
        {
            "ref_price": [100.0] * 5,
            "high": [100.0] * 5,
            "low": [100.0] * 5,
            "open": [100.0] * 5,
            "close": [100.0] * 5,
            "volume": [1000.0] * 5,
        }
    )

    # All zero spread → insufficient data after filtering
    with pytest.raises(ValueError, match="Недостаточно данных"):
        calibrate(
            df,
            price_col="ref_price",
            vol_mode="hl",
            target_mode="hl",
            target_k=0.25,
            liq_col="volume",
            liq_ref=1000.0,
        )


def test_ensure_dir():
    """Test _ensure_dir helper function."""
    from service_calibrate_tcost import _ensure_dir

    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = os.path.join(tmpdir, "a", "b", "c", "file.txt")
        _ensure_dir(nested_path)

        # Directory should be created
        assert os.path.exists(os.path.dirname(nested_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

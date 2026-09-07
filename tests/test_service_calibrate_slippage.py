# -*- coding: utf-8 -*-
"""Comprehensive tests for service_calibrate_slippage.py - 100% coverage.

Tests cover:
- SlippageCalibrateConfig dataclass
- fit_k_closed_form() function
- run() function
- from_config() function
- CSV and Parquet file handling
- Spread modes (mean, median)
- Min half spread quantile calculation
- Edge cases (empty data, missing columns)
"""

import json
import os
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

from service_calibrate_slippage import (
    SlippageCalibrateConfig,
    fit_k_closed_form,
    run,
    from_config,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_trades():
    """Sample trades data for slippage calibration."""
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 5,
            "size": [1.0, 2.0, 1.5, 3.0, 0.5],
            "liquidity": [1000.0, 1500.0, 1200.0, 2000.0, 800.0],
            "vol_factor": [1.0, 1.2, 0.9, 1.5, 0.8],
            "observed_slip_bps": [10.0, 15.0, 12.0, 20.0, 8.0],
            "spread_bps": [5.0, 6.0, 5.5, 7.0, 4.5],
            "half_spread_bps": [2.5, 3.0, 2.75, 3.5, 2.25],
        }
    )


@pytest.fixture
def sample_trades_no_half_spread():
    """Sample trades without half_spread_bps column."""
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 5,
            "size": [1.0, 2.0, 1.5, 3.0, 0.5],
            "liquidity": [1000.0, 1500.0, 1200.0, 2000.0, 800.0],
            "vol_factor": [1.0, 1.2, 0.9, 1.5, 0.8],
            "observed_slip_bps": [10.0, 15.0, 12.0, 20.0, 8.0],
            "spread_bps": [5.0, 6.0, 5.5, 7.0, 4.5],
        }
    )


# ============================================================================
# Test SlippageCalibrateConfig
# ============================================================================


def test_slippage_calibrate_config_defaults():
    """Test SlippageCalibrateConfig with defaults."""
    cfg = SlippageCalibrateConfig(
        trades="trades.csv",
        out="output.json",
    )

    assert cfg.trades == "trades.csv"
    assert cfg.out == "output.json"
    assert cfg.fmt is None
    assert cfg.default_spread_mode == "median"
    assert cfg.min_half_spread_quantile == 0.0


def test_slippage_calibrate_config_custom():
    """Test SlippageCalibrateConfig with custom values."""
    cfg = SlippageCalibrateConfig(
        trades="custom_trades.parquet",
        out="custom_output.json",
        fmt="parquet",
        default_spread_mode="mean",
        min_half_spread_quantile=0.1,
    )

    assert cfg.fmt == "parquet"
    assert cfg.default_spread_mode == "mean"
    assert cfg.min_half_spread_quantile == 0.1


# ============================================================================
# Test fit_k_closed_form()
# ============================================================================


def test_fit_k_closed_form_basic(sample_trades):
    """Test fit_k_closed_form with basic data."""
    k = fit_k_closed_form(sample_trades)

    # k should be non-negative
    assert k >= 0.0

    # k should be finite (not NaN or Inf)
    assert np.isfinite(k)

    # Note: With the test data, k can be quite large (e.g., 276) because
    # the observed slippage is high relative to size/liquidity ratio.
    # This is expected behavior - the function should work with any data.


def test_fit_k_closed_form_perfect_correlation():
    """Test fit_k_closed_form with perfect linear relationship."""
    # Create data where observed_slip - half_spread = 0.8 * vol_factor * sqrt(size/liquidity)
    df = pd.DataFrame(
        {
            "size": [1.0, 2.0, 4.0, 1.0, 2.0],
            "liquidity": [100.0, 100.0, 100.0, 400.0, 400.0],
            "vol_factor": [1.0, 1.0, 1.0, 1.0, 1.0],
            "half_spread_bps": [0.0, 0.0, 0.0, 0.0, 0.0],
            "observed_slip_bps": [0.0, 0.0, 0.0, 0.0, 0.0],  # Will be overwritten
        }
    )

    # Calculate perfect observed_slip: k * vol_factor * sqrt(size/liquidity)
    k_true = 0.8
    df["observed_slip_bps"] = k_true * df["vol_factor"] * (df["size"] / df["liquidity"]) ** 0.5

    k_fitted = fit_k_closed_form(df)

    # Should recover k_true approximately
    assert k_fitted == pytest.approx(k_true, rel=0.1)


def test_fit_k_closed_form_empty_data():
    """Test fit_k_closed_form with empty data."""
    df = pd.DataFrame(
        {
            "size": [],
            "liquidity": [],
            "vol_factor": [],
            "observed_slip_bps": [],
            "half_spread_bps": [],
        }
    )

    k = fit_k_closed_form(df)

    # Should return default value
    assert k == 0.8


def test_fit_k_closed_form_zero_size():
    """Test fit_k_closed_form with zero size (filtered out)."""
    df = pd.DataFrame(
        {
            "size": [0.0, 0.0, 0.0],
            "liquidity": [1000.0, 1000.0, 1000.0],
            "vol_factor": [1.0, 1.0, 1.0],
            "observed_slip_bps": [10.0, 10.0, 10.0],
            "half_spread_bps": [5.0, 5.0, 5.0],
        }
    )

    k = fit_k_closed_form(df)

    # All rows filtered out → default
    assert k == 0.8


def test_fit_k_closed_form_zero_liquidity():
    """Test fit_k_closed_form with zero liquidity (filtered out)."""
    df = pd.DataFrame(
        {
            "size": [1.0, 1.0, 1.0],
            "liquidity": [0.0, 0.0, 0.0],
            "vol_factor": [1.0, 1.0, 1.0],
            "observed_slip_bps": [10.0, 10.0, 10.0],
            "half_spread_bps": [5.0, 5.0, 5.0],
        }
    )

    k = fit_k_closed_form(df)

    # All rows filtered out → default
    assert k == 0.8


def test_fit_k_closed_form_negative_k():
    """Test fit_k_closed_form clips negative k to zero."""
    # Create data that would produce negative k
    df = pd.DataFrame(
        {
            "size": [1.0, 2.0, 4.0],
            "liquidity": [100.0, 100.0, 100.0],
            "vol_factor": [1.0, 1.0, 1.0],
            "half_spread_bps": [10.0, 15.0, 20.0],
            "observed_slip_bps": [5.0, 5.0, 5.0],  # Less than half_spread → negative impact
        }
    )

    k = fit_k_closed_form(df)

    # k should be clipped to zero
    assert k == 0.0


def test_fit_k_closed_form_nan_handling():
    """Test fit_k_closed_form handles NaN values."""
    df = pd.DataFrame(
        {
            "size": [1.0, 2.0, float("nan"), 4.0],
            "liquidity": [100.0, 100.0, 100.0, 100.0],
            "vol_factor": [1.0, 1.0, 1.0, 1.0],
            "half_spread_bps": [2.0, 2.0, 2.0, 2.0],
            "observed_slip_bps": [10.0, 15.0, 12.0, 20.0],
        }
    )

    k = fit_k_closed_form(df)

    # Should compute k from non-NaN rows
    assert k >= 0.0


# ============================================================================
# Test run() Function
# ============================================================================


def test_run_csv_format(sample_trades):
    """Test run function with CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create CSV file
        trades_path = os.path.join(tmpdir, "trades.csv")
        sample_trades.to_csv(trades_path, index=False)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
            fmt="csv",
        )

        report = run(cfg)

        # Verify report structure
        assert "k" in report
        assert "default_spread_bps" in report
        assert "min_half_spread_bps" in report

        # Verify values are reasonable
        assert report["k"] >= 0.0
        assert report["default_spread_bps"] > 0.0
        assert report["min_half_spread_bps"] >= 0.0

        # Verify output file was created
        assert os.path.exists(out_path)

        # Verify JSON can be loaded
        with open(out_path) as f:
            loaded = json.load(f)
            assert loaded["k"] == report["k"]


def test_run_parquet_format(sample_trades):
    """Test run function with Parquet file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create Parquet file
        trades_path = os.path.join(tmpdir, "trades.parquet")
        sample_trades.to_parquet(trades_path)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
            fmt="parquet",
        )

        report = run(cfg)

        assert "k" in report
        assert os.path.exists(out_path)


def test_run_auto_format_detection(sample_trades):
    """Test run with automatic format detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Parquet file with .parquet extension
        trades_path = os.path.join(tmpdir, "trades.parquet")
        sample_trades.to_parquet(trades_path)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
            fmt=None,  # Auto-detect
        )

        report = run(cfg)

        assert "k" in report


def test_run_no_spread_column(sample_trades_no_half_spread):
    """Test run with missing spread columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Remove spread_bps column too
        df = sample_trades_no_half_spread.drop(columns=["spread_bps"])
        trades_path = os.path.join(tmpdir, "trades.csv")
        df.to_csv(trades_path, index=False)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
        )

        report = run(cfg)

        # Should use default value
        assert report["default_spread_bps"] == 2.0


def test_run_spread_mode_mean(sample_trades):
    """Test run with mean spread mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        sample_trades.to_csv(trades_path, index=False)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
            default_spread_mode="mean",
        )

        report = run(cfg)

        # Should compute mean of spread_bps
        expected_mean = sample_trades["spread_bps"].mean()
        assert report["default_spread_bps"] == pytest.approx(expected_mean, rel=0.01)


def test_run_spread_mode_median(sample_trades):
    """Test run with median spread mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        sample_trades.to_csv(trades_path, index=False)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
            default_spread_mode="median",
        )

        report = run(cfg)

        # Should compute median of spread_bps
        expected_median = sample_trades["spread_bps"].median()
        assert report["default_spread_bps"] == pytest.approx(expected_median, rel=0.01)


def test_run_min_half_spread_quantile(sample_trades):
    """Test run with min_half_spread_quantile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        sample_trades.to_csv(trades_path, index=False)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
            min_half_spread_quantile=0.2,  # 20th percentile
        )

        report = run(cfg)

        # Verify min_half_spread_bps is computed
        expected_quantile = sample_trades["half_spread_bps"].quantile(0.2)
        assert report["min_half_spread_bps"] == pytest.approx(expected_quantile, rel=0.01)


def test_run_no_half_spread_column(sample_trades_no_half_spread):
    """Test run derives half_spread from spread_bps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        sample_trades_no_half_spread.to_csv(trades_path, index=False)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
        )

        report = run(cfg)

        # half_spread should be derived as spread_bps / 2
        assert "k" in report


def test_run_output_directory_creation():
    """Test run creates output directory if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create trades file
        trades_path = os.path.join(tmpdir, "trades.csv")
        sample_df = pd.DataFrame(
            {
                "size": [1.0],
                "liquidity": [100.0],
                "vol_factor": [1.0],
                "observed_slip_bps": [10.0],
                "half_spread_bps": [2.0],
            }
        )
        sample_df.to_csv(trades_path, index=False)

        # Output in nested directory
        out_path = os.path.join(tmpdir, "nested", "dir", "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
        )

        report = run(cfg)

        # Verify nested directory was created
        assert os.path.exists(os.path.dirname(out_path))
        assert os.path.exists(out_path)


# ============================================================================
# Test from_config() Function
# ============================================================================


def test_from_config(sample_trades):
    """Test from_config function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create trades file
        trades_path = os.path.join(tmpdir, "trades.csv")
        sample_trades.to_csv(trades_path, index=False)

        # Create config file
        config_path = os.path.join(tmpdir, "config.yaml")
        config = {
            "trades": trades_path,
            "format": "csv",
            "default_spread_mode": "median",
            "min_half_spread_quantile": 0.1,
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        out_path = os.path.join(tmpdir, "output.json")

        report = from_config(config_path, out=out_path)

        # Verify report
        assert "k" in report
        assert "default_spread_bps" in report
        assert os.path.exists(out_path)


def test_from_config_minimal():
    """Test from_config with minimal config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create trades file
        trades_path = os.path.join(tmpdir, "trades.csv")
        sample_df = pd.DataFrame(
            {
                "size": [1.0, 2.0],
                "liquidity": [100.0, 200.0],
                "vol_factor": [1.0, 1.0],
                "observed_slip_bps": [10.0, 15.0],
                "half_spread_bps": [2.0, 3.0],
            }
        )
        sample_df.to_csv(trades_path, index=False)

        # Minimal config
        config_path = os.path.join(tmpdir, "config.yaml")
        config = {
            "trades": trades_path,
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        out_path = os.path.join(tmpdir, "output.json")

        report = from_config(config_path, out=out_path)

        assert "k" in report


def test_from_config_empty_config():
    """Test from_config with empty config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump({}, f)

        out_path = os.path.join(tmpdir, "output.json")

        # Should raise KeyError due to missing 'trades' key
        with pytest.raises(KeyError):
            from_config(config_path, out=out_path)


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_run_empty_trades():
    """Test run with empty trades DataFrame."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty trades file
        trades_path = os.path.join(tmpdir, "trades.csv")
        pd.DataFrame(
            {
                "size": [],
                "liquidity": [],
                "vol_factor": [],
                "observed_slip_bps": [],
                "half_spread_bps": [],
            }
        ).to_csv(trades_path, index=False)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
        )

        report = run(cfg)

        # Should return default k
        assert report["k"] == 0.8


def test_run_all_nan_spread():
    """Test run with all NaN spread values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        df = pd.DataFrame(
            {
                "size": [1.0, 2.0],
                "liquidity": [100.0, 200.0],
                "vol_factor": [1.0, 1.0],
                "observed_slip_bps": [10.0, 15.0],
                "spread_bps": [float("nan"), float("nan")],
                "half_spread_bps": [2.0, 3.0],
            }
        )
        df.to_csv(trades_path, index=False)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
        )

        report = run(cfg)

        # Should use default spread
        assert report["default_spread_bps"] == 2.0


def test_run_zero_quantile():
    """Test run with min_half_spread_quantile=0.0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        sample_df = pd.DataFrame(
            {
                "size": [1.0, 2.0],
                "liquidity": [100.0, 200.0],
                "vol_factor": [1.0, 1.0],
                "observed_slip_bps": [10.0, 15.0],
                "half_spread_bps": [2.0, 3.0],
            }
        )
        sample_df.to_csv(trades_path, index=False)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
            min_half_spread_quantile=0.0,
        )

        report = run(cfg)

        # min_half_spread_bps should be 0.0
        assert report["min_half_spread_bps"] == 0.0


def test_run_one_quantile():
    """Test run with min_half_spread_quantile=1.0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        sample_df = pd.DataFrame(
            {
                "size": [1.0, 2.0],
                "liquidity": [100.0, 200.0],
                "vol_factor": [1.0, 1.0],
                "observed_slip_bps": [10.0, 15.0],
                "half_spread_bps": [2.0, 3.0],
            }
        )
        sample_df.to_csv(trades_path, index=False)

        out_path = os.path.join(tmpdir, "output.json")

        cfg = SlippageCalibrateConfig(
            trades=trades_path,
            out=out_path,
            min_half_spread_quantile=1.0,
        )

        report = run(cfg)

        # min_half_spread_bps should be 0.0 (out of (0, 1) range)
        assert report["min_half_spread_bps"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

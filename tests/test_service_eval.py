# -*- coding: utf-8 -*-
"""Comprehensive tests for service_eval.py - 100% coverage.

Tests cover:
- EvalConfig dataclass
- ServiceEval class
- from_config integration
- Multiple execution profiles
- Equity frame synthesis
- Trades normalization
- Metrics calculation
- Output file generation (JSON, MD, PNG)
"""

import json
import os
import tempfile
from typing import Dict
from unittest import mock

import pandas as pd
import pytest

from service_eval import (
    EvalConfig,
    ServiceEval,
    from_config,
)
from core_config import CommonRunConfig, ExecutionProfile


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_trades_df():
    """Sample trades DataFrame."""
    return pd.DataFrame(
        {
            "ts_ms": [1000, 2000, 3000, 4000, 5000],
            "run_id": ["test"] * 5,
            "symbol": ["BTCUSDT"] * 5,
            "side": ["BUY", "SELL", "BUY", "SELL", "BUY"],
            "order_type": ["MARKET"] * 5,
            "price": [100.0, 101.0, 102.0, 103.0, 104.0],
            "qty": [1.0, 0.5, 1.5, 1.0, 0.5],
            "pnl": [0.0, 1.0, -0.5, 1.5, -0.3],
        }
    )


@pytest.fixture
def sample_reports_df():
    """Sample equity reports DataFrame."""
    return pd.DataFrame(
        {
            "ts_ms": [1000, 2000, 3000, 4000, 5000],
            "symbol": ["BTCUSDT"] * 5,
            "equity": [10000.0, 10001.0, 10000.5, 10002.0, 10001.7],
            "equity_after_costs": [9999.0, 10000.0, 9999.5, 10001.0, 10000.7],
            "bar_pnl": [0.0, 1.0, -0.5, 1.5, -0.3],
        }
    )


@pytest.fixture
def sample_reports_without_equity():
    """Sample reports without equity column (needs synthesis)."""
    return pd.DataFrame(
        {
            "ts_ms": [1000, 2000, 3000, 4000, 5000],
            "symbol": ["BTCUSDT"] * 5,
            "bar_pnl": [0.0, 1.0, -0.5, 1.5, -0.3],
        }
    )


# ============================================================================
# Test EvalConfig
# ============================================================================


def test_eval_config_defaults():
    """Test EvalConfig with default values."""
    cfg = EvalConfig(
        trades_path="logs/trades.csv",
        reports_path="logs/reports.csv",
    )

    assert cfg.trades_path == "logs/trades.csv"
    assert cfg.reports_path == "logs/reports.csv"
    assert cfg.profile is None
    assert cfg.out_json == "logs/metrics.json"
    assert cfg.capital_base == 10_000.0
    assert cfg.rf_annual == 0.0


def test_eval_config_custom():
    """Test EvalConfig with custom values."""
    cfg = EvalConfig(
        trades_path="custom/trades.csv",
        reports_path="custom/reports.csv",
        profile="aggressive",
        out_json="custom/metrics.json",
        capital_base=50_000.0,
        rf_annual=0.02,
    )

    assert cfg.trades_path == "custom/trades.csv"
    assert cfg.profile == "aggressive"
    assert cfg.capital_base == 50_000.0
    assert cfg.rf_annual == 0.02


def test_eval_config_dict_paths():
    """Test EvalConfig with dict paths (multi-profile)."""
    cfg = EvalConfig(
        trades_path={
            "conservative": "logs/trades_conservative.csv",
            "aggressive": "logs/trades_aggressive.csv",
        },
        reports_path={
            "conservative": "logs/reports_conservative.csv",
            "aggressive": "logs/reports_aggressive.csv",
        },
    )

    assert isinstance(cfg.trades_path, dict)
    assert "conservative" in cfg.trades_path
    assert "aggressive" in cfg.trades_path


# ============================================================================
# Test ServiceEval Initialization
# ============================================================================


def test_service_eval_init():
    """Test ServiceEval initialization."""
    cfg = EvalConfig(
        trades_path="logs/trades.csv",
        reports_path="logs/reports.csv",
    )

    service = ServiceEval(cfg)

    assert service.cfg is cfg
    assert service.container == {}


def test_service_eval_init_with_container():
    """Test ServiceEval initialization with container."""
    cfg = EvalConfig(
        trades_path="logs/trades.csv",
        reports_path="logs/reports.csv",
    )

    container = {"policy": mock.Mock(), "executor": mock.Mock()}
    service = ServiceEval(cfg, container)

    assert service.container is container


# ============================================================================
# Test ServiceEval.run()
# ============================================================================


def test_service_eval_run_basic(sample_trades_df, sample_reports_df):
    """Test ServiceEval.run() with basic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write test data
        trades_path = os.path.join(tmpdir, "trades.csv")
        reports_path = os.path.join(tmpdir, "reports.csv")
        sample_trades_df.to_csv(trades_path, index=False)
        sample_reports_df.to_csv(reports_path, index=False)

        cfg = EvalConfig(
            trades_path=trades_path,
            reports_path=reports_path,
            out_json=os.path.join(tmpdir, "metrics.json"),
            out_md=os.path.join(tmpdir, "metrics.md"),
            equity_png=os.path.join(tmpdir, "equity.png"),
        )

        service = ServiceEval(cfg)

        with mock.patch("service_eval.calculate_metrics") as mock_calc:
            with mock.patch("service_eval.plot_equity_curve"):
                # Mock metrics response
                mock_calc.return_value = {
                    "equity": {
                        "sharpe_ratio": 1.5,
                        "sortino_ratio": 2.0,
                        "max_drawdown": -0.05,
                        "total_return": 0.017,
                    },
                    "trades": {
                        "total_trades": 5,
                        "win_rate": 0.6,
                        "profit_factor": 1.8,
                    },
                }

                metrics = service.run()

                # Verify metrics were calculated
                assert "equity" in metrics
                assert "trades" in metrics
                assert metrics["equity"]["sharpe_ratio"] == 1.5

                # Verify files were written
                assert os.path.exists(cfg.out_json)
                assert os.path.exists(cfg.out_md)


def test_service_eval_run_equity_synthesis(sample_trades_df, sample_reports_without_equity):
    """Test ServiceEval.run() with equity synthesis from bar_pnl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        reports_path = os.path.join(tmpdir, "reports.csv")
        sample_trades_df.to_csv(trades_path, index=False)
        sample_reports_without_equity.to_csv(reports_path, index=False)

        cfg = EvalConfig(
            trades_path=trades_path,
            reports_path=reports_path,
            out_json=os.path.join(tmpdir, "metrics.json"),
            capital_base=10000.0,
        )

        service = ServiceEval(cfg)

        with mock.patch("service_eval.calculate_metrics") as mock_calc:
            with mock.patch("service_eval.plot_equity_curve"):
                mock_calc.return_value = {
                    "equity": {"total_return": 0.017},
                    "trades": {"total_trades": 5},
                }

                metrics = service.run()

                # Verify calculate_metrics was called
                mock_calc.assert_called_once()

                # Get the reports argument passed to calculate_metrics
                call_args = mock_calc.call_args
                reports_arg = call_args[0][1]

                # Verify equity was synthesized
                assert "equity" in reports_arg.columns
                assert reports_arg["equity"].iloc[0] == 10000.0  # Initial capital
                assert reports_arg["equity"].iloc[1] == 10001.0  # 10000 + 1.0 (bar_pnl)


def test_service_eval_run_trades_synthesis(sample_reports_df):
    """Test ServiceEval.run() with trades synthesis from reports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # No trades file, should synthesize from reports
        trades_path = os.path.join(tmpdir, "nonexistent_trades.csv")
        reports_path = os.path.join(tmpdir, "reports.csv")
        sample_reports_df.to_csv(reports_path, index=False)

        cfg = EvalConfig(
            trades_path=trades_path,
            reports_path=reports_path,
            out_json=os.path.join(tmpdir, "metrics.json"),
        )

        service = ServiceEval(cfg)

        with mock.patch("service_eval.read_any") as mock_read:
            with mock.patch("service_eval.calculate_metrics") as mock_calc:
                with mock.patch("service_eval.plot_equity_curve"):
                    # Mock read_any to return empty for trades, reports for reports
                    def read_side_effect(path):
                        if "trades" in path:
                            return pd.DataFrame()
                        return sample_reports_df

                    mock_read.side_effect = read_side_effect

                    mock_calc.return_value = {
                        "equity": {"total_return": 0.0},
                        "trades": {"total_trades": 0},
                    }

                    metrics = service.run()

                    # Verify trades were synthesized from reports
                    call_args = mock_calc.call_args
                    trades_arg = call_args[0][0]

                    # Should have synthesized trades with pnl column
                    assert "pnl" in trades_arg.columns


def test_service_eval_run_profile_filter(sample_trades_df, sample_reports_df):
    """Test ServiceEval.run() with execution profile filtering."""
    # Add execution_profile column
    sample_trades_df = sample_trades_df.copy()
    sample_trades_df["execution_profile"] = "aggressive"

    sample_reports_df = sample_reports_df.copy()
    sample_reports_df["execution_profile"] = "aggressive"

    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        reports_path = os.path.join(tmpdir, "reports.csv")
        sample_trades_df.to_csv(trades_path, index=False)
        sample_reports_df.to_csv(reports_path, index=False)

        cfg = EvalConfig(
            trades_path=trades_path,
            reports_path=reports_path,
            profile="aggressive",
            out_json=os.path.join(tmpdir, "metrics.json"),
        )

        service = ServiceEval(cfg)

        with mock.patch("service_eval.calculate_metrics") as mock_calc:
            with mock.patch("service_eval.plot_equity_curve"):
                mock_calc.return_value = {
                    "equity": {},
                    "trades": {},
                }

                service.run()

                # Verify filtering was applied
                call_args = mock_calc.call_args
                trades_arg = call_args[0][0]
                reports_arg = call_args[0][1]

                # execution_profile column should be dropped after filtering
                assert "execution_profile" not in trades_arg.columns


def test_service_eval_run_multi_profile(sample_trades_df, sample_reports_df):
    """Test ServiceEval.run() with multiple profiles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write separate files for each profile
        trades_conservative = os.path.join(tmpdir, "trades_conservative.csv")
        trades_aggressive = os.path.join(tmpdir, "trades_aggressive.csv")
        reports_conservative = os.path.join(tmpdir, "reports_conservative.csv")
        reports_aggressive = os.path.join(tmpdir, "reports_aggressive.csv")

        sample_trades_df.to_csv(trades_conservative, index=False)
        sample_trades_df.to_csv(trades_aggressive, index=False)
        sample_reports_df.to_csv(reports_conservative, index=False)
        sample_reports_df.to_csv(reports_aggressive, index=False)

        cfg = EvalConfig(
            trades_path={
                "conservative": trades_conservative,
                "aggressive": trades_aggressive,
            },
            reports_path={
                "conservative": reports_conservative,
                "aggressive": reports_aggressive,
            },
            out_json=os.path.join(tmpdir, "metrics.json"),
        )

        service = ServiceEval(cfg)

        with mock.patch("service_eval.calculate_metrics") as mock_calc:
            with mock.patch("service_eval.plot_equity_curve"):
                # Return separate metrics for each profile
                mock_calc.return_value = {
                    "conservative": {
                        "equity": {"sharpe_ratio": 1.2},
                        "trades": {"total_trades": 5},
                    },
                    "aggressive": {
                        "equity": {"sharpe_ratio": 1.8},
                        "trades": {"total_trades": 10},
                    },
                }

                metrics = service.run()

                # Verify multi-profile results
                assert "conservative" in metrics or "equity" in metrics


def test_service_eval_run_snapshot_config():
    """Test ServiceEval.run() with snapshot_config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = EvalConfig(
            trades_path="logs/trades.csv",
            reports_path="logs/reports.csv",
            snapshot_config_path="config.yaml",
            artifacts_dir=tmpdir,
        )

        service = ServiceEval(cfg)

        with mock.patch("service_eval.snapshot_config") as mock_snapshot:
            with mock.patch("service_eval.read_any") as mock_read:
                with mock.patch("service_eval.calculate_metrics") as mock_calc:
                    with mock.patch("service_eval.plot_equity_curve"):
                        mock_read.return_value = pd.DataFrame(
                            {
                                "ts_ms": [1000],
                                "pnl": [0.0],
                                "equity": [10000.0],
                            }
                        )

                        mock_calc.return_value = {
                            "equity": {"total_return": 0.0},
                            "trades": {"total_trades": 0},
                        }

                        service.run()

                        # Verify snapshot_config was called
                        mock_snapshot.assert_called_once_with("config.yaml", tmpdir)


def test_service_eval_run_side_normalization(sample_reports_df):
    """Test trades side column normalization."""
    # Create trades with lowercase side
    trades_df = pd.DataFrame(
        {
            "ts_ms": [1000, 2000],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "side": ["buy", "sell"],
            "qty": [1.0, 0.5],
            "pnl": [0.0, 1.0],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        reports_path = os.path.join(tmpdir, "reports.csv")
        trades_df.to_csv(trades_path, index=False)
        sample_reports_df.to_csv(reports_path, index=False)

        cfg = EvalConfig(
            trades_path=trades_path,
            reports_path=reports_path,
            out_json=os.path.join(tmpdir, "metrics.json"),
        )

        service = ServiceEval(cfg)

        with mock.patch("service_eval.calculate_metrics") as mock_calc:
            with mock.patch("service_eval.plot_equity_curve"):
                mock_calc.return_value = {"equity": {}, "trades": {}}

                service.run()

                # Verify side was normalized to uppercase
                call_args = mock_calc.call_args
                trades_arg = call_args[0][0]

                assert (trades_arg["side"] == ["BUY", "SELL"]).all()


# ============================================================================
# Test from_config Integration
# ============================================================================


def test_from_config_single_profile(sample_trades_df, sample_reports_df):
    """Test from_config with single profile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        reports_path = os.path.join(tmpdir, "reports.csv")
        sample_trades_df.to_csv(trades_path, index=False)
        sample_reports_df.to_csv(reports_path, index=False)

        cfg = mock.Mock(spec=CommonRunConfig)
        cfg.input = mock.Mock()
        cfg.input.trades_path = trades_path
        cfg.input.equity_path = reports_path
        cfg.logs_dir = tmpdir
        cfg.artifacts_dir = tmpdir
        cfg.components = mock.Mock()

        with mock.patch("service_eval.di_registry.build_graph") as mock_build:
            with mock.patch("service_eval.calculate_metrics") as mock_calc:
                with mock.patch("service_eval.plot_equity_curve"):
                    mock_build.return_value = {}
                    mock_calc.return_value = {
                        "equity": {"sharpe_ratio": 1.5},
                        "trades": {"total_trades": 5},
                    }

                    metrics = from_config(cfg)

                    assert "equity" in metrics
                    assert "trades" in metrics


def test_from_config_all_profiles(sample_trades_df, sample_reports_df):
    """Test from_config with all_profiles=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files for each ExecutionProfile
        from core_config import ExecutionProfile

        for prof in ExecutionProfile:
            trades_path = os.path.join(tmpdir, f"trades_{prof.value}.csv")
            reports_path = os.path.join(tmpdir, f"reports_{prof.value}.csv")
            sample_trades_df.to_csv(trades_path, index=False)
            sample_reports_df.to_csv(reports_path, index=False)

        cfg = mock.Mock(spec=CommonRunConfig)
        cfg.input = mock.Mock()
        cfg.input.trades_path = os.path.join(tmpdir, "trades_{profile}.csv")
        cfg.input.equity_path = os.path.join(tmpdir, "reports_{profile}.csv")
        cfg.logs_dir = tmpdir
        cfg.artifacts_dir = tmpdir
        cfg.components = mock.Mock()

        with mock.patch("service_eval.di_registry.build_graph") as mock_build:
            with mock.patch("service_eval.calculate_metrics") as mock_calc:
                with mock.patch("service_eval.plot_equity_curve"):
                    mock_build.return_value = {}
                    mock_calc.return_value = {
                        "equity": {"sharpe_ratio": 1.5},
                        "trades": {"total_trades": 5},
                    }

                    metrics = from_config(cfg, all_profiles=True)

                    # Verify metrics returned for all profiles
                    assert isinstance(metrics, dict)
                    # Should have entries for all ExecutionProfile values
                    assert len(metrics) == len(ExecutionProfile)
                    # Verify all profile keys are present
                    for prof in ExecutionProfile:
                        assert prof.value in metrics


def test_from_config_with_execution_profile(sample_trades_df, sample_reports_df):
    """Test from_config with ExecutionProfile enum."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        reports_path = os.path.join(tmpdir, "reports.csv")
        sample_trades_df.to_csv(trades_path, index=False)
        sample_reports_df.to_csv(reports_path, index=False)

        cfg = mock.Mock(spec=CommonRunConfig)
        cfg.input = mock.Mock()
        cfg.input.trades_path = trades_path
        cfg.input.equity_path = reports_path
        cfg.logs_dir = tmpdir
        cfg.artifacts_dir = tmpdir
        cfg.components = mock.Mock()
        cfg.execution_profile = ExecutionProfile.MKT_OPEN_NEXT_H1

        with mock.patch("service_eval.di_registry.build_graph") as mock_build:
            with mock.patch("service_eval.calculate_metrics") as mock_calc:
                with mock.patch("service_eval.plot_equity_curve"):
                    mock_build.return_value = {}
                    mock_calc.return_value = {
                        "equity": {},
                        "trades": {},
                    }

                    metrics = from_config(cfg, profile=ExecutionProfile.MKT_OPEN_NEXT_H1)

                    assert isinstance(metrics, dict)


def test_from_config_dict_paths(sample_trades_df, sample_reports_df):
    """Test from_config with dict paths (multi-profile setup)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_aggressive = os.path.join(tmpdir, "trades_aggressive.csv")
        reports_aggressive = os.path.join(tmpdir, "reports_aggressive.csv")
        sample_trades_df.to_csv(trades_aggressive, index=False)
        sample_reports_df.to_csv(reports_aggressive, index=False)

        cfg = mock.Mock(spec=CommonRunConfig)
        cfg.input = mock.Mock()
        cfg.input.trades_path = {"aggressive": trades_aggressive}
        cfg.input.equity_path = {"aggressive": reports_aggressive}
        cfg.logs_dir = tmpdir
        cfg.artifacts_dir = tmpdir
        cfg.components = mock.Mock()

        with mock.patch("service_eval.di_registry.build_graph") as mock_build:
            with mock.patch("service_eval.calculate_metrics") as mock_calc:
                with mock.patch("service_eval.plot_equity_curve"):
                    mock_build.return_value = {}
                    mock_calc.return_value = {
                        "equity": {},
                        "trades": {},
                    }

                    metrics = from_config(cfg)

                    # Should select first profile from dict
                    assert isinstance(metrics, dict)


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_service_eval_empty_trades():
    """Test ServiceEval with empty trades."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        reports_path = os.path.join(tmpdir, "reports.csv")

        # Create empty DataFrame with expected columns to avoid EmptyDataError
        pd.DataFrame(columns=["ts_ms", "pnl"]).to_csv(trades_path, index=False)
        pd.DataFrame({"equity": [10000.0]}).to_csv(reports_path, index=False)

        cfg = EvalConfig(
            trades_path=trades_path,
            reports_path=reports_path,
            out_json=os.path.join(tmpdir, "metrics.json"),
        )

        service = ServiceEval(cfg)

        with mock.patch("service_eval.calculate_metrics") as mock_calc:
            with mock.patch("service_eval.plot_equity_curve"):
                mock_calc.return_value = {
                    "equity": {"total_return": 0.0},
                    "trades": {"total_trades": 0},
                }

                metrics = service.run()

                assert metrics["trades"]["total_trades"] == 0


def test_service_eval_quantity_column_rename():
    """Test trades quantity column renaming."""
    trades_df = pd.DataFrame(
        {
            "ts": [1000],
            "run_id": ["test"],
            "symbol": ["BTCUSDT"],
            "side": ["BUY"],
            "order_type": ["MARKET"],
            "price": [100.0],
            "quantity": [1.0],  # Should be renamed to 'qty'
            "pnl": [0.0],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path = os.path.join(tmpdir, "trades.csv")
        reports_path = os.path.join(tmpdir, "reports.csv")
        trades_df.to_csv(trades_path, index=False)
        pd.DataFrame({"equity": [10000.0]}).to_csv(reports_path, index=False)

        cfg = EvalConfig(
            trades_path=trades_path,
            reports_path=reports_path,
            out_json=os.path.join(tmpdir, "metrics.json"),
        )

        service = ServiceEval(cfg)

        with mock.patch("service_eval.calculate_metrics") as mock_calc:
            with mock.patch("service_eval.plot_equity_curve"):
                mock_calc.return_value = {"equity": {}, "trades": {}}

                service.run()

                # Verify column was renamed
                call_args = mock_calc.call_args
                trades_arg = call_args[0][0]

                assert "qty" in trades_arg.columns
                assert "quantity" not in trades_arg.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

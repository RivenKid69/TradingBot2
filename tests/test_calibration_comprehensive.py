"""Comprehensive unit tests for calibration services."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Check if parquet engine is available for tests that need it
try:
    pd.DataFrame().to_parquet  # Check if method exists
    import pyarrow  # noqa: F401

    HAS_PARQUET = True
except (ImportError, AttributeError):
    HAS_PARQUET = False

from scripts.calibrate_dynamic_spread import (
    _read_table,
    _pick_column,
    _compute_mid,
    _compute_spread_bps,
    _prepare_dataframe,
    _select_volatility,
    _clip_percentiles,
    _linear_regression,
    _fallback_parameters,
    _derive_spread_bounds,
    _normalise_smoothing_alpha,
    parse_args,
    main,
)


class TestReadTable:
    """Test _read_table function."""

    def test_read_csv(self):
        """Test reading CSV file."""
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
            data.to_csv(path, index=False)

        try:
            result = _read_table(path)
            pd.testing.assert_frame_equal(result, data)
        finally:
            path.unlink()

    @pytest.mark.skipif(not HAS_PARQUET, reason="pyarrow/fastparquet not installed")
    def test_read_parquet(self):
        """Test reading Parquet file."""
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
            data.to_parquet(path)

        try:
            result = _read_table(path)
            pd.testing.assert_frame_equal(result, data)
        finally:
            path.unlink()

    def test_read_unsupported_format(self):
        """Test raises ValueError for unsupported format."""
        path = Path("test.xyz")
        with pytest.raises(ValueError, match="Unsupported file type"):
            _read_table(path)


class TestPickColumn:
    """Test _pick_column helper function."""

    def test_pick_column_first_match(self):
        """Test picks first matching column."""
        df = pd.DataFrame({"bid": [1], "ask": [2], "price": [3]})
        result = _pick_column(df, ["bid", "ask", "price"])
        assert result == "bid"

    def test_pick_column_second_match(self):
        """Test picks second when first not found."""
        df = pd.DataFrame({"ask": [2], "price": [3]})
        result = _pick_column(df, ["bid", "ask", "price"])
        assert result == "ask"

    def test_pick_column_no_match(self):
        """Test returns None when no match."""
        df = pd.DataFrame({"other": [1]})
        result = _pick_column(df, ["bid", "ask", "price"])
        assert result is None


class TestComputeMid:
    """Test _compute_mid function."""

    def test_compute_mid_from_bid_ask(self):
        """Test computes mid from bid/ask."""
        df = pd.DataFrame({"bid": [99.0, 100.0], "ask": [101.0, 102.0]})
        result = _compute_mid(df)
        expected = pd.Series([100.0, 101.0])
        pd.testing.assert_series_equal(result, expected)

    def test_compute_mid_from_close(self):
        """Test uses close when bid/ask not available."""
        df = pd.DataFrame({"close": [100.0, 101.0]})
        result = _compute_mid(df)
        expected = pd.Series([100.0, 101.0])
        pd.testing.assert_series_equal(result, expected)

    def test_compute_mid_raises_when_no_columns(self):
        """Test raises KeyError when no valid columns."""
        df = pd.DataFrame({"other": [1, 2]})
        with pytest.raises(KeyError):
            _compute_mid(df)


class TestComputeSpreadBps:
    """Test _compute_spread_bps function."""

    def test_compute_spread_from_existing_column(self):
        """Test uses existing spread_bps column."""
        df = pd.DataFrame({"spread_bps": [10.0, 20.0]})
        mid = pd.Series([100.0, 100.0])
        result = _compute_spread_bps(df, mid)
        expected = pd.Series([10.0, 20.0])
        pd.testing.assert_series_equal(result, expected)

    def test_compute_spread_from_bid_ask(self):
        """Test computes spread from bid/ask."""
        df = pd.DataFrame({"bid": [99.0], "ask": [101.0]})
        mid = pd.Series([100.0])
        result = _compute_spread_bps(df, mid)
        expected = pd.Series([(101.0 - 99.0) / 100.0 * 10000])
        pd.testing.assert_series_equal(result, expected)

    def test_compute_spread_returns_none_when_unavailable(self):
        """Test returns None when spread cannot be computed."""
        df = pd.DataFrame({"other": [1, 2]})
        mid = pd.Series([100.0, 100.0])
        result = _compute_spread_bps(df, mid)
        assert result is None


class TestPrepareDataframe:
    """Test _prepare_dataframe function."""

    def test_prepare_dataframe_basic(self):
        """Test basic dataframe preparation."""
        df = pd.DataFrame(
            {
                "high": [102.0, 103.0],
                "low": [98.0, 97.0],
                "close": [100.0, 101.0],
            }
        )
        result = _prepare_dataframe(df, symbol=None, timeframe=None)

        assert "mid" in result.columns
        assert "range_ratio_bps" in result.columns
        assert len(result) == 2

    def test_prepare_dataframe_filters_by_symbol(self):
        """Test filters by symbol when provided."""
        df = pd.DataFrame(
            {
                "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
                "high": [102.0, 103.0, 104.0],
                "low": [98.0, 97.0, 96.0],
                "close": [100.0, 101.0, 102.0],
            }
        )
        result = _prepare_dataframe(df, symbol="BTCUSDT", timeframe=None)

        assert len(result) == 2
        assert all(result["symbol"] == "BTCUSDT")

    def test_prepare_dataframe_computes_range_ratio(self):
        """Test computes range_ratio_bps correctly."""
        df = pd.DataFrame(
            {
                "high": [110.0],
                "low": [90.0],
                "close": [100.0],
            }
        )
        result = _prepare_dataframe(df, symbol=None, timeframe=None)

        # Range = 20, mid = 100, ratio = 0.2, bps = 2000
        assert result["range_ratio_bps"].iloc[0] == pytest.approx(2000.0, abs=1.0)

    def test_prepare_dataframe_handles_missing_columns(self):
        """Test raises KeyError for missing high/low columns."""
        df = pd.DataFrame({"close": [100.0]})
        with pytest.raises(KeyError):
            _prepare_dataframe(df, symbol=None, timeframe=None)

    def test_prepare_dataframe_filters_invalid_prices(self):
        """Test filters rows with invalid prices."""
        df = pd.DataFrame(
            {
                "high": [102.0, np.nan, 104.0],
                "low": [98.0, 97.0, np.inf],
                "close": [100.0, 101.0, 102.0],
            }
        )
        result = _prepare_dataframe(df, symbol=None, timeframe=None)

        # Only first row should remain
        assert len(result) == 1


class TestSelectVolatility:
    """Test _select_volatility function."""

    def test_select_volatility_range_ratio(self):
        """Test selects range_ratio_bps."""
        df = pd.DataFrame({"range_ratio_bps": [100.0, 200.0]})
        result = _select_volatility(df, "range_ratio_bps")
        expected = pd.Series([100.0, 200.0])
        pd.testing.assert_series_equal(result, expected)

    def test_select_volatility_custom_column(self):
        """Test selects custom volatility column."""
        df = pd.DataFrame({"custom_vol": [50.0, 60.0]})
        result = _select_volatility(df, "custom_vol")
        expected = pd.Series([50.0, 60.0])
        pd.testing.assert_series_equal(result, expected)

    def test_select_volatility_raises_for_missing(self):
        """Test raises KeyError for missing column."""
        df = pd.DataFrame({"other": [1, 2]})
        with pytest.raises(KeyError):
            _select_volatility(df, "nonexistent")


class TestClipPercentiles:
    """Test _clip_percentiles function."""

    def test_clip_percentiles_basic(self):
        """Test clips to percentile range."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = _clip_percentiles(series, lower=10, upper=90)

        # Should clip to [1, 9] range
        assert result.min() >= 1.0
        assert result.max() <= 9.0

    def test_clip_percentiles_empty_series(self):
        """Test returns empty series unchanged."""
        series = pd.Series([], dtype=float)
        result = _clip_percentiles(series, lower=10, upper=90)
        assert len(result) == 0

    def test_clip_percentiles_handles_inf(self):
        """Test handles inf values."""
        series = pd.Series([1, 2, np.inf, 4, 5])
        result = _clip_percentiles(series, lower=10, upper=90)

        # Should not crash, inf handled
        assert np.isfinite(result).all()


class TestLinearRegression:
    """Test _linear_regression function."""

    def test_linear_regression_basic(self):
        """Test basic linear regression."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])  # y = 2x

        alpha, beta = _linear_regression(x, y)

        assert alpha == pytest.approx(0.0, abs=0.1)
        assert beta == pytest.approx(2.0, abs=0.1)

    def test_linear_regression_with_intercept(self):
        """Test regression with non-zero intercept."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([3, 5, 7, 9, 11])  # y = 2x + 1

        alpha, beta = _linear_regression(x, y)

        assert alpha == pytest.approx(1.0, abs=0.1)
        assert beta == pytest.approx(2.0, abs=0.1)


class TestFallbackParameters:
    """Test _fallback_parameters function."""

    def test_fallback_parameters_basic(self):
        """Test fallback parameter computation."""
        volatility = pd.Series([1.0, 2.0, 3.0])
        spread = pd.Series([10.0, 20.0, 30.0])

        alpha, beta = _fallback_parameters(volatility, spread)

        # Alpha should be median spread
        assert alpha == 20.0

        # Beta should be median of spread/vol ratio
        assert beta > 0

    def test_fallback_parameters_empty_input(self):
        """Test handles empty input."""
        volatility = pd.Series([], dtype=float)
        spread = pd.Series([], dtype=float)

        alpha, beta = _fallback_parameters(volatility, spread)

        assert alpha == 0.0
        assert beta == 0.0

    def test_fallback_parameters_invalid_volatility(self):
        """Test handles invalid volatility."""
        volatility = pd.Series([0.0, np.nan, -1.0])
        spread = pd.Series([10.0, 20.0, 30.0])

        alpha, beta = _fallback_parameters(volatility, spread)

        # Should handle gracefully
        assert np.isfinite(alpha)
        assert np.isfinite(beta)


class TestDeriveSpreadBounds:
    """Test _derive_spread_bounds function."""

    def test_derive_spread_bounds_basic(self):
        """Test derives bounds from percentiles."""
        spread = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lower, upper = _derive_spread_bounds(spread, lower_pct=10, upper_pct=90)

        assert lower is not None
        assert upper is not None
        assert lower < upper
        assert lower == pytest.approx(1.9, abs=0.5)
        assert upper == pytest.approx(9.1, abs=0.5)

    def test_derive_spread_bounds_empty_series(self):
        """Test returns None for empty series."""
        spread = pd.Series([], dtype=float)
        lower, upper = _derive_spread_bounds(spread, lower_pct=10, upper_pct=90)

        assert lower is None
        assert upper is None

    def test_derive_spread_bounds_inverted_percentiles(self):
        """Test handles inverted percentiles."""
        spread = pd.Series([5, 10, 15, 20])
        lower, upper = _derive_spread_bounds(spread, lower_pct=90, upper_pct=10)

        # Upper should be at least equal to lower
        assert upper >= lower


class TestNormaliseSmoothingAlpha:
    """Test _normalise_smoothing_alpha function."""

    def test_normalise_smoothing_alpha_none(self):
        """Test returns None for None input."""
        assert _normalise_smoothing_alpha(None) is None

    def test_normalise_smoothing_alpha_zero(self):
        """Test returns None for zero."""
        assert _normalise_smoothing_alpha(0.0) is None

    def test_normalise_smoothing_alpha_negative(self):
        """Test returns None for negative."""
        assert _normalise_smoothing_alpha(-0.5) is None

    def test_normalise_smoothing_alpha_valid(self):
        """Test returns value for valid input."""
        assert _normalise_smoothing_alpha(0.5) == 0.5

    def test_normalise_smoothing_alpha_above_one(self):
        """Test caps at 1.0."""
        assert _normalise_smoothing_alpha(1.5) == 1.0


class TestParseArgs:
    """Test parse_args function."""

    def test_parse_args_minimal(self):
        """Test parsing minimal arguments."""
        args = parse_args(["data.csv"])
        assert args.data_path == Path("data.csv")
        assert args.volatility_metric == "range_ratio_bps"
        assert args.clip_lower == 5.0
        assert args.clip_upper == 95.0

    def test_parse_args_with_symbol(self):
        """Test parsing with symbol filter."""
        args = parse_args(["data.csv", "--symbol", "BTCUSDT"])
        assert args.symbol == "BTCUSDT"

    def test_parse_args_with_timeframe(self):
        """Test parsing with timeframe filter."""
        args = parse_args(["data.csv", "--timeframe", "1m"])
        assert args.timeframe == "1m"

    def test_parse_args_with_output(self):
        """Test parsing with output file."""
        args = parse_args(["data.csv", "--output", "calibration.yaml"])
        assert args.output == Path("calibration.yaml")

    def test_parse_args_with_spread_override(self):
        """Test parsing with target spread override."""
        args = parse_args(["data.csv", "--target-spread-bps", "15.0"])
        assert args.target_spread_bps == 15.0

    def test_parse_args_with_bounds(self):
        """Test parsing with explicit bounds."""
        args = parse_args(
            [
                "data.csv",
                "--min-spread-bps",
                "5.0",
                "--max-spread-bps",
                "50.0",
            ]
        )
        assert args.min_spread_bps == 5.0
        assert args.max_spread_bps == 50.0

    def test_parse_args_with_smoothing(self):
        """Test parsing with smoothing alpha."""
        args = parse_args(["data.csv", "--smoothing-alpha", "0.1"])
        assert args.smoothing_alpha == 0.1


class TestMainFunction:
    """Test main() function."""

    def test_main_with_valid_data(self):
        """Test main() runs successfully with valid data."""
        # Create test data
        data = pd.DataFrame(
            {
                "high": [105.0] * 100,
                "low": [95.0] * 100,
                "close": [100.0] * 100,
                "bid": [99.0] * 100,
                "ask": [101.0] * 100,
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
            data.to_csv(path, index=False)

        try:
            result = main([str(path)])
            assert result == 0
        finally:
            path.unlink()

    def test_main_with_output_file(self):
        """Test main() writes output YAML."""
        data = pd.DataFrame(
            {
                "high": [105.0] * 100,
                "low": [95.0] * 100,
                "close": [100.0] * 100,
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as data_file:
            data_path = Path(data_file.name)
            data.to_csv(data_path, index=False)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as out_file:
            out_path = Path(out_file.name)

        try:
            # Run with output
            result = main([str(data_path), "--output", str(out_path)])
            assert result == 0

            # Verify output file created
            assert out_path.exists()

            # Verify YAML structure
            import yaml

            with out_path.open("r") as f:
                config = yaml.safe_load(f)

            assert "slippage" in config
            assert "dynamic_spread" in config["slippage"]
            assert "alpha_bps" in config["slippage"]["dynamic_spread"]
            assert "beta_coef" in config["slippage"]["dynamic_spread"]
        finally:
            data_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)

    def test_main_with_missing_file(self):
        """Test main() raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            main(["/nonexistent/file.csv"])

    def test_main_with_invalid_percentiles(self):
        """Test main() raises error for invalid percentiles."""
        data = pd.DataFrame(
            {
                "high": [105.0],
                "low": [95.0],
                "close": [100.0],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
            data.to_csv(path, index=False)

        try:
            with pytest.raises(ValueError):
                main([str(path), "--clip-lower", "50", "--clip-upper", "40"])
        finally:
            path.unlink()

    def test_main_with_target_spread(self):
        """Test main() uses target spread when provided."""
        data = pd.DataFrame(
            {
                "high": [105.0] * 50,
                "low": [95.0] * 50,
                "close": [100.0] * 50,
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
            data.to_csv(path, index=False)

        try:
            result = main([str(path), "--target-spread-bps", "10.0"])
            assert result == 0
        finally:
            path.unlink()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_prepare_dataframe_with_negative_range(self):
        """Test handles inverted high/low gracefully."""
        df = pd.DataFrame(
            {
                "high": [90.0],  # Lower than low
                "low": [110.0],
                "close": [100.0],
            }
        )
        result = _prepare_dataframe(df, symbol=None, timeframe=None)

        # Should handle gracefully (abs value)
        assert result["range_ratio_bps"].iloc[0] >= 0

    def test_linear_regression_with_constant_x(self):
        """Test regression handles constant input."""
        x = pd.Series([5.0, 5.0, 5.0])
        y = pd.Series([10.0, 20.0, 30.0])

        # Should not crash (though result may be degenerate)
        try:
            alpha, beta = _linear_regression(x, y)
            assert np.isfinite(alpha) or np.isnan(alpha)
        except np.linalg.LinAlgError:
            pass  # Expected for degenerate case

    def test_fallback_parameters_with_inf(self):
        """Test fallback handles inf values."""
        volatility = pd.Series([1.0, 2.0, np.inf])
        spread = pd.Series([10.0, 20.0, 30.0])

        alpha, beta = _fallback_parameters(volatility, spread)

        # Should filter inf and compute on valid data
        assert np.isfinite(alpha)
        assert np.isfinite(beta)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

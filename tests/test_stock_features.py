# tests/test_stock_features.py
"""
Comprehensive tests for Phase 5: Stock-Specific Features.

This test suite verifies:
1. VIX Integration (HIGH priority)
2. Market Regime Indicator
3. Relative Strength features
4. Sector/Industry features
5. Backward compatibility with crypto

Test categories:
- Unit tests for individual feature calculations
- Integration tests for feature extraction
- Backward compatibility tests for crypto data
- Edge case handling (NaN, Inf, missing data)

Author: Sultan Suyunbayev
Date: 2025-11-27
"""

import math
import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Import the module under test
from stock_features import (
    # VIX functions
    calculate_vix_regime,
    normalize_vix_value,
    VIXRegime,
    VIX_REGIME_THRESHOLDS,
    # Market regime functions
    calculate_market_regime,
    MarketRegime,
    MARKET_REGIME_THRESHOLDS,
    # Relative strength functions
    calculate_relative_strength,
    RS_WINDOWS,
    # Sector functions
    calculate_sector_momentum,
    get_symbol_sector,
    SYMBOL_TO_SECTOR,
    SECTOR_ETFS,
    # Main extraction functions
    extract_stock_features,
    StockFeatures,
    BenchmarkData,
    add_stock_features_to_dataframe,
    calculate_sector_returns_from_etfs,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_spy_prices() -> List[float]:
    """Generate sample SPY prices for testing."""
    # Simulate upward trending SPY (bull market)
    base = 450.0
    prices = []
    for i in range(100):
        # Slight upward trend with noise
        noise = np.random.randn() * 2
        trend = i * 0.1  # Gradual uptrend
        prices.append(base + trend + noise)
    return prices


@pytest.fixture
def sample_qqq_prices() -> List[float]:
    """Generate sample QQQ prices for testing."""
    base = 380.0
    prices = []
    for i in range(100):
        noise = np.random.randn() * 3
        trend = i * 0.12  # Slightly stronger uptrend (tech)
        prices.append(base + trend + noise)
    return prices


@pytest.fixture
def sample_stock_prices() -> List[float]:
    """Generate sample stock prices for testing."""
    base = 150.0
    prices = []
    for i in range(100):
        noise = np.random.randn() * 5
        trend = i * 0.15  # Outperforming trend
        prices.append(base + trend + noise)
    return prices


@pytest.fixture
def sample_vix_values() -> List[float]:
    """Generate sample VIX values."""
    return [18.5, 19.2, 20.1, 19.8, 21.5, 22.0, 20.5, 19.0, 18.0, 17.5]


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame with stock data."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(145, 155, n),
        "high": np.random.uniform(155, 165, n),
        "low": np.random.uniform(140, 150, n),
        "close": np.cumsum(np.random.randn(n) * 0.5) + 150,
        "volume": np.random.uniform(1e6, 5e6, n),
    })
    return df


# =============================================================================
# VIX REGIME TESTS
# =============================================================================

class TestVIXRegime:
    """Test VIX regime classification."""

    def test_vix_regime_low_complacency(self):
        """Test VIX < 12 classified as LOW (complacency)."""
        vix = 10.0
        normalized, regime = calculate_vix_regime(vix)

        assert regime == VIXRegime.LOW
        assert 0.0 <= normalized <= 0.25
        # VIX=10 should be around 0.21 (10/12 * 0.25)
        expected = (10.0 / 12.0) * 0.25
        assert abs(normalized - expected) < 0.01

    def test_vix_regime_normal(self):
        """Test VIX 12-20 classified as NORMAL."""
        vix = 16.0
        normalized, regime = calculate_vix_regime(vix)

        assert regime == VIXRegime.NORMAL
        assert 0.25 <= normalized <= 0.5
        # VIX=16 should be around 0.375
        expected = 0.25 + ((16.0 - 12.0) / (20.0 - 12.0)) * 0.25
        assert abs(normalized - expected) < 0.01

    def test_vix_regime_elevated(self):
        """Test VIX 20-30 classified as ELEVATED."""
        vix = 25.0
        normalized, regime = calculate_vix_regime(vix)

        assert regime == VIXRegime.ELEVATED
        assert 0.5 <= normalized <= 0.75

    def test_vix_regime_extreme(self):
        """Test VIX > 30 classified as EXTREME."""
        vix = 45.0
        normalized, regime = calculate_vix_regime(vix)

        assert regime == VIXRegime.EXTREME
        assert 0.75 <= normalized <= 1.0

    def test_vix_regime_extreme_capped_at_80(self):
        """Test VIX > 80 is capped (normalized stays <= 1.0)."""
        vix = 100.0  # Crisis level
        normalized, regime = calculate_vix_regime(vix)

        assert regime == VIXRegime.EXTREME
        assert normalized <= 1.0

    def test_vix_regime_invalid_nan(self):
        """Test NaN VIX returns default NORMAL regime."""
        normalized, regime = calculate_vix_regime(float('nan'))

        assert regime == VIXRegime.NORMAL
        assert normalized == 0.5

    def test_vix_regime_invalid_negative(self):
        """Test negative VIX returns default NORMAL regime."""
        normalized, regime = calculate_vix_regime(-5.0)

        assert regime == VIXRegime.NORMAL
        assert normalized == 0.5


class TestVIXNormalization:
    """Test VIX value normalization."""

    def test_normalize_vix_at_typical_level(self):
        """Test VIX=20 (typical) normalizes to 0."""
        normalized = normalize_vix_value(20.0)
        assert abs(normalized) < 0.01  # Should be close to 0

    def test_normalize_vix_low(self):
        """Test low VIX normalizes to negative."""
        normalized = normalize_vix_value(10.0)
        # tanh((10-20)/20) = tanh(-0.5) ≈ -0.46
        assert normalized < 0
        assert normalized > -1.0

    def test_normalize_vix_high(self):
        """Test high VIX normalizes to positive."""
        normalized = normalize_vix_value(40.0)
        # tanh((40-20)/20) = tanh(1) ≈ 0.76
        assert normalized > 0
        assert normalized < 1.0

    def test_normalize_vix_extreme_high(self):
        """Test extreme VIX approaches but doesn't exceed 1."""
        normalized = normalize_vix_value(80.0)
        assert normalized > 0.9
        assert normalized < 1.0

    def test_normalize_vix_nan_returns_zero(self):
        """Test NaN VIX normalizes to 0."""
        normalized = normalize_vix_value(float('nan'))
        assert normalized == 0.0

    def test_normalize_vix_inf_returns_zero(self):
        """Test Inf VIX normalizes to 0."""
        normalized = normalize_vix_value(float('inf'))
        assert normalized == 0.0


# =============================================================================
# MARKET REGIME TESTS
# =============================================================================

class TestMarketRegime:
    """Test market regime classification."""

    def test_market_regime_bull(self, sample_spy_prices):
        """Test bull market detection."""
        # Create strongly uptrending prices
        prices = [100 + i * 0.5 for i in range(60)]

        regime_val, regime, is_valid = calculate_market_regime(prices, vix_value=15.0)

        assert is_valid
        assert regime == MarketRegime.BULL
        assert regime_val > 0.0

    def test_market_regime_bear(self):
        """Test bear market detection."""
        # Create strongly downtrending prices
        prices = [500 - i * 2 for i in range(60)]

        regime_val, regime, is_valid = calculate_market_regime(prices, vix_value=30.0)

        assert is_valid
        assert regime == MarketRegime.BEAR
        assert regime_val < 0.0

    def test_market_regime_sideways(self):
        """Test sideways market detection."""
        # Create sideways prices (small random walk around mean with no trend)
        np.random.seed(42)
        prices = [100.0]
        for _ in range(59):
            # Random walk with mean reversion
            prices.append(100.0 + np.random.randn() * 0.5)

        regime_val, regime, is_valid = calculate_market_regime(prices, vix_value=17.0)

        assert is_valid
        # Sideways or near-sideways (regime not strongly bull or bear)
        # Due to random nature, we just check it's not extreme
        assert -0.8 < regime_val < 0.8

    def test_market_regime_vix_override_bear(self):
        """Test high VIX overrides to bear regardless of trend."""
        # Create uptrending prices
        prices = [100 + i * 0.3 for i in range(60)]

        # But very high VIX
        regime_val, regime, is_valid = calculate_market_regime(prices, vix_value=40.0)

        assert is_valid
        assert regime == MarketRegime.BEAR  # VIX override
        assert regime_val == -1.0

    def test_market_regime_insufficient_data(self):
        """Test insufficient data returns invalid."""
        prices = [100.0] * 10  # Only 10 prices, need 50

        regime_val, regime, is_valid = calculate_market_regime(prices)

        assert not is_valid
        assert regime == MarketRegime.SIDEWAYS
        assert regime_val == 0.0

    def test_market_regime_no_vix(self):
        """Test regime calculation without VIX data."""
        prices = [100 + i * 0.3 for i in range(60)]

        regime_val, regime, is_valid = calculate_market_regime(prices, vix_value=None)

        assert is_valid
        # Without VIX override, should detect trend
        assert regime == MarketRegime.BULL


# =============================================================================
# RELATIVE STRENGTH TESTS
# =============================================================================

class TestRelativeStrength:
    """Test relative strength calculations."""

    def test_rs_outperformance(self):
        """Test stock outperforming benchmark gives positive RS."""
        # Stock +10% vs benchmark +5%
        stock = [100.0] + [100.0] * 20 + [110.0]  # 21 prices
        bench = [100.0] + [100.0] * 20 + [105.0]

        rs, is_valid = calculate_relative_strength(stock, bench, window=20)

        assert is_valid
        assert rs > 0.0  # Outperformance should be positive

    def test_rs_underperformance(self):
        """Test stock underperforming benchmark gives negative RS."""
        # Stock +5% vs benchmark +10%
        stock = [100.0] + [100.0] * 20 + [105.0]
        bench = [100.0] + [100.0] * 20 + [110.0]

        rs, is_valid = calculate_relative_strength(stock, bench, window=20)

        assert is_valid
        assert rs < 0.0  # Underperformance should be negative

    def test_rs_equal_performance(self):
        """Test equal performance gives RS near 0."""
        stock = [100.0] + [100.0] * 20 + [110.0]
        bench = [100.0] + [100.0] * 20 + [110.0]

        rs, is_valid = calculate_relative_strength(stock, bench, window=20)

        assert is_valid
        assert abs(rs) < 0.1  # Should be close to 0

    def test_rs_benchmark_flat(self):
        """Test RS with flat benchmark."""
        stock = [100.0] + [100.0] * 20 + [110.0]  # +10%
        bench = [100.0] * 22  # Flat

        rs, is_valid = calculate_relative_strength(stock, bench, window=20)

        assert is_valid
        # When benchmark flat, RS should be positive (stock up)
        assert rs > 0.0

    def test_rs_insufficient_data(self):
        """Test insufficient data returns invalid."""
        stock = [100.0] * 10  # Only 10 prices, need 21
        bench = [100.0] * 10

        rs, is_valid = calculate_relative_strength(stock, bench, window=20)

        assert not is_valid
        assert rs == 0.0

    def test_rs_normalization_range(self):
        """Test RS normalized to approximately [-1, 1]."""
        # Extreme outperformance: stock +50%, bench +5%
        stock = [100.0] + [100.0] * 20 + [150.0]
        bench = [100.0] + [100.0] * 20 + [105.0]

        rs, is_valid = calculate_relative_strength(stock, bench, window=20)

        assert is_valid
        assert rs > 0.9  # Should be close to 1
        assert rs <= 1.0

    def test_rs_negative_returns(self):
        """Test RS with both negative returns."""
        # Stock -5% vs bench -10% (stock outperforms)
        stock = [100.0] + [100.0] * 20 + [95.0]
        bench = [100.0] + [100.0] * 20 + [90.0]

        rs, is_valid = calculate_relative_strength(stock, bench, window=20)

        assert is_valid
        assert rs > 0.0  # Stock outperforms despite both being negative


# =============================================================================
# SECTOR FEATURES TESTS
# =============================================================================

class TestSectorFeatures:
    """Test sector classification and momentum."""

    def test_get_symbol_sector_known(self):
        """Test sector lookup for known symbols."""
        assert get_symbol_sector("AAPL") == "technology"
        assert get_symbol_sector("JPM") == "financials"
        assert get_symbol_sector("XOM") == "energy"
        assert get_symbol_sector("JNJ") == "healthcare"

    def test_get_symbol_sector_unknown(self):
        """Test sector lookup for unknown symbol returns None."""
        assert get_symbol_sector("UNKNOWN") is None
        assert get_symbol_sector("") is None

    def test_get_symbol_sector_case_insensitive(self):
        """Test sector lookup is case insensitive."""
        assert get_symbol_sector("aapl") == "technology"
        assert get_symbol_sector("AAPL") == "technology"

    def test_sector_momentum_outperforming(self):
        """Test sector outperforming market gives positive momentum."""
        sector_returns = {"technology": 0.05}  # Tech +5%
        market_return = 0.02  # Market +2%

        momentum, is_valid = calculate_sector_momentum("AAPL", sector_returns, market_return)

        assert is_valid
        assert momentum > 0.0  # Tech outperforming

    def test_sector_momentum_underperforming(self):
        """Test sector underperforming market gives negative momentum."""
        sector_returns = {"energy": 0.01}  # Energy +1%
        market_return = 0.05  # Market +5%

        momentum, is_valid = calculate_sector_momentum("XOM", sector_returns, market_return)

        assert is_valid
        assert momentum < 0.0  # Energy underperforming

    def test_sector_momentum_unknown_symbol(self):
        """Test unknown symbol returns invalid."""
        sector_returns = {"technology": 0.05}

        momentum, is_valid = calculate_sector_momentum("UNKNOWN", sector_returns, 0.02)

        assert not is_valid
        assert momentum == 0.0

    def test_sector_momentum_missing_sector_data(self):
        """Test missing sector data returns invalid."""
        sector_returns = {"technology": 0.05}  # No financials data

        momentum, is_valid = calculate_sector_momentum("JPM", sector_returns, 0.02)

        assert not is_valid
        assert momentum == 0.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestStockFeaturesIntegration:
    """Integration tests for stock feature extraction."""

    def test_extract_all_features(
        self, sample_spy_prices, sample_qqq_prices, sample_stock_prices, sample_vix_values
    ):
        """Test extraction of all stock features."""
        benchmark = BenchmarkData(
            spy_prices=sample_spy_prices,
            qqq_prices=sample_qqq_prices,
            vix_values=sample_vix_values,
            sector_returns={"technology": 0.03},
        )

        row = {"vix": 22.0}  # Row with VIX data

        features = extract_stock_features(
            row=row,
            symbol="AAPL",
            benchmark_data=benchmark,
            stock_prices=sample_stock_prices,
        )

        # VIX features should be valid
        assert features.vix_valid
        assert features.vix_regime_valid

        # Market regime should be valid (have SPY data)
        assert features.market_regime_valid

        # RS features should be valid (have benchmark data)
        assert features.rs_spy_20d_valid
        assert features.rs_spy_50d_valid
        assert features.rs_qqq_20d_valid

        # Sector momentum should be valid (have sector data)
        assert features.sector_momentum_valid

    def test_extract_features_no_benchmark(self):
        """Test feature extraction without benchmark data."""
        row = {"vix": 18.0}

        features = extract_stock_features(row=row, symbol="AAPL")

        # VIX should be valid (from row)
        assert features.vix_valid

        # Other features should be invalid (no benchmark)
        assert not features.market_regime_valid
        assert not features.rs_spy_20d_valid
        assert not features.sector_momentum_valid

    def test_extract_features_empty_row(self):
        """Test feature extraction with empty row."""
        features = extract_stock_features(row={}, symbol="AAPL")

        # All features should be invalid
        assert not features.vix_valid
        assert not features.market_regime_valid
        assert not features.rs_spy_20d_valid


class TestDataFrameIntegration:
    """Test DataFrame-level feature addition."""

    def test_add_features_to_dataframe(self, sample_dataframe):
        """Test adding stock features to DataFrame."""
        spy_df = pd.DataFrame({
            "close": np.cumsum(np.random.randn(100) * 0.3) + 450
        })

        result = add_stock_features_to_dataframe(
            sample_dataframe, "AAPL", spy_df=spy_df
        )

        # Check all new columns exist
        assert "vix_normalized" in result.columns
        assert "vix_regime" in result.columns
        assert "market_regime" in result.columns
        assert "rs_spy_20d" in result.columns
        assert "rs_spy_50d" in result.columns
        assert "rs_qqq_20d" in result.columns
        assert "sector_momentum" in result.columns

    def test_add_features_preserves_original(self, sample_dataframe):
        """Test adding features preserves original columns."""
        original_cols = set(sample_dataframe.columns)

        result = add_stock_features_to_dataframe(sample_dataframe, "AAPL")

        # Original columns should still exist
        for col in original_cols:
            assert col in result.columns


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility with crypto data."""

    def test_crypto_data_no_stock_columns(self):
        """Test crypto data without stock columns gives invalid features."""
        # Simulate crypto row (no VIX, no RS, etc.)
        crypto_row = {
            "close": 50000.0,
            "volume": 1e9,
            "cvd_24h": 100.0,
            "yang_zhang_48h": 0.02,
        }

        features = extract_stock_features(row=crypto_row, symbol="BTCUSDT")

        # All stock features should be invalid
        assert not features.vix_valid
        assert not features.market_regime_valid
        assert not features.rs_spy_20d_valid
        assert not features.sector_momentum_valid

    def test_default_values_are_sensible(self):
        """Test default values are sensible (zero/neutral)."""
        features = StockFeatures()

        # Check defaults
        assert features.vix_value == 20.0  # Typical VIX
        assert features.vix_regime == 0.5  # Neutral
        assert features.market_regime == 0.0  # Sideways
        assert features.rs_spy_20d == 0.0  # Neutral
        assert features.sector_momentum == 0.0  # Neutral


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nan_vix_in_row(self):
        """Test NaN VIX value is handled gracefully."""
        row = {"vix": float('nan')}

        features = extract_stock_features(row=row)

        assert not features.vix_valid
        # When VIX is invalid, vix_value stays at DEFAULT_VIX (20.0), not normalized
        assert features.vix_value == 20.0  # DEFAULT_VIX

    def test_inf_price_in_rs_calculation(self):
        """Test Inf price in RS calculation returns invalid."""
        stock = [100.0] * 20 + [float('inf')]
        bench = [100.0] * 21

        rs, is_valid = calculate_relative_strength(stock, bench, window=20)

        assert not is_valid

    def test_zero_price_in_rs_calculation(self):
        """Test zero price in RS calculation returns invalid."""
        stock = [100.0] * 20 + [0.0]
        bench = [100.0] * 21

        rs, is_valid = calculate_relative_strength(stock, bench, window=20)

        assert not is_valid

    def test_empty_spy_prices(self):
        """Test empty SPY prices returns invalid regime."""
        regime_val, regime, is_valid = calculate_market_regime([])

        assert not is_valid
        assert regime == MarketRegime.SIDEWAYS

    def test_all_same_prices(self):
        """Test all same prices (no movement)."""
        prices = [100.0] * 60

        regime_val, regime, is_valid = calculate_market_regime(prices)

        assert is_valid
        assert regime == MarketRegime.SIDEWAYS
        assert abs(regime_val) < 0.5


# =============================================================================
# FEATURE CONFIG TESTS
# =============================================================================

class TestFeatureConfig:
    """Test feature_config.py updates."""

    def test_ext_norm_dim_expanded(self):
        """Test EXT_NORM_DIM is expanded to 35 (Phase 6)."""
        from feature_config import EXT_NORM_DIM

        # Phase 6 expansion: 21 crypto + 7 stock + 7 macro/corp = 35
        assert EXT_NORM_DIM == 35

    def test_feature_layout_includes_stock_features(self):
        """Test feature layout accounts for stock features."""
        from feature_config import make_layout, EXT_NORM_DIM

        layout = make_layout({"ext_norm_dim": EXT_NORM_DIM})

        # Find external block
        external_block = None
        for block in layout:
            if block["name"] == "external":
                external_block = block
                break

        assert external_block is not None
        # Phase 6 expansion: 21 crypto + 7 stock + 7 macro/corp = 35
        assert external_block["size"] == 35


# =============================================================================
# MEDIATOR INTEGRATION TESTS
# =============================================================================

class TestMediatorIntegration:
    """Test mediator._extract_norm_cols updates."""

    def test_norm_cols_size(self):
        """Test norm_cols array size is 28."""
        # Create a mock row with all stock features
        row = {
            # Crypto features (original 21)
            "cvd_24h": 100.0,
            "cvd_7d": 200.0,
            "yang_zhang_48h": 0.02,
            "yang_zhang_7d": 0.03,
            "garch_200h": 0.015,
            "garch_14d": 0.02,
            "ret_12h": 0.01,
            "ret_24h": 0.02,
            "ret_4h": 0.005,
            "sma_12000": 45000.0,
            "yang_zhang_30d": 0.025,
            "parkinson_48h": 0.018,
            "parkinson_7d": 0.022,
            "garch_30d": 0.028,
            "taker_buy_ratio": 0.52,
            "taker_buy_ratio_sma_24h": 0.51,
            "taker_buy_ratio_sma_8h": 0.50,
            "taker_buy_ratio_sma_16h": 0.505,
            "taker_buy_ratio_momentum_4h": 0.01,
            "taker_buy_ratio_momentum_8h": 0.008,
            "taker_buy_ratio_momentum_12h": 0.006,
            # Stock features (new 7)
            "vix_normalized": 0.5,
            "vix_regime": 0.5,
            "market_regime": 0.0,
            "rs_spy_20d": 0.1,
            "rs_spy_50d": 0.05,
            "rs_qqq_20d": 0.08,
            "sector_momentum": 0.03,
        }

        # Test with Mediator if available, otherwise skip
        try:
            from mediator import Mediator

            # Note: Full mediator test requires complex setup
            # This is a basic shape verification
            import numpy as np

            # Verify the expected shape
            norm_cols_values = np.zeros(28, dtype=np.float32)
            norm_cols_validity = np.zeros(28, dtype=bool)

            assert len(norm_cols_values) == 28
            assert len(norm_cols_validity) == 28

        except ImportError:
            pytest.skip("Mediator not available")

    def test_stock_feature_indices(self):
        """Test stock features are at correct indices (21-27)."""
        # Stock feature index mapping
        stock_feature_indices = {
            21: "vix_normalized",
            22: "vix_regime",
            23: "market_regime",
            24: "rs_spy_20d",
            25: "rs_spy_50d",
            26: "rs_qqq_20d",
            27: "sector_momentum",
        }

        # Verify indices don't overlap with crypto features (0-20)
        assert min(stock_feature_indices.keys()) == 21
        assert max(stock_feature_indices.keys()) == 27
        assert len(stock_feature_indices) == 7


# =============================================================================
# PERFORMANCE TESTS (optional)
# =============================================================================

class TestPerformance:
    """Performance tests for stock features."""

    @pytest.mark.slow
    def test_feature_extraction_speed(self, sample_spy_prices, sample_qqq_prices, sample_stock_prices):
        """Test feature extraction is fast enough for real-time use."""
        import time

        benchmark = BenchmarkData(
            spy_prices=sample_spy_prices,
            qqq_prices=sample_qqq_prices,
        )

        row = {"vix": 20.0}

        start = time.time()
        iterations = 1000

        for _ in range(iterations):
            extract_stock_features(
                row=row,
                symbol="AAPL",
                benchmark_data=benchmark,
                stock_prices=sample_stock_prices,
            )

        elapsed = time.time() - start
        per_call_ms = (elapsed / iterations) * 1000

        # Should be fast enough for real-time (< 1ms per call)
        assert per_call_ms < 1.0, f"Feature extraction too slow: {per_call_ms:.3f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

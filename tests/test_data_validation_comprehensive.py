"""Comprehensive tests for data_validation.py - 100% coverage.

This test suite provides complete coverage of the DataValidator including:
- Null and inf value detection
- Positive value checks
- OHLC invariants validation
- Timestamp continuity checks
- Schema and column order validation
- PII detection
"""

import numpy as np
import pandas as pd
import pytest

from data_validation import DataValidator


class TestDataValidatorNulls:
    """Test null and inf value detection."""

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()

    @pytest.fixture
    def valid_df(self):
        """Create valid dataframe."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
                "symbol": ["BTCUSDT"] * 10,
                "open": [50000.0] * 10,
                "high": [50100.0] * 10,
                "low": [49900.0] * 10,
                "close": [50050.0] * 10,
                "volume": [100.0] * 10,
                "quote_asset_volume": [5000000.0] * 10,
                "number_of_trades": [1000] * 10,
                "taker_buy_base_asset_volume": [50.0] * 10,
                "taker_buy_quote_asset_volume": [2500000.0] * 10,
            }
        ).set_index("timestamp")

    def test_no_nulls(self, validator, valid_df):
        """Test validation passes with no nulls."""
        validator._check_for_nulls(valid_df)

    def test_nan_in_open(self, validator, valid_df):
        """Test detection of NaN in open column."""
        valid_df.loc[valid_df.index[0], "open"] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            validator._check_for_nulls(valid_df)

    def test_nan_in_close(self, validator, valid_df):
        """Test detection of NaN in close column."""
        valid_df.loc[valid_df.index[5], "close"] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            validator._check_for_nulls(valid_df)

    def test_inf_in_high(self, validator, valid_df):
        """Test detection of inf in high column."""
        valid_df.loc[valid_df.index[0], "high"] = np.inf

        with pytest.raises(ValueError, match="inf"):
            validator._check_for_nulls(valid_df)

    def test_negative_inf_in_volume(self, validator, valid_df):
        """Test detection of negative inf in volume."""
        valid_df.loc[valid_df.index[0], "quote_asset_volume"] = -np.inf

        with pytest.raises(ValueError, match="inf"):
            validator._check_for_nulls(valid_df)

    def test_missing_columns_handled(self, validator):
        """Test handling of missing columns gracefully."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        validator._check_for_nulls(df)  # Should not crash


class TestDataValidatorPositiveValues:
    """Test positive value checks."""

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()

    @pytest.fixture
    def valid_df(self):
        """Create valid dataframe."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
                "open": [50000.0] * 10,
                "high": [50100.0] * 10,
                "low": [49900.0] * 10,
                "close": [50050.0] * 10,
                "quote_asset_volume": [5000000.0] * 10,
            }
        )

    def test_all_positive(self, validator, valid_df):
        """Test validation passes with all positive values."""
        validator._check_values_are_positive(valid_df)

    def test_zero_open(self, validator, valid_df):
        """Test detection of zero in open column."""
        valid_df.loc[0, "open"] = 0.0

        with pytest.raises(ValueError, match="нулевые или отрицательные"):
            validator._check_values_are_positive(valid_df)

    def test_negative_close(self, validator, valid_df):
        """Test detection of negative in close column."""
        valid_df.loc[5, "close"] = -100.0

        with pytest.raises(ValueError, match="нулевые или отрицательные"):
            validator._check_values_are_positive(valid_df)

    def test_zero_volume(self, validator, valid_df):
        """Test detection of zero volume."""
        valid_df.loc[0, "quote_asset_volume"] = 0.0

        with pytest.raises(ValueError, match="нулевые или отрицательные"):
            validator._check_values_are_positive(valid_df)


class TestDataValidatorOHLCInvariants:
    """Test OHLC invariants validation."""

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()

    @pytest.fixture
    def valid_df(self):
        """Create valid OHLC dataframe."""
        return pd.DataFrame(
            {
                "open": [50000.0, 50100.0, 50200.0],
                "high": [50200.0, 50300.0, 50400.0],
                "low": [49900.0, 50000.0, 50100.0],
                "close": [50100.0, 50200.0, 50300.0],
            }
        )

    def test_valid_ohlc(self, validator, valid_df):
        """Test validation passes with valid OHLC."""
        validator._check_ohlc_invariants(valid_df)

    def test_high_less_than_low(self, validator, valid_df):
        """Test detection of high < low."""
        valid_df.loc[0, "high"] = 49000.0  # Lower than low

        with pytest.raises(ValueError, match="high >= low"):
            validator._check_ohlc_invariants(valid_df)

    def test_high_less_than_open(self, validator, valid_df):
        """Test detection of high < open."""
        # Set high between low and open so it only violates high >= open, not high >= low
        # open[1]=50100, low[1]=50000, close[1]=50200
        # Set high to 50050 which is > low but < open
        valid_df.loc[1, "high"] = 50050.0  # Greater than low[1]=50000, but less than open[1]=50100

        with pytest.raises(ValueError, match="high >= open"):
            validator._check_ohlc_invariants(valid_df)

    def test_high_less_than_close(self, validator, valid_df):
        """Test detection of high < close."""
        # For row 2: open=50200, close=50300, low=50100
        # Set high between max(open, low) and close so only high >= close fails
        # high must be >= low (50100) and >= open (50200) but < close (50300)
        valid_df.loc[2, "high"] = 50250.0  # > low and > open, but < close

        with pytest.raises(ValueError, match="high >= close"):
            validator._check_ohlc_invariants(valid_df)

    def test_low_greater_than_open(self, validator, valid_df):
        """Test detection of low > open."""
        # For row 0: open=50000, high=50200, low=49900, close=50100
        # Set low between open and high so only low <= open fails
        # low must be <= high (50200) and <= close (50100) but > open (50000)
        valid_df.loc[0, "low"] = 50050.0  # > open, but < high and < close

        with pytest.raises(ValueError, match="low <= open"):
            validator._check_ohlc_invariants(valid_df)

    def test_low_greater_than_close(self, validator, valid_df):
        """Test detection of low > close."""
        # For row 1: open=50100, high=50300, low=50000, close=50200
        # To test "low <= close" specifically, we need low > close but low <= open
        # Since close > open in the fixture, we must create a down-bar first
        # Set close to be < open, then set low between close and open
        valid_df.loc[1, "close"] = 50050.0  # Now: open=50100, close=50050 (down-bar)
        valid_df.loc[1, "low"] = 50075.0  # > close (50050), < open (50100), < high (50300)

        with pytest.raises(ValueError, match="low <= close"):
            validator._check_ohlc_invariants(valid_df)

    def test_missing_ohlc_columns(self, validator):
        """Test detection of missing OHLC columns."""
        df = pd.DataFrame({"open": [1, 2, 3]})

        with pytest.raises(ValueError, match="Отсутствуют обязательные OHLC-колонки"):
            validator._check_ohlc_invariants(df)


class TestDataValidatorTimestampContinuity:
    """Test timestamp continuity checks."""

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()

    def test_continuous_datetime_index(self, validator):
        """Test validation passes with continuous datetime index."""
        df = pd.DataFrame(
            {"value": [1, 2, 3, 4, 5]}, index=pd.date_range("2024-01-01", periods=5, freq="1min")
        )

        validator._check_timestamp_continuity(df, frequency="1min")

    def test_gap_in_datetime_index(self, validator):
        """Test detection of gap in datetime index."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1min")
        dates = dates.delete(5)  # Remove one timestamp

        df = pd.DataFrame({"value": list(range(len(dates)))}, index=dates)

        with pytest.raises(ValueError, match="разрыв в непрерывности"):
            validator._check_timestamp_continuity(df, frequency="1min")

    def test_non_monotonic_datetime_index(self, validator):
        """Test detection of non-monotonic datetime index."""
        dates = list(pd.date_range("2024-01-01", periods=10, freq="1min"))
        dates[5], dates[6] = dates[6], dates[5]  # Swap two timestamps

        df = pd.DataFrame({"value": list(range(len(dates)))}, index=pd.DatetimeIndex(dates))

        with pytest.raises(ValueError, match="монотонность"):
            validator._check_timestamp_continuity(df, frequency="1min")

    def test_auto_frequency_detection(self, validator):
        """Test automatic frequency detection."""
        df = pd.DataFrame(
            {"value": [1, 2, 3, 4, 5]}, index=pd.date_range("2024-01-01", periods=5, freq="5min")
        )

        # Should auto-detect 5min frequency
        validator._check_timestamp_continuity(df)

    def test_numeric_index_continuous(self, validator):
        """Test numeric index continuity."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4])

        validator._check_timestamp_continuity(df)

    def test_numeric_index_gap(self, validator):
        """Test detection of gap in numeric index."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=[0, 1, 2, 4, 5])  # Gap at 3

        with pytest.raises(ValueError, match="timestamp gap detected"):
            validator._check_timestamp_continuity(df)

    def test_single_row_dataframe(self, validator):
        """Test handling of single row dataframe."""
        df = pd.DataFrame({"value": [1]}, index=[0])

        validator._check_timestamp_continuity(df)  # Should not raise


class TestDataValidatorSchemaAndOrder:
    """Test schema and column order validation."""

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()

    @pytest.fixture
    def valid_df(self):
        """Create dataframe with correct schema."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
                "symbol": ["BTCUSDT"] * 10,
                "open": [50000.0] * 10,
                "high": [50100.0] * 10,
                "low": [49900.0] * 10,
                "close": [50050.0] * 10,
                "volume": [100.0] * 10,
                "quote_asset_volume": [5000000.0] * 10,
                "number_of_trades": [1000] * 10,
                "taker_buy_base_asset_volume": [50.0] * 10,
                "taker_buy_quote_asset_volume": [2500000.0] * 10,
            }
        )

    def test_correct_schema(self, validator, valid_df):
        """Test validation passes with correct schema."""
        validator._check_schema_and_order(valid_df)

    def test_missing_required_column(self, validator, valid_df):
        """Test detection of missing required column."""
        df = valid_df.drop(columns=["volume"])

        with pytest.raises(ValueError, match="Отсутствуют обязательные колонки"):
            validator._check_schema_and_order(df)

    def test_wrong_column_order(self, validator, valid_df):
        """Test detection of wrong column order."""
        # Swap two columns
        cols = list(valid_df.columns)
        cols[2], cols[3] = cols[3], cols[2]
        df = valid_df[cols]

        with pytest.raises(ValueError, match="Нарушен порядок колонок"):
            validator._check_schema_and_order(df)

    def test_extra_columns_allowed(self, validator, valid_df):
        """Test that extra columns after prefix are allowed."""
        valid_df["extra_column"] = 1
        validator._check_schema_and_order(valid_df)


class TestDataValidatorPII:
    """Test PII detection."""

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()

    def test_no_pii(self, validator):
        """Test validation passes without PII."""
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "value": [1, 2, 3]})

        validator._check_no_pii(df)

    def test_email_detection(self, validator):
        """Test detection of email addresses."""
        df = pd.DataFrame({"comment": ["Hello", "Contact me at john@example.com", "Thanks"]})

        with pytest.raises(ValueError, match="email|персональные данные"):
            validator._check_no_pii(df)

    def test_ssn_detection(self, validator):
        """Test detection of SSN."""
        df = pd.DataFrame({"comment": ["Hello", "SSN: 123-45-6789", "Thanks"]})

        with pytest.raises(ValueError, match="персональные данные"):
            validator._check_no_pii(df)

    def test_phone_detection(self, validator):
        """Test detection of phone numbers."""
        df = pd.DataFrame({"comment": ["Hello", "Call me at +1 555-123-4567", "Thanks"]})

        with pytest.raises(ValueError, match="персональные данные"):
            validator._check_no_pii(df)

    def test_numeric_columns_ignored(self, validator):
        """Test that numeric columns are not checked for PII."""
        df = pd.DataFrame({"value": [123456789, 555123456, 111111111]})  # Looks like SSN

        validator._check_no_pii(df)  # Should pass


class TestDataValidatorIntegration:
    """Test full validation pipeline."""

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()

    def test_full_validation_pass(self, validator):
        """Test full validation pipeline passes."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1min"),
                "symbol": ["BTCUSDT"] * 100,
                "open": np.random.uniform(49000, 51000, 100),
                "high": np.random.uniform(51000, 52000, 100),
                "low": np.random.uniform(48000, 49000, 100),
                "close": np.random.uniform(49000, 51000, 100),
                "volume": np.random.uniform(100, 1000, 100),
                "quote_asset_volume": np.random.uniform(5000000, 50000000, 100),
                "number_of_trades": np.random.randint(500, 2000, 100),
                "taker_buy_base_asset_volume": np.random.uniform(50, 500, 100),
                "taker_buy_quote_asset_volume": np.random.uniform(2500000, 25000000, 100),
            }
        ).set_index(
            "timestamp", drop=False
        )  # Keep timestamp as column for schema check

        # Fix OHLC invariants
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        result = validator.validate(df, frequency="1min")
        assert result is df

    def test_validation_stops_at_first_error(self, validator):
        """Test validation stops at first error."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
                "symbol": ["BTCUSDT"] * 10,
                "open": [np.nan] * 10,  # Will fail null check
                "high": [-100.0] * 10,  # Would fail positive check
                "low": [100.0] * 10,
                "close": [0.0] * 10,
                "volume": [100.0] * 10,
                "quote_asset_volume": [5000000.0] * 10,
                "number_of_trades": [1000] * 10,
                "taker_buy_base_asset_volume": [50.0] * 10,
                "taker_buy_quote_asset_volume": [2500000.0] * 10,
            }
        ).set_index(
            "timestamp", drop=False
        )  # Keep timestamp as column for schema check

        # Should fail at null check, not positive check
        with pytest.raises(ValueError, match="NaN"):
            validator.validate(df, frequency="1min")


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()

    def test_empty_dataframe(self, validator):
        """Test validation with empty dataframe."""
        df = pd.DataFrame()

        # Should handle gracefully or raise informative error
        try:
            validator.validate(df)
        except (ValueError, KeyError):
            pass  # Expected

    def test_single_row_dataframe(self, validator):
        """Test validation with single row."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=1, freq="1min"),
                "symbol": ["BTCUSDT"],
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [100.0],
                "quote_asset_volume": [5000000.0],
                "number_of_trades": [1000],
                "taker_buy_base_asset_volume": [50.0],
                "taker_buy_quote_asset_volume": [2500000.0],
            }
        ).set_index(
            "timestamp", drop=False
        )  # Keep timestamp as column for schema check

        validator.validate(df)

    def test_very_large_dataframe(self, validator):
        """Test validation with very large dataframe."""
        n = 100000
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
                "symbol": ["BTCUSDT"] * n,
                "open": [50000.0] * n,
                "high": [50100.0] * n,
                "low": [49900.0] * n,
                "close": [50050.0] * n,
                "volume": [100.0] * n,
                "quote_asset_volume": [5000000.0] * n,
                "number_of_trades": [1000] * n,
                "taker_buy_base_asset_volume": [50.0] * n,
                "taker_buy_quote_asset_volume": [2500000.0] * n,
            }
        ).set_index(
            "timestamp", drop=False
        )  # Keep timestamp as column for schema check

        validator.validate(df, frequency="1min")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

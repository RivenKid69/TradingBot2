"""
Tests for backtest disclaimer injection service.

Verifies that all backtest results include mandatory disclaimers
and are clearly marked as simulations.

References:
    - SEC Rule 206(4)-1: Performance advertising
    - FCA COBS 4.6: Past performance
    - ESMA Guidelines on Marketing Communications
"""

import pytest
from datetime import datetime, timezone

from services.backtest.disclaimer_injection import (
    BacktestDisclaimerService,
    BacktestDisclaimer,
    BacktestResultWithDisclaimer,
    BACKTEST_DISCLAIMER,
    inject_disclaimer,
    format_backtest_result,
    validate_backtest_output,
)


class TestBacktestDisclaimer:
    """Tests for BacktestDisclaimer content."""

    def test_disclaimer_has_warning(self):
        """Verify disclaimer has a short warning message."""
        assert BACKTEST_DISCLAIMER.warning is not None
        assert len(BACKTEST_DISCLAIMER.warning) > 0
        assert "simulation" in BACKTEST_DISCLAIMER.warning.lower()

    def test_disclaimer_has_legal_text(self):
        """Verify disclaimer has full legal text."""
        assert BACKTEST_DISCLAIMER.legal is not None
        assert len(BACKTEST_DISCLAIMER.legal) > 100

    def test_disclaimer_legal_contains_required_elements(self):
        """Verify legal text contains required regulatory elements."""
        legal_lower = BACKTEST_DISCLAIMER.legal.lower()

        # Must mention historical/simulation nature
        assert "historical" in legal_lower or "simulation" in legal_lower

        # Must have past performance warning
        assert "past performance" in legal_lower

        # Must mention limitations
        assert "limitation" in legal_lower

        # Must mention risk
        assert "risk" in legal_lower

    def test_disclaimer_has_version(self):
        """Verify disclaimer has version."""
        assert BACKTEST_DISCLAIMER.version is not None
        assert "." in BACKTEST_DISCLAIMER.version

    def test_disclaimer_has_limitations_list(self):
        """Verify disclaimer has list of limitations."""
        assert BACKTEST_DISCLAIMER.limitations is not None
        assert len(BACKTEST_DISCLAIMER.limitations) > 0

    def test_disclaimer_has_risk_factors(self):
        """Verify disclaimer has list of risk factors."""
        assert BACKTEST_DISCLAIMER.risk_factors is not None
        assert len(BACKTEST_DISCLAIMER.risk_factors) > 0

    def test_disclaimer_to_dict(self):
        """Verify disclaimer serialization."""
        data = BACKTEST_DISCLAIMER.to_dict()

        assert "warning" in data
        assert "legal" in data
        assert "version" in data
        assert "limitations" in data
        assert "risk_factors" in data


class TestBacktestResultWithDisclaimer:
    """Tests for BacktestResultWithDisclaimer class."""

    @pytest.fixture
    def sample_results(self):
        """Sample backtest results."""
        return {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": -0.12,
            "win_rate": 0.55,
        }

    @pytest.fixture
    def wrapped_result(self, sample_results):
        """Create wrapped result."""
        return BacktestResultWithDisclaimer(
            disclaimer=BACKTEST_DISCLAIMER,
            results=sample_results,
            generated_at=datetime.now(timezone.utc),
            strategy_name="test_strategy",
            strategy_version="1.0",
            data_period={"start": "2020-01-01", "end": "2023-12-31"},
        )

    def test_is_simulation_always_true(self, wrapped_result):
        """Verify is_simulation is set to True on construction."""
        assert wrapped_result.is_simulation is True

    def test_is_investment_advice_always_false(self, wrapped_result):
        """Verify is_investment_advice is set to False on construction."""
        assert wrapped_result.is_investment_advice is False

    def test_to_dict_includes_disclaimer_first(self, wrapped_result):
        """Verify disclaimer is prominently included in serialization."""
        data = wrapped_result.to_dict()

        assert "disclaimer" in data
        assert "is_simulation" in data
        assert "is_investment_advice" in data
        assert "results" in data

        # Verify simulation flags
        assert data["is_simulation"] is True
        assert data["is_investment_advice"] is False

    def test_to_dict_includes_results(self, wrapped_result):
        """Verify results are included in serialization."""
        data = wrapped_result.to_dict()
        assert data["results"]["sharpe_ratio"] == 1.5
        assert data["results"]["total_return"] == 0.25

    def test_to_dict_includes_metadata(self, wrapped_result):
        """Verify metadata is included."""
        data = wrapped_result.to_dict()
        assert data["strategy_name"] == "test_strategy"
        assert data["data_period"]["start"] == "2020-01-01"

    def test_get_short_disclaimer(self, wrapped_result):
        """Verify short disclaimer accessor."""
        short = wrapped_result.get_short_disclaimer()
        assert "simulation" in short.lower()

    def test_get_full_disclaimer(self, wrapped_result):
        """Verify full disclaimer accessor."""
        full = wrapped_result.get_full_disclaimer()
        assert "past performance" in full.lower()


class TestBacktestDisclaimerService:
    """Tests for BacktestDisclaimerService."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return BacktestDisclaimerService()

    @pytest.fixture
    def sample_results(self):
        """Sample raw backtest results."""
        return {
            "sharpe_ratio": 1.2,
            "total_return": 0.18,
            "max_drawdown": -0.15,
        }

    def test_wrap_result_returns_disclaimer(self, service, sample_results):
        """Verify wrap_result includes disclaimer."""
        result = service.wrap_result(sample_results)

        assert result.disclaimer is not None
        assert result.disclaimer == BACKTEST_DISCLAIMER

    def test_wrap_result_preserves_results(self, service, sample_results):
        """Verify wrap_result preserves original results."""
        result = service.wrap_result(sample_results)

        assert result.results == sample_results
        assert result.results["sharpe_ratio"] == 1.2

    def test_wrap_result_with_metadata(self, service, sample_results):
        """Verify wrap_result handles metadata correctly."""
        result = service.wrap_result(
            raw_results=sample_results,
            strategy_name="momentum_v1",
            strategy_version="2.0",
            start_date="2021-01-01",
            end_date="2023-12-31",
            metadata={"notes": "test run"}
        )

        assert result.strategy_name == "momentum_v1"
        assert result.strategy_version == "2.0"
        assert result.data_period["start"] == "2021-01-01"
        assert result.data_period["end"] == "2023-12-31"
        assert result.metadata["notes"] == "test run"

    def test_wrap_result_has_timestamp(self, service, sample_results):
        """Verify wrap_result includes timestamp."""
        result = service.wrap_result(sample_results)

        assert result.generated_at is not None
        assert isinstance(result.generated_at, datetime)

    def test_inject_into_dict(self, service, sample_results):
        """Verify inject_into_dict returns proper dictionary."""
        result = service.inject_into_dict(sample_results, strategy_name="test")

        assert isinstance(result, dict)
        assert "disclaimer" in result
        assert "is_simulation" in result
        assert result["is_simulation"] is True
        assert result["results"] == sample_results

    def test_validate_result_has_disclaimer_valid(self, service, sample_results):
        """Verify validation passes for properly formatted result."""
        result = service.inject_into_dict(sample_results)
        assert service.validate_result_has_disclaimer(result) is True

    def test_validate_result_has_disclaimer_missing_disclaimer(self, service):
        """Verify validation fails when disclaimer is missing."""
        result = {
            "results": {"sharpe": 1.0},
            "is_simulation": True,
            "is_investment_advice": False,
        }
        assert service.validate_result_has_disclaimer(result) is False

    def test_validate_result_has_disclaimer_wrong_simulation_flag(self, service):
        """Verify validation fails when is_simulation is False."""
        result = {
            "disclaimer": BACKTEST_DISCLAIMER.to_dict(),
            "results": {"sharpe": 1.0},
            "is_simulation": False,  # Wrong!
            "is_investment_advice": False,
        }
        assert service.validate_result_has_disclaimer(result) is False

    def test_validate_result_has_disclaimer_wrong_advice_flag(self, service):
        """Verify validation fails when is_investment_advice is True."""
        result = {
            "disclaimer": BACKTEST_DISCLAIMER.to_dict(),
            "results": {"sharpe": 1.0},
            "is_simulation": True,
            "is_investment_advice": True,  # Wrong!
        }
        assert service.validate_result_has_disclaimer(result) is False

    def test_get_api_warnings(self, service):
        """Verify get_api_warnings returns proper warnings."""
        warnings = service.get_api_warnings()

        assert "risk_warning" in warnings
        assert "simulation_notice" in warnings
        assert "not_advice" in warnings

        # Each should have content
        for key, value in warnings.items():
            assert len(value) > 20

    def test_custom_disclaimer(self):
        """Verify custom disclaimer can be used."""
        custom_disclaimer = BacktestDisclaimer(
            warning="CUSTOM WARNING",
            legal="Custom legal text for testing.",
            version="custom",
            limitations=["Custom limitation"],
            risk_factors=["Custom risk"],
        )
        service = BacktestDisclaimerService(disclaimer=custom_disclaimer)

        result = service.wrap_result({"test": 1})
        assert result.disclaimer.warning == "CUSTOM WARNING"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def sample_results(self):
        """Sample backtest results."""
        return {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.10,
        }

    def test_inject_disclaimer(self, sample_results):
        """Verify inject_disclaimer function works."""
        result = inject_disclaimer(sample_results)

        assert "disclaimer" in result
        assert result["is_simulation"] is True
        assert result["is_investment_advice"] is False

    def test_format_backtest_result(self, sample_results):
        """Verify format_backtest_result function works."""
        result = format_backtest_result(
            results=sample_results,
            strategy_name="momentum_v1",
            start_date="2020-01-01",
            end_date="2023-12-31"
        )

        assert "disclaimer" in result
        assert result["strategy_name"] == "momentum_v1"
        assert result["data_period"]["start"] == "2020-01-01"
        assert result["is_simulation"] is True

    def test_validate_backtest_output_valid(self, sample_results):
        """Verify validate_backtest_output for valid output."""
        result = inject_disclaimer(sample_results)
        assert validate_backtest_output(result) is True

    def test_validate_backtest_output_invalid(self):
        """Verify validate_backtest_output for invalid output."""
        result = {"results": {"test": 1}}  # Missing disclaimer
        assert validate_backtest_output(result) is False


class TestDisclaimerLimitations:
    """Tests verifying disclaimer limitations content."""

    def test_limitations_mention_hindsight(self):
        """Verify limitations mention hindsight bias."""
        limitations_text = " ".join(BACKTEST_DISCLAIMER.limitations).lower()
        assert "hindsight" in limitations_text or "historical" in limitations_text

    def test_limitations_mention_execution(self):
        """Verify limitations mention execution assumptions."""
        limitations_text = " ".join(BACKTEST_DISCLAIMER.limitations).lower()
        assert "execution" in limitations_text

    def test_limitations_mention_slippage(self):
        """Verify limitations mention slippage."""
        limitations_text = " ".join(BACKTEST_DISCLAIMER.limitations).lower()
        assert "slippage" in limitations_text

    def test_limitations_mention_fees(self):
        """Verify limitations mention fees."""
        limitations_text = " ".join(BACKTEST_DISCLAIMER.limitations).lower()
        assert "fee" in limitations_text

    def test_limitations_are_substantial(self):
        """Verify there are enough limitations listed."""
        assert len(BACKTEST_DISCLAIMER.limitations) >= 5


class TestRiskFactors:
    """Tests verifying risk factors content."""

    def test_risk_factors_mention_past_performance(self):
        """Verify risk factors mention past performance."""
        risks_text = " ".join(BACKTEST_DISCLAIMER.risk_factors).lower()
        assert "past performance" in risks_text

    def test_risk_factors_mention_loss(self):
        """Verify risk factors mention potential loss."""
        risks_text = " ".join(BACKTEST_DISCLAIMER.risk_factors).lower()
        assert "lose" in risks_text or "loss" in risks_text

    def test_risk_factors_mention_market_changes(self):
        """Verify risk factors mention market changes."""
        risks_text = " ".join(BACKTEST_DISCLAIMER.risk_factors).lower()
        assert "market" in risks_text and "change" in risks_text

    def test_risk_factors_are_substantial(self):
        """Verify there are enough risk factors listed."""
        assert len(BACKTEST_DISCLAIMER.risk_factors) >= 5

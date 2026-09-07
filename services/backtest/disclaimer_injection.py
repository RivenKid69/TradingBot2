"""
Automatic disclaimer injection into backtest results.

Ensures all backtest results include mandatory risk warnings and
disclaimers to comply with financial regulations and protect users.

References:
    - SEC Rule 206(4)-1: Performance advertising requirements
    - FCA COBS 4.6: Past performance disclosures
    - ESMA Guidelines on Marketing Communications
    - MiFID II Article 24: Client information requirements

Features:
    - Automatic disclaimer injection into all backtest outputs
    - Multiple warning levels (summary, full legal, risk metrics)
    - Standardized result format with clear simulation flags
    - Version tracking for audit trail

Example:
    >>> service = BacktestDisclaimerService()
    >>> raw_results = {"sharpe_ratio": 1.5, "total_return": 0.25}
    >>> result = service.wrap_result(raw_results, strategy_name="momentum_v1")
    >>> assert result.is_simulation == True
    >>> assert "past performance" in result.disclaimer.legal.lower()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ============================================================================
# DISCLAIMER CONTENT
# ============================================================================


@dataclass(frozen=True)
class BacktestDisclaimer:
    """
    Standardized disclaimer for backtest results.

    Attributes:
        warning: Short warning message (for UI display)
        legal: Full legal disclaimer text
        version: Version of the disclaimer
        limitations: List of specific limitations
        risk_factors: List of risk factors to consider
    """

    warning: str
    legal: str
    version: str
    limitations: List[str]
    risk_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "warning": self.warning,
            "legal": self.legal,
            "version": self.version,
            "limitations": self.limitations,
            "risk_factors": self.risk_factors,
        }


# Standard disclaimer instance
BACKTEST_DISCLAIMER = BacktestDisclaimer(
    warning="SIMULATION ONLY - NOT ACTUAL TRADING RESULTS",
    legal=(
        "IMPORTANT: These results are based on historical simulation and do NOT "
        "represent actual trading. Past performance does NOT guarantee future results. "
        "Simulated results have inherent limitations: (1) they are prepared with the benefit "
        "of hindsight, (2) they do not reflect actual slippage, liquidity constraints, or "
        "market impact, (3) they assume order execution which may not occur in live trading. "
        "Trading involves substantial risk of loss. You should not rely solely on backtest "
        "results when making trading decisions."
    ),
    version="1.0.0",
    limitations=[
        "Results are based on historical data and may not predict future performance",
        "Simulation assumes perfect execution which rarely occurs in live trading",
        "Backtests do not account for market impact of your actual trades",
        "Slippage models are approximations and may underestimate real costs",
        "Historical data may contain errors, gaps, or survivorship bias",
        "Fee calculations are estimates and may differ from actual broker fees",
        "Results do not account for taxes or other real-world costs",
        "Liquidity constraints are modeled but may not reflect actual availability",
    ],
    risk_factors=[
        "Past performance does not guarantee future results",
        "Markets can change, making historical patterns unreliable",
        "Black swan events and market crashes are underrepresented in backtests",
        "You may lose more than your initial investment",
        "Leverage amplifies both gains and losses",
        "Strategy may perform differently across different market regimes",
        "Technical issues may cause execution failures in live trading",
    ],
)


# ============================================================================
# RESULT WRAPPER
# ============================================================================


@dataclass
class BacktestResultWithDisclaimer:
    """
    Backtest result wrapped with mandatory disclaimers.

    This class ensures all backtest results include proper warnings
    and are clearly marked as simulations.

    Attributes:
        disclaimer: Standardized disclaimer object
        results: The actual backtest results
        generated_at: Timestamp of result generation
        is_simulation: Always True (explicit flag)
        is_investment_advice: Always False (explicit flag)
        strategy_name: Name of the tested strategy
        strategy_version: Version of the strategy (if available)
        data_period: Time period of the backtest
        metadata: Additional context
    """

    disclaimer: BacktestDisclaimer
    results: Dict[str, Any]
    generated_at: datetime
    is_simulation: bool = True
    is_investment_advice: bool = False
    strategy_name: Optional[str] = None
    strategy_version: Optional[str] = None
    data_period: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure simulation flags are always correct."""
        # These must always be True/False respectively
        # Prevents any accidental modification
        self.is_simulation = True
        self.is_investment_advice = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for API response or storage.

        The disclaimer is always included at the top level
        to ensure it's visible in any output format.
        """
        return {
            # Disclaimer prominently at top
            "disclaimer": self.disclaimer.to_dict(),
            # Explicit simulation flags
            "is_simulation": True,
            "is_investment_advice": False,
            # Metadata
            "generated_at": self.generated_at.isoformat(),
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "data_period": self.data_period,
            # Actual results last (after disclaimers)
            "results": self.results,
            # Additional metadata
            "metadata": self.metadata,
        }

    def get_short_disclaimer(self) -> str:
        """Get the short warning message for UI display."""
        return self.disclaimer.warning

    def get_full_disclaimer(self) -> str:
        """Get the full legal disclaimer text."""
        return self.disclaimer.legal


# ============================================================================
# SERVICE CLASS
# ============================================================================


class BacktestDisclaimerService:
    """
    Service for injecting disclaimers into backtest results.

    Ensures all backtest outputs include mandatory risk warnings
    and are clearly marked as simulations (not real trading results).

    Usage:
        >>> service = BacktestDisclaimerService()
        >>>
        >>> # In your backtest execution code:
        >>> raw_results = run_backtest(strategy, data)
        >>> result = service.wrap_result(
        ...     raw_results,
        ...     strategy_name="my_strategy",
        ...     start_date="2020-01-01",
        ...     end_date="2023-12-31"
        ... )
        >>>
        >>> # Result is now wrapped with disclaimers
        >>> return result.to_dict()
    """

    def __init__(self, disclaimer: Optional[BacktestDisclaimer] = None):
        """
        Initialize the service.

        Args:
            disclaimer: Custom disclaimer (or use default)
        """
        self._disclaimer = disclaimer or BACKTEST_DISCLAIMER

    @property
    def disclaimer(self) -> BacktestDisclaimer:
        """Get the current disclaimer."""
        return self._disclaimer

    def wrap_result(
        self,
        raw_results: Dict[str, Any],
        strategy_name: Optional[str] = None,
        strategy_version: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BacktestResultWithDisclaimer:
        """
        Wrap raw backtest results with disclaimers.

        Args:
            raw_results: The computed backtest results
            strategy_name: Name of the strategy tested
            strategy_version: Version of the strategy
            start_date: Start date of backtest period (ISO format)
            end_date: End date of backtest period (ISO format)
            metadata: Additional context to include

        Returns:
            BacktestResultWithDisclaimer with all warnings included
        """
        data_period = None
        if start_date or end_date:
            data_period = {
                "start": start_date,
                "end": end_date,
            }

        return BacktestResultWithDisclaimer(
            disclaimer=self._disclaimer,
            results=raw_results,
            generated_at=datetime.now(timezone.utc),
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            data_period=data_period,
            metadata=metadata or {},
        )

    def inject_into_dict(
        self, results_dict: Dict[str, Any], strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Inject disclaimers into an existing results dictionary.

        Alternative to wrap_result when you need a dict output directly.

        Args:
            results_dict: Existing results dictionary
            strategy_name: Name of the strategy

        Returns:
            Dictionary with disclaimers injected
        """
        wrapped = self.wrap_result(
            raw_results=results_dict,
            strategy_name=strategy_name,
        )
        return wrapped.to_dict()

    def validate_result_has_disclaimer(self, result: Dict[str, Any]) -> bool:
        """
        Validate that a result dictionary contains required disclaimers.

        Useful for checking results before returning to users.

        Args:
            result: Result dictionary to validate

        Returns:
            True if disclaimers are present and valid
        """
        if "disclaimer" not in result:
            return False

        disclaimer = result.get("disclaimer", {})

        required_fields = ["warning", "legal", "version"]
        for field_name in required_fields:
            if field_name not in disclaimer:
                return False

        # Check simulation flags
        if result.get("is_simulation") is not True:
            return False

        if result.get("is_investment_advice") is not False:
            return False

        return True

    def get_api_warnings(self) -> Dict[str, Any]:
        """
        Get warning messages for API documentation.

        Returns standardized warning text for inclusion in
        API responses and documentation.
        """
        return {
            "risk_warning": (
                "RISK WARNING: Trading involves substantial risk of loss. "
                "Past performance does not guarantee future results."
            ),
            "simulation_notice": (
                "SIMULATION NOTICE: All backtest results are simulations "
                "based on historical data and do not represent actual trading."
            ),
            "not_advice": (
                "NOT INVESTMENT ADVICE: This platform provides software tools, "
                "not investment recommendations or financial advice."
            ),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def inject_disclaimer(backtest_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to inject disclaimers into backtest results.

    Args:
        backtest_result: Raw backtest result dictionary

    Returns:
        Result dictionary with disclaimers added

    Example:
        >>> raw_results = {"sharpe_ratio": 1.2, "return": 0.15}
        >>> result = inject_disclaimer(raw_results)
        >>> assert "disclaimer" in result
    """
    service = BacktestDisclaimerService()
    return service.inject_into_dict(backtest_result)


def format_backtest_result(
    results: Dict[str, Any], strategy_name: str, start_date: str, end_date: str
) -> Dict[str, Any]:
    """
    Format backtest results with full metadata and disclaimers.

    Convenience function for common use case.

    Args:
        results: Raw backtest results
        strategy_name: Name of the strategy
        start_date: Backtest start date (ISO format)
        end_date: Backtest end date (ISO format)

    Returns:
        Fully formatted result dictionary

    Example:
        >>> results = {"sharpe_ratio": 1.5, "max_drawdown": -0.12}
        >>> formatted = format_backtest_result(
        ...     results, "momentum_v1", "2020-01-01", "2023-12-31"
        ... )
    """
    service = BacktestDisclaimerService()
    wrapped = service.wrap_result(
        raw_results=results,
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date,
    )
    return wrapped.to_dict()


def validate_backtest_output(result: Dict[str, Any]) -> bool:
    """
    Validate that backtest output includes required disclaimers.

    Args:
        result: Result dictionary to validate

    Returns:
        True if properly formatted with disclaimers
    """
    service = BacktestDisclaimerService()
    return service.validate_result_has_disclaimer(result)

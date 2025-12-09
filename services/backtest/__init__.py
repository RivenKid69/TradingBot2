"""
Backtest Services Module.

Provides backtest-related services including disclaimer injection
and result formatting with mandatory risk warnings.

Components:
    - BacktestDisclaimerService: Automatic disclaimer injection into backtest results

References:
    - SEC Rule 206(4)-1: Performance advertising
    - FCA COBS 4.6: Past performance disclaimers
    - ESMA Guidelines on Marketing Communications
"""

from services.backtest.disclaimer_injection import (
    BacktestDisclaimerService,
    BacktestDisclaimer,
    BacktestResultWithDisclaimer,
    BACKTEST_DISCLAIMER,
    inject_disclaimer,
    format_backtest_result,
    validate_backtest_output,
)

__all__ = [
    "BacktestDisclaimerService",
    "BacktestDisclaimer",
    "BacktestResultWithDisclaimer",
    "BACKTEST_DISCLAIMER",
    "inject_disclaimer",
    "format_backtest_result",
    "validate_backtest_output",
]

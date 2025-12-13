# -*- coding: utf-8 -*-
"""
CCEA Cloud Research Module.

Provides research and backtesting capabilities.
Uses SimExecutionEngine from shared package.

IMPORTANT: This module does NOT execute live trades.
It runs simulations using historical data.

Key Components:
- BacktestRunner: Runs historical backtests
- ResearchJobRunner: Runs research jobs in isolation
"""

from typing import Final

ZONE: Final[str] = "cloud"

from .backtest import (
    BacktestRunner,
    BacktestConfig,
    BacktestResult,
)

__all__ = [
    "BacktestRunner",
    "BacktestConfig",
    "BacktestResult",
]

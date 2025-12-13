# -*- coding: utf-8 -*-
"""
Strategy Runner Package.

Provides unified interface for running strategies in different zones:
- Cloud: SimulationRunner (backtest, paper trading simulation)
- Agent: LiveRunner (live execution with risk management)

Key Principle:
    Strategies produce OrderIntents -> Runners process them
    Cloud runners NEVER execute real trades
    Agent runners enforce risk controls before execution
"""

from typing import Final, List

__version__ = "1.0.0"

# Runner types
RUNNER_TYPES: Final[List[str]] = [
    "SimulationRunner",
    "RunnerResult",
    "RunnerConfig",
]

__all__ = RUNNER_TYPES

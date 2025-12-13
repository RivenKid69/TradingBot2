# -*- coding: utf-8 -*-
"""
CCEA Shared Simulation Module.

Provides simulation/backtest execution engine that can be used in Cloud zone.
This processes OrderIntents for simulation WITHOUT connecting to real brokers.

Key Components:
- SimExecutionEngine: Executes intents in simulation mode
- BacktestRunner: Runs historical backtests
- PaperTradingEngine: Simulates live trading without real orders

IMPORTANT: This module contains NO real broker connections.
It only simulates order execution for research purposes.
"""

from typing import Final, List

ZONE: Final[str] = "shared"

# What this module provides
SIMULATION_COMPONENTS: Final[List[str]] = [
    "SimExecutionEngine",
    "BacktestRunner",
    "PaperTradingEngine",
    "SimulatedFill",
    "SimulatedPosition",
]

# Re-export from existing simulation modules
# These will be imported when the modules exist
# from execution_sim import SimExecutionEngine

from .engine import (
    SimExecutionEngine,
    SimulatedFill,
    SimulatedPosition,
    SimulationConfig,
)

__all__ = [
    "SimExecutionEngine",
    "SimulatedFill",
    "SimulatedPosition",
    "SimulationConfig",
    "ZONE",
    "SIMULATION_COMPONENTS",
]

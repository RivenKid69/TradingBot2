# -*- coding: utf-8 -*-
"""
Agent Runner Package - AGENT ZONE ONLY.

Provides LiveRunner for executing strategies with real order submission.
This package is PROHIBITED in Cloud zone.

Key Features:
- Live order execution via LiveExecutionEngine
- Policy firewall enforcement
- Hard cap validation
- Risk checks before every order
- Reconciliation and recovery
"""

from typing import Final, List

__version__ = "1.0.0"

# Runner types provided by this package
AGENT_RUNNER_TYPES: Final[List[str]] = [
    "LiveRunner",
    "LiveRunnerConfig",
]

__all__ = AGENT_RUNNER_TYPES

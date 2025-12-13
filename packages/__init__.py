# -*- coding: utf-8 -*-
"""
CCEA Packages - Phase 2 Implementation.

Three-zone architecture:
- packages/shared: Safe contracts, models, simulation, data-only adapters
- packages/agent: Live execution, broker connectors, vault, policy firewall
- packages/cloud: Research, backtest, training, control plane (NO trading code)

Key Principle (non-negotiable):
    Cloud = research/build/monitoring/control plane (lifecycle requests)
    Agent = secrets + live loop + risk enforce + order creation/sending

Cloud NEVER:
    - Stores broker API keys
    - Generates or sends orders
    - Has access to trading endpoints
    - Contains order-like payloads in protocol
"""

__version__ = "2.0.0"

from typing import Final

# Phase 2 implementation complete flag
PHASE_2_COMPLETE: Final[bool] = True

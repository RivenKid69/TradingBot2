# -*- coding: utf-8 -*-
"""
CCEA Test Configuration.

Phase 2 Implementation: Pytest fixtures for CCEA tests.
"""

from __future__ import annotations

import pytest
import sys
import tempfile
from pathlib import Path
from decimal import Decimal

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_order_intent():
    """Create a sample OrderIntent."""
    from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

    return OrderIntent(
        strategy_id="test_strategy",
        symbol="AAPL",
        intent_type=IntentType.OPEN,
        side=IntentSide.LONG,
        target_quantity=Decimal("100"),
    )


@pytest.fixture
def sample_risk_config():
    """Create a sample RiskConfig."""
    from packages.shared.contracts.config import RiskConfig

    return RiskConfig(
        max_position_pct=Decimal("0.05"),
        max_drawdown_pct=Decimal("0.10"),
        max_daily_loss_pct=Decimal("0.02"),
    )


@pytest.fixture
def sim_engine():
    """Create a SimExecutionEngine."""
    from packages.shared.simulation.engine import SimExecutionEngine

    return SimExecutionEngine(
        initial_capital=Decimal("100000"),
        commission_rate=Decimal("0.001"),
    )


@pytest.fixture
def temp_vault(temp_directory):
    """Create a temporary LocalVault."""
    from packages.agent.vault.local_vault import LocalVault

    return LocalVault(vault_path=temp_directory / "vault.db")


@pytest.fixture
def policy_firewall():
    """Create a PolicyFirewall with default config."""
    from packages.agent.policy.firewall import PolicyFirewall, PolicyConfig

    return PolicyFirewall(
        policy_config=PolicyConfig(
            max_position_pct_ceiling=Decimal("0.10"),
            max_daily_loss_pct_ceiling=Decimal("0.05"),
            require_approval_for_trading_impacting=True,
        )
    )


@pytest.fixture
def hard_cap_enforcer():
    """Create a HardCapEnforcer."""
    from packages.agent.policy.hard_caps import HardCapEnforcer, HardCaps

    return HardCapEnforcer(
        hard_caps=HardCaps(
            max_position_size=Decimal("10000"),
            max_order_size=Decimal("1000"),
            max_daily_orders=100,
        )
    )

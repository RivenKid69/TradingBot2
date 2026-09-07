# -*- coding: utf-8 -*-
import numpy as np
import pytest

pytest.importorskip("sortedcontainers")

try:
    from lob_state_cython import (
        EnvState,
        run_full_step_logic_cython,
        SimulationWorkspace,
        CythonLOB,
        CyMicrostructureGenerator,
    )

    HAVE_LOB_STATE_CYTHON = True
except ImportError:
    HAVE_LOB_STATE_CYTHON = False
    pytest.skip("lob_state_cython (Cython module) not available", allow_module_level=True)


def test_risk_penalty_stable_with_negative_net_worth():
    """
    Tests that the risk_penalty calculation in _compute_reward_cython remains stable
    even when net_worth is negative. This validates the fix that changed the
    denominator from abs(net_worth) to the more stable reward_scale (based on prev_net_worth).
    """
    # 1. Setup
    workspace = SimulationWorkspace(10)
    lob = CythonLOB()
    generator = CyMicrostructureGenerator()
    state = EnvState()

    # --- State that could cause issues with the old logic ---
    # `net_worth` is negative
    # `prev_net_worth` is a stable positive number
    state.net_worth = -1000.0
    state.prev_net_worth = 10000.0
    state.peak_value = 11000.0

    # Parameters to trigger risk penalty calculation
    state.units = 5.0
    state.use_potential_shaping = True
    state.risk_aversion_variance = 0.1
    state.potential_shaping_coef = 1.0
    state.gamma = 0.99
    state.last_potential = 0.0

    # Market-related data
    bar_price = 2000.0
    bar_atr = 50.0

    # Dummy action
    action = np.array([0.0, 0.0], dtype=np.float32)

    # 2. Execute the function
    # The core logic being tested is inside _compute_reward_cython, which is called by this function
    reward, done, info = run_full_step_logic_cython(
        workspace,
        lob,
        generator,
        bar_price,
        bar_price,  # open
        100000.0,  # volume_usd
        50000.0,  # taker_buy_volume
        bar_atr,
        bar_atr,  # long_term_atr
        100,  # bar_trade_count
        50.0,  # fear_greed
        action,
        state,
    )

    # 3. Assertions
    # The main goal is to ensure the reward is a stable, finite number.
    # The old logic `... / (abs(net_worth) + 1e-9)` was okay for negative net_worth,
    # but would explode if net_worth was close to zero.
    # The new logic `... / reward_scale` where reward_scale is based on prev_net_worth is always stable.
    assert np.isfinite(reward), "Reward should be a finite number"
    assert reward > -10.0 and reward < 10.0, "Reward should be within a reasonable range"

    # Test with near-zero net_worth, which would have caused explosions before
    state.net_worth = 1e-7
    state.prev_net_worth = 10000.0
    state.peak_value = 11000.0

    reward_near_zero, _, _ = run_full_step_logic_cython(
        workspace,
        lob,
        generator,
        bar_price,
        bar_price,
        100000.0,
        50000.0,
        bar_atr,
        bar_atr,
        100,
        50.0,
        action,
        state,
    )

    assert np.isfinite(reward_near_zero), "Reward should be stable even with near-zero net worth"
    assert (
        reward_near_zero > -10.0 and reward_near_zero < 10.0
    ), "Reward should not explode with near-zero net worth"

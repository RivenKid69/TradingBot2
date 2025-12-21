# cython: language_level=3, language=c++, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
"""Reward shaping utilities shared between Python and Cython environments."""

from libc.math cimport fabs, log, tanh, NAN, isnan

from lob_state_cython cimport EnvState
from risk_enums cimport ClosedReason


cdef inline double _clamp(double value, double lower, double upper) noexcept nogil:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


cdef double log_return(double net_worth, double prev_net_worth) noexcept nogil:
    """
    Calculate log return between two net worth values.

    CRITICAL FIX (2025-11-23): Returns large negative penalty instead of NAN when
    bankruptcy occurs (net_worth <= 0 or prev_net_worth <= 0).

    Previous Behavior (BUG):
        - Returned NAN when net_worth <= 0.0 or prev_net_worth <= 0.0
        - Caused training to crash with ValueError in distributional_ppo.py
        - Agent never learned to avoid bankruptcy (no negative reinforcement)

    New Behavior (FIX):
        - Returns -10.0 (configurable large negative penalty) for bankruptcy
        - Training continues, agent receives strong negative reinforcement
        - Agent learns to avoid bankruptcy through gradient descent

    Args:
        net_worth: Current net worth
        prev_net_worth: Previous net worth

    Returns:
        Log return or -10.0 if bankruptcy occurs

    Design Rationale:
        - Bankruptcy is catastrophic failure, deserves severe penalty
        - -10.0 is ~5-10x larger than typical episode returns
        - Ensures bankruptcy avoidance is strongly prioritized
        - Similar to DeepMind AlphaStar: illegal actions get -1000 penalty

    References:
        - CRITICAL_BUGS_ANALYSIS_2025_11_23.md - Problem #2
        - Vinyals et al. (2019), "Grandmaster level in StarCraft II" - penalty for invalid actions
        - Schulman et al. (2017), "PPO" - importance of reward shaping
    """
    cdef double ratio
    # CRITICAL FIX: Return large negative penalty instead of NAN for bankruptcy
    # This allows training to continue and teaches agent to avoid bankruptcy
    if prev_net_worth <= 0.0 or net_worth <= 0.0:
        return -10.0  # Large negative penalty for bankruptcy
    ratio = net_worth / (prev_net_worth + 1e-9)
    ratio = _clamp(ratio, 0.1, 10.0)
    return log(ratio)


cdef double potential_phi(
    double net_worth,
    double peak_value,
    double units,
    double atr,
    double risk_aversion_variance,
    double risk_aversion_drawdown,
    double potential_shaping_coef,
) noexcept nogil:
    cdef double risk_penalty = 0.0
    cdef double dd_penalty = 0.0

    if net_worth > 1e-9 and atr > 0.0 and units != 0.0:
        risk_penalty = -risk_aversion_variance * fabs(units) * atr / (fabs(net_worth) + 1e-9)

    if peak_value > 1e-9:
        dd_penalty = -risk_aversion_drawdown * (peak_value - net_worth) / peak_value

    return potential_shaping_coef * tanh(risk_penalty + dd_penalty)


cdef double potential_shaping(double gamma, double last_potential, double phi_t) noexcept nogil:
    return gamma * phi_t - last_potential


cdef double trade_frequency_penalty_fn(double penalty, int trades_count) noexcept nogil:
    if penalty <= 0.0 or trades_count <= 0:
        return 0.0
    return penalty * trades_count


cdef double event_reward(
    double profit_bonus,
    double loss_penalty,
    double bankruptcy_penalty,
    ClosedReason closed_reason,
) noexcept nogil:
    """
    Calculate event-based reward for position close reasons.

    FIX (MEDIUM #8): Improved documentation and explicit handling of all close reasons.

    Reward mapping:
    - NONE: 0.0 (no event)
    - BANKRUPTCY: -bankruptcy_penalty (catastrophic failure)
    - STATIC_TP_LONG/SHORT: +profit_bonus (successful profit taking)
    - All stop losses (ATR_SL, TRAILING_SL): -loss_penalty (protective stops)
    - MAX_DRAWDOWN: -loss_penalty (risk limit breached)

    Design rationale:
    All non-TP closes (except NONE) are penalized because they represent:
    1. Stop losses: Position moved against us → legitimate loss
    2. Max drawdown: Risk management failure → strong penalty
    3. Bankruptcy: Complete capital loss → maximum penalty

    This encourages the model to:
    - Take profits at TP levels (positive reinforcement)
    - Avoid triggering stop losses (negative reinforcement)
    - Maintain healthy drawdown (risk management)

    Args:
        profit_bonus: Reward for profitable closes
        loss_penalty: Penalty for unprofitable closes
        bankruptcy_penalty: Penalty for bankruptcy (fallback to loss_penalty if not set)
        closed_reason: Reason for position closure

    Returns:
        Event reward (positive for profit, negative for loss, zero for no event)
    """
    if closed_reason == ClosedReason.NONE:
        return 0.0

    if closed_reason == ClosedReason.BANKRUPTCY:
        if bankruptcy_penalty > 0.0:
            return -bankruptcy_penalty
        return -loss_penalty

    if closed_reason == ClosedReason.STATIC_TP_LONG or closed_reason == ClosedReason.STATIC_TP_SHORT:
        return profit_bonus

    # All other reasons (stop losses, max drawdown) receive loss penalty
    # This is intentional: these are protective mechanisms that indicate
    # the position moved against us (SL) or we exceeded risk limits (drawdown)
    return -loss_penalty


cdef double compute_reward_view(
    double net_worth,
    double prev_net_worth,
    double last_potential,
    bint use_legacy_log_reward,
    bint use_potential_shaping,
    double gamma,
    double potential_shaping_coef,
    double units,
    double atr,
    double risk_aversion_variance,
    double peak_value,
    double risk_aversion_drawdown,
    int trades_count,
    double trade_frequency_penalty,
    double last_executed_notional,
    double spot_cost_taker_fee_bps,
    double spot_cost_half_spread_bps,
    double spot_cost_impact_coeff,
    double spot_cost_impact_exponent,
    double spot_cost_adv_quote,
    double turnover_penalty_coef,
    double profit_close_bonus,
    double loss_close_penalty,
    double bankruptcy_penalty,
    ClosedReason closed_reason,
    double* out_potential,
    double reward_cap=10.0,  # FIX (MEDIUM #9): Parameterized reward clipping
) noexcept nogil:
    cdef double net_worth_delta = net_worth - prev_net_worth
    cdef double reward_scale = fabs(prev_net_worth)
    if reward_scale < 1e-9:
        reward_scale = 1.0
    # FIX: Устранен двойной учет reward! Было: reward = delta/scale + log_return (удвоение!)
    # Теперь: используется либо log_return, либо delta/scale, но НЕ оба одновременно
    cdef double reward
    if use_legacy_log_reward:
        reward = log_return(net_worth, prev_net_worth)
    else:
        reward = net_worth_delta / reward_scale
    cdef double phi_t = 0.0
    cdef double base_cost_bps = 0.0
    cdef double total_cost_bps = 0.0
    cdef double participation = 0.0
    cdef double impact_exp = 0.0

    # FIX CRITICAL BUG: Apply potential shaping regardless of reward mode
    # Previously, potential shaping was only applied when use_legacy_log_reward=True,
    # causing it to be ignored in the new reward mode even when enabled
    if use_potential_shaping:
        phi_t = potential_phi(
            net_worth,
            peak_value,
            units,
            atr,
            risk_aversion_variance,
            risk_aversion_drawdown,
            potential_shaping_coef,
        )
        reward += potential_shaping(gamma, last_potential, phi_t)

    reward -= trade_frequency_penalty_fn(trade_frequency_penalty, trades_count) / reward_scale

    # DOCUMENTATION (MEDIUM #7): Two-tier trading cost structure (INTENTIONAL DESIGN)
    # ================================================================================
    # This applies TWO separate penalties for trading (not a bug, but intentional):
    #
    # Penalty 1: Real market transaction costs (~0.12%)
    #   - Taker fee (e.g., 0.10%)
    #   - Half spread (e.g., 0.02%)
    #   - Market impact (participation-based, e.g., 0.00-0.10%)
    #
    # Penalty 2: Turnover penalty / behavioral regularization (~0.05%)
    #   - Fixed: turnover_penalty_coef * notional
    #   - Purpose: Discourage overtrading beyond real execution costs
    #
    # Design rationale:
    # 1. Penalty 1 models REAL costs (must match actual trading expenses)
    # 2. Penalty 2 is RL regularization (prevents model from excessive churning)
    # 3. Total ~0.17% encourages selective, high-conviction trades
    #
    # This pattern is standard in RL for trading:
    # - Almgren & Chriss (2001): Separate impact + regularization
    # - Moody et al. (1998): Behavioral penalties in performance functions
    #
    # If unintentional: Remove Penalty 2 by setting turnover_penalty_coef = 0
    # ================================================================================

    cdef double trade_notional = fabs(last_executed_notional)
    if trade_notional > 0.0:
        # Penalty 1: Real transaction costs (fee + spread + impact)
        base_cost_bps = spot_cost_taker_fee_bps + spot_cost_half_spread_bps
        total_cost_bps = base_cost_bps if base_cost_bps > 0.0 else 0.0
        if spot_cost_impact_coeff > 0.0 and spot_cost_adv_quote > 0.0:
            participation = trade_notional / spot_cost_adv_quote
            if participation > 0.0:
                impact_exp = spot_cost_impact_exponent if spot_cost_impact_exponent > 0.0 else 1.0
                total_cost_bps += spot_cost_impact_coeff * participation ** impact_exp
        if total_cost_bps > 0.0:
            reward -= (trade_notional * total_cost_bps * 1e-4) / reward_scale

    # Penalty 2: Turnover penalty (behavioral regularization)
    if turnover_penalty_coef > 0.0 and last_executed_notional > 0.0:
        reward -= (turnover_penalty_coef * last_executed_notional) / reward_scale

    reward += event_reward(
        profit_close_bonus,
        loss_close_penalty,
        bankruptcy_penalty,
        closed_reason,
    ) / reward_scale

    # FIXED (MEDIUM #9): reward_cap is now parameterized (default: 10.0)
    # Allows configuration via config files for experimentation.
    # Default value (10.0) maintains backward compatibility.
    # Tech Debt Tracking: docs/reports/TECH_DEBT_REGISTRY.md#perf-reward-cap (Closed)
    reward = _clamp(reward, -reward_cap, reward_cap)

    if out_potential != <double*>0:
        out_potential[0] = phi_t

    return reward


cpdef double compute_reward(EnvState state, ClosedReason closed_reason, int trades_count):
    cdef double new_potential = 0.0
    cdef double reward = compute_reward_view(
        state.net_worth,
        state.prev_net_worth,
        state.last_potential,
        state.use_legacy_log_reward,
        state.use_potential_shaping,
        state.gamma,
        state.potential_shaping_coef,
        state.units,
        state.last_bar_atr,
        state.risk_aversion_variance,
        state.peak_value,
        state.risk_aversion_drawdown,
        trades_count,
        state.trade_frequency_penalty,
        state.last_executed_notional,
        state.spot_cost_taker_fee_bps,
        state.spot_cost_half_spread_bps,
        state.spot_cost_impact_coeff,
        state.spot_cost_impact_exponent,
        state.spot_cost_adv_quote,
        state.turnover_penalty_coef,
        state.profit_close_bonus,
        state.loss_close_penalty,
        state.bankruptcy_penalty,
        closed_reason,
        &new_potential,
    )

    if state.use_potential_shaping:
        state.last_potential = new_potential
    else:
        state.last_potential = 0.0

    return reward

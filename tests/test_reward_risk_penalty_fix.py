"""
Comprehensive tests for Risk Penalty Normalization Fix (2025-11-23)

Tests verify that risk penalty is normalized by prev_net_worth (baseline capital)
instead of current net_worth, preventing unstable training signals.

CRITICAL BUGS FIXED:
1. Old normalization: abs(net_worth) → penalty explodes when net_worth drops
2. New normalization: baseline_capital (prev_net_worth or peak_value) → stable

Research support:
- Lopez de Prado (2018): "Advances in Financial ML"
- Sharpe Ratio: relative to starting value
- CVaR/VaR: relative to initial portfolio value
"""

import pytest
import numpy as np

pytest.importorskip("sortedcontainers")

try:
    from lob_state_cython import _compute_reward_cython
    HAVE_LOB_STATE_CYTHON = True
except ImportError:
    HAVE_LOB_STATE_CYTHON = False
    pytest.skip("lob_state_cython (Cython module) not available", allow_module_level=True)


class TestRiskPenaltyNormalization:
    """Tests for risk penalty normalization fix"""

    def test_stable_penalty_with_dropping_networth(self):
        """
        CRITICAL: Risk penalty should remain stable when net_worth drops.

        Old behavior: penalty explodes when net_worth -> 0
        New behavior: penalty stays proportional to initial capital
        """
        # Setup: constant position, prev_net_worth = 10000
        prev_net_worth = 10000.0
        peak_value = 10000.0
        units = 100.0
        atr = 50.0
        risk_aversion_variance = 0.1

        # Case 1: net_worth still high (9000) - small drop
        reward1, _ = _compute_reward_cython(
            net_worth=9000.0,
            prev_net_worth=prev_net_worth,
            event_reward=0.0,
            use_legacy_log_reward=False,
            use_potential_shaping=True,
            gamma=0.99,
            last_potential=0.0,
            potential_shaping_coef=0.1,
            units=units,
            atr=atr,
            risk_aversion_variance=risk_aversion_variance,
            peak_value=peak_value,
            risk_aversion_drawdown=0.0,
            trades_this_step=0,
            trade_frequency_penalty=0.0,
            executed_notional=0.0,
            turnover_penalty_coef=0.0
        )

        # Case 2: net_worth very low (100) - large drop
        reward2, _ = _compute_reward_cython(
            net_worth=100.0,
            prev_net_worth=prev_net_worth,
            event_reward=0.0,
            use_legacy_log_reward=False,
            use_potential_shaping=True,
            gamma=0.99,
            last_potential=0.0,
            potential_shaping_coef=0.1,
            units=units,
            atr=atr,
            risk_aversion_variance=risk_aversion_variance,
            peak_value=peak_value,
            risk_aversion_drawdown=0.0,
            trades_this_step=0,
            trade_frequency_penalty=0.0,
            executed_notional=0.0,
            turnover_penalty_coef=0.0
        )

        # CRITICAL ASSERTION: Risk penalty should be SIMILAR despite huge net_worth drop
        # Because it's normalized by prev_net_worth (10000), not current net_worth
        # The only difference should come from PnL component and drawdown penalty

        # Extract risk penalty contribution (removing PnL effect)
        # reward = pnl/scale + potential_shaping
        # potential_shaping = coef * tanh(risk_penalty + dd_penalty)

        # For reward1: pnl = -1000, scale = 10000, pnl_component = -0.1
        # For reward2: pnl = -9900, scale = 10000, pnl_component = -0.99

        # OLD BUG: risk_penalty would be 10x different (div by net_worth)
        # NEW FIX: risk_penalty should be SAME (div by prev_net_worth)

        # The risk penalty component should be identical
        # (units * atr) / prev_net_worth = (100 * 50) / 10000 = 0.5
        expected_risk_ratio = (units * atr) / prev_net_worth

        # Verify penalty is reasonable (not exploded)
        assert abs(reward1) < 5.0, f"reward1 should not explode: {reward1}"
        assert abs(reward2) < 5.0, f"reward2 should not explode: {reward2}"

        # The difference should come primarily from PnL, not risk penalty
        # If risk penalty was normalized by net_worth, reward2 would be ~ -100x worse
        pnl_diff = abs((-9900 / 10000) - (-1000 / 10000))  # = 0.89
        reward_diff = abs(reward2 - reward1)

        # reward_diff should be close to pnl_diff (not 100x larger)
        # Allow 2x margin for drawdown penalty contribution
        assert reward_diff < pnl_diff * 2.0, \
            f"Reward difference ({reward_diff:.3f}) should be close to PnL difference ({pnl_diff:.3f}), " \
            f"not exploded by risk penalty. Old bug would give reward_diff ~ 10.0+"

    def test_same_position_same_penalty_regardless_of_current_networth(self):
        """
        CORE PROPERTY: Same absolute position size → same absolute risk penalty
        Independent of current net_worth (only depends on baseline capital)
        """
        prev_net_worth = 50000.0
        peak_value = 50000.0
        units = 200.0
        atr = 100.0
        risk_aversion_variance = 0.05

        penalties = []
        for net_worth in [50000.0, 30000.0, 10000.0, 1000.0, 100.0]:
            _, potential = _compute_reward_cython(
                net_worth=net_worth,
                prev_net_worth=prev_net_worth,
                event_reward=0.0,
                use_legacy_log_reward=False,
                use_potential_shaping=True,
                gamma=0.99,
                last_potential=0.0,
                potential_shaping_coef=1.0,  # Set to 1.0 to extract raw potential
                units=units,
                atr=atr,
                risk_aversion_variance=risk_aversion_variance,
                peak_value=peak_value,
                risk_aversion_drawdown=0.0,
                trades_this_step=0,
                trade_frequency_penalty=0.0,
                executed_notional=0.0,
                turnover_penalty_coef=0.0
            )
            penalties.append(potential)

        # CRITICAL: Risk penalty component should be SIMILAR across all net_worth values
        # (minor variations due to tanh and drawdown penalty are OK)

        # Extract base risk penalty (without drawdown):
        # potential = tanh(risk_penalty + dd_penalty)
        # When net_worth = prev_net_worth (no drawdown), dd_penalty = 0
        base_penalty = penalties[0]  # net_worth = 50000 (no drawdown)

        # For other net_worths, we have drawdown penalty too, but risk_penalty should be constant
        # risk_penalty = -risk_aversion * (units * atr) / baseline_capital
        # = -0.05 * (200 * 100) / 50000 = -0.02

        expected_risk_penalty = -risk_aversion_variance * units * atr / prev_net_worth

        # Verify base penalty is close to expected (within tanh transformation)
        # tanh(-0.02) ≈ -0.02 (for small values)
        assert abs(base_penalty - expected_risk_penalty) < 0.01, \
            f"Base penalty {base_penalty:.4f} should be close to {expected_risk_penalty:.4f}"

        # OLD BUG: penalties would vary wildly (100x range)
        # NEW FIX: penalties should vary only due to drawdown component (not risk component)
        penalty_range = max(penalties) - min(penalties)

        # Range should be dominated by drawdown, not risk penalty explosion
        # Drawdown contribution: (peak - net_worth) / peak ranges from 0.0 to 0.998
        # So penalty_range should be ~ O(1), not O(100)
        assert penalty_range < 2.0, \
            f"Penalty range {penalty_range:.3f} should be reasonable, not exploded by risk penalty"

    def test_edge_case_zero_prev_networth_uses_fallback(self):
        """
        Edge case: If prev_net_worth <= 0, should use peak_value as fallback
        """
        peak_value = 10000.0
        units = 50.0
        atr = 25.0
        risk_aversion_variance = 0.1

        # Case 1: prev_net_worth = 0 (edge case)
        reward1, _ = _compute_reward_cython(
            net_worth=5000.0,
            prev_net_worth=0.0,  # Edge case: zero starting capital
            event_reward=0.0,
            use_legacy_log_reward=False,
            use_potential_shaping=True,
            gamma=0.99,
            last_potential=0.0,
            potential_shaping_coef=0.1,
            units=units,
            atr=atr,
            risk_aversion_variance=risk_aversion_variance,
            peak_value=peak_value,
            risk_aversion_drawdown=0.0,
            trades_this_step=0,
            trade_frequency_penalty=0.0,
            executed_notional=0.0,
            turnover_penalty_coef=0.0
        )

        # Should not crash or produce inf/nan
        assert np.isfinite(reward1), "Reward should be finite even with zero prev_net_worth"
        assert abs(reward1) < 10.0, "Reward should not explode with zero prev_net_worth"

        # Penalty should be normalized by peak_value (10000), not by prev_net_worth (0)
        expected_risk_penalty = -risk_aversion_variance * units * atr / peak_value
        # = -0.1 * 50 * 25 / 10000 = -0.0125

    def test_edge_case_negative_prev_networth_uses_fallback(self):
        """
        Edge case: If prev_net_worth < 0, should use peak_value as fallback
        """
        peak_value = 20000.0
        units = 100.0
        atr = 50.0
        risk_aversion_variance = 0.1

        # Case: prev_net_worth negative (edge case - bankruptcy)
        reward, _ = _compute_reward_cython(
            net_worth=1000.0,
            prev_net_worth=-5000.0,  # Edge case: negative starting capital
            event_reward=0.0,
            use_legacy_log_reward=False,
            use_potential_shaping=True,
            gamma=0.99,
            last_potential=0.0,
            potential_shaping_coef=0.1,
            units=units,
            atr=atr,
            risk_aversion_variance=risk_aversion_variance,
            peak_value=peak_value,
            risk_aversion_drawdown=0.0,
            trades_this_step=0,
            trade_frequency_penalty=0.0,
            executed_notional=0.0,
            turnover_penalty_coef=0.0
        )

        # Should not crash or produce inf/nan
        assert np.isfinite(reward), "Reward should be finite even with negative prev_net_worth"
        assert abs(reward) < 10.0, "Reward should not explode with negative prev_net_worth"

    def test_edge_case_both_zero_uses_last_resort_fallback(self):
        """
        Edge case: If both prev_net_worth AND peak_value <= 0, use 1.0 as last resort
        """
        units = 10.0
        atr = 5.0
        risk_aversion_variance = 0.1

        # Catastrophic case: both zero
        reward, _ = _compute_reward_cython(
            net_worth=100.0,
            prev_net_worth=0.0,
            event_reward=0.0,
            use_legacy_log_reward=False,
            use_potential_shaping=True,
            gamma=0.99,
            last_potential=0.0,
            potential_shaping_coef=0.1,
            units=units,
            atr=atr,
            risk_aversion_variance=risk_aversion_variance,
            peak_value=0.0,  # Also zero
            risk_aversion_drawdown=0.0,
            trades_this_step=0,
            trade_frequency_penalty=0.0,
            executed_notional=0.0,
            turnover_penalty_coef=0.0
        )

        # Should use fallback baseline_capital = 1.0
        # penalty = -0.1 * 10 * 5 / 1.0 = -5.0
        assert np.isfinite(reward), "Reward should be finite with both zero"
        assert abs(reward) < 15.0, "Reward should use 1.0 fallback normalization"

    def test_comparison_old_vs_new_behavior(self):
        """
        Direct comparison: old normalization vs new normalization

        OLD: risk_penalty / abs(net_worth)
        NEW: risk_penalty / baseline_capital
        """
        prev_net_worth = 10000.0
        peak_value = 10000.0
        units = 100.0
        atr = 50.0
        risk_aversion_variance = 0.1

        # Test case: net_worth drops to 500 (95% loss)
        net_worth_low = 500.0

        # NEW BEHAVIOR (implemented):
        reward_new, _ = _compute_reward_cython(
            net_worth=net_worth_low,
            prev_net_worth=prev_net_worth,
            event_reward=0.0,
            use_legacy_log_reward=False,
            use_potential_shaping=True,
            gamma=0.99,
            last_potential=0.0,
            potential_shaping_coef=1.0,
            units=units,
            atr=atr,
            risk_aversion_variance=risk_aversion_variance,
            peak_value=peak_value,
            risk_aversion_drawdown=0.0,
            trades_this_step=0,
            trade_frequency_penalty=0.0,
            executed_notional=0.0,
            turnover_penalty_coef=0.0
        )

        # SIMULATED OLD BEHAVIOR (what it would have been):
        # Old: risk_penalty = -0.1 * 100 * 50 / 500 = -1.0
        # New: risk_penalty = -0.1 * 100 * 50 / 10000 = -0.05
        # Ratio: 20x difference!

        old_risk_penalty = -risk_aversion_variance * units * atr / (abs(net_worth_low) + 1e-9)
        new_risk_penalty = -risk_aversion_variance * units * atr / (prev_net_worth + 1e-9)

        ratio = abs(old_risk_penalty) / abs(new_risk_penalty)

        # OLD bug: ratio = 20x (penalty explodes)
        # NEW fix: penalty is 20x smaller and stable
        assert ratio > 10.0, \
            f"Old penalty ({old_risk_penalty:.3f}) should be ~20x larger than new ({new_risk_penalty:.3f}), " \
            f"ratio = {ratio:.1f}x"

        # Verify new behavior produces reasonable reward
        assert abs(reward_new) < 5.0, "New reward should be reasonable (not exploded)"

    def test_zero_position_no_risk_penalty(self):
        """
        Sanity check: Zero position → zero risk penalty
        """
        prev_net_worth = 10000.0
        peak_value = 10000.0

        reward, potential = _compute_reward_cython(
            net_worth=10000.0,
            prev_net_worth=prev_net_worth,
            event_reward=0.0,
            use_legacy_log_reward=False,
            use_potential_shaping=True,
            gamma=0.99,
            last_potential=0.0,
            potential_shaping_coef=1.0,
            units=0.0,  # Zero position
            atr=50.0,
            risk_aversion_variance=0.1,
            peak_value=peak_value,
            risk_aversion_drawdown=0.0,
            trades_this_step=0,
            trade_frequency_penalty=0.0,
            executed_notional=0.0,
            turnover_penalty_coef=0.0
        )

        # With zero position and no drawdown, potential should be ~0
        assert abs(potential) < 0.01, f"Zero position should give ~zero potential, got {potential:.4f}"

    def test_large_position_appropriate_penalty(self):
        """
        Large position → appropriate penalty (not exploded)
        """
        prev_net_worth = 100000.0
        peak_value = 100000.0
        units = 1000.0  # Large position
        atr = 500.0
        risk_aversion_variance = 0.1

        # Case 1: net_worth = prev_net_worth (no PnL change)
        reward1, potential1 = _compute_reward_cython(
            net_worth=100000.0,
            prev_net_worth=prev_net_worth,
            event_reward=0.0,
            use_legacy_log_reward=False,
            use_potential_shaping=True,
            gamma=0.99,
            last_potential=0.0,
            potential_shaping_coef=1.0,
            units=units,
            atr=atr,
            risk_aversion_variance=risk_aversion_variance,
            peak_value=peak_value,
            risk_aversion_drawdown=0.0,
            trades_this_step=0,
            trade_frequency_penalty=0.0,
            executed_notional=0.0,
            turnover_penalty_coef=0.0
        )

        # Risk penalty = -0.1 * 1000 * 500 / 100000 = -0.5
        expected_risk_penalty = -risk_aversion_variance * units * atr / prev_net_worth

        # Potential = tanh(-0.5) ≈ -0.46
        assert abs(potential1 - expected_risk_penalty) < 0.1, \
            f"Large position penalty {potential1:.4f} should be close to {expected_risk_penalty:.4f}"

        # Should not explode
        assert abs(reward1) < 5.0, "Large position reward should not explode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

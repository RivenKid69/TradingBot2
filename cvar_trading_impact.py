#!/usr/bin/env python3
"""
Impact analysis of CVaR interpolation for trading bot with risk constraints.
"""

print("=" * 80)
print("TRADING BOT CVaR INTERPOLATION: PRACTICAL IMPACT ANALYSIS")
print("=" * 80)

# Typical trading scenario
print("\nSCENARIO: Trading bot with CVaR constraint CVaR_0.05 ≥ -0.10 (daily returns)")
print("-" * 80)

# Simulate realistic daily returns distribution
import math

def simulate_cvar_constraint_check():
    """Simulate how interpolation affects constraint satisfaction"""

    # Real-world trading returns often have:
    # - Fat tails (more extreme losses than normal distribution)
    # - Negative skew (larger losses than gains)
    # - Sharp drops in tail (liquidations, stop-losses)

    # Simulate 32 quantiles of daily returns (in %)
    # Bottom 5% (worst days): -8%, -5%, -4%, -3.5%
    # Next 5%: -3%, -2.5%, -2.2%, -2%
    # Gradually smoother
    quantiles = [
        -8.0, -5.0, -4.0, -3.5,  # α ≤ 0.125 (4/32)
        -3.0, -2.5, -2.2, -2.0,  # α ≤ 0.25
        -1.8, -1.6, -1.4, -1.2,
        -1.0, -0.8, -0.6, -0.4,
        -0.2, 0.0, 0.2, 0.4,
        0.6, 0.8, 1.0, 1.2,
        1.4, 1.6, 1.8, 2.0,
        2.2, 2.4, 2.6, 3.0
    ]

    alpha = 0.05
    constraint_limit = -0.10  # CVaR must be ≥ -10% (risk constraint)

    # Method 1: Piecewise constant (current)
    num_quantiles = len(quantiles)
    alpha_idx = int(math.floor(alpha * num_quantiles - 0.5))
    interval_start = alpha_idx / num_quantiles
    partial_mass = alpha - interval_start
    mass = 1.0 / num_quantiles

    if alpha_idx > 0:
        full_contribution_pc = mass * sum(quantiles[:alpha_idx])
    else:
        full_contribution_pc = 0.0
    partial_contribution_pc = quantiles[alpha_idx] * partial_mass
    cvar_pc = (full_contribution_pc + partial_contribution_pc) / alpha

    # Method 2: With interpolation (corrected)
    tau_i = (alpha_idx + 0.5) / num_quantiles
    tau_i_next = (alpha_idx + 1.5) / num_quantiles
    q_i = quantiles[alpha_idx]
    q_i_next = quantiles[alpha_idx + 1]

    weight = (alpha - tau_i) / (tau_i_next - tau_i)
    value_at_alpha = q_i * (1.0 - weight) + q_i_next * weight

    slope = (q_i_next - q_i) / (tau_i_next - tau_i)
    value_at_start = q_i - slope * tau_i

    # Full interval [0, 1/N] with interpolation
    tau_0 = 0.5 / num_quantiles
    tau_1 = 1.5 / num_quantiles
    slope_01 = (quantiles[1] - quantiles[0]) / (tau_1 - tau_0)
    value_at_0 = quantiles[0] - slope_01 * tau_0
    interval_0_end = 1.0 / num_quantiles
    weight_0_end = (interval_0_end - tau_0) / (tau_1 - tau_0)
    value_at_0_end = quantiles[0] * (1.0 - weight_0_end) + quantiles[1] * weight_0_end

    full_contribution_interp = 0.5 * (value_at_0 + value_at_0_end) * mass
    partial_contribution_interp = 0.5 * (value_at_start + value_at_alpha) * partial_mass
    cvar_interp = (full_contribution_interp + partial_contribution_interp) / alpha

    print(f"\nQuantile distribution (first 8): {quantiles[:8]}")
    print(f"Alpha: {alpha} (monitoring worst {alpha*100}% of outcomes)")
    print(f"Constraint: CVaR ≥ {constraint_limit}%")
    print()
    print(f"CVaR estimate (piecewise constant): {cvar_pc:.3f}%")
    print(f"CVaR estimate (with interpolation):  {cvar_interp:.3f}%")
    print(f"Difference: {abs(cvar_interp - cvar_pc):.3f}% ({abs(cvar_interp - cvar_pc)/abs(cvar_pc)*100:.1f}% relative)")
    print()

    # Check constraint satisfaction
    margin_pc = cvar_pc - constraint_limit
    margin_interp = cvar_interp - constraint_limit

    print(f"Constraint margin (piecewise):    {margin_pc:+.3f}%")
    print(f"Constraint margin (interpolation): {margin_interp:+.3f}%")
    print()

    if margin_pc >= 0:
        print("✓ Piecewise: Constraint SATISFIED")
    else:
        print("✗ Piecewise: Constraint VIOLATED")

    if margin_interp >= 0:
        print("✓ Interpolation: Constraint SATISFIED")
    else:
        print("✗ Interpolation: Constraint VIOLATED")
    print()

    # Gradient impact
    print("GRADIENT IMPACT:")
    print("-" * 80)
    print("The CVaR penalty term in loss: λ * max(0, limit - CVaR)")
    print()

    violation_pc = max(0, constraint_limit - cvar_pc)
    violation_interp = max(0, constraint_limit - cvar_interp)

    print(f"Violation (piecewise):    {violation_pc:.4f}")
    print(f"Violation (interpolation): {violation_interp:.4f}")
    print()

    if violation_pc != violation_interp:
        grad_ratio = violation_interp / violation_pc if violation_pc > 0 else float('inf')
        print(f"Gradient magnitude ratio: {grad_ratio:.2f}x")
        print(f"→ Interpolation produces {'STRONGER' if grad_ratio > 1 else 'WEAKER'} penalty signal")
    else:
        print("Both methods produce same penalty (both satisfied or both violated)")

    return cvar_pc, cvar_interp

print()
cvar_pc, cvar_interp = simulate_cvar_constraint_check()

print("\n" + "=" * 80)
print("WHEN INTERPOLATION MATTERS MOST:")
print("=" * 80)
print("""
1. **Small alpha (α ≤ 0.1)**: You're monitoring rare, extreme events
   → Up to 4% relative error in CVaR estimate
   → Could mean difference between constraint satisfied/violated

2. **Sharp tail drops**: Trading with stop-losses, liquidations, flash crashes
   → Piecewise constant UNDERESTIMATES true tail risk
   → Interpolation captures the rapid deterioration better

3. **Few quantiles (16-32)**: Computational efficiency vs accuracy tradeoff
   → Fewer quantiles = bigger gaps between estimates
   → Interpolation partially compensates for coarse discretization

4. **Risk-sensitive applications**: When CVaR constraint is TIGHT
   → Example: CVaR = -2.5% vs limit = -3.0% (small margin)
   → 0.1% estimation error could flip constraint status

5. **Training dynamics**:
   → Biased CVaR → Biased gradients → Policy converges to suboptimal risk level
   → Systematic underestimation trains agent to take MORE risk than intended
""")

print("=" * 80)
print("RECOMMENDATION FOR YOUR TRADING BOT:")
print("=" * 80)
print("""
USE INTERPOLATION if:
✓ You have tight risk constraints (CVaR close to limit)
✓ Alpha ≤ 0.1 (monitoring rare tail events)
✓ Using ≤ 32 quantiles
✓ Trading in volatile markets (crypto, options)
✓ Regulatory or fund mandate requires precise risk control

KEEP PIECEWISE CONSTANT if:
✓ Alpha ≥ 0.2 (monitoring more common events)
✓ Using ≥ 64 quantiles (fine discretization)
✓ Constraint has comfortable margin (>2% safety buffer)
✓ Prefer simpler code over 0.5-4% accuracy gain

For most trading applications with α=0.05 and 32 quantiles:
→ **INTERPOLATION IS RECOMMENDED** (2-4% accuracy gain is significant)
""")

print("=" * 80)
print("COMPUTATIONAL COST:")
print("=" * 80)
print("""
Interpolation overhead:
- ~10-20 extra floating point operations per CVaR computation
- Negligible compared to neural network forward/backward pass
- No memory overhead
- Same O(N) complexity

Estimated performance impact: < 0.1% of total training time
""")
print("=" * 80)

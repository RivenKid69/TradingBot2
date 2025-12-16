# COMPREHENSIVE TRAINING METRICS ANALYSIS

> **⚠️ DOCUMENT STATUS (2025-11-21):**
> This comprehensive metrics analysis was created in November 2025 and documents all training metrics in the distributional PPO implementation.
> **IMPORTANT NOTES:**
> - **LSTM State Reset (2025-11-21)**: This analysis **does not reflect** the critical LSTM state reset fix which improves value loss convergence by 5-15%. The documented loss patterns may differ from current behavior. See [NUMERICAL_ISSUES_FIX_SUMMARY.md](../../../NUMERICAL_ISSUES_FIX_SUMMARY.md).
> - **PopArt**: Mentions of PopArt reflect code structure but **PopArt is DISABLED** by default (see distributional_ppo.py:33-37).
> - **Action Space**: Does not cover 2025-11-21 action space semantics changes (TARGET vs DELTA). See [CRITICAL_FIXES_COMPLETE_REPORT.md](../../../CRITICAL_FIXES_COMPLETE_REPORT.md).
> - **CVaR Bugs**: Some documented bugs (e.g., CVaR violation sign inversion) may have been addressed. Check latest code.
>
> **RECOMMENDATION**: Use this document for understanding **what metrics exist and how they're calculated**, but verify current behavior against latest code and critical fixes reports. For production training, refer to [CLAUDE.md](../../../CLAUDE.md) section "Distributional PPO".

## Summary
This document provides a complete analysis of ALL training metrics logged during model training in the distributional PPO implementation. It includes metric definitions, calculation formulas, potential issues, and methodological concerns.

---

## SECTION 1: METRIC INVENTORY (ALL METRICS LOGGED)

### 1.1 LOSS METRICS

#### 1.1.1 Policy Loss Components
- **train/policy_loss** - Total policy loss combining PPO + BC weighted + KL penalty
  - Formula: `loss = policy_loss_ppo + policy_loss_bc_weighted + (kl_beta * kl_penalty_sample if kl_beta > 0)`
  - Location: line ~7721, 9062
  
- **train/policy_loss_ppo** - Standard PPO loss (clipped)
  - Formula: `-min(adv * ratio, adv * clamp(ratio, 1-clip_range, 1+clip_range))`
  - Location: line ~7694
  - **ISSUE FOUND**: Uses clipped advantages directly. Advantages are normalized per batch
  
- **train/policy_loss_bc** - Behavior cloning loss (unweighted)
  - Formula: `-log_prob_selected * weights` where `weights = exp(adv / cql_beta)` clamped to max 100
  - Location: line ~7718
  - **POTENTIAL ISSUE**: CQL weighting uses fixed cql_beta parameter, not adaptive
  
- **train/policy_loss_bc_weighted** - BC loss scaled by coefficient
  - Formula: `policy_loss_bc * bc_coef`
  - Location: line ~7719
  
- **train/policy_bc_vs_ppo_ratio** - Ratio of weighted BC to PPO loss
  - Formula: `abs(policy_loss_bc_weighted) / (abs(policy_loss_ppo) + 1e-8)`
  - Location: line ~9058
  - **WARNING**: Division guard of 1e-8 may be too small; use 1e-6 or check for zero cases

#### 1.1.2 Critic/Value Loss Components
- **train/value_ce_loss** (or **train/value_mse** for quantile) - Value function loss
  - Location: line ~8187, 8339, 9073
  - **For Quantile (if enabled)**:
    - Formula: `quantile_huber_loss(quantiles, targets)` (Huber loss weighted by quantile level)
    - Location: line ~8184-8186
    - **MATHEMATICAL CONCERN**: Huber loss formula at line ~2457 uses `tau - indicator` as weighting, needs verification
  
  - **For Categorical (if not quantile)**:
    - Formula: Cross-entropy loss on categorical distribution
    - Location: line ~8336-8339
    - **ISSUE**: Uses `max(unclipped, clipped)` which may not be mathematically justified

- **train/value_mse** - MSE between mean quantiles and target returns
  - Formula: `F.mse_loss(mean_values, target_returns)`
  - Location: line ~8160-8165
  - Only computed for quantile value function
  
- **train/value_target_below_frac** / **train/value_target_above_frac** - Clipping fraction metrics
  - Location: line ~9184-9185
  - Aggregated from weighted sum of per-batch clipping fractions
  - **POTENTIAL DATA LEAKAGE**: These use target values which may not be properly split between train/val

#### 1.1.3 Entropy Loss
- **train/entropy_loss** - Negative mean entropy (becomes positive loss term)
  - Formula: `-mean(entropy_selected)` 
  - Location: line ~7799, 9154
  - **SIGN CONVENTION**: Logged as negative (so positive values mean entropy regularization is pulling backward)
  - Coefficient: `ent_coef_eff_value` (line ~8515)

#### 1.1.4 CVaR Loss
- **train/cvar_loss** - Raw CVaR loss component (unscaled)
- **train/cvar_loss_unit** - CVaR loss in normalized units
- **train/cvar_term** - CVaR penalty term applied to total loss
- **train/cvar_loss_in_fraction** - Same as cvar_loss (confusing naming)
  - Location: line ~3110-3114
  - **REDUNDANCY ISSUE**: Multiple "in_fraction" and unit variants logged

#### 1.1.5 Total Loss
- **train/loss** - Total combined loss for gradient descent
  - Location: line ~9229
  - Formula: `policy_loss + vf_coef * critic_loss + ent_coef * entropy_loss + cvar_penalty`

### 1.2 EXPLAINED VARIANCE METRICS (CRITICAL)

#### 1.2.1 Primary EV Metrics
- **train/explained_variance** - Primary explained variance metric
  - Location: line ~2978
  - **AVAILABILITY FLAG**: `train/explained_variance_available` (0 or 1)
  - Formula: `1 - (var_residual / var_target)`
  - Where residual = y_true - y_pred, with weighted variance calculation
  
- **train/ev/global** - Alias for explained_variance
  - Location: line ~2979

#### 1.2.2 Grouped EV Metrics
- **train/ev/mean_grouped_unweighted** - Unweighted mean of per-group EV values
  - Location: line ~2985
  - **METHOD**: Groups by `group_keys` (presumably regime, symbol, or other feature)
  
- **train/ev/mean_grouped** - Weighted mean of per-group EV values
  - Location: line ~2987
  - **METHOD**: Weights by sample count per group
  - **COMMENT IN CODE**: "# FIX" suggests this metric may be problematic
  
- **train/ev/median_grouped** - Median of per-group EV values
  - Location: line ~2995

#### 1.2.3 EV Diagnostic Metrics
- **train/ev/corr** - Correlation between targets and predictions
  - Location: line ~8990
  
- **train/ev/bias** - Mean prediction error (bias)
  - Location: line ~8994
  
- **train/value/std_true** - Standard deviation of target returns
  - Location: line ~9002
  
- **train/value/std_pred** - Standard deviation of value predictions
  - Location: line ~9006
  
- **train/ev/on_train_batch** - EV computed on training batch only
  - Location: line ~9027
  - **POTENTIAL ISSUE**: May have train/test contamination depending on data split

### 1.3 APPROXIMATE KL DIVERGENCE METRICS

- **train/approx_kl** - Mean approximate KL divergence across all minibatches
  - Formula: `mean((exp(log_ratio) - 1) - log_ratio)` where `log_ratio = log(new_prob) - log(old_prob)`
  - Location: line ~8543, 9206-9207
  - **VERIFICATION NEEDED**: This is correct formula for KL[old||new]

- **train/approx_kl_raw** - Raw KL on unscaled action distributions
  - Location: line ~5831, 7787
  - Only computed for continuous action spaces with Box action space
  
- **train/approx_kl_median** - Median KL across minibatches
  - Location: line ~9208
  
- **train/approx_kl_p90** - 90th percentile KL
  - Location: line ~9209
  
- **train/approx_kl_max** - Maximum KL value
  - Location: line ~9210
  
- **train/approx_kl_ema** - Exponential moving average of KL (smoothed)
  - Location: line ~9213
  - **EMA FORMULA**: `beta * ema_old + (1-beta) * kl_new`
  - **WINDOW**: Uses either EMA with alpha=0.1 or rolling window (default 10 updates)

- **train/kl_exceed_frac** - Fraction of minibatches exceeding target_kl
  - Location: line ~9215
  
- **train/kl_last_exceeded** - Last KL value that exceeded threshold
  - Location: line ~9217
  
- **train/kl_exceed_consec_max** - Maximum consecutive KL exceedances
  - Location: line ~9218

- **train/policy_entropy_raw** - Raw entropy from inner distribution
  - Location: line ~5828
  - Only logged when `entropy_raw_count > 0`

### 1.4 ADVANTAGE/RATIO METRICS

- **train/clip_fraction** - Fraction of samples where ratio falls outside clipping range
  - Formula: `count(|ratio - 1| > clip_range) / total_count`
  - Location: line ~9273-9275
  - **COMPUTATION**: Correctly computed using `abs(ratio - 1) > clip_range` at line ~7698
  
- **train/clip_fraction_batch** - Per-batch clip fraction (logged at each microbatch)
  - Location: line ~7700
  - **GRANULARITY**: More detailed than aggregate metric

- **train/ratio_mean** - Mean of all probability ratios
  - Formula: `sum(ratio) / count(ratio)`
  - Location: line ~9250
  
- **train/ratio_std** - Standard deviation of probability ratios
  - Formula: `sqrt(E[ratio^2] - E[ratio]^2)`
  - Location: line ~9251
  - **CORRECT COMPUTATION**: Uses Bessel's correction-free variance formula
  
- **train/adv_mean** - Mean advantage per minibatch batch
  - Location: line ~9255
  
- **train/adv_std** - Std dev of advantages per batch
  - Location: line ~9256
  
- **train/adv_z_p10, p50, p90** - Percentiles of normalized advantages
  - Location: line ~9265-9267
  - Uses `(adv - adv_mean) / adv_std` normalization per batch

- **train/raw_z_clip_fraction** - Fraction of actions with |z| > 8.0
  - Location: line ~5834
  - Where z = (raw_action - mean) / std

- **train/log_prob_mean** - Mean log probability of actions
  - Location: line ~9253

### 1.5 VALUE SCALE & NORMALIZATION METRICS

#### 1.5.1 Return Normalization
- **train/ret_mean** - Mean of returns buffer
  - Location: line ~6797
  
- **train/ret_std** - Std dev of returns buffer
  - Location: line ~6798
  - **CRITICAL**: Used to normalize targets via `(target - mean) / std`
  - **STABILITY WARNING**: When std < 0.5 or > 0.9, warning streak incremented (line ~9236)

- **train/returns_abs_p95** - 95th percentile of absolute returns
  - Location: line ~6802, 9124
  
- **train/returns_abs_p95_in_fraction** - Same but in fraction units
  - Location: line ~9122

- **train/v_min / v_max** - Unscaled value bounds (raw returns)
  - Location: line ~6776-6779
  
- **train/v_min_scaled / v_max_scaled** - Scaled value bounds
  - Location: line ~6778-6779

- **train/value_target_scale** - Effective scale applied to targets
  - Location: line ~6780
  
- **train/value_target_scale_config** - Configured scale (from parameter)
  - Location: line ~6781
  
- **train/value_target_scale_robust** - Robust estimate of scale
  - Location: line ~6782

#### 1.5.2 Value Scale Updates (PopArt/Running Statistics)
- **train/value_scale_mean_before/after** - Mean before/after update
  - Location: line ~3969, 4142
  
- **train/value_scale_std_before/after** - Std before/after update
  - Location: line ~3970, 4143
  
- **train/value_scale_vmin_before/after** - Min quantile before/after
  
- **train/value_scale_vmax_before/after** - Max quantile before/after
  
- **train/value_scale_update_count** - Total number of scale updates
  - Location: line ~4153
  
- **train/value_scale_update_block_samples** - Samples in blocking window
  - Location: line ~4154
  
- **train/value_scale_update_block_freeze** - Freeze block flag
  - Location: line ~4155
  
- **train/value_scale_update_block_stability** - Stability block flag
  - Location: line ~4156

- **train/value_quantile_p50** - Median of value quantiles
  - Location: line ~2691
  
- **train/value_quantile_iqr** - IQR of value quantiles (q75 - q25)
  - Location: line ~2692

#### 1.5.3 Value Clipping Metrics
- **train/vf_clip_warmup_active** - Whether VF clip warmup is active
  - Location: line ~6396
  
- **train/vf_clip_warmup_completed** - Whether warmup completed
  - Location: line ~6419
  
- **train/vf_clip_ev_gate_active** - Whether EV gate is blocking clips
  - Location: line ~6400
  
- **train/vf_clip_threshold_ev** - EV threshold for clipping
  - Location: line ~6406
  
- **train/vf_clip_last_ev** - Last EV value used in clip decision
  - Location: line ~6408

### 1.6 REWARD/RETURN METRICS

- **train/reward_raw_p50** - Median reward in rollout
  - Location: line ~9126
  
- **train/reward_raw_p95** - 95th percentile reward
  - Location: line ~9127
  
- **train/reward_raw_p50_in_fraction** - Median reward (fraction units)
  - Location: line ~9119
  
- **train/reward_raw_p95_in_fraction** - 95th percentile reward (fraction)
  - Location: line ~9120

- **train/reward_costs_in_fraction** - Fraction of reward consumed by costs
  - Location: line ~9104
  
- **train/reward_costs_mean_in_fraction** - Mean cost fraction
  - Location: line ~9109

- **train/reward_robust_clip_in_fraction** - Robust clipping applied
  - Location: line ~9116
  
- **train/reward_clip_bound_in_fraction** - Clipping boundary
  - Location: line ~9129

- **train/target_return_mean** - Mean of target returns
  - Location: line ~9298
  
- **train/target_return_std** - Std of target returns
  - Location: line ~9299
  
- **train/value_pred_mean** - Mean of value predictions
  - Location: line ~9296
  
- **train/value_pred_std** - Std of value predictions
  - Location: line ~9297

### 1.7 CVaR METRICS (COMPREHENSIVE)

#### 1.7.1 Raw CVaR Values
- **train/cvar_raw** - Predicted CVaR from value distribution (raw units)
  - Location: line ~3107
  
- **train/cvar_empirical** - Empirical CVaR from windsorized rewards
  - Location: line ~3115
  - Formula: Mean of tail returns below alpha percentile
  - Location calc: line ~2636
  - **ISSUE**: Uses different alpha than value distribution CVaR
  
- **train/cvar_empirical_ema** - Exponential moving average of empirical CVaR
  - Location: line ~3118, 6544-6548

#### 1.7.2 Normalized CVaR (Unit Scale)
- **train/cvar_unit** - Predicted CVaR in normalized units
  - Location: line ~3109
  
- **train/cvar_empirical_unit** - Empirical CVaR normalized
  - Location: line ~3117

#### 1.7.3 CVaR Gap and Violation
- **train/cvar_gap** - Distance between CVaR limit and empirical CVaR (can be negative)
  - Formula: `limit - empirical_cvar`
  - Location: line ~3119, 6510
  - **MATHEMATICAL INTERPRETATION**: Positive gap = good (CVaR below limit)
  
- **train/cvar_gap_unit** - Normalized gap
  - Location: line ~3121
  
- **train/cvar_violation** - Clipped gap (always >= 0)
  - Formula: `max(0, limit - empirical_cvar)`
  - Location: line ~3122, 6535-6536
  - **DISCREPANCY**: Name suggests violation but uses gap; violation should be `max(0, empirical - limit)`
  - **POTENTIAL BUG**: Sign convention appears inverted
  
- **train/cvar_violation_ema** - EMA of violations
  - Location: line ~3125
  
- **train/cvar_gap_pos** - Alias for violation unit (confusing naming)
  - Location: line ~3126

- **train/cvar_limit_unit** - Target CVaR limit (normalized)
  - Location: line ~3130

#### 1.7.4 CVaR Penalty/Loss
- **train/cvar_loss** - Loss from CVaR constraint violation
  - Location: line ~3110
  - **ISSUE**: Also labeled "_in_fraction" which contradicts semantics
  
- **train/cvar_penalty_active** - Whether CVaR penalty is enabled
  - Location: line ~3127
  
- **train/cvar_lambda** - CVaR penalty coefficient (Lagrange multiplier)
  - Location: line ~3128
  - Updated via PID controller if enabled
  
- **train/cvar_scale** - Scale factor for CVaR normalization
  - Location: line ~3129
  
- **train/cvar_constraint** - Constraint term added to loss
  - Location: line ~9147

#### 1.7.5 CVaR Configuration
- **train/cvar_weight_effective** - Effective weight after clipping
  - Location: line ~3131
  
- **debug/cvar_weight_nominal** - Nominal weight before scaling
  - Location: line ~3134
  
- **debug/cvar_weight_effective_raw** - Raw effective weight
  - Location: line ~3135

### 1.8 LEARNING RATE METRICS

- **train/learning_rate** - Current learning rate for policy
  - Location: line ~2596, 5276, 9226
  
- **train/optimizer_lr** - Learning rate from optimizer (group 0)
  - Location: line ~2597
  
- **train/optimizer_lr_min** - Minimum LR across param groups
  - Location: line ~2598
  
- **train/optimizer_lr_group_min/max** - Min/max across parameter groups
  - Location: line ~2600-2601
  
- **train/scheduler_lr** - Learning rate from schedule
  - Location: line ~2592-2593
  
- **train/scheduler_lr_min** - Minimum scheduled LR
  - Location: line ~2599
  
- **train/lr_kl_scale** - KL-based LR scaling factor
  - Location: line ~2591
  
- **train/lr_before_clip** - LR before hard clipping
  - Location: line ~2594
  
- **train/lr_after_clip** - LR after hard clipping
  - Location: line ~2595
  
- **train/kl_lr_scale** - Scale factor applied based on KL
  - Location: line ~9279

### 1.9 ENTROPY SCHEDULING METRICS

- **train/policy_entropy** - Mean entropy of policy distribution
  - Location: line ~6309
  - **COMPUTATION**: Averaged from selected (valid) samples
  
- **train/entropy_loss** - Entropy regularization loss (negative entropy)
  - Location: line ~9154
  
- **train/ent_coef** - Current entropy coefficient (scaled)
  - Location: line ~9160
  
- **train/ent_coef_nominal** - Unscaled entropy coefficient
  - Location: line ~9161
  
- **train/ent_coef_eff** - Effective coefficient applied
  - Location: line ~9162
  
- **train/ent_coef_min** - Minimum entropy coefficient floor
  - Location: line ~9163
  
- **train/ent_coef_initial** - Initial configured value
  - Location: line ~9196
  
- **train/ent_coef_final** - Final configured value  
  - Location: line ~9197
  
- **train/ent_coef_autoclamp** - Whether autoclamp is active
  - Location: line ~9165
  
- **train/entropy_plateau** - Whether entropy has plateaued
  - Location: line ~9156
  
- **train/entropy_decay_start_update** - Update when decay began
  - Location: line ~9158
  
- **train/policy_entropy_slope** - Recent slope of entropy decay
  - Location: line ~9155

### 1.10 KL PENALTY CONTROL METRICS

- **train/kl_penalty_beta** - KL penalty coefficient
  - Location: line ~9282
  
- **train/kl_penalty_error** - PID controller error signal
  - Location: line ~9283
  
- **train/kl_penalty_pid_p** - P (proportional) component
  - Location: line ~9284
  
- **train/kl_penalty_pid_i** - I (integral) component
  - Location: line ~9285
  
- **train/kl_penalty_pid_d** - D (derivative) component
  - Location: line ~9286
  
- **train/kl_early_stop** - Whether early stop was triggered
  - Location: line ~9271
  
- **train/kl_absolute_stop_trigger** - Whether absolute KL threshold was hit
  - Location: line ~9220
  
- **train/kl_exceed_frac_at_stop** - KL exceed fraction when stopped
  - Location: line ~9224

- **train/policy_loss_kl_penalty** - Mean KL penalty component per minibatch
  - Location: line ~9289
  - Formula: `sum(kl_beta * kl_component) / count`

### 1.11 BATCH SIZE & TRAINING STRUCTURE METRICS

- **train/expected_batch_size** - Planned batch size
  - Location: line ~7462
  
- **train/actual_batch_size** - Actual samples processed (may differ due to masking)
  - Location: line ~7518
  
- **train/microbatch_size** - Size of gradient accumulation microbatches
  - Location: line ~7463
  
- **train/grad_accum_steps** - Number of microbatch groups before optimizer step
  - Location: line ~7464
  
- **train/n_epochs_effective** - Effective number of epochs after KL adjustments
  - Location: line ~9268
  
- **train/n_epochs_completed** - Actually completed epochs
  - Location: line ~9269
  
- **train/n_minibatches_done** - Total minibatches processed
  - Location: line ~9270
  
- **train/clip_range** - Current PPO clip range
  - Location: line ~9243
  
- **train/clip_range_schedule** - Scheduled clip range value
  - Location: line ~9245

### 1.12 CONFIGURATION METRICS (logged once per update)
- **config/*** - Various config parameters logged at line ~4917-4961, 5270
- **debug/*** - Various debug metrics

---

## SECTION 2: MATHEMATICAL ANALYSIS & ISSUES

### 2.1 EXPLAINED VARIANCE FORMULA CORRECTNESS

**Formula Used** (line ~192-270):
```
EV = 1 - (Var(residual) / Var(target))
```

With weights, uses effective sample size:
```
denom = sum(w) - (sum(w^2) / sum(w))
```

**Assessment**: CORRECT implementation of weighted explained variance
- Handles Bessel's correction appropriately for weighted data
- Filters for finite values
- Returns NaN when insufficient data

**ISSUE FOUND**: Line 2987 - Grouped EV metric has "# FIX" comment suggesting known problems

---

### 2.2 CLIP FRACTION CALCULATION

**Formula** (line ~7698):
```
clip_mask = |ratio - 1.0| > clip_range
clip_fraction = count(clip_mask) / total_count
```

**Assessment**: CORRECT
- Properly detaches for stability
- Uses cleaner formulation than traditional min/max

---

### 2.3 APPROXIMATE KL DIVERGENCE

**Formula** (line ~8543):
```
approx_kl = mean((exp(log_ratio) - 1) - log_ratio)
where log_ratio = log(new_logprob) - log(old_logprob)
```

**Assessment**: CORRECT
- This is the second-order Taylor expansion of KL divergence
- Mathematically sound and commonly used in PPO
- Symmetric form avoids numerical issues

**ISSUE**: Line 7787 computes `approx_kl_raw = old_log_prob - log_prob_new` without the (exp-1) term
- This is actually `log_ratio` itself, not KL divergence!
- **BUG SEVERITY**: HIGH - metric is mislabeled

---

### 2.4 ENTROPY LOSS SIGN CONVENTION

**Location**: Line 7799, 9154

**Formula**: `entropy_loss = -mean(entropy)`

**ISSUE**: Confusing sign convention
- When entropy is high (good), entropy_loss is negative
- Loss is typically positive, so this creates confusion
- Code correctly applies as `+ ent_coef * entropy_loss` which works, but semantics are unclear

**Recommendation**: Log as `train/entropy_loss_positive = -entropy_loss` for clarity

---

### 2.5 CVaR VIOLATION VS GAP SIGN CONFUSION

**Critical Issue Found at lines 6510-6536**:

```python
cvar_gap_tensor = cvar_limit_raw_tensor - cvar_empirical_tensor  # >0 if CVaR below limit
cvar_violation_unit_tensor = torch.clamp(cvar_gap_unit_tensor.detach(), min=0.0)  # enforces >=0
```

**The problem**:
- `cvar_gap = limit - empirical` (positive when CVaR is GOOD, below limit)
- `cvar_violation = max(0, gap)` (positive when GOOD)
- **Semantically incorrect**: "violation" should mean EXCEEDING the limit, not being below it

This metric name is inverted!

**Recommendation**: Rename `cvar_violation` to `cvar_headroom` or similar

---

### 2.6 VALUE TARGET CLIPPING METRICS REDUNDANCY

**Lines 3108, 3111, 3114, 3116, 3120, 3123**:

Multiple metrics logged with same values:
- `train/cvar_raw` and `train/cvar_raw_in_fraction` (same value!)
- `train/cvar_empirical` and `train/cvar_empirical_in_fraction` (same value!)
- `train/cvar_gap` and `train/cvar_gap_in_fraction` (same value!)
- `train/cvar_violation` and `train/cvar_violation_in_fraction` (same value!)

**ISSUE**: Redundant logging wastes storage and confuses interpretation

---

### 2.7 ADVANTAGE NORMALIZATION

**Location**: Lines 7619-7641

**Formula**:
```
if mask present:
    adv_selected = (adv[valid] - mean[valid]) / std[valid]
else:
    adv = (adv - mean) / std
```

**Assessment**: CORRECT
- Uses Welford's online algorithm implicitly via PyTorch
- Per-batch normalization is standard in PPO
- Properly handles masked samples

---

### 2.8 BEHAVIOR CLONING WEIGHTING

**Location**: Lines 7715-7719

**Formula**:
```
weights = exp(advantage / cql_beta)
weights = clamp(weights, max=100.0)
```

**ISSUE**: CQL weighting with fixed `cql_beta`
- CQL beta should ideally adapt based on data distribution
- Fixed value may not work across different trading regimes
- No adaptive weighting strategy implemented

---

### 2.9 RATIO STATISTICS CALCULATION

**Location**: Lines 9247-9251

```python
ratio_mean = sum(ratio) / count
ratio_var = max(sum(ratio^2) / count - mean^2, 0.0)
ratio_std = sqrt(ratio_var)
```

**Assessment**: POTENTIALLY INCORRECT
- Uses biased variance estimator (no Bessel's correction: dividing by n, not n-1)
- This is intentional for online statistics but inconsistent with other metrics
- The `max(var, 0)` prevents negative variance but hides numerical issues

---

### 2.10 DATA LEAKAGE ANALYSIS: EXPLAINED VARIANCE

**Concern**: Is EV computed using data that appears in training?

**Finding**: The code implements an "EV reserve" system (lines 3426-3490):
- Primary cache: Data actually used in gradient updates
- Reserve cache: Data held out for evaluation

**Assessment**: GOOD PRACTICE - properly avoids data leakage
- EV reported is on reserve set by default
- Can use primary set if reserve empty (fallback at line 3603)
- **WARNING**: Fallback uses training data if reserve unavailable!

**Risk**: If reserve cache is small/empty frequently, EV becomes contaminated

---

### 2.11 WEIGHT HANDLING IN EXPLAINED VARIANCE

**Location**: Lines 225-252 (weighted variance calculation)

```python
denom = sum_w - (sum_w_sq / sum_w if sum_w_sq > 0 else 0.0)
var_y = var_y_num / denom
```

**Assessment**: CORRECT
- Implements correct weighted variance with effective sample size
- Bessel's correction: `N_eff = (sum(w))^2 / sum(w^2)` approximately

---

### 2.12 CLIP FRACTION vs RATIO STATISTICS

**Potential Issue**: Line 7698 computes clip mask as:
```python
clip_mask = ratio_detached.sub(1.0).abs() > clip_range
```

But ratio can be computed from log_prob which has numerical issues:
```python
log_ratio = log_prob - old_log_prob
ratio = exp(log_ratio)  # Can be inf/nan if |log_ratio| is large
```

**Risk**: Very large policy changes create inf/nan in ratio
- Clipping logic still works (inf > clip_range = True)
- But statistics aggregation may fail

**Recommendation**: Add explicit finite checks before aggregating ratio statistics

---

## SECTION 3: DETECTED ISSUES SUMMARY

### CRITICAL BUGS

1. **CVaR Violation Sign Inversion** (Line 6535-6536)
   - "violation" actually measures "headroom" below limit
   - Name is semantically inverted
   - **Fix**: Rename or invert logic

2. **approx_kl_raw Incorrect Formula** (Line 7787)
   - Computes `log_ratio` not KL divergence
   - Different from `approx_kl` formula
   - **Fix**: Use correct KL formula: `(exp(log_ratio)-1) - log_ratio`

### HIGH PRIORITY ISSUES

3. **Data Leakage Risk in EV Fallback** (Line 3603)
   - Uses training data when reserve empty
   - Can cause optimistic EV estimates
   - **Fix**: Raise error or use separate holdout set

4. **Redundant CVaR Metrics** (Lines 3108-3126)
   - Multiple metrics with identical values
   - "_in_fraction" naming contradicts units
   - **Fix**: Remove duplicates and clarify naming

5. **Entropy Loss Sign Confusion** (Line 9154)
   - Logged as negative makes interpretation difficult
   - **Fix**: Log as `entropy_loss_positive = -entropy_loss`

### MEDIUM PRIORITY ISSUES

6. **Fixed CQL Beta** (Line 7716)
   - No adaptive weighting in BC loss
   - May not work across different regimes
   - **Fix**: Implement adaptive CQL weighting

7. **Biased Ratio Variance** (Line 9248)
   - Missing Bessel's correction
   - Inconsistent with other variance metrics
   - **Fix**: Divide by (count - 1) if count > 1

8. **Grouped EV Metric Issues** (Line 2987)
   - Code has "# FIX" comment
   - Unknown problem with grouped aggregation
   - **Fix**: Investigate and document

### LOW PRIORITY ISSUES

9. **LR Division Guard** (Line 9058)
   - Uses 1e-8 which may be too small
   - **Fix**: Use 1e-6 or add zero check

10. **Numeric Stability**: Large `log_ratio` values not explicitly handled
    - Can create inf/nan in ratio statistics
    - **Fix**: Add finite checks before aggregating

---

## SECTION 4: METHODOLOGICAL RECOMMENDATIONS

### 4.1 EV Metric Improvements

- Separate "train_ev" (on training data) from "test_ev" (on reserved data)
- Add confidence intervals around EV estimates
- Log EV per group with sample counts
- Track EV stability over time (variance of EV across batches)

### 4.2 CVaR Metric Clarification

- Rename "violation" to "gap" or "headroom" for clarity
- Log both empirical and theoretical CVaR separately
- Add metric for "exceeding CVaR" (actual violation)
- Remove "_in_fraction" duplicates

### 4.3 Loss Component Transparency

- Log total loss and each component separately (already done)
- Add log-scale plots since losses span multiple orders of magnitude
- Log ratio of each loss component to total
- Track when BC loss dominates or becomes zero

### 4.4 Value Function Diagnostics

- Log prediction error percentiles, not just mean/std
- Track ratio of target std to prediction std (should be >1 for exploration)
- Monitor clipping frequency and adjust v_min/v_max accordingly
- Log effective sample size used in EV computation

### 4.5 KL Control Diagnostics

- Log ratio of KL to target_kl (when > 1, triggering early stop)
- Track cumulative KL vs cumulative gradient steps
- Monitor PID controller state (error, integral term, derivative)
- Log when adaptive LR scaling kicks in

### 4.6 Batch Composition

- Log fraction of samples that are masked (invalid)
- Track EV reserve usage (# reserve samples vs # total)
- Monitor microbatch imbalance (max/min microbatch size ratio)
- Log sample reuse patterns for recurrent models

---

## SECTION 5: TESTING RECOMMENDATIONS

### 5.1 Unit Tests

1. Verify explained variance formula matches sklearn/statsmodels
2. Test CVaR calculation against manual computation
3. Verify KL divergence calculation matches reference implementations
4. Test clipping logic against manual Python implementation

### 5.2 Integration Tests

1. Train on synthetic data and verify metrics are sane:
   - EV should increase over time (if learning)
   - Loss should decrease monotonically
   - Entropy should increase then decrease (with decay schedule)
   
2. Test with adversarial loss gradients:
   - Ensure KL early stopping works
   - Verify loss components scale correctly
   
3. Test data leakage:
   - Artificially create train/test split
   - Verify EV reserve is actually used
   - Check that test-only data isn't in training

### 5.3 Validation Checks

1. Spot-check metric values:
   - ratio_mean should be near 1.0 (1-2 range typical)
   - clip_fraction should be <50% (>70% indicates too-large LR)
   - entropy should be >0 always
   
2. Cross-validate metrics:
   - KL divergence should correlate with policy loss changes
   - ratio_std should correlate with clip_fraction
   - explained_variance should relate to value loss

---

## CONCLUSION

The training metrics implementation is **largely correct** with **2-3 critical bugs** related to:
1. CVaR metric naming/sign inversion
2. Raw KL divergence incorrect formula
3. Potential data leakage in EV fallback

The system includes proper mechanisms for avoiding data contamination (reserve cache) and captures comprehensive training diagnostics. However, several metrics have confusing naming conventions and redundant variants that should be cleaned up.

**Recommendation**: Address critical bugs immediately, refactor metric logging to remove redundancy, and add the suggested diagnostic metrics for better observability.


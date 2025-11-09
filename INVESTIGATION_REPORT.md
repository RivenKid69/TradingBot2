# Investigation Report: Root Cause of Low EV and Training Issues

## Executive Summary

**Problem:** Training shows EV ≈ 0.0001, value predictions have extremely low variance (std=0.008 vs target std=1.07), and negative returns.

**Root Cause:** Synthetic training data is pure Geometric Brownian Motion with ZERO drift → no predictable patterns → optimal value function is a constant → value network correctly learns to predict constant → training collapses.

**Solution:** Use data with predictable patterns (momentum, drift, or real market data).

---

## Detailed Analysis

### 1. Symptoms Observed

```
EV (Explained Variance): 0.0001  ❌ (should be > 0.5)
value_pred_std: 0.008            ❌ (133x smaller than target)
target_return_std: 1.07          ✓ (normalized correctly)
value_mse: 1.03                  ❌ (predictions don't correlate)
ret_mean: -0.0007                ❌ (negative returns)
reward_p50: -0.000688            ❌ (median reward negative)
```

### 2. Investigation Path

#### Step 1: Checked Value Network Architecture
- ✓ QuantileValueHead with 32 quantiles: **CORRECT**
- ✓ CustomMlpExtractor with residual connections: **CORRECT**
- ✓ Quantile Huber loss function: **CORRECT**

#### Step 2: Checked Normalization & Scaling
- ✓ PopArt return normalization: **WORKING**
- ✓ Value scaling (ret_std=0.0322): **REASONABLE**
- ✓ Target normalization (target_std=1.07): **CORRECT**

#### Step 3: Found Smoking Gun
```
vf_clip/pred/norm_std_pre: 0.0    ← Predictions have ZERO variance!
vf_clip/pred/norm_std_post: 0.0   ← Even before clipping
value_pred_mean: 0.112            ← Always same value
value_pred_std: 0.008             ← Tiny noise from SGD
```

**Value network predicts almost constant value for ALL states!**

#### Step 4: Analyzed Training Data
```python
# prepare_demo_data.py:44-45
returns = np.random.normal(0, volatility, num_hours)  # ZERO drift!
prices = base_price * np.exp(np.cumsum(returns))
```

**Data autocorrelation:**
```
Returns Autocorrelation:
  Lag 1: 0.009685  ≈ 0  ← No momentum
  Lag 2: -0.009284 ≈ 0  ← No patterns
  Lag 5: -0.002849 ≈ 0  ← Pure noise
```

### 3. Root Cause Explanation

**Geometric Brownian Motion with zero drift:**
- Returns ~ IID Normal(μ=0, σ=0.02)
- No autocorrelation → no momentum
- No predictable patterns
- **E[Return | State] = E[Return] = 0 for ALL states**

**Why value network predicts constant:**
1. In RL, value function learns V(s) = E[Σ γ^t r_t | s_0 = s]
2. For random walk: E[r_t | s] = E[r_t] = constant
3. Therefore, optimal V(s) = constant for all s
4. Value network correctly learns this!

**Cascade of failures:**
```
Random walk data
  ↓
Value function = constant
  ↓
var(value_predictions) ≈ 0
  ↓
EV = 1 - var(residuals)/var(targets) ≈ 1 - var(targets)/var(targets) ≈ 0
  ↓
GAE advantages = noise (no useful baseline)
  ↓
Policy gradient = noise (no learning signal)
  ↓
Agent doesn't learn → negative returns (fees dominate)
```

### 4. Why This Is Actually Correct Behavior

The value network is **not broken** - it's correctly learning that:
- In a random walk, past states don't predict future returns
- The optimal prediction is the unconditional mean
- Predicting a constant minimizes MSE

This is mathematically correct but useless for training!

---

## Solutions

### Option 1: Generate Predictable Synthetic Data (RECOMMENDED FOR TESTING)

**Use provided script:**
```bash
python prepare_demo_data_with_drift.py --rows 20000 --drift 0.0005 --momentum 0.15
```

**What it does:**
- Adds **positive drift** (expected return > 0) → agent can profit
- Adds **momentum** (autocorrelation=0.15) → past returns predict future
- Creates exploitable patterns → value function can learn useful predictions

**Expected results:**
- EV should improve to > 0.3
- value_pred_std should match target_std
- Returns should become positive
- Agent should learn to follow momentum

### Option 2: Use Real Market Data

Download actual crypto OHLCV data with:
- Real trends and momentum
- Mean reversion patterns
- Regime switches
- News/sentiment effects

### Option 3: Add Technical Indicators to Random Walk

Even with random walk prices, you can add features that CREATE predictable patterns:
- Moving average crossovers
- RSI oversold/overbought signals
- Bollinger Band bounces

These create **exploitable** patterns even if underlying prices are random.

---

## Verification

After generating new data, verify it's learnable:

```python
import pandas as pd
df = pd.read_feather('data/processed/BTCUSDT.feather')
df['ret'] = df['close'].pct_change()

# Check autocorrelation
print(f"Lag 1 autocorr: {df['ret'].autocorr(lag=1):.4f}")
# Should be > 0.1 for momentum

# Check mean return
print(f"Mean return: {df['ret'].mean():.6f}")
# Should be > 0 for profit potential
```

After training with predictable data, you should see:
```
✓ EV > 0.5 (good value function)
✓ value_pred_std ≈ target_return_std (network learns variation)
✓ ret_mean > 0 (positive returns)
✓ policy converges to profitable strategy
```

---

## Key Insights

1. **Not a bug, it's a feature**: Value network correctly identifies that random walk has no state-dependent value

2. **Garbage in, garbage out**: RL needs data where actions matter and states predict outcomes

3. **Test with care**: Always validate that your training data has learnable patterns before debugging the model

4. **This is why real traders use:**
   - Technical analysis (creates patterns)
   - Fundamental analysis (predicts trends)
   - Sentiment analysis (forecasts momentum)
   - NOT pure random walk simulation!

---

## Files Changed/Created

- `prepare_demo_data_with_drift.py` - New data generator with predictable patterns
- `INVESTIGATION_REPORT.md` - This report
- `distributional_ppo.py` - Fixed IndexError in EV calculation (separate issue)

## Next Steps

1. Generate new predictable data:
   ```bash
   python prepare_demo_data_with_drift.py --rows 20000
   ```

2. Re-run training:
   ```bash
   python train_model_multi_patch.py \
     --config configs/config_train_spot_bar.yaml \
     --dataset-split none \
     --n-envs 4
   ```

3. Monitor metrics:
   - EV should increase from 0.0001 to > 0.5
   - value_pred_std should match target_std (~1.0)
   - Returns should become positive
   - Agent should learn profitable strategies

4. If EV is still low with predictable data, THEN investigate:
   - Learning rate tuning
   - Gradient clipping settings
   - Network architecture
   - Hyperparameter optimization

But with random walk data, **these won't help** because the problem is fundamental!

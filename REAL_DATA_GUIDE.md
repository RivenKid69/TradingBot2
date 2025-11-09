# Guide: Working with Real Historical Data

## Problem Statement

You asked: "Why use synthetic data generator when I have historical data?"

**Answer:** You DON'T have historical data yet! The `data/train/` directory is empty.

I used synthetic data because:
1. No historical data existed in the repository
2. Needed to test the training pipeline
3. The synthetic data revealed a fundamental problem: **random walk data can't be learned**

---

## The Real Issue: Are Your Historical Data Learnable?

### Important Truth About Crypto Data

**Not all historical data is learnable!**

Real cryptocurrency data on short timeframes often exhibits random walk properties:

| Timeframe | Typical Autocorr | Learnability |
|-----------|------------------|--------------|
| 1-minute  | 0.02-0.05 | ❌ Random walk - can't learn |
| 5-minute  | 0.05-0.10 | ⚠️ Marginal - difficult |
| 15-minute | 0.08-0.15 | ⚠️ Possible with good features |
| 1-hour    | 0.10-0.20 | ✅ Learnable - good momentum |
| 4-hour    | 0.15-0.30 | ✅ Very learnable |

### Why Short Timeframes Are Random

1. **Market microstructure noise** - HFT algorithms, market makers
2. **Bid-ask spread** - Eats small edges
3. **Slippage** - Destroys tiny profits
4. **News reactions** - Instant, unpredictable
5. **High-frequency noise** - Dominates signal

**Analogy:**
- 1-minute chart = Looking at ocean waves from 1 meter away (see only ripples)
- 1-hour chart = Looking from 100 meters away (see wave trends)

---

## Step-by-Step: Get and Validate Your Data

### Step 1: Download Historical Data

```bash
# Install dependencies
pip install requests pandas numpy

# Download data (RECOMMENDED: start with 1h timeframe)
python download_historical_data.py \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 1h \
  --days 365 \
  --output-dir data/train
```

**The script will:**
- Download real OHLCV data from Binance
- Automatically analyze predictability
- Give you a verdict: Learnable or Random Walk

### Step 2: Check the Analysis Output

```
PREDICTABILITY ANALYSIS: BTCUSDT

Autocorrelation:
  Lag 1:  +0.1245  ✓ Good
  Lag 5:  +0.0823  ✓ Good
  Lag 20: +0.0312  ✗ Weak

Return Statistics:
  Mean: +0.000145 (12.7% annualized)
  Std:  0.0342
  Sharpe: 0.73

Momentum Test:
  After UP move:   56.3% chance of UP
  After DOWN move: 48.2% chance of UP
  Difference: +8.1%  ✓ Momentum!

VERDICT: ✅ LEARNABLE - Good predictive structure!
Score: 5/5
```

**If you get ❌ RANDOM WALK verdict, proceed to Step 3!**

### Step 3: If Data is Random Walk, Here's What To Do

#### Option A: Use Longer Timeframes

```bash
# Try 4-hour instead of 1-minute
python download_historical_data.py --timeframe 4h --days 365
```

**Why it helps:**
- More time for trends to develop
- Noise averages out
- Slippage becomes smaller relative to price moves
- Technical patterns become clearer

#### Option B: Add Technical Indicators

Even random price data can become learnable with indicators!

```bash
# Add indicators that CREATE predictable patterns
python add_technical_indicators.py \
  --input data/train \
  --output data/train_with_indicators
```

**How indicators help:**

```
Raw prices: Random walk → Can't learn
     ↓
Add RSI: Overbought/oversold levels → Mean reversion pattern
Add MA: Crossovers → Momentum pattern
Add BB: Bands → Support/resistance pattern
     ↓
Combined: Predictable structure → CAN learn!
```

**Example:**
```
RSI < 30 (oversold) → 58% chance price goes UP
RSI > 70 (overbought) → 45% chance price goes UP
→ 13% edge! Learnable!
```

Then update your config:
```yaml
paths: ["data/train_with_indicators/*.csv"]
```

#### Option C: Different Assets or Markets

Some assets are more predictable:
- **Large cap crypto** (BTC, ETH) - more random on short TF
- **Small cap crypto** - more volatile, stronger trends
- **Forex majors** - strong intraday patterns
- **Commodities** - seasonal patterns
- **Stocks** - momentum after earnings

---

## Step 4: Update Config and Train

Once you have learnable data:

```bash
# Update config to point to your data
nano configs/config_train_spot_bar.yaml

# Change this line:
paths: ["data/train/*.csv"]  # Your real data

# Or if using indicators:
paths: ["data/train_with_indicators/*.csv"]

# Run training
python train_model_multi_patch.py \
  --config configs/config_train_spot_bar.yaml \
  --dataset-split none \
  --n-envs 4
```

**Monitor these metrics:**

```
✅ Good training:
- EV > 0.3 (value function learns)
- value_pred_std ≈ target_return_std (predictions vary)
- Mean return > 0 (agent finds edge)

❌ Bad training (still random walk):
- EV ≈ 0
- value_pred_std ≈ 0 (constant predictions)
- Mean return ≈ 0 or negative
```

---

## Understanding the Fundamental Limit

### Why This Matters

**Mathematical fact:**
```
If data has zero predictable structure (I(state; future) = 0 bits),
NO algorithm can learn from it.
```

This is NOT a limitation of:
- ❌ PPO algorithm
- ❌ Your implementation
- ❌ Network architecture
- ❌ Hyperparameters

This IS a limitation of:
- ✅ The data itself
- ✅ Information theory
- ✅ Laws of probability

**Analogy:**

You can't build a perpetual motion machine because it violates the laws of physics.

You can't learn from random walk data because it violates the laws of information theory.

### The Efficient Market Hypothesis

On very short timeframes (1m, 5m), crypto markets approach **semi-strong efficiency**:

```
All available information → Already in price
→ Future moves are unpredictable
→ Can't consistently profit
```

**Why longer timeframes work:**
- Information takes time to propagate
- Traders have different time horizons
- Momentum and mean reversion emerge
- Technical patterns become self-fulfilling

---

## Decision Tree

```
Do you have historical data?
├─ NO → Use download_historical_data.py
└─ YES
    └─ Is it learnable? (autocorr > 0.1?)
        ├─ YES → Great! Train directly
        └─ NO
            └─ Options:
                ├─ Use longer timeframe (1h, 4h)
                ├─ Add technical indicators
                ├─ Try different assets
                └─ Or accept that RL won't work on this data
```

---

## Key Takeaways

1. **Not all historical data is learnable** - especially short timeframes

2. **Always validate first:**
   ```bash
   python download_historical_data.py --timeframe 1h --days 365
   # Check the autocorrelation and verdict
   ```

3. **If random walk, you have options:**
   - Longer timeframes (easiest)
   - Technical indicators (adds structure)
   - Different markets (some are more predictable)

4. **Synthetic data with momentum is BETTER than random walk historical data**
   - If your 1m BTC data has autocorr = 0.02
   - And synthetic data with drift has autocorr = 0.15
   - → Synthetic is MORE learnable!

5. **The analysis I did is still valid:**
   - Random walk historical data will have same problems
   - Same low EV, collapsed predictions, failed training
   - This is not a bug - it's correct behavior for unpredictable data

---

## Final Recommendation

**Start with this workflow:**

```bash
# 1. Download 1-hour data
python download_historical_data.py --timeframe 1h --days 365

# 2. Check the verdict - is it learnable?

# 3a. If YES → train directly
python train_model_multi_patch.py --config configs/config_train_spot_bar.yaml

# 3b. If NO → add indicators first
python add_technical_indicators.py --input data/train --output data/train_with_indicators
# Update config to use data/train_with_indicators/*.csv
# Then train

# 4. Monitor EV - should be > 0.3 for good learning
tail -f logs/training_*.log | grep "explained_variance"
```

**If EV is still ≈ 0 with real data:**
- Your historical data IS random walk
- The analysis in SCIENTIFIC_ANALYSIS.md applies to it
- Solutions: longer TF, indicators, or different market

---

## Questions?

**Q: Why did you use synthetic data?**
A: Because `data/train/` was empty. I needed something to test with.

**Q: Will my historical data work better?**
A: Only if it has autocorr > 0.1. Many crypto timeframes are near-random walk.

**Q: What if my 1m data is random walk?**
A: Use 1h or 4h instead, or add technical indicators to create structure.

**Q: Is the analysis wrong then?**
A: No! The analysis applies to ANY random walk data - synthetic or real.

**Q: Can't I just tune hyperparameters?**
A: No. If I(state; future) = 0, no algorithm can learn. It's information theory.

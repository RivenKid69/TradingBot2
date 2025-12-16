# CRITICAL BUG REPORT: RSI Returns NaN When Loss = 0

## Problem
RSI feature returns **NaN** when there are no losses (all prices increasing).

## Root Cause
In `transformers.py` OnlineFeatureTransformer.update() lines 620-628:

```python
if (
    st["avg_gain"] is not None
    and st["avg_loss"] is not None
    and float(st["avg_loss"]) > 0.0  # ❌ PROBLEM: excludes zero loss
):
    rs = float(st["avg_gain"]) / float(st["avg_loss"])
    feats["rsi"] = float(100.0 - (100.0 / (1.0 + rs)))
else:
    feats["rsi"] = float("nan")
```

**When avg_loss = 0 (all gains, no losses):**
- Condition `avg_loss > 0.0` is False
- Returns NaN instead of 100

## Impact
- ❌ RSI = NaN during strong uptrends (no losses)
- ❌ RSI = NaN during strong downtrends (no gains -> avg_gain=0, same issue)
- ✅ RSI valid only when both gains and losses present

**Model trained on NaN values!**

## Evidence
```python
Bar 1: price=28950.0, rsi=NaN  # avg_gain=50.0, avg_loss=0.0
Bar 2: price=29000.0, rsi=NaN  # avg_gain=50.0, avg_loss=0.0
Bar 3: price=29050.0, rsi=NaN  # avg_gain=50.0, avg_loss=0.0
Bar 4: price=29100.0, rsi=NaN  # avg_gain=50.0, avg_loss=0.0
Bar 5: price=28900.0, rsi=76.47  # First loss appears -> RSI valid
```

## Correct Behavior (Wilder's RSI)
When avg_loss = 0:
- RS = avg_gain / 0 = infinity
- RSI = 100 - (100 / (1 + infinity)) = 100 - 0 = 100

When avg_gain = 0:
- RS = 0 / avg_loss = 0
- RSI = 100 - (100 / (1 + 0)) = 100 - 100 = 0

## Fix Required
```python
if st["avg_gain"] is not None and st["avg_loss"] is not None:
    if float(st["avg_loss"]) == 0.0:
        feats["rsi"] = 100.0  # All gains -> RSI = 100
    elif float(st["avg_gain"]) == 0.0:
        feats["rsi"] = 0.0    # All losses -> RSI = 0
    else:
        rs = float(st["avg_gain"]) / float(st["avg_loss"])
        feats["rsi"] = float(100.0 - (100.0 / (1.0 + rs)))
else:
    feats["rsi"] = float("nan")
```

## Status
🚨 **UNFIXED** - This bug exists in production code!

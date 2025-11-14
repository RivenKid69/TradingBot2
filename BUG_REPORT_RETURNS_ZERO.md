# CRITICAL BUG REPORT: Returns Feature Always Zero

## Problem
Returns features (ret_4h, ret_12h, etc.) are **ALWAYS 0.0** when window size = 1 bar.

## Root Cause
In `transformers.py` OnlineFeatureTransformer.update() lines 606-618:

```python
for i, lb in enumerate(self.spec.lookbacks_prices):
    if len(seq) >= lb:
        window = seq[-lb:]  # Get last lb bars
        first = float(window[0])  # First bar of window
        ret_name = f"ret_{_format_window_name(lb_minutes)}"
        feats[ret_name] = (
            float(math.log(price / first)) if first > 0 else 0.0
        )
```

**When lb=1 (1 bar window):**
1. Current price is appended to deque (line 569)
2. window = seq[-1:] gets the LAST bar (current)
3. first = window[0] = CURRENT price
4. ret = log(current_price / current_price) = log(1) = 0

## Impact
- ❌ ret_4h (240 min / 240 min = 1 bar) **ALWAYS 0**
- ❌ ret_12h (720 min / 240 min = 3 bars) - correct
- ❌ ret_24h (1440 min / 240 min = 6 bars) - correct

**ALL 4h returns are corrupted!**

## Evidence
```bash
$ python test_returns_bug.py
BAR 0: price=29000.0 -> ret_4h=0.0
BAR 1: price=29100.0 -> ret_4h=0.0  # Should be log(29100/29000) = 0.00344
BAR 2: price=29200.0 -> ret_4h=0.0  # Should be log(29200/29100) = 0.00343
```

## Correct Behavior
For 4h returns on 4h timeframe:
- Should calculate: log(current_price / price_1_bar_ago)
- Requires: lb >= 2 bars minimum
- Current: lb = 1 bar (incorrect!)

## Fix Required
```python
# Option 1: Increase minimum window to 2 bars for returns
window = seq[-max(lb, 2):]
first = window[0]
current = window[-1]
ret = log(current / first)

# Option 2: Use previous close explicitly
ret = log(price / st["last_close"]) if st["last_close"] else 0.0
```

## Status
🚨 **UNFIXED** - This bug exists in production code!

"""
Test suite for CRITICAL FIX: RSI should return 100 when avg_loss = 0 (pure uptrend).

Bug description:
- When prices rise consecutively (4+ bars), avg_loss = 0.0
- Original code: if avg_loss > 0.0: ... else: return NaN
- Problem: 0.0 > 0.0 = False → returns NaN instead of 100.0
- Fix: Handle edge cases according to Wilder's RSI formula
  - avg_loss = 0, avg_gain > 0 → RSI = 100 (pure uptrend)
  - avg_gain = 0, avg_loss > 0 → RSI = 0 (pure downtrend)
  - both = 0 → RSI = 50 (no movement)
  - both > 0 → RSI = 100 - (100 / (1 + RS))
"""

import math
from typing import List, Dict, Any
from transformers import FeatureTransformer, TransformerSpec


def create_bars(prices: List[float]) -> List[Dict[str, Any]]:
    """Create bar data from price list."""
    bars = []
    for i, price in enumerate(prices):
        bars.append({
            "symbol": "BTCUSDT",
            "time_key": f"2024-01-01T{i:02d}:00:00",
            "close": price,
            "open": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "volume": 1000.0,
            "quote_volume": 1000.0 * price,
            "trade_count": 100,
            "taker_buy_base_volume": 500.0,
            "taker_buy_quote_volume": 500.0 * price,
        })
    return bars


def create_transformer_with_rsi() -> FeatureTransformer:
    """Create transformer configured to calculate RSI."""
    spec = TransformerSpec(
        sma_windows=[],
        ret_windows=[],
        yang_zhang_windows=[],
        parkinson_windows=[],
        bid_ask_spread_windows=[],
        volume_windows=[],
        vwap_windows=[],
        intrabar_range_windows=[],
        taker_buy_ratio_windows=[],
        rsi_period=14,  # Enable RSI calculation
    )
    return FeatureTransformer(spec)


def test_rsi_pure_uptrend():
    """
    Test: RSI should return 100.0 when all prices rise (avg_loss = 0).

    Scenario: 20 bars with consecutive price increases
    Expected: RSI = 100.0 (not NaN)
    """
    # Create 20 bars with increasing prices (29000 → 30900 in steps of 100)
    prices = [29000 + i * 100 for i in range(20)]
    bars = create_bars(prices)

    transformer = create_transformer_with_rsi()

    # Process all bars
    for bar in bars:
        transformer.transform(bar)

    # Get final state
    state = transformer.get_state()

    # After 15+ bars, RSI should be calculated (period=14)
    # In pure uptrend: avg_loss = 0, avg_gain > 0 → RSI = 100
    assert state["avg_gain"] is not None, "avg_gain should be calculated"
    assert state["avg_loss"] is not None, "avg_loss should be calculated"
    assert state["avg_gain"] > 0, "avg_gain should be positive in uptrend"
    assert state["avg_loss"] == 0.0, "avg_loss should be 0 in pure uptrend"

    # Process final bar and check RSI
    final_features = transformer.transform(bars[-1])

    assert "rsi" in final_features, "RSI should be in features"
    assert not math.isnan(final_features["rsi"]), "RSI should NOT be NaN"
    assert final_features["rsi"] == 100.0, f"RSI should be 100.0 in pure uptrend, got {final_features['rsi']}"


def test_rsi_pure_downtrend():
    """
    Test: RSI should return 0.0 when all prices fall (avg_gain = 0).

    Scenario: 20 bars with consecutive price decreases
    Expected: RSI = 0.0 (not NaN)
    """
    # Create 20 bars with decreasing prices (30000 → 28100 in steps of 100)
    prices = [30000 - i * 100 for i in range(20)]
    bars = create_bars(prices)

    transformer = create_transformer_with_rsi()

    # Process all bars
    for bar in bars:
        transformer.transform(bar)

    # Get final state
    state = transformer.get_state()

    # In pure downtrend: avg_gain = 0, avg_loss > 0 → RSI = 0
    assert state["avg_gain"] is not None, "avg_gain should be calculated"
    assert state["avg_loss"] is not None, "avg_loss should be calculated"
    assert state["avg_gain"] == 0.0, "avg_gain should be 0 in pure downtrend"
    assert state["avg_loss"] > 0, "avg_loss should be positive in downtrend"

    # Process final bar and check RSI
    final_features = transformer.transform(bars[-1])

    assert "rsi" in final_features, "RSI should be in features"
    assert not math.isnan(final_features["rsi"]), "RSI should NOT be NaN"
    assert final_features["rsi"] == 0.0, f"RSI should be 0.0 in pure downtrend, got {final_features['rsi']}"


def test_rsi_no_movement():
    """
    Test: RSI should return 50.0 when price doesn't move (both gains and losses = 0).

    Scenario: 20 bars with constant price
    Expected: RSI = 50.0 (neutral)
    """
    # Create 20 bars with constant price
    prices = [29000] * 20
    bars = create_bars(prices)

    transformer = create_transformer_with_rsi()

    # Process all bars
    for bar in bars:
        transformer.transform(bar)

    # Get final state
    state = transformer.get_state()

    # In no movement: avg_gain = 0, avg_loss = 0 → RSI = 50
    assert state["avg_gain"] is not None, "avg_gain should be calculated"
    assert state["avg_loss"] is not None, "avg_loss should be calculated"
    assert state["avg_gain"] == 0.0, "avg_gain should be 0 with no movement"
    assert state["avg_loss"] == 0.0, "avg_loss should be 0 with no movement"

    # Process final bar and check RSI
    final_features = transformer.transform(bars[-1])

    assert "rsi" in final_features, "RSI should be in features"
    assert not math.isnan(final_features["rsi"]), "RSI should NOT be NaN"
    assert final_features["rsi"] == 50.0, f"RSI should be 50.0 with no movement, got {final_features['rsi']}"


def test_rsi_normal_case():
    """
    Test: RSI should calculate correctly when both gains and losses exist.

    Scenario: Mixed price movements (ups and downs)
    Expected: RSI calculated via formula: 100 - (100 / (1 + RS))
    """
    # Create realistic price movements (some ups, some downs)
    prices = [
        29000, 29100, 29050, 29200, 29150,  # Mixed
        29300, 29250, 29400, 29350, 29500,  # Mixed
        29450, 29600, 29550, 29700, 29650,  # Mixed
        29800, 29750, 29900, 29850, 30000,  # Overall uptrend
    ]
    bars = create_bars(prices)

    transformer = create_transformer_with_rsi()

    # Process all bars
    for bar in bars:
        transformer.transform(bar)

    # Get final state
    state = transformer.get_state()

    # In normal case: both avg_gain and avg_loss > 0
    assert state["avg_gain"] is not None, "avg_gain should be calculated"
    assert state["avg_loss"] is not None, "avg_loss should be calculated"
    assert state["avg_gain"] > 0, "avg_gain should be positive in mixed movements"
    assert state["avg_loss"] > 0, "avg_loss should be positive in mixed movements"

    # Process final bar and check RSI
    final_features = transformer.transform(bars[-1])

    assert "rsi" in final_features, "RSI should be in features"
    assert not math.isnan(final_features["rsi"]), "RSI should NOT be NaN"

    # Calculate expected RSI manually
    rs = state["avg_gain"] / state["avg_loss"]
    expected_rsi = 100.0 - (100.0 / (1.0 + rs))

    assert abs(final_features["rsi"] - expected_rsi) < 0.01, (
        f"RSI should match formula: expected {expected_rsi:.2f}, got {final_features['rsi']:.2f}"
    )

    # RSI should be between 0 and 100
    assert 0 <= final_features["rsi"] <= 100, f"RSI should be in range [0, 100], got {final_features['rsi']}"


def test_rsi_original_bug_scenario():
    """
    Test: Original bug scenario from issue description.

    Scenario: 4 bars rising by +100 each
    Original bug: Returns NaN for first 3 bars
    Expected: RSI = 100.0 for all bars after RSI period is met
    """
    # Exact scenario from bug report
    prices = [29000, 29100, 29200, 29300]

    # Need to add more bars to reach RSI period (14)
    # Continue pattern: +100 each bar
    for i in range(4, 20):
        prices.append(29000 + i * 100)

    bars = create_bars(prices)

    transformer = create_transformer_with_rsi()

    # Process bars and track RSI values
    rsi_values = []
    for i, bar in enumerate(bars):
        features = transformer.transform(bar)
        if i >= 14:  # After RSI period
            rsi_values.append((i, features.get("rsi", float("nan"))))

    # All RSI values after period should be 100.0 (not NaN)
    for i, rsi in rsi_values:
        assert not math.isnan(rsi), f"Bar {i}: RSI should NOT be NaN in pure uptrend"
        assert rsi == 100.0, f"Bar {i}: RSI should be 100.0, got {rsi}"


if __name__ == "__main__":
    # Run all tests
    test_rsi_pure_uptrend()
    print("✓ test_rsi_pure_uptrend passed")

    test_rsi_pure_downtrend()
    print("✓ test_rsi_pure_downtrend passed")

    test_rsi_no_movement()
    print("✓ test_rsi_no_movement passed")

    test_rsi_normal_case()
    print("✓ test_rsi_normal_case passed")

    test_rsi_original_bug_scenario()
    print("✓ test_rsi_original_bug_scenario passed")

    print("\n✅ All RSI NaN fix tests passed!")

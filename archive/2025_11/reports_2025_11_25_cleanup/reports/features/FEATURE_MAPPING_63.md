# Feature Mapping (63 features) - FINAL with ATR validity flag

## Summary
Total observation size: **63 features** (was 62, added 1 validity flag for ATR)

## Critical Change (62→63)
**Added ATR validity flag** to fix critical bug where vol_proxy became NaN during warmup period.

### Problem Solved
Before this change:
- ATR was the ONLY indicator without a validity flag
- `vol_proxy = tanh(log1p(atr / price))` used raw `atr` variable
- When ATR was NaN (first ~14 bars), vol_proxy became NaN
- This violated the core guarantee "no NaN in observation vector"

After this change:
- ATR now has `atr_valid` flag (index 16)
- vol_proxy checks `atr_valid` before calculation
- If invalid, uses fallback ATR (price * 0.01) for vol_proxy
- vol_proxy is NEVER NaN, even during warmup

## All Validity Flags (7 total)

| Indicator | Warmup Period | Fallback Value | Value Index | Flag Index | Problem Solved |
|-----------|---------------|----------------|-------------|------------|----------------|
| RSI | ~14 bars | 50.0 | 7 | 8 | Distinguish neutral (50) from no data (50) |
| MACD | ~26 bars | 0.0 | 9 | 10 | Distinguish no divergence from no data |
| MACD Signal | ~35 bars | 0.0 | 11 | 12 | Distinguish no signal from no data |
| Momentum | ~10 bars | 0.0 | 13 | 14 | Distinguish no movement from no data |
| **ATR** | **~14 bars** | **price\*0.01** | **15** | **16** | **Prevent NaN in vol_proxy (CRITICAL)** |
| CCI | ~20 bars | 0.0 | 17 | 18 | Distinguish average level from no data |
| OBV | 1 bar | 0.0 | 19 | 20 | Distinguish balance from no data |

## Feature Index Mapping (0-62)

### Bar Level Features (0-20)
| Index | Feature Name | Has Validity Flag | Fallback Value | Notes |
|-------|--------------|-------------------|----------------|-------|
| 0 | price | No | N/A | Always valid |
| 1 | log_volume_norm | No | N/A | Always valid |
| 2 | rel_volume | No | N/A | Always valid |
| 3 | ma5 | YES | 0.0 | Requires ~5 bars |
| 4 | ma5_valid | FLAG | 1.0/0.0 | Validity flag for ma5 |
| 5 | ma20 | YES | 0.0 | Requires ~20 bars |
| 6 | ma20_valid | FLAG | 1.0/0.0 | Validity flag for ma20 |
| 7 | rsi14 | YES ✅ | 50.0 | Requires ~14 bars |
| 8 | rsi_valid | FLAG ✅ | 1.0/0.0 | Validity flag for rsi14 |
| 9 | macd | YES ✅ | 0.0 | Requires ~26 bars |
| 10 | macd_valid | FLAG ✅ | 1.0/0.0 | Validity flag for macd |
| 11 | macd_signal | YES ✅ | 0.0 | Requires ~35 bars |
| 12 | macd_signal_valid | FLAG ✅ | 1.0/0.0 | Validity flag for macd_signal |
| 13 | momentum | YES ✅ | 0.0 | Requires ~10 bars |
| 14 | momentum_valid | FLAG ✅ | 1.0/0.0 | Validity flag for momentum |
| 15 | atr | **YES** ⭐ | price*0.01 | Requires ~14 bars (Wilder's EMA) |
| 16 | **atr_valid** | **FLAG** ⭐ | 1.0/0.0 | **NEW: Critical for vol_proxy** |
| 17 | cci | YES ✅ | 0.0 | Requires ~20 bars |
| 18 | cci_valid | FLAG ✅ | 1.0/0.0 | Validity flag for cci |
| 19 | obv | YES ✅ | 0.0 | Requires 1 bar |
| 20 | obv_valid | FLAG ✅ | 1.0/0.0 | Validity flag for obv |

### Derived Price/Volatility Features (21-22)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 21 | ret_bar | Bar-to-bar return (tanh normalized) |
| 22 | **vol_proxy** | **Volatility proxy from ATR (uses atr_valid!)** |

### Agent State Features (23-28)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 23 | cash_ratio | Cash / Total worth |
| 24 | position_ratio | Position value / Total worth |
| 25 | vol_imbalance | Volume imbalance (tanh) |
| 26 | trade_intensity | Trade intensity (tanh) |
| 27 | realized_spread | Realized spread (clipped) |
| 28 | agent_fill_ratio | Agent fill ratio |

### Technical Indicators for 4h Timeframe (29-31)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 29 | price_momentum | Uses momentum_valid flag |
| 30 | bb_squeeze | Uses bb_valid flag |
| 31 | trend_strength | Uses macd_valid AND macd_signal_valid flags |

### Bollinger Bands Context (32-33)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 32 | bb_position | Price position within Bollinger Bands |
| 33 | bb_width | Normalized Bollinger Band width |

### Event Metadata (34-36)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 34 | is_high_importance | Event importance flag |
| 35 | time_since_event | Time since last event (tanh) |
| 36 | risk_off_flag | Risk-off regime flag |

### Fear & Greed (37-38)
| Index | Feature Name | Has Validity Flag | Notes |
|-------|--------------|-------------------|-------|
| 37 | fear_greed_value | YES | Normalized 0-100 → [-3, 3] |
| 38 | fear_greed_indicator | FLAG | has_fear_greed flag |

### External Normalized Columns (39-59)
| Index Range | Feature Name | Count | Notes |
|-------------|--------------|-------|-------|
| 39-59 | norm_cols_values[0-20] | 21 | External features (4h timeframe) |

### Token Metadata (60-62)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 60 | token_count_ratio | num_tokens / max_num_tokens |
| 61 | token_id_norm | token_id / max_num_tokens |
| 62 | token_one_hot[0] | First position of one-hot encoding |

---

## Index Shift Summary (62 → 63)

All features AFTER index 15 (atr) have been shifted by **+1 position**:

| Feature | OLD Index (62) | NEW Index (63) | Shift |
|---------|----------------|----------------|-------|
| atr | 15 | 15 | 0 |
| **atr_valid** | - | **16** | NEW |
| cci | 16 | 17 | +1 |
| cci_valid | 17 | 18 | +1 |
| obv | 18 | 19 | +1 |
| obv_valid | 19 | 20 | +1 |
| ret_bar | 20 | 21 | +1 |
| vol_proxy | 21 | 22 | +1 |
| cash_ratio | 22 | 23 | +1 |
| position_ratio | 23 | 24 | +1 |
| ... | ... | ... | +1 |
| agent_fill_ratio | 27 | 28 | +1 |
| price_momentum | 28 | 29 | +1 |
| bb_squeeze | 29 | 30 | +1 |
| trend_strength | 30 | 31 | +1 |
| bb_position | 31 | 32 | +1 |
| bb_width | 32 | 33 | +1 |
| ... | ... | ... | +1 |
| external[0] | 38 | 39 | +1 |
| external[20] | 58 | 59 | +1 |
| token_count_ratio | 59 | 60 | +1 |
| token_id_norm | 60 | 61 | +1 |
| token_one_hot[0] | 61 | 62 | +1 |

---

## Total Feature Count
**Current: 63 features (indices 0-62)**
**Previous: 62 features (indices 0-61)**
**Original: 56 features (indices 0-55)**

## Migration History
- **v56 → v62**: Added 6 validity flags (RSI, MACD, MACD Signal, Momentum, CCI, OBV)
- **v62 → v63**: Added 1 validity flag (ATR) to fix vol_proxy NaN bug

## Breaking Changes
⚠️ **CRITICAL**: This change is NOT backwards compatible with models trained on 62-feature observations.
- All saved model checkpoints using 62 features must be retrained
- Observation shape changed from (62,) to (63,)
- Feature indices shifted for all features after index 15

See `MIGRATION_GUIDE_62_TO_63.md` for migration instructions.

---

## Research References
- Wilder, J. Welles (1978): "New Concepts in Technical Trading Systems" - ATR definition
- IEEE 754: NaN propagation through arithmetic operations
- "Defense in Depth" (OWASP): Multiple validation layers prevent failures
- "Fail-Fast Validation" (Martin Fowler): Validate before use, not just at storage
- "Incomplete Data - Machine Learning Trading" (OMSCS): Distinguish missing from neutral

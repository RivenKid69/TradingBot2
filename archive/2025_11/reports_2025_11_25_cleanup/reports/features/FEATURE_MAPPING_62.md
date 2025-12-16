# Feature Mapping (62 features) - AFTER adding validity flags

## Summary
Total observation size: **62 features** (was 56, added 6 validity flags)

## NEW Validity Flags Added
The following indicators now have explicit validity flags to eliminate ambiguity:

| Indicator | Warmup Period | Fallback Value | Validity Flag Index | Problem Solved |
|-----------|---------------|----------------|---------------------|----------------|
| RSI | ~14 bars | 50.0 | 8 | Distinguish neutral (50) from no data (50) |
| MACD | ~26 bars | 0.0 | 10 | Distinguish no divergence (0) from no data (0) |
| MACD Signal | ~35 bars | 0.0 | 12 | Distinguish no signal (0) from no data (0) |
| Momentum | ~10 bars | 0.0 | 14 | Distinguish no movement (0) from no data (0) |
| CCI | ~20 bars | 0.0 | 16 | Distinguish average level (0) from no data (0) |
| OBV | 1 bar | 0.0 | 18 | Distinguish balance (0) from no data (0) |

## Feature Index Mapping (0-61)

### Bar Level Features (0-19)
| Index | Feature Name | Has Validity Flag | Fallback Value | Notes |
|-------|--------------|-------------------|----------------|-------|
| 0 | price | No | N/A | Always valid |
| 1 | log_volume_norm | No | N/A | Always valid |
| 2 | rel_volume | No | N/A | Always valid |
| 3 | ma5 | **YES** | 0.0 | Requires ~5 bars |
| 4 | ma5_valid | FLAG | 1.0/0.0 | **Validity flag for ma5** |
| 5 | ma20 | **YES** | 0.0 | Requires ~20 bars |
| 6 | ma20_valid | FLAG | 1.0/0.0 | **Validity flag for ma20** |
| 7 | rsi14 | **YES** ✅ | 50.0 | Requires ~14 bars |
| 8 | rsi_valid | FLAG ✅ | 1.0/0.0 | **NEW: Validity flag for rsi14** |
| 9 | macd | **YES** ✅ | 0.0 | Requires ~26 bars |
| 10 | macd_valid | FLAG ✅ | 1.0/0.0 | **NEW: Validity flag for macd** |
| 11 | macd_signal | **YES** ✅ | 0.0 | Requires ~35 bars |
| 12 | macd_signal_valid | FLAG ✅ | 1.0/0.0 | **NEW: Validity flag for macd_signal** |
| 13 | momentum | **YES** ✅ | 0.0 | Requires ~10 bars |
| 14 | momentum_valid | FLAG ✅ | 1.0/0.0 | **NEW: Validity flag for momentum** |
| 15 | atr | No | price*0.01 | Intelligent fallback |
| 16 | cci | **YES** ✅ | 0.0 | Requires ~20 bars |
| 17 | cci_valid | FLAG ✅ | 1.0/0.0 | **NEW: Validity flag for cci** |
| 18 | obv | **YES** ✅ | 0.0 | Requires 1 bar |
| 19 | obv_valid | FLAG ✅ | 1.0/0.0 | **NEW: Validity flag for obv** |

### Derived Price/Volatility Features (20-21)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 20 | ret_bar | Bar-to-bar return (tanh normalized) |
| 21 | vol_proxy | Volatility proxy from ATR |

### Agent State Features (22-27)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 22 | cash_ratio | Cash / Total worth |
| 23 | position_ratio | Position value / Total worth |
| 24 | vol_imbalance | Volume imbalance (tanh) |
| 25 | trade_intensity | Trade intensity (tanh) |
| 26 | realized_spread | Realized spread (clipped) |
| 27 | agent_fill_ratio | Agent fill ratio |

### Technical Indicators for 4h Timeframe (28-30)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 28 | price_momentum | Replaces ofi_proxy (uses momentum indicator) |
| 29 | bb_squeeze | Replaces qimb (Bollinger Band width) |
| 30 | trend_strength | Replaces micro_dev (MACD divergence) |

### Bollinger Band Context (31-32)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 31 | bb_position | Price position within BB (0-1) |
| 32 | bb_width | Normalized band width |

### Event Metadata (33-35)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 33 | is_high_importance | Event importance flag |
| 34 | time_since_event | Time since last event (tanh) |
| 35 | risk_off_flag | Risk-off regime flag |

### Fear & Greed (36-37)
| Index | Feature Name | Has Validity Flag | Notes |
|-------|--------------|-------------------|-------|
| 36 | fear_greed_value | **YES** | Normalized 0-100 → [-3, 3] |
| 37 | fear_greed_indicator | FLAG | **has_fear_greed flag** |

### External Normalized Columns (38-58)
| Index Range | Feature Name | Count | Notes |
|-------------|--------------|-------|-------|
| 38-58 | norm_cols_values[0-20] | 21 | External features (4h timeframe) |

### Token Metadata (59-61)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 59 | token_count_ratio | num_tokens / max_num_tokens |
| 60 | token_id_norm | token_id / max_num_tokens |
| 61 | token_one_hot[0] | First position of one-hot encoding |

---

## Index Shift Summary (56 → 62)

All features AFTER index 7 (rsi14) have been shifted by **+6 positions**:

| Feature | OLD Index (56) | NEW Index (62) | Shift |
|---------|----------------|----------------|-------|
| price | 0 | 0 | 0 |
| log_volume_norm | 1 | 1 | 0 |
| rel_volume | 2 | 2 | 0 |
| ma5 | 3 | 3 | 0 |
| ma5_valid | 4 | 4 | 0 |
| ma20 | 5 | 5 | 0 |
| ma20_valid | 6 | 6 | 0 |
| rsi14 | 7 | 7 | 0 |
| **rsi_valid** | - | **8** | NEW |
| macd | 8 | 9 | +1 |
| **macd_valid** | - | **10** | NEW |
| macd_signal | 9 | 11 | +2 |
| **macd_signal_valid** | - | **12** | NEW |
| momentum | 10 | 13 | +3 |
| **momentum_valid** | - | **14** | NEW |
| atr | 11 | 15 | +4 |
| cci | 12 | 16 | +4 |
| **cci_valid** | - | **17** | NEW |
| obv | 13 | 18 | +5 |
| **obv_valid** | - | **19** | NEW |
| ret_bar | 14 | 20 | +6 |
| vol_proxy | 15 | 21 | +6 |
| ... | ... | ... | +6 |
| external[0] | 32 | 38 | +6 |
| external[20] | 52 | 58 | +6 |
| token_count_ratio | 53 | 59 | +6 |
| token_id_norm | 54 | 60 | +6 |
| token_one_hot[0] | 55 | 61 | +6 |

---

## Total Feature Count
**Current: 62 features (indices 0-61)**
**Previous: 56 features (indices 0-55)**

## Breaking Changes
⚠️ **CRITICAL**: This change is NOT backwards compatible with models trained on 56-feature observations.
- All saved model checkpoints using 56 features must be retrained
- Observation shape changed from (56,) to (62,)
- Feature indices shifted for all features after index 7

See `MIGRATION_GUIDE_56_TO_62.md` for migration instructions.

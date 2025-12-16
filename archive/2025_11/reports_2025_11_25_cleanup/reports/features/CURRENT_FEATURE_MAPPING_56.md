# Current Feature Mapping (56 features) - BEFORE adding validity flags

## Feature Index Mapping (0-55)

### Bar Level Features (0-13)
| Index | Feature Name | Has Validity Flag | Fallback Value | Notes |
|-------|--------------|-------------------|----------------|-------|
| 0 | price | No | N/A | Always valid |
| 1 | log_volume_norm | No | N/A | Always valid |
| 2 | rel_volume | No | N/A | Always valid |
| 3 | ma5 | **YES** | 0.0 | Requires ~5 bars |
| 4 | ma5_valid | FLAG | 1.0/0.0 | **Validity flag for ma5** |
| 5 | ma20 | **YES** | 0.0 | Requires ~20 bars |
| 6 | ma20_valid | FLAG | 1.0/0.0 | **Validity flag for ma20** |
| 7 | rsi14 | **NO** ❌ | 50.0 | ⚠️ AMBIGUOUS: 50.0 = neutral OR no data |
| 8 | macd | **NO** ❌ | 0.0 | ⚠️ AMBIGUOUS: 0.0 = no divergence OR no data |
| 9 | macd_signal | **NO** ❌ | 0.0 | ⚠️ AMBIGUOUS: 0.0 = no signal OR no data |
| 10 | momentum | **NO** ❌ | 0.0 | ⚠️ AMBIGUOUS: 0.0 = no movement OR no data |
| 11 | atr | No | price*0.01 | Intelligent fallback |
| 12 | cci | **NO** ❌ | 0.0 | ⚠️ AMBIGUOUS: 0.0 = average OR no data |
| 13 | obv | **NO** ❌ | 0.0 | ⚠️ AMBIGUOUS: 0.0 = balance OR no data |

### Derived Price/Volatility Features (14-15)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 14 | ret_bar | Bar-to-bar return (tanh normalized) |
| 15 | vol_proxy | Volatility proxy from ATR |

### Agent State Features (16-21)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 16 | cash_ratio | Cash / Total worth |
| 17 | position_ratio | Position value / Total worth |
| 18 | vol_imbalance | Volume imbalance (tanh) |
| 19 | trade_intensity | Trade intensity (tanh) |
| 20 | realized_spread | Realized spread (clipped) |
| 21 | agent_fill_ratio | Agent fill ratio |

### Technical Indicators for 4h Timeframe (22-24)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 22 | price_momentum | Replaces ofi_proxy (uses momentum indicator) |
| 23 | bb_squeeze | Replaces qimb (Bollinger Band width) |
| 24 | trend_strength | Replaces micro_dev (MACD divergence) |

### Bollinger Band Context (25-26)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 25 | bb_position | Price position within BB (0-1) |
| 26 | bb_width | Normalized band width |

### Event Metadata (27-29)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 27 | is_high_importance | Event importance flag |
| 28 | time_since_event | Time since last event (tanh) |
| 29 | risk_off_flag | Risk-off regime flag |

### Fear & Greed (30-31)
| Index | Feature Name | Has Validity Flag | Notes |
|-------|--------------|-------------------|-------|
| 30 | fear_greed_value | **YES** | Normalized 0-100 → [-3, 3] |
| 31 | fear_greed_indicator | FLAG | **has_fear_greed flag** |

### External Normalized Columns (32-52)
| Index Range | Feature Name | Count | Notes |
|-------------|--------------|-------|-------|
| 32-52 | norm_cols_values[0-20] | 21 | External features (4h timeframe) |

### Token Metadata (53-55)
| Index | Feature Name | Notes |
|-------|--------------|-------|
| 53 | token_count_ratio | num_tokens / max_num_tokens |
| 54 | token_id_norm | token_id / max_num_tokens |
| 55 | token_one_hot[0] | First position of one-hot encoding |

---

## CRITICAL ISSUES TO FIX

### Missing Validity Flags (6 indicators)
These indicators have **AMBIGUOUS fallback values** that create data interpretation problems:

1. **rsi14** (index 7): 50.0 fallback
   - Problem: 50.0 = neutral RSI OR insufficient data
   - Need flag to distinguish

2. **macd** (index 8): 0.0 fallback
   - Problem: 0.0 = no divergence OR insufficient data
   - Need flag to distinguish

3. **macd_signal** (index 9): 0.0 fallback
   - Problem: 0.0 = no signal OR insufficient data
   - Need flag to distinguish

4. **momentum** (index 10): 0.0 fallback
   - Problem: 0.0 = no movement OR insufficient data
   - Need flag to distinguish

5. **cci** (index 12): 0.0 fallback
   - Problem: 0.0 = average level OR insufficient data
   - Need flag to distinguish

6. **obv** (index 13): 0.0 fallback
   - Problem: 0.0 = volume balance OR insufficient data
   - Need flag to distinguish

### Solution
Add validity flags for ALL 6 indicators → increases observation size from **56 → 62**

---

## Total Feature Count
**Current: 56 features (indices 0-55)**
**After adding flags: 62 features (indices 0-61)**

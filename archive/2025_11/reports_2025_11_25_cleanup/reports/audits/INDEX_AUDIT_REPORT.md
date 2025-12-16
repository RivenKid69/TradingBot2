# CRITICAL: Index Audit Report - 63 Features

## Summary

**Status**: ⚠️ CRITICAL ERRORS FOUND
**Issue**: Documentation indices DO NOT match actual obs_builder.pyx code
**Impact**: Tests and documentation reference wrong feature indices

---

## Correct Feature Index Mapping (Based on obs_builder.pyx)

### Bar Level (0-2)
| Index | Feature | Status |
|-------|---------|--------|
| 0 | price | ✅ Correct |
| 1 | log_volume_norm | ✅ Correct |
| 2 | rel_volume | ✅ Correct |

### Moving Averages (3-6)
| Index | Feature | Status |
|-------|---------|--------|
| 3 | ma5 | ✅ Correct |
| 4 | ma5_valid | ✅ Correct |
| 5 | ma20 | ✅ Correct |
| 6 | ma20_valid | ✅ Correct |

### Technical Indicators with Validity Flags (7-20)
| Index | Feature | Status |
|-------|---------|--------|
| 7 | rsi14 | ✅ Correct |
| 8 | rsi_valid | ✅ Correct |
| 9 | macd | ✅ Correct |
| 10 | macd_valid | ✅ Correct |
| 11 | macd_signal | ✅ Correct |
| 12 | macd_signal_valid | ✅ Correct |
| 13 | momentum | ✅ Correct |
| 14 | momentum_valid | ✅ Correct |
| 15 | atr | ✅ Correct |
| 16 | **atr_valid** | ✅ NEW in 62→63 |
| 17 | cci | ✅ Correct |
| 18 | cci_valid | ✅ Correct |
| 19 | obv | ✅ Correct |
| 20 | obv_valid | ✅ Correct |

### Derived Price/Volatility (21-22) ⚠️ WRONG IN DOCS
| Index | Feature | Code | Docs (WRONG) |
|-------|---------|------|--------------|
| 21 | ret_bar | ✅ 21 | ❌ Says 23 |
| 22 | vol_proxy | ✅ 22 | ❌ Says 24 |

**ERROR**: Documentation says these are at 23-24, but code shows 21-22!

### Agent State (23-28) ⚠️ WRONG IN DOCS
| Index | Feature | Code | Docs (WRONG) |
|-------|---------|------|--------------|
| 23 | cash_ratio | ✅ 23 | ❌ Says 25 |
| 24 | position_ratio | ✅ 24 | ❌ Says 26 |
| 25 | vol_imbalance | ✅ 25 | ❌ Says 27 |
| 26 | trade_intensity | ✅ 26 | ❌ Says 28 |
| 27 | realized_spread | ✅ 27 | ❌ Says 29 |
| 28 | agent_fill_ratio | ✅ 28 | ❌ Says 30 |

**ERROR**: Documentation says 25-30, but code shows 23-28!

### Microstructure / Technical 4h (29-31) ⚠️ WRONG IN DOCS
| Index | Feature | Code | Docs (WRONG) |
|-------|---------|------|--------------|
| 29 | price_momentum | ✅ 29 | ❌ Says 32 |
| 30 | bb_squeeze | ✅ 30 | ❌ Says 31 |
| 31 | trend_strength | ✅ 31 | ❌ Says 33 |

**ERROR**: Documentation says 31-33, code shows 29-31!

### Bollinger Bands (32-33) ⚠️ CRITICAL ERROR IN DOCS
| Index | Feature | Code | Docs (WRONG) |
|-------|---------|------|--------------|
| 32 | bb_position | ✅ 32 | ❌ Says 21 |
| 33 | bb_width | ✅ 33 | ❌ Says 22 |

**CRITICAL ERROR**: Documentation says BB features are at 21-22, but they're actually at 32-33!
**Impact**: This swaps derived features (21-22) with BB features (32-33)!

### Event Metadata (34-36)
| Index | Feature | Status |
|-------|---------|--------|
| 34 | is_high_importance | ✅ Correct |
| 35 | time_since_event | ✅ Correct |
| 36 | risk_off_flag | ✅ Correct |

### Fear & Greed (37-38)
| Index | Feature | Status |
|-------|---------|--------|
| 37 | fear_greed_value | ✅ Correct |
| 38 | fear_greed_indicator | ✅ Correct |

### External Normalized Columns (39-59)
| Index Range | Feature | Status |
|-------------|---------|--------|
| 39-59 | norm_cols[0-20] | ✅ Correct |

### Token Metadata (60-61)
| Index | Feature | Status |
|-------|---------|--------|
| 60 | token_count_ratio | ✅ Correct |
| 61 | token_id_norm | ✅ Correct |

### Token One-Hot (62)
| Index | Feature | Status |
|-------|---------|--------|
| 62 | token_one_hot[0] | ✅ Correct |

---

## Root Cause Analysis

### Why This Happened

The documentation was written based on LOGICAL GROUPING (BB features grouped with other BB-related features), but the CODE implements a DIFFERENT ORDER:

**Documentation Logic** (WRONG):
1. Indicators (0-20)
2. **BB features (21-22)** ← Grouped with indicators
3. Derived (23-24)
4. Agent (25-30)
5. Microstructure (31-33)

**Code Implementation** (CORRECT):
1. Indicators (0-20)
2. **Derived (21-22)** ← Placed immediately after indicators
3. Agent (23-28)
4. Microstructure (29-31)
5. **BB features (32-33)** ← Placed after microstructure

### The Confusion

The **bb_position** and **bb_width** features were placed AFTER the microstructure block in the code, not before. This caused a cascading shift in all feature indices between positions 21-33.

---

## Impact Assessment

### Files with WRONG indices:
1. ❌ **FEATURE_MAPPING_63.md** - Major sections have wrong indices (21-33)
2. ❌ **OBSERVATION_MAPPING.md** - Same errors
3. ❌ **MIGRATION_GUIDE_62_TO_63.md** - Index shift table is wrong
4. ✅ **test_atr_validity_flag.py** - FIXED (vol_proxy 23→22)

### Tests Status:
- ✅ test_atr_validity_flag.py - NOW FIXED
- ⚠️ Other tests need verification

---

## Correct Feature Block Summary

| Block | Indices | Count | Features |
|-------|---------|-------|----------|
| Bar | 0-2 | 3 | price, log_volume_norm, rel_volume |
| MA | 3-6 | 4 | ma5, ma5_valid, ma20, ma20_valid |
| Indicators | 7-20 | 14 | 7 indicators × 2 (value + valid) |
| **Derived** | **21-22** | **2** | **ret_bar, vol_proxy** |
| **Agent** | **23-28** | **6** | **cash_ratio, position_ratio, ...** |
| **Microstructure** | **29-31** | **3** | **price_momentum, bb_squeeze, trend_strength** |
| **BB** | **32-33** | **2** | **bb_position, bb_width** |
| Metadata | 34-36 | 3 | event-related features |
| Fear/Greed | 37-38 | 2 | fear_greed_value, indicator |
| External | 39-59 | 21 | norm_cols[0-20] |
| Token Meta | 60-61 | 2 | token ratios |
| Token OH | 62 | 1 | token_one_hot[0] |
| **TOTAL** | **0-62** | **63** | **All features** |

---

## Action Items

### ✅ COMPLETED:
1. Fixed test_atr_validity_flag.py (vol_proxy: 23→22)

### ⏳ TODO:
1. Rewrite FEATURE_MAPPING_63.md with correct indices
2. Rewrite OBSERVATION_MAPPING.md with correct indices
3. Update MIGRATION_GUIDE_62_TO_63.md index shift table
4. Verify all other tests use correct indices
5. Create new commit with fixes

---

## Critical Indices Reference (for quick verification)

| Index | Feature | Use Case |
|-------|---------|----------|
| 15 | atr | Value |
| 16 | **atr_valid** | **NEW flag** |
| 17 | cci | Next indicator after ATR |
| 21 | ret_bar | Bar return (NOT BB!) |
| 22 | **vol_proxy** | **Uses atr_valid** |
| 23 | cash_ratio | Agent state start |
| 29 | price_momentum | Microstructure start |
| 32 | bb_position | BB features start (NOT 21!) |
| 33 | bb_width | BB features end |
| 39 | norm_cols[0] | External start |
| 62 | token_one_hot[0] | Last feature |

---

**Date**: 2025-11-16
**Audit**: Comprehensive index verification
**Status**: CRITICAL ERRORS FOUND AND PARTIALLY FIXED
**Next**: Fix all documentation files

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# This module defines the observation feature layout for the trading environment.
#
# НЕ БАГ #50: FEATURE DIMENSION SYNCHRONIZATION RISK IS ADDRESSED BY DESIGN
# ═══════════════════════════════════════════════════════════════════════════════
# FEATURES_LAYOUT defines block sizes, but the ACTUAL observation vector is ALWAYS
# built by obs_builder.pyx:build_observation_vector_c(). This file is ONLY used for:
# 1. Size calculation via compute_n_features()
# 2. Documentation of feature order
#
# DEFENSE IN DEPTH:
# - P0: mediator validation (validates inputs before obs_builder)
# - P1: obs_builder Python wrapper validates array sizes
# - P2: obs_builder.pyx builds actual vector (Cython boundscheck=False for speed)
# - P3: observation_space.shape is set from compute_n_features(FEATURES_LAYOUT)
#
# If layout changes:
# 1. Update BOTH this file AND obs_builder.pyx:build_observation_vector_c()
# 2. Run pytest tests/test_feature_*.py to verify consistency
#
# Reference: docs/NOT_BUGS_AND_FAQ.md → "НЕ БАГИ" → #50
# ═══════════════════════════════════════════════════════════════════════════════

# Default normalization constants
DEFAULT_TANH_CLIP = 0.999
OBS_EPS = 1e-8
# Default maximum sizes
MAX_NUM_TOKENS = 1  # Changed from 16 to match mediator and lob_state_cython
# EXT_NORM_DIM: External normalized features dimension
# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 (2025-11-27): Expanded from 21 to 28 for stock-specific features
# Phase 6 (2025-11-28): Expanded from 28 to 35 for macro data & earnings/dividend features
#
# Indices 0-20:  Original crypto features (cvd, garch, yang_zhang, returns, taker_buy_ratio)
# Indices 21-27: Stock-specific features (Phase 5 - backward compatible)
#   [21] vix_normalized     - VIX index value (normalized via tanh)
#   [22] vix_regime         - VIX regime indicator (0-1 scale: low/normal/elevated/extreme)
#   [23] market_regime      - Bull/Bear/Sideways indicator (-1 to 1)
#   [24] rs_spy_20d         - 20-day relative strength vs SPY
#   [25] rs_spy_50d         - 50-day relative strength vs SPY
#   [26] rs_qqq_20d         - 20-day relative strength vs QQQ
#   [27] sector_momentum    - Sector momentum relative to market
#
# Indices 28-34: Macro & Corporate Features (Phase 6 - L3 Stock Simulation)
#   [28] dxy_normalized     - Dollar Index (normalized via tanh, center at 100)
#   [29] treasury_10y_yield - 10-Year Treasury Yield (normalized, typical range 2-5%)
#   [30] real_yield_proxy   - Real yield approximation (10y yield - inflation proxy)
#   [31] days_until_earnings_norm - Days to next earnings (normalized 0-1, 90 days = 1)
#   [32] dividend_yield_norm - Trailing 12-month dividend yield (normalized)
#   [33] earnings_surprise_norm - Last earnings surprise (normalized via tanh)
#   [34] in_earnings_blackout - 0/1 flag for earnings blackout period (<14 days)
#
# Backward Compatibility:
# - Crypto data won't have stock features → validity flags = False
# - Default values are sensible (0.0)
# - No changes to existing crypto observation building
# ═══════════════════════════════════════════════════════════════════════════════
EXT_NORM_DIM = 35  # 21 crypto + 7 stock + 7 macro/corp features

# Initialize feature layout (to be populated by make_layout)
FEATURES_LAYOUT = []


def make_layout(obs_params=None):
    """Build the observation feature layout list based on parameters."""
    global FEATURES_LAYOUT, N_FEATURES
    if obs_params is None:
        obs_params = {}
    # Determine actual dimensions from parameters or defaults
    max_tokens = obs_params.get("max_num_tokens", MAX_NUM_TOKENS)
    ext_dim = obs_params.get("ext_norm_dim", EXT_NORM_DIM)
    include_fear = obs_params.get("include_fear_greed", obs_params.get("use_dynamic_risk", False))
    # Define feature blocks
    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL: This order MUST match obs_builder.pyx build_observation_vector_c()!
    # See obs_builder.pyx:236-590 for the exact order of feature construction.
    # If you modify this, update obs_builder.pyx AND run feature tests!
    # ═══════════════════════════════════════════════════════════════════════════
    layout = []

    # Block 1: Bar-level features (indices 0-2)
    layout.append(
        {
            "name": "bar",
            "size": 3,
            "dtype": "float32",
            "clip": None,
            "scale": 1.0,
            "bias": 0.0,
            "source": "bars",
            "description": "price, log_volume_norm, rel_volume",
        }
    )

    # Block 2: MA5 (indices 3-4)
    layout.append(
        {
            "name": "ma5",
            "size": 2,
            "dtype": "float32",
            "clip": None,
            "scale": 1.0,
            "bias": 0.0,
            "source": "indicators",
            "description": "ma5, is_ma5_valid",
        }
    )

    # Block 3: MA20 (indices 5-6)
    layout.append(
        {
            "name": "ma20",
            "size": 2,
            "dtype": "float32",
            "clip": None,
            "scale": 1.0,
            "bias": 0.0,
            "source": "indicators",
            "description": "ma20, is_ma20_valid",
        }
    )

    # Block 4: Technical indicators with validity flags (indices 7-20)
    # Includes: rsi14, macd, macd_signal, momentum, atr, cci, obv + validity flags
    # Total: 7 indicators × 2 (value + flag) = 14 features
    layout.append(
        {
            "name": "indicators",
            "size": 14,
            "dtype": "float32",
            "clip": None,
            "scale": 1.0,
            "bias": 0.0,
            "source": "indicators",
            "description": "rsi14, macd, macd_signal, momentum, atr, cci, obv (each with validity flag)",
        }
    )

    # Block 5: Derived price/volatility signals (indices 21-22)
    # NOTE: This comes AFTER indicators, not before! (Fixed from previous incorrect ordering)
    layout.append(
        {
            "name": "derived",
            "size": 2,
            "dtype": "float32",
            "clip": DEFAULT_TANH_CLIP,
            "scale": 1.0,
            "bias": 0.0,
            "source": "derived",
            "description": "ret_bar (bar-to-bar return), vol_proxy (volatility proxy from ATR)",
        }
    )

    # Block 6: Agent state features (indices 23-29)
    # FIX (2025-11-24): Added signal_pos (index 29) - critical for signal_only mode
    layout.append(
        {
            "name": "agent",
            "size": 7,
            "dtype": "float32",
            "clip": DEFAULT_TANH_CLIP,
            "scale": 1.0,
            "bias": 0.0,
            "source": "agent",
            "description": "cash_ratio, position_ratio, vol_imbalance, trade_intensity, realized_spread, fill_ratio, signal_pos",
        }
    )

    # Block 7: Microstructure proxies (indices 30-32) - shifted +1 after signal_pos
    layout.append(
        {
            "name": "microstructure",
            "size": 3,
            "dtype": "float32",
            "clip": DEFAULT_TANH_CLIP,
            "scale": 1.0,
            "bias": 0.0,
            "source": "micro",
            "description": "price_momentum, bb_squeeze, trend_strength (MACD divergence)",
        }
    )

    # Block 8: Bollinger Bands context (indices 33-34) - shifted +1 after signal_pos
    # NOTE: This was missing from previous versions!
    layout.append(
        {
            "name": "bb_context",
            "size": 2,
            "dtype": "float32",
            "clip": None,
            "scale": 1.0,
            "bias": 0.0,
            "source": "indicators",
            "description": "bb_position (price within bands), bb_width_norm (band width normalized)",
        }
    )

    # Block 9: Event metadata (indices 35-39) - shifted +1 after signal_pos
    layout.append(
        {
            "name": "metadata",
            "size": 5,
            "dtype": "float32",
            "clip": DEFAULT_TANH_CLIP,
            "scale": 1.0,
            "bias": 0.0,
            "source": "meta",
            "description": "is_high_importance, time_since_event, risk_off_flag, fear_greed_value, fear_greed_indicator",
        }
    )
    # External normalized columns block (if any)
    if ext_dim and ext_dim > 0:
        layout.append(
            {
                "name": "external",
                "size": ext_dim,
                "dtype": "float32",
                "clip": None,
                "scale": 1.0,
                "bias": 0.0,
                "source": "external",
            }
        )
    # External validity flags block (NEW - Phase 2 of ISSUE #2 fix)
    # One validity flag per external feature to distinguish missing data (NaN) from zero values
    if ext_dim and ext_dim > 0:
        layout.append(
            {
                "name": "external_validity",
                "size": ext_dim,
                "dtype": "float32",
                "clip": None,
                "scale": 1.0,
                "bias": 0.0,
                "source": "external",
                "description": "Validity flags for external features (1.0=valid, 0.0=NaN/missing)",
            }
        )
    # Token metadata block (num_tokens_norm, token_id_norm)
    if max_tokens > 0:
        layout.append(
            {
                "name": "token_meta",
                "size": 2,
                "dtype": "float32",
                "clip": None,
                "scale": 1.0,
                "bias": 0.0,
                "source": "token_meta",
            }
        )
    # Token one-hot block
    if max_tokens > 0:
        layout.append(
            {
                "name": "token",
                "size": max_tokens,
                "dtype": "float32",
                "clip": None,
                "scale": 1.0,
                "bias": 0.0,
                "source": "token",
            }
        )
    FEATURES_LAYOUT = layout
    # Compute total feature vector length
    N_FEATURES = sum(block["size"] for block in layout)
    return FEATURES_LAYOUT


# Build default layout with default parameters on import
make_layout({})

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from cython cimport Py_ssize_t
from libc.math cimport tanh, log1p, isnan


cdef inline float _clipf(double value, double lower, double upper) nogil:
    if value < lower:
        value = lower
    elif value > upper:
        value = upper
    return <float>value


cpdef int compute_n_features(list layout):
    """Utility used by legacy Python code to count feature slots."""
    cdef int total = 0
    cdef dict block
    for block in layout:
        total += <int>block.get("size", 0)
    return total


cdef void build_observation_vector_c(
    float price,
    float prev_price,
    float log_volume_norm,
    float rel_volume,
    float ma5,
    float ma20,
    float rsi14,
    float macd,
    float macd_signal,
    float momentum,
    float atr,
    float cci,
    float obv,
    float bb_lower,
    float bb_upper,
    float is_high_importance,
    float time_since_event,
    float fear_greed_value,
    bint has_fear_greed,
    bint risk_off_flag,
    float cash,
    float units,
    float last_vol_imbalance,
    float last_trade_intensity,
    float last_realized_spread,
    float last_agent_fill_ratio,
    int token_id,
    int max_num_tokens,
    int num_tokens,
    float[::1] norm_cols_values,
    float[::1] out_features
) noexcept nogil:
    """Populate ``out_features`` with the observation vector without acquiring the GIL."""

    cdef int feature_idx = 0
    cdef float feature_val
    cdef float indicator
    cdef double price_d = price
    cdef double prev_price_d = prev_price
    cdef double position_value
    cdef double total_worth
    cdef double ret_bar
    cdef double vol_proxy
    cdef double price_momentum
    cdef double bb_squeeze
    cdef double trend_strength
    cdef double bb_width
    cdef bint ma5_valid
    cdef bint ma20_valid
    cdef bint bb_valid
    cdef double min_bb_width
    cdef int padded_tokens
    cdef Py_ssize_t i

    # --- Bar level block ---------------------------------------------------
    out_features[feature_idx] = price
    feature_idx += 1
    out_features[feature_idx] = log_volume_norm
    feature_idx += 1
    out_features[feature_idx] = rel_volume
    feature_idx += 1

    ma5_valid = not isnan(ma5)
    out_features[feature_idx] = ma5 if ma5_valid else 0.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if ma5_valid else 0.0
    feature_idx += 1

    ma20_valid = not isnan(ma20)
    out_features[feature_idx] = ma20 if ma20_valid else 0.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if ma20_valid else 0.0
    feature_idx += 1

    out_features[feature_idx] = rsi14
    feature_idx += 1
    out_features[feature_idx] = macd
    feature_idx += 1
    out_features[feature_idx] = macd_signal
    feature_idx += 1
    out_features[feature_idx] = momentum
    feature_idx += 1
    out_features[feature_idx] = atr
    feature_idx += 1
    out_features[feature_idx] = cci
    feature_idx += 1
    out_features[feature_idx] = obv
    feature_idx += 1

    # Derived price/volatility signals (bar-to-bar return for current timeframe)
    ret_bar = tanh((price_d - prev_price_d) / (prev_price_d + 1e-8))
    out_features[feature_idx] = <float>ret_bar
    feature_idx += 1

    vol_proxy = tanh(log1p(atr / (price_d + 1e-8)))
    out_features[feature_idx] = <float>vol_proxy
    feature_idx += 1

    # --- Agent state block -------------------------------------------------
    position_value = units * price_d
    total_worth = cash + position_value

    if total_worth <= 1e-8:
        feature_val = 1.0
    else:
        feature_val = _clipf(cash / total_worth, 0.0, 1.0)
    out_features[feature_idx] = feature_val
    feature_idx += 1

    if total_worth <= 1e-8:
        feature_val = 0.0
    else:
        feature_val = <float>tanh(position_value / (total_worth + 1e-8))
    out_features[feature_idx] = feature_val
    feature_idx += 1

    out_features[feature_idx] = <float>tanh(last_vol_imbalance)
    feature_idx += 1
    out_features[feature_idx] = <float>tanh(last_trade_intensity)
    feature_idx += 1

    feature_val = _clipf(last_realized_spread, -0.1, 0.1)
    out_features[feature_idx] = feature_val
    feature_idx += 1

    out_features[feature_idx] = last_agent_fill_ratio
    feature_idx += 1

    # --- Technical indicators for 4h timeframe (replaces microstructure) ---
    # Microstructure features (ofi_proxy, qimb, micro_dev) are not applicable for 4h timeframe
    # as they require high-frequency order flow data. Replaced with candlestick-based indicators.

    # 1. Price momentum (replaces ofi_proxy) - captures trend direction and strength
    # Uses normalized momentum indicator to measure price movement strength
    # Normalized by 1% of price (price_d * 0.01) for sensitivity to typical intraday moves
    price_momentum = tanh(momentum / (price_d * 0.01 + 1e-8))
    out_features[feature_idx] = <float>price_momentum
    feature_idx += 1

    # 2. Bollinger Bands squeeze (replaces qimb) - measures volatility regime
    # High value = high volatility (wide bands), low value = low volatility (squeeze)
    # Normalized by full price (price_d) not 1% because bb_width is typically 1-5% of price
    # This ensures the normalized value is in a reasonable range for tanh
    bb_squeeze = tanh((bb_upper - bb_lower) / (price_d + 1e-8))
    out_features[feature_idx] = <float>bb_squeeze
    feature_idx += 1

    # 3. Trend strength via MACD divergence (replaces micro_dev) - measures trend strength
    # Positive = bullish trend, negative = bearish trend, magnitude = strength
    # Normalized by 1% of price (price_d * 0.01) similar to price_momentum for consistency
    trend_strength = tanh((macd - macd_signal) / (price_d * 0.01 + 1e-8))
    out_features[feature_idx] = <float>trend_strength
    feature_idx += 1

    # --- Bollinger band context -------------------------------------------
    bb_width = bb_upper - bb_lower
    bb_valid = not isnan(bb_lower)
    min_bb_width = price_d * 0.0001
    if (not bb_valid) or bb_width <= min_bb_width:
        feature_val = 0.5
    else:
        feature_val = _clipf((price_d - bb_lower) / (bb_width + 1e-9), -1.0, 2.0)
    out_features[feature_idx] = feature_val
    feature_idx += 1

    if bb_valid:
        feature_val = _clipf(bb_width / (price_d + 1e-8), 0.0, 10.0)
    else:
        feature_val = 0.0
    out_features[feature_idx] = feature_val
    feature_idx += 1

    # --- Event metadata ----------------------------------------------------
    out_features[feature_idx] = is_high_importance
    feature_idx += 1

    out_features[feature_idx] = <float>tanh(time_since_event / 24.0)
    feature_idx += 1

    out_features[feature_idx] = 1.0 if risk_off_flag else 0.0
    feature_idx += 1

    # --- Fear & Greed ------------------------------------------------------
    if has_fear_greed:
        feature_val = _clipf(fear_greed_value / 100.0, -3.0, 3.0)
        indicator = 1.0
    else:
        feature_val = 0.0
        indicator = 0.0
    out_features[feature_idx] = feature_val
    feature_idx += 1
    out_features[feature_idx] = indicator
    feature_idx += 1

    # --- External normalised columns --------------------------------------
    for i in range(norm_cols_values.shape[0]):
        # Apply tanh normalization first, then clip to safe range
        feature_val = _clipf(tanh(norm_cols_values[i]), -3.0, 3.0)
        out_features[feature_idx] = feature_val
        feature_idx += 1

    # --- Token metadata ----------------------------------------------------
    if max_num_tokens > 0:
        # Normalised statistics to keep vector length fixed
        feature_val = _clipf(num_tokens / (<double>max_num_tokens), 0.0, 1.0)
        out_features[feature_idx] = feature_val
        feature_idx += 1

        if 0 <= token_id < max_num_tokens:
            feature_val = _clipf(token_id / (<double>max_num_tokens), 0.0, 1.0)
        else:
            feature_val = 0.0
        out_features[feature_idx] = feature_val
        feature_idx += 1

        padded_tokens = max_num_tokens
        for i in range(padded_tokens):
            out_features[feature_idx + i] = 0.0

        if 0 <= token_id < num_tokens and token_id < max_num_tokens:
            out_features[feature_idx + token_id] = 1.0

        feature_idx += padded_tokens


cpdef void build_observation_vector(
    float price,
    float prev_price,
    float log_volume_norm,
    float rel_volume,
    float ma5,
    float ma20,
    float rsi14,
    float macd,
    float macd_signal,
    float momentum,
    float atr,
    float cci,
    float obv,
    float bb_lower,
    float bb_upper,
    float is_high_importance,
    float time_since_event,
    float fear_greed_value,
    bint has_fear_greed,
    bint risk_off_flag,
    float cash,
    float units,
    float last_vol_imbalance,
    float last_trade_intensity,
    float last_realized_spread,
    float last_agent_fill_ratio,
    int token_id,
    int max_num_tokens,
    int num_tokens,
    float[::1] norm_cols_values,
    float[::1] out_features
) noexcept:
    """Python-callable wrapper that forwards to the ``nogil`` implementation."""

    build_observation_vector_c(
        price,
        prev_price,
        log_volume_norm,
        rel_volume,
        ma5,
        ma20,
        rsi14,
        macd,
        macd_signal,
        momentum,
        atr,
        cci,
        obv,
        bb_lower,
        bb_upper,
        is_high_importance,
        time_since_event,
        fear_greed_value,
        has_fear_greed,
        risk_off_flag,
        cash,
        units,
        last_vol_imbalance,
        last_trade_intensity,
        last_realized_spread,
        last_agent_fill_ratio,
        token_id,
        max_num_tokens,
        num_tokens,
        norm_cols_values,
        out_features,
    )

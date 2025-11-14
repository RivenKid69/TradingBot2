# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from cython cimport Py_ssize_t
from libc.math cimport tanh, log1p, isnan, isinf, isfinite


cdef inline float _clipf(double value, double lower, double upper) nogil:
    """
    Clip value to [lower, upper] range with NaN handling.

    CRITICAL: NaN comparisons are always False in C/Cython, so we must check explicitly.
    If value is NaN, we return 0.0 as a safe default to prevent NaN propagation.
    """
    if isnan(value):
        return 0.0
    if value < lower:
        value = lower
    elif value > upper:
        value = upper
    return <float>value


cdef inline void _validate_price(float price, str param_name) except *:
    """
    Validate that price is finite and positive.

    This function enforces critical data integrity constraints for price data:
    1. Price must not be NaN (indicates missing or corrupted data)
    2. Price must not be Inf/-Inf (indicates computation overflow)
    3. Price must be strictly positive (negative/zero prices are invalid)

    Args:
        price: The price value to validate
        param_name: Name of the parameter for error messages (e.g., "price", "prev_price")

    Raises:
        ValueError: If price fails validation with detailed diagnostic message

    Research references:
    - "Data validation best practices" (Cube Software)
    - "Incomplete Data - Machine Learning Trading" (OMSCS)
    - Financial data standards require positive, finite prices
    """
    if isnan(price):
        raise ValueError(
            f"Invalid {param_name}: NaN (Not a Number). "
            f"This indicates missing or corrupted market data. "
            f"All price inputs must be valid finite numbers. "
            f"Check data source integrity and preprocessing pipeline."
        )

    if isinf(price):
        sign = "positive" if price > 0 else "negative"
        raise ValueError(
            f"Invalid {param_name}: {sign} infinity. "
            f"This indicates arithmetic overflow in upstream calculations. "
            f"All price inputs must be finite values. "
            f"Review data transformations and numerical stability."
        )

    if price <= 0.0:
        raise ValueError(
            f"Invalid {param_name}: {price:.10f}. "
            f"Price must be strictly positive (> 0). "
            f"Negative or zero prices are invalid in trading systems. "
            f"This may indicate data errors, incorrect units, or "
            f"issues with price normalization/denormalization."
        )


cdef inline void _validate_portfolio_value(float value, str param_name) except *:
    """
    Validate portfolio value (cash or units) - finite but can be zero.

    Portfolio values have different validation rules than prices:
    - CAN be 0.0 (valid state: no cash or no position)
    - CAN be negative for cash (short positions, margin debt)
    - CANNOT be NaN (indicates missing/corrupted data)
    - CANNOT be Inf (indicates calculation overflow)

    Args:
        value: The portfolio value to validate (cash or units)
        param_name: Parameter name for error messages

    Raises:
        ValueError: If value is NaN or Inf

    Best practices:
    - Zero cash/units is valid portfolio state
    - Negative cash can be valid (margin, short positions)
    - NaN/Inf indicate data corruption and must be caught

    References:
    - "Investment Model Validation" (CFA Institute)
    - "Best Practices for Ensuring Financial Data Accuracy" (Paystand)
    """
    if isnan(value):
        raise ValueError(
            f"Invalid {param_name}: NaN (Not a Number). "
            f"Portfolio values must be finite numbers. "
            f"NaN indicates missing or corrupted portfolio state. "
            f"Check state management and data pipeline integrity."
        )

    if isinf(value):
        sign = "positive" if value > 0 else "negative"
        raise ValueError(
            f"Invalid {param_name}: {sign} infinity. "
            f"Portfolio values must be finite. "
            f"Infinity indicates arithmetic overflow in calculations. "
            f"Check portfolio valuation logic and numerical stability."
        )


cdef inline void _validate_volume_metric(float value, str param_name) except *:
    """
    Validate volume-derived metrics (log_volume_norm, rel_volume).

    Volume metrics are derived from market data transformations and must be finite.
    Unlike prices, they CAN be zero (no volume) or negative (theoretical edge case).
    However, they CANNOT be NaN or Inf as this indicates upstream calculation errors.

    Typical range: [-1, 1] due to tanh normalization, but we validate finitude only.

    Args:
        value: The volume metric to validate
        param_name: Parameter name for error messages (e.g., "log_volume_norm")

    Raises:
        ValueError: If value is NaN or Inf with detailed diagnostic message

    Research references:
    - "Defense in Depth" (OWASP): Multiple validation layers prevent NaN propagation
    - "Data Validation Best Practices" (Cube Software): Validate at boundaries
    - "Fail-fast validation" (Martin Fowler): Catch errors early in pipeline

    Design rationale:
    - Volume metrics computed in mediator._extract_market_data()
    - Formula: tanh(log1p(volume / normalizer)) with volume >= 0 guaranteed by P0
    - P0 layer (_get_safe_float with min_value=0.0) prevents negative volumes
    - With volume >= 0, tanh(log1p(...)) always yields finite result in [-1, 1]
    - This P2 validation catches any remaining edge cases or pipeline errors
    - Fail-fast approach prevents silent NaN propagation to observation vector
    """
    if isnan(value):
        raise ValueError(
            f"Invalid {param_name}: NaN (Not a Number). "
            f"Volume metrics must be finite numbers. "
            f"NaN indicates corrupted market data or calculation error. "
            f"Check volume data source and normalization pipeline. "
            f"Common causes: missing volume data, division by zero, log of negative."
        )

    if isinf(value):
        sign = "positive" if value > 0 else "negative"
        raise ValueError(
            f"Invalid {param_name}: {sign} infinity. "
            f"Volume metrics must be finite. "
            f"Infinity indicates numerical overflow in volume normalization. "
            f"Check volume scaling factors and log1p/tanh computations. "
            f"Review mediator.py _extract_market_data() for calculation errors."
        )


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

    # Technical indicators with NaN handling (early bars may not have enough history)
    # RSI: 50.0 = neutral (no trend signal)
    out_features[feature_idx] = rsi14 if not isnan(rsi14) else 50.0
    feature_idx += 1

    # MACD: 0.0 = no divergence signal
    out_features[feature_idx] = macd if not isnan(macd) else 0.0
    feature_idx += 1
    out_features[feature_idx] = macd_signal if not isnan(macd_signal) else 0.0
    feature_idx += 1

    # Momentum: 0.0 = no price movement
    out_features[feature_idx] = momentum if not isnan(momentum) else 0.0
    feature_idx += 1

    # ATR: default to 1% of price (small volatility estimate)
    out_features[feature_idx] = atr if not isnan(atr) else <float>(price_d * 0.01)
    feature_idx += 1

    # CCI: 0.0 = at average level
    out_features[feature_idx] = cci if not isnan(cci) else 0.0
    feature_idx += 1

    # OBV: always valid, but handle NaN defensively
    out_features[feature_idx] = obv if not isnan(obv) else 0.0
    feature_idx += 1

    # CRITICAL: Derived price/volatility signals (bar-to-bar return for current timeframe)
    # ret_bar calculation (feature index 14):
    # - Formula: tanh((price_d - prev_price_d) / (prev_price_d + 1e-8))
    # - Numerator: price_d - prev_price_d (price change)
    # - Denominator: prev_price_d + 1e-8 (epsilon prevents division by zero)
    # - tanh normalization: maps (-inf, +inf) → (-1, 1)
    #
    # Safety guarantees:
    # 1. Division by zero: Impossible due to +1e-8 epsilon
    #    Even if prev_price_d = 0.0: division = x / 1e-8 = large finite number
    #
    # 2. NaN/Inf protection: Enforced by fail-fast validation at entry points
    #    - P0: Mediator validation (_validate_critical_price at mediator.py:1015)
    #    - P1: Wrapper validation (_validate_price at obs_builder.pyx:469-470)
    #    Both price AND prev_price are validated as finite, positive, non-zero
    #    If validation fails → ValueError raised immediately (fail-fast)
    #
    # 3. No silent failures: Invalid data causes immediate exception, not silent corruption
    #
    # Direct call path (lob_state_cython.pyx:62):
    # - Only used for feature vector size calculation with dummy zeros
    # - price=0.0, prev_price=0.0 → ret_bar = tanh(0/1e-8) = tanh(0) = 0.0
    # - Safe and correct for initialization purposes
    #
    # Design philosophy: Fail-fast at entry (P0/P1) > Silent fallbacks in computation
    # Research references:
    # - "Fail-fast validation" (Martin Fowler): Catch errors early, fail loudly
    # - IEEE 754: NaN propagation requires explicit handling at data boundaries
    # - Financial data standards: Validation at ingestion, not in calculations
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
    # NaN handling: if momentum is NaN (first 10 bars), use 0.0 (no momentum)
    if not isnan(momentum):
        price_momentum = tanh(momentum / (price_d * 0.01 + 1e-8))
    else:
        price_momentum = 0.0
    out_features[feature_idx] = <float>price_momentum
    feature_idx += 1

    # 2. Bollinger Bands squeeze (replaces qimb) - measures volatility regime
    # High value = high volatility (wide bands), low value = low volatility (squeeze)
    # Normalized by full price (price_d) not 1% because bb_width is typically 1-5% of price
    # This ensures the normalized value is in a reasonable range for tanh
    # NaN handling: if BB not ready (first 20 bars), use 0.0 (neutral volatility)
    #
    # CRITICAL: Validate BOTH bb_lower AND bb_upper for completeness
    # - Bollinger Bands require both bounds to be valid
    # - If only one is NaN/Inf, derived features (bb_width, bb_position) become NaN
    # - bb_width = bb_upper - bb_lower: if either is NaN/Inf → NaN propagation
    # - Must check finitude (not just NaN) to catch Inf values
    # - Logical consistency: bb_upper should be >= bb_lower (sanity check)
    #
    # Research references:
    # - "Bollinger Bands" (John Bollinger): Upper band > Lower band by definition
    # - "Defense in Depth" (OWASP): Validate all required inputs, not just subset
    # - "Data Validation Best Practices": Complete validation prevents partial failures
    bb_valid = (not isnan(bb_lower) and not isnan(bb_upper) and
                isfinite(bb_lower) and isfinite(bb_upper) and
                bb_upper >= bb_lower)
    if bb_valid:
        bb_squeeze = tanh((bb_upper - bb_lower) / (price_d + 1e-8))
    else:
        bb_squeeze = 0.0
    out_features[feature_idx] = <float>bb_squeeze
    feature_idx += 1

    # 3. Trend strength via MACD divergence (replaces micro_dev) - measures trend strength
    # Positive = bullish trend, negative = bearish trend, magnitude = strength
    # Normalized by 1% of price (price_d * 0.01) similar to price_momentum for consistency
    # NaN handling: if MACD not ready (first ~26 bars), use 0.0 (no trend signal)
    if not isnan(macd) and not isnan(macd_signal):
        trend_strength = tanh((macd - macd_signal) / (price_d * 0.01 + 1e-8))
    else:
        trend_strength = 0.0
    out_features[feature_idx] = <float>trend_strength
    feature_idx += 1

    # --- Bollinger band context -------------------------------------------
    # Position within bands and band width - critical features for volatility-based strategies
    # NOTE: bb_valid is already computed above for bb_squeeze with FULL validation:
    #       - Validates both bb_lower AND bb_upper are finite (not NaN/Inf)
    #       - Ensures logical consistency: bb_upper >= bb_lower
    #       - If validation fails, both features default to safe values
    #
    # Defense-in-depth: Double-check bb_width calculation
    # Even with bb_valid check, explicitly verify bb_width is finite
    # This catches any remaining edge cases or calculation errors
    bb_width = bb_upper - bb_lower
    min_bb_width = price_d * 0.0001

    # Feature 1: Price position within Bollinger Bands
    # 0.5 = at the middle (default when bands not available)
    # 0.0 = at lower band, 1.0 = at upper band
    #
    # Defense-in-depth validation:
    # 1. Primary: bb_valid check (both bands finite and consistent)
    # 2. Secondary: bb_width > min_bb_width (avoid division by near-zero)
    # 3. Tertiary: _clipf handles any remaining NaN via NaN-to-zero conversion
    #
    # This triple-layer approach ensures bb_position CANNOT be NaN:
    # - Layer 1: Prevents invalid inputs from being used
    # - Layer 2: Prevents division by zero/near-zero
    # - Layer 3: Final safety net converts any NaN to 0.0
    if (not bb_valid) or bb_width <= min_bb_width:
        feature_val = 0.5
    else:
        # Additional safety: verify bb_width is finite before division
        if not isfinite(bb_width):
            feature_val = 0.5
        else:
            feature_val = _clipf((price_d - bb_lower) / (bb_width + 1e-9), -1.0, 2.0)
    out_features[feature_idx] = feature_val
    feature_idx += 1

    # Feature 2: Normalized band width (volatility measure)
    # 0.0 = bands not available or zero width
    #
    # Defense-in-depth validation:
    # 1. Primary: bb_valid check ensures inputs are finite
    # 2. Secondary: Verify bb_width is finite before normalization
    # 3. Tertiary: _clipf handles any remaining NaN
    if bb_valid:
        # Additional safety: verify bb_width is finite
        if not isfinite(bb_width):
            feature_val = 0.0
        else:
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
):
    """
    Python-callable wrapper that forwards to the ``nogil`` implementation.

    CRITICAL: Validates critical inputs before processing to prevent NaN/Inf propagation.
    This is the entry point for all observation vector construction and must enforce
    data integrity constraints.

    Validation performed:
    - price must be finite (not NaN/Inf) and positive (> 0)
    - prev_price must be finite (not NaN/Inf) and positive (> 0)
    - log_volume_norm must be finite (not NaN/Inf), can be 0 or negative
    - rel_volume must be finite (not NaN/Inf), can be 0 or negative
    - cash must be finite (not NaN/Inf), can be 0 or negative
    - units must be finite (not NaN/Inf), can be 0 or negative

    If validation fails, ValueError is raised with diagnostic information.
    This fail-fast approach prevents silent data corruption in the observation vector.

    Best practices implemented:
    - Price validation: Strict (must be > 0)
    - Volume metrics validation: Must be finite (can be 0 or negative)
    - Portfolio validation: Allows 0/negative but not NaN/Inf
    - Fail-fast approach catches data issues early
    - Clear error messages for debugging
    """
    # CRITICAL: Validate price inputs before any computation
    # This prevents NaN/Inf propagation through 15+ calculations downstream
    _validate_price(price, "price")
    _validate_price(prev_price, "prev_price")

    # CRITICAL: Validate volume metrics to prevent NaN propagation
    # These are computed from raw volume data and can be corrupted upstream
    # Without this check, corrupted values would be written directly to observation array
    _validate_volume_metric(log_volume_norm, "log_volume_norm")
    _validate_volume_metric(rel_volume, "rel_volume")

    # Validate portfolio state (cash and units)
    # These can be 0 or negative (valid states) but not NaN/Inf
    _validate_portfolio_value(cash, "cash")
    _validate_portfolio_value(units, "units")

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

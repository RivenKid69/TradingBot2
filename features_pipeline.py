# -*- coding: utf-8 -*-
"""
features_pipeline.py
------------------------------------------------------------------
Single source of truth for feature normalization both in training
and inference. Works over dict[str, pandas.DataFrame] where each DF
follows the canonical schema established in prepare_and_run.py.
- Adds standardized columns with suffix '_z' (z-score).
- Leaves original columns intact.
- Saves/loads stats to/from JSON for reproducibility.

FIXES:
- FIX (MEDIUM #3): Added outlier detection via winsorization for robust statistics.
- FIX (2025-11-21): Winsorization consistency - bounds from fit() applied in transform()
- FIX (2025-11-21): Close shift consistency - always shift to prevent look-ahead bias

Usage:
    pipe = FeaturePipeline()
    pipe.fit(all_dfs_dict)                 # during training
    pipe.save("models/preproc_pipeline.json")
    all_dfs_dict = pipe.transform_dict(all_dfs_dict, add_suffix="_z")

    pipe = FeaturePipeline.load("models/preproc_pipeline.json")  # inference
    all_dfs_dict = pipe.transform_dict(all_dfs_dict, add_suffix="_z")
"""
import os
import json
from datetime import UTC, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

CANON_PREFIX = [
    "timestamp","symbol","open","high","low","close","volume","quote_asset_volume",
    "number_of_trades","taker_buy_base_asset_volume","taker_buy_quote_asset_volume"
]

# Metadata columns that should NOT be shifted (not features)
METADATA_COLUMNS = {
    "timestamp", "symbol", "wf_role", "close_orig", "_close_shifted",
    # Add other metadata columns here if needed
}

# Target columns that should NOT be shifted (labels for prediction)
TARGET_COLUMNS = {
    "target", "target_return", "target_log_return", "target_volatility",
    "target_sharpe", "target_sortino", "target_max_drawdown",
    # Add other target columns here if needed
}


# ==============================================================================
# Data Leakage Prevention - Feature Column Identification
# ==============================================================================

def _columns_to_shift(df: pd.DataFrame) -> List[str]:
    """
    Identify all feature columns that must be shifted to prevent look-ahead bias.

    CRITICAL: All features derived from price/volume (technical indicators, normalized
    features, etc.) MUST be shifted by 1 period to ensure they represent information
    available BEFORE the current decision point.

    This prevents DATA LEAKAGE where:
    - Model sees indicators calculated on future prices
    - Features at time t contain information about close[t] or close[t+1]
    - Training learns spurious correlations with unavailable future data

    Returns columns to shift (excludes metadata and targets).

    Args:
        df: DataFrame to analyze

    Returns:
        List of column names to shift (all numeric features except metadata/targets)

    Examples:
        >>> df = pd.DataFrame({
        ...     "timestamp": [1, 2, 3],
        ...     "close": [100, 101, 102],
        ...     "rsi_14": [50, 55, 60],  # Technical indicator - MUST shift!
        ...     "target": [0.01, 0.02, 0.03],  # Target - DO NOT shift
        ... })
        >>> _columns_to_shift(df)
        ['close', 'rsi_14']  # Excludes timestamp (metadata) and target
    """
    cols: List[str] = []
    for c in df.columns:
        # Skip metadata columns (timestamp, symbol, etc.)
        if c in METADATA_COLUMNS:
            continue

        # Skip target columns (labels for prediction)
        if c in TARGET_COLUMNS:
            continue

        # Skip already-normalized columns (they will be recomputed from shifted features)
        if c.endswith("_z"):
            continue

        # Include all numeric columns (prices, volumes, indicators)
        if _is_numeric(df[c]):
            cols.append(c)

    return cols


# ==============================================================================
# Outlier Detection Utilities
# ==============================================================================

def winsorize_array(data: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    """
    Winsorize array: cap extreme values at specified percentiles.

    Winsorization is preferred over outlier removal in finance because it:
    - Preserves data points (no removal → no gaps in time series)
    - Bounds extreme values (flash crashes, fat-finger errors)
    - Maintains distribution shape in bulk (99% of data unchanged)

    Common use: Handle crypto market anomalies (flash wicks, exchange glitches).

    Args:
        data: Input array (may contain NaN)
        lower_percentile: Lower percentile bound (default: 1st percentile)
        upper_percentile: Upper percentile bound (default: 99th percentile)

    Returns:
        Winsorized array with same shape as input

    References:
        Dixon, W. J. (1960). "Simplified Estimation from Censored Normal Samples"
        Cont, R. (2001). "Empirical Properties of Asset Returns"

    Examples:
        >>> data = np.array([0.01, 0.02, 0.03, -0.50, 0.04])  # -50% flash crash
        >>> winsorize_array(data, 1, 99)
        array([0.01, 0.02, 0.03, 0.01, 0.04])  # Crash clipped to 1st percentile
    """
    if len(data) == 0:
        return data

    # Ignore NaN for percentile calculation
    lower_bound = np.nanpercentile(data, lower_percentile)
    upper_bound = np.nanpercentile(data, upper_percentile)

    return np.clip(data, lower_bound, upper_bound)

# Additional optional features we may standardize if present
OPTIONAL_NUMERIC = [
    "fear_greed_value","fear_greed_value_norm",
    "recent_event_high_96h","recent_event_medium_96h","time_since_last_event_hours",
]

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s)

def _columns_to_scale(df: pd.DataFrame) -> List[str]:
    # Key columns which are numeric but shouldn't be z-scored directly:
    # _close_shifted is a marker column for shift detection, not a feature
    exclude = {"timestamp", "_close_shifted"}  # 'symbol' non-numeric already excluded
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if c == "symbol":
            continue
        if c.endswith("_z"):  # already standardized
            continue
        if _is_numeric(df[c]):
            cols.append(c)
    return cols

class FeaturePipeline:
    """
    Feature preprocessing pipeline with asset-class awareness.

    This pipeline handles:
    - Feature normalization (z-scores)
    - Data leakage prevention (feature shifting)
    - Asset-class specific features (VIX, sector momentum for equities)
    - Winsorization for outlier handling

    FIX (2025-11-28): Added asset_class parameter to automatically apply
    stock-specific features when processing equity data. This prevents
    the common mistake of forgetting to call add_stock_features() manually.

    Reference: docs/PLATFORM_REFERENCE.md → Issue #5 "FeaturePipeline не asset_class aware"
    """

    # Valid asset classes
    VALID_ASSET_CLASSES = ("crypto", "equity", "forex", None)

    def __init__(
        self,
        stats: Optional[Dict[str, Dict[str, float]]] = None,
        metadata: Optional[Dict[str, object]] = None,
        enable_winsorization: bool = True,  # Winsorization mitigates outliers from flash crashes
        winsorize_percentiles: Tuple[float, float] = (1.0, 99.0),
        strict_idempotency: bool = True,  # Fail on repeated transform_df() to prevent double-shift
        preserve_close_orig: bool = True,  # REQUIRED for TradingEnv reward calculation
        asset_class: Optional[str] = None,  # FIX: Asset class awareness
        auto_stock_features: bool = True,  # FIX: Auto-apply stock features for equity
        auto_forex_features: bool = True,  # FIX: Auto-apply forex features for forex
    ):
        """Container for feature normalization statistics.

        Parameters
        ----------
        stats:
            Mapping from column name to ``{"mean": float, "std": float, "is_constant": bool}``.
        metadata:
            Additional information persisted alongside the statistics (for
            example the training window bounds or split version).
        enable_winsorization:
            If True, apply winsorization to mitigate outliers before computing statistics.
            Recommended for financial data to handle flash crashes, fat wicks, etc.
            Default: True
        winsorize_percentiles:
            Tuple of (lower_percentile, upper_percentile) for winsorization.
            Default: (1.0, 99.0) clips extreme 1% on each tail.
        strict_idempotency:
            If True (default), raise ValueError on repeated transform_df() to prevent
            double-shifting and data corruption. If False, make transform_df() idempotent
            (return already-transformed DataFrame unchanged).
            Default: True (strict mode for data integrity)
        preserve_close_orig:
            If True (default), create 'close_orig' column with unshifted close prices
            before shifting features.
            Default: True (REQUIRED for correct reward calculation in TradingEnv)

            CRITICAL: This MUST be True for training/backtest to ensure TradingEnv
            has access to original (unshifted) close prices for reward calculation.
            Without close_orig, the first bar of each episode will have reward=0
            because shifted close[0] = NaN → _last_reward_price = 0.0.

            Set to False ONLY if you're certain close_orig is not needed (e.g.,
            pure feature analysis without TradingEnv).
        asset_class:
            Asset class for data being processed. Options: "crypto", "equity", "forex", None.
            When set to "equity", automatically applies stock-specific features during
            transform (VIX regime, sector momentum, relative strength) unless disabled.
            Default: None (backward compatible - no automatic stock features)

            FIX (2025-11-28): Added to prevent common mistake of forgetting
            to call add_stock_features() for equity data.
        auto_stock_features:
            If True (default) and asset_class="equity", automatically add stock-specific
            features (gap analysis, earnings proximity, dividend proximity) during transform.
            Set to False to disable automatic feature addition (e.g., if features already added).
            Default: True

            Note: This only affects equity asset class. Crypto/forex are unchanged.
        auto_forex_features:
            If True (default) and asset_class="forex", automatically add forex-specific
            features (carry, session, spread regime, etc.) during transform.
            Set to False to disable automatic feature addition (e.g., if features already added).
            Default: True

            Note: This only affects forex asset class. Crypto/equity are unchanged.
        """
        # Validate asset_class
        if asset_class is not None and asset_class.lower() not in self.VALID_ASSET_CLASSES:
            raise ValueError(
                f"Invalid asset_class '{asset_class}'. "
                f"Valid options: {self.VALID_ASSET_CLASSES}"
            )

        # stats: {col: {"mean": float, "std": float, "is_constant": bool, "winsorize_bounds": tuple}}
        self.stats: Dict[str, Dict[str, float]] = stats or {}
        self.metadata: Dict[str, object] = metadata or {}
        self.enable_winsorization = enable_winsorization
        self.winsorize_percentiles = winsorize_percentiles
        self.strict_idempotency = strict_idempotency
        self.preserve_close_orig = preserve_close_orig
        self.asset_class = asset_class.lower() if asset_class else None
        self.auto_stock_features = auto_stock_features
        self.auto_forex_features = auto_forex_features

    def reset(self) -> None:
        """Drop previously computed statistics.

        Creating a fresh instance for each ``TradingEnv`` or clearing the
        state on episode reset avoids cross‑environment leakage of
        normalization parameters.
        """

        self.stats.clear()
        self.metadata.clear()

    def _add_stock_features_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to add stock-specific features.

        Called automatically by transform_df() when asset_class="equity"
        and auto_stock_features=True.

        Features added:
        - Gap analysis (gap_pct, gap_direction, etc.)
        - Earnings proximity (if data available)
        - Dividend proximity (if data available)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with stock features added

        Note:
            This is a lightweight version. For VIX regime, sector momentum,
            and relative strength, use add_stock_features() from stock_features.py
            which requires benchmark data (SPY, QQQ, VIX).
        """
        # Extract symbol from DataFrame if present
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns and len(df) > 0 else "UNKNOWN"

        try:
            # Use the existing add_stock_features function
            return add_stock_features(
                df,
                symbol=symbol,
                add_gaps=True,
                add_earnings=True,
                add_dividends=True,
            )
        except Exception as e:
            # Log warning but don't fail - return original DataFrame
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Failed to add stock features to {symbol}: {e}. "
                "Returning DataFrame without additional stock features."
            )
            return df

    def _add_forex_features_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to add forex-specific features.

        Called automatically by transform_df() when asset_class="forex"
        and auto_forex_features=True.

        Features added:
        - Carry differential (normalized interest rate spread)
        - Session indicators (sydney, tokyo, london, new_york, overlap)
        - Session liquidity multiplier
        - Spread regime (if spread column available)
        - Volatility features (5d, 20d, ratio)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with forex features added

        Note:
            For DXY relative strength, COT positioning, and economic calendar
            features, use add_forex_features_to_dataframe() from forex_features.py
            which requires benchmark data (DXY, rates, COT).
        """
        # Extract symbol from DataFrame if present
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns and len(df) > 0 else "EUR_USD"

        try:
            from forex_features import (
                add_forex_features_to_dataframe,
                ForexFeatureConfig,
            )

            # Use the forex features function with minimal config
            # (no benchmark data - just session and basic features)
            config = ForexFeatureConfig(
                session_check_enabled=True,
                vol_window_short=5,
                vol_window_long=20,
            )

            return add_forex_features_to_dataframe(
                df,
                symbol=symbol,
                dxy_df=None,  # Optional benchmark
                rates_df=None,  # Optional rates
                cot_df=None,  # Optional COT
                calendar_events=None,  # Optional calendar
                config=config,
            )
        except ImportError:
            # forex_features module not available - return as-is
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                "forex_features module not available, skipping forex feature addition"
            )
            return df
        except Exception as e:
            # Log warning but don't fail - return original DataFrame
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Failed to add forex features to {symbol}: {e}. "
                "Returning DataFrame without additional forex features."
            )
            return df

    def fit(
        self,
        dfs: Dict[str, pd.DataFrame],
        *,
        train_mask_column: Optional[str] = None,
        train_mask_values: Optional[Iterable] = None,
        train_start_ts: Optional[int] = None,
        train_end_ts: Optional[int] = None,
        timestamp_column: str = "timestamp",
        split_version: Optional[str] = None,
        train_intervals: Optional[Sequence[Tuple[Optional[int], Optional[int]]]] = None,
    ) -> "FeaturePipeline":
        """Fit normalization statistics from the provided dataframes.

        The caller may either provide a boolean/role mask identifying the
        training rows (for example a ``wf_role`` column equal to ``"train"``)
        or explicit ``train_start_ts``/``train_end_ts`` bounds. When both are
        supplied the intersection is used.
        """

        frames: List[pd.DataFrame] = []
        per_symbol_counts: Dict[str, int] = {}

        if train_mask_column is None and (train_start_ts is not None or train_end_ts is not None):
            # Ensure the timestamp column exists up-front when time bounds are used.
            for name, df in dfs.items():
                if timestamp_column not in df.columns:
                    raise KeyError(
                        f"DataFrame '{name}' is missing timestamp column '{timestamp_column}' required for training window filter."
                    )

        mask_values: Optional[Sequence] = None
        if train_mask_values is not None:
            mask_values = tuple(train_mask_values)

        for name, df in dfs.items():
            if df is None:
                continue
            cur = df
            if train_mask_column is not None:
                if train_mask_column not in cur.columns:
                    raise KeyError(
                        f"DataFrame '{name}' is missing training mask column '{train_mask_column}'."
                    )
                mask_series = cur[train_mask_column]
                if mask_values is None:
                    if pd.api.types.is_bool_dtype(mask_series):
                        mask = mask_series.astype(bool)
                    else:
                        mask = mask_series.astype(str).str.lower() == "train"
                else:
                    mask = mask_series.isin(mask_values)
                cur = cur.loc[mask]

            if train_start_ts is not None or train_end_ts is not None:
                ts = pd.to_numeric(cur[timestamp_column], errors="coerce")
                time_mask = pd.Series(True, index=cur.index)
                if train_start_ts is not None:
                    time_mask &= ts >= int(train_start_ts)
                if train_end_ts is not None:
                    time_mask &= ts <= int(train_end_ts)
                cur = cur.loc[time_mask]

            if not cur.empty:
                per_symbol_counts[name] = int(len(cur))
                frames.append(cur)

        if not frames:
            raise ValueError("No rows available to fit FeaturePipeline after applying training filters.")

        # Shift ALL feature columns to prevent data leakage (look-ahead bias)
        # Per-symbol shift to prevent cross-symbol contamination
        # Each frame corresponds to one symbol, so we shift each independently
        #
        # IMPORTANT: All technical indicators (RSI, MA, BB, etc.) MUST be shifted
        # together with price/volume data to ensure they represent information
        # available BEFORE the current decision point.
        #
        # Example of data leakage WITHOUT this fix:
        #   t=0: close=100, rsi_14=50 (calculated from close[t-13:t])
        #   t=1: close=105, rsi_14=60 (calculated from close[t-12:t+1])
        #   After shift: close[t]=100 (from t-1), rsi_14[t]=60 (from t)
        #   → Model sees RSI calculated on FUTURE prices (close[t+1])!
        #
        # Correct behavior WITH this fix:
        #   After shift: close[t]=100 (from t-1), rsi_14[t]=50 (from t-1)
        #   → Model sees only PAST information (consistent temporal alignment)
        shifted_frames: List[pd.DataFrame] = []
        for frame in frames:
            # Check if shift already applied (close_orig marker present)
            # If close_orig exists, data is already shifted - skip shifting
            if "close_orig" in frame.columns:
                shifted_frames.append(frame)
                continue

            frame_copy = frame.copy()

            # Identify all feature columns to shift (excludes metadata and targets)
            cols_to_shift = _columns_to_shift(frame_copy)

            if cols_to_shift:
                # ═══════════════════════════════════════════════════════════════════════
                # НЕ БАГ: ВСЕ FEATURES СДВИГАЮТСЯ ВМЕСТЕ (НЕТ TEMPORAL MISMATCH)
                # ═══════════════════════════════════════════════════════════════════════
                # SMA, Return, RSI, и все остальные features сдвигаются на 1 период
                # ОДНОВРЕМЕННО. Нет рассинхронизации между разными типами features.
                #
                # До shift: SMA[t] использует bars [t-lb:t], Return[t] = close[t]/close[t-1]
                # После shift: SMA[t] и Return[t] оба представляют данные на момент t-1
                # → Temporal alignment сохраняется!
                #
                # Reference: docs/NOT_BUGS_AND_FAQ.md → "НЕ БАГИ" → #24
                # ═══════════════════════════════════════════════════════════════════════
                for col in cols_to_shift:
                    frame_copy[col] = frame_copy[col].shift(1)

                # Add column-based marker for TradingEnv compatibility
                # This marker tells TradingEnv that data is already shifted
                frame_copy["_close_shifted"] = True

            shifted_frames.append(frame_copy)

        big = pd.concat(shifted_frames, axis=0, ignore_index=True)
        cols = _columns_to_scale(big)
        stats = {}
        all_nan_columns = []  # Track all-NaN columns for warning

        for c in cols:
            v = big[c].astype(float).to_numpy()

            # Detect all-NaN columns BEFORE winsorization
            # If column is entirely NaN, np.nanpercentile returns NaN bounds
            # which leads to silent NaN → 0.0 conversion (semantic ambiguity)
            is_all_nan = v.size > 0 and np.isnan(v).all()

            # Apply winsorization to mitigate outliers
            # This prevents flash crashes, fat-finger errors, and data anomalies
            # from contaminating normalization statistics (mean, std)
            # Store winsorization bounds for consistent application in transform
            winsorize_bounds = None
            if self.enable_winsorization and not is_all_nan:
                # Only compute bounds if column has at least one non-NaN value
                lower_bound = np.nanpercentile(v, self.winsorize_percentiles[0])
                upper_bound = np.nanpercentile(v, self.winsorize_percentiles[1])

                # Validate bounds are finite
                # If bounds are NaN (edge case: very few non-NaN values), skip winsorization
                if np.isfinite(lower_bound) and np.isfinite(upper_bound):
                    v_clean = np.clip(v, lower_bound, upper_bound)
                    winsorize_bounds = (float(lower_bound), float(upper_bound))
                else:
                    # Bounds invalid → treat as all-NaN
                    v_clean = v
                    is_all_nan = True
            else:
                v_clean = v

            m = float(np.nanmean(v_clean))
            # IMPROVEMENT: Use population std (ddof=0) for ML consistency
            # This aligns with ML frameworks: scikit-learn StandardScaler, PyTorch normalization
            # use ddof=0 (population std) for feature scaling, not ddof=1 (sample std).
            # For large datasets (n > 100), difference is negligible: sqrt(n/(n-1)) ≈ 1.005
            # For consistency with standard ML pipelines, we use ddof=0.
            # Reference: Pedregosa et al. (2011), "Scikit-learn: Machine Learning in Python"
            s = float(np.nanstd(v_clean, ddof=0))

            # Store zero variance indicator for proper handling
            # When s == 0 (constant feature), we mark it explicitly so transform can return zeros
            # instead of applying (value - mean) / 1.0 which may not be zero for NaN values
            is_constant = (not np.isfinite(s)) or (s == 0.0)
            if is_constant:
                s = 1.0  # avoid division by zero (will be handled specially in transform)
            if not np.isfinite(m):
                m = 0.0

            stats[c] = {"mean": m, "std": s, "is_constant": is_constant}

            # Mark all-NaN columns explicitly
            # This prevents silent NaN → 0.0 conversion and provides clear semantics
            if is_all_nan:
                stats[c]["is_all_nan"] = True
                all_nan_columns.append(c)
                # Do NOT store winsorize_bounds for all-NaN columns
                # This signals to transform_df() to skip winsorization
            elif winsorize_bounds is not None:
                # Store winsorization bounds for train/inference consistency
                stats[c]["winsorize_bounds"] = winsorize_bounds

        # Warn about all-NaN columns
        # These columns cannot provide useful information for training
        # and should be investigated for data quality issues
        if all_nan_columns:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Found {len(all_nan_columns)} column(s) with ALL NaN values: {all_nan_columns}. "
                f"These columns will be normalized to NaN (not zeros) to preserve semantic meaning. "
                f"Consider: (1) Checking data quality, (2) Imputing missing values, or (3) Removing these features."
            )

        intervals_payload: Optional[List[Dict[str, Optional[int]]]] = None
        if train_intervals:
            intervals_payload = [
                {
                    "start_ts": int(start) if start is not None else None,
                    "end_ts": int(end) if end is not None else None,
                }
                for start, end in train_intervals
            ]

        metadata: Dict[str, object] = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "filters": {
                "train_mask_column": train_mask_column,
                "train_mask_values": list(mask_values) if mask_values is not None else None,
                "train_start_ts": int(train_start_ts) if train_start_ts is not None else None,
                "train_end_ts": int(train_end_ts) if train_end_ts is not None else None,
                "timestamp_column": timestamp_column,
                "train_intervals": intervals_payload,
            },
            "train_rows_by_symbol": per_symbol_counts,
            "train_rows_total": int(sum(per_symbol_counts.values())),
        }
        if split_version is not None:
            metadata["split_version"] = str(split_version)

        self.stats = stats
        self.metadata = metadata
        return self

    def transform_df(self, df: pd.DataFrame, add_suffix: str = "_z") -> pd.DataFrame:
        """Transform DataFrame by applying normalization statistics.

        IMPORTANT: This method should be called ONLY ONCE per DataFrame.
        Repeated calls will cause double-shifting of 'close' column, leading to
        data misalignment and look-ahead bias accumulation.

        To apply transform multiple times:
        1. Preserve original close: df["close_orig"] = df["close"].copy()
        2. Or use fresh copy from original data source

        Args:
            df: DataFrame to transform
            add_suffix: Suffix for normalized columns (default: "_z")

        Returns:
            Transformed DataFrame with normalized columns

        Raises:
            ValueError: If pipeline is empty (call fit() or load() first)
            ValueError: If repeated application detected (strict_idempotency=True, default)
        """
        if not self.stats:
            raise ValueError("FeaturePipeline is empty; call fit() or load().")

        # Detect repeated transform_df() application
        # Check for marker in DataFrame attrs (metadata introduced in pandas 1.0)
        # This prevents silent data corruption from double-shifting
        if hasattr(df, 'attrs') and df.attrs.get('_feature_pipeline_transformed', False):
            if self.strict_idempotency:
                # STRICT MODE (default): Fail immediately to prevent data corruption
                raise ValueError(
                    "transform_df() called on already-transformed DataFrame! "
                    "This would cause DOUBLE SHIFT of 'close' column, leading to:\n"
                    "  1. Data misalignment (close lag = 2 instead of 1)\n"
                    "  2. Accumulated look-ahead bias (features based on t+1 close)\n"
                    "  3. Scale mismatch (z-scores applied twice with different stats)\n\n"
                    "To fix:\n"
                    "  - Option A: Preserve 'close_orig' before first transform\n"
                    "  - Option B: Use fresh copy from original data source\n"
                    "  - Option C: Set strict_idempotency=False for idempotent behavior\n\n"
                    "This error prevents silent data corruption in training loop."
                )
            else:
                # IDEMPOTENT MODE: Return already-transformed DataFrame unchanged
                import warnings
                warnings.warn(
                    "transform_df() called on already-transformed DataFrame. "
                    "Returning without changes (idempotent mode). "
                    "Set strict_idempotency=True to fail immediately and catch errors.",
                    RuntimeWarning,
                    stacklevel=2
                )
                return df  # Return original (already transformed) without changes

        out = df.copy()

        # ═══════════════════════════════════════════════════════════════════════
        # FIX (2025-11-28): AUTO STOCK FEATURES FOR EQUITY
        # ═══════════════════════════════════════════════════════════════════════
        # When asset_class="equity" and auto_stock_features=True, automatically
        # add stock-specific features (gaps, earnings proximity, dividends).
        # This prevents the common mistake of forgetting to call add_stock_features().
        #
        # Reference: docs/PLATFORM_REFERENCE.md → Issue #5 "FeaturePipeline не asset_class aware"
        # ═══════════════════════════════════════════════════════════════════════
        if self.asset_class == "equity" and self.auto_stock_features:
            # Only add if not already present (check for gap_pct as marker)
            if "gap_pct" not in out.columns:
                out = self._add_stock_features_internal(out)

        # ═══════════════════════════════════════════════════════════════════════
        # FIX (2025-11-30): AUTO FOREX FEATURES FOR FOREX
        # ═══════════════════════════════════════════════════════════════════════
        # When asset_class="forex" and auto_forex_features=True, automatically
        # add forex-specific features (carry, session indicators, spread regime).
        # ═══════════════════════════════════════════════════════════════════════
        if self.asset_class == "forex" and self.auto_forex_features:
            # Only add if not already present (check for carry_diff as marker)
            if "carry_diff" not in out.columns:
                out = self._add_forex_features_internal(out)

        # Shift ALL feature columns to prevent data leakage (look-ahead bias)
        # This must match the shifting logic in fit() for consistency
        #
        # Rationale:
        # 1. Statistics were computed on shifted features during fit()
        # 2. Transform must operate on shifted features for scale consistency
        # 3. All features must be temporally aligned to prevent look-ahead bias
        #
        # Example WITHOUT this fix (data leakage):
        #   Training: Stats computed on shifted features (correct)
        #   Inference: Only close shifted, indicators NOT shifted (WRONG!)
        #   → Model sees indicators based on CURRENT prices (future info!)
        #
        # Example WITH this fix (no leakage):
        #   Training: Stats computed on shifted features
        #   Inference: ALL features shifted (consistent temporal alignment)
        #   → Model sees only PAST information (correct behavior)

        # Check if shift already applied (close_orig marker present)
        # If close_orig exists, data is already shifted - skip shifting
        if "close_orig" not in out.columns:
            # ENHANCEMENT (2025-11-25): Optionally preserve original close price
            # before shifting for post-training analysis and debugging
            if self.preserve_close_orig and "close" in out.columns:
                out["close_orig"] = out["close"].copy()

            # Identify all feature columns to shift (excludes metadata and targets)
            cols_to_shift = _columns_to_shift(out)

            if cols_to_shift:
                if "symbol" in out.columns:
                    # Per-symbol shift to prevent cross-symbol contamination
                    # Use groupby to shift each symbol's features independently
                    for col in cols_to_shift:
                        out[col] = out.groupby("symbol", group_keys=False)[col].shift(1)
                else:
                    # Single symbol case - standard shift
                    for col in cols_to_shift:
                        out[col] = out[col].shift(1)

                # Add column-based marker for TradingEnv compatibility
                # TradingEnv checks for "_close_shifted" column, not attrs
                # This prevents double-shifting when data flows: features_pipeline → TradingEnv
                out["_close_shifted"] = True

        for c, ms in self.stats.items():
            if c not in out.columns:
                # silently skip columns missing in this DF
                continue
            v = out[c].astype(float).to_numpy()

            # Handle all-NaN columns explicitly
            # If column was all-NaN during training, preserve NaN semantics
            # (do NOT convert to zeros, as zeros have different meaning)
            if ms.get("is_all_nan", False):
                # Column was entirely NaN during training
                # Keep as NaN to preserve semantic distinction from zero values
                # Model should handle NaN appropriately (skip, impute, or use validity flags)
                z = np.full_like(v, np.nan, dtype=float)
                out[c + add_suffix] = z
                continue  # Skip winsorization and standardization

            # ═══════════════════════════════════════════════════════════════════════
            # НЕ БАГ: WINSORIZATION PREVENTS UNBOUNDED Z-SCORES
            # ═══════════════════════════════════════════════════════════════════════
            # Winsorization bounds из training применяются ДО вычисления z-score.
            # Это предотвращает экстремальные z-scores (50+ sigma) при flash crashes.
            #
            # Пример: training bounds [95, 105], mean=100, std=5
            #   - Flash crash: raw_price = 70 → clipped to 95
            #   - z = (95 - 100) / 5 = -1.0 (reasonable, not -6.0!)
            #
            # Best practice references:
            # - Huber (1981) "Robust Statistics": Apply same robust procedure on train/test
            # - Scikit-learn RobustScaler: Clips test data using train quantiles
            # - De Prado (2018) "Advances in Financial ML": Consistent winsorization
            #
            # Reference: docs/NOT_BUGS_AND_FAQ.md → "НЕ БАГИ" → #25
            # ═══════════════════════════════════════════════════════════════════════
            if "winsorize_bounds" in ms:
                lower, upper = ms["winsorize_bounds"]
                v = np.clip(v, lower, upper)

            # Handle constant features explicitly
            # For zero-variance features, return zeros instead of (value - mean) / 1.0
            # This prevents NaN propagation and ensures semantic correctness
            if ms.get("is_constant", False):
                # Feature had zero variance during training → always normalize to 0
                z = np.zeros_like(v, dtype=float)
            else:
                z = (v - ms["mean"]) / ms["std"]
            out[c + add_suffix] = z

        # Mark DataFrame as transformed to detect repeated applications
        # Use DataFrame.attrs (metadata dict introduced in pandas 1.0)
        # This marker survives copy() operations and helps prevent silent data corruption
        if hasattr(out, 'attrs'):
            out.attrs['_feature_pipeline_transformed'] = True

        return out

    def transform_dict(self, dfs: Dict[str, pd.DataFrame], add_suffix: str = "_z") -> Dict[str, pd.DataFrame]:
        return {k: self.transform_df(v, add_suffix=add_suffix) for k, v in dfs.items()}

    def get_metadata(self) -> Dict[str, object]:
        """Return metadata captured during :meth:`fit`."""

        return dict(self.metadata)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Save configuration flags for reproducibility
        payload = {
            "stats": self.stats,
            "metadata": self.metadata,
            "config": {
                "enable_winsorization": self.enable_winsorization,
                "winsorize_percentiles": self.winsorize_percentiles,
                "strict_idempotency": self.strict_idempotency,
                "preserve_close_orig": self.preserve_close_orig,
                # FIX (2025-11-28): Persist asset_class for reproducibility
                "asset_class": self.asset_class,
                "auto_stock_features": self.auto_stock_features,
                "auto_forex_features": self.auto_forex_features,
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "stats" in payload:
            stats = payload.get("stats", {})
            metadata = payload.get("metadata", {})
            # Load configuration flags if available
            config = payload.get("config", {})
            # FIX (2025-11-25): Changed preserve_close_orig default from False to True
            # to match constructor default. Legacy artifacts without this config key
            # will now correctly enable close_orig preservation by default.
            # This fixes first-bar reward=0 bug when loading old preproc_pipeline.json.
            #
            # FIX (2025-11-28): Load asset_class and auto_stock_features.
            # Default to None for backward compatibility with legacy artifacts.
            return cls(
                stats=stats,
                metadata=metadata,
                enable_winsorization=config.get("enable_winsorization", True),
                winsorize_percentiles=tuple(config.get("winsorize_percentiles", [1.0, 99.0])),
                strict_idempotency=config.get("strict_idempotency", True),
                preserve_close_orig=config.get("preserve_close_orig", True),
                asset_class=config.get("asset_class", None),
                auto_stock_features=config.get("auto_stock_features", True),
                auto_forex_features=config.get("auto_forex_features", True),
            )
        else:
            # Backwards compatibility for legacy artifacts containing only stats.
            stats = payload
            metadata = {}
            return cls(stats=stats, metadata=metadata)


# ==============================================================================
# Gap Analysis Features (Phase 7 - Stock Support)
# ==============================================================================

def compute_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-market gap analysis features.

    Gap analysis is critical for equity trading:
    - Overnight news creates opening gaps
    - Gap fill probability varies by gap size
    - Large gaps often indicate significant events

    Features computed:
    - gap_pct: Gap percentage from previous close to open
    - gap_abs: Absolute gap (log-scaled for normalization)
    - gap_direction: Direction (1 = up, -1 = down, 0 = flat)
    - gap_magnitude: Categorized size (0-3)
    - gap_filled: Whether gap was filled intrabar
    - gap_fill_ratio: How much of gap was filled (0-1+)

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with gap features added

    Note:
        For crypto, gaps are less common (24/7 trading) but can occur
        on weekends or during exchange maintenance.
    """
    df = df.copy()

    if "close" not in df.columns or "open" not in df.columns:
        # Missing required columns, return as-is
        return df

    # Compute previous close
    prev_close = df["close"].shift(1)

    # Gap percentage: (open - prev_close) / prev_close * 100
    # Handle division by zero gracefully
    gap_pct = np.where(
        prev_close > 0,
        (df["open"] - prev_close) / prev_close * 100,
        0.0
    )
    df["gap_pct"] = gap_pct

    # Absolute gap (log-scaled with sign preserved)
    # log1p(abs(x)) * sign(x) provides bounded feature for ML
    df["gap_abs"] = np.log1p(np.abs(gap_pct)) * np.sign(gap_pct)

    # Gap direction: 1 for up, -1 for down, 0 for flat
    df["gap_direction"] = np.sign(gap_pct)

    # Gap magnitude category:
    # 0 = No gap (< 0.5%)
    # 1 = Small (0.5% - 2%)
    # 2 = Medium (2% - 5%)
    # 3 = Large (> 5%)
    abs_gap = np.abs(gap_pct)
    df["gap_magnitude"] = np.select(
        [abs_gap < 0.5, abs_gap < 2.0, abs_gap < 5.0],
        [0, 1, 2],
        default=3
    )

    # Gap filled detection:
    # Up gap is filled if low goes below previous close
    # Down gap is filled if high goes above previous close
    df["gap_filled"] = 0

    if "high" in df.columns and "low" in df.columns:
        up_gap_mask = gap_pct > 0
        down_gap_mask = gap_pct < 0

        # Up gap filled when low reaches prev_close
        df.loc[up_gap_mask & (df["low"] <= prev_close), "gap_filled"] = 1

        # Down gap filled when high reaches prev_close
        df.loc[down_gap_mask & (df["high"] >= prev_close), "gap_filled"] = 1

        # Gap fill ratio: how much of the gap was retraced
        # For up gap: (open - low) / (open - prev_close)
        # For down gap: (high - open) / (prev_close - open)
        gap_fill_ratio = np.zeros(len(df))

        # Up gap fill ratio
        up_gap_size = df["open"] - prev_close
        up_gap_fill = df["open"] - df["low"]
        valid_up = up_gap_mask & (up_gap_size > 0)
        gap_fill_ratio = np.where(
            valid_up,
            up_gap_fill / up_gap_size,
            gap_fill_ratio
        )

        # Down gap fill ratio
        down_gap_size = prev_close - df["open"]
        down_gap_fill = df["high"] - df["open"]
        valid_down = down_gap_mask & (down_gap_size > 0)
        gap_fill_ratio = np.where(
            valid_down,
            down_gap_fill / down_gap_size,
            gap_fill_ratio
        )

        df["gap_fill_ratio"] = np.clip(gap_fill_ratio, 0, 2)  # Cap at 200%

    # Handle NaN in first row (no previous close)
    df["gap_pct"] = df["gap_pct"].fillna(0)
    df["gap_abs"] = df["gap_abs"].fillna(0)
    df["gap_direction"] = df["gap_direction"].fillna(0)
    df["gap_magnitude"] = df["gap_magnitude"].fillna(0)
    df["gap_filled"] = df["gap_filled"].fillna(0)
    if "gap_fill_ratio" in df.columns:
        df["gap_fill_ratio"] = df["gap_fill_ratio"].fillna(0)

    return df


def compute_earnings_proximity_features(
    df: pd.DataFrame,
    earnings_dates: Sequence[str],
) -> pd.DataFrame:
    """
    Compute features related to earnings proximity.

    Important for:
    - Volatility modeling (implied vol rises before earnings)
    - Position sizing (reduce risk around earnings)
    - Regime detection (earnings season patterns)

    Features:
    - days_to_earnings: Days until next earnings (-90 to +90)
    - earnings_week: Binary flag for week of earnings
    - post_earnings: First 3 days after earnings

    Args:
        df: DataFrame with timestamp column
        earnings_dates: List of earnings dates (ISO format)

    Returns:
        DataFrame with earnings features
    """
    df = df.copy()

    if "timestamp" not in df.columns or not earnings_dates:
        df["days_to_earnings"] = 90
        df["earnings_week"] = 0
        df["post_earnings"] = 0
        return df

    from datetime import datetime, timedelta

    # Convert timestamps to dates
    df["_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")

    # Sort earnings dates
    sorted_earnings = sorted(earnings_dates)

    days_to_earnings = []
    earnings_week = []
    post_earnings = []

    for date_str in df["_date"]:
        current_date = datetime.fromisoformat(date_str)

        # Find next and previous earnings
        next_earnings = None
        prev_earnings = None

        for e_date in sorted_earnings:
            e_dt = datetime.fromisoformat(e_date)
            if e_dt > current_date:
                next_earnings = e_dt
                break
            prev_earnings = e_dt

        # Days to next earnings
        if next_earnings:
            days = (next_earnings - current_date).days
            days_to_earnings.append(min(days, 90))  # Cap at 90
        else:
            days_to_earnings.append(90)

        # Earnings week (within 3 days before)
        is_earnings_week = False
        if next_earnings:
            days = (next_earnings - current_date).days
            is_earnings_week = 0 <= days <= 3
        earnings_week.append(1 if is_earnings_week else 0)

        # Post earnings (within 3 days after)
        is_post_earnings = False
        if prev_earnings:
            days = (current_date - prev_earnings).days
            is_post_earnings = 0 <= days <= 3
        post_earnings.append(1 if is_post_earnings else 0)

    df["days_to_earnings"] = days_to_earnings
    df["earnings_week"] = earnings_week
    df["post_earnings"] = post_earnings

    # Clean up temp column
    df.drop(columns=["_date"], inplace=True)

    return df


def compute_dividend_proximity_features(
    df: pd.DataFrame,
    dividend_dates: Sequence[str],
) -> pd.DataFrame:
    """
    Compute features related to dividend proximity.

    Important for:
    - Price adjustment timing
    - Dividend capture strategies
    - Ex-date behavior modeling

    Features:
    - days_to_ex_div: Days until next ex-dividend date
    - ex_div_week: Binary flag for week of ex-div

    Args:
        df: DataFrame with timestamp column
        dividend_dates: List of ex-dividend dates (ISO format)

    Returns:
        DataFrame with dividend features
    """
    df = df.copy()

    if "timestamp" not in df.columns or not dividend_dates:
        df["days_to_ex_div"] = 90
        df["ex_div_week"] = 0
        return df

    from datetime import datetime

    # Convert timestamps to dates
    df["_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")

    # Sort dividend dates
    sorted_divs = sorted(dividend_dates)

    days_to_ex_div = []
    ex_div_week = []

    for date_str in df["_date"]:
        current_date = datetime.fromisoformat(date_str)

        # Find next ex-div date
        next_ex_div = None
        for d_date in sorted_divs:
            d_dt = datetime.fromisoformat(d_date)
            if d_dt > current_date:
                next_ex_div = d_dt
                break

        # Days to next ex-div
        if next_ex_div:
            days = (next_ex_div - current_date).days
            days_to_ex_div.append(min(days, 90))
            ex_div_week.append(1 if days <= 5 else 0)
        else:
            days_to_ex_div.append(90)
            ex_div_week.append(0)

    df["days_to_ex_div"] = days_to_ex_div
    df["ex_div_week"] = ex_div_week

    # Clean up
    df.drop(columns=["_date"], inplace=True)

    return df


def add_stock_features(
    df: pd.DataFrame,
    symbol: str,
    earnings_dates: Optional[Sequence[str]] = None,
    dividend_dates: Optional[Sequence[str]] = None,
    add_gaps: bool = True,
    add_earnings: bool = True,
    add_dividends: bool = True,
) -> pd.DataFrame:
    """
    Add all stock-specific features to a DataFrame.

    This is a convenience function that combines:
    - Gap analysis features
    - Earnings proximity features
    - Dividend proximity features

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol (for logging)
        earnings_dates: List of earnings dates (optional)
        dividend_dates: List of ex-dividend dates (optional)
        add_gaps: Whether to compute gap features
        add_earnings: Whether to compute earnings features
        add_dividends: Whether to compute dividend features

    Returns:
        DataFrame with all stock features added
    """
    if add_gaps:
        df = compute_gap_features(df)

    if add_earnings:
        if earnings_dates is None:
            # Try to fetch from service
            try:
                from services.corporate_actions import get_service
                service = get_service()
                earnings = service.get_earnings(symbol)
                earnings_dates = [e["report_date"] for e in earnings if e.get("report_date")]
            except Exception:
                earnings_dates = []

        df = compute_earnings_proximity_features(df, earnings_dates or [])

    if add_dividends:
        if dividend_dates is None:
            # Try to fetch from service
            try:
                from services.corporate_actions import get_service
                service = get_service()
                divs = service.get_dividends(symbol)
                dividend_dates = [d["ex_date"] for d in divs if d.get("ex_date")]
            except Exception:
                dividend_dates = []

        df = compute_dividend_proximity_features(df, dividend_dates or [])

    return df

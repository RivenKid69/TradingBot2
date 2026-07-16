"""Self-contained EDA / Data Quality profiler for quant datasets.

This module is intentionally standalone: it imports only the standard library,
numpy, and pandas. scipy is used opportunistically for skew/kurtosis but a
pure-numpy fallback is provided so the module works without it.

Public API:
    profile_dataset(df, time_col="ts_ms", symbol_col="symbol", n_bins=30) -> dict

CLI:
    python research/eda_profiler.py --in <parquet/csv> --out <json> \
        [--time-col ts_ms] [--symbol-col symbol]
    python research/eda_profiler.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# scipy is optional. If available we use it for skew/kurtosis; otherwise we
# fall back to numpy-based implementations.
try:  # pragma: no cover - exercised implicitly depending on environment
    from scipy import stats as _scipy_stats  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_stats = None
    _HAS_SCIPY = False


# Maximum number of numeric feature columns to build histograms for.
_MAX_DISTRIBUTION_COLS = 12


def _json_safe_float(value: Any) -> Optional[float]:
    """Convert a value to a JSON-safe float (NaN/inf -> None)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _json_safe_int(value: Any) -> Optional[int]:
    """Convert a value to a JSON-safe int (None on failure)."""
    try:
        if value is None:
            return None
        f = float(value)
        if not math.isfinite(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


def _skewness(arr: np.ndarray) -> Optional[float]:
    """Sample skewness with numpy fallback when scipy is unavailable."""
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return None
    if _HAS_SCIPY:
        try:
            return _json_safe_float(_scipy_stats.skew(arr, bias=False))
        except Exception:
            pass
    n = arr.size
    mean = arr.mean()
    std = arr.std(ddof=0)
    if std == 0 or not math.isfinite(std):
        return None
    m3 = np.mean((arr - mean) ** 3)
    g1 = m3 / (std ** 3)
    # Adjust to the unbiased (Fisher-Pearson) estimator to match scipy bias=False
    adj = math.sqrt(n * (n - 1)) / (n - 2)
    return _json_safe_float(adj * g1)


def _kurtosis(arr: np.ndarray) -> Optional[float]:
    """Excess kurtosis (Fisher) with numpy fallback when scipy unavailable."""
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return None
    if _HAS_SCIPY:
        try:
            return _json_safe_float(_scipy_stats.kurtosis(arr, fisher=True, bias=False))
        except Exception:
            pass
    n = arr.size
    mean = arr.mean()
    std = arr.std(ddof=0)
    if std == 0 or not math.isfinite(std):
        return None
    m4 = np.mean((arr - mean) ** 4)
    g2 = m4 / (std ** 4) - 3.0
    # Unbiased estimator matching scipy bias=False
    num = (n - 1) / ((n - 2) * (n - 3))
    adj = num * ((n + 1) * g2 + 6)
    return _json_safe_float(adj)


def _profile_numeric_column(name: str, series: pd.Series) -> Dict[str, Any]:
    """Build the descriptive stats block for a numeric column."""
    n_total = len(series)
    missing_pct = (
        100.0 * float(series.isna().sum()) / n_total if n_total else 0.0
    )
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]

    col: Dict[str, Any] = {
        "name": name,
        "dtype": str(series.dtype),
        "missing_pct": _json_safe_float(missing_pct),
        "n_unique": int(series.nunique(dropna=True)),
    }

    if finite.size == 0:
        for key in (
            "mean",
            "std",
            "min",
            "p25",
            "p50",
            "p75",
            "max",
            "skew",
            "kurtosis",
        ):
            col[key] = None
        col["n_outliers_iqr"] = 0
        col["n_outliers_z3"] = 0
        return col

    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    q1 = float(np.percentile(finite, 25))
    q2 = float(np.percentile(finite, 50))
    q3 = float(np.percentile(finite, 75))

    col["mean"] = _json_safe_float(mean)
    col["std"] = _json_safe_float(std)
    col["min"] = _json_safe_float(finite.min())
    col["p25"] = _json_safe_float(q1)
    col["p50"] = _json_safe_float(q2)
    col["p75"] = _json_safe_float(q3)
    col["max"] = _json_safe_float(finite.max())
    col["skew"] = _skewness(finite)
    col["kurtosis"] = _kurtosis(finite)

    # IQR-based outliers
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    col["n_outliers_iqr"] = int(np.sum((finite < lower) | (finite > upper)))

    # Z-score (>3) outliers
    if std > 0 and math.isfinite(std):
        z = (finite - mean) / std
        col["n_outliers_z3"] = int(np.sum(np.abs(z) > 3.0))
    else:
        col["n_outliers_z3"] = 0

    return col


def _profile_non_numeric_column(name: str, series: pd.Series) -> Dict[str, Any]:
    """Build the minimal block for a non-numeric column."""
    n_total = len(series)
    missing_pct = (
        100.0 * float(series.isna().sum()) / n_total if n_total else 0.0
    )
    return {
        "name": name,
        "dtype": str(series.dtype),
        "missing_pct": _json_safe_float(missing_pct),
        "n_unique": int(series.nunique(dropna=True)),
    }


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _per_symbol_block(
    sub: pd.DataFrame, symbol: str, time_col: Optional[str]
) -> Dict[str, Any]:
    """Compute interval/coverage/gap stats for one symbol partition."""
    block: Dict[str, Any] = {
        "symbol": symbol,
        "n_rows": int(len(sub)),
        "ts_start": None,
        "ts_end": None,
        "median_interval_ms": None,
        "n_gaps": 0,
        "max_gap_ms": None,
        "expected_rows": None,
        "coverage_pct": None,
    }

    if time_col is None or time_col not in sub.columns:
        return block

    ts = pd.to_numeric(sub[time_col], errors="coerce").to_numpy(dtype="float64")
    ts = ts[np.isfinite(ts)]
    ts = np.sort(ts)
    if ts.size == 0:
        return block

    block["ts_start"] = _json_safe_float(ts.min())
    block["ts_end"] = _json_safe_float(ts.max())

    if ts.size < 2:
        return block

    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]  # ignore duplicate timestamps for interval estimate
    if diffs.size == 0:
        return block

    median_interval = float(np.median(diffs))
    block["median_interval_ms"] = _json_safe_float(median_interval)

    if median_interval > 0:
        gap_threshold = 1.5 * median_interval
        block["n_gaps"] = int(np.sum(diffs > gap_threshold))
        block["max_gap_ms"] = _json_safe_float(diffs.max())

        span = float(ts.max() - ts.min())
        expected = span / median_interval + 1.0
        block["expected_rows"] = _json_safe_float(expected)
        if expected > 0:
            block["coverage_pct"] = _json_safe_float(
                100.0 * float(len(sub)) / expected
            )

    return block


def _find_ohlc_columns(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """Locate open/high/low/close columns case-insensitively."""
    lookup = {c.lower(): c for c in df.columns}
    needed = ("open", "high", "low", "close")
    if all(n in lookup for n in needed):
        return {n: lookup[n] for n in needed}
    return None


def _ohlc_violations(df: pd.DataFrame) -> Optional[int]:
    """Count rows where the OHLC bar is internally inconsistent.

    A row is valid when:
        high >= max(open, close, low) AND low <= min(open, close, high)
    A violation is any row that does NOT satisfy both conditions.
    """
    cols = _find_ohlc_columns(df)
    if cols is None:
        return None

    o = pd.to_numeric(df[cols["open"]], errors="coerce").to_numpy(dtype="float64")
    h = pd.to_numeric(df[cols["high"]], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(df[cols["low"]], errors="coerce").to_numpy(dtype="float64")
    c = pd.to_numeric(df[cols["close"]], errors="coerce").to_numpy(dtype="float64")

    valid_present = (
        np.isfinite(o) & np.isfinite(h) & np.isfinite(low) & np.isfinite(c)
    )

    high_ok = h >= np.maximum.reduce([o, c, low])
    low_ok = low <= np.minimum.reduce([o, c, h])
    is_valid = high_ok & low_ok

    # Only evaluate rows where all four values are present.
    violations = valid_present & ~is_valid
    return int(np.sum(violations))


def _duplicate_timestamps(
    df: pd.DataFrame, time_col: Optional[str], symbol_col: Optional[str]
) -> int:
    """Count duplicate (symbol, time) pairs.

    The count is the number of rows that are duplicates (i.e. rows beyond the
    first occurrence of each (symbol, time) key).
    """
    if time_col is None or time_col not in df.columns:
        return 0

    keys: List[str] = []
    if symbol_col is not None and symbol_col in df.columns:
        keys.append(symbol_col)
    keys.append(time_col)

    return int(df.duplicated(subset=keys, keep="first").sum())


def _distributions(
    df: pd.DataFrame, numeric_cols: List[str], n_bins: int
) -> Dict[str, Any]:
    """Build histograms for up to _MAX_DISTRIBUTION_COLS numeric columns."""
    out: Dict[str, Any] = {}
    for name in numeric_cols[:_MAX_DISTRIBUTION_COLS]:
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            out[name] = {"bin_edges": [], "counts": []}
            continue
        try:
            counts, edges = np.histogram(finite, bins=n_bins)
        except Exception:
            out[name] = {"bin_edges": [], "counts": []}
            continue
        out[name] = {
            "bin_edges": [_json_safe_float(e) for e in edges.tolist()],
            "counts": [int(c) for c in counts.tolist()],
        }
    return out


def profile_dataset(
    df: pd.DataFrame,
    time_col: str = "ts_ms",
    symbol_col: str = "symbol",
    n_bins: int = 30,
) -> Dict[str, Any]:
    """Profile a dataset and return a JSON-safe report dict.

    See the module docstring for the report schema.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    has_symbol = symbol_col in df.columns
    has_time = time_col in df.columns
    effective_time = time_col if has_time else None

    report: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
    }

    # ---- Column profiles -------------------------------------------------
    columns: List[Dict[str, Any]] = []
    numeric_feature_cols: List[str] = []
    excluded = {c for c in (time_col, symbol_col) if c in df.columns}
    for name in df.columns:
        series = df[name]
        if _is_numeric(series):
            columns.append(_profile_numeric_column(name, series))
            if name not in excluded:
                numeric_feature_cols.append(name)
        else:
            columns.append(_profile_non_numeric_column(name, series))
    report["columns"] = columns

    # ---- Per-symbol coverage --------------------------------------------
    per_symbol: List[Dict[str, Any]] = []
    if has_symbol:
        for symbol, sub in df.groupby(symbol_col, sort=True):
            per_symbol.append(
                _per_symbol_block(sub, str(symbol), effective_time)
            )
    else:
        per_symbol.append(_per_symbol_block(df, "ALL", effective_time))
    report["per_symbol"] = per_symbol

    # ---- OHLC violations -------------------------------------------------
    report["ohlc_violations"] = _ohlc_violations(df)

    # ---- Duplicate timestamps -------------------------------------------
    report["duplicate_timestamps"] = _duplicate_timestamps(
        df, effective_time, symbol_col if has_symbol else None
    )

    # ---- Distributions ---------------------------------------------------
    report["distributions"] = _distributions(df, numeric_feature_cols, n_bins)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported input extension '{ext}'. Use .parquet or .csv."
    )


def _run_cli(args: argparse.Namespace) -> int:
    df = _read_dataframe(args.input)
    report = profile_dataset(
        df,
        time_col=args.time_col,
        symbol_col=args.symbol_col,
        n_bins=args.n_bins,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)

    n_symbols = len(report["per_symbol"])
    total_gaps = sum(int(s.get("n_gaps") or 0) for s in report["per_symbol"])
    print(
        f"EDA report: {report['n_rows']} rows x {report['n_cols']} cols, "
        f"{n_symbols} symbol(s), {total_gaps} gap(s), "
        f"dup_ts={report['duplicate_timestamps']}, "
        f"ohlc_violations={report['ohlc_violations']} -> {args.out}"
    )
    return 0


def _build_synthetic_df() -> pd.DataFrame:
    """Build a synthetic dataset for the self-test."""
    rng = np.random.default_rng(42)
    step = 4 * 60 * 60 * 1000  # 4h in ms
    base = 1_700_000_000_000

    frames = []
    for sym in ("AAA", "BBB"):
        n = 50
        ts = base + np.arange(n) * step
        # Inject a deliberate gap: remove a chunk in the middle of AAA.
        if sym == "AAA":
            keep = np.ones(n, dtype=bool)
            keep[20:25] = False  # creates a ~5x interval gap
            ts = ts[keep]
        n_actual = ts.size

        open_ = 100.0 + rng.normal(0, 1, n_actual).cumsum()
        close = open_ + rng.normal(0, 0.5, n_actual)
        high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n_actual))
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n_actual))
        volume = np.abs(rng.normal(1000, 100, n_actual))

        feat_a = rng.normal(0, 1, n_actual)
        feat_b = rng.normal(5, 2, n_actual)
        feat_c = rng.normal(-3, 1, n_actual)

        # NaNs in feat_a
        if n_actual > 5:
            feat_a[1] = np.nan
            feat_a[3] = np.nan
        # Outliers in feat_b
        if n_actual > 4:
            feat_b[2] = 1000.0
            feat_b[4] = -1000.0

        frames.append(
            pd.DataFrame(
                {
                    "ts_ms": ts,
                    "symbol": sym,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "feat_a": feat_a,
                    "feat_b": feat_b,
                    "feat_c": feat_c,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def _selftest() -> int:
    df = _build_synthetic_df()
    report = profile_dataset(df)

    # JSON round-trip must succeed (proves all floats are JSON-safe).
    json.dumps(report, allow_nan=False)

    # Gap detection
    total_gaps = sum(int(s["n_gaps"] or 0) for s in report["per_symbol"])
    assert total_gaps >= 1, f"expected at least one gap, got {total_gaps}"

    # Missing pct > 0 for the NaN column (feat_a)
    feat_a = next(c for c in report["columns"] if c["name"] == "feat_a")
    assert feat_a["missing_pct"] and feat_a["missing_pct"] > 0, (
        f"expected missing_pct>0 for feat_a, got {feat_a['missing_pct']}"
    )

    # Outliers detected for feat_b
    feat_b = next(c for c in report["columns"] if c["name"] == "feat_b")
    assert feat_b["n_outliers_iqr"] >= 1, (
        f"expected IQR outliers for feat_b, got {feat_b['n_outliers_iqr']}"
    )
    assert feat_b["n_outliers_z3"] >= 1, (
        f"expected z3 outliers for feat_b, got {feat_b['n_outliers_z3']}"
    )

    # OHLC violations must be computed (0 expected for clean synthetic OHLC).
    assert report["ohlc_violations"] is not None, "OHLC violations not computed"

    print("EDA SELFTEST OK")
    print(
        f"  n_rows={report['n_rows']} n_cols={report['n_cols']} "
        f"symbols={len(report['per_symbol'])}"
    )
    print(
        f"  total_gaps={total_gaps} "
        f"feat_a.missing_pct={feat_a['missing_pct']:.2f} "
        f"feat_b.n_outliers_iqr={feat_b['n_outliers_iqr']} "
        f"feat_b.n_outliers_z3={feat_b['n_outliers_z3']}"
    )
    print(
        f"  ohlc_violations={report['ohlc_violations']} "
        f"dup_ts={report['duplicate_timestamps']} "
        f"distributions={list(report['distributions'].keys())}"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="EDA / Data Quality profiler for quant datasets."
    )
    parser.add_argument("--selftest", action="store_true", help="run self-test")
    parser.add_argument(
        "--in", dest="input", help="input parquet/csv file path"
    )
    parser.add_argument(
        "--out",
        dest="out",
        default="models/eda_report.json",
        help="output JSON path (default: models/eda_report.json)",
    )
    parser.add_argument("--time-col", dest="time_col", default="ts_ms")
    parser.add_argument("--symbol-col", dest="symbol_col", default="symbol")
    parser.add_argument("--n-bins", dest="n_bins", type=int, default=30)

    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.input:
        parser.error("--in is required (or use --selftest)")

    return _run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())

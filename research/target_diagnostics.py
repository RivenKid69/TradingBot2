"""Target/Label Diagnostics.

Self-contained diagnostic panel for trading targets/labels.

Reports class balance (binary) or distribution stats (continuous),
autocorrelation (ACF), stationarity (ADF or variance-ratio fallback),
and per-symbol summaries.

Dependencies: stdlib + numpy + pandas (required).
              statsmodels (optional, for ACF Ljung-Box + ADF).
              scipy (optional, for skew/kurtosis; numpy fallback otherwise).

No project-internal imports. All outputs are JSON-safe.

CLI:
    python research/target_diagnostics.py --in <file> \
        --out <json default models/target_diagnostics.json> \
        --target <col> [--time-col ts_ms] [--symbol-col symbol]

Self-test:
    python research/target_diagnostics.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------
try:  # statsmodels for ADF + Ljung-Box
    from statsmodels.stats.diagnostic import acorr_ljungbox as _acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller as _adfuller

    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_STATSMODELS = False
    _acorr_ljungbox = None
    _adfuller = None

try:  # scipy for skew/kurtosis
    from scipy import stats as _scipy_stats

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_SCIPY = False
    _scipy_stats = None


# ---------------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Recursively convert numpy / non-finite values into JSON-safe types."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if not math.isfinite(f):
            return None
        return f
    if isinstance(value, (np.ndarray,)):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (str,)):
        return value
    # Fallback: best effort
    try:
        return float(value)
    except Exception:
        return str(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _skewness(x: np.ndarray) -> Optional[float]:
    if _HAS_SCIPY:
        try:
            return _safe_float(_scipy_stats.skew(x, bias=False))
        except Exception:
            pass
    n = x.size
    if n < 3:
        return None
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return None
    g1 = np.mean(((x - m) / s) ** 3)
    # Adjusted Fisher-Pearson standardized moment coefficient
    return _safe_float(math.sqrt(n * (n - 1)) / (n - 2) * g1)


def _kurtosis(x: np.ndarray) -> Optional[float]:
    """Excess kurtosis (normal -> 0)."""
    if _HAS_SCIPY:
        try:
            return _safe_float(_scipy_stats.kurtosis(x, fisher=True, bias=False))
        except Exception:
            pass
    n = x.size
    if n < 4:
        return None
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return None
    g2 = np.mean(((x - m) / s) ** 4) - 3.0
    # Sample-corrected excess kurtosis
    num = (n - 1) / ((n - 2) * (n - 3))
    return _safe_float(num * ((n + 1) * g2 + 6))


def _acf_numpy(x: np.ndarray, max_lag: int) -> List[float]:
    """ACF via acf[k] = cov(x_t, x_{t-k}) / var(x), biased (divide by n)."""
    n = x.size
    out: List[float] = []
    if n < 2:
        return [float("nan")] * max_lag
    xm = x - x.mean()
    var = np.dot(xm, xm) / n
    if var == 0:
        return [float("nan")] * max_lag
    for k in range(1, max_lag + 1):
        if k >= n:
            out.append(float("nan"))
            continue
        cov = np.dot(xm[k:], xm[:-k]) / n
        out.append(cov / var)
    return out


def _compute_acf(
    df: pd.DataFrame,
    target_col: str,
    time_col: Optional[str],
    symbol_col: Optional[str],
    max_lag: int,
) -> List[float]:
    """Compute ACF; per-symbol then averaged, or global if no symbol_col."""
    if symbol_col is not None and symbol_col in df.columns:
        per_symbol_acfs: List[List[float]] = []
        for _, grp in df.groupby(symbol_col, sort=False):
            g = grp
            if time_col is not None and time_col in g.columns:
                g = g.sort_values(time_col)
            series = pd.to_numeric(g[target_col], errors="coerce").dropna().to_numpy(
                dtype=float
            )
            if series.size >= 2:
                per_symbol_acfs.append(_acf_numpy(series, max_lag))
        if per_symbol_acfs:
            arr = np.array(per_symbol_acfs, dtype=float)
            with np.errstate(all="ignore"):
                avg = np.nanmean(arr, axis=0)
            return [float(v) for v in avg]
        # fall through to global if no usable symbol groups

    g = df
    if time_col is not None and time_col in g.columns:
        g = g.sort_values(time_col)
    series = pd.to_numeric(g[target_col], errors="coerce").dropna().to_numpy(dtype=float)
    return _acf_numpy(series, max_lag)


def _global_sorted_series(
    df: pd.DataFrame,
    target_col: str,
    time_col: Optional[str],
    symbol_col: Optional[str],
) -> np.ndarray:
    """Time-sorted target series (global, but stable per-symbol grouping)."""
    g = df
    sort_cols: List[str] = []
    if symbol_col is not None and symbol_col in g.columns:
        sort_cols.append(symbol_col)
    if time_col is not None and time_col in g.columns:
        sort_cols.append(time_col)
    if sort_cols:
        g = g.sort_values(sort_cols)
    return pd.to_numeric(g[target_col], errors="coerce").dropna().to_numpy(dtype=float)


def _ljung_box_pvalue(series: np.ndarray) -> Optional[float]:
    if not _HAS_STATSMODELS or series.size < 12:
        return None
    try:
        res = _acorr_ljungbox(series, lags=[10], return_df=True)
        return _safe_float(res["lb_pvalue"].iloc[-1])
    except Exception:
        return None


def _stationarity(series: np.ndarray) -> Dict[str, Any]:
    if _HAS_STATSMODELS and series.size >= 12:
        try:
            res = _adfuller(series, autolag="AIC")
            adf_stat = _safe_float(res[0])
            pvalue = _safe_float(res[1])
            return {
                "method": "adf",
                "adf_stat": adf_stat,
                "pvalue": pvalue,
                "stationary": bool(pvalue is not None and pvalue < 0.05),
            }
        except Exception:
            pass

    # Variance-ratio fallback: compare first vs second half
    n = series.size
    if n < 4:
        return {
            "method": "variance_ratio_fallback",
            "first_half_mean": None,
            "second_half_mean": None,
            "first_half_std": None,
            "second_half_std": None,
            "mean_shift_z": None,
            "stationary": None,
        }
    half = n // 2
    first = series[:half]
    second = series[half:]
    m1 = float(first.mean())
    m2 = float(second.mean())
    s1 = float(first.std(ddof=1)) if first.size > 1 else 0.0
    s2 = float(second.std(ddof=1)) if second.size > 1 else 0.0
    se = math.sqrt(
        (s1 ** 2 / max(first.size, 1)) + (s2 ** 2 / max(second.size, 1))
    )
    if se > 0:
        z = (m2 - m1) / se
    else:
        z = 0.0 if m1 == m2 else float("inf")
    z_safe = _safe_float(z)
    return {
        "method": "variance_ratio_fallback",
        "first_half_mean": _safe_float(m1),
        "second_half_mean": _safe_float(m2),
        "first_half_std": _safe_float(s1),
        "second_half_std": _safe_float(s2),
        "mean_shift_z": z_safe,
        "stationary": bool(z_safe is not None and abs(z_safe) < 2),
    }


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------
def diagnose_target(
    df: pd.DataFrame,
    target_col: str,
    time_col: str = "ts_ms",
    symbol_col: str = "symbol",
    acf_lags: int = 40,
    n_bins: int = 40,
) -> Dict[str, Any]:
    """Diagnose a target/label column. Returns JSON-safe dict."""
    if target_col not in df.columns:
        raise KeyError(f"target_col {target_col!r} not in DataFrame columns")

    has_symbol = symbol_col is not None and symbol_col in df.columns
    has_time = time_col is not None and time_col in df.columns
    eff_symbol = symbol_col if has_symbol else None
    eff_time = time_col if has_time else None

    raw = pd.to_numeric(df[target_col], errors="coerce")
    n_missing = int(raw.isna().sum())
    values = raw.dropna().to_numpy(dtype=float)

    result: Dict[str, Any] = {}
    result["methods"] = {
        "statsmodels_available": bool(_HAS_STATSMODELS),
        "scipy_available": bool(_HAS_SCIPY),
    }

    # Detect type: binary if non-NaN values subset of {0, 1}
    unique_vals = np.unique(values) if values.size else np.array([])
    is_binary = values.size > 0 and np.all(np.isin(unique_vals, [0.0, 1.0]))
    result["target_type"] = "binary" if is_binary else "continuous"

    if is_binary:
        n1 = int(np.sum(values == 1.0))
        n0 = int(np.sum(values == 0.0))
        total = n0 + n1
        result["class_counts"] = {"0": n0, "1": n1}
        result["positive_rate"] = _safe_float(n1 / total) if total > 0 else None
        mn = min(n0, n1)
        mx = max(n0, n1)
        result["imbalance_ratio"] = _safe_float(mx / mn) if mn > 0 else None
        result["n_missing"] = n_missing
    else:
        if values.size > 0:
            q = np.percentile(values, [1, 25, 50, 75, 99])
            hist_counts, hist_edges = np.histogram(values, bins=n_bins)
            result_stats = {
                "mean": _safe_float(values.mean()),
                "std": _safe_float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "skew": _skewness(values),
                "kurtosis": _kurtosis(values),
                "min": _safe_float(values.min()),
                "p1": _safe_float(q[0]),
                "p25": _safe_float(q[1]),
                "p50": _safe_float(q[2]),
                "p75": _safe_float(q[3]),
                "p99": _safe_float(q[4]),
                "max": _safe_float(values.max()),
                "positive_fraction": _safe_float(np.mean(values > 0)),
                "n_missing": n_missing,
            }
            result.update(result_stats)
            result["histogram"] = {
                "bin_edges": [_safe_float(e) for e in hist_edges],
                "counts": [int(c) for c in hist_counts],
            }
        else:
            for k in (
                "mean",
                "std",
                "skew",
                "kurtosis",
                "min",
                "p1",
                "p25",
                "p50",
                "p75",
                "p99",
                "max",
                "positive_fraction",
            ):
                result[k] = None
            result["n_missing"] = n_missing
            result["histogram"] = {"bin_edges": [], "counts": []}

    # Autocorrelation
    acf_vals = _compute_acf(df, target_col, eff_time, eff_symbol, acf_lags)
    result["autocorrelation"] = [
        {"lag": lag, "acf": _safe_float(acf_vals[lag - 1])}
        for lag in range(1, acf_lags + 1)
    ]

    global_series = _global_sorted_series(df, target_col, eff_time, eff_symbol)
    result["ljung_box_pvalue"] = _ljung_box_pvalue(global_series)

    # Stationarity
    result["stationarity"] = _stationarity(global_series)

    # Per-symbol summary
    if has_symbol:
        per_symbol: List[Dict[str, Any]] = []
        for sym, grp in df.groupby(symbol_col, sort=False):
            s = pd.to_numeric(grp[target_col], errors="coerce").dropna().to_numpy(
                dtype=float
            )
            entry: Dict[str, Any] = {
                "symbol": str(sym),
                "n": int(s.size),
            }
            if is_binary:
                entry["positive_rate"] = (
                    _safe_float(np.mean(s == 1.0)) if s.size else None
                )
            else:
                entry["mean"] = _safe_float(s.mean()) if s.size else None
            entry["std"] = (
                _safe_float(s.std(ddof=1)) if s.size > 1 else (0.0 if s.size == 1 else None)
            )
            per_symbol.append(entry)
        result["per_symbol"] = per_symbol

    return _json_safe(result)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def _load_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    # default: try parquet then csv
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _run_selftest() -> int:
    rng = np.random.default_rng(42)

    # (a) Binary target with 85/15 imbalance
    n = 20000
    n_pos = int(round(n * 0.15))
    y = np.zeros(n, dtype=float)
    y[:n_pos] = 1.0
    rng.shuffle(y)
    df_bin = pd.DataFrame(
        {
            "ts_ms": np.arange(n),
            "symbol": ["AAA"] * n,
            "target": y,
        }
    )
    res_bin = diagnose_target(df_bin, "target")
    assert res_bin["target_type"] == "binary", res_bin["target_type"]
    imbalance = res_bin["imbalance_ratio"]
    pos_rate = res_bin["positive_rate"]
    assert abs(imbalance - 5.6) < 0.5, f"imbalance_ratio={imbalance}"
    assert abs(pos_rate - 0.15) < 0.02, f"positive_rate={pos_rate}"

    # (b) Continuous AR(1) with phi=0.8
    phi = 0.8
    m = 30000
    eps = rng.standard_normal(m)
    x = np.zeros(m)
    for t in range(1, m):
        x[t] = phi * x[t - 1] + eps[t]
    df_ar = pd.DataFrame(
        {
            "ts_ms": np.arange(m),
            "symbol": ["BBB"] * m,
            "target": x,
        }
    )
    res_ar = diagnose_target(df_ar, "target")
    assert res_ar["target_type"] == "continuous", res_ar["target_type"]
    acf1 = res_ar["autocorrelation"][0]["acf"]
    assert abs(acf1 - 0.8) < 0.1, f"acf[1]={acf1}"
    if _HAS_STATSMODELS:
        assert res_ar["stationarity"]["stationary"] is True, res_ar["stationarity"]

    print("TARGET SELFTEST OK")
    print(f"  acf[1] (AR(1) phi=0.8) = {acf1:.4f}")
    print(f"  imbalance_ratio        = {imbalance:.4f}")
    print(f"  positive_rate          = {pos_rate:.4f}")
    print(f"  statsmodels_available  = {_HAS_STATSMODELS}")
    print(f"  scipy_available        = {_HAS_SCIPY}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Target/Label Diagnostics panel."
    )
    parser.add_argument("--selftest", action="store_true", help="Run self-test.")
    parser.add_argument("--in", dest="in_path", help="Input parquet/csv file.")
    parser.add_argument(
        "--out",
        dest="out_path",
        default="models/target_diagnostics.json",
        help="Output JSON path.",
    )
    parser.add_argument("--target", dest="target", help="Target column name.")
    parser.add_argument("--time-col", dest="time_col", default="ts_ms")
    parser.add_argument("--symbol-col", dest="symbol_col", default="symbol")
    parser.add_argument("--acf-lags", dest="acf_lags", type=int, default=40)
    parser.add_argument("--n-bins", dest="n_bins", type=int, default=40)
    args = parser.parse_args(argv)

    if args.selftest:
        return _run_selftest()

    if not args.in_path or not args.target:
        parser.error("--in and --target are required (unless --selftest).")

    df = _load_df(args.in_path)
    result = diagnose_target(
        df,
        target_col=args.target,
        time_col=args.time_col,
        symbol_col=args.symbol_col,
        acf_lags=args.acf_lags,
        n_bins=args.n_bins,
    )

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote diagnostics to {args.out_path}")
    print(f"  target_type = {result.get('target_type')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

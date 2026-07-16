#!/usr/bin/env python3
"""Advanced financial ML features (López de Prado techniques).

Self-contained module implementing:
  1. Fractional differentiation (Fixed-Width Window / FFD)
  2. Optimal `d` search via stationarity (ADF) + memory retention
  3. Sample uniqueness / concurrency weighting for overlapping labels
  4. Meta-labeling (was the primary bet correct?)
  5. Point-in-time membership (survivorship-bias-free universe)

Dependencies: stdlib + numpy + pandas. statsmodels is OPTIONAL (used for the
Augmented Dickey-Fuller test); a graceful fallback is provided.

References:
    López de Prado, M. (2018). "Advances in Financial Machine Learning."
        - Ch. 5 (Fractionally Differentiated Features)
        - Ch. 3 (Labeling / Meta-labeling)
        - Ch. 4 (Sample Weights / Uniqueness)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependency: statsmodels (ADF test)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from statsmodels.tsa.stattools import adfuller as _adfuller  # type: ignore

    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    _adfuller = None  # type: ignore
    _HAS_STATSMODELS = False


# ---------------------------------------------------------------------------
# 1. Fractional Differentiation (Fixed-Width Window)
# ---------------------------------------------------------------------------
def _ffd_weights(d: float, thresh: float = 1e-5, max_width: int = 100_000) -> np.ndarray:
    """Compute FFD weights via the recursion w_0 = 1, w_k = -w_{k-1} (d-k+1)/k.

    Generation stops once |w_k| < thresh. Weights are returned in *descending lag*
    order, i.e. ``w[0]`` multiplies the most recent observation ``x[t]`` and
    ``w[-1]`` multiplies the oldest ``x[t - (width-1)]``.
    """
    w: List[float] = [1.0]
    k = 1
    while k < max_width:
        w_k = -w[-1] * (d - k + 1.0) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.asarray(w, dtype=float)


def frac_diff_ffd(series: Sequence[float], d: float, thresh: float = 1e-5) -> np.ndarray:
    """Fixed-Width-Window fractional differentiation.

    Parameters
    ----------
    series : 1-D sequence of floats (may contain NaNs).
    d : differentiation order in [0, 1] (or beyond).
    thresh : weight cutoff controlling window width.

    Returns
    -------
    np.ndarray of same length as ``series``. The first ``width-1`` entries are
    NaN (insufficient history). NaNs inside the input window also propagate to
    NaN for the affected output points.
    """
    x = np.asarray(series, dtype=float).ravel()
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out

    w = _ffd_weights(d, thresh=thresh)
    width = w.shape[0]
    if width > n:
        # Window wider than data: nothing computable.
        return out

    for t in range(width - 1, n):
        window = x[t - width + 1 : t + 1]  # oldest .. newest
        if not np.all(np.isfinite(window)):
            continue  # NaN-robust: skip windows with missing data
        # w[0] applies to newest (window[-1]); reverse window to align with w.
        out[t] = float(np.dot(w, window[::-1]))
    return out


# ---------------------------------------------------------------------------
# 2. Optimal d search
# ---------------------------------------------------------------------------
def _adf_pvalue(arr: np.ndarray) -> Optional[float]:
    """Return ADF p-value, or None if statsmodels unavailable / test fails."""
    finite = arr[np.isfinite(arr)]
    if finite.shape[0] < 10 or np.nanstd(finite) == 0.0:
        return None
    if not _HAS_STATSMODELS:
        return None
    try:  # pragma: no cover - depends on optional lib
        result = _adfuller(finite, maxlag=1, regression="c", autolag=None)
        return float(result[1])
    except Exception:  # pragma: no cover
        return None


def find_min_ffd_d(
    series: Sequence[float],
    d_grid: Optional[Sequence[float]] = None,
    thresh: float = 1e-5,
    pvalue_cut: float = 0.05,
) -> Dict[str, Any]:
    """Search a grid of ``d`` values for minimum differentiation that yields
    stationarity, while tracking memory retention (correlation with original).
    """
    if d_grid is None:
        d_grid = np.arange(0.0, 1.01, 0.05)
    d_grid = [float(d) for d in d_grid]

    x = np.asarray(series, dtype=float).ravel()
    method = "adf" if _HAS_STATSMODELS else "no_statsmodels"

    table: List[Dict[str, Any]] = []
    min_d_stationary: Optional[float] = None

    for d in d_grid:
        fd = frac_diff_ffd(x, d, thresh=thresh)
        mask = np.isfinite(fd) & np.isfinite(x)
        if mask.sum() >= 2:
            xa = x[mask]
            fa = fd[mask]
            if np.std(xa) > 0 and np.std(fa) > 0:
                corr = float(np.corrcoef(xa, fa)[0, 1])
            else:
                corr = float("nan")
        else:
            corr = float("nan")

        pval = _adf_pvalue(fd)
        stationary = bool(pval is not None and pval < pvalue_cut)
        if stationary and min_d_stationary is None:
            min_d_stationary = d

        table.append(
            {
                "d": round(d, 6),
                "adf_pvalue": (None if pval is None else round(pval, 6)),
                "corr_with_original": (None if not np.isfinite(corr) else round(corr, 6)),
                "stationary": stationary,
            }
        )

    return {
        "table": table,
        "min_d_stationary": min_d_stationary,
        "method": method,
    }


# ---------------------------------------------------------------------------
# 3. Sample uniqueness / concurrency
# ---------------------------------------------------------------------------
def sample_uniqueness(
    t0: Sequence[int],
    t1: Sequence[int],
    n_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """Average uniqueness of overlapping labels.

    Each label ``i`` spans bar indices ``[t0_i, t1_i]`` (inclusive). Concurrency
    ``c[t]`` counts how many labels span bar ``t``. A label's average uniqueness
    is the mean of ``1/c[t]`` over its span.
    """
    a0 = np.asarray(t0, dtype=int).ravel()
    a1 = np.asarray(t1, dtype=int).ravel()
    if a0.shape != a1.shape:
        raise ValueError("t0 and t1 must have the same length")
    n_labels = a0.shape[0]

    if n_labels == 0:
        return {"avg_uniqueness": [], "mean_avg_uniqueness": float("nan"), "concurrency_max": 0}

    span_max = int(a1.max()) + 1
    total_bars = span_max if n_bars is None else max(int(n_bars), span_max)

    concurrency = np.zeros(total_bars, dtype=float)
    for i in range(n_labels):
        lo = max(0, int(a0[i]))
        hi = int(a1[i])
        if hi < lo:
            continue
        concurrency[lo : hi + 1] += 1.0

    avg_uniqueness: List[float] = []
    for i in range(n_labels):
        lo = max(0, int(a0[i]))
        hi = int(a1[i])
        if hi < lo:
            avg_uniqueness.append(float("nan"))
            continue
        c = concurrency[lo : hi + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(c > 0, 1.0 / c, np.nan)
        avg_uniqueness.append(float(np.nanmean(inv)))

    valid = [u for u in avg_uniqueness if np.isfinite(u)]
    mean_u = float(np.mean(valid)) if valid else float("nan")

    return {
        "avg_uniqueness": avg_uniqueness,
        "mean_avg_uniqueness": mean_u,
        "concurrency_max": int(concurrency.max()) if total_bars > 0 else 0,
    }


# ---------------------------------------------------------------------------
# 4. Meta-labeling
# ---------------------------------------------------------------------------
def meta_label(
    primary_signal: Sequence[float],
    realized_return: Sequence[float],
) -> Dict[str, Any]:
    """Meta-labels = was the primary bet correct?

    ``meta_label_i = 1`` iff ``primary_signal_i != 0`` and
    ``sign(primary_signal_i) * realized_return_i > 0``.
    """
    sig = np.asarray(primary_signal, dtype=float).ravel()
    ret = np.asarray(realized_return, dtype=float).ravel()
    if sig.shape != ret.shape:
        raise ValueError("primary_signal and realized_return must have the same length")

    n = sig.shape[0]
    meta = np.zeros(n, dtype=int)
    nonzero = (sig != 0) & np.isfinite(sig) & np.isfinite(ret)
    correct = nonzero & (np.sign(sig) * ret > 0)
    meta[correct] = 1

    n_signals = int(nonzero.sum())
    hit_rate = float(meta[nonzero].mean()) if n_signals > 0 else float("nan")
    # precision_if_traded: fraction of taken bets that were correct (== hit_rate here)
    precision_if_traded = hit_rate

    return {
        "meta_labels": meta.tolist(),
        "n_signals": n_signals,
        "hit_rate": hit_rate,
        "precision_if_traded": precision_if_traded,
    }


# ---------------------------------------------------------------------------
# 5. Point-in-time membership (survivorship-bias-free)
# ---------------------------------------------------------------------------
def point_in_time_membership(
    timestamps: Sequence[int],
    listings: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Boolean membership matrix [n_ts x n_assets] respecting list/delist dates.

    ``membership[t, a] = True`` iff ``list_date <= ts < delist_date`` (delist None
    means still active).
    """
    ts = np.asarray(timestamps, dtype=float).ravel()
    n_ts = ts.shape[0]
    n_assets = len(listings)

    membership = np.zeros((n_ts, n_assets), dtype=bool)
    assets: List[Any] = []
    for a, item in enumerate(listings):
        assets.append(item.get("asset", a))
        list_ms = item.get("list_date_ms", None)
        delist_ms = item.get("delist_date_ms", None)
        lo = -np.inf if list_ms is None else float(list_ms)
        hi = np.inf if delist_ms is None else float(delist_ms)
        membership[:, a] = (ts >= lo) & (ts < hi)

    as_of_counts = membership.sum(axis=1).astype(int).tolist()

    return {
        "membership": membership,
        "assets": assets,
        "as_of_counts": as_of_counts,
    }


# ---------------------------------------------------------------------------
# CLI / I/O helpers
# ---------------------------------------------------------------------------
def _load_series(path: str, price_col: str, time_col: Optional[str]) -> np.ndarray:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.lower().endswith((".json",)):
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)

    if time_col and time_col in df.columns:
        df = df.sort_values(time_col)
    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found. Available: {list(df.columns)}")
    return np.asarray(df[price_col], dtype=float)


def _run_cli(args: argparse.Namespace) -> int:
    series = _load_series(args.in_path, args.price_col, args.time_col)

    report: Dict[str, Any] = {
        "input": args.in_path,
        "price_col": args.price_col,
        "n_obs": int(series.shape[0]),
        "has_statsmodels": _HAS_STATSMODELS,
    }

    if args.op in ("fracdiff", "all"):
        search = find_min_ffd_d(series)
        report["find_min_ffd_d"] = search
        chosen_d = search["min_d_stationary"]
        if chosen_d is None:
            chosen_d = 0.5
        report["chosen_d"] = chosen_d
        fd = frac_diff_ffd(series, chosen_d)
        finite = fd[np.isfinite(fd)]
        report["fracdiff_sample"] = {
            "d": chosen_d,
            "n_finite": int(finite.shape[0]),
            "first_values": [round(float(v), 6) for v in finite[:10].tolist()],
        }

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote report to {args.out_path}")
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest() -> int:
    rng = np.random.default_rng(42)

    # --- frac_diff_ffd on a random walk -------------------------------------
    rw = np.cumsum(rng.standard_normal(2000))
    search = find_min_ffd_d(rw)
    min_d = search["min_d_stationary"]

    # corr_with_original should generally decrease as d increases
    corrs = [
        row["corr_with_original"]
        for row in search["table"]
        if row["corr_with_original"] is not None
    ]
    if len(corrs) >= 2:
        # compare a low-d corr vs high-d corr
        assert corrs[0] >= corrs[-1] - 1e-6, "corr should decrease with larger d"

    if _HAS_STATSMODELS:
        assert min_d is not None, "random walk should become stationary for some d"
        assert 0.0 < min_d < 1.0, f"min_d expected in (0,1), got {min_d}"

    # --- sample_uniqueness --------------------------------------------------
    su = sample_uniqueness(t0=[0, 1, 2], t1=[5, 6, 7])
    assert su["concurrency_max"] >= 3, f"concurrency_max {su['concurrency_max']} < 3"
    for u in su["avg_uniqueness"]:
        assert 0.0 < u <= 1.0, f"uniqueness {u} not in (0,1]"

    # --- meta_label ---------------------------------------------------------
    ml = meta_label(primary_signal=[1, -1, 1, 0], realized_return=[0.02, -0.01, -0.03, 0.05])
    assert ml["meta_labels"] == [1, 1, 0, 0], ml["meta_labels"]
    assert abs(ml["hit_rate"] - (2.0 / 3.0)) < 1e-9, ml["hit_rate"]

    # --- point_in_time_membership -------------------------------------------
    pit = point_in_time_membership(
        timestamps=[100, 200, 300],
        listings=[
            {"asset": "A", "list_date_ms": 150, "delist_date_ms": None},
            {"asset": "B", "list_date_ms": 0, "delist_date_ms": 250},
        ],
    )
    assert pit["as_of_counts"] == [1, 2, 1], pit["as_of_counts"]

    print("ADVFEAT SELFTEST OK")
    print(f"min_d={min_d} hit_rate={ml['hit_rate']:.6f} concurrency_max={su['concurrency_max']}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Advanced financial ML features (López de Prado).")
    parser.add_argument("--in", dest="in_path", help="Input file (csv/parquet/json).")
    parser.add_argument(
        "--out",
        dest="out_path",
        default="models/advanced_features.json",
        help="Output JSON report path.",
    )
    parser.add_argument("--price-col", dest="price_col", default="close")
    parser.add_argument("--time-col", dest="time_col", default="ts_ms")
    parser.add_argument("--op", dest="op", choices=["fracdiff", "find_min_d", "all"], default="all")
    parser.add_argument("--selftest", action="store_true", help="Run self-tests and exit.")

    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.in_path:
        parser.error("--in is required unless --selftest is given")

    # normalize find_min_d alias
    if args.op == "find_min_d":
        args.op = "fracdiff"
    return _run_cli(args)


if __name__ == "__main__":
    sys.exit(main())

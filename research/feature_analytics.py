"""Feature Analysis & Selection quant panel.

Self-contained analytics for evaluating predictive features against a target:
correlation/collinearity, VIF, information coefficient (IC) with rolling
stability, mutual information, model-based importance, and PSI drift.

Design constraints:
- stdlib + numpy + pandas required.
- scipy optional (used for Spearman; numpy fallback otherwise).
- sklearn optional (used for mutual info + RandomForest importance; graceful
  fallbacks otherwise).
- No project-internal imports.
- Every numeric value emitted is JSON-safe: NaN/inf are replaced with ``None``.
- The method actually used for each pluggable computation is recorded.

CLI:
    python research/feature_analytics.py --in <file> --out <json> --target <col> \
        [--features a,b,c] [--time-col ts_ms]
    python research/feature_analytics.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------
try:  # scipy is optional
    from scipy import stats as _scipy_stats  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - exercised only when scipy missing
    _scipy_stats = None  # type: ignore
    _HAS_SCIPY = False

try:  # sklearn is optional
    from sklearn.ensemble import (  # type: ignore
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.feature_selection import (  # type: ignore
        mutual_info_classif,
        mutual_info_regression,
    )

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - exercised only when sklearn missing
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------------
def _safe_float(x) -> Optional[float]:
    """Return a JSON-safe float or ``None`` for NaN/inf/non-numeric."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _safe_list(arr) -> List[Optional[float]]:
    return [_safe_float(v) for v in np.asarray(arr).ravel()]


# ---------------------------------------------------------------------------
# Rank-correlation primitives (NaN-robust, scipy-or-numpy)
# ---------------------------------------------------------------------------
def _spearman_pair(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Spearman rank correlation over rows where both are finite."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    xv, yv = x[mask], y[mask]
    # Degenerate (constant) inputs have undefined correlation.
    if np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return None
    if _HAS_SCIPY:
        try:
            rho, _ = _scipy_stats.spearmanr(xv, yv)
            return _safe_float(rho)
        except Exception:
            pass
    # numpy fallback: Pearson on ranks.
    rx = pd.Series(xv).rank().to_numpy()
    ry = pd.Series(yv).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return _safe_float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------
_EXCLUDE_NAME_TOKENS = {
    "ts_ms", "timestamp", "time", "date", "datetime",
    "symbol", "ticker", "asset", "instrument", "id",
    "open", "high", "low", "close", "volume", "ohlcv", "vwap",
}


def _auto_feature_cols(
    df: pd.DataFrame, target_col: str, time_col: str
) -> List[str]:
    out: List[str] = []
    for c in df.columns:
        if c in (target_col, time_col):
            continue
        if c.lower() in _EXCLUDE_NAME_TOKENS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        out.append(c)
    return out


def _is_binary_target(y: pd.Series) -> bool:
    vals = pd.unique(y.dropna())
    try:
        nums = set(float(v) for v in vals)
    except (TypeError, ValueError):
        return False
    return nums.issubset({0.0, 1.0}) and len(nums) <= 2


# ---------------------------------------------------------------------------
# Individual analytics blocks
# ---------------------------------------------------------------------------
def _compute_correlation(df: pd.DataFrame, feats: Sequence[str]) -> dict:
    n = len(feats)
    matrix: List[List[Optional[float]]] = [[None] * n for _ in range(n)]
    pairs: List[Tuple[str, str, float]] = []
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            rho = _spearman_pair(
                df[feats[i]].to_numpy(), df[feats[j]].to_numpy()
            )
            matrix[i][j] = rho
            matrix[j][i] = rho
            if rho is not None:
                pairs.append((feats[i], feats[j], rho))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    top = [
        {"a": a, "b": b, "corr": _safe_float(c)}
        for a, b, c in pairs[:15]
    ]
    return {"labels": list(feats), "matrix": matrix, "top_collinear": top}


def _standardize(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    return (mat - mu) / sd_safe


def _compute_vif(df: pd.DataFrame, feats: Sequence[str]) -> List[dict]:
    out: List[dict] = []
    # Use only fully-observed rows for the joint regression design.
    sub = df[list(feats)].apply(pd.to_numeric, errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < len(feats) + 2:
        return [{"feature": f, "vif": None, "flag": "insufficient_data"}
                for f in feats]
    X = _standardize(sub.to_numpy(dtype=float))
    nrows = X.shape[0]
    for i, f in enumerate(feats):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        if others.shape[1] == 0:
            out.append({"feature": f, "vif": 1.0, "flag": None})
            continue
        A = np.column_stack([others, np.ones(nrows)])
        flag = None
        try:
            # Rank deficiency => near-singular design (collinear feature).
            rank = np.linalg.matrix_rank(A)
            if rank < A.shape[1]:
                out.append({"feature": f, "vif": None, "flag": "singular"})
                continue
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            if ss_tot <= 0:
                out.append({"feature": f, "vif": None, "flag": "constant"})
                continue
            r2 = 1.0 - ss_res / ss_tot
            denom = 1.0 - r2
            if denom <= 1e-10:
                vif = None
                flag = "near_singular"
            else:
                vif = _safe_float(1.0 / denom)
                if vif is None:
                    flag = "near_singular"
            out.append({"feature": f, "vif": vif, "flag": flag})
        except np.linalg.LinAlgError:
            out.append({"feature": f, "vif": None, "flag": "singular"})
    return out


def _compute_ic(
    df: pd.DataFrame, feats: Sequence[str], target_col: str,
    time_col: Optional[str], n_time_bins: int,
) -> List[dict]:
    # Time-sort if we have a usable time column; else keep given order.
    if time_col and time_col in df.columns:
        order = df[time_col].to_numpy()
        sort_idx = np.argsort(order, kind="stable")
    else:
        sort_idx = np.arange(len(df))
    sdf = df.iloc[sort_idx].reset_index(drop=True)
    y = sdf[target_col].to_numpy(dtype=float)

    nbins = max(1, int(n_time_bins))
    chunks = np.array_split(np.arange(len(sdf)), nbins) if len(sdf) else []

    out: List[dict] = []
    for f in feats:
        x = sdf[f].to_numpy(dtype=float)
        ic = _spearman_pair(x, y)
        rolling: List[Optional[float]] = []
        for ch in chunks:
            if len(ch) == 0:
                rolling.append(None)
                continue
            rolling.append(_spearman_pair(x[ch], y[ch]))
        valid = [r for r in rolling if r is not None]
        if valid:
            ic_mean = _safe_float(np.mean(valid))
            ic_std = _safe_float(np.std(valid, ddof=0))
        else:
            ic_mean = ic_std = None
        if ic_mean is not None and ic_std is not None and ic_std > 1e-12:
            ic_ir = _safe_float(ic_mean / ic_std)
        else:
            ic_ir = None
        out.append({
            "feature": f,
            "ic": ic,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "rolling_ic": rolling,
        })
    return out


def _compute_mutual_info(
    df: pd.DataFrame, feats: Sequence[str], target_col: str, is_binary: bool,
) -> Tuple[List[dict], str]:
    sub = df[list(feats) + [target_col]].apply(pd.to_numeric, errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if _HAS_SKLEARN and len(sub) >= 5:
        try:
            X = sub[list(feats)].to_numpy(dtype=float)
            y = sub[target_col].to_numpy(dtype=float)
            if is_binary:
                mi = mutual_info_classif(
                    X, y.astype(int), random_state=0
                )
                method = "sklearn_mutual_info_classif"
            else:
                mi = mutual_info_regression(X, y, random_state=0)
                method = "sklearn_mutual_info_regression"
            rows = [
                {"feature": f, "mi": _safe_float(m)}
                for f, m in zip(feats, mi)
            ]
            return rows, method
        except Exception:
            pass
    # Fallback: |spearman| as a monotonic-dependence proxy.
    rows = []
    for f in feats:
        rho = _spearman_pair(df[f].to_numpy(), df[target_col].to_numpy())
        rows.append({
            "feature": f,
            "mi": _safe_float(abs(rho)) if rho is not None else None,
            "method": "spearman_fallback",
        })
    return rows, "spearman_fallback"


def _compute_importance(
    df: pd.DataFrame, feats: Sequence[str], target_col: str,
    is_binary: bool, ic_rows: List[dict],
) -> Tuple[List[dict], str]:
    sub = df[list(feats) + [target_col]].apply(pd.to_numeric, errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if _HAS_SKLEARN and len(sub) >= 10:
        try:
            X = sub[list(feats)].to_numpy(dtype=float)
            y = sub[target_col].to_numpy(dtype=float)
            if is_binary:
                model = RandomForestClassifier(
                    n_estimators=100, random_state=0
                )
                model.fit(X, y.astype(int))
                method = "sklearn_random_forest_classifier"
            else:
                model = RandomForestRegressor(
                    n_estimators=100, random_state=0
                )
                model.fit(X, y)
                method = "sklearn_random_forest_regressor"
            imp = model.feature_importances_
            rows = [
                {"feature": f, "importance": _safe_float(v)}
                for f, v in zip(feats, imp)
            ]
            rows.sort(
                key=lambda r: (r["importance"] is not None, r["importance"]),
                reverse=True,
            )
            return rows, method
        except Exception:
            pass
    # Fallback: normalized |ic|.
    ic_map = {r["feature"]: r["ic"] for r in ic_rows}
    abs_ic = {f: (abs(ic_map[f]) if ic_map.get(f) is not None else 0.0)
              for f in feats}
    total = sum(abs_ic.values())
    rows = []
    for f in feats:
        val = abs_ic[f] / total if total > 0 else None
        rows.append({"feature": f, "importance": _safe_float(val)})
    rows.sort(
        key=lambda r: (r["importance"] is not None, r["importance"]),
        reverse=True,
    )
    return rows, "abs_ic_normalized_fallback"


def _psi(expected: np.ndarray, actual: np.ndarray) -> Optional[float]:
    """Population Stability Index using 10 quantile buckets from expected."""
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return None
    quantiles = np.linspace(0, 1, 11)
    edges = np.quantile(expected, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = e_counts / max(1, e_counts.sum())
    a_pct = a_counts / max(1, a_counts.sum())
    eps = 1e-6
    e_pct = np.where(e_pct == 0, eps, e_pct)
    a_pct = np.where(a_pct == 0, eps, a_pct)
    psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return _safe_float(psi)


def _compute_stability(
    df: pd.DataFrame, feats: Sequence[str], time_col: Optional[str],
) -> List[dict]:
    if time_col and time_col in df.columns:
        sort_idx = np.argsort(df[time_col].to_numpy(), kind="stable")
    else:
        sort_idx = np.arange(len(df))
    sdf = df.iloc[sort_idx].reset_index(drop=True)
    half = len(sdf) // 2
    out: List[dict] = []
    for f in feats:
        x = sdf[f].to_numpy(dtype=float)
        psi = _psi(x[:half], x[half:])
        if psi is None:
            status = "unknown"
        elif psi < 0.1:
            status = "stable"
        elif psi <= 0.25:
            status = "warning"
        else:
            status = "drifted"
        out.append({"feature": f, "psi": psi, "status": status})
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_features(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]],
    target_col: str,
    time_col: str = "ts_ms",
    n_time_bins: int = 10,
) -> Dict:
    """Run the full feature-analysis suite and return a JSON-safe dict."""
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} not in dataframe")

    if feature_cols is None or len(feature_cols) == 0:
        feats = _auto_feature_cols(df, target_col, time_col)
        feature_detection = "auto"
    else:
        feats = [c for c in feature_cols if c in df.columns]
        feature_detection = "explicit"
    if not feats:
        raise ValueError("no usable feature columns")

    y = df[target_col]
    is_binary = _is_binary_target(y)
    target_type = "binary" if is_binary else "continuous"

    ic_rows = _compute_ic(df, feats, target_col, time_col, n_time_bins)
    mi_rows, mi_method = _compute_mutual_info(df, feats, target_col, is_binary)
    imp_rows, imp_method = _compute_importance(
        df, feats, target_col, is_binary, ic_rows
    )

    result: Dict = {
        "meta": {
            "n_rows": int(len(df)),
            "n_features": len(feats),
            "feature_cols": list(feats),
            "target_col": target_col,
            "time_col": time_col if time_col in df.columns else None,
            "n_time_bins": int(n_time_bins),
            "feature_detection": feature_detection,
            "target_type": target_type,
            "has_scipy": _HAS_SCIPY,
            "has_sklearn": _HAS_SKLEARN,
            "spearman_method": "scipy" if _HAS_SCIPY else "numpy_rank",
        },
        "correlation": _compute_correlation(df, feats),
        "vif": _compute_vif(df, feats),
        "ic": ic_rows,
        "mutual_info": mi_rows,
        "mi_method": mi_method,
        "importance": imp_rows,
        "importance_method": imp_method,
        "stability": _compute_stability(df, feats, time_col),
    }
    return result


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _load_df(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path)
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    # Default: try parquet then csv.
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


def _summary_line(result: Dict) -> str:
    ic_sorted = sorted(
        result["ic"],
        key=lambda r: (r["ic"] is not None, abs(r["ic"]) if r["ic"] else 0.0),
        reverse=True,
    )
    top = ic_sorted[0] if ic_sorted else None
    top_str = (
        f"{top['feature']}(IC={top['ic']:.3f})"
        if top and top["ic"] is not None else "n/a"
    )
    return (
        f"Analyzed {result['meta']['n_features']} features / "
        f"{result['meta']['n_rows']} rows | top-IC {top_str} | "
        f"mi={result['mi_method']} imp={result['importance_method']} | "
        f"scipy={result['meta']['has_scipy']} sklearn={result['meta']['has_sklearn']}"
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _make_synthetic(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    feat_signal_true = rng.normal(0, 1, n)
    feat_signal = feat_signal_true + rng.normal(0, 0.2, n)
    feat_noise = rng.normal(0, 1, n)
    feat_collinear = 0.99 * feat_signal + rng.normal(0, 0.01, n)
    target = feat_signal_true + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "ts_ms": np.arange(n) * 1000,
        "feat_signal": feat_signal,
        "feat_noise": feat_noise,
        "feat_collinear": feat_collinear,
        "target": target,
    })


def _run_selftest() -> int:
    df = _make_synthetic(2000)
    result = analyze_features(
        df,
        feature_cols=["feat_signal", "feat_noise", "feat_collinear"],
        target_col="target",
        time_col="ts_ms",
        n_time_bins=10,
    )

    ic_map = {r["feature"]: r["ic"] for r in result["ic"]}
    ic_signal = abs(ic_map["feat_signal"] or 0.0)
    ic_noise = abs(ic_map["feat_noise"] or 0.0)

    # 1. signal IC must dominate noise IC.
    assert ic_signal > 0.5, f"signal IC too low: {ic_signal}"
    assert ic_noise < 0.1, f"noise IC too high: {ic_noise}"
    assert ic_signal > ic_noise * 5, "signal IC not >> noise IC"

    # 2. collinear pair must show high pairwise corr in top_collinear.
    pairs = result["correlation"]["top_collinear"]
    coll = [p for p in pairs
            if {p["a"], p["b"]} == {"feat_signal", "feat_collinear"}]
    assert coll and abs(coll[0]["corr"]) > 0.95, \
        f"collinear pair not flagged: {pairs}"

    # 3. collinear feature should have high VIF or singular flag.
    vif_map = {r["feature"]: r for r in result["vif"]}
    vc = vif_map["feat_collinear"]
    assert (vc["vif"] is None and vc["flag"]) or (vc["vif"] and vc["vif"] > 10), \
        f"collinear VIF not high/flagged: {vc}"

    # 4. signal should be the top-importance / top-MI feature.
    mi_sorted = sorted(
        result["mutual_info"],
        key=lambda r: (r["mi"] is not None, r["mi"] or 0.0), reverse=True,
    )
    assert mi_sorted[0]["feature"] in ("feat_signal", "feat_collinear"), \
        f"top MI not signal-like: {mi_sorted[0]}"
    assert result["importance"][0]["feature"] in (
        "feat_signal", "feat_collinear"
    ), f"top importance not signal-like: {result['importance'][0]}"

    print("FEATURE SELFTEST OK")
    print("IC ranking (|IC| desc):")
    ranking = sorted(
        result["ic"],
        key=lambda r: (r["ic"] is not None, abs(r["ic"]) if r["ic"] else 0.0),
        reverse=True,
    )
    for i, r in enumerate(ranking, 1):
        print(
            f"  {i}. {r['feature']:<16} IC={r['ic']:.4f} "
            f"IR={r['ic_ir'] if r['ic_ir'] is not None else float('nan'):.3f}"
        )
    print(_summary_line(result))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Feature Analysis & Selection quant panel."
    )
    parser.add_argument("--in", dest="in_path", help="input parquet/csv file")
    parser.add_argument(
        "--out", dest="out_path", default="models/feature_analytics.json",
        help="output JSON path (default: models/feature_analytics.json)",
    )
    parser.add_argument("--target", dest="target", help="target column name")
    parser.add_argument(
        "--features", dest="features", default=None,
        help="comma-separated feature columns (auto-detect if omitted)",
    )
    parser.add_argument("--time-col", dest="time_col", default="ts_ms")
    parser.add_argument(
        "--n-time-bins", dest="n_time_bins", type=int, default=10
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return _run_selftest()

    if not args.in_path or not args.target:
        parser.error("--in and --target are required (or use --selftest)")

    df = _load_df(args.in_path)
    feats = (
        [c.strip() for c in args.features.split(",") if c.strip()]
        if args.features else None
    )
    result = analyze_features(
        df, feats, args.target,
        time_col=args.time_col, n_time_bins=args.n_time_bins,
    )

    import os
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(_summary_line(result) + f" -> {args.out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

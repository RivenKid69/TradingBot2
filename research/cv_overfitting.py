#!/usr/bin/env python
"""Gold-standard overfitting controls for backtesting / model selection.

Self-contained module implementing three rigorously-defined tools from the
financial machine-learning literature:

  1. CPCV  -- Combinatorial Purged Cross-Validation
             (Lopez de Prado, "Advances in Financial Machine Learning", 2018)
  2. PBO   -- Probability of Backtest Overfitting via CSCV
             (Bailey, Borwein, Lopez de Prado, Zhu, 2017)
  3. DSR   -- Deflated Sharpe Ratio and related statistics
             (Bailey & Lopez de Prado, 2014)

Dependencies: stdlib + numpy + pandas. scipy is OPTIONAL (used only for the
normal CDF / inverse-CDF when present); a numpy/math fallback is provided.

No project-internal imports. Run ``python research/cv_overfitting.py --selftest``
to exercise every component.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Normal CDF / inverse-CDF (scipy optional)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised depending on environment
    from scipy.stats import norm as _scipy_norm  # type: ignore

    def norm_cdf(x: float) -> float:
        return float(_scipy_norm.cdf(x))

    def norm_ppf(p: float) -> float:
        return float(_scipy_norm.ppf(p))

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_SCIPY = False

    def norm_cdf(x: float) -> float:
        """Standard-normal CDF via the error function."""
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))

    def norm_ppf(p: float) -> float:
        """Inverse standard-normal CDF (Acklam's rational approximation).

        Relative error < 1.15e-9 over the full support; refined with one
        Halley step using the error-function CDF.
        """
        p = float(p)
        if p <= 0.0:
            return -math.inf
        if p >= 1.0:
            return math.inf

        # Coefficients (Peter Acklam).
        a = [-3.969683028665376e+01, 2.209460984245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798598866e+02, 6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01,
             2.445134137142996e+00, 3.754408661907416e+00]

        plow = 0.02425
        phigh = 1.0 - plow
        if p < plow:
            q = math.sqrt(-2.0 * math.log(p))
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        elif p <= phigh:
            q = p - 0.5
            r = q * q
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        else:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

        # One Halley refinement step.
        e = norm_cdf(x) - p
        u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
        x = x - u / (1.0 + x * u / 2.0)
        return x


EULER_MASCHERONI = 0.5772156649015329


# ===========================================================================
# 1. CPCV -- Combinatorial Purged Cross-Validation
# ===========================================================================
def _contiguous_groups(n_samples: int, n_groups: int) -> List[np.ndarray]:
    """Partition 0..n_samples-1 into ``n_groups`` contiguous index groups."""
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if n_groups > n_samples:
        raise ValueError("n_groups cannot exceed n_samples")
    return [g for g in np.array_split(np.arange(n_samples, dtype=int), n_groups)]


def cpcv_splits(
    n_samples: int,
    n_groups: int = 6,
    k_test: int = 2,
    horizon: int = 0,
    embargo: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate Combinatorial Purged Cross-Validation train/test splits.

    Parameters
    ----------
    n_samples : total number of ordered observations.
    n_groups  : number of contiguous groups to partition into.
    k_test    : number of groups assigned to the TEST set per combination.
    horizon   : label horizon (bars). Train samples whose label window would
                overlap a test block (i.e. within ``horizon`` bars BEFORE the
                start of any test block) are purged.
    embargo   : number of bars AFTER each test block to embargo from train.

    Returns
    -------
    list of (train_idx, test_idx) numpy arrays. There are C(n_groups, k_test)
    splits; the number of backtest paths is C(n_groups,k_test)*k_test/n_groups.
    """
    if k_test <= 0 or k_test >= n_groups:
        raise ValueError("require 0 < k_test < n_groups")

    groups = _contiguous_groups(n_samples, n_groups)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for combo in combinations(range(n_groups), k_test):
        test_groups = [groups[i] for i in combo]
        test_idx = np.sort(np.concatenate(test_groups))

        # Build contiguous test blocks (test groups may be non-contiguous).
        blocks: List[Tuple[int, int]] = []
        for g in test_groups:
            if len(g) == 0:
                continue
            blocks.append((int(g[0]), int(g[-1])))

        # Start with all non-test groups as candidate train.
        train_idx = np.sort(
            np.concatenate([groups[i] for i in range(n_groups) if i not in combo])
        )

        # Purge + embargo: remove forbidden indices.
        forbidden = np.zeros(n_samples, dtype=bool)
        forbidden[test_idx] = True  # test ranges themselves
        for (b0, b1) in blocks:
            # PURGE: horizon bars BEFORE the start of the test block.
            p0 = max(0, b0 - horizon)
            forbidden[p0:b0] = True
            # EMBARGO: embargo bars AFTER the end of the test block.
            e1 = min(n_samples, b1 + 1 + embargo)
            forbidden[b1 + 1:e1] = True

        train_idx = train_idx[~forbidden[train_idx]]
        splits.append((train_idx, test_idx))

    return splits


def cpcv_n_paths(n_groups: int, k_test: int) -> int:
    """Number of distinct backtest paths = C(n_groups,k_test)*k_test/n_groups."""
    return int(math.comb(n_groups, k_test) * k_test // n_groups)


def cpcv_report(
    n_samples: int,
    n_groups: int = 6,
    k_test: int = 2,
    horizon: int = 0,
    embargo: int = 0,
) -> Dict[str, Any]:
    """Summary statistics for a CPCV configuration."""
    splits = cpcv_splits(n_samples, n_groups, k_test, horizon, embargo)
    n_splits = len(splits)

    # Without purge/embargo, train size = n_samples - len(test).
    total_purged = 0
    total_embargoed = 0
    train_sizes = []
    test_sizes = []
    groups = _contiguous_groups(n_samples, n_groups)
    for (train_idx, test_idx), combo in zip(splits, combinations(range(n_groups), k_test)):
        train_sizes.append(len(train_idx))
        test_sizes.append(len(test_idx))
        baseline_train = n_samples - len(test_idx)
        removed = baseline_train - len(train_idx)

        # Decompose removed into purge vs embargo regions.
        test_groups = [groups[i] for i in combo]
        purge_mask = np.zeros(n_samples, dtype=bool)
        embargo_mask = np.zeros(n_samples, dtype=bool)
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[test_idx] = True
        for g in test_groups:
            if len(g) == 0:
                continue
            b0, b1 = int(g[0]), int(g[-1])
            purge_mask[max(0, b0 - horizon):b0] = True
            embargo_mask[b1 + 1:min(n_samples, b1 + 1 + embargo)] = True
        # Only count indices that were actually in candidate train (non-test).
        candidate = ~test_mask
        total_purged += int(np.sum(purge_mask & candidate & ~embargo_mask))
        total_embargoed += int(np.sum(embargo_mask & candidate))

    return {
        "n_groups": int(n_groups),
        "k_test": int(k_test),
        "n_splits": int(n_splits),
        "n_paths": cpcv_n_paths(n_groups, k_test),
        "avg_train_size": float(np.mean(train_sizes)) if train_sizes else 0.0,
        "avg_test_size": float(np.mean(test_sizes)) if test_sizes else 0.0,
        "total_purged": int(total_purged),
        "total_embargoed": int(total_embargoed),
        "horizon": int(horizon),
        "embargo": int(embargo),
    }


# ===========================================================================
# 2. PBO -- Probability of Backtest Overfitting via CSCV
# ===========================================================================
def _sharpe_columns(mat: np.ndarray) -> np.ndarray:
    """Non-annualized Sharpe (mean/std) per column; 0 where std==0."""
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(mat.shape[1])
    out = np.zeros_like(mean, dtype=float)
    nz = std > 0
    out[nz] = mean[nz] / std[nz]
    return out


def pbo_cscv(perf_matrix: np.ndarray, S: int = 16) -> Dict[str, Any]:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric CV.

    Parameters
    ----------
    perf_matrix : array of shape (T, N) -- T time observations of per-period
                  returns for N candidate strategy configurations.
    S           : number of disjoint contiguous row-submatrices (must be even).

    Returns
    -------
    dict with PBO, lambda statistics, and a logit histogram.
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("perf_matrix must be 2D (T x N)")
    T, N = M.shape
    if S % 2 != 0:
        raise ValueError("S must be even")
    if S < 2 or S > T:
        raise ValueError("require 2 <= S <= T")

    # Split into S disjoint contiguous submatrices; trim remainder rows.
    block = T // S
    usable = block * S
    M = M[:usable]
    sub = [M[i * block:(i + 1) * block] for i in range(S)]

    all_combos = list(combinations(range(S), S // 2))
    sampled = False
    if len(all_combos) > 20000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(all_combos), size=20000, replace=False)
        all_combos = [all_combos[i] for i in idx]
        sampled = True

    lambdas: List[float] = []
    for is_groups in all_combos:
        is_set = set(is_groups)
        oos_groups = [i for i in range(S) if i not in is_set]

        is_mat = np.concatenate([sub[i] for i in is_groups], axis=0)
        oos_mat = np.concatenate([sub[i] for i in oos_groups], axis=0)

        is_sr = _sharpe_columns(is_mat)
        oos_sr = _sharpe_columns(oos_mat)

        n_star = int(np.argmax(is_sr))

        # Relative rank of n* in OOS (ascending: 1 = worst, N = best).
        order = np.argsort(oos_sr, kind="mergesort")
        ranks = np.empty(N, dtype=float)
        ranks[order] = np.arange(1, N + 1, dtype=float)
        omega = ranks[n_star] / (N + 1.0)
        omega = min(max(omega, 1e-12), 1.0 - 1e-12)
        lam = math.log(omega / (1.0 - omega))
        lambdas.append(lam)

    lambdas_arr = np.asarray(lambdas, dtype=float)
    pbo = float(np.mean(lambdas_arr <= 0.0))

    counts, bin_edges = np.histogram(lambdas_arr, bins=20)

    return {
        "S": int(S),
        "n_combinations": int(len(all_combos)),
        "pbo": pbo,
        "lambda_mean": float(np.mean(lambdas_arr)),
        "lambda_hist": {
            "bin_edges": [float(x) for x in bin_edges],
            "counts": [int(c) for c in counts],
        },
        "sampled": bool(sampled),
        "T": int(T),
        "N": int(N),
    }


# ===========================================================================
# 3. Deflated Sharpe Ratio and related statistics
# ===========================================================================
def _moments(returns: np.ndarray) -> Tuple[float, float, float, float, int]:
    """Return (SR, skew, non_excess_kurt, std, T) for a return series."""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    T = int(len(r))
    if T < 2:
        raise ValueError("returns must have at least 2 finite observations")
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    if std <= 0:
        raise ValueError("return series has zero variance")
    sr = mean / std
    # Population skewness / kurtosis (biased) about the mean, normalized by
    # population std (ddof=0) -- standard for the PSR/DSR formulas.
    std_pop = float(np.std(r, ddof=0))
    centered = r - mean
    skew = float(np.mean(centered ** 3) / std_pop ** 3)
    kurt = float(np.mean(centered ** 4) / std_pop ** 4)  # NON-excess kurtosis
    return sr, skew, kurt, std, T


def probabilistic_sharpe_ratio(returns: Sequence[float], sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark)."""
    sr, skew, kurt, _, T = _moments(np.asarray(returns, dtype=float))
    denom = math.sqrt(max(1e-300, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))
    z = (sr - sr_benchmark) * math.sqrt(T - 1) / denom
    return float(norm_cdf(z))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """Expected maximum Sharpe under the null (independent trials).

    SR0 = sqrt(var_sr) * [ (1-gamma)*Phi^-1(1 - 1/N) + gamma*Phi^-1(1 - 1/(N*e)) ]
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    e = math.e
    term = (1.0 - g) * norm_ppf(1.0 - 1.0 / n_trials) + \
        g * norm_ppf(1.0 - 1.0 / (n_trials * e))
    return float(math.sqrt(max(0.0, var_sr)) * term)


def deflated_sharpe_ratio(
    returns: Sequence[float], n_trials: int, var_sr: float
) -> Dict[str, Any]:
    """Deflated Sharpe Ratio = PSR evaluated against expected_max_sharpe."""
    sr, _, _, _, T = _moments(np.asarray(returns, dtype=float))
    sr0 = expected_max_sharpe(var_sr, n_trials)
    dsr = probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)
    psr0 = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
    return {
        "sr": float(sr),
        "sr0": float(sr0),
        "dsr": float(dsr),
        "psr_vs_0": float(psr0),
        "n_trials": int(n_trials),
        "var_sr": float(var_sr),
        "T": int(T),
    }


def min_track_record_length(
    returns: Sequence[float], sr_benchmark: float = 0.0, prob: float = 0.95
) -> float:
    """Minimum track-record length to reject SR <= sr_benchmark at ``prob``."""
    sr, skew, kurt, _, _ = _moments(np.asarray(returns, dtype=float))
    if sr == sr_benchmark:
        return math.inf
    factor = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    return float(1.0 + factor * (norm_ppf(prob) / (sr - sr_benchmark)) ** 2)


# ===========================================================================
# Report assembly + IO helpers
# ===========================================================================
def _load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_report(
    *,
    cpcv_args: Optional[Dict[str, int]] = None,
    perf_matrix: Optional[np.ndarray] = None,
    pbo_S: int = 16,
    returns: Optional[np.ndarray] = None,
    n_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble a report dict from whichever inputs are supplied."""
    report: Dict[str, Any] = {}

    if cpcv_args is not None:
        report["cpcv"] = cpcv_report(**cpcv_args)

    if perf_matrix is not None:
        report["pbo"] = pbo_cscv(perf_matrix, S=pbo_S)

    if returns is not None:
        # Estimate var_sr across configs if a matrix is present, else use a
        # conservative default of 1/T (variance of SR under the null).
        if perf_matrix is not None:
            srs = _sharpe_columns(np.asarray(perf_matrix, dtype=float))
            var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 1.0 / max(1, len(returns))
        else:
            var_sr = 1.0 / max(1, len(returns) - 1)
        nt = int(n_trials) if n_trials is not None else (
            int(perf_matrix.shape[1]) if perf_matrix is not None else 1
        )
        dsr = deflated_sharpe_ratio(returns, n_trials=max(1, nt), var_sr=var_sr)
        dsr["min_track_record_length"] = min_track_record_length(returns)
        report["dsr"] = dsr

    return report


# ===========================================================================
# Self-test
# ===========================================================================
def _build_perf_matrix(T: int, N: int, seed: int, drift_idx: Optional[int] = None,
                       drift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 1.0, size=(T, N))
    if drift_idx is not None:
        M[:, drift_idx] += drift
    return M


def selftest() -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # --- CPCV -------------------------------------------------------------
    rep = cpcv_report(n_samples=120, n_groups=6, k_test=2, horizon=3, embargo=2)
    assert rep["n_splits"] == 15, f"n_splits={rep['n_splits']}"
    assert rep["n_paths"] == 5, f"n_paths={rep['n_paths']}"
    assert rep["total_purged"] > 0, f"total_purged={rep['total_purged']}"
    out["cpcv"] = rep
    print(f"[CPCV] n_splits={rep['n_splits']} n_paths={rep['n_paths']} "
          f"total_purged={rep['total_purged']} total_embargoed={rep['total_embargoed']}")

    # --- PBO: pure noise --------------------------------------------------
    # PBO ~= 0.5 holds IN EXPECTATION over realizations; a single (T,N) draw
    # has high sampling variance, so we average over several noise matrices to
    # exercise the law-of-large-numbers behaviour faithfully.
    n_real = 12
    noise_pbos = [pbo_cscv(_build_perf_matrix(T=1000, N=20, seed=s), S=16)["pbo"]
                  for s in range(n_real)]
    pbo_noise_single = pbo_cscv(_build_perf_matrix(T=1000, N=20, seed=42), S=16)
    pbo_noise = float(np.mean(noise_pbos))
    print(f"[PBO] noise pbo(avg over {n_real} realizations)={pbo_noise:.3f} "
          f"single-seed42={pbo_noise_single['pbo']:.3f} "
          f"combos={pbo_noise_single['n_combinations']}")
    assert 0.35 <= pbo_noise <= 0.65, f"noise PBO out of range: {pbo_noise}"

    # --- PBO: one genuinely better strategy -------------------------------
    better_pbos = [
        pbo_cscv(_build_perf_matrix(T=1000, N=20, seed=s, drift_idx=0, drift=0.12),
                 S=16)["pbo"]
        for s in range(n_real)
    ]
    pbo_better = float(np.mean(better_pbos))
    print(f"[PBO] better pbo(avg over {n_real} realizations)={pbo_better:.3f}")
    assert pbo_better < 0.4, f"better PBO not low enough: {pbo_better}"
    assert pbo_better < pbo_noise, "better PBO should drop below noise PBO"
    out["pbo_noise"] = pbo_noise
    out["pbo_better"] = pbo_better

    # --- DSR --------------------------------------------------------------
    rng = np.random.default_rng(7)
    # Clearly positive Sharpe series: mean 0.10, std 0.4 -> SR ~ 0.25 per period
    # over T=1000, giving a strongly significant track record.
    rets = rng.normal(0.10, 0.4, size=1000)
    dsr = deflated_sharpe_ratio(rets, n_trials=50, var_sr=1.0 / 1000.0)
    print(f"[DSR] sr={dsr['sr']:.4f} sr0={dsr['sr0']:.4f} dsr={dsr['dsr']:.4f} "
          f"psr_vs_0={dsr['psr_vs_0']:.4f}")
    assert dsr["psr_vs_0"] > 0.9, f"PSR(0) too low: {dsr['psr_vs_0']}"
    out["dsr"] = dsr

    print(f"CVOVERFIT SELFTEST OK  (scipy={'yes' if _HAVE_SCIPY else 'no'})")
    return out


# ===========================================================================
# CLI
# ===========================================================================
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overfitting controls: CPCV, PBO, DSR")
    p.add_argument("--out", default="models/cv_overfitting.json",
                   help="output JSON path")
    p.add_argument("--selftest", action="store_true", help="run self-test and exit")

    p.add_argument("--returns-matrix", default=None,
                   help="parquet/csv: columns = candidate strategies (T x N)")
    p.add_argument("--returns", default=None,
                   help="parquet/csv with return series for DSR")
    p.add_argument("--returns-col", default=None,
                   help="column name to use from --returns")
    p.add_argument("--pbo-S", type=int, default=16, help="number of CSCV submatrices")
    p.add_argument("--n-trials", type=int, default=None, help="n_trials for DSR")

    # CPCV
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--n-groups", type=int, default=6)
    p.add_argument("--k-test", type=int, default=2)
    p.add_argument("--horizon", type=int, default=0)
    p.add_argument("--embargo", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    cpcv_args: Optional[Dict[str, int]] = None
    if args.n_samples is not None:
        cpcv_args = {
            "n_samples": args.n_samples,
            "n_groups": args.n_groups,
            "k_test": args.k_test,
            "horizon": args.horizon,
            "embargo": args.embargo,
        }

    perf_matrix: Optional[np.ndarray] = None
    if args.returns_matrix:
        df = _load_table(args.returns_matrix)
        perf_matrix = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    returns: Optional[np.ndarray] = None
    if args.returns:
        df = _load_table(args.returns)
        if args.returns_col:
            returns = df[args.returns_col].to_numpy(dtype=float)
        else:
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] < 1:
                raise ValueError("no numeric column found in --returns")
            returns = num.iloc[:, 0].to_numpy(dtype=float)

    report = build_report(
        cpcv_args=cpcv_args,
        perf_matrix=perf_matrix,
        pbo_S=args.pbo_S,
        returns=returns,
        n_trials=args.n_trials,
    )

    if not report:
        print("No sections requested. Use --selftest or supply inputs "
              "(--n-samples, --returns-matrix, --returns).")
        return 1

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote report -> {args.out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

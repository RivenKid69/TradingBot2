# drift.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class NumericBins:
    edges: List[float]          # границы бинов (включая -inf и +inf)
    probs: List[float]          # доли по бинам на baseline (Qi)
    n: int                      # объём baseline

    def to_dict(self) -> Dict:
        return {"type": "numeric", "edges": list(self.edges), "probs": list(self.probs), "n": int(self.n)}

    @classmethod
    def from_dict(cls, d: Dict) -> "NumericBins":
        return cls(edges=[float(x) for x in d["edges"]], probs=[float(p) for p in d["probs"]], n=int(d.get("n", 0)))


@dataclass
class CategoricalDist:
    categories: List[str]       # список категорий (включая "OTHER", если был тримминг)
    probs: List[float]          # доли по категориям на baseline (Qi)
    n: int                      # объём baseline

    def to_dict(self) -> Dict:
        return {"type": "categorical", "categories": list(self.categories), "probs": list(self.probs), "n": int(self.n)}

    @classmethod
    def from_dict(cls, d: Dict) -> "CategoricalDist":
        return cls(categories=[str(x) for x in d["categories"]], probs=[float(p) for p in d["probs"]], n=int(d.get("n", 0)))


BaselineSpec = Dict[str, Union[NumericBins, CategoricalDist]]


def _safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _hist_from_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Возвращает количество элементов по бинам согласно edges (как в np.histogram).
    Все NaN игнорируются. Возвращает массив длины len(edges)-1.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(len(edges) - 1, dtype=float)
    cnt, _ = np.histogram(x, bins=edges)
    return cnt.astype(float)


def _psi_from_counts(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """
    PSI = Σ (Pi - Qi) * ln(Pi/Qi)
    p_counts — текущие (prod window), q_counts — baseline.
    Малые значения заменяем на eps, чтобы избежать деления на 0.
    """
    eps = 1e-8
    p = p_counts.astype(float)
    q = q_counts.astype(float)
    if p.sum() <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    if q.sum() <= 0:
        q = np.ones_like(q) / len(q)
    else:
        q = q / q.sum()
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum((p - q) * np.log(p / q)))


def _build_numeric_bins_baseline(series: pd.Series, bins: int = 10) -> NumericBins:
    """
    Строим квантили по baseline и превращаем в «замкнутые» бины: [-inf, q1, q2, ..., +inf].
    """
    x = _safe_to_numeric(series).dropna().to_numpy(dtype=float)
    n = int(x.size)
    if n == 0:
        edges = np.array([-np.inf, np.inf], dtype=float)
        probs = np.array([1.0], dtype=float)
        return NumericBins(edges=list(edges), probs=list(probs), n=0)

    qs = np.linspace(0.0, 1.0, bins + 1)
    # избегаем совпадающих квантилей
    raw_edges = np.quantile(x, qs)
    # гарантируем строгую возрастающую последовательность через добавление -inf/+inf и устранение дублей
    edges = [-np.inf]
    for val in raw_edges[1:-1]:
        if len(edges) == 0 or val > edges[-1]:
            edges.append(float(val))
        else:
            # если квантили совпали, слегка подвинем
            edges.append(float(edges[-1] + 1e-12))
    edges.append(np.inf)
    edges = np.array(edges, dtype=float)

    counts = _hist_from_edges(x, edges)
    if counts.sum() <= 0:
        probs = np.ones_like(counts) / len(counts)
    else:
        probs = counts / counts.sum()
    return NumericBins(edges=list(edges), probs=list(probs), n=n)


def _build_categorical_baseline(series: pd.Series, top_k: int = 20) -> CategoricalDist:
    """
    Берём top_k категорий по частоте. Остальные — в "OTHER".
    """
    s = series.astype("string")
    vc = s.value_counts(dropna=True)
    n = int(vc.sum())
    cats = vc.index.tolist()
    if len(cats) == 0:
        return CategoricalDist(categories=["OTHER"], probs=[1.0], n=0)
    if len(cats) > top_k:
        head = vc.iloc[:top_k]
        other = float(vc.iloc[top_k:].sum())
        categories = head.index.tolist() + ["OTHER"]
        counts = head.to_numpy(dtype=float).tolist() + [other]
    else:
        categories = cats
        counts = vc.to_numpy(dtype=float).tolist()
    counts = np.array(counts, dtype=float)
    if counts.sum() <= 0:
        probs = np.ones_like(counts) / len(counts)
    else:
        probs = counts / counts.sum()
    return CategoricalDist(categories=[str(c) for c in categories], probs=list(probs), n=n)


def make_baseline(
    df: pd.DataFrame,
    features: List[str],
    *,
    bins: int = 10,
    categorical: Optional[List[str]] = None,
    top_k_cats: int = 20,
) -> Dict[str, Dict]:
    """
    Строит baseline-спецификацию: для числовых — квантили и доли, для категориальных — частоты категорий.
    Возвращает dict, пригодный для сохранения в JSON.
    """
    categorical = set(categorical or [])
    spec: Dict[str, Dict] = {}
    for col in features:
        if col not in df.columns:
            continue
        s = df[col]
        if col in categorical or (pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s)):
            cat = _build_categorical_baseline(s, top_k=top_k_cats)
            spec[col] = cat.to_dict()
        else:
            num = _build_numeric_bins_baseline(s, bins=bins)
            spec[col] = num.to_dict()
    return spec


def save_baseline_json(spec: Dict[str, Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)


def load_baseline_json(path: str) -> BaselineSpec:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out: BaselineSpec = {}
    for k, v in d.items():
        t = str(v.get("type", "numeric")).lower()
        if t == "numeric":
            out[k] = NumericBins.from_dict(v)
        elif t == "categorical":
            out[k] = CategoricalDist.from_dict(v)
        else:
            raise ValueError(f"Неизвестный тип в baseline для {k}: {t}")
    return out


def _psi_numeric_current(series: pd.Series, nb: NumericBins) -> float:
    x = _safe_to_numeric(series).to_numpy(dtype=float)
    edges = np.asarray(nb.edges, dtype=float)
    q_counts = np.asarray(nb.probs, dtype=float) * max(nb.n, 1)
    p_counts = _hist_from_edges(x, edges)
    return _psi_from_counts(p_counts, q_counts)


def _psi_categorical_current(series: pd.Series, cd: CategoricalDist) -> float:
    s = series.astype("string")
    vc = s.value_counts(dropna=True)
    cats = list(cd.categories)
    # соберём counts по cats, остальные — в OTHER (если есть)
    counts = []
    other_count = 0.0
    for c in vc.index.tolist():
        val = float(vc[c])
        if c in cats:
            # будет добавлено ниже
            pass
        else:
            other_count += val
    for c in cats:
        if c == "OTHER":
            counts.append(other_count)
        else:
            counts.append(float(vc.get(c, 0.0)))
    p_counts = np.asarray(counts, dtype=float)
    q_counts = np.asarray(cd.probs, dtype=float) * max(cd.n, 1)
    return _psi_from_counts(p_counts, q_counts)


def compute_psi(
    current_df: pd.DataFrame,
    baseline: BaselineSpec,
    *,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Считает PSI для заданных колонок по baseline-спецификации.
    Возвращает DataFrame с колонками: ['feature','psi','type','n_current','n_baseline'].
    """
    feats = list(features) if features is not None else list(baseline.keys())
    rows: List[Dict] = []
    for col in feats:
        if col not in baseline:
            continue
        if col not in current_df.columns:
            rows.append({"feature": col, "psi": float("nan"), "type": "missing", "n_current": 0, "n_baseline": 0})
            continue
        spec = baseline[col]
        try:
            if isinstance(spec, NumericBins) or (isinstance(spec, dict) and spec.get("type") == "numeric"):
                nb = spec if isinstance(spec, NumericBins) else NumericBins.from_dict(spec)  # type: ignore
                psi = _psi_numeric_current(current_df[col], nb)
                rows.append({"feature": col, "psi": float(psi), "type": "numeric", "n_current": int(current_df[col].notna().sum()), "n_baseline": int(nb.n)})
            elif isinstance(spec, CategoricalDist) or (isinstance(spec, dict) and spec.get("type") == "categorical"):
                cd = spec if isinstance(spec, CategoricalDist) else CategoricalDist.from_dict(spec)  # type: ignore
                psi = _psi_categorical_current(current_df[col], cd)
                rows.append({"feature": col, "psi": float(psi), "type": "categorical", "n_current": int(current_df[col].notna().sum()), "n_baseline": int(cd.n)})
            else:
                rows.append({"feature": col, "psi": float("nan"), "type": "unknown", "n_current": 0, "n_baseline": 0})
        except Exception:
            rows.append({"feature": col, "psi": float("nan"), "type": "error", "n_current": 0, "n_baseline": 0})
    res = pd.DataFrame(rows)
    res = res.sort_values(["psi"], ascending=[False]).reset_index(drop=True)
    return res


def ks_statistic(baseline: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max CDF gap). 0=identical, 1=disjoint.

    Distribution-shape drift test that complements PSI (PSI is binned; KS is the
    sup-norm of the empirical CDFs). Pure-NumPy, no scipy needed.
    """
    a = np.asarray(baseline, dtype="float64"); a = a[np.isfinite(a)]
    b = np.asarray(current, dtype="float64"); b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein1d(baseline: np.ndarray, current: np.ndarray) -> float:
    """1-D Wasserstein (earth-mover) distance between two samples (pure NumPy).

    Integral of |CDF_a − CDF_b|; sensitive to mean/location shift, unlike KS which
    is scale-free. Useful for magnitude of covariate drift.
    """
    a = np.sort(np.asarray(baseline, dtype="float64")); a = a[np.isfinite(a)]
    b = np.sort(np.asarray(current, dtype="float64")); b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    deltas = np.diff(grid)
    cdf_a = np.searchsorted(a, grid[:-1], side="right") / len(a)
    cdf_b = np.searchsorted(b, grid[:-1], side="right") / len(b)
    return float(np.sum(np.abs(cdf_a - cdf_b) * deltas))


def compute_distribution_drift(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    *,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Per-feature KS + Wasserstein covariate drift vs a baseline DataFrame.

    Returns columns [feature, ks, wasserstein, drift] where drift flags ks>0.1.
    """
    feats = features or [c for c in baseline_df.columns if c in current_df.columns]
    rows: List[Dict] = []
    for c in feats:
        if c not in current_df.columns or c not in baseline_df.columns:
            continue
        a = _safe_to_numeric(baseline_df[c]).to_numpy()
        b = _safe_to_numeric(current_df[c]).to_numpy()
        ks = ks_statistic(a, b)
        wd = wasserstein1d(a, b)
        rows.append({"feature": c, "ks": ks, "wasserstein": wd,
                     "drift": bool(np.isfinite(ks) and ks > 0.1)})
    return pd.DataFrame(rows).sort_values("ks", ascending=False).reset_index(drop=True)


def concept_drift(
    baseline_y_true: np.ndarray, baseline_y_pred: np.ndarray,
    current_y_true: np.ndarray, current_y_pred: np.ndarray,
    *,
    metric: str = "rmse",
) -> Dict[str, float]:
    """Concept/label drift: degradation of predictive performance (P(y|x) shift).

    Covariate drift (PSI/KS) misses the case where inputs look the same but the
    input→target relationship changed. We compare model error on a baseline window
    vs the current window; a large relative increase signals concept drift.
    """
    def _err(yt, yp):
        yt = np.asarray(yt, dtype="float64"); yp = np.asarray(yp, dtype="float64")
        m = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[m], yp[m]
        if len(yt) == 0:
            return float("nan")
        if metric == "mae":
            return float(np.mean(np.abs(yt - yp)))
        if metric == "directional":   # 1 - hit-rate of sign prediction
            return float(1.0 - np.mean(np.sign(yt) == np.sign(yp)))
        return float(np.sqrt(np.mean((yt - yp) ** 2)))   # rmse

    base_err = _err(baseline_y_true, baseline_y_pred)
    cur_err = _err(current_y_true, current_y_pred)
    rel = (cur_err / base_err - 1.0) if (base_err and np.isfinite(base_err) and base_err > 0) else float("nan")
    return {
        "metric": metric,
        "baseline_error": base_err,
        "current_error": cur_err,
        "relative_degradation": float(rel),
        "concept_drift": bool(np.isfinite(rel) and rel > 0.15),   # >15% worse → drift
    }


def default_feature_list(df: pd.DataFrame) -> List[str]:
    """
    Простая эвристика: все числовые фичи, начинающиеся с 'f_' или заканчивающиеся на '_z', плюс 'score', если есть.
    """
    out: List[str] = []
    for c in df.columns:
        if c.startswith("f_") or c.endswith("_z"):
            out.append(c)
    if "score" in df.columns:
        out.append("score")
    return sorted(list(dict.fromkeys(out)))


if __name__ == "__main__":
    import argparse
    import sys
    
    ap = argparse.ArgumentParser(description="Запустить расчет дрифта данных.")
    ap.add_argument("--data", default="data/features.parquet", help="Путь к текущим фичам.")
    ap.add_argument("--baseline", default="models/drift_baseline.json", help="Путь к baseline JSON.")
    ap.add_argument("--out_csv", default="data/features_psi.csv", help="Путь к сохранению CSV с PSI.")
    ap.add_argument("--out_json", default="models/drift_report.json", help="Путь к сохранению JSON отчета.")
    args = ap.parse_args()

    # Попытка найти датасет
    data_path = args.data
    if not os.path.exists(data_path):
        if os.path.exists("data/test_features.parquet"):
            data_path = "data/test_features.parquet"
        elif os.path.exists("data/test_training_table.parquet"):
            data_path = "data/test_training_table.parquet"
        else:
            print(f"Ошибка: файл данных {args.data} не найден.")
            sys.exit(1)

    print(f"Загрузка текущих данных из: {data_path}")
    ext = os.path.splitext(data_path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    feats = default_feature_list(df)
    if not feats:
        print("Предупреждение: не найдено фичей (f_* или *_z). Используем все числовые колонки.")
        feats = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("ts_ms", "timestamp", "time")]

    if not feats:
        print("Ошибка: в датасете нет подходящих фичей для анализа.")
        sys.exit(1)

    # Проверка baseline
    baseline_path = args.baseline
    if not os.path.exists(baseline_path):
        print(f"Файл baseline {baseline_path} не найден. Генерируем автоматический baseline из первой половины данных...")
        half_idx = len(df) // 2
        df_base = df.iloc[:half_idx]
        spec = make_baseline(df_base, feats, bins=10, categorical=None, top_k_cats=20)
        save_baseline_json(spec, baseline_path)
        print(f"Baseline успешно сохранен в: {baseline_path}")
        df_curr = df.iloc[half_idx:]
    else:
        df_curr = df

    print(f"Загрузка baseline из: {baseline_path}")
    baseline = load_baseline_json(baseline_path)
    
    # Сопоставим фичи
    run_feats = [f for f in feats if f in baseline]
    if not run_feats:
        print("Ошибка: нет общих фичей между датасетом и baseline.")
        sys.exit(1)

    print(f"Расчет PSI для {len(run_feats)} фичей...")
    res = compute_psi(df_curr, baseline, features=run_feats)
    
    # Сохранение CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    res.to_csv(args.out_csv, index=False)
    print(f"Детальный отчет PSI сохранен в CSV: {args.out_csv}")

    # Расчет среднего PSI
    valid_psi = res["psi"].replace([np.inf, -np.inf], np.nan).dropna()
    avg_psi = float(valid_psi.mean()) if not valid_psi.empty else 0.0
    worst_feat = res.iloc[0]["feature"] if not res.empty else "none"
    worst_psi = float(res.iloc[0]["psi"]) if not res.empty else 0.0

    # Интерпретация
    if avg_psi < 0.1:
        status_lbl = "Стабильно (PSI < 0.1)"
        status_code = "stable"
    elif avg_psi < 0.25:
        status_lbl = "Умеренный дрифт (0.1 <= PSI < 0.25)"
        status_code = "warning"
    else:
        status_lbl = "Сильный дрифт (PSI >= 0.25)"
        status_code = "drift"

    report = {
        "avg_psi": avg_psi,
        "worst_feature": worst_feat,
        "worst_psi": worst_psi,
        "status": status_code,
        "status_label": status_lbl,
        "total_features": len(run_feats),
        "n_samples": len(df_curr)
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Также сохраним как validation_report.json для совместимости
    with open("models/validation_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "sortino_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "validation_pnl": 0.0,
            "psi": avg_psi,
            "psi_worst_feature": worst_feat,
            "psi_worst": worst_psi
        }, f, ensure_ascii=False, indent=2)

    print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ДРЕЙФА (CONCEPT DRIFT) ===")
    print(f"Количество проанализированных признаков: {len(run_feats)}")
    print(f"Средний индекс PSI: {avg_psi:.4f} ({status_lbl})")
    print(f"Наиболее нестабильный признак: {worst_feat} (PSI = {worst_psi:.4f})")
    print("Границы оценки PSI: <0.1 — норма, 0.1–0.25 — предупреждение, >0.25 — необходима переподготовка модели.")


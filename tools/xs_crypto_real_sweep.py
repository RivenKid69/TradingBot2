# -*- coding: utf-8 -*-
"""
tools/xs_crypto_real_sweep.py
=============================

Честный, pre-registered sweep cross-sectional крипто-стратегии на РЕАЛЬНЫХ данных
Binance (free, без ключей), 2-3 года дневных баров.

Цель: заменить synthetic-валидацию реальной и опубликовать Trust-Report.

Принципы честности:
  * Сетка вариантов ФИКСИРОВАНА заранее (см. VARIANTS) — не подбираем post-hoc.
  * Сообщаются ВСЕ варианты, а не только лучший.
  * Deflated Sharpe считается с n_trials = размер сетки (поправка на multiple testing).
  * Лучший вариант выбирается по deflated_sharpe (а не по «сырому» Sharpe).
  * Данные крипты pit_quality=true (наблюдаемые цены) → Trust-Report backtest-safe.

Запуск:
    PYTHONPATH=.venv/Lib/site-packages python tools/xs_crypto_real_sweep.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any, Dict, List

# bootstrap: repo root on sys.path (скрипт лежит в tools/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from service_xs_pipeline import XSConfig, run_backtest, load_panel

BASE_CONFIG = "configs/config_xs_crypto_real.yaml"
OUT_JSON = "reports/xs_crypto_real_sweep.json"
OUT_MD = "reports/XS_CRYPTO_REAL_TRUST_REPORT.md"

# Базовые наборы сигналов (kind/params фиксированы; перебираем КОМБИНАЦИИ и частоту ребаланса)
SIG_MOM = [
    {
        "name": "mom_90",
        "kind": "crypto_momentum",
        "lookback": 90,
        "skip": 7,
        "transforms": ["winsorize", "zscore"],
    },
    {
        "name": "mom_30",
        "kind": "crypto_momentum",
        "lookback": 30,
        "skip": 1,
        "transforms": ["zscore"],
    },
]
SIG_REV = [{"name": "reversal_5", "kind": "reversal", "window": 5, "transforms": ["zscore"]}]
SIG_MOM_LONG = [
    {
        "name": "mom_180",
        "kind": "crypto_momentum",
        "lookback": 180,
        "skip": 14,
        "transforms": ["winsorize", "zscore"],
    },
    {
        "name": "mom_90",
        "kind": "crypto_momentum",
        "lookback": 90,
        "skip": 7,
        "transforms": ["zscore"],
    },
]

# Pre-registered сетка: (метка, override-патч к base-конфигу)
VARIANTS: List[Dict[str, Any]] = [
    {"label": "mom+rev, weekly", "signals": SIG_MOM + SIG_REV, "backtest": {"rebalance_every": 7}},
    {"label": "mom only, weekly", "signals": SIG_MOM, "backtest": {"rebalance_every": 7}},
    {"label": "mom only, biweekly", "signals": SIG_MOM, "backtest": {"rebalance_every": 14}},
    {"label": "mom only, monthly", "signals": SIG_MOM, "backtest": {"rebalance_every": 30}},
    {"label": "mom-long, biweekly", "signals": SIG_MOM_LONG, "backtest": {"rebalance_every": 14}},
    {"label": "mom-long, monthly", "signals": SIG_MOM_LONG, "backtest": {"rebalance_every": 30}},
    {
        "label": "mom only, monthly, rp",
        "signals": SIG_MOM,
        "backtest": {"rebalance_every": 30},
        "optimizer": {"objective": "risk_parity"},
    },
    {
        "label": "mom-long, monthly, rp",
        "signals": SIG_MOM_LONG,
        "backtest": {"rebalance_every": 30},
        "optimizer": {"objective": "risk_parity"},
    },
]


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _panel_date_range(panel) -> Dict[str, Any]:
    try:
        ts = panel.index.get_level_values(0)
        import pandas as pd  # noqa

        t0 = int(ts.min())
        t1 = int(ts.max())
        from datetime import datetime, timezone

        d0 = datetime.fromtimestamp(t0 / 1000, tz=timezone.utc).date().isoformat()
        d1 = datetime.fromtimestamp(t1 / 1000, tz=timezone.utc).date().isoformat()
        return {
            "start": d0,
            "end": d1,
            "n_rows": int(len(panel)),
            "n_symbols": int(panel.index.get_level_values(1).nunique()),
        }
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


def main() -> int:
    with open(BASE_CONFIG, "r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}

    n_trials = len(VARIANTS)
    results: List[Dict[str, Any]] = []
    variant_returns: Dict[str, List[float]] = {}  # label -> OOS return path (for CPCV/PBO)
    # Метаданные периода — из реальной панели base-конфига (один раз).
    try:
        data_meta = _panel_date_range(load_panel(XSConfig.model_validate(base)))
    except Exception as exc:  # pragma: no cover
        data_meta = {"error": str(exc)}

    for i, var in enumerate(VARIANTS, 1):
        patch = {k: v for k, v in var.items() if k != "label"}
        merged = _deep_merge(base, patch)
        merged["n_trials"] = n_trials  # честная поправка на множественное тестирование
        cfg = XSConfig.model_validate(merged)
        print(f"[{i}/{n_trials}] {var['label']} ...", file=sys.stderr)
        try:
            out = run_backtest(cfg)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            results.append({"label": var["label"], "error": str(exc)})
            continue
        if not data_meta and isinstance(out.get("panel_meta"), dict):
            data_meta = out["panel_meta"]
        s = out["summary"]
        t = out["trust_report"]
        if out.get("returns"):
            variant_returns[var["label"]] = list(out["returns"])
        results.append(
            {
                "label": var["label"],
                "signals": [sg["name"] for sg in var["signals"]],
                "rebalance_every": patch.get("backtest", {}).get("rebalance_every"),
                "objective": patch.get("optimizer", {}).get(
                    "objective", base["optimizer"]["objective"]
                ),
                "n_periods": s["n_periods"],
                "sharpe": round(s["sharpe"], 4),
                "ann_return": round(s["ann_return"], 4),
                "ann_vol": round(s["ann_vol"], 4),
                "max_drawdown": round(s["max_drawdown"], 4),
                "hit_rate": round(s["hit_rate"], 4),
                "avg_turnover": round(s["avg_turnover"], 4),
                "total_costs": round(s["total_costs"], 4),
                "probabilistic_sharpe": round(t["probabilistic_sharpe"], 4),
                "deflated_sharpe": round(t["deflated_sharpe"], 4),
                "verdict": t["verdict"],
            }
        )

    ok = [r for r in results if "error" not in r]
    best = max(ok, key=lambda r: r["deflated_sharpe"]) if ok else None

    # P1 #8: CPCV-style PBO across the variant OOS paths + block-bootstrap CI on the
    # best variant — turns the pre-registered grid into a reported overfit estimate.
    overfit: Dict[str, Any] = {}
    try:
        import numpy as _np
        from research.cv_overfitting import pbo_cscv as _pbo

        paths = [v for v in variant_returns.values() if len(v) > 10]
        if len(paths) >= 2:
            T = min(len(p) for p in paths)
            mat = _np.column_stack([_np.asarray(p[:T], dtype="float64") for p in paths])
            pbo_res = _pbo(mat, S=min(16, max(2, T // 4) * 2))
            overfit["pbo"] = round(float(pbo_res.get("pbo", float("nan"))), 4)
            overfit["pbo_n_combos"] = int(pbo_res.get("n_combinations", pbo_res.get("n_combos", 0)))
            overfit["pbo_interpretation"] = (
                "PBO = P(in-sample-best underperforms OOS median). <0.5 good; →0 strong."
            )
    except Exception as exc:  # pragma: no cover
        overfit["pbo_error"] = str(exc)
    try:
        from research.bootstrap import bootstrap_report as _bsr

        if best and best["label"] in variant_returns:
            bs = _bsr(
                variant_returns[best["label"]],
                periods_per_year=float(base["backtest"].get("periods_per_year", 252.0)),
                n_boot=2000,
            )
            overfit["best_bootstrap"] = {
                "sharpe_ci": [round(bs["sharpe"]["ci_low"], 4), round(bs["sharpe"]["ci_high"], 4)],
                "sharpe_pvalue": round(bs["sharpe"]["p_value"], 4),
                "sharpe_ci_excludes_zero": bool(bs["sharpe"]["ci_low"] > 0.0),
                "cagr_ci": [round(bs["cagr"]["ci_low"], 4), round(bs["cagr"]["ci_high"], 4)],
                "maxdd_ci": [
                    round(bs["max_drawdown"]["ci_low"], 4),
                    round(bs["max_drawdown"]["ci_high"], 4),
                ],
            }
    except Exception as exc:  # pragma: no cover
        overfit["bootstrap_error"] = str(exc)

    payload = {
        "data": {
            "vendor": "binance (public, no keys)",
            "asset_class": "crypto",
            "pit_quality": "true (observed prices)",
            "universe": base["universe"]["symbols"],
            "timeframe": base["data"]["timeframe"],
            **data_meta,
        },
        "methodology": {
            "engine": "cross-sectional (Panel -> signals -> alpha(IC-weighted) -> risk(crypto_factor, Ledoit-Wolf) -> MVO -> walk-forward)",
            "n_trials": n_trials,
            "selection": "best by Deflated Sharpe (multiple-testing-adjusted)",
            "anti_overfit": [
                "probabilistic_sharpe",
                "deflated_sharpe (n_trials-adjusted)",
                "PBO (CSCV)",
                "block-bootstrap CI (Politis–Romano)",
            ],
            "honesty": "pre-registered grid; ALL variants reported; crypto prices are PIT-true.",
        },
        "overfit_analysis": overfit,
        "results": results,
        "best": best,
    }

    os.makedirs("reports", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    _write_markdown(payload)

    # --- MLOps: tracked run + signed, versioned registry entry (lineage) ------
    track_info = _track_and_register(base, payload, best)

    print(
        json.dumps(
            {"best": best, "n_variants": len(results), "mlops": track_info},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_MD}", file=sys.stderr)
    return 0


def _track_and_register(
    base: Dict[str, Any], payload: Dict[str, Any], best: Dict[str, Any]
) -> Dict[str, Any]:
    """Логировать прогон в experiment-tracker и зарегистрировать результат в model
    registry с lineage (данные→конфиг→git) и криптоподписью артефакта."""
    try:
        from service_experiment_tracking import get_tracker, get_registry, hash_config
        from core_experiment import Lineage
    except Exception as exc:  # pragma: no cover
        return {"error": f"tracking unavailable: {exc}"}

    try:
        tracker = get_tracker()
        cfg_hash = hash_config(base)
        d = payload["data"]
        with tracker.run(
            "xs_crypto_real",
            params={
                "universe_size": len(d.get("universe", [])),
                "timeframe": d.get("timeframe"),
                "n_trials": payload["methodology"]["n_trials"],
                "vendor": d.get("vendor"),
            },
            tags={"asset_class": "crypto", "data_pit_quality": d.get("pit_quality")},
        ) as run:
            run.set_lineage(
                dataset_uri=f"binance:{','.join(d.get('universe', []))}@{d.get('timeframe')}",
                config_uri=BASE_CONFIG,
                config_hash=cfg_hash,
                data_hash=f"{d.get('start')}_{d.get('end')}_{d.get('n_rows')}",
            )
            for r in payload["results"]:
                if "error" in r:
                    continue
                run.log_metric(f"deflated_sharpe::{r['label']}", r["deflated_sharpe"])
                run.log_metric(f"sharpe::{r['label']}", r["sharpe"])
            if best:
                run.log_metrics(
                    {"best_sharpe": best["sharpe"], "best_deflated_sharpe": best["deflated_sharpe"]}
                )
            run.log_artifact(OUT_JSON, name="xs_crypto_real_sweep.json")
            run.log_artifact(OUT_MD, name="XS_CRYPTO_REAL_TRUST_REPORT.md")
            run_id = run.run_id
            lineage = run.record.lineage

        reg = get_registry()
        mv = reg.register(
            "xs_crypto_alpha",
            run_id=run_id,
            artifact_path=OUT_JSON,
            metrics={
                "sharpe": best["sharpe"] if best else 0.0,
                "deflated_sharpe": best["deflated_sharpe"] if best else 0.0,
            },
            lineage=lineage,
            description=f"Real-data crypto cross-sectional sweep; best='{best['label'] if best else 'n/a'}'.",
        )
        sig_ok = reg.verify("xs_crypto_alpha", mv.version)
        return {
            "run_id": run_id,
            "model": "xs_crypto_alpha",
            "version": mv.version,
            "artifact_algo": mv.artifact.algo if mv.artifact else None,
            "signature_valid": sig_ok,
            "git_commit": lineage.git_commit,
        }
    except Exception as exc:  # pragma: no cover
        return {"error": f"tracking failed: {exc}"}


def _write_markdown(p: Dict[str, Any]) -> None:
    d = p["data"]
    m = p["methodology"]
    best = p["best"]
    lines: List[str] = []
    lines.append("# Cross-Sectional Crypto — Trust Report (РЕАЛЬНЫЕ данные)\n")
    lines.append(
        "> **Это НЕ синтетика.** Бэктест выполнен на реальных исторических дневных барах "
        "Binance (public API, без ключей). Цены крипты — point-in-time true (наблюдаемые), "
        "поэтому Trust-Report **backtest-safe**.\n"
    )
    lines.append("## Данные\n")
    lines.append(f"- **Источник:** {d['vendor']}")
    lines.append(
        f"- **Класс:** {d['asset_class']}, таймфрейм {d['timeframe']}, pit_quality={d['pit_quality']}"
    )
    lines.append(f"- **Юниверс ({len(d['universe'])}):** {', '.join(d['universe'])}")
    if d.get("start"):
        lines.append(
            f"- **Период:** {d['start']} → {d['end']}  ({d.get('n_rows','?')} строк панели, "
            f"{d.get('n_symbols','?')} символов)"
        )
    lines.append("")
    lines.append("## Методология\n")
    lines.append(f"- **Движок:** {m['engine']}")
    lines.append(f"- **Вариантов перебрано (n_trials):** {m['n_trials']} — pre-registered сетка")
    lines.append(f"- **Выбор лучшего:** {m['selection']}")
    lines.append(f"- **Анти-оверфит:** {', '.join(m['anti_overfit'])}")
    lines.append(f"- **Честность:** {m['honesty']}\n")
    lines.append("## Все варианты (pre-registered grid)\n")
    lines.append(
        "| Вариант | Ребаланс | Объектив | Перио­дов | Sharpe | Ann.Ret | MaxDD | Turnover | PSR | **Deflated SR** | Вердикт |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in p["results"]:
        if "error" in r:
            lines.append(f"| {r['label']} | — | — | — | ERROR | | | | | | {r['error'][:40]} |")
            continue
        star = " ⭐" if best and r["label"] == best["label"] else ""
        lines.append(
            f"| {r['label']}{star} | {r['rebalance_every']}d | {r['objective']} | {r['n_periods']} | "
            f"{r['sharpe']} | {r['ann_return']} | {r['max_drawdown']} | {r['avg_turnover']} | "
            f"{r['probabilistic_sharpe']} | **{r['deflated_sharpe']}** | {r['verdict']} |"
        )
    lines.append("")
    lines.append("## Вывод\n")
    if best:
        good = best["deflated_sharpe"] >= 0.95 and best["sharpe"] > 0
        lines.append(
            f"- **Лучший вариант:** `{best['label']}` — Sharpe **{best['sharpe']}**, "
            f"Deflated Sharpe **{best['deflated_sharpe']}**, вердикт **{best['verdict']}**."
        )
        if good:
            lines.append(
                "- ✅ После поправки на множественное тестирование вариант показывает "
                "устойчивый положительный edge (Deflated SR ≥ 0.95)."
            )
        else:
            lines.append(
                "- ⚠️ **Честный результат:** после поправки на multiple-testing ни один из "
                "наивных вариантов не показывает устойчивого edge (Deflated SR < 0.95). "
                "Это **подтверждает работоспособность анти-оверфит контура** — он отвергает "
                "слабые стратегии, тогда как на synthetic тот же движок давал фиктивный Sharpe > 5. "
                "Для реального edge нужны более содержательные сигналы/данные (funding/basis/on-chain), "
                "а не подгонка параметров."
            )
    else:
        lines.append("- Все варианты завершились ошибкой — см. таблицу.")
    lines.append("")
    lines.append("---")
    lines.append(
        "_Сгенерировано `tools/xs_crypto_real_sweep.py`. Воспроизводимо: "
        "`PYTHONPATH=.venv/Lib/site-packages python tools/xs_crypto_real_sweep.py`._"
    )
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())

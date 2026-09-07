# -*- coding: utf-8 -*-
"""
service_tearsheet.py
====================

LP-grade rendered tear-sheet (P1 #10). The platform already produces a JSON
tear-sheet (`service_attribution.tear_sheet`) but nothing rendered. This module
renders a self-contained **HTML** tear-sheet (no external libs — browser-printable
to PDF) from a ``run_backtest`` output, including:

  * headline metrics: Sharpe, Sortino, Calmar, max drawdown, hit-rate;
  * benchmark-relative: information ratio, tracking error, beta, alpha;
  * **GIPS-style gross-vs-net** dual presentation (gross = net + costs);
  * equity curve + drawdown (inline SVG), period-return bars;
  * factor P&L attribution (tied to the fitted risk model);
  * capacity curve (AUM → Sharpe) + capacity AUM;
  * anti-overfit Trust Report (Deflated Sharpe, PSR, PBO);
  * an honest SYNTHETIC banner when the data is not real.

Slой ``service_`` (depends on core/impl only).
"""

from __future__ import annotations

import html
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core_xs_results import compute_metrics


def _fmt(x: Any, pct: bool = False, dp: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v * 100:.{dp}f}%" if pct else f"{v:.{dp}f}"


def _svg_line(
    values: Sequence[float],
    *,
    w: int = 720,
    h: int = 180,
    color: str = "#3b82f6",
    fill: Optional[str] = None,
    baseline: Optional[float] = None,
) -> str:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return f'<svg width="{w}" height="{h}"></svg>'
    lo, hi = min(vals), max(vals)
    if baseline is not None:
        lo, hi = min(lo, baseline), max(hi, baseline)
    rng = (hi - lo) or 1.0
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = i / (n - 1) * (w - 8) + 4
        y = h - 4 - (v - lo) / rng * (h - 8)
        pts.append((x, y))
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    out = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none">']
    if baseline is not None:
        by = h - 4 - (baseline - lo) / rng * (h - 8)
        out.append(
            f'<line x1="4" y1="{by:.1f}" x2="{w-4}" y2="{by:.1f}" stroke="#9ca3af" stroke-dasharray="3,3" stroke-width="1"/>'
        )
    if fill:
        area = path + f" L {pts[-1][0]:.1f},{h-4} L {pts[0][0]:.1f},{h-4} Z"
        out.append(f'<path d="{area}" fill="{fill}" opacity="0.15"/>')
    out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
    out.append("</svg>")
    return "".join(out)


def _bars(values: Sequence[float], *, w: int = 720, h: int = 140) -> str:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return f'<svg width="{w}" height="{h}"></svg>'
    mx = max(abs(min(vals)), abs(max(vals))) or 1.0
    n = len(vals)
    bw = max(1.0, (w - 8) / n - 1)
    mid = h / 2
    out = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    out.append(f'<line x1="4" y1="{mid}" x2="{w-4}" y2="{mid}" stroke="#9ca3af" stroke-width="1"/>')
    for i, v in enumerate(vals):
        x = 4 + i / n * (w - 8)
        bh = abs(v) / mx * (h / 2 - 4)
        y = mid - bh if v >= 0 else mid
        col = "#10b981" if v >= 0 else "#ef4444"
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}"/>'
        )
    out.append("</svg>")
    return "".join(out)


def _metric_card(label: str, value: str, *, good: Optional[bool] = None) -> str:
    color = "#e5e7eb"
    if good is True:
        color = "#10b981"
    elif good is False:
        color = "#ef4444"
    return (
        f'<div class="card"><div class="k">{html.escape(label)}</div>'
        f'<div class="v" style="color:{color}">{html.escape(value)}</div></div>'
    )


def render_html_tearsheet(
    out: Dict[str, Any], *, title: str = "Cross-Sectional Strategy — Tear Sheet"
) -> str:
    """Render an HTML tear-sheet from a ``run_backtest`` output dict."""
    res = out.get("result")
    summary: Dict[str, Any] = dict(out.get("summary") or {})
    trust: Dict[str, Any] = dict(out.get("trust_report") or {})
    fa: Dict[str, Any] = dict(out.get("factor_attribution") or {})
    cap: Dict[str, Any] = dict(out.get("capacity") or {})
    is_syn = bool(out.get("is_synthetic"))

    # equity / drawdown / gross-net (GIPS-style)
    nav_vals: List[float] = []
    dd_vals: List[float] = []
    ret_vals: List[float] = []
    gross_summary: Dict[str, Any] = {}
    if res is not None and getattr(res, "nav", None) is not None and len(res.nav):
        nav = res.nav.to_numpy(dtype="float64")
        nav_vals = list(nav)
        peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
        dd_vals = list(nav / peak - 1.0)
        ret_vals = list(res.returns.to_numpy(dtype="float64"))
        # GIPS gross = net + costs
        try:
            costs = res.costs.reindex(res.returns.index).fillna(0.0)
            gross_ret = res.returns + costs
            gross_summary = compute_metrics(
                gross_ret,
                periods_per_year=float(
                    (res.meta or {}).get("config", {}).get("periods_per_year", 252.0)
                ),
            )
        except Exception:
            gross_summary = {}

    def g(key: str, pct: bool = False, dp: int = 2) -> str:
        return _fmt(summary.get(key), pct=pct, dp=dp)

    cards = [
        _metric_card("Sharpe (net)", g("sharpe"), good=(summary.get("sharpe") or 0) > 1),
        _metric_card("Sortino", g("sortino")),
        _metric_card("Calmar", g("calmar")),
        _metric_card("Ann. return", g("ann_return", pct=True)),
        _metric_card("Ann. vol", g("ann_vol", pct=True)),
        _metric_card("Max drawdown", g("max_drawdown", pct=True), good=False),
        _metric_card("Hit rate", g("hit_rate", pct=True)),
        _metric_card("Avg turnover", _fmt(summary.get("avg_turnover"), pct=True)),
    ]
    # benchmark-relative
    if "information_ratio" in summary:
        cards += [
            _metric_card("Info ratio", g("information_ratio")),
            _metric_card("Tracking error", g("tracking_error", pct=True)),
            _metric_card("Beta", g("beta")),
            _metric_card("Alpha (ann)", g("alpha", pct=True)),
        ]

    # GIPS gross/net dual table
    gips_rows = ""
    if gross_summary:
        for k, lbl in [
            ("ann_return", "Annualized return"),
            ("sharpe", "Sharpe"),
            ("max_drawdown", "Max drawdown"),
        ]:
            pct = k != "sharpe"
            gips_rows += (
                f"<tr><td>{lbl}</td><td>{_fmt(gross_summary.get(k), pct=pct)}</td>"
                f"<td>{_fmt(summary.get(k), pct=pct)}</td></tr>"
            )

    # factor attribution table
    fa_rows = ""
    for f, v in (fa.get("factor_pnl") or {}).items():
        fa_rows += f"<tr><td>{html.escape(str(f))}</td><td>{_fmt(v, pct=True)}</td></tr>"
    if fa:
        fa_rows += f'<tr class="tot"><td>specific</td><td>{_fmt(fa.get("specific_pnl"), pct=True)}</td></tr>'
        fa_rows += (
            f'<tr class="tot"><td>total</td><td>{_fmt(fa.get("total_pnl"), pct=True)}</td></tr>'
        )

    # capacity
    cap_curve = cap.get("curve") or []
    cap_sharpes = [p.get("sharpe") for p in cap_curve]
    cap_rows = ""
    for p in cap_curve:
        cap_rows += (
            f"<tr><td>${p.get('aum',0):,.0f}</td><td>{_fmt(p.get('sharpe'))}</td>"
            f"<td>{_fmt(p.get('avg_cost_bps'), dp=1)} bps</td></tr>"
        )

    # trust report
    trust_rows = ""
    for k, lbl, pct in [
        ("sharpe_annual", "Sharpe (annual)", False),
        ("probabilistic_sharpe", "Probabilistic Sharpe", False),
        ("deflated_sharpe", "Deflated Sharpe", False),
        ("pbo", "PBO (overfit prob.)", False),
        ("n_trials", "Trials tested", False),
    ]:
        if k in trust:
            trust_rows += f"<tr><td>{lbl}</td><td>{_fmt(trust.get(k))}</td></tr>"
    verdict = trust.get("verdict") or trust.get("trust_verdict")

    syn_banner = ""
    if is_syn:
        syn_banner = (
            '<div class="banner">⚠️ SYNTHETIC DATA — these results are a demo and '
            "NOT a real edge. Do not present as performance.</div>"
        )

    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0b0b0f;color:#e5e7eb;margin:0;padding:24px}
    h1{font-size:20px;margin:0 0 4px} h2{font-size:14px;color:#9ca3af;margin:24px 0 8px;text-transform:uppercase;letter-spacing:.05em}
    .sub{color:#9ca3af;font-size:12px;margin-bottom:16px}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
    .card{background:#15151c;border:1px solid #26262f;border-radius:10px;padding:12px}
    .card .k{font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.04em}
    .card .v{font-size:22px;font-weight:600;margin-top:4px}
    .chart{background:#15151c;border:1px solid #26262f;border-radius:10px;padding:12px;margin-top:8px}
    table{width:100%;border-collapse:collapse;background:#15151c;border:1px solid #26262f;border-radius:10px;overflow:hidden}
    th,td{padding:8px 12px;text-align:left;font-size:13px;border-bottom:1px solid #26262f}
    th{color:#9ca3af;font-weight:500;text-transform:uppercase;font-size:11px}
    tr.tot td{font-weight:600;color:#fff}
    .banner{background:rgba(245,158,11,.15);border:1px solid rgba(245,158,11,.5);color:#fbbf24;padding:10px 14px;border-radius:10px;margin-bottom:16px;font-weight:600}
    .two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    @media print{body{background:#fff;color:#111}.card,.chart,table{background:#fff;border-color:#ddd}}
    """
    html_doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{css}</style></head><body>
{syn_banner}
<h1>{html.escape(title)}</h1>
<div class="sub">{int(summary.get('n_rebalances',0))} rebalances · {int(summary.get('n_periods',0))} periods · data_source: {html.escape(str(out.get('data_source','?')))}</div>
<h2>Performance</h2>
<div class="grid">{''.join(cards)}</div>
<h2>Equity curve (net)</h2>
<div class="chart">{_svg_line(nav_vals, color='#3b82f6', fill='#3b82f6', baseline=1.0)}</div>
<h2>Drawdown</h2>
<div class="chart">{_svg_line(dd_vals, color='#ef4444', fill='#ef4444', baseline=0.0)}</div>
<h2>Period returns</h2>
<div class="chart">{_bars(ret_vals)}</div>
<div class="two">
<div><h2>GIPS gross vs net</h2><table><tr><th>Metric</th><th>Gross</th><th>Net</th></tr>{gips_rows or '<tr><td colspan=3>—</td></tr>'}</table></div>
<div><h2>Factor attribution ({html.escape(str(fa.get('risk_model','—')))})</h2><table><tr><th>Factor</th><th>P&amp;L</th></tr>{fa_rows or '<tr><td colspan=2>—</td></tr>'}</table></div>
</div>
<h2>Capacity (AUM → Sharpe after impact)</h2>
<div class="chart">{_svg_line(cap_sharpes, color='#a78bfa')}</div>
<table><tr><th>AUM</th><th>Sharpe</th><th>Avg cost</th></tr>{cap_rows or '<tr><td colspan=3>—</td></tr>'}</table>
<div class="sub" style="margin-top:8px">Capacity AUM (Sharpe ≥ {_fmt(cap.get('sharpe_threshold_frac'))}× base): <b>${cap.get('capacity_aum',0):,.0f}</b></div>
<h2>Trust report (anti-overfit){f' — verdict: {html.escape(str(verdict))}' if verdict else ''}</h2>
<table>{trust_rows or '<tr><td colspan=2>—</td></tr>'}</table>
</body></html>"""
    return html_doc


def render_tearsheet_from_config(cfg: Any) -> str:
    """Run a backtest from an XSConfig and render its HTML tear-sheet."""
    from service_xs_pipeline import run_backtest

    out = run_backtest(cfg)
    return render_html_tearsheet(out)


__all__ = ["render_html_tearsheet", "render_tearsheet_from_config"]

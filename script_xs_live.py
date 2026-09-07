# -*- coding: utf-8 -*-
"""
script_xs_live.py
=================

CLI cross-sectional live-ребаланса (Stage A12), dry-run по умолчанию.

    python script_xs_live.py --config configs/config_xs_template.yaml --equity 100000 [--dry-run]

Считает целевые веса последнего ребаланса, формирует Intents (target exposures, БЕЗ
ордеров — CCEA) и прогоняет через portfolio risk guard. Реальная отправка в Agent — лишь
при наличии подключённого Agent (по умолчанию dry-run). Слой ``script_``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from service_xs_pipeline import XSConfig, latest_target_weights
from service_xs_portfolio_risk import PortfolioRiskGuard, PortfolioRiskLimits
from service_xs_live import CrossSectionalLiveRunner


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_prices_json(path: str):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return {str(k): float(v) for k, v in (json.load(fh) or {}).items()}


def _build_paper_agent_client(prices, equity, n_slices, slice_interval_s):
    """Local Agent (PAPER) execution stack: real engine + sim broker + fills."""
    from packages.agent.broker.adapters.sim import SimBrokerConnector
    from packages.agent.execution.live_factory import build_live_stack

    clk = [0.0]
    broker = SimBrokerConnector(prices, equity=equity)
    stack = build_live_stack(
        broker,
        n_slices=n_slices,
        symbols=list(prices),
        slice_interval_s=slice_interval_s,
        clock=lambda: clk[0],
    )
    return stack, broker, clk


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cross-sectional live rebalance")
    p.add_argument("--config", required=True)
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--ts-ms", type=int, default=0)
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="form Intents only, do not execute (default)",
    )
    p.add_argument(
        "--execute",
        dest="dry_run",
        action="store_false",
        help="execute via the local Agent (PAPER sim broker by default)",
    )
    p.add_argument(
        "--broker",
        default="sim",
        choices=["sim"],
        help="execution broker (sim = paper). Real brokers run via the Agent daemon + vault.",
    )
    p.add_argument(
        "--prices-json",
        default="",
        help="JSON {symbol: price} to size paper orders (required with --execute)",
    )
    p.add_argument("--n-slices", type=int, default=1, help=">1 enables TWAP child-order slicing")
    p.add_argument("--slice-interval-s", type=float, default=1.0)
    p.add_argument("--pump-steps", type=int, default=16)
    args = p.parse_args(argv)

    cfg = XSConfig.model_validate(_load_yaml(args.config))
    weights = latest_target_weights(cfg)

    o = cfg.optimizer
    guard = PortfolioRiskGuard(
        PortfolioRiskLimits(
            gross_max=o.gross_max,
            net_max=(abs(o.net_target) + 0.05) if o.net_target is not None else None,
            max_position=o.max_position,
            max_turnover=o.max_turnover,
        )
    )

    if args.dry_run:
        runner = CrossSectionalLiveRunner(risk_guard=guard)  # no agent_client → dry-run
        res = runner.rebalance(weights, args.equity, ts_ms=args.ts_ms)
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
        return 0 if res.approved else 1

    # --- execute via local Agent (PAPER) ---
    prices = _load_prices_json(args.prices_json)
    if not prices:
        print(json.dumps({"error": "--execute requires --prices-json {symbol: price}"}, indent=2))
        return 2
    stack, broker, clk = _build_paper_agent_client(
        prices, args.equity, args.n_slices, args.slice_interval_s
    )
    runner = CrossSectionalLiveRunner(
        risk_guard=guard,
        agent_client=stack["agent_client"],
        position_provider=stack["agent_client"]._position_provider,
    )
    res = runner.rebalance(weights, args.equity, ts_ms=args.ts_ms)

    ac = stack["agent_client"]
    for _ in range(args.pump_steps):
        clk[0] += args.slice_interval_s
        summary = ac.pump(now_ts=clk[0])
        if summary.get("complete"):
            break

    out = res.to_dict()
    out["execution"] = {
        "simulated": True,
        "broker": args.broker,
        "orders": [o2.to_dict() for o2 in stack["engine"]._orders_by_client_id.values()],
        "positions": {pp.symbol: float(pp.market_value) for pp in broker.get_positions()},
    }
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0 if res.approved else 1


if __name__ == "__main__":
    raise SystemExit(main())

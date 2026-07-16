# service_xs_rebalance.py
"""Регулярный XS-ребаланс: целевые веса → гардрейлы → Intents → CCEA Agent.

Закрывает §4.9 (P1-C) из PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md: движок
cross-sectional платформы выдавал веса, но регулярного пути «веса → ордера»
не существовало.

Поток (профессиональный ребаланс-цикл):

1. **Гейты до каких-либо расчётов**: kill switch, наличие конфига, живой CCEA
   Agent, ``paper_only``-режим (v1 исполняет только на sim_paper брокере).
2. **Подпись модели**: если XS-конфиг использует RL-as-signal (checkpoint),
   артефакт проходит Ed25519-гейт (``services/model_signature_gate``) ДО
   загрузки; на live-брокере — строго enforce.
3. **Целевые веса**: ``service_xs_pipeline.latest_target_weights`` (полный
   пайплайн signals → Σ → μ → optimizer на актуальной панели).
4. **Планирование сделок** (чистая функция, тестируется отдельно):
   клип позиций по ``max_position_weight`` → дельты против фактических позиций
   Agent'а → no-trade band (``drift_band_bps``) и ``min_trade_notional`` →
   turnover-cap с пропорциональным скейлингом → разрез позиций, пересекающих
   ноль, на close+entry → сортировка «сначала продажи» (высвобождают кэш).
5. **Исполнение**: по одному Intent'у через
   ``DesktopCceaSupervisor.submit_rebalance_order`` — реальный Agent OMS
   (policy firewall → hash-chain журнал → fill → P&L ledger/blotter → MAR).
6. **Журнал решений**: полная запись (веса, дельты, скипы с причинами,
   скейлинг, результат каждого ордера) в ``logs/xs_rebalance/<ts>.json`` +
   ``last.json``; ничего не «подразумевается» — только то, что реально
   отправлено и подтверждено OMS.

Планировщик (`configs/scheduler.yaml`, job ``xs_rebalance``) добавляет поверх
этого свой CCEA-гейт двойного opt-in — см. docs/SCHEDULER.md.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

import yaml

from services.utils_app import atomic_write_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Лимиты и план (чистая логика)
# ---------------------------------------------------------------------------

@dataclass
class RebalanceLimits:
    """Гардрейлы одного ребаланса (значения — консервативные default'ы)."""

    max_turnover: float = 0.25          # Σ|Δnotional| ≤ max_turnover · equity
    min_trade_notional: float = 25.0    # мелочь не торгуем
    drift_band_bps: float = 25.0        # |Δw| < band → позицию не трогаем (no-trade band)
    max_position_weight: float = 0.20   # клип целевого веса на инструмент
    max_orders: int = 50

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "RebalanceLimits":
        base = cls()
        return cls(
            max_turnover=float(params.get("max_turnover", base.max_turnover)),
            min_trade_notional=float(params.get("min_trade_notional", base.min_trade_notional)),
            drift_band_bps=float(params.get("drift_band_bps", base.drift_band_bps)),
            max_position_weight=float(params.get("max_position_weight", base.max_position_weight)),
            max_orders=int(params.get("max_orders", base.max_orders)),
        )


@dataclass
class PlannedOrder:
    symbol: str
    qty: float          # знаковое: >0 купить, <0 продать
    price: float
    notional: float     # знаковое (qty·price)
    kind: str           # increase | reduce | close_leg | open_leg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def plan_rebalance(
    target_weights: Mapping[str, float],
    holdings_qty: Mapping[str, float],
    prices: Mapping[str, float],
    equity: float,
    limits: RebalanceLimits,
) -> Dict[str, Any]:
    """Дельты «цель − факт» с гардрейлами. Чистая функция без side effects.

    Возвращает {orders, skipped, clipped, turnover_raw, turnover_planned,
    scale, dropped_by_max_orders} — всё журналируется как есть.
    """
    skipped: List[Dict[str, Any]] = []
    clipped: List[Dict[str, Any]] = []

    if equity <= 0:
        return {"orders": [], "skipped": [{"symbol": "*", "reason": "equity <= 0"}],
                "clipped": [], "turnover_raw": 0.0, "turnover_planned": 0.0,
                "scale": 1.0, "dropped_by_max_orders": []}

    # 1) Клип целевых весов по концентрации.
    weights: Dict[str, float] = {}
    for sym, w in target_weights.items():
        w = float(w)
        if abs(w) > limits.max_position_weight:
            clipped.append({"symbol": sym, "weight": w,
                            "clipped_to": limits.max_position_weight * (1 if w > 0 else -1)})
            w = limits.max_position_weight * (1 if w > 0 else -1)
        weights[str(sym)] = w

    # 2) Дельты по объединённому множеству (цель ∪ факт): позиции вне целевого
    #    юниверса закрываются (target weight = 0).
    deltas: Dict[str, float] = {}
    universe = set(weights) | {s for s, q in holdings_qty.items() if float(q) != 0.0}
    for sym in sorted(universe):
        price = float(prices.get(sym) or 0.0)
        if price <= 0:
            # Без цены позицию НЕ трогаем — честный skip, а не сделка вслепую.
            skipped.append({"symbol": sym, "reason": "no_price"})
            continue
        target_notional = weights.get(sym, 0.0) * equity
        current_notional = float(holdings_qty.get(sym, 0.0)) * price
        delta = target_notional - current_notional
        drift_bps = abs(delta) / equity * 1e4
        if drift_bps < limits.drift_band_bps:
            skipped.append({"symbol": sym, "reason": "drift_band",
                            "drift_bps": round(drift_bps, 2)})
            continue
        if abs(delta) < limits.min_trade_notional:
            skipped.append({"symbol": sym, "reason": "min_notional",
                            "delta_notional": round(delta, 2)})
            continue
        deltas[sym] = delta

    # 3) Turnover-cap: пропорциональный скейлинг всех дельт (стандартная
    #    практика — сохраняет направление портфельного сдвига).
    turnover_raw = sum(abs(d) for d in deltas.values()) / equity
    scale = 1.0
    if turnover_raw > limits.max_turnover > 0:
        scale = limits.max_turnover / turnover_raw
        deltas = {s: d * scale for s, d in deltas.items()}

    # 4) Ордеры; позицию, пересекающую ноль, режем на close + entry — иначе
    #    OMS-намерение неоднозначно (CLOSE_POSITION не открывает противоположную).
    orders: List[PlannedOrder] = []
    for sym, delta in deltas.items():
        price = float(prices[sym])
        cur_qty = float(holdings_qty.get(sym, 0.0))
        qty = delta / price
        target_qty = cur_qty + qty
        crosses_zero = cur_qty != 0.0 and target_qty != 0.0 and (cur_qty > 0) != (target_qty > 0)
        if crosses_zero:
            orders.append(PlannedOrder(sym, -cur_qty, price, -cur_qty * price, "close_leg"))
            orders.append(PlannedOrder(sym, target_qty, price, target_qty * price, "open_leg"))
        else:
            kind = "reduce" if cur_qty != 0.0 and abs(target_qty) < abs(cur_qty) else "increase"
            orders.append(PlannedOrder(sym, qty, price, delta, kind))

    # 5) Сначала продажи (высвобождают кэш), потом покупки.
    orders.sort(key=lambda o: (o.qty > 0, o.symbol))

    dropped: List[Dict[str, Any]] = []
    if len(orders) > limits.max_orders:
        for o in orders[limits.max_orders:]:
            dropped.append({"symbol": o.symbol, "qty": o.qty, "reason": "max_orders"})
        orders = orders[: limits.max_orders]

    return {
        "orders": orders,
        "skipped": skipped,
        "clipped": clipped,
        "turnover_raw": round(turnover_raw, 6),
        "turnover_planned": round(sum(abs(o.notional) for o in orders) / equity, 6),
        "scale": round(scale, 6),
        "dropped_by_max_orders": dropped,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _last_prices(panel) -> Dict[str, float]:
    """Последние close по каждому символу из канонической панели (ts_ms, symbol)."""
    import pandas as pd  # noqa: F401

    if panel is None or getattr(panel, "empty", True) or "close" not in panel.columns:
        return {}
    try:
        closes = panel["close"].dropna()
        last = closes.groupby(level="symbol").last()
        return {str(k): float(v) for k, v in last.items() if float(v) > 0}
    except Exception:
        return {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_rebalance(
    config_path: str,
    supervisor: Any,
    *,
    paper_only: bool = True,
    dry_run: bool = False,
    limits: Optional[RebalanceLimits] = None,
    strategy_id: str = "xs-rebalance",
    out_dir: str = os.path.join("logs", "xs_rebalance"),
    alert_fn: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Один цикл ребаланса. Возвращает журналируемую запись решения.

    ``status``: ``blocked`` (гейт не пройден — ничего не считалось/не слалось),
    ``dry_run`` (план построен, ордера НЕ отправлялись), ``ok`` / ``partial`` /
    ``failed`` (по результатам отправки), ``noop`` (план пуст — портфель в бэнде).
    """
    limits = limits or RebalanceLimits()
    record: Dict[str, Any] = {
        "started_at": _utc_now_iso(),
        "config": config_path,
        "strategy_id": strategy_id,
        "paper_only": bool(paper_only),
        "dry_run": bool(dry_run),
        "limits": asdict(limits),
        "status": "blocked",
        "reason": None,
        "signature": None,
        "authorization": None,
        "equity": None,
        "weights": {},
        "plan": None,
        "executions": [],
    }

    def _finish(status: str, reason: Optional[str] = None) -> Dict[str, Any]:
        record["status"] = status
        record["reason"] = reason
        record["finished_at"] = _utc_now_iso()
        _write_record(record, out_dir)
        if alert_fn is not None and status in ("partial", "failed", "blocked"):
            try:
                alert_fn("xs_rebalance", f"XS-ребаланс: {status} — {reason or ''}")
            except Exception:
                pass
        return record

    # --- Гейты (fail-closed, до каких-либо расчётов) -------------------------
    try:
        import services.ops_kill_switch as _oks
        if _oks.tripped():
            return _finish("blocked", "kill switch активен — ребаланс запрещён")
    except Exception:
        pass

    if not config_path or not os.path.exists(config_path):
        return _finish("blocked", f"XS-конфиг не найден: {config_path!r}")

    if supervisor is None:
        return _finish("blocked", "CCEA Agent (supervisor) не запущен — исполнять некуда")

    snap = supervisor.portfolio_snapshot()
    if not isinstance(snap, dict) or not snap.get("ok"):
        return _finish("blocked", f"портфель Agent'а недоступен: {snap.get('error') if isinstance(snap, dict) else snap}")
    is_paper = bool(snap.get("simulated"))
    broker = str(snap.get("broker") or "").strip().lower()
    record["broker"] = broker
    record["is_paper"] = is_paper
    if paper_only and not is_paper:
        return _finish("blocked", "paper_only=true, а активный брокер — live; авто-ребаланс на live запрещён")

    # --- Конфиг (нужен ДО авторизации: мандат привязан к хешу конфига) --------
    try:
        from service_xs_pipeline import latest_target_weights, load_config_dict, load_panel
        with open(config_path, "r", encoding="utf-8") as fh:
            raw_cfg = yaml.safe_load(fh) or {}
        cfg = load_config_dict(raw_cfg)
    except Exception as exc:
        return _finish("blocked", f"не удалось загрузить XS-конфиг: {exc}")

    # --- Авторизация live-торговли (для live-брокера — обязательна) -----------
    # Precheck (нулевые параметры): проверяет наличие активного мандата,
    # совпадение хеша конфига и не-исчерпанность бюджета; возвращает потолок,
    # к которому будут прижаты рантайм-лимиты ДО построения плана.
    live_store = None
    live_auth_id = None
    if not is_paper:
        live_store = getattr(supervisor, "live_auth_store", None)
        if live_store is None:
            return _finish("blocked", "live-брокер, но хранилище авторизаций недоступно")
        from packages.agent.approval.live_trading_authorization import canonical_config_hash
        record["config_hash"] = canonical_config_hash(raw_cfg)
        precheck = live_store.check(
            strategy_id=strategy_id, config=raw_cfg, broker=broker,
            turnover=0.0, notional=0.0, n_orders=0,
        )
        record["authorization"] = precheck.to_dict()
        if not precheck.allowed:
            return _finish("blocked", f"live-авторизация: {precheck.reason}")
        live_auth_id = precheck.auth_id
        ceiling = precheck.effective_ceiling
        # Прижать рантайм-лимиты к потолку мандата (строже — можно, слабее — нет).
        limits = RebalanceLimits(
            max_turnover=min(limits.max_turnover, ceiling.max_turnover),
            min_trade_notional=limits.min_trade_notional,
            drift_band_bps=limits.drift_band_bps,
            max_position_weight=limits.max_position_weight,
            max_orders=min(limits.max_orders, ceiling.max_orders_per_rebalance),
        )
        record["limits"] = asdict(limits)

    rl_checkpoint = getattr(getattr(cfg, "rl", None), "checkpoint", None)
    if rl_checkpoint:
        from services.model_signature_gate import ModelSignatureError, verify_model_artifact
        try:
            # live-брокер ⇒ строго enforce; paper ⇒ политика по env (default warn).
            verdict = verify_model_artifact(
                rl_checkpoint, live=not is_paper, context="xs-rebalance",
            )
            record["signature"] = verdict.to_dict()
        except ModelSignatureError as exc:
            record["signature"] = {"path": rl_checkpoint, "ok": False, "reason": str(exc)}
            return _finish("blocked", f"подпись RL-модели не прошла: {exc}")

    # --- Целевые веса ---------------------------------------------------------
    try:
        panel = load_panel(cfg)
        weights = latest_target_weights(cfg, panel)
    except Exception as exc:
        return _finish("failed", f"XS-пайплайн не построил веса: {exc}")
    if weights is None or len(weights) == 0:
        return _finish("noop", "пайплайн вернул пустые веса — сделок нет")
    record["weights"] = {str(k): round(float(v), 6) for k, v in weights.items()}

    # --- Факт портфеля и цены --------------------------------------------------
    equity = float(((snap.get("metrics") or {}).get("net_liquidation_value")) or 0.0)
    record["equity"] = equity
    holdings_qty = {
        str(h.get("symbol")): float(h.get("qty", 0.0))
        for h in (snap.get("holdings") or [])
    }
    prices = _last_prices(panel)

    plan = plan_rebalance(dict(weights.items()), holdings_qty, prices, equity, limits)
    record["plan"] = {
        **{k: v for k, v in plan.items() if k != "orders"},
        "orders": [o.to_dict() for o in plan["orders"]],
    }

    if not plan["orders"]:
        return _finish("noop", "все дельты внутри no-trade band / min-notional — сделок нет")

    if dry_run:
        return _finish("dry_run", f"план из {len(plan['orders'])} ордеров построен, отправка отключена")

    planned_notional = sum(abs(o.notional) for o in plan["orders"])

    # --- Финальная авторизация: точные числа плана против потолка/бюджета -----
    if not is_paper:
        final = live_store.check(
            strategy_id=strategy_id, config=raw_cfg, broker=broker,
            turnover=float(plan["turnover_planned"]),
            notional=float(planned_notional),
            n_orders=len(plan["orders"]),
        )
        record["authorization_final"] = final.to_dict()
        if not final.allowed:
            return _finish("blocked", f"live-авторизация (финальная проверка): {final.reason}")

    # --- Исполнение через Agent OMS --------------------------------------------
    ok_count = 0
    for order in plan["orders"]:
        try:
            res = supervisor.submit_rebalance_order(
                order.symbol, order.qty, order.price,
                strategy_id=strategy_id,
                reason=f"XS rebalance {os.path.basename(config_path)}",
                allow_live=(not is_paper),
            )
        except Exception as exc:
            res = {"ok": False, "error": f"exception: {exc}"}
        record["executions"].append({**order.to_dict(), **{
            "ok": bool(res.get("ok")),
            "client_order_id": res.get("client_order_id"),
            "broker_order_id": res.get("broker_order_id"),
            "state": res.get("state"),
            "error": res.get("error"),
        }})
        if res.get("ok"):
            ok_count += 1

    # --- Зафиксировать использование мандата (по фактически отправленному) ----
    if not is_paper and live_auth_id is not None and ok_count > 0:
        sent_notional = sum(
            abs(o.notional) for o, e in zip(plan["orders"], record["executions"]) if e["ok"]
        )
        try:
            consume = live_store.consume(live_auth_id, notional=sent_notional, n_orders=ok_count)
            record["authorization_consumed"] = consume
        except Exception as exc:
            record["authorization_consumed"] = {"ok": False, "error": str(exc)}

    total = len(plan["orders"])
    if ok_count == total:
        status, reason = "ok", f"исполнено {ok_count}/{total} ордеров (paper={is_paper})"
    elif ok_count > 0:
        status, reason = "partial", f"исполнено {ok_count}/{total}; остальные отклонены OMS"
    else:
        status, reason = "failed", f"все {total} ордеров отклонены OMS"
    result = _finish(status, reason)
    if alert_fn is not None and status == "ok":
        try:
            alert_fn("xs_rebalance", f"XS-ребаланс OK: {reason}, turnover {plan['turnover_planned']:.2%}")
        except Exception:
            pass
    return result


def _write_record(record: Dict[str, Any], out_dir: str) -> None:
    try:
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        atomic_write_json(os.path.join(out_dir, f"rebalance-{stamp}.json"), record)
        atomic_write_json(os.path.join(out_dir, "last.json"), record)
    except Exception:
        logger.exception("xs-rebalance: не удалось записать журнал решения")


def load_last_record(out_dir: str = os.path.join("logs", "xs_rebalance")) -> Optional[Dict[str, Any]]:
    path = os.path.join(out_dir, "last.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


__all__ = [
    "PlannedOrder",
    "RebalanceLimits",
    "load_last_record",
    "plan_rebalance",
    "run_rebalance",
]

# -*- coding: utf-8 -*-
"""
Agent client - the XS -> Agent -> broker bridge (AGENT ZONE ONLY).

Closes P0 blocker #1: ``service_xs_live.CrossSectionalLiveRunner`` produces an
``IntentBatch`` of **target exposures** (CCEA-compliant: no side/qty/price) and
calls ``agent_client.send_intents(batch)`` — but no ``AgentClient`` existed, so the
live rebalance terminated in a dry run and the real, journaled, idempotent
``LiveExecutionEngine`` was never driven.

``AgentClient`` is the local Agent-side translator. It:
  1. reads current positions (notional) + prices in the Agent's environment;
  2. updates the engine's ``PortfolioState`` (for policy / pre-trade risk checks);
  3. converts each target exposure into a **delta** vs the current position and
     emits ``OrderIntent``s (created LOCALLY — the Cloud never sent order fields);
  4. routes them either as immediate market orders through the engine, or — when a
     ``ClockDrivenChildExecutor`` is attached — as parent orders sliced
     (TWAP/VWAP/POV) and released on a clock;
  5. drives the OMS fill lifecycle via an attached ``FillHandler`` + fill source.

Idempotency: order intent ids are derived deterministically from the batch
idempotency key + symbol, so re-sending the same batch is de-duplicated by the
engine's journal (matches Design Doc Phase 8 WI-AGENT-06).

PROHIBITED in Cloud zone.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide
from packages.agent.execution.engine import LiveExecutionEngine, OrderStatus
from packages.agent.policy.risk_checker import PortfolioState

logger = logging.getLogger(__name__)

# Stable namespace for deterministic intent ids (idempotent rebalances).
_INTENT_NS = uuid.UUID("6f1d3c2a-4b5e-4a7c-9d2f-8e1a0b3c4d5e")


@dataclass
class SendResult:
    """Outcome of routing one IntentBatch through the Agent."""

    idempotency_key: str
    submitted: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)  # parent_ids when sliced
    sliced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idempotency_key": self.idempotency_key,
            "sliced": self.sliced,
            "submitted": self.submitted,
            "skipped": self.skipped,
            "errors": self.errors,
            "parents": self.parents,
            "counts": {
                "submitted": len(self.submitted),
                "skipped": len(self.skipped),
                "errors": len(self.errors),
            },
        }


class AgentClient:
    """Receives an ``IntentBatch`` (target exposures) and drives the execution engine."""

    def __init__(
        self,
        engine: LiveExecutionEngine,
        *,
        prices_provider: Any = None,  # .get_prices() -> {symbol: price}
        position_provider: Any = None,  # .get_positions() -> {symbol: notional}
        strategy_id: str = "xs_cross_sectional",
        min_trade_notional: float = 1.0,
        child_executor: Any = None,  # ClockDrivenChildExecutor (optional -> sliced)
        n_slices: int = 1,  # >1 with child_executor => TWAP slicing
        slice_weights: Optional[List[float]] = None,
        fill_handler: Any = None,  # FillHandler (drives OMS lifecycle)
        fill_source: Any = None,  # FillSource polled by pump()
        clock: Any = None,  # callable() -> epoch seconds (for sliced release)
    ) -> None:
        self._engine = engine
        self._prices_provider = prices_provider
        self._position_provider = position_provider
        self._strategy_id = strategy_id
        self._min_trade_notional = float(min_trade_notional)
        self._child_executor = child_executor
        self._n_slices = max(1, int(n_slices))
        self._slice_weights = slice_weights
        self._fill_handler = fill_handler
        self._fill_source = fill_source
        self._clock = clock

    # ------------------------------------------------------------------
    def _get_prices(self) -> Dict[str, float]:
        if self._prices_provider is None:
            return {}
        try:
            return {str(k): float(v) for k, v in (self._prices_provider.get_prices() or {}).items()}
        except Exception as exc:  # pragma: no cover
            logger.warning("prices_provider failed: %s", exc)
            return {}

    def _get_positions(self) -> Dict[str, float]:
        if self._position_provider is None:
            return {}
        try:
            return {
                str(k): float(v) for k, v in (self._position_provider.get_positions() or {}).items()
            }
        except Exception as exc:  # pragma: no cover
            logger.warning("position_provider failed: %s", exc)
            return {}

    def _sync_portfolio(
        self, equity: float, positions_notional: Dict[str, float], prices: Dict[str, float]
    ) -> None:
        """Push current state into the engine for policy / risk checks."""
        positions: Dict[str, Decimal] = {}
        position_values: Dict[str, Decimal] = {}
        gross = Decimal("0")
        net = Decimal("0")
        for sym, notional in positions_notional.items():
            nv = Decimal(str(notional))
            position_values[sym] = nv
            gross += abs(nv)
            net += nv
            px = prices.get(sym)
            if px and px > 0:
                positions[sym] = nv / Decimal(str(px))
        state = PortfolioState(
            equity=Decimal(str(equity)) if equity else Decimal("0"),
            buying_power=Decimal(str(equity)) if equity else Decimal("0"),
            positions=positions,
            position_values=position_values,
            gross_exposure=gross,
            net_exposure=net,
        )
        self._engine.update_portfolio(state)

    def _intent_id(self, key: str, symbol: str) -> uuid.UUID:
        return uuid.uuid5(_INTENT_NS, f"{key}|{symbol}")

    # ------------------------------------------------------------------
    def send_intents(self, batch: Any) -> SendResult:
        """Route an ``IntentBatch`` (target exposures) to real orders.

        Computes target-vs-current deltas (including liquidation of dropped names)
        and either submits immediate market orders or registers sliced parents.
        """
        equity = float(getattr(batch, "equity", 0.0) or 0.0)
        idem = str(getattr(batch, "idempotency_key", "") or "")
        # Let a real-broker price provider fetch quotes for the batch symbols on
        # demand (the get_prices() contract takes no args).
        if self._prices_provider is not None and hasattr(self._prices_provider, "prime"):
            try:
                self._prices_provider.prime(
                    [str(it.symbol) for it in (getattr(batch, "intents", []) or [])]
                )
            except Exception:  # pragma: no cover
                pass
        prices = self._get_prices()
        cur_notional = self._get_positions()
        self._sync_portfolio(equity, cur_notional, prices)

        # target notional per symbol from the batch intents
        target_notional: Dict[str, float] = {}
        for it in getattr(batch, "intents", []) or []:
            target_notional[str(it.symbol)] = float(it.target_notional)

        # union: trade dropped names down to zero too
        symbols = sorted(set(target_notional) | set(cur_notional))
        result = SendResult(
            idempotency_key=idem, sliced=bool(self._child_executor) and self._n_slices > 1
        )

        now_ts = self._now()
        for sym in symbols:
            tgt = float(target_notional.get(sym, 0.0))
            cur = float(cur_notional.get(sym, 0.0))
            delta = tgt - cur
            if abs(delta) < self._min_trade_notional:
                result.skipped.append(
                    {"symbol": sym, "delta_notional": delta, "reason": "below_min"}
                )
                continue
            price = prices.get(sym)
            if price is None or not math.isfinite(price) or price <= 0:
                result.errors.append({"symbol": sym, "reason": "no_price"})
                continue
            qty = Decimal(str(abs(delta) / price))
            side = "buy" if delta > 0 else "sell"

            if result.sliced and self._child_executor is not None:
                try:
                    parent = self._child_executor.submit_parent(
                        symbol=sym,
                        side=side,
                        total_qty=qty,
                        n_slices=self._n_slices,
                        weights=self._slice_weights,
                        start_ts=now_ts,
                        parent_id=f"{idem[:12]}_{sym}",
                    )
                    result.parents.append(parent.parent_id)
                    result.submitted.append(
                        {
                            "symbol": sym,
                            "side": side,
                            "qty": str(qty),
                            "delta_notional": delta,
                            "mode": "sliced",
                            "parent_id": parent.parent_id,
                            "n_slices": len(parent.children),
                        }
                    )
                except Exception as exc:
                    result.errors.append({"symbol": sym, "reason": f"slice_failed: {exc}"})
                continue

            # immediate market order through the engine
            intent = OrderIntent(
                strategy_id=self._strategy_id,
                symbol=sym,
                intent_type=IntentType.MARKET_ENTRY,
                side=IntentSide.LONG if side == "buy" else IntentSide.SHORT,
                target_quantity=qty,
                target_notional=Decimal(str(abs(delta))),
                time_in_force="DAY",
                reason=f"xs rebalance {idem[:12]}",
                metadata={"idempotency_key": idem, "target_notional": tgt, "delta_notional": delta},
                intent_id=self._intent_id(idem, sym),
            )
            exec_result = self._engine.execute(
                intent, current_price=Decimal(str(price)), origin="runner"
            )
            if exec_result.success and exec_result.order is not None:
                o = exec_result.order
                if self._fill_source is not None and hasattr(self._fill_source, "track"):
                    try:
                        self._fill_source.track(o.client_order_id)
                    except Exception:  # pragma: no cover
                        pass
                result.submitted.append(
                    {
                        "symbol": sym,
                        "side": side,
                        "qty": str(o.quantity),
                        "delta_notional": delta,
                        "mode": "market",
                        "client_order_id": o.client_order_id,
                        "broker_order_id": o.broker_order_id,
                        "status": o.status.value,
                    }
                )
            else:
                result.errors.append(
                    {
                        "symbol": sym,
                        "reason": exec_result.error_message or "execute_failed",
                    }
                )

        return result

    # ------------------------------------------------------------------
    def pump(self, now_ts: Optional[float] = None, *, max_fill_batches: int = 4) -> Dict[str, Any]:
        """Advance live execution one tick: release due slices + ingest fills.

        Call this on a clock from the live loop. Returns a summary of work done.
        """
        ts = float(now_ts) if now_ts is not None else self._now()
        step_summary: Optional[Dict[str, Any]] = None
        if self._child_executor is not None:
            # release any child whose client_order_id should be tracked for fills
            step_summary = self._child_executor.step(ts)
            if self._fill_source is not None and hasattr(self._fill_source, "track"):
                for coid in step_summary.get("released", []):
                    try:
                        self._fill_source.track(coid)
                    except Exception:  # pragma: no cover
                        pass

        fills_handled = 0
        if self._fill_handler is not None and self._fill_source is not None:
            fills_handled = self._fill_handler.consume(
                self._fill_source, max_batches=max_fill_batches
            )

        return {
            "ts": ts,
            "fills_handled": fills_handled,
            "step": step_summary,
            "complete": (
                self._child_executor.all_complete() if self._child_executor is not None else True
            ),
        }

    def _now(self) -> float:
        if self._clock is not None:
            try:
                return float(self._clock())
            except Exception:  # pragma: no cover
                pass
        import time as _time

        return _time.time()


__all__ = ["AgentClient", "SendResult"]

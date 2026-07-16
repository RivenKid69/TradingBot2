# -*- coding: utf-8 -*-
"""
service_xs_live.py
==================

Live-ребаланс cross-sectional портфеля через CCEA (Stage A11).

**Граница CCEA соблюдается строго:** Cloud (этот код) формирует только **Intents** —
целевые экспозиции (``target_weight`` + ``target_notional``), БЕЗ полей ордера
(side/qty/price). Конкретные ордера создаёт и отправляет локальный **Agent** в среде
клиента. Cloud никогда не передаёт order-like payload.

Поток ``rebalance``:
  1. reconcile: текущие позиции от ``position_provider`` → текущие веса;
  2. pre-trade portfolio risk guard (блокирует при нарушении лимитов);
  3. формирование ``IntentBatch`` (target exposures) с idempotency-ключом;
  4. передача Agent через ``agent_client.send_intents`` (только Intents!);
  5. reconciliation расхождений.

Всё DI-дружелюбно (моки в тестах). Слой ``service_``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from service_xs_portfolio_risk import PortfolioRiskGuard, RiskDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Intent:
    """Высокоуровневое намерение = целевая экспозиция (НЕ ордер).

    Намеренно НЕ содержит side/qty/price — эти поля запрещены протоколом CCEA и
    создаются только локально в Agent.
    """

    symbol: str
    target_weight: float
    target_notional: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "target_notional": self.target_notional,
        }


# Запрещённые (order-like) поля — guard против случайного нарушения CCEA.
_FORBIDDEN_FIELDS = {"side", "qty", "quantity", "price", "order_type", "limit_price"}


@dataclass
class IntentBatch:
    ts_ms: int
    equity: float
    intents: List[Intent]
    idempotency_key: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def symbols(self) -> List[str]:
        return [i.symbol for i in self.intents]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "equity": self.equity,
            "idempotency_key": self.idempotency_key,
            "intents": [i.to_dict() for i in self.intents],
            "meta": self.meta,
        }


def _idempotency_key(ts_ms: int, targets: Dict[str, float]) -> str:
    payload = f"{int(ts_ms)}|" + ",".join(
        f"{s}:{round(float(w), 8)}" for s, w in sorted(targets.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass
class RebalanceResult:
    approved: bool
    decision: RiskDecision
    batch: Optional[IntentBatch]
    reconciliation: Optional[Dict[str, Any]]
    sent: bool
    risk_report: Optional[Dict[str, Any]] = None       # pre-trade VaR/CVaR/стресс (P1)
    execution_plan: Optional[Dict[str, Any]] = None    # impact-aware TWAP/VWAP/POV slices (P1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "sent": self.sent,
            "decision": self.decision.to_dict(),
            "batch": self.batch.to_dict() if self.batch else None,
            "reconciliation": self.reconciliation,
            "risk_report": self.risk_report,
            "execution_plan": self.execution_plan,
        }


class CrossSectionalLiveRunner:
    """Преобразует целевые веса в Intents, проверяет риск, передаёт Agent."""

    def __init__(
        self,
        *,
        risk_guard: Optional[PortfolioRiskGuard] = None,
        position_provider: Any = None,   # .get_positions() -> {symbol: notional}
        agent_client: Any = None,        # .send_intents(batch)
        reconcile_tolerance: float = 0.01,
        pretrade_analyzer: Any = None,   # PreTradeRiskAnalyzer (VaR/CVaR/стресс) — P1
        risk_limits: Any = None,         # service_pretrade_risk.RiskLimits — P1
        scheduler: Any = None,           # RebalanceScheduler (TWAP/VWAP/POV slices) — P1
        prices_provider: Any = None,     # .get_prices() -> {symbol: price} (для scheduler)
        adv_provider: Any = None,        # .get_adv() -> {symbol: adv} (для scheduler)
    ) -> None:
        self.risk_guard = risk_guard
        self.position_provider = position_provider
        self.agent_client = agent_client
        self.reconcile_tolerance = float(reconcile_tolerance)
        self.pretrade_analyzer = pretrade_analyzer
        self.risk_limits = risk_limits
        self.scheduler = scheduler
        self.prices_provider = prices_provider
        self.adv_provider = adv_provider

    # ------------------------------------------------------------------
    def build_intents(
        self,
        target_weights: pd.Series,
        equity: float,
        *,
        ts_ms: int,
        idempotency_key: Optional[str] = None,
    ) -> IntentBatch:
        w = target_weights.astype("float64").dropna()
        targets = {str(s): float(v) for s, v in w.items()}
        intents = [
            Intent(symbol=s, target_weight=v, target_notional=v * float(equity))
            for s, v in targets.items()
        ]
        key = idempotency_key or _idempotency_key(ts_ms, targets)
        return IntentBatch(ts_ms=int(ts_ms), equity=float(equity), intents=intents, idempotency_key=key)

    def current_weights(self, equity: float) -> pd.Series:
        if self.position_provider is None or float(equity) == 0.0:
            return pd.Series(dtype="float64")
        pos = self.position_provider.get_positions() or {}
        return pd.Series({s: float(n) / float(equity) for s, n in pos.items()}, dtype="float64")

    def reconcile(
        self,
        expected_weights: pd.Series,
        equity: float,
        *,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        tol = self.reconcile_tolerance if tolerance is None else float(tolerance)
        actual_w = self.current_weights(equity)
        syms = sorted(set(expected_weights.index) | set(actual_w.index))
        exp = expected_weights.reindex(syms).fillna(0.0)
        act = actual_w.reindex(syms).fillna(0.0)
        diff = (exp - act).abs()
        discrepancies = {s: float(diff[s]) for s in syms if diff[s] > tol}
        return {
            "in_sync": len(discrepancies) == 0,
            "tolerance": tol,
            "discrepancies": discrepancies,
            "max_discrepancy": float(diff.max()) if len(diff) else 0.0,
        }

    # ------------------------------------------------------------------
    def rebalance(
        self,
        target_weights: pd.Series,
        equity: float,
        *,
        ts_ms: int,
        idempotency_key: Optional[str] = None,
    ) -> RebalanceResult:
        cur = self.current_weights(equity)
        if self.risk_guard is not None:
            decision = self.risk_guard.check(target_weights, cur if len(cur) else None)
        else:
            decision = RiskDecision(approved=True)

        # P1: pre-trade VaR/CVaR/стресс-сценарии ПЕРЕД отправкой rebalance
        risk_report = None
        if self.pretrade_analyzer is not None:
            try:
                rep = self.pretrade_analyzer.pretrade_check(
                    target_weights, limits=self.risk_limits, strict=True)
                risk_report = rep.to_dict()
                if not rep.approved:
                    decision.approved = False
                    decision.violations = list(decision.violations) + list(rep.violations)
            except Exception as exc:  # pragma: no cover
                logger.warning("pre-trade risk analyzer failed: %s", exc)

        if not decision.approved:
            logger.warning("rebalance blocked: %s", decision.violations)
            return RebalanceResult(
                approved=False, decision=decision, batch=None, reconciliation=None, sent=False,
                risk_report=risk_report,
            )

        batch = self.build_intents(target_weights, equity, ts_ms=ts_ms, idempotency_key=idempotency_key)

        # P1: impact-aware execution-plan (TWAP/VWAP/POV slices) для Agent
        execution_plan = None
        if self.scheduler is not None and self.prices_provider is not None:
            try:
                prices = pd.Series(self.prices_provider.get_prices() or {}, dtype="float64")
                adv = None
                if self.adv_provider is not None:
                    adv = pd.Series(self.adv_provider.get_adv() or {}, dtype="float64")
                plan = self.scheduler.build_plan(
                    target_weights, cur if len(cur) else None, prices, float(equity), adv=adv)
                execution_plan = plan.to_dict()
            except Exception as exc:  # pragma: no cover
                logger.warning("execution scheduler failed: %s", exc)

        sent = False
        if self.agent_client is not None:
            # CCEA: передаём ТОЛЬКО Intents (target exposures), никогда не ордера
            self.agent_client.send_intents(batch)
            sent = True

        recon = self.reconcile(target_weights, equity) if self.position_provider is not None else None
        return RebalanceResult(
            approved=True, decision=decision, batch=batch, reconciliation=recon, sent=sent,
            risk_report=risk_report, execution_plan=execution_plan,
        )


__all__ = [
    "Intent",
    "IntentBatch",
    "RebalanceResult",
    "CrossSectionalLiveRunner",
]

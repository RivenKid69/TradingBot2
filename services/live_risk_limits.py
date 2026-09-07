"""Enforcement риск-лимитов в реальном торговом контуре (P0-B gap closure).

Закрывает §3.6 / P0-B из PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md: форма
риск-лимитов Lite сохраняла ``lite_limits`` (дневной лимит убытка, макс.
просадку, плечо, концентрацию) в ``configs/risk.yaml``, но НИКАКОЙ рантайм-код
их не применял — трейдер думал, что защищён, а стопа по убытку не было.

Дизайн следует практике institutional risk management (двухуровневая защита):

1. **Pre-trade gate** — перед каждым ордером (уже вызывается движком
   ``LiveExecutionEngine`` через ``RiskChecker``). Здесь мы строим RiskChecker
   ИЗ пользовательских лимитов: leverage cap, concentration, position, daily
   loss, max drawdown. Блокирует ордер, который бы превысил лимит.

2. **Intra-day monitor** (circuit breaker) — pre-trade недостаточно: убыток
   может расти от движения рынка БЕЗ новых ордеров. ``LiveRiskMonitor``
   отслеживает trailing peak equity и при пробое дневного лимита убытка ИЛИ
   макс. просадки триггерит halt (kill switch + отмена/закрытие) — это
   account-level стоп-лосс, стандарт у профессиональных десков.

Peak equity долговечен (переживает рестарт в пределах торгового дня) и
сбрасывается на EOD. Монитор идемпотентен: сработав, не триггерит повторно до
явного reset.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_RISK_PATH = os.path.join("configs", "risk.yaml")
DEFAULT_PEAK_STATE = os.path.join("state", "live_risk_peak.json")


@dataclass
class LiveRiskLimits:
    """Пользовательские лимиты (из configs/risk.yaml → lite_limits)."""

    daily_loss_limit_usd: Optional[float] = None  # день -X$ → halt
    max_drawdown_pct: Optional[float] = None  # % от пика equity → halt (напр. 15 = 15%)
    max_leverage: Optional[float] = None  # gross exposure / equity cap
    max_concentration_pct: Optional[float] = None  # % equity на инструмент
    pdt_guard_enabled: bool = False
    span_guard_enabled: bool = False
    greeks_guard_enabled: bool = False

    def as_public(self) -> Dict[str, Any]:
        return {
            "daily_loss_limit_usd": self.daily_loss_limit_usd,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_leverage": self.max_leverage,
            "max_concentration_pct": self.max_concentration_pct,
            "pdt_guard_enabled": self.pdt_guard_enabled,
            "span_guard_enabled": self.span_guard_enabled,
            "greeks_guard_enabled": self.greeks_guard_enabled,
        }

    @property
    def any_enforced(self) -> bool:
        return any(
            v is not None
            for v in (
                self.daily_loss_limit_usd,
                self.max_drawdown_pct,
                self.max_leverage,
                self.max_concentration_pct,
            )
        )


def load_live_risk_limits(path: Optional[str] = None) -> LiveRiskLimits:
    """Читает lite_limits из configs/risk.yaml. Пустое/битое → пустые лимиты.

    ``path=None`` резолвит DEFAULT_RISK_PATH в момент вызова (не в момент
    определения функции) — иначе monkeypatch/переопределение пути не работали бы.
    """
    if path is None:
        path = DEFAULT_RISK_PATH
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            logger.warning("live-risk: не удалось прочитать %s", path, exc_info=True)
    lite = data.get("lite_limits") if isinstance(data.get("lite_limits"), dict) else {}

    def _num(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    conc = _num(lite.get("max_concentration_pct"))
    # Если lite_limits нет, но задан max_total_exposure_pct (доля) — берём как
    # ориентир концентрации (обратная совместимость с прежней формой).
    if conc is None and isinstance(data.get("max_total_exposure_pct"), (int, float)):
        conc = float(data["max_total_exposure_pct"]) * 100.0

    return LiveRiskLimits(
        daily_loss_limit_usd=_num(lite.get("daily_loss_limit_usd")),
        max_drawdown_pct=_num(lite.get("max_drawdown_pct")),
        max_leverage=_num(lite.get("max_leverage")),
        max_concentration_pct=conc,
        pdt_guard_enabled=bool(lite.get("pdt_guard_enabled", False)),
        span_guard_enabled=bool(lite.get("span_guard_enabled", False)),
        greeks_guard_enabled=bool(lite.get("greeks_guard_enabled", False)),
    )


def build_risk_checker(limits: LiveRiskLimits, *, equity: float = 100_000.0) -> Any:
    """Строит RiskChecker, питаемый пользовательскими лимитами.

    Не заданные лимиты остаются на безопасных дефолтах RiskChecker (или None
    для leverage/drawdown = не проверять). Это pre-trade половина enforcement.
    """
    from packages.agent.policy.risk_checker import RiskChecker

    kwargs: Dict[str, Any] = {}
    if limits.daily_loss_limit_usd is not None and limits.daily_loss_limit_usd > 0:
        kwargs["max_daily_loss"] = Decimal(str(limits.daily_loss_limit_usd))
    if limits.max_concentration_pct is not None and limits.max_concentration_pct > 0:
        kwargs["max_concentration_pct"] = Decimal(str(limits.max_concentration_pct / 100.0))
    if limits.max_leverage is not None and limits.max_leverage >= 1.0:
        kwargs["max_leverage"] = Decimal(str(limits.max_leverage))
    if limits.max_drawdown_pct is not None and limits.max_drawdown_pct > 0:
        kwargs["max_drawdown_pct"] = Decimal(str(limits.max_drawdown_pct / 100.0))
    # Position-size cap — производный ориентир: leverage × equity (если задано),
    # иначе оставляем дефолт RiskChecker.
    if limits.max_leverage is not None and equity > 0:
        kwargs["max_position_size"] = Decimal(str(limits.max_leverage * equity))
    return RiskChecker(**kwargs)


# ---------------------------------------------------------------------------
# Intra-day monitor / circuit breaker
# ---------------------------------------------------------------------------

BREACH_NONE = "ok"
BREACH_DAILY_LOSS = "daily_loss"
BREACH_DRAWDOWN = "max_drawdown"


class LiveRiskMonitor:
    """Отслеживает peak equity и триггерит halt при пробое дневного лимита
    убытка или макс. просадки. Питается snapshot'ом P&L-леджера Agent'а."""

    def __init__(
        self,
        *,
        limits_loader: Callable[[], LiveRiskLimits] = load_live_risk_limits,
        halt_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        peak_state_path: str = DEFAULT_PEAK_STATE,
        time_fn: Callable[[], float] = None,  # type: ignore[assignment]
    ) -> None:
        import time as _time

        self._loader = limits_loader
        self._halt = halt_callback
        self._peak_path = peak_state_path
        self._time = time_fn or _time.time
        self._lock = threading.RLock()
        self._peak_equity: Optional[float] = None
        self._breached: Optional[str] = None  # какой лимит пробит (идемпотентность)
        self._last_status: Dict[str, Any] = {"status": "no_data"}
        self._load_peak()

    # --------------------------------------------------------------- peak

    def _load_peak(self) -> None:
        try:
            from services.utils_app import read_json

            d = read_json(self._peak_path)
            if isinstance(d, dict) and isinstance(d.get("peak_equity"), (int, float)):
                self._peak_equity = float(d["peak_equity"])
        except Exception:
            pass

    def _save_peak(self) -> None:
        try:
            from services.utils_app import atomic_write_json

            atomic_write_json(
                self._peak_path, {"peak_equity": self._peak_equity, "at": self._time()}
            )
        except Exception:
            pass

    def reset_day(self, *, equity: Optional[float] = None) -> None:
        """Сброс на EOD: peak = текущий equity, снятие breach-флага."""
        with self._lock:
            self._peak_equity = float(equity) if equity is not None else None
            self._breached = None
            self._save_peak()

    def reset_breach(self) -> None:
        """Снять breach-флаг (после ручного reset kill switch)."""
        with self._lock:
            self._breached = None

    # ------------------------------------------------------------ evaluate

    def evaluate(
        self, ledger_snapshot: Dict[str, Any], *, auto_halt: bool = True
    ) -> Dict[str, Any]:
        """Оценить текущее состояние против лимитов. При hard breach (и
        ``auto_halt``) один раз вызывает halt_callback."""
        limits = self._loader()
        with self._lock:
            equity = float(ledger_snapshot.get("equity") or ledger_snapshot.get("nav") or 0.0)
            day_pnl = float(ledger_snapshot.get("day_pnl") or 0.0)
            gross = float(ledger_snapshot.get("gross_exposure") or 0.0)

            # trailing high-water mark
            if self._peak_equity is None or equity > self._peak_equity:
                self._peak_equity = equity
                self._save_peak()
            peak = self._peak_equity or equity
            drawdown_pct = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
            leverage = (gross / equity) if equity > 0 else 0.0

            breaches = []
            if (
                limits.daily_loss_limit_usd is not None
                and limits.daily_loss_limit_usd > 0
                and day_pnl <= -limits.daily_loss_limit_usd
            ):
                breaches.append(BREACH_DAILY_LOSS)
            if (
                limits.max_drawdown_pct is not None
                and limits.max_drawdown_pct > 0
                and drawdown_pct >= limits.max_drawdown_pct
            ):
                breaches.append(BREACH_DRAWDOWN)

            def _usage(cur, lim):
                return (abs(cur) / lim * 100.0) if (lim and lim > 0) else None

            status = {
                "status": (
                    "breached" if breaches else ("armed" if limits.any_enforced else "no_limits")
                ),
                "enforced": limits.any_enforced,
                "limits": limits.as_public(),
                "equity": round(equity, 2),
                "peak_equity": round(peak, 2),
                "day_pnl": round(day_pnl, 2),
                "drawdown_pct": round(drawdown_pct, 3),
                "leverage": round(leverage, 3),
                "breaches": breaches,
                "usage": {
                    "daily_loss_pct": _usage(min(0.0, day_pnl), limits.daily_loss_limit_usd),
                    "drawdown_pct_of_limit": _usage(drawdown_pct, limits.max_drawdown_pct),
                    "leverage_pct_of_cap": _usage(leverage, limits.max_leverage),
                },
                "already_halted": self._breached is not None,
            }
            self._last_status = status

            # Idempotent auto-halt: триггерим только на НОВОМ пробое.
            if breaches and auto_halt and self._breached is None and self._halt is not None:
                self._breached = breaches[0]
                reason = (
                    "Дневной лимит убытка пробит"
                    if BREACH_DAILY_LOSS in breaches
                    else "Макс. просадка пробита"
                )
                payload = {
                    "reason": reason,
                    "breaches": breaches,
                    "day_pnl": day_pnl,
                    "drawdown_pct": drawdown_pct,
                    "limits": limits.as_public(),
                }
                logger.error("live-risk: CIRCUIT BREAKER — %s (%s)", reason, payload)
                try:
                    self._halt(payload)
                except Exception:
                    logger.exception("live-risk: halt_callback failed")
                status["halt_triggered"] = True
            return status

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._last_status)


__all__ = [
    "BREACH_DAILY_LOSS",
    "BREACH_DRAWDOWN",
    "BREACH_NONE",
    "LiveRiskLimits",
    "LiveRiskMonitor",
    "build_risk_checker",
    "load_live_risk_limits",
]

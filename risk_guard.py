# risk_guard.py
"""
Risk management guards for trading systems.

This module provides:
1. RiskGuard - Core risk checks (position limits, drawdown, bankruptcy)
2. SimpleRiskGuard - Lightweight per-symbol risk guard
3. PortfolioLimitGuard - Aggregate portfolio exposure limits
4. StockRiskGuard - Stock-specific risk management (PDT, margin, short sale)

Architecture:
    For crypto trading: Use RiskGuard alone
    For stock trading: Use RiskGuard + StockRiskGuard

Stock-specific guards (Phase 6):
- PDTRuleGuard: Pattern Day Trader rule enforcement (< $25k accounts)
- MarginGuard: Reg T margin requirements (50% initial, 25% maintenance)
- ShortSaleGuard: Uptick rule, HTB list, circuit breakers
- CorporateActionsHandler: Dividends, stock splits
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import IntEnum, auto
import math
import logging
from typing import Optional, Deque, Tuple, Dict, Any, TYPE_CHECKING, Sequence, Callable, Mapping, List
from collections import deque
from collections.abc import Mapping as MappingABC
from clock import now_ms
from datetime import date
from decimal import Decimal

if TYPE_CHECKING:
    from core_contracts import RiskGuards
    from adapters.models import MarketType

try:
    import event_bus as eb
except Exception:  # на случай отсутствия event_bus в окружении
    class _Stub:
        def configure(self, *a, **k): return ""
        def log_trade(self, *a, **k): pass
        def log_risk(self, *a, **k): pass
        def flush(self): pass
        def run_dir(self): return ""
    eb = _Stub()  # type: ignore

from action_proto import ActionProto, ActionType

logger = logging.getLogger(__name__)


class RiskEvent(IntEnum):
    NONE = 0
    POSITION_LIMIT = 1        # превышение лимита по абсолютной позиции (pre/post)
    NOTIONAL_LIMIT = 2        # превышение лимита по ноционалу (post)
    DRAWDOWN = 3              # превышение лимита по дроудауну (post)
    BANKRUPTCY = 4            # cash ниже порога банкротства (post)
    # Stock-specific events (Phase 6)
    PDT_VIOLATION = 5         # Pattern Day Trader rule violation
    MARGIN_CALL = 6           # Margin requirement violation
    SHORT_SALE_RESTRICTED = 7  # Short sale restriction (uptick rule, HTB)
    CORPORATE_ACTION = 8      # Corporate action affecting position
    # Futures-specific events (Phase 6A)
    LEVERAGE_VIOLATION = 9    # Leverage exceeds maximum allowed
    FUTURES_MARGIN_WARNING = 10  # Margin ratio below warning threshold
    FUTURES_MARGIN_DANGER = 11   # Margin ratio critical - position reduction required
    FUTURES_MARGIN_LIQUIDATION = 12  # Margin ratio at liquidation level
    FUNDING_EXPOSURE = 13     # Funding rate exposure exceeds threshold
    CONCENTRATION_LIMIT = 14  # Position concentration exceeds limits
    ADL_RISK = 15             # Auto-deleveraging risk detected


@dataclass
class RiskConfig:
    # Прямые жёсткие лимиты
    max_abs_position: float = 1e12
    max_notional: float = 2e12
    max_total_notional: float = 0.0
    max_total_exposure_pct: float = 0.0
    exposure_buffer_frac: float = 0.0

    # Дроудаун/устойчивость
    max_drawdown_pct: float = 1.00        # разрешённая просадка (0.30 => 30%)
    intrabar_dd_pct: float = 0.30         # «жёсткий» интра-барный триггер
    dd_window: int = 500                  # размер окна для оценки пика equity

    # Ликвидация/банкротство
    bankruptcy_cash_th: float = -1e12     # порог банкротства по кэшу

    # Технические опции
    ts_provider: Callable[[], int] = lambda: now_ms()


class RiskGuard:
    """
    Единая точка risk-контроля:
      * on_action_proposed(state, proto) — pre-trade проверка (возможной позиции)
      * on_post_trade(state, mid_price) — post-trade инварианты (ноционал, дроудаун, банкротство)
    Ожидается, что state имеет поля: units (float), cash (float), max_position (float | опционально).
    """

    def __init__(self, cfg: Optional[RiskConfig] = None):
        self.cfg = cfg or RiskConfig()
        self._nw_hist: Deque[Tuple[int, float]] = deque(maxlen=self.cfg.dd_window)  # (ts, net_worth)
        self._peak_nw_window: Deque[float] = deque(maxlen=self.cfg.dd_window)
        self._last_event: RiskEvent = RiskEvent.NONE

    def reset(self) -> None:
        """Reset internal statistics collected during an episode."""
        self._nw_hist.clear()
        self._peak_nw_window.clear()
        self._last_event = RiskEvent.NONE

    # ---------- ВСПОМОГАТЕЛЬНЫЕ РАСЧЁТЫ ----------

    @staticmethod
    def _get_max_position_from_state_or_cfg(state, cfg: RiskConfig) -> float:
        mp = float(getattr(state, "max_position", 0.0) or 0.0)
        if mp > 0.0 and math.isfinite(mp):
            return mp

        cfg_mp = float(cfg.max_abs_position)
        if not math.isfinite(cfg_mp) or cfg_mp <= 0.0:
            cfg_mp = 1.0
        return float(cfg_mp)

    @staticmethod
    def _notional(state, mid_price: float) -> float:
        # Абсолютная экспозиция позиции (в денежных единицах)
        price = float(mid_price)
        if not math.isfinite(price) or price <= 0.0:
            return 0.0
        return abs(float(state.units)) * price

    def _update_equity_windows(self, ts: int, state, mid_price: float) -> Tuple[float, float, float]:
        """
        Возвращает (nw, peak, dd_pct) - net worth, rolling peak, drawdown percentage.

        ═══════════════════════════════════════════════════════════════════════════
        НЕ БАГ: ROLLING WINDOW DRAWDOWN (BY DESIGN)
        ═══════════════════════════════════════════════════════════════════════════
        Peak вычисляется как max(NW) в пределах СКОЛЬЗЯЩЕГО ОКНА (dd_window баров).
        Это НАМЕРЕННОЕ поведение для "recent drawdown" метрики.

        После заполнения окна старые значения уходят и peak может УМЕНЬШИТЬСЯ.
        Это корректно для trading bot, который оценивает текущую просадку
        относительно недавнего максимума, а не исторического.

        Для ГЛОБАЛЬНОГО drawdown увеличьте dd_window в configs/risk.yaml:
            risk_guard:
              dd_window: 999999  # Практически бесконечное окно

        Двойной max() выглядит избыточным но корректен:
          max(max(window), nw) = max(всех элементов окна + текущий nw)

        Reference: docs/NOT_BUGS_AND_FAQ.md → "НЕ БАГИ" → #22
        ═══════════════════════════════════════════════════════════════════════════
        """
        nw = float(state.cash) + float(state.units) * float(mid_price)
        self._nw_hist.append((ts, nw))
        # Rolling window peak: max NW in last dd_window bars
        if not self._peak_nw_window:
            self._peak_nw_window.append(nw)
            peak = nw
        else:
            peak = max(max(self._peak_nw_window, default=nw), nw)
            self._peak_nw_window.append(nw)
        dd_pct = 0.0 if peak <= 0 else max(0.0, (peak - nw) / peak)
        return nw, peak, dd_pct

    # ---------- PRE-TRADE ----------

    def on_action_proposed(self, state, proto: ActionProto) -> RiskEvent:
        """
        Проверяет, не приведёт ли ДЕЙСТВИЕ к нарушению лимита по абсолютной позиции.
        Возвращает RiskEvent (NONE или POSITION_LIMIT).

        CRITICAL FIX (2025-11-21):
        - volume_frac теперь интерпретируется как **TARGET position**, а не DELTA
        - Это предотвращает risk of position doubling при повторных действиях
        """
        cfg = self.cfg
        ts = cfg.ts_provider()

        # volume_frac ∈ [-1, 1] представляет TARGET position as fraction of max
        max_pos = self._get_max_position_from_state_or_cfg(state, cfg)

        # политики типа HOLD не изменяют позицию
        if proto.action_type == ActionType.HOLD:
            self._last_event = RiskEvent.NONE
            return self._last_event

        # ✅ FIXED: Interpret volume_frac as TARGET, not delta
        # target_units = volume_frac * max_position
        target_units = float(proto.volume_frac) * float(max_pos)
        next_units = target_units  # Direct target, not adding to current

        if abs(next_units) > cfg.max_abs_position + 1e-12:
            evt = RiskEvent.POSITION_LIMIT
            eb.log_risk({
                "ts": ts,
                "type": "POSITION_LIMIT",
                "stage": "pre_trade",
                "units_curr": float(state.units),
                "units_next": float(next_units),
                "max_abs_position": float(cfg.max_abs_position),
                "proto": {
                    "type": int(proto.action_type),
                    "volume_frac": float(proto.volume_frac),
                    "ttl_steps": int(getattr(proto, "ttl_steps", 0) or 0),
                    "client_order_id": int(getattr(proto, "client_order_id", 0) or 0),
                },
            })
            self._last_event = evt
            return evt

        self._last_event = RiskEvent.NONE
        return self._last_event

    # ---------- POST-TRADE ----------

    def on_post_trade(self, state, mid_price: float) -> RiskEvent:
        """
        Пост-фактум проверки: лимит по ноционалу, интрабарный дроудаун, общий дроудаун и банкротство.
        Возвращает первый сработавший RiskEvent (приоритет: BANKRUPTCY > NOTIONAL_LIMIT > DRAWDOWN > POSITION_LIMIT).
        """
        cfg = self.cfg
        ts = cfg.ts_provider()

        # 1) Банкротство (по кэшу)
        if float(state.cash) < cfg.bankruptcy_cash_th:
            evt = RiskEvent.BANKRUPTCY
            eb.log_risk({
                "ts": ts,
                "type": "BANKRUPTCY",
                "cash": float(state.cash),
                "threshold": float(cfg.bankruptcy_cash_th),
            })
            self._last_event = evt
            return evt

        # 2) Лимит по ноционалу
        notion = self._notional(state, float(mid_price))
        if notion > cfg.max_notional + 1e-9:
            evt = RiskEvent.NOTIONAL_LIMIT
            eb.log_risk({
                "ts": ts,
                "type": "NOTIONAL_LIMIT",
                "notional": float(notion),
                "max_notional": float(cfg.max_notional),
                "units": float(state.units),
                "mid": float(mid_price),
                "cash": float(state.cash),
            })
            self._last_event = evt
            return evt

        # 3) Дроудаун (интрабарный быстрый триггер + оконный)
        nw, peak, dd_pct = self._update_equity_windows(ts, state, float(mid_price))
        if dd_pct >= cfg.intrabar_dd_pct - 1e-12:
            evt = RiskEvent.DRAWDOWN
            eb.log_risk({
                "ts": ts,
                "type": "DRAWDOWN_INTRABAR",
                "drawdown_pct": float(dd_pct),
                "intrabar_dd_pct": float(cfg.intrabar_dd_pct),
                "nw": float(nw),
                "peak": float(peak),
            })
            self._last_event = evt
            return evt

        if dd_pct >= cfg.max_drawdown_pct - 1e-12:
            evt = RiskEvent.DRAWDOWN
            eb.log_risk({
                "ts": ts,
                "type": "DRAWDOWN",
                "drawdown_pct": float(dd_pct),
                "max_drawdown_pct": float(cfg.max_drawdown_pct),
                "nw": float(nw),
                "peak": float(peak),
            })
            self._last_event = evt
            return evt

        # 4) Контроль «на всякий» по абсолютной позиции (post) — на случай внешних модификаций состояния
        if abs(float(state.units)) > cfg.max_abs_position + 1e-12:
            evt = RiskEvent.POSITION_LIMIT
            eb.log_risk({
                "ts": ts,
                "type": "POSITION_LIMIT",
                "stage": "post_trade",
                "units": float(state.units),
                "max_abs_position": float(cfg.max_abs_position),
            })
            self._last_event = evt
            return evt

        self._last_event = RiskEvent.NONE
        return self._last_event

    # ---------- ВСПОМОГАТЕЛЬНОЕ ----------

    def last_event(self) -> RiskEvent:
        return self._last_event

    def snapshot(self) -> Dict[str, Any]:
        """Для отладки/логов."""
        return {
            "cfg": asdict(self.cfg),
            "last_event": int(self._last_event),
            "nw_window_len": len(self._nw_hist),
        }


# ----------- PIPELINE SUPPORT -----------


@dataclass
class _SymbolState:
    """Internal per-symbol bookkeeping for lightweight risk checks."""

    last_ts: int = 0
    exposure: float = 0.0


class SimpleRiskGuard:
    """Minimal per-symbol risk guard used by the pipeline.

    The guard tracks the last processed timestamp and cumulative exposure for
    each symbol.  ``apply`` returns filtered decisions and an optional reason
    string beginning with ``"RISK_"`` if all decisions should be dropped.
    """

    def __init__(self) -> None:
        self._states: Dict[str, _SymbolState] = {}

    def _state(self, symbol: str) -> _SymbolState:
        return self._states.setdefault(symbol, _SymbolState())

    def apply(
        self, ts_ms: int, symbol: str, decisions: Sequence[Any]
    ) -> tuple[Sequence[Any], str | None]:
        st = self._state(symbol)
        if ts_ms <= st.last_ts:
            # Reject stale timestamps outright
            return [], "RISK_STALE_TS"

        def _signal_leg(order: Any) -> str:
            meta = getattr(order, "meta", None)
            if isinstance(meta, MappingABC):
                return str(meta.get("signal_leg") or "").lower()
            if meta is not None:
                getter = getattr(meta, "get", None)
                if callable(getter):
                    try:
                        value = getter("signal_leg")
                    except Exception:
                        value = None
                    else:
                        return str(value or "").lower()
            return ""

        exp = 0.0
        checked: list[Any] = []
        for d in decisions:
            leg = _signal_leg(d)
            if leg == "exit":
                checked.append(d)
                continue
            vol = getattr(d, "volume_frac", getattr(d, "quantity", 0.0)) or 0.0
            try:
                exp += abs(float(vol))
            except Exception:
                continue
            checked.append(d)

        st.last_ts = int(ts_ms)
        st.exposure += exp
        return checked, None


@dataclass
class PortfolioLimitConfig:
    """Configuration for aggregate portfolio exposure limits."""

    max_total_notional: float | None = None
    max_total_exposure_pct: float | None = None
    exposure_buffer_frac: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PortfolioLimitConfig":
        payload: Dict[str, Any] = {}
        if isinstance(data, Mapping):
            payload.update(data)
        return cls(
            max_total_notional=cls._coerce_float(payload.get("max_total_notional")),
            max_total_exposure_pct=cls._coerce_float(payload.get("max_total_exposure_pct")),
            exposure_buffer_frac=cls._coerce_float(
                payload.get("exposure_buffer_frac"), default=0.0, minimum=0.0
            )
            or 0.0,
        )

    @staticmethod
    def _coerce_float(
        value: Any,
        *,
        default: float | None = None,
        minimum: float | None = None,
    ) -> float | None:
        if value is None:
            return default
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(coerced):
            return default
        if minimum is not None and coerced < minimum:
            return default
        return coerced

    @property
    def enabled(self) -> bool:
        return any(
            v is not None and v > 0.0
            for v in (self.max_total_notional, self.max_total_exposure_pct)
        )


class PortfolioLimitGuard:
    """Guard enforcing aggregate portfolio exposure limits."""

    _EPS = 1e-12

    def __init__(
        self,
        *,
        config: PortfolioLimitConfig,
        get_positions: Callable[[], Mapping[str, Any]],
        get_total_notional: Callable[[], float | None] | None = None,
        get_price: Callable[[str], float | None],
        get_equity: Callable[[], float | None] | None = None,
        leg_getter: Callable[[Any], str] | None = None,
    ) -> None:
        self._cfg = config
        self._get_positions = get_positions
        self._get_total_notional = get_total_notional
        self._get_price = get_price
        self._get_equity = get_equity or (lambda: None)
        self._leg_getter = leg_getter or self._default_leg_getter

    @staticmethod
    def _default_leg_getter(order: Any) -> str:
        meta = getattr(order, "meta", None)
        value: Any = None
        if isinstance(meta, MappingABC):
            value = meta.get("signal_leg")
        elif meta is not None:
            getter = getattr(meta, "get", None)
            if callable(getter):
                try:
                    value = getter("signal_leg")
                except Exception:
                    value = None
        return str(value or "").lower()

    @staticmethod
    def _normalize_symbol(symbol: Any, fallback: str | None = None) -> str:
        if symbol is None and fallback is not None:
            symbol = fallback
        if symbol is None:
            return ""
        return str(symbol).upper()

    @staticmethod
    def _extract_side(order: Any) -> str:
        side = getattr(order, "side", "")
        if hasattr(side, "value"):
            try:
                side = side.value
            except Exception:
                pass
        return str(side or "").upper()

    @staticmethod
    def _extract_quantity(order: Any) -> float | None:
        candidates = (
            getattr(order, "quantity", None),
            getattr(order, "volume", None),
            getattr(order, "size", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return None

    def _snapshot_positions(self) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        try:
            positions = self._get_positions()
        except Exception:
            positions = {}
        if not isinstance(positions, MappingABC):
            return snapshot
        for sym, qty in positions.items():
            try:
                value = float(qty)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if math.isclose(value, 0.0, abs_tol=self._EPS):
                continue
            snapshot[str(sym).upper()] = value
        return snapshot

    def _base_total_notional(self, positions: Mapping[str, float]) -> float:
        if self._get_total_notional is not None:
            try:
                value = self._get_total_notional()
            except Exception:
                value = None
            else:
                if value is not None and math.isfinite(float(value)):
                    return max(0.0, float(value))
        total = 0.0
        for sym, qty in positions.items():
            price = self._safe_price(sym)
            if price is None:
                continue
            total += abs(float(qty)) * price
        return total

    def _safe_price(self, symbol: str) -> float | None:
        try:
            price = self._get_price(symbol)
        except Exception:
            return None
        if price is None:
            return None
        try:
            value = float(price)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value

    def _effective_limits(self) -> tuple[float | None, float | None]:
        def _sanitize(limit: float | None) -> float | None:
            if limit is None:
                return None
            try:
                value = float(limit)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value) or value <= 0.0:
                return None
            return value

        notional_limit = _sanitize(self._cfg.max_total_notional)
        equity_limit: float | None = None
        pct = self._cfg.max_total_exposure_pct
        if pct is not None and pct > 0.0:
            try:
                equity = self._get_equity()
            except Exception:
                equity = None
            if equity is not None:
                try:
                    equity_val = float(equity)
                except (TypeError, ValueError):
                    equity_val = None
                if equity_val is not None and math.isfinite(equity_val) and equity_val > 0.0:
                    equity_limit = _sanitize(equity_val * pct)
        return notional_limit, equity_limit

    def _classify_orders(
        self,
        orders: Sequence[Any],
        default_symbol: str,
    ) -> list[Dict[str, Any]]:
        infos: list[Dict[str, Any]] = []
        for idx, order in enumerate(orders):
            sym = self._normalize_symbol(getattr(order, "symbol", None), default_symbol)
            leg = self._leg_getter(order)
            reduce_only = bool(getattr(order, "reduce_only", False))
            qty = self._extract_quantity(order)
            side = self._extract_side(order)
            delta: float | None
            if qty is None:
                delta = None
            elif side == "BUY":
                delta = qty
            elif side == "SELL":
                delta = -qty
            else:
                delta = None
            infos.append(
                {
                    "index": idx,
                    "order": order,
                    "symbol": sym,
                    "delta": delta,
                    "price": self._safe_price(sym) if sym else None,
                    "leg": leg,
                    "reduce_only": reduce_only,
                }
            )
        return infos

    @staticmethod
    def _is_exit(info: Mapping[str, Any]) -> bool:
        if bool(info.get("reduce_only")):
            return True
        leg = str(info.get("leg") or "").lower()
        return leg == "exit"

    def apply(
        self, ts_ms: int, symbol: str, decisions: Sequence[Any]
    ) -> tuple[list[Any], str | None]:
        orders = list(decisions or [])
        if not orders:
            return [], None
        if not self._cfg.enabled:
            return orders, None
        positions = self._snapshot_positions()
        working_positions = dict(positions)
        current_total = self._base_total_notional(working_positions)
        notional_limit, equity_limit = self._effective_limits()
        if notional_limit is None and equity_limit is None:
            return orders, None
        buffer_raw = float(self._cfg.exposure_buffer_frac or 0.0)
        if not math.isfinite(buffer_raw) or buffer_raw < 0.0:
            buffer_raw = 0.0
        buffer_mult = 1.0 + buffer_raw
        infos = self._classify_orders(orders, symbol)
        accepted: set[int] = set()
        blocked: set[int] = set()

        # Process exits first to free up capacity.
        for info in infos:
            if not self._is_exit(info):
                continue
            idx = int(info["index"])
            accepted.add(idx)
            sym = info.get("symbol") or ""
            delta = info.get("delta")
            if not sym or delta is None:
                continue
            prev = working_positions.get(sym, 0.0)
            new = prev + float(delta)
            working_positions[sym] = new
            price = info.get("price")
            if price is not None:
                current_total += (abs(new) - abs(prev)) * float(price)
                if current_total < 0.0:
                    current_total = 0.0

        def _within_limits(total: float) -> bool:
            if notional_limit is not None and total > notional_limit + self._EPS:
                return False
            if equity_limit is not None and total > equity_limit + self._EPS:
                return False
            return True

        # Process remaining orders.
        for info in infos:
            idx = int(info["index"])
            if idx in accepted or idx in blocked:
                continue
            sym = info.get("symbol") or ""
            delta = info.get("delta")
            price = info.get("price")
            if not sym:
                blocked.add(idx)
                continue
            if delta is None:
                blocked.add(idx)
                continue
            prev = working_positions.get(sym, 0.0)
            new = prev + float(delta)
            exposure_delta = abs(new) - abs(prev)
            if exposure_delta <= self._EPS:
                accepted.add(idx)
                working_positions[sym] = new
                if price is not None:
                    current_total += exposure_delta * float(price)
                    if current_total < 0.0:
                        current_total = 0.0
                continue
            if price is None:
                blocked.add(idx)
                continue
            notional_delta = exposure_delta * float(price)
            if exposure_delta > self._EPS:
                buffered_delta = notional_delta * buffer_mult
            else:
                buffered_delta = notional_delta
            prospective_total = current_total + buffered_delta
            if not _within_limits(prospective_total):
                blocked.add(idx)
                continue
            accepted.add(idx)
            working_positions[sym] = new
            current_total += notional_delta
            if current_total < 0.0:
                current_total = 0.0

        approved: list[Any] = []
        for idx, order in enumerate(orders):
            if idx in accepted:
                approved.append(order)

        if approved:
            return approved, None
        return [], "RISK_PORTFOLIO_LIMIT"


# =========================
# Stock Risk Guard (Phase 6)
# =========================

@dataclass
class StockRiskConfig:
    """Configuration for stock-specific risk management."""

    # Asset class detection
    market_type: str = "EQUITY"  # EQUITY, CRYPTO_SPOT, etc.

    # PDT settings
    pdt_enabled: bool = True
    pdt_account_equity: float = 30_000.0  # Account equity for PDT check
    pdt_threshold: float = 25_000.0  # PDT exemption threshold
    pdt_max_day_trades: int = 3  # Max day trades in rolling window
    pdt_rolling_days: int = 5  # Rolling window size

    # Margin settings
    margin_enabled: bool = True
    initial_margin: float = 0.50  # Reg T initial margin (50%)
    maintenance_margin: float = 0.25  # Reg T maintenance margin (25%)
    margin_buffer: float = 0.05  # House buffer above maintenance

    # Short sale settings
    short_sale_enabled: bool = True
    enforce_uptick_rule: bool = True
    check_htb_list: bool = True
    circuit_breaker_threshold: float = -0.10  # 10% drop triggers Rule 201

    # Corporate actions settings
    corporate_actions_enabled: bool = True
    warn_on_ex_dividend: bool = True
    adjust_positions_on_split: bool = True

    # General settings
    simulation_mode: bool = False  # Warn but don't block
    strict_mode: bool = True  # Block vs warn

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StockRiskConfig":
        """Create config from mapping."""
        return cls(
            market_type=str(data.get("market_type", "EQUITY")),
            pdt_enabled=bool(data.get("pdt_enabled", True)),
            pdt_account_equity=float(data.get("pdt_account_equity", 30_000.0)),
            pdt_threshold=float(data.get("pdt_threshold", 25_000.0)),
            margin_enabled=bool(data.get("margin_enabled", True)),
            initial_margin=float(data.get("initial_margin", 0.50)),
            maintenance_margin=float(data.get("maintenance_margin", 0.25)),
            short_sale_enabled=bool(data.get("short_sale_enabled", True)),
            enforce_uptick_rule=bool(data.get("enforce_uptick_rule", True)),
            corporate_actions_enabled=bool(data.get("corporate_actions_enabled", True)),
            simulation_mode=bool(data.get("simulation_mode", False)),
            strict_mode=bool(data.get("strict_mode", True)),
        )

    @property
    def is_stock_trading(self) -> bool:
        """Check if this is stock trading (not crypto)."""
        return self.market_type.upper() in ("EQUITY", "EQUITY_OPTIONS")


class StockRiskGuard:
    """
    Combined stock-specific risk management guard.

    Integrates:
    1. PDT Rule Enforcement (< $25k accounts)
    2. Reg T Margin Requirements (50%/25%)
    3. Short Sale Rules (uptick, HTB, circuit breaker)
    4. Corporate Actions (dividends, splits)

    BACKWARD COMPATIBLE: For crypto trading, all checks are skipped
    automatically based on market_type configuration.

    Usage:
        # For stock trading
        stock_guard = StockRiskGuard(StockRiskConfig(market_type="EQUITY"))

        # Pre-trade check
        event = stock_guard.check_trade(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            is_day_trade=True,
        )

        if event != RiskEvent.NONE:
            reject_trade(event)

        # Post-trade update
        stock_guard.record_trade(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            timestamp_ms=now_ms(),
        )

    For crypto trading:
        # All checks automatically disabled
        crypto_guard = StockRiskGuard(StockRiskConfig(market_type="CRYPTO_SPOT"))
        event = crypto_guard.check_trade(...)  # Always returns NONE
    """

    def __init__(
        self,
        config: Optional[StockRiskConfig] = None,
    ) -> None:
        """
        Initialize StockRiskGuard.

        Args:
            config: Stock risk configuration
        """
        self._config = config or StockRiskConfig()
        self._last_event: RiskEvent = RiskEvent.NONE
        self._last_event_reason: str = ""

        # Lazy initialization of sub-guards
        self._pdt_tracker = None
        self._margin_guard = None
        self._short_sale_guard = None
        self._corporate_actions = None

        # Only initialize if this is stock trading
        if self._config.is_stock_trading:
            self._initialize_guards()

        logger.debug(
            f"StockRiskGuard initialized: market_type={self._config.market_type}, "
            f"is_stock={self._config.is_stock_trading}"
        )

    def _initialize_guards(self) -> None:
        """Initialize sub-guards for stock trading."""
        try:
            # Import stock risk modules
            from services.pdt_tracker import PDTTracker, PDTTrackerConfig
            from services.stock_risk_guards import (
                MarginGuard, MarginGuardConfig,
                ShortSaleGuard, ShortSaleGuardConfig,
                CorporateActionsHandler, CorporateActionsConfig,
            )

            # PDT Tracker
            if self._config.pdt_enabled:
                pdt_config = PDTTrackerConfig(
                    initial_equity=self._config.pdt_account_equity,
                    pdt_threshold=self._config.pdt_threshold,
                    max_day_trades=self._config.pdt_max_day_trades,
                    rolling_days=self._config.pdt_rolling_days,
                    simulation_mode=self._config.simulation_mode,
                    strict_mode=self._config.strict_mode,
                )
                self._pdt_tracker = PDTTracker(
                    account_equity=self._config.pdt_account_equity,
                    config=pdt_config,
                )

            # Margin Guard
            if self._config.margin_enabled:
                margin_config = MarginGuardConfig(
                    initial_margin=self._config.initial_margin,
                    maintenance_margin=self._config.maintenance_margin,
                    house_margin_buffer=self._config.margin_buffer,
                    strict_mode=self._config.strict_mode,
                )
                self._margin_guard = MarginGuard(config=margin_config)

            # Short Sale Guard
            if self._config.short_sale_enabled:
                short_config = ShortSaleGuardConfig(
                    enforce_uptick_rule=self._config.enforce_uptick_rule,
                    check_htb_list=self._config.check_htb_list,
                    circuit_breaker_threshold=self._config.circuit_breaker_threshold,
                    strict_mode=self._config.strict_mode,
                )
                self._short_sale_guard = ShortSaleGuard(config=short_config)

            # Corporate Actions Handler
            if self._config.corporate_actions_enabled:
                ca_config = CorporateActionsConfig(
                    adjust_positions_on_split=self._config.adjust_positions_on_split,
                    warn_on_ex_dividend=self._config.warn_on_ex_dividend,
                )
                self._corporate_actions = CorporateActionsHandler(config=ca_config)

            logger.info("StockRiskGuard: All sub-guards initialized")

        except ImportError as e:
            logger.warning(f"StockRiskGuard: Could not import sub-guards: {e}")

    # =========================
    # Properties
    # =========================

    @property
    def config(self) -> StockRiskConfig:
        """Get current configuration."""
        return self._config

    @property
    def last_event(self) -> RiskEvent:
        """Get last risk event."""
        return self._last_event

    @property
    def last_event_reason(self) -> str:
        """Get reason for last risk event."""
        return self._last_event_reason

    @property
    def pdt_tracker(self):
        """Access PDT tracker directly."""
        return self._pdt_tracker

    @property
    def margin_guard(self):
        """Access margin guard directly."""
        return self._margin_guard

    @property
    def short_sale_guard(self):
        """Access short sale guard directly."""
        return self._short_sale_guard

    @property
    def corporate_actions(self):
        """Access corporate actions handler directly."""
        return self._corporate_actions

    # =========================
    # Pre-Trade Checks
    # =========================

    def check_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        is_day_trade: bool = False,
        timestamp_ms: Optional[int] = None,
    ) -> RiskEvent:
        """
        Check if a trade is allowed under stock risk rules.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Trade quantity
            price: Trade price
            is_day_trade: Whether this would complete a day trade
            timestamp_ms: Current timestamp

        Returns:
            RiskEvent indicating any violation (NONE if OK)
        """
        # Skip all checks for non-stock trading
        if not self._config.is_stock_trading:
            self._last_event = RiskEvent.NONE
            self._last_event_reason = "Not stock trading - checks skipped"
            return RiskEvent.NONE

        if timestamp_ms is None:
            timestamp_ms = now_ms()

        side = side.upper()
        symbol = symbol.upper()

        # 1. PDT Check (for potential day trades)
        if is_day_trade and self._pdt_tracker is not None:
            can_trade, reason = self._pdt_tracker.can_day_trade(symbol, timestamp_ms)
            if not can_trade:
                self._last_event = RiskEvent.PDT_VIOLATION
                self._last_event_reason = reason
                eb.log_risk({
                    "ts": timestamp_ms,
                    "type": "PDT_VIOLATION",
                    "symbol": symbol,
                    "reason": reason,
                })
                if self._config.strict_mode and not self._config.simulation_mode:
                    return RiskEvent.PDT_VIOLATION

        # 2. Margin Check (for buys)
        if side == "BUY" and self._margin_guard is not None:
            can_buy, reason = self._margin_guard.can_open_position(
                symbol, quantity, price, timestamp_ms
            )
            if not can_buy:
                self._last_event = RiskEvent.MARGIN_CALL
                self._last_event_reason = reason
                eb.log_risk({
                    "ts": timestamp_ms,
                    "type": "MARGIN_VIOLATION",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "reason": reason,
                })
                if self._config.strict_mode and not self._config.simulation_mode:
                    return RiskEvent.MARGIN_CALL

        # 3. Short Sale Check (for sells that open/increase short position)
        if side == "SELL" and self._short_sale_guard is not None:
            # Check if this would be a short sale
            can_short, status = self._short_sale_guard.can_short(
                symbol, price, quantity, timestamp_ms
            )
            if not can_short:
                self._last_event = RiskEvent.SHORT_SALE_RESTRICTED
                self._last_event_reason = f"Short sale restricted: {status.restriction.value}"
                eb.log_risk({
                    "ts": timestamp_ms,
                    "type": "SHORT_SALE_RESTRICTED",
                    "symbol": symbol,
                    "restriction": status.restriction.value,
                })
                if self._config.strict_mode and not self._config.simulation_mode:
                    return RiskEvent.SHORT_SALE_RESTRICTED

        # 4. Corporate Actions Check
        if self._corporate_actions is not None:
            is_short = side == "SELL"  # Simplification
            warning = self._corporate_actions.check_dividend_warning(symbol, is_short)
            if warning:
                logger.warning(f"StockRiskGuard: {warning}")
                # Don't block, just warn

        self._last_event = RiskEvent.NONE
        self._last_event_reason = "All stock risk checks passed"
        return RiskEvent.NONE

    def check_short_sale(
        self,
        symbol: str,
        price: float,
        quantity: float,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Specifically check if a short sale is allowed.

        Args:
            symbol: Trading symbol
            price: Short sale price
            quantity: Short quantity
            timestamp_ms: Current timestamp

        Returns:
            (can_short, reason) tuple
        """
        if not self._config.is_stock_trading:
            return True, "Not stock trading"

        if self._short_sale_guard is None:
            return True, "Short sale guard not enabled"

        can_short, status = self._short_sale_guard.can_short(
            symbol, price, quantity, timestamp_ms
        )
        return can_short, status.restriction.value

    def check_day_trade(
        self,
        symbol: str,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Check if a day trade is allowed.

        Args:
            symbol: Trading symbol
            timestamp_ms: Current timestamp

        Returns:
            (can_day_trade, reason) tuple
        """
        if not self._config.is_stock_trading:
            return True, "Not stock trading"

        if self._pdt_tracker is None:
            return True, "PDT tracker not enabled"

        return self._pdt_tracker.can_day_trade(symbol, timestamp_ms)

    def check_margin(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Check if a position can be opened within margin requirements.

        Args:
            symbol: Trading symbol
            quantity: Position size
            price: Price
            timestamp_ms: Current timestamp

        Returns:
            (can_open, reason) tuple
        """
        if not self._config.is_stock_trading:
            return True, "Not stock trading"

        if self._margin_guard is None:
            return True, "Margin guard not enabled"

        return self._margin_guard.can_open_position(symbol, quantity, price, timestamp_ms)

    # =========================
    # Post-Trade Recording
    # =========================

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp_ms: int,
        is_opening: bool = True,
    ) -> None:
        """
        Record a completed trade for tracking.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Trade quantity
            price: Trade price
            timestamp_ms: Trade timestamp
            is_opening: Whether this opens (vs closes) a position
        """
        if not self._config.is_stock_trading:
            return

        symbol = symbol.upper()
        side = side.upper()

        # Update PDT tracking
        if self._pdt_tracker is not None:
            if is_opening:
                pdt_side = "LONG" if side == "BUY" else "SHORT"
                self._pdt_tracker.record_open(
                    symbol, pdt_side, quantity, price, timestamp_ms
                )
            else:
                self._pdt_tracker.record_close(
                    symbol, quantity, price, timestamp_ms
                )

        # Update short sale last price
        if self._short_sale_guard is not None:
            self._short_sale_guard.update_last_sale(symbol, price, timestamp_ms)

    def record_day_trade(
        self,
        symbol: str,
        timestamp_ms: int,
        buy_price: float = 0.0,
        sell_price: float = 0.0,
        quantity: float = 1.0,
    ) -> None:
        """
        Directly record a completed day trade.

        Args:
            symbol: Trading symbol
            timestamp_ms: Trade timestamp
            buy_price: Buy price
            sell_price: Sell price
            quantity: Trade quantity
        """
        if not self._config.is_stock_trading:
            return

        if self._pdt_tracker is not None:
            from services.pdt_tracker import DayTradeType
            self._pdt_tracker.record_day_trade(
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                trade_type=DayTradeType.LONG_ROUND_TRIP,
                buy_price=buy_price,
                sell_price=sell_price,
                quantity=quantity,
            )

    # =========================
    # State Updates
    # =========================

    def update_account_equity(self, equity: float, cash: float = 0.0) -> None:
        """
        Update account equity for margin and PDT calculations.

        Args:
            equity: Total account equity
            cash: Cash balance (optional)
        """
        if not self._config.is_stock_trading:
            return

        if self._pdt_tracker is not None:
            self._pdt_tracker.account_equity = equity

        if self._margin_guard is not None:
            self._margin_guard.set_equity(equity, cash)

    def trigger_circuit_breaker(
        self,
        symbol: str,
        timestamp_ms: int,
        duration_ms: Optional[int] = None,
    ) -> None:
        """
        Trigger short sale circuit breaker (Rule 201).

        Args:
            symbol: Symbol to restrict
            timestamp_ms: Current timestamp
            duration_ms: Custom duration (default: ~1.5 trading days)
        """
        if not self._config.is_stock_trading:
            return

        if self._short_sale_guard is not None:
            self._short_sale_guard.trigger_circuit_breaker(
                symbol, timestamp_ms, duration_ms
            )

    def add_corporate_action(
        self,
        symbol: str,
        action_type: str,
        ex_date: date,
        dividend_amount: float = 0.0,
        split_ratio: Tuple[int, int] = (1, 1),
    ) -> None:
        """
        Add a corporate action.

        Args:
            symbol: Trading symbol
            action_type: "DIVIDEND", "STOCK_SPLIT", etc.
            ex_date: Ex-dividend or effective date
            dividend_amount: Dividend per share (for dividends)
            split_ratio: (new_shares, old_shares) for splits
        """
        if not self._config.is_stock_trading:
            return

        if self._corporate_actions is not None:
            from services.stock_risk_guards import CorporateAction, CorporateActionType
            try:
                action = CorporateAction(
                    symbol=symbol,
                    action_type=CorporateActionType(action_type),
                    ex_date=ex_date,
                    dividend_amount=dividend_amount,
                    split_ratio=split_ratio,
                )
                self._corporate_actions.add_action(action)
            except ValueError:
                logger.warning(f"Unknown corporate action type: {action_type}")

    # =========================
    # Status and Reporting
    # =========================

    def get_pdt_status(self, timestamp_ms: Optional[int] = None) -> Dict[str, Any]:
        """Get PDT status summary."""
        if not self._config.is_stock_trading or self._pdt_tracker is None:
            return {"enabled": False}

        return self._pdt_tracker.get_summary(timestamp_ms)

    def get_margin_status(self) -> Dict[str, Any]:
        """Get margin status summary."""
        if not self._config.is_stock_trading or self._margin_guard is None:
            return {"enabled": False}

        status = self._margin_guard.get_margin_status()
        return {
            "enabled": True,
            "equity": status.equity,
            "buying_power": status.buying_power,
            "margin_used": status.margin_used,
            "margin_call_type": status.margin_call_type.value,
            "margin_call_amount": status.margin_call_amount,
        }

    def get_short_status(self, symbol: str, timestamp_ms: Optional[int] = None) -> Dict[str, Any]:
        """Get short sale status for a symbol."""
        if not self._config.is_stock_trading or self._short_sale_guard is None:
            return {"enabled": False}

        status = self._short_sale_guard.get_short_status(symbol, timestamp_ms)
        return {
            "enabled": True,
            "symbol": status.symbol,
            "restriction": status.restriction.value,
            "is_shortable": status.is_shortable,
            "is_easy_to_borrow": status.is_easy_to_borrow,
            "borrow_rate": status.borrow_rate,
            "circuit_breaker_active": status.circuit_breaker_active,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Get comprehensive status snapshot."""
        return {
            "market_type": self._config.market_type,
            "is_stock_trading": self._config.is_stock_trading,
            "last_event": self._last_event.name,
            "last_event_reason": self._last_event_reason,
            "pdt_enabled": self._pdt_tracker is not None,
            "margin_enabled": self._margin_guard is not None,
            "short_sale_enabled": self._short_sale_guard is not None,
            "corporate_actions_enabled": self._corporate_actions is not None,
        }

    # =========================
    # Reset
    # =========================

    def reset(self) -> None:
        """Reset all stock risk guards state."""
        self._last_event = RiskEvent.NONE
        self._last_event_reason = ""

        if self._pdt_tracker is not None:
            self._pdt_tracker.reset()

        if self._margin_guard is not None:
            self._margin_guard.clear_positions()

        if self._short_sale_guard is not None:
            self._short_sale_guard.clear_htb_cache()

        logger.debug("StockRiskGuard: State reset")


# =========================
# Factory Functions
# =========================

def create_stock_risk_guard(
    market_type: str = "EQUITY",
    account_equity: float = 30_000.0,
    simulation_mode: bool = False,
    strict_mode: bool = True,
) -> StockRiskGuard:
    """
    Create a StockRiskGuard with common defaults.

    Args:
        market_type: "EQUITY" for stocks, "CRYPTO_SPOT" for crypto
        account_equity: Account equity for PDT/margin calculations
        simulation_mode: If True, warn but don't block
        strict_mode: If True, enforce limits strictly

    Returns:
        Configured StockRiskGuard instance
    """
    config = StockRiskConfig(
        market_type=market_type,
        pdt_account_equity=account_equity,
        simulation_mode=simulation_mode,
        strict_mode=strict_mode,
    )
    return StockRiskGuard(config)


def create_combined_risk_guard(
    risk_config: Optional[RiskConfig] = None,
    stock_config: Optional[StockRiskConfig] = None,
) -> Tuple[RiskGuard, Optional[StockRiskGuard]]:
    """
    Create both RiskGuard and StockRiskGuard.

    Returns:
        (RiskGuard, StockRiskGuard) tuple
        StockRiskGuard is None if market_type is crypto

    Usage:
        risk_guard, stock_guard = create_combined_risk_guard(
            stock_config=StockRiskConfig(market_type="EQUITY")
        )

        # Use risk_guard for core checks
        event = risk_guard.on_action_proposed(state, proto)

        # Use stock_guard for stock-specific checks
        if stock_guard:
            event = stock_guard.check_trade(...)
    """
    risk_guard = RiskGuard(risk_config)

    if stock_config is None:
        return risk_guard, None

    if stock_config.is_stock_trading:
        stock_guard = StockRiskGuard(stock_config)
        return risk_guard, stock_guard

    return risk_guard, None


# =========================
# Crypto Futures Risk Guard (Phase 6A)
# =========================

@dataclass
class CryptoFuturesRiskConfig:
    """Configuration for crypto futures risk management.

    Reference: Binance USDT-M Futures risk management
    Documentation: docs/FUTURES_INTEGRATION_PLAN.md (Phase 6A)
    """

    # Asset class detection
    market_type: str = "CRYPTO_FUTURES"  # CRYPTO_FUTURES, CRYPTO_PERP

    # Leverage settings
    leverage_enabled: bool = True
    max_account_leverage: float = 20.0  # Maximum account-wide leverage
    default_leverage: int = 10  # Default leverage for new positions

    # Margin settings
    margin_enabled: bool = True
    margin_warning_threshold: float = 2.0  # 200% margin ratio
    margin_danger_threshold: float = 1.5   # 150% margin ratio
    margin_critical_threshold: float = 1.2  # 120% margin ratio
    margin_liquidation_threshold: float = 1.05  # 105% margin ratio

    # Funding rate exposure
    funding_enabled: bool = True
    max_funding_exposure_pct: float = 0.1  # 10% of equity
    funding_rate_warning_threshold: float = 0.001  # 0.1% per interval

    # Concentration limits
    concentration_enabled: bool = True
    max_single_symbol_pct: float = 0.25  # 25% max in single symbol
    max_correlated_group_pct: float = 0.40  # 40% in correlated assets

    # ADL risk
    adl_enabled: bool = True
    adl_warning_percentile: float = 70.0  # Warn if above 70th percentile
    adl_critical_percentile: float = 90.0  # Critical if above 90th percentile

    # Notification settings
    notification_enabled: bool = True
    notification_cooldown_sec: float = 300.0  # 5 minutes between notifications
    escalation_enabled: bool = True

    # General settings
    simulation_mode: bool = False  # Warn but don't block
    strict_mode: bool = True  # Block vs warn

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CryptoFuturesRiskConfig":
        """Create config from mapping."""
        return cls(
            market_type=str(data.get("market_type", "CRYPTO_FUTURES")),
            leverage_enabled=bool(data.get("leverage_enabled", True)),
            max_account_leverage=float(data.get("max_account_leverage", 20.0)),
            default_leverage=int(data.get("default_leverage", 10)),
            margin_enabled=bool(data.get("margin_enabled", True)),
            margin_warning_threshold=float(data.get("margin_warning_threshold", 2.0)),
            margin_danger_threshold=float(data.get("margin_danger_threshold", 1.5)),
            margin_critical_threshold=float(data.get("margin_critical_threshold", 1.2)),
            margin_liquidation_threshold=float(data.get("margin_liquidation_threshold", 1.05)),
            funding_enabled=bool(data.get("funding_enabled", True)),
            max_funding_exposure_pct=float(data.get("max_funding_exposure_pct", 0.1)),
            concentration_enabled=bool(data.get("concentration_enabled", True)),
            max_single_symbol_pct=float(data.get("max_single_symbol_pct", 0.25)),
            max_correlated_group_pct=float(data.get("max_correlated_group_pct", 0.40)),
            adl_enabled=bool(data.get("adl_enabled", True)),
            notification_enabled=bool(data.get("notification_enabled", True)),
            notification_cooldown_sec=float(data.get("notification_cooldown_sec", 300.0)),
            simulation_mode=bool(data.get("simulation_mode", False)),
            strict_mode=bool(data.get("strict_mode", True)),
        )

    @property
    def is_futures_trading(self) -> bool:
        """Check if this is futures trading."""
        return self.market_type.upper() in ("CRYPTO_FUTURES", "CRYPTO_PERP", "FUTURES")


class CryptoFuturesRiskGuard:
    """
    Crypto futures risk management guard for Binance USDT-M Perpetuals.

    Integrates:
    1. Leverage Guard - Tiered bracket leverage limits
    2. Margin Guard - Margin ratio monitoring with levels
    3. Margin Call Notifier - Escalation and notification
    4. Funding Exposure Guard - Funding rate risk management
    5. Concentration Guard - Position concentration limits
    6. ADL Risk Guard - Auto-deleveraging risk monitoring

    BACKWARD COMPATIBLE: For spot trading, all checks are skipped
    automatically based on market_type configuration.

    Usage:
        # For futures trading
        futures_guard = CryptoFuturesRiskGuard(
            CryptoFuturesRiskConfig(market_type="CRYPTO_FUTURES")
        )

        # Pre-trade check
        event = futures_guard.check_trade(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.1,
            leverage=20,
            mark_price=50000.0,
        )

        if event != RiskEvent.NONE:
            reject_trade(event)

        # Get comprehensive risk summary
        summary = futures_guard.get_risk_summary()

    For spot trading:
        # All checks automatically disabled
        spot_guard = CryptoFuturesRiskGuard(
            CryptoFuturesRiskConfig(market_type="CRYPTO_SPOT")
        )
        event = spot_guard.check_trade(...)  # Always returns NONE

    Reference:
        - Binance Futures Risk Management: https://www.binance.com/en/support/faq/360033162192
        - Margin and Leverage: https://www.binance.com/en/support/faq/360033162192
        - ADL System: https://www.binance.com/en/support/faq/360033525271
    """

    def __init__(
        self,
        config: Optional[CryptoFuturesRiskConfig] = None,
    ) -> None:
        """
        Initialize CryptoFuturesRiskGuard.

        Args:
            config: Crypto futures risk configuration
        """
        self._config = config or CryptoFuturesRiskConfig()
        self._last_event: RiskEvent = RiskEvent.NONE
        self._last_event_reason: str = ""

        # Lazy initialization of sub-guards
        self._leverage_guard = None
        self._margin_guard = None
        self._margin_notifier = None
        self._funding_guard = None
        self._concentration_guard = None
        self._adl_guard = None
        self._unified_guard = None

        # Only initialize if this is futures trading
        if self._config.is_futures_trading:
            self._initialize_guards()

        logger.debug(
            f"CryptoFuturesRiskGuard initialized: market_type={self._config.market_type}, "
            f"is_futures={self._config.is_futures_trading}"
        )

    def _initialize_guards(self) -> None:
        """Initialize sub-guards for futures trading."""
        try:
            from services.futures_risk_guards import (
                FuturesLeverageGuard,
                FuturesMarginGuard,
                MarginCallNotifier,
                FundingExposureGuard,
                ConcentrationGuard,
                ADLRiskGuard,
                FuturesRiskGuard as UnifiedFuturesGuard,
                LeverageConfig,
                MarginGuardConfig,
                NotifierConfig,
                FundingGuardConfig,
                ConcentrationConfig,
                ADLConfig,
            )

            # Leverage Guard
            if self._config.leverage_enabled:
                # If concentration is disabled, use 1.0 (100%) as limit to effectively disable it
                concentration_limit = (
                    self._config.max_single_symbol_pct
                    if self._config.concentration_enabled
                    else 1.0
                )
                correlated_limit = (
                    self._config.max_correlated_group_pct
                    if self._config.concentration_enabled
                    else 1.0
                )
                self._leverage_guard = FuturesLeverageGuard(
                    max_account_leverage=int(self._config.max_account_leverage),
                    max_symbol_leverage=125,  # Default max symbol leverage
                    concentration_limit=concentration_limit,
                    correlated_limit=correlated_limit,
                )

            # Margin Guard
            if self._config.margin_enabled:
                self._margin_guard = FuturesMarginGuard(
                    margin_calculator=None,  # Will be set when margin_calculator is available
                    warning_level=Decimal(str(self._config.margin_warning_threshold)),
                    danger_level=Decimal(str(self._config.margin_danger_threshold)),
                    critical_level=Decimal(str(self._config.margin_critical_threshold)),
                )

            # Margin Call Notifier
            if self._config.notification_enabled:
                self._margin_notifier = MarginCallNotifier(
                    cooldown_seconds=self._config.notification_cooldown_sec,
                )

            # Funding Exposure Guard
            if self._config.funding_enabled:
                self._funding_guard = FundingExposureGuard(
                    warning_threshold=Decimal(str(self._config.funding_rate_warning_threshold)),
                )

            # Concentration Guard
            if self._config.concentration_enabled:
                self._concentration_guard = ConcentrationGuard(
                    single_symbol_limit=self._config.max_single_symbol_pct,
                    correlated_group_limit=self._config.max_correlated_group_pct,
                )

            # ADL Risk Guard
            if self._config.adl_enabled:
                self._adl_guard = ADLRiskGuard(
                    warning_percentile=self._config.adl_warning_percentile,
                    critical_percentile=self._config.adl_critical_percentile,
                )

            logger.info("CryptoFuturesRiskGuard: All sub-guards initialized")

        except ImportError as e:
            logger.warning(f"CryptoFuturesRiskGuard: Could not import sub-guards: {e}")

    # =========================
    # Properties
    # =========================

    @property
    def config(self) -> CryptoFuturesRiskConfig:
        """Get current configuration."""
        return self._config

    @property
    def last_event(self) -> RiskEvent:
        """Get last risk event."""
        return self._last_event

    @property
    def last_event_reason(self) -> str:
        """Get reason for last risk event."""
        return self._last_event_reason

    @property
    def leverage_guard(self):
        """Access leverage guard directly."""
        return self._leverage_guard

    @property
    def margin_guard(self):
        """Access margin guard directly."""
        return self._margin_guard

    @property
    def funding_guard(self):
        """Access funding guard directly."""
        return self._funding_guard

    @property
    def concentration_guard(self):
        """Access concentration guard directly."""
        return self._concentration_guard

    @property
    def adl_guard(self):
        """Access ADL guard directly."""
        return self._adl_guard

    # =========================
    # Pre-Trade Checks
    # =========================

    def check_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: int,
        mark_price: float,
        entry_price: Optional[float] = None,
        position_notional: Optional[float] = None,
        account_equity: Optional[float] = None,
        total_margin_used: Optional[float] = None,
        funding_rate: Optional[float] = None,
        positions: Optional[List[Dict[str, Any]]] = None,
        timestamp_ms: Optional[int] = None,
    ) -> RiskEvent:
        """
        Check if a futures trade is allowed under risk rules.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            side: "LONG" or "SHORT"
            quantity: Trade quantity in base currency
            leverage: Position leverage
            mark_price: Current mark price
            entry_price: Entry price (default: mark_price)
            position_notional: Total position notional (default: calculated)
            account_equity: Account equity (default: None)
            total_margin_used: Total margin used (default: None)
            funding_rate: Current funding rate (default: None)
            positions: List of current positions (default: None)
            timestamp_ms: Current timestamp

        Returns:
            RiskEvent indicating any violation (NONE if OK)
        """
        # Skip all checks for non-futures trading
        if not self._config.is_futures_trading:
            self._last_event = RiskEvent.NONE
            self._last_event_reason = "Not futures trading - checks skipped"
            return RiskEvent.NONE

        if timestamp_ms is None:
            timestamp_ms = now_ms()

        side = side.upper()
        symbol = symbol.upper()

        if entry_price is None:
            entry_price = mark_price

        if position_notional is None:
            position_notional = quantity * mark_price

        # 1. Leverage Check
        if self._leverage_guard is not None:
            # Create a position-like object for validation
            from types import SimpleNamespace
            proposed_position = SimpleNamespace(
                symbol=symbol,
                qty=Decimal(str(quantity)),
                entry_price=Decimal(str(entry_price)),
                leverage=leverage,
            )
            current_pos_list = positions if positions else []
            result = self._leverage_guard.validate_new_position(
                proposed_position=proposed_position,
                current_positions=current_pos_list,
                account_balance=Decimal(str(account_equity)) if account_equity else Decimal("0"),
            )
            if not result.is_valid:
                self._last_event = RiskEvent.LEVERAGE_VIOLATION
                self._last_event_reason = result.error_message or "Leverage violation"
                eb.log_risk({
                    "ts": timestamp_ms,
                    "type": "LEVERAGE_VIOLATION",
                    "symbol": symbol,
                    "requested_leverage": leverage,
                    "max_leverage": result.max_allowed_leverage,
                    "reason": result.error_message,
                })
                if self._config.strict_mode and not self._config.simulation_mode:
                    return RiskEvent.LEVERAGE_VIOLATION

        # 2. Margin Check
        if self._margin_guard is not None and account_equity is not None:
            required_margin = position_notional / leverage if leverage > 0 else position_notional
            margin_used = (total_margin_used or 0) + required_margin
            margin_ratio = account_equity / margin_used if margin_used > 0 else float('inf')

            result = self._margin_guard.check_margin_ratio(
                margin_ratio=margin_ratio,
                account_equity=account_equity,
                total_margin_used=margin_used,
                symbol=symbol,
            )

            # Map margin level to risk event
            if result.requires_liquidation:
                self._last_event = RiskEvent.FUTURES_MARGIN_LIQUIDATION
                self._last_event_reason = f"Margin ratio {margin_ratio:.2%} at liquidation level"
                eb.log_risk({
                    "ts": timestamp_ms,
                    "type": "FUTURES_MARGIN_LIQUIDATION",
                    "symbol": symbol,
                    "margin_ratio": margin_ratio,
                    "level": result.level.value,
                })
                if self._config.strict_mode and not self._config.simulation_mode:
                    return RiskEvent.FUTURES_MARGIN_LIQUIDATION

            elif result.requires_reduction:
                self._last_event = RiskEvent.FUTURES_MARGIN_DANGER
                self._last_event_reason = f"Margin ratio {margin_ratio:.2%} requires position reduction"
                eb.log_risk({
                    "ts": timestamp_ms,
                    "type": "FUTURES_MARGIN_DANGER",
                    "symbol": symbol,
                    "margin_ratio": margin_ratio,
                    "level": result.level.value,
                })
                if self._config.strict_mode and not self._config.simulation_mode:
                    return RiskEvent.FUTURES_MARGIN_DANGER

            elif result.level.value in ("warning", "danger"):
                self._last_event = RiskEvent.FUTURES_MARGIN_WARNING
                self._last_event_reason = f"Margin ratio {margin_ratio:.2%} below warning threshold"
                eb.log_risk({
                    "ts": timestamp_ms,
                    "type": "FUTURES_MARGIN_WARNING",
                    "symbol": symbol,
                    "margin_ratio": margin_ratio,
                    "level": result.level.value,
                })
                # Don't block on warning, just log

        # 3. Funding Exposure Check
        if self._funding_guard is not None and funding_rate is not None and account_equity is not None:
            result = self._funding_guard.check_funding_exposure(
                funding_rate=funding_rate,
                position_notional=position_notional,
                account_equity=account_equity,
                side=side,
            )
            if not result.is_acceptable:
                self._last_event = RiskEvent.FUNDING_EXPOSURE
                self._last_event_reason = f"Funding exposure {result.exposure_pct:.2%} exceeds limit"
                eb.log_risk({
                    "ts": timestamp_ms,
                    "type": "FUNDING_EXPOSURE",
                    "symbol": symbol,
                    "funding_rate": funding_rate,
                    "exposure_pct": result.exposure_pct,
                    "level": result.level.value,
                })
                if self._config.strict_mode and not self._config.simulation_mode:
                    return RiskEvent.FUNDING_EXPOSURE

        # 4. Concentration Check
        if self._concentration_guard is not None and positions is not None and account_equity is not None:
            result = self._concentration_guard.check_concentration(
                symbol=symbol,
                position_notional=position_notional,
                all_positions=positions,
                account_equity=account_equity,
            )
            if not result.is_within_limits:
                self._last_event = RiskEvent.CONCENTRATION_LIMIT
                self._last_event_reason = f"Position concentration {result.symbol_concentration:.2%} exceeds limit"
                eb.log_risk({
                    "ts": timestamp_ms,
                    "type": "CONCENTRATION_LIMIT",
                    "symbol": symbol,
                    "symbol_concentration": result.symbol_concentration,
                    "group_concentration": result.group_concentration,
                })
                if self._config.strict_mode and not self._config.simulation_mode:
                    return RiskEvent.CONCENTRATION_LIMIT

        self._last_event = RiskEvent.NONE
        self._last_event_reason = "All futures risk checks passed"
        return RiskEvent.NONE

    def check_adl_risk(
        self,
        pnl_percentile: float,
        leverage_percentile: float,
        timestamp_ms: Optional[int] = None,
    ) -> RiskEvent:
        """
        Check ADL (Auto-Deleveraging) risk.

        Args:
            pnl_percentile: PnL percentile (0-100)
            leverage_percentile: Leverage percentile (0-100)
            timestamp_ms: Current timestamp

        Returns:
            RiskEvent indicating ADL risk level
        """
        if not self._config.is_futures_trading:
            return RiskEvent.NONE

        if self._adl_guard is None:
            return RiskEvent.NONE

        if timestamp_ms is None:
            timestamp_ms = now_ms()

        result = self._adl_guard.check_adl_risk(
            pnl_percentile=pnl_percentile,
            leverage_percentile=leverage_percentile,
        )

        if result.level.value in ("critical", "high"):
            self._last_event = RiskEvent.ADL_RISK
            self._last_event_reason = f"ADL risk {result.level.value}: score={result.adl_score:.2f}"
            eb.log_risk({
                "ts": timestamp_ms,
                "type": "ADL_RISK",
                "adl_score": result.adl_score,
                "pnl_percentile": pnl_percentile,
                "leverage_percentile": leverage_percentile,
                "level": result.level.value,
            })
            return RiskEvent.ADL_RISK

        return RiskEvent.NONE

    # =========================
    # Risk Summary
    # =========================

    def get_risk_summary(
        self,
        positions: Optional[List[Dict[str, Any]]] = None,
        account_equity: Optional[float] = None,
        total_margin_used: Optional[float] = None,
        funding_rates: Optional[Dict[str, float]] = None,
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive risk summary.

        Args:
            positions: List of current positions
            account_equity: Account equity
            total_margin_used: Total margin used
            funding_rates: Dict of symbol -> funding rate
            timestamp_ms: Current timestamp

        Returns:
            Dict with risk summary information
        """
        if not self._config.is_futures_trading:
            return {"enabled": False, "market_type": self._config.market_type}

        if self._unified_guard is not None:
            return self._unified_guard.get_risk_summary(
                positions=positions,
                account_equity=account_equity,
                total_margin_used=total_margin_used,
                funding_rates=funding_rates,
                timestamp_ms=timestamp_ms,
            ).to_dict()

        return {
            "enabled": True,
            "market_type": self._config.market_type,
            "leverage_guard": self._leverage_guard is not None,
            "margin_guard": self._margin_guard is not None,
            "funding_guard": self._funding_guard is not None,
            "concentration_guard": self._concentration_guard is not None,
            "adl_guard": self._adl_guard is not None,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Get status snapshot."""
        return {
            "market_type": self._config.market_type,
            "is_futures_trading": self._config.is_futures_trading,
            "last_event": self._last_event.name,
            "last_event_reason": self._last_event_reason,
            "leverage_enabled": self._leverage_guard is not None,
            "margin_enabled": self._margin_guard is not None,
            "funding_enabled": self._funding_guard is not None,
            "concentration_enabled": self._concentration_guard is not None,
            "adl_enabled": self._adl_guard is not None,
            "notification_enabled": self._margin_notifier is not None,
        }

    # =========================
    # Reset
    # =========================

    def reset(self) -> None:
        """Reset all futures risk guards state."""
        self._last_event = RiskEvent.NONE
        self._last_event_reason = ""

        if self._margin_notifier is not None:
            self._margin_notifier.reset_cooldowns()

        logger.debug("CryptoFuturesRiskGuard: State reset")


# =========================
# Crypto Futures Factory Functions
# =========================

def create_crypto_futures_risk_guard(
    market_type: str = "CRYPTO_FUTURES",
    max_account_leverage: float = 20.0,
    simulation_mode: bool = False,
    strict_mode: bool = True,
) -> CryptoFuturesRiskGuard:
    """
    Create a CryptoFuturesRiskGuard with common defaults.

    Args:
        market_type: "CRYPTO_FUTURES", "CRYPTO_PERP", "CRYPTO_SPOT" (disabled)
        max_account_leverage: Maximum account-wide leverage
        simulation_mode: If True, warn but don't block
        strict_mode: If True, enforce limits strictly

    Returns:
        Configured CryptoFuturesRiskGuard instance
    """
    config = CryptoFuturesRiskConfig(
        market_type=market_type,
        max_account_leverage=max_account_leverage,
        simulation_mode=simulation_mode,
        strict_mode=strict_mode,
    )
    return CryptoFuturesRiskGuard(config)


def create_full_risk_guard(
    risk_config: Optional[RiskConfig] = None,
    stock_config: Optional[StockRiskConfig] = None,
    futures_config: Optional[CryptoFuturesRiskConfig] = None,
) -> Tuple[RiskGuard, Optional[StockRiskGuard], Optional[CryptoFuturesRiskGuard]]:
    """
    Create all risk guards (core, stock, futures).

    Returns:
        (RiskGuard, StockRiskGuard, CryptoFuturesRiskGuard) tuple
        Asset-specific guards are None if market_type doesn't match

    Usage:
        # For crypto futures
        risk_guard, _, futures_guard = create_full_risk_guard(
            futures_config=CryptoFuturesRiskConfig(market_type="CRYPTO_FUTURES")
        )

        # For stocks
        risk_guard, stock_guard, _ = create_full_risk_guard(
            stock_config=StockRiskConfig(market_type="EQUITY")
        )

        # For crypto spot (core only)
        risk_guard, _, _ = create_full_risk_guard()
    """
    risk_guard = RiskGuard(risk_config)

    stock_guard = None
    if stock_config is not None and stock_config.is_stock_trading:
        stock_guard = StockRiskGuard(stock_config)

    futures_guard = None
    if futures_config is not None and futures_config.is_futures_trading:
        futures_guard = CryptoFuturesRiskGuard(futures_config)

    return risk_guard, stock_guard, futures_guard


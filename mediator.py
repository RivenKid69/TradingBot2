from __future__ import annotations

"""
Mediator — координатор между средой, LOB/симулятором исполнения, RiskGuard и EventBus.
Держит TTL ордеров, прокидывает действия агента и обновляет портфельное состояние.

Контракт ожиданий:
- env_ref.state имеет как минимум атрибуты: units: float, cash: float, max_position: float (опционально).
- env_ref.lob (опционально): объект с методами add_limit_order, remove_order, match_market_order.
  Если отсутствует — используется _DummyLOB (ничего не делает, но не ломает пайплайн).
- ExecutionSimulator (если используется) должен предоставлять внутренний SimStepReport (см. execution_sim.py),
  а наружу для логирования и анализа использовать единые core_models.ExecReport через compat_shims/sim_adapter.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from core_models import Order, ExecReport, Position, TradeLogRow
    from core_contracts import TradeExecutor, RiskGuards

import math
import numpy as np

from core_models import ExecReport, TradeLogRow, Side, OrderType, Liquidity, ExecStatus

# Import obs_builder for observation vector construction
try:
    from obs_builder import build_observation_vector
    _HAVE_OBS_BUILDER = True
except ImportError:
    _HAVE_OBS_BUILDER = False
from core_events import EventType, OrderEvent, FillEvent
from compat_shims import sim_report_dict_to_core_exec_reports
import event_bus as eb
from impl_latency import LatencyImpl
from core_constants import PRICE_SCALE
from utils import SignalRateLimiter
from clock import now_ms
try:
    from quantizer import Quantizer, load_filters
except Exception:  # pragma: no cover - soft dependency
    Quantizer = None  # type: ignore

    def load_filters(path: str):  # type: ignore
        return {}

try:
    from impl_quantizer import QuantizerImpl  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    QuantizerImpl = None  # type: ignore

try:
    import event_bus
except Exception:
    # мягкая деградация при отсутствии event_bus
    class _Stub:
        def configure(self, *a, **k): return ""
        def log_trade(self, *a, **k): pass
        def log_risk(self, *a, **k): pass
        def flush(self): pass
        def run_dir(self): return ""
    event_bus = _Stub()  # type: ignore

from action_proto import ActionProto, ActionType
from risk_guard import RiskConfig, RiskGuard
from compat_shims import sim_report_dict_to_core_exec_reports
from core_models import as_dict
from order_shims import actionproto_to_order, legacy_decision_to_order, OrderContext

# ExecutionSimulator и SimStepReport (как внутренний тип отчёта, импортируемый как ExecReport)
# опциональны; при отсутствии работаем напрямую с LOB
try:
    from execution_sim import SimStepReport as ExecReport, ExecutionSimulator  # type: ignore
    _HAVE_EXEC_SIM = True
except Exception:
    ExecReport = None  # type: ignore
    ExecutionSimulator = None  # type: ignore
    _HAVE_EXEC_SIM = False


# ------------------------------ Вспомогательная заглушка LOB ------------------------------

class _DummyLOB:
    """Минималистичная заглушка для разработки без Cython."""
    _next_id: int

    def __init__(self):
        self._next_id = 1

    def add_limit_order(self, is_buy_side: bool, price_ticks: int, volume: float, timestamp: int,
                        taker_is_agent: bool = True) -> Tuple[int, int]:
        oid = self._next_id
        self._next_id += 1
        # (order_id, fake_queue_position)
        return int(oid), 0

    def remove_order(self, is_buy_side: bool, price_ticks: int, order_id: int) -> bool:
        return True

    def match_market_order(self, is_buy_side: bool, volume: float, timestamp: int,
                           taker_is_agent: bool, out_prices=None, out_volumes=None,
                           out_is_buy=None, out_is_self=None, out_ids=None, max_len: int = 0):
        # Заглушка: не исполняем, возвращаем ноль сделок и нулевую комиссию
        return 0, 0.0


# ------------------------------ Mediator ------------------------------

@dataclass
class _EnvStateView:
    units: float
    cash: float
    max_position: float = 0.0


class Mediator:
    def __init__(
        self,
        env_ref: Any,
        *,
        event_level: int = 0,
        use_exec_sim: Optional[bool] = None,
        latency_steps: int = 0,
        slip_k: float = 0.0,
        seed: int = 0,
        latency_cfg: dict | None = None,
        rate_limit: float | None = None,
        backoff_base: float | None = None,
        max_backoff: float | None = None,
    ):
        """
        env_ref — ссылка на «среду» (должна держать .state и, опционально, .lob)
        event_level — уровень логов EventBus (0/1/2)
        use_exec_sim — если None, выбираем автоматически по наличию execution_sim
        latency_steps/slip_k/seed — параметры ExecutionSimulator (если используется)
        latency_cfg — параметры модели латентности для ExecutionSimulator
        """
        self.env = env_ref
        try:
            self.event_level = int(event_level)
        except Exception:
            self.event_level = 0
        rc = getattr(env_ref, "run_config", None)

        if rate_limit is None and rc is not None:
            rate_limit = getattr(rc, "max_signals_per_sec", None)
        if backoff_base is None:
            backoff_base = getattr(rc, "backoff_base_s", 2.0) if rc is not None else 2.0
        if max_backoff is None:
            max_backoff = getattr(rc, "max_backoff_s", 60.0) if rc is not None else 60.0

        # EventBus
        try:
            event_bus.configure(level=event_level)
        except Exception:
            pass

        # RiskGuard c параметрами из env_ref (если заданы)
        self.risk = RiskGuard(
            RiskConfig(
                max_abs_position=float(getattr(env_ref, "max_abs_position", 1e12)),
                max_notional=float(getattr(env_ref, "max_notional", 2e12)),
                max_drawdown_pct=float(getattr(env_ref, "max_drawdown_pct", 1.0)),
                intrabar_dd_pct=float(getattr(env_ref, "intrabar_dd_pct", 0.30)),
                dd_window=int(getattr(env_ref, "dd_window", 500)),
                bankruptcy_cash_th=float(getattr(env_ref, "bankruptcy_cash_th", -1e12)),
            )
        )

        # LOB (реальный или заглушка)
        self.lob = getattr(env_ref, "lob", None) or _DummyLOB()

        # Rate limiter for outbound signals
        self._rate_limiter = (
            SignalRateLimiter(rate_limit, backoff_base, max_backoff)
            if rate_limit and rate_limit > 0
            else None
        )

        # Counters for signal statistics
        self.total_signals = 0
        self.delayed_signals = 0
        self.rejected_signals = 0

        # ExecutionSimulator — опционально
        if use_exec_sim is None:
            use_exec_sim = _HAVE_EXEC_SIM
        self._use_exec = bool(use_exec_sim and _HAVE_EXEC_SIM)

        if latency_cfg is None:
            if rc is not None:
                latency_cfg = getattr(rc, "latency", None)

        self._latency_impl: LatencyImpl | None = None
        if self._use_exec:
            self.exec = ExecutionSimulator(latency_steps=latency_steps, slip_k=slip_k, seed=seed)  # type: ignore
            try:
                cfg = dict(latency_cfg or {})
                cfg.setdefault(
                    "symbol",
                    str(getattr(env_ref, "symbol", getattr(env_ref, "base_symbol", ""))),
                )
                l_impl = LatencyImpl.from_dict(cfg)
                l_impl.attach_to(self.exec)
                self._latency_impl = l_impl
            except Exception:
                self._latency_impl = None
        else:
            self.exec = None
            self._latency_impl = None

        # Quantizer (shared with ExecutionSimulator)
        self.quantizer_impl = None
        self.quantizer = None
        self.enforce_ppbs = True

        def _plain_mapping(obj: Any) -> Dict[str, Any]:
            if isinstance(obj, Mapping):
                try:
                    return {str(k): v for k, v in obj.items()}
                except Exception:
                    return dict(obj)
            if hasattr(obj, "dict"):
                try:
                    data = obj.dict(exclude_unset=False)  # type: ignore[call-arg]
                except Exception:
                    data = {}
                if isinstance(data, Mapping):
                    return _plain_mapping(data)
            if hasattr(obj, "__dict__"):
                try:
                    return {
                        str(k): getattr(obj, k)
                        for k in vars(obj)
                        if not str(k).startswith("_")
                    }
                except Exception:
                    return {}
            return {}

        qcfg_raw = getattr(rc, "quantizer", None) if rc is not None else None
        qcfg: Dict[str, Any] = _plain_mapping(qcfg_raw) if qcfg_raw is not None else {}

        filters_path = str(
            qcfg.get("filters_path")
            or qcfg.get("filtersPath")
            or qcfg.get("path")
            or ""
        ).strip()

        strict_raw = qcfg.get("strict_filters")
        if strict_raw is None:
            strict_raw = qcfg.get("strict")
        strict = bool(strict_raw if strict_raw is not None else True)

        enforce_raw = qcfg.get("enforce_percent_price_by_side")
        if enforce_raw is None:
            enforce_raw = qcfg.get("enforcePercentPriceBySide")
        self.enforce_ppbs = bool(enforce_raw if enforce_raw is not None else True)

        quantizer_impl = None
        if QuantizerImpl is not None and filters_path:
            cfg_payload = dict(qcfg)
            cfg_payload.setdefault("path", filters_path)
            cfg_payload.setdefault("filters_path", filters_path)
            try:
                quantizer_impl = QuantizerImpl.from_dict(cfg_payload)
            except Exception:
                quantizer_impl = None

        self.quantizer_impl = quantizer_impl
        if quantizer_impl is not None:
            self.quantizer = getattr(quantizer_impl, "quantizer", None)
            cfg_obj = getattr(quantizer_impl, "cfg", None)
            enforce_attr = getattr(cfg_obj, "enforce_percent_price_by_side", None)
            if enforce_attr is not None:
                self.enforce_ppbs = bool(enforce_attr)
            if self._use_exec and self.exec is not None:
                attach_api = getattr(self.exec, "attach_quantizer", None)
                attached = False
                if callable(attach_api):
                    try:
                        attach_api(impl=quantizer_impl)
                        attached = True
                    except TypeError:
                        attach_api = None
                    except Exception:
                        attach_api = None
                if not attached:
                    try:
                        quantizer_impl.attach_to(self.exec)
                    except Exception:
                        pass
        else:
            if filters_path and Quantizer is not None:
                try:
                    filters, _meta = load_filters(filters_path)
                    if filters:
                        self.quantizer = Quantizer(filters, strict=strict)
                except Exception:
                    self.quantizer = None
            if self._use_exec and self.exec is not None and self.quantizer is not None:
                try:
                    self.exec.set_quantizer(self.quantizer)  # type: ignore[attr-defined]
                    setattr(self.exec, "enforce_ppbs", self.enforce_ppbs)
                    setattr(self.exec, "strict_filters", strict)
                except Exception:
                    pass

        # TTL-очередь: [(order_id, expire_ts)]
        self._ttl_queue: List[Tuple[int, int]] = []

        # Внутренние «ожидаемые» объёмы по последним операциям (для согласования с отчётом)
        self._pending_buy_volume: float = 0.0
        self._pending_sell_volume: float = 0.0
        self._context_row: Any | None = None
        self._context_row_idx: int | None = None
        self._context_timestamp: int | None = None
        self._last_signal_position: float = 0.0

    def _check_rate_limit(self) -> bool:
        """Apply rate limiter using wall-clock milliseconds."""
        if self._rate_limiter is None:
            return True
        self.total_signals += 1
        ts_s = now_ms() / 1000.0
        allowed, status = self._rate_limiter.can_send(ts_s)
        try:
            eb.log_signal_metric(status)
        except Exception:
            pass
        if not allowed:
            if status == "delayed":
                self.delayed_signals += 1
            else:
                self.rejected_signals += 1
        return allowed

    # ------------------------------ Служебное ------------------------------

    def reset(self) -> None:
        """Очистить внутреннее состояние посредника (портфельное состояние живёт в env.state)."""
        # логируем накопленную статистику латентности за предыдущий эпизод
        self.on_episode_end()
        try:
            self.risk.reset()
        except Exception:
            pass
        self._ttl_queue.clear()
        self._pending_buy_volume = 0.0
        self._pending_sell_volume = 0.0
        self.total_signals = 0
        self.delayed_signals = 0
        self.rejected_signals = 0
        self._context_row = None
        self._context_row_idx = None
        self._context_timestamp = None
        self._last_signal_position = 0.0
        self._latest_log_ret_prev = 0.0

    def set_market_context(self, *, row: Any | None = None, row_idx: int | None = None, timestamp: int | None = None) -> None:
        """Store per-step market context passed from the environment."""
        self._context_row = row
        self._context_row_idx = int(row_idx) if row_idx is not None else None
        self._context_timestamp = int(timestamp) if timestamp is not None else None

    def on_episode_end(self) -> None:
        """Запросить и вывести статистику латентности и ограничителя сигналов."""
        if self._latency_impl is not None:
            try:
                stats = self._latency_impl.get_stats()
            except Exception:
                stats = None
            if stats:
                try:
                    event_bus.log_risk({"etype": "LATENCY_STATS", **stats})
                except Exception:
                    pass
                try:
                    self._latency_impl.reset_stats()
                except Exception:
                    pass

        # log signal rate limiter stats
        if self.total_signals > 0:
            delayed_ratio = float(self.delayed_signals) / float(self.total_signals)
            rejected_ratio = float(self.rejected_signals) / float(self.total_signals)
            try:
                event_bus.log_risk(
                    {
                        "etype": "SIGNAL_RATE_STATS",
                        "total": int(self.total_signals),
                        "delayed_ratio": delayed_ratio,
                        "rejected_ratio": rejected_ratio,
                    }
                )
            except Exception:
                pass

    def _state_view(self) -> _EnvStateView:
        st = getattr(self.env, "state", None)
        if st is None:
            # fallback: минимальный стейт
            return _EnvStateView(units=0.0, cash=0.0, max_position=0.0)
        return _EnvStateView(
            units=float(getattr(st, "units", 0.0)),
            cash=float(getattr(st, "cash", 0.0)),
            max_position=float(getattr(st, "max_position", 0.0) or 0.0),
        )

    def _normalize_trades(self, trades: list[Any]) -> list[tuple[float, float, bool, bool]]:
        """Convert heterogeneous trade payloads into (price, volume, is_buy, maker)."""
        normalized: list[tuple[float, float, bool, bool]] = []
        for trade in trades:
            price_val = getattr(trade, "price", None)
            qty_val = getattr(trade, "qty", getattr(trade, "quantity", None))
            side_flag = getattr(trade, "side", None)
            liquidity_flag = getattr(trade, "liquidity", None)
            if price_val is None or qty_val is None:
                if isinstance(trade, Mapping):
                    price_val = trade.get("price")
                    qty_val = trade.get("qty", trade.get("quantity"))
                    if side_flag is None:
                        side_flag = trade.get("side")
                    if liquidity_flag is None:
                        liquidity_flag = trade.get("liquidity")
                elif isinstance(trade, (list, tuple)) and len(trade) >= 2:
                    price_val, qty_val = trade[0], trade[1]
                    if len(trade) > 2:
                        side_flag = trade[2]
                    if len(trade) > 3:
                        liquidity_flag = trade[3]
            try:
                price_f = float(price_val)
                qty_f = float(qty_val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(price_f) or not math.isfinite(qty_f):
                continue
            side_txt = str(side_flag or "").upper()
            liquidity_txt = str(liquidity_flag or "").lower()
            is_buy = side_txt == "BUY"
            maker_flag = liquidity_txt == "maker"
            normalized.append((price_f, qty_f, is_buy, maker_flag))
        return normalized

    def _apply_trades_to_state(self, trades: List[Tuple[float, float, bool, bool]]) -> None:
        """
        Обновить env.state по списку сделок: (price, volume, is_buy, maker_is_agent).
        Buy → увеличивает units, уменьшает cash; Sell → уменьшает units, увеличивает cash.
        """
        st = getattr(self.env, "state", None)
        if st is None:
            return
        for price, vol, is_buy, _maker_is_agent in trades:
            if is_buy:
                st.units = float(st.units) + float(vol)
                st.cash = float(st.cash) - float(price) * float(vol)
            else:
                st.units = float(st.units) - float(vol)
                st.cash = float(st.cash) + float(price) * float(vol)

    def _process_ttl_queue(self, now_ts: int) -> None:
        """Отменить просроченные ордера."""
        if not self._ttl_queue:
            return
        keep: List[Tuple[int, int]] = []
        for order_id, exp_ts in self._ttl_queue:
            if now_ts >= exp_ts:
                # Нет информации о стороне/цене в очереди — отменяем «вслепую», игнорируя результат
                try:
                    self.lob.remove_order(True, 0, order_id)
                except Exception:
                    pass
                try:
                    self.lob.remove_order(False, 0, order_id)
                except Exception:
                    pass
            else:
                keep.append((order_id, exp_ts))
        self._ttl_queue = keep

    # ------------------------------ Публичный API ------------------------------

    def add_limit_order(self, *, is_buy_side: bool, price_ticks: int, volume: float,
                        timestamp: int, ttl_steps: int = 0, taker_is_agent: bool = True) -> Tuple[int, int]:
        """
        Разместить лимитный ордер напрямую в LOB.
        Возвращает (order_id, queue_position). При ttl_steps>0 — ордер будет отменён после истечения.
        """
        # pre-trade риск по ожидаемой позиции
        st = self._state_view()
        proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=float(volume) / max(1.0, st.max_position))
        evt = self.risk.on_action_proposed(self.env.state, proto)  # type: ignore[attr-defined]
        if evt.name != "NONE":
            return 0, 0
        price_ticks_q = int(price_ticks)
        volume_q = float(volume)
        symbol = str(getattr(self.env, "symbol", getattr(self.env, "base_symbol", ""))).upper()
        ref_price = getattr(self.env, "last_mtm_price", getattr(self.env, "last_mid", None))
        if self.quantizer is not None:
            try:
                price_abs = float(price_ticks_q) / PRICE_SCALE
                p_abs = self.quantizer.quantize_price(symbol, price_abs)
                q = self.quantizer.quantize_qty(symbol, volume_q)
                q_clamped = self.quantizer.clamp_notional(symbol, p_abs if p_abs > 0 else (ref_price or 0.0), q)
                p_ticks = int(round(p_abs * PRICE_SCALE))
                if p_ticks != price_ticks_q or abs(q - volume_q) > 1e-12 or abs(q_clamped - q) > 1e-12:
                    return 0, 0
                if self.enforce_ppbs and ref_price is not None:
                    side = "BUY" if is_buy_side else "SELL"
                    if not self.quantizer.check_percent_price_by_side(symbol, side, p_abs, ref_price):
                        return 0, 0
                price_ticks_q = p_ticks
                volume_q = q_clamped
            except Exception:
                return 0, 0

        if not self._check_rate_limit():
            return 0, 0

        order_id, qpos = self.lob.add_limit_order(
            bool(is_buy_side),
            int(price_ticks_q),
            float(volume_q),
            int(timestamp),
            bool(taker_is_agent),
        )
        if int(ttl_steps) > 0:
            ttl_set = False
            if hasattr(self.lob, "set_order_ttl"):
                try:
                    ttl_set = bool(self.lob.set_order_ttl(int(order_id), int(ttl_steps)))
                except Exception:
                    ttl_set = False
            if not ttl_set:
                self._ttl_queue.append((int(order_id), int(timestamp) + int(ttl_steps)))
        # учёт «ожидаемого» объёма
        if is_buy_side:
            self._pending_buy_volume += float(volume_q)
        else:
            self._pending_sell_volume += float(volume_q)
        return int(order_id), int(qpos)

    def remove_order(self, *, is_buy_side: bool, price_ticks: int, order_id: int) -> bool:
        """Отменить ордер по ID и цене (грубый контракт, для реального LOB достаточно ID)."""
        ok = False
        try:
            ok = bool(self.lob.remove_order(bool(is_buy_side), int(price_ticks), int(order_id)))
        finally:
            # убрать из TTL-очереди, если там есть
            self._ttl_queue = [(oid, ts) for (oid, ts) in self._ttl_queue if oid != int(order_id)]
        return ok

    def match_market_order(self, *, is_buy_side: bool, volume: float, timestamp: int,
                           taker_is_agent: bool = True) -> List[Tuple[float, float, bool, bool]]:
        """
        Исполнить маркет-заявку через LOB.
        Возвращает список сделок [(price, volume, is_buy, maker_is_agent)].
        """
        # pre-trade риск по ожидаемой позиции
        st = self._state_view()
        proto = ActionProto(action_type=ActionType.MARKET, volume_frac=float(volume) / max(1.0, st.max_position))
        evt = self.risk.on_action_proposed(self.env.state, proto)  # type: ignore[attr-defined]
        if evt.name != "NONE":
            return []

        # Попробуем использовать Cython LOB сигнатуру (с буферами), если она доступна
        trades: List[Tuple[float, float, bool, bool]] = []
        if self._check_rate_limit():
            try:
                max_len = 1024
                prices = np.empty(max_len, dtype=np.float64)
                vols = np.empty(max_len, dtype=np.float64)
                is_buy_arr = np.empty(max_len, dtype=np.int32)
                is_self_arr = np.empty(max_len, dtype=np.int32)
                ids = np.empty(max_len, dtype=np.int64)
                n, fee_total = self.lob.match_market_order(
                    bool(is_buy_side),
                    float(volume),
                    int(timestamp),
                    bool(taker_is_agent),
                    prices,
                    vols,
                    is_buy_arr,
                    is_self_arr,
                    ids,
                    int(max_len),
                )
                for i in range(int(n)):
                    trades.append(
                        (
                            float(prices[i]),
                            float(vols[i]),
                            bool(is_buy_arr[i]),
                            bool(is_self_arr[i]),
                        )
                    )
            except Exception:
                # Заглушечный путь: ничего не исполнилось
                trades = []

        # применяем сделки к состоянию и логируем
        if trades:
            self._apply_trades_to_state(trades)
            for (px, vol, is_buy, is_self) in trades:
                try:
                    # формируем ExecReport и логируем единообразно
                    _rid = str(getattr(event_bus, "_STATE").run_id if hasattr(event_bus, "_STATE") else "")
                    _sym = str(getattr(event_bus, "_STATE").default_symbol if hasattr(event_bus, "_STATE") else "UNKNOWN")
                    _er = ExecReport(
                        ts=int(timestamp),
                        run_id=_rid,
                        symbol=_sym,
                        side=Side.BUY if bool(is_buy) else Side.SELL,
                        order_type=OrderType.MARKET,
                        price=Decimal(str(float(px))),
                        quantity=Decimal(str(float(vol))),
                        fee=Decimal("0"),
                        fee_asset=None,
                        exec_status=ExecStatus.FILLED,
                        liquidity=Liquidity.UNKNOWN,
                        client_order_id=None,
                        order_id=None,
                        trade_id=None,
                        pnl=None,
                        meta={},
                    )
                    event_bus.log_trade(_er)
                except Exception:
                    pass

        # post-trade проверки
        mid_for_risk = trades[-1][0] if trades else float(
            getattr(self.env, "last_mtm_price", getattr(self.env, "last_mid", 0.0))
        )
        try:
            self.risk.on_post_trade(self.env.state, float(mid_for_risk))  # type: ignore[attr-defined]
        except Exception:
            pass

        return trades

    def step_action(self, proto: ActionProto, *, timestamp: int) -> dict:
        """
        Унифицированная точка для выполнения действия агента.
        Возвращает краткий отчёт dict (совместим с ExecReport.to_dict при наличии execution_sim).
        """
        # локальный буфер событий
        events: list[dict] = []
        pre_cancelled: list[int] = []

        now_ts = int(timestamp)
        try:
            self._process_ttl_queue(now_ts)
        except Exception:
            pass
        try:
            if not self._use_exec:
                decay_fn = getattr(self.lob, "decay_ttl_and_cancel", None)
                if callable(decay_fn):
                    pre_cancelled = [int(x) for x in decay_fn()]
        except Exception:
            pass

        # Сформировать Order из proto (или legacy dict)
        ctx = OrderContext(
            ts_ms=int(timestamp),
            symbol=str(getattr(self.env, "symbol", "UNKNOWN")),
            ref_price=float(
                getattr(self.env, "last_mtm_price", getattr(self.env, "last_mid", 0.0))
            )
            if hasattr(self.env, "last_mtm_price") or hasattr(self.env, "last_mid")
            else None,
            max_position_abs_base=float(getattr(getattr(self.env, "state", None), "max_position", 0.0) or getattr(self.env, "max_abs_position", 0.0) or 0.0),
            tick_size=None,  # квантование делается ниже по контуру
            price_offset_ticks=int(getattr(proto, "price_offset_ticks", 0)),
            tif=str(getattr(proto, "tif", "GTC")),
            client_tag=str(getattr(proto, "client_tag", "") or ""),
        )

        order_obj: Order | None = None
        try:
            # Если это ActionProto
            order_obj = actionproto_to_order(proto, ctx)
        except Exception:
            # Если пришёл legacy dict
            if isinstance(proto, dict):
                order_obj = legacy_decision_to_order(proto, ctx)

        # публикация факта подачи действия
        try:
            submitted_event = OrderEvent(
                etype=EventType.ORDER_SUBMITTED,
                ts=int(timestamp),
                order=(order_obj.to_dict() if hasattr(order_obj, "to_dict") else None),
                meta={"action": getattr(proto, "to_dict", lambda: {"type": int(getattr(proto, "action_type", 0))})()}
            ).to_dict()
            events.append(submitted_event)
        except Exception:
            pass

        # pre-trade
        evt = self.risk.on_action_proposed(self.env.state, proto)  # type: ignore[attr-defined]
        info: dict = {}
        if evt.name != "NONE":
            info["risk_event"] = evt.name
            return {"trades": [], "cancelled_ids": [], "new_order_ids": [], "fee_total": 0.0,
                    "new_order_pos": [], "info": info, "events": events}

        # если есть ExecutionSimulator — используем его
        if self._use_exec and self.exec is not None:
            if proto.action_type != ActionType.HOLD and not self._check_rate_limit():
                info["rate_limited"] = True
                return {"trades": [], "cancelled_ids": [], "new_order_ids": [], "fee_total": 0.0,
                        "new_order_pos": [], "info": info, "events": events}
            try:
                bid = getattr(self.env, "last_bid", None)
                ask = getattr(self.env, "last_ask", None)
                try:
                    self.exec.set_market_snapshot(bid=bid, ask=ask)  # type: ignore[union-attr]
                except Exception:
                    pass
                mid = None
                if bid is not None and ask is not None:
                    mid = (float(bid) + float(ask)) / 2.0
                else:
                    mid = getattr(
                        self.env, "last_mtm_price", getattr(self.env, "last_mid", None)
                    )
                try:
                    if mid is not None:
                        self.exec.set_ref_price(float(mid))  # type: ignore[union-attr]
                except Exception:
                    pass
                cli_id = self.exec.submit(proto)  # type: ignore[union-attr]
                # в простом варианте считаем, что latency=0 и сразу «поп» (если latency>0 — поп произойдёт на тик)
                report: ExecReport = self.exec.pop_ready()  # type: ignore  # ExecReport — это alias на SimStepReport
                try:
                    setattr(
                        self.env,
                        "last_bid",
                        float(getattr(report, "bid", getattr(self.env, "last_bid", 0.0))),
                    )
                    setattr(
                        self.env,
                        "last_ask",
                        float(getattr(report, "ask", getattr(self.env, "last_ask", 0.0))),
                    )
                    mtm = float(
                        getattr(
                            report,
                            "mtm_price",
                            getattr(
                                self.env,
                                "last_mtm_price",
                                getattr(self.env, "last_mid", 0.0),
                            ),
                        )
                    )
                    setattr(self.env, "last_mtm_price", mtm)
                    setattr(self.env, "last_mid", mtm)
                except Exception:
                    pass
                # применить и пост-проверки
                raw_trades = list(getattr(report, "trades", []))
                simple_trades = self._normalize_trades(raw_trades)
                self._apply_trades_to_state(simple_trades)
                mid_for_risk = float(
                    getattr(
                        report,
                        "mtm_price",
                        getattr(
                            self.env,
                            "last_mtm_price",
                            getattr(self.env, "last_mid", 0.0),
                        ),
                    )
                )
                self.risk.on_post_trade(self.env.state, mid_for_risk)  # type: ignore[attr-defined]
                d = report.to_dict()
                try:
                    exec_reports = sim_report_dict_to_core_exec_reports(
                        d,
                        symbol=str(getattr(self.env, "symbol", getattr(self.env, "base_symbol", "UNKNOWN"))),
                        client_order_id=None
                    )
                    d["core_exec_reports"] = [as_dict(er) for er in exec_reports]

                    # добавлено: публикация FillEvent и запись в unified CSV
                    try:
                        lvl = int(getattr(self, "event_level", 0))
                    except Exception:
                        lvl = 0
                    for _er in exec_reports:
                        if lvl >= 2:
                            try:
                                events.append(FillEvent(etype=EventType.EXEC_FILLED, ts=_er.ts, exec_report=_er).to_dict())
                            except Exception:
                                pass
                        # лог в unified-CSV
                        try:
                            run_id_val = getattr(event_bus, "_STATE").run_id if hasattr(event_bus, "_STATE") else ""
                            symbol_val = getattr(self.env, "symbol", getattr(self.env, "base_symbol", "UNKNOWN"))
                            event_bus.log_trade(_er)
                        except Exception:
                            pass
                except Exception:
                    d["core_exec_reports"] = []

                # возвращаем также события
                d["events"] = events
                d["info"] = info
                d["trades"] = simple_trades
                return d
            except Exception:
                # запасной путь — выполнить напрямую по типу действия
                pass

        # иначе — прямое исполнение через LOB
        trades: List[Tuple[float, float, bool, bool]] = []
        new_order_ids: List[int] = []
        new_order_pos: List[int] = []
        cancelled_ids: List[int] = list(pre_cancelled)
        fee_total: float = 0.0

        if proto.action_type == ActionType.HOLD:
            pass
        elif proto.action_type == ActionType.MARKET:
            trades = self.match_market_order(is_buy_side=(proto.volume_frac > 0.0),
                                             volume=abs(proto.volume_frac) * max(1.0, self._state_view().max_position),
                                             timestamp=int(timestamp), taker_is_agent=True)
        elif proto.action_type == ActionType.LIMIT:
            ttl_steps = int(getattr(proto, "ttl_steps", 0))
            vol = abs(proto.volume_frac) * max(1.0, self._state_view().max_position)
            price_ticks = int(getattr(proto, "price_offset_ticks", 0))
            abs_price = getattr(proto, "abs_price", None)
            if abs_price is not None:
                symbol = str(getattr(self.env, "symbol", getattr(self.env, "base_symbol", ""))).upper()
                if self.quantizer is not None:
                    p_abs = self.quantizer.quantize_price(symbol, float(abs_price))
                else:
                    p_abs = float(abs_price)
                price_ticks = int(round(p_abs * PRICE_SCALE))
            oid, qpos = self.add_limit_order(is_buy_side=(proto.volume_frac > 0.0),
                                             price_ticks=price_ticks, volume=vol,
                                             timestamp=int(timestamp), ttl_steps=ttl_steps, taker_is_agent=True)
            if oid:
                new_order_ids.append(int(oid))
                new_order_pos.append(int(qpos))

        # пост-проверки и отчёт
        mid_for_risk = trades[-1][0] if trades else float(
            getattr(self.env, "last_mtm_price", getattr(self.env, "last_mid", 0.0))
        )
        try:
            self.risk.on_post_trade(self.env.state, float(mid_for_risk))  # type: ignore[attr-defined]
        except Exception:
            pass

        for (px, vol, is_buy, is_self) in trades:
            try:
                # формируем ExecReport и логируем единообразно
                _rid = str(getattr(event_bus, "_STATE").run_id if hasattr(event_bus, "_STATE") else "")
                _sym = str(getattr(event_bus, "_STATE").default_symbol if hasattr(event_bus, "_STATE") else "UNKNOWN")
                _er = ExecReport(
                    ts=int(timestamp),
                    run_id=_rid,
                    symbol=_sym,
                    side=Side.BUY if bool(is_buy) else Side.SELL,
                    order_type=OrderType.MARKET,
                    price=Decimal(str(float(px))),
                    quantity=Decimal(str(float(vol))),
                    fee=Decimal("0"),
                    fee_asset=None,
                    exec_status=ExecStatus.FILLED,
                    liquidity=Liquidity.UNKNOWN,
                    client_order_id=None,
                    order_id=None,
                    trade_id=None,
                    pnl=None,
                    meta={},
                )
                event_bus.log_trade(_er)
            except Exception:
                pass

        return {
            "trades": trades,
            "cancelled_ids": cancelled_ids,
            "new_order_ids": new_order_ids,
            "fee_total": float(fee_total),
            "new_order_pos": new_order_pos,
            "info": info,
            "events": events,
        }

    @staticmethod
    def _coerce_finite(value: Any, default: float = 0.0) -> float:
        """Cast ``value`` to ``float`` returning ``default`` when non‑finite."""

        if value is None:
            return float(default)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(numeric):
            return float(default)
        return numeric

    @staticmethod
    def _get_safe_float(row: Any, col: str, default: float = 0.0) -> float:
        """Safely extract float value from row with fallback."""
        if row is None:
            return default
        try:
            val = row.get(col) if hasattr(row, "get") else getattr(row, col, None)
            if val is None:
                return default
            result = float(val)
            if not math.isfinite(result):
                return default
            return result
        except (TypeError, ValueError, KeyError, AttributeError):
            return default

    def _extract_market_data(self, row: Any, state: Any, mark_price: float, prev_price: float) -> Dict[str, float]:
        """Extract basic market data from row."""
        price = self._coerce_finite(mark_price, default=0.0)
        prev = self._coerce_finite(prev_price, default=price)

        # Volume normalization
        volume = self._get_safe_float(row, "volume", 1.0)
        quote_volume = self._get_safe_float(row, "quote_asset_volume", 1.0)

        log_volume_norm = 0.0
        if quote_volume > 0:
            log_volume_norm = float(np.tanh(np.log1p(quote_volume / 1e6)))

        rel_volume = 0.0
        if volume > 0:
            rel_volume = float(np.tanh(np.log1p(volume / 100.0)))

        return {
            "price": price,
            "prev_price": prev,
            "log_volume_norm": log_volume_norm,
            "rel_volume": rel_volume,
        }

    def _extract_technical_indicators(self, row: Any, sim: Any, row_idx: int) -> Dict[str, float]:
        """Extract technical indicators from row or simulator."""
        # Try to get from row first (from prepare_and_run.py features)
        ma5 = self._get_safe_float(row, "sma_5", float('nan'))
        # NOTE: For 4h timeframe using sma_21 (21 bars = 84h ≈ weekly trend)
        # config_4h_timeframe.py specifies SMA_LOOKBACKS = [5, 21, 50] bars
        ma20 = self._get_safe_float(row, "sma_21", float('nan'))
        rsi14 = self._get_safe_float(row, "rsi", 50.0)

        # For MACD and other indicators, try from simulator if available
        macd = 0.0
        macd_signal = 0.0
        momentum = 0.0
        atr = 0.0
        cci = 0.0
        obv = 0.0
        bb_lower = float('nan')
        bb_upper = float('nan')

        # Try to get from MarketSimulator if available
        if sim is not None and hasattr(sim, "get_macd"):
            try:
                if hasattr(sim, "get_macd"):
                    macd = float(sim.get_macd(row_idx))
                if hasattr(sim, "get_macd_signal"):
                    macd_signal = float(sim.get_macd_signal(row_idx))
                if hasattr(sim, "get_momentum"):
                    momentum = float(sim.get_momentum(row_idx))
                if hasattr(sim, "get_atr"):
                    atr = float(sim.get_atr(row_idx))
                if hasattr(sim, "get_cci"):
                    cci = float(sim.get_cci(row_idx))
                if hasattr(sim, "get_obv"):
                    obv = float(sim.get_obv(row_idx))
                if hasattr(sim, "get_bb_lower"):
                    bb_lower = float(sim.get_bb_lower(row_idx))
                if hasattr(sim, "get_bb_upper"):
                    bb_upper = float(sim.get_bb_upper(row_idx))
            except Exception:
                pass

        return {
            "ma5": ma5,
            "ma20": ma20,
            "rsi14": rsi14,
            "macd": macd,
            "macd_signal": macd_signal,
            "momentum": momentum,
            "atr": atr,
            "cci": cci,
            "obv": obv,
            "bb_lower": bb_lower,
            "bb_upper": bb_upper,
        }

    def _extract_norm_cols(self, row: Any) -> np.ndarray:
        """Extract normalized columns for external features (cvd, garch, yang_zhang, etc.).

        Adapted for 4h timeframe:
        - GARCH windows: 200h/14d/30d (50/84/180 bars) instead of 500m/12h/24h
        - Returns: 4h/12h/24h (1/3/6 bars) instead of 5m/15m/60m
        - SMA: sma_12000 (50 bars = 12000 минут = 200h) instead of sma_60 (60 minutes)
        - Taker Buy Ratio Momentum: 4h (1 bar) instead of 1h
        """
        norm_cols = np.zeros(21, dtype=np.float32)

        # Map technical indicators from prepare_and_run.py to norm_cols
        # Original 8 features (adapted for 4h)
        norm_cols[0] = self._get_safe_float(row, "cvd_24h", 0.0)
        norm_cols[1] = self._get_safe_float(row, "cvd_7d", 0.0)  # 10080 минут = 7 дней
        norm_cols[2] = self._get_safe_float(row, "yang_zhang_48h", 0.0)  # 12 bars = 48h
        norm_cols[3] = self._get_safe_float(row, "yang_zhang_7d", 0.0)  # 10080 минут = 7 дней
        norm_cols[4] = self._get_safe_float(row, "garch_200h", 0.0)  # 50 bars = 12000 min = 200h (минимум для GARCH на 4h)
        norm_cols[5] = self._get_safe_float(row, "garch_14d", 0.0)  # 84 bars = 14 days
        norm_cols[6] = self._get_safe_float(row, "ret_12h", 0.0)  # 3 bars
        norm_cols[7] = self._get_safe_float(row, "ret_24h", 0.0)  # 6 bars

        # Additional 8 features for complete coverage (43 -> 51) - adapted for 4h
        norm_cols[8] = self._get_safe_float(row, "ret_4h", 0.0)  # 1 bar
        norm_cols[9] = self._get_safe_float(row, "sma_12000", 0.0)  # 50 bars = 12000 минут = 200h
        norm_cols[10] = self._get_safe_float(row, "yang_zhang_30d", 0.0)  # 43200 минут = 30 дней
        norm_cols[11] = self._get_safe_float(row, "parkinson_48h", 0.0)  # 12 bars = 48h
        norm_cols[12] = self._get_safe_float(row, "parkinson_7d", 0.0)  # 10080 минут = 7 дней
        norm_cols[13] = self._get_safe_float(row, "garch_30d", 0.0)  # 180 bars = 30 days
        norm_cols[14] = self._get_safe_float(row, "taker_buy_ratio", 0.0)
        norm_cols[15] = self._get_safe_float(row, "taker_buy_ratio_sma_24h", 0.0)  # 6 bars

        # Additional 5 features for complete taker_buy_ratio coverage (51 -> 56) - adapted for 4h
        norm_cols[16] = self._get_safe_float(row, "taker_buy_ratio_sma_8h", 0.0)  # 2 bars
        norm_cols[17] = self._get_safe_float(row, "taker_buy_ratio_sma_16h", 0.0)  # 4 bars
        norm_cols[18] = self._get_safe_float(row, "taker_buy_ratio_momentum_4h", 0.0)  # 1 bar
        norm_cols[19] = self._get_safe_float(row, "taker_buy_ratio_momentum_8h", 0.0)  # 2 bars
        norm_cols[20] = self._get_safe_float(row, "taker_buy_ratio_momentum_12h", 0.0)  # 3 bars

        # NOTE: Normalization (tanh, clip) is applied in obs_builder.pyx when available.
        # In legacy fallback mode (when obs_builder is not available), normalization
        # is applied in the fallback path to ensure consistent behavior.
        return norm_cols

    def _build_observation(self, *, row: Any | None, state: Any, mark_price: float) -> np.ndarray:
        """Build observation vector using obs_builder infrastructure with technical indicators."""
        obs_shape = getattr(getattr(self.env, "observation_space", None), "shape", None)
        if not obs_shape:
            return np.zeros(0, dtype=np.float32)

        # If obs_builder is not available, fall back to legacy implementation
        if not _HAVE_OBS_BUILDER:
            return self._build_observation_legacy(row=row, state=state, mark_price=mark_price)

        # Initialize observation array
        obs = np.zeros(obs_shape, dtype=np.float32)

        # Get environment and dataframe
        env = self.env
        df = getattr(env, "df", None)

        # Determine row index
        row_idx: int | None = getattr(self, "_context_row_idx", None)
        if row_idx is None and row is not None:
            try:
                row_idx = int(getattr(row, "name"))
            except Exception:
                row_idx = None
        if row_idx is None:
            try:
                step_idx = getattr(state, "step_idx", None)
                if step_idx is not None:
                    row_idx = int(step_idx)
            except Exception:
                row_idx = 0

        if row_idx is not None:
            if row_idx < 0:
                row_idx = 0
            if df is not None and row_idx >= len(df):
                row_idx = len(df) - 1

        # Calculate previous price and current price
        resolve_reward_price = getattr(env, "_resolve_reward_price", None)
        prev_price_val = self._coerce_finite(getattr(env, "_last_reward_price", 0.0), default=0.0)
        curr_price = mark_price

        if callable(resolve_reward_price):
            try:
                curr_price = float(resolve_reward_price(row_idx, row))
            except Exception:
                pass

        if not math.isfinite(curr_price) or curr_price <= 0.0:
            curr_price = mark_price if mark_price > 0.0 else 1.0

        if (prev_price_val is None or prev_price_val <= 0.0) and callable(resolve_reward_price):
            prev_idx = max(row_idx - 1, 0) if row_idx is not None else 0
            prev_row = None
            if df is not None and prev_idx < len(df):
                try:
                    prev_row = df.iloc[prev_idx]
                except Exception:
                    pass
            try:
                prev_price_candidate = float(resolve_reward_price(prev_idx, prev_row))
                if math.isfinite(prev_price_candidate) and prev_price_candidate > 0.0:
                    prev_price_val = float(prev_price_candidate)
            except Exception:
                pass

        if prev_price_val <= 0.0:
            prev_price_val = curr_price

        # Extract market data
        market_data = self._extract_market_data(row, state, curr_price, prev_price_val)

        # Extract technical indicators
        sim = getattr(env, "sim", None)
        indicators = self._extract_technical_indicators(row, sim, row_idx or 0)

        # Extract normalized columns (cvd, garch, yang_zhang, etc.)
        norm_cols_values = self._extract_norm_cols(row)

        # Get state values
        units = self._coerce_finite(getattr(state, "units", 0.0), default=0.0)
        cash = self._coerce_finite(getattr(state, "cash", 0.0), default=0.0)

        # Get microstructure metrics from state if available
        last_vol_imbalance = self._coerce_finite(getattr(state, "last_vol_imbalance", 0.0), default=0.0)
        last_trade_intensity = self._coerce_finite(getattr(state, "last_trade_intensity", 0.0), default=0.0)
        last_realized_spread = self._coerce_finite(getattr(state, "last_realized_spread", 0.0), default=0.0)
        last_agent_fill_ratio = self._coerce_finite(getattr(state, "last_agent_fill_ratio", 0.0), default=0.0)

        # Fear & Greed
        fear_greed_value = self._get_safe_float(row, "fear_greed_value", 50.0)
        has_fear_greed = abs(fear_greed_value - 50.0) > 0.1  # Check if we have real FG data

        # Event metadata
        is_high_importance = self._get_safe_float(row, "is_high_importance", 0.0)
        time_since_event = self._get_safe_float(row, "time_since_event", 0.0)

        # Risk-off flag (simplified: based on fear & greed)
        risk_off_flag = fear_greed_value < 25.0

        # Token metadata (single token by default)
        token_id = getattr(state, "token_index", 0)
        max_num_tokens = 1
        num_tokens = 1

        # Call obs_builder to construct observation vector
        try:
            build_observation_vector(
                float(market_data["price"]),
                float(market_data["prev_price"]),
                float(market_data["log_volume_norm"]),
                float(market_data["rel_volume"]),
                float(indicators["ma5"]),
                float(indicators["ma20"]),
                float(indicators["rsi14"]),
                float(indicators["macd"]),
                float(indicators["macd_signal"]),
                float(indicators["momentum"]),
                float(indicators["atr"]),
                float(indicators["cci"]),
                float(indicators["obv"]),
                float(indicators["bb_lower"]),
                float(indicators["bb_upper"]),
                float(is_high_importance),
                float(time_since_event),
                float(fear_greed_value),
                bool(has_fear_greed),
                bool(risk_off_flag),
                float(cash),
                float(units),
                float(last_vol_imbalance),
                float(last_trade_intensity),
                float(last_realized_spread),
                float(last_agent_fill_ratio),
                int(token_id),
                int(max_num_tokens),
                int(num_tokens),
                norm_cols_values,
                obs,
            )
        except Exception as e:
            # If obs_builder fails, fall back to legacy
            import logging
            logging.getLogger(__name__).warning(f"obs_builder failed: {e}, falling back to legacy")
            return self._build_observation_legacy(row=row, state=state, mark_price=mark_price)

        return obs

    def _build_observation_legacy(self, *, row: Any | None, state: Any, mark_price: float) -> np.ndarray:
        """Legacy observation builder (fallback when obs_builder is not available)."""
        obs_shape = getattr(getattr(self.env, "observation_space", None), "shape", None)
        if not obs_shape:
            return np.zeros(0, dtype=np.float32)
        obs = np.zeros(obs_shape, dtype=np.float32)
        # Try to populate front slots with common market columns when available
        if row is not None:
            try:
                columns = getattr(row, "index", [])
            except Exception:
                columns = []
            col_order = [
                "open",
                "high",
                "low",
                "close",
                "price",
                "bid",
                "ask",
                "quote_asset_volume",
            ]
            pos = 0
            tail_slots = 4
            tail_reserve = tail_slots if obs.shape[0] >= tail_slots else min(obs.shape[0], tail_slots)
            for name in col_order:
                if pos >= max(0, obs.shape[0] - tail_reserve):
                    break
                try:
                    if columns is not None and name not in columns:
                        continue
                except Exception:
                    # fallback to getattr
                    pass
                try:
                    val = row.get(name) if hasattr(row, "get") else getattr(row, name, None)
                except Exception:
                    val = None
                if val is None:
                    continue
                coerced = self._coerce_finite(val, default=0.0)
                obs[pos] = coerced
                pos += 1
        # Always include mark price, units and cash in the tail slots if possible
        mark_value = self._coerce_finite(mark_price, default=0.0)
        if obs.size:
            obs[0] = mark_value
        units = self._coerce_finite(getattr(state, "units", 0.0), default=0.0)
        cash = self._coerce_finite(getattr(state, "cash", 0.0), default=0.0)
        signal_source = getattr(
            self,
            "_last_signal_position",
            getattr(self.env, "_last_signal_position", 0.0),
        )
        signal_pos = self._coerce_finite(signal_source, default=0.0)
        log_ret_prev = 0.0
        env = self.env
        df = getattr(env, "df", None)
        row_idx: int | None = getattr(self, "_context_row_idx", None)
        if row_idx is None and row is not None:
            try:
                row_idx = int(getattr(row, "name"))
            except Exception:
                row_idx = None
        if row_idx is None:
            try:
                step_idx = getattr(state, "step_idx", None)
                if step_idx is not None:
                    row_idx = int(step_idx)
            except Exception:
                row_idx = None
        if row_idx is not None:
            if row_idx < 0:
                row_idx = 0
            if df is not None and row_idx >= len(df):
                row_idx = len(df) - 1
            resolve_reward_price = getattr(env, "_resolve_reward_price", None)
            prev_price = self._coerce_finite(getattr(env, "_last_reward_price", 0.0), default=0.0)
            curr_price: float | None = None
            if callable(resolve_reward_price):
                try:
                    curr_price = float(resolve_reward_price(row_idx, row))
                except Exception:
                    curr_price = None
            if curr_price is None or not math.isfinite(curr_price) or curr_price <= 0.0:
                curr_price = mark_value if mark_value > 0.0 else None
            if (prev_price is None or prev_price <= 0.0) and callable(resolve_reward_price):
                prev_idx = max(row_idx - 1, 0)
                prev_row = None
                if df is not None and prev_idx < len(df):
                    try:
                        prev_row = df.iloc[prev_idx]
                    except Exception:
                        prev_row = None
                try:
                    prev_price_candidate = float(resolve_reward_price(prev_idx, prev_row))
                except Exception:
                    prev_price_candidate = None
                if prev_price_candidate is not None and math.isfinite(prev_price_candidate) and prev_price_candidate > 0.0:
                    prev_price = float(prev_price_candidate)
            if curr_price is not None and math.isfinite(curr_price) and curr_price > 0.0 and prev_price > 0.0:
                log_ret_prev = math.log(curr_price / prev_price)
                mark_value = float(curr_price)
                if obs.size:
                    obs[0] = float(mark_value)
        self._latest_log_ret_prev = float(log_ret_prev)

        tail_values = (units, cash, signal_pos, float(log_ret_prev))
        tail_count = min(len(tail_values), obs.size)
        for offset in range(1, tail_count + 1):
            obs[-offset] = float(tail_values[-offset])
        return obs

    def step(self, proto: ActionProto):
        env = self.env
        state = getattr(env, "state", None)
        if state is None:
            raise RuntimeError("Mediator requires environment state")

        current_idx = int(getattr(state, "step_idx", 0) or 0)
        df = getattr(env, "df", None)
        row_idx = self._context_row_idx if self._context_row_idx is not None else current_idx
        row = self._context_row
        if row is None and df is not None:
            try:
                if 0 <= row_idx < len(df):
                    row = df.iloc[row_idx]
            except Exception:
                row = None

        timestamp = self._context_timestamp
        if timestamp is None:
            if row is not None and hasattr(env, "_resolve_snapshot_timestamp"):
                try:
                    timestamp = int(env._resolve_snapshot_timestamp(row))
                except Exception:
                    timestamp = None
        if timestamp is None:
            timestamp = int(now_ms())

        report = self.step_action(proto, timestamp=timestamp)

        trades = list(report.get("trades", []))
        cancelled_ids = [int(x) for x in report.get("cancelled_ids", [])]
        new_order_ids = [int(x) for x in report.get("new_order_ids", [])]
        fee_total = float(report.get("fee_total", 0.0) or 0.0)
        events = list(report.get("events", []))

        executed_notional = 0.0
        for price, volume, _is_buy, _maker_is_agent in trades:
            try:
                executed_notional += abs(float(price) * float(volume))
            except Exception:
                continue

        # Update agent order tracker if env.state holds it
        agent_orders = getattr(state, "agent_orders", None)
        if agent_orders is not None:
            try:
                for cid in cancelled_ids:
                    agent_orders.discard(int(cid))
                for oid in new_order_ids:
                    agent_orders.add(int(oid))
            except Exception:
                pass

        # Reduce pending expected volume counters
        for _price, vol, is_buy, _maker_is_agent in trades:
            try:
                vol_f = float(vol)
            except Exception:
                continue
            if is_buy:
                self._pending_buy_volume = max(0.0, float(self._pending_buy_volume) - vol_f)
            else:
                self._pending_sell_volume = max(0.0, float(self._pending_sell_volume) - vol_f)

        mark_price = getattr(env, "last_mtm_price", None)
        if mark_price is None:
            mark_price = getattr(env, "last_mid", None)
        if mark_price is None and row is not None:
            for key in ("close", "price", "open"):
                if hasattr(row, "get"):
                    candidate = row.get(key)
                else:
                    candidate = getattr(row, key, None)
                if candidate is not None:
                    try:
                        mark_price = float(candidate)
                        break
                    except Exception:
                        continue
        mark_price = self._coerce_finite(mark_price, default=0.0)

        cash = self._coerce_finite(getattr(state, "cash", 0.0), default=0.0)
        units = self._coerce_finite(getattr(state, "units", 0.0), default=0.0)
        net_worth = cash + units * mark_price
        try:
            state.net_worth = float(net_worth)
        except Exception:
            pass
        peak_value = float(getattr(state, "peak_value", net_worth) or net_worth)
        if net_worth > peak_value:
            try:
                state.peak_value = float(net_worth)
            except Exception:
                pass

        max_steps = int(getattr(env, "_max_steps", 0) or 0)
        next_idx = current_idx + 1
        truncated = False
        if max_steps > 0 and next_idx >= max_steps:
            truncated = True
            next_idx = max_steps
        try:
            state.step_idx = int(next_idx)
        except Exception:
            pass

        bankruptcy_th = float(getattr(env, "bankruptcy_cash_th", -1e12) or -1e12)
        is_bankrupt = bool(getattr(state, "is_bankrupt", False))
        if not is_bankrupt and cash <= bankruptcy_th:
            is_bankrupt = True
            try:
                state.is_bankrupt = True
            except Exception:
                pass

        info = dict(report.get("info", {}))
        info.setdefault("executed_notional", executed_notional)
        info.setdefault("turnover", executed_notional)
        info.setdefault("fee_total", fee_total)
        info.setdefault("mark_price", mark_price)
        info.setdefault("cash", cash)
        info.setdefault("units", units)
        info.setdefault("net_worth", net_worth)
        info.setdefault("step_idx", current_idx)
        info.setdefault(
            "signal_pos",
            self._coerce_finite(
                getattr(self, "_last_signal_position", getattr(env, "_last_signal_position", 0.0)),
                default=0.0,
            ),
        )
        info["trades"] = trades
        info["cancelled_ids"] = cancelled_ids
        info["new_order_ids"] = new_order_ids
        info["events"] = events

        obs = self._build_observation(row=row, state=state, mark_price=mark_price)

        info.setdefault(
            "log_ret_prev",
            self._coerce_finite(getattr(self, "_latest_log_ret_prev", 0.0), default=0.0),
        )

        terminated = is_bankrupt
        if info.get("risk_event") in {"BANKRUPT", "STOP_TRADE"}:
            terminated = True

        self._context_row = None
        self._context_row_idx = None
        self._context_timestamp = None

        reward = 0.0
        return obs, float(reward), bool(terminated), bool(truncated), info

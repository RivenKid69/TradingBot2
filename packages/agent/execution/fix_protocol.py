# -*- coding: utf-8 -*-
"""
packages/agent/execution/fix_protocol.py
========================================

Минимальный, самодостаточный **FIX 4.4** энкодер/декодер (P2): институциональная
интеграция без внешних зависимостей (simplefix/quickfix недоступны). Корректные
**BodyLength (9)** и **CheckSum (10)** — обязательны для совместимости с брокером.

Поддержано: NewOrderSingle (35=D), OrderCancelRequest (35=F), ExecutionReport (35=8),
парсинг + верификация checksum. FIX-сессия (логон/seqnum) — каркас (Agent-зона; реальный
транспорт/TLS остаётся в Agent, CCEA). Стандартные теги в ``Tag``.

Замечание: это корректный encode/decode уровня сообщений; полноценный FIX-движок (rec/resend,
gap-fill, persistence) — поверх этого. Слой Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

SOH = "\x01"
BEGIN_STRING = "FIX.4.4"


class Tag:
    BeginString = "8"
    BodyLength = "9"
    MsgType = "35"
    SenderCompID = "49"
    TargetCompID = "56"
    MsgSeqNum = "34"
    SendingTime = "52"
    CheckSum = "10"
    ClOrdID = "11"
    OrigClOrdID = "41"
    OrderID = "37"
    ExecID = "17"
    Symbol = "55"
    Side = "54"
    OrderQty = "38"
    OrdType = "40"
    Price = "44"
    TimeInForce = "59"
    TransactTime = "60"
    OrdStatus = "39"
    ExecType = "150"
    CumQty = "14"
    AvgPx = "6"
    LeavesQty = "151"


class Side(str, Enum):
    BUY = "1"
    SELL = "2"


class OrdType(str, Enum):
    MARKET = "1"
    LIMIT = "2"


class MsgType(str, Enum):
    NEW_ORDER_SINGLE = "D"
    ORDER_CANCEL_REQUEST = "F"
    ORDER_CANCEL_REPLACE_REQUEST = "G"  # amend qty/price of a working order
    EXECUTION_REPORT = "8"
    LOGON = "A"
    HEARTBEAT = "0"


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------
def _checksum(s: str) -> str:
    return f"{sum(s.encode('latin-1')) % 256:03d}"


def encode_message(
    msg_type: str, fields: List[Tuple[str, Any]], *, begin: str = BEGIN_STRING
) -> str:
    """Собрать FIX-сообщение с корректными BodyLength и CheckSum.

    ``fields`` — упорядоченный список (tag, value), БЕЗ 8/9/35/10 (добавляются здесь).
    """
    body = SOH.join([f"{Tag.MsgType}={msg_type}"] + [f"{t}={v}" for t, v in fields]) + SOH
    header = f"{Tag.BeginString}={begin}{SOH}{Tag.BodyLength}={len(body)}{SOH}"
    pre = header + body
    return pre + f"{Tag.CheckSum}={_checksum(pre)}{SOH}"


def parse_message(raw: Any) -> Dict[str, str]:
    """Разобрать FIX-сообщение в словарь tag->value (последнее значение тега выигрывает)."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("latin-1")
    out: Dict[str, str] = {}
    for part in raw.split(SOH):
        if not part:
            continue
        tag, sep, val = part.partition("=")
        if sep:
            out[tag] = val
    return out


def verify_checksum(raw: Any) -> bool:
    """Проверить CheckSum (10) сообщения."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("latin-1")
    idx = raw.rfind(f"{SOH}{Tag.CheckSum}=")
    if idx < 0:
        # 10= может быть в начале только теоретически; ищем без ведущего SOH
        idx = raw.find(f"{Tag.CheckSum}=")
        if idx < 0:
            return False
        pre = raw[:idx]
        got = raw[idx:].split(SOH)[0].split("=")[1]
    else:
        pre = raw[: idx + 1]  # включая SOH перед 10=
        got = raw[idx + 1 :].split(SOH)[0].split("=")[1]
    return _checksum(pre) == got


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------
@dataclass
class FixSession:
    sender: str = "RIVEN"
    target: str = "BROKER"
    seq: int = 1

    def next_seq(self) -> int:
        s = self.seq
        self.seq += 1
        return s

    def _session_fields(self, sending_time: str) -> List[Tuple[str, Any]]:
        return [
            (Tag.SenderCompID, self.sender),
            (Tag.TargetCompID, self.target),
            (Tag.MsgSeqNum, self.next_seq()),
            (Tag.SendingTime, sending_time),
        ]


def new_order_single(
    *,
    cl_ord_id: str,
    symbol: str,
    side: Side,
    qty: float,
    ord_type: OrdType = OrdType.MARKET,
    price: Optional[float] = None,
    tif: str = "0",
    transact_time: str = "20200101-00:00:00.000",
    session: Optional[FixSession] = None,
) -> str:
    """NewOrderSingle (35=D)."""
    fields: List[Tuple[str, Any]] = []
    if session is not None:
        fields += session._session_fields(transact_time)
    fields += [
        (Tag.ClOrdID, cl_ord_id),
        (Tag.Symbol, symbol),
        (Tag.Side, side.value if isinstance(side, Side) else side),
        (Tag.OrderQty, _num(qty)),
        (Tag.OrdType, ord_type.value if isinstance(ord_type, OrdType) else ord_type),
    ]
    if price is not None:
        fields.append((Tag.Price, _num(price)))
    fields += [(Tag.TimeInForce, tif), (Tag.TransactTime, transact_time)]
    return encode_message(MsgType.NEW_ORDER_SINGLE.value, fields)


def order_cancel_request(
    *,
    orig_cl_ord_id: str,
    cl_ord_id: str,
    symbol: str,
    side: Side,
    transact_time: str = "20200101-00:00:00.000",
    session: Optional[FixSession] = None,
) -> str:
    fields: List[Tuple[str, Any]] = []
    if session is not None:
        fields += session._session_fields(transact_time)
    fields += [
        (Tag.OrigClOrdID, orig_cl_ord_id),
        (Tag.ClOrdID, cl_ord_id),
        (Tag.Symbol, symbol),
        (Tag.Side, side.value if isinstance(side, Side) else side),
        (Tag.TransactTime, transact_time),
    ]
    return encode_message(MsgType.ORDER_CANCEL_REQUEST.value, fields)


def order_cancel_replace_request(
    *,
    orig_cl_ord_id: str,
    cl_ord_id: str,
    symbol: str,
    side: Side,
    qty: float,
    ord_type: OrdType = OrdType.LIMIT,
    price: Optional[float] = None,
    tif: str = "0",
    transact_time: str = "20200101-00:00:00.000",
    session: Optional[FixSession] = None,
) -> str:
    """OrderCancelReplaceRequest (35=G) — amend a working order's qty and/or price.

    Carries OrigClOrdID (the order being replaced) + a new ClOrdID, the new
    OrderQty/Price/OrdType. This is the standard FIX 4.4 order-amendment message
    (was missing — only cancel 35=F existed)."""
    fields: List[Tuple[str, Any]] = []
    if session is not None:
        fields += session._session_fields(transact_time)
    fields += [
        (Tag.OrigClOrdID, orig_cl_ord_id),
        (Tag.ClOrdID, cl_ord_id),
        (Tag.Symbol, symbol),
        (Tag.Side, side.value if isinstance(side, Side) else side),
        (Tag.OrderQty, _num(qty)),
        (Tag.OrdType, ord_type.value if isinstance(ord_type, OrdType) else ord_type),
    ]
    if price is not None:
        fields.append((Tag.Price, _num(price)))
    fields += [(Tag.TimeInForce, tif), (Tag.TransactTime, transact_time)]
    return encode_message(MsgType.ORDER_CANCEL_REPLACE_REQUEST.value, fields)


def execution_report(
    *,
    order_id: str,
    cl_ord_id: str,
    exec_id: str,
    symbol: str,
    side: Side,
    ord_status: str,
    exec_type: str,
    cum_qty: float,
    avg_px: float,
    leaves_qty: float = 0.0,
) -> str:
    """ExecutionReport (35=8) — обычно от брокера; тут для симуляции/тестов."""
    fields = [
        (Tag.OrderID, order_id),
        (Tag.ClOrdID, cl_ord_id),
        (Tag.ExecID, exec_id),
        (Tag.Symbol, symbol),
        (Tag.Side, side.value if isinstance(side, Side) else side),
        (Tag.OrdStatus, ord_status),
        (Tag.ExecType, exec_type),
        (Tag.CumQty, _num(cum_qty)),
        (Tag.AvgPx, _num(avg_px)),
        (Tag.LeavesQty, _num(leaves_qty)),
    ]
    return encode_message(MsgType.EXECUTION_REPORT.value, fields)


def _num(x: float) -> str:
    f = float(x)
    return str(int(f)) if f == int(f) else repr(f)


__all__ = [
    "SOH",
    "BEGIN_STRING",
    "Tag",
    "Side",
    "OrdType",
    "MsgType",
    "FixSession",
    "encode_message",
    "parse_message",
    "verify_checksum",
    "new_order_single",
    "order_cancel_request",
    "order_cancel_replace_request",
    "execution_report",
]

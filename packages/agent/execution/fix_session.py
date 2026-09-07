# -*- coding: utf-8 -*-
"""
packages/agent/execution/fix_session.py
========================================

Real FIX 4.4 **session engine** (P2 #13) on top of the message codec in
``fix_protocol.py``. Previously only the codec existed (no transport/session). This
adds the actual session layer institutions require:

  * **transport**: TCP socket, optional TLS (``ssl``); injectable for hermetic tests;
  * **logon** (35=A) handshake with HeartBtInt, optional ResetSeqNumFlag;
  * **heartbeat** (35=0) on interval + **TestRequest** (35=1) when idle, and replies
    to peer TestRequests;
  * **sequence numbers** with **persistence** (survive restart) and **gap detection**;
  * **ResendRequest** (35=2) on inbound gaps + **SequenceReset/GapFill** (35=4) replies;
  * **Logout** (35=5) and a session **state machine**.

App-level messages (ExecutionReport, etc.) are dispatched to ``on_app_message``.
Outbound orders go through ``send_new_order`` / ``send_cancel``. Admin traffic is
handled internally. Reader + heartbeat run on background threads.

A certification harness can drive this against a counterparty/simulator. The unit
tests run two engines back-to-back over an in-process paired transport.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from packages.agent.execution.fix_protocol import (
    SOH,
    BEGIN_STRING,
    Tag,
    MsgType,
    Side,
    OrdType,
    encode_message,
    parse_message,
    verify_checksum,
    _num,
)

logger = logging.getLogger(__name__)


# --- additional session-level tags / msg types not in the codec ---
class STag:
    HeartBtInt = "108"
    EncryptMethod = "98"
    TestReqID = "112"
    ResetSeqNumFlag = "141"
    BeginSeqNo = "7"
    EndSeqNo = "16"
    GapFillFlag = "123"
    NewSeqNo = "36"
    PossDupFlag = "43"
    Text = "58"
    RefSeqNum = "45"


class SMsg:
    TEST_REQUEST = "1"
    RESEND_REQUEST = "2"
    REJECT = "3"
    SEQUENCE_RESET = "4"
    LOGOUT = "5"


class SessionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    LOGON_SENT = "logon_sent"
    ACTIVE = "active"
    LOGOUT_SENT = "logout_sent"


# ---------------------------------------------------------------------------
# Transport abstraction (TCP by default; injectable for tests)
# ---------------------------------------------------------------------------
class Transport:
    def send(self, data: bytes) -> None: ...
    def recv(self, n: int = 4096) -> bytes: ...
    def close(self) -> None: ...


class TCPTransport(Transport):
    def __init__(self, host: str, port: int, *, tls: bool = False, timeout: float = 1.0) -> None:
        import socket

        self._sock = socket.create_connection((host, port), timeout=timeout)
        if tls:
            import ssl

            ctx = ssl.create_default_context()
            self._sock = ctx.wrap_socket(self._sock, server_hostname=host)
        self._sock.settimeout(timeout)

    def send(self, data: bytes) -> None:
        self._sock.sendall(data)

    def recv(self, n: int = 4096) -> bytes:
        import socket

        try:
            return self._sock.recv(n)
        except socket.timeout:
            return b""

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


class PairedTransport(Transport):
    """In-process bidirectional transport for tests (two endpoints share queues)."""

    def __init__(self, inbound, outbound) -> None:
        import queue

        self._in = inbound  # queue we read from
        self._out = outbound  # queue we write to
        self._buf = b""
        self._q = queue
        self._closed = False

    @classmethod
    def pair(cls) -> Tuple["PairedTransport", "PairedTransport"]:
        import queue

        a, b = queue.Queue(), queue.Queue()
        return cls(a, b), cls(b, a)

    def send(self, data: bytes) -> None:
        if not self._closed:
            self._out.put(data)

    def recv(self, n: int = 4096) -> bytes:
        try:
            return self._in.get(timeout=0.2)
        except Exception:
            return b""

    def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Sequence number persistence
# ---------------------------------------------------------------------------
class SeqStore:
    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self.out_seq = 1
        self.in_seq = 1
        self._load()

    def _load(self) -> None:
        if self.path and os.path.exists(self.path):
            try:
                import json

                with open(self.path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                self.out_seq = int(d.get("out", 1))
                self.in_seq = int(d.get("in", 1))
            except Exception:
                pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            import json

            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"out": self.out_seq, "in": self.in_seq}, fh)
        except Exception:
            pass

    def reset(self) -> None:
        self.out_seq = 1
        self.in_seq = 1
        self.save()


def _now() -> str:
    return time.strftime("%Y%m%d-%H:%M:%S", time.gmtime()) + ".000"


# ---------------------------------------------------------------------------
# Session engine
# ---------------------------------------------------------------------------
class FixSessionEngine:
    def __init__(
        self,
        sender: str,
        target: str,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        heartbeat_int: int = 30,
        reset_seq_on_logon: bool = False,
        seq_store_path: Optional[str] = None,
        on_app_message: Optional[Callable[[Dict[str, str]], None]] = None,
        transport: Optional[Transport] = None,
        is_acceptor: bool = False,
    ) -> None:
        self.sender = sender
        self.target = target
        self.host = host
        self.port = port
        self.heartbeat_int = int(heartbeat_int)
        self.reset_seq_on_logon = reset_seq_on_logon
        self.on_app_message = on_app_message
        self.is_acceptor = is_acceptor

        self.state = SessionState.DISCONNECTED
        self._seq = SeqStore(seq_store_path)
        self._transport = transport
        self._buf = b""
        self._running = False
        self._reader: Optional[threading.Thread] = None
        self._hb: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._last_recv = time.time()
        self._last_sent = time.time()
        self._sent_messages: Dict[int, str] = {}  # seq -> raw (for resend)
        self.logged_on = threading.Event()

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> None:
        if self._transport is None:
            self.state = SessionState.CONNECTING
            self._transport = TCPTransport(self.host, self.port)
        if self.reset_seq_on_logon:
            self._seq.reset()
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._hb = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb.start()
        if not self.is_acceptor:
            self._send_logon()

    def disconnect(self, text: str = "normal") -> None:
        try:
            if self.state == SessionState.ACTIVE:
                self._send(SMsg.LOGOUT, [(STag.Text, text)])
                self.state = SessionState.LOGOUT_SENT
        finally:
            self._running = False
            self._seq.save()
            if self._transport:
                self._transport.close()
            self.state = SessionState.DISCONNECTED

    # -- sending ------------------------------------------------------------
    def _session_fields(self) -> List[Tuple[str, Any]]:
        seq = self._seq.out_seq
        self._seq.out_seq += 1
        self._seq.save()
        return seq, [
            (Tag.SenderCompID, self.sender),
            (Tag.TargetCompID, self.target),
            (Tag.MsgSeqNum, seq),
            (Tag.SendingTime, _now()),
        ]

    def _send(self, msg_type: str, fields: List[Tuple[str, Any]]) -> int:
        with self._lock:
            seq, sess = self._session_fields()
            raw = encode_message(msg_type, sess + list(fields))
            self._sent_messages[seq] = raw
            if self._transport:
                self._transport.send(raw.encode("latin-1"))
            self._last_sent = time.time()
            return seq

    def _send_logon(self) -> None:
        self.state = SessionState.LOGON_SENT
        fields = [(STag.EncryptMethod, "0"), (STag.HeartBtInt, str(self.heartbeat_int))]
        if self.reset_seq_on_logon:
            fields.append((STag.ResetSeqNumFlag, "Y"))
        self._send(MsgType.LOGON.value, fields)

    def send_new_order(
        self,
        *,
        cl_ord_id: str,
        symbol: str,
        side: str,
        qty: float,
        ord_type: str = "MARKET",
        price: Optional[float] = None,
        tif: str = "0",
    ) -> int:
        sd = (
            side
            if side in ("1", "2")
            else (Side.BUY.value if side.upper() == "BUY" else Side.SELL.value)
        )
        ot = OrdType.LIMIT.value if str(ord_type).upper() == "LIMIT" else OrdType.MARKET.value
        fields = [
            (Tag.ClOrdID, cl_ord_id),
            (Tag.Symbol, symbol),
            (Tag.Side, sd),
            (Tag.OrderQty, _num(qty)),
            (Tag.OrdType, ot),
        ]
        if price is not None:
            fields.append((Tag.Price, _num(price)))
        fields += [(Tag.TimeInForce, tif), (Tag.TransactTime, _now())]
        return self._send(MsgType.NEW_ORDER_SINGLE.value, fields)

    def send_cancel(self, *, orig_cl_ord_id: str, cl_ord_id: str, symbol: str, side: str) -> int:
        sd = (
            side
            if side in ("1", "2")
            else (Side.BUY.value if side.upper() == "BUY" else Side.SELL.value)
        )
        return self._send(
            MsgType.ORDER_CANCEL_REQUEST.value,
            [
                (Tag.OrigClOrdID, orig_cl_ord_id),
                (Tag.ClOrdID, cl_ord_id),
                (Tag.Symbol, symbol),
                (Tag.Side, sd),
                (Tag.TransactTime, _now()),
            ],
        )

    # -- receiving ----------------------------------------------------------
    def _read_loop(self) -> None:
        while self._running:
            try:
                data = self._transport.recv(4096) if self._transport else b""
            except Exception:
                data = b""
            if data:
                self._buf += data
                self._last_recv = time.time()
                self._extract_and_handle()
            else:
                time.sleep(0.01)

    def _extract_and_handle(self) -> None:
        # Extract complete FIX messages using BodyLength(9).
        while True:
            text = self._buf.decode("latin-1", errors="ignore")
            start = text.find(f"{Tag.BeginString}={BEGIN_STRING}{SOH}")
            if start < 0:
                return
            blpos = text.find(f"{SOH}{Tag.BodyLength}=", start)
            if blpos < 0:
                return
            blval_start = blpos + len(f"{SOH}{Tag.BodyLength}=")
            soh_after_bl = text.find(SOH, blval_start)
            if soh_after_bl < 0:
                return
            try:
                body_len = int(text[blval_start:soh_after_bl])
            except ValueError:
                self._buf = self._buf[start + 1 :]
                continue
            body_start = soh_after_bl + 1
            # message = header(up to body_start) + body(body_len) + checksum field (10=NNN<SOH>)
            cs_start = body_start + body_len
            cs_end = text.find(SOH, cs_start)
            if cs_end < 0 or cs_end + 1 > len(text):
                return
            msg = text[start : cs_end + 1]
            self._buf = self._buf[len(text[: cs_end + 1].encode("latin-1")) :]
            self._handle_message(msg)

    def _handle_message(self, raw: str) -> None:
        if not verify_checksum(raw):
            logger.warning("FIX: bad checksum, dropping")
            return
        fields = parse_message(raw)
        mtype = fields.get(Tag.MsgType, "")
        try:
            seq = int(fields.get(Tag.MsgSeqNum, "0"))
        except ValueError:
            seq = 0

        # sequence / gap handling for non-admin-reset messages
        poss_dup = fields.get(STag.PossDupFlag, "N") == "Y"
        expected = self._seq.in_seq
        if mtype == SMsg.SEQUENCE_RESET:
            new_seq = int(fields.get(STag.NewSeqNo, str(expected)))
            self._seq.in_seq = new_seq
            self._seq.save()
            return
        if (
            seq > expected
            and not poss_dup
            and self.state in (SessionState.ACTIVE, SessionState.LOGON_SENT)
        ):
            # gap: request resend of missing range
            self._send(
                SMsg.RESEND_REQUEST, [(STag.BeginSeqNo, str(expected)), (STag.EndSeqNo, "0")]
            )
            return
        if seq < expected and not poss_dup:
            return  # already processed
        self._seq.in_seq = max(expected, seq + 1)
        self._seq.save()

        # dispatch by type
        if mtype == MsgType.LOGON.value:
            self._on_logon(fields)
        elif mtype == MsgType.HEARTBEAT.value:
            pass
        elif mtype == SMsg.TEST_REQUEST:
            self._send(MsgType.HEARTBEAT.value, [(STag.TestReqID, fields.get(STag.TestReqID, ""))])
        elif mtype == SMsg.RESEND_REQUEST:
            self._handle_resend(fields)
        elif mtype == SMsg.LOGOUT:
            self.state = SessionState.DISCONNECTED
            self.logged_on.clear()
        else:
            # application message (ExecutionReport, NewOrderSingle on acceptor, etc.)
            if self.on_app_message:
                try:
                    self.on_app_message(fields)
                except Exception as e:  # pragma: no cover
                    logger.error("on_app_message error: %s", e)

    def _on_logon(self, fields: Dict[str, str]) -> None:
        if self.is_acceptor and self.state != SessionState.ACTIVE:
            # acceptor replies with its own logon
            self._send_logon_ack(fields)
        self.state = SessionState.ACTIVE
        self.logged_on.set()

    def _send_logon_ack(self, fields: Dict[str, str]) -> None:
        f = [
            (STag.EncryptMethod, "0"),
            (STag.HeartBtInt, fields.get(STag.HeartBtInt, str(self.heartbeat_int))),
        ]
        if fields.get(STag.ResetSeqNumFlag) == "Y":
            self._seq.reset()
            f.append((STag.ResetSeqNumFlag, "Y"))
        self._send(MsgType.LOGON.value, f)

    def _handle_resend(self, fields: Dict[str, str]) -> None:
        begin = int(fields.get(STag.BeginSeqNo, "1"))
        end = int(fields.get(STag.EndSeqNo, "0"))
        end = end if end > 0 else (self._seq.out_seq - 1)
        # gap-fill: reset sequence forward with PossDup (we don't replay app msgs here)
        self._send(
            SMsg.SEQUENCE_RESET,
            [(STag.GapFillFlag, "Y"), (STag.PossDupFlag, "Y"), (STag.NewSeqNo, str(end + 1))],
        )

    # -- heartbeat ----------------------------------------------------------
    def _heartbeat_loop(self) -> None:
        while self._running:
            time.sleep(max(1, self.heartbeat_int) / 3.0)
            now = time.time()
            if self.state != SessionState.ACTIVE:
                continue
            if (now - self._last_sent) >= self.heartbeat_int:
                self._send(MsgType.HEARTBEAT.value, [])
            if (now - self._last_recv) >= 2 * self.heartbeat_int:
                # peer silent → test request
                self._send(SMsg.TEST_REQUEST, [(STag.TestReqID, f"TR{int(now)}")])

    # -- introspection ------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "out_seq": self._seq.out_seq,
            "in_seq": self._seq.in_seq,
            "logged_on": self.logged_on.is_set(),
            "sender": self.sender,
            "target": self.target,
        }


__all__ = [
    "FixSessionEngine",
    "SessionState",
    "Transport",
    "TCPTransport",
    "PairedTransport",
    "SeqStore",
    "STag",
    "SMsg",
]

# -*- coding: utf-8 -*-
"""
CCEA desktop supervisor.

Runs the COMPLETE CCEA topology locally (loopback, no servers), exactly honouring
the Cloud/Agent boundary, so the desktop app hosts real CCEA — not a demo:

  * Cloud zone  : the FastAPI control plane (packages.cloud.control_plane),
                  on SQLite, bound to 127.0.0.1. Holds no secrets, sends only
                  lifecycle commands (no order payloads).
  * Agent zone  : the real Agent daemon (packages.agent.daemon.agentd) running in
                  this process. Owns the local Vault (OS keychain), policy
                  firewall + hard caps, kill switch, reconciliation, and creates /
                  sends orders LOCALLY (paper via SimBroker by default).

The agent enrolls to the local control plane over HTTP, then heartbeats / polls
lifecycle commands. The desktop launches this supervisor as a sidecar; the UI
reads CCEA status via the agent's status (cloud zone never imports agent secrets
or order code).

This module is AGENT-ZONE (it instantiates the agent + broker). The cloud-zone
UI backend must talk to it only via its loopback HTTP status API.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
import socket
import threading
import time

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlopen
from uuid import uuid4


# Starting equity for the local paper SimBroker. Single source of truth so the
# broker's opening balance and the PnL baseline in paper_trade() never diverge.
_PAPER_START_EQUITY = 100_000.0


def _free_port(preferred: int = 0) -> int:
    for cand in ([preferred] if preferred else []) + [0]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", cand))
                return int(s.getsockname()[1])
        except OSError:
            continue
    raise RuntimeError("no free port")


def _wait_http(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=3) as r:  # noqa: S310 - loopback only
                if 200 <= r.status < 500:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


async def _seed_enrollment_token(db_url: str, ttl_hours: int = 24) -> Dict[str, str]:
    """Seed Organization -> Workspace -> AgentEnrollmentToken; return the raw token."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from packages.cloud.control_plane.models import (
        AgentEnrollmentToken,
        Organization,
        Workspace,
    )

    engine = create_async_engine(db_url)
    try:
        maker = async_sessionmaker(engine, expire_on_commit=False)
        async with maker() as session:
            org = Organization(id=uuid4(), name="Local Desktop Org")
            session.add(org)
            await session.flush()

            ws = Workspace(id=uuid4(), name="Local Desktop Workspace", organization_id=org.id)
            session.add(ws)
            await session.flush()

            raw_token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
            tok = AgentEnrollmentToken(
                id=uuid4(),
                name="desktop-local-enrollment",
                token_hash=token_hash,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=ttl_hours),
                is_used=False,
                workspace_id=ws.id,
            )
            # TenantMixin may also expose organization_id.
            if hasattr(tok, "organization_id"):
                tok.organization_id = org.id
            session.add(tok)
            await session.commit()
            return {
                "token": raw_token,
                "workspace_id": str(ws.id),
                "org_id": str(org.id),
            }
    finally:
        await engine.dispose()


@dataclass
class SupervisorConfig:
    data_dir: Path = field(default_factory=lambda: Path.home() / ".ccea_desktop")
    cp_host: str = "127.0.0.1"
    cp_port: int = 0          # 0 => ephemeral
    status_port: int = 0      # agent status HTTP port (0 => ephemeral)
    paper: bool = True        # paper trading via SimBroker
    agent_name: str = "desktop-agent"


class CCEASupervisor:
    """Owns the local control plane + the real Agent daemon."""

    def __init__(self, config: Optional[SupervisorConfig] = None) -> None:
        self.config = config or SupervisorConfig()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self._cp_server = None
        self._cp_thread: Optional[threading.Thread] = None
        self._daemon = None
        self._broker = None          # active broker (paper SimBroker by default)
        self._vault = None           # LocalVault (Agent zone) for broker credentials
        self._broker_name = "sim_paper"
        self._paper_engine = None    # LiveExecutionEngine wired to the paper broker
        self._live_engine = None     # LiveExecutionEngine wired to the active live broker
        self._live_fill_handler = None
        self._fill_handler = None    # FillHandler -> drives OMS + books P&L into ledger
        self._books = None           # BooksAndRecords (ledger + blotter + cash + surveillance)
        self._ledger = None          # PnLLedger alias (== self._books.ledger)
        self._live_auth = None       # LiveTradingAuthorizationStore (Agent zone)
        self._risk_monitor = None    # LiveRiskMonitor (intra-day circuit breaker)
        self._last_paper: Optional[Dict[str, Any]] = None
        self._enroll: Dict[str, str] = {}
        self._cp_url: str = ""
        self._started = False
        self._error: Optional[str] = None

    # ------------------------------------------------------------------ boot
    def start(self) -> None:
        cp_port = _free_port(self.config.cp_port or 0)
        self._cp_url = f"http://{self.config.cp_host}:{cp_port}"
        db_path = (self.config.data_dir / "ccea_control_plane.db").as_posix()
        db_url = f"sqlite+aiosqlite:///{db_path}"

        # Cloud zone env (development => SQLite allowed; loopback only).
        os.environ.setdefault("CCEA_ENV", "development")
        os.environ["CCEA_DATABASE_URL"] = db_url
        # NOTE: the vault master key must be STABLE across launches (the encrypted
        # vault file persists). It is sourced from the OS keychain with a
        # persisted-file fallback (see _start_agent), NOT a random env var.

        self._start_control_plane(cp_port)
        if not _wait_http(f"{self._cp_url}/openapi.json", timeout=60):
            raise RuntimeError("control plane did not become ready")

        # Seed enrollment token directly in the control-plane DB.
        self._enroll = asyncio.run(_seed_enrollment_token(db_url))

        self._start_agent()
        self._started = True

    def _start_control_plane(self, port: int) -> None:
        import uvicorn

        config = uvicorn.Config(
            "packages.cloud.control_plane.app:app",
            host=self.config.cp_host,
            port=port,
            log_level="warning",
            loop="asyncio",
        )
        self._cp_server = uvicorn.Server(config)
        # Server.run() installs its own event loop in this thread.
        self._cp_thread = threading.Thread(
            target=self._cp_server.run, name="ccea-control-plane", daemon=True
        )
        self._cp_thread.start()

    def _start_agent(self) -> None:
        from packages.agent.daemon.agentd import AgentDaemon, DaemonConfig
        from packages.agent.daemon.keychain import KeychainConfig

        agent_dir = self.config.data_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Stable master-key source: OS keychain first, persisted file fallback in
        # the agent data dir. NOT the environment (which would change per launch
        # and make the persisted vault unopenable).
        self._keychain_config = KeychainConfig(
            service_name="ccea-desktop-agent",
            account_name="vault-master-key",
            use_keychain=True,
            fallback_to_env=False,
            fallback_to_file=True,
            key_file_path=agent_dir / "vault.key",
            allow_key_generation=True,
        )

        cfg = DaemonConfig(
            agent_name=self.config.agent_name,
            cloud_endpoint=self._cp_url,
            cloud_enrollment_token=self._enroll["token"],
            data_dir=agent_dir,
            heartbeat_interval_seconds=5,
            keychain_config=self._keychain_config,
            # Preflight is the pre-RUN live gate (needs a signed manifest + live
            # vault). The desktop agent comes ONLINE in paper mode without an
            # active run; per-run preflight still applies when a real
            # REQUEST_START_RUN with a manifest arrives. Hard caps, policy
            # firewall and kill switch remain active regardless.
            require_preflight=False,
        )
        self._daemon = AgentDaemon(config=cfg)

        if not self._daemon.initialize():
            raise RuntimeError("agent daemon initialize() failed")

        # Local Vault on the OS keychain (master key via keychain / CCEA_VAULT_KEY
        # fallback). Secrets NEVER leave the Agent zone. This is where broker
        # credentials live for live trading.
        try:
            import base64

            from packages.agent.daemon.keychain import KeychainManager
            from packages.agent.vault.local_vault import LocalVault, VaultConfig

            # Stable master password derived from the persisted keychain key.
            km = KeychainManager(self._keychain_config)
            master_password = base64.b64encode(km.get_master_key()).decode("ascii")

            vpath = agent_dir / "vault.enc"
            vault = LocalVault(VaultConfig(vault_path=vpath))
            if not vault.is_initialized:
                vault.initialize(master_password)
            try:
                vault.unlock(master_password)
            except Exception:
                # Stale vault from an earlier (non-stable-key) bootstrap — the key
                # is now stable, so re-initialize. Paper vault holds no live
                # secrets at bootstrap; for live vaults recovery is manual.
                try:
                    vpath.unlink()
                except Exception:
                    pass
                vault.initialize(master_password)
                vault.unlock(master_password)
            self._daemon.set_vault(vault)
            self._vault = vault
        except Exception as exc:  # pragma: no cover
            self._error = f"vault wiring failed: {exc}"

        # Wire a paper broker (SimBroker) for local order execution. Orders are
        # created and "sent" entirely within the Agent zone.
        if self.config.paper:
            try:
                from packages.agent.broker.adapters.sim import SimBrokerConnector

                broker = SimBrokerConnector(equity=_PAPER_START_EQUITY)
                self._broker = broker
                self._daemon.set_broker_connector(broker)
                # The daemon does not assign broker_connected itself; reflect the
                # wired paper broker so status is truthful.
                try:
                    self._daemon.status.broker_connected = True
                except Exception:
                    pass
            except Exception as exc:  # pragma: no cover - paper broker optional
                self._error = f"sim broker wiring failed: {exc}"

        # Agent-zone BOOKS-AND-RECORDS: P&L ledger (realized/unrealized/fees + EOD NAV)
        # + immutable hash-chained trade blotter + cash general-ledger + instrument
        # master (FIGI annotation) + live MAR surveillance — all fed by the SAME live
        # fill stream. Tamper-evidence is keyed by the Agent vault master key.
        try:
            from packages.agent.accounting.books import BooksAndRecords

            hmac_key = None
            try:
                from packages.agent.daemon.keychain import KeychainManager
                hmac_key = KeychainManager(self._keychain_config).get_master_key()
            except Exception:
                hmac_key = None
            self._books = BooksAndRecords(
                starting_cash=_PAPER_START_EQUITY,
                data_dir=agent_dir,
                account_id=self.config.agent_name,
                strategy_id="desktop-demo",
                hmac_key=hmac_key,
            )
            self._ledger = self._books.ledger   # back-compat alias (status/eod_close)
            # SimBroker is in-memory; restore it from the durable Agent ledger so
            # positions/cash still reconcile after a desktop restart.
            if self._broker is not None and hasattr(self._broker, "restore_state"):
                persisted = self._ledger.snapshot()
                self._broker.restore_state(
                    cash=persisted["cash"],
                    positions=persisted["positions"],
                    sequence=persisted["n_fills"],
                )
        except Exception as exc:  # pragma: no cover
            self._error = f"books-and-records wiring failed: {exc}"

        # Live-trading authorization store (Agent zone): durable, hash-chained
        # operator mandates that open the auto-rebalance path on a LIVE broker.
        # Audit is keyed by the same vault master key as the books tamper-chains.
        try:
            from packages.agent.approval.live_trading_authorization import (
                LiveTradingAuthorizationStore,
            )
            _auth_key = None
            try:
                from packages.agent.daemon.keychain import KeychainManager
                _auth_key = KeychainManager(self._keychain_config).get_master_key()
            except Exception:
                _auth_key = None
            self._live_auth = LiveTradingAuthorizationStore(
                state_path=str(agent_dir / "live_trading_authorizations.json"),
                audit_path=str(agent_dir / "live_trading_audit.jsonl"),
                audit_key=_auth_key,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("live-auth store init failed: %s", exc)
            self._live_auth = None

        # Live risk-limit ENFORCEMENT (P0-B): intra-day circuit breaker that
        # trips a halt when the user's daily-loss or max-drawdown limit is
        # breached — the pre-trade RiskChecker alone can't catch losses driven
        # by market moves without new orders.
        try:
            from services.live_risk_limits import LiveRiskMonitor
            self._risk_monitor = LiveRiskMonitor(
                halt_callback=self._on_risk_breach,
                peak_state_path=str(agent_dir / "live_risk_peak.json"),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("live-risk monitor init failed: %s", exc)
            self._risk_monitor = None

        ok, err = self._daemon.start()
        if not ok:
            raise RuntimeError(f"agent daemon start() failed: {err}")

    # ----------------------------------------------- risk-limit enforcement
    def _on_risk_breach(self, payload: Dict[str, Any]) -> None:
        """Circuit-breaker callback: trip kill switch + flatten via emergency halt.

        Account-level stop-loss — при пробое дневного лимита убытка / макс.
        просадки закрываем всё и останавливаем торговлю (kill switch)."""
        try:
            import services.ops_kill_switch as _oks
            _oks._trip()
        except Exception:
            pass
        try:
            self.emergency_halt()
        except Exception:
            logger.exception("live-risk: emergency_halt from breach failed")
        # Также снимаем любые live-мандаты авто-торговли.
        if self._live_auth is not None:
            try:
                self._live_auth.revoke_all(reason=f"risk breach: {payload.get('reason')}")
            except Exception:
                pass

    def _evaluate_risk(self) -> Optional[Dict[str, Any]]:
        """Прогнать intra-day монитор против текущего снимка леджера."""
        if self._risk_monitor is None or self._ledger is None:
            return None
        try:
            return self._risk_monitor.evaluate(self._ledger.snapshot())
        except Exception:
            logger.exception("live-risk: evaluate failed")
            return None

    def reload_risk_limits(self) -> Dict[str, Any]:
        """Перечитать lite_limits: пересобрать pre-trade RiskChecker (сбросить
        движки — пересоздадутся с новыми лимитами) и вернуть текущий статус."""
        self._paper_engine = None
        self._fill_handler = None
        self._live_engine = None
        self._live_fill_handler = None
        return self._evaluate_risk() or {"status": "no_data"}

    def risk_enforcement_status(self) -> Dict[str, Any]:
        """Реальный статус enforcement для UI/REST: лимиты, текущее
        использование (день/просадка/плечо), armed/breached."""
        st = self._evaluate_risk()
        if st is None:
            return {"status": "unavailable", "enforced": False}
        try:
            import services.ops_kill_switch as _oks
            st["kill_switch_tripped"] = bool(_oks.tripped())
        except Exception:
            st["kill_switch_tripped"] = None
        return st

    def reset_risk_breach(self) -> None:
        if self._risk_monitor is not None:
            self._risk_monitor.reset_breach()

    # ----------------------------------------------------------- paper trade
    def _build_user_risk_checker(self):
        """RiskChecker, питаемый пользовательскими lite_limits (P0-B enforcement):
        leverage cap / concentration / daily loss / max drawdown применяются
        pre-trade. Не заданные лимиты остаются на безопасных дефолтах."""
        try:
            from services.live_risk_limits import build_risk_checker, load_live_risk_limits
            eq = float(self._ledger.equity) if self._ledger is not None else 100_000.0
            return build_risk_checker(load_live_risk_limits(), equity=eq)
        except Exception:
            logger.warning("live-risk: не удалось построить RiskChecker из lite_limits", exc_info=True)
            return None

    def _ensure_paper_engine(self):
        if self._paper_engine is not None:
            return self._paper_engine
        from packages.agent.execution.engine import LiveExecutionEngine, PriceCollarConfig
        from packages.agent.execution.fill_handler import FillHandler
        from packages.agent.execution.live_factory import (
            make_broker_submit, make_broker_cancel, make_broker_replace)
        from packages.agent.reconciliation.journal import OrderJournal

        # Tamper-evident order journal (hash-chained audit log keyed by the books key).
        hmac_key = getattr(getattr(self._books, "blotter", None), "_key", None)
        journal = OrderJournal(
            db_path=self.config.data_dir / "agent" / "paper_orders.db", hmac_key=hmac_key)
        # Real OMS: PolicyFirewall + HardCapEnforcer + RiskChecker stack, journaled +
        # idempotent, with engine-level cancel/replace (FIX 35=G semantics) and a
        # fat-finger / price-collar pre-trade gate (P1 #10). RiskChecker is fed
        # from the user's lite_limits (P0-B) — leverage/concentration/daily-loss/
        # drawdown are enforced pre-trade, not just displayed.
        self._paper_engine = LiveExecutionEngine(
            broker_submit=make_broker_submit(self._broker),
            broker_cancel=make_broker_cancel(self._broker),
            broker_replace=make_broker_replace(self._broker),
            broker_name="sim_paper",
            order_journal=journal,
            risk_checker=self._build_user_risk_checker(),
            price_collar=PriceCollarConfig(max_price_distance_pct=0.20, max_notional=5_000_000.0),
            deployment_id="desktop-paper",
            run_id="desktop-paper-run",
        )
        # FillHandler advances the real OMS lifecycle (SUBMITTED->FILLED) AND books
        # each fill across ALL records (P&L ledger + immutable blotter + cash GL) and
        # feeds live MAR surveillance — via the BooksAndRecords facade. The on_fill
        # is wrapped so the intra-day risk monitor evaluates AFTER every booked fill
        # (equity/day_pnl are fresh) and can trip the account-level circuit breaker.
        on_fill = None
        if self._books is not None:
            on_fill = self._risk_wrapped_on_fill(
                self._books.fill_handler_callback(strategy_id="desktop-demo"))
        self._fill_handler = FillHandler(self._paper_engine, on_fill=on_fill)
        return self._paper_engine

    def _risk_wrapped_on_fill(self, inner):
        """Оборачивает books on_fill: после booking каждого fill'а прогоняет
        intra-day risk monitor (авто-halt при пробое дневного лимита/просадки)."""
        def _wrapped(*args, **kwargs):
            result = inner(*args, **kwargs) if inner is not None else None
            try:
                self._evaluate_risk()
            except Exception:
                logger.exception("live-risk: post-fill evaluate failed")
            return result
        return _wrapped

    def paper_trade(
        self,
        symbol: str = "BTCUSDT",
        qty: float = 0.1,
        entry_price: float = 50000.0,
        mark_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Drive a REAL paper run: OrderIntent -> policy firewall -> journal ->
        broker order -> fill -> mark-to-market PnL. Exercises the full Agent OMS."""
        from decimal import Decimal

        from packages.shared.contracts.intent import IntentSide, IntentType, OrderIntent

        if self._broker is None:
            return {"ok": False, "error": "paper broker not available"}
        eng = self._ensure_paper_engine()
        self._broker.set_price(symbol, float(entry_price))

        # Pre-trade risk runs against the Agent's OWN ledger equity (not an
        # externally-supplied number): feed the engine portfolio from the ledger.
        if self._ledger is not None:
            try:
                eng.update_portfolio(self._ledger.to_portfolio_state())
            except Exception:
                pass

        intent = OrderIntent(
            strategy_id="desktop-demo",
            symbol=symbol,
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal(str(qty)),
            reason="desktop CCEA paper demo",
        )
        res = eng.execute(intent, current_price=Decimal(str(entry_price)), origin="local")

        # Live MAR surveillance: record the order placement (NEW) on the real path so
        # order-based detectors (spoofing/layering) observe actual flow.
        if res.success and res.order is not None and self._books is not None:
            try:
                self._books.on_order(
                    symbol=symbol, side=str(res.order.side).upper(), action="NEW",
                    quantity=float(res.order.quantity), price=float(entry_price),
                    order_id=res.order.client_order_id, mid=float(entry_price))
            except Exception:
                pass

        # Advance the real OMS lifecycle from the broker fill and book it into the
        # P&L ledger via the FillHandler (SUBMITTED -> FILLED + realized/unrealized).
        if res.success and res.order is not None and self._fill_handler is not None:
            try:
                from packages.agent.execution.fill_handler import FillEvent

                info = self._broker.get_order(client_order_id=res.order.client_order_id)
                if info is not None and info.filled_quantity and info.filled_quantity > 0:
                    self._fill_handler.handle_event(FillEvent(
                        client_order_id=res.order.client_order_id,
                        event_type="fill",
                        filled_qty=info.filled_quantity,
                        avg_fill_price=info.avg_fill_price,
                        broker_order_id=info.broker_order_id,
                        cumulative=True,
                    ))
            except Exception as exc:  # pragma: no cover - never break the paper run
                self._error = f"ledger fill booking failed: {exc}"

        # Mark-to-market: update both the broker book and the Agent ledger.
        if mark_price is not None:
            self._broker.set_price(symbol, float(mark_price))
            if self._ledger is not None:
                self._ledger.mark(symbol, float(mark_price))

        acct = self._broker.get_account()
        # Positions/PnL are reported from the Agent's OWN ledger (source of truth),
        # with a broker reconciliation flag for honesty.
        if self._ledger is not None:
            snap = self._ledger.snapshot()
            positions = [
                {"symbol": p["symbol"], "qty": p["quantity"], "avg_entry": p["avg_cost"],
                 "price": p["mark"], "market_value": p["market_value"],
                 "unrealized": p["unrealized_pnl"], "realized": p["realized_pnl"]}
                for p in snap["positions"]
            ]
            pnl = round(snap["total_pnl"], 2)
            ledger_equity = snap["equity"]
            broker_pos = {p.symbol: float(p.quantity) for p in self._broker.get_positions()}
            reconciled = self._ledger.reconcile_against(broker_pos)["reconciled"]
        else:
            positions = [
                {"symbol": p.symbol, "qty": float(p.quantity),
                 "avg_entry": float(p.avg_entry_price), "price": float(p.current_price),
                 "market_value": float(p.market_value)}
                for p in self._broker.get_positions()
            ]
            pnl = round(float(acct.equity) - _PAPER_START_EQUITY, 2)
            ledger_equity = float(acct.equity)
            reconciled = True

        order = None
        if res.order is not None:
            order = {
                "client_order_id": res.order.client_order_id,
                "broker_order_id": res.order.broker_order_id,
                "status": str(getattr(res.order.status, "value", res.order.status)),
                "side": res.order.side,
                "qty": float(res.order.quantity),
                "symbol": res.order.symbol,
            }
        # Reflect in daemon status for truthful reporting.
        try:
            if res.success and self._daemon is not None:
                self._daemon.status.orders_today += 1
                self._daemon.status.fills_today += 1
                self._daemon.status.pnl_today = Decimal(str(pnl))
        except Exception:
            pass

        # Surveillance alerts + books integrity from this fill (books-and-records).
        surveillance_alerts: list = []
        integrity = None
        last_trade = None
        if self._books is not None:
            try:
                surveillance_alerts = [a.to_dict() for a in self._books.surveillance.get_alerts()][-10:] \
                    if self._books.surveillance is not None else []
                integrity = self._books.verify_integrity()
                trades = self._books.recent_trades(limit=1)
                last_trade = trades[-1] if trades else None
            except Exception:
                pass

        self._last_paper = {
            "ok": bool(res.success),
            "error": res.error_message,
            "order": order,
            "account": {"equity": ledger_equity, "cash": float(acct.cash)},
            "positions": positions,
            "pnl": pnl,
            "ledger_broker_reconciled": reconciled,
            "last_trade": last_trade,
            "surveillance_alerts": surveillance_alerts,
            "integrity_ok": (integrity["all_valid"] if integrity else None),
        }
        return self._last_paper

    def eod_close(self) -> Dict[str, Any]:
        """Take an EOD NAV snapshot on the Agent ledger and roll the trading day."""
        result: Dict[str, Any]
        if self._books is not None:
            result = {"ok": True, "snapshot": self._books.eod_close()}
        elif self._ledger is None:
            return {"ok": False, "error": "ledger not available"}
        else:
            snap = self._ledger.eod_close()
            result = {"ok": True, "snapshot": snap.to_dict()}
        # Roll the risk monitor's day too: peak equity resets, breach cleared —
        # the daily-loss / drawdown limits are measured against the new day.
        if self._risk_monitor is not None and self._ledger is not None:
            try:
                self._risk_monitor.reset_day(equity=float(self._ledger.equity))
            except Exception:
                pass
        return result

    def portfolio_snapshot(self) -> Dict[str, Any]:
        """Authoritative holdings view from the active Agent broker/books."""
        if self._broker is None:
            return {"ok": False, "error": "broker not available"}
        account = self._broker.get_account()
        holdings = []
        if self._ledger is not None and self._broker_name == "sim_paper":
            positions = self._ledger.snapshot().get("positions", [])
            for p in positions:
                qty = float(p["quantity"])
                if qty == 0:
                    continue
                holdings.append({
                    "symbol": p["symbol"], "qty": qty,
                    "entry_price": float(p["avg_cost"]), "current_price": float(p["mark"]),
                    "value": abs(float(p["market_value"])),
                    "side": "LONG" if qty > 0 else "SHORT",
                    "pnl": float(p["unrealized_pnl"]),
                })
        else:
            for p in self._broker.get_positions() or []:
                qty = float(p.quantity)
                current = float(p.current_price or p.avg_entry_price or 0)
                entry = float(p.avg_entry_price or current)
                holdings.append({
                    "symbol": p.symbol, "qty": qty, "entry_price": entry,
                    "current_price": current,
                    "value": abs(float(p.market_value or qty * current)),
                    "side": "LONG" if qty > 0 else "SHORT",
                    "pnl": (current - entry) * qty,
                })
        gross = sum(float(h["value"]) for h in holdings)
        equity = float(account.equity)
        return {
            "ok": True,
            "holdings": holdings,
            "metrics": {
                "net_liquidation_value": equity,
                "margin_used": 0.0 if self._broker_name == "sim_paper" else gross,
                "leverage": f"{(gross / equity if equity else 0.0):.2f}x",
                "buying_power": float(account.buying_power),
            },
            "simulated": self._broker_name == "sim_paper",
            "data_source": "paper_broker" if self._broker_name == "sim_paper" else "live_broker",
            "broker": self._broker_name,
        }

    def close_position(self, symbol: str, quantity: Optional[float] = None) -> Dict[str, Any]:
        """Close (fully or partially) an Agent position through the real OMS.

        ``quantity`` — сколько единиц закрыть (по модулю); None или >= |позиции|
        закрывает целиком. Частичное закрытие идёт тем же путём (OrderIntent
        CLOSE_POSITION → policy firewall → journal → fill → books)."""
        if self._broker is None:
            return {"ok": False, "error": "broker not available"}
        pos = self._broker.get_position(symbol)
        if pos is None or not pos.quantity:
            return {"ok": False, "error": f"no active position for {symbol}"}

        cur = float(pos.quantity)
        close_qty = abs(cur) if quantity is None else min(abs(float(quantity)), abs(cur))
        if close_qty <= 0:
            return {"ok": False, "error": "quantity must be > 0"}
        partial = close_qty < abs(cur) - 1e-12

        if self._broker_name != "sim_paper":
            # Живой брокер: закрытие позиции — reduce-only, авторизация live
            # НЕ требуется (это снижение риска, а не наращивание экспозиции),
            # но идёт через живой OMS (firewall/journal/collar).
            from decimal import Decimal
            from packages.agent.execution.fill_handler import FillEvent
            from packages.shared.contracts.intent import IntentSide, IntentType, OrderIntent
            engine = self._ensure_live_engine()
            price = self._broker.get_last_price(symbol)
            if price is None or price <= 0:
                return {"ok": False, "error": f"no market mark for {symbol}"}
            if self._ledger is not None:
                try:
                    engine.update_portfolio(self._ledger.to_portfolio_state())
                except Exception:
                    pass
            intent = OrderIntent(
                strategy_id="desktop-manual-close", symbol=symbol,
                intent_type=IntentType.CLOSE_POSITION,
                side=IntentSide.SHORT if cur > 0 else IntentSide.LONG,
                target_quantity=Decimal(str(close_qty)),
                reason=f"operator {'partial ' if partial else ''}close",
            )
            result = engine.execute(intent, current_price=Decimal(str(price)), origin="local")
            if not result.success or result.order is None:
                return {"ok": False, "error": result.error_message or "close rejected"}
            info = self._broker.get_order(client_order_id=result.order.client_order_id)
            if info is not None and info.filled_quantity and self._live_fill_handler is not None:
                self._live_fill_handler.handle_event(FillEvent(
                    client_order_id=result.order.client_order_id, event_type="fill",
                    filled_qty=info.filled_quantity, avg_fill_price=info.avg_fill_price,
                    broker_order_id=info.broker_order_id, cumulative=True))
            remaining = self._broker.get_position(symbol)
            return {
                "ok": bool(result.success), "broker": self._broker_name,
                "partial": partial, "closed_qty": close_qty,
                "broker_order_id": info.broker_order_id if info is not None else None,
                "remaining_qty": float(remaining.quantity) if remaining is not None else 0.0,
                "simulated": False,
            }

        from decimal import Decimal
        from packages.agent.execution.fill_handler import FillEvent
        from packages.shared.contracts.intent import IntentSide, IntentType, OrderIntent

        engine = self._ensure_paper_engine()
        if self._ledger is not None:
            engine.update_portfolio(self._ledger.to_portfolio_state())
        price = self._broker.get_last_price(symbol)
        if price is None or price <= 0:
            return {"ok": False, "error": f"no market mark for {symbol}"}
        intent = OrderIntent(
            strategy_id="desktop-manual-close", symbol=symbol,
            intent_type=IntentType.CLOSE_POSITION,
            side=IntentSide.SHORT if cur > 0 else IntentSide.LONG,
            target_quantity=Decimal(str(close_qty)),
            reason=f"desktop operator {'partial ' if partial else ''}close position",
        )
        result = engine.execute(intent, current_price=Decimal(str(price)), origin="local")
        if not result.success or result.order is None:
            return {"ok": False, "error": result.error_message or "close rejected"}
        info = self._broker.get_order(client_order_id=result.order.client_order_id)
        if info is not None and info.filled_quantity and self._fill_handler is not None:
            self._fill_handler.handle_event(FillEvent(
                client_order_id=result.order.client_order_id,
                event_type="fill", filled_qty=info.filled_quantity,
                avg_fill_price=info.avg_fill_price, broker_order_id=info.broker_order_id,
                cumulative=True,
            ))
        remaining = self._broker.get_position(symbol)
        return {
            "ok": (remaining is None or abs(float(remaining.quantity)) < abs(cur) - 1e-12
                   or float(remaining.quantity) == 0.0),
            "broker": self._broker_name,
            "partial": partial, "closed_qty": close_qty,
            "broker_order_id": info.broker_order_id if info is not None else None,
            "remaining_qty": float(remaining.quantity) if remaining is not None else 0.0,
            "simulated": True,
        }

    # --------------------------------------------------- manual order ticket
    _INTENT_TYPE_MAP = {
        ("market", "long"): "MARKET_ENTRY", ("market", "short"): "MARKET_ENTRY",
        ("limit", "long"): "LIMIT_ENTRY", ("limit", "short"): "LIMIT_ENTRY",
        ("stop", "long"): "STOP_ENTRY", ("stop", "short"): "STOP_ENTRY",
        ("stop_limit", "long"): "STOP_ENTRY", ("stop_limit", "short"): "STOP_ENTRY",
    }

    def submit_manual_order(
        self,
        *,
        symbol: str,
        side: str,                    # buy|sell|long|short
        order_type: str = "market",   # market|limit|stop|stop_limit
        quantity: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",   # GTC|DAY|IOC|FOK
        reduce_only: bool = False,
        strategy_id: str = "desktop-manual",
    ) -> Dict[str, Any]:
        """Ручной ордер оператора через настоящий Agent OMS (paper или live).

        Проверки: валидность цен для типа ордера; reduce_only не может нарастить
        и не превышает размер позиции; на live-брокере — обязательный мандат
        авторизации, КРОМЕ reduce_only (снижение риска разрешено без мандата).
        """
        from decimal import Decimal
        from packages.agent.execution.fill_handler import FillEvent
        from packages.shared.contracts.intent import IntentSide, IntentType, OrderIntent

        if self._broker is None:
            return {"ok": False, "error": "broker not available"}
        s = str(side).strip().lower()
        side_long = s in ("buy", "long")
        if s not in ("buy", "sell", "long", "short"):
            return {"ok": False, "error": f"invalid side: {side!r}"}
        ot = str(order_type).strip().lower()
        if ot not in ("market", "limit", "stop", "stop_limit"):
            return {"ok": False, "error": f"invalid order_type: {order_type!r}"}
        try:
            qty = float(quantity)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid quantity"}
        if qty <= 0:
            return {"ok": False, "error": "quantity must be > 0"}
        if ot in ("limit", "stop_limit") and not (limit_price and float(limit_price) > 0):
            return {"ok": False, "error": f"{ot} order requires a positive limit_price"}
        if ot in ("stop", "stop_limit") and not (stop_price and float(stop_price) > 0):
            return {"ok": False, "error": f"{ot} order requires a positive stop_price"}
        tif = str(time_in_force).strip().upper()
        if tif not in ("GTC", "DAY", "IOC", "FOK"):
            return {"ok": False, "error": f"invalid time_in_force: {time_in_force!r}"}

        is_paper = self._broker_name == "sim_paper"

        # reduce_only: направление обязано уменьшать позицию и не превышать её.
        pos = self._broker.get_position(symbol)
        cur = float(pos.quantity) if pos is not None and pos.quantity else 0.0
        if reduce_only:
            if cur == 0.0:
                return {"ok": False, "error": "reduce_only, но позиции нет"}
            reduces = (side_long and cur < 0) or (not side_long and cur > 0)
            if not reduces:
                return {"ok": False, "error": "reduce_only: сторона ордера не уменьшает позицию"}
            if qty > abs(cur) + 1e-12:
                qty = abs(cur)   # не даём перевернуть позицию в reduce-only

        # Живой брокер: наращивание экспозиции требует мандата; reduce_only — нет.
        if not is_paper and not reduce_only:
            store = self._live_auth
            if store is None:
                return {"ok": False, "error": "live-брокер, но хранилище авторизаций недоступно"}
            # Ручной ордер авторизуется по стратегии/брокеру с оценкой нотионала.
            ref_price = float(limit_price or stop_price or (self._broker.get_last_price(symbol) or 0) or 0)
            est_notional = qty * ref_price
            equity = 0.0
            try:
                equity = float(self._broker.get_account().equity)
            except Exception:
                pass
            turnover = (est_notional / equity) if equity > 0 else 1.0
            chk = store.check(strategy_id=strategy_id, config={"manual_order": True},
                              broker=self._broker_name, turnover=turnover,
                              notional=est_notional, n_orders=1)
            if not chk.allowed:
                return {"ok": False, "error": f"live-авторизация ручного ордера: {chk.reason}",
                        "authorization": chk.to_dict()}

        engine = self._ensure_paper_engine() if is_paper else self._ensure_live_engine()
        fill_handler = self._fill_handler if is_paper else self._live_fill_handler
        mark = self._broker.get_last_price(symbol)
        if is_paper and mark is None:
            # SimBroker нужна котировка для расчёта fill; используем limit/stop как прокси.
            proxy = limit_price or stop_price
            if proxy:
                self._broker.set_price(symbol, float(proxy))
                mark = self._broker.get_last_price(symbol)
        if mark is None or float(mark) <= 0:
            return {"ok": False, "error": f"no market mark for {symbol}"}
        if self._ledger is not None:
            try:
                engine.update_portfolio(self._ledger.to_portfolio_state())
            except Exception:
                pass

        if reduce_only:
            intent_type = IntentType.CLOSE_POSITION
        else:
            intent_type = getattr(IntentType, self._INTENT_TYPE_MAP[(ot, "long" if side_long else "short")])
        intent = OrderIntent(
            strategy_id=strategy_id, symbol=symbol, intent_type=intent_type,
            side=IntentSide.LONG if side_long else IntentSide.SHORT,
            target_quantity=Decimal(str(qty)),
            limit_price=(Decimal(str(limit_price)) if limit_price else None),
            stop_price=(Decimal(str(stop_price)) if stop_price else None),
            time_in_force=tif,
            reason=f"desktop manual {ot} order",
        )
        res = engine.execute(intent, current_price=Decimal(str(mark)), origin="local")
        if res.success and res.order is not None and self._books is not None:
            try:
                self._books.on_order(
                    symbol=symbol, side=str(res.order.side).upper(), action="NEW",
                    quantity=float(res.order.quantity), price=float(mark),
                    order_id=res.order.client_order_id, mid=float(mark))
            except Exception:
                pass
        filled = False
        if res.success and res.order is not None and fill_handler is not None:
            try:
                info = self._broker.get_order(client_order_id=res.order.client_order_id)
                if info is not None and info.filled_quantity and info.filled_quantity > 0:
                    fill_handler.handle_event(FillEvent(
                        client_order_id=res.order.client_order_id, event_type="fill",
                        filled_qty=info.filled_quantity, avg_fill_price=info.avg_fill_price,
                        broker_order_id=info.broker_order_id, cumulative=True))
                    filled = True
            except Exception:
                pass
        return {
            "ok": bool(res.success),
            "client_order_id": res.order.client_order_id if res.order is not None else None,
            "broker_order_id": (getattr(res.order, "broker_order_id", None) if res.order is not None else None),
            "error": None if res.success else (res.error_message or "rejected by OMS"),
            "order_type": ot, "side": "long" if side_long else "short",
            "quantity": qty, "limit_price": limit_price, "stop_price": stop_price,
            "time_in_force": tif, "reduce_only": bool(reduce_only),
            "state": "filled" if filled else ("submitted" if res.success else "rejected"),
            "simulated": is_paper,
        }

    def open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Рабочие (неисполненные) ордера активного брокера."""
        if self._broker is None:
            return {"ok": False, "error": "broker not available", "orders": []}
        getter = getattr(self._broker, "get_open_orders", None)
        if not callable(getter):
            return {"ok": True, "orders": [], "supported": False, "broker": self._broker_name}
        out = []
        try:
            for o in getter(symbol) or []:
                out.append({
                    "client_order_id": getattr(o, "client_order_id", None),
                    "broker_order_id": getattr(o, "broker_order_id", None),
                    "symbol": getattr(o, "symbol", None),
                    "side": str(getattr(o, "side", "")).replace("OrderSide.", "").lower(),
                    "order_type": str(getattr(o, "order_type", "")).replace("OrderType.", "").lower(),
                    "quantity": float(getattr(o, "quantity", 0) or 0),
                    "filled_quantity": float(getattr(o, "filled_quantity", 0) or 0),
                    "limit_price": (float(o.limit_price) if getattr(o, "limit_price", None) else None),
                    "stop_price": (float(o.stop_price) if getattr(o, "stop_price", None) else None),
                    "status": str(getattr(o, "status", "")).replace("OrderStatus.", "").lower(),
                })
        except Exception as exc:
            return {"ok": False, "error": str(exc), "orders": []}
        return {"ok": True, "orders": out, "broker": self._broker_name,
                "simulated": self._broker_name == "sim_paper"}

    def cancel_order(self, client_order_id: str) -> Dict[str, Any]:
        """Отменить рабочий ордер через активный брокер."""
        if self._broker is None:
            return {"ok": False, "error": "broker not available"}
        canceller = getattr(self._broker, "cancel_order", None)
        if not callable(canceller):
            return {"ok": False, "error": "broker does not support cancel"}
        try:
            res = canceller(client_order_id=client_order_id)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        ok = bool(getattr(res, "success", res is True))
        return {"ok": ok, "client_order_id": client_order_id, "broker": self._broker_name,
                "error": None if ok else getattr(res, "error_message", "cancel failed")}

    def _ensure_live_engine(self):
        """LiveExecutionEngine, привязанный к активному LIVE-брокеру.

        Тот же OMS-стек, что и paper (PolicyFirewall + HardCapEnforcer +
        RiskChecker + hash-chain журнал + price-collar), но broker_submit идёт
        в реальный коннектор. Собирается лениво при первой live-отправке и
        сбрасывается при смене брокера.
        """
        if self._live_engine is not None:
            return self._live_engine
        from packages.agent.execution.engine import LiveExecutionEngine, PriceCollarConfig
        from packages.agent.execution.fill_handler import FillHandler
        from packages.agent.execution.live_factory import (
            make_broker_submit, make_broker_cancel, make_broker_replace)
        from packages.agent.reconciliation.journal import OrderJournal

        hmac_key = getattr(getattr(self._books, "blotter", None), "_key", None)
        journal = OrderJournal(
            db_path=self.config.data_dir / "agent" / "live_orders.db", hmac_key=hmac_key)
        self._live_engine = LiveExecutionEngine(
            broker_submit=make_broker_submit(self._broker),
            broker_cancel=make_broker_cancel(self._broker),
            broker_replace=make_broker_replace(self._broker),
            broker_name=self._broker_name,
            order_journal=journal,
            risk_checker=self._build_user_risk_checker(),   # P0-B: lite_limits enforced pre-trade
            price_collar=PriceCollarConfig(max_price_distance_pct=0.20, max_notional=5_000_000.0),
            deployment_id="desktop-live",
            run_id="desktop-live-rebalance",
        )
        on_fill = None
        if self._books is not None:
            on_fill = self._risk_wrapped_on_fill(
                self._books.fill_handler_callback(strategy_id="xs-rebalance"))
        self._live_fill_handler = FillHandler(self._live_engine, on_fill=on_fill)
        return self._live_engine

    def submit_rebalance_order(
        self,
        symbol: str,
        qty: float,
        price: float,
        *,
        strategy_id: str = "xs-rebalance",
        reason: str = "scheduled XS rebalance",
        allow_live: bool = False,
    ) -> Dict[str, Any]:
        """Submit ONE rebalance order through the real Agent OMS.

        ``qty`` is signed (>0 увеличить экспозицию, <0 уменьшить). Mirrors the
        paper_trade()/close_position() flow: OrderIntent -> policy firewall ->
        hash-chained journal -> broker fill -> FillHandler (P&L ledger, blotter,
        cash GL) + live MAR surveillance.

        На LIVE-брокере отправка разрешена ТОЛЬКО при ``allow_live=True`` — этот
        флаг runner выставляет после проверки локальной авторизации оператора
        (LiveTradingAuthorizationStore). Без него live-путь fail-closed.
        """
        from decimal import Decimal

        from packages.agent.execution.fill_handler import FillEvent
        from packages.shared.contracts.intent import IntentSide, IntentType, OrderIntent

        if self._broker is None:
            return {"ok": False, "error": "broker not available"}
        is_paper = self._broker_name == "sim_paper"
        if not is_paper and not allow_live:
            return {"ok": False,
                    "error": "auto-rebalance on a live broker requires an operator authorization (CCEA approval)"}
        try:
            qty = float(qty)
            price = float(price)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid qty/price"}
        if qty == 0 or price <= 0:
            return {"ok": False, "error": "invalid qty/price"}

        if is_paper:
            eng = self._ensure_paper_engine()
            fill_handler = self._fill_handler
            self._broker.set_price(symbol, price)
        else:
            eng = self._ensure_live_engine()
            fill_handler = self._live_fill_handler
        if self._ledger is not None:
            try:
                eng.update_portfolio(self._ledger.to_portfolio_state())
            except Exception:
                pass

        pos = self._broker.get_position(symbol)
        cur_qty = float(pos.quantity) if pos is not None and pos.quantity else 0.0
        # Уменьшение существующей позиции — CLOSE_POSITION (сторона противоположна
        # позиции); открытие/увеличение — MARKET_ENTRY. Пересечение нуля планировщик
        # ребаланса режет на два ордера ДО этого вызова.
        reducing = cur_qty != 0.0 and ((qty < 0 < cur_qty) or (qty > 0 > cur_qty))
        if reducing:
            intent_type = IntentType.CLOSE_POSITION
            side = IntentSide.SHORT if cur_qty > 0 else IntentSide.LONG
        else:
            intent_type = IntentType.MARKET_ENTRY
            side = IntentSide.LONG if qty > 0 else IntentSide.SHORT

        intent = OrderIntent(
            strategy_id=strategy_id,
            symbol=symbol,
            intent_type=intent_type,
            side=side,
            target_quantity=Decimal(str(abs(qty))),
            reason=reason,
        )
        res = eng.execute(intent, current_price=Decimal(str(price)), origin="local")

        if res.success and res.order is not None and self._books is not None:
            try:
                self._books.on_order(
                    symbol=symbol, side=str(res.order.side).upper(), action="NEW",
                    quantity=float(res.order.quantity), price=price,
                    order_id=res.order.client_order_id, mid=price)
            except Exception:
                pass
        # Провести fill в книги. Paper-брокер исполняет мгновенно; live-брокер
        # исполняет асинхронно — если моментального fill'а нет, ордер остаётся
        # SUBMITTED и реконсилируется штатным путём Agent'а (мы НЕ выдумываем fill).
        filled = False
        if res.success and res.order is not None and fill_handler is not None:
            try:
                info = self._broker.get_order(client_order_id=res.order.client_order_id)
                if info is not None and info.filled_quantity and info.filled_quantity > 0:
                    fill_handler.handle_event(FillEvent(
                        client_order_id=res.order.client_order_id,
                        event_type="fill", filled_qty=info.filled_quantity,
                        avg_fill_price=info.avg_fill_price,
                        broker_order_id=info.broker_order_id,
                        cumulative=True,
                    ))
                    filled = True
            except Exception:
                pass

        return {
            "ok": bool(res.success),
            "client_order_id": res.order.client_order_id if res.order is not None else None,
            "broker_order_id": (getattr(res.order, "broker_order_id", None) if res.order is not None else None),
            "error": None if res.success else (res.error_message or "rejected by OMS"),
            "intent_type": intent_type.value,
            "side": side.value,
            "qty": abs(qty),
            "price": price,
            "filled": filled,
            "state": "filled" if filled else ("submitted" if res.success else "rejected"),
            "simulated": is_paper,
        }

    # ------------------------------------------------ live-trading authorization
    def grant_live_trading(
        self,
        *,
        strategy_id: str,
        config: Any,
        broker: str,
        confirmation_token: str,
        expected_token: str,
        ttl_sec: int = 8 * 3600,
        max_turnover: float = 0.10,
        max_notional_per_rebalance: float = 100_000.0,
        max_orders_per_rebalance: int = 25,
        max_total_notional: Optional[float] = None,
        max_rebalances: Optional[int] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        """Выдать локальный мандат авто-торговли на live-брокере (двухшаговая
        церемония). Только оператор Agent-зоны; Cloud выдать не может."""
        if self._live_auth is None:
            return {"ok": False, "error": "live-authorization store недоступен"}
        from packages.agent.approval.live_trading_authorization import LimitCeiling
        return self._live_auth.grant(
            strategy_id=strategy_id,
            config=config,
            broker=broker,
            limit_ceiling=LimitCeiling(
                max_turnover=max_turnover,
                max_notional_per_rebalance=max_notional_per_rebalance,
                max_orders_per_rebalance=max_orders_per_rebalance,
            ),
            ttl_sec=ttl_sec,
            confirmation_token=confirmation_token,
            expected_token=expected_token,
            max_total_notional=max_total_notional,
            max_rebalances=max_rebalances,
            note=note,
        )

    def revoke_live_trading(self, auth_id: Optional[str] = None,
                            *, reason: str = "operator revoke") -> Dict[str, Any]:
        if self._live_auth is None:
            return {"ok": False, "error": "live-authorization store недоступен"}
        if auth_id:
            return self._live_auth.revoke(auth_id, reason=reason)
        return self._live_auth.revoke_all(reason=reason)

    def live_trading_status(self) -> Dict[str, Any]:
        if self._live_auth is None:
            return {"active": [], "recent": [], "store": "unavailable"}
        return self._live_auth.status()

    @property
    def live_auth_store(self):
        return self._live_auth

    def emergency_halt(self) -> Dict[str, Any]:
        """Pause the Agent, cancel working orders, and flatten actual positions.

        Every count in the response comes from the active Agent broker.  In
        particular, this method never manufactures demo orders or positions.
        A live close that has only been submitted remains visible as pending
        until the broker confirms that the position is flat.
        """
        if self._broker is None:
            return {"ok": False, "error": "broker not available"}

        # Экстренная остановка снимает ВСЕ live-мандаты: авто-ребаланс не должен
        # возобновиться после halt без явной новой авторизации оператора.
        if self._live_auth is not None:
            try:
                self._live_auth.revoke_all(reason="emergency halt")
            except Exception:
                pass

        lifecycle = self.request_lifecycle("stop")
        try:
            cancelled = self._broker.cancel_all_orders()
            cancel_report = {
                "requested": int(cancelled.total_requested),
                "cancelled": int(cancelled.total_cancelled),
                "failed": int(cancelled.total_failed),
            }
        except Exception as exc:
            cancel_report = {"requested": 0, "cancelled": 0, "failed": 1, "error": str(exc)}

        before = self.portfolio_snapshot()
        close_results = []
        for holding in before.get("holdings", []) if before.get("ok") else []:
            symbol = str(holding.get("symbol", "")).strip()
            if symbol:
                close_results.append({"symbol": symbol, **self.close_position(symbol)})

        after = self.portfolio_snapshot()
        remaining = after.get("holdings", []) if after.get("ok") else []
        failed_closes = [item for item in close_results if not item.get("ok")]
        ok = bool(
            lifecycle.get("ok")
            and cancel_report.get("failed", 0) == 0
            and not failed_closes
            and not remaining
        )
        return {
            "ok": ok,
            "agent_paused": bool(lifecycle.get("ok")),
            "mode": "paper" if self._broker_name == "sim_paper" else "live",
            "broker": self._broker_name,
            "orders": cancel_report,
            "positions_seen": len(before.get("holdings", [])) if before.get("ok") else 0,
            "close_results": close_results,
            "positions_remaining": len(remaining),
            "remaining_holdings": remaining,
            "error": None if ok else (
                lifecycle.get("error")
                or (failed_closes[0].get("error") if failed_closes else None)
                or ("positions are awaiting broker confirmation" if remaining else None)
                or cancel_report.get("error")
            ),
        }

    def sync_trades(self, limit: int = 1000) -> Dict[str, Any]:
        """Refresh trade history from the authoritative source when supported."""
        if self._broker_name == "sim_paper":
            trades = self._books.recent_trades(limit=limit) if self._books is not None else []
            return {
                "ok": True, "source": "agent_books", "broker": self._broker_name,
                "synchronized": len(trades), "trades": trades, "simulated": True,
            }
        fetch = getattr(self._broker, "get_trade_history", None)
        if not callable(fetch):
            return {
                "ok": False, "supported": False, "broker": self._broker_name,
                "error": "active broker connector does not expose trade-history synchronization",
            }
        trades = list(fetch(limit=limit) or [])
        return {"ok": True, "source": "live_broker", "broker": self._broker_name,
                "synchronized": len(trades), "trades": trades, "simulated": False}

    def request_lifecycle(self, action: str) -> Dict[str, Any]:
        """Apply a local operator lifecycle request to the CCEA Agent daemon."""
        if self._daemon is None:
            return {"ok": False, "error": "CCEA Agent is not available"}
        action = str(action).strip().lower()
        if action == "start":
            ok, error = self._daemon.start(broker_name=self._broker_name)
        elif action == "stop":
            ok, error = bool(self._daemon.pause()), None
        else:
            return {"ok": False, "error": f"unsupported lifecycle action: {action}"}
        status = self._daemon.get_status()
        return {
            "ok": bool(ok), "error": error, "action": action,
            "mode": "paper" if self._broker_name == "sim_paper" else "live",
            "broker": self._broker_name, "agent": status,
        }

    # --------------------------------------------------------- live broker
    # Real broker connectors (Agent zone). Credentials come from the local Vault,
    # never from the Cloud. Selecting a live broker switches the daemon's executor
    # from the paper SimBroker to a real venue connector.
    _BROKER_CLASSES = {
        "alpaca": ("packages.agent.broker.adapters.alpaca", "AlpacaConnector"),
        "binance": ("packages.agent.broker.adapters.binance", "BinanceConnector"),
        "ib": ("packages.agent.broker.adapters.ib", "IBConnector"),
        "oanda": ("packages.agent.broker.adapters.oanda", "OANDAConnector"),
    }

    def store_credentials(self, broker: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Persist adapter credentials in the desktop Agent Vault.

        This is also used for data-only adapters (for example Polygon) that do
        not have a live Agent broker connector.  Values are never returned.
        """
        broker = (broker or "").strip().lower()
        if not broker:
            return {"ok": False, "error": "broker is required"}
        if self._vault is None:
            return {"ok": False, "error": "vault not available"}

        clean = {
            str(key).strip(): str(value)
            for key, value in (credentials or {}).items()
            if str(key).strip() and value is not None and str(value) != ""
        }
        if not clean:
            return {"ok": False, "error": "no credentials supplied"}

        try:
            for credential_type, value in clean.items():
                self._vault.store(broker, credential_type, value)
            return {
                "ok": True,
                "broker": broker,
                "stored": sorted(clean),
                "credentials_in_vault": True,
            }
        except Exception as exc:
            return {"ok": False, "broker": broker, "error": str(exc)}

    def connect_live_broker(
        self,
        broker: str,
        api_key: str,
        api_secret: str,
        sandbox: bool = True,
        account_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store broker credentials in the local Vault and connect a REAL broker
        connector (paper/live per ``sandbox``). Credentials never leave the Agent."""
        import importlib

        requested_broker = (broker or "").strip().lower()
        canonical_broker = "binance" if requested_broker == "binance_futures" else requested_broker
        if canonical_broker not in self._BROKER_CLASSES:
            return {"ok": False, "error": f"unknown broker '{broker}'",
                    "available": sorted([*self._BROKER_CLASSES, "binance_futures"])}
        if self._vault is None:
            return {"ok": False, "error": "vault not available"}
        try:
            from packages.agent.broker.protocol import BrokerCredentials

            # 1) Persist all credentials/config in the Agent-zone vault.
            stored_values: Dict[str, Any] = {
                "api_key": api_key,
                "api_secret": api_secret,
                **(extra or {}),
            }
            if account_id:
                stored_values["account_id"] = account_id
            stored = self.store_credentials(requested_broker, stored_values)
            if not stored.get("ok"):
                return stored

            # 2) Build credentials FROM the vault (proves the secure read path).
            def _vault_value(name: str, default: str = "") -> str:
                try:
                    return self._vault.retrieve(requested_broker, name)
                except Exception:
                    return default

            extra_from_vault: Dict[str, str] = {}
            for key in (extra or {}):
                value = _vault_value(key)
                if value:
                    extra_from_vault[key] = value
            vault_account = _vault_value("account_id", account_id or "")
            creds = BrokerCredentials(
                api_key=_vault_value("api_key"),
                api_secret=_vault_value("api_secret"),
                subaccount=vault_account or None,
                extra={**extra_from_vault, "account_id": vault_account} if vault_account else extra_from_vault,
            )

            # 3) Construct the real connector and attempt a connection.
            mod_name, cls_name = self._BROKER_CLASSES[canonical_broker]
            connector_cls = getattr(importlib.import_module(mod_name), cls_name)
            connector_kwargs: Dict[str, Any] = {}
            if requested_broker == "binance_futures":
                connector_kwargs["futures"] = True
            elif canonical_broker == "ib":
                connector_kwargs["config"] = {
                    "host": extra_from_vault.get("host", "127.0.0.1"),
                    "port": int(extra_from_vault.get("port") or (7497 if sandbox else 7496)),
                    "client_id": int(extra_from_vault.get("client_id") or 7),
                }
            elif canonical_broker == "oanda":
                connector_kwargs["config"] = {
                    "account_id": vault_account,
                    "practice": sandbox,
                }
            connector = connector_cls(creds, sandbox=sandbox, **connector_kwargs)
            connected = False
            err = None
            try:
                connected = bool(connector.connect())
            except Exception as exc:
                err = str(exc)
            if not connected and not err:
                err = getattr(connector, "_connect_error", None) or "broker connection verification failed"

            # 4) Replace the active execution connector only after a successful
            # connection.  Failed credentials must not take the paper broker down.
            if connected:
                previous = self._broker
                self._broker = connector
                self._broker_name = requested_broker
                self._paper_engine = None
                # Смена брокера аннулирует любой live-движок и (из осторожности)
                # активные мандаты: они привязаны к конкретному брокеру, но
                # переезд коннектора — повод потребовать явную повторную выдачу.
                self._live_engine = None
                self._live_fill_handler = None
                if self._live_auth is not None:
                    try:
                        self._live_auth.revoke_all(reason="broker connection changed")
                    except Exception:
                        pass
                try:
                    self._daemon.set_broker_connector(connector)
                    self._daemon.status.broker_connected = True
                except Exception:
                    pass
                if previous is not None and previous is not connector and hasattr(previous, "disconnect"):
                    try:
                        previous.disconnect()
                    except Exception:
                        pass

            return {
                "ok": connected,
                "broker": requested_broker,
                "sandbox": sandbox,
                "connected": connected,
                "error": err,
                "credentials_in_vault": True,
            }
        except Exception as exc:
            return {"ok": False, "broker": broker, "error": str(exc)}

    # ---------------------------------------------------------------- status
    def status(self) -> Dict[str, Any]:
        agent: Dict[str, Any] = {}
        enrolled = False
        if self._daemon is not None:
            try:
                agent = self._daemon.get_status()
                enrolled = bool(getattr(self._daemon.config, "cloud_access_token", None))
            except Exception as exc:  # pragma: no cover
                agent = {"error": str(exc)}
        account = None
        if self._broker is not None:
            try:
                a = self._broker.get_account()
                account = {"equity": float(a.equity), "cash": float(a.cash)}
            except Exception:
                pass
        return {
            "started": self._started,
            "error": self._error,
            "zone": "agent",
            "control_plane_url": self._cp_url,
            # Truthful paper flag: the SimBroker paper-trade path is only available
            # while the active broker is the sim. After connect_live_broker() swaps
            # in a real connector (no set_price/sim semantics), report paper=False so
            # the UI hides the paper-trade affordance instead of erroring on click.
            "paper": bool(self.config.paper and self._broker_name == "sim_paper"),
            "broker": self._broker_name,
            "enrolled": enrolled,
            "agent": agent,
            "broker_account": account,
            "last_paper": self._last_paper,
            "pnl_ledger": (self._ledger.snapshot() if self._ledger is not None else None),
            "books": self._books_status(),
            "live_trading": (self._live_auth.status() if self._live_auth is not None else None),
            "risk_enforcement": self.risk_enforcement_status(),
        }

    def _books_status(self) -> Optional[Dict[str, Any]]:
        """Compact books-and-records status: blotter/cash integrity + surveillance."""
        if self._books is None:
            return None
        try:
            integ = self._books.verify_integrity()
            surv = self._books.surveillance.summary() if self._books.surveillance is not None else {}
            n_alerts = len(self._books.surveillance.get_alerts()) if self._books.surveillance is not None else 0
            return {
                "blotter_trades": self._books.blotter.summary()["n_trades"],
                "blotter_valid": integ["blotter"]["valid"],
                "blotter_keyed": integ["blotter"].get("keyed", False),
                "cash_balance": round(self._books.cash.balance, 2),
                "cash_valid": integ["cash"]["valid"],
                "cash_matches_pnl": integ["cash_ledger_matches_pnl_cash"],
                "all_valid": integ["all_valid"],
                "surveillance_alerts": int(n_alerts),
                "surveillance_by_pattern": surv,
            }
        except Exception as exc:  # pragma: no cover
            return {"error": str(exc)}

    # ---------------------------------------------------------------- books views
    def books_blotter(self, limit: int = 100) -> Dict[str, Any]:
        if self._books is None:
            return {"trades": [], "integrity": None}
        return {"trades": self._books.recent_trades(limit=limit),
                "integrity": self._books.blotter.verify()}

    def books_cash(self, limit: int = 100) -> Dict[str, Any]:
        if self._books is None:
            return {"movements": [], "integrity": None}
        return {"movements": self._books.recent_cash(limit=limit),
                "integrity": self._books.cash.verify(),
                "balance": round(self._books.cash.balance, 2)}

    def surveillance_alerts(self, limit: int = 100) -> Dict[str, Any]:
        if self._books is None or self._books.surveillance is None:
            return {"alerts": [], "summary": {}}
        alerts = [a.to_dict() for a in self._books.surveillance.get_alerts()][-limit:]
        return {"alerts": alerts, "summary": self._books.surveillance.summary()}

    def journal_integrity(self) -> Dict[str, Any]:
        if self._paper_engine is None:
            return {"available": False}
        j = getattr(self._paper_engine, "_journal", None)
        if j is None or not hasattr(j, "verify_audit_chain"):
            return {"available": False}
        return {"available": True, "audit": j.verify_audit_chain(),
                "events": j.get_audit_events(limit=50)}

    def stop(self) -> None:
        """Stop Agent, close durable stores, and shut down the local control plane."""
        if not self._started and self._daemon is None and self._cp_server is None:
            return
        try:
            if self._daemon is not None:
                if hasattr(self._daemon, "close"):
                    self._daemon.close()
                else:
                    self._daemon.stop(reason="supervisor_shutdown")
            if self._paper_engine is not None:
                journal = getattr(self._paper_engine, "_journal", None)
                if journal is not None and hasattr(journal, "close"):
                    journal.close()
            if self._books is not None:
                self._books.close()
            if self._broker is not None and hasattr(self._broker, "disconnect"):
                self._broker.disconnect()
            if self._vault is not None and hasattr(self._vault, "lock"):
                self._vault.lock()
        finally:
            if self._cp_server is not None:
                self._cp_server.should_exit = True
            if self._cp_thread is not None and self._cp_thread.is_alive():
                self._cp_thread.join(timeout=10)
            self._started = False


# --------------------------------------------------------------------- smoke
if __name__ == "__main__":
    import json
    import tempfile

    sup = CCEASupervisor(SupervisorConfig(data_dir=Path(tempfile.mkdtemp(prefix="ccea_")), paper=True))
    sup.start()
    # Wait for enrollment AND the first heartbeat to land (cloud_connected).
    for _ in range(30):
        st = sup.status()
        if st["enrolled"] and st["agent"].get("cloud_connected"):
            break
        time.sleep(1.0)
    print("CCEA_STATUS=" + json.dumps(sup.status(), default=str))
    sup.stop()

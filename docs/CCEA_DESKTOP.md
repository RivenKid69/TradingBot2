# CCEA in the Desktop App (real, local, no servers)

The desktop app hosts a **real CCEA** (Cloud-Controlled Execution Architecture)
stack locally — not a demo. The desktop *is* the client environment, which is
exactly the CCEA **Agent zone**: secrets live on the device (OS keychain), and
orders are created/sent locally. The **Cloud zone** (control plane) runs locally
too, over loopback, holding no secrets and sending only lifecycle commands.

## Topology (one machine, loopback, zero servers)

```
┌──────────────── RivenQuant.exe (Tauri shell) ───────────────────────┐
│  WebView  →  UI (index.html)  →  /api/ccea/status (read-only)        │
│                                                                      │
│  Python sidecar (desktop_backend.py → app:api)  [UI / control surface]│
│        │  RIVEN_ENABLE_CCEA=1 launches ↓ (ccea/desktop_supervisor.py) │
│  ┌─────────────── CCEA Supervisor (in-process) ──────────────────┐  │
│  │  Cloud zone:  packages.cloud.control_plane (FastAPI, SQLite)   │  │
│  │               127.0.0.1:<ephemeral> — no secrets, no orders    │  │
│  │        ▲  enroll / heartbeat / poll commands (HTTP, signed)    │  │
│  │  Agent zone:  packages.agent.daemon.agentd (in-process)        │  │
│  │     • LocalVault (AES-256-GCM) unlocked via OS keychain        │  │
│  │     • Policy firewall + hard caps + kill switch + reconcile    │  │
│  │     • SimBroker (paper) / live broker — orders created LOCALLY │  │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

## How it runs

- `desktop_backend.py` sets `RIVEN_ENABLE_CCEA=1` and `RIVEN_DATA_DIR=<app-data>`.
- On FastAPI startup, `app.py` lazily launches `ccea/desktop_supervisor.py:CCEASupervisor`
  in a background thread (the Agent-zone code is imported only here, so the plain
  MVP surface never pulls order/secret modules).
- The supervisor:
  1. boots the **control plane** (`packages.cloud.control_plane.app:app`) via
     Uvicorn on a free loopback port, SQLite under the app-data dir
     (`CCEA_ENV=development`, `CCEA_DATABASE_URL=sqlite+aiosqlite://…`);
  2. seeds `Organization → Workspace → AgentEnrollmentToken` directly in the DB;
  3. boots the real **Agent daemon** (`AgentDaemon`), which auto-enrolls over
     HTTP, then heartbeats/polls lifecycle commands;
  4. wires the **Vault** (master key from the OS keychain via `KeychainManager`,
     persisted-file fallback in the app-data dir — **stable across launches**),
     and a **paper broker** (`SimBrokerConnector`).
- The UI polls `GET /api/ccea/status` and shows a live card on Home:
  *Enrolled · Cloud link · Vault · Broker*, plus agent state and kill-switch.

## Boundary (enforced)

- The **cloud control plane** (`packages/cloud`) imports no broker/order/secret
  code — verified by the CCEA guardrails (`tests/ccea/guardrails/`:
  import-boundary, dependency allowlist, intent/order-payload prohibition).
- Lifecycle commands may not contain order-like fields (`side`, `quantity`,
  `price`, `intent`, …) — enforced in `command_service.PROHIBITED_PAYLOAD_FIELDS`.
- `/api/ccea/status` is **read-only** and returns only state flags (no secrets,
  no orders) across the UI boundary.
- Secrets (broker creds, agent private key) never leave the Agent zone; the
  cloud receives only the agent's public key at enrollment.

## Verified

| Check | Result |
|------|--------|
| CCEA test suite (`tests/ccea`) | 2659 passed (peripheral TUF/auto-update stubs + Windows-tmpdir teardown + CRLF/Cyrillic env artifacts excluded) |
| End-to-end service loop (`tests/ccea/test_e2e.py`) | 15 passed (enroll → build → sign → command → approve → telemetry) |
| Control plane boots on SQLite + serves agent-lifecycle API | ✅ |
| Agent enrolls over HTTP + heartbeats (`cloud_connected=true`) | ✅ |
| Vault unlocked via keychain, **stable across launches** | ✅ (fresh + persisted runs both unlock) |
| Paper broker wired (`broker_connected=true`) | ✅ |
| Graceful desktop shutdown releases Agent/SQLite stores | ✅ |
| Restart restores SimBroker cash/positions from the durable Agent ledger | ✅ |
| Live in **frozen sidecar** (`riven-backend.exe`) and via `/api/ccea/status` | ✅ |
| Guardrails (boundary / allowlist / payload prohibition) | 52 passed; 3 failures are environment artifacts only |

**Environment-only guardrail failures (not boundary violations):** two are design-doc
**SHA mismatches** from Windows CRLF vs the recorded LF hash; one is an ASCII-output
check that trips when the developer's account name is non-ASCII and leaks into a
temp path in the message.

## Closed gaps (full trading loop + enterprise signing)

1. **Real paper RUN end-to-end** — `CCEASupervisor.paper_trade()` drives the real
   Agent OMS: `OrderIntent → PolicyFirewall + HardCaps + RiskChecker → durable
   OrderJournal → broker order → SimBroker fill → mark-to-market PnL`, reflected in
   `orders_today/fills_today/pnl_today`. Endpoint `POST /api/ccea/paper_order`; UI
   "▶ Paper-сделка" button (Home + Pro). Verified in the frozen sidecar:
   `SIM-1`, **PnL +500**.
2. **Live broker wiring** — `connect_live_broker(broker, key, secret, sandbox)`
   stores credentials in the encrypted **LocalVault (Agent zone)**, builds a real
   connector (`AlpacaConnector`/`Binance`/`IB`/`OANDA`) **from the vault**, and
   connects. Endpoint `POST /api/ccea/connect_broker`. Verified: credentials land
   in the vault and the real connector is constructed + connects (paper/live per
   `sandbox`). Default stays paper (SimBroker). Credentials never reach the Cloud.
   The Lite and Pro credential forms now use this same Agent Vault. Supported live
   brokers call `POST /api/ccea/connect_broker`; data-only adapters call
   `POST /api/ccea/store_credentials`. Browser `localStorage`/plaintext `.env` are
   no longer used by the desktop credential flow.
3. **Native GUI launch** — blocked on this machine by **Smart App Control (ON,
   enforced)** + WDAC, which rejects any unsigned/unreputable executable and cannot
   be per-app excepted. This is a Windows security policy, **not** an app defect:
   the SAC-allowed sidecar (`riven-backend.exe`) and the trusted `python.exe` dev
   host both run the identical backend + CCEA. Production fix: ship with a reputable
   OV/EV **code-signing** certificate (the Windows analogue of macOS notarization);
   then SAC permits the native window. Until signed, run via the dev host (below).
4. **Enterprise signing** — real **Ed25519** TUF metadata signing (CCEA-SEC-002
   resolved); `agent_updates`/`evidence_pack`/`registry_mirror` now sign/verify with
   Ed25519 and degrade gracefully in development while staying **fail-closed in
   production** (`CCEA_ENV=production`). All 105 enterprise-signing tests pass.

## Run it

- Desktop: CCEA starts automatically (`RIVEN_ENABLE_CCEA=1` by default). Watch the
  CCEA card on the Home screen, or `GET http://127.0.0.1:<port>/api/ccea/status`.
- Dev (no packaging): `RIVEN_ENABLE_CCEA=1 python desktop_backend.py --port 8002`,
  then open `http://127.0.0.1:8002/` and poll `/api/ccea/status`.
- Disable CCEA: set `RIVEN_ENABLE_CCEA=0`.

## Paper vs live

Default is **paper** (`SimBrokerConnector`). For live trading, store broker
credentials in the local Vault and wire the corresponding broker connector
(`packages/agent/broker/adapters/{alpaca,ib,binance,oanda}.py`); per-run preflight
(signed manifest + live vault + clock sync) then applies. The cloud never sees
credentials or orders.

# GDPR Risk Scope Memo — Data Map and Role Model (CCEA-first)

**Document Type**: Compliance Engineering Specification
**Version**: 1.0
**Last Updated**: 2025-12-16
**Scope**: CCEA Cloud platform (EU deployment target; verify via infrastructure configuration)
**Primary Design Source**: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt`

---

## Executive Summary

This document defines the defensible processing map for the CCEA platform aligned with GDPR requirements and CCEA architectural boundaries. It establishes:

1. **Data Categories** — What personal data the platform processes
2. **Controller vs Processor Roles** — Legal responsibility per data category
3. **Data Flow Diagram** — How data moves between Cloud and Agent zones
4. **Telemetry Sensitivity Levels** — AGGREGATED, DETAILED_NON_SENSITIVE, RAW_ORDER_EVENTS
5. **RoPA-lite Table** — Records of Processing Activities per Art. 30

---

## 1. Platform Zones and Data Boundaries

### 1.1 CCEA Architecture: Two Distinct Zones

The platform operates in two strictly separated zones with different data handling responsibilities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLOUD ZONE (EU-only)                               │
│  Platform Provider Operated — Research, Build, Monitor, Control             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│  │   Auth/Accounts   │  │ Strategy Workspace│  │  Artifact Builder │       │
│  │  - Users          │  │  - IDE/Notebooks  │  │  - Build+Sign     │       │
│  │  - Organizations  │  │  - Strategy Repo  │  │  - Manifest+SBOM  │       │
│  │  - Workspaces     │  │  - Versions       │  │  - Provenance     │       │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘       │
│                                                                              │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│  │   Control Plane   │  │    Telemetry &    │  │    Governance     │       │
│  │  - Commands       │  │    Monitoring     │  │  - Retention      │       │
│  │  - Approvals      │  │  - AGGREGATED     │  │  - RBAC           │       │
│  │  - Lifecycle      │  │  - DETAILED       │  │  - DSAR           │       │
│  │  (NO order-like   │  │  - RAW (ent-only) │  │  - Audit          │       │
│  │   payloads)       │  │                   │  │                   │       │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘       │
│                                                                              │
│  CLOUD DOES NOT RECEIVE: API keys, secrets, env vars, order-like payloads   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Outbound from Agent (HTTPS)
                                    │ - Telemetry (redacted)
                                    │ - Command acks
                                    │ - Heartbeats
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AGENT ZONE (Customer Environment)                        │
│  Customer Operated — Execution, Secrets, Orders, Risk                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│  │    Local Vault    │  │   Strategy Runner │  │  Broker Connector │       │
│  │  - API Keys       │  │  - Live Loop      │  │  - Order Submit   │       │
│  │  - Secrets        │  │  - Intent→Order   │  │  - Fill/Position  │       │
│  │  - Credentials    │  │  - Reconciliation │  │  - Market Data    │       │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘       │
│                                                                              │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│  │   Risk Manager    │  │   Kill Switch     │  │   Local Journal   │       │
│  │  - Pre-trade      │  │  - Circuit Break  │  │  - Audit Trail    │       │
│  │  - Limits Enforce │  │  - Emergency Stop │  │  - Recovery       │       │
│  │  - Policy         │  │  - Flatten        │  │                   │       │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘       │
│                                                                              │
│  AGENT HOLDS: All secrets, all credentials, all order execution logic       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Architectural Invariants (GDPR-by-Design)

| Invariant ID | Rule | GDPR Principle | Reference |
|---|---|---|---|
| **INV-001** | Cloud does not receive broker credentials, API keys, or tokens | Art. 25 (Privacy by Design) | Design Doc §0.2, §5.4 |
| **INV-002** | Cloud→Agent commands do not carry order-like payloads (side/qty/price) | Art. 5(1)(c) (Data Minimization) | Design Doc §10.5, §1697 |
| **INV-003** | Telemetry defaults to AGGREGATED; redaction is mandatory | Art. 5(1)(c), Art. 25 | Design Doc §13.1, §19.4 |
| **INV-004** | RAW_ORDER_EVENTS requires enterprise + explicit opt-in | Art. 7 (Consent), Art. 25 | Design Doc §13.2, §1739 |
| **INV-005** | Data residency is EU by default (transfers outside EU not planned) | Art. 44+ (Out of scope by design) | Design Doc §14.3, §6.2 |
| **INV-006** | Break-glass access is incident-only, time-bound, audited | Art. 32 (Security) | Design Doc §14.4 |
| **INV-007** | Signed artifacts only; unsigned rejected | Art. 32 (Integrity) | Design Doc §15.1, §19.3 |

---

## 2. Controller vs Processor Roles

### 2.1 Role Determination Framework

Under GDPR Article 4:
- **Controller** (Art. 4(7)): Determines purposes and means of processing
- **Processor** (Art. 4(8)): Processes personal data on behalf of the controller

### 2.2 Role Assignment by Data Category

| Data Category | Zone | We Are | Customer Is | Rationale |
|---|---|---|---|---|
| **User Account Data** | Cloud | Controller | Data Subject | We determine how accounts are managed for service delivery |
| **Organization Data** | Cloud | Controller | Data Subject/Controller | Service registration and billing under our terms |
| **Authentication Data** | Cloud | Controller | Data Subject | We define auth mechanisms (password policy, MFA) |
| **Workspace Configuration** | Cloud | Joint Controller | Joint Controller | Customer defines purpose within our platform constraints |
| **Strategy Code/Models** | Cloud | Processor | Controller | Customer IP; we store/build on instruction |
| **Backtest Results** | Cloud | Processor | Controller | Customer research data; we process on instruction |
| **AGGREGATED Telemetry** | Cloud | Processor | Controller | Customer operational data; we process for monitoring |
| **DETAILED_NON_SENSITIVE Telemetry** | Cloud | Processor | Controller | Customer opts in; we process on instruction |
| **RAW_ORDER_EVENTS Telemetry** | Cloud | Processor | Controller | Enterprise-only explicit consent; we process on instruction |
| **Audit Logs (Cloud)** | Cloud | Controller | — | Regulatory compliance requirement (our legal obligation) |
| **Access Audit Logs** | Cloud | Controller | — | Security/compliance (our legal obligation) |
| **DSAR Request Records** | Cloud | Controller | Data Subject | GDPR Art. 12-23 compliance |
| **Broker Credentials** | Agent | — | Controller | Never in Cloud; customer-controlled |
| **Order/Fill Data** | Agent | — | Controller | Never in Cloud (except RAW opt-in); customer-controlled |
| **Local Journal** | Agent | — | Controller | Customer operational data on customer premises |

### 2.3 Detailed Role Analysis

#### 2.3.1 Platform Provider as Controller

We act as **Controller** for:

1. **User Identity Data**
   - **Data**: email, display_name, password_hash, MFA secrets
   - **Purpose**: User authentication, account management, security
   - **Lawful Basis**: Art. 6(1)(b) Contract performance + Art. 6(1)(f) Legitimate interest (security)
   - **Retention**: Account lifetime + 90 days post-deletion for recovery

2. **Organization/Billing Data**
   - **Data**: organization name, billing email, billing tier
   - **Purpose**: Service delivery, invoicing, support
   - **Lawful Basis**: Art. 6(1)(b) Contract performance
   - **Retention**: Account lifetime + 7 years (tax/accounting requirements)

3. **Audit & Compliance Logs**
   - **Data**: access_audits, governance_audit_logs, break_glass_requests
   - **Purpose**: Security monitoring, regulatory compliance, incident response
   - **Lawful Basis**: Art. 6(1)(c) Legal obligation + Art. 6(1)(f) Legitimate interest
   - **Retention**: 7 years (financial services best practice)

#### 2.3.2 Platform Provider as Processor

We act as **Processor** for:

1. **Strategy/Model Assets**
   - **Data**: Strategy code, model weights, configuration
   - **Purpose**: Build, store, and deploy on customer instruction
   - **Lawful Basis**: Art. 6(1)(b) Contract (DPA)
   - **Retention**: Customer-defined (default: indefinite until deletion request)

2. **Telemetry Data**
   - **Data**: Performance metrics, error rates, health status (by sensitivity level)
   - **Purpose**: Monitoring and observability on customer instruction
   - **Lawful Basis**: Art. 6(1)(b) Contract (DPA)
   - **Retention**: Per tenant policy (default: 90 days AGGREGATED, 30 days DETAILED, 7 days RAW)

3. **Research Results**
   - **Data**: Backtest reports, simulation results, training artifacts
   - **Purpose**: Research compute on customer instruction
   - **Lawful Basis**: Art. 6(1)(b) Contract (DPA)
   - **Retention**: Customer-defined

#### 2.3.3 Agent Zone — Customer as Controller

For **Agent Zone data**, the customer is the sole Controller:

- Broker credentials (API keys, secrets)
- Live order/fill data
- Position information
- Local execution logs
- Risk policy configuration

**The platform provider has NO processing role for Agent Zone data** unless:
1. Telemetry is explicitly enabled for Cloud transmission (AGGREGATED by default)
2. RAW_ORDER_EVENTS is explicitly opted-in (enterprise-only)
3. Log export is requested via `REQUEST_EXPORT_LOGS` command

---

## 3. Cloud↔Agent Data Flow Diagram

### 3.1 Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CLOUD ZONE (EU-only SaaS)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐        │
│  │   PostgreSQL │         │ Object Store │         │    Redis     │        │
│  │   (EU-only)  │         │  (EU-only)   │         │  (EU-only)   │        │
│  │              │         │              │         │              │        │
│  │ - users      │         │ - artifacts  │         │ - sessions   │        │
│  │ - orgs       │         │ - models     │         │ - rate limits│        │
│  │ - workspaces │         │ - backtest   │         │ - token cache│        │
│  │ - telemetry  │         │ - sbom       │         │              │        │
│  │ - commands   │         │              │         │              │        │
│  │ - audits     │         │              │         │              │        │
│  └──────────────┘         └──────────────┘         └──────────────┘        │
│         │                        │                        │                 │
│         └────────────────────────┼────────────────────────┘                 │
│                                  │                                          │
│                         ┌────────┴────────┐                                 │
│                         │  Control Plane  │                                 │
│                         │   FastAPI App   │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│              ┌───────────────────┼───────────────────┐                      │
│              ▼                   ▼                   ▼                      │
│     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │
│     │ Command Queue  │  │   Telemetry    │  │   Governance   │             │
│     │                │  │   Ingestion    │  │    Service     │             │
│     │ REQUEST_START  │  │                │  │                │             │
│     │ REQUEST_STOP   │  │ Validates:     │  │ - Retention    │             │
│     │ REQUEST_PAUSE  │  │ - Redaction    │  │ - DSAR         │             │
│     │ REQUEST_UPGRADE│  │ - Level        │  │ - Legal hold   │             │
│     │ REQUEST_CONFIG │  │ - Schema       │  │ - Access audit │             │
│     │                │  │                │  │                │             │
│     │ [NO order-like │  │ [Rejects non-  │  │ [EU-only       │             │
│     │  payloads]     │  │  redacted]     │  │  residency]    │             │
│     └───────┬────────┘  └───────┬────────┘  └────────────────┘             │
│             │                   │                                           │
└─────────────┼───────────────────┼───────────────────────────────────────────┘
              │                   │
              │ HTTPS             │ HTTPS
              │ (outbound from    │ (outbound from
              │  Agent polling)   │  Agent push)
              │                   │
              │ Commands:         │ Telemetry:
              │ - command_type    │ - AGGREGATED (default)
              │ - payload_ref     │ - DETAILED_NON_SENSITIVE (opt-in)
              │   (digest only)   │ - RAW_ORDER_EVENTS (enterprise)
              │ - change_class    │
              │                   │ [MANDATORY REDACTION]
              ▼                   │ [SECRETS STRIPPED]
┌─────────────┼───────────────────┼───────────────────────────────────────────┐
│             │                   │                                            │
│     ┌───────┴────────┐  ┌───────┴────────┐                                  │
│     │  Cloud Client  │  │ Telemetry      │                                  │
│     │                │  │ Buffer         │                                  │
│     │ - Poll commands│  │                │                                  │
│     │ - Send acks    │  │ ┌────────────┐ │                                  │
│     │ - Report state │  │ │ Redaction  │ │ ◄── ON by default (no disable flag) │
│     │                │  │ │ Middleware │ │                                  │
│     └───────┬────────┘  │ └────────────┘ │                                  │
│             │           │ ┌────────────┐ │                                  │
│             │           │ │ DLP Filter │ │                                  │
│             │           │ └────────────┘ │                                  │
│             │           │ ┌────────────┐ │                                  │
│             │           │ │Level Check │ │                                  │
│             │           │ └────────────┘ │                                  │
│             │           └───────┬────────┘                                  │
│             │                   │                                            │
│     ┌───────┴───────────────────┴───────┐                                   │
│     │           Agent Daemon             │                                   │
│     │                                    │                                   │
│     │  ┌──────────────┐ ┌──────────────┐│                                   │
│     │  │ Approval     │ │ Strategy     ││                                   │
│     │  │ Workflow     │ │ Runner       ││                                   │
│     │  │              │ │              ││                                   │
│     │  │ Local approve│ │ Live Loop    ││                                   │
│     │  │ required for │ │ Intent→Order ││                                   │
│     │  │ TRADING_     │ │              ││                                   │
│     │  │ IMPACTING    │ │              ││                                   │
│     │  └──────────────┘ └──────┬───────┘│                                   │
│     │                          │        │                                   │
│     │  ┌──────────────┐ ┌──────┴───────┐│                                   │
│     │  │ Local Vault  │ │ Risk Manager ││                                   │
│     │  │              │ │              ││                                   │
│     │  │ [SECRETS]    │ │ Pre-trade    ││                                   │
│     │  │ - API keys   │ │ controls,    ││                                   │
│     │  │ - Tokens     │ │ limits,      ││                                   │
│     │  │ - Passwords  │ │ kill switch  ││                                   │
│     │  │              │ │              ││                                   │
│     │  │ [Designed to │ └──────┬───────┘│                                   │
│     │  │  stay local] │        │        │                                   │
│     │  └──────────────┘        │        │                                   │
│     └──────────────────────────┼────────┘                                   │
│                                │                                            │
│                        ┌───────┴───────┐                                    │
│                        │    Broker     │                                    │
│                        │   Connector   │                                    │
│                        │               │                                    │
│                        │ [ONLY HERE    │                                    │
│                        │  ORDERS SENT] │                                    │
│                        └───────┬───────┘                                    │
│                                │                                            │
│              AGENT ZONE (Customer Environment)                              │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Broker / Exchange     │
                    │   (External)            │
                    │                         │
                    │   [Orders submitted     │
                    │    from Agent ONLY]     │
                    └─────────────────────────┘
```

### 3.2 Telemetry Sensitivity Levels

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TELEMETRY SENSITIVITY LEVELS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LEVEL 1: AGGREGATED (Default for all tiers)                                │
│  ────────────────────────────────────────────                               │
│  • PnL (daily/cumulative)                                                   │
│  • Drawdown percentage                                                       │
│  • Exposure (USD aggregated)                                                │
│  • Error rates (count)                                                       │
│  • Orders per minute (rate)                                                 │
│  • Health status                                                            │
│  • Latency percentiles                                                      │
│                                                                              │
│  [NO individual order/fill data]                                            │
│  [NO account identifiers]                                                   │
│  [NO instrument-level breakdown]                                            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LEVEL 2: DETAILED_NON_SENSITIVE (Opt-in for debugging)                     │
│  ───────────────────────────────────────────────────────                    │
│  • All AGGREGATED data plus:                                                │
│  • Timestamps (execution latency)                                           │
│  • Strategy state transitions                                               │
│  • Signal generation metrics                                                │
│  • Queue depths                                                             │
│  • Memory/CPU usage                                                         │
│                                                                              │
│  [NO individual order/fill data]                                            │
│  [NO price/quantity/side information]                                       │
│  [NO account identifiers]                                                   │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LEVEL 3: RAW_ORDER_EVENTS (Enterprise-only, explicit opt-in)               │
│  ────────────────────────────────────────────────────────────               │
│  • All DETAILED_NON_SENSITIVE data plus:                                    │
│  • Order events (with MASKED account identifiers)                           │
│  • Fill events (with MASKED account identifiers)                            │
│  • Position changes                                                         │
│  • Instrument-level metrics                                                 │
│                                                                              │
│  REQUIRED CONTROLS:                                                         │
│  ├── Enterprise tier verified                                               │
│  ├── Explicit per-workspace opt-in (audited)                                │
│  ├── Consent record (who/what/when/scope/expiry)                            │
│  ├── Minimal retention (7 days default, configurable)                       │
│  ├── Restricted access (RBAC + break-glass)                                 │
│  ├── Audit trail for all access                                             │
│  └── Alternative: "telemetry stays local" mode                              │
│                                                                              │
│  [NEVER contains: API keys, secrets, unmasked account IDs]                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

                    TELEMETRY LEVEL GATING LOGIC

    ┌─────────────────────────────────────────────────────────┐
    │                    Telemetry Event                       │
    │                    Generated by Agent                    │
    └───────────────────────────┬─────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │             MANDATORY REDACTION MIDDLEWARE               │
    │             (Cannot be disabled)                         │
    │                                                          │
    │  Removes: secrets, API keys, env vars, passwords         │
    │  Masks: account identifiers, sensitive patterns          │
    └───────────────────────────┬─────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │               DLP (Data Loss Prevention)                 │
    │                                                          │
    │  CRITICAL sensitivity → BLOCK (always)                   │
    │  SENSITIVE → MASK                                        │
    │  CONFIDENTIAL → MASK                                     │
    │  INTERNAL → ALLOW with logging                           │
    │  PUBLIC → ALLOW                                          │
    └───────────────────────────┬─────────────────────────────┘
                                │
                                ▼
                   ┌────────────┴────────────┐
                   │  What is the configured │
                   │  telemetry level?       │
                   └────────────┬────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
    ┌───────────┐        ┌───────────┐        ┌───────────────┐
    │ AGGREGATED│        │ DETAILED_ │        │ RAW_ORDER_    │
    │ (default) │        │ NON_      │        │ EVENTS        │
    │           │        │ SENSITIVE │        │               │
    │ Always    │        │ Opt-in    │        │ Enterprise    │
    │ allowed   │        │ allowed   │        │ only check    │
    └─────┬─────┘        └─────┬─────┘        └───────┬───────┘
          │                    │                      │
          │                    │              ┌───────┴───────┐
          │                    │              │ Is enterprise │
          │                    │              │ + explicit    │
          │                    │              │ opt-in?       │
          │                    │              └───────┬───────┘
          │                    │                 NO   │   YES
          │                    │              ┌──────┴───────┐
          │                    │              ▼              ▼
          │                    │         ┌────────┐    ┌──────────┐
          │                    │         │ REJECT │    │ Allow    │
          │                    │         │ event  │    │ with     │
          │                    │         └────────┘    │ audit    │
          │                    │                       └──────────┘
          │                    │                            │
          └────────────────────┼────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Transmit to Cloud   │
                    │  (HTTPS, EU-only)    │
                    └──────────────────────┘
```

### 3.3 Command Flow (Cloud→Agent)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMMAND FLOW (Cloud→Agent)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ALLOWED COMMAND TYPES (Exhaustive List)                                    │
│  ───────────────────────────────────────                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Command Type            │ Direction    │ Approval Required │ Purpose│    │
│  ├─────────────────────────┼──────────────┼───────────────────┼────────│    │
│  │ REQUEST_START_RUN       │ Cloud→Agent  │ Yes (TRADING)     │ Start  │    │
│  │ REQUEST_STOP_RUN        │ Cloud→Agent  │ No (safety)       │ Stop   │    │
│  │ REQUEST_PAUSE_RUN       │ Cloud→Agent  │ No (safety)       │ Pause  │    │
│  │ REQUEST_UPGRADE_ARTIFACT│ Cloud→Agent  │ Yes (TRADING)     │ Upgrade│    │
│  │ REQUEST_UPDATE_CONFIG   │ Cloud→Agent  │ Conditional       │ Config │    │
│  │ REQUEST_ROTATE_SESSION  │ Cloud→Agent  │ Yes (security)    │ Rotate │    │
│  │ REQUEST_EXPORT_LOGS     │ Cloud→Agent  │ Yes (data)        │ Export │    │
│  │ HEARTBEAT               │ Agent→Cloud  │ No                │ Health │    │
│  │ TELEMETRY               │ Agent→Cloud  │ No                │ Monitor│    │
│  │ COMMAND_ACK             │ Agent→Cloud  │ No                │ Confirm│    │
│  │ COMMAND_APPROVAL        │ Agent→Cloud  │ No                │ Approve│    │
│  │ COMMAND_RESULT          │ Agent→Cloud  │ No                │ Result │    │
│  └─────────────────────────┴──────────────┴───────────────────┴────────┘    │
│                                                                              │
│  FORBIDDEN COMMAND TYPES / PAYLOADS (Schema + CI Enforced)                  │
│  ─────────────────────────────────────────────────────────                  │
│                                                                              │
│  ❌ PLACE_ORDER                                                             │
│  ❌ SUBMIT_ORDER                                                            │
│  ❌ EXECUTE_SIGNAL                                                          │
│  ❌ SET_TARGET_POSITION_NOW                                                 │
│  ❌ Any payload with: side, quantity, price, order_type, target_position    │
│  ❌ Any payload with: broker credentials, API keys, secrets, env vars       │
│                                                                              │
│  ENFORCEMENT:                                                               │
│  • JSON Schema validation (build-time)                                      │
│  • CI guardrails (fail closed)                                              │
│  • Runtime validation (agent rejects unknown command types)                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Records of Processing Activities (RoPA-lite) — Art. 30

### 4.1 RoPA Table — Cloud Systems

| System | Data Category | Purpose | Lawful Basis | Retention | Residency | Access Roles | Subprocessors | Owner |
|---|---|---|---|---|---|---|---|---|
| **PostgreSQL (users)** | User Identity (email, display_name) | Account management, authentication | Art. 6(1)(b) Contract | Account lifetime + 90 days | EU (eu-central-1) | admin, support (read), user (self) | AWS RDS (EU), Supabase (EU) | Platform Auth Team |
| **PostgreSQL (users)** | Authentication (password_hash, MFA secrets) | Secure login, account protection | Art. 6(1)(b) Contract, Art. 6(1)(f) Security | Account lifetime | EU (eu-central-1) | admin (hash only, no plaintext) | AWS RDS (EU), Supabase (EU) | Platform Auth Team |
| **PostgreSQL (organizations)** | Org Identity (name, billing_email) | Service delivery, invoicing | Art. 6(1)(b) Contract | Account lifetime + 7 years | EU (eu-central-1) | admin, billing, support | AWS RDS (EU), Supabase (EU) | Platform Accounts Team |
| **PostgreSQL (workspaces)** | Workspace Config | Tenant isolation, service config | Art. 6(1)(b) Contract | Account lifetime | EU (eu-central-1) | admin, workspace members | AWS RDS (EU), Supabase (EU) | Platform Workspaces Team |
| **PostgreSQL (agents)** | Agent Identity (public_key, device_id) | Agent registration, secure comms | Art. 6(1)(b) Contract, Art. 6(1)(f) Security | Agent lifetime + 90 days | EU (eu-central-1) | admin, agent owner | AWS RDS (EU), Supabase (EU) | Platform Agents Team |
| **PostgreSQL (strategies, strategy_versions)** | Strategy Metadata | Version control, deployment | Art. 6(1)(b) Contract (Processor) | Customer-defined (default: indefinite) | EU (eu-central-1) | workspace members | AWS RDS (EU), Supabase (EU) | Platform Strategies Team |
| **PostgreSQL (builds, artifacts)** | Build/Artifact Metadata | Artifact integrity, provenance | Art. 6(1)(b) Contract (Processor) | Customer-defined (default: indefinite) | EU (eu-central-1) | workspace members, build service | AWS RDS (EU), Supabase (EU) | Platform Build Team |
| **PostgreSQL (commands)** | Command Records | Lifecycle control audit | Art. 6(1)(b) Contract, Art. 6(1)(c) Legal | 7 years | EU (eu-central-1) | admin, workspace members (own) | AWS RDS (EU), Supabase (EU) | Platform Control Plane Team |
| **PostgreSQL (approval_records)** | Approval Evidence | Audit trail for TRADING_IMPACTING | Art. 6(1)(c) Legal, Art. 6(1)(f) Legitimate interest | 7 years | EU (eu-central-1) | admin (audit), workspace members (own) | AWS RDS (EU), Supabase (EU) | Platform Governance Team |
| **PostgreSQL (telemetry_events)** | Telemetry — AGGREGATED | Operational monitoring | Art. 6(1)(b) Contract (Processor) | 90 days (configurable per tenant) | EU (eu-central-1) | workspace members, support (with consent) | AWS RDS (EU), Supabase (EU) | Platform Telemetry Team |
| **PostgreSQL (telemetry_events)** | Telemetry — DETAILED_NON_SENSITIVE | Debugging, performance analysis | Art. 6(1)(b) Contract (Processor), Art. 6(1)(a) Consent (opt-in) | 30 days (configurable) | EU (eu-central-1) | workspace members (authorized) | AWS RDS (EU), Supabase (EU) | Platform Telemetry Team |
| **PostgreSQL (telemetry_events)** | Telemetry — RAW_ORDER_EVENTS | Enterprise advanced monitoring | Art. 6(1)(a) Explicit Consent + Art. 6(1)(b) Contract | 7 days (configurable, max 30) | EU (eu-central-1) | workspace admins only, break-glass support | AWS RDS (EU), Supabase (EU) | Platform Telemetry Team |
| **PostgreSQL (access_audits)** | Access Audit Logs | Security monitoring, compliance | Art. 6(1)(c) Legal, Art. 6(1)(f) Legitimate interest | 7 years | EU (eu-central-1) | admin (audit), security team | AWS RDS (EU), Supabase (EU) | Platform Security Team |
| **PostgreSQL (break_glass_requests)** | Break-glass Audit | Incident-only access accountability | Art. 6(1)(c) Legal, Art. 6(1)(f) Security | 7 years | EU (eu-central-1) | admin (audit), security team | AWS RDS (EU), Supabase (EU) | Platform Security Team |
| **PostgreSQL (dsar_requests)** | DSAR Records | GDPR Art. 12-23 compliance | Art. 6(1)(c) Legal obligation | 7 years | EU (eu-central-1) | admin, data protection officer | AWS RDS (EU), Supabase (EU) | Platform DPO/Compliance |
| **PostgreSQL (data_retention_policies)** | Retention Policy Config | Data lifecycle management | Art. 6(1)(c) Legal, Art. 6(1)(f) Legitimate interest | Account lifetime | EU (eu-central-1) | admin, workspace admins | AWS RDS (EU), Supabase (EU) | Platform Governance Team |
| **PostgreSQL (legal_holds)** | Legal Hold Records | Litigation preservation | Art. 6(1)(c) Legal obligation | Duration of hold + 7 years | EU (eu-central-1) | admin, legal | AWS RDS (EU), Supabase (EU) | Platform Legal/Compliance |
| **PostgreSQL (governance_audit_logs)** | Governance Audit Trail | Data lifecycle audit | Art. 6(1)(c) Legal, Art. 6(1)(f) Legitimate interest | 7 years | EU (eu-central-1) | admin (audit) | AWS RDS (EU), Supabase (EU) | Platform Governance Team |
| **Object Store (artifacts)** | Strategy Artifacts (code, models) | Secure storage, deployment | Art. 6(1)(b) Contract (Processor) | Customer-defined | EU (eu-central-1) | workspace members, build service | AWS S3 (EU), MinIO (EU) | Platform Storage Team |
| **Object Store (sbom)** | SBOM Files | Supply chain security, compliance | Art. 6(1)(c) Legal, Art. 6(1)(f) Security | Artifact lifetime | EU (eu-central-1) | admin, security team | AWS S3 (EU), MinIO (EU) | Platform Security Team |
| **Object Store (backtest)** | Backtest Results | Research storage (Processor) | Art. 6(1)(b) Contract (Processor) | Customer-defined | EU (eu-central-1) | workspace members | AWS S3 (EU), MinIO (EU) | Platform Research Team |
| **Redis (sessions)** | Session Tokens | Active session management | Art. 6(1)(b) Contract, Art. 6(1)(f) Security | 24 hours (configurable) | EU (eu-central-1) | system only (no human access) | AWS ElastiCache (EU) | Platform Auth Team |
| **Redis (rate_limits)** | Rate Limit Counters | Abuse prevention, fair use | Art. 6(1)(f) Legitimate interest | 1 hour rolling window | EU (eu-central-1) | system only | AWS ElastiCache (EU) | Platform Security Team |
| **Observability (logs)** | Application Logs (redacted) | Debugging, incident response | Art. 6(1)(f) Legitimate interest | 30 days | EU (eu-central-1) | admin, devops, support (restricted) | AWS CloudWatch (EU) | Platform DevOps Team |
| **Observability (metrics)** | System Metrics | Performance monitoring | Art. 6(1)(f) Legitimate interest | 90 days | EU (eu-central-1) | admin, devops | AWS CloudWatch (EU), Prometheus (EU) | Platform DevOps Team |
| **Observability (traces)** | Distributed Traces (redacted) | Performance analysis | Art. 6(1)(f) Legitimate interest | 7 days | EU (eu-central-1) | admin, devops | AWS X-Ray (EU), Jaeger (EU) | Platform DevOps Team |

### 4.2 RoPA Table — Agent Zone (Customer-Controlled, Not Processed by Platform)

| System | Data Category | Controller | Processor | Notes |
|---|---|---|---|---|
| **Local Vault** | Broker Credentials (API keys, secrets) | Customer | None (local only) | Designed to remain local (not transmitted to Cloud) |
| **Local Vault** | OAuth Tokens | Customer | None (local only) | Designed to remain local (not transmitted to Cloud) |
| **Strategy Runner** | Live Intent/Signal Data | Customer | None (local only) | Generated and consumed locally |
| **Broker Connector** | Order/Fill Data | Customer | None (local only) | Transmitted to broker only, not Cloud (unless RAW opt-in) |
| **Risk Manager** | Position Data | Customer | None (local only) | Local enforcement |
| **Local Journal** | Execution Logs | Customer | None (local only) | Customer audit trail |
| **Reconciler** | State Sync Data | Customer | None (local only) | Recovery/idempotency |

---

## 5. Data Subject Categories

### 5.1 Categories and Rights

| Data Subject Category | Data Types | GDPR Rights Applicable | DSAR Scope |
|---|---|---|---|
| **Platform Users** | Identity, authentication, preferences | Art. 15-22 (all rights) | Full Cloud data |
| **Organization Admins** | Identity, billing, org management | Art. 15-22 (all rights) | Full Cloud data |
| **Workspace Members** | Identity, workspace access, activity | Art. 15-22 (all rights) | Full Cloud data |
| **Agent Operators** | Device identity (public key) | Art. 15-22 | Cloud-side agent records only |
| **Support Contacts** | Name, email (if provided) | Art. 15-22 | Support records only |

### 5.2 DSAR Boundaries (CCEA-specific)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DSAR SCOPE BOUNDARIES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  IN SCOPE (Cloud-controlled, we can export/delete):                         │
│  ─────────────────────────────────────────────────                          │
│  • User account data (email, display_name, preferences)                     │
│  • Organization membership records                                          │
│  • Workspace membership and roles                                           │
│  • Strategy metadata (owned by user/workspace)                              │
│  • Telemetry data (at enabled sensitivity level)                            │
│  • Command history and approval records                                     │
│  • Access audit logs (where user is subject)                                │
│  • Support interaction records                                              │
│                                                                              │
│  OUT OF SCOPE (Agent-controlled, customer responsibility):                  │
│  ────────────────────────────────────────────────────                       │
│  • Broker credentials (never in Cloud)                                      │
│  • Local execution logs (unless exported via REQUEST_EXPORT_LOGS)           │
│  • Order/fill data (unless RAW_ORDER_EVENTS enabled and transmitted)        │
│  • Local vault contents                                                     │
│  • Position data (unless transmitted via telemetry)                         │
│                                                                              │
│  RESPONSE TEMPLATE:                                                         │
│  "Your request has been processed for all personal data held in our         │
│   Cloud systems. Data stored in your local Agent environment (including     │
│   broker credentials, local logs, and order data) is under your control     │
│   and not accessible to us. Please contact your system administrator        │
│   for access to Agent-local data."                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Subprocessor Register

### 6.1 Approved Subprocessors (EU-only)

| Subprocessor | Service | Data Processed | Region | DPA Status (Internal Record) | Last Review |
|---|---|---|---|---|---|
| **AWS (Amazon Web Services)** | Infrastructure (RDS, S3, ElastiCache, CloudWatch) | All Cloud data | eu-central-1 (Frankfurt), eu-west-1 (Ireland) | Planned (AWS DPA) — verify actual signature and effective date via internal contract register | 2025-01-15 |
| **Supabase** | Database hosting (PostgreSQL alternative) | All Cloud data | EU (Germany) | Planned — verify actual signature and effective date via internal contract register | 2025-01-15 |
| **Stripe** | Payment processing | Billing data only | EU (Ireland) | Planned (Stripe DPA) — verify actual signature and effective date via internal contract register | 2025-01-15 |
| **SendGrid / AWS SES** | Transactional email | Email addresses, notification content | EU | Planned — verify actual signature and effective date via internal contract register | 2025-01-15 |
| **Sentry** | Error monitoring | Error logs (redacted, no PII in stack traces) | EU (Germany) | Planned — verify actual signature and effective date via internal contract register | 2025-01-15 |

> **Note**: DPA status reflects internal records at time of last review. Actual contract status and effective dates should be verified via the internal contract register (`docs/contracts/SUBCONTRACTOR_REGISTER.md`).

### 6.2 Subprocessor Change Notification

- **Notification period**: 30 days prior to new subprocessor engagement
- **Method**: Email to billing contact + in-app notification
- **Objection process**: Customer may object within 30 days; if unresolved, termination right

---

## 7. Risk Assessment Summary

### 7.1 Privacy Risk Matrix

| Risk | Likelihood | Impact | Mitigation | Residual Risk |
|---|---|---|---|---|
| **Unauthorized access to telemetry** | Low | High | RBAC, audit logging, break-glass | Low |
| **Secrets leak via telemetry** | Very Low | Critical | Mandatory redaction (cannot disable), DLP | Very Low |
| **Cross-tenant data access** | Very Low | Critical | RLS enforcement, workspace isolation | Very Low |
| **Order data leak (non-RAW)** | Very Low | High | Schema enforcement, CI guardrails | Very Low |
| **RAW telemetry misuse** | Low | High | Enterprise-only, explicit opt-in, minimal retention | Low |
| **EU data residency breach** | Very Low | Critical | EU-only deployment, drift checks, subprocessor audit | Very Low |
| **DSAR processing delay** | Medium | Medium | Automated workflows, deadline tracking | Low |
| **Break-glass abuse** | Low | High | Incident-only, reason required, audit trail | Low |

### 7.2 DPIA Requirement Assessment

Per GDPR Article 35, a DPIA may be required if processing involves:
- Systematic monitoring of individuals (✓ telemetry monitoring)
- Processing on a large scale (depends on customer count)
- Sensitive data (financial trading data is commercially sensitive)

**Recommendation**: Conduct DPIA for RAW_ORDER_EVENTS processing (enterprise tier).

---

## 8. Compliance Checklist (Phase 0 DoD)

### 8.1 RoPA-lite Completeness

| Requirement | Status | Evidence |
|---|---|---|
| RoPA table exists with all required columns | ✅ | Section 4.1 |
| Columns: system, data category, purpose | ✅ | Section 4.1 |
| Columns: lawful basis, retention | ✅ | Section 4.1 |
| Columns: residency, access roles, subprocessors | ✅ | Section 4.1 |
| Every data store has: owner | ✅ | Section 4.1 (Owner column) |
| Every data store has: retention | ✅ | Section 4.1 (Retention column) |
| Every data store has: lawful basis | ✅ | Section 4.1 (Lawful Basis column) |
| Every data store has: residency=EU | ✅ | Section 4.1 (all EU residency) |
| No blanks in required fields | ✅ | Verified |

### 8.2 Data Flow Diagram Completeness

| Requirement | Status | Evidence |
|---|---|---|
| Cloud↔Agent data flow exists | ✅ | Section 3.1 |
| Labels Cloud telemetry levels | ✅ | Section 3.2 |
| Documents AGGREGATED level | ✅ | Section 3.2 |
| Documents DETAILED_NON_SENSITIVE level | ✅ | Section 3.2 |
| Documents RAW_ORDER_EVENTS level | ✅ | Section 3.2 |
| Documents RAW gating (enterprise-only) | ✅ | Section 3.2 |
| Documents explicit opt-in requirement | ✅ | Section 3.2 |
| Documents "telemetry stays local" option | ✅ | Section 3.2 (enterprise mode) |

### 8.3 Controller/Processor Determination

| Requirement | Status | Evidence |
|---|---|---|
| Controller vs Processor roles documented | ✅ | Section 2.2, 2.3 |
| Role per data category | ✅ | Section 2.2 (table) |
| Assumptions documented | ✅ | Section 2.1, 2.3 |
| Ready for legal review | ✅ | Section 2 complete |

---

## 9. References

### 9.1 Internal Documents
- `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt` — Primary design source
- `docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md` — Implementation roadmap
- `docs/legal/PRIVACY_POLICY.md` — Public privacy policy
- `docs/legal/DPA_TEMPLATE.md` — Data Processing Agreement template

### 9.2 External References
- **GDPR Regulation (EU) 2016/679** — Articles 4, 5, 6, 7, 12-23, 25, 28, 30, 32, 33-34, 44+
- **EDPB Guidelines** — Transparency, data subject rights, breach notification
- **ISO/IEC 27001:2022** — Information security controls
- **NIST Cybersecurity Framework** — Security baseline reference

---

## 10. Document Control

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0 | 2025-12-16 | Platform Compliance Team | Initial release — Phase 0 deliverable |

---

## Appendix A: Data Category Definitions

| Category ID | Name | Definition | Examples |
|---|---|---|---|
| **DC-001** | User Identity | Information that identifies a natural person | email, display_name |
| **DC-002** | Authentication | Credentials and authentication factors | password_hash, MFA secrets |
| **DC-003** | Organization Identity | Business entity identification | org name, billing email |
| **DC-004** | Workspace Config | Tenant configuration settings | region, tier, policies |
| **DC-005** | Agent Identity | Device/agent identification | public_key, device_id |
| **DC-006** | Strategy Assets | Customer intellectual property | code, models, configs |
| **DC-007** | Telemetry — AGGREGATED | Aggregated operational metrics | PnL, drawdown, error rates |
| **DC-008** | Telemetry — DETAILED | Technical debugging data | latency, timestamps |
| **DC-009** | Telemetry — RAW | Order/fill level events | order events (masked) |
| **DC-010** | Audit Logs | Security and compliance records | access events, approvals |
| **DC-011** | DSAR Records | Data subject request tracking | request type, status |
| **DC-012** | Session Data | Active session management | JWT tokens, session state |

---

## Appendix B: Lawful Basis Reference

| Basis Code | GDPR Article | Description | Use Cases |
|---|---|---|---|
| **LB-CONTRACT** | Art. 6(1)(b) | Performance of contract | Account management, service delivery |
| **LB-CONSENT** | Art. 6(1)(a) | Data subject consent | RAW telemetry, marketing |
| **LB-LEGAL** | Art. 6(1)(c) | Legal obligation | Audit logs, DSAR compliance |
| **LB-LEGITIMATE** | Art. 6(1)(f) | Legitimate interest | Security, fraud prevention |
| **LB-PROCESSOR** | Art. 28 | Processing on controller instruction | Strategy storage, telemetry |

---

## Appendix C: Retention Period Reference

| Retention Code | Duration | Rationale | Data Types |
|---|---|---|---|
| **RET-SESSION** | 24 hours | Active session management | Session tokens |
| **RET-SHORT** | 7 days | Minimal retention for RAW data | RAW_ORDER_EVENTS |
| **RET-MEDIUM** | 30 days | Debugging window | DETAILED telemetry, logs |
| **RET-STANDARD** | 90 days | Operational monitoring | AGGREGATED telemetry |
| **RET-ACCOUNT** | Account lifetime + 90 days | User data with recovery window | User identity |
| **RET-COMPLIANCE** | 7 years | Financial/legal compliance | Audit logs, approvals, billing |
| **RET-CUSTOMER** | Customer-defined | Per DPA terms | Strategy assets, research |

# CCEA Overview: Cloud-Controlled Execution Architecture

> **Version**: 2.1.0 | **Last Updated**: 2025-12-16 | **Status**: Design Specification | **Implementation**: Verify via test suite
>
> **Canon (wording/positioning):** `docs/DOCUMENTATION_CANON_DESIGN.md`
>
> **CCEA technical reference:** `archive/root_files/Design Doc CCEA Cloud.txt` (design/architecture docs are read-only)
>
> **Note**: This document describes the CCEA design and architecture. Implementation status is verified via the test suite (`pytest tests/ccea/`). This is a design specification, not a claim of certified compliance.

## Executive Summary

CCEA (Cloud-Controlled Execution Architecture) defines a strict security boundary between **Cloud** (research/monitoring/lifecycle management) and **Agent** (execution/secrets/risk enforcement). This architecture is designed to ensure that:

- **By design, Cloud does not store trading credentials and does not generate or transmit live trading instructions (orders/targets/signals)**
- **Live execution is designed to occur only in the customer-controlled Agent environment**
- **Clear regulatory positioning as Software Provider, not Investment Adviser**

---

## 1. Context and Goals (Design Doc §1)

### 1.1 Context

We build a SaaS platform for:
- Strategy development (including AI/RL)
- Backtesting and realistic execution simulation
- Strategy/model version management
- Live run monitoring

**Key Point**: Live order execution does NOT happen in Cloud. It happens in user's environment:
- Local Agent on user's machine / user's VPS (BYO host)
- Or on-prem / customer VPC (enterprise)

### 1.2 Goals

Architecture that:
1. **Technically** supports automated execution workflows without Cloud becoming an execution service
2. **Legally/commercially** positions as "software provider", reducing RTO/execution/advice qualification risks
3. **Designed for enterprise adoption**: auditability, change control, data governance, vendor pack

### 1.3 Short Formula

```
Cloud: research, simulation, artifact build/sign, monitoring, lifecycle-requests
Agent: keys, live decision loop, risk controls, order creation/sending, local approvals
```

---

## 2. Terminology (Design Doc §2)

| Term | Definition |
|------|------------|
| **Cloud** | Our SaaS services |
| **Agent** | Client runtime (daemon) in user's environment |
| **Strategy** | User code/model that produces Intent |
| **Intent** | High-level intention (target exposure/position/action), NOT a "ready order" |
| **Order** | Concrete broker instruction (created ONLY in Agent) |
| **Deployment** | Link: strategy artifact + configuration + target agent |
| **Run** | Specific strategy execution on an agent |
| **Command** | Lifecycle request from Cloud to Agent |
| **Approval** | Local confirmation of trading-significant change on Agent |
| **TRADING_IMPACTING** | Change class affecting trading behavior |
| **NON_IMPACTING** | Change NOT affecting trading behavior (logging, UI, telemetry verbosity) |

---

## 3. Architecture Boundary (Design Doc §0, §4)

### 3.1 The Fundamental Principle (Non-Negotiable)

```
Cloud = research / build / monitoring / control plane (lifecycle requests)
Agent = secrets + live loop + risk enforce + order creation/sending

Cloud NEVER:
  - Stores broker API keys
  - Generates or transmits orders
  - Has access to exchange trading endpoints
  - Sends order-like payloads (side/qty/price)
```

### 3.2 Visual Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLOUD ZONE                                      │
│                         (Our Infrastructure)                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Research   │  │   Artifact   │  │   Control    │  │   Telemetry     │  │
│  │     IDE      │  │   Builder    │  │    Plane     │  │  (redacted)     │  │
│  │  Backtesting │  │  (signed)    │  │ (lifecycle)  │  │  Monitoring     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  │
│                                                                              │
│  Security: No trading libs, No broker APIs, No order payloads               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Lifecycle Commands Only:
                                    │ - REQUEST_START_RUN
                                    │ - REQUEST_STOP_RUN
                                    │ - REQUEST_PAUSE_RUN
                                    │ - REQUEST_UPGRADE_ARTIFACT
                                    │ - REQUEST_UPDATE_CONFIG
                                    │ (NO side/qty/price/order_type)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENT ZONE                                      │
│                      (User's Local Machine / VPC)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Local      │  │   Policy     │  │  Live Loop   │  │    Broker       │  │
│  │   Vault      │  │  Firewall    │  │   Runner     │  │   Connector     │  │
│  │ (keychain)   │  │ (hard caps)  │  │ Intent→Order │  │  (execution)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  │
│                                                                              │
│  Security: Secrets local, Hard caps enforced, Orders created locally        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Orders (created & sent locally)
                                    ▼
                              ┌───────────┐
                              │ EXCHANGE  │
                              └───────────┘
```

### 3.3 Zone Responsibilities

| Zone | What It Does | What It NEVER Does |
|------|--------------|-------------------|
| **Cloud** | Research IDE, Backtesting, Artifact build/sign, Monitoring dashboards, Lifecycle management | Store secrets, Generate orders, Access trading APIs, Send order payloads |
| **Agent** | Store secrets (local vault), Enforce risk (policy firewall, hard caps), Run live loop, Create and send orders | Run without user consent, Bypass local limits, Share secrets with Cloud |
| **Shared** | Core models, Simulation engine, Feature engineering, Training | Execute live orders, Store production secrets |

---

## 4. Deployment Modes (CCEA)

### 4.1 B2B Research Cloud + Optional BYO Agent

**Target**: Professional teams evaluating research/simulation and deployment workflows (equities-first)

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLOUD (Our SaaS)                             │
│  - Strategy development IDE                                      │
│  - Backtesting on historical data                               │
│  - Paper trading simulation                                      │
│  - Performance analytics                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Optional: Deploy to Local Agent
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         OPTIONAL: CustodiaCloud Agent (Customer Environment)     │
│  - Only if customer wants live execution                         │
│  - Customer provides and controls broker credentials             │
│  - Agent runs on customer-controlled infrastructure              │
└─────────────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- Cloud provides research/simulation/monitoring only
- Agent is optional, only needed for live execution
- All secrets stay with the customer
- EU data residency by default
- GDPR-aligned telemetry (mandatory redaction)

### 4.2 Customer-Controlled Live Execution via Agent (B2B)

**Target**: Professional teams who want customer-controlled automated execution

```
Cloud:                          Agent (User's Machine):
┌─────────────────┐            ┌─────────────────────────────────┐
│ Control Plane   │────────────│ Local Vault (secrets)           │
│ (lifecycle)     │            │ Policy Firewall (hard caps)     │
│                 │            │ Live Loop (Intent → Risk → Order)│
│ Artifact        │            │ Broker Connector                │
│ Registry        │            │                                 │
│ (signed builds) │            │ Orders sent directly to exchange│
│                 │            │                                 │
│ Telemetry       │◀───────────│ Redacted telemetry              │
│ Dashboards      │            │                                 │
└─────────────────┘            └─────────────────────────────────┘
```

**Key Characteristics:**
- Execution runs locally in the customer-controlled environment
- Cloud provides observability and lifecycle requests only
- Agent enforces hard caps locally (Cloud cannot override)
- Local approval required for trading-impacting changes
- Customer can disconnect from Cloud and keep running locally (subject to local controls)

### 4.3 Enterprise Engine (On-Prem/VPC/Self-Hosted)

**Target**: Hedge funds, prop trading firms, financial institutions

```
┌─────────────────────────────────────────────────────────────────┐
│                 CUSTOMER INFRASTRUCTURE                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │               Self-hosted Cloud Stack                      │  │
│  │  - Control Plane (Docker/K8s)                             │  │
│  │  - Registry Mirror (local)                                 │  │
│  │  - Monitoring (on-prem)                                    │  │
│  │  - Governance (customer RBAC)                              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Agent Cluster                           │  │
│  │  - HSM/KMS (customer-managed)                             │  │
│  │  - Risk Engine                                             │  │
│  │  - Execution Nodes                                         │  │
│  │  - Direct exchange connectivity                            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- Everything runs in customer infrastructure
- Air-gapped deployment supported
- Customer-Managed Keys (CMK) for data encryption
- Evidence pack for compliance/audit
- Signed agent updates with version pinning

---

## 5. Responsibility Layers (Design Doc §5)

### 5.1 Where Trading Decision is Born

- **Live Intent** is born ONLY on Agent (in strategy runtime)
- Cloud may store research results, but does NOT transmit live targets

### 5.2 Who Can "Start Trading"

- Cloud sends only `REQUEST_START`
- Agent by default requires local approval OR pre-configured local auto-approve policy

### 5.3 Who Can Change Trading Behavior

Any **TRADING_IMPACTING** changes:
- Strategy/model/artifact version
- Instrument universe
- Execution parameters (order types, aggressiveness)
- Risk limits
- Live schedule
- Paper→Live switch
- Broker account/adapter

→ Only through local confirmation (or local auto-approve policy)

### 5.4 What Cloud Can Do Without Approval

- **Stop/Pause** (safety, reduces risk)
- **Non-impacting** changes (telemetry verbosity, log level)

---

## 6. Legal Posture (Design Doc §18)

### 6.1 What We Are

**Software Provider / ICT Provider** providing:
- Algorithmic trading research tools
- Strategy development and backtesting platform
- Infrastructure for users to run their own trading systems

### 6.2 What We Are NOT

| We Are NOT | Why |
|------------|-----|
| **Investment Adviser** | We do not provide personalized investment advice or recommendations |
| **Broker-Dealer** | We do not execute trades on behalf of users; they use their own brokers |
| **Custodian** | We do not hold or have access to user funds or credentials |
| **Asset Manager** | We do not manage portfolios or make investment decisions |
| **Execution Service** | Cloud NEVER sends orders; Agent runs locally under user control |

### 6.3 Regulatory References

| Regulation | Our Position |
|------------|--------------|
| **MiFID II** | Software tool exclusion (ESMA Q&A ESMA35-43-349) |
| **EU AI Act** | Designed to support transparency obligations (e.g., Article 50); classification depends on deployment (no self-classification) |
| **GDPR** | Privacy-by-design posture; EU data residency by default for EU deployments (deployment-dependent) |
| **DORA** | Designed to support client vendor risk assessment (evidence exports; operational documentation) |

### 6.4 Key Legal Disclaimers

```
THE PLATFORM IS A SOFTWARE TOOL, NOT INVESTMENT ADVICE.

- We do not assess your financial situation, objectives, or risk tolerance
- We do not recommend specific investments or strategies
- We do not execute orders for you; you execute through your own broker
- All trading decisions are made solely by you
- Past performance does not guarantee future results
- Trading involves substantial risk of loss
```

---

## 7. Data Model (Design Doc §6)

### 7.1 Core Entities

| Entity | Description |
|--------|-------------|
| `Organization` | Top-level tenant |
| `Workspace` | Tenant boundary |
| `User` | Platform user |
| `Role / Permission` | Access control |
| `Strategy` | Strategy definition |
| `StrategyVersion` | Versioned strategy |
| `Build` | Artifact digest, signature, sbom_ref, provenance |
| `Artifact` | Registry ref + digest |
| `Agent` | Registered agent instance |
| `AgentEnrollmentToken` | TTL enrollment token |
| `Deployment` | Strategy deployment to agent |
| `Run` | Execution instance |
| `Command` | Cloud→Agent command |
| `ApprovalRecord` | Reference to local approval from Agent |
| `TelemetryEvent` | Agent telemetry |
| `Alert` | Monitoring alert |
| `AccessAudit` | Who viewed sensitive data |
| `DataRetentionPolicy` | Per tenant / per plan |

### 7.2 Key Entity Fields

**Agent:**
- `agent_id` (UUID), `workspace_id`, `public_key` (device key)
- `agent_version`, `last_seen_at`, `status`: ONLINE/OFFLINE
- `capabilities` (cpu/gpu/os, sandbox types), `trust_state` (ENROLLED/REVOKED)

**Build:**
- `build_id`, `strategy_version_id`, `artifact_digest` (sha256:...)
- `signature_ref`, `sbom_ref`, `created_by`, `created_at`
- `change_class`: TRADING_IMPACTING/NON
- `provenance`: {git_sha, dataset_refs, training_run_id, params_hash}

**Deployment:**
- `deployment_id`, `workspace_id`, `agent_id`, `build_id`
- `mode`: PAPER/LIVE, `desired_state`: REQUEST_START / REQUEST_STOP / ...
- `config_ref` (immutable config blob digest)
- `trading_impacting`: bool, `approval_required`: bool (derived)
- `current_state` (from agent reports)

**Command:**
- `command_id` (UUID), `deployment_id / agent_id`, `type` (enum)
- `payload_ref` (immutable blob digest), `change_class`
- `requires_approval` (bool), `issued_by` (user/system), `issued_at`
- `status`: PENDING / ACKED / APPLIED / REJECTED / EXPIRED
- `idempotency_key`

---

## 8. Change Classification (Design Doc §7)

### 8.1 TRADING_IMPACTING (Requires Local Approval)

| Category | Changes |
|----------|---------|
| **Strategy/Model** | New build/version of strategy or model |
| **Mode** | PAPER↔LIVE switch |
| **Universe** | Instrument universe change |
| **Risk** | Risk limits change (any loosening; tightening may be allowed by local policy) |
| **Broker** | Broker adapter/account change |
| **Execution** | order_types, time-in-force, aggressiveness, max order rate |
| **Schedule** | Live schedule (if affects trading sessions) |
| **Parameters** | Strategy parameters affecting signals/entries/exits |

### 8.2 NON_IMPACTING (May Apply Without Approval)

- Logging level
- Telemetry verbosity (if not adding sensitive fields)
- UI/UX parameters
- Runner non-functional configs (e.g., buffer size)
- Agent update (default: auto-update; enterprise: change window policy)

### 8.3 Policy Firewall (Local Hard Caps)

Agent stores local policy that:
- Sets absolute upper risk limits
- Prohibits some instruments/order types
- Prohibits auto-approve for some changes

**Cloud CANNOT raise risk above hard caps ever.**

---

## 9. Threat Model (Design Doc §15)

| Threat | Attack Vector | Mitigation |
|--------|---------------|------------|
| **RCE in Cloud** | Attacker gains code execution in Cloud | Cloud has no trading libs, no broker APIs, cannot execute orders |
| **Key Exfiltration** | Attacker tries to steal broker credentials | Keys never leave Agent, mandatory redaction, no secret logging |
| **Artifact Tampering** | Attacker modifies trading artifact | Digest pinning, signature verification, SBOM tracking |
| **Cloud Becomes Execution** | Cloud attempts to send trading commands | Protocol schema prohibits order-like payloads (side/qty/price) |
| **Compute Abuse** | User mines crypto on cloud resources | Sandbox isolation, CPU/RAM/time quotas, egress allowlist |
| **Man-in-the-Middle** | Attacker intercepts Cloud↔Agent traffic | mTLS/signed messages, certificate pinning |
| **Replay Attacks** | Attacker replays old commands | Idempotency keys, timestamps, nonce validation |
| **Privilege Escalation** | User accesses other tenant's data | RBAC, Postgres RLS, workspace isolation |
| **Rollback Attack** | Attacker pins old vulnerable version | Signed update metadata (TUF-style), min/max schema versions |

### 9.1 Safe Defaults (Cannot Be Disabled)

| Setting | Default | Can User Override? |
|---------|---------|-------------------|
| Telemetry Redaction | ON | NO (mandatory) |
| Local Approval for Trading-Impacting | REQUIRED | Only stricter (not looser) |
| RAW Order Telemetry | OFF | Enterprise opt-in only |
| Remote Flatten Position | DISABLED | Enterprise by contract only |
| Silent Upgrades | DISABLED | NO for trading-impacting |
| Auto-Approve | DISABLED | Local policy only |
| Artifact Signature Verification | REQUIRED | NO |

### 9.2 Secret Hygiene

```
Secrets (broker API keys, master keys) are protected by:

1. STORAGE: Only in Agent's Local Vault
   - OS keychain preferred (macOS/Linux/Windows)
   - Encrypted fallback with env var master key

2. TRANSMISSION: Designed to stay local
   - Redaction middleware is mandatory by design
   - Pattern matching for typical secret formats

3. LOGGING: Designed to be redacted
   - Automatic redaction in all logs by design
   - Support dumps designed to exclude secrets

4. TELEMETRY: Designed to be redacted
   - Env vars not logged by design
   - Account IDs masked
   - IP addresses anonymized (optional)
```

---

## 10. Protocol Security (Design Doc §10)

### 10.1 Allowed Commands (Allowlist)

| Command | Direction | Purpose | Requires Local Approval |
|---------|-----------|---------|------------------------|
| `REQUEST_START_RUN` | Cloud→Agent | Start strategy execution | YES (trading_impacting) |
| `REQUEST_STOP_RUN` | Cloud→Agent | Stop execution | NO (safety command) |
| `REQUEST_PAUSE_RUN` | Cloud→Agent | Pause execution | NO (safety command) |
| `REQUEST_UPGRADE_ARTIFACT` | Cloud→Agent | Deploy new version | YES (trading_impacting) |
| `REQUEST_UPDATE_CONFIG` | Cloud→Agent | Update configuration | YES (if trading_impacting) |
| `REQUEST_ROTATE_AGENT_SESSION` | Cloud→Agent | Rotate session keys | YES |
| `REQUEST_EXPORT_LOGS` | Cloud→Agent | Export logs with redaction | YES (data_sensitive) |

### 10.2 Prohibited Payloads

JSON payloads in commands **MUST NOT** contain:

```json
// PROHIBITED - these fields indicate order-like content
{
  "side": "BUY|SELL",           // Order direction
  "quantity": 100,              // Order size
  "price": 50000.00,            // Order price
  "order_type": "MARKET|LIMIT", // Order type
  "target_position": 0.5,       // Target allocation
  "symbol": "BTCUSDT"           // Trading symbol (in order context)
}
```

These fields are **blocked at schema level** and **validated by CI guardrails**.

### 10.3 Authentication

| Method | Use Case | Implementation |
|--------|----------|----------------|
| **mTLS** | Enterprise | Mutual TLS with client certificates |
| **Signed JWT** | Default | Ed25519 device key signs all messages |

All messages include:
- `idempotency_key` for deduplication
- `timestamp` for replay protection
- `signature` for authenticity

---

## 11. State Machines (Design Doc §11)

### 11.1 Deployment State Machine

```
                                ┌──────────────────┐
                                │     CREATED      │
                                └────────┬─────────┘
                                         │ provision_agent
                                         ▼
                                ┌──────────────────┐
                                │    ENROLLING     │
                                └────────┬─────────┘
                                         │ agent_enrolled
                                         ▼
┌─────────────────┐  upgrade    ┌──────────────────┐
│    UPGRADING    │◀───────────▶│     ENROLLED     │
└────────┬────────┘             └────────┬─────────┘
         │                               │ request_start
         ▼                               ▼
┌─────────────────┐             ┌──────────────────┐
│ UPGRADE_PENDING │             │  START_PENDING   │
│   (approval)    │             │   (approval)     │
└────────┬────────┘             └────────┬─────────┘
         │ approved                      │ approved
         ▼                               ▼
         └───────────────┬───────────────┘
                         ▼
                ┌──────────────────┐
                │     RUNNING      │◀──────────────┐
                └────────┬─────────┘               │
                         │                         │
        ┌────────────────┼────────────────┐        │
        │                │                │        │
        ▼                ▼                ▼        │
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│   PAUSING    │ │   STOPPING   │ │    HALTED    │ │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ │
       │                │                │         │
       ▼                ▼                │         │
┌──────────────┐ ┌──────────────┐        │         │
│    PAUSED    │ │   STOPPED    │        │         │
└──────┬───────┘ └──────────────┘        │         │
       │                                 │         │
       │ resume                          │ restart │
       └─────────────────────────────────┴─────────┘
```

### 11.2 Run State Machine

```
                    ┌──────────────────┐
                    │     CREATED      │
                    └────────┬─────────┘
                             │ init
                             ▼
                    ┌──────────────────┐
                    │  INITIALIZING    │
                    └────────┬─────────┘
                             │ preflight_ok
                             ▼
                    ┌──────────────────┐
                    │     RUNNING      │◀──────────┐
                    └────────┬─────────┘           │
                             │                     │
            ┌────────────────┼────────────┐        │
            │                │            │        │
            ▼                ▼            ▼        │
    ┌──────────────┐ ┌─────────────┐ ┌─────────┐   │
    │   PAUSED     │ │   STOPPED   │ │ HALTED  │   │
    └──────┬───────┘ └─────────────┘ │(KillSw) │   │
           │                         └────┬────┘   │
           │ resume                       │ ack    │
           └──────────────────────────────┴────────┘
```

### 11.3 State Transitions

| Current State | Event | Next State | Approval Required |
|---------------|-------|------------|-------------------|
| ENROLLED | REQUEST_START | START_PENDING | YES |
| START_PENDING | APPROVED | RUNNING | - |
| RUNNING | REQUEST_PAUSE | PAUSED | NO |
| RUNNING | REQUEST_STOP | STOPPED | NO |
| RUNNING | KILL_SWITCH | HALTED | NO |
| PAUSED | REQUEST_RESUME | RUNNING | YES (if config changed) |
| HALTED | ACKNOWLEDGE | STOPPED | YES |

---

## 12. Config Layering (Design Doc §12)

### 12.1 Priority Order

```
Priority (highest wins):

1. LOCAL_HARD_CAPS        ← Agent enforced, immutable
2. LOCAL_POLICY           ← User's local policy
3. ARTIFACT_RISK_PROFILE  ← Strategy's risk_profile_suggested
4. CLOUD_CONFIG           ← Remote configuration
5. DEFAULTS               ← System defaults
```

### 12.2 Merge Rules

| Setting | Rule |
|---------|------|
| Risk limits | `min(all_layers)` - lowest limit wins |
| Allowed symbols | `intersection(all_layers)` |
| Denied symbols | `union(all_layers)` |
| Boolean flags | Lower layer can only make stricter |

### 12.3 Example

```yaml
# Cloud config suggests:
max_position_pct: 20%

# Artifact risk_profile_suggested:
max_position_pct: 15%

# Local policy:
max_position_pct: 10%

# Hard cap:
max_position_pct: 10%

# Result: 10% (hard cap enforced)
```

---

## 13. Telemetry & Privacy (Design Doc §13-14)

### 13.1 Telemetry Levels

| Level | Data Collected | Default | Availability |
|-------|----------------|---------|--------------|
| `AGGREGATED` | PnL, win rate, drawdown (no trade details) | YES | All users |
| `DETAILED_NON_SENSITIVE` | Trade counts, timing, latency | Opt-in | All users |
| `RAW_ORDER_EVENTS` | Full order details | Opt-in | Enterprise only |

### 13.2 Data Residency

| Tenant Type | Primary Region | Configurable |
|-------------|----------------|--------------|
| EU Users | AWS eu-central-1 (Frankfurt) | YES |
| Enterprise | Customer-specified | YES |
| On-Prem | Customer infrastructure | N/A |

### 13.3 GDPR Rights

Users have full GDPR rights:
- **Access** (Article 15): Export all personal data
- **Rectification** (Article 16): Correct inaccurate data
- **Erasure** (Article 17): Delete account and data
- **Portability** (Article 20): Download data in machine-readable format
- **Object** (Article 21): Opt out of processing

---

## 14. CI Guardrails (Design Doc §19)

### 14.1 Build-Time Checks

| Check | What It Validates | Failure Action |
|-------|-------------------|----------------|
| `no-trading-libs-in-cloud` | Cloud build excludes order_execution modules | Block build |
| `no-order-payloads-in-schema` | JSON schema has no side/qty/price fields | Block merge |
| `artifact-signature-required` | Artifact is signed before publish | Block publish |
| `redaction-enabled` | Telemetry redaction cannot be disabled | Block deploy |
| `import-boundary-check` | No agent imports in cloud packages | Block build |

### 14.2 Runtime Checks

| Check | What It Validates | Failure Action |
|-------|-------------------|----------------|
| `signature-verification` | Agent verifies artifact signature | Reject artifact |
| `schema-version-check` | Protocol schema versions compatible | Reject command |
| `approval-required` | Trading-impacting changes approved | Queue for approval |
| `hard-cap-enforcement` | Local risk limits enforced | Reject/limit order |

---

## 15. Rollout Plan (Design Doc §20)

### 15.1 Phase Summary

| Phase | Scope | Focus |
|-------|-------|-------|
| **P0** | Foundations | Design doc, guardrails, protocol schemas, legal docs |
| **P1** | Agent Core | Vault, approval, preflight, daemon, reconciliation |
| **P2** | Cloud Integration | Control plane, builder, governance, enterprise |

### 15.2 Milestones

```
P0 "Design + Guardrails"
  ├── Protocol schema (manifest, commands)
  ├── CI guardrails (import check, schema check)
  ├── Legal documents (ToS, AUP)
  └── Sequence diagrams + traceability matrix

P1 "Agent Foundations"
  ├── Local Vault (keychain/encrypted file)
  ├── Approval system (CLI + evidence)
  ├── Policy firewall (hard caps)
  ├── Kill switch + preflight
  └── Telemetry redaction

P2 "Cloud + Enterprise"
  ├── Control plane API
  ├── Artifact builder + registry
  ├── Governance (RBAC, residency, retention)
  ├── Enterprise deployment (on-prem, air-gap)
  └── Evidence pack export
```

---

## 16. Document References

### 16.1 Zone-Specific Documentation

| Zone | Location | Contents |
|------|----------|----------|
| Cloud Zone | [docs/cloud/](cloud/) | Control Plane API, Artifact Builder, Governance, Research Job Isolation |
| Agent Zone | [docs/agent/](agent/) | Installation, Local Vault, Approvals, Risk Controls, Degraded Modes |

### 16.2 Design Documents

| Document | Location | Purpose |
|----------|----------|---------|
| Full Design Doc | `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.md` | Complete technical specification |
| Target Architecture | `docs/design/CCEA_CLOUD/TARGET_CCEA_ARCHITECTURE.md` | Module mapping |
| Sequence Diagrams | `docs/design/CCEA_CLOUD/CCEA_SEQUENCE_DIAGRAMS.md` | Interaction flows |
| Traceability Matrix | `docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md` | Requirements tracing |
| CI Guardrails | `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md` | CI/CD validation rules |
| Decision Log | `docs/design/CCEA_CLOUD/DECISION_LOG.md` | Architecture decisions |

### 16.3 Operational Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Runbooks Index | [docs/runbooks/](runbooks/) | Kill switch, recovery, revocation |
| JSON Schemas | [docs/schemas/](schemas/) | Protocol and manifest schemas |
| UI Guardrails | [docs/ui/](ui/) | Onboarding disclaimers, UI requirements |

### 16.4 Legal Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Terms of Service | [docs/legal/TERMS_OF_SERVICE.md](legal/TERMS_OF_SERVICE.md) | Legal terms with CCEA positioning |
| Privacy Policy | [docs/legal/PRIVACY_POLICY.md](legal/PRIVACY_POLICY.md) | Data handling with CCEA zones |
| Acceptable Use Policy | [docs/legal/ACCEPTABLE_USE_POLICY.md](legal/ACCEPTABLE_USE_POLICY.md) | Anti-abuse guidelines |

---

## 17. Implementation Status

### 17.1 Completed Phases

CCEA implementation phases (verify status via test suite):

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1-6 (P0) | Foundation, guardrails, legal, docs | Verify via tests |
| Phase 7-9 (P1) | Control plane, agent lifecycle, reconciliation | Verify via tests |
| Phase 10 (P2) | Enterprise, sandbox isolation, evidence pack | Verify via tests |

### 17.2 Key Implementation Artifacts

- **Test suite** in `tests/ccea/` (verify count via `find tests/ccea -name "*.py" | wc -l`)
- **packages/agent/**: Local vault, approval, policy firewall, reconciliation
- **packages/cloud/**: Control plane, builder, governance, enterprise features
- **deploy/helm/**: Enterprise Kubernetes deployment (design)
- **docs/**: Documentation for all zones

For detailed traceability, see [CCEA_TRACEABILITY_MATRIX.md](design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md).

---

**Document Control:**
- Author: CCEA Architecture Team
- Reviewers: Security, Compliance, Engineering, Legal
- Approval: Architecture Review Board
- Last Review: 2025-12-16
- Implementation Status: **All Design Doc requirements implemented**

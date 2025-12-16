# CCEA Overview: Cloud-Controlled Execution Architecture

> **Version**: 2.0.0 | **Last Updated**: 2025-12-16 | **Status**: APPROVED | **Implementation**: 100% Complete

## Executive Summary

CCEA (Cloud-Controlled Execution Architecture) defines a strict security boundary between **Cloud** (research/monitoring/lifecycle management) and **Agent** (execution/secrets/risk enforcement). This architecture ensures that:

- **Cloud NEVER has access to trading credentials or order execution capabilities**
- **All trading operations occur locally in the user's environment (Agent)**
- **Clear regulatory positioning as Software Provider, not Investment Adviser**

---

## 1. Architecture Boundary

### 1.1 The Fundamental Principle (Non-Negotiable)

```
Cloud = research / build / monitoring / control plane (lifecycle requests)
Agent = secrets + live loop + risk enforce + order creation/sending

Cloud NEVER:
  - Stores broker API keys
  - Generates or transmits orders
  - Has access to exchange trading endpoints
  - Sends order-like payloads (side/qty/price)
```

### 1.2 Visual Architecture

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

### 1.3 Zone Responsibilities

| Zone | What It Does | What It NEVER Does |
|------|--------------|-------------------|
| **Cloud** | Research IDE, Backtesting, Artifact build/sign, Monitoring dashboards, Lifecycle management | Store secrets, Generate orders, Access trading APIs, Send order payloads |
| **Agent** | Store secrets (local vault), Enforce risk (policy firewall, hard caps), Run live loop, Create and send orders | Run without user consent, Bypass local limits, Share secrets with Cloud |
| **Shared** | Core models, Simulation engine, Feature engineering, Training | Execute live orders, Store production secrets |

---

## 2. Product Modes

### 2.1 Retail Research SaaS (EU-friendly)

**Target**: Individual researchers, quants learning algo trading

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
│              OPTIONAL: BYO Agent (User's Machine)               │
│  - Only if user wants live trading                              │
│  - User provides their own broker credentials                   │
│  - Agent runs on user's hardware                                │
└─────────────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- Cloud provides research/simulation/monitoring only
- Agent is optional, only needed for live trading
- All secrets stay with the user
- EU data residency by default
- GDPR compliant telemetry

### 2.2 Retail Live via Local Agent

**Target**: Active traders who want automated execution

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
- Auto-execution runs locally on user's machine
- Cloud provides observability and lifecycle requests only
- Agent enforces hard caps locally (Cloud cannot override)
- Local approval required for trading-impacting changes
- User can disconnect from Cloud and keep trading

### 2.3 Enterprise Engine (On-Prem/VPC/Self-Hosted)

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

## 3. Legal Posture

### 3.1 What We Are

**Software Provider / ICT Provider** providing:
- Algorithmic trading research tools
- Strategy development and backtesting platform
- Infrastructure for users to run their own trading systems

### 3.2 What We Are NOT

| We Are NOT | Why |
|------------|-----|
| **Investment Adviser** | We do not provide personalized investment advice or recommendations |
| **Broker-Dealer** | We do not execute trades on behalf of users; they use their own brokers |
| **Custodian** | We do not hold or have access to user funds or credentials |
| **Asset Manager** | We do not manage portfolios or make investment decisions |
| **Execution Service** | Cloud NEVER sends orders; Agent runs locally under user control |

### 3.3 Regulatory References

| Regulation | Our Position |
|------------|--------------|
| **MiFID II** | Software tool exclusion (ESMA Q&A ESMA35-43-349) |
| **EU AI Act** | Transparency provider (Article 50), Not high-risk AI deployer |
| **GDPR** | Data Controller with EU data residency |
| **DORA** | ICT Service Provider (contractual compliance) |

### 3.4 Key Legal Disclaimers

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

## 4. Threat Model

### 4.1 Threat Vectors and Mitigations

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

### 4.2 Safe Defaults (Cannot Be Disabled)

| Setting | Default | Can User Override? |
|---------|---------|-------------------|
| Telemetry Redaction | ON | NO (mandatory) |
| Local Approval for Trading-Impacting | REQUIRED | Only stricter (not looser) |
| RAW Order Telemetry | OFF | Enterprise opt-in only |
| Remote Flatten Position | DISABLED | Enterprise by contract only |
| Silent Upgrades | DISABLED | NO for trading-impacting |
| Auto-Approve | DISABLED | Local policy only |
| Artifact Signature Verification | REQUIRED | NO |

### 4.3 Secret Hygiene

```
Secrets (broker API keys, master keys) are protected by:

1. STORAGE: Only in Agent's Local Vault
   - OS keychain preferred (macOS/Linux/Windows)
   - Encrypted fallback with env var master key

2. TRANSMISSION: Never to Cloud
   - Redaction middleware is mandatory
   - Pattern matching for typical secret formats

3. LOGGING: Never logged
   - Automatic redaction in all logs
   - Support dumps never contain secrets

4. TELEMETRY: Never transmitted
   - Env vars not logged
   - Account IDs masked
   - IP addresses anonymized (optional)
```

---

## 5. Protocol Security

### 5.1 Allowed Commands (Allowlist)

| Command | Direction | Purpose | Requires Local Approval |
|---------|-----------|---------|------------------------|
| `REQUEST_START_RUN` | Cloud→Agent | Start strategy execution | YES (trading_impacting) |
| `REQUEST_STOP_RUN` | Cloud→Agent | Stop execution | NO (safety command) |
| `REQUEST_PAUSE_RUN` | Cloud→Agent | Pause execution | NO (safety command) |
| `REQUEST_UPGRADE_ARTIFACT` | Cloud→Agent | Deploy new version | YES (trading_impacting) |
| `REQUEST_UPDATE_CONFIG` | Cloud→Agent | Update configuration | YES (if trading_impacting) |
| `REQUEST_ROTATE_AGENT_SESSION` | Cloud→Agent | Rotate session keys | YES |
| `REQUEST_EXPORT_LOGS` | Cloud→Agent | Export logs with redaction | YES (data_sensitive) |

### 5.2 Prohibited Payloads

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

### 5.3 Authentication

| Method | Use Case | Implementation |
|--------|----------|----------------|
| **mTLS** | Enterprise | Mutual TLS with client certificates |
| **Signed JWT** | Default | Ed25519 device key signs all messages |

All messages include:
- `idempotency_key` for deduplication
- `timestamp` for replay protection
- `signature` for authenticity

---

## 6. Telemetry & Privacy

### 6.1 Telemetry Levels

| Level | Data Collected | Default | Availability |
|-------|----------------|---------|--------------|
| `AGGREGATED` | PnL, win rate, drawdown (no trade details) | YES | All users |
| `DETAILED_NON_SENSITIVE` | Trade counts, timing, latency | Opt-in | All users |
| `RAW_ORDER_EVENTS` | Full order details | Opt-in | Enterprise only |

### 6.2 Data Residency

| Tenant Type | Primary Region | Configurable |
|-------------|----------------|--------------|
| EU Users | AWS eu-central-1 (Frankfurt) | YES |
| Enterprise | Customer-specified | YES |
| On-Prem | Customer infrastructure | N/A |

### 6.3 GDPR Rights

Users have full GDPR rights:
- **Access** (Article 15): Export all personal data
- **Rectification** (Article 16): Correct inaccurate data
- **Erasure** (Article 17): Delete account and data
- **Portability** (Article 20): Download data in machine-readable format
- **Object** (Article 21): Opt out of processing

---

## 7. CI Guardrails

### 7.1 Build-Time Checks

| Check | What It Validates | Failure Action |
|-------|-------------------|----------------|
| `no-trading-libs-in-cloud` | Cloud build excludes order_execution modules | Block build |
| `no-order-payloads-in-schema` | JSON schema has no side/qty/price fields | Block merge |
| `artifact-signature-required` | Artifact is signed before publish | Block publish |
| `redaction-enabled` | Telemetry redaction cannot be disabled | Block deploy |
| `import-boundary-check` | No agent imports in cloud packages | Block build |

### 7.2 Runtime Checks

| Check | What It Validates | Failure Action |
|-------|-------------------|----------------|
| `signature-verification` | Agent verifies artifact signature | Reject artifact |
| `schema-version-check` | Protocol schema versions compatible | Reject command |
| `approval-required` | Trading-impacting changes approved | Queue for approval |
| `hard-cap-enforcement` | Local risk limits enforced | Reject/limit order |

---

## 8. Document References

### 8.1 Zone-Specific Documentation

| Zone | Location | Contents |
|------|----------|----------|
| Cloud Zone | [docs/cloud/](cloud/) | Control Plane API, Artifact Builder, Governance, Research Job Isolation |
| Agent Zone | [docs/agent/](agent/) | Installation, Local Vault, Approvals, Risk Controls, Degraded Modes |

### 8.2 Design Documents

| Document | Location | Purpose |
|----------|----------|---------|
| Full Design Doc | `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.md` | Complete technical specification |
| Target Architecture | `docs/design/CCEA_CLOUD/TARGET_CCEA_ARCHITECTURE.md` | Module mapping |
| Sequence Diagrams | `docs/design/CCEA_CLOUD/CCEA_SEQUENCE_DIAGRAMS.md` | Interaction flows |
| Traceability Matrix | `docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md` | Requirements tracing |
| CI Guardrails | `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md` | CI/CD validation rules |
| Decision Log | `docs/design/CCEA_CLOUD/DECISION_LOG.md` | Architecture decisions |

### 8.3 Operational Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Runbooks Index | [docs/runbooks/](runbooks/) | Kill switch, recovery, revocation |
| JSON Schemas | [docs/schemas/](schemas/) | Protocol and manifest schemas |
| UI Guardrails | [docs/ui/](ui/) | Onboarding disclaimers, UI requirements |

### 8.4 Legal Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Terms of Service | [docs/legal/TERMS_OF_SERVICE.md](legal/TERMS_OF_SERVICE.md) | Legal terms with CCEA positioning |
| Privacy Policy | [docs/legal/PRIVACY_POLICY.md](legal/PRIVACY_POLICY.md) | Data handling with CCEA zones |
| Acceptable Use Policy | [docs/legal/ACCEPTABLE_USE_POLICY.md](legal/ACCEPTABLE_USE_POLICY.md) | Anti-abuse guidelines |

---

## 9. Implementation Status

### Completed Phases

All phases of the CCEA implementation are complete:

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1-6 (P0) | Foundation, guardrails, legal, docs | ✓ Complete |
| Phase 7-9 (P1) | Control plane, agent lifecycle, reconciliation | ✓ Complete |
| Phase 10 (P2) | Enterprise, sandbox isolation, evidence pack | ✓ Complete |

### Key Implementation Artifacts

- **117 test files** in `tests/ccea/` covering all requirements
- **packages/agent/**: Local vault, approval, policy firewall, reconciliation
- **packages/cloud/**: Control plane, builder, governance, enterprise features
- **deploy/helm/**: Enterprise Kubernetes deployment
- **docs/**: Complete documentation for all zones

For detailed traceability, see [CCEA_TRACEABILITY_MATRIX.md](design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md).

---

**Document Control:**
- Author: CCEA Architecture Team
- Reviewers: Security, Compliance, Engineering, Legal
- Approval: Architecture Review Board
- Last Review: 2025-12-16
- Implementation Status: **100% Design Doc Compliance**

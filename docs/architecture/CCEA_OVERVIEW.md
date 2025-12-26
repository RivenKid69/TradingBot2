# CCEA Platform - Cloud-Controlled Execution Architecture

## Overview

**Document Version:** 3.0
**Date:** December 2025
**Classification:** Technical Reference Documentation

---

## 0. Document Purpose

This document describes the **Cloud-Controlled Execution Architecture (CCEA)** - the foundational architecture for the CCEA Platform. It defines:

- The separation of concerns between Cloud and Agent
- Security boundaries and data flow
- Regulatory positioning
- Terminology and concepts
- Marketing and ToS guidelines

All other documentation should reference this document for architectural consistency.

---

## 1. Executive Summary

CCEA is a **two-tier architecture** designed to deliver AI-powered quantitative research and execution tools while maintaining:

1. **Regulatory Compliance** - Cloud is designed not to touch trading execution
2. **Security** - Customer credentials designed to remain in customer environment
3. **Transparency** - Open-source Agent enables customer audit
4. **Control** - Customer approves all trading-impacting changes locally

### Core Principle

> **"Cloud provides intelligence, Agent provides execution. The two are deliberately separated by an air gap that protects both the customer and the company."**

---

## 2. Terminology (Canonical Definitions)

| Term | Definition |
|------|------------|
| **Cloud** | Our SaaS services hosted in our infrastructure (Training, Backtest, Artifacts, Control Plane) |
| **Agent** | Customer-deployed runtime in customer's environment (executes strategies, holds credentials) |
| **Strategy** | User code/model that produces Intents based on market data |
| **Intent** | High-level trading intention (e.g., "target 10% BTC allocation") - NOT a ready order |
| **Order** | Concrete broker instruction (designed to be created only in Agent) |
| **Deployment** | Binding of strategy artifact + configuration + target agent |
| **Run** | Specific execution instance of a strategy on an agent |
| **Command** | Lifecycle request from Cloud to Agent (start, stop, pause, upgrade) |
| **Approval** | Local confirmation on Agent for any TRADING_IMPACTING change |
| **TRADING_IMPACTING** | Actions that can result in position/balance changes |
| **NON_IMPACTING** | Purely observational or research actions (no position changes) |

---

## 3. Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CCEA PLATFORM                                  │
├─────────────────────────────────┬──────────────────────────────────────────┤
│           CLOUD                 │              AGENT                        │
│     (Our Infrastructure)        │     (Customer Environment)                │
├─────────────────────────────────┼──────────────────────────────────────────┤
│                                 │                                           │
│  ┌─────────────────────────┐   │   ┌─────────────────────────────────┐    │
│  │   Auth & Accounts       │   │   │        Agent Daemon              │    │
│  │   - User management     │   │   │   - Connects to Cloud            │    │
│  │   - Team permissions    │   │   │   - Receives Commands            │    │
│  └─────────────────────────┘   │   │   - Manages local components     │    │
│                                 │   └─────────────────────────────────┘    │
│  ┌─────────────────────────┐   │                                           │
│  │   Strategy Workspace    │   │   ┌─────────────────────────────────┐    │
│  │   - Code editing        │   │   │        Local Vault               │    │
│  │   - Version control     │   │   │   - API keys (encrypted)        │    │
│  └─────────────────────────┘   │   │   - Never sent to Cloud         │    │
│                                 │   └─────────────────────────────────┘    │
│  ┌─────────────────────────┐   │                                           │
│  │   Backtest & Sim        │   │   ┌─────────────────────────────────┐    │
│  │   - L1/L2/L3 simulation │   │   │        Strategy Runner          │    │
│  │   - Paper trading       │   │   │   - Executes artifacts          │    │
│  └─────────────────────────┘   │   │   - Produces Intents → Orders   │    │
│                                 │   └─────────────────────────────────┘    │
│  ┌─────────────────────────┐   │                                           │
│  │   Training Service      │   │   ┌─────────────────────────────────┐    │
│  │   - RL model training   │   │   │        Risk Manager             │    │
│  │   - CVaR optimization   │◄──┼───│   - Position limits             │    │
│  └─────────────────────────┘   │   │   - Drawdown guards             │    │
│                                 │   │   - Kill Switch                 │    │
│  ┌─────────────────────────┐   │   └─────────────────────────────────┘    │
│  │   Artifact Builder      │   │                                           │
│  │   - Builds strategy     │   │   ┌─────────────────────────────────┐    │
│  │   - Creates manifest    │   │   │        Broker Connectors        │    │
│  │   - Signs artifacts     │   │   │   - Binance, Alpaca, OANDA      │    │
│  └─────────────────────────┘   │   │   - CREATES ORDERS HERE         │    │
│                                 │   └─────────────────────────────────┘    │
│  ┌─────────────────────────┐   │                                           │
│  │   Artifact Registry     │   │   ┌─────────────────────────────────┐    │
│  │   - Stores artifacts    │   │   │        Approval UI              │    │
│  │   - Version management  │   │   │   - Local approval prompt       │    │
│  └─────────────────────────┘   │   │   - TRADING_IMPACTING gate      │    │
│                                 │   └─────────────────────────────────┘    │
│  ┌─────────────────────────┐   │                                           │
│  │   Control Plane         │   │   ┌─────────────────────────────────┐    │
│  │   - Sends Commands      │───┼──►│        State Reconciler         │    │
│  │   - NOT orders          │   │   │   - Syncs with broker           │    │
│  └─────────────────────────┘   │   │   - Tracks positions            │    │
│                                 │   └─────────────────────────────────┘    │
│  ┌─────────────────────────┐   │                                           │
│  │   Telemetry Service     │   │   ┌─────────────────────────────────┐    │
│  │   - Aggregated metrics  │◄──┼───│        Telemetry Buffer         │    │
│  │   - No raw order data   │   │   │   - Local event journal         │    │
│  └─────────────────────────┘   │   │   - Privacy filtering           │    │
│                                 │   └─────────────────────────────────┘    │
└─────────────────────────────────┴──────────────────────────────────────────┘
```

---

## 4. Security Boundaries

### 4.1 What Cloud Does Not Do (by design)

| Prohibited Action | Why |
|-------------------|-----|
| Store broker/exchange API keys | Security + regulatory compliance |
| Generate live trading orders | Would require investment advisor license |
| Transmit ready-to-execute orders | Same as above |
| Access customer's live positions directly | Privacy + regulatory |
| Execute trades on behalf of customer | Broker/dealer license required |

### 4.2 What Agent Controls (by design)

| Exclusive Agent Responsibility | Rationale |
|-------------------------------|-----------|
| Store and use API credentials | Customer owns and controls |
| Create Orders from Intents | Execution is customer's decision |
| Apply local risk limits | Customer's risk appetite |
| Approve TRADING_IMPACTING changes | Customer control |
| Kill switch activation | Safety is customer's prerogative |

---

## 5. Command Protocol (Cloud → Agent)

### 5.1 Allowed Commands

| Command | Description | Impact Type |
|---------|-------------|-------------|
| `REQUEST_START_RUN` | Request agent to start a strategy run | TRADING_IMPACTING |
| `REQUEST_STOP_RUN` | Request agent to stop a running strategy | TRADING_IMPACTING |
| `REQUEST_PAUSE_RUN` | Request agent to pause execution | TRADING_IMPACTING |
| `REQUEST_RESUME_RUN` | Request agent to resume execution | TRADING_IMPACTING |
| `REQUEST_UPGRADE_ARTIFACT` | Request agent to upgrade strategy version | TRADING_IMPACTING |
| `REQUEST_UPDATE_CONFIG` | Request agent to update configuration | TRADING_IMPACTING |
| `PING` / `HEARTBEAT` | Health check | NON_IMPACTING |
| `REQUEST_STATUS` | Request current agent status | NON_IMPACTING |
| `REQUEST_LOGS` | Request log excerpt | NON_IMPACTING |

### 5.2 Prohibited Commands (Not Sent by Cloud)

| Prohibited Command | Why Prohibited |
|-------------------|----------------|
| `PLACE_ORDER` | Would make Cloud an execution venue |
| `SUBMIT_ORDER` | Same as above |
| `EXECUTE_SIGNAL` | Same as above |
| `SET_TARGET_POSITION_NOW` | Same as above |
| `CANCEL_ORDER` | Order management is Agent's domain |
| `MODIFY_ORDER` | Same as above |

### 5.3 Approval Protocol

All TRADING_IMPACTING commands require local approval:

```
┌─────────┐      REQUEST_START_RUN      ┌─────────┐
│  Cloud  │ ────────────────────────────► │  Agent  │
└─────────┘                               └────┬────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │   Approval UI     │
                                    │                   │
                                    │  "Cloud requests  │
                                    │   to start run    │
                                    │   'BTC_Strategy'  │
                                    │                   │
                                    │  [Approve] [Deny] │
                                    └──────────────────┘
                                               │
                             ┌─────────────────┴─────────────────┐
                             │                                   │
                      User clicks              User clicks
                       [Approve]                 [Deny]
                             │                                   │
                             ▼                                   ▼
                    Run starts locally              Command rejected
                    Agent notifies Cloud           Cloud receives rejection
```

---

## 6. Data Flow

### 6.1 Strategy Development Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     STRATEGY DEVELOPMENT FLOW                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1. DEVELOP                    2. TRAIN                   3. BACKTEST    │
│  ┌─────────────────┐          ┌─────────────────┐       ┌─────────────┐ │
│  │ Strategy        │          │ Training        │       │ Backtest    │ │
│  │ Workspace       │────────► │ Service         │──────►│ Service     │ │
│  │ (Cloud)         │          │ (Cloud)         │       │ (Cloud)     │ │
│  └─────────────────┘          └─────────────────┘       └─────────────┘ │
│                                                                           │
│  4. BUILD                      5. REGISTER               6. DEPLOY       │
│  ┌─────────────────┐          ┌─────────────────┐       ┌─────────────┐ │
│  │ Artifact        │          │ Artifact        │       │ Agent       │ │
│  │ Builder         │────────► │ Registry        │──────►│ (Customer)  │ │
│  │ (Cloud)         │          │ (Cloud)         │       └─────────────┘ │
│  └─────────────────┘          └─────────────────┘                        │
│                                                                           │
│  Artifact contains:                                                       │
│  ├── strategy.pkl (model weights)                                        │
│  ├── manifest.json (metadata, version, hash)                            │
│  ├── config.yaml (non-sensitive configuration)                          │
│  └── signature (cryptographic signature)                                 │
│                                                                           │
│  Artifact does NOT contain:                                               │
│  ├── API keys                                                            │
│  ├── Broker credentials                                                  │
│  └── Order instructions                                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Live Execution Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        LIVE EXECUTION FLOW                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────┐          ┌─────────────────┐          ┌────────────────┐  │
│   │ Market  │          │     Agent       │          │    Broker      │  │
│   │  Data   │────────► │                 │          │                │  │
│   └─────────┘          │  ┌───────────┐  │          │                │  │
│                        │  │ Strategy  │  │          │                │  │
│                        │  │  Runner   │  │          │                │  │
│                        │  └─────┬─────┘  │          │                │  │
│                        │        │        │          │                │  │
│                        │        ▼        │          │                │  │
│                        │   ┌────────┐    │          │                │  │
│                        │   │ Intent │    │          │                │  │
│                        │   └────┬───┘    │          │                │  │
│                        │        │        │          │                │  │
│                        │        ▼        │          │                │  │
│                        │  ┌──────────┐   │          │                │  │
│                        │  │   Risk   │   │          │                │  │
│                        │  │ Manager  │   │          │                │  │
│                        │  └────┬─────┘   │          │                │  │
│                        │       │         │          │                │  │
│                        │       ▼         │          │                │  │
│                        │  ┌──────────┐   │          │                │  │
│                        │  │ Broker   │   │  ORDER   │                │  │
│                        │  │Connector │───┼─────────►│    EXECUTED    │  │
│                        │  └──────────┘   │          │                │  │
│                        │                 │          │                │  │
│                        └─────────────────┘          └────────────────┘  │
│                                                                           │
│   Key Points:                                                             │
│   • Market data can come from Cloud (simulation) or direct (live)        │
│   • Intent is high-level ("target 10% BTC")                              │
│   • Risk Manager validates against local limits                          │
│   • Order is designed to be created only in Agent, by Broker Connector   │
│   • Cloud does NOT see or create live Orders                             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Product Modes

### 7.1 Three Deployment Modes

| Mode | Description | Target User | Regulatory Position |
|------|-------------|-------------|---------------------|
| **Pro Research SaaS (B2B)** | Cloud-only access to backtest, simulation, training. No live execution. | Professional systematic teams (research workflows only) | Software Provider posture (licensing depends on activities and jurisdiction; verify with counsel) |
| **Pro Live via Customer Agent (B2B)** | Cloud services + customer-deployed Agent for live trading | Professional systematic teams, prop firms | Software Provider (customer executes) |
| **Enterprise Engine (B2B)** | On-premise or VPC deployment of full stack | Hedge funds, banks, large prop firms | Software License (customer owns entire stack) |

### 7.2 Mode Comparison

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCT MODE COMPARISON                             │
├──────────────────────┬──────────────────┬───────────────┬─────────────────┤
│       Feature        │ Pro Research     │ Pro Live      │ Enterprise      │
├──────────────────────┼──────────────────┼───────────────┼─────────────────┤
│ Backtest/Simulation  │        ✅        │      ✅       │       ✅        │
│ Training Service     │        ✅        │      ✅       │       ✅        │
│ Artifact Builder     │        ✅        │      ✅       │       ✅        │
│ Live Execution       │        ❌        │      ✅       │       ✅        │
│ Agent Location       │       N/A        │   Customer    │    Customer     │
│ Cloud Location       │       Our        │    Our        │   Customer VPC  │
│ API Keys Location    │       N/A        │   Customer    │    Customer     │
│ SLA                  │     Standard     │   Standard    │    Custom       │
│ Support              │     Community    │   Priority    │    Dedicated    │
└──────────────────────┴──────────────────┴───────────────┴─────────────────┘
```

---

## 8. Telemetry & Privacy

### 8.1 Telemetry Levels

| Level | Data Sent to Cloud | Purpose |
|-------|-------------------|---------|
| `AGGREGATED` | Metrics only (latency, error rate, uptime) | Service health monitoring |
| `DETAILED_NON_SENSITIVE` | Strategy performance (anonymized P&L curves, Sharpe) | Platform improvement |
| `RAW_ORDER_EVENTS` | Not sent (by default) | Enterprise opt-in only |

### 8.2 Data That Stays in Agent (by design)

| Data Type | Reason |
|-----------|--------|
| Broker API keys | Security |
| Exchange credentials | Security |
| Raw order events | Privacy + regulatory |
| Exact position sizes | Privacy |
| Account balances | Privacy |
| Individual trade details | Privacy |

### 8.3 GDPR Alignment

**Status**: Designed to support GDPR requirements (privacy-by-design; not independently audited) - See [GDPR_COMPLIANCE_SUMMARY.md](../compliance/GDPR_COMPLIANCE_SUMMARY.md)

| Requirement | Implementation |
|-------------|----------------|
| Data minimization (Art. 5) | Telemetry contracts, CI guardrails |
| Storage limitation (Art. 5) | Auto-purge with legal hold |
| Purpose limitation (Art. 5) | RoPA documented |
| Right to access (Art. 15) | DSAR workflow documented (30-day target per GDPR; actual terms per executed agreements) |
| Right to erasure (Art. 17) | DSAR with legal hold check |
| Data portability (Art. 20) | JSON export |
| Privacy by design (Art. 25) | CCEA architecture |
| Processor obligations (Art. 28) | DPA template |
| Security (Art. 32) | Encryption, RBAC, break-glass |
| Breach notification (Art. 33-34) | 72-hour workflow |
| EU data residency | By design for EU customers; drift checks designed to fail closed (verify via residency dashboard and CI tests) |

**CCEA Privacy Design Commitments** *(verify via architecture review and CI tests)*:
- Cloud is **designed not to** store or receive broker credentials or API keys (secrets designed to stay in customer-controlled Agent)
- Cloud is **designed not to** receive order-like payloads in commands (protocol-level design prohibition)
- Telemetry redaction is **on by default** (mandatory by design); raw order events require explicit opt-in
- DSAR scope is Cloud-only; Agent data is customer-controlled

See [CCEA_PRIVACY.md](./CCEA_PRIVACY.md) for full privacy architecture.

---

## 9. Risk Management Architecture

### 9.1 Two-Layer Risk System

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      TWO-LAYER RISK SYSTEM                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LAYER 1: CLOUD (Research/Training Phase)                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  • CVaR-aware policy optimization                                    │  │
│   │  • Backtest drawdown analysis                                        │  │
│   │  • Sim-to-live parity monitoring                                     │  │
│   │  • Strategy risk scoring                                             │  │
│   │                                                                       │  │
│   │  Purpose: Train strategies that are inherently risk-aware            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   LAYER 2: AGENT (Execution Phase) - ENFORCEMENT                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  • Position size limits (hard caps)                                  │  │
│   │  • Drawdown limits (portfolio and strategy level)                    │  │
│   │  • Concentration limits (per asset)                                  │  │
│   │  • Order rate limits (orders per minute)                             │  │
│   │  • Kill Switch (manual and automatic)                                │  │
│   │                                                                       │  │
│   │  Purpose: ENFORCE limits regardless of what strategy requests        │  │
│   │  Note: These limits are set LOCALLY by customer, not by Cloud        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   KEY PRINCIPLE: Cloud SUGGESTS risk parameters. Agent ENFORCES them.       │
│   Customer can override Cloud suggestions with stricter local limits.       │
└────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Kill Switch Hierarchy

| Level | Trigger | Action |
|-------|---------|--------|
| **Strategy Kill** | Strategy-level drawdown exceeded | Stop specific strategy, close positions |
| **Agent Kill** | Agent-level loss exceeded | Stop all strategies, close all positions |
| **Manual Kill** | User presses kill switch | Immediate halt, cancel pending orders |
| **Broker Kill** | Broker margin call | Positions liquidated by broker |

---

## 10. Open-Source Strategy

### 10.1 Open Components (MIT License)

| Component | Repository | Purpose |
|-----------|------------|---------|
| **CCEA Agent** | `ccea-agent` | Customer-deployed execution runtime |
| **CCEA SDK** | `ccea-sdk` | Python client library for Cloud API |
| **Broker Connectors** | Part of Agent | Exchange/broker integrations |
| **Examples** | Part of SDK | Example notebooks and strategies |

### 10.2 Proprietary Components (Trade Secret)

| Component | Protection | Purpose |
|-----------|------------|---------|
| Training Service | Trade Secret | CVaR-RL training |
| Backtest Engine | Trade Secret | L3 LOB simulation |
| Artifact Builder | Trade Secret | Strategy packaging |
| Control Plane | Trade Secret | Agent orchestration |
| Feature Pipeline | Trade Secret | 63+ proprietary features |

### 10.3 Benefits of Open Agent

1. **Customer Trust** - Can audit execution code
2. **Regulatory Clarity** - Customer owns and runs execution
3. **Community Contribution** - Broker connectors, bug fixes
4. **Lower CAC** - Self-service evaluation
5. **Talent Pipeline** - Hire from contributor community

---

## 11. Marketing & Communication Guidelines

### 11.1 Approved Language

| Context | Approved Phrasing |
|---------|-------------------|
| Product description | "AI-powered quantitative research and simulation platform" |
| Execution model | "Customer-controlled execution via local Agent" |
| Deployment | "Deploy strategies to your own infrastructure" |
| Relationship | "We provide tools, you control execution" |
| Risk | "Built-in risk management with customer-configurable limits" |

### 11.2 Prohibited Language

| Never Say | Why |
|-----------|-----|
| "We trade for you" | Implies broker/advisor relationship |
| "Cloud-side execution" | Incorrect - Agent executes, not Cloud |
| "Guaranteed profit" | Financial advice, illegal |
| "No risk" | Misleading |
| "We manage your portfolio" | Investment advisor language |
| "Our algorithm places orders" | Incorrect - Agent places orders |

### 11.3 Disclaimer Requirements

All marketing materials must include:

> **Disclaimer:** CCEA Platform provides software tools for quantitative research and simulation. We are not a broker, investment advisor, or portfolio manager. All trading decisions and execution are made by the customer using their own infrastructure and accounts. Past performance does not guarantee future results. Trading involves risk of loss.

---

## 12. Terms of Service Guidelines

### 12.1 Required ToS Provisions

| Provision | Language |
|-----------|----------|
| **Not Investment Advice** | "CCEA Platform does not provide investment advice. All content is for informational and educational purposes only." |
| **Not a Broker** | "We are not a broker, dealer, or custodian. We do not execute trades or hold customer funds." |
| **Execution Responsibility** | "All trade execution occurs in the customer's environment using customer's accounts and credentials. Customer is solely responsible for trading decisions." |
| **Risk Acknowledgment** | "Customer acknowledges that trading involves substantial risk of loss and may not be suitable for all investors." |
| **Regulatory Compliance** | "Customer is responsible for compliance with all applicable laws and regulations in their jurisdiction." |

### 12.2 Liability Limitations

| Scenario | Our Liability |
|----------|---------------|
| Strategy underperformance | None - customer's trading decision |
| Broker execution issues | None - customer's broker relationship |
| API key compromise | None if compromise outside our systems |
| Agent malfunction | Limited to platform fees paid |
| Cloud service outage | SLA credits only |

---

## 13. Regulatory Positioning

### 13.1 What We Are

| Classification | Rationale |
|----------------|-----------|
| **Software Provider** | We provide tools, not financial services |
| **Technology Licensor** | We license software for customer's use |
| **SaaS Platform** | We host research and simulation services |

### 13.2 What We Are NOT

| Classification | Why Not |
|----------------|---------|
| **Investment Advisor** | We don't provide investment advice |
| **Broker/Dealer** | We don't execute trades or hold funds |
| **Custodian** | We don't hold customer assets |
| **Exchange** | We don't match buyers and sellers |
| **Portfolio Manager** | We don't make investment decisions |

### 13.3 Jurisdictional Notes

| Jurisdiction | Positioning |
|--------------|-------------|
| **EU/MiFID II** | Software provider, not investment firm |
| **US/SEC** | Technology vendor, not broker-dealer |
| **UK/FCA** | Technology provider, not authorized firm |
| **Singapore/MAS** | Technology provider, not licensed financial advisor |

---

## 14. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01 | CCEA Team | Initial architecture |
| 2.0 | 2025-06 | CCEA Team | Added Agent architecture |
| 3.0 | 2025-12 | CCEA Team | Complete rewrite per Design Doc |

---

## 15. Related Documents

### Architecture Documentation

| Document | Purpose |
|----------|---------|
| [CCEA_DATA_MODEL.md](./CCEA_DATA_MODEL.md) | Database entities and relationships |
| [CCEA_STATE_MACHINE.md](./CCEA_STATE_MACHINE.md) | Deployment and Run state machines |
| [CCEA_PROTOCOL.md](./CCEA_PROTOCOL.md) | Agent↔Cloud communication protocol |
| [CCEA_ROLLOUT_PLAN.md](./CCEA_ROLLOUT_PLAN.md) | Implementation phases and open questions |
| [CCEA_CI_GUARDRAILS.md](./CCEA_CI_GUARDRAILS.md) | CI/CD security guardrails |
| [CCEA_PRIVACY.md](./CCEA_PRIVACY.md) | Privacy, GDPR, and data governance |

### Agent Documentation

| Document | Purpose |
|----------|---------|
| [Agent README](../agent/README.md) | Agent zone overview |
| [INSTALLATION.md](../agent/INSTALLATION.md) | Agent installation guide |
| [LOCAL_VAULT.md](../agent/LOCAL_VAULT.md) | Credential storage |
| [APPROVALS.md](../agent/APPROVALS.md) | Local approval system |
| [RISK_CONTROLS.md](../agent/RISK_CONTROLS.md) | Policy firewall and hard caps |
| [DEGRADED_MODES.md](../agent/DEGRADED_MODES.md) | Safe degradation handling |

### Cloud Documentation

| Document | Purpose |
|----------|---------|
| [Cloud README](../cloud/README.md) | Cloud zone overview |
| [CONTROL_PLANE_API.md](../cloud/CONTROL_PLANE_API.md) | REST API reference |
| [ARTIFACT_BUILDER.md](../cloud/ARTIFACT_BUILDER.md) | Build and signing guide |
| [GOVERNANCE.md](../cloud/GOVERNANCE.md) | RBAC, retention, residency |
| [EVIDENCE_PACK.md](../cloud/EVIDENCE_PACK.md) | Enterprise audit export |
| [ENTERPRISE.md](../cloud/ENTERPRISE.md) | Enterprise deployment |

### Business Documentation

| Document | Purpose |
|----------|---------|
| [CCEA_MARKETING_GUIDELINES.md](../business/CCEA_MARKETING_GUIDELINES.md) | Approved marketing language |
| [CCEA_TERMS_OF_SERVICE_GUIDELINES.md](../business/CCEA_TERMS_OF_SERVICE_GUIDELINES.md) | ToS requirements |
| [OPEN_CORE_BUSINESS_MODEL.md](../business/OPEN_CORE_BUSINESS_MODEL.md) | Business model details |
| [IP_PROTECTION_STRATEGY.md](../business/IP_PROTECTION_STRATEGY.md) | IP and trade secret protection |
| [PRICING_DIFFERENTIATION_STRATEGY.md](../business/PRICING_DIFFERENTIATION_STRATEGY.md) | Pricing by product mode |
| [COMPETITIVE_MOAT.md](../business/COMPETITIVE_MOAT.md) | Competitive advantage analysis |

### Operational Documentation

| Document | Purpose |
|----------|---------|
| [Runbooks README](../runbooks/README.md) | Operational procedures |
| [KILL_SWITCH.md](../runbooks/KILL_SWITCH.md) | Emergency halt procedures |
| [INCIDENT_RESPONSE.md](../runbooks/INCIDENT_RESPONSE.md) | Incident handling |
| [RECOVERY.md](../runbooks/RECOVERY.md) | Recovery procedures |
| [Protocol Schemas](../schemas/README.md) | JSON schema reference |

---

**Document Classification:** INTERNAL - Technical Reference
**Owner:** CTO
**Review Cycle:** Quarterly
**Next Review:** Q2 2025

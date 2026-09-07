# CCEA Rollout Plan

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16
>
> **Reference**: Design Doc CCEA Cloud.txt (canonical source) - Section 20-21

## Overview

This document defines the phased rollout plan for the CCEA Platform and tracks open questions that require resolution before production deployment.

---

## 1. Rollout Phases

### Phase 1: Skeleton

**Goal:** Basic infrastructure and connectivity

| Component | Deliverable | Status |
|-----------|-------------|--------|
| Agent enrollment | Enrollment token flow | Complete |
| Agent heartbeat | Health reporting to Cloud | Complete |
| Artifact Builder | Sign and publish artifacts | Complete |
| Agent pull | Fetch artifact by digest | Complete |
| Hello Strategy | Run "hello world" (no broker) | Complete |

**Success Criteria:**

- [ ] Agent can enroll with Cloud using token
- [ ] Heartbeat visible in Cloud dashboard
- [ ] Artifact builds and signs successfully
- [ ] Agent verifies signature before running
- [ ] "Hello Strategy" prints output in Agent logs

**No broker connectivity in this phase.**

---

### Phase 2: Lifecycle Requests + Local Approvals

**Goal:** Command protocol and approval flow

| Component | Deliverable | Status |
|-----------|-------------|--------|
| Control Plane commands | Full command protocol | Complete |
| Local Approval UI/CLI | Approval interface | Complete |
| Deployment state machine | State tracking | Complete |
| Run state machine | Run lifecycle | Complete |
| Command idempotency | Deduplication | Complete |

**Success Criteria:**

- [ ] Cloud can send REQUEST_START_RUN
- [ ] Agent shows approval prompt locally
- [ ] User can approve/reject in CLI/UI
- [ ] Approval evidence recorded
- [ ] State transitions visible in Cloud
- [ ] Commands are idempotent (retry-safe)

**Commands implemented:**

- REQUEST_START_RUN
- REQUEST_STOP_RUN
- REQUEST_PAUSE_RUN
- REQUEST_RESUME_RUN
- REQUEST_UPGRADE_ARTIFACT

---

### Phase 3: Live Execution (Local) + Risk Enforcement

**Goal:** Full trading capability in Agent

| Component | Deliverable | Status |
|-----------|-------------|--------|
| Broker connectors | Binance, Alpaca adapters | Complete |
| Local Vault | Credential storage | Complete |
| Risk Manager | Pre-trade controls | Complete |
| Kill Switch | Emergency halt | Complete |
| Reconciliation | Position/order sync | Complete |
| Idempotency | Client order IDs | Complete |

**Success Criteria:**

- [ ] Credentials stored in Local Vault
- [ ] Strategy produces Intent locally
- [ ] Risk Manager validates Intent
- [ ] Order created and sent from Agent
- [ ] Kill switch halts on threshold breach
- [ ] Restart reconciles with broker state
- [ ] No duplicate orders after retry

**Broker Connectors:**

- Binance Spot/Futures
- Alpaca Stocks
- OANDA Forex
- Deribit Options (beta)

---

### Phase 4: Telemetry + Alerts + Privacy Defaults

**Goal:** Observability with privacy protection

| Component | Deliverable | Status |
|-----------|-------------|--------|
| Telemetry buffer | SQLite local buffer | Complete |
| Redaction middleware | Mandatory filtering | Complete |
| AGGREGATED telemetry | Default level | Complete |
| Cloud dashboards | Metrics visualization | Complete |
| Alerting | Threshold-based alerts | Complete |
| Retention policies | Configurable retention | Complete |

**Success Criteria:**

- [ ] Telemetry buffered locally
- [ ] All telemetry passes through redaction
- [ ] AGGREGATED is default (no raw orders)
- [ ] Dashboards show PnL, drawdown, health
- [ ] Alerts fire on configured thresholds
- [ ] Data deleted after retention period

**Privacy Features:**

- Mandatory redaction middleware
- No RAW_ORDER_EVENTS level
- Configurable telemetry level
- EU data residency option

---

### Phase 5: Enterprise Pack

**Goal:** Features designed for enterprise adoption

| Component | Deliverable | Status |
|-----------|-------------|--------|
| On-prem mode | Self-hosted deployment | Complete |
| Evidence pack export | Audit artifacts | Complete |
| RBAC/audit access | Role-based control | Complete |
| Version pinning | Agent version lock | Complete |
| Air-gap support | Offline operation | In Progress |
| Custom SLAs | Contract support | In Progress |

**Success Criteria:**

- [ ] Full stack deploys in customer VPC
- [ ] Evidence pack exports to customer storage
- [ ] Access audit log captures all reads
- [ ] Enterprise can pin Agent version
- [ ] Air-gapped mode works without Cloud
- [ ] Custom SLA terms implemented

**Enterprise Features:**

- On-premises deployment
- Air-gapped mode
- Evidence pack export
- Custom retention policies
- Dedicated support

---

## 2. Open Questions

### 2.1 Must Resolve Before Production

| # | Question | Options | Status |
|---|----------|---------|--------|
| 1 | **Minimum sandbox for retail Agent** | Process-only vs Docker-required | **Decision: Process-only** (Docker optional) |
| 2 | **Raw order telemetry level** | Enterprise-only vs disabled entirely | **Decision: Disabled** (can enable per contract) |
| 3 | **"Flatten position" as remote request** | Allow with approval vs local-only | **Decision: Local-only** by default |
| 4 | **"Remote brain, local finger" mode** | Cloud inference → local execution | **DEFERRED** - Requires legal review |
| 5 | **Personal data definition** | What's GDPR sensitive? | **Decision: See Privacy doc** |
| 6 | **Customer-managed host verification** | How to verify VPS is user-controlled? | **Decision: Attestation + ToS** |

### 2.2 Detailed Analysis

#### Q1: Minimum Sandbox for Retail Agent

**Context:** Should we require Docker for all retail users, or allow process-only sandbox?

**Decision:** Process-only is acceptable for retail

- Docker optional but recommended
- Process sandbox provides basic isolation
- Resource limits via OS controls
- Enterprise can mandate Docker

**Rationale:**

- Lower barrier to entry for retail
- Docker not available on all systems
- Process sandbox sufficient for trusted user code

#### Q2: Raw Order Telemetry

**Context:** Should we ever allow raw order events in telemetry?

**Decision:** Disabled by default, enterprise-only with contract

- `RAW_ORDER_EVENTS` level does not exist in protocol
- Enterprise can request via custom contract
- Requires explicit data processing agreement
- Additional legal review required

**Rationale:**

- Privacy risk: order data is highly sensitive
- IP risk: reveals strategy logic
- Regulatory risk: could be construed as advisory data

#### Q3: Remote Flatten Position

**Context:** Can Cloud request Agent to flatten all positions?

**Decision:** Local-only by default

- No `FLATTEN_POSITION` command in protocol
- Kill switch can flatten (local decision)
- Enterprise can enable via policy
- Requires explicit local configuration

**Rationale:**

- Flatten is a trading action
- Cloud should not control trading
- Local kill switch provides this capability
- Enterprise may need for compliance

#### Q4: Remote Brain, Local Finger

**Context:** Pattern where Cloud runs inference, Agent executes

```
Cloud: "The model says BUY 100 AAPL"
Agent: Receives intent, creates order
```

**Decision:** DEFERRED

- Legally/reputation risky
- Could be construed as Cloud sending orders
- Requires thorough legal review
- May pursue in future with proper structuring

**Concerns:**

- "Intent" vs "Order" distinction blurs
- Cloud becomes de facto decision maker
- Regulatory risk increases significantly

#### Q5: Personal Data Definition

**Context:** What data is GDPR "personal data" for our platform?

**Decision:** Conservative interpretation

- Email, name, IP address: Personal
- Trading activity: Personal (reveals behavior)
- Strategy performance: Pseudonymous (with user ID)
- Aggregated metrics: Not personal (no identification)

**Implementation:**

- See [CCEA_PRIVACY.md](./CCEA_PRIVACY.md)
- DPA template includes data categories
- Retention policies per data type

#### Q6: Customer-Managed Host Verification

**Context:** How do we verify user actually controls their VPS?

**Decision:** Attestation + ToS

- User attests in ToS they control the host
- Agent generates device key on first run
- No verification of VPS ownership required
- Enterprise: can require attestation documents

**Rationale:**

- We cannot technically verify VPS ownership
- ToS places responsibility on user
- Device key proves consistent agent identity
- Enterprise can add additional requirements

---

## 3. Sequence Diagrams

### 3.1 Enrollment

```
User (Cloud UI)         Cloud                      Agent                User (Agent host)
      │                    │                          │                       │
      │ Request token      │                          │                       │
      │───────────────────►│                          │                       │
      │                    │                          │                       │
      │◄───────────────────│                          │                       │
      │ Token (TTL 15min)  │                          │                       │
      │                    │                          │                       │
      │ Copy token to agent host                      │                       │
      │──────────────────────────────────────────────────────────────────────►│
      │                    │                          │                       │
      │                    │                          │ agent enroll --token  │
      │                    │                          │◄──────────────────────│
      │                    │                          │                       │
      │                    │ enroll(public_key, token)│                       │
      │                    │◄─────────────────────────│                       │
      │                    │                          │                       │
      │                    │ Validate token           │                       │
      │                    │ Create agent record      │                       │
      │                    │ Store public key         │                       │
      │                    │                          │                       │
      │                    │ enrolled(agent_id)       │                       │
      │                    │─────────────────────────►│                       │
      │                    │                          │                       │
      │                    │◄─────────────────────────│                       │
      │                    │ HEARTBEAT                │                       │
      │                    │                          │                       │
      │ Agent visible      │                          │                       │
      │◄───────────────────│                          │                       │
```

### 3.2 Deploy & Start (with Approval)

```
User (Cloud UI)         Cloud                   Builder              Agent           User (Local)
      │                    │                       │                   │                  │
      │ Deploy(strategy,   │                       │                   │                  │
      │  agent, mode=LIVE) │                       │                   │                  │
      │───────────────────►│                       │                   │                  │
      │                    │                       │                   │                  │
      │                    │ build+sign(version)   │                   │                  │
      │                    │──────────────────────►│                   │                  │
      │                    │                       │                   │                  │
      │                    │ artifact(digest)      │                   │                  │
      │                    │◄──────────────────────│                   │                  │
      │                    │                       │                   │                  │
      │                    │                       │ publish(registry) │                  │
      │                    │                       │                   │                  │
      │                    │ Create deployment     │                   │                  │
      │                    │ Create command        │                   │                  │
      │                    │                       │                   │                  │
      │                    │ REQUEST_START_RUN     │                   │                  │
      │                    │ (via poll)            │                   │                  │
      │                    │───────────────────────────────────────────►                  │
      │                    │                       │                   │                  │
      │                    │                       │                   │ Show approval    │
      │                    │                       │                   │ diff + prompt    │
      │                    │                       │                   │─────────────────►│
      │                    │                       │                   │                  │
      │                    │                       │                   │                  │ Approve
      │                    │                       │                   │◄─────────────────│
      │                    │                       │                   │                  │
      │                    │ COMMAND_APPROVAL      │                   │                  │
      │                    │◄──────────────────────────────────────────│                  │
      │                    │                       │                   │                  │
      │                    │                       │      pull(digest) │                  │
      │                    │                       │◄──────────────────│                  │
      │                    │                       │                   │                  │
      │                    │                       │      artifact     │                  │
      │                    │                       │──────────────────►│                  │
      │                    │                       │                   │                  │
      │                    │                       │                   │ verify signature │
      │                    │                       │                   │ start run        │
      │                    │                       │                   │                  │
      │                    │ COMMAND_RESULT        │                   │                  │
      │                    │ (APPLIED, run_id)     │                   │                  │
      │                    │◄──────────────────────────────────────────│                  │
      │                    │                       │                   │                  │
      │                    │ TELEMETRY(AGGREGATED) │                   │                  │
      │                    │◄──────────────────────────────────────────│                  │
      │                    │                       │                   │                  │
      │ Deployment ACTIVE  │                       │                   │                  │
      │◄───────────────────│                       │                   │                  │
```

### 3.3 Upgrade Build (TRADING_IMPACTING)

```
User (Cloud UI)         Cloud                      Agent              User (Local)
      │                    │                          │                     │
      │ Request upgrade    │                          │                     │
      │ to new version     │                          │                     │
      │───────────────────►│                          │                     │
      │                    │                          │                     │
      │                    │ REQUEST_UPGRADE_ARTIFACT │                     │
      │                    │ change_class:            │                     │
      │                    │ TRADING_IMPACTING        │                     │
      │                    │────────────────────────► │                     │
      │                    │                          │                     │
      │                    │                          │ Show diff:          │
      │                    │                          │ - Model changed     │
      │                    │                          │ - Params changed    │
      │                    │                          │ [Approve] [Reject]  │
      │                    │                          │─────────────────────►
      │                    │                          │                     │
      │                    │                          │                     │ Approve
      │                    │                          │◄────────────────────│
      │                    │                          │                     │
      │                    │ COMMAND_APPROVAL         │                     │
      │                    │ (APPROVED)               │                     │
      │                    │◄─────────────────────────│                     │
      │                    │                          │                     │
      │                    │                          │ 1. Stop old run     │
      │                    │                          │ 2. Pull new artifact│
      │                    │                          │ 3. Verify signature │
      │                    │                          │ 4. Start new run    │
      │                    │                          │                     │
      │                    │ COMMAND_RESULT(APPLIED)  │                     │
      │                    │◄─────────────────────────│                     │
      │                    │                          │                     │
      │ Upgrade complete   │                          │                     │
      │◄───────────────────│                          │                     │
```

### 3.4 Kill Switch Activation

```
                         Agent                    Broker               Cloud
                           │                        │                    │
   [Max daily loss exceeded]                        │                    │
                           │                        │                    │
                           │ 1. Trigger kill switch │                    │
                           │                        │                    │
                           │ CANCEL_ALL_ORDERS      │                    │
                           │───────────────────────►│                    │
                           │                        │                    │
                           │ Orders cancelled       │                    │
                           │◄───────────────────────│                    │
                           │                        │                    │
                           │ 2. Optional: Flatten   │                    │
                           │ (if local policy)      │                    │
                           │                        │                    │
                           │ 3. Halt run            │                    │
                           │ State → HALTED         │                    │
                           │                        │                    │
                           │ 4. Log halt evidence   │                    │
                           │                        │                    │
                           │ TELEMETRY(HALT)        │                    │
                           │ reason: MAX_DAILY_LOSS │                    │
                           │────────────────────────────────────────────►│
                           │                        │                    │
                           │                        │                    │ Create alert
                           │                        │                    │ Notify user
                           │                        │                    │
```

---

## 4. Milestones & Timeline

| Phase | Target | Dependencies |
|-------|--------|--------------|
| Phase 1 | Complete | - |
| Phase 2 | Complete | Phase 1 |
| Phase 3 | Complete | Phase 2 |
| Phase 4 | Complete | Phase 3 |
| Phase 5 | In Progress | Phase 4 |

**Note:** Timelines are implementation-order only. Actual dates depend on team capacity and priorities.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | CCEA Team | Initial rollout plan per Design Doc |

---

**Related Documentation:**

- [CCEA Overview](./CCEA_OVERVIEW.md)
- [State Machine](./CCEA_STATE_MACHINE.md)
- [Protocol](./CCEA_PROTOCOL.md)

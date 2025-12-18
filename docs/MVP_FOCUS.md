# CustodiaCloud MVP Focus: Core Use-Case Definition

*What We're Building First — And Why*

---

## Architecture: CCEA (Cloud-Controlled Execution Architecture)

> **Key Principle**: This MVP follows the **CCEA architecture** where Cloud handles research/simulation/monitoring, while Agent (running locally in customer environment) handles all live execution, secrets, and order creation.

| Component | Responsibility | Secrets | Orders |
|-----------|---------------|---------|--------|
| **Cloud** | Research, backtesting, artifact build, monitoring, lifecycle commands | **NEVER** | **NEVER** |
| **Agent** | Live execution, local vault, risk enforcement, order creation/sending | **LOCAL ONLY** | **YES** |

**Legal Posture**: We are a **Software Provider / ICT Provider**, NOT an investment adviser, broker-dealer, or execution service.

## Asset Coverage (Foundation vs MVP)

**Foundation (multi-asset by design)**: listed **equities**, listed **futures**, listed **options**, **FX**, and **digital assets** (spot/perpetuals) as an optional expansion path.

**MVP / Beachhead (equities-first)**: default production support and positioning start with listed equities. Adjacent asset classes remain available as an expansion path, but are not a default MVP support commitment.

## Regulatory Posture (Design Intent)

| Framework | What customers need | How CustodiaCloud supports | What we do not do |
|----------|----------------------|----------------------------|-------------------|
| **MiFID II** (and EU algo trading expectations) | Controls + governance + testing evidence | CCEA separation, local approvals for trading-impacting changes, risk controls/kill switch, audit trails & exports | No custody, no client secrets in Cloud, no Cloud live trading instructions, no execution on behalf of clients |
| **GDPR** | Privacy-by-design, minimization, retention, EU residency | Telemetry sensitivity levels, redaction, tenant isolation, retention/DSAR hooks, EU-region defaults | No collection of unnecessary personal data; no secrets in telemetry |
| **DORA** | Vendor risk assessment, operational resilience evidence | Evidence exports, change control posture, incident/runbook documentation, roadmap for enterprise controls | Not claiming “DORA certified”; clients run their vendor due diligence |
| **EU AI Act** | AI governance & transparency posture | Model/version provenance, logging/auditability, human control via local approvals, avoid “personalized recommendations” posture | Not positioning as an AI adviser; no claims about risk classification without legal review |

---

## The One Problem We Solve

**Proprietary trading firms spend 6-12 months building infrastructure before deploying their first strategy.**

This is our singular focus. Everything in MVP serves this problem — while maintaining strict Cloud/Agent separation for regulatory safety.

---

## Target Customer Profile

### Primary Persona: The Quant CTO

| Attribute | Profile |
|-----------|---------|
| **Title** | CTO, Head of Technology, Lead Developer |
| **Company** | Systematic equities team (prop firm or small fund) |
| **Location** | EU (Netherlands, Germany, Ireland, France) |
| **Background** | Quantitative finance, software engineering |
| **Experience** | Built trading systems at banks or hedge funds |
| **Current State** | Evaluating build vs. buy for new firm/strategy |

### Their Day-to-Day Pain

| Pain Point | Frequency | Intensity |
|------------|-----------|-----------|
| Building execution infrastructure from scratch | Every new firm | High |
| Implementing risk management that satisfies compliance | Every system | High |
| Backtesting with realistic execution simulation | Daily | Medium |
| Managing broker + market data integrations | Ongoing | Medium |
| Explaining technical architecture to non-technical partners | Weekly | Low |

### Jobs to Be Done

1. **Primary Job**: Get a new quantitative strategy live in production with proper risk controls
2. **Secondary Job**: Validate strategy performance before committing capital
3. **Tertiary Job**: Demonstrate to investors/partners that infrastructure is institutional-grade

---

## Product Modes (CCEA Architecture)

### Three Modes Aligned with CCEA

| Mode | Description | Cloud Role | Agent Role |
|------|-------------|------------|------------|
| **Research SaaS (Pro)** | EU-friendly research + simulation | Full (IDE, backtest, sim) | Optional (for live) |
| **Pro Live via Customer Agent** | Auto-execution in customer environment, cloud observability | Lifecycle, Telemetry | Local vault + execution |
| **Enterprise Engine** | On-prem/VPC, all in customer infra | Self-hosted option | HSM/KMS, air-gapped |

**MVP Focus**: **Research SaaS (Pro)** + **Pro Live via Customer Agent** for **systematic equities** (equities-first).

---

## MVP Scope Definition

### In Scope (Must Have)

| Feature | Why Essential | Customer Value | CCEA Zone |
|---------|---------------|----------------|-----------|
| **Cloud Research IDE** | Strategy development | Days, not months to research | Cloud |
| **Backtest & Simulation** | Strategy validation | Confidence before capital | Cloud |
| **Artifact Builder (signed)** | Immutable deployable strategies | Version control + audit | Cloud |
| **Agent with Local Vault** | Secure key storage | Keys never leave user's machine | Agent |
| **Risk-aware execution** | Regulatory expectation | MiFID II-oriented controls + evidence | Agent |
| **CVaR-constrained optimization** | Key differentiator | Tail-risk constrained optimization | Cloud (training) |
| **Real-time monitoring** | Operational necessity | Know what's happening | Cloud (telemetry) |
| **Local approval for TRADING_IMPACTING** | Regulatory safety | User controls trading changes | Agent |
| **Kill switch** | Safety requirement | Emergency halt | Agent |

### Out of Scope (Deferred)

Note: the platform is **multi-asset by design**. MVP commercial support and positioning remain **equities-first**; additional asset classes are deferred as a default support commitment until validated customer pull.

| Feature | Why Deferred | Reintroduce When |
|---------|--------------|------------------|
| Digital assets (spot/perpetuals) | Not in equities-first narrative | Only if enterprise demand exists |
| CME Futures (IB) | Complex, institutional-only | After 10 paying customers |
| Options pricing | Specialized user base | Customer requests (3+) |
| L3 LOB simulation | Advanced research feature | Power user demand |
| Additional brokers beyond IBKR | Integration + support burden | After repeatable onboarding |
| Multi-strategy orchestration | Complexity | Single-strategy validated |
| Managed Agent Hosting | Separate legal review required | Enterprise contracts |
| Copy-trading / Social trading | Heavy advice/portfolio regulations | NOT planned |

### The MVP Feature Boundary (CCEA Architecture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MVP BOUNDARY                                   │
│                                                                          │
│  ┌────────────────────── CLOUD ZONE ──────────────────────┐             │
│  │  Research IDE ──► Backtest/Sim ──► Artifact Builder    │             │
│  │       │                │                  │            │             │
│  │       ▼                ▼                  ▼            │             │
│  │  Strategy Dev     CVaR Training      Sign + Publish    │             │
│  │  Notebooks        Historical Sim     Immutable Digest  │             │
│  │                                                        │             │
│  │  Control Plane: REQUEST_START, REQUEST_STOP, etc.      │             │
│  │  Telemetry Ingestion: Aggregated metrics (redacted)    │             │
│  └────────────────────────────────────────────────────────┘             │
│                              │                                           │
│                              │ Lifecycle Commands (NOT orders)           │
│                              ▼                                           │
│  ┌────────────────────── AGENT ZONE ──────────────────────┐             │
│  │  Local Vault ──► Risk Manager ──► Broker Connector     │             │
│  │       │                │                  │            │             │
│  │       ▼                ▼                  ▼            │             │
│  │  API Keys         CVaR Limits        Broker API        │             │
│  │  (encrypted)      Kill Switch        Order Creation    │             │
│  │                   Local Approval     Order Sending     │             │
│  └────────────────────────────────────────────────────────┘             │
│                                                                          │
│  ════════════════════════ DEFERRED ════════════════════════════════════ │
│                                                                          │
│  [ Digital assets ] [ Options ] [ CME ] [ L3 LOB ] [ Managed Agent Hosting ] │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Value Proposition (MVP)

### For European Prop Trading Firms

**Before Our Product:**
- 6-12 months to build execution infrastructure
- €200K-500K in development costs
- Risk management built as afterthought
- Backtesting accuracy questionable
- Regulatory uncertainty (who controls trading?)

**After Our Product:**
- Days to first live strategy (via Cloud research + Agent deployment)
- €2,000-5,000/month subscription
- Risk management built-in (CVaR, limits, kill switch)
- Research-grade execution simulation
- **Clear regulatory posture**: Cloud = research tools, Agent = YOUR execution

### The Pitch (30 Seconds)

> "CustodiaCloud helps systematic equity trading teams go from strategy idea to live trading in days, not months. Our Cloud platform handles research and simulation — but **your Agent** running on **your infrastructure** handles all live execution. Your keys never leave your machine. We start with **equities-first** validation in Europe."

### CCEA Value Proposition

| Stakeholder | Value |
|-------------|-------|
| **CTO** | Full-stack research infrastructure without building from scratch |
| **Compliance** | Clear boundary: Cloud = software tools, Agent = client-controlled execution |
| **Trader** | Fast iteration: research in Cloud, deploy signed artifacts to Agent |
| **Risk Manager** | Local risk enforcement, kill switch, approval workflow for changes |

---

## Success Metrics for MVP

### North Star Metric

**Time from signup to first live trade**

Target: < 1 week

### Supporting Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Activation Rate** | % completing onboarding | > 80% |
| **Time to First Backtest** | Hours from signup | < 4 hours |
| **Time to First Live Trade** | Days from signup | < 7 days |
| **Weekly Active Users** | Users with 1+ action/week | > 70% |
| **Strategies Deployed** | Live strategies per user | > 2 |
| **NPS Score** | Net Promoter Score | > 40 |

---

## What Success Looks Like

### Month 1: Validation

- 3-5 pilot firms onboarded
- 80%+ complete setup
- First feedback collected
- Top 3 friction points identified

### Month 3: Product-Market Fit Signals

- 70%+ weekly active rate maintained
- NPS > 40
- 2+ firms express willingness to pay
- Referral from existing pilot

### Month 6: Revenue Validation

- 3+ paying customers (€2K+/month each)
- < 20% monthly churn
- Clear feature roadmap from customer input
- Repeatable sales process documented

---

## Feature Prioritization Framework

### How We Decide What to Build Next

```
                    HIGH CUSTOMER DEMAND
                           │
              ┌────────────┼────────────┐
              │            │            │
              │   BUILD    │   BUILD    │
              │   NEXT     │   NOW      │
              │            │            │
    LOW ──────┼────────────┼────────────┼────── HIGH
    EFFORT    │            │            │       EFFORT
              │   CONSIDER │   DEFER    │
              │   LATER    │            │
              │            │            │
              └────────────┼────────────┘
                           │
                    LOW CUSTOMER DEMAND
```

### Validation Requirements for New Features

| Requirement | Threshold |
|-------------|-----------|
| Customer requests | 3+ independent requests |
| Revenue impact | Affects conversion or retention |
| Competitive necessity | Losing deals without it |
| Strategic alignment | Fits European market focus |

---

## Competitive Positioning (MVP)

### We Are NOT Competing With:

| Competitor | Their Focus | Why We're Different |
|------------|-------------|---------------------|
| QuantConnect | Retail algo traders | We target institutional |
| Alpaca | Broker/API | We're full platform |
| Trading Technologies | Established institutions | We serve emerging firms |
| In-house development | Custom everything | We reduce build time |

### Our MVP Positioning

**"The fastest path from quant strategy to live trading for European prop firms."**

We don't compete on features. We compete on time-to-market.

---

## Risk Mitigation

### What Could Go Wrong

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Customers need futures/options day 1 | Medium | High | Keep MVP equities-first; add only after clear demand |
| Pricing too high | Medium | Medium | Test €500/month pilot pricing |
| Regulatory concerns (MiFID II) | Low | High | Legal review before paid launch |
| Technical reliability issues | Low | High | Extensive testing already done |
| Competitive response | Low | Medium | Speed of execution, customer focus |

### Pivot Triggers

We will consider pivoting if:

1. **Asset class mismatch**: 70%+ of prospects require futures/options/FX beyond equities
2. **Price sensitivity**: Zero conversion at €2K/month after 10 prospects
3. **Feature gap**: Consistent loss to competitors on specific capability
4. **Market timing**: Market structure constraints require different initial venue/broker

---

## The Build vs. Validate Commitment

### What We Will Do

- Weekly customer conversations (minimum 3/week)
- Bi-weekly releases based on feedback
- Monthly business reviews with pilot customers
- Quarterly strategy assessment

### What We Will NOT Do

- Build features without customer validation
- Expand asset classes without proving current ones
- Add complexity before simplicity works
- Chase enterprise deals before SMB validation

---

## Appendix: Customer Interview Guide

### Questions for Discovery Calls

**Problem Exploration:**
1. Walk me through how you built your last trading system. What took the longest?
2. What's the most frustrating part of your current infrastructure?
3. How do you handle risk management today? What's missing?

**Solution Validation:**
4. If you could wave a magic wand, what would your ideal platform do?
5. What would make you switch from your current solution?
6. How much would you pay to save 6 months of development time?

**Competitive Landscape:**
7. What tools/platforms do you use today? What do you like/dislike?
8. Have you evaluated other solutions? Why did/didn't you choose them?
9. What would prevent you from using a third-party platform?

---

## CCEA Terminology (Reference)

| Term | Definition |
|------|------------|
| **Cloud** | Our SaaS services (research, backtesting, monitoring, control plane) |
| **Agent** | Client's runtime daemon in their environment (BYO host / VPS / on-prem) |
| **Strategy** | User code/model that produces Intent |
| **Intent** | High-level intention (target exposure/position), NOT a ready order |
| **Order** | Concrete broker instruction (created ONLY in Agent) |
| **Deployment** | Link between artifact + config + target Agent |
| **Run** | Specific strategy execution on Agent |
| **Command** | Lifecycle request from Cloud to Agent (REQUEST_START, REQUEST_STOP, etc.) |
| **TRADING_IMPACTING** | Change class requiring local approval (new version, mode switch, risk changes) |
| **NON_IMPACTING** | Change class that can auto-apply (log level, telemetry verbosity) |

---

## Related Documents

- [BEACHHEAD_MARKET_STRATEGY.md](BEACHHEAD_MARKET_STRATEGY.md) — Beachhead market selection analysis (Geoffrey Moore methodology)
- [LEAN_VALIDATION_STRATEGY.md](LEAN_VALIDATION_STRATEGY.md) — Customer-centric validation approach
- [PILOT_PROGRAM.md](PILOT_PROGRAM.md) — Structured pilot program design
- [PRODUCT_OVERVIEW.md](PRODUCT_OVERVIEW.md) — One-pager for pitches
- [design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt](design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt) — Master CCEA architecture document

---

*Document Version: 2.1*
*Last Updated: 2025-12-18*
*Owner: Product Team*
*Aligned with: Design Doc CCEA Cloud v1.0*

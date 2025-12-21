# CustodiaCloud Investor Brief

## Risk-First Systematic Equities Research & Deployment Platform

*Last Updated: 2025-12-21 | Pre-Seed Stage*

**Changelog:**
- 2025-12-21: **Due diligence corrections**: Replaced "✅ Complete" status markers with "Foundation implemented" and "Defined" per Canon §4.5 (avoid absolute claims). Affected: Technical Foundation and Customer Validation tables.
- 2025-12-19: **Internal due diligence review corrections**: Changed "Now Testing with Customers" to "Entering Customer Validation Phase" to avoid implying active pilots; added explicit disclaimer to validation milestones clarifying targets are planned/aspirational and no customers/pilots/LOIs currently signed
- 2025-12-18: Updated regulatory posture, asset coverage

**Canonical positioning / safe wording**: see `docs/DOCUMENTATION_CANON_DESIGN.md`.

> **Document Status**: This brief describes an early-stage technology company. Financial projections are illustrative and forward-looking. The platform is technically mature but commercially pre-revenue. This document is for informational purposes only and does not constitute an offer to sell securities.

---

## Executive Summary

| Aspect | Status |
|--------|--------|
| **Stage** | Pre-seed, seeking seed funding |
| **Phase** | **Customer validation** (core foundation implemented) |
| **Revenue** | Pre-revenue; pilot program launching in **Phase 1 (0–3 months)** |
| **Ask** | Seed funding for customer validation & go-to-market |
| **Primary ICP** | European systematic equities teams (prop firms + small funds) |

**What we've built**: A trading infrastructure platform with built-in risk management designed to reduce time-to-market (target: days vs. industry-typical months; pending customer validation).

**Where we are now**: Core foundation implemented. We are entering the **customer validation phase** — planning pilot programs with European systematic equities teams to validate product-market fit before scaling. **Note**: No pilots are currently active; no signed LOIs, customers, or revenue exist as of this document's date.

**Why it matters**: Based on founder experience and informal industry conversations, prop trading firms may spend months and significant budget building trading infrastructure (actual figures vary widely by team size, scope, and region). We aim to reduce time-to-production significantly (target: days; pending customer case study validation).

---

## Architecture: Cloud-Controlled Execution Architecture (CCEA)

**Regulatory-First Design**: CustodiaCloud is a **software/ICT provider**. Cloud does not hold secrets and does not send live trading instructions (orders/targets/signals); execution remains customer-controlled via the Agent.

Our platform implements **CCEA** - a strict architectural separation designed to support regulatory clarity:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLOUD ZONE                                │
│        (Our Infrastructure - Research & Monitoring)              │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────┐ │
│  │  Research │  │  Builder  │  │  Control  │  │  Monitoring  │ │
│  │    IDE    │  │  Registry │  │   Plane   │  │  Telemetry   │ │
│  └───────────┘  └───────────┘  └───────────┘  └──────────────┘ │
│                                                                  │
│  Does NOT: store secrets │ execute orders │ send live instructions │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │ Lifecycle commands only
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT ZONE                                │
│        (User Infrastructure - Execution & Secrets)               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────┐ │
│  │   Local   │  │   Policy  │  │ Live Loop │  │    Broker    │ │
│  │   Vault   │  │  Firewall │  │   Runner  │  │  Connector   │ │
│  └───────────┘  └───────────┘  └───────────┘  └──────────────┘ │
│                                                                  │
│  All secrets local │ User controls hard caps │ Orders local only│
└─────────────────────────────────────────────────────────────────┘
```

**Why This Matters for Investors:**

| Benefit | Description |
|---------|-------------|
| **Regulatory Clarity** | Software / ICT provider posture under MiFID II (licensing depends on activities; Cloud does not execute orders or hold credentials/assets) |
| **Enterprise-Grade Design** | Designed to satisfy institutional security requirements (secrets designed to stay in user infra) |
| **Defensible Moat** | Complex architecture that competitors cannot easily replicate |
| **Multiple Revenue Streams** | B2B SaaS + Enterprise deployments (on-prem/VPC) |

**Legal positioning (design intent):**
- B2B software/ICT product; **not** investment advice, portfolio management, or trade recommendations
- **Not** an execution service: Cloud does not execute orders and does not send live trading instructions (orders/targets/signals); execution remains customer-controlled via the Agent
- Designed for a software/ICT provider posture: Cloud has **no secrets** and **no live trading instructions** (orders/targets/signals)
- Enterprise posture: auditability, change control, evidence exports (DORA-aware vendor requirements)
- Customers remain responsible for market data licensing/terms (bring-your-own data providers)

### Asset Coverage (Foundation vs MVP)

**Foundation (multi-asset by design)**: listed **equities**, listed **futures**, listed **options**, **FX**, and **digital assets** (spot/perpetuals) as an optional expansion path.

**MVP / Beachhead (equities-first)**: we lead with listed equities for repeatable onboarding; adjacent asset classes are enabled based on validated customer pull.

### Regulatory Posture (Design Intent)

| Framework | What customers need | How CustodiaCloud supports | What we do not do |
|----------|----------------------|----------------------------|-------------------|
| **MiFID II** (and EU algo trading expectations) | Controls + governance + testing evidence | CCEA separation, local approvals for trading-impacting changes, risk controls/kill switch, audit trails & exports | No client secrets/assets held in Cloud, no Cloud live trading instructions, execution remains customer-controlled via the Agent |
| **GDPR** | Privacy-by-design, minimization, retention, EU residency | Telemetry sensitivity levels, redaction, tenant isolation, retention/DSAR hooks, EU-region defaults | No collection of unnecessary personal data; no secrets in telemetry |
| **DORA** | Vendor risk assessment, operational resilience evidence | Evidence exports, change control posture, incident/runbook documentation, roadmap for enterprise controls | Not claiming certification; clients run their vendor due diligence |
| **EU AI Act** | AI governance & transparency posture | Model/version provenance, logging/auditability, human control via local approvals, avoid “personalized recommendations” posture | Not positioning as an AI adviser; no claims about risk classification without legal review |

---

## Current Phase: Lean Validation

### Foundation Built — Entering Customer Validation Phase

We have implemented the core technical foundation. Our focus now is entering the **customer validation phase** (structured pilot program planned for Phase 1), not feature expansion.

```
┌─────────────────────────────────────────────────────────────┐
│                    OUR JOURNEY                               │
│                                                              │
│   [✓] Technical       [→] Customer        [ ] Revenue       │
│       Foundation          Validation          Scale         │
│                                                              │
│   • Core platform     • Interviews (plan) • Paying clients  │
│   • 5 asset classes   • Pilot program     • Repeatable      │
│   • Risk management   • Feature freeze      sales          │
│   • Testing infra     • Iterate on PMF                      │
│                                                              │
│   COMPLETED           CURRENT PHASE       POST-VALIDATION   │
└─────────────────────────────────────────────────────────────┘
```

### Validation Milestones

| Milestone | Timeline | Success Criteria (Illustrative Target) |
|-----------|----------|------------------|
| Customer interviews (20+) | Phase 1 (0–1 months) (planned) | Target: Pain points validated and ranked |
| Pilot launch (3-5 teams) | Phase 1 (1–3 months) (planned) | Target: 80% complete pilot onboarding |
| Feature iteration | Phase 2 (3–6 months) (planned) | Target: Top 3 pilot requests addressed |
| Conversion validation | Phase 2 (3–6 months) (planned) | Target: Willingness-to-pay validated at target pricing range |
| First paying customers | Phase 2 (3–6 months) (planned) | Target: First pilot conversions (aspirational) |

> **Note**: These are **planned milestones** for a pre-revenue company. Customer validation has not yet commenced. Actual pilot acquisition, conversion rates, and revenue timing will depend on market response, execution, and factors beyond our control. No customers, pilots, or LOIs currently signed.

### What We're NOT Doing (Until Validated)

- **No new asset classes** until current ones proven with customers
- **No enterprise features** before SMB validation
- **No geographic expansion** before EU product-market fit
- **No hiring spree** before revenue validates demand
- **Features gated by customer demand** (minimum 3 firms requesting)

---

## Investment Highlights

### Market Opportunity

> **Note**: Market sizing figures are from third-party research and should be independently verified. These figures are illustrative context, not the basis of investment claims.

The global algorithmic trading market is projected to reach **$31.49 billion by 2028** (CAGR 12.2%, per Allied Market Research report; verify current figures at source).

**Our focus: Systematic Equities Teams (EU)**

Per FIA/SIFMA data (verify at source), there are 500+ active prop trading firms in the US alone, with similar density in EU (London, Amsterdam) and Asia (Singapore, Hong Kong). Key drivers:

- Institutional adoption of ML-based strategies (per Greenwich Associates research; verify current figures at source)
- Regulatory push for systematic risk management (MiFID II best execution, SEC 15c3-5 market access rule)
- Demand for systematic equities infrastructure with auditability (MiFID II best execution, governance, and monitoring expectations)
- Execution automation and risk controls becoming mandatory as teams scale

### Our Position

We've built a **technically developed platform** designed to solve infrastructure fragmentation:

| Challenge | Traditional Approach | Our Solution |
|-----------|---------------------|--------------|
| Multi-asset support | Separate systems per asset class | Unified architecture |
| Execution modeling | Basic slippage estimates | Research-grade L2+/L3 models |
| Risk management | Manual oversight | Automated, real-time guards |
| Strategy development | Months of infrastructure work | Hours from idea to backtest |

---

## Technical Foundation

### Why Our Technical Depth is an Asset (Not Over-Engineering)

**The technical foundation enables rapid customer-driven iteration:**

| Asset | Benefit for Validation Phase |
|-------|------------------------------|
| Robust architecture | Fast feature changes without breaking production |
| Extensive test coverage | Confidence to iterate quickly on customer feedback |
| Multi-asset support | Pivot capability if customers prefer different asset classes |
| Research-backed algorithms | Credibility with technical buyers (prop firm CTOs) |

### Development Metrics

| Metric | Value | Why It Matters for Customers |
|--------|-------|------------------------------|
| **Automated Tests** | Extensive automated test suite | Designed for production use — comprehensive internal QA (not independently audited) |
| **CI Validation** | Continuous | Internal unit/integration/regression suite; CI test reports available under NDA (not third-party audited) |
| **Asset Classes** | 5 (MVP: Equities) | Extensibility proven; MVP focused on equities |
| **Connectivity** | Multi-provider architecture | Flexibility for customer requirements |

### Technology Differentiation

**Core Innovation: Risk-Aware Reinforcement Learning**

Unlike traditional algorithmic trading platforms that optimize average returns, our platform provides an **implementation of CVaR-constrained reinforcement learning designed for production use** in trading:

```
Traditional: maximize E[Return]
Our Approach: maximize E[Return] subject to CVaR₅%[Return] ≥ threshold
```

This means strategies explicitly avoid catastrophic tail losses, not just maximize gains.

**Three Technical Differentiators**

| Innovation | What It Does | Why It Matters |
|------------|--------------|----------------|
| **Twin Critics + CVaR** | Dual value networks with pessimistic aggregation + tail-risk constraints | Reduced overestimation bias (established technique: Fujimoto et al., 2018); explicit worst-case optimization |
| **AdaptiveUPGD** | Utility-weighted gradient descent preventing catastrophic forgetting | Models remain robust across market regime changes (bull→bear→sideways) |
| **Conformal Prediction** | Distribution-free uncertainty bounds on value estimates | Valid uncertainty even when model assumptions are wrong; automatic position scaling |

**Academic Research Integration**

Our execution models are informed by 7+ peer-reviewed papers (implementation fidelity varies; see code for details):

| Model | Publication | Application |
|-------|-------------|-------------|
| Almgren-Chriss (2001) | J. Risk | Market impact estimation |
| Kyle Lambda (1985) | Econometrica | Price impact model |
| Gatheral (2010) | Quant Finance | Transient impact decay |
| Moallemi & Yuan (2017) | Operations Research | Queue value optimization |
| Dabney et al. (2018) | AAAI | Distributional RL |
| Chow et al. (2015) | JMLR | CVaR optimization |
| Romano et al. (2019) | NeurIPS | Conformal prediction |

**Machine Learning Innovation Stack**

- **Distributional PPO**: 21-51 quantile value estimation (not single-point)
- **Twin Critics**: Reduces value overestimation (well-established in RL literature)
- **CVaR Learning**: Penalizes worst 5% of outcomes
- **VGS v3.2**: Per-parameter gradient variance tracking
- **Conformal Prediction**: Distribution-free uncertainty bounds
- **Adversarial Training**: Robust to market regime changes

**For detailed innovation documentation, see [INNOVATION_STATEMENT.md](INNOVATION_STATEMENT.md)**

---

## Beachhead Market Strategy

### Why Focus Matters

We deliberately constrain our initial market scope following Geoffrey Moore's "Crossing the Chasm" methodology. While our platform supports 5 asset classes, we focus validation efforts on a single, well-defined beachhead segment.

### Defined Beachhead: European Systematic Equities Teams

| Attribute | Specification |
|-----------|---------------|
| **Geography** | UK/EU (Estonia, Netherlands, Germany, Ireland, France, Luxembourg) |
| **Team Size** | 5-50 (prop firms + small funds) |
| **Asset Class** | Listed equities (equities-first) |
| **Need** | Fast infrastructure + institutional-grade risk controls |
| **Budget** | ~€2,000–€5,000/month (initial, illustrative) + enterprise tier for larger firms |

### Expansion Roadmap (Bowling Alley Strategy)

```
Beachhead (Equities)         Adjacent Segments
├─ EU systematic equities ─► ├─ Larger EU funds
│                             ├─ Listed derivatives (futures)
└─ Success Metrics:           └─ Geographic expansion via referrals
   • Reference customers
   • Repeatable onboarding
   • Procurement / risk review sign-off
```

*For canonical positioning, GTM facts, and legally safe language, see `docs/DOCUMENTATION_CANON_DESIGN.md`.*

---

## Go-to-Market Strategy

### Phase 1: European Pilot Program (0–3 months)

**Structured customer validation** before scaling.

| Element | Specification |
|---------|---------------|
| **Target** | 3-5 European systematic equities teams |
| **Duration** | 3 months |
| **Pricing** | €500/month (80% discount) |
| **Commitment** | Weekly feedback, usage data sharing |
| **Success criteria** | conversion willingness at target price range; repeatable onboarding (illustrative) |

**Why Europe first:**
- MiFID II regulatory clarity
- Lower competition vs US
- Startup visa pathway for team expansion
- Strong prop trading ecosystem (Amsterdam, Frankfurt, Dublin)

**Target firm profile:**
- 5-50 traders
- Systematic equities focus (listed markets)
- Building or evaluating new infrastructure
- Budget: €2,000-5,000/month for validated solution

### Phase 2: Early Adopter Revenue (3–6 months)

**Post-pilot conversion** to paying customers.

| Milestone | Target |
|-----------|--------|
| Paying customers | first conversions from pilots (illustrative) |
| Price point | €2,000–€5,000/month (illustrative) |
| ARR | €40K–€50K ARR (illustrative) |

### Phase 3: Scale (9–18 months)

- Feature expansion based on validated demand
- Geographic expansion within EU
- Enterprise tier for larger firms

### Revenue Model

| Segment | Model | Illustrative Pricing |
|---------|-------|---------------------|
| **SMB Prop Firms** (5-20 traders) | Monthly subscription | €2,000-3,000/month |
| **Mid-Size Prop Firms** (20-50 traders) | Annual license | €36,000-60,000/year |
| **Enterprise** (50+ traders) | Custom | €100,000+/year |

### Sales Approach

| Phase | Approach | Rationale |
|-------|----------|-----------|
| **Pilot** | Founder-led | Direct feedback loop |
| **Early Adopters** | Referrals + direct | Leverage pilot success |
| **Scale** | Sales hire | After repeatable process proven |

---

## Competitive Landscape

### Direct Competitors

| Competitor | What They Do | Our Differentiation |
|------------|--------------|---------------------|
| **QuantConnect** | Developer platform + community | Strong platform, but governance/evidence exports and client-controlled execution boundaries are typically engineered in-house for institutional use. |
| **Zipline / OSS backtesting** | Open-source backtesting libraries | Useful building blocks, but productionization (data, execution realism, monitoring, governance) is typically on the customer. |
| **Alpaca** | Commission-free broker API | They provide pipes; we provide intelligence. No ML, no execution modeling. |
| **In-House Development** | Custom systems at prop firms | $500K-2M cost, 12+ months. We reduce to days at fraction of cost. |

### Why We Are Not a Clone

**Fundamental Difference in Approach**:

| Aspect | Traditional Platforms | Our Platform |
|--------|----------------------|--------------|
| **Objective** | maximize E[Return] | maximize E[Return] s.t. CVaR₅% ≥ threshold |
| **Value Estimation** | Single point | 21-51 quantile distribution |
| **Execution Model** | Fixed spread | Market-adaptive 6-9 factors |
| **Uncertainty** | Assumed known | Conformal prediction bounds |
| **Learning** | Prone to forgetting | Continual learning (UPGD) |

### Competitive Moats

1. **Technical Depth**: 7+ peer-reviewed papers implemented (Almgren-Chriss, Kyle, Dabney, Chow, Romano, Gatheral, Moallemi)
2. **Novel Algorithms**: Twin Critics + CVaR, AdaptiveUPGD, VGS — not available anywhere else
3. **Multi-Asset Unity**: Single codebase for 5 asset classes (vs 1-2 typical)
4. **Testing Rigor**: Extensive automated tests + CI validation
5. **Complexity Barrier**: Multi-year development and deep systems integration effort

---

## Technical Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Applications                           │
│    Backtesting │ Live Trading │ Strategy Development │ Research │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                     Platform Core                                │
│  ML Engine │ Execution Sim │ Risk Management │ Data Pipeline    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                   Exchange Connectivity                          │
│  Interactive Brokers │ Alpaca │ Polygon │ OANDA               │
└─────────────────────────────────────────────────────────────────┘
```

### Scalability

| Dimension | Current (internal) | Design Target |
|-----------|---------|-------------|
| Concurrent strategies | Tens | Hundreds (workload-dependent) |
| Assets monitored | Tens to hundreds | Thousands (workload-dependent) |
| Trades per day | Varies by venue/strategy | High-volume (venue/strategy dependent) |
| Historical data | Multi-year | Multi-decade (data provider dependent) |

---

## Traction & Validation

### Stage Acknowledgment

**Current stage: Pre-revenue, entering customer validation**

We are transparent that this is an early-stage opportunity. The core technical foundation is implemented; we are now focused on customer validation following lean startup principles.

### Technical Foundation (Implemented)

| Phase | Status | Evidence |
|-------|--------|----------|
| Core platform | Foundation implemented | Equities execution + simulation functional (verify via tests) |
| Multi-asset architecture | Foundation implemented | 5 asset classes supported (verify via tests) |
| Risk management | Foundation implemented | CVaR optimization, risk guards (verify via tests) |
| Testing infrastructure | Foundation implemented | Automated tests; CI validation reports available |

### What We're Doing Now (Customer Validation)

| Activity | Status | Target |
|----------|--------|--------|
| Customer discovery interviews | 🔄 In progress | 20+ (Phase 1) |
| Pain point validation | 🔄 In progress | Ranked problem list |
| Pilot program design | Defined | 3-month structured program |
| MVP scope definition | Defined | Equities execution + risk mgmt |

### What's Next (Revenue Validation)

| Milestone | Timeline | Success Criteria |
|-----------|----------|------------------|
| Pilot launch | Phase 1 (0–3 months) | 3-5 teams onboarded |
| Pilot completion | Phase 2 (3–6 months) | willingness to pay at target range (illustrative) |
| First paying customers | Phase 2 (3–6 months) | first pilot conversions (illustrative) |
| Product-market fit indicators | Phase 3 (6–12 months) | 2+ customers expanding usage |

### Technical Validation (Internal)

Internal QA includes paper-trading, parity instrumentation (backtest vs. live), and reliability monitoring. Detailed results can be shared under NDA; customer validation is the current priority.

### Lean Validation Approach

**Build-Measure-Learn cycle:**

1. **Build**: MVP deployed (equities execution + risk management)
2. **Measure**: Activation, retention, NPS from pilot customers
3. **Learn**: Iterate based on feedback, not assumptions

**Pivot criteria defined**: If 70%+ of prospects want different asset classes or price sensitivity prevents conversion, we will pivot based on data.

---

## Team

### Current Team

| Role | Background | Focus |
|------|------------|-------|
| **Founder/CTO** | Quantitative development, ML/RL research | Platform architecture, execution models |

**Technical capabilities demonstrated:**
- Extensive automated testing and CI validation
- 5 asset class integrations (designed for production use)
- Academic research implementation (7+ peer-reviewed papers)
- Multiple broker/data connectivity adapters (see Appendix B)

### Team Gaps (To Be Filled Post-Funding)

| Role | Priority | Why Needed |
|------|----------|------------|
| **Sales Lead** | Critical | Founder-led sales not scalable past 10 customers |
| **DevOps Engineer** | High | Cloud deployment, multi-tenant infrastructure |
| **Frontend Engineer** | Medium | Dashboard MVP for enterprise clients |
| **Quant Researcher** | Medium | Strategy templates, customer success |

### Advisory Board (Seeking)

Actively seeking advisors with:
- Prop trading firm operational experience
- Enterprise B2B sales in fintech
- Regulatory/compliance expertise (MiFID II, SEC)

*Note: Current team size is small. This is a typical pre-seed configuration. Technical depth has been prioritized over headcount.*

---

## Economic Impact & Job Creation (EU Focus)

### Employment Commitment

Our European expansion plan targets meaningful economic contribution while building a world-class fintech team:

| Year | Direct FTEs | Cumulative Salary Investment | Notes |
|------|-------------|------------------------------|------------------|
| **Year 1** | 5 | €375,000 | Initial EU entity + first hires |
| **Year 2** | 12 | €930,000 | GTM + engineering growth |
| **Year 3** | 22 | €1,760,000 | Scaling customer success + security |
| **Year 5** | 50 | €4,250,000 | Multi-team EU growth (scenario) |

### EU Visa / Relocation Readiness (Non-Legal Summary)

- We will pursue an EU base via an applicable startup/entrepreneur pathway (jurisdiction-dependent). Primary establishment path is Estonia; Netherlands is a secondary path if needed. We will engage local immigration counsel and an approved facilitator/incubator where required.
- Our case rests on **innovation** (risk-first ML + CCEA), **scalability** (B2B SaaS + enterprise), and **local job creation** (technical roles + go-to-market).
- We avoid making claims about specific statutory thresholds in investor materials; requirements differ by country and case specifics.

*See `docs/BUSINESS_PLAN_EU_VISA.md` for the detailed hiring plan and assumptions.*

---

## Use of Funds

### Funding Ask

**Target raise**: **€500K–€750K** (customer validation + EU go-to-market runway; see `docs/BUSINESS_PLAN_EU_VISA.md`)

**Use of funds priority:**

| Priority | Category | Allocation | Purpose |
|----------|----------|------------|---------|
| 1 | **Sales/GTM** | 40% | Sales lead hire, pilot customer acquisition |
| 2 | **Engineering** | 35% | DevOps, frontend, infrastructure |
| 3 | **Operations** | 15% | Legal/compliance operations, vendor due diligence preparedness |
| 4 | **Reserve** | 10% | Contingency buffer (runway target 18–24 months) |

### Key Milestones (12 months post-funding)

| Phase | Milestone | Success Metric |
|-------|-----------|----------------|
| **Phase 1 (0–3 months)** | First pilot customers | 3 signed pilots |
| **Phase 2 (3–6 months)** | Dashboard MVP, first revenue | €50K ARR (illustrative) |
| **Phase 2 (3–6 months)** | Cloud deployment | Multi-tenant infrastructure |
| **Phase 3 (6–12 months)** | Product-market fit indicators | 2+ customers expanding |
| **Phase 3 (9–18 months)** | Series A readiness milestones | milestone-based; revenue/retention dependent (illustrative) |

### Runway Consideration

Targeting 18-24 month runway to reach Series A milestones. Conservative burn assumed until product-market fit indicators.

---

## Risk Factors

### Execution Risks (Primary)

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Sales execution** | High | Founder-led initially; hire sales lead post-funding |
| **First customer acquisition** | High | Target warm network; offer extended pilots |
| **Team scaling** | Medium | Structured hiring plan; competitive compensation |
| **Founder dependency** | High | Document architecture; hire CTO-track engineer |

### Market Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **EU regulatory expectations** | Medium | Clear software-provider posture; CCEA boundary; compliance documentation |
| **Competition** | Medium | Technical depth moat; niche focus on prop firms |
| **Bear markets** | Medium | Subscription model less affected than AUM-based |

### Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Exchange API changes** | Low | Adapter abstraction layer; connector monitoring |
| **Model degradation** | Low | Continuous retraining pipelines built-in |
| **Security breach** | Medium | Security best practices; no client funds/secrets in Cloud |

### Investment Risks

**This is a high-risk, early-stage investment.** Key risks include:

1. **Pre-revenue**: No validated customer willingness to pay
2. **Single founder**: Key person risk until team expansion
3. **Competitive market**: Well-funded incumbents (QuantConnect, Alpaca)
4. **Regulatory expectations**: MiFID II/DORA-driven procurement and auditability expectations

*Investors should expect a 10-year horizon with significant loss potential.*

---

## Financial Model (Illustrative)

> **Important**: These projections are illustrative scenarios grounded in bottom-up analysis, not top-down aspirations. As a pre-revenue company, actual results will depend on execution and market conditions. We present conservative and downside scenarios to demonstrate business viability under various conditions.

### Bottom-Up Revenue Logic

**Target Market**: European systematic equities teams (prop firms + small funds) across major hubs (Amsterdam, London, Frankfurt, Paris, Dublin)
**Serviceable Market**: ~60 firms (5-50 traders, tech-forward, budget available)

**Funnel Assumptions** (industry-benchmarked):
| Stage | Our Rate | Industry Range | Source |
|-------|----------|----------------|--------|
| Outreach→Meeting | 10% | 5-15% | [Gradient Works](https://www.gradient.works/blog/2024-b2b-sales-benchmarks) |
| Meeting→Pilot | 25% | 20-30% | [First Page Sage](https://firstpagesage.com/seo-blog/b2b-saas-funnel-conversion-benchmarks-fc/) |
| Pilot→Paid | 60% | 50-70% | [SaaStr](https://www.saastr.com/what-is-the-typical-conversion-from-paid-pilot-to-annual-contract-in-b2b-saas/) |
| Sales Cycle | 4-6 mo | 3-9 mo | [Databox](https://databox.com/saas-sales-benchmarks) |

### Scenario Analysis

#### Conservative Scenario (50% below base)

| Year | Customers | ARR (EUR) | Key Assumptions |
|------|-----------|-----------|-----------------|
| Y1 | 2 | €48K | 2 customers × €2K/month × 12 |
| Y2 | 8 | €200K | +6 net new, slower expansion |
| Y3 | 18 | €500K | Founder-led sales, 50% pilot conversion |

#### Base Scenario

| Year | Customers | ARR (EUR) | Key Assumptions |
|------|-----------|-----------|-----------------|
| Y1 | 3 | €80K | 3 customers × ~€2.2K/month × 12 (blended) |
| Y2 | 12 | €360K | +9 net new, sales hire |
| Y3 | 25 | €850K | 60% pilot conversion, 110% NRR |

#### Stress Test: Downside Scenario (70% below base)

| Year | Customers | ARR (EUR) | Monthly Burn | Response |
|------|-----------|-----------|--------------|----------|
| Y1 | 1 | €24K | €35K | Reduce to €25K/mo |
| Y2 | 4 | €100K | €40K | Delay sales hire |
| Y3 | 10 | €280K | €45K | Break-even at €300K ARR |

**Downside assumptions**: 40% pilot conversion, 8-10 month sales cycle, conservative market conditions

### Contingency Measures

| Trigger | Response |
|---------|----------|
| Y1 ARR < €50K | Cut burn 45% (€45K→€25K/mo), extend runway to 20 months |
| Pilot conversion <40% | Pivot to adjacent segment (hedge funds, family offices) |
| Market downturn | Emphasize multi-asset flexibility, cost-savings narrative |

**Burn Reduction Levers**: Delay sales hire (-€80K/yr), remote-first (-€24K/yr), founder salary cut (-€30K/yr) = **€134K savings**

### Unit Economics (Industry-Benchmarked)

| Metric | Our Target | Industry Benchmark | Source |
|--------|------------|-------------------|--------|
| **CAC** | €8-12K | €5-15K (SMB), €15K+ (Enterprise) | [Powered by Search](https://www.poweredbysearch.com/learn/b2b-saas-cac-benchmarks/), [First Page Sage](https://firstpagesage.com/reports/average-customer-acquisition-cost-cac-by-industry-b2b-edition-fc) |
| **LTV** | €45-60K | €30-100K | [ProfitWell](https://profitwell.com) |
| **LTV:CAC** | 4:1-5:1 | 3:1 (min), 4:1 (B2B SaaS), 5:1 (Fintech) | [Phoenix Strategy Group](https://www.phoenixstrategy.group/blog/ltvcac-ratio-saas-benchmarks-and-insights) |
| **Gross Margin** | 82-85% | 70-85% | [SaaS Capital](https://www.saas-capital.com/blog-posts/benchmarking-metrics-for-bootstrapped-saas-companies/) |
| **Payback** | 12-15 mo | 12-18 mo (healthy) | [OpenView Partners](https://openviewpartners.com) |
| **Annual Churn** | 8-12% | 5-7% (ent), 10-15% (SMB) | [SaaS Capital](https://www.saas-capital.com) |
| **NRR** | 105-110% | 100-120% (enterprise) | [KeyBanc](https://www.key.com/kco/images/2023_SaaS_Survey_Results.pdf) |

### Runway Analysis

| Funding | Burn Rate | Runway | Milestone |
|---------|-----------|--------|-----------|
| €500K | Burn-managed (€25K/mo) | 20 mo | Validation + pilot conversion |
| €750K | Burn-managed (€35K/mo) | 21 mo | Validation + earlier key hires |
| €750K | Accelerated (€45K/mo) | 18 mo | Faster hiring / higher spend |

### Why These Numbers Are Credible

| Projection | Industry Norm | Assessment |
|------------|---------------|------------|
| Y1 customers: 2-3 | Early SaaS: 1-10 | ✅ Conservative |
| Y1 ARR: €48-80K | Pre-seed: €0-200K | ✅ Realistic |
| CAC: €10K | Fintech SMB: €5-15K | ✅ Within range |
| LTV:CAC: 5:1 | Fintech: 4-6:1 | ✅ Industry standard |

*Growth rates benchmarked against [SaaS Capital 2024](https://www.saas-capital.com/blog-posts/growth-benchmarks-for-private-saas-companies/), [High Alpha 2024](https://www.highalpha.com/2024-saas-benchmarks-report), [Growth Unhinged](https://www.growthunhinged.com/p/your-guide-to-the-2024-saas-benchmarks)*

**Our commitment**: Sustainable unit economics over growth-at-all-costs. Detailed projections in [BUSINESS_PLAN_EU_VISA.md](BUSINESS_PLAN_EU_VISA.md) Section 8.

---

## Why Now?

### Market Timing

1. **Regulatory and governance push**: stronger best-execution evidence and operational resilience expectations (MiFID II / DORA)
2. **AI/ML Maturity**: ML frameworks designed for production now available
3. **Market Complexity**: Multi-asset strategies require sophisticated tools
4. **Talent Availability**: Quant talent seeking modern platforms

### Our Advantage

- **2+ years of development** completed
- **Designed for production use** with live execution via customer-controlled Agent
- **Research-grade** execution models
- **Extensive** internal testing infrastructure (not independently audited)

---

## Next Steps

### For Interested Investors

1. **Technical Demo**: Live walkthrough of platform capabilities
2. **Due Diligence**: Code review, architecture deep-dive
3. **Customer References**: Introductions to early users
4. **Term Sheet Discussion**: Investment structure

### Contact

For more information or to schedule a demo, please contact:
[Contact Information]

---

## Appendix

### A. Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Core | Python 3.12, Cython, C++ | Performance-critical code |
| ML | PyTorch, Stable-Baselines3 | Reinforcement learning |
| Data | Pandas, NumPy, Parquet | Data processing |
| Testing | Pytest, CI/CD | Quality assurance |
| Config | YAML, Pydantic | Type-safe configuration |

### B. Connectivity Support Matrix (MVP: Equities)

| Venue/Broker | Asset Class | Data | Trading | Status |
|----------|-------------|------|---------|--------|
| Interactive Brokers | Equities (MVP), Futures (optional) | ✓ | ✓ | Ready (internal) |
| Alpaca | Equities (optional) | ✓ | ✓ | Implemented |
| Polygon | Equities data (optional) | ✓ | - | Implemented |
| OANDA | FX (optional) | ✓ | ✓ | Implemented |

### C. Regulatory Positioning

**Our position: Software vendor, not regulated financial entity**

We provide technology tools to trading firms who are themselves regulated. We do not:
- Provide execution-as-a-service or discretionary execution on behalf of clients
- Manage client assets
- Provide investment advice or recommendations
- Handle client funds

**Regulatory framework by jurisdiction:**

| Jurisdiction | Our Position | Client's Responsibility |
|--------------|--------------|------------------------|
| **USA** | Software provider posture (classification depends on activities) | Client must be registered if required |
| **EU** | Technology vendor posture (designed to avoid MiFID-regulated activities) | Client handles MiFID II obligations (incl. best execution where applicable) |
| **UK** | Software-as-a-Service posture (classification depends on activities) | Client handles FCA obligations |
| **Singapore** | Technology vendor | Client handles MAS requirements |

**Key distinctions:**
- We are similar to Bloomberg Terminal, Refinitiv Eikon, or QuantConnect in regulatory positioning
- Clients use our tools to implement *their* strategies with *their* regulatory obligations
- We do not recommend specific trades or strategies

**Compliance tooling (for enterprise clients):**
- Vendor due diligence documentation (designed to support client procurement reviews)
- GDPR data handling documentation (available now)
- Audit logs + export tooling to support client record-keeping and best-execution analysis (where applicable)

*No certification claims are made. Clients run their own compliance and legal review per their requirements.*

*Legal review recommended for specific client engagements. This section describes our current understanding and is not legal advice.*

---

## Important Disclaimers

### Forward-Looking Statements

This document contains forward-looking statements including projections, targets, and expectations. These statements are based on current assumptions and are subject to significant risks and uncertainties. Actual results may differ materially.

### Investment Risk

An investment in this company is speculative and involves substantial risk. Investors may lose their entire investment. This is an early-stage company with no revenue history.

### No Offer of Securities

This document is for informational purposes only and does not constitute an offer to sell or a solicitation of an offer to buy any securities. Any offer will be made only by means of a definitive offering document.

### Trading Risk

The platform is a software tool for algorithmic trading. Trading in financial instruments carries significant risk of loss. Past performance, whether actual or simulated, is not indicative of future results.

---

*This document contains forward-looking statements and illustrative projections. Actual results may vary.*

*Confidential - For Investor Use Only*

*Last Updated: 2025-12-18*

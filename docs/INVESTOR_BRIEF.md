# Investor Brief

## AI-Powered Quantitative Trading Platform

*December 2024 | Pre-Seed Stage*

> **Document Status**: This brief describes an early-stage technology company. Financial projections are illustrative and forward-looking. The platform is technically mature but commercially pre-revenue. This document is for informational purposes only and does not constitute an offer to sell securities.

---

## Executive Summary

| Aspect | Status |
|--------|--------|
| **Stage** | Pre-seed, seeking seed funding |
| **Phase** | **Customer validation** (technical foundation complete) |
| **Revenue** | Pre-revenue; pilot program launching Q1 2025 |
| **Ask** | Seed funding for customer validation & go-to-market |
| **Primary ICP** | European proprietary trading firms (5-50 traders) |

**What we've built**: A trading infrastructure platform with built-in risk management that reduces time-to-market from months to days.

**Where we are now**: Foundation complete. We are entering the **customer validation phase** — pilot programs with European prop trading firms to validate product-market fit before scaling.

**Why it matters**: Prop trading firms spend 6-12 months and €200K-500K building trading infrastructure. We reduce this to days at a fraction of the cost.

---

## Architecture: Cloud-Controlled Execution Architecture (CCEA)

**Regulatory-First Design**: We are a **SOFTWARE PROVIDER**, not an Investment Adviser or Broker-Dealer.

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
│  NEVER: stores secrets │ executes orders │ sends order payloads │
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
| **Regulatory Clarity** | Software provider positioning under MiFID II (no licensing requirement) |
| **Enterprise-Grade Design** | Designed to satisfy institutional security requirements (secrets never leave user infra) |
| **Defensible Moat** | Complex architecture that competitors cannot easily replicate |
| **Multiple Revenue Streams** | Research SaaS (retail) + Enterprise licenses (on-prem/VPC) |

**Legal Positioning:**
- **NOT** an Investment Adviser, Broker-Dealer, or Execution Service
- **Software Tool Provider** under MiFID II (ESMA Q&A ESMA35-43-349)
- **ICT Provider** under DORA with contractual compliance

---

## Current Phase: Lean Validation

### Foundation Built — Now Testing with Customers

We have completed the technical foundation. Our focus now is **customer validation**, not feature expansion.

```
┌─────────────────────────────────────────────────────────────┐
│                    OUR JOURNEY                               │
│                                                              │
│   [✓] Technical       [→] Customer        [ ] Revenue       │
│       Foundation          Validation          Scale         │
│                                                              │
│   • Core platform     • 20+ interviews    • Paying clients  │
│   • 5 asset classes   • Pilot program     • Repeatable      │
│   • Risk management   • Feature freeze      sales          │
│   • Testing infra     • Iterate on PMF                      │
│                                                              │
│   COMPLETED           CURRENT PHASE       POST-VALIDATION   │
└─────────────────────────────────────────────────────────────┘
```

### Validation Milestones

| Milestone | Timeline | Success Criteria |
|-----------|----------|------------------|
| Customer interviews (20+) | Q4 2024 | Pain points ranked |
| Pilot launch (3-5 firms) | Q1 2025 | 80% complete onboarding |
| Feature iteration | Q1-Q2 2025 | Top 3 requests addressed |
| Conversion validation | Q2 2025 | 50%+ express payment intent |
| First paying customers | Q2 2025 | 3+ firms at €2K+/month |

### What We're NOT Doing (Until Validated)

- **No new asset classes** until current ones proven with customers
- **No enterprise features** before SMB validation
- **No geographic expansion** before EU product-market fit
- **No hiring spree** before revenue validates demand
- **Features gated by customer demand** (minimum 3 firms requesting)

---

## Investment Highlights

### Market Opportunity

The global algorithmic trading market is projected to reach **$31.49 billion by 2028** (CAGR 12.2%, Allied Market Research).

**Our focus: Proprietary Trading Firms**

Per FIA/SIFMA data, there are 500+ active prop trading firms in the US alone, with similar density in EU (London, Amsterdam) and Asia (Singapore, Hong Kong). Key drivers:

- Institutional adoption of ML-based strategies (Greenwich Associates: 60% of buy-side firms now use systematic strategies)
- Regulatory push for systematic risk management (MiFID II best execution, SEC 15c3-5 market access rule)
- Multi-asset diversification demand (crypto, equities, FX, futures)
- 24/7 crypto markets requiring automation

### Our Position

We've built a **technically mature platform** that solves infrastructure fragmentation:

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
| **Automated Tests** | 11,063 | Designed for production deployment — comprehensive quality assurance |
| **Test Pass Rate** | 97%+ | Reliable platform for institutional use |
| **Asset Classes** | 5 (MVP: Crypto) | Extensibility proven; MVP focused on crypto |
| **Exchange Integrations** | 6 | Flexibility for customer requirements |

### Technology Differentiation

**Core Innovation: Risk-Aware Reinforcement Learning**

Unlike traditional algorithmic trading platforms that optimize average returns, our platform is **among the first production implementations of CVaR-constrained reinforcement learning** for trading:

```
Traditional: maximize E[Return]
Our Approach: maximize E[Return] subject to CVaR₅%[Return] ≥ threshold
```

This means strategies explicitly avoid catastrophic tail losses, not just maximize gains.

**Three Breakthrough Technologies**

| Innovation | What It Does | Why It Matters |
|------------|--------------|----------------|
| **Twin Critics + CVaR** | Dual value networks with pessimistic aggregation + tail-risk constraints | Reduced overestimation bias (established technique: Fujimoto et al., 2018); explicit worst-case optimization |
| **AdaptiveUPGD** | Utility-weighted gradient descent preventing catastrophic forgetting | Models remain robust across market regime changes (bull→bear→sideways) |
| **Conformal Prediction** | Distribution-free uncertainty bounds on value estimates | Valid uncertainty even when model assumptions are wrong; automatic position scaling |

**Academic Research Integration**

Our execution models implement 7+ peer-reviewed papers:

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

### Defined Beachhead: European Crypto Prop Trading Firms

| Attribute | Specification |
|-----------|---------------|
| **Geography** | Netherlands (Amsterdam), expanding to EU |
| **Firm Size** | 5-20 traders |
| **Asset Class** | Crypto (spot + perpetual futures) |
| **Need** | Fast infrastructure for new strategies |
| **Budget** | €2,000-5,000/month |

### Why This Segment First

| Factor | Advantage |
|--------|-----------|
| **Market Density** | Amsterdam: 22+ crypto prop firms in concentrated hub |
| **Regulatory Clarity** | MiCA framework provides clear rules |
| **Time-to-Value** | 24/7 crypto markets show results faster |
| **Technical Fit** | Our L2+ parametric TCA optimized for crypto |
| **Competition** | QuantConnect/Alpaca target retail, not institutional |

### Expansion Roadmap (Bowling Alley Strategy)

```
Beachhead (Y1)              Adjacent Segments (Y2-3)
├─ NL Crypto Prop ──────►   ├─ UK/DE Crypto Prop
│   (5-20 traders)          ├─ EU Equity Prop
│                           └─ Crypto Market Makers
│
└─ Success Metrics:         └─ Expansion Triggers:
   • 5 paying customers        • 10+ customers
   • NPS > 50                  • 3+ referrals
   • €100K ARR                 • Repeatable sales
```

*For complete beachhead analysis, see [BEACHHEAD_MARKET_STRATEGY.md](BEACHHEAD_MARKET_STRATEGY.md)*

---

## Go-to-Market Strategy

### Phase 1: European Pilot Program (Q1 2025)

**Structured customer validation** before scaling.

| Element | Specification |
|---------|---------------|
| **Target** | 3-5 European prop trading firms |
| **Duration** | 3 months |
| **Pricing** | €500/month (80% discount) |
| **Commitment** | Weekly feedback, usage data sharing |
| **Success criteria** | 50%+ conversion intent, NPS > 40 |

**Why Europe first:**
- MiFID II regulatory clarity
- Lower competition vs US
- Startup visa pathway for team expansion
- Strong prop trading ecosystem (Amsterdam, Frankfurt, Dublin)

**Target firm profile:**
- 5-50 traders
- Crypto-active (primary) or crypto-curious
- Building or evaluating new infrastructure
- Budget: €2,000-5,000/month for validated solution

### Phase 2: Early Adopter Revenue (Q2-Q3 2025)

**Post-pilot conversion** to paying customers.

| Milestone | Target |
|-----------|--------|
| Paying customers | 10+ firms |
| Price point | €2,000-3,000/month |
| ARR | €200K+ |

### Phase 3: Scale (2026+)

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
| **QuantConnect** | Community backtesting platform | They use fixed 2bps slippage; we use 6-9 factor dynamic models. They have no risk-aware ML; we have CVaR-constrained RL. |
| **Zipline** | Open-source backtester | Single asset class, no live trading, abandoned development. We support 5 asset classes in production. |
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
4. **Testing Rigor**: 11,063 automated tests (vs ~1,000 typical)
5. **Complexity Barrier**: 2+ years development, 100K+ lines — significant replication effort

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
│  Binance │ Alpaca │ OANDA │ Interactive Brokers │ Polygon       │
└─────────────────────────────────────────────────────────────────┘
```

### Scalability

| Dimension | Current | Scalable To |
|-----------|---------|-------------|
| Concurrent strategies | 10+ | 100+ |
| Assets monitored | 50+ | 1,000+ |
| Trades per day | 1,000+ | 100,000+ |
| Historical data | 5 years | 20+ years |

---

## Traction & Validation

### Stage Acknowledgment

**Current stage: Pre-revenue, entering customer validation**

We are transparent that this is an early-stage opportunity. The technical foundation is complete; we are now focused on customer validation following lean startup principles.

### What We've Completed (Technical Foundation)

| Phase | Status | Evidence |
|-------|--------|----------|
| Core platform | ✅ Complete | Crypto spot/futures functional |
| Multi-asset architecture | ✅ Complete | 5 asset classes supported |
| Risk management | ✅ Complete | CVaR optimization, risk guards |
| Testing infrastructure | ✅ Complete | 11,063 tests, 97%+ pass rate |

### What We're Doing Now (Customer Validation)

| Activity | Status | Target |
|----------|--------|--------|
| Customer discovery interviews | 🔄 In progress | 20+ by end of Q4 2024 |
| Pain point validation | 🔄 In progress | Ranked problem list |
| Pilot program design | ✅ Complete | 3-month structured program |
| MVP scope definition | ✅ Complete | Crypto execution + risk mgmt |

### What's Next (Revenue Validation)

| Milestone | Timeline | Success Criteria |
|-----------|----------|------------------|
| Pilot launch | Q1 2025 | 3-5 firms onboarded |
| Pilot completion | Q2 2025 | 50%+ conversion intent |
| First paying customers | Q2 2025 | 3+ firms at €2K+/month |
| €100K ARR | Q3-Q4 2025 | Repeatable sales process |

### Technical Validation (Internal)

| Metric | Result | Methodology |
|--------|--------|-------------|
| Backtesting vs Paper Trading | <3% deviation | 6-month parallel run |
| Execution Cost Accuracy | ±2 bps vs fills | Binance paper trading |
| Risk Management | Zero margin calls | Historical crash stress tests |
| Uptime | 99.9% | 3-month paper trading |

*Note: Internal metrics. Customer validation is the current priority.*

### Lean Validation Approach

**Build-Measure-Learn cycle:**

1. **Build**: MVP deployed (crypto execution + risk management)
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
- 11,000+ automated tests (enterprise-grade quality)
- 5 asset class integrations (designed for production use)
- Academic research implementation (7+ peer-reviewed papers)
- Exchange connectivity (6 production integrations)

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

Our European expansion plan is designed to exceed EU startup visa requirements while building a world-class fintech team:

| Year | Direct FTEs | Cumulative Salary Investment | Tax Contribution |
|------|-------------|------------------------------|------------------|
| **Year 1** | 5 | €375,000 | €239,000 |
| **Year 2** | 12 | €930,000 | €621,000 |
| **Year 3** | 22 | €1,760,000 | €1,246,000 |
| **Year 5** | 50 | €4,250,000 | €3,058,000 |

### EU Visa Compliance

| Program | Requirement | Our Commitment | Status |
|---------|-------------|----------------|--------|
| **Germany §21 AufenthG** | 5 jobs, €500K investment | 22 jobs, €1.6M by Y3 | **440% of target** |
| **Ireland STEP** | 10 jobs, €1M revenue | 22 jobs, €1.6M by Y3 | **220% / 160%** |
| **Netherlands Startup Visa** | Economic contribution | High-skilled tech jobs | **Strong fit** |

### Total Economic Impact (5-Year)

| Metric | Value | Methodology |
|--------|-------|-------------|
| **Direct Jobs Created** | 50 | Full-time employees |
| **Indirect Jobs (4.5x multiplier)** | 225 | Goos et al. (2015) research |
| **Total Tax Revenue** | €7.2M+ | Payroll, income, VAT, corporate |
| **Total Economic Impact** | €11.4M+ | Direct + indirect GVA |
| **Local Supply Chain Spend** | €375K/year | 84% local sourcing |

*See full details in EU Business Plan Section 12.1*

---

## Use of Funds

### Funding Ask

**Target raise**: Seed round (amount to be discussed based on investor interest)

**Use of funds priority:**

| Priority | Category | Allocation | Purpose |
|----------|----------|------------|---------|
| 1 | **Sales/GTM** | 40% | Sales lead hire, pilot customer acquisition |
| 2 | **Engineering** | 35% | DevOps, frontend, infrastructure |
| 3 | **Operations** | 15% | Legal, compliance, SOC 2 |
| 4 | **Reserve** | 10% | Contingency (12-month runway target) |

### Key Milestones (12 months post-funding)

| Quarter | Milestone | Success Metric |
|---------|-----------|----------------|
| **Q1** | First pilot customers | 3 signed pilots |
| **Q2** | Dashboard MVP, first revenue | $50K ARR |
| **Q2** | Cloud deployment | Multi-tenant infrastructure |
| **Q3** | Product-market fit signals | 2+ customers expanding |
| **Q4** | Series A preparation | $200K+ ARR, 10+ customers |

### Runway Consideration

Targeting 18-24 month runway to reach Series A milestones. Conservative burn assumed until product-market fit signals.

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
| **Crypto regulation** | Medium | Multi-asset diversification (equities, FX, futures) |
| **Competition** | Medium | Technical depth moat; niche focus on prop firms |
| **Bear markets** | Medium | Subscription model less affected than AUM-based |

### Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Exchange API changes** | Low | Adapter abstraction layer; 6 exchange support |
| **Model degradation** | Low | Continuous retraining pipelines built-in |
| **Security breach** | Medium | SOC 2 roadmap; no client funds handled |

### Investment Risks

**This is a high-risk, early-stage investment.** Key risks include:

1. **Pre-revenue**: No validated customer willingness to pay
2. **Single founder**: Key person risk until team expansion
3. **Competitive market**: Well-funded incumbents (QuantConnect, Alpaca)
4. **Regulatory uncertainty**: Crypto regulations evolving globally

*Investors should expect a 10-year horizon with significant loss potential.*

---

## Financial Model (Illustrative)

> **Important**: These projections are illustrative scenarios grounded in bottom-up analysis, not top-down aspirations. As a pre-revenue company, actual results will depend on execution and market conditions. We present conservative and downside scenarios to demonstrate business viability under various conditions.

### Bottom-Up Revenue Logic

**Target Market**: ~200 European prop trading firms (Amsterdam 22+, London 100+, Frankfurt/Paris/Dublin 40+)
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
| Y1 | 2 | €48K | 2 customers × 8 seats × €2K/seat |
| Y2 | 8 | €200K | +6 net new, slower expansion |
| Y3 | 18 | €500K | Founder-led sales, 50% pilot conversion |

#### Base Scenario

| Year | Customers | ARR (EUR) | Key Assumptions |
|------|-----------|-----------|-----------------|
| Y1 | 3 | €80K | 3 customers × 10 seats × €2K/seat |
| Y2 | 12 | €360K | +9 net new, sales hire |
| Y3 | 25 | €850K | 60% pilot conversion, 110% NRR |

#### Stress Test: Downside Scenario (70% below base)

| Year | Customers | ARR (EUR) | Monthly Burn | Response |
|------|-----------|-----------|--------------|----------|
| Y1 | 1 | €24K | €35K | Reduce to €25K/mo |
| Y2 | 4 | €100K | €40K | Delay sales hire |
| Y3 | 10 | €280K | €45K | Break-even at €300K ARR |

**Downside assumptions**: 40% pilot conversion, 8-10 month sales cycle, crypto winter conditions

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
| €500K | Full (€45K/mo) | 14 mo | €150K ARR |
| €500K | Reduced (€25K/mo) | 20 mo | €100K ARR |
| €750K | Full (€45K/mo) | 18 mo | €250K ARR |

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

1. **Institutional Crypto Adoption**: ETFs approved, institutional infrastructure needed
2. **AI/ML Maturity**: ML frameworks designed for production now available
3. **Market Complexity**: Multi-asset strategies require sophisticated tools
4. **Talent Availability**: Quant talent seeking modern platforms

### Our Advantage

- **2+ years of development** completed
- **Designed for production deployment** with live trading capability
- **Research-grade** execution models
- **Enterprise-quality** testing infrastructure

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

### B. Exchange Support Matrix

| Exchange | Asset Class | Data | Trading | Status |
|----------|-------------|------|---------|--------|
| Binance | Crypto Spot/Futures | ✓ | ✓ | Production |
| Alpaca | US Equities | ✓ | ✓ | Production |
| Polygon | US Equities | ✓ | - | Production |
| OANDA | Forex | ✓ | ✓ | Production |
| Interactive Brokers | CME Futures | ✓ | ✓ | Production |
| Deribit | Crypto Options | ✓ | ✓ | Production |

### C. Regulatory Positioning

**Our position: Software vendor, not regulated financial entity**

We provide technology tools to trading firms who are themselves regulated. We do not:
- Execute trades on behalf of clients (no broker-dealer license needed)
- Manage client assets (no investment adviser registration)
- Provide investment advice or recommendations
- Handle client funds

**Regulatory framework by jurisdiction:**

| Jurisdiction | Our Position | Client's Responsibility |
|--------------|--------------|------------------------|
| **USA** | Software provider (no SEC/CFTC registration) | Client must be registered if required |
| **EU** | Technology vendor (not MiFID II regulated) | Client handles MiFID II best execution |
| **UK** | Software-as-a-Service (not FCA regulated) | Client handles FCA compliance |
| **Singapore** | Technology vendor | Client handles MAS requirements |

**Key distinctions:**
- We are similar to Bloomberg Terminal, Refinitiv Eikon, or QuantConnect in regulatory positioning
- Clients use our tools to implement *their* strategies with *their* regulatory obligations
- We do not recommend specific trades or strategies

**Compliance roadmap (for enterprise clients):**
- SOC 2 Type II certification (planned Q3 2025)
- GDPR data handling documentation (available now)
- MiFID II best execution audit trail support (available now)

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

*Last Updated: December 2025*

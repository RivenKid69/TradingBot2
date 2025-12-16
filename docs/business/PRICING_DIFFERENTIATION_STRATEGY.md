# Pricing Differentiation Strategy

## CCEA Platform - Segment-Based Pricing Framework

**Document Version:** 2.0
**Date:** December 2025
**Classification:** Internal Strategy Document

---

## Architecture Reference

This pricing document is aligned with the **Cloud-Controlled Execution Architecture (CCEA)**. See [CCEA_OVERVIEW.md](../architecture/CCEA_OVERVIEW.md) for complete architectural details.

**Key Principle:** Cloud provides intelligence (training, backtest, artifacts), Agent provides execution. Pricing varies by deployment mode and capability access.

---

## Executive Summary

This document establishes a **comprehensive pricing differentiation strategy** for CCEA Platform, addressing the critical weakness of one-size-fits-all pricing. Our framework is built on:

- **Value-based pricing principles** (Pearson Ham Group methodology)
- **Multi-metric pricing** aligned with fintech best practices (McKinsey research: 40% more market share)
- **Segment-specific packaging** for three distinct customer profiles
- **Clear upgrade paths** to maximize customer lifetime value

**Key Reference:** According to [SBI Growth's 2024 State of B2B SaaS Pricing](https://sbigrowth.com/tools-and-solutions/pricing-benchmarks-report-2024), financial SaaS companies using 3-4 pricing metrics capture significantly more value than those using 1-2 metrics.

---

## 1. Customer Segmentation Framework

### 1.1 Target Segments

Based on industry analysis of quantitative trading market structure ([QuantBlueprint](https://www.quantblueprint.com/post/top-quant-firms-list-comp-up-to-500k), [WallStreetQuants](https://www.thewallstreetquants.com/firm-list)):

| Segment | AUM Range | Team Size | Decision Maker | Sales Motion |
|---------|-----------|-----------|----------------|--------------|
| **Indie/Small Prop** | $0-10M | 1-5 | Trader/Founder | Self-serve + PLG |
| **Mid-Size Fund** | $10M-500M | 5-50 | CTO/Head of Quant | Inside sales |
| **Enterprise/Bank** | $500M+ | 50-1000+ | C-Suite + Procurement | Field sales |

### 1.2 Segment Characteristics

#### Segment A: Indie Traders & Small Prop Firms

**Profile:**
- Solo traders or small teams (1-5 people)
- AUM: $100K - $10M proprietary capital
- Price-sensitive, value-conscious
- Technical sophistication: High
- Decision cycle: 1-7 days
- Support needs: Low (self-serve)

**Key Value Drivers:**
- Cost efficiency
- Rapid deployment
- API access
- No minimum commitments

**Pricing Psychology:**
- Monthly billing preferred (cash flow)
- Transparent, predictable costs
- Free tier for evaluation

**Reference Examples:**
- [QuantConnect](https://www.quantconnect.com/) at $20-40/month for quant researchers
- [Alpaca Elite](https://alpaca.markets/) for sophisticated algorithmic traders

#### Segment B: Mid-Size Hedge Funds & Trading Desks

**Profile:**
- Team size: 5-50 people
- AUM: $10M - $500M
- Mix of technical and business stakeholders
- Decision cycle: 2-8 weeks
- Support needs: Moderate (email, onboarding)

**Key Value Drivers:**
- Performance improvement over existing systems
- Team collaboration features
- Compliance and audit trails
- Integration with existing infrastructure

**Pricing Psychology:**
- Annual contracts acceptable
- ROI-focused (willing to pay for measurable alpha)
- Value customization options

**Reference Examples:**
- [Refinitiv Eikon](https://www.refinitiv.com/) at $22K/year (stripped: $3.6K/year)
- Professional-tier data providers

#### Segment C: Enterprise Institutions & Banks

**Profile:**
- Global banks, large asset managers, tier-1 hedge funds
- AUM: $500M - $100B+
- Team size: 50-1000+ users
- Complex procurement processes
- Decision cycle: 3-12 months
- Support needs: High (dedicated CSM, SLA)

**Key Value Drivers:**
- Enterprise security (SOC 2, ISO 27001)
- On-premise/private cloud deployment
- Custom integrations
- Regulatory compliance
- Executive sponsorship and partnership

**Pricing Psychology:**
- Multi-year contracts preferred
- Value-based not seat-based
- Total cost of ownership important

**Reference Examples:**
- [Bloomberg Terminal](https://www.bloomberg.com/professional/products/bloomberg-terminal/) at $24K-30K/user/year
- Enterprise data platforms with custom pricing

---

## 2. Product Modes (CCEA Architecture)

### 2.1 Three Deployment Modes

Per the CCEA Design Document, the platform operates in three distinct modes:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         CCEA PRODUCT MODES                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  MODE 1: RETAIL RESEARCH SaaS (EU-Friendly)                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  • Cloud-only access: Backtest, Simulation, Training                     │ │
│  │  • NO live execution capability                                          │ │
│  │  • Paper trading and research only                                       │ │
│  │  • Target: Students, researchers, strategy developers                    │ │
│  │  • Regulatory: Pure software provider (no broker license needed)         │ │
│  │  • Pricing: Starter, Pro tiers ($0-$2,000/month)                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  MODE 2: RETAIL LIVE via Local Agent (BYO Host)                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  • Cloud services + Customer-deployed Agent                              │ │
│  │  • Agent runs on customer's machine/server                               │ │
│  │  • Customer controls API keys, execution, kill switch                    │ │
│  │  • Target: Active traders, small prop firms                              │ │
│  │  • Regulatory: Software provider (customer executes)                     │ │
│  │  • Pricing: Pro, Team tiers ($500-$10,000/month)                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  MODE 3: ENTERPRISE ENGINE (On-Prem/VPC)                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  • Full stack deployed in customer environment                           │ │
│  │  • Cloud components run in customer's VPC/on-premise                     │ │
│  │  • Complete air-gap option available                                     │ │
│  │  • Target: Banks, large hedge funds, asset managers                      │ │
│  │  • Regulatory: Software license (customer owns entire stack)             │ │
│  │  • Pricing: Enterprise tier ($100K-$2M+/year)                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Mode Feature Matrix

| Feature | Research SaaS | Live via Agent | Enterprise Engine |
|---------|---------------|----------------|-------------------|
| **Cloud Services** | | | |
| Strategy Workspace | ✅ | ✅ | ✅ (in customer VPC) |
| Backtest & Simulation | ✅ | ✅ | ✅ |
| Training Service | ✅ | ✅ | ✅ |
| Artifact Builder | ✅ | ✅ | ✅ |
| **Agent Capabilities** | | | |
| Live Execution | ❌ | ✅ | ✅ |
| Local API Key Storage | N/A | ✅ | ✅ |
| Local Risk Manager | N/A | ✅ | ✅ |
| Kill Switch | N/A | ✅ | ✅ |
| **Deployment** | | | |
| Location | Our Cloud | Our Cloud + Customer | Customer VPC/On-Prem |
| Agent Host | N/A | Customer | Customer |
| Air-Gap Option | ❌ | ❌ | ✅ |
| **Regulatory** | | | |
| Our Position | Software Provider | Software Provider | Software Licensor |
| Execution Owner | N/A | Customer | Customer |
| API Keys Holder | N/A | Customer (local) | Customer (local) |

### 2.3 Mode-to-Tier Mapping

| Product Mode | Available Pricing Tiers | Target AUM |
|--------------|------------------------|------------|
| **Research SaaS** | Free, Starter, Starter Pro | N/A (research only) |
| **Live via Agent** | Pro, Pro Plus, Pro Max, Team, Team Plus | $100K - $500M |
| **Enterprise Engine** | Team Enterprise, Enterprise (all variants) | $500M+ |

---

## 3. Pricing Model Architecture

### 3.1 Multi-Metric Pricing Framework

Per [Financial Services SaaS Pricing in 2024](https://www.getmonetizely.com/articles/financial-services-saas-pricing-in-2024-strategies-for-success), fintech companies typically use 3-4 pricing metrics. Our model uses:

| Metric | Description | Applies To |
|--------|-------------|------------|
| **Seat License** | Per-user access | All tiers |
| **Compute Usage** | Backtest hours, training time | Pro+ tiers |
| **Data Consumption** | API calls, historical data | Pro+ tiers |
| **AUM-Based** | Percentage of managed assets | Enterprise only |

### 3.2 Tier Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENTERPRISE TIER                               │
│   Custom Pricing ($100K-$1M+/year) - Banks, Tier-1 Funds            │
│   ├── On-premise deployment                                         │
│   ├── Custom integrations                                            │
│   ├── Dedicated success team                                         │
│   └── SLA with financial guarantees                                  │
├─────────────────────────────────────────────────────────────────────┤
│                         TEAM TIER                                    │
│   $3,000-10,000/month - Mid-Size Funds                              │
│   ├── Up to 20 seats                                                 │
│   ├── L3 LOB simulation                                              │
│   ├── Custom model training                                          │
│   └── Priority support                                               │
├─────────────────────────────────────────────────────────────────────┤
│                          PRO TIER                                    │
│   $500-2,000/month - Small Prop Firms                               │
│   ├── 1-5 seats                                                      │
│   ├── L2 execution simulation                                        │
│   ├── Multi-asset strategies                                         │
│   └── Email support                                                  │
├─────────────────────────────────────────────────────────────────────┤
│                         STARTER TIER                                 │
│   $0-99/month - Indie Traders                                        │
│   ├── 1 seat                                                         │
│   ├── Basic backtesting                                              │
│   ├── Community support                                              │
│   └── Limited API access                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Pricing Tables

### 3.1 Starter Tier (Indie Traders)

**Target:** Solo traders, quant researchers, students

| Plan | Monthly | Annual (20% off) | Features |
|------|---------|------------------|----------|
| **Free** | $0 | $0 | SDK access, 100 backtests/mo, community support |
| **Starter** | $49 | $39/mo ($468/yr) | 500 backtests/mo, basic signals, email support |
| **Starter Pro** | $99 | $79/mo ($948/yr) | Unlimited backtests, L1 sim, priority queue |

**Compute Add-ons:**
- Additional backtest hours: $0.10/hour
- GPU training: $2.00/hour
- Historical data: $0.001/API call (first 10K free)

### 3.2 Pro Tier (Small Prop Firms)

**Target:** Prop trading firms, small quant teams (1-5 people)

| Plan | Monthly | Annual | Features |
|------|---------|--------|----------|
| **Pro** | $500 | $5,400/yr ($450/mo) | 3 seats, L2 simulation, multi-asset |
| **Pro Plus** | $1,000 | $10,800/yr ($900/mo) | 5 seats, L2+ parametric TCA, custom features |
| **Pro Max** | $2,000 | $21,600/yr ($1,800/mo) | 10 seats, priority training, dedicated onboarding |

**Included Compute:**
| Plan | Backtest Hours | Training Hours | API Calls |
|------|---------------|----------------|-----------|
| Pro | 1,000/mo | 50/mo | 1M/mo |
| Pro Plus | 5,000/mo | 200/mo | 5M/mo |
| Pro Max | 20,000/mo | 1,000/mo | 20M/mo |

**Overage Rates:**
- Backtest: $0.05/hour (50% discount from Starter)
- Training: $1.50/hour
- API: $0.0005/call

### 3.3 Team Tier (Mid-Size Funds)

**Target:** Hedge funds $10M-500M AUM, trading desks at banks

| Plan | Monthly | Annual | Features |
|------|---------|--------|----------|
| **Team** | $3,000 | $32,400/yr | 10 seats, L3 LOB simulation, SSO |
| **Team Plus** | $6,000 | $64,800/yr | 20 seats, custom model training, audit logs |
| **Team Enterprise** | $10,000 | $108,000/yr | 50 seats, dedicated CSM, custom SLA |

**Enterprise Features by Tier:**
| Feature | Team | Team Plus | Team Enterprise |
|---------|------|-----------|-----------------|
| L3 LOB Simulation | ✅ | ✅ | ✅ |
| Custom Model Training | ❌ | ✅ | ✅ |
| SSO/SAML | ✅ | ✅ | ✅ |
| Audit Logging | ❌ | ✅ | ✅ |
| Custom Integrations | ❌ | ❌ | ✅ |
| Dedicated CSM | ❌ | ❌ | ✅ |
| SLA (Uptime) | 99.5% | 99.9% | 99.95% |
| Support | Business hours | 12x5 | 24x7 |

### 3.4 Enterprise Tier (Banks, Large Funds)

**Target:** Tier-1 hedge funds, global banks, asset managers $500M+ AUM

**Pricing Model: Hybrid (Base + Usage + Success)**

```
Total Annual Cost = Base License + Compute Usage + Success Fee (optional)
```

**Base License (by deployment):**
| Deployment | Annual Base | Includes |
|------------|-------------|----------|
| Cloud Multi-Tenant | $100,000 | 50 seats, standard SLA |
| Cloud Dedicated | $250,000 | 100 seats, dedicated infrastructure |
| Private Cloud (AWS/GCP/Azure) | $500,000 | Unlimited seats, customer VPC |
| On-Premise | $750,000+ | Air-gapped, full source access |

**AUM-Based Component (Optional):**
| AUM Tier | Annual Fee | Notes |
|----------|------------|-------|
| $500M - $2B | 2 bps | 0.02% of AUM |
| $2B - $10B | 1.5 bps | Declining rate |
| $10B - $50B | 1 bp | Volume discount |
| $50B+ | 0.5 bps | Strategic partnership |

**Example Enterprise Pricing:**
- **$1B AUM fund, cloud dedicated:** $250K base + $200K AUM fee = **$450K/year**
- **$10B AUM bank, on-premise:** $750K base + $1M AUM fee = **$1.75M/year**
- **$50B AUM manager, strategic:** Custom negotiated, typically $2-5M/year

**Success Fee Option:**
- 5-10% of documented alpha improvement
- Requires 12-month baseline comparison
- Capped at 50% of base license

---

## 4. Competitive Positioning

### 4.1 Market Pricing Benchmarks

| Competitor | Target Segment | Pricing | Our Advantage |
|------------|---------------|---------|---------------|
| [Bloomberg Terminal](https://www.bloomberg.com/professional/products/bloomberg-terminal/) | Enterprise | $24-30K/seat/yr | ML-first, lower per-seat at scale |
| [Refinitiv Eikon](https://www.refinitiv.com/) | Mid-Market | $22K/yr (full) | Purpose-built for quant trading |
| [QuantConnect](https://www.quantconnect.com/) | Indie | $20-40/mo | Enterprise features, proprietary RL |
| Capital IQ | Enterprise | $20K+/seat/yr | Execution simulation, not just data |

### 4.2 Value-Based Positioning

Per [Pearson Ham Group](https://www.pearsonhamgroup.com/why-value-based-pricing-makes-sense-for-financial-services-software-providers/) on value-based pricing for fintech:

> "The first step is always an assessment of cost structure... establish a floor price below which you cannot price in the long-term."

**Our Value Quantification:**

| Customer Segment | Expected Alpha Improvement | Value Created (1yr) | Our Price | Value Capture |
|-----------------|---------------------------|---------------------|-----------|---------------|
| $10M prop firm | 50-200 bps | $50K-200K | $12K-25K | 5-25% |
| $100M hedge fund | 20-100 bps | $200K-1M | $65K-108K | 10-32% |
| $1B institution | 10-50 bps | $1M-5M | $450K-1M | 20-45% |
| $10B manager | 5-25 bps | $5M-25M | $1.75M-3M | 10-35% |

**Price-to-Value Ratio Target:** 10-35% (industry standard for enterprise software)

---

## 5. Packaging Strategy

### 5.1 Good-Better-Best Framework

Per [HubiFi Enterprise SaaS Pricing Guide](https://www.hubifi.com/blog/enterprise-saas-pricing-guide):

> "Each tier should provide a logical upgrade path, encouraging users to move to higher tiers without feeling forced."

**Feature Gating Logic:**

| Feature Category | Starter | Pro | Team | Enterprise |
|-----------------|---------|-----|------|------------|
| **Execution Sim** | L1 | L2 | L3 | L3 + Custom |
| **Assets** | Crypto only | Multi-asset | Multi-asset | Custom |
| **Strategies** | Signal only | Full training | Custom models | Bespoke |
| **Support** | Community | Email | Dedicated | 24x7 + CSM |
| **Compliance** | None | Basic | Audit logs | Full |
| **Deployment** | Cloud | Cloud | Cloud + VPC | Any |

### 5.2 Add-On Strategy

**Horizontal Add-Ons (Any Tier):**
| Add-On | Price | Description |
|--------|-------|-------------|
| Additional Seats | $100-500/seat/mo | Scale team access |
| Premium Data | $500-2,000/mo | Alternative data feeds |
| GPU Cluster | $2,000-10,000/mo | Dedicated compute |

**Vertical Add-Ons (Tier-Specific):**
| Add-On | Available Tier | Price |
|--------|---------------|-------|
| L3 LOB Module | Pro+ | $1,000/mo |
| Options Pricing | Team+ | $2,000/mo |
| Forex Module | Pro+ | $500/mo |
| CME Futures | Team+ | $1,500/mo |
| Custom Integrations | Enterprise | $25K+ one-time |

### 5.3 Bundling Strategy

**Segment-Specific Bundles:**

**Crypto Trader Bundle (Pro):**
- Crypto spot + perpetuals
- L2+ parametric TCA
- Binance/Bybit integrations
- **$750/mo** (vs $1,200 a la carte, 37% savings)

**Equity Quant Bundle (Team):**
- US Equity + Options
- L3 LOB simulation
- Alpaca/IB integrations
- **$5,000/mo** (vs $7,500 a la carte, 33% savings)

**Multi-Asset Pro Bundle (Enterprise):**
- All asset classes
- All execution tiers
- Custom integrations
- **$200K/yr** (vs $350K a la carte, 43% savings)

---

## 6. Pricing Governance

### 6.1 Discount Authority Matrix

| Discount Level | Approver | Max Discount | Conditions |
|---------------|----------|--------------|------------|
| 0-10% | AE | 10% | Annual commit, case-by-case |
| 11-20% | Sales Manager | 20% | Multi-year, strategic account |
| 21-30% | VP Sales | 30% | Enterprise only, board approval |
| 31%+ | CEO | 40% max | Exceptional cases only |

### 6.2 Price Exception Process

1. **Request:** AE submits discount request with business case
2. **Review:** Manager reviews competitive situation, account potential
3. **Approval:** Appropriate authority approves/modifies
4. **Documentation:** All exceptions logged for pricing analysis
5. **Review:** Quarterly audit of discount patterns

### 6.3 Annual Price Review

Per [SBI Growth research](https://sbigrowth.com/tools-and-solutions/pricing-benchmarks-report-2024):

> "94% of B2B SaaS pricing leaders update pricing at least once yearly, with almost 40% updating as often as once per quarter."

**Review Cadence:**
- **Quarterly:** Package/feature adjustments
- **Annually:** Full pricing review
- **Ad-hoc:** Competitive response, cost changes

**Grandfathering Policy:**
- Existing customers: 12-month price protection
- Renewals: Max 10% increase per year
- New features: Available at current pricing

---

## 7. Segment-Specific Sales Playbooks

### 7.1 Indie/Small Prop (Self-Serve)

**Motion:** Product-Led Growth (PLG)

**Funnel:**
```
Free SDK → Starter Trial → Starter Paid → Pro Upgrade
```

**Key Metrics:**
- Free-to-paid conversion: 5-10%
- Starter-to-Pro upgrade: 15-25%
- Time-to-value: <7 days

**Tactics:**
- In-product upgrade prompts
- Usage-based upgrade triggers (hitting limits)
- Email nurture with case studies
- Community showcase of successful traders

### 7.2 Mid-Size Funds (Inside Sales)

**Motion:** Sales-Assisted PLG

**Funnel:**
```
Inbound Lead → Demo → Trial → POC → Proposal → Close
```

**Cycle:** 2-8 weeks

**Key Metrics:**
- Demo-to-trial: 40-50%
- Trial-to-close: 25-35%
- Average deal size: $50K-150K ACV

**Tactics:**
- Personalized demo with relevant asset class
- 14-day technical POC
- ROI calculator with their data
- Executive sponsor introduction

### 7.3 Enterprise (Field Sales)

**Motion:** Enterprise Field Sales

**Funnel:**
```
Target Account → Multi-Thread → Technical Win → Business Case → Procurement → Close
```

**Cycle:** 3-12 months

**Key Metrics:**
- Technical win rate: 60-70%
- Business case approval: 50-60%
- Average deal size: $300K-2M ACV

**Tactics:**
- Named account strategy
- Multi-threaded engagement (CTO, Head of Quant, CFO)
- Custom POC with production data
- Security/compliance deep-dive
- Executive alignment meeting
- Reference customer calls

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Q1 2025)

- [ ] Implement tiered pricing in billing system
- [ ] Create segment-specific landing pages
- [ ] Build pricing calculator tool
- [ ] Train sales team on new pricing
- [ ] Document discount approval process

### Phase 2: Optimization (Q2 2025)

- [ ] A/B test pricing page layouts
- [ ] Implement usage-based billing
- [ ] Launch add-on modules
- [ ] Create segment-specific case studies
- [ ] Establish pricing committee

### Phase 3: Scale (Q3-Q4 2025)

- [ ] Automate enterprise quoting
- [ ] Implement CPQ system
- [ ] Launch partner pricing program
- [ ] Expand geographic pricing
- [ ] Quarterly pricing review process

---

## 9. Success Metrics

### 9.1 Pricing KPIs

| Metric | Current | Target Y1 | Target Y2 |
|--------|---------|-----------|-----------|
| ACV (Average Contract Value) | N/A | $25K | $50K |
| Gross Margin | N/A | 75% | 80% |
| Net Revenue Retention | N/A | 110% | 125% |
| Price Realization | N/A | 85% | 90% |
| Discount Rate | N/A | <15% | <10% |
| Upgrade Rate | N/A | 20% | 30% |

### 9.2 Segment Mix Targets

| Segment | % of Revenue Y1 | % of Revenue Y3 |
|---------|-----------------|-----------------|
| Starter/Free | 5% | 3% |
| Pro | 25% | 15% |
| Team | 40% | 35% |
| Enterprise | 30% | 47% |

---

## 10. References

### Industry Research

1. [SBI Growth: State of B2B SaaS Pricing 2024](https://sbigrowth.com/tools-and-solutions/pricing-benchmarks-report-2024)
2. [Financial Services SaaS Pricing in 2024](https://www.getmonetizely.com/articles/financial-services-saas-pricing-in-2024-strategies-for-success)
3. [HubiFi: Enterprise SaaS Pricing Guide](https://www.hubifi.com/blog/enterprise-saas-pricing-guide)
4. [Pearson Ham: Value-Based Pricing for Financial Services](https://www.pearsonhamgroup.com/why-value-based-pricing-makes-sense-for-financial-services-software-providers/)

### Competitive Intelligence

5. [Bloomberg Terminal Pricing](https://stockstotrade.com/cost-of-bloomberg-terminal/)
6. [Bloomberg vs Capital IQ vs Refinitiv](https://www.wallstreetprep.com/knowledge/bloomberg-vs-capital-iq-vs-factset-vs-thomson-reuters-eikon/)
7. [QuantConnect Review](https://www.luxalgo.com/blog/quantconnect-review-best-platform-for-algo-trading-2/)
8. [Quant Firm Market Structure](https://www.quantblueprint.com/post/top-quant-firms-list-comp-up-to-500k)

### Best Practices

9. [Top SaaS Pricing Models 2024](https://www.getcacheflow.com/post/top-5-saas-pricing-models)
10. [B2B SaaS Pricing Strategies](https://aventigroup.com/blog/b2b-saas-pricing-strategies-how-to-build-a-model-that-drives-growth/)
11. [SaaS Pricing Models Guide](https://www.cobloom.com/blog/saas-pricing-models)

---

**Document Classification:** INTERNAL STRATEGY
**Owner:** CEO / Head of Revenue
**Review Cycle:** Quarterly
**Next Review:** Q2 2025

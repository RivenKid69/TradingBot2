# CustodiaCloud Product Overview

*One-pager for startup visa applications & investor pitches (equities-first).*

**Canonical positioning**: see [POSITIONING_CANONICAL.md](POSITIONING_CANONICAL.md).

> **Architecture**: CCEA (Cloud-Controlled Execution Architecture) | **Regulatory Posture**: software/ICT provider by design (MiFID II/DORA-aware; AI governance by design)

---

## Architecture: Cloud-Controlled Execution Architecture (CCEA)

**We are a SOFTWARE PROVIDER, not an Investment Adviser or Broker-Dealer.**

Our platform implements **CCEA** - a strict security separation:

| Zone | Responsibility | Secrets Access | Order Execution |
|------|---------------|----------------|-----------------|
| **Cloud** | Research, backtesting, monitoring, lifecycle | **NEVER** | **NEVER** |
| **Agent** | Live execution, credential storage, risk enforcement | **LOCAL ONLY** | **LOCAL ONLY** |

**Security Design Commitments** (enforced at architecture level):
- Cloud **NEVER** stores broker API keys or credentials
- Cloud **NEVER** generates, transmits, or executes live trading instructions (orders/targets/signals)
- All trading happens **ONLY** in the user's local Agent
- User retains **full control** over hard caps (Cloud cannot override)

**What This Means for Customers:**
- Your credentials stay on YOUR machine (or VPC)
- You control all trading-impacting changes via local approval
- Cloud provides research tools and observability - nothing more

**Legal positioning (design intent):**
- We provide software/tools; customers trade through **their own** broker accounts via **their own** Agent
- Cloud has **no secrets** and **no live trading instructions** (orders/targets/signals)
- Regulatory classification depends on activities and jurisdiction; this is **not legal advice**

---

## The Problem

**Algorithmic trading firms spend 6-12 months building infrastructure before deploying their first strategy.**

Current solutions are inadequate:

| Solution | Limitation |
|----------|------------|
| **QuantConnect** | Basic execution models, no risk-aware ML |
| **Alpaca** | Broker only, no intelligence |
| **In-house development** | €200K-500K cost, 12+ months |
| **Academic tools** | Not deployment-ready |

---

## Our Solution

**A platform that reduces trading infrastructure development from months to days, with built-in risk management unavailable elsewhere.**

### Core Innovation: Risk-Aware Execution

**Traditional ML**: Optimizes average returns → ignores catastrophic risks

**Our Approach**: Optimizes returns **while constraining worst-case losses**

```
We implement Conditional Value-at-Risk (CVaR) optimization:
Instead of: maximize E[Return]
We solve:   maximize E[Return] subject to CVaR₅%[Return] ≥ threshold
```

**Result**: Strategies that avoid large drawdowns, not just maximize gains.

**Academic basis**: Chow et al. (2015, JMLR), Dabney et al. (2018, AAAI)

---

## Current Phase: Validation

### Foundation Built — Now Testing with Customers

We have completed the technical foundation. Our focus now is **customer validation**, not feature expansion.

| Phase | Status | Focus |
|-------|--------|-------|
| ~~Technical Development~~ | ✅ Complete | Core platform built |
| **Customer Discovery** | 🔄 Active | 20+ interviews planned |
| **Pilot Program** | 🔜 Phase 1 (0–3 months) | 3-5 European firms |
| Revenue Validation | Planned | Post-pilot |

### MVP Scope (What We're Launching)

| Feature | Status | Customer Value |
|---------|--------|----------------|
| Equities execution (Interactive Brokers) | ✅ Ready | Days to go live |
| Risk-aware position sizing | ✅ Ready | Compliance-supporting risk controls |
| CVaR optimization | ✅ Ready | Downside protection |
| Backtesting | ✅ Ready | Strategy validation |
| Real-time monitoring | ✅ Ready | Operational visibility |

### Deferred Features (Post-Validation)

Note: the underlying engine is **multi-asset** (equities + adjacent listed derivatives + FX; digital assets optional). MVP commercial support and positioning remain **equities-first** until validated by customer demand.

| Feature | When | Trigger |
|---------|------|---------|
| Additional brokers/venues | After repeatable onboarding | Customer demand |
| CME Futures | After 10 paying customers | Enterprise requests |
| Options | Based on pilot feedback | 3+ firm requests |
| Digital assets | Only if demanded | Enterprise / expansion requests |

---

## Target Market: European Systematic Equities Teams

### Why Europe First

| Factor | Europe | US |
|--------|--------|-----|
| Regulatory clarity | MiFID II framework | Fragmented |
| Competition | Lower density | Saturated |
| Market access | Startup visa pathway | Complex |
| Language | English common | English native |

### Target Customer

| Attribute | Profile |
|-----------|---------|
| **Company** | Prop firm or small systematic fund |
| **Size** | 5-50 traders |
| **Location** | Netherlands, Germany, Ireland, France |
| **Need** | Fast, compliance-supporting infrastructure for equities strategies |
| **Budget** | €2,000-5,000/month |

---

## Competitive Positioning

| Capability | Our Platform | QuantConnect | Alpaca |
|------------|--------------|--------------|--------|
| Risk-aware ML | **CVaR-constrained RL** | None | None |
| Execution modeling | **Multi-factor TCA** | Fixed spread | N/A |
| Time to market | **Days** | Weeks | N/A |
| Target customer | Institutional | Retail | Retail/SMB |

---

## Technical Foundation

Our technical depth is an **asset for fast iteration**, not a goal in itself.

| Asset | Benefit |
|-------|---------|
| Robust architecture | Rapid feature changes without breaking production |
| Extensive test coverage | Confidence to iterate quickly |
| Research-backed algorithms | Credibility with technical buyers |
| MiFID II / DORA posture | Controls + evidence tooling to support regulated clients |

*Technical depth enables customer focus, not delays it.*

---

## Intellectual Property

### Novel Algorithms

1. **CVaR-Constrained RL**: Risk-aware decision making
2. **Parametric TCA**: Market-adaptive cost modeling
3. **Twin Critics**: Reduced overestimation in value learning

### Academic Foundation

- Research-backed approach (Almgren-Chriss, Kyle, Dabney, Chow)
- Defensible through complexity and trade secrets

---

## Market Opportunity

**European algorithmic trading market**: Growing segment of €31B global market

**Our focus**: European systematic equities teams seeking:
- Faster time-to-market
- Superior risk management
- Compliance-supporting controls and evidence (MiFID II-style expectations)

---

## Go-to-Market Strategy

### Phase 1: Pilot (0–3 months)

| Element | Specification |
|---------|---------------|
| Cohort size | 3-5 firms |
| Duration | 3 months |
| Pricing | €500/month (discounted) |
| Commitment | Weekly feedback |

### Phase 2: Early Adopters (3–9 months)

| Element | Target |
|---------|--------|
| Paying customers | 10+ firms |
| Price point | €2,000-3,000/month |
| ARR target | €200K+ |

### Phase 3: Scale (9–18 months)

- Expand feature set based on validated demand
- Geographic expansion within EU
- Enterprise tier development

---

## What We Will NOT Do (Until Validated)

- Add new asset classes without customer demand
- Build enterprise features before SMB validation
- Expand geographically before EU product-market fit
- Prioritize features over customer feedback

---

## Summary

| Question | Answer |
|----------|--------|
| **What is it?** | Trading infrastructure platform with built-in risk management |
| **Who is it for?** | European systematic equities teams (prop firms + small funds) |
| **What problem?** | 6-12 months to build infrastructure → days |
| **What's different?** | Risk-aware ML, multi-factor execution modeling |
| **What's next?** | Pilot program with 3-5 European firms |
| **What's the ask?** | Introductions to systematic equities CTOs/PMs in EU |

---

*For detailed validation strategy, see [LEAN_VALIDATION_STRATEGY.md](LEAN_VALIDATION_STRATEGY.md)*

*For pilot program details, see [PILOT_PROGRAM.md](PILOT_PROGRAM.md)*

*For investor materials, see [INVESTOR_BRIEF.md](INVESTOR_BRIEF.md)*

*For beachhead market strategy, see [BEACHHEAD_MARKET_STRATEGY.md](BEACHHEAD_MARKET_STRATEGY.md)*

---

*Last Updated: 2025-12-18*

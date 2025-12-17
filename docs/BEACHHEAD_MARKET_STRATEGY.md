# Beachhead Market Strategy: Risk-First Quantitative Trading Platform

*Strategic Market Focus Based on Geoffrey Moore's "Crossing the Chasm" Framework*

---

## Architecture Foundation: CCEA (Cloud-Controlled Execution Architecture)

> **Key Principle**: All go-to-market strategy is built on the CCEA architecture — **Cloud for research and simulation, Agent for execution**. This architecture enables our legal positioning as a **Software Provider / ICT Provider**, not an execution service.

| Component | Role | Secrets | Orders |
|-----------|------|---------|--------|
| **Cloud** | Research IDE, backtesting, artifact builder, monitoring | **NEVER** | **NEVER** |
| **Agent** | Local execution, vault, risk enforcement, order creation | **LOCAL** | **YES** |

**Legal Posture**: We provide **algorithmic trading research and infrastructure tools**. Users trade through **THEIR OWN broker accounts** via **their local Agent**. We do NOT hold client assets, credentials, or execute orders.

---

## Executive Summary

**Our Beachhead**: Mid-size European systematic equity funds (€10M-200M AUM) in the UK/EU who need institutional-grade risk management and regulatory-compliant algorithmic trading infrastructure — deployed via **CCEA architecture** for clear regulatory boundaries.

**Why Equity, Not Crypto**:
- **Institutional credibility**: Equities are the language of serious capital markets
- **Regulatory alignment**: MiFID II, MAR, DORA compliance demonstrates maturity
- **Market size**: €8.5T European asset management market vs €987B crypto
- **Risk-first positioning**: Our CVaR-RL technology addresses the #1 institutional concern
- **CCEA advantage**: Clear "Cloud = research tools, Agent = your execution" story resonates with compliance officers

**Why This Focus Matters**: While our platform supports 5 asset classes (crypto, equities, futures, forex, options), we deliberately lead with **institutional equity trading** because:

1. **Credibility with allocators**: Pension funds, family offices, and fund-of-funds evaluate technology partners based on equity track records
2. **Regulatory proof**: MiFID II Article 17 compliance for algo trading is non-negotiable for EU institutional clients
3. **Risk management differentiation**: Our CVaR-RL engine is unique in the market — this advantage is most visible in equity drawdown control
4. **Expansion pathway**: Once established in equities, we expand to CME futures, then crypto — NOT the reverse

---

## Theoretical Foundation: Why Beachhead Strategy Works

### Geoffrey Moore's "Crossing the Chasm"

> "A beachhead market is the place where, once you gain a dominant market share, you will have the strength to attack adjacent markets with different opportunities, building a larger company with each new following."
>
> — Bill Aulet, MIT Sloan

The "chasm" is the gap between early adopters (who buy on vision) and the mainstream market (who buy on proof). **Crossing this chasm requires:**

1. **Focused resources** on a single, well-defined segment
2. **Whole product** delivery that fully solves the segment's problem
3. **Word-of-mouth** that spreads within the segment's community
4. **Reference customers** who become advocates

**Key Insight**: Startups that try to serve everyone before dominating one segment almost always fail to cross the chasm.

**Sources**:
- [Geoffrey Moore on finding your beachhead, crossing the chasm](https://www.lennysnewsletter.com/p/geoffrey-moore-on-finding-your-beachhead)
- [The Complete Guide to Crossing the Chasm for SaaS Startups](https://leanb2bbook.com/blog/crossing-chasm-saas-startups/)
- [MIT Sloan: Launching a Successful Startup — The Beachhead Market](https://executive.mit.edu/launching-a-successful-start-up-3-the-beachhead-market-MC7FUMDZ6IU5AIPP4WGIPN2PZJI4.html)

---

## Our Beachhead: European Systematic Equity Funds

### Segment Definition

| Attribute | Specification | Rationale |
|-----------|---------------|-----------|
| **Firm Type** | Systematic/quantitative funds (own capital or third-party) | Sophisticated buyers who evaluate technology rigorously |
| **Size** | €10M-200M AUM | Large enough to pay enterprise pricing, small enough for direct sales |
| **Asset Class** | European/US equities (Alpaca, Interactive Brokers) | Institutional credibility, regulatory familiarity |
| **Geography** | UK, Ireland, Luxembourg, Netherlands | MiFID II jurisdiction, English-speaking, strong fund ecosystem |
| **Regulatory Status** | AIFM-licensed or FCA-regulated | Compliance-conscious buyers value our MiFID II alignment |
| **Technology** | Upgrading from Excel/Python scripts to production infrastructure | Clear pain point we solve |
| **Budget** | €3,000-10,000/month | Validated willingness to pay for institutional-grade tools |
| **Deployment** | BYO host (VPS/on-prem) for Agent | CCEA requirement: execution in client environment |

### CCEA Deployment Model for Beachhead

| Product Mode | Description | Target Segment |
|--------------|-------------|----------------|
| **Retail Research SaaS** | Cloud research + optional Agent | Quants evaluating strategies |
| **Retail Live via Local Agent** | Full deployment with local execution | Active trading firms |
| **Enterprise Engine** | On-prem/VPC, all in client infra | Regulated funds (AIFM, UCITS) |

### Why Equity Focus Over Crypto?

#### Strategic Positioning Analysis

| Factor | Crypto First | Equity First (Our Choice) |
|--------|--------------|---------------------------|
| **Investor Perception** | "Crypto casino", speculative | "Serious capital markets player" |
| **Regulatory Pathway** | MiCA (new, uncertain) | MiFID II (established, understood) |
| **Reference Value** | Limited for institutional expansion | Opens doors to pension funds, family offices |
| **Risk Story** | "Volatile markets need risk management" | "Proven drawdown control in regulated markets" |
| **Competition** | Crowded (Binance, FTX refugees, QuantConnect) | Underserved (no SMB-friendly CVaR solutions) |
| **Expansion Logic** | Crypto → Equity is hard (credibility gap) | Equity → Crypto is natural (technology reuse) |

#### The Credibility Ladder

```
YEAR 3+: CRYPTO & ALTERNATIVE ASSETS
┌─────────────────────────────────────────────────────────────────┐
│  Crypto spot/futures, DeFi integration                          │
│  • Leverage equity track record for credibility                 │
│  • Cross-sell to existing equity clients expanding to crypto    │
│  • "Same risk engine, different asset class"                    │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
YEAR 2: CME FUTURES & COMMODITIES
┌─────────────────────────────────────────────────────────────────┐
│  ES, NQ, GC, CL futures (via Interactive Brokers)               │
│  • Adjacent to equity clients (macro overlay strategies)        │
│  • Higher leverage = higher risk = CVaR value proposition       │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
YEAR 1: EQUITY BEACHHEAD (START HERE)
┌─────────────────────────────────────────────────────────────────┐
│  European/US equities via Alpaca, Interactive Brokers           │
│  • MiFID II-aligned design, institutional credibility           │
│  • Risk management is the story (not speculation)               │
│  • Build case studies, testimonials, regulatory track record    │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Segment?

#### 1. Acute Pain Point: Risk Management Gap

> **80% of quantitative funds cite risk management as their top technology challenge.**
> — Preqin "Alternative Asset Manager Technology Survey" (2024)

The mid-market systematic fund has a specific, unsolved problem:

| Pain | Current Solution | Why It Fails |
|------|------------------|--------------|
| **Drawdown control** | Simple stop-losses | No tail risk awareness |
| **Position sizing** | Fixed percentages | Ignores volatility regime |
| **Regulatory reporting** | Manual Excel | MiFID II Art.17 requires algo-level monitoring |
| **Backtest-to-live gap** | "It worked in backtest" | No sim-to-live parity measurement |

**Our Solution**: CVaR-aware position sizing with regulatory-compliant risk monitoring solves ALL of these.

#### 2. Word-of-Mouth Network Exists

The European systematic fund community is **tight-knit and networked**:

| Network | Description | Our Access Strategy |
|---------|-------------|---------------------|
| **CFA Societies** | London, Dublin, Luxembourg chapters | Speaking engagements on ML risk |
| **AIMA (Alternative Investment Management Association)** | 2,000+ member firms | Conference sponsorship, thought leadership |
| **Hedge Fund Club UK** | Quarterly events, 500+ attendees | Networking, panel participation |
| **Quantitative Finance LinkedIn Groups** | 50,000+ members in EU-focused groups | Content marketing, DM outreach |
| **Prime Broker Networks** | Goldman Sachs, Morgan Stanley, BNP | Referral partnerships (Year 2) |

**Moore's Criterion Met**: "Customers within the market communicate with each other — word of mouth is possible."

#### 3. Competition is Fragmented in Risk-First Space

| Competitor | Focus | Gap We Fill |
|------------|-------|-------------|
| **Bloomberg Terminal** | Data + analytics | Not a trading platform, €24K/year per seat |
| **QuantConnect** | Retail algo trading, education | No institutional-grade risk management |
| **Alpaca** | Broker/API, US equities | Not a full platform, no risk engine |
| **Eze Software** | Enterprise OMS/EMS | €500K+ implementation, not for SMB |
| **In-house** | Custom Python/R scripts | No production-grade risk controls |

**No competitor has focused on the mid-market systematic fund with CVaR-based risk management.** This is our "unoccupied beachhead."

#### 4. Market Size Validation

| Metric | Data | Source |
|--------|------|--------|
| **European AUM (alternative assets)** | €2.5T (2024) | EFAMA |
| **Systematic/quant funds (EU)** | €350B AUM | HFR Global Hedge Fund Report |
| **Mid-market funds (€10M-200M AUM)** | 2,500-4,000 funds | Preqin |
| **Technology budget (% of AUM)** | 0.5-1.5% | Deloitte "Cost of Running a Hedge Fund" |

**Market Size Estimate (Beachhead)**:
- 2,500-4,000 mid-market systematic funds in EU
- 20% addressable with quantitative equity focus = 500-800 funds
- At €5,000/month average: **€30-48M ARR** opportunity in beachhead alone

#### 5. Regulatory Alignment (Key Differentiator)

| Regulation | Requirement | Our Compliance | CCEA Advantage |
|------------|-------------|----------------|----------------|
| **MiFID II Article 17** | Algo trading risk controls, kill switch, testing requirements | ✅ Built-in kill switch, pre-trade risk checks | Kill switch in Agent (local enforcement) |
| **MAR 596/2014** | Market abuse prevention | ✅ Unusual activity monitoring, audit trails | Audit logs in both Cloud and Agent |
| **DORA** | ICT risk management, third-party oversight | ✅ SOC 2 roadmap, secure-by-design | Clear Cloud/Agent boundary aids vendor assessment |
| **ESMA Guidelines** | Algo trading testing, validation | ✅ Sim-to-live parity metrics, backtesting validation | Sim in Cloud, live in Agent (separation) |

**CCEA Legal Posture for Regulated Clients**:

| We Are | We Are NOT |
|--------|-----------|
| Software Provider / ICT Provider | Investment Adviser |
| Algorithmic trading research tools | Broker-Dealer |
| Strategy development platform | Custodian |
| Infrastructure for client-controlled execution | Execution Service |

**Positioning Statement**: *"The only mid-market trading platform with MiFID II Article 17 compliance built-in, not bolted-on — with clear Cloud/Agent separation for regulatory clarity."*

---

## Risk Management as the Core Value Proposition

### Why Risk is Our Story

| Traditional Pitch | Our Risk-First Pitch |
|-------------------|---------------------|
| "Trade faster" | "Protect capital during drawdowns" |
| "Better backtest" | "Confidence your backtest will work live" |
| "More asset classes" | "Same risk discipline across all assets" |
| "AI-powered trading" | "Risk-aware AI that knows when NOT to trade" |

### CVaR-RL: The Technical Moat

**Conditional Value-at-Risk (CVaR)** is the gold standard for institutional risk measurement — it captures tail risk that VaR misses.

**Our Innovation**: We embedded CVaR directly into the reinforcement learning reward function, so the AI **learns to avoid tail risk**, not just maximize returns.

```
Traditional RL:    Maximize E[returns]           → Risk-blind
Our CVaR-RL:       Maximize E[returns] - λ×CVaR  → Risk-aware by construction
```

**Why This Matters to Institutional Buyers**:

| Buyer Concern | Our Answer |
|---------------|------------|
| "What if it blows up?" | CVaR-aware policy won't take tail-risk bets |
| "How do I explain to my investors?" | Industry-standard risk metrics (CVaR, Sharpe, Max DD) |
| "Is it tested?" | 11,063 test cases, sim-to-live parity monitoring |
| "Is it compliant?" | MiFID II risk controls built-in, audit trails available |

### Risk Metrics Dashboard (MVP Feature)

| Metric | Description | Regulatory Relevance |
|--------|-------------|----------------------|
| **Real-time CVaR** | 95% worst-case loss estimate | ESMA algo trading guidelines |
| **Drawdown Monitor** | Rolling max drawdown with alerts | MiFID II Art.17 risk controls |
| **Position Sizing** | Volatility-adjusted, CVaR-constrained | Best execution requirements |
| **Kill Switch Status** | One-click halt all trading | MiFID II Art.17 mandatory |
| **Sim-to-Live Parity** | Confidence bounds on backtest vs live | ESMA testing requirements |

---

## Staged Expansion: From Equities to Multi-Asset

### Expansion Roadmap

```
YEAR 1: EQUITY BEACHHEAD DOMINANCE
┌─────────────────────────────────────────────────────────────────┐
│  Mid-Market Systematic Equity Funds (€10-200M AUM)               │
│  • UK, Ireland, Luxembourg, Netherlands                          │
│  • European/US equities via Alpaca, Interactive Brokers          │
│  • Target: 15+ paying customers, €500K+ ARR                      │
│  • Risk-first positioning, MiFID II compliance story             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
YEAR 2: ADJACENT SEGMENTS (Bowling Pins)
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│   Pin 1: Larger   │  │  Pin 2: Adjacent  │  │  Pin 3: Adjacent  │
│   EU Equity Funds │  │  Asset Class      │  │  Geography        │
│   (€200M-1B AUM)  │  │  (CME Futures)    │  │  (Switzerland,    │
│                   │  │                   │  │   Germany, France)│
└───────────────────┘  └───────────────────┘  └───────────────────┘
                              │
                              ▼
YEAR 3: MULTI-ASSET EXPANSION
┌─────────────────────────────────────────────────────────────────┐
│  Full Multi-Asset Platform for Quantitative Trading Firms        │
│  • Equities, Futures, Forex, Options, Crypto                     │
│  • Global markets (EU, US, APAC)                                 │
│  • SMB to Enterprise tiers                                       │
│  • Crypto offered as "additional asset class" with equity        │
│    credibility established                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Expansion Triggers (NOT Before)

| Expansion | Trigger Condition | Evidence Required |
|-----------|-------------------|-------------------|
| **Larger EU Funds** | 15+ paying customers in beachhead | Repeatable sales process, 2+ case studies |
| **CME Futures** | 5+ customer requests | Validated demand signal, existing client expansion |
| **Additional Geographies** | Referrals from EU customers | Word-of-mouth working |
| **Forex** | 3+ specific requests | Feature demand validated |
| **Crypto** | 5+ requests from equity clients | Cross-sell opportunity, credibility established |

### Why Crypto is Year 3, Not Year 1

| Reason | Explanation |
|--------|-------------|
| **Credibility sequence** | Equity → Futures → Crypto is the institutional trust ladder |
| **Regulatory maturity** | MiCA still uncertain; MiFID II is established |
| **Reference value** | "They manage €50M in equities" > "They trade Bitcoin" for institutional allocators |
| **Cross-sell logic** | Equity client expands to crypto (natural) vs crypto client "upgrades" to equities (unnatural) |
| **Competitive dynamics** | Crypto infrastructure is commoditized; equity risk management is not |

---

## Competitive Positioning in Beachhead

### Competitive Matrix: Mid-Market Systematic Equity Funds

| Capability | Our Platform | Bloomberg | QuantConnect | Alpaca | In-House |
|------------|--------------|-----------|--------------|--------|----------|
| **CVaR Risk Management** | ✅ Built-in | ❌ Separate tools | ❌ None | ❌ None | ❌ Build from scratch |
| **MiFID II Alignment** | ✅ Designed-in | ⚠️ Manual setup | ❌ US-centric | ❌ US-centric | ❓ Depends |
| **Execution Modeling** | Multi-factor TCA | Via EMSX | Fixed spread | Basic | Varies |
| **Time to Live** | Days | Months | Weeks | Weeks | 6-12 months |
| **Price Point** | €3-10K/month | €24K+/seat/year | Free-$250/month | Free-$99/month | €200K+ build cost |
| **Target Customer** | SMB Institutional | Enterprise | Retail/prosumer | Retail/SMB | Enterprise |

### Positioning Statement

**For** mid-market European systematic equity funds (€10M-200M AUM)
**Who** need institutional-grade risk management with regulatory compliance
**Our platform** is a risk-first quantitative trading infrastructure
**That** provides CVaR-aware execution and MiFID II-compliant risk monitoring
**Unlike** Bloomberg (too expensive, not risk-native) or QuantConnect (retail-focused, no compliance)
**We** combine academic-grade risk research with deployment-ready infrastructure at SMB-accessible pricing

---

## Multi-Asset Platform: Strategic Asset, Not Liability

### Reframing the Narrative

| Perceived Concern | Our Reframe |
|-------------------|-------------|
| "Why build 5 asset classes?" | "Risk engine is asset-agnostic; equities are the entry point, not the limit" |
| "Isn't crypto risky?" | "Crypto is a Year 3 expansion for clients who start with equities" |
| "Are you spreading too thin?" | "Equity focus for GTM; architecture enables rapid expansion once PMF achieved" |

### Why Multi-Asset is a Moat (Not a Distraction)

1. **Cross-Sell Potential**: Equity fund adds macro overlay with CME futures → upsell
2. **Technology Reuse**: Same CVaR engine, same risk dashboard, different asset class
3. **Competitive Barrier**: 2+ years of development, 11,000+ tests, 5 exchange integrations = hard to replicate
4. **Regulatory Ready**: Single compliance framework covers MiFID II, EMIR, and future MiCA

### How We Communicate This

**To Accelerators/Investors**:
> "Our platform is built for multi-asset quantitative trading with institutional-grade risk management. Our **go-to-market starts with European systematic equity funds** — the most compliance-conscious, risk-aware segment of the market. This positions us as a serious infrastructure provider, not a speculative trading tool. As we establish credibility in equities, we expand to futures, forex, and eventually crypto — following the institutional trust ladder."

**To Customers**:
> "We specialize in helping systematic funds deploy equity strategies with CVaR-aware risk management and MiFID II-compliant monitoring. As you grow into other asset classes, our platform scales with you — same risk discipline, same dashboard, new markets."

---

## Action Plan: Beachhead Execution

### Phase 1: Customer Discovery (Month 1-2)

| Activity | Target | Output |
|----------|--------|--------|
| Customer interviews | 20+ systematic fund CTOs/PMs | Pain point validation |
| LinkedIn outreach (Quant + Risk roles) | 100 targeted connections | 15 discovery calls |
| CFA Society event attendance | 2 events (London, Dublin) | Network building |
| AIMA content contribution | 1 whitepaper on CVaR for algo trading | Thought leadership |

### Phase 2: Pilot Launch (Month 3-5)

| Activity | Target | Output |
|----------|--------|--------|
| Pilot cohort | 5-8 firms | Usage data, feedback |
| Weekly check-ins | 100% participation | Feature priorities |
| Regulatory review | 2 compliance officers | Compliance validation |
| NPS measurement | >40 score | Validation signal |

### Phase 3: Early Revenue (Month 6-12)

| Activity | Target | Output |
|----------|--------|--------|
| Conversion | 60%+ of pilots | 4-6 paying customers |
| Referrals | 1+ per customer | Word-of-mouth working |
| ARR | €200K+ | Revenue validation |
| Case studies | 2+ written, 1+ video | Social proof |

### Phase 4: Expansion (Year 2+)

| Trigger | Action |
|---------|--------|
| 15+ paying customers | Expand to larger EU funds (€200M-1B) |
| 5+ CME futures requests | Add CME futures support |
| Referral from EU to Switzerland | Expand geographic focus |
| 5+ crypto requests from equity clients | Add crypto as expansion asset class |

---

## Key Metrics: Beachhead Success Criteria

### North Star

**Become the default risk management platform for mid-market European systematic equity funds** = 10%+ market share of addressable segment within 3 years

### Leading Indicators

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Activation Rate** | >80% | Firms completing onboarding |
| **Time to First Live Trade** | <14 days | From signup to production |
| **Risk Dashboard Engagement** | >5x/week | CVaR/drawdown dashboard views |
| **NPS Score** | >45 | Quarterly survey |
| **Referral Rate** | >25% | Customers referring peers |
| **Compliance Validation** | Target: All | Pilot firms pass their internal compliance review |

### Lagging Indicators (Year 1)

| Metric | Target | Evidence |
|--------|--------|----------|
| **Paying Customers** | 15+ | Product-market fit |
| **ARR** | €500K+ | Revenue validation |
| **Customer Retention** | >85% annual | Value delivery |
| **Case Studies** | 3+ | Social proof |
| **Regulatory Validation** | 2+ compliance letters | MiFID II credibility |

---

## Investor FAQ: Addressing Concerns

### Q: Why lead with equities when crypto is growing faster?

**A**: Three reasons:
1. **Credibility sequence**: Institutional allocators evaluate technology partners based on equity track records. Crypto-first companies face an uphill battle for institutional credibility.
2. **Risk story clarity**: "We help you control drawdowns in volatile equity markets" is more defensible than "We help you trade crypto."
3. **Regulatory proof**: MiFID II compliance demonstrates maturity; MiCA is still evolving.

### Q: Isn't the equity market saturated with competition?

**A**: The *retail* equity market is saturated. The *mid-market institutional* segment with CVaR-native risk management is **underserved**. Bloomberg is too expensive (€24K/seat), QuantConnect is too retail, and in-house solutions lack production-grade risk controls.

### Q: What if customers want crypto, not equities?

**A**: We track this as a **pivot signal**. If 70%+ of prospects want crypto, we reconsider. But initial research shows systematic funds care more about risk management and compliance than asset class novelty.

### Q: How do you know this beachhead is big enough?

**A**:
- **2,500-4,000 mid-market systematic funds** in EU
- **€30-48M ARR** opportunity in beachhead segment alone
- **Big enough to matter, small enough to lead** (Moore's criterion)

### Q: Won't larger competitors just copy your CVaR approach?

**A**: CVaR is not new. Our moat is **integration**: CVaR embedded in the RL training loop, not bolted on as a post-hoc filter. This requires:
- 2+ years of research (published academic references)
- 11,000+ test cases validating behavior
- Sim-to-live parity measurement (unique to us)

Copying requires recreating our research and testing infrastructure — an 18-24 month effort.

---

## Appendix: References & Sources

### Academic & Practitioner Sources

1. **Moore, G. (1991/2014)**. *Crossing the Chasm: Marketing and Selling Disruptive Products to Mainstream Customers*. Harper Business.

2. **Aulet, B. (2013)**. *Disciplined Entrepreneurship: 24 Steps to a Successful Startup*. Wiley.

3. **Rockafellar, R.T. & Uryasev, S. (2000)**. "Optimization of Conditional Value-at-Risk." *Journal of Risk*.

4. **ESMA (2022)**. "Guidelines on MiFID II Product Governance Requirements."

5. **Preqin (2024)**. "Alternative Asset Manager Technology Survey."

### Case Studies & Examples

- [Geoffrey Moore on Beachhead Strategy](https://www.lennysnewsletter.com/p/geoffrey-moore-on-finding-your-beachhead)
- [MIT Beachhead Market Framework](https://executive.mit.edu/launching-a-successful-start-up-3-the-beachhead-market-MC7FUMDZ6IU5AIPP4WGIPN2PZJI4.html)
- [Crossing the Chasm for SaaS](https://leanb2bbook.com/blog/crossing-chasm-saas-startups/)

### Market Data

- [EFAMA European Asset Management Report](https://www.efama.org)
- [HFR Global Hedge Fund Report](https://www.hfr.com)
- [Deloitte "Cost of Running a Hedge Fund"](https://www2.deloitte.com)
- [AIMA Regulatory Resources](https://www.aima.org)

### Regulatory References

- [MiFID II Article 17 — Algorithmic Trading](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32014L0065)
- [MAR 596/2014 — Market Abuse Regulation](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32014R0596)
- [ESMA Algo Trading Guidelines](https://www.esma.europa.eu/policy-rules/mifid-ii-and-mifir)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | December 2024 | Product/Strategy | Initial creation (crypto focus) |
| 2.0 | December 2025 | Product/Strategy | **Complete rewrite**: Equity-first beachhead, risk management positioning, MiFID II compliance focus |
| 2.1 | December 2025 | Product/Strategy | **CCEA alignment**: Added CCEA architecture foundation, legal posture, Cloud/Agent separation |

---

*Related Documents*:
- [MVP_FOCUS.md](MVP_FOCUS.md) — Feature scope definition
- [REGULATORY_COMPLIANCE_STRATEGY.md](REGULATORY_COMPLIANCE_STRATEGY.md) — Compliance approach
- [IP_PROTECTION_STRATEGY.md](docs/business/IP_PROTECTION_STRATEGY.md) — IP protection framework
- [COMPETITIVE_MOAT.md](docs/business/COMPETITIVE_MOAT.md) — Competitive analysis
- [../Design Doc CCEA Cloud.txt](../Design%20Doc%20CCEA%20Cloud.txt) — Master CCEA architecture document

*Aligned with: Design Doc CCEA Cloud v1.0*

# CustodiaCloud European Pilot Program

*Structured customer validation for systematic equities teams (equities-first).*

**Canonical positioning**: see [POSITIONING_CANONICAL.md](POSITIONING_CANONICAL.md).

---

## CCEA Architecture Overview

> This pilot program follows the **CCEA (Cloud-Controlled Execution Architecture)**. Pilots will deploy:
> - **Cloud**: Research IDE, backtesting, monitoring (SaaS)
> - **Agent**: Local execution runtime on pilot firm's infrastructure (BYO host)

| Component | Pilot Responsibility | Our Responsibility |
|-----------|---------------------|-------------------|
| **Cloud** | Use research/sim tools | Provide SaaS platform |
| **Agent** | Host on their VPS/machine | Provide daemon + support |
| **Secrets** | Store in local vault | **NEVER** access or store |
| **Orders** | Created/sent by Agent | **NEVER** create or send |

**Why CCEA matters for pilots**: Clear regulatory boundary — we provide software tools, pilots control their own execution.

---

## Program Overview

A 3-month structured pilot program designed to validate product-market fit with European systematic equities teams before scaling — using the CCEA architecture for clean Cloud/Agent separation.

| Attribute | Specification |
|-----------|---------------|
| **Duration** | 3 months |
| **Cohort Size** | 3-5 firms |
| **Pricing** | €500/month (80% discount from target) |
| **Commitment** | Weekly feedback, usage data sharing |
| **Target Start** | Phase 1 (0–3 months) |
| **Deployment Model** | Cloud SaaS + Local Agent (BYO host) |

---

## Pilot Objectives

### What We're Testing

| Hypothesis | Validation Method | Success Criteria |
|------------|-------------------|------------------|
| Firms need faster time-to-market | Time to first live trade | < 7 days |
| Risk-aware execution is valued | Feature usage metrics | 80%+ use CVaR limits |
| Equities-first is the right wedge | Asset class preference | 60%+ satisfied with equities-first |
| €2-5K/month is acceptable | Conversion interviews | 50%+ willing to pay |
| Platform is reliable enough | Uptime, error rates | 99.5%+ uptime |

### What We're NOT Testing (Yet)

- Broadening default production support beyond equities (we will record demand signals for futures/options/FX during the pilot)
- Enterprise features (white-label, API)
- High-frequency trading use cases
- Retail trader market

---

## Ideal Pilot Customer

### Firm Profile

| Criterion | Specification | Why |
|-----------|---------------|-----|
| **Size** | 5-20 traders | Large enough to validate, small enough to engage |
| **Location** | EU (NL, DE, IE, FR) | Regulatory alignment, timezone |
| **Stage** | New or expanding | Need for new infrastructure |
| **Capital** | €1-10M AUM | Serious but not institutional |
| **Technology** | In-house quant team | Can evaluate technical quality |

### Qualification Questions

1. Are you currently building or evaluating trading infrastructure?
2. Do you trade or plan to trade listed equities systematically?
3. What is your timeline to deploy new strategies?
4. Who makes technology decisions at your firm?
5. What is your budget for trading infrastructure?

### Disqualification Criteria

- Retail/individual traders (not our segment)
- Firms requiring non-equities support as a hard gate for the pilot
- Firms with < 6 months runway
- Firms unwilling to share feedback

---

## Program Structure

### Timeline

```
Week 0-1: Onboarding
├── Account setup
├── Technical orientation
├── First strategy configuration
└── Success criteria alignment

Week 2-4: Active Usage
├── Strategy deployment
├── Risk limit configuration
├── Daily monitoring
└── Weekly check-in calls

Week 5-8: Iteration
├── Feature feedback implementation
├── Workflow optimization
├── Performance review
└── Bi-weekly calls

Week 9-12: Evaluation
├── Full usage assessment
├── NPS survey
├── Conversion discussion
└── Reference request
```

### Weekly Cadence

| Day | Activity | Participants |
|-----|----------|--------------|
| Monday | Usage metrics review | Internal team |
| Wednesday | Pilot check-in call | 1:1 with each firm |
| Friday | Week summary email | All pilot firms |

---

## Onboarding Process (CCEA)

### Day 1: Cloud Onboarding

- [ ] Welcome email with Cloud credentials
- [ ] Access to documentation portal
- [ ] Slack channel invitation
- [ ] Calendar invite for kickoff call
- [ ] Cloud account activated

### Day 2-3: Agent Setup (Local Infrastructure)

- [ ] Agent installation on pilot's VPS/machine
- [ ] Agent registration with Cloud (device key exchange)
- [ ] Local vault initialization
- [ ] Broker API keys stored in **LOCAL VAULT** (never sent to Cloud)
- [ ] Agent health check (heartbeat visible in Cloud)

### Day 4-5: Research & Strategy Development (Cloud)

- [ ] First backtest completed in Cloud
- [ ] Strategy artifact built and signed
- [ ] Risk parameter configuration reviewed
- [ ] Paper trading environment test

### Day 6-7: Live Deployment via Agent

- [ ] Cloud sends REQUEST_START_RUN to Agent
- [ ] Local approval of TRADING_IMPACTING changes
- [ ] Strategy running on Agent
- [ ] Telemetry visible in Cloud monitoring dashboard
- [ ] Kill switch tested

### Success Checkpoint (Day 7)

| Metric | Target | Zone |
|--------|--------|------|
| Cloud account activated | Yes | Cloud |
| Agent installed and registered | Yes | Agent |
| Local vault configured | Yes | Agent |
| First backtest run | Yes | Cloud |
| Risk limits set | Yes | Agent |
| Strategy artifact deployed | Yes | Cloud → Agent |
| Live run started (with local approval) | Yes | Agent |
| Telemetry visible | Yes | Cloud |

---

## Feedback Collection

### Weekly Feedback Form

**Sent every Friday, 5 questions:**

1. How satisfied are you with the platform this week? (1-10)
2. What worked well?
3. What frustrated you?
4. What feature would make the biggest difference?
5. Would you recommend us to a peer? (NPS)

### Monthly Deep Dive

**30-minute call covering:**

- Overall experience assessment
- Feature priority discussion
- Competitive comparison
- Pricing sensitivity exploration
- Roadmap input

### Exit Interview (Month 3)

**Key questions:**

1. Did the platform deliver on the promise of faster time-to-market?
2. What would need to change for you to become a paying customer?
3. What price point would you consider fair value?
4. Would you provide a reference or case study?
5. What features would you prioritize for next year?

---

## Support Model

### Pilot-Specific Support

| Channel | Response Time | Availability |
|---------|---------------|--------------|
| Dedicated Slack | < 2 hours | Business hours (EU) |
| Email | < 24 hours | 24/7 |
| Weekly call | Scheduled | Fixed time |
| Emergency hotline | < 30 minutes | Trading hours |

### Escalation Path

```
Tier 1: Slack/Email (daily issues)
    │
    ▼
Tier 2: Scheduled call (feature requests)
    │
    ▼
Tier 3: Founder involvement (critical issues)
```

---

## Success Metrics

### Activation Metrics (Week 1)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Account created | 100% | System log |
| Broker connected | 100% | API validation |
| First backtest | 90% | Usage log |
| Risk limits configured | 90% | Configuration audit |

### Engagement Metrics (Ongoing)

| Metric | Target | Frequency |
|--------|--------|-----------|
| Weekly active users | 80% | Weekly |
| Strategies deployed | 2+ per firm | Monthly |
| Backtests run | 10+ per firm/month | Monthly |
| Feature adoption | 70%+ using core features | Monthly |

### Outcome Metrics (Month 3)

| Metric | Target | Method |
|--------|--------|--------|
| NPS Score | > 40 | Survey |
| Conversion intent | 50% | Exit interview |
| Willing to refer | 60% | Exit interview |
| Case study consent | 40% | Request |

---

## Pricing & Terms

### Pilot Pricing

| Component | Price | Notes |
|-----------|-------|-------|
| Platform access | €500/month | 80% discount from target |
| Setup fee | €0 | Waived for pilot |
| Support | Included | Premium support included |
| Commitment | 3 months | Minimum term |

### Post-Pilot Options

| Option | Price | Conditions |
|--------|-------|------------|
| Early adopter rate | €2,000/month | 12-month commitment |
| Standard rate | €3,000/month | Month-to-month |
| Annual prepay | €30,000/year | 17% discount |

### Pilot Agreement Terms

- Usage data may be used for product improvement (anonymized)
- Participation in monthly feedback calls required
- Reference/case study participation encouraged (not required)
- No SLA guarantees during pilot period
- 30-day notice for early termination

---

## Recruitment Strategy

### Target List Development

**Sources:**
- LinkedIn Sales Navigator (CTO, Head of Trading titles)
- Trading conference attendee lists
- Prop trading firm directories (PropTradingFirms.com)
- Quantitative finance communities (Wilmott, QuantNet)
- Referrals from advisors/investors

### Outreach Sequence

**Email 1 (Day 0): Introduction**
> Subject: Faster path to live trading for [Firm Name]?
>
> [Personalized intro based on research]
>
> We're building a platform that helps prop firms go from strategy to live trading in days, not months. We're looking for 3-5 European firms to join our pilot program.
>
> Would you be open to a 15-minute call to see if there's a fit?

**Email 2 (Day 3): Value Prop**
> Subject: Re: Faster path to live trading
>
> Quick follow-up — one of our early testers deployed their first strategy in 4 days (vs. their estimate of 3 months to build in-house).
>
> If you're evaluating trading infrastructure, happy to share how we do it.

**Email 3 (Day 7): Social Proof**
> Subject: How [Similar Firm] reduced their time-to-market
>
> [Brief case study or testimonial]
>
> Last chance to connect this week — I'll assume timing isn't right if I don't hear back.

### Qualification Call Script

**Opening (2 min):**
> "Thanks for taking the time. I'd love to learn about your trading operation and see if our pilot program might be a fit. Can you tell me about your current infrastructure setup?"

**Discovery (10 min):**
- Current trading infrastructure
- Pain points and frustrations
- Evaluation criteria for new tools
- Timeline and budget

**Pitch (5 min):**
> "Based on what you've shared, here's how our platform might help..."

**Close (3 min):**
> "We have 3 spots left in our pilot cohort starting [date]. The commitment is €500/month for 3 months with weekly feedback. Is that something you'd like to explore?"

---

## Risk Management

### Pilot Risks

| Risk | Mitigation |
|------|------------|
| Low engagement | Weekly check-ins, early intervention |
| Technical issues | Premium support, fast iteration |
| Negative feedback | Frame as learning, not failure |
| Pilot churn | Clear expectations upfront |
| Scope creep | Defined MVP boundary |

### Go/No-Go Criteria

**Continue to paid launch if:**
- 3+ firms complete pilot
- NPS > 30
- 2+ firms express conversion intent
- No critical technical failures

**Extend pilot if:**
- Mixed feedback requiring iteration
- Engagement below target but improving
- Feature gaps identified and addressable

**Pivot if:**
- Zero conversion intent
- Consistent negative feedback
- Fundamental value proposition rejected

---

## Post-Pilot Transition

### For Converting Customers

| Week | Activity |
|------|----------|
| Week 12 | Conversion discussion |
| Week 13 | Contract negotiation |
| Week 14 | Payment setup |
| Week 15 | Transition to production support |

### For Non-Converting Participants

- Thank you for participation
- Access extended 30 days for transition
- Exit feedback incorporated
- Door open for future engagement

---

## Appendix: Templates

### Welcome Email Template

```
Subject: Welcome to the CustodiaCloud Pilot Program

Dear [Name],

Welcome to our pilot program! We're excited to have [Firm Name] as one of our founding customers.

Here's what happens next:
1. Your account is ready: [Login URL]
2. Join our Slack: [Invite Link]
3. Schedule your kickoff call: [Calendly Link]

Your dedicated contact: [Name], [Email], [Phone]

Looking forward to working together.

Best,
[Founder Name]
```

### Weekly Check-in Agenda

```
1. How's the week going? (5 min)
2. Any blockers or issues? (5 min)
3. Feature/feedback discussion (10 min)
4. Priorities for next week (5 min)
5. Any questions for us? (5 min)
```

### Exit Survey Questions

```
1. Overall satisfaction (1-10)
2. Likelihood to recommend (NPS)
3. Most valuable feature
4. Biggest disappointment
5. Fair price point
6. Would you continue as paying customer?
7. What would change your answer?
8. May we use you as a reference?
```

---

## Related Documents

- [BEACHHEAD_MARKET_STRATEGY.md](BEACHHEAD_MARKET_STRATEGY.md) — Beachhead market selection
- [MVP_FOCUS.md](MVP_FOCUS.md) — MVP feature scope definition
- [LEAN_VALIDATION_STRATEGY.md](LEAN_VALIDATION_STRATEGY.md) — Customer validation framework
- [PRODUCT_OVERVIEW.md](PRODUCT_OVERVIEW.md) — One-pager for pitches
- [design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt](design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt) — Master CCEA architecture document

---

*Document Version: 2.0*
*Last Updated: 2025-12-18*
*Owner: Founder / Head of Sales*
*Aligned with: Design Doc CCEA Cloud v1.0*

# CCEA Marketing Guidelines

## Communication Standards for CCEA Platform

**Document Version:** 1.0
**Date:** December 2025
**Classification:** Internal - Marketing & Sales Reference

---

## Purpose

This document establishes **mandatory communication standards** for all CCEA Platform marketing, sales, and public communications. These guidelines ensure:

1. **Regulatory Compliance** - Avoid language that implies investment advice or brokerage services
2. **Accurate Positioning** - Clearly communicate our role as a software provider
3. **Customer Clarity** - Help customers understand our Cloud/Agent architecture
4. **Legal Protection** - Minimize liability exposure through precise language

**Reference:** See [CCEA_OVERVIEW.md](../architecture/CCEA_OVERVIEW.md) for complete architectural documentation.

---

## 1. Core Messaging Framework

### 1.1 What We Are

| Category | Approved Description |
|----------|---------------------|
| **Company Type** | Software provider, technology company |
| **Product Type** | AI-powered quantitative research and simulation platform |
| **Service Model** | SaaS platform with optional customer-deployed execution agent |
| **Industry Position** | Infrastructure provider for quantitative trading |

### 1.2 One-Line Descriptions (Approved)

**For General Audiences:**
> "CCEA Platform provides AI-powered tools for quantitative trading research and strategy development."

**For Technical Audiences:**
> "CCEA Platform offers cloud-based RL training, backtesting, and simulation with customer-controlled local execution."

**For Investors:**
> "CCEA Platform is an open-core AI infrastructure company serving quantitative trading firms and researchers."

**For Regulators:**
> "CCEA Platform is a software provider offering research and simulation tools. Trading execution occurs in customer environments using customer accounts."

---

## 2. Approved Language Matrix

### 2.1 When Describing the Platform

| Context | Approved Language | Prohibited Language |
|---------|-------------------|---------------------|
| **Platform Function** | "AI-powered research tools" | "Trading platform" |
| | "Quantitative simulation environment" | "Execution venue" |
| | "Strategy development software" | "Automated trading service" |
| **Execution Model** | "Customer-controlled execution" | "We trade for you" |
| | "Deploy strategies to your infrastructure" | "Cloud auto-execution" |
| | "Your Agent, your execution" | "We place orders" |
| **Performance** | "Backtested performance" | "Guaranteed returns" |
| | "Historical simulation results" | "Expected profit" |
| | "Research-based insights" | "Investment recommendations" |
| **Risk** | "Configurable risk limits" | "Risk-free" |
| | "Customer-controlled kill switch" | "No losses possible" |
| | "Built-in risk management tools" | "Protected capital" |

### 2.2 When Describing the Architecture

| Component | Approved Description | Avoid |
|-----------|---------------------|-------|
| **Cloud** | "Provides training, simulation, and artifact management" | "Controls trading" |
| **Agent** | "Customer-deployed execution runtime" | "Our trading software" |
| **API Keys** | "Stored locally in your Agent" | "Managed by us" |
| **Orders** | "Created by your Agent from Intents" | "Sent by our Cloud" |
| **Kill Switch** | "Local control in your environment" | "Remote shutdown" |

---

## 3. Prohibited Statements

### 3.1 Absolute Prohibitions (Never Use)

| Prohibited Statement | Why Prohibited | Alternative |
|---------------------|----------------|-------------|
| "We trade for you" | Implies broker/advisor relationship | "Deploy strategies to your infrastructure" |
| "Cloud auto-execution" | Architecturally incorrect | "Customer-controlled local execution" |
| "Guaranteed profit" | Illegal financial claim | "Backtested performance of X%" |
| "No risk" | Misleading | "Risk management tools included" |
| "We manage your portfolio" | Investment advisor language | "Tools for portfolio analysis" |
| "Our algorithm places orders" | Architecturally incorrect | "Your Agent executes your strategies" |
| "Zero losses" | Misleading | "Configurable loss limits" |
| "Free money" | Misleading | Never use any variant |
| "Beat the market guaranteed" | Illegal claim | "Research tools for alpha generation" |
| "Our signals generate profit" | Investment advice | "Backtested signal performance" |

### 3.2 Context-Dependent Restrictions

| Statement | Allowed Context | Prohibited Context |
|-----------|-----------------|-------------------|
| "Alpha generation" | Research/educational content | Sales guarantees |
| "Performance improvement" | With disclaimers, historical data | Without qualifications |
| "Higher returns" | Comparative backtest data | Absolute promises |
| "Beat your benchmark" | Research tools positioning | Performance guarantee |

---

## 4. Required Disclaimers

### 4.1 Mandatory Disclaimers by Content Type

**Marketing Materials (Website, Brochures, Ads):**
```
Disclaimer: CCEA Platform provides software tools for quantitative research
and simulation. We are not a broker, investment advisor, or portfolio manager.
All trading decisions and execution are made by the customer using their own
infrastructure and accounts. Past performance does not guarantee future results.
Trading involves risk of loss.
```

**Performance Claims (Any Backtest Results):**
```
IMPORTANT: These results are from historical backtests and simulations.
Past performance does not guarantee future results. Actual trading involves
significant risks including loss of capital. Results shown do not include
transaction costs, slippage, or market impact that would affect live trading.
```

**Social Media / Short-Form Content:**
```
Not investment advice. Trading involves risk. Backtest results shown.
```

**Email Marketing:**
```
This communication is for informational purposes only and does not constitute
investment advice. CCEA Platform is a software provider, not a financial advisor.
```

### 4.2 Placement Requirements

| Content Type | Disclaimer Location | Font Size |
|--------------|-------------------|-----------|
| **Website Pages** | Footer of every page + above any performance data | Readable (min 10pt) |
| **PDF Documents** | First page and last page | Same as body text |
| **Videos** | On-screen during performance claims + end card | Readable 3+ seconds |
| **Social Media** | In post body or first comment | Platform standard |
| **Email** | Footer of every email | Same as body text |
| **Presentations** | Footer of each slide with performance data | Readable |

---

## 5. Channel-Specific Guidelines

### 5.1 Website Content

**Homepage:**
- Lead with research/simulation capabilities
- Show architecture diagram with Cloud/Agent separation
- Include regulatory positioning statement
- Link to detailed architecture documentation

**Product Pages:**
- Clearly separate Research (Cloud) and Execution (Agent) capabilities
- Emphasize customer control and ownership
- Include technical specifications
- Show compliance badges (if applicable)

**Blog Posts:**
- Educational focus, not promotional
- All performance claims with disclaimers
- Link to methodology documentation
- Avoid "tip" or "signal" language

### 5.2 Social Media

**LinkedIn (Primary B2B Channel):**
- Technical thought leadership
- Industry research and analysis
- Company milestones and hiring
- NO performance claims without full context

**Twitter/X:**
- Product updates and features
- Community engagement
- Technical tips
- Always include disclaimers for any data

**YouTube:**
- Educational content
- Product tutorials
- Webinar recordings
- On-screen disclaimers for all performance visuals

### 5.3 Sales Communications

**Cold Outreach:**
```
Template: "Hi [Name], I'm [Your Name] from CCEA Platform. We provide AI-powered
research and simulation tools for quantitative trading teams. Unlike traditional
platforms, our architecture keeps execution in your environment - you control
your API keys and trading decisions. Would you be open to a brief conversation
about how firms like [Reference] are using our tools?"
```

**Demo Scripts:**
- Emphasize research and simulation capabilities
- Clearly explain Cloud/Agent separation
- Show how customers control execution
- Never promise specific performance outcomes

**Proposals:**
- Include full Terms of Service reference
- Clearly state we are software provider, not advisor
- Specify that execution is customer's responsibility
- Include all relevant disclaimers

---

## 6. Performance Presentation Standards

### 6.1 Backtest Results

**Required Context:**
- Time period clearly stated
- Asset class and symbols specified
- Simulation assumptions documented
- "Backtest" or "Simulated" clearly labeled

**Prohibited:**
- Presenting backtests as live results
- Cherry-picking best periods
- Hiding drawdown data
- Claiming results are replicable

**Example (Compliant):**
```
Backtest Results (BTC-USD, 2022-2024)
Simulation using L2 market data, 10bps slippage model
- Sharpe Ratio: 1.8
- Max Drawdown: -22%
- Annual Return: 45%

DISCLAIMER: These are simulated historical results. Actual trading
performance may differ significantly due to market conditions, execution
quality, and other factors. Past performance is not indicative of future results.
```

### 6.2 Customer Outcomes

**Before Publishing:**
- [ ] Written permission from customer
- [ ] Results verified by customer
- [ ] Appropriate anonymization if requested
- [ ] All required disclaimers included
- [ ] Legal review for material claims

**Format:**
```
"[Customer Type] achieved [metric] improvement in [specific area]
using CCEA Platform's [feature]. Individual results may vary."
```

---

## 7. Regulatory Keyword Avoidance

### 7.1 Investment Advisor Language (Avoid)

| Trigger Phrase | Issue | Alternative |
|----------------|-------|-------------|
| "We recommend" | Investment advice | "Our tools suggest" / "Analysis shows" |
| "You should buy/sell" | Direct advice | "Signal indicates" / "Model output" |
| "Best investment" | Investment advice | "Top-performing backtest" |
| "Portfolio management" | Advisory service | "Portfolio analysis tools" |
| "Financial planning" | Advisory service | "Research and simulation" |

### 7.2 Broker/Dealer Language (Avoid)

| Trigger Phrase | Issue | Alternative |
|----------------|-------|-------------|
| "Execute trades" | Broker activity | "Your Agent executes" |
| "Place orders" | Broker activity | "Orders created locally" |
| "Custody" | Custodial service | Never use |
| "Hold assets" | Custodial service | "Track positions" |
| "Best execution" | Broker duty | "Execution simulation" |

### 7.3 Safe Harbor Language

When discussing performance or capabilities, use:
- "Research indicates..."
- "Historical data shows..."
- "Backtests demonstrate..."
- "Simulations suggest..."
- "Our tools enable..."
- "Customers have reported..."

---

## 8. Visual Guidelines

### 8.1 Architecture Diagrams

**Required Elements:**
- Clear separation between Cloud and Agent
- Label showing execution occurs in Agent (customer environment)
- Note that API keys are stored locally
- Arrow showing Commands (not Orders) from Cloud to Agent

**Prohibited:**
- Diagrams implying Cloud sends trading orders
- Diagrams showing API keys in Cloud
- Any visualization suggesting we execute trades

### 8.2 Screenshot Standards

**Trading Interface Screenshots:**
- Always show simulation/paper trading mode
- Include "SIMULATION" or "BACKTEST" watermark
- Never show live trading interface without clear labeling
- Blur or anonymize any real account data

---

## 9. Compliance Checklist

### 9.1 Before Publishing Any Content

**Mandatory Review:**
- [ ] No prohibited statements (Section 3)
- [ ] Required disclaimers included (Section 4)
- [ ] Performance claims properly contextualized (Section 6)
- [ ] Architecture accurately represented
- [ ] Regulatory keywords avoided (Section 7)
- [ ] Customer data properly anonymized/approved

**Additional Review for:**
- [ ] Press releases: Legal review required
- [ ] Performance claims: CTO sign-off
- [ ] Customer quotes: Customer approval
- [ ] Regulatory statements: Legal review

### 9.2 Annual Training Requirement

All marketing and sales personnel must complete:
- Initial training within 30 days of hire
- Annual refresher training
- Sign acknowledgment of these guidelines

---

## 10. Violation Response

### 10.1 Severity Levels

| Level | Example | Response |
|-------|---------|----------|
| **Minor** | Missing disclaimer on social post | Immediate correction, verbal reminder |
| **Moderate** | Incorrect architecture description | Content removal, written guidance |
| **Severe** | Performance guarantee statement | Immediate removal, legal review, formal warning |
| **Critical** | Regulatory violation | Immediate escalation to Legal/CEO |

### 10.2 Reporting

Report potential violations to:
- Marketing Lead (initial)
- Legal Counsel (if regulatory concern)
- CEO (if critical)

---

## 11. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12 | Marketing + Legal | Initial release |

**Review Cycle:** Quarterly
**Next Review:** Q2 2025
**Owner:** Marketing Lead + Legal Counsel

---

## Quick Reference Card

### Always Say:
- "Research and simulation platform"
- "Customer-controlled execution"
- "Deploy to your own infrastructure"
- "Your Agent, your execution"
- "Backtested/simulated results"
- "Tools for quantitative trading"

### Never Say:
- "We trade for you"
- "Guaranteed profit"
- "Cloud auto-execution"
- "Our algorithm places orders"
- "Risk-free"
- "We manage your portfolio"

### Always Include:
- Disclaimers for all performance data
- Clear labeling of backtest vs. live results
- Architecture explanation for execution model
- Reference to Terms of Service for detailed terms

---

**Classification:** INTERNAL - Marketing & Sales Reference
**Distribution:** All marketing, sales, and customer-facing personnel

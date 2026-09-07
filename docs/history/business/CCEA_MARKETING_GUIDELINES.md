# CustodiaCloud Marketing Guidelines (CCEA-Safe)

## Communication Standards for CustodiaCloud

**Document Version:** 1.0
**Date:** December 2025
**Classification:** Internal - Marketing & Sales Reference

---

## Purpose

This document establishes **mandatory communication standards** for all CustodiaCloud marketing, sales, and public communications. These guidelines ensure:

1. **Regulatory-safe messaging** - Avoid language that implies investment advice, brokerage, or execution-as-a-service
2. **Accurate Positioning** - Clearly communicate our role as a software provider
3. **Customer Clarity** - Help customers understand our Cloud/Agent architecture
4. **Legal Protection** - Minimize liability exposure through precise language

**Canon (single source of truth for wording/positioning):** `docs/DOCUMENTATION_CANON_DESIGN.md`

**Technical reference for the CCEA boundary:** `archive/root_files/Design Doc CCEA Cloud.txt`

---

## 1. Core Messaging Framework

### 1.1 What We Are

| Category | Approved Description |
|----------|---------------------|
| **Company Type** | Software provider, technology company |
| **Product Type** | Risk-first quantitative research and deployment platform (equities-first) |
| **Service Model** | Cloud research/monitoring + customer-controlled execution via Agent |
| **Industry Position** | B2B infrastructure platform for professional systematic trading organizations |

### 1.2 One-Line Descriptions (Approved)

**For General Audiences:**
> "CustodiaCloud provides risk-first tools for quantitative research and deployment for professional teams."

**For Technical Audiences:**
> "CustodiaCloud offers cloud-based research/backtesting/simulation with customer-controlled local execution via an Agent (CCEA)."

**For Investors:**
> "CustodiaCloud is a B2B research & deployment platform with a defensible Cloud/Agent execution boundary (CCEA)."

**For Regulators:**
> "CustodiaCloud is a B2B software provider. Live execution occurs only in customer environments via customer-controlled Agent and customer broker accounts; Cloud does not store credentials and does not send live trading instructions."

---

## 2. Approved Language Matrix

### 2.1 When Describing the Platform

| Context | Approved Language | Prohibited Language |
|---------|-------------------|---------------------|
| **Platform Function** | "AI-powered research tools" | "Trading platform" |
| | "Quantitative simulation environment" | "Execution venue" |
| | "Strategy development software" | "Automated trading service" |
| **Execution Model** | "Customer-controlled execution" | "We execute on your behalf" |
| | "Deploy strategies to your infrastructure" | "Cloud-side execution" |
| | "Your Agent, your execution" | "We place orders" |
| **Performance** | "Backtested performance" | "Guaranteed returns" |
| | "Historical simulation results" | "Expected profit" |
| | "Research-based insights" | "Investment advice" |
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
| "We execute on your behalf" | Implies broker/advisor relationship | "Deploy strategies to your infrastructure" |
| "Cloud-side execution" | Architecturally incorrect | "Customer-controlled local execution" |
| "Guaranteed profit" | Illegal financial claim | "Backtested performance of X%" |
| "No risk" | Misleading | "Risk management tools included" |
| "We manage your portfolio" | Investment advisor language | "Tools for portfolio analysis" |
| "Our algorithm places orders" | Architecturally incorrect | "Your Agent executes your strategies" |
| "Zero losses" | Misleading | "Configurable loss limits" |
| "Free money" | Misleading | Never use any variant |
| "Beat the market guaranteed" | Illegal claim | "Research tools for alpha generation" |
| "Our research outputs generate profit" | Advice/performance promise | "Backtested simulation results (with disclaimers)" |

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
Disclaimer: CustodiaCloud is a B2B software/ICT product for professional trading
organizations. CustodiaCloud does not provide investment advice, portfolio
management, or trade recommendations. Live execution occurs only via the
customer-controlled Agent and the customer's own broker accounts; the Cloud does
not store credentials and does not send live trading instructions (orders/targets/signals).
Past performance does not guarantee future results. Trading involves risk of loss.
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
Not investment advice. Trading involves risk. Historical simulation results shown.
```

**Email Marketing:**

```
This communication is for informational purposes only and does not constitute
investment advice. CustodiaCloud is a software provider, not a financial advisor.
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
- Use evidence exports and privacy-by-design language; avoid “certification/compliance badge” framing unless independently verified and counsel-approved

**Blog Posts:**

- Educational focus, not promotional
- All performance claims with disclaimers
- Link to methodology documentation
- Avoid advice-like language ("tips", "recommendations", "execution-on-behalf framing")

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
Template: "Hi [Name], I'm [Your Name] from CustodiaCloud. We provide AI-powered
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

**Example (Committee-safe):**

```
Backtest Results (SYMBOL, YYYY–YYYY)
Simulation assumptions: data source, slippage/fees/market impact model (illustrative placeholders)
- Sharpe Ratio: [X]
- Max Drawdown: [Y]
- Annualized Return: [Z]

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
using CustodiaCloud's [feature]. Individual results may vary."
```

---

## 7. Regulatory Keyword Avoidance

### 7.1 Investment Advisor Language (Avoid)

| Trigger Phrase | Issue | Alternative |
|----------------|-------|-------------|
| "We recommend" | Investment advice | "Our tools suggest" / "Analysis shows" |
| "You should buy/sell" | Direct advice | "Model output indicates" / "Simulation output" |
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

### 7.4 Compliance & Readiness Language (Avoid Without Independent Validation)

When communicating with committees, regulators, or immigration counsel, avoid absolute claims unless we have independent evidence (e.g., an external audit, a signed legal opinion, or a contractual attestation).

| Trigger Phrase | Issue | Alternative |
|----------------|-------|-------------|
| Any statement asserting full regulatory compliance or external certification/approval | Absolute claim without independent validation | "designed to support alignment" / "alignment & evidence exports" / "mapped to requirements (scope- and deployment-dependent)" |
| "production ready" / "production-grade" | Implies deployment fitness guarantee | "designed for production use" / "ready for production review" / "pilot-ready" |

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

### Always Say

- "Research and simulation platform"
- "Customer-controlled execution"
- "Deploy to your own infrastructure"
- "Your Agent, your execution"
- "Backtested/simulated results"
- "Tools for quantitative trading"

### Never Say

- "We execute on your behalf"
- "Guaranteed profit"
- "Cloud-side execution"
- "Our algorithm places orders"
- "Risk-free"
- "We manage your portfolio"

### Always Include

- Disclaimers for all performance data
- Clear labeling of backtest vs. live results
- Architecture explanation for execution model
- Reference to Terms of Service for detailed terms

---

**Classification:** INTERNAL - Marketing & Sales Reference
**Distribution:** All marketing, sales, and customer-facing personnel

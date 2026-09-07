# CustodiaCloud Documentation Canon (Design Doc)

This document defines the **documentation standard** for CustodiaCloud: naming, legally safe language, positioning, and committee-friendly narratives. It is intended to be the **single source of truth** used to correct and align the rest of the repository documentation (excluding architecture/design docs).

**Audience priority**: European startup visa committees (primary), investors (secondary).

**Non-legal notice**: This document is a writing and product-positioning guide. It is **not** legal, tax, immigration, or investment advice.

---

## 1) Scope and precedence (single source of truth)

Use this document when editing any document that describes the product externally or semi-externally (visa, incubators, partners, pilots, investors, public docs).

**Precedence rules** (when documents disagree):

1. This document for **facts, naming, legally safe wording, banned phrases, and “how to write” rules**.
2. Supporting technical/compliance documents for deeper implementation detail (they must not contradict the guardrails here).

---

## 2) Canonical identity and terminology

Use these terms consistently across the repo:

- **Company / product**: `CustodiaCloud`
- **Architecture**: `CCEA` (Cloud-Controlled Execution Architecture)
- **Components**:
  - `CustodiaCloud Cloud`: research, simulation/backtesting, artifact building, monitoring/telemetry, lifecycle control plane
  - `CustodiaCloud Agent`: runs in the customer environment; holds secrets locally; enforces risk controls; performs any live execution via the customer’s own broker accounts
- **Repository (internal)**: `AI-Powered-Quantitative-Research-Platform` (do not label it as a “trading bot”)

Avoid introducing additional brand/product names unless explicitly required (e.g., a future legal entity name).

---

## 3) Core product truth (what we can safely and concretely claim)

### 3.1 What CustodiaCloud is

CustodiaCloud is a **B2B** risk-first quantitative **research and deployment platform** for professional trading organizations, with an equities-first go-to-market.

### 3.2 What problem we solve (committee-friendly)

Professional systematic teams often spend **months** building research, execution, and risk infrastructure before deploying their first strategy. CustodiaCloud reduces time-to-production by providing:

- repeatable research and simulation workflows
- production deployment tooling
- risk controls and governance evidence exports

We measure success via onboarding and operational KPIs (time-to-first-backtest, time-to-first-live-run, stability, evidence exports), **not** by promising trading performance.

### 3.3 Architecture summary (CCEA)

CCEA is the core posture that makes the product procurement-friendly:

- **Cloud**: research/simulation/monitoring + artifact building/registry + lifecycle control plane (non-orders)
- **Agent**: customer-controlled execution and secrets

**Hard rule** (must appear consistently): Cloud **does not** store customer broker credentials and **does not** generate, transmit, or execute **live trading instructions** (orders/targets/signals). Execution remains under the customer’s control.

**Allowed phrasing for lifecycle control** (to avoid ambiguity):
> “Cloud may send lifecycle commands and signed artifacts to the Agent; Cloud never sends live trading instructions (orders/targets/signals).”

### 3.4 Asset coverage (5 asset types, correctly framed)

We can truthfully state **two layers** of scope:

- **Foundation (multi-asset by design)**: listed **equities**, listed **options**, listed **futures**, **FX**, and **digital assets** (spot/perpetuals) as an optional expansion path (jurisdiction- and customer-dependent).
- **MVP / default commercial scope (equities-first)**: we lead with listed equities for institutional credibility and repeatable onboarding; additional asset classes are enabled based on validated customer pull, support capacity, and deployment risk review.

Approved phrasing (use verbatim if needed):
> “The core engine is multi-asset by design (equities, options, futures, FX, and optional digital assets). Our MVP and beachhead are equities-first; additional asset classes are enabled based on validated customer demand and support capacity.”

---

## 4) Legal- and committee-safe messaging guardrails

These rules prevent red flags in EU startup visa reviews and reduce regulatory risk in narratives.

### 4.1 Always state these boundaries (B2B + no advice + no execution-as-a-service)

Use this as a standard disclaimer block in external-facing docs:

> CustodiaCloud is a B2B software/ICT product for professional trading organizations. CustodiaCloud does not provide investment advice, portfolio management, or trade recommendations. Live execution occurs only via the customer-controlled Agent and the customer’s own broker accounts; the Cloud does not store credentials and does not send live trading instructions (orders/targets/signals).

Add this sentence in committee-facing docs when space allows:
> “Clients remain responsible for their own regulatory obligations and for broker/market-data relationships and licensing.”

### 4.2 Avoid certification / compliance claims

Do not claim:

- “MiFID compliant”, “GDPR certified”, “DORA compliant/certified”, “EU AI Act compliant”, “conformity assessment completed”

Use instead:

- “designed to support”
- “privacy-by-design”
- “evidence exports for client governance”
- “vendor due diligence friendly”
- “clients run their own compliance and legal review”

### 4.3 Avoid performance promises (investor + committee red flag)

Do not claim:

- guaranteed returns, “profitability”, “prevents losses”, “risk-free”, “beats the market”, “will reduce drawdown” as a certainty

Use instead:

- “risk controls”, “tail-risk awareness”, “measurable sim-to-live parity instrumentation”, “designed to reduce operational risk”

### 4.4 Market data licensing and client responsibility

Be explicit:

- Customers remain responsible for market data licensing/terms and broker relationships.
- CustodiaCloud supports bring-your-own data providers and does not resell data unless explicitly covered by a future agreement.

### 4.5 Avoid absolute / unprovable claims

Do not claim:

- “best”, “unique”, “first/only”, “no competitors”, “guaranteed”, “proven to”

Use instead:

- “designed to”, “intended to”, “differentiates by”, “in our experience”, “based on customer feedback”, “evidence available under NDA”

---

## 5) Canonical positioning (what to say, not just what to avoid)

### 5.1 One-liner

> “CustodiaCloud is a risk-first research and deployment platform for professional systematic equities teams, built on CCEA so execution and secrets stay customer-controlled.”

### 5.2 30-second pitch (committee-safe)

> “CustodiaCloud helps professional systematic teams go from strategy idea to production in days rather than months. The Cloud provides research, simulation, and monitoring; the customer-controlled Agent runs in the customer environment and performs any live execution via the customer’s own broker accounts. This keeps credentials and execution control client-side and makes the product procurement-friendly for regulated and compliance-conscious buyers. We start equities-first and expand to other asset classes based on validated demand.”

### 5.3 Differentiators we can defend

Use differentiators as “design facts” rather than hype:

- **CCEA boundary**: Cloud≠execution; secrets stay local; client-controlled approvals for trading-impacting changes.
- **Risk-first workflows**: CVaR-aware constraints and local enforcement (limits, kill switch) as a first-class product workflow.
- **Execution realism**: multi-factor transaction cost / execution modeling; sim-to-live parity instrumentation.
- **Governance and evidence exports**: logs, change control posture, evidence packages designed to support client procurement and operational reviews.

### 5.4 Canonical go-to-market facts (use consistently)

These are the concrete “what we do / for whom / how we validate” facts that should stay consistent across all committee and investor docs.

**Beachhead / ICP (equities-first)**:

- Professional systematic equities teams (prop firms + small funds), typically 5–50 people.
- Geography: EU/UK-focused; cross-border customers are expected even if the company is established in a single host country.
- Budget: ~€2,000–€5,000/month (initial, illustrative), with an enterprise tier for larger firms.
- Deployment: BYO host (VPS/on-prem/VPC) for the Agent; Cloud is research/monitoring + lifecycle control (non-orders).

**MVP scope (equities-first)**:

- Cloud: research IDE/workflows, backtesting & simulation, signed artifact builder/registry, monitoring/telemetry, lifecycle control plane.
- Agent: local vault for secrets, local risk enforcement (limits/kill switch), local approvals for trading-impacting changes, broker connectors under customer control.
- Out of scope (default): retail workflows, “signals product”, copy-trading, HFT promises, broad multi-asset support as a default commitment.

**Pilot program (customer validation)**:

- Format: 3-month structured pilot cohort (3–5 firms).
- Pricing: ~€500/month during pilot (discounted; illustrative), in exchange for weekly feedback and structured onboarding participation.
- Success metrics (examples): <7 days to first live run (via customer Agent), high adoption of risk controls, conversion willingness at target price range.

**Funding ask (investor-facing, but must not contradict committee docs)**:

- Target raise: €500K–€750K (illustrative).
- Use of proceeds (illustrative): 40% GTM, 35% engineering, 15% operations (legal/compliance/vendor due diligence preparedness), 10% reserve.
- Runway target: 18–24 months to reach “Series A readiness” milestones (burn and revenue traction dependent; milestone-based spending).

**12-month milestones (illustrative, not forecasts)**:

- Phase 1 (0–3 months): 3 signed pilot agreements; repeatable onboarding.
- Phase 2 (3–6 months): dashboard MVP; first revenue (illustrative €40K–€50K ARR).
- Phase 3 (6–12 months): product-market fit signals (2+ customers expanding usage); readiness for a larger raise.

**EU establishment (multi-country, Estonia first)**:

- Establish an EU entity via an applicable startup/entrepreneur pathway (jurisdiction-dependent).
- Estonia is the first application/primary path; other EU countries remain viable options depending on program fit, facilitator/incubator availability, and counsel guidance.

---

## 6) Startup visa committee narrative (Estonia-first)

Committees typically look for credible innovation, a realistic plan, and contribution to the local economy. Keep the story concrete and implementation-oriented.

Important: we plan to apply to **multiple European countries**. Estonia is the first application, but this narrative should be reusable: swap the “host country” details (incubator/facilitator, local partners, hiring plan timing) while keeping the core facts identical.

### 6.1 Committee-friendly “read first” bullets (canonical)

- **Innovation**: CCEA architecture (Cloud research/monitoring + customer-controlled Agent execution) + risk-first ML (CVaR constraints) + governance/evidence exports by design.
- **Implementation plan (EU, Estonia-first)**: primary establishment path is an **Estonia OÜ** (first application). We will tailor the same plan for other EU countries if needed. Customer validation (pilot cohort) is EU/UK-wide and can be executed cross-border while operations are established in the host country.
- **Economic contribution**: high-skilled job creation roadmap (engineering, DevOps, product, sales/BD, security/compliance operations) as revenue scales.
- **Partners / ecosystem**: engage the relevant local startup ecosystem and (where required) an approved facilitator/incubator; collaborate with local universities/meetups for talent and knowledge transfer.
- **Realistic budget**: seed funding targets an **18–24 month runway** to cover pilot execution, EU go-to-market, and initial hiring (no reliance on “multiplier” claims).

### 6.2 What committees must *not* hear

Avoid phrasing that triggers red flags:

- “trading bot”, “signals”, “copy trading”, “we trade for clients”, “we manage money”
- “we are regulated / licensed” (unless verified and specific)
- “we are compliant with X law” (unless counsel-reviewed and scoped)
- “we are a high-risk AI system under the EU AI Act” (do not self-classify in docs without legal review)

Crypto/digital-assets guidance for committees:

- If mentioned, frame digital assets only as **optional expansion** after the equities-first validation, and always as jurisdiction- and customer-dependent (with legal review for concrete deployments).

---

## 7) Investor narrative (consistent with committee safety)

Investors want ambition, but visa committees punish over-claims. Use the same facts; change emphasis:

- **GTM wedge**: equities-first in EU/UK, 3–5 firm pilot, measurable conversion and onboarding metrics.
- **Business model**: B2B subscription per firm (tiered), with an enterprise tier.
- **Why now**: increasing demand for governance, controls, and vendor-friendly deployment models in financial services.
- **Moat**: operational + architectural moat (CCEA, evidence exports, deployment tooling), not “secret AI”.
- **Ask**: seed funding for validation, go-to-market, and key hires.

---

## 8) Naming cleanup spec (repo-wide)

This section is the canonical replacement guide when editing older docs.

### 8.1 Replace legacy product names

- Replace `TradingBot2`, `TradingBot`, “trading bot”, “algo trading bot” with:
  - “CustodiaCloud platform”
  - “systematic trading infrastructure”
  - “research and deployment platform”
  - “CCEA Cloud/Agent system”

### 8.2 Avoid product framing that implies regulated activity

Avoid “execution service”, “broker”, “custody”, “adviser”, “portfolio manager”, “recommendations”.

Use “software tools”, “infrastructure”, “client-controlled execution”, “research/simulation/monitoring”.

---

## 9) Red-flag checklist (use before sharing any doc)

Before sending a doc to a committee, incubator, partner, or investor, confirm:

- It states **B2B only** and avoids retail language.
- It explicitly states **no investment advice** and **no execution by Cloud**.
- It avoids **certification/compliance** claims and avoids self-classifying under the EU AI Act.
- It avoids **performance promises** and focuses on measurable operational outcomes.
- It keeps **asset coverage** accurate: 5 asset types in the foundation; equities-first MVP.
- It keeps **Estonia-first** establishment narrative consistent.

---

## 10) Where details live (do not duplicate inconsistently)

When a document needs detail, link to this document instead of re-deriving:

- Positioning, naming, legal-safe language, and committee narrative: `docs/DOCUMENTATION_CANON_DESIGN.md`

For deeper technical/compliance detail (supporting documents; not canonical for messaging):

- Architecture overview: `docs/CCEA_OVERVIEW.md`
- Innovation narrative: `docs/INNOVATION_STATEMENT.md`
- Regulatory posture (non-legal): `docs/REGULATORY_COMPLIANCE_STRATEGY.md`
- Data protection posture: `docs/DATA_PROTECTION_POLICY.md`

*Last Updated: 2025-12-18*

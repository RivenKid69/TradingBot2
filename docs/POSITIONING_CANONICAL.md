# CustodiaCloud Positioning (Canonical)

This document is the **single source of truth** for visa/investor-facing positioning, naming, and market focus.

---

## 1) What we are

**CustodiaCloud** is a risk-first quantitative research and deployment platform for **systematic equities**.

**Scope note**: the core engine is **multi-asset** by design (equities + adjacent listed derivatives + FX; digital assets optional), but our MVP and beachhead go-to-market are deliberately **equities-first** for institutional credibility.

## Asset Coverage (Foundation vs MVP)

**Foundation (multi-asset by design)**: listed **equities**, listed **futures**, listed **options**, **FX**, and **digital assets** (spot/perpetuals) as an optional expansion path.

**MVP / Beachhead (equities-first)**: we lead with listed equities for credibility and repeatable onboarding. Adjacent asset classes are enabled **only** based on validated customer pull and support capacity.

The platform is built on **CCEA (Cloud-Controlled Execution Architecture)**:
- **Cloud**: research, simulation/backtesting, artifact building, monitoring/telemetry
- **Agent**: runs on the customer’s infrastructure; holds secrets locally; enforces risk controls; performs any live execution via the customer’s own broker accounts

**Regulatory posture** (core message): we are a **software / ICT provider**, not a broker, custodian, or investment adviser.

## Regulatory Posture (Design Intent)

This is how CustodiaCloud is designed to support EU-facing deployments. Regulatory classification depends on activities and jurisdiction (not legal advice).

| Framework | What customers need | How CustodiaCloud supports | What we do not do |
|----------|----------------------|----------------------------|-------------------|
| **MiFID II** (and EU algo trading expectations) | Controls + governance + testing evidence | CCEA separation (Cloud≠execution), local approvals for trading-impacting changes, risk controls/kill switch, audit trails & exports | No custody, no client secrets in Cloud, no Cloud live trading instructions, no execution on behalf of clients |
| **GDPR** | Privacy-by-design, minimization, retention, EU residency | Telemetry sensitivity levels, redaction, tenant isolation, retention/DSAR hooks, EU-region defaults | No collection of unnecessary personal data; no secrets in telemetry |
| **DORA** | Vendor risk assessment, operational resilience evidence | Evidence exports, change control posture, incident/runbook documentation, roadmap for enterprise controls | Not claiming “DORA certified”; clients run their vendor due diligence |
| **EU AI Act** | AI governance & transparency posture | Model/version provenance, logging/auditability, human control via local approvals, avoid “personalized recommendations” posture | Not positioning as an AI adviser; no claims about risk classification without legal review |

Naming intent: **CustodiaCloud** refers to custody/control staying with the customer (via the Agent). It does **not** imply that we custody assets, keys, or credentials.

## Messaging Guardrails (Visa/Investor-Facing)

Use these rules to avoid regulatory, legal, and credibility issues in external narratives:

- **B2B only**: we target professional trading organizations (prop firms/funds). We do **not** target retail consumers.
- **No investment advice**: do not describe CustodiaCloud as “giving recommendations”, “managing portfolios”, or “making trades” for customers.
- **No execution service**: Cloud does not send orders (or order payloads). Execution occurs only via the customer-controlled Agent and the customer’s own broker accounts.
- **No certification claims**: avoid “MiFID compliant”, “DORA compliant”, “GDPR certified”, “EU AI Act compliant”. Use “designed to support”, “evidence exports”, “privacy-by-design”, and “vendor due diligence friendly”.
- **No performance promises**: avoid language implying guaranteed returns, risk elimination, or “will prevent losses”.
- **Data licensing**: customers remain responsible for market data licenses/terms; CustodiaCloud is designed to work with bring-your-own data providers.

---

## 2) MVP and Beachhead (equities-first)

**MVP focus**: enable a systematic equities team to go from strategy idea → tested → deployed with strong risk controls **in days, not months**, while preserving client control (CCEA).

**Beachhead customer**: European systematic equities teams (prop firms and small funds) that:
- need institutional-grade risk controls and auditability
- must explain architecture and controls to partners/investors/compliance
- prefer a vendor model where secrets and execution remain under their control

**What we avoid in positioning**: “digital-assets-first” messaging. If digital assets are mentioned at all, they are framed as **future optional expansion**, not part of MVP, not the beachhead.

---

## 3) Naming and branding rules

Use these names consistently in visa/investor documents:

- **Company / Product brand**: CustodiaCloud
- **Architecture / core technical concept**: CCEA (Cloud-Controlled Execution Architecture)
- **Public OSS component**: CCEA SDK (protocol + schemas + verification/guardrails)
- **Customer runtime**: CustodiaCloud Agent (client-controlled execution + secrets)
- **Managed services** (if referenced): CustodiaCloud Cloud (research + monitoring only; no secrets; no orders)

Avoid mixing in legacy/internal names in external narratives:
- **TradingBot2** is an internal repository codename and may still appear in technical notes; it is **not** a customer-facing product name.

---

## 4) Timeline language (avoid conflicting “2024/2025/2026” stories)

In narrative docs, prefer **phases** over hard calendar years:
- **Phase 1 (0–3 months)**: pilot onboarding + validation
- **Phase 2 (3–9 months)**: early adopters + repeatable sales
- **Phase 3 (9–18 months)**: scale + enterprise hardening

If dates are required, use a single “**Last updated**” field per document and keep it consistent.

---

*Last Updated: 2025-12-18*

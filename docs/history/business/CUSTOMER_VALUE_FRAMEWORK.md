# Customer Value Framework

## CustodiaCloud — Operational ROI & Value Quantification (B2B)

**Canon (positioning / legal-safe wording / committee narrative):** `docs/DOCUMENTATION_CANON_DESIGN.md`

This document is an **internal** framework to quantify customer value in a way that is consistent with CustodiaCloud’s posture as a **software/ICT provider**. It intentionally focuses on **operational** outcomes (time-to-production, engineering cost, governance/evidence exports, risk controls) and avoids trading performance promises.

---

## Executive Summary

CustodiaCloud creates customer value primarily by reducing the time and cost required to:

- build repeatable research/simulation workflows,
- package and deploy strategies safely via customer-controlled environments (CCEA Agent),
- operate with risk-first guardrails and governance evidence exports that support procurement and operational reviews.

Success is measured via onboarding and operational KPIs (time-to-first-backtest, time-to-first-live-run, stability, evidence exports), not trading performance metrics.

---

## 1. Value pillars (canonical framing)

1. **Time-to-production compression**: faster path from research to controlled deployment.
2. **Engineering cost reduction**: fewer months of bespoke infrastructure work and maintenance.
3. **Risk-first operations**: controls, approvals, and kill-switch patterns reduce operational incidents.
4. **Governance & evidence exports**: documentation + exportable artifacts support procurement and audits.
5. **CCEA boundary (procurement-friendly)**: Cloud lifecycle control and monitoring without secrets or live trading instructions.

---

## 2. Time value quantification (illustrative)

### 2.1 Baseline vs with CustodiaCloud

Use the customer’s own baseline numbers (interviews) and compute deltas:

- Baseline time-to-first-backtest: `T_backtest_baseline`
- Baseline time-to-first-live-run: `T_live_baseline`
- With CustodiaCloud: `T_backtest_custodia`, `T_live_custodia`

**Time saved (weeks)**:

- `ΔT_backtest = T_backtest_baseline - T_backtest_custodia`
- `ΔT_live = T_live_baseline - T_live_custodia`

### 2.2 Convert time saved to cost saved

Use a loaded engineering cost (customer-provided):

- `cost_per_engineer_week` (EUR/week)
- `team_size` (engineers involved)

**Engineering cost saved**:

- `Savings_time = (ΔT_backtest + ΔT_live) * team_size * cost_per_engineer_week`

---

## 3. Cost value quantification (illustrative)

### 3.1 Build vs buy categories

Common cost buckets:

- research/simulation tooling (build + maintain)
- deployment tooling (packaging, rollbacks, configuration governance)
- operational tooling (monitoring, incident runbooks, change control)
- procurement/governance effort (policies, evidence exports, documentation upkeep)

### 3.2 Subscription framing (canon-aligned)

Use the canonical ranges as **illustrative** starting points (actual pricing is program- and scope-dependent):

- Pilot pricing: ~€500/month during pilot (discounted; illustrative)
- Initial target pricing: ~€2,000–€5,000/month (illustrative), enterprise tier by scope

---

## 4. Governance & evidence exports value (non-performance)

Quantify in terms of reduced internal effort and faster approvals:

- fewer staff-hours to answer vendor questionnaires and security reviews
- repeatable evidence exports for change windows, incidents, and controls
- clearer separation of responsibilities via CCEA (Cloud/Agent boundary)

**Illustrative metric** (customer-provided):

- `hours_saved_per_review * reviews_per_year * hourly_cost`

---

## 5. ROI template (operational)

### 5.1 Simple ROI formula

`ROI = (Savings_time + Savings_governance + Savings_incident_avoidance - Subscription_cost) / Subscription_cost`

Where each component is customer-calibrated and treated as **illustrative** (no guarantees).

### 5.2 What to measure during a pilot (3 months)

For a 3–5 firm pilot cohort, track:

- onboarding timeline milestones (first backtest, first controlled run)
- incident rate / stability indicators (crashes, retries, kill-switch usage)
- evidence export completeness and customer procurement feedback
- deployment friction (time to update artifacts/config under policy gates)

---

## 6. Sales enablement guardrails (do not violate canon)

- Do **not** promise trading performance or “performance uplift”.
- Do **not** claim regulatory compliance/certification (MiFID II / DORA / EU AI Act); use “designed to support” and evidence exports instead.
- Always repeat the CCEA boundary: Cloud has no secrets and no live trading instructions (orders/targets/signals); execution is customer-controlled via the Agent.

---

## Related documents

- `docs/DOCUMENTATION_CANON_DESIGN.md`
- `docs/INVESTOR_BRIEF.md`
- `docs/INNOVATION_STATEMENT.md`
- `docs/REGULATORY_COMPLIANCE_STRATEGY.md`

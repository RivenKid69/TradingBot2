# CustodiaCloud — Product Overview

**Canon (positioning, naming, legal-safe wording, committee narrative):** `docs/DOCUMENTATION_CANON_DESIGN.md`

CustodiaCloud is a **B2B**, risk-first quantitative **research and deployment platform** for professional systematic trading organizations. CustodiaCloud is built around the `CCEA` architecture (Cloud/Agent separation) to support enterprise procurement, operational governance, and customer-controlled execution.

---

## CCEA in one paragraph

- **CustodiaCloud Cloud**: research, simulation/backtesting, artifact building/registry, monitoring/telemetry, and lifecycle control plane.
- **CustodiaCloud Agent**: runs in the customer environment; holds secrets locally; enforces risk controls; performs any live execution via the customer’s own broker accounts.

**Hard rule (CCEA boundary):** Cloud does **not** store customer broker credentials and does **not** generate, transmit, or execute **live trading instructions** (orders/targets/signals). Execution remains under the customer’s control via the Agent.

---

## Asset scope (correctly framed)

Approved phrasing (canonical):
> “The core engine is multi-asset by design (equities, options, futures, FX, and optional digital assets). Our MVP and beachhead are equities-first; additional asset classes are enabled based on validated customer demand and support capacity.”

---

## What CustodiaCloud provides (non-performance)

- Repeatable research and simulation workflows (backtesting and validation)
- Signed artifacts, versioning, and deployment packaging for customer-controlled environments
- Risk-first controls (policy gates, approvals for trading-impacting changes, kill-switch patterns) and evidence exports
- Monitoring and telemetry with privacy-by-design (aggregation/redaction; enterprise can choose local-only telemetry modes)
- A procurement-friendly operating model: clear Cloud/Agent boundary and auditable lifecycle controls

---

## Go-to-market (committee-friendly)

- **MVP/beachhead**: equities-first deployment workflows for professional systematic teams
- **Pilot**: 3–5 firms / ~3 months, measured via onboarding and operational KPIs (time-to-first-backtest, time-to-first-live-run, stability, evidence exports), not trading performance promises
- **EU plan**: Estonia-first establishment, with the same plan adaptable for other EU countries as required by program fit

---

## Pricing, funding, runway (illustrative)

- Pilot pricing: ~€500/month during pilot (discounted; illustrative)
- Initial target pricing: ~€2,000–€5,000/month (illustrative), enterprise tier by scope
- Seed ask: €500K–€750K (illustrative)
- Runway target: 18–24 months

---

## Legal-safe notes (non-legal)

CustodiaCloud is a B2B software/ICT product for professional trading organizations. CustodiaCloud does not provide investment advice, portfolio management, or trade recommendations. Live execution occurs only via the customer-controlled Agent and the customer's own broker accounts; the Cloud does not store credentials and does not send live trading instructions (orders/targets/signals).

Clients remain responsible for their own regulatory obligations and for broker/market-data relationships and licensing.

This document does not assert regulatory compliance or certification; CustodiaCloud is designed to support customer assessments via controls and evidence exports (not audited or certified).

---

## Licensing

This repository is proprietary (see `LICENSE`). Any future open-core/repo split (e.g., Agent/SDK) would be published as separate repositories with explicit licensing and trademark terms.

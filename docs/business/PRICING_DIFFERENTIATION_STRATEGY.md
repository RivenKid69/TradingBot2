# Pricing Differentiation Strategy — CustodiaCloud (CCEA)

**Document Version:** 2.1  
**Date:** 2025-12-18  
**Classification:** Internal Strategy Document

**Canon (single source of truth for pricing ranges and committee-safe wording):** `docs/DOCUMENTATION_CANON_DESIGN.md`.

**CCEA boundary reminder:** Cloud may send lifecycle commands and signed artifacts to the Agent; Cloud never sends live trading instructions (orders/targets/signals). Execution remains customer-controlled via the Agent.

---

## 1) Pricing principles (B2B, procurement-friendly)

- **B2B only**: professional systematic trading organizations (equities-first beachhead).
- **Value is operational**: reduced time-to-production, safer deployments, governance/evidence exports, and risk-first controls.
- **Pricing is scenario-based**: workload/support/deployment mode varies across customers; ranges are illustrative.

---

## 2) Canonical ranges (illustrative)

These ranges must remain consistent with `docs/DOCUMENTATION_CANON_DESIGN.md`:

- **Pilot cohort (3 months; 3–5 firms)**: ~**€500/month** (discounted; illustrative)
- **Beachhead subscription**: ~**€2,000–€5,000/month** (initial, illustrative)
- **Enterprise**: custom pricing (deployment mode, support, procurement/security requirements)

---

## 3) Packaging (recommended)

### 3.1 Pilot (validation)
- Structured onboarding + weekly feedback
- Target outcome: repeatable onboarding + willingness to pay at target range (not performance promises)

### 3.2 Core (equities-first)
- Research/backtesting/simulation workflows (Cloud)
- Monitoring/telemetry (redacted by design) + governance/evidence exports
- Deploy-to-Agent lifecycle tooling (non-orders)

### 3.3 Enterprise (on‑prem/VPC options)
- Customer-hosted deployments where required
- Extended retention/controls and enterprise operations support (customer- and deployment-dependent)

---

## 4) Pricing drivers (what increases/decreases price)

- **Workspaces / environments** (dev/stage/prod separation)
- **Number of Agents** and deployment topology (VPS/on‑prem/VPC)
- **Telemetry retention and sensitivity level** (redacted by default; higher sensitivity requires explicit opt-in)
- **Support tier** (response windows, onboarding intensity)
- **Procurement/security requirements** (evidence pack scope, review cycles, contractual addenda)

---

## 5) What we must avoid in pricing narratives

- Consumer pricing, “signals product”, copy-trading, portfolio management framing.
- Performance guarantees or outcome promises.
- Claims of being “compliant/certified” with MiFID II, DORA, EU AI Act.

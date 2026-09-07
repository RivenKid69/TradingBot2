# CustodiaCloud — Repository Overview (Internal)

This repository (`AI-Powered-Quantitative-Research-Platform`) contains the codebase and documentation for **CustodiaCloud**.

**External positioning / legally safe wording / committee narrative (canon):** `docs/DOCUMENTATION_CANON_DESIGN.md`

---

## Product posture (CCEA)

CustodiaCloud uses the `CCEA` architecture (Cloud/Agent separation):

- **Cloud**: research/simulation/monitoring + artifact building/registry + lifecycle control plane (non-orders)
- **Agent**: runs in the customer environment; holds secrets locally; enforces risk controls; performs any live execution via the customer’s own broker accounts

**Hard rule (CCEA boundary):** Cloud is designed so it does **not** store customer broker credentials and does **not** generate, transmit, or execute **live trading instructions** (orders/targets/signals). This is enforced via schema validation, CI guardrails, and protocol allowlists (verify via `tests/ccea/` and CI pipeline).

---

## Asset scope (correctly framed)

Use the canonical phrasing:
> “The core engine is multi-asset by design (equities, options, futures, FX, and optional digital assets). Our MVP and beachhead are equities-first; additional asset classes are enabled based on validated customer demand and support capacity.”

---

## What’s in the repo (high level)

- `docs/`: documentation hub (canon, enterprise docs, compliance/alignment tooling docs, runbooks)
- `ccea/` and `packages/agent/`: CCEA protocol/guardrails and Agent runtime components
- `services/`, `core_*`, `impl_*`, `execution_*`: research/simulation/execution-modeling components used for backtesting, validation, and deployment artifacts
- `tests/`: extensive automated test suite (run `pytest` to verify current coverage/status)

---

## Licensing

This repository is proprietary (see `LICENSE`). Any future open-core or repo-split strategy (Agent/SDK) must be published as separate repositories with explicit licensing/trademark terms.

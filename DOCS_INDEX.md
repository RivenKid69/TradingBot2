# Documentation index

Everything under `docs/` plus the files kept in the repository root. Start with the
README; this page is for finding a specific document.

## Start here

| Document | What it covers |
|---|---|
| [README.md](README.md) | What the project is, the CCEA boundary, quick start |
| [QUICK_START.md](QUICK_START.md) | Command cheat sheet per asset class |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Zero to a first backtest, step by step |
| [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) | Native C++/Cython build, per platform |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Branches, tests, formatting, guardrails |
| [docs/AUDIT_2026-09.md](docs/AUDIT_2026-09.md) | Current state of the repository and its known open issues |

## Architecture and reference

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module map and system design |
| [docs/CCEA_OVERVIEW.md](docs/CCEA_OVERVIEW.md) | Cloud/Agent boundary, threat model |
| [docs/architecture/](docs/architecture/) | CCEA privacy, telemetry, deployment topologies |
| [docs/history/design-doc-ccea-cloud.txt](docs/history/design-doc-ccea-cloud.txt) | The original CCEA design document, hash-verified in CI |
| [docs/PLATFORM_REFERENCE.md](docs/PLATFORM_REFERENCE.md) | Full platform reference (Russian) |
| [docs/NOT_BUGS_AND_FAQ.md](docs/NOT_BUGS_AND_FAQ.md) | Patterns that look like bugs and are not, with the reasoning |
| [docs/reference/](docs/reference/) | Pro mode design, cross-sectional platform, system requirements |
| [docs/schemas/](docs/schemas/) | JSON schemas and their versioning rules |

## Running it

| Document | What it covers |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | Train → backtest → paper → live |
| [docs/agent/](docs/agent/) | Agent installation, vault, approvals, risk controls |
| [docs/cloud/](docs/cloud/) | Control plane API, builder, governance |
| [docs/runbooks/](docs/runbooks/) | Incident response, kill switch, recovery |
| [docs/OPERATIONS_RUNBOOK.md](docs/OPERATIONS_RUNBOOK.md) | Day-to-day operations |
| [docs/CCEA_DESKTOP.md](docs/CCEA_DESKTOP.md) | Desktop shell (Tauri) |

## Asset classes

| Document | What it covers |
|---|---|
| [docs/CRYPTO_SPOT_EXECUTION.md](docs/CRYPTO_SPOT_EXECUTION.md) | Crypto spot execution |
| [docs/FOREX_INTEGRATION_PLAN.md](docs/FOREX_INTEGRATION_PLAN.md) | FX: sessions, swaps, dealer simulation |
| [docs/futures/](docs/futures/) | Futures: continuous contracts, margin, settlement |
| [docs/options/](docs/options/) | Options: Greeks, IV, exercise |
| [docs/l3_simulator/](docs/l3_simulator/) | L3 order book simulation |
| [docs/DUKASCOPY_ADAPTER.md](docs/DUKASCOPY_ADAPTER.md) | Public FX tick feed |

## Research and modelling

| Document | What it covers |
|---|---|
| [docs/twin_critics.md](docs/twin_critics.md) | Twin critics in the distributional PPO |
| [docs/seasonality.md](docs/seasonality.md) | Liquidity seasonality pipeline (see the other `seasonality_*` pages) |
| [docs/universe.md](docs/universe.md) | Universe construction |
| [docs/MODEL_SIGNATURE_AND_REBALANCE.md](docs/MODEL_SIGNATURE_AND_REBALANCE.md) | Model signing and rebalancing |
| [docs/RISK_LIMIT_ENFORCEMENT.md](docs/RISK_LIMIT_ENFORCEMENT.md) | Where each risk limit is enforced |

## Security and compliance

| Document | What it covers |
|---|---|
| [SECURITY.md](SECURITY.md) | Reporting a vulnerability |
| [docs/security/](docs/security/) | Threat model, hardening, key management |
| [docs/compliance/](docs/compliance/) | GDPR, DORA, MiFID II and EU AI Act integration plans |
| [docs/legal/](docs/legal/) | Terms of service, data processing |
| [docs/CYBERSECURITY_FRAMEWORK.md](docs/CYBERSECURITY_FRAMEWORK.md) | Controls mapping |

These are engineering plans and internal mappings, not certifications or legal advice.

## History

| Path | What is there |
|---|---|
| [docs/history/](docs/history/) | Closed audits, gap analyses and blocker reports, kept for context |
| [docs/history/business/](docs/history/business/) | Go-to-market, pricing and investor material from when the project was pursued commercially. Superseded by the Apache-2.0 release; kept as background. |
| [docs/reports/](docs/reports/) | Technical-debt register and generated reports |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

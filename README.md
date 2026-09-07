# CustodiaCloud — Quantitative Research & Deployment Platform

CustodiaCloud is a **B2B** risk-first quantitative **research and deployment platform** with an **equities-first** go-to-market. This repository contains its research/simulation stack and the Cloud/Agent boundary (CCEA) used for customer-controlled execution.

**Canonical positioning / naming / legally safe wording**: `docs/DOCUMENTATION_CANON_DESIGN.md`.

## Licensing

This monorepo is **proprietary**. For the open-core split plan (public `ccea-sdk` + `ccea-agent`, private `ccea-cloud`), see `LICENSING.md`.

## Architecture: Cloud-Controlled Execution Architecture (CCEA)

> **Technical reference**: `archive/root_files/Design Doc CCEA Cloud.txt` | Additional overview: `docs/CCEA_OVERVIEW.md`

CustodiaCloud implements **CCEA** — a strict separation between Cloud (research/monitoring/lifecycle) and Agent (execution/secrets/risk):

| Component | Responsibility | Secrets Access | Order Execution |
|-----------|---------------|----------------|-----------------|
| **Cloud** | Research, backtesting, monitoring, lifecycle management | **None** (by design) | **None** (by design) |
| **Agent** | Live execution, risk enforcement, local vault, order creation | **Local only** | **Yes** (customer-controlled) |

**Key Security Design Commitments:**

- Cloud is designed not to store customer broker API keys or credentials (secrets are intended to stay in the customer-controlled Agent)
- Cloud is designed not to generate, transmit, or execute live trading instructions (orders/targets/signals)
- Cloud may send lifecycle commands and signed artifacts to the Agent; the Agent performs any live execution via customer accounts
- Telemetry redaction is designed to be mandatory (redaction middleware before transmission; verify via CI tests and deployment audits); default telemetry level is **AGGREGATED**; RAW order events are enterprise-only with explicit opt-in (deployment- and customer-dependent; see Design Doc section 13.2)

**Deployment Modes (B2B):**

1. **Cloud + BYO Agent**: Cloud research/simulation/monitoring + customer-controlled Agent execution
2. **Enterprise on‑prem/VPC**: customer-hosted deployments (where required by procurement/security)

**CCEA Terminology:**

- **Intent**: High-level trading intention (target exposure), produced by Strategy
- **Order**: Concrete broker instruction (designed to be created only in Agent, from Intent)
- **Command**: Lifecycle request (REQUEST_START, REQUEST_STOP, etc.) - NOT an order
- **TRADING_IMPACTING**: Changes requiring local approval (new version, risk limits)
- **NON_IMPACTING**: Changes that auto-apply (log level, telemetry verbosity)

## Overview

- Distributional PPO with twin critics, adaptive UPGD optimizer, and population-based tuning for robust policies.
- Market-structure-aware execution: limit/market routing, TWAP/POV, slippage and fee modeling, and risk guards.
- Multi-asset adapters (equities/options/futures/FX; optional digital assets) behind a unified YAML configuration and dependency injection registry.
- Shared pipeline for training, backtesting, paper trading, and live execution with reproducible artifacts.
- Observability and safety: structured logs, KPI benchmarks, sanity checks, and doctor tooling.
- **Strict zone separation**: Cloud/Agent/Shared packages with CI-enforced import boundaries.

## Installation

1. Prerequisites: Python 3.12+, git, compiler toolchain for C++/Cython extensions (see `SYSTEM_REQUIREMENTS.md`).
2. Create a virtual environment and install dependencies (PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel
pip install -r requirements-dev.txt
```

- GPU build: `pip install -r requirements-gpu.lock.txt`
- CPU-only runtime: `pip install -r requirements-cpu.lock.txt`

3. Optional: build native extensions for maximum performance:

```powershell
python setup.py build_ext --inplace
```

## Quick Start

Configuration examples live in `configs/examples/README.md`.

- Train (example config; equities-first beachhead):

```bash
cp configs/examples/example_train_stocks.yaml configs/my_train.yaml
python train_model_multi_patch.py --config configs/my_train.yaml
```

- Backtest / simulation:

```bash
cp configs/examples/example_backtest_crypto.yaml configs/my_backtest.yaml
python script_backtest.py --config configs/my_backtest.yaml --offline-config configs/offline.yaml --dataset-split val
```

- Live execution via local Agent (CCEA architecture):

```bash
# 1. Deploy Agent locally (credentials stay on YOUR machine; architecture designed so they are not sent to cloud)
#    See docs/agent/INSTALLATION.md for full setup
python -m packages.agent.daemon.agentd --config configs/agent.yaml

# 2. (Optional) Use Cloud control plane to manage runs
#    Cloud sends lifecycle commands only - designed not to trade or store keys
```

**Important**: Live execution runs only in your local Agent. Cloud manages lifecycle (start/stop/deploy) and is designed so it does not execute orders or store your credentials. See [CCEA Overview](docs/CCEA_OVERVIEW.md).

For legacy/development dry-run testing:

```bash
# Development/testing only (not production CCEA architecture)
python script_live.py --config configs/my_live.yaml --dry-run
```

Run `python scripts/doctor.py --verbose` before the first training or trading run.

## Status

**Automated test suite** (verify via `pytest`) | **CCEA implemented** | **Evidence exports & alignment tooling** (designed to support customer procurement/operational reviews; not audited or certified)

## CI Status

[![Docs quality](https://github.com/RivenKid69/AI-Powered-Quantitative-Research-Platform/actions/workflows/docs-quality.yml/badge.svg)](https://github.com/RivenKid69/AI-Powered-Quantitative-Research-Platform/actions/workflows/docs-quality.yml)
[![Security SAST](https://github.com/RivenKid69/AI-Powered-Quantitative-Research-Platform/actions/workflows/security-sast.yml/badge.svg)](https://github.com/RivenKid69/AI-Powered-Quantitative-Research-Platform/actions/workflows/security-sast.yml)

- Docs quality: markdown lint/render checks for user-facing docs.
- Security SAST: static analysis of adapters/core for regressions.

## Module Architecture

### ICT Provider Positioning (CCEA Architecture)

CustodiaCloud is designed to support a **software / ICT provider** posture (classification depends on activities and jurisdiction):

- We provide quantitative research, simulation, deployment, and governance tooling
- Customers execute via **their own broker accounts** through the customer-controlled **Agent**
- Cloud is designed not to hold customer broker credentials and does not execute orders (verify via architecture review)

**Legal Position:**

- **NOT** investment advice / portfolio management / trade recommendations
- **NOT** an execution service: Cloud is designed not to execute orders or hold credentials/assets
- B2B software platform for professional trading organizations

See: `docs/DOCUMENTATION_CANON_DESIGN.md` for canonical wording.

### Module Structure (Post-Migration v2.0)

| Package | Purpose | Load |
|---------|---------|------|
| `services.core.risk_controls` | Universal risk controls (kill switch, pre-trade, audit) | Always |
| `services.algo_integration` | Alignment/evidence tooling for regulated clients | Enterprise tier (target segment) |
| `services.archive.mifid_financial_entity` | Investment Firm modules | Archived (not loaded) |

#### For ICT Providers (Default)

```python
from services.core.risk_controls import (
    EnhancedKillSwitch, PreTradeControls, AuditTrailWriter,
    RealTimeMonitor, BusinessContinuityPlan
)
```

#### Migration Note

The legacy `services.compliance` module has been removed. Use the new modular structure:

```python
# Current (canonical import path)
from services.core.risk_controls import EnhancedKillSwitch

# The old services.compliance path is no longer supported
```

## Regulatory Compliance

CustodiaCloud includes documentation, controls, and evidence export patterns intended to **support** customer procurement and operational reviews (jurisdiction- and customer-dependent; not a certification claim).

**CCEA Privacy Design Goals:**

- Cloud **designed not to** store or receive broker credentials or API keys (secrets designed to stay in customer-controlled Agent)
- Cloud **designed not to** receive order-like payloads in commands (protocol-level design prohibition)
- Telemetry redaction is mandatory by design; default telemetry is **AGGREGATED** and RAW order events are enterprise-only with explicit opt-in
- EU data residency **by design** for EU customers (design target; deployment- and contract-specific; enterprise: on-prem/customer-managed options available)
- DSAR scope is Cloud-only (by design); Agent data is customer-controlled

Details: `docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md`

## Supported Exchanges

| Asset class | Vendor(s) | Path | Modes | Status |
| --- | --- | --- | --- | --- |
| Equities execution (MVP/beachhead) | Alpaca | adapters/alpaca/ | sim, paper, live | Implemented |
| Equities data | Polygon, Yahoo | adapters/polygon/, adapters/yahoo/ | data, sim | Implemented |
| FX | OANDA | adapters/oanda/ | sim, live | Implemented (beta) |
| FX (historical) | Dukascopy | adapters/dukascopy/ | historical data only | Stub (Phase 0) |
| Listed options / futures (optional) | Interactive Brokers, ThetaData | adapters/ib/, adapters/theta_data/ | paper/sim, live | Experimental |
| Digital assets (optional) | Binance, Deribit | adapters/binance/, adapters/deribit/ | sim, live | Implemented (beta) |

> **Note**: Status reflects current implementation state. Stubs indicate interface definition without full integration. See individual adapter READMEs for details.

## Guides

### CCEA Architecture Documentation

- `docs/CCEA_OVERVIEW.md` — Cloud/Agent boundary, threat model, legal posture
- `docs/cloud/README.md` — Control plane API, builder, governance
- `docs/agent/README.md` — Installation, vault, approvals, risk controls
- `docs/schemas/README.md` — JSON schemas with versioning guide
- `docs/runbooks/` — Incident response, kill-switch, recovery procedures

### Business & Legal Documentation

- `docs/business/CCEA_MARKETING_GUIDELINES.md` — Approved language and disclaimers (CustodiaCloud/CCEA-safe)
- `docs/business/CCEA_TERMS_OF_SERVICE_GUIDELINES.md` — ToS requirements, liability
- `docs/business/PRICING_DIFFERENTIATION_STRATEGY.md` — Product modes, pricing tiers
- `docs/business/OPEN_CORE_BUSINESS_MODEL.md` — Open-source strategy, licensing
- `docs/business/COMPETITIVE_MOAT.md` — Competitive advantage analysis

### General Documentation

- `docs/PLATFORM_REFERENCE.md` — complete project guide (RU).
- `ARCHITECTURE.md` — system architecture and module map.
- `DOCS_INDEX.md` — documentation hub.
- `QUICK_START.md` — command cheat sheet.
- `configs/examples/README.md` — ready-to-copy configs for train/backtest/live.
- `BUILD_INSTRUCTIONS.md` — native build notes.

## Runbooks

### Simulation / backtest

1. Copy an example config and set dataset paths (`offline-config`, `dataset-split`).
2. Run doctor without network if needed: `python scripts/doctor.py --skip-network`.
3. Execute `python script_backtest.py --config <cfg> --offline-config configs/offline.yaml --dataset-split val`.
4. Validate outputs: compare KPI to `benchmarks/sim_kpi_thresholds.json`, review reports in `artifacts/` and logs in `logs/`.

### Live execution (CCEA Architecture)

**Production: Via Local Agent**

1. **Agent Setup**: Install and configure Agent locally (`docs/agent/INSTALLATION.md`).
2. **Credentials**: Store broker API keys in Agent's local vault (designed not to be uploaded to Cloud).
3. **Deploy Strategy**: Use Cloud control plane to deploy strategy artifact to Agent.
4. **Start Run**: Cloud sends `REQUEST_START_RUN`; Agent executes locally with your credentials.
5. **Monitor**: Cloud receives redacted telemetry; full position data stays in Agent.
6. **Safety**: Agent enforces local hard caps; kill switch available in `docs/runbooks/KILL_SWITCH.md`.

**Development/Testing Only**:

```bash
# For local testing without full Agent setup
python script_live.py --config <cfg> --dry-run
```

See `docs/runbooks/` for full operational procedures.

### Adapter debugging

1. Validate configuration: `python -m pytest tests/test_adapters_config_validation.py -k <vendor>` and align YAML with `configs/examples`.
2. Run vendor smoke tests: e.g., `python -m pytest tests/test_alpaca_adapters.py` or `python -m pytest tests/test_deribit_options.py`.
3. Reproduce with the live runner in dry-run mode: `python script_live.py --config <cfg> --dry-run --asset-class <equity|forex|crypto>` and watch `logs/` for adapter traces.
4. Refresh exchange metadata when applicable (for Binance: `python scripts/fetch_binance_filters.py`) and re-run doctor to confirm connectivity.

### Pre-release checklist

- [ ] `python scripts/doctor.py --verbose` (environment, credentials, clocks).
- [ ] `python -m pytest tests/test_service_mode_smoke.py tests/test_dry_run_executor.py` (or targeted smoke tests relevant to the change).
- [ ] `bash tools/test_markdown_render.sh` (pandoc render check).
- [ ] Verify configs referenced in README/Quick Start exist and secrets are environment-backed.
- [ ] Confirm supported exchange table and badges reflect current coverage.

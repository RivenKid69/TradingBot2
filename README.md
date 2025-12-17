# AI-Powered Quantitative Research Platform

RL-first research and trading stack for crypto, equities, FX, and derivatives designed to minimize simulator-to-live deviation.

## Licensing

This monorepo is **proprietary**. For the open-core split plan (public `ccea-sdk` + `ccea-agent`, private `ccea-cloud`), see `LICENSING.md`.

## Architecture: Cloud-Controlled Execution Architecture (CCEA)

> **Reference**: `Design Doc CCEA Cloud.txt` (canonical source) | [CCEA Overview](docs/architecture/CCEA_OVERVIEW.md)

This platform implements **CCEA** - a strict separation between Cloud (research/monitoring/lifecycle) and Agent (execution/secrets/risk):

| Component | Responsibility | Secrets Access | Order Execution |
|-----------|---------------|----------------|-----------------|
| **Cloud** | Research, backtesting, monitoring, lifecycle management | **NEVER** | **NEVER** |
| **Agent** | Live execution, risk enforcement, local vault, order creation | **LOCAL ONLY** | **YES** |

**Key Security Design Commitments:**
- Cloud **NEVER** stores broker API keys or credentials
- Cloud **NEVER** generates, transmits, or executes trading orders
- Cloud **NEVER** has access to exchange trading endpoints
- All trading operations occur **ONLY** in the Agent running locally or in user's VPC
- Telemetry is **ALWAYS** redacted before transmission to Cloud

**Product Modes:**
1. **Retail Research SaaS (EU-friendly)**: Cloud research/simulation + optional BYO Agent
2. **Retail Live via Local Agent**: Local auto-execution, cloud observability
3. **Enterprise Engine (on-prem/VPC)**: Full stack in customer infrastructure

**CCEA Terminology:**
- **Intent**: High-level trading intention (target exposure), produced by Strategy
- **Order**: Concrete broker instruction, created ONLY in Agent from Intent
- **Command**: Lifecycle request (REQUEST_START, REQUEST_STOP, etc.) - NOT an order
- **TRADING_IMPACTING**: Changes requiring local approval (new version, risk limits)
- **NON_IMPACTING**: Changes that auto-apply (log level, telemetry verbosity)

## Overview
- Distributional PPO with twin critics, adaptive UPGD optimizer, and population-based tuning for robust policies.
- Market-structure-aware execution: limit/market routing, TWAP/POV, slippage and fee modeling, and risk guards.
- Multi-asset adapters (crypto, equities, FX, options) behind a unified YAML configuration and dependency injection registry.
- Shared pipeline for training, backtesting, paper trading, and live trading with reproducible artifacts.
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

- Train (crypto demo):
```bash
cp configs/examples/example_train_crypto.yaml configs/my_train.yaml
python train_model_multi_patch.py --config configs/my_train.yaml
```

- Backtest / simulation:
```bash
cp configs/examples/example_backtest_crypto.yaml configs/my_backtest.yaml
python script_backtest.py --config configs/my_backtest.yaml --offline-config configs/offline.yaml --dataset-split val
```

- Live trading via local Agent (CCEA architecture):
```bash
# 1. Deploy Agent locally (credentials stay on YOUR machine, never sent to cloud)
#    See docs/agent/INSTALLATION.md for full setup
python -m packages.agent.daemon.agentd --config configs/agent.yaml

# 2. (Optional) Use Cloud control plane to manage runs
#    Cloud sends lifecycle commands only - NEVER trades or stores keys
```
**Important**: Live trading runs ONLY in your local Agent. Cloud manages lifecycle (start/stop/deploy) but NEVER executes orders or stores your credentials. See [CCEA Overview](docs/architecture/CCEA_OVERVIEW.md).

For legacy/development dry-run testing:
```bash
# Development/testing only (not production CCEA architecture)
python script_live.py --config configs/my_live.yaml --dry-run
```

Run `python scripts/doctor.py --verbose` before the first training or trading run.

## Status

**14,000+ automated tests** | **MiFID II: Compliance-Ready Toolkit** | **EU AI Act: Compliance-Ready Toolkit** | **DORA: Compliance-Ready Toolkit** | **GDPR: Compliance-Ready Controls** | **CCEA Implemented** | **Built to Support Production Use**

*Note: "Compliance-ready" means the technical features are implemented and designed to align with regulatory requirements. These are tools to support compliance efforts, not compliance certifications. Actual regulatory compliance requires independent third-party assessment, proper configuration, legal review, and validation specific to your jurisdiction and use case.*

## CI Status
[![Docs quality](https://github.com/RivenKid69/AI-Powered-Quantitative-Research-Platform/actions/workflows/docs-quality.yml/badge.svg)](https://github.com/RivenKid69/AI-Powered-Quantitative-Research-Platform/actions/workflows/docs-quality.yml)
[![Security SAST](https://github.com/RivenKid69/AI-Powered-Quantitative-Research-Platform/actions/workflows/security-sast.yml/badge.svg)](https://github.com/RivenKid69/AI-Powered-Quantitative-Research-Platform/actions/workflows/security-sast.yml)

- Docs quality: markdown lint/render checks for user-facing docs.
- Security SAST: static analysis of adapters/core for regressions.

## Module Architecture

### ICT Provider Positioning (CCEA Architecture)

This platform is designed for **ICT Providers / Software Providers** under MiFID II scope:
- We provide algorithmic trading **research and infrastructure tools**
- Users trade through **THEIR OWN broker accounts** via **local Agent**
- Platform does NOT hold client assets or credentials in Cloud
- Cloud **NEVER** executes orders - only lifecycle management
- MiFID II does NOT apply directly to us

**Legal Position:**
- **NOT** an investment adviser or broker-dealer
- **NOT** providing investment recommendations
- **NOT** custodian of assets or credentials
- Software vendor providing tools for independent traders

See: `docs/architecture/CCEA_OVERVIEW.md` for full architecture and legal posture

### Module Structure (Post-Migration v2.0)

| Package | Purpose | Load |
|---------|---------|------|
| `services.core.risk_controls` | Universal risk controls (kill switch, pre-trade, audit) | Always |
| `services.algo_integration` | MiFID II B2B compliance toolkit | Enterprise clients |
| `services.archive.mifid_financial_entity` | Investment Firm modules | Archived (not loaded) |

#### For ICT Providers (Default)
```python
from services.core.risk_controls import (
    EnhancedKillSwitch, PreTradeControls, AuditTrailWriter,
    RealTimeMonitor, BusinessContinuityPlan
)
```

#### For B2B Clients (Financial Institutions)
Enable `services.algo_integration` for MiFID II compliance tools:
```python
from services.algo_integration import (
    BestExecutionAnalyzer,      # Article 27
    TCAComplianceWrapper,       # Transaction Cost Analysis
    ConformanceTestRunner,      # RTS 6 Article 5
    AlgorithmRegistry,          # Article 17(2)
    CertificateManager          # Deployment certification
)
```

#### Migration Note
The old `services.compliance` module is now a deprecated facade that emits warnings:
```python
# Old (deprecated - emits DeprecationWarning)
from services.compliance import EnhancedKillSwitch

# New (recommended)
from services.core.risk_controls import EnhancedKillSwitch
```

## Regulatory Compliance

### MiFID II (Directive 2014/65/EU)
All 7 compliance toolkit phases implemented (designed to align with requirements, not independently certified):
- Kill Switch & Pre-Trade Controls (RTS 6)
- Transaction Reporting (RTS 22)
- Record Keeping & Audit Trail (5-7 years retention)
- Best Execution & TCA (Article 27)
- Governance & Self-Assessment

Details: `docs/compliance/MIFID_II_COMPLIANCE_ROADMAP.md`

### EU AI Act (Regulation 2024/1689)
High-Risk AI System - all 4 compliance toolkit phases implemented and designed to align with requirements (1,007 tests, not independently certified):
- Risk Management System (Article 9)
- Data Governance & Technical Documentation (Article 10, 11)
- Human Oversight & Transparency (Article 13, 14)
- Quality Management System (Article 17)
- Conformity Assessment & EU Declaration (Article 43, 47)

Details: `docs/compliance/EU_AI_ACT_INTEGRATION_PLAN.md`

### DORA (Regulation 2022/2554)
Digital Operational Resilience Act - all 5 compliance toolkit phases implemented and designed to align with requirements (~1,015 tests, not independently certified):
- Phase 1: ICT Risk Management Framework (Articles 5-16)
- Phase 2: ICT Incident Management & Reporting (Articles 17-23)
- Phase 3: Digital Resilience Testing (Articles 24-27)
- Phase 4: Third-Party ICT Risk Management (Articles 28-44)
- Phase 5: Information Sharing, Dashboard & Unified Reporting

Details: `docs/compliance/DORA_INTEGRATION_PLAN.md`

### GDPR (Regulation 2016/679)
General Data Protection Regulation - all 9 compliance toolkit phases implemented and designed to align with requirements (CCEA-aligned, not independently certified):
- Phase 0: Data mapping, RoPA, Controller/Processor roles
- Phase 1: Transparency, Privacy Policy, DPA, DSAR SOP
- Phase 2: Data minimization, telemetry contracts, CI guardrails
- Phase 3: EU-only data residency enforcement
- Phase 4: Retention policies, auto-purge, legal holds
- Phase 5: DSAR workflows (access, portability, erasure)
- Phase 6: RBAC, access audit, break-glass procedures
- Phase 7: Security controls (Art. 32), breach workflow (Art. 33-34)
- Phase 8: Continuous compliance, privacy-by-design CI checks
- Phase 9: Enterprise/on-prem/VPC posture

**CCEA Privacy Design Commitments:**
- Cloud **NEVER** receives broker credentials or API keys
- Cloud **NEVER** receives order-like payloads in commands
- Telemetry **ALWAYS** redacted before transmission
- **EU-only** data residency enforced at runtime
- DSAR scope is Cloud-only; Agent data is customer-controlled

Details: `docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md`

## Supported Exchanges
| Asset class | Vendor(s) | Path | Modes | Status |
| --- | --- | --- | --- | --- |
| Crypto spot/futures | Binance | adapters/binance/ | sim, live | Production |
| Options/futures (crypto) | Deribit | adapters/deribit/ | sim, live | Beta |
| US equities execution | Alpaca | adapters/alpaca/ | sim, paper, live | Production |
| US equities data | Polygon, Yahoo | adapters/polygon/, adapters/yahoo/ | data, sim | Production |
| Forex | OANDA, Dukascopy | adapters/oanda/, adapters/dukascopy/ | sim, live (OANDA), historical (Dukascopy) | Beta |
| Traditional futures/options | Interactive Brokers, ThetaData | adapters/ib/, adapters/theta_data/ | paper/sim, live | Experimental |

## Guides

### CCEA Architecture Documentation
- `docs/architecture/CCEA_OVERVIEW.md` — Cloud/Agent boundary, threat model, legal posture
- `docs/cloud/README.md` — Control plane API, builder, governance
- `docs/agent/README.md` — Installation, vault, approvals, risk controls
- `docs/schemas/README.md` — JSON schemas with versioning guide
- `docs/runbooks/` — Incident response, kill-switch, recovery procedures

### Business & Legal Documentation
- `docs/business/CCEA_MARKETING_GUIDELINES.md` — Approved language, disclaimers, compliance
- `docs/business/CCEA_TERMS_OF_SERVICE_GUIDELINES.md` — ToS requirements, liability
- `docs/business/PRICING_DIFFERENTIATION_STRATEGY.md` — Product modes, pricing tiers
- `docs/business/OPEN_CORE_BUSINESS_MODEL.md` — Open-source strategy, licensing
- `docs/business/COMPETITIVE_MOAT.md` — Competitive advantage analysis
- `docs/business/IP_PROTECTION_STRATEGY.md` — IP protection, patents, trade secrets

### General Documentation
- `claude.md` — complete project guide (RU).
- `ARCHITECTURE.md` — system architecture and module map.
- `DOCS_INDEX.md` — documentation hub.
- `QUICK_START_REFERENCE.md` — command cheat sheet.
- `configs/examples/README.md` — ready-to-copy configs for train/backtest/live.
- `BUILD_INSTRUCTIONS.md` — native build notes.
- `docs/AI_GUIDE.md` — AI-assistant instructions.

## Runbooks
### Simulation / backtest
1. Copy an example config and set dataset paths (`offline-config`, `dataset-split`).
2. Run doctor without network if needed: `python scripts/doctor.py --skip-network`.
3. Execute `python script_backtest.py --config <cfg> --offline-config configs/offline.yaml --dataset-split val`.
4. Validate outputs: compare KPI to `benchmarks/sim_kpi_thresholds.json`, review reports in `artifacts/` and logs in `logs/`.

### Live trading (CCEA Architecture)
**Production: Via Local Agent**
1. **Agent Setup**: Install and configure Agent locally (`docs/agent/INSTALLATION.md`).
2. **Credentials**: Store broker API keys in Agent's local vault (NEVER upload to Cloud).
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
3. Reproduce with the live runner in dry-run mode: `python script_live.py --config <cfg> --dry-run --asset-class <crypto|equity|forex>` and watch `logs/` for adapter traces.
4. Refresh exchange metadata when applicable (for Binance: `python scripts/fetch_binance_filters.py`) and re-run doctor to confirm connectivity.

### Pre-release checklist
- [ ] `python scripts/doctor.py --verbose` (environment, credentials, clocks).
- [ ] `python -m pytest tests/test_service_mode_smoke.py tests/test_dry_run_executor.py` (or targeted smoke tests relevant to the change).
- [ ] `bash tools/test_markdown_render.sh` (pandoc render check).
- [ ] Verify configs referenced in README/Quick Start exist and secrets are environment-backed.
- [ ] Confirm supported exchange table and badges reflect current coverage.

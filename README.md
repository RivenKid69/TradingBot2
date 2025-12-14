# AI-Powered Quantitative Research Platform

RL-first research and trading stack for crypto, equities, FX, and derivatives with simulator-to-live parity.

## Architecture: Cloud-Controlled Execution Architecture (CCEA)

This platform implements **CCEA** - a strict separation between Cloud (research/monitoring/lifecycle) and Agent (execution/secrets/risk):

| Component | Responsibility | Secrets Access | Order Execution |
|-----------|---------------|----------------|-----------------|
| **Cloud** | Research, backtesting, monitoring, lifecycle management | **NEVER** | **NEVER** |
| **Agent** | Live execution, risk enforcement, local vault, order creation | **LOCAL ONLY** | **YES** |

**Key Security Guarantees:**
- Cloud **NEVER** stores broker API keys or credentials
- Cloud **NEVER** generates, transmits, or executes trading orders
- Cloud **NEVER** has access to exchange trading endpoints
- All trading operations occur **ONLY** in the Agent running locally or in user's VPC
- Telemetry is **ALWAYS** redacted before transmission to Cloud

**Product Modes:**
1. **Retail Research SaaS (EU-friendly)**: Cloud research/simulation + optional BYO Agent
2. **Retail Live via Local Agent**: Local auto-execution, cloud observability
3. **Enterprise Engine (on-prem/VPC)**: Full stack in customer infrastructure

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

- Live trading dry-run:
```bash
cp configs/examples/example_live_crypto.yaml configs/my_live.yaml
export BINANCE_API_KEY=...   # use environment variables, never commit secrets
export BINANCE_API_SECRET=...
python script_live.py --config configs/my_live.yaml --dry-run
```

Run `python scripts/doctor.py --verbose` before the first training or trading run.

## Status

**14,000+ automated tests** | **MiFID II 100%** | **EU AI Act 100%** | **DORA 100%** | **Production Ready**

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

See: `docs/CCEA_OVERVIEW.md` for full architecture and legal posture

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
All 7 phases complete:
- Kill Switch & Pre-Trade Controls (RTS 6)
- Transaction Reporting (RTS 22)
- Record Keeping & Audit Trail (5-7 years retention)
- Best Execution & TCA (Article 27)
- Governance & Self-Assessment

Details: `docs/compliance/MIFID_II_COMPLIANCE_ROADMAP.md`

### EU AI Act (Regulation 2024/1689)
High-Risk AI System compliance - all 4 phases complete (1,007 tests):
- Risk Management System (Article 9)
- Data Governance & Technical Documentation (Article 10, 11)
- Human Oversight & Transparency (Article 13, 14)
- Quality Management System (Article 17)
- Conformity Assessment & EU Declaration (Article 43, 47)

Details: `docs/compliance/EU_AI_ACT_INTEGRATION_PLAN.md`

### DORA (Regulation 2022/2554)
Digital Operational Resilience Act - all 5 phases complete (~1,015 tests):
- Phase 1: ICT Risk Management Framework (Articles 5-16)
- Phase 2: ICT Incident Management & Reporting (Articles 17-23)
- Phase 3: Digital Resilience Testing (Articles 24-27)
- Phase 4: Third-Party ICT Risk Management (Articles 28-44)
- Phase 5: Information Sharing, Dashboard & Unified Reporting

Details: `docs/compliance/DORA_INTEGRATION_PLAN.md`

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
- `docs/CCEA_OVERVIEW.md` — Cloud/Agent boundary, threat model, legal posture
- `docs/cloud/README.md` — Control plane API, builder, governance
- `docs/agent/README.md` — Installation, vault, approvals, risk controls
- `docs/schemas/README.md` — JSON schemas with versioning guide
- `docs/runbooks/` — Incident response, kill-switch, recovery procedures

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

### Live trading
1. Preflight: `python scripts/doctor.py --verbose`; sync system clock; set API keys via environment; ensure `risk.*` and `execution.*` limits are conservative.
2. Paper/dry run: `python script_live.py --config <cfg> --dry-run` and inspect `logs/live_*`.
3. Go live: remove `--dry-run`, pin `asset_class`/`vendor` in the config, and monitor metrics in `artifacts/live/` plus alerts in `logs/`.
4. Safety: keep the kill switch enabled (`runtime.kill_switch_enabled: true`), rotate keys periodically, and back up runtime state.

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

# Archived DORA Financial Entity Modules

**Status:** Phase 7 Complete - All FE modules archived
**Migration Date:** 2025-01-17
**Version:** 1.0.0

These modules implement DORA requirements for **Financial Entities** (Art. 2),
not for ICT Third-Party Service Providers (Art. 30).

## Why Archived?

As an ICT service provider, we:
- Comply with Art. 30 (contractual requirements for ICT providers)
- Support client due diligence (Art. 28)
- **DO NOT** implement the full FE DORA framework ourselves

Our active DORA code lives in:
- `services/core/` - Operational resilience (14 modules)
- `services/dora_integration/` - Client-facing interfaces (21 modules)

## When to Use These Modules?

If you're building a product **FOR financial entities** to manage their own
DORA compliance, these modules provide a comprehensive reference implementation.

**Use cases:**
- Building a DORA compliance SaaS for banks
- Creating a regulatory reporting tool for investment firms
- Developing an ICT risk management platform for insurance companies

## Architecture Overview

```
services/
├── core/                          # CORE - Operational resilience (14 modules)
│   └── [healthcheck, backup, dr, alerting, etc.]
│
├── dora_integration/              # INTEGRATION - Client interfaces (21 modules)
│   ├── due_diligence/             # Audit & due diligence support
│   ├── incident_interface/        # Client incident notifications
│   ├── third_party/               # Subcontractor management
│   ├── contracts/                 # Contractual requirements
│   ├── reporting/                 # ROI data generation
│   └── sharing/                   # Information sharing
│
├── dora/                          # FACADE - Re-exports from integration
│   └── __init__.py                # Backward compatibility layer
│
└── archive/                       # ARCHIVE - FE modules (this directory)
    └── dora_financial_entity/     # 23 FE modules
        ├── configs/               # 2 FE configuration files
        │   ├── entity_classification.yaml
        │   └── nca_identification.yaml
        ├── __init__.py            # Deprecation-aware exports
        ├── README.md              # This file
        └── [23 Python modules]
```

## Archived Modules (23 total)

### Phase 0: Proportionality Assessment (Articles 2-4, 16)

| # | Module | DORA Article | Description |
|---|--------|--------------|-------------|
| 1 | `scope_verification.py` | Art. 2 | Determines if entity falls under DORA scope |
| 2 | `function_classification.py` | Art. 3(22) | Classifies critical/important functions |
| 3 | `proportionality.py` | Art. 4, 16 | Determines applicable regime (full/simplified) |

### Phase 1: ICT Risk Management Framework (Articles 5-16)

| # | Module | DORA Article | Description |
|---|--------|--------------|-------------|
| 4 | `governance.py` | Art. 5 | Management body oversight, ICT governance |
| 5 | `ict_risk_framework.py` | Art. 6 | ICT risk management framework |
| 6 | `ict_systems.py` | Art. 7 | ICT systems, protocols, tools management |
| 7 | `ict_identification.py` | Art. 8 | ICT asset and risk identification |
| 8 | `protection.py` | Art. 9 | Protection and prevention controls |
| 9 | `detection.py` | Art. 10 | Anomaly detection and monitoring |
| 10 | `response_recovery.py` | Art. 11 | Incident response and recovery |
| 11 | `backup_recovery.py` | Art. 12 | Backup policies and recovery procedures |
| 12 | `learning.py` | Art. 13 | Post-incident learning and evolution |
| 13 | `ict_business_continuity.py` | Art. 15 | ICT business continuity management |
| 14 | `simplified_framework.py` | Art. 16 | Simplified ICT framework for smaller entities |

### Phase 2: ICT Incident Management (Articles 17-23)

| # | Module | DORA Article | Description |
|---|--------|--------------|-------------|
| 15 | `incident_management.py` | Art. 17 | ICT incident management process |
| 16 | `supervisory_feedback.py` | Art. 22 | Handling NCA feedback on incidents |

### Phase 3: Digital Resilience Testing (Articles 24-27)

| # | Module | DORA Article | Description |
|---|--------|--------------|-------------|
| 17 | `resilience_testing.py` | Art. 24 | Digital operational resilience testing programme |
| 18 | `ict_testing.py` | Art. 25 | Testing of ICT tools and systems |
| 19 | `tlpt.py` | Art. 26 | Threat-led penetration testing (TLPT) |
| 20 | `tester_management.py` | Art. 27 | Requirements for TLPT testers |
| 21 | `pooled_testing.py` | Art. 26(3) | Pooled TLPT arrangements |

### Phase 5: Information Sharing & Integration (Article 45+)

| # | Module | DORA Article | Description |
|---|--------|--------------|-------------|
| 22 | `cross_regulation.py` | - | Cross-regulation integration (AI Act, MiFID II) |
| 23 | `training_participation.py` | Art. 30(2)(i) | FE requests for provider training participation |

## Archived Configurations (2 files)

| Config | Description | Used By |
|--------|-------------|---------|
| `entity_classification.yaml` | Entity type classification rules | `scope_verification.py`, `proportionality.py` |
| `nca_identification.yaml` | NCA contact mapping per jurisdiction | `incident_reporting.py`, `supervisory_feedback.py` |

## DORA Article Reference

| Article | Title | Related Modules |
|---------|-------|-----------------|
| Art. 2 | Scope | `scope_verification.py` |
| Art. 3(22) | Critical/Important Functions | `function_classification.py` |
| Art. 4 | Proportionality | `proportionality.py` |
| Art. 5 | Governance | `governance.py` |
| Art. 6 | ICT Risk Management | `ict_risk_framework.py` |
| Art. 7 | ICT Systems | `ict_systems.py` |
| Art. 8 | Identification | `ict_identification.py` |
| Art. 9 | Protection | `protection.py` |
| Art. 10 | Detection | `detection.py` |
| Art. 11 | Response & Recovery | `response_recovery.py` |
| Art. 12 | Backup & Recovery | `backup_recovery.py` |
| Art. 13 | Learning | `learning.py` |
| Art. 15 | Business Continuity | `ict_business_continuity.py` |
| Art. 16 | Simplified Framework | `simplified_framework.py` |
| Art. 17 | Incident Management | `incident_management.py` |
| Art. 22 | Supervisory Feedback | `supervisory_feedback.py` |
| Art. 24 | Resilience Testing | `resilience_testing.py` |
| Art. 25 | ICT Testing | `ict_testing.py` |
| Art. 26 | TLPT | `tlpt.py`, `pooled_testing.py` |
| Art. 27 | Tester Requirements | `tester_management.py` |
| Art. 30(2)(i) | Training Participation | `training_participation.py` |

## Usage

### Importing Archived Modules (with Deprecation Warning)

```python
# Import will work but emit a DeprecationWarning
from services.archive.dora_financial_entity import (
    DORAScope,
    DORAGovernanceFramework,
    DORAICTRiskFramework,
    # ... etc
)
```

### Direct Import (No Warning)

```python
# Direct import from archived module
from services.archive.dora_financial_entity.governance import (
    DORAGovernanceFramework,
    GovernanceRole,
    create_governance_framework,
)
```

### Example: Building a Financial Entity Compliance Tool

```python
from services.archive.dora_financial_entity import (
    # Scope verification
    DORAScope, create_scope_verifier, DORAEntityType,
    # Proportionality
    ProportionalityAssessor, DORARegime,
    # Governance
    DORAGovernanceFramework, GovernanceRole,
    # Risk Management
    DORAICTRiskFramework,
)

# Step 1: Verify DORA scope
scope_verifier = create_scope_verifier()
scope_result = scope_verifier.verify_scope(
    entity_type=DORAEntityType.INVESTMENT_FIRM,
    is_authorized=True,
    member_state="DE"
)

if scope_result.in_scope:
    # Step 2: Determine proportionality regime
    assessor = ProportionalityAssessor()
    regime = assessor.assess(entity_data)

    if regime == DORARegime.FULL:
        # Step 3: Implement full governance framework
        governance = DORAGovernanceFramework()
        governance.assign_role(GovernanceRole.ICT_RISK_OWNER, "CISO")
    else:
        # Use simplified framework
        from services.archive.dora_financial_entity import DORASimplifiedFramework
        framework = DORASimplifiedFramework()
```

## Recovery Instructions

To restore any module to active development:

```bash
# The modules are still in the archive, just copy back
cp services/archive/dora_financial_entity/<module>.py services/dora/

# Or restore from git history if needed
git log --oneline -- services/dora/<module>.py  # Find last commit
git checkout <commit>^ -- services/dora/<module>.py
```

## Testing

Archived modules have their own test suite:

```bash
# Run all archive tests
pytest tests/archive/dora_financial_entity/ -v

# Run specific module tests
pytest tests/archive/dora_financial_entity/test_governance.py -v

# Run with coverage
pytest tests/archive/dora_financial_entity/ --cov=services.archive.dora_financial_entity
```

## Technical Standards Implemented

These modules implement requirements from:

| Standard | Description | Status |
|----------|-------------|--------|
| DORA (EU) 2022/2554 | Main regulation | Full implementation |
| CDR 2024/1774 | RTS on ICT Risk Management | `ict_risk_framework.py` |
| CDR 2024/1772 | RTS on Incident Classification | `incident_management.py` |
| CDR 2025/301 | RTS on Incident Reporting | `incident_management.py` |
| CIR 2024/2956 | ITS on Register of Information | Referenced in configs |

## Key Compliance Dates

| Date | Milestone | Relevant Module |
|------|-----------|-----------------|
| 17 Jan 2025 | DORA Application Date | All modules |
| 31 Mar 2025 | ROI Reference Date | `register_of_information.py` |
| 30 Apr 2025 | First ROI Submission | `register_of_information.py` |
| Ongoing | Annual Resilience Testing | `resilience_testing.py` |
| Every 3 Years | TLPT (if designated) | `tlpt.py` |

## References

- [DORA Regulation (EU) 2022/2554](https://eur-lex.europa.eu/eli/reg/2022/2554/oj)
- [DORA Article 2 - Scope](https://www.digital-operational-resilience-act.com/Article_2.html)
- [DORA Article 30 - ICT Provider Requirements](https://www.digital-operational-resilience-act.com/Article_30.html)
- [ESA DORA Implementation Hub](https://www.eba.europa.eu/regulation-and-policy/operational-resilience)
- [ESMA DORA Technical Standards](https://www.esma.europa.eu/publications-and-data/dora)

## Maintainer Notes

**Why These Modules Were Archived:**

1. **Regulatory Clarity**: DORA distinguishes between obligations for Financial Entities (Art. 2-27) and ICT Providers (Art. 30). As an ICT provider, we only need Art. 30 compliance.

2. **Separation of Concerns**: The integration layer (`services/dora_integration/`) provides what our FE clients need from us. The archived modules are for FE internal use.

3. **Product Potential**: These modules represent significant IP that could be productized as a separate DORA compliance offering for financial entities.

**Key Decision**: These modules are archived, NOT deleted. They remain fully functional and can be restored or used for building client-facing compliance tools.

# Archived DORA Financial Entity Modules

**Status:** Archive prepared for Phase 7 migration

These modules implement DORA requirements for **Financial Entities** (Art. 2),
not for ICT Third-Party Service Providers (Art. 30).

## Why Archived?

As an ICT service provider, we:
- Comply with Art. 30 (contractual requirements)
- Support client due diligence (Art. 28)
- DO NOT implement full FE DORA framework

Our active DORA code lives in:
- `services/core/` - Operational resilience (14 modules)
- `services/dora_integration/` - Client-facing interfaces (21 modules)

## When to Use?

If you're building a product FOR financial entities to manage their own
DORA compliance, these modules provide a reference implementation.

## Archived Modules (23 total - to be moved in Phase 7)

| # | Module | Article | Description |
|---|--------|---------|-------------|
| 1 | `scope_verification.py` | Art. 2 | DORA scope determination |
| 2 | `function_classification.py` | Art. 3(22) | Critical function classification |
| 3 | `proportionality.py` | Art. 4, 16 | Proportionality regime |
| 4 | `governance.py` | Art. 5 | ICT governance framework |
| 5 | `ict_risk_framework.py` | Art. 6 | ICT risk management |
| 6 | `ict_systems.py` | Art. 7 | ICT systems management |
| 7 | `ict_identification.py` | Art. 8 | ICT asset identification |
| 8 | `protection.py` | Art. 9 | Protection controls |
| 9 | `detection.py` | Art. 10 | Anomaly detection |
| 10 | `response_recovery.py` | Art. 11 | Incident response |
| 11 | `backup_recovery.py` | Art. 12 | Backup policies |
| 12 | `learning.py` | Art. 13 | Learning & evolving |
| 13 | `ict_business_continuity.py` | Art. 15 | Business continuity |
| 14 | `simplified_framework.py` | Art. 16 | Simplified ICT framework |
| 15 | `incident_management.py` | Art. 17 | Incident management |
| 16 | `supervisory_feedback.py` | Art. 22 | NCA feedback handling |
| 17 | `resilience_testing.py` | Art. 24 | Testing programme |
| 18 | `ict_testing.py` | Art. 25 | ICT tools testing |
| 19 | `tlpt.py` | Art. 26 | Threat-led penetration testing |
| 20 | `tester_management.py` | Art. 27 | Tester requirements |
| 21 | `pooled_testing.py` | Art. 26(3) | Pooled TLPT |
| 22 | `cross_regulation.py` | - | Cross-regulation integration |
| 23 | `training_participation.py` | Art. 30(2)(i) | FE training requests |

## Archived Configs (in configs/)

| Config | Description |
|--------|-------------|
| `entity_classification.yaml` | Entity type classification |
| `nca_identification.yaml` | NCA contact mapping |

## Recovery

To restore any module to active development:

```bash
# Find last commit before archival
git log --oneline -- services/dora/<module>.py

# Restore from specific commit
git checkout <commit>^ -- services/dora/<module>.py
```

## Architecture Diagram

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
└── archive/                       # ARCHIVE - FE modules
    └── dora_financial_entity/     # 23 FE modules (this directory)
        ├── configs/
        │   ├── entity_classification.yaml
        │   └── nca_identification.yaml
        ├── README.md (this file)
        └── [23 Python modules]
```

## References

- [DORA Regulation (EU) 2022/2554](https://eur-lex.europa.eu/eli/reg/2022/2554/oj)
- [DORA Article 2 - Scope](https://www.digital-operational-resilience-act.com/Article_2.html)
- [DORA Article 30 - ICT Provider Requirements](https://www.digital-operational-resilience-act.com/Article_30.html)
- [ESA DORA Implementation Hub](https://www.eba.europa.eu/regulation-and-policy/operational-resilience)

# DORA Integration Layer Configuration

This directory contains configuration files for the DORA Integration Layer.

## Configuration Files (to be created in subsequent phases)

| File | Phase | Description |
|------|-------|-------------|
| `incident_notification.yaml` | Phase 2 | Client notification settings |
| `third_party_management.yaml` | Phase 3 | Subcontractor management config |
| `information_sharing.yaml` | Phase 6 | Information sharing settings |
| `digital_resilience_strategy.yaml` | Phase 6 | Resilience strategy config |

## Migration Plan

Configuration files will be migrated from `config/dora/` during the respective phases.

### Files to Remain in `config/dora/`

- `proportionality_assessment.yaml` - Internal toggle for provider proportionality

### Files to Move to Integration Layer

- `third_party_management.yaml` - Phase 3
- `information_sharing.yaml` - Phase 6
- `digital_resilience_strategy.yaml` - Phase 6

### Files to Archive (Phase 7)

- `entity_classification.yaml` -> `services/archive/dora_financial_entity/configs/`
- `nca_identification.yaml` -> `services/archive/dora_financial_entity/configs/`

## Directory Structure After Migration

```
config/
├── dora/
│   └── proportionality_assessment.yaml  # KEEP - internal toggle
│
└── dora_integration/
    ├── incident_notification.yaml       # NEW - Phase 2
    ├── third_party_management.yaml      # MOVE - Phase 3
    ├── information_sharing.yaml         # MOVE - Phase 6
    └── digital_resilience_strategy.yaml # MOVE - Phase 6
```

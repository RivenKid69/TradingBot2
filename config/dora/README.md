# DORA Configuration Directory

This directory previously contained Financial Entity (FE) configuration files.

## Migration Notice

**As of 2025-01-17**, FE-specific configurations have been moved to the archive:

| File | New Location | Reason |
|------|--------------|--------|
| `entity_classification.yaml` | `services/archive/dora_financial_entity/configs/` | FE scope determination (Art. 2) |
| `nca_identification.yaml` | `services/archive/dora_financial_entity/configs/` | FE NCA contacts |
| `proportionality_assessment.yaml` | `services/archive/dora_financial_entity/configs/` | FE regime determination (Art. 16) |

## Why?

We are an **ICT Third-Party Service Provider** (Art. 30), not a **Financial Entity** (Art. 2).

- FE configurations are for entities that fall under DORA scope directly
- ICT Providers comply with Art. 30 contractual requirements
- We don't have a "regime" (full/simplified) — Art. 30 applies uniformly

## Active DORA Configurations

Active ICT Provider configurations are located in `/configs/dora/`:

```
configs/dora/
├── digital_resilience_strategy.yaml   # Our operational resilience strategy
├── third_party_management.yaml        # Subcontractor management
└── information_sharing.yaml           # Art. 45 threat intel sharing
```

## See Also

- [DORA Integration Layer](../../services/dora_integration/)
- [Archived FE Modules](../../services/archive/dora_financial_entity/README.md)

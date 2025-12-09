# Archived DORA Modules - Not Applicable to ICT Provider Role

**Archive Date**: 2025-01-17
**Reason**: These modules implement DORA obligations for Financial Entities, not ICT Third-Party Providers

---

## Background

Our platform is an **ICT Third-Party Service Provider**, NOT a Financial Entity under DORA Article 2(1)(a-t).

DORA applies to us **indirectly** through:
- Contractual requirements (Article 28-30)
- Client audit rights
- NCA inspection rights

The modules listed below were designed for direct DORA compliance by Financial Entities and are **not applicable** to our role as an ICT provider.

---

## Archived Modules

### 1. scope_verification.py
**Original Location**: `services/dora/scope_verification.py`
**Purpose**: Determines if DORA applies to an entity
**Why Not Applicable**: We already know DORA applies to us via client contracts. This module helps entities determine if they're in scope - irrelevant for providers.

### 2. proportionality.py
**Original Location**: `services/dora/proportionality.py`
**Purpose**: Assesses entity size for simplified regime eligibility
**Why Not Applicable**: Proportionality regimes (Art. 16) apply to financial entities based on their size/complexity. As a provider, we don't use these classifications.

### 3. supervisory_feedback.py
**Original Location**: `services/dora/supervisory_feedback.py`
**Purpose**: Manages NCA feedback and corrective actions
**Why Not Applicable**: Direct NCA supervision applies to financial entities. We interact with NCAs only through client audits, not direct supervisory feedback loops.

---

## Archived Configurations

### 4. nca_identification.yaml
**Original Location**: `configs/dora/nca_identification.yaml`
**Purpose**: Maps entity types to competent authorities
**Why Not Applicable**: Financial entities use this to identify their NCA. We don't have a designated NCA (unless designated as CTPP).

### 5. entity_classification.yaml
**Original Location**: `configs/dora/entity_classification.yaml`
**Purpose**: Classifies financial entity types per Article 2(1)
**Why Not Applicable**: We are not classified as a financial entity under these categories.

---

## Modules NOT Archived (Initially Considered)

The following modules were initially considered for archiving but are **KEPT** because they serve valid ICT provider purposes:

| Module | Reason to Keep |
|--------|----------------|
| `concentration_risk.py` | Monitors our market concentration for CTPP designation risk |
| `ctpp_oversight.py` | Prepares for potential CTPP designation if client base grows |
| `pooled_testing.py` | Adapted to `pooled_audit_support.py` for Art. 30(4) |
| `register_of_information.py` | Adapted to `provider_info_package.py` for client ROI support |
| `exit_strategies.py` | Required for Art. 28(8) exit obligations |

---

## Migration Notes

If these modules are needed for other purposes:

1. **For testing/reference**: Modules remain in original locations but are marked as "archived" in this document
2. **For actual archiving**: Copy to `services/archive/dora_not_applicable/` directory
3. **For imports**: Update `__init__.py` if modules are physically moved

---

## ICT Provider vs Financial Entity

```
┌─────────────────────────────────────────────────────────────────┐
│                   DORA APPLICABILITY MATRIX                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FINANCIAL ENTITY                 ICT PROVIDER (Us)             │
│  ─────────────────                ─────────────────             │
│  ✓ scope_verification             ✗ scope_verification          │
│  ✓ proportionality                ✗ proportionality             │
│  ✓ supervisory_feedback           ✗ supervisory_feedback        │
│  ✓ nca_identification             ✗ nca_identification          │
│  ✓ entity_classification          ✗ entity_classification       │
│                                                                  │
│  ✓ incident_reporting (to NCA)    ✓ incident_reporting (to client)│
│  ✓ resilience_testing (own)       ✓ resilience_testing (support)│
│  ✓ tlpt (coordination)            ✓ tlpt (cooperation)          │
│  ✓ register_of_information        ✓ provider_info_package       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

- DORA Article 2: Scope of application
- DORA Article 28-30: Third-party ICT risk management
- DORA Article 31-44: CTPP oversight (if designated)
- DORA_OPERATIONAL_RESILIENCE_PLAN.md Section 4.E

---

*This README documents the rationale for archiving modules that are not applicable to our ICT provider role under DORA.*

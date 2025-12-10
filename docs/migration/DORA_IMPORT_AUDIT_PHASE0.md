# DORA Import Audit - Phase 0

**Date:** 2025-01-17
**Status:** Complete
**Branch:** refactor/dora-integration-layer

## Summary

This document records all imports from `services.dora` module identified during Phase 0 preparation.

## Import Locations

### 1. Internal Module Imports (`services/dora/__init__.py`)

The main `__init__.py` imports from 44 submodules to provide a unified API.
These are internal re-exports and will be updated in Phase 8.

### 2. Test File Imports

| Test File | Imports From |
|-----------|--------------|
| `tests/test_dora_phase0_proportionality.py` | `scope_verification`, `function_classification`, `proportionality` |
| `tests/test_dora_phase1_ict_risk_management.py` | `governance`, `ict_risk_framework`, `ict_systems`, `ict_identification`, `protection`, `detection`, `response_recovery`, `backup_recovery`, `learning`, `communication`, `ict_business_continuity`, `simplified_framework` |
| `tests/test_dora_phase2_incident_management.py` | `incident_management`, `incident_classification`, `incident_reporting`, `cyber_threat_notification`, `reporting_templates`, `supervisory_feedback`, `third_party_incidents` |
| `tests/test_dora_phase3_tlpt.py` | `tlpt` |
| `tests/test_dora_phase3_resilience_testing.py` | `resilience_testing` |
| `tests/test_dora_phase3_ict_testing.py` | `ict_testing` |
| `tests/test_dora_phase3_tester_management.py` | `tester_management` |
| `tests/test_dora_phase3_pooled_testing.py` | `pooled_testing` |
| `tests/dora/test_dora_exit_strategies.py` | `exit_strategies` |
| `tests/dora/test_dora_unified_reporting.py` | `unified_reporting` |
| `tests/dora/test_dora_ctpp_oversight.py` | `ctpp_oversight` |
| `tests/dora/test_dora_information_sharing.py` | `information_sharing` |
| `tests/dora/test_dora_compliance_dashboard.py` | `compliance_dashboard` |
| `tests/dora/test_dora_register_of_information.py` | `register_of_information` |
| `tests/dora/test_dora_cross_regulation.py` | `cross_regulation` |
| `tests/dora/test_dora_contractual_requirements.py` | `contractual_requirements` |
| `tests/dora/test_dora_pooled_audit_support.py` | `pooled_audit_support` |
| `tests/dora/test_dora_concentration_risk.py` | `concentration_risk` |
| `tests/dora/test_dora_sla_guardrails.py` | `sla_guardrails` |
| `tests/dora/test_dora_third_party_risk.py` | `third_party_risk` |

### 3. Documentation Imports (`services/dora/incident_reporting.py`)

Contains example import in docstring:
```python
>>> from services.dora.client_incident_notification import DORAClientNotification
>>> from services.dora.incident_reporting import DORAIncidentReporter
```

## Module Categories

### Integration Layer Modules (21 - to move in Phases 1-6)

**Due Diligence (Phase 1):**
- `audit_readiness.py`
- `provider_info_package.py`
- `pooled_audit_support.py`
- `compliance_dashboard.py`

**Incident Interface (Phase 2):**
- `client_incident_notification.py`
- `incident_classification.py`
- `incident_reporting.py`
- `cyber_threat_notification.py`
- `communication.py`

**Third Party (Phase 3):**
- `concentration_risk.py`
- `ctpp_oversight.py`
- `third_party_risk.py`
- `third_party_incidents.py`
- `subcontractor_management.py`

**Contracts (Phase 4):**
- `contractual_requirements.py`
- `sla_guardrails.py`
- `exit_strategies.py`

**Reporting (Phase 5):**
- `unified_reporting.py`
- `reporting_templates.py`
- `register_of_information.py`

**Sharing (Phase 6):**
- `information_sharing.py`

### Archive Modules (23 - to move in Phase 7)

- `scope_verification.py`
- `function_classification.py`
- `proportionality.py`
- `governance.py`
- `ict_risk_framework.py`
- `ict_systems.py`
- `ict_identification.py`
- `protection.py`
- `detection.py`
- `response_recovery.py`
- `backup_recovery.py`
- `learning.py`
- `ict_business_continuity.py`
- `simplified_framework.py`
- `incident_management.py`
- `supervisory_feedback.py`
- `resilience_testing.py`
- `ict_testing.py`
- `tlpt.py`
- `tester_management.py`
- `pooled_testing.py`
- `cross_regulation.py`
- `training_participation.py`

## Migration Script Template

```bash
#!/bin/bash
# scripts/migrate_dora_imports.sh

FILES=$(grep -rl "from services.dora" --include="*.py" | grep -v "services/dora/")

for file in $FILES; do
    echo "Processing: $file"

    # Phase 1 - Due Diligence
    sed -i 's/from services\.dora\.audit_readiness/from services.dora_integration.due_diligence.audit_readiness/g' "$file"
    sed -i 's/from services\.dora\.provider_info_package/from services.dora_integration.due_diligence.provider_info_package/g' "$file"
    # ... continue for all modules
done
```

## Verification Commands

```bash
# Count all imports
grep -r "from services.dora" --include="*.py" | wc -l

# Find external imports (excluding services/dora/)
grep -r "from services.dora" --include="*.py" | grep -v "services/dora/"

# Verify no circular imports
python -c "from services.dora import *"
```

## Next Steps

1. Phase 1: Move due_diligence modules and update imports
2. Phase 2-6: Continue module migration
3. Phase 7: Archive FE modules
4. Phase 8: Update facade and all remaining imports
5. Phase 9: Final validation and CI/CD integration

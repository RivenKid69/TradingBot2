# Archive

This directory contains archived documentation, reports, and deprecated code that is no longer actively maintained but kept for historical reference.

## Directory Structure

```
archive/
├── 2025_11/                    # November 2025 completed work
│   ├── reports_2025_11/        # Bug fixes, audits, analysis reports
│   ├── reports_2025_11_24/     # Daily reports
│   ├── reports_2025_11_25_cleanup/  # Cleanup phase reports
│   ├── reports_2025_11_27/     # Analysis reports
│   └── verification_2025_11/   # Verification summaries
│
├── 2025_12_cleanup/            # December 2025 cleanup
│   ├── completed_plans/        # Completed alignment plans (internal)
│   ├── migration_reports/      # Phase 8 completion report
│   └── audits/                 # DOC_AUDIT_PHASE_11, DOCUMENTATION_ALIGNMENT_REPORT
│
├── deprecated/                 # Deprecated code and modules
│   ├── audits/                 # Old audit reports
│   ├── debug_archive/          # Debug scripts (historical)
│   ├── documentation_meta/     # Documentation metadata
│   ├── high_priority/          # Resolved high priority issues
│   ├── medium_issues/          # Resolved medium priority issues
│   ├── old_reports/            # Completed validation reports
│   ├── pbt/                    # PBT optimizer fix reports
│   ├── reports/                # Various completed reports
│   ├── services_archive/       # Deprecated DORA/MIFID services
│   ├── tests_archive/          # Deprecated test modules
│   ├── twin_critics/           # Twin critics audit
│   └── verification/           # Final verification summaries
│
└── root_files/                 # Old files from repository root
    ├── test_output.txt         # Old test outputs
    ├── test_vgs_pbt_output.txt # VGS PBT test outputs
    ├── ENCODING_TEST_RESULTS.md
    ├── METRICS_QUICK_REFERENCE.txt
    ├── Design Doc CCEA Cloud.txt
    └── Список проблем и несогсасованностей.txt
```

## Archive Date

**Created**: 2025-12-16

## Size Summary

| Directory | Size | Contents |
|-----------|------|----------|
| `2025_11/` | ~5 MB | November 2025 reports, audits, fixes |
| `2025_12_cleanup/` | ~500 KB | December 2025 completed plans, reports |
| `deprecated/` | ~6 MB | Deprecated services, tests, old reports |
| `root_files/` | ~340 KB | Old root-level files |
| **Total** | ~12 MB | |

## Notes

- All files here were marked as COMPLETED, FINAL, or DEPRECATED
- These files are kept for audit trail and historical reference
- Terminology like "Production Ready", "compliant", or "100%" may appear as informal internal shorthand and should not be treated as an independently validated claim (audit/certification/legal opinion)
- Active documentation remains in `docs/`
- Active tests remain in `tests/`
- Active services remain in `services/`

## Restoration

If you need to restore any archived content:

```bash
# Example: restore a specific file
cp archive/deprecated/old_reports/L3_VALIDATION_REPORT.md docs/

# Example: restore entire November 2025 reports
cp -r archive/2025_11/reports_2025_11 docs/archive/
```

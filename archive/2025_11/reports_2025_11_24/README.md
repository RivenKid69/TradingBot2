# Archived Reports - November 2025 (Phase 2)

**Archived Date**: 2025-11-24
**Purpose**: Superseded intermediate reports from indicator bugs and documentation modernization

## Summary

This archive contains 10 reports that were superseded by more comprehensive final documents. These reports represent the evolution of bug analysis and fixes during November 2025.

## Reports by Category

### Indicator Bugs Analysis (4 reports)

**Superseded by**: [INDICATOR_INITIALIZATION_FIXES_SUMMARY.md](../../../INDICATOR_INITIALIZATION_FIXES_SUMMARY.md) + [CONCEPTUAL_BUGS_VERIFICATION_REPORT_2025_11_24.md](../../../CONCEPTUAL_BUGS_VERIFICATION_REPORT_2025_11_24.md)

- **INDICATOR_BUGS_COMPREHENSIVE_ANALYSIS.md** - Initial quick analysis
- **INDICATOR_BUGS_FINAL_REPORT.md** - Intermediate final report
- **INDICATOR_BUGS_VERIFICATION_REPORT_2025_11_24.md** - Verification step
- **INDICATOR_INITIALIZATION_BUGS_REPORT.md** - Detailed bug report

**Evolution**: Quick analysis → Final report → Verification → Comprehensive fixes summary

### Conceptual Analysis (1 report)

**Superseded by**: [CONCEPTUAL_ANALYSIS_REPORT_2025_11_24.md](../../../CONCEPTUAL_ANALYSIS_REPORT_2025_11_24.md)

- **CRITICAL_ANALYSIS_THREE_PROBLEMS_2025_11_24.md** - Deep dive into 3 alleged problems

**Key Finding**: All 3 problems were either already fixed or false positives (system working as designed)

### Test Coverage (1 report)

**Superseded by**: [TEST_COVERAGE_REPORT_2025_11_24.md](../../../TEST_COVERAGE_REPORT_2025_11_24.md)

- **COMPREHENSIVE_TEST_COVERAGE_REPORT.md** - Early comprehensive coverage report

**Evolution**: Initial report → Expanded with services coverage → Full report with 200+ tests

### Documentation Meta (4 reports)

**Purpose**: Historical documentation of the modernization process itself

- **CONSOLIDATION_SUMMARY.md** - Summary of advantage normalization consolidation
- **DOCUMENTATION_MODERNIZATION_PLAN_2025_11_24.md** - The modernization plan (completed)
- **DOCUMENTATION_UPDATE_SUMMARY_2025_11_24.md** - Update summary
- **TARGET_CLIPPING_FIX_SUMMARY_2025_11_24.md** - Target clipping bug fix

**Status**: Documentation modernization completed successfully

## Key Outcomes

### Indicator Bugs
- ✅ **2 CRITICAL bugs fixed**: RSI initialization, CCI mean deviation
- ✅ **1 FALSE ALARM**: ATR (SMA variant is correct)
- ⚠️ **ACTION**: Models trained before 2025-11-24 should be retrained

### Conceptual Analysis
- ✅ **Problem #1** (Look-ahead bias): Already fixed 2025-11-23
- ✅ **Problem #2** (VGS formula): Not a bug (design choice)
- ✅ **Problem #3** (Reward discontinuity): Not a bug (standard practice)

### Test Coverage
- ✅ **200+ tests** added across all critical systems
- ✅ **98%+ pass rate** maintained
- ✅ Coverage expanded from core to services and infrastructure

### Documentation
- ✅ **43% reduction** in root directory clutter (44 → 31 files)
- ✅ **22 reports archived** (verification_2025_11/ + reports_2025_11_24/)
- ✅ Clear separation: Active fixes vs Historical analysis

## Active References

For current active documentation, see root directory:

### Critical Fixes
- [INDICATOR_INITIALIZATION_FIXES_SUMMARY.md](../../../INDICATOR_INITIALIZATION_FIXES_SUMMARY.md) - Indicator fixes ⭐
- [CONCEPTUAL_BUGS_VERIFICATION_REPORT_2025_11_24.md](../../../CONCEPTUAL_BUGS_VERIFICATION_REPORT_2025_11_24.md) - Full verification ⭐
- [CONCEPTUAL_ANALYSIS_REPORT_2025_11_24.md](../../../CONCEPTUAL_ANALYSIS_REPORT_2025_11_24.md) - Conceptual analysis ⭐
- [CRITICAL_ANALYSIS_REPORT_2025_11_24.md](../../../CRITICAL_ANALYSIS_REPORT_2025_11_24.md) - Twin Critics loss fix ⭐
- [DATA_LEAKAGE_FIX_REPORT_2025_11_23.md](../../../DATA_LEAKAGE_FIX_REPORT_2025_11_23.md) - Data leakage fix ⚠️ Requires retraining

### Test Coverage
- [TEST_COVERAGE_REPORT_2025_11_24.md](../../../TEST_COVERAGE_REPORT_2025_11_24.md) - Full test coverage
- [TEST_COVERAGE_SERVICES_SUMMARY.md](../../../TEST_COVERAGE_SERVICES_SUMMARY.md) - Services coverage
- [TEST_FIXES_SUMMARY_2025_11_24.md](../../../TEST_FIXES_SUMMARY_2025_11_24.md) - Test fixes

### Master Documentation
- [CLAUDE.md](../../../CLAUDE.md) - Complete project documentation (Russian)
- [README.md](../../../README.md) - Project overview
- [DOCS_INDEX.md](../../../DOCS_INDEX.md) - Documentation index

---

## Archive Timeline

**2025-11-24 (Phase 1)**: Archived 12 verification reports → `docs/archive/verification_2025_11/`
**2025-11-24 (Phase 2)**: Archived 10 superseded reports → `docs/archive/reports_2025_11_24/` (this archive)

**Total Archived**: 22 reports (134 KB) - cleaned up root directory by 43%

---

**For questions about archived reports, consult the active references above or git history.**

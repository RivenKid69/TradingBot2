# Archived Root Test Files - November 2025

## Purpose

This directory contains test files that were in the project root and have been superseded by more comprehensive tests in the `tests/` directory.

## Organization

Tests are archived here for historical reference. All archived tests have been replaced by improved versions.

## Archived Tests

### test_bug_twin_critics_vf_clipping.py
- **Replaced by**: `tests/test_twin_critics_vf_clipping_correctness.py` (11 tests, 100% pass)
- **Reason**: Old exploratory test replaced by comprehensive verification suite
- **Date**: 2025-11-23

### test_normalization_consistency.py
- **Status**: Incomplete test with TODO placeholders
- **Reason**: Never finished, functionality covered by other tests
- **Date**: 2025-11-23

## Note on Root Test Files

The project currently has **277 test files in the root directory**. This archive contains only the clearly obsolete tests.

**Recommendation for future cleanup**:
1. Move all active tests to `tests/` directory
2. Update import paths and references
3. Update CI/CD configuration
4. Keep root directory clean

This should be done carefully to avoid breaking existing test workflows.

## Archive Date

2025-11-23

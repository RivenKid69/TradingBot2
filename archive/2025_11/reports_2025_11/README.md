# Archived Reports - November 2025

This directory contains historical reports, audits, and analyses from the intensive bug-fixing and verification period (2025-11-20 to 2025-11-23).

## Purpose

These reports document the discovery and resolution of critical bugs in the AI-Powered Quantitative Research Platform system. While the fixes are still active and documented in the main documentation, these detailed reports are archived for historical reference.

## Organization

- **audits/** - Deep audits and comprehensive analyses
- **analysis/** - Detailed technical analyses
- **fixes/** - Individual fix reports and summaries
- **verification/** - Verification and validation reports
- **integration/** - Integration testing reports

## Key Fixes Documented Here

1. **Action Space Fixes** (2025-11-21)
   - Position doubling bug (DELTA→TARGET semantics)
   - LongOnlyWrapper sign convention
   - Action space range unification

2. **LSTM State Reset Fix** (2025-11-21)
   - Temporal leakage between episodes
   - 5-15% improvement in model accuracy

3. **Numerical Stability Fixes** (2025-11-20)
   - Gradient explosion prevention
   - VGS-UPGD noise interaction
   - CVaR quantile clipping

4. **Twin Critics Fixes** (2025-11-21/22)
   - GAE computation with Twin Critics
   - VF clipping independent critic updates

5. **VGS Fix v3.1** (2025-11-23)
   - E[g²] computation corrected
   - Stochastic variance properly calculated

## Current Status

**All fixes are ACTIVE and PRODUCTION READY**. See main documentation:
- [CLAUDE.md](../../../CLAUDE.md) - Complete project documentation
- [DOCS_INDEX.md](../../../DOCS_INDEX.md) - Documentation index

## Archive Date

2025-11-23

**Note**: These reports are kept for historical reference. For current implementation details, always refer to the main documentation.

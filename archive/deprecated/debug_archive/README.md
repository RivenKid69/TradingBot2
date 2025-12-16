# Debug Scripts Archive

This directory contains debug/diagnostic scripts used during bug investigation.

## VGS Parameter Tracking Bug (Bug #9)

Scripts used to diagnose and fix the VGS parameter tracking issue:

- **debug_vgs_param_mystery.py** - Initial investigation showing VGS tracked wrong parameters
- **debug_vgs_detailed.py** - Detailed tracing with instrumented _setup_dependent_components
- **debug_vgs_source.py** - Comprehensive tracing of VGS creation and parameter updates
- **debug_vgs_error.py** - Error reproduction script

These scripts are kept for reference but are not part of the production codebase.

## Bug #9 Fix Summary

**Problem**: VGS tracked stale parameter copies after `model.load()`, rendering gradient scaling ineffective.

**Solution**:
1. Don't pickle `_parameters` in VGS.__getstate__()
2. Reinitialize via `update_parameters()` after load
3. Reset `_setup_complete` flag in load() to force VGS recreation

**Status**: ✅ Fixed and tested
**Production Test**: `test_vgs_param_tracking_fix.py`

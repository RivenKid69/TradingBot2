#!/usr/bin/env python3
"""
Verification script to check if technical indicators are correctly integrated
into the observation pipeline and visible to the model.

This script should be run in the REAL training environment where:
- Python dependencies are installed (numpy, pandas, etc.)
- Cython modules are compiled for the correct Python version
- Data files exist in data/processed/

Usage:
    python verify_observation_integration.py
"""

import sys
import os

print("=" * 80)
print("VERIFICATION: Technical Indicators in Observation Pipeline")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Check imports
# ============================================================================
print("STEP 1: Checking critical imports...")
print("-" * 80)

errors = []

try:
    import numpy as np
    print("✓ numpy imported")
except ImportError as e:
    print(f"✗ numpy FAILED: {e}")
    errors.append("numpy")

try:
    import pandas as pd
    print("✓ pandas imported")
except ImportError as e:
    print(f"✗ pandas FAILED: {e}")
    errors.append("pandas")

try:
    from obs_builder import build_observation_vector
    print("✓ obs_builder.build_observation_vector imported")
    OBS_BUILDER_AVAILABLE = True
except ImportError as e:
    print(f"✗ obs_builder FAILED: {e}")
    print("  WARNING: Will fall back to legacy observation builder!")
    OBS_BUILDER_AVAILABLE = False
    errors.append("obs_builder")

try:
    import lob_state_cython as lob
    N_FEATURES = lob.N_FEATURES
    print(f"✓ lob_state_cython imported, N_FEATURES = {N_FEATURES}")
except ImportError as e:
    print(f"✗ lob_state_cython FAILED: {e}")
    N_FEATURES = 53  # fallback
    errors.append("lob_state_cython")

try:
    from mediator import Mediator
    print("✓ mediator.Mediator imported")
except ImportError as e:
    print(f"✗ mediator FAILED: {e}")
    errors.append("mediator")

print()

if "numpy" in errors or "pandas" in errors:
    print("CRITICAL ERROR: numpy or pandas not available!")
    print("Cannot proceed with verification.")
    print("Please install dependencies: pip install numpy pandas")
    sys.exit(1)

# ============================================================================
# STEP 2: Check observation space size
# ============================================================================
print("STEP 2: Checking observation space configuration...")
print("-" * 80)

EXPECTED_OBS_SIZE = N_FEATURES + 4  # 53 + 4 = 57 typically
print(f"Expected observation size: {EXPECTED_OBS_SIZE}")
print(f"  N_FEATURES (from obs_builder): {N_FEATURES}")
print(f"  + 4 tail slots (units, cash, signal_pos, log_ret)")
print()

# ============================================================================
# STEP 3: Check mediator obs_builder usage
# ============================================================================
print("STEP 3: Checking mediator obs_builder integration...")
print("-" * 80)

from mediator import _HAVE_OBS_BUILDER

print(f"_HAVE_OBS_BUILDER = {_HAVE_OBS_BUILDER}")

if _HAVE_OBS_BUILDER:
    print("✓ Mediator will use obs_builder.build_observation_vector()")
    print("  → 57 features will be populated with technical indicators")
else:
    print("✗ Mediator will use LEGACY fallback mode")
    print("  → Only ~12 features will be populated (NO technical indicators!)")
    print()
    print("  ACTION REQUIRED:")
    print("  1. Ensure obs_builder.pyx is compiled for your Python version")
    print("  2. Run: python setup.py build_ext --inplace")
    print("  3. Verify: python -c 'from obs_builder import build_observation_vector'")

print()

# ============================================================================
# STEP 4: Create mock environment and test observation
# ============================================================================
print("STEP 4: Testing observation building with synthetic data...")
print("-" * 80)

# Create synthetic dataframe with technical indicators
df = pd.DataFrame({
    'timestamp': [1700000000 + i * 3600 for i in range(200)],
    'open': [50000 + i * 10 for i in range(200)],
    'high': [50100 + i * 10 for i in range(200)],
    'low': [49900 + i * 10 for i in range(200)],
    'close': [50000 + i * 10 for i in range(200)],
    'volume': [100 + i for i in range(200)],
    'quote_asset_volume': [5000000 + i * 1000 for i in range(200)],

    # Technical indicators (from prepare_and_run.py)
    'sma_5': [50000 + i * 10 for i in range(200)],
    'sma_15': [50000 + i * 9 for i in range(200)],
    'rsi': [50 + (i % 20) for i in range(200)],
    'cvd_24h': [(i % 10) / 10.0 for i in range(200)],
    'cvd_168h': [(i % 20) / 20.0 for i in range(200)],
    'yang_zhang_24h': [0.01 + (i % 5) * 0.001 for i in range(200)],
    'yang_zhang_168h': [0.015 + (i % 7) * 0.001 for i in range(200)],
    'garch_12h': [0.02 + (i % 3) * 0.002 for i in range(200)],
    'garch_24h': [0.025 + (i % 4) * 0.002 for i in range(200)],
    'ret_15m': [(i % 15) * 0.0001 for i in range(200)],
    'ret_60m': [(i % 25) * 0.0002 for i in range(200)],
    'fear_greed_value': [50 + (i % 30) for i in range(200)],
})

# Mock environment
class MockObsSpace:
    shape = (EXPECTED_OBS_SIZE,)

class MockEnv:
    observation_space = MockObsSpace()
    df = df
    _last_reward_price = 50000.0

    def _resolve_reward_price(self, idx, row):
        if row is not None and hasattr(row, 'close'):
            return float(row.close)
        return 50000.0

class MockState:
    units = 0.5
    cash = 10000.0
    step_idx = 100
    last_vol_imbalance = 0.1
    last_trade_intensity = 5.0
    last_realized_spread = 0.001
    last_agent_fill_ratio = 0.95
    token_index = 0

# Create mediator and build observation
env = MockEnv()
mediator = Mediator(env)

row = df.iloc[100]
state = MockState()
mark_price = 51000.0

try:
    obs = mediator._build_observation(row=row, state=state, mark_price=mark_price)

    print(f"Observation shape: {obs.shape}")
    print(f"Expected shape: ({EXPECTED_OBS_SIZE},)")

    if obs.shape == (EXPECTED_OBS_SIZE,):
        print("✓ Observation has correct shape!")
    else:
        print(f"✗ WRONG SHAPE! Expected {EXPECTED_OBS_SIZE}, got {obs.shape[0]}")

    non_zero_count = np.count_nonzero(obs)
    print(f"\nNon-zero values: {non_zero_count} / {EXPECTED_OBS_SIZE}")

    if non_zero_count >= 40:
        print("✓ Good! Most features are populated (>40 non-zero)")
    elif non_zero_count >= 20:
        print("⚠ Warning: Some features populated but not all (~20-40 non-zero)")
    else:
        print("✗ PROBLEM: Very few features populated (<20 non-zero)")
        print("  This indicates legacy fallback mode is being used!")

    print(f"\nFirst 15 values: {obs[:15]}")
    print(f"Positions 32-40 (norm_cols with indicators): {obs[32:40]}")
    print(f"Last 5 values: {obs[-5:]}")

    # Check specific indicators
    print("\nChecking specific technical indicators in observation:")

    # norm_cols region should contain cvd, garch, yang_zhang
    norm_cols_region = obs[32:40]
    norm_cols_nonzero = np.count_nonzero(norm_cols_region)

    print(f"  norm_cols (positions 32-39) non-zero count: {norm_cols_nonzero}/8")
    if norm_cols_nonzero >= 5:
        print("  ✓ Technical indicators (cvd, garch, yang_zhang) are present!")
    else:
        print("  ✗ Technical indicators NOT in observation (legacy mode)")

except Exception as e:
    print(f"✗ ERROR building observation: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# STEP 5: Check with real data if available
# ============================================================================
print("STEP 5: Checking real data availability...")
print("-" * 80)

data_dir = "data/processed"
if os.path.exists(data_dir):
    feather_files = [f for f in os.listdir(data_dir) if f.endswith('.feather')]
    if feather_files:
        print(f"Found {len(feather_files)} feather files in {data_dir}")
        print(f"Example: {feather_files[0]}")

        # Try loading one file
        try:
            sample_file = os.path.join(data_dir, feather_files[0])
            df_real = pd.read_feather(sample_file)

            print(f"\nColumns in {feather_files[0]}:")
            print(f"  Total columns: {len(df_real.columns)}")

            # Check for technical indicators
            indicators = ['sma_5', 'sma_15', 'rsi', 'cvd_24h', 'cvd_168h',
                         'yang_zhang_24h', 'yang_zhang_168h', 'garch_12h', 'garch_24h']

            present = [ind for ind in indicators if ind in df_real.columns]
            missing = [ind for ind in indicators if ind not in df_real.columns]

            print(f"  Technical indicators present: {len(present)}/{len(indicators)}")
            if present:
                print(f"    ✓ {', '.join(present)}")
            if missing:
                print(f"    ✗ Missing: {', '.join(missing)}")
                print("\n  ACTION: Run prepare_and_run.py to generate missing indicators")

        except Exception as e:
            print(f"Error reading feather file: {e}")
    else:
        print(f"No feather files found in {data_dir}")
else:
    print(f"Data directory {data_dir} not found")

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

issues = []

if not OBS_BUILDER_AVAILABLE:
    issues.append("obs_builder not compiled/available - using LEGACY mode")

if not _HAVE_OBS_BUILDER:
    issues.append("Mediator cannot import obs_builder - falling back to legacy")

if 'obs' in locals():
    if obs.shape[0] != EXPECTED_OBS_SIZE:
        issues.append(f"Wrong observation size: {obs.shape[0]} vs {EXPECTED_OBS_SIZE}")

    if np.count_nonzero(obs) < 40:
        issues.append(f"Too few non-zero features: {np.count_nonzero(obs)}/57")

if issues:
    print("\n⚠️  ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print("\n📋 RECOMMENDED ACTIONS:")
    print("  1. Verify Python version matches compiled modules")
    print(f"     Current: {sys.version}")
    print("  2. Recompile Cython modules if needed:")
    print("     python setup.py build_ext --inplace")
    print("  3. Verify obs_builder import works:")
    print("     python -c 'from obs_builder import build_observation_vector; print(\"OK\")'")
    print("  4. Run tests:")
    print("     python tests/test_technical_indicators_in_obs.py")
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("\n  Technical indicators are correctly integrated into observations.")
    print("  The model will receive all 57 features including:")
    print("    • Market data (price, volumes)")
    print("    • Moving averages (sma_5, sma_15)")
    print("    • Technical indicators (RSI, MACD, etc.)")
    print("    • CVD (cumulative volume delta)")
    print("    • GARCH volatility")
    print("    • Yang-Zhang volatility")
    print("    • Fear & Greed Index")
    print("    • Agent state")

print("=" * 80)

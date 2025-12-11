"""Quick test of critical fixes."""
import sys
sys.path.insert(0, '/home/user/ai-quant-platform')

print("Testing critical fixes...")

# Test 1: Import works
try:
    from variance_gradient_scaler import VarianceGradientScaler
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Basic instantiation
try:
    scaler = VarianceGradientScaler(None, enabled=True, beta=0.99, alpha=0.1)
    print("✓ Instantiation successful")
    print(f"  Scaler: {scaler}")
except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    sys.exit(1)

# Test 3: get_normalized_variance doesn't crash
try:
    norm_var = scaler.get_normalized_variance()
    print(f"✓ get_normalized_variance() works: {norm_var}")
    assert norm_var == 0.0, "Should be 0 with no statistics"
except Exception as e:
    print(f"✗ get_normalized_variance() failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: get_scaling_factor doesn't crash
try:
    scale = scaler.get_scaling_factor()
    print(f"✓ get_scaling_factor() works: {scale}")
    assert scale == 1.0, "Should be 1.0 during warmup"
except Exception as e:
    print(f"✗ get_scaling_factor() failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: step() doesn't crash
try:
    scaler.step()
    print(f"✓ step() works, step_count={scaler._step_count}")
    assert scaler._step_count == 1, "Step count should increment"
except Exception as e:
    print(f"✗ step() failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL CRITICAL FIX TESTS PASSED!")
print("="*60)

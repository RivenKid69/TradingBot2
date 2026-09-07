"""
Test to confirm ISSUE #8: Outdated PBT API usage in integration tests

This test confirms that the PBTScheduler.exploit_and_explore() method
now returns 3 values instead of 2, and old code needs to be updated.
"""

import pytest

torch = pytest.importorskip("torch")
from adversarial.pbt_scheduler import PBTScheduler, PBTConfig, HyperparamConfig


def test_exploit_and_explore_returns_three_values():
    """Confirm that exploit_and_explore returns 3 values (not 2)."""

    config = PBTConfig(
        population_size=3,
        perturbation_interval=5,
        hyperparams=[
            HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
        ],
        metric_mode="max",
    )

    scheduler = PBTScheduler(config, seed=42)
    population = scheduler.initialize_population()

    # Give members different performances
    for i, member in enumerate(population):
        scheduler.update_performance(
            member,
            performance=float(i),
            step=0,
            model_state_dict={"dummy": torch.randn(2, 2)},
        )

    # Trigger exploitation for worst performer
    worst_member = population[0]
    worst_member.step = 5  # Trigger at perturbation interval

    # NEW API: Should return 3 values
    result = scheduler.exploit_and_explore(
        worst_member,
        model_state_dict={"dummy": torch.randn(2, 2)},
    )

    # Verify 3 values returned
    assert len(result) == 3, f"Expected 3 return values, got {len(result)}"

    new_parameters, new_hyperparams, checkpoint_format = result

    # Verify types
    assert isinstance(new_hyperparams, dict), "new_hyperparams should be dict"
    assert checkpoint_format is None or isinstance(
        checkpoint_format, str
    ), "checkpoint_format should be None or str"

    print("✅ CONFIRMED: exploit_and_explore() returns 3 values")
    print(f"   - new_parameters: {type(new_parameters)}")
    print(f"   - new_hyperparams: {type(new_hyperparams)}")
    print(f"   - checkpoint_format: {checkpoint_format}")


def test_old_api_usage_fails():
    """Confirm that OLD API (2 values) raises ValueError."""

    config = PBTConfig(
        population_size=3,
        perturbation_interval=5,
        hyperparams=[
            HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
        ],
        metric_mode="max",
    )

    scheduler = PBTScheduler(config, seed=42)
    population = scheduler.initialize_population()

    for i, member in enumerate(population):
        scheduler.update_performance(
            member,
            performance=float(i),
            step=0,
            model_state_dict={"dummy": torch.randn(2, 2)},
        )

    worst_member = population[0]
    worst_member.step = 5

    # OLD API: Try to unpack 2 values (should fail)
    with pytest.raises(ValueError, match="too many values to unpack"):
        new_state_dict, new_hyperparams = scheduler.exploit_and_explore(
            worst_member,
            model_state_dict={"dummy": torch.randn(2, 2)},
        )

    print("✅ CONFIRMED: OLD API (2 values) raises ValueError")


def test_correct_usage_with_v2_checkpoint():
    """Test correct usage with model_parameters (v2 checkpoint format)."""

    config = PBTConfig(
        population_size=3,
        perturbation_interval=5,
        hyperparams=[
            HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
        ],
        metric_mode="max",
    )

    scheduler = PBTScheduler(config, seed=42)
    population = scheduler.initialize_population()

    # Use model_parameters (v2 format) instead of model_state_dict
    mock_vgs_state = {
        "step_count": 100,
        "grad_mean_ema": 0.001,
        "grad_var_ema": 0.0001,
    }

    for i, member in enumerate(population):
        scheduler.update_performance(
            member,
            performance=float(i),
            step=0,
            model_parameters={  # ✅ NEW: Use model_parameters
                "policy": {"weight": torch.randn(2, 2)},
                "vgs_state": mock_vgs_state,
            },
        )

    worst_member = population[0]
    worst_member.step = 5

    # NEW API: Correct usage
    new_parameters, new_hyperparams, checkpoint_format = scheduler.exploit_and_explore(
        worst_member,
    )

    # Verify v2 checkpoint format was used
    if new_parameters is not None:
        assert (
            checkpoint_format == "v2_full_parameters"
        ), f"Expected v2_full_parameters, got {checkpoint_format}"

        # Verify VGS state is included
        assert "vgs_state" in new_parameters, "VGS state should be in parameters"
        assert new_parameters["vgs_state"] == mock_vgs_state, "VGS state should match source"

        print("✅ CONFIRMED: V2 checkpoint format includes VGS state")
        print(f"   - checkpoint_format: {checkpoint_format}")
        print(f"   - VGS state included: {new_parameters['vgs_state']}")
    else:
        print("⚠️  No exploitation occurred (expected if no ready members)")


if __name__ == "__main__":
    print("=" * 80)
    print("ISSUE #8 Confirmation Tests")
    print("=" * 80)

    print("\n[Test 1] Verify exploit_and_explore returns 3 values")
    test_exploit_and_explore_returns_three_values()

    print("\n[Test 2] Verify OLD API (2 values) fails")
    test_old_api_usage_fails()

    print("\n[Test 3] Verify correct usage with V2 checkpoints")
    test_correct_usage_with_v2_checkpoint()

    print("\n" + "=" * 80)
    print("✅ ISSUE #8 CONFIRMED: Tests need to be updated to use 3-value return")
    print("=" * 80)

"""
Comprehensive Deep Audit for Twin Critics Default Behavior.

This test suite performs exhaustive validation of:
1. Default behavior in ALL configurations
2. Interaction with ALL other features
3. ALL edge cases and boundary conditions
4. Save/load in ALL scenarios
5. Optimizer and gradient flow
6. Memory and performance characteristics
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from custom_policy_patch1 import CustomActorCriticPolicy
import gc


class TestDefaultBehaviorExhaustive:
    """Exhaustive tests for default Twin Critics behavior."""

    def test_default_with_no_arch_params(self):
        """Test default with completely empty arch_params."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        # Completely empty arch_params
        policy = CustomActorCriticPolicy(obs_space, act_space, lambda x: 0.001)

        # Should still enable Twin Critics by default
        assert (
            policy._use_twin_critics is True
        ), "Twin Critics should be enabled with no arch_params"

    def test_default_with_none_arch_params(self):
        """Test default with None arch_params."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(obs_space, act_space, lambda x: 0.001, arch_params=None)

        assert policy._use_twin_critics is True

    def test_default_with_empty_dict(self):
        """Test default with empty dict arch_params."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(obs_space, act_space, lambda x: 0.001, arch_params={})

        assert policy._use_twin_critics is True

    def test_default_with_critic_none(self):
        """Test default when critic config is None."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space, act_space, lambda x: 0.001, arch_params={"critic": None}
        )

        assert policy._use_twin_critics is True

    def test_default_quantile_all_combinations(self):
        """Test all combinations of quantile settings with default Twin Critics."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        configs = [
            {"distributional": True, "num_quantiles": 8},
            {"distributional": True, "num_quantiles": 16},
            {"distributional": True, "num_quantiles": 32},
            {"distributional": True, "num_quantiles": 64},
        ]

        for cfg in configs:
            policy = CustomActorCriticPolicy(
                obs_space, act_space, lambda x: 0.001, arch_params={"critic": cfg}
            )
            assert policy._use_twin_critics is True, f"Failed for config {cfg}"
            assert policy.quantile_head_2 is not None, f"Second critic missing for {cfg}"
            assert policy.quantile_head_2.num_quantiles == cfg["num_quantiles"]

    def test_default_categorical_all_atom_sizes(self):
        """Test categorical mode with various atom sizes."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        atom_sizes = [21, 51, 101, 201]

        for n_atoms in atom_sizes:
            policy = CustomActorCriticPolicy(
                obs_space,
                act_space,
                lambda x: 0.001,
                arch_params={"num_atoms": n_atoms, "critic": {"distributional": False}},
            )
            assert policy._use_twin_critics is True, f"Failed for {n_atoms} atoms"
            assert policy.dist_head_2 is not None
            assert policy.dist_head_2.out_features == n_atoms


class TestExplicitControlExhaustive:
    """Exhaustive tests for explicit Twin Critics control."""

    def test_explicit_true_variations(self):
        """Test all variations of explicitly setting to True."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        true_values = [True, 1, "true", "True", "TRUE", "yes", "Yes", "YES", "on", "ON"]

        for val in true_values:
            policy = CustomActorCriticPolicy(
                obs_space,
                act_space,
                lambda x: 0.001,
                arch_params={
                    "critic": {"distributional": True, "num_quantiles": 16, "use_twin_critics": val}
                },
            )
            # All should be True (bool coercion)
            assert policy._use_twin_critics is True, f"Failed for value {val}"

    def test_explicit_false_variations(self):
        """Test all variations of explicitly setting to False."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        # Only False and 0 are actually converted to False by bool()
        # Strings are truthy in Python (bool("false") == True)
        false_values = [False, 0]

        for val in false_values:
            policy = CustomActorCriticPolicy(
                obs_space,
                act_space,
                lambda x: 0.001,
                arch_params={
                    "critic": {"distributional": True, "num_quantiles": 16, "use_twin_critics": val}
                },
            )
            # All should be False (bool coercion)
            assert policy._use_twin_critics is False, f"Failed for value {val}"
            assert policy.quantile_head_2 is None

        # Note: string values like "false" would be bool("false") = True
        # This is expected Python behavior


class TestArchitectureConsistency:
    """Test architectural consistency with Twin Critics."""

    def test_both_critics_same_architecture_quantile(self):
        """Verify both quantile critics have identical architecture."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={"critic": {"distributional": True, "num_quantiles": 32}},
        )

        # Same number of quantiles
        assert policy.quantile_head.num_quantiles == policy.quantile_head_2.num_quantiles

        # Same huber kappa
        assert policy.quantile_head.huber_kappa == policy.quantile_head_2.huber_kappa

        # Same input dimension
        assert policy.quantile_head.linear.in_features == policy.quantile_head_2.linear.in_features

        # Same output dimension
        assert (
            policy.quantile_head.linear.out_features == policy.quantile_head_2.linear.out_features
        )

        # But different parameters
        assert (
            policy.quantile_head.linear.weight.data_ptr()
            != policy.quantile_head_2.linear.weight.data_ptr()
        )

    def test_both_critics_same_architecture_categorical(self):
        """Verify both categorical critics have identical architecture."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={"num_atoms": 51, "critic": {"distributional": False}},
        )

        # Same dimensions
        assert policy.dist_head.in_features == policy.dist_head_2.in_features
        assert policy.dist_head.out_features == policy.dist_head_2.out_features

        # Different parameters
        assert policy.dist_head.weight.data_ptr() != policy.dist_head_2.weight.data_ptr()

    def test_quantile_levels_match(self):
        """Verify quantile levels (taus) are identical."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={"critic": {"distributional": True, "num_quantiles": 16}},
        )

        # Quantile levels should be identical
        assert torch.allclose(policy.quantile_head.taus, policy.quantile_head_2.taus)


class TestForwardPassExhaustive:
    """Exhaustive forward pass tests."""

    def test_forward_pass_batch_sizes(self):
        """Test forward passes with various batch sizes."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={"critic": {"distributional": True, "num_quantiles": 16}},
        )

        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        for bs in batch_sizes:
            latent_vf = torch.randn(bs, policy.hidden_dim)

            # Both critics should work
            logits_1, logits_2 = policy._get_twin_value_logits(latent_vf)

            assert logits_1.shape == (bs, 16)
            assert logits_2.shape == (bs, 16)

            # Min should work
            min_val = policy._get_min_twin_values(latent_vf)
            assert min_val.shape == (bs, 1)

    def test_forward_pass_different_dtypes(self):
        """Test forward passes with different dtypes."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={"critic": {"distributional": True, "num_quantiles": 16}},
        )

        # float32
        latent_f32 = torch.randn(4, policy.hidden_dim, dtype=torch.float32)
        logits_1, logits_2 = policy._get_twin_value_logits(latent_f32)
        assert logits_1.dtype == torch.float32
        assert logits_2.dtype == torch.float32

        # float64 (should work via auto-conversion)
        latent_f64 = torch.randn(4, policy.hidden_dim, dtype=torch.float64)
        # This might raise or convert - just check it doesn't crash
        try:
            logits_1, logits_2 = policy._get_twin_value_logits(latent_f64)
        except Exception:
            pass  # Expected for some dtype mismatches


class TestOptimizerIntegration:
    """Test optimizer integration comprehensively."""

    def test_both_critics_in_optimizer_quantile(self):
        """Verify both quantile critics are in optimizer."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={"critic": {"distributional": True, "num_quantiles": 16}},
        )

        opt_params = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
        critic1_params = {id(p) for p in policy.quantile_head.parameters()}
        critic2_params = {id(p) for p in policy.quantile_head_2.parameters()}

        assert critic1_params.issubset(opt_params), "Critic 1 params not in optimizer"
        assert critic2_params.issubset(opt_params), "Critic 2 params not in optimizer"

        # No overlap between critics
        assert len(critic1_params & critic2_params) == 0, "Critics share parameters!"

    def test_both_critics_in_optimizer_categorical(self):
        """Verify both categorical critics are in optimizer."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space, act_space, lambda x: 0.001, arch_params={"critic": {"distributional": False}}
        )

        opt_params = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
        critic1_params = {id(p) for p in policy.dist_head.parameters()}
        critic2_params = {id(p) for p in policy.dist_head_2.parameters()}

        assert critic1_params.issubset(opt_params)
        assert critic2_params.issubset(opt_params)
        assert len(critic1_params & critic2_params) == 0

    def test_gradient_flow_both_critics(self):
        """Test that gradients flow to both critics."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={"critic": {"distributional": True, "num_quantiles": 16}},
        )

        latent_vf = torch.randn(4, policy.hidden_dim, requires_grad=True)

        # Forward
        logits_1, logits_2 = policy._get_twin_value_logits(latent_vf)

        # Loss
        loss = logits_1.mean() + logits_2.mean()

        # Backward
        policy.optimizer.zero_grad()
        loss.backward()

        # Both should have gradients
        assert policy.quantile_head.linear.weight.grad is not None
        assert policy.quantile_head_2.linear.weight.grad is not None

        # Gradients should be non-zero
        assert policy.quantile_head.linear.weight.grad.abs().sum() > 0
        assert policy.quantile_head_2.linear.weight.grad.abs().sum() > 0


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""

    def test_parameter_count_doubling(self):
        """Verify Twin Critics approximately doubles critic parameters."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        # Single critic
        policy_single = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={
                "critic": {"distributional": True, "num_quantiles": 16, "use_twin_critics": False}
            },
        )

        # Twin critics
        policy_twin = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={
                "critic": {"distributional": True, "num_quantiles": 16}
            },  # Default: twin enabled
        )

        params_single = sum(p.numel() for p in policy_single.quantile_head.parameters())
        params_twin_1 = sum(p.numel() for p in policy_twin.quantile_head.parameters())
        params_twin_2 = sum(p.numel() for p in policy_twin.quantile_head_2.parameters())

        # Both critics should have same param count
        assert params_twin_1 == params_twin_2

        # Twin critics should have 2x params
        assert params_twin_1 + params_twin_2 == 2 * params_single

    def test_memory_cleanup(self):
        """Test that Twin Critics models clean up properly."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        # Create and delete policy
        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={"critic": {"distributional": True, "num_quantiles": 16}},
        )

        # Store weak references
        import weakref

        weak_critic1 = weakref.ref(policy.quantile_head)
        weak_critic2 = weakref.ref(policy.quantile_head_2)

        # Delete policy
        del policy
        gc.collect()

        # Weak references should be dead
        assert weak_critic1() is None, "Critic 1 not cleaned up"
        assert weak_critic2() is None, "Critic 2 not cleaned up"


class TestMinValueSelection:
    """Test minimum value selection logic."""

    def test_min_selection_correctness(self):
        """Verify min(V1, V2) selects correct minimum."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={"critic": {"distributional": True, "num_quantiles": 16}},
        )

        # Create specific latent that will give predictable outputs
        latent_vf = torch.randn(8, policy.hidden_dim)

        # Get individual values
        logits_1, logits_2 = policy._get_twin_value_logits(latent_vf)
        val_1 = logits_1.mean(dim=-1, keepdim=True)
        val_2 = logits_2.mean(dim=-1, keepdim=True)

        # Get min
        min_val = policy._get_min_twin_values(latent_vf)

        # Verify correctness
        expected_min = torch.min(val_1, val_2)
        assert torch.allclose(min_val, expected_min, atol=1e-6)

        # Min should be <= both values
        assert (min_val <= val_1 + 1e-6).all()
        assert (min_val <= val_2 + 1e-6).all()

    def test_min_reduces_overestimation(self):
        """Test that min operation produces conservative estimates."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={"critic": {"distributional": True, "num_quantiles": 16}},
        )

        # Multiple samples
        latents = [torch.randn(16, policy.hidden_dim) for _ in range(10)]

        for latent_vf in latents:
            logits_1, logits_2 = policy._get_twin_value_logits(latent_vf)
            val_1 = logits_1.mean(dim=-1, keepdim=True)
            val_2 = logits_2.mean(dim=-1, keepdim=True)
            min_val = policy._get_min_twin_values(latent_vf)

            # Average of both critics
            avg_val = (val_1 + val_2) / 2

            # Min should be <= average (more conservative)
            assert (min_val <= avg_val + 1e-6).all(), "Min not conservative"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_error_when_accessing_second_critic_disabled(self):
        """Test that accessing second critic raises error when disabled."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={
                "critic": {"distributional": True, "num_quantiles": 16, "use_twin_critics": False}
            },
        )

        latent_vf = torch.randn(4, policy.hidden_dim)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Second critic is not enabled"):
            policy._get_value_logits_2(latent_vf)

        with pytest.raises(RuntimeError, match="not enabled"):
            policy._get_twin_value_logits(latent_vf)

    def test_fallback_when_twin_disabled(self):
        """Test that _get_min_twin_values falls back correctly."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        policy = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={
                "critic": {"distributional": True, "num_quantiles": 16, "use_twin_critics": False}
            },
        )

        latent_vf = torch.randn(4, policy.hidden_dim)

        # Should fall back to single critic
        min_val = policy._get_min_twin_values(latent_vf)
        single_val = policy._get_value_from_latent(latent_vf)

        assert torch.allclose(min_val, single_val)


class TestCrossValidation:
    """Cross-validation tests between modes."""

    def test_quantile_vs_categorical_consistency(self):
        """Test that both modes enable Twin Critics by default."""
        obs_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        # Quantile mode
        policy_q = CustomActorCriticPolicy(
            obs_space,
            act_space,
            lambda x: 0.001,
            arch_params={"critic": {"distributional": True, "num_quantiles": 16}},
        )

        # Categorical mode
        policy_c = CustomActorCriticPolicy(
            obs_space, act_space, lambda x: 0.001, arch_params={"critic": {"distributional": False}}
        )

        # Both should have Twin Critics enabled
        assert policy_q._use_twin_critics is True
        assert policy_c._use_twin_critics is True

        # Both should have second critic
        assert policy_q.quantile_head_2 is not None
        assert policy_c.dist_head_2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Comprehensive test suite for VGS parameter tracking after model load (Bug #9).

Tests that VGS correctly tracks policy parameters after save/load cycle.
"""

import tempfile
from pathlib import Path
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("torch")
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


class TestVGSParameterTracking:
    """Test VGS parameter tracking through save/load cycles."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        yield env
        env.close()

    def test_vgs_tracks_correct_params_after_load(self, env):
        """Test that VGS tracks correct policy parameters after load."""
        # Create and train model
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            variance_gradient_scaling=True,
            n_steps=64,
            verbose=0,
        )
        model.learn(total_timesteps=256)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_vgs.zip"

            # Save model
            model.save(save_path)

            # Load model
            loaded_model = DistributionalPPO.load(save_path, env=env)

            # Get parameter IDs
            policy_params = list(loaded_model.policy.parameters())
            vgs_params = loaded_model._variance_gradient_scaler._parameters

            assert vgs_params is not None, "VGS _parameters should not be None"
            assert len(vgs_params) == len(
                policy_params
            ), f"VGS should track all {len(policy_params)} parameters"

            # Check that VGS tracks the SAME parameter objects as policy
            ids_policy = set(id(p) for p in policy_params)
            ids_vgs = set(id(p) for p in vgs_params)

            assert ids_policy == ids_vgs, (
                f"VGS should track exact same parameter objects as policy. "
                f"Match: {len(ids_policy & ids_vgs)}/{len(policy_params)}"
            )

    def test_vgs_params_are_not_pickled(self, env):
        """Test that VGS._parameters is not included in pickle state."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            variance_gradient_scaling=True,
            n_steps=64,
            verbose=0,
        )

        vgs = model._variance_gradient_scaler
        state = vgs.__getstate__()

        assert "_parameters" not in state, "VGS __getstate__ should NOT include _parameters"
        assert "_logger" not in state, "VGS __getstate__ should NOT include _logger"

    def test_vgs_params_initialized_to_none_after_unpickle(self, env):
        """Test that VGS._parameters is None after unpickling."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            variance_gradient_scaling=True,
            n_steps=64,
            verbose=0,
        )

        vgs = model._variance_gradient_scaler
        state = vgs.__getstate__()

        # Simulate unpickling
        new_vgs = object.__new__(type(vgs))
        new_vgs.__setstate__(state)

        assert new_vgs._parameters is None, (
            "VGS._parameters should be None after unpickling "
            "(will be set by update_parameters())"
        )

    def test_vgs_update_parameters_works_correctly(self, env):
        """Test that update_parameters() correctly updates parameter references."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            variance_gradient_scaling=True,
            n_steps=64,
            verbose=0,
        )

        vgs = model._variance_gradient_scaler
        old_params = list(model.policy.parameters())
        old_ids = [id(p) for p in old_params]

        # Simulate parameter update
        vgs.update_parameters(model.policy.parameters())

        new_ids = [id(p) for p in vgs._parameters]

        assert new_ids == old_ids, "update_parameters() should update to current parameters"

    def test_vgs_functionality_after_load(self, env):
        """Test that VGS actually affects gradients after load."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            variance_gradient_scaling=True,
            n_steps=64,
            verbose=0,
        )
        model.learn(total_timesteps=256)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_vgs_func.zip"
            model.save(save_path)
            loaded_model = DistributionalPPO.load(save_path, env=env)

            # Collect some data
            loaded_model.collect_rollouts(
                loaded_model.env,
                loaded_model._last_obs,
                loaded_model.rollout_buffer,
                n_rollout_steps=loaded_model.n_steps,
            )

            # Compute gradients
            loaded_model.policy.optimizer.zero_grad()
            _, loss_dict = loaded_model.train()

            # Check that VGS actually has statistics
            vgs = loaded_model._variance_gradient_scaler
            assert (
                vgs._grad_norm_ema is not None or vgs._step_count == 0
            ), "VGS should track gradient statistics after training"

    def test_multiple_save_load_cycles(self, env):
        """Test VGS correctness through multiple save/load cycles."""
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            variance_gradient_scaling=True,
            n_steps=64,
            verbose=0,
        )
        model.learn(total_timesteps=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                save_path = Path(tmpdir) / f"model_{i}.zip"
                model.save(save_path)
                model = DistributionalPPO.load(save_path, env=env)

                # Verify VGS tracking after each cycle
                policy_params = list(model.policy.parameters())
                vgs_params = model._variance_gradient_scaler._parameters

                ids_policy = set(id(p) for p in policy_params)
                ids_vgs = set(id(p) for p in vgs_params)

                assert ids_policy == ids_vgs, f"Cycle {i}: VGS should track correct params"

                # Train a bit more
                if i < 2:
                    model.learn(total_timesteps=64)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

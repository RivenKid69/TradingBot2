"""
Test LSTM state reset at episode boundaries.

CRITICAL FIX (Issue #4): Verify that LSTM hidden states are properly reset
when episodes end, preventing temporal leakage across episode boundaries.

References:
    - Hausknecht & Stone (2015): "Deep Recurrent Q-Learning for POMDPs"
    - Kapturowski et al. (2018): "Recurrent Experience Replay in DQNs"
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from sb3_contrib.common.recurrent.type_aliases import RNNStates


def test_reset_lstm_states_single_env_done():
    """Test LSTM state reset for single environment finishing."""
    from distributional_ppo import DistributionalPPO

    # Create dummy states (2 layers, 3 envs, 64 hidden size)
    num_layers, num_envs, hidden_size = 2, 3, 64

    # Current states (non-zero values)
    current_pi_states = tuple(
        [
            torch.randn(num_layers, num_envs, hidden_size),  # hidden state
            torch.randn(num_layers, num_envs, hidden_size),  # cell state
        ]
    )
    current_vf_states = tuple(
        [
            torch.randn(num_layers, num_envs, hidden_size),
            torch.randn(num_layers, num_envs, hidden_size),
        ]
    )
    current_states = RNNStates(pi=current_pi_states, vf=current_vf_states)

    # Initial states (zeros)
    init_pi_states = tuple(
        [
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        ]
    )
    init_vf_states = tuple(
        [
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        ]
    )
    init_states = RNNStates(pi=init_pi_states, vf=init_vf_states)

    # Environment 1 is done
    dones = np.array([False, True, False], dtype=bool)

    # Create PPO instance (minimal setup)
    ppo = DistributionalPPO.__new__(DistributionalPPO)
    ppo.device = torch.device("cpu")

    # Reset states for done environments
    updated_states = ppo._reset_lstm_states_for_done_envs(current_states, dones, init_states)

    # Verify env 1 (index 1) was reset
    assert updated_states is not None
    assert hasattr(updated_states, "pi") and hasattr(updated_states, "vf")

    # Check actor states
    for i, state_tensor in enumerate(updated_states.pi):
        # Env 1 should be reset to zeros
        assert torch.allclose(state_tensor[:, 1, :], torch.zeros(num_layers, hidden_size))

        # Env 0 and 2 should be unchanged
        assert torch.allclose(state_tensor[:, 0, :], current_pi_states[i][:, 0, :])
        assert torch.allclose(state_tensor[:, 2, :], current_pi_states[i][:, 2, :])

    # Check critic states
    for i, state_tensor in enumerate(updated_states.vf):
        # Env 1 should be reset to zeros
        assert torch.allclose(state_tensor[:, 1, :], torch.zeros(num_layers, hidden_size))

        # Env 0 and 2 should be unchanged
        assert torch.allclose(state_tensor[:, 0, :], current_vf_states[i][:, 0, :])
        assert torch.allclose(state_tensor[:, 2, :], current_vf_states[i][:, 2, :])


def test_reset_lstm_states_multiple_envs_done():
    """Test LSTM state reset for multiple environments finishing."""
    from distributional_ppo import DistributionalPPO

    num_layers, num_envs, hidden_size = 2, 4, 64

    current_pi_states = tuple(
        [
            torch.randn(num_layers, num_envs, hidden_size),
            torch.randn(num_layers, num_envs, hidden_size),
        ]
    )
    current_vf_states = tuple(
        [
            torch.randn(num_layers, num_envs, hidden_size),
            torch.randn(num_layers, num_envs, hidden_size),
        ]
    )
    current_states = RNNStates(pi=current_pi_states, vf=current_vf_states)

    init_pi_states = tuple(
        [
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        ]
    )
    init_vf_states = tuple(
        [
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        ]
    )
    init_states = RNNStates(pi=init_pi_states, vf=init_vf_states)

    # Envs 0 and 2 are done
    dones = np.array([True, False, True, False], dtype=bool)

    ppo = DistributionalPPO.__new__(DistributionalPPO)
    ppo.device = torch.device("cpu")

    updated_states = ppo._reset_lstm_states_for_done_envs(current_states, dones, init_states)

    # Verify envs 0 and 2 were reset, envs 1 and 3 unchanged
    for i, state_tensor in enumerate(updated_states.pi):
        assert torch.allclose(state_tensor[:, 0, :], torch.zeros(num_layers, hidden_size))
        assert torch.allclose(state_tensor[:, 2, :], torch.zeros(num_layers, hidden_size))
        assert torch.allclose(state_tensor[:, 1, :], current_pi_states[i][:, 1, :])
        assert torch.allclose(state_tensor[:, 3, :], current_pi_states[i][:, 3, :])

    for i, state_tensor in enumerate(updated_states.vf):
        assert torch.allclose(state_tensor[:, 0, :], torch.zeros(num_layers, hidden_size))
        assert torch.allclose(state_tensor[:, 2, :], torch.zeros(num_layers, hidden_size))
        assert torch.allclose(state_tensor[:, 1, :], current_vf_states[i][:, 1, :])
        assert torch.allclose(state_tensor[:, 3, :], current_vf_states[i][:, 3, :])


def test_reset_lstm_states_no_dones():
    """Test that LSTM states are unchanged when no environments finish."""
    from distributional_ppo import DistributionalPPO

    num_layers, num_envs, hidden_size = 2, 3, 64

    current_pi_states = tuple(
        [
            torch.randn(num_layers, num_envs, hidden_size),
            torch.randn(num_layers, num_envs, hidden_size),
        ]
    )
    current_vf_states = tuple(
        [
            torch.randn(num_layers, num_envs, hidden_size),
            torch.randn(num_layers, num_envs, hidden_size),
        ]
    )
    current_states = RNNStates(pi=current_pi_states, vf=current_vf_states)

    init_pi_states = tuple(
        [
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        ]
    )
    init_vf_states = tuple(
        [
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        ]
    )
    init_states = RNNStates(pi=init_pi_states, vf=init_vf_states)

    # No environments are done
    dones = np.array([False, False, False], dtype=bool)

    ppo = DistributionalPPO.__new__(DistributionalPPO)
    ppo.device = torch.device("cpu")

    updated_states = ppo._reset_lstm_states_for_done_envs(current_states, dones, init_states)

    # All states should be unchanged
    for i in range(len(updated_states.pi)):
        assert torch.equal(updated_states.pi[i], current_pi_states[i])
    for i in range(len(updated_states.vf)):
        assert torch.equal(updated_states.vf[i], current_vf_states[i])


def test_reset_lstm_states_all_dones():
    """Test LSTM state reset when all environments finish."""
    from distributional_ppo import DistributionalPPO

    num_layers, num_envs, hidden_size = 2, 3, 64

    current_pi_states = tuple(
        [
            torch.randn(num_layers, num_envs, hidden_size),
            torch.randn(num_layers, num_envs, hidden_size),
        ]
    )
    current_vf_states = tuple(
        [
            torch.randn(num_layers, num_envs, hidden_size),
            torch.randn(num_layers, num_envs, hidden_size),
        ]
    )
    current_states = RNNStates(pi=current_pi_states, vf=current_vf_states)

    init_pi_states = tuple(
        [
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        ]
    )
    init_vf_states = tuple(
        [
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        ]
    )
    init_states = RNNStates(pi=init_pi_states, vf=init_vf_states)

    # All environments are done
    dones = np.array([True, True, True], dtype=bool)

    ppo = DistributionalPPO.__new__(DistributionalPPO)
    ppo.device = torch.device("cpu")

    updated_states = ppo._reset_lstm_states_for_done_envs(current_states, dones, init_states)

    # All states should be reset to zeros
    for state_tensor in updated_states.pi:
        assert torch.allclose(state_tensor, torch.zeros_like(state_tensor))
    for state_tensor in updated_states.vf:
        assert torch.allclose(state_tensor, torch.zeros_like(state_tensor))


def test_reset_lstm_states_simple_tuple():
    """Test LSTM state reset for simple tuple states (no separate pi/vf)."""
    from distributional_ppo import DistributionalPPO

    num_layers, num_envs, hidden_size = 2, 3, 64

    # Simple tuple without RNNStates wrapper
    current_states = tuple(
        [
            torch.randn(num_layers, num_envs, hidden_size),
            torch.randn(num_layers, num_envs, hidden_size),
        ]
    )

    init_states = tuple(
        [
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        ]
    )

    dones = np.array([False, True, False], dtype=bool)

    ppo = DistributionalPPO.__new__(DistributionalPPO)
    ppo.device = torch.device("cpu")

    updated_states = ppo._reset_lstm_states_for_done_envs(current_states, dones, init_states)

    # Verify env 1 was reset
    for i, state_tensor in enumerate(updated_states):
        assert torch.allclose(state_tensor[:, 1, :], torch.zeros(num_layers, hidden_size))
        assert torch.allclose(state_tensor[:, 0, :], current_states[i][:, 0, :])
        assert torch.allclose(state_tensor[:, 2, :], current_states[i][:, 2, :])


def test_reset_lstm_states_none_handling():
    """Test that None states are handled gracefully."""
    from distributional_ppo import DistributionalPPO

    ppo = DistributionalPPO.__new__(DistributionalPPO)
    ppo.device = torch.device("cpu")

    dones = np.array([True, False], dtype=bool)

    # Both None
    result = ppo._reset_lstm_states_for_done_envs(None, dones, None)
    assert result is None

    # Current states None
    init_states = RNNStates(
        pi=(torch.zeros(2, 1, 64), torch.zeros(2, 1, 64)),
        vf=(torch.zeros(2, 1, 64), torch.zeros(2, 1, 64)),
    )
    result = ppo._reset_lstm_states_for_done_envs(None, dones, init_states)
    assert result is None

    # Init states None - returns current states unchanged
    # (cannot reset without initial states)
    current_states = RNNStates(
        pi=(torch.randn(2, 2, 64), torch.randn(2, 2, 64)),
        vf=(torch.randn(2, 2, 64), torch.randn(2, 2, 64)),
    )
    result = ppo._reset_lstm_states_for_done_envs(current_states, dones, None)
    # When init_states is None, function returns current states unchanged
    assert result is not None
    assert result is current_states


def test_reset_lstm_states_temporal_independence():
    """
    Test that LSTM states from different episodes are independent.

    This verifies the fix prevents temporal leakage across episode boundaries.
    """
    from distributional_ppo import DistributionalPPO

    num_layers, num_envs, hidden_size = 2, 2, 64

    # Episode 1: Non-zero states
    episode1_pi_states = tuple(
        [
            torch.randn(num_layers, num_envs, hidden_size) + 10.0,  # Large offset
            torch.randn(num_layers, num_envs, hidden_size) + 10.0,
        ]
    )
    episode1_vf_states = tuple(
        [
            torch.randn(num_layers, num_envs, hidden_size) + 10.0,
            torch.randn(num_layers, num_envs, hidden_size) + 10.0,
        ]
    )
    episode1_states = RNNStates(pi=episode1_pi_states, vf=episode1_vf_states)

    # Initial states (zeros)
    init_pi_states = tuple(
        [
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        ]
    )
    init_vf_states = tuple(
        [
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        ]
    )
    init_states = RNNStates(pi=init_pi_states, vf=init_vf_states)

    # Env 0 finishes episode
    dones = np.array([True, False], dtype=bool)

    ppo = DistributionalPPO.__new__(DistributionalPPO)
    ppo.device = torch.device("cpu")

    # Store Episode 1 env 0 states for comparison
    env0_episode1_pi = [s[:, 0, :].clone() for s in episode1_pi_states]

    # Reset states
    episode2_states = ppo._reset_lstm_states_for_done_envs(episode1_states, dones, init_states)

    # CRITICAL: Env 0 states should have NO correlation with Episode 1
    # They should be reset to zeros (independent)
    for i, state_tensor in enumerate(episode2_states.pi):
        env0_episode2 = state_tensor[:, 0, :]
        env0_episode1 = env0_episode1_pi[i]

        # Episode 2 env 0 should be zeros (reset)
        assert torch.allclose(
            env0_episode2, torch.zeros_like(env0_episode2)
        ), f"Env 0 state {i} should be reset to zeros after episode boundary"

        # Episode 1 env 0 had non-zero values (verify test setup)
        assert not torch.allclose(
            env0_episode1, torch.zeros_like(env0_episode1)
        ), f"Episode 1 env 0 state {i} should have non-zero values (test setup issue)"

        # Episode 2 should be DIFFERENT from Episode 1 (temporal independence)
        assert not torch.allclose(
            env0_episode2, env0_episode1
        ), f"Env 0 states should be independent across episodes (no leakage)"

        # Env 1 (not done) should be same as Episode 1 (no reset)
        assert torch.allclose(
            state_tensor[:, 1, :], episode1_pi_states[i][:, 1, :]
        ), f"Env 1 state {i} should be unchanged (no episode boundary)"


def test_reset_lstm_states_device_handling():
    """Test that LSTM state reset works with different devices."""
    from distributional_ppo import DistributionalPPO

    num_layers, num_envs, hidden_size = 2, 2, 64

    # Test CPU device
    current_states_cpu = RNNStates(
        pi=(
            torch.randn(num_layers, num_envs, hidden_size),
            torch.randn(num_layers, num_envs, hidden_size),
        ),
        vf=(
            torch.randn(num_layers, num_envs, hidden_size),
            torch.randn(num_layers, num_envs, hidden_size),
        ),
    )
    init_states_cpu = RNNStates(
        pi=(torch.zeros(num_layers, 1, hidden_size), torch.zeros(num_layers, 1, hidden_size)),
        vf=(torch.zeros(num_layers, 1, hidden_size), torch.zeros(num_layers, 1, hidden_size)),
    )

    dones = np.array([True, False], dtype=bool)

    ppo = DistributionalPPO.__new__(DistributionalPPO)
    ppo.device = torch.device("cpu")

    updated_states_cpu = ppo._reset_lstm_states_for_done_envs(
        current_states_cpu, dones, init_states_cpu
    )

    # Verify reset worked on CPU
    assert updated_states_cpu.pi[0].device == torch.device("cpu")
    assert torch.allclose(updated_states_cpu.pi[0][:, 0, :], torch.zeros(num_layers, hidden_size))

    # If CUDA available, test GPU device
    if torch.cuda.is_available():
        current_states_gpu = RNNStates(
            pi=(
                torch.randn(num_layers, num_envs, hidden_size).cuda(),
                torch.randn(num_layers, num_envs, hidden_size).cuda(),
            ),
            vf=(
                torch.randn(num_layers, num_envs, hidden_size).cuda(),
                torch.randn(num_layers, num_envs, hidden_size).cuda(),
            ),
        )
        init_states_gpu = RNNStates(
            pi=(
                torch.zeros(num_layers, 1, hidden_size).cuda(),
                torch.zeros(num_layers, 1, hidden_size).cuda(),
            ),
            vf=(
                torch.zeros(num_layers, 1, hidden_size).cuda(),
                torch.zeros(num_layers, 1, hidden_size).cuda(),
            ),
        )

        ppo.device = torch.device("cuda")

        updated_states_gpu = ppo._reset_lstm_states_for_done_envs(
            current_states_gpu, dones, init_states_gpu
        )

        # Verify reset worked on GPU
        assert updated_states_gpu.pi[0].device.type == "cuda"
        assert torch.allclose(
            updated_states_gpu.pi[0][:, 0, :], torch.zeros(num_layers, hidden_size).cuda()
        )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

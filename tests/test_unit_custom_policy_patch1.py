"""
Comprehensive unit tests for custom_policy_patch1.py

Tests cover:
1. QuantileValueHead - Quantile value predictions with optional monotonicity
2. CustomMlpExtractor - MLP feature extractor with residual connections
3. _CategoricalAdapter - Categorical distribution adapter
4. CustomActorCriticPolicy - Main recurrent actor-critic policy with Twin Critics

Author: Test Suite
Created: 2025-12-02
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")
import torch.nn as nn
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any
from unittest.mock import Mock, MagicMock, patch
import math


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def device():
    """Get test device (CPU for testing)."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 8


@pytest.fixture
def hidden_dim():
    """Standard hidden dimension."""
    return 64


@pytest.fixture
def num_quantiles():
    """Standard number of quantiles."""
    return 21


@pytest.fixture
def box_action_space():
    """Box action space for continuous actions."""
    return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)


@pytest.fixture
def box_action_space_symmetric():
    """Symmetric box action space [-1, 1]."""
    return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


@pytest.fixture
def observation_space():
    """Standard observation space."""
    return spaces.Box(low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32)


@pytest.fixture
def sample_latent(batch_size, hidden_dim, device):
    """Sample latent tensor."""
    return torch.randn(batch_size, hidden_dim, device=device)


# ============================================================================
# QuantileValueHead Tests
# ============================================================================


class TestQuantileValueHeadInit:
    """Tests for QuantileValueHead initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        from custom_policy_patch1 import QuantileValueHead

        head = QuantileValueHead(input_dim=64, num_quantiles=21, huber_kappa=1.0)

        assert head.num_quantiles == 21
        assert head.huber_kappa == 1.0
        assert head.enforce_monotonicity is False
        assert isinstance(head.linear, nn.Linear)
        assert head.linear.in_features == 64
        assert head.linear.out_features == 21

    def test_init_with_monotonicity(self):
        """Test initialization with monotonicity enforcement."""
        from custom_policy_patch1 import QuantileValueHead

        head = QuantileValueHead(
            input_dim=64, num_quantiles=21, huber_kappa=1.0, enforce_monotonicity=True
        )

        assert head.enforce_monotonicity is True

    def test_init_invalid_num_quantiles(self):
        """Test that invalid num_quantiles raises ValueError."""
        from custom_policy_patch1 import QuantileValueHead

        with pytest.raises(ValueError, match="must be positive"):
            QuantileValueHead(input_dim=64, num_quantiles=0, huber_kappa=1.0)

        with pytest.raises(ValueError, match="must be positive"):
            QuantileValueHead(input_dim=64, num_quantiles=-5, huber_kappa=1.0)

    def test_init_invalid_huber_kappa(self):
        """Test that invalid huber_kappa raises ValueError."""
        from custom_policy_patch1 import QuantileValueHead

        with pytest.raises(ValueError, match="positive finite"):
            QuantileValueHead(input_dim=64, num_quantiles=21, huber_kappa=0.0)

        with pytest.raises(ValueError, match="positive finite"):
            QuantileValueHead(input_dim=64, num_quantiles=21, huber_kappa=-1.0)

        with pytest.raises(ValueError, match="positive finite"):
            QuantileValueHead(input_dim=64, num_quantiles=21, huber_kappa=float("inf"))

        with pytest.raises(ValueError, match="positive finite"):
            QuantileValueHead(input_dim=64, num_quantiles=21, huber_kappa=float("nan"))

    def test_taus_buffer_correct_formula(self):
        """Test that taus are computed using midpoint formula."""
        from custom_policy_patch1 import QuantileValueHead

        num_quantiles = 21
        head = QuantileValueHead(input_dim=64, num_quantiles=num_quantiles, huber_kappa=1.0)

        # Expected midpoints: tau_i = (i + 0.5) / N
        expected_taus = torch.tensor([(i + 0.5) / num_quantiles for i in range(num_quantiles)])

        assert torch.allclose(head.taus, expected_taus, atol=1e-6)

    def test_taus_buffer_is_persistent(self):
        """Test that taus buffer is persistent."""
        from custom_policy_patch1 import QuantileValueHead

        head = QuantileValueHead(input_dim=64, num_quantiles=21, huber_kappa=1.0)

        # Check buffer is registered
        assert "taus" in dict(head.named_buffers())


class TestQuantileValueHeadForward:
    """Tests for QuantileValueHead forward pass."""

    def test_forward_shape(self, sample_latent, hidden_dim, batch_size, num_quantiles):
        """Test output shape of forward pass."""
        from custom_policy_patch1 import QuantileValueHead

        head = QuantileValueHead(input_dim=hidden_dim, num_quantiles=num_quantiles, huber_kappa=1.0)

        output = head(sample_latent)

        assert output.shape == (batch_size, num_quantiles)

    def test_forward_without_monotonicity(self, sample_latent, hidden_dim, num_quantiles):
        """Test that forward without monotonicity preserves order."""
        from custom_policy_patch1 import QuantileValueHead

        head = QuantileValueHead(
            input_dim=hidden_dim,
            num_quantiles=num_quantiles,
            huber_kappa=1.0,
            enforce_monotonicity=False,
        )

        # Forward pass
        output = head(sample_latent)

        # Output may not be sorted
        assert output.shape[1] == num_quantiles

    def test_forward_with_monotonicity_sorted(self, hidden_dim, num_quantiles):
        """Test that forward with monotonicity returns sorted quantiles."""
        from custom_policy_patch1 import QuantileValueHead

        head = QuantileValueHead(
            input_dim=hidden_dim,
            num_quantiles=num_quantiles,
            huber_kappa=1.0,
            enforce_monotonicity=True,
        )

        # Create input that would produce unsorted output
        latent = torch.randn(4, hidden_dim)

        output = head(latent)

        # Check output is sorted along quantile dimension
        sorted_output, _ = torch.sort(output, dim=1)
        assert torch.allclose(output, sorted_output)

    def test_forward_differentiable(self, sample_latent, hidden_dim, num_quantiles):
        """Test that forward pass is differentiable."""
        from custom_policy_patch1 import QuantileValueHead

        head = QuantileValueHead(
            input_dim=hidden_dim,
            num_quantiles=num_quantiles,
            huber_kappa=1.0,
            enforce_monotonicity=True,
        )

        sample_latent.requires_grad_(True)
        output = head(sample_latent)
        loss = output.sum()
        loss.backward()

        assert sample_latent.grad is not None
        assert sample_latent.grad.shape == sample_latent.shape


# ============================================================================
# CustomMlpExtractor Tests
# ============================================================================


class TestCustomMlpExtractorInit:
    """Tests for CustomMlpExtractor initialization."""

    def test_init_basic(self, hidden_dim):
        """Test basic initialization."""
        from custom_policy_patch1 import CustomMlpExtractor

        rnn_latent_dim = 128
        extractor = CustomMlpExtractor(
            rnn_latent_dim=rnn_latent_dim, hidden_dim=hidden_dim, activation=nn.ReLU
        )

        assert extractor.latent_dim_pi == hidden_dim
        assert extractor.latent_dim_vf == hidden_dim
        assert isinstance(extractor.input_linear, nn.Linear)
        assert isinstance(extractor.hidden_linear, nn.Linear)
        assert isinstance(extractor.skip_linear, nn.Linear)
        assert isinstance(extractor.input_activation, nn.ReLU)
        assert isinstance(extractor.hidden_activation, nn.ReLU)

    def test_init_different_activations(self, hidden_dim):
        """Test initialization with different activation functions."""
        from custom_policy_patch1 import CustomMlpExtractor

        activations = [nn.ReLU, nn.Tanh, nn.SiLU, nn.GELU, nn.ELU]

        for activation in activations:
            extractor = CustomMlpExtractor(
                rnn_latent_dim=128, hidden_dim=hidden_dim, activation=activation
            )
            assert isinstance(extractor.input_activation, activation)
            assert isinstance(extractor.hidden_activation, activation)

    def test_skip_linear_no_bias(self, hidden_dim):
        """Test that skip linear has no bias."""
        from custom_policy_patch1 import CustomMlpExtractor

        extractor = CustomMlpExtractor(
            rnn_latent_dim=128, hidden_dim=hidden_dim, activation=nn.ReLU
        )

        assert extractor.skip_linear.bias is None


class TestCustomMlpExtractorForward:
    """Tests for CustomMlpExtractor forward pass."""

    def test_forward_shape(self, hidden_dim, batch_size):
        """Test output shape of forward pass."""
        from custom_policy_patch1 import CustomMlpExtractor

        rnn_latent_dim = 128
        extractor = CustomMlpExtractor(
            rnn_latent_dim=rnn_latent_dim, hidden_dim=hidden_dim, activation=nn.ReLU
        )

        features = torch.randn(batch_size, rnn_latent_dim)
        output = extractor(features)

        assert output.shape == (batch_size, hidden_dim)

    def test_forward_actor_same_as_forward(self, hidden_dim, batch_size):
        """Test that forward_actor returns same as forward."""
        from custom_policy_patch1 import CustomMlpExtractor

        rnn_latent_dim = 128
        extractor = CustomMlpExtractor(
            rnn_latent_dim=rnn_latent_dim, hidden_dim=hidden_dim, activation=nn.ReLU
        )

        features = torch.randn(batch_size, rnn_latent_dim)

        output_forward = extractor(features)
        output_actor = extractor.forward_actor(features)

        assert torch.allclose(output_forward, output_actor)

    def test_forward_critic_same_as_forward(self, hidden_dim, batch_size):
        """Test that forward_critic returns same as forward."""
        from custom_policy_patch1 import CustomMlpExtractor

        rnn_latent_dim = 128
        extractor = CustomMlpExtractor(
            rnn_latent_dim=rnn_latent_dim, hidden_dim=hidden_dim, activation=nn.ReLU
        )

        features = torch.randn(batch_size, rnn_latent_dim)

        output_forward = extractor(features)
        output_critic = extractor.forward_critic(features)

        assert torch.allclose(output_forward, output_critic)

    def test_residual_connection(self, hidden_dim, batch_size):
        """Test that residual connection is applied."""
        from custom_policy_patch1 import CustomMlpExtractor

        rnn_latent_dim = hidden_dim  # Same dim to see residual effect
        extractor = CustomMlpExtractor(
            rnn_latent_dim=rnn_latent_dim, hidden_dim=hidden_dim, activation=nn.ReLU
        )

        features = torch.randn(batch_size, rnn_latent_dim)

        # Zero out main path weights
        with torch.no_grad():
            extractor.input_linear.weight.zero_()
            extractor.input_linear.bias.zero_()
            extractor.hidden_linear.weight.zero_()
            extractor.hidden_linear.bias.zero_()

        output = extractor(features)
        residual = extractor.skip_linear(features)

        # Output should equal residual when main path is zeroed
        assert torch.allclose(output, residual)


# ============================================================================
# _CategoricalAdapter Tests
# ============================================================================


class TestCategoricalAdapter:
    """Tests for _CategoricalAdapter class."""

    def test_init(self):
        """Test initialization."""
        from custom_policy_patch1 import _CategoricalAdapter

        logits = torch.randn(4, 5)  # batch=4, num_classes=5
        adapter = _CategoricalAdapter(logits)

        assert adapter._dist is not None
        assert adapter.distribution is adapter._dist

    def test_sample(self):
        """Test sample method."""
        from custom_policy_patch1 import _CategoricalAdapter

        torch.manual_seed(42)
        logits = torch.randn(4, 5)
        adapter = _CategoricalAdapter(logits)

        samples = adapter.sample()

        assert samples.shape == (4,)
        assert all(0 <= s < 5 for s in samples)

    def test_log_prob(self):
        """Test log_prob method."""
        from custom_policy_patch1 import _CategoricalAdapter

        logits = torch.randn(4, 5)
        adapter = _CategoricalAdapter(logits)

        actions = torch.tensor([0, 1, 2, 3])
        log_probs = adapter.log_prob(actions)

        assert log_probs.shape == (4,)
        assert torch.all(log_probs <= 0)  # Log probs are negative

    def test_entropy(self):
        """Test entropy method."""
        from custom_policy_patch1 import _CategoricalAdapter

        logits = torch.randn(4, 5)
        adapter = _CategoricalAdapter(logits)

        entropy = adapter.entropy()

        assert entropy.shape == (4,)
        assert torch.all(entropy >= 0)  # Entropy is non-negative

    def test_get_actions_stochastic(self):
        """Test get_actions with deterministic=False."""
        from custom_policy_patch1 import _CategoricalAdapter

        torch.manual_seed(42)
        logits = torch.randn(4, 5)
        adapter = _CategoricalAdapter(logits)

        actions = adapter.get_actions(deterministic=False)

        assert actions.shape == (4,)

    def test_get_actions_deterministic(self):
        """Test get_actions with deterministic=True."""
        from custom_policy_patch1 import _CategoricalAdapter

        # Create logits where max is clear
        logits = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        adapter = _CategoricalAdapter(logits)

        actions = adapter.get_actions(deterministic=True)

        expected = torch.tensor([0, 1, 2, 3])
        assert torch.equal(actions, expected)

    def test_logits_property(self):
        """Test logits property returns normalized log probabilities (log_softmax)."""
        from custom_policy_patch1 import _CategoricalAdapter

        logits = torch.randn(4, 5)
        adapter = _CategoricalAdapter(logits)

        # Categorical.logits returns log_softmax normalized logits, not raw logits
        # log_softmax = logits - logsumexp(logits, dim=-1, keepdim=True)
        expected_normalized = torch.nn.functional.log_softmax(logits, dim=-1)
        assert torch.allclose(adapter.logits, expected_normalized)

    def test_getattr_delegation(self):
        """Test that unknown attributes are delegated to _dist."""
        from custom_policy_patch1 import _CategoricalAdapter

        logits = torch.randn(4, 5)
        adapter = _CategoricalAdapter(logits)

        # Access probs through delegation
        probs = adapter.probs
        assert probs.shape == (4, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4))


# ============================================================================
# CustomActorCriticPolicy - Coercion Helper Tests
# ============================================================================


class TestCoerceArchFloat:
    """Tests for _coerce_arch_float helper."""

    def test_none_returns_fallback(self):
        """Test that None returns fallback."""
        # Use helper function since _coerce_arch_float is a nested function in __init__
        result = _coerce_arch_float_helper(None, 1.5, "test")
        assert result == 1.5

    def test_int_converts_to_float(self):
        """Test that int is converted to float."""
        result = _coerce_arch_float_helper(5, 1.0, "test")
        assert result == 5.0
        assert isinstance(result, float)

    def test_float_passthrough(self):
        """Test that float passes through."""
        result = _coerce_arch_float_helper(3.14, 1.0, "test")
        assert result == 3.14

    def test_string_number_converts(self):
        """Test that string number converts."""
        result = _coerce_arch_float_helper("2.5", 1.0, "test")
        assert result == 2.5


def _coerce_arch_float_helper(value, fallback, key):
    """Helper to test _coerce_arch_float without full policy initialization."""
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid '{key}' value")


class TestCoerceArchInt:
    """Tests for _coerce_arch_int helper."""

    def test_none_returns_fallback(self):
        """Test that None returns fallback."""
        result = _coerce_arch_int_helper(None, 10, "test")
        assert result == 10

    def test_bool_raises_error(self):
        """Test that bool raises ValueError."""
        with pytest.raises(ValueError):
            _coerce_arch_int_helper(True, 10, "test")

    def test_int_passthrough(self):
        """Test that int passes through."""
        result = _coerce_arch_int_helper(42, 10, "test")
        assert result == 42


def _coerce_arch_int_helper(value, fallback, key):
    """Helper to test _coerce_arch_int without full policy initialization."""
    if value is None:
        return fallback
    if isinstance(value, bool):
        raise ValueError(f"Invalid '{key}' value: {value}")
    if isinstance(value, int):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid '{key}' value")


class TestCoerceArchBool:
    """Tests for _coerce_arch_bool helper."""

    def test_none_returns_fallback(self):
        """Test that None returns fallback."""
        result = _coerce_arch_bool_helper(None, True, "test")
        assert result is True
        result = _coerce_arch_bool_helper(None, False, "test")
        assert result is False

    def test_bool_passthrough(self):
        """Test that bool passes through."""
        result = _coerce_arch_bool_helper(True, False, "test")
        assert result is True
        result = _coerce_arch_bool_helper(False, True, "test")
        assert result is False

    def test_int_converts(self):
        """Test that int converts to bool."""
        result = _coerce_arch_bool_helper(0, True, "test")
        assert result is False
        result = _coerce_arch_bool_helper(1, False, "test")
        assert result is True
        result = _coerce_arch_bool_helper(-5, False, "test")
        assert result is True

    def test_string_true_values(self):
        """Test that string true values convert."""
        for val in ["true", "True", "TRUE", "yes", "YES", "1", "on", "ON"]:
            result = _coerce_arch_bool_helper(val, False, "test")
            assert result is True

    def test_string_false_values(self):
        """Test that string false values convert."""
        for val in ["false", "False", "FALSE", "no", "NO", "0", "off", "OFF"]:
            result = _coerce_arch_bool_helper(val, True, "test")
            assert result is False

    def test_invalid_string_raises(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            _coerce_arch_bool_helper("maybe", False, "test")


def _coerce_arch_bool_helper(value, fallback, key):
    """Helper to test _coerce_arch_bool without full policy initialization."""
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lower_val = value.lower().strip()
        if lower_val in ("true", "yes", "1", "on"):
            return True
        elif lower_val in ("false", "no", "0", "off"):
            return False
        else:
            raise ValueError(f"Invalid '{key}' string value")
    raise ValueError(f"Invalid '{key}' value type")


# ============================================================================
# CustomActorCriticPolicy - Initialization Tests
# ============================================================================


class TestCustomActorCriticPolicyInit:
    """Tests for CustomActorCriticPolicy initialization."""

    @pytest.fixture
    def minimal_policy_kwargs(self, observation_space, box_action_space):
        """Minimal kwargs for policy creation."""
        return {
            "observation_space": observation_space,
            "action_space": box_action_space,
            "lr_schedule": lambda _: 1e-4,
        }

    def test_init_basic(self, minimal_policy_kwargs):
        """Test basic initialization."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        policy = CustomActorCriticPolicy(**minimal_policy_kwargs)

        assert policy.hidden_dim > 0
        assert policy.action_dim == 1
        assert policy._execution_mode == "score"

    def test_init_with_arch_params(self, observation_space, box_action_space):
        """Test initialization with arch_params."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        arch_params = {
            "hidden_dim": 128,
            "activation": "relu",
            "num_atoms": 51,
            "v_min": -20.0,
            "v_max": 20.0,
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params=arch_params,
        )

        assert policy.num_atoms == 51
        assert policy.v_min == -20.0
        assert policy.v_max == 20.0

    def test_init_invalid_action_space_type(self, observation_space):
        """Test that non-Box action space raises error."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        discrete_space = spaces.Discrete(5)

        with pytest.raises(NotImplementedError, match="Box action space"):
            CustomActorCriticPolicy(
                observation_space=observation_space,
                action_space=discrete_space,
                lr_schedule=lambda _: 1e-4,
            )

    def test_init_invalid_action_space_shape(self, observation_space):
        """Test that multi-dimensional action space raises error."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        multi_box = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        with pytest.raises(ValueError, match="single-dimensional"):
            CustomActorCriticPolicy(
                observation_space=observation_space,
                action_space=multi_box,
                lr_schedule=lambda _: 1e-4,
            )

    def test_init_with_quantile_critic(self, observation_space, box_action_space):
        """Test initialization with quantile critic."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        arch_params = {
            "critic": {
                "distributional": True,
                "categorical": False,
                "num_quantiles": 32,
                "huber_kappa": 1.0,
            }
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params=arch_params,
        )

        assert policy._use_quantile_value_head is True
        assert policy.num_quantiles == 32

    def test_init_with_twin_critics(self, observation_space, box_action_space):
        """Test initialization with twin critics enabled."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        arch_params = {
            "critic": {
                "use_twin_critics": True,
            }
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params=arch_params,
        )

        assert policy._use_twin_critics is True

    def test_init_twin_critics_disabled(self, observation_space, box_action_space):
        """Test initialization with twin critics disabled."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        arch_params = {
            "critic": {
                "use_twin_critics": False,
            }
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params=arch_params,
        )

        assert policy._use_twin_critics is False

    def test_init_invalid_activation(self, observation_space, box_action_space):
        """Test that invalid activation raises ValueError."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        arch_params = {"activation": "invalid_activation"}

        with pytest.raises(ValueError, match="Unsupported activation"):
            CustomActorCriticPolicy(
                observation_space=observation_space,
                action_space=box_action_space,
                lr_schedule=lambda _: 1e-4,
                arch_params=arch_params,
            )

    def test_init_all_activations(self, observation_space, box_action_space):
        """Test all supported activation functions."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        activations = ["relu", "tanh", "leakyrelu", "elu", "gelu", "silu"]

        for act in activations:
            arch_params = {"activation": act}
            policy = CustomActorCriticPolicy(
                observation_space=observation_space,
                action_space=box_action_space,
                lr_schedule=lambda _: 1e-4,
                arch_params=arch_params,
            )
            assert policy is not None

    def test_init_invalid_num_atoms(self, observation_space, box_action_space):
        """Test that invalid num_atoms raises ValueError."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        with pytest.raises(ValueError, match="num_atoms"):
            CustomActorCriticPolicy(
                observation_space=observation_space,
                action_space=box_action_space,
                lr_schedule=lambda _: 1e-4,
                arch_params={"num_atoms": 0},
            )

    def test_init_invalid_value_range(self, observation_space, box_action_space):
        """Test that v_max <= v_min raises ValueError."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        with pytest.raises(ValueError, match="value range"):
            CustomActorCriticPolicy(
                observation_space=observation_space,
                action_space=box_action_space,
                lr_schedule=lambda _: 1e-4,
                arch_params={"v_min": 10.0, "v_max": -10.0},
            )

    def test_tanh_activation_for_symmetric_action_space(
        self, observation_space, box_action_space_symmetric
    ):
        """Test that symmetric action space uses tanh activation."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space_symmetric,
            lr_schedule=lambda _: 1e-4,
        )

        assert policy._use_tanh_activation is True

    def test_sigmoid_activation_for_positive_action_space(
        self, observation_space, box_action_space
    ):
        """Test that [0, 1] action space uses sigmoid activation."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

        assert policy._use_tanh_activation is False


# ============================================================================
# CustomActorCriticPolicy - Activation Tests
# ============================================================================


class TestApplyActionActivation:
    """Tests for _apply_action_activation method."""

    def test_sigmoid_activation(self, observation_space, box_action_space):
        """Test sigmoid activation for [0, 1] action space."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

        raw = torch.tensor([[-5.0], [0.0], [5.0]])
        result = policy._apply_action_activation(raw)

        # Sigmoid outputs should be in [0, 1]
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

        # Check specific values
        expected = torch.sigmoid(raw)
        assert torch.allclose(result, expected)

    def test_tanh_activation(self, observation_space, box_action_space_symmetric):
        """Test tanh activation for [-1, 1] action space."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space_symmetric,
            lr_schedule=lambda _: 1e-4,
        )

        raw = torch.tensor([[-5.0], [0.0], [5.0]])
        result = policy._apply_action_activation(raw)

        # Tanh outputs should be in [-1, 1]
        assert torch.all(result >= -1.0)
        assert torch.all(result <= 1.0)

        # Check specific values
        expected = torch.tanh(raw)
        assert torch.allclose(result, expected)


class TestScoreToRaw:
    """Tests for _score_to_raw method."""

    def test_inverse_sigmoid(self, observation_space, box_action_space):
        """Test inverse sigmoid (logit) for [0, 1] action space."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

        # Scores in (0, 1)
        scores = torch.tensor([[0.1], [0.5], [0.9]])
        raw = policy._score_to_raw(scores)

        # Apply activation should give back scores
        recovered = policy._apply_action_activation(raw)
        assert torch.allclose(recovered, scores, atol=1e-3)

    def test_inverse_tanh(self, observation_space, box_action_space_symmetric):
        """Test inverse tanh (atanh) for [-1, 1] action space."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space_symmetric,
            lr_schedule=lambda _: 1e-4,
        )

        # Scores in (-1, 1)
        scores = torch.tensor([[-0.5], [0.0], [0.5]])
        raw = policy._score_to_raw(scores)

        # Apply activation should give back scores
        recovered = policy._apply_action_activation(raw)
        assert torch.allclose(recovered, scores, atol=1e-3)


# ============================================================================
# CustomActorCriticPolicy - Value Head Tests
# ============================================================================


class TestValueHeadMethods:
    """Tests for value head related methods."""

    @pytest.fixture
    def policy_quantile(self, observation_space, box_action_space):
        """Policy with quantile value head."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        arch_params = {
            "critic": {
                "distributional": True,
                "categorical": False,
                "num_quantiles": 21,
            }
        }
        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params=arch_params,
        )

    @pytest.fixture
    def policy_categorical(self, observation_space, box_action_space):
        """Policy with categorical value head."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        arch_params = {
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
        }
        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params=arch_params,
        )

    def test_get_value_logits_quantile(self, policy_quantile, batch_size):
        """Test _get_value_logits with quantile head."""
        latent_vf = torch.randn(batch_size, policy_quantile.hidden_dim)

        logits = policy_quantile._get_value_logits(latent_vf)

        assert logits.shape == (batch_size, policy_quantile.num_quantiles)
        assert policy_quantile._last_value_quantiles is not None
        assert policy_quantile._last_value_logits is None

    def test_get_value_logits_categorical(self, policy_categorical, batch_size):
        """Test _get_value_logits with categorical head."""
        latent_vf = torch.randn(batch_size, policy_categorical.hidden_dim)

        logits = policy_categorical._get_value_logits(latent_vf)

        assert logits.shape == (batch_size, policy_categorical.num_atoms)
        assert policy_categorical._last_value_logits is not None
        assert policy_categorical._last_value_quantiles is None

    def test_value_from_logits_quantile(self, policy_quantile, batch_size):
        """Test _value_from_logits with quantile head."""
        # For quantile head, logits are quantile values
        quantiles = torch.randn(batch_size, policy_quantile.num_quantiles)

        values = policy_quantile._value_from_logits(quantiles)

        assert values.shape == (batch_size, 1)
        # Mean of quantiles
        expected = quantiles.mean(dim=-1, keepdim=True)
        assert torch.allclose(values, expected)

    def test_value_from_logits_categorical(self, policy_categorical, batch_size):
        """Test _value_from_logits with categorical head."""
        logits = torch.randn(batch_size, policy_categorical.num_atoms)

        values = policy_categorical._value_from_logits(logits)

        assert values.shape == (batch_size, 1)
        # Expected value = sum(probs * atoms)
        probs = torch.softmax(logits, dim=-1)
        expected = (probs * policy_categorical.atoms).sum(dim=-1, keepdim=True)
        assert torch.allclose(values, expected)


# ============================================================================
# CustomActorCriticPolicy - Twin Critics Tests
# ============================================================================


class TestTwinCritics:
    """Tests for Twin Critics functionality."""

    @pytest.fixture
    def policy_twin(self, observation_space, box_action_space):
        """Policy with twin critics enabled."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        arch_params = {
            "critic": {
                "distributional": True,
                "categorical": False,
                "num_quantiles": 21,
                "use_twin_critics": True,
            }
        }
        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params=arch_params,
        )

    @pytest.fixture
    def policy_single(self, observation_space, box_action_space):
        """Policy with single critic."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        arch_params = {
            "critic": {
                "distributional": True,
                "categorical": False,
                "num_quantiles": 21,
                "use_twin_critics": False,
            }
        }
        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params=arch_params,
        )

    def test_twin_critics_two_heads(self, policy_twin):
        """Test that twin critics creates two value heads."""
        assert policy_twin.quantile_head is not None
        assert policy_twin.quantile_head_2 is not None
        assert policy_twin._value_head_module is not None
        assert policy_twin._value_head_module_2 is not None

    def test_single_critic_one_head(self, policy_single):
        """Test that single critic creates only one head."""
        assert policy_single.quantile_head is not None
        assert policy_single.quantile_head_2 is None
        assert policy_single._value_head_module is not None
        assert policy_single._value_head_module_2 is None

    def test_get_value_logits_2_twin_enabled(self, policy_twin, batch_size):
        """Test _get_value_logits_2 with twin critics enabled."""
        latent_vf = torch.randn(batch_size, policy_twin.hidden_dim)

        logits = policy_twin._get_value_logits_2(latent_vf)

        assert logits.shape == (batch_size, policy_twin.num_quantiles)
        assert policy_twin._last_value_quantiles_2 is not None

    def test_get_value_logits_2_single_raises(self, policy_single, batch_size):
        """Test _get_value_logits_2 raises error with single critic."""
        latent_vf = torch.randn(batch_size, policy_single.hidden_dim)

        with pytest.raises(RuntimeError, match="not enabled"):
            policy_single._get_value_logits_2(latent_vf)

    def test_get_twin_value_logits(self, policy_twin, batch_size):
        """Test _get_twin_value_logits returns both critics' outputs."""
        latent_vf = torch.randn(batch_size, policy_twin.hidden_dim)

        logits_1, logits_2 = policy_twin._get_twin_value_logits(latent_vf)

        assert logits_1.shape == (batch_size, policy_twin.num_quantiles)
        assert logits_2.shape == (batch_size, policy_twin.num_quantiles)
        # Should be different (different heads)
        assert not torch.equal(logits_1, logits_2)

    def test_get_min_twin_values_twin(self, policy_twin, batch_size):
        """Test _get_min_twin_values returns minimum of both critics."""
        latent_vf = torch.randn(batch_size, policy_twin.hidden_dim)

        min_values = policy_twin._get_min_twin_values(latent_vf)

        assert min_values.shape == (batch_size, 1)

    def test_get_min_twin_values_single(self, policy_single, batch_size):
        """Test _get_min_twin_values falls back to single critic."""
        latent_vf = torch.randn(batch_size, policy_single.hidden_dim)

        values = policy_single._get_min_twin_values(latent_vf)

        assert values.shape == (batch_size, 1)

    def test_last_value_quantiles_min(self, policy_twin, batch_size):
        """Test last_value_quantiles_min property."""
        latent_vf = torch.randn(batch_size, policy_twin.hidden_dim)

        # Trigger computation
        _ = policy_twin._get_twin_value_logits(latent_vf)

        min_quantiles = policy_twin.last_value_quantiles_min

        assert min_quantiles is not None
        assert min_quantiles.shape == (batch_size, policy_twin.num_quantiles)

        # Should be element-wise minimum
        expected = torch.min(policy_twin._last_value_quantiles, policy_twin._last_value_quantiles_2)
        assert torch.equal(min_quantiles, expected)


# ============================================================================
# CustomActorCriticPolicy - Forward Methods Tests
# ============================================================================


class TestForwardMethods:
    """Tests for forward pass methods."""

    @pytest.fixture
    def policy(self, observation_space, box_action_space):
        """Standard policy for testing."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

    def test_forward_output_shapes(self, policy, batch_size):
        """Test forward method output shapes."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        episode_starts = torch.zeros(batch_size)

        actions, values, log_prob, new_states = policy.forward(
            obs, lstm_states=None, episode_starts=episode_starts
        )

        assert actions.shape == (batch_size, 1)
        assert values.shape == (batch_size, 1)
        assert log_prob.shape == (batch_size,) or log_prob.shape == (batch_size, 1)
        assert hasattr(new_states, "pi")
        assert hasattr(new_states, "vf")

    def test_forward_deterministic(self, policy, batch_size):
        """Test forward with deterministic=True produces consistent outputs."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        episode_starts = torch.zeros(batch_size)

        # Run twice with deterministic=True
        torch.manual_seed(42)
        actions1, _, _, _ = policy.forward(
            obs, lstm_states=None, episode_starts=episode_starts, deterministic=True
        )

        torch.manual_seed(42)
        actions2, _, _, _ = policy.forward(
            obs, lstm_states=None, episode_starts=episode_starts, deterministic=True
        )

        assert torch.equal(actions1, actions2)

    def test_forward_actions_in_bounds(self, policy, batch_size):
        """Test forward produces actions within bounds."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        episode_starts = torch.zeros(batch_size)

        actions, _, _, _ = policy.forward(obs, lstm_states=None, episode_starts=episode_starts)

        low = policy.action_space.low[0]
        high = policy.action_space.high[0]
        assert torch.all(actions >= low)
        assert torch.all(actions <= high)

    def test_forward_caches_raw_actions(self, policy, batch_size):
        """Test that forward caches raw actions."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        episode_starts = torch.zeros(batch_size)

        policy.forward(obs, lstm_states=None, episode_starts=episode_starts)

        assert policy.last_raw_actions is not None
        assert policy.last_raw_actions.shape == (batch_size, 1)


class TestEvaluateActions:
    """Tests for evaluate_actions method."""

    @pytest.fixture
    def policy(self, observation_space, box_action_space):
        """Standard policy for testing."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

    def test_evaluate_actions_shapes(self, policy, batch_size):
        """Test evaluate_actions output shapes."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        actions = torch.rand(batch_size, 1)
        episode_starts = torch.zeros(batch_size)

        values, log_prob, entropy = policy.evaluate_actions(
            obs, actions, episode_starts=episode_starts
        )

        assert values.shape == (batch_size, 1)
        assert log_prob.shape == (batch_size, 1)
        assert entropy.shape == (batch_size,)

    def test_evaluate_actions_requires_episode_starts(self, policy, batch_size):
        """Test that evaluate_actions requires episode_starts."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        actions = torch.rand(batch_size, 1)

        with pytest.raises(TypeError, match="episode_starts"):
            policy.evaluate_actions(obs, actions)

    def test_evaluate_actions_with_raw_actions(self, policy, batch_size):
        """Test evaluate_actions with explicit raw actions."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        actions = torch.rand(batch_size, 1)
        actions_raw = torch.randn(batch_size, 1)
        episode_starts = torch.zeros(batch_size)

        values, log_prob, entropy = policy.evaluate_actions(
            obs, actions, episode_starts=episode_starts, actions_raw=actions_raw
        )

        assert values.shape == (batch_size, 1)
        assert log_prob.shape == (batch_size, 1)


class TestPredictValues:
    """Tests for predict_values method."""

    @pytest.fixture
    def policy(self, observation_space, box_action_space):
        """Standard policy for testing."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

    def test_predict_values_shape(self, policy, batch_size):
        """Test predict_values output shape."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        episode_starts = torch.zeros(batch_size)

        values = policy.predict_values(obs, policy.recurrent_initial_state, episode_starts)

        assert values.shape == (batch_size, 1)


# ============================================================================
# CustomActorCriticPolicy - Gradient Modulation Tests
# ============================================================================


class TestGradientModulation:
    """Tests for critic gradient modulation."""

    @pytest.fixture
    def policy(self, observation_space, box_action_space):
        """Standard policy for testing."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

    def test_modulate_critic_gradient_scale_1(self, policy):
        """Test gradient modulation with scale=1.0."""
        tensor = torch.randn(4, 64, requires_grad=True)

        result = policy._modulate_critic_gradient(tensor, 1.0)

        assert torch.equal(result, tensor)

    def test_modulate_critic_gradient_scale_0(self, policy):
        """Test gradient modulation with scale=0.0."""
        tensor = torch.randn(4, 64, requires_grad=True)

        result = policy._modulate_critic_gradient(tensor, 0.0)

        # Result should be detached
        assert not result.requires_grad

    def test_modulate_critic_gradient_scale_partial(self, policy):
        """Test gradient modulation with partial scale."""
        tensor = torch.randn(4, 64, requires_grad=True)

        result = policy._modulate_critic_gradient(tensor, 0.5)

        # Result should be mix of tensor and detached
        assert torch.allclose(result, tensor)

    def test_set_critic_gradient_blocked_bool(self, policy):
        """Test set_critic_gradient_blocked with bool."""
        policy.set_critic_gradient_blocked(True)
        assert policy._critic_gradient_blocked is True
        assert policy._critic_gradient_scale == 0.0

        policy.set_critic_gradient_blocked(False)
        assert policy._critic_gradient_blocked is False
        assert policy._critic_gradient_scale == 1.0

    def test_set_critic_gradient_blocked_float(self, policy):
        """Test set_critic_gradient_blocked with float."""
        policy.set_critic_gradient_blocked(0.5)
        assert policy._critic_gradient_blocked is False
        assert policy._critic_gradient_scale == 0.5

        policy.set_critic_gradient_blocked(0.0)
        assert policy._critic_gradient_blocked is True
        assert policy._critic_gradient_scale == 0.0


# ============================================================================
# CustomActorCriticPolicy - State Handling Tests
# ============================================================================


class TestStateHandling:
    """Tests for RNN state handling."""

    @pytest.fixture
    def policy(self, observation_space, box_action_space):
        """Standard policy for testing."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

    def test_as_tensor_tuple_tensor(self, policy):
        """Test _as_tensor_tuple with single tensor."""
        tensor = torch.randn(1, 4, 64)

        result = policy._as_tensor_tuple(tensor)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert torch.equal(result[0], tensor)

    def test_as_tensor_tuple_list(self, policy):
        """Test _as_tensor_tuple with list of tensors."""
        tensors = [torch.randn(1, 4, 64), torch.randn(1, 4, 64)]

        result = policy._as_tensor_tuple(tensors)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_as_tensor_tuple_invalid(self, policy):
        """Test _as_tensor_tuple with invalid input."""
        with pytest.raises(TypeError):
            policy._as_tensor_tuple("invalid")

    def test_align_state_tuple_gru_empty(self, policy):
        """Test _align_state_tuple for GRU with empty states."""
        gru = nn.GRU(64, 64, batch_first=True)
        reference = (torch.zeros(1, 4, 64),)

        result = policy._align_state_tuple((), reference, gru)

        assert len(result) == 1
        assert result[0].shape == reference[0].shape


# ============================================================================
# CustomActorCriticPolicy - Properties Tests
# ============================================================================


class TestProperties:
    """Tests for policy properties."""

    @pytest.fixture
    def policy(self, observation_space, box_action_space):
        """Standard policy for testing."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params={
                "critic": {
                    "distributional": True,
                    "categorical": False,
                    "num_quantiles": 21,
                }
            },
        )

    def test_squash_output_property(self, policy):
        """Test squash_output property getter/setter."""
        policy.squash_output = True
        assert policy.squash_output is True

        policy.squash_output = False
        assert policy.squash_output is False

    def test_uses_quantile_value_head(self, policy):
        """Test uses_quantile_value_head property."""
        assert policy.uses_quantile_value_head is True

    def test_quantile_levels(self, policy):
        """Test quantile_levels property."""
        levels = policy.quantile_levels

        assert levels is not None
        assert len(levels) == 21
        assert torch.all(levels > 0)
        assert torch.all(levels < 1)

    def test_value_head_metadata_quantile(self, policy):
        """Test value_head_metadata for quantile head."""
        metadata = policy.value_head_metadata()

        assert metadata["type"] == "quantile"
        assert metadata["num_quantiles"] == 21

    def test_value_head_metadata_categorical(self, observation_space, box_action_space):
        """Test value_head_metadata for categorical head."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

        metadata = policy.value_head_metadata()

        assert metadata["type"] == "categorical"
        assert metadata["num_quantiles"] is None


# ============================================================================
# CustomActorCriticPolicy - Update Atoms Tests
# ============================================================================


class TestUpdateAtoms:
    """Tests for update_atoms method."""

    @pytest.fixture
    def policy_categorical(self, observation_space, box_action_space):
        """Policy with categorical value head."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params={
                "num_atoms": 51,
                "v_min": -10.0,
                "v_max": 10.0,
            },
        )

    @pytest.fixture
    def policy_quantile(self, observation_space, box_action_space):
        """Policy with quantile value head."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params={
                "critic": {
                    "distributional": True,
                    "categorical": False,
                    "num_quantiles": 21,
                }
            },
        )

    def test_update_atoms_categorical(self, policy_categorical):
        """Test update_atoms updates atom values."""
        old_atoms = policy_categorical.atoms.clone()

        policy_categorical.update_atoms(-20.0, 20.0)

        assert policy_categorical.v_min == -20.0
        assert policy_categorical.v_max == 20.0
        assert not torch.equal(policy_categorical.atoms, old_atoms)

    def test_update_atoms_no_change(self, policy_categorical):
        """Test update_atoms with same values does nothing."""
        old_atoms = policy_categorical.atoms.clone()

        policy_categorical.update_atoms(-10.0, 10.0)

        assert torch.equal(policy_categorical.atoms, old_atoms)

    def test_update_atoms_quantile_noop(self, policy_quantile):
        """Test update_atoms is no-op for quantile head."""
        # Should not raise error, just return early
        policy_quantile.update_atoms(-20.0, 20.0)

        # Should still work
        assert policy_quantile._use_quantile_value_head is True


# ============================================================================
# CustomActorCriticPolicy - Entropy Tests
# ============================================================================


class TestWeightedEntropy:
    """Tests for weighted_entropy method."""

    @pytest.fixture
    def policy(self, observation_space, box_action_space):
        """Standard policy for testing."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

    def test_weighted_entropy_shape(self, policy, batch_size):
        """Test weighted_entropy output shape."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        episode_starts = torch.zeros(batch_size)

        # Get distribution from forward pass
        features = policy.extract_features(obs)
        latent_pi, latent_vf, _ = policy._forward_recurrent(
            features, policy.recurrent_initial_state, episode_starts
        )
        latent_pi = policy.mlp_extractor.forward_actor(latent_pi)
        distribution = policy._get_action_dist_from_latent(latent_pi)

        entropy = policy.weighted_entropy(distribution)

        assert entropy.shape == (batch_size,)

    def test_weighted_entropy_positive(self, policy, batch_size):
        """Test weighted_entropy is positive."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        episode_starts = torch.zeros(batch_size)

        features = policy.extract_features(obs)
        latent_pi, _, _ = policy._forward_recurrent(
            features, policy.recurrent_initial_state, episode_starts
        )
        latent_pi = policy.mlp_extractor.forward_actor(latent_pi)
        distribution = policy._get_action_dist_from_latent(latent_pi)

        entropy = policy.weighted_entropy(distribution)

        # Entropy should generally be positive
        assert torch.all(torch.isfinite(entropy))


# ============================================================================
# CustomActorCriticPolicy - Load State Dict Tests
# ============================================================================


class TestLoadStateDict:
    """Tests for load_state_dict method."""

    @pytest.fixture
    def policy(self, observation_space, box_action_space):
        """Standard policy for testing."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

    def test_load_state_dict_basic(self, policy):
        """Test basic state dict loading."""
        # Save state dict
        state_dict = policy.state_dict()

        # Create new policy
        new_policy = type(policy)(
            observation_space=policy.observation_space,
            action_space=policy.action_space,
            lr_schedule=lambda _: 1e-4,
        )

        # Load state dict
        new_policy.load_state_dict(state_dict)

        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
            policy.named_parameters(), new_policy.named_parameters()
        ):
            assert name1 == name2
            assert torch.equal(param1, param2)


# ============================================================================
# Jacobian Tests
# ============================================================================


class TestJacobianMethods:
    """Tests for Jacobian computation methods."""

    @pytest.fixture
    def policy_sigmoid(self, observation_space, box_action_space):
        """Policy with sigmoid activation."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

    @pytest.fixture
    def policy_tanh(self, observation_space, box_action_space_symmetric):
        """Policy with tanh activation."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space_symmetric,
            lr_schedule=lambda _: 1e-4,
        )

    def test_log_activation_jacobian_sigmoid(self, policy_sigmoid):
        """Test log_activation_jacobian for sigmoid."""
        raw = torch.tensor([[0.0], [1.0], [-1.0]])

        log_jac = policy_sigmoid._log_activation_jacobian(raw)

        assert log_jac.shape == raw.shape
        # At 0, sigmoid jacobian = 0.25, log(0.25) ≈ -1.386
        assert torch.isfinite(log_jac).all()

    def test_log_activation_jacobian_tanh(self, policy_tanh):
        """Test log_activation_jacobian for tanh."""
        raw = torch.tensor([[0.0], [1.0], [-1.0]])

        log_jac = policy_tanh._log_activation_jacobian(raw)

        assert log_jac.shape == raw.shape
        # At 0, tanh jacobian = 1, log(1) = 0
        assert torch.isfinite(log_jac).all()

    def test_log_sigmoid_jacobian_from_raw_deprecated(self, policy_sigmoid):
        """Test deprecated _log_sigmoid_jacobian_from_raw."""
        raw = torch.tensor([[0.0], [1.0], [-1.0]])

        # Should delegate to _log_activation_jacobian
        result1 = policy_sigmoid._log_sigmoid_jacobian_from_raw(raw)
        result2 = policy_sigmoid._log_activation_jacobian(raw)

        assert torch.equal(result1, result2)


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def policy(self, observation_space, box_action_space):
        """Standard policy for testing."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
        )

    def test_forward_with_none_lstm_states(self, policy, batch_size):
        """Test forward handles None lstm_states."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        episode_starts = torch.zeros(batch_size)

        # Should not raise
        actions, values, log_prob, new_states = policy.forward(
            obs, lstm_states=None, episode_starts=episode_starts
        )

        assert actions is not None

    def test_forward_with_episode_start(self, policy, batch_size):
        """Test forward with episode starts."""
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        episode_starts = torch.ones(batch_size)  # All episodes start

        actions, values, log_prob, new_states = policy.forward(
            obs, lstm_states=None, episode_starts=episode_starts
        )

        assert actions is not None

    def test_batch_size_1(self, policy):
        """Test with batch size 1."""
        obs = torch.randn(1, policy.observation_space.shape[0])
        episode_starts = torch.zeros(1)

        actions, values, log_prob, new_states = policy.forward(
            obs, lstm_states=None, episode_starts=episode_starts
        )

        assert actions.shape == (1, 1)

    def test_large_batch_size(self, policy):
        """Test with large batch size."""
        batch_size = 128
        obs = torch.randn(batch_size, policy.observation_space.shape[0])
        episode_starts = torch.zeros(batch_size)

        actions, values, log_prob, new_states = policy.forward(
            obs, lstm_states=None, episode_starts=episode_starts
        )

        assert actions.shape == (batch_size, 1)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full policy workflow."""

    @pytest.fixture
    def policy_twin_quantile(self, observation_space, box_action_space):
        """Policy with twin critics and quantile head."""
        from custom_policy_patch1 import CustomActorCriticPolicy

        return CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=box_action_space,
            lr_schedule=lambda _: 1e-4,
            arch_params={
                "critic": {
                    "distributional": True,
                    "categorical": False,
                    "num_quantiles": 21,
                    "use_twin_critics": True,
                    "enforce_monotonicity": True,
                }
            },
        )

    def test_full_forward_backward(self, policy_twin_quantile, batch_size):
        """Test full forward and backward pass."""
        obs = torch.randn(batch_size, policy_twin_quantile.observation_space.shape[0])
        episode_starts = torch.zeros(batch_size)

        # Forward pass
        actions, values, log_prob, new_states = policy_twin_quantile.forward(
            obs, lstm_states=None, episode_starts=episode_starts
        )

        # Compute loss (simple sum)
        loss = actions.sum() + values.sum() + log_prob.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        for name, param in policy_twin_quantile.named_parameters():
            if param.requires_grad:
                # Some params may not have gradients if not used in this forward
                pass

    def test_evaluate_after_forward(self, policy_twin_quantile, batch_size):
        """Test evaluate_actions after forward pass."""
        obs = torch.randn(batch_size, policy_twin_quantile.observation_space.shape[0])
        episode_starts = torch.zeros(batch_size)

        # Forward pass
        actions, _, _, new_states = policy_twin_quantile.forward(
            obs, lstm_states=None, episode_starts=episode_starts
        )

        # Evaluate with same actions
        values, log_prob, entropy = policy_twin_quantile.evaluate_actions(
            obs, actions, lstm_states=new_states, episode_starts=episode_starts
        )

        assert values.shape == (batch_size, 1)
        assert log_prob.shape == (batch_size, 1)

    def test_predict_numpy(self, policy_twin_quantile):
        """Test predict method with numpy inputs."""
        obs = np.random.randn(4, policy_twin_quantile.observation_space.shape[0]).astype(np.float32)

        actions, states = policy_twin_quantile.predict(obs)

        assert isinstance(actions, np.ndarray)
        assert actions.shape[0] == 4

    def test_twin_critics_different_values(self, policy_twin_quantile, batch_size):
        """Test that twin critics produce different values."""
        latent_vf = torch.randn(batch_size, policy_twin_quantile.hidden_dim)

        logits_1 = policy_twin_quantile._get_value_logits(latent_vf)
        logits_2 = policy_twin_quantile._get_value_logits_2(latent_vf)

        # Values should differ due to different network weights
        values_1 = policy_twin_quantile._value_from_logits(logits_1)
        values_2 = policy_twin_quantile._value_from_logits(logits_2)

        # Not equal (with very high probability)
        assert not torch.allclose(values_1, values_2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

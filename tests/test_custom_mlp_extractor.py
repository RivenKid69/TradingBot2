import sys
import types
from typing import Callable, Tuple

import torch
from torch import nn


def _install_policy_stubs() -> None:
    if 'sb3_contrib.common.recurrent.policies' not in sys.modules:
        sb3_contrib = types.ModuleType('sb3_contrib')
        sb3_contrib.__path__ = []  # type: ignore[attr-defined]
        sb3_contrib_common = types.ModuleType('sb3_contrib.common')
        sb3_contrib_common.__path__ = []  # type: ignore[attr-defined]
        sb3_contrib_recurrent = types.ModuleType('sb3_contrib.common.recurrent')
        sb3_contrib_recurrent.__path__ = []  # type: ignore[attr-defined]
        sb3_contrib_policies = types.ModuleType('sb3_contrib.common.recurrent.policies')
        sb3_contrib_type_aliases = types.ModuleType('sb3_contrib.common.recurrent.type_aliases')

        class _DummyRecurrentPolicy(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__()

        sb3_contrib_policies.RecurrentActorCriticPolicy = _DummyRecurrentPolicy
        sb3_contrib_type_aliases.RNNStates = Tuple[torch.Tensor, ...]

        sys.modules['sb3_contrib'] = sb3_contrib
        sb3_contrib.common = sb3_contrib_common
        sb3_contrib_common.recurrent = sb3_contrib_recurrent
        sb3_contrib_recurrent.policies = sb3_contrib_policies
        sb3_contrib_recurrent.type_aliases = sb3_contrib_type_aliases

        sys.modules['sb3_contrib.common'] = sb3_contrib_common
        sys.modules['sb3_contrib.common.recurrent'] = sb3_contrib_recurrent
        sys.modules['sb3_contrib.common.recurrent.policies'] = sb3_contrib_policies
        sys.modules['sb3_contrib.common.recurrent.type_aliases'] = sb3_contrib_type_aliases

    if 'stable_baselines3.common.policies' not in sys.modules:
        sb3 = types.ModuleType('stable_baselines3')
        sb3.__path__ = []  # type: ignore[attr-defined]
        sb3_common = types.ModuleType('stable_baselines3.common')
        sb3_common.__path__ = []  # type: ignore[attr-defined]
        sb3_policies = types.ModuleType('stable_baselines3.common.policies')
        sb3_type_aliases = types.ModuleType('stable_baselines3.common.type_aliases')
        sb3_utils = types.ModuleType('stable_baselines3.common.utils')

        class _DummyActorCriticPolicy(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__()

        def _zip_strict(*iterables):
            return zip(*iterables)

        sb3_policies.ActorCriticPolicy = _DummyActorCriticPolicy
        sb3_type_aliases.Schedule = Callable[[float], float]
        sb3_utils.zip_strict = _zip_strict

        sb3.common = sb3_common
        sb3_common.policies = sb3_policies
        sb3_common.type_aliases = sb3_type_aliases
        sb3_common.utils = sb3_utils

        sys.modules['stable_baselines3'] = sb3
        sys.modules['stable_baselines3.common'] = sb3_common
        sys.modules['stable_baselines3.common.policies'] = sb3_policies
        sys.modules['stable_baselines3.common.type_aliases'] = sb3_type_aliases
        sys.modules['stable_baselines3.common.utils'] = sb3_utils


_install_policy_stubs()

from custom_policy_patch1 import CustomMlpExtractor


def _assign_linear(linear: nn.Linear, value: float) -> None:
    with torch.no_grad():
        linear.weight.fill_(value)
        if linear.bias is not None:
            linear.bias.fill_(value)


def test_default_activation_preserves_gradients_for_negative_inputs() -> None:
    extractor = CustomMlpExtractor(rnn_latent_dim=4, hidden_dim=3, activation=nn.SiLU)
    _assign_linear(extractor.input_linear, 0.5)
    _assign_linear(extractor.hidden_linear, 0.25)
    _assign_linear(extractor.skip_linear, 0.1)

    features = torch.full((2, 4), -1.0)
    output = extractor.forward_critic(features).sum()
    output.backward()

    assert extractor.input_linear.weight.grad is not None
    assert torch.count_nonzero(extractor.input_linear.weight.grad) > 0


def test_skip_connection_keeps_signal_with_relu_activation() -> None:
    extractor = CustomMlpExtractor(rnn_latent_dim=3, hidden_dim=2, activation=nn.ReLU)

    _assign_linear(extractor.input_linear, -0.5)
    _assign_linear(extractor.hidden_linear, 0.75)
    with torch.no_grad():
        extractor.skip_linear.weight.zero_()
        extractor.skip_linear.weight[0, 0] = 0.3
        extractor.skip_linear.weight[1, 1] = -0.2

    negative_batch = torch.tensor(
        [[-1.0, -1.5, -2.0], [-1.05, -1.45, -1.95]], dtype=torch.float32
    )

    # При заданных весах нелинейная ветка ReLU гасится, и сигнал идёт через skip-путь.
    outputs = extractor.forward_critic(negative_batch)
    assert not torch.allclose(outputs[0], outputs[1])

    outputs.sum().backward()
    assert extractor.skip_linear.weight.grad is not None
    assert torch.count_nonzero(extractor.skip_linear.weight.grad) > 0

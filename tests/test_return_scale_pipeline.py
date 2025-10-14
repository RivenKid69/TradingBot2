import inspect

import pytest
import torch
import trading_patchnew

distributional_ppo = pytest.importorskip(
    "distributional_ppo", reason="distributional PPO module requires optional sb3_contrib dependency"
)
DistributionalPPO = distributional_ppo.DistributionalPPO


def _discounted_series(rewards, gamma, last):
    running = float(last)
    out: list[float] = []
    for reward in reversed([float(r) for r in rewards]):
        running = reward + gamma * running
        out.append(running)
    return list(reversed(out))


def test_return_scale_pipeline_fraction_units():
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = False
    algo.value_target_scale = 1.0
    algo._value_target_scale_effective = 4.0
    algo._ret_mean_snapshot = 0.0
    algo._ret_std_snapshot = 1.0
    algo._value_clip_limit_unscaled = None
    algo._value_clip_limit_scaled = None

    base_scale = float(algo.value_target_scale)
    eff_scale = float(algo._value_target_scale_effective)
    gamma = 0.9

    rewards_raw = torch.tensor([0.03, -0.025, 0.01], dtype=torch.float32)
    values_raw = torch.tensor([0.045, 0.02, 0.005], dtype=torch.float32)

    mean_values_norm = values_raw / base_scale * eff_scale
    buffer_values = [
        float(algo._to_raw_returns(v.unsqueeze(0)).item() / base_scale)
        for v in mean_values_norm[:-1]
    ]
    last_scalar_value = float(
        algo._to_raw_returns(mean_values_norm[-1].unsqueeze(0)).item() / base_scale
    )

    buffer_rewards = [float(r / base_scale) for r in rewards_raw]
    buffer_returns = _discounted_series(buffer_rewards, gamma, last_scalar_value)
    returns_tensor = torch.tensor(buffer_returns, dtype=torch.float32)

    decoded, scale_out = algo._decode_returns_scale_only(returns_tensor)
    expected_raw = torch.tensor(
        _discounted_series(rewards_raw, gamma, float(values_raw[-1])), dtype=torch.float32
    )
    assert scale_out == base_scale
    assert torch.allclose(decoded, expected_raw, atol=1e-6)

    target_norm = (decoded / scale_out) * eff_scale
    expected_norm = expected_raw / base_scale * eff_scale
    assert torch.allclose(target_norm, expected_norm, atol=1e-6)

    returns_abs_p99 = torch.quantile(decoded.abs(), 0.99)
    assert returns_abs_p99.item() < 0.2

    buggy_last = float(mean_values_norm[-1] / base_scale)
    buggy_returns = _discounted_series(buffer_rewards, gamma, buggy_last)
    buggy_raw, _ = algo._decode_returns_scale_only(
        torch.tensor(buggy_returns, dtype=torch.float32)
    )
    assert not torch.allclose(buggy_raw, expected_raw, atol=1e-6)


def test_return_scale_pipeline_with_normalization():
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = True
    algo.value_target_scale = 1.0
    algo._value_target_scale_effective = 1.0
    algo._ret_mean_snapshot = 0.05
    algo._ret_std_snapshot = 0.5
    algo._value_clip_limit_unscaled = None
    algo._value_clip_limit_scaled = None

    base_scale = float(algo.value_target_scale)
    gamma = 0.9

    rewards_raw = torch.tensor([0.012, -0.018, 0.008], dtype=torch.float32)
    values_raw = torch.tensor([0.025, 0.015, 0.002], dtype=torch.float32)
    mean_values_norm = (values_raw - algo._ret_mean_snapshot) / algo._ret_std_snapshot

    last_scalar_value = float(
        algo._to_raw_returns(mean_values_norm[-1].unsqueeze(0)).item() / base_scale
    )
    buffer_rewards = [float(r / base_scale) for r in rewards_raw]
    buffer_returns = _discounted_series(buffer_rewards, gamma, last_scalar_value)
    returns_tensor = torch.tensor(buffer_returns, dtype=torch.float32)

    decoded, scale_out = algo._decode_returns_scale_only(returns_tensor)
    expected_raw = torch.tensor(
        _discounted_series(rewards_raw, gamma, float(values_raw[-1])), dtype=torch.float32
    )
    assert torch.allclose(decoded, expected_raw, atol=1e-6)

    target_norm = (decoded - algo._ret_mean_snapshot) / algo._ret_std_snapshot
    expected_norm = (expected_raw - algo._ret_mean_snapshot) / algo._ret_std_snapshot
    assert torch.allclose(target_norm, expected_norm, atol=1e-6)

    returns_abs_p99 = torch.quantile(decoded.abs(), 0.99)
    assert returns_abs_p99.item() < 0.2


def test_reward_pipeline_has_no_percent_scaling():
    step_src = inspect.getsource(trading_patchnew.TradingEnv.step)
    for pattern in ("*100", "* 100", "/100", "/ 100"):
        assert pattern not in step_src

    collect_src = inspect.getsource(DistributionalPPO._collect_rollouts)
    for pattern in ("*100", "* 100", "/100", "/ 100"):
        assert pattern not in collect_src


def test_distributional_ppo_source_has_no_action_nvec_logging():
    src = inspect.getsource(DistributionalPPO)
    assert "action_nvec" not in src

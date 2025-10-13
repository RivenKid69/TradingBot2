import pytest
import torch

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


def test_return_scale_pipeline_percent_units():
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = False
    algo.value_target_scale = 100.0
    algo._value_target_scale_effective = 400.0
    algo._ret_mean_snapshot = 0.0
    algo._ret_std_snapshot = 1.0
    algo._value_clip_limit_unscaled = None
    algo._value_clip_limit_scaled = None

    base_scale = float(algo.value_target_scale)
    eff_scale = float(algo._value_target_scale_effective)
    gamma = 0.9

    rewards_raw = torch.tensor([0.3, -0.25, 0.1], dtype=torch.float32)
    values_raw = torch.tensor([0.45, 0.2, 0.05], dtype=torch.float32)

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

    returns_abs_p95 = torch.quantile(decoded.abs(), 0.95)
    assert returns_abs_p95.item() < 10.0

    buggy_last = float(mean_values_norm[-1] / base_scale)
    buggy_returns = _discounted_series(buffer_rewards, gamma, buggy_last)
    buggy_raw, _ = algo._decode_returns_scale_only(
        torch.tensor(buggy_returns, dtype=torch.float32)
    )
    assert not torch.allclose(buggy_raw, expected_raw, atol=1e-6)


def test_return_scale_pipeline_with_normalization():
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.normalize_returns = True
    algo.value_target_scale = 100.0
    algo._value_target_scale_effective = 100.0
    algo._ret_mean_snapshot = 0.05
    algo._ret_std_snapshot = 0.5
    algo._value_clip_limit_unscaled = None
    algo._value_clip_limit_scaled = None

    base_scale = float(algo.value_target_scale)
    gamma = 0.9

    rewards_raw = torch.tensor([0.12, -0.18, 0.08], dtype=torch.float32)
    values_raw = torch.tensor([0.25, 0.15, 0.02], dtype=torch.float32)
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

    returns_abs_p95 = torch.quantile(decoded.abs(), 0.95)
    assert returns_abs_p95.item() < 10.0

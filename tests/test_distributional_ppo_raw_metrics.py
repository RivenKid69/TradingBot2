import pytest

import test_distributional_ppo_raw_outliers  # noqa: F401  # ensures RL stubs are installed

from distributional_ppo import DistributionalPPO


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value: float, **_: object) -> None:
        self.records[key] = float(value)


def test_record_raw_policy_metrics_logs_expected_keys() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.logger = _CaptureLogger()
    algo._last_rollout_entropy_raw = 0.0

    algo._record_raw_policy_metrics(avg_policy_entropy_raw=0.42, entropy_raw_count=8, kl_raw_sum=0.12, kl_raw_count=6)

    assert algo.logger.records["train/policy_entropy_raw"] == pytest.approx(0.42)
    assert algo.logger.records["train/approx_kl_raw"] == pytest.approx(0.12 / 6)

    # When no fresh statistics are available, the helper should keep previous values intact.
    algo._record_raw_policy_metrics(
        avg_policy_entropy_raw=0.99,
        entropy_raw_count=0,
        kl_raw_sum=0.0,
        kl_raw_count=0,
    )

    assert algo.logger.records["train/policy_entropy_raw"] == pytest.approx(0.42)
    assert algo.logger.records["train/approx_kl_raw"] == pytest.approx(0.12 / 6)

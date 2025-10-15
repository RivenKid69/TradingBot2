import math
import types

import pytest

try:  # noqa: SIM105
    import test_distributional_ppo_raw_outliers  # noqa: F401  # ensure RL stubs are installed
except ImportError:  # pragma: no cover - optional dependency for local pytest setups
    pass

from distributional_ppo import DistributionalPPO


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value: float | None, **_: object) -> None:
        if value is None:
            return
        self.records[key] = float(value)


def test_optimizer_lr_floor_dry_run_logs_chain() -> None:
    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.logger = _CaptureLogger()
    algo._optimizer_lr_min = 5e-6
    algo._optimizer_lr_max = 1e-3
    algo._optimizer_lr_floor_warned = False
    algo._kl_lr_scale = 1.0
    algo._kl_min_lr = 0.0
    algo._current_progress_remaining = 0.5
    algo._base_lr_schedule = lambda progress: 1e-5
    algo.lr_scheduler = None

    param_group = {"lr": 1e-6, "initial_lr": 1e-6}
    optimizer = types.SimpleNamespace(param_groups=[param_group])
    algo.policy = types.SimpleNamespace(optimizer=optimizer, lr_scheduler=None)

    algo._enforce_optimizer_lr_bounds(scheduler_lr=1e-5, log_values=True, warn_on_floor=True)

    logs = algo.logger.records
    assert math.isclose(logs["train/lr_base"], 1e-5, rel_tol=0.0, abs_tol=1e-12)
    assert logs["train/lr_kl_scale"] == pytest.approx(1.0)
    assert logs["train/lr_scheduler"] == pytest.approx(1e-5)
    assert logs["train/lr_before_clip"] == pytest.approx(1e-6)
    assert logs["train/lr_after_clip"] == pytest.approx(5e-6)
    assert logs["train/optimizer_lr"] == pytest.approx(5e-6)
    assert logs["train/optimizer_lr_group_min"] == pytest.approx(5e-6)
    assert logs["train/optimizer_lr_group_max"] == pytest.approx(5e-6)
    assert "warn/optimizer_lr_floor_hit" in logs
    assert param_group["lr"] == pytest.approx(5e-6)
    assert param_group["initial_lr"] == pytest.approx(5e-6)

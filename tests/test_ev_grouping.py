"""Unit tests for grouped explained variance computations."""  # FIX-TEST

from __future__ import annotations

import math

import numpy as np

import sys
import types

if "sb3_contrib" not in sys.modules:  # FIX-TEST
    sb3_contrib = types.ModuleType("sb3_contrib")  # FIX-TEST
    sb3_contrib.__path__ = []  # type: ignore[attr-defined]  # FIX-TEST
    sb3_contrib.RecurrentPPO = object  # type: ignore[attr-defined]  # FIX-TEST
    sys.modules["sb3_contrib"] = sb3_contrib  # FIX-TEST
    sys.modules["sb3_contrib.common"] = types.ModuleType("sb3_contrib.common")  # FIX-TEST
    sys.modules["sb3_contrib.common.recurrent"] = types.ModuleType("sb3_contrib.common.recurrent")  # FIX-TEST
    buffers_mod = types.ModuleType("sb3_contrib.common.recurrent.buffers")  # FIX-TEST
    buffers_mod.RecurrentRolloutBuffer = object  # type: ignore[attr-defined]  # FIX-TEST
    sys.modules["sb3_contrib.common.recurrent.buffers"] = buffers_mod  # FIX-TEST
    policies_mod = types.ModuleType("sb3_contrib.common.recurrent.policies")  # FIX-TEST
    policies_mod.RecurrentActorCriticPolicy = object  # type: ignore[attr-defined]  # FIX-TEST
    sys.modules["sb3_contrib.common.recurrent.policies"] = policies_mod  # FIX-TEST
    type_aliases_mod = types.ModuleType("sb3_contrib.common.recurrent.type_aliases")  # FIX-TEST
    type_aliases_mod.RNNStates = tuple  # type: ignore[attr-defined]  # FIX-TEST
    sys.modules["sb3_contrib.common.recurrent.type_aliases"] = type_aliases_mod  # FIX-TEST

from distributional_ppo import compute_grouped_explained_variance


def test_compute_grouped_explained_variance_mean_matches_average() -> None:  # FIX-TEST
    y_true = np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float32)
    y_pred = np.array([0.9, 2.1, -0.5, -1.0], dtype=np.float32)
    group_keys = ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"]

    grouped, mean_value = compute_grouped_explained_variance(
        y_true,
        y_pred,
        group_keys,
    )

    assert grouped  # at least two groups present
    assert math.isfinite(grouped["BTCUSDT"]) and math.isfinite(grouped["ETHUSDT"])  # FIX-TEST
    assert not math.isclose(grouped["BTCUSDT"], grouped["ETHUSDT"])  # FIX-TEST
    expected_mean = float(np.mean([grouped["BTCUSDT"], grouped["ETHUSDT"]]))
    assert mean_value is not None
    assert math.isclose(mean_value, expected_mean, rel_tol=1e-9, abs_tol=1e-9)  # FIX-TEST

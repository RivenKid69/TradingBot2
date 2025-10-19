import math

import pytest

from TradingBot2.winrate_stats import (
    WinRateStats,
    _clopper_pearson_interval,
    compute_win_rate_stats,
    extract_episode_win_payload,
)


def test_compute_win_rate_stats_basic() -> None:
    stats = compute_win_rate_stats([True, False, True, True], [10, 12, 8, 15])
    assert isinstance(stats, WinRateStats)
    assert stats.total_episodes == 4
    assert stats.total_wins == 3
    assert stats.win_rate == pytest.approx(0.75)
    assert stats.steps_to_win_mean == pytest.approx(11.0)
    assert stats.steps_to_win_median == pytest.approx(10.0)
    assert stats.steps_to_win_min == pytest.approx(8.0)
    assert stats.steps_to_win_max == pytest.approx(15.0)
    assert stats.wilson_low == pytest.approx(0.300641842582402, rel=1e-12)
    assert stats.wilson_high == pytest.approx(0.9544127391902995, rel=1e-12)
    assert stats.clopper_pearson_low == pytest.approx(0.19412044968339615, rel=1e-12)
    assert stats.clopper_pearson_high == pytest.approx(0.9936905367903819, rel=1e-12)


def test_compute_win_rate_stats_empty_and_mismatch() -> None:
    assert compute_win_rate_stats([], []) is None
    with pytest.raises(ValueError):
        compute_win_rate_stats([True], [1, 2])


def test_extract_episode_win_payload() -> None:
    info = {"episode": {"r": 2.5, "l": 7}}
    win_flag, steps = extract_episode_win_payload(info)
    assert win_flag is True
    assert steps == 7

    info_success = {"episode_info": {"win": False, "length": 42}}
    win_flag, steps = extract_episode_win_payload(info_success)
    assert win_flag is False
    assert steps == 42

    missing = extract_episode_win_payload({"episode_stats": {"reward": -1.0}})
    assert missing == (False, None)


def test_clopper_pearson_extreme_bounds() -> None:
    confidence = 0.95
    alpha = 1.0 - confidence
    low, high = _clopper_pearson_interval(0, 10, confidence)
    assert low == pytest.approx(0.0)
    expected_high = 1.0 - math.pow(alpha / 2.0, 1.0 / 10.0)
    assert high == pytest.approx(expected_high, rel=1e-12)

    low_full, high_full = _clopper_pearson_interval(10, 10, confidence)
    expected_low = math.pow(alpha / 2.0, 1.0 / 10.0)
    assert low_full == pytest.approx(expected_low, rel=1e-12)
    assert high_full == pytest.approx(1.0)

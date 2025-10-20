import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.getcwd())

from no_trade import compute_no_trade_mask, estimate_block_ratio, _load_maintenance_calendar
from no_trade_config import NoTradeConfig, get_no_trade_config


def test_compute_no_trade_mask_returns_all_false():
    df = pd.DataFrame({"ts_ms": np.arange(0, 6, dtype=np.int64) * 60_000})

    mask = compute_no_trade_mask(df, sandbox_yaml_path="configs/legacy_sandbox.yaml")

    assert mask.name == "no_trade_block"
    assert not mask.any()

    reasons = mask.attrs.get("reasons")
    assert isinstance(reasons, pd.DataFrame)
    assert reasons.empty

    assert mask.attrs.get("meta") == {"disabled": True}
    assert mask.attrs.get("state") == {}
    assert mask.attrs.get("reason_labels") == {}


def test_estimate_block_ratio_zero_even_with_config():
    cfg = get_no_trade_config("configs/legacy_sandbox.yaml")
    df = pd.DataFrame({"ts_ms": np.arange(0, 24, dtype=np.int64) * 60_000})

    ratio = estimate_block_ratio(df, cfg)

    assert ratio == pytest.approx(0.0)


def test_load_maintenance_calendar_empty_when_disabled():
    cfg = NoTradeConfig()

    calendar, meta = _load_maintenance_calendar(cfg)

    assert calendar == {"global": [], "per_symbol": {}, "windows": []}
    assert meta == {}

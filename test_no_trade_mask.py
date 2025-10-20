import os
import sys
import types
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

lob_state_stub = types.ModuleType("lob_state_cython")
lob_state_stub.N_FEATURES = 1
sys.modules["lob_state_cython"] = lob_state_stub

mediator_stub = types.ModuleType("mediator")
class _Mediator:
    def __init__(self, env):
        self.env = env
        self.calls = []
    def step(self, proto):
        self.calls.append(proto)
        return np.zeros(1), 0.0, False, False, {}
    def reset(self):
        return np.zeros(1, dtype=np.float32), {}
    def _build_observation(self, row, state, mark_price):
        return np.zeros(1, dtype=np.float32)
mediator_stub.Mediator = _Mediator
sys.modules["mediator"] = mediator_stub

from trading_patchnew import TradingEnv
from action_proto import ActionProto, ActionType

def _make_df(ts_minutes):
    ts = np.array(ts_minutes, dtype=np.int64) * 60_000
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "price": 1.0,
            "quote_asset_volume": 1.0,
            "ts_ms": ts,
        }
    )

def test_funding_buffer_mask_applies():
    df = _make_df([360, 410, 470])  # minutes since midnight
    env = TradingEnv(df, no_trade={"funding_buffer_min": 30}, no_trade_enabled=True)
    env.reset()
    act = ActionProto(ActionType.MARKET, 1.0)
    for i in range(len(df)):
        env.state.step_idx = i
        env.step(act)
    assert env._no_trade_enabled
    assert env.no_trade_blocks == 1
    assert env.no_trade_hits == 1
    assert env._no_trade_mask.tolist() == [False, False, True]


def test_custom_window_mask_applies():
    df = _make_df([0, 20, 40])
    env = TradingEnv(
        df,
        no_trade={"custom_ms": [{"start_ts_ms": 19 * 60_000, "end_ts_ms": 21 * 60_000}]},
        no_trade_enabled=True,
    )
    env.reset()
    act = ActionProto(ActionType.MARKET, 1.0)
    env.state.step_idx = 0
    env.step(act)
    env.state.step_idx = 1
    env.step(act)
    assert env._no_trade_enabled
    assert env.no_trade_blocks == 1
    assert env.no_trade_hits == 1
    assert env._no_trade_mask.tolist() == [False, True, False]


def test_mask_disabled_by_default():
    df = _make_df([0, 10, 20])
    env = TradingEnv(df)
    env.reset()

    assert not env._no_trade_enabled
    assert env._no_trade_cfg is None
    assert not bool(env._no_trade_mask.any())

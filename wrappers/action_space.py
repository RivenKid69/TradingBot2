from __future__ import annotations
from typing import Any
from collections.abc import Mapping

import numpy as np
from gymnasium import spaces
from gymnasium import ActionWrapper  # <-- ключевое: наследуемся от ActionWrapper

from action_proto import ActionType


class DictToMultiDiscreteActionWrapper(ActionWrapper):
    """
    Convert Dict action space:
      { price_offset_ticks: Discrete(201),
        ttl_steps:          Discrete(33),
        type:               Discrete(4),
        volume_frac:        Box(-1,1,(1,),float32) }
    -> MultiDiscrete([201, 33, 4, bins_vol])

    Agent outputs [i_price, i_ttl, i_type, i_vol]; wrapper maps to Dict and
    delegates to underlying env.step(...). Observation space is proxied unchanged.
    """

    def __init__(
        self,
        env: Any,
        bins_vol: int = 101,
        action_overrides: Mapping[str, Any] | None = None,
    ):
        # делаем класс полноценным Gymnasium-энвом
        super().__init__(env)
        assert int(bins_vol) >= 2, "bins_vol must be >= 2"
        self.bins_vol = int(bins_vol)
        normalized = self._normalize_overrides(action_overrides)
        self._lock_price_offset = normalized["lock_price_offset"]
        self._lock_ttl = normalized["lock_ttl"]
        self._fixed_type = normalized["fixed_type"]

        # обновляем action_space на MultiDiscrete; observation_space оставляем как у базовой среды
        self.action_space = spaces.MultiDiscrete([201, 33, 4, self.bins_vol])
        self.observation_space = env.observation_space

    @staticmethod
    def _normalize_overrides(
        overrides: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if overrides is None:
            return {"lock_price_offset": False, "lock_ttl": False, "fixed_type": None}

        if hasattr(overrides, "dict"):
            try:
                overrides = overrides.dict()  # type: ignore[assignment]
            except TypeError:
                pass

        data: Mapping[str, Any]
        if isinstance(overrides, Mapping):
            data = overrides
        else:
            data = {}

        lock_price_offset = bool(data.get("lock_price_offset", False))
        lock_ttl = bool(data.get("lock_ttl", False))

        fixed_type_raw = data.get("fixed_type", None)
        fixed_type = None
        if fixed_type_raw is not None:
            fixed_type = DictToMultiDiscreteActionWrapper._coerce_action_type(
                fixed_type_raw
            )

        return {
            "lock_price_offset": lock_price_offset,
            "lock_ttl": lock_ttl,
            "fixed_type": fixed_type,
        }

    @staticmethod
    def _coerce_action_type(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, ActionType):
            member = value
            return int(getattr(member, "value", member))
        if isinstance(value, str):
            name = value.strip().upper()
            if not name:
                return None
            try:
                member = ActionType[name]
            except KeyError as exc:
                raise ValueError(f"Unknown action type name: {value}") from exc
            return int(getattr(member, "value", member))
        try:
            member = ActionType(int(value))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Unsupported action type value: {value!r}") from exc
        return int(getattr(member, "value", member))

    def _vol_center(self, idx: int) -> float:
        idx = int(np.clip(idx, 0, self.bins_vol - 1))
        step = 2.0 / (self.bins_vol - 1)
        return float(-1.0 + step * idx)

    # Метод ActionWrapper.action(a) преобразует действие ПЕРЕД вызовом env.step(...)
    def action(self, action):
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        if a.size != 4:
            raise ValueError(f"Expected 4-dim MultiDiscrete action, got shape {a.shape}")
        price_i, ttl_i, type_i, vol_i = a.tolist()

        # Собираем dict-действие для исходной среды
        dict_action = {
            "price_offset_ticks": int(np.clip(price_i, 0, 200)),
            "ttl_steps":          int(np.clip(ttl_i,   0, 32)),
            "type":               int(np.clip(type_i,  0, 3)),
            "volume_frac":        np.array([self._vol_center(vol_i)], dtype=np.float32),
        }
        if self._lock_price_offset:
            dict_action["price_offset_ticks"] = 0
        if self._lock_ttl:
            dict_action["ttl_steps"] = 0
        if self._fixed_type is not None:
            dict_action["type"] = self._fixed_type
        return dict_action


__all__ = ["DictToMultiDiscreteActionWrapper"]

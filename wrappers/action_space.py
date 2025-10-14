from __future__ import annotations

"""Utilities for enforcing the score-based action space."""

from dataclasses import replace
from typing import Any

import numpy as np
from gymnasium import ActionWrapper, spaces

from action_proto import ActionProto


SCORE_LOW: float = 0.0
SCORE_HIGH: float = 1.0
SCORE_SHAPE: tuple[int, ...] = (1,)


class ScoreActionWrapper(ActionWrapper):
    """Project all outgoing actions to the ``[0, 1]`` score interval."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(
            low=SCORE_LOW,
            high=SCORE_HIGH,
            shape=SCORE_SHAPE,
            dtype=np.float32,
        )
        self.observation_space = env.observation_space

    def action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 1:
            raise ValueError(
                f"ScoreActionWrapper expects a single scalar action, got shape {arr.shape}"
            )
        score = float(arr[0])
        if not np.isfinite(score):
            raise ValueError(f"Received non-finite score action: {score}")
        clipped = np.clip(score, SCORE_LOW, SCORE_HIGH)
        return np.asarray([clipped], dtype=np.float32)


class LongOnlyActionWrapper(ActionWrapper):
    """Clamp outgoing ``score`` actions to keep the policy long-only."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.action_space = getattr(env, "action_space", None)
        self.observation_space = getattr(env, "observation_space", None)

    def action(self, action: Any) -> Any:
        if action is None:
            return action
        if isinstance(action, np.ndarray):
            if action.size == 0:
                return action
            clipped = np.clip(action.astype(np.float32, copy=False), SCORE_LOW, SCORE_HIGH)
            return clipped
        if isinstance(action, (list, tuple)):
            arr = np.asarray(action, dtype=np.float32)
            return np.clip(arr, SCORE_LOW, SCORE_HIGH)
        if isinstance(action, ActionProto):
            clipped = float(np.clip(action.volume_frac, SCORE_LOW, SCORE_HIGH))
            if clipped == action.volume_frac:
                return action
            return replace(action, volume_frac=clipped)
        try:
            value = float(action)
        except (TypeError, ValueError):
            return action
        if not np.isfinite(value):
            raise ValueError(f"Non-finite long-only score: {value}")
        return float(np.clip(value, SCORE_LOW, SCORE_HIGH))


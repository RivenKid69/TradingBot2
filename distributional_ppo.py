import io
import itertools
import logging
import math
from collections import deque
from collections.abc import Mapping
from typing import Any, Callable, Generator, Iterable, NamedTuple, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.save_util import load_from_zip_file


logger = logging.getLogger(__name__)

PadFn = Callable[[Union[np.ndarray, torch.Tensor]], np.ndarray]

try:
    from stable_baselines3.common.vec_env.vec_normalize import unwrap_vec_normalize as _sb3_unwrap
except Exception:  # pragma: no cover - backcompat guard
    _sb3_unwrap = None


torch.set_float32_matmul_precision("high")


def unwrap_vec_normalize(env: VecEnv) -> Optional[VecNormalize]:
    """Backwards compatible helper to locate VecNormalize inside wrappers."""

    if _sb3_unwrap is not None:
        return _sb3_unwrap(env)

    try:
        from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
    except Exception:  # pragma: no cover - optional dependency guard
        return None

    candidate: Optional[VecEnv] = env
    while candidate is not None:
        if isinstance(candidate, VecNormalize):
            return candidate
        if not isinstance(candidate, VecEnvWrapper):
            break
        candidate = getattr(candidate, "venv", None)
    return None


def safe_explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Stable explained variance that guards against zero variance targets."""

    y_true64 = y_true.astype(np.float64)
    y_pred64 = y_pred.astype(np.float64)
    var_y = np.var(y_true64)
    if var_y == 0.0:
        return float("nan")
    return float(1.0 - np.var(y_true64 - y_pred64) / var_y)


def calculate_cvar(probs: torch.Tensor, atoms: torch.Tensor, alpha: float) -> torch.Tensor:
    """Vectorized Conditional Value at Risk for batched categorical distributions."""

    alpha_float = float(alpha)
    if not math.isfinite(alpha_float) or not (0.0 < alpha_float <= 1.0):
        raise ValueError("'alpha' must be a finite probability in the interval (0, 1]")

    if probs.dim() != 2:
        raise ValueError("'probs' must be a 2D tensor")

    batch_size, num_atoms = probs.shape
    if atoms.numel() != num_atoms:
        raise ValueError("'atoms' length must match probability dimension")

    device = probs.device
    dtype = torch.float32

    sorted_atoms, sort_indices = torch.sort(atoms.to(dtype=dtype, device=device))
    expanded_indices = sort_indices.view(1, -1).expand(batch_size, -1)
    sorted_probs = torch.gather(probs.to(dtype=dtype), 1, expanded_indices)
    cumulative_probs = torch.cumsum(sorted_probs, dim=1)

    alpha_tensor = torch.full((batch_size, 1), alpha_float, dtype=dtype, device=device)
    var_indices = torch.searchsorted(cumulative_probs.detach(), alpha_tensor).clamp(max=num_atoms - 1)

    atom_positions = torch.arange(num_atoms, device=device).view(1, -1)
    tail_mask = atom_positions < var_indices
    masked_probs = sorted_probs * tail_mask.to(dtype)
    tail_expectation = torch.sum(masked_probs * sorted_atoms.view(1, -1), dim=1)

    prev_indices = (var_indices - 1).clamp(min=0)
    prev_cum = torch.where(
        var_indices > 0,
        torch.gather(cumulative_probs, 1, prev_indices),
        torch.zeros_like(alpha_tensor),
    ).squeeze(1)

    weight_on_var = (alpha_tensor.squeeze(1) - prev_cum).clamp(min=0.0)
    var_values = sorted_atoms[var_indices.squeeze(1)]

    cvar = (tail_expectation + weight_on_var * var_values) / (alpha_float + 1e-8)
    return cvar


def create_sequencers(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: Union[str, torch.device],
) -> Tuple[np.ndarray, PadFn, PadFn]:
    """Utility that builds padding helpers for variable length RNN rollouts.

    Parameters
    ----------
    episode_starts:
        Flat array indicating when a new episode begins inside the sampled
        rollout segment.
    env_change:
        Flat array indicating when we jump to the next environment chunk.
    device:
        Unused placeholder, kept for backwards compatibility with older
        implementations that expected a device argument.

    Returns
    -------
    seq_start_indices:
        Indices in the flattened batch where each RNN sequence begins.
    pad:
        Function that pads an array according to the computed sequences,
        returning a ``(n_seq, max_len, *rest)`` tensor.
    pad_and_flatten:
        Convenience wrapper that first pads and then flattens the leading
        dimensions back to ``(n_seq * max_len, *rest)``.
    """

    del device  # The helpers operate purely on numpy arrays.

    episode_starts_np = np.asarray(episode_starts, dtype=bool)
    env_change_np = np.asarray(env_change, dtype=bool)

    if episode_starts_np.shape != env_change_np.shape:
        raise ValueError("'episode_starts' and 'env_change' must share the same shape")

    episode_starts_np = np.squeeze(episode_starts_np)
    env_change_np = np.squeeze(env_change_np)

    if episode_starts_np.ndim != 1:
        raise ValueError("'episode_starts' and 'env_change' must be 1D arrays")

    combined_flags = np.logical_or(episode_starts_np, env_change_np)
    if combined_flags.size == 0:
        raise ValueError("Cannot create sequencers from empty rollout segments")

    combined_flags[0] = True
    seq_start_indices = np.flatnonzero(combined_flags).astype(np.int64, copy=False)

    # Determine the unpadded length of each sequence so we can pad consistently.
    seq_ends = np.concatenate((seq_start_indices[1:], np.array([combined_flags.size], dtype=np.int64)))
    seq_lengths = seq_ends - seq_start_indices
    max_length = int(seq_lengths.max()) if seq_lengths.size > 0 else 0

    def pad(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        arr_np = array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else np.asarray(array)
        if arr_np.shape[0] != combined_flags.size:
            raise ValueError("Input has incompatible leading dimension for padding")

        trailing_shape = arr_np.shape[1:]
        padded_shape = (len(seq_start_indices), max_length) + trailing_shape
        padded = np.zeros(padded_shape, dtype=arr_np.dtype)

        for i, (start, length) in enumerate(zip(seq_start_indices, seq_lengths)):
            padded[i, :length, ...] = arr_np[start : start + length, ...]

        return padded

    def pad_and_flatten(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        padded = pad(array)
        return padded.reshape((len(seq_start_indices) * max_length, *padded.shape[2:]))

    return seq_start_indices, pad, pad_and_flatten


class RawRecurrentRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    lstm_states: RNNStates
    episode_starts: torch.Tensor
    mask: torch.Tensor
    actions_raw: torch.Tensor
    old_log_prob_raw: torch.Tensor


class RawRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    def reset(self) -> None:
        super().reset()
        self.actions_raw = np.zeros_like(self.actions, dtype=self.actions.dtype)
        self.old_log_prob_raw = np.zeros_like(self.log_probs, dtype=self.log_probs.dtype)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def add(
        self,
        *args: Any,
        lstm_states: RNNStates,
        actions_raw: Any,
        log_prob_raw: Any,
        **kwargs: Any,
    ) -> None:
        if actions_raw is None or log_prob_raw is None:
            raise TypeError("'actions_raw' and 'log_prob_raw' must be provided when adding to the rollout buffer")

        super().add(*args, lstm_states=lstm_states, **kwargs)

        pos = (self.pos - 1) % self.buffer_size

        raw_np = self._to_numpy(actions_raw)
        raw_np = np.reshape(raw_np, self.actions_raw[pos].shape)
        self.actions_raw[pos] = raw_np.astype(self.actions_raw.dtype, copy=False)

        log_prob_raw_np = self._to_numpy(log_prob_raw)
        log_prob_raw_np = np.reshape(log_prob_raw_np, self.old_log_prob_raw[pos].shape)
        self.old_log_prob_raw[pos] = log_prob_raw_np.astype(self.old_log_prob_raw.dtype, copy=False)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RawRecurrentRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        if not self.generator_ready:
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
                "actions_raw",
                "old_log_prob_raw",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RawRecurrentRolloutBufferSamples:
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length

        lstm_states_pi = (
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (
            self.to_torch(lstm_states_pi[0]).contiguous(),
            self.to_torch(lstm_states_pi[1]).contiguous(),
        )
        lstm_states_vf = (
            self.to_torch(lstm_states_vf[0]).contiguous(),
            self.to_torch(lstm_states_vf[1]).contiguous(),
        )

        def _to_tensor(
            array: np.ndarray,
            *,
            convert_floats: bool = False,
            force_float32: bool = False,
        ) -> torch.Tensor:
            tensor = torch.as_tensor(array, device=self.device)
            if force_float32:
                return tensor.to(dtype=torch.float32)
            if convert_floats and tensor.is_floating_point() and tensor.dtype != torch.float32:
                tensor = tensor.to(dtype=torch.float32)
            return tensor

        observations_np = self.pad(self.observations[batch_inds]).reshape((padded_batch_size, *self.obs_shape))
        actions_np = self.pad(self.actions[batch_inds]).reshape((padded_batch_size, *self.actions.shape[1:]))
        actions_raw_np = self.pad(self.actions_raw[batch_inds]).reshape(
            (padded_batch_size, *self.actions_raw.shape[1:])
        )

        old_values_np = self.pad_and_flatten(self.values[batch_inds])
        old_log_prob_np = self.pad_and_flatten(self.log_probs[batch_inds])
        advantages_np = self.pad_and_flatten(self.advantages[batch_inds])
        returns_np = self.pad_and_flatten(self.returns[batch_inds])
        episode_starts_np = self.pad_and_flatten(self.episode_starts[batch_inds])
        mask_np = self.pad_and_flatten(np.ones_like(self.returns[batch_inds]))
        old_log_prob_raw_np = self.pad_and_flatten(self.old_log_prob_raw[batch_inds])

        observations = _to_tensor(observations_np, convert_floats=True)
        actions = _to_tensor(actions_np, convert_floats=True)
        actions_raw = _to_tensor(actions_raw_np, convert_floats=True)
        old_values = _to_tensor(old_values_np, convert_floats=True)
        old_log_prob = _to_tensor(old_log_prob_np, convert_floats=True)
        advantages = _to_tensor(advantages_np, convert_floats=True)
        returns = _to_tensor(returns_np, convert_floats=True)
        episode_starts = _to_tensor(episode_starts_np, force_float32=True)
        mask = _to_tensor(mask_np, force_float32=True)
        old_log_prob_raw = _to_tensor(old_log_prob_raw_np, convert_floats=True)

        return RawRecurrentRolloutBufferSamples(
            observations=observations,
            actions=actions,
            old_values=old_values,
            old_log_prob=old_log_prob,
            advantages=advantages,
            returns=returns,
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=episode_starts,
            mask=mask,
            actions_raw=actions_raw,
            old_log_prob_raw=old_log_prob_raw,
        )


class DistributionalPPO(RecurrentPPO):
    """Distributional PPO with CVaR regularisation and entropy scheduling."""

    def _setup_model(self) -> None:
        super()._setup_model()

        if isinstance(self.rollout_buffer, RecurrentRolloutBuffer) and not isinstance(
            self.rollout_buffer, RawRecurrentRolloutBuffer
        ):
            lstm = getattr(self.policy, "lstm_actor", None) or getattr(
                self.policy, "lstm_critic", None
            )
            if lstm is None:
                hidden_state_buffer_shape = (self.n_steps, 1, self.n_envs, 1)
            else:
                hidden_state_buffer_shape = (
                    self.n_steps,
                    lstm.num_layers,
                    self.n_envs,
                    lstm.hidden_size,
                )
            self.rollout_buffer = RawRecurrentRolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                hidden_state_buffer_shape,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )

    @property
    def cvar_winsor_pct(self) -> float:
        return float(getattr(self, "_cvar_winsor_pct", 0.0))

    @cvar_winsor_pct.setter
    def cvar_winsor_pct(self, value: float) -> None:
        pct_value = float(value)
        if not math.isfinite(pct_value) or pct_value < 0.0:
            raise ValueError("'cvar_winsor_pct' must be a non-negative finite value")
        if pct_value <= 0.005:
            pct_value *= 100.0
        if pct_value > 50.0:
            pct_value = 50.0
        if pct_value >= 50.0:
            pct_value = 50.0 - 1e-6
        fraction = pct_value / 100.0
        fraction = float(min(max(fraction, 0.0), 0.5 - 1e-6))
        self._cvar_winsor_pct = pct_value
        self._cvar_winsor_fraction = fraction

    @staticmethod
    def _clone_states_to_device(
        states: Optional[RNNStates | tuple[torch.Tensor, ...]], device: torch.device
    ) -> Optional[RNNStates | tuple[torch.Tensor, ...]]:
        if states is None:
            return None

        if hasattr(states, "pi") and hasattr(states, "vf"):
            pi_states = tuple(state.to(device) for state in states.pi)
            vf_states = tuple(state.to(device) for state in states.vf)
            return RNNStates(pi=pi_states, vf=vf_states)

        return tuple(state.to(device) for state in states)

    @staticmethod
    def _extract_actor_states(
        states: Optional[RNNStates | tuple[torch.Tensor, ...]]
    ) -> Optional[tuple[torch.Tensor, ...]]:
        if states is None:
            return None

        if hasattr(states, "pi"):
            return tuple(states.pi)

        return tuple(states)

    @staticmethod
    def _value_target_outlier_fractions(
        values: torch.Tensor, support_min: float, support_max: float
    ) -> tuple[float, float]:
        if values.numel() == 0:
            return 0.0, 0.0

        values_fp32 = values.to(dtype=torch.float32)
        below_frac = (values_fp32 < support_min).float().mean().item()
        above_frac = (values_fp32 > support_max).float().mean().item()
        return float(below_frac), float(above_frac)

    def _quantile_levels_tensor(self, device: torch.device) -> torch.Tensor:
        levels = getattr(self.policy, "quantile_levels", None)
        if levels is None:
            raise RuntimeError("Quantile levels are not available on the policy")
        return levels.to(device=device, dtype=torch.float32)

    def _quantile_huber_loss(
        self, predicted_quantiles: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        kappa = max(float(self._quantile_huber_kappa), 1e-6)
        tau = self._quantile_levels_tensor(predicted_quantiles.device).view(1, -1)
        delta = targets.unsqueeze(1) - predicted_quantiles
        abs_delta = delta.abs()
        huber = torch.where(
            abs_delta <= kappa,
            0.5 * delta.pow(2),
            kappa * (abs_delta - 0.5 * kappa),
        )
        indicator = (delta.detach() < 0.0).float()
        loss = torch.abs(tau - indicator) * (huber / kappa)
        return loss.mean()

    def _cvar_from_quantiles(self, predicted_quantiles: torch.Tensor) -> torch.Tensor:
        alpha = float(self.cvar_alpha)
        if alpha <= 0.0:
            raise ValueError("CVaR alpha must be positive for quantile critic")
        num_quantiles = predicted_quantiles.shape[1]
        if num_quantiles == 0:
            return predicted_quantiles.new_zeros(predicted_quantiles.shape[0])
        mass = 1.0 / float(num_quantiles)
        # TODO(quantile-critic): When migrating to non-uniform quantile levels
        # (e.g. IQN), replace the uniform ``mass`` assumption with explicit
        # integration over the τ intervals.
        k_float = alpha * num_quantiles
        full_mass = int(min(num_quantiles, math.floor(k_float)))
        frac = float(k_float - full_mass)
        device = predicted_quantiles.device
        dtype = predicted_quantiles.dtype
        tail_sum = predicted_quantiles.new_zeros(predicted_quantiles.shape[0], dtype=dtype, device=device)
        if full_mass > 0:
            tail_sum = predicted_quantiles[:, :full_mass].sum(dim=1)
        partial = predicted_quantiles.new_zeros(predicted_quantiles.shape[0], dtype=dtype, device=device)
        if frac > 1e-8 and full_mass < num_quantiles:
            partial = predicted_quantiles[:, full_mass] * frac
        expectation = mass * (tail_sum + partial)
        tail_mass = max(alpha, mass * (full_mass + frac))
        return expectation / tail_mass

    def _enforce_optimizer_lr_bounds(
        self,
        *,
        scheduler_lr: Optional[float] = None,
        log_values: bool = True,
        warn_on_floor: bool = True,
    ) -> None:
        optimizer = getattr(self.policy, "optimizer", None)
        if optimizer is None:
            return

        param_groups = getattr(optimizer, "param_groups", [])
        if not param_groups:
            return

        min_lr = float(getattr(self, "_optimizer_lr_min", 0.0))
        max_lr = float(getattr(self, "_optimizer_lr_max", float("inf")))
        if max_lr < min_lr:
            max_lr = min_lr

        scheduler_min_lr = float(getattr(self, "_scheduler_min_lr", min_lr))
        if scheduler_min_lr < min_lr:
            scheduler_min_lr = min_lr

        before_lrs = [float(group.get("lr", 0.0)) for group in param_groups]
        if not before_lrs:
            return

        lr_before_clip = float(before_lrs[0])
        clipped_lrs: list[float] = []
        floor_hit = False
        for lr_value in before_lrs:
            clipped = float(lr_value)
            if math.isfinite(max_lr):
                clipped = min(clipped, max_lr)
            if clipped < min_lr:
                floor_hit = True
                clipped = min_lr
            clipped_lrs.append(clipped)

        for group, new_lr in zip(param_groups, clipped_lrs):
            group["lr"] = float(new_lr)
            if "initial_lr" in group:
                initial_lr = float(group["initial_lr"])
                if math.isfinite(max_lr):
                    initial_lr = min(max(initial_lr, min_lr), max_lr)
                else:
                    initial_lr = max(initial_lr, min_lr)
                group["initial_lr"] = initial_lr

        lr_after_clip = float(clipped_lrs[0])

        if warn_on_floor and floor_hit and lr_before_clip < min_lr - 1e-12:
            if not getattr(self, "_optimizer_lr_floor_warned", False):
                logger.warning(
                    "Optimizer LR %.6e fell below floor %.6e; clamping to floor",
                    lr_before_clip,
                    min_lr,
                )
                self._optimizer_lr_floor_warned = True
            if getattr(self, "logger", None) is not None:
                self.logger.record("warn/optimizer_lr_floor_hit", float(lr_before_clip))

        base_schedule = getattr(self, "_base_lr_schedule", None)
        progress = getattr(self, "_current_progress_remaining", None)
        base_lr = None
        if callable(base_schedule) and progress is not None:
            try:
                base_lr = float(base_schedule(progress))
            except Exception:
                base_lr = None
        if base_lr is None:
            kl_scale = float(getattr(self, "_kl_lr_scale", 1.0))
            if kl_scale > 0.0:
                base_lr = lr_after_clip / kl_scale
            else:
                base_lr = lr_after_clip
        kl_scale_value = float(getattr(self, "_kl_lr_scale", 1.0))

        scheduler_lr_value: Optional[float]
        if scheduler_lr is not None:
            scheduler_lr_value = float(scheduler_lr)
        else:
            scheduler_lr_value = None
            scheduler = getattr(self.policy, "lr_scheduler", None) or getattr(self, "lr_scheduler", None)
            if scheduler is not None:
                get_last_lr = getattr(scheduler, "get_last_lr", None)
                if callable(get_last_lr):
                    try:
                        last_values = get_last_lr()
                    except TypeError:
                        last_values = None
                    if last_values:
                        scheduler_lr_value = float(last_values[0])
        if scheduler_lr_value is None:
            scheduler_lr_value = float(lr_after_clip)
        scheduler_lr_value = max(float(scheduler_lr_value), scheduler_min_lr)

        min_group_lr = float(min(clipped_lrs))
        max_group_lr = float(max(clipped_lrs))

        if not log_values or getattr(self, "logger", None) is None:
            return

        self.logger.record("train/lr_base", float(base_lr))
        self.logger.record("train/lr_kl_scale", float(kl_scale_value))
        self.logger.record("train/lr_scheduler", float(scheduler_lr_value))
        self.logger.record("train/scheduler_lr", float(scheduler_lr_value))
        self.logger.record("train/lr_before_clip", float(lr_before_clip))
        self.logger.record("train/lr_after_clip", float(lr_after_clip))
        self.logger.record("train/learning_rate", float(lr_after_clip))
        self.logger.record("train/optimizer_lr", float(clipped_lrs[0]))
        self.logger.record("train/optimizer_lr_min", float(min_lr))
        self.logger.record("train/scheduler_lr_min", float(scheduler_min_lr))
        self.logger.record("train/optimizer_lr_group_min", min_group_lr)
        self.logger.record("train/optimizer_lr_group_max", max_group_lr)

    def _compute_empirical_cvar(
        self, raw_rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Winsorise raw rewards and estimate empirical CVaR in fraction units."""

        if raw_rewards.numel() == 0:
            zero = raw_rewards.new_tensor(0.0)
            return raw_rewards.to(dtype=torch.float32), zero

        rewards_fp32 = raw_rewards.to(dtype=torch.float32)
        winsor_pct = float(
            min(max(getattr(self, "_cvar_winsor_fraction", 0.0), 0.0), 0.5 - 1e-6)
        )
        if winsor_pct > 0.0:
            quantiles = torch.quantile(
                rewards_fp32,
                torch.tensor(
                    [winsor_pct, 1.0 - winsor_pct],
                    dtype=rewards_fp32.dtype,
                    device=rewards_fp32.device,
                ),
            )
            lower, upper = quantiles[0], quantiles[1]
            rewards_winsor = rewards_fp32.clamp(min=float(lower.item()), max=float(upper.item()))
        else:
            rewards_winsor = rewards_fp32

        alpha = float(self.cvar_alpha)
        if not math.isfinite(alpha) or alpha <= 0.0:
            cvar_empirical = rewards_winsor.new_tensor(0.0)
        else:
            tail_count = max(int(math.ceil(alpha * rewards_winsor.numel())), 1)
            tail, _ = torch.topk(rewards_winsor, tail_count, largest=False)
            cvar_empirical = tail.mean() if tail.numel() > 0 else rewards_winsor.new_tensor(0.0)

        return rewards_winsor, cvar_empirical

    def _compute_cvar_statistics(
        self, raw_rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if raw_rewards.numel() == 0:
            zero = raw_rewards.new_tensor(0.0)
            return raw_rewards.to(dtype=torch.float32), zero, zero, zero, zero

        rewards_winsor, cvar_empirical = self._compute_empirical_cvar(raw_rewards)
        rewards_fp32 = raw_rewards.to(dtype=torch.float32)

        quantiles_raw = torch.quantile(
            rewards_fp32,
            torch.tensor([0.5, 0.95], dtype=rewards_fp32.dtype, device=rewards_fp32.device),
        )
        reward_p50 = quantiles_raw[0]
        reward_p95 = quantiles_raw[1]
        returns_abs_p95 = torch.quantile(rewards_fp32.abs(), 0.95).clamp_min(0.0)

        return rewards_winsor, cvar_empirical, reward_p50, reward_p95, returns_abs_p95

    def _compute_cvar_violation(self, cvar_empirical: float) -> float:
        """Return positive CVaR violation in fraction-per-bar units."""

        limit = float(self._get_cvar_limit_raw())
        violation = limit - float(cvar_empirical)
        if not math.isfinite(violation) or violation <= 0.0:
            return 0.0
        return float(violation)

    def _record_quantile_summary(
        self,
        quantiles_unscaled: list[torch.Tensor],
        quantiles_norm: list[torch.Tensor],
    ) -> None:
        if not quantiles_unscaled:
            return
        levels = self._quantile_levels_tensor(quantiles_unscaled[0].device)
        concat_unscaled = torch.cat(quantiles_unscaled, dim=0)
        concat_norm = torch.cat(quantiles_norm, dim=0) if quantiles_norm else None
        def _index_for(target: float) -> int:
            idx = torch.argmin(torch.abs(levels - target)).item()
            return int(idx)

        idx_q25 = _index_for(0.25)
        idx_q50 = _index_for(0.5)
        idx_q75 = _index_for(0.75)

        q50 = concat_unscaled[:, idx_q50].mean().item()
        q25 = concat_unscaled[:, idx_q25]
        q75 = concat_unscaled[:, idx_q75]
        iqr = (q75 - q25).mean().item()
        self.logger.record("train/value_quantile_p50", float(q50))
        self.logger.record("train/value_quantile_iqr", float(iqr))

        if concat_norm is not None:
            q25_norm = concat_norm[:, idx_q25]
            q75_norm = concat_norm[:, idx_q75]
            iqr_norm = (q75_norm - q25_norm).mean().item()
            self.logger.record("train/value_quantile_iqr_norm", float(iqr_norm))

    def _ensure_score_action_space(self) -> None:
        """Validate that policy and environment expose a score-based Box(1) action."""

        def _check_box(space: gym.Space, *, owner: str) -> None:
            if not isinstance(space, gym.spaces.Box):
                raise RuntimeError(
                    f"{owner} action space must be gym.spaces.Box for score actions"
                )
            shape = tuple(int(x) for x in getattr(space, "shape", tuple()))
            if shape != (1,):
                raise RuntimeError(
                    f"{owner} action space must have shape (1,), got {shape}"
                )
            low = np.asarray(space.low, dtype=np.float32)
            high = np.asarray(space.high, dtype=np.float32)
            if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
                raise RuntimeError(f"{owner} action bounds must be finite")
            if np.min(low) > 0.0 or np.max(high) < 1.0:
                raise RuntimeError(
                    f"{owner} action bounds must cover [0, 1], got [{float(np.min(low))}, {float(np.max(high))}]"
                )

        action_space = getattr(self, "action_space", None)
        if action_space is not None:
            _check_box(action_space, owner="Environment")

        policy = getattr(self, "policy", None)
        if policy is not None:
            policy_space = getattr(policy, "action_space", None)
            if policy_space is not None:
                _check_box(policy_space, owner="Policy")
            action_dim = getattr(policy, "action_dim", None)
            if action_dim is not None and int(action_dim) != 1:
                raise RuntimeError(
                    f"Policy action_dim must be 1 for score actions, got {action_dim}"
                )

    @staticmethod
    def _coerce_value_target_scale(value_target_scale: Union[str, float, None]) -> float:
        if value_target_scale is None:
            return 1.0

        if isinstance(value_target_scale, str):
            normalized = value_target_scale.strip().lower()
            if normalized in {"percent", "percents", "percentage", "%"}:
                scale = 100.0
            elif normalized in {"bps", "basis_points", "basis-point", "basis points"}:
                scale = 10000.0
            else:
                raise ValueError(
                    "Unsupported value_target_scale string. Use 'percent', 'bps', or provide a positive float."
                )
        else:
            try:
                scale = float(value_target_scale)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid value_target_scale={value_target_scale!r}; expected positive float or recognised alias."
                ) from exc

        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"'value_target_scale' must be a positive finite value, got {value_target_scale!r}"
            )
        return scale

    def _activate_return_scale_snapshot(self) -> None:
        """Freeze return statistics for the upcoming optimisation step."""

        if not self.normalize_returns:
            self._ret_mean_snapshot = float(self._ret_mean_value)
            self._ret_std_snapshot = float(self._ret_std_value)
            self._pending_rms = None
            self._pending_ret_mean = None
            self._pending_ret_std = None
            return

        if getattr(self, "_value_scale_frozen", False):
            self._ret_mean_snapshot = float(self._ret_mean_value)
            self._ret_std_snapshot = max(
                float(self._ret_std_value), self._value_scale_std_floor
            )
            self._pending_rms = None
            self._pending_ret_mean = float(self._ret_mean_snapshot)
            self._pending_ret_std = float(self._ret_std_snapshot)
            return

        self._ret_mean_snapshot = float(self._ret_mean_value)
        self._ret_std_snapshot = max(
            float(self._ret_std_value), self._value_scale_std_floor
        )
        self._pending_rms = RunningMeanStd(shape=())
        self._pending_ret_mean = float(self._ret_mean_snapshot)
        self._pending_ret_std = float(self._ret_std_snapshot)

    def _to_raw_returns(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_returns:
            mean = x.new_tensor(self._ret_mean_snapshot)
            std = x.new_tensor(self._ret_std_snapshot)
            return x * std + mean

        eff = float(getattr(self, "_value_target_scale_effective", self.value_target_scale))
        base = float(self.value_target_scale)
        eff = eff if abs(eff) > 1e-8 else 1.0
        base = base if abs(base) > 1e-8 else 1.0
        return (x / eff) * base

    def _resolve_value_scale_safe(self) -> float:
        base_scale = float(self.value_target_scale)
        return base_scale if abs(base_scale) > 1e-8 else 1.0

    def _decode_returns_scale_only(
        self, returns_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Convert rollout-buffer returns back to raw fraction units without RMS factors."""

        scale_safe = self._resolve_value_scale_safe()
        return returns_tensor * float(scale_safe), scale_safe

    def _get_cvar_limit_raw(self) -> float:
        """Return the CVaR limit expressed in the same units as raw rewards."""

        return float(self.cvar_limit)

    def _get_cvar_normalization_params(self) -> tuple[float, float]:
        """Return offset and scale for normalising CVaR statistics."""

        if getattr(self, "normalize_returns", False):
            mean_snapshot = float(getattr(self, "_ret_mean_snapshot", 0.0))
            std_snapshot = float(getattr(self, "_ret_std_snapshot", 1.0))
            std_floor = float(getattr(self, "_value_scale_std_floor", 1e-8))
            scale = max(abs(std_snapshot), std_floor, 1e-8)
            if not math.isfinite(scale):
                scale = 1.0
            return float(mean_snapshot), float(scale)

        robust_scale = float(getattr(self, "_value_target_scale_robust", 1.0))
        if not math.isfinite(robust_scale) or robust_scale <= 0.0:
            robust_scale = float(self._resolve_value_scale_safe())
        return 0.0, float(max(abs(robust_scale), 1e-8))

    @staticmethod
    def _bounded_dual_update(lambda_value: float, lr: float, gap_unit: float) -> float:
        """Project dual variable updates into the unit interval."""

        lambda_float = float(lambda_value)
        lr_float = float(lr)
        gap_float = float(gap_unit)
        if not math.isfinite(lambda_float):
            lambda_float = 0.0
        if not math.isfinite(lr_float):
            lr_float = 0.0
        if not math.isfinite(gap_float):
            gap_float = 0.0
        candidate = lambda_float + lr_float * gap_float
        if candidate <= 0.0:
            return 0.0
        if candidate >= 1.0:
            return 1.0
        return float(candidate)

    def _resolve_cvar_penalty_state(
        self,
        current_weight_nominal: float,
        current_weight_raw: float,
        violation_unit_value: float,
    ) -> tuple[float, float, bool]:
        """Determine CVaR penalty activation and fallback weights in normalised units."""

        penalty_active = float(violation_unit_value) > 0.0
        if not penalty_active:
            return 0.0, 0.0, False

        if current_weight_raw > 0.0:
            return float(current_weight_nominal), float(current_weight_raw), True

        ramp_updates = max(1.0, float(getattr(self, "_cvar_ramp_updates", 0)))
        fallback_weight = float(getattr(self, "_cvar_weight_target", 0.0)) / ramp_updates
        fallback_weight = max(fallback_weight, 1e-6)
        penalty_cap = float(getattr(self, "cvar_penalty_cap", float("inf")))
        if math.isfinite(penalty_cap):
            fallback_weight = min(fallback_weight, penalty_cap)
            fallback_weight = max(fallback_weight, 1e-6)
        fallback = float(fallback_weight)
        return fallback, fallback, True

    def _record_cvar_logs(
        self,
        *,
        cvar_raw_value: float,
        cvar_unit_value: float,
        cvar_loss_raw_value: float,
        cvar_loss_unit_value: float,
        cvar_term_raw_value: float,
        cvar_term_unit_value: float,
        cvar_empirical_value: float,
        cvar_empirical_unit_value: float,
        cvar_empirical_ema_value: float,
        cvar_violation_raw_value: float,
        cvar_violation_raw_unclipped_value: float,
        cvar_violation_unit_value: float,
        cvar_violation_ema_value: float,
        cvar_gap_raw_value: float,
        cvar_gap_unit_value: float,
        cvar_penalty_active_value: float,
        cvar_lambda_value: float,
        cvar_scale_value: float,
        cvar_limit_raw_value: float,
        cvar_limit_unit_value: float,
        current_cvar_weight_scaled: float,
        current_cvar_weight_nominal: float,
        current_cvar_weight_raw: float,
        cvar_penalty_cap_value: float,
    ) -> None:
        """Emit CVaR telemetry in both raw fraction and normalised units."""

        self.logger.record("train/cvar_raw", float(cvar_raw_value))
        self.logger.record("train/cvar_raw_in_fraction", float(cvar_raw_value))
        self.logger.record("train/cvar_unit", float(cvar_unit_value))
        self.logger.record("train/cvar_loss", float(cvar_loss_raw_value))
        self.logger.record("train/cvar_loss_in_fraction", float(cvar_loss_raw_value))
        self.logger.record("train/cvar_loss_unit", float(cvar_loss_unit_value))
        self.logger.record("train/cvar_term", float(cvar_term_unit_value))
        self.logger.record("train/cvar_term_in_fraction", float(cvar_term_raw_value))
        self.logger.record("train/cvar_empirical", float(cvar_empirical_value))
        self.logger.record("train/cvar_empirical_in_fraction", float(cvar_empirical_value))
        self.logger.record("train/cvar_empirical_unit", float(cvar_empirical_unit_value))
        self.logger.record("train/cvar_empirical_ema", float(cvar_empirical_ema_value))
        self.logger.record("train/cvar_gap", float(cvar_gap_raw_value))
        self.logger.record("train/cvar_gap_in_fraction", float(cvar_gap_raw_value))
        self.logger.record("train/cvar_gap_unit", float(cvar_gap_unit_value))
        self.logger.record("train/cvar_violation", float(cvar_violation_raw_value))
        self.logger.record("train/cvar_violation_in_fraction", float(cvar_violation_raw_value))
        self.logger.record("train/cvar_violation_unit", float(cvar_violation_unit_value))
        self.logger.record("train/cvar_violation_ema", float(cvar_violation_ema_value))
        self.logger.record("train/cvar_gap_pos", float(cvar_violation_unit_value))
        self.logger.record("train/cvar_penalty_active", float(cvar_penalty_active_value))
        self.logger.record("train/cvar_lambda", float(cvar_lambda_value))
        self.logger.record("train/cvar_scale", float(cvar_scale_value))
        self.logger.record("train/cvar_limit_unit", float(cvar_limit_unit_value))
        self.logger.record("train/cvar_weight_effective", float(current_cvar_weight_scaled))
        self.logger.record("debug/cvar_empirical_raw", float(cvar_empirical_value))
        self.logger.record("debug/cvar_violation_raw", float(cvar_violation_raw_unclipped_value))
        self.logger.record("debug/cvar_weight_nominal", float(current_cvar_weight_nominal))
        self.logger.record("debug/cvar_weight_effective_raw", float(current_cvar_weight_raw))
        self.logger.record("debug/cvar_penalty_cap", float(cvar_penalty_cap_value))
        self.logger.record("debug/cvar_lambda", float(cvar_lambda_value))
        self.logger.record("debug/cvar_limit", float(cvar_limit_raw_value))
        self.logger.record("debug/cvar_limit_unit", float(cvar_limit_unit_value))

    def _limit_mean_step(self, old_value: float, proposed: float, reference_std: float) -> float:
        max_rel_step = float(self._value_scale_max_rel_step)
        if max_rel_step <= 0.0 or not math.isfinite(max_rel_step):
            return float(proposed)

        scale = max(abs(old_value), abs(reference_std), float(self._ret_std_snapshot), 1e-8)
        max_delta = scale * max_rel_step
        lower = old_value - max_delta
        upper = old_value + max_delta
        if lower > upper:
            lower, upper = upper, lower
        return float(min(max(proposed, lower), upper))

    def _limit_std_step(self, old_value: float, proposed: float) -> float:
        proposed = max(float(proposed), 1e-8)
        max_rel_step = float(self._value_scale_max_rel_step)
        if max_rel_step <= 0.0 or not math.isfinite(max_rel_step):
            return proposed

        base = max(float(old_value), 1e-8)
        upper = base * (1.0 + max_rel_step)
        lower = base / (1.0 + max_rel_step)
        if lower > upper:
            lower, upper = upper, lower
        ratio = proposed / base
        clamped_ratio = min(max(ratio, lower / base), upper / base)
        return float(max(base * clamped_ratio, 1e-8))

    def _limit_v_range_step(
        self,
        old_min: float,
        old_max: float,
        proposed_min: float,
        proposed_max: float,
    ) -> tuple[float, float]:
        """Clamp the change of the distributional value support per update."""

        max_rel = float(self._value_scale_range_max_rel_step)
        if max_rel <= 0.0 or not math.isfinite(max_rel):
            return float(proposed_min), float(proposed_max)

        old_min = float(old_min)
        old_max = float(old_max)
        proposed_min = float(proposed_min)
        proposed_max = float(proposed_max)

        if proposed_max <= proposed_min:
            span = max(abs(proposed_max - proposed_min), 1e-6)
            center = 0.5 * (proposed_max + proposed_min)
            proposed_min = center - 0.5 * span
            proposed_max = center + 0.5 * span

        span_old = float(old_max - old_min)
        if not math.isfinite(span_old) or span_old <= 1e-8:
            return proposed_min, proposed_max

        max_delta = span_old * max_rel
        limited_min = max(min(proposed_min, old_min + max_delta), old_min - max_delta)
        limited_max = max(min(proposed_max, old_max + max_delta), old_max - max_delta)

        if limited_max <= limited_min:
            center = 0.5 * (limited_max + limited_min)
            half_span = max(span_old * 0.5, 5e-7)
            limited_min = center - half_span
            limited_max = center + half_span

        return float(limited_min), float(limited_max)

    def _robust_std_from_returns(self, returns_raw: torch.Tensor) -> float:
        returns_abs = returns_raw.abs().to(dtype=torch.float32)
        if returns_abs.numel() == 0:
            return float(self._value_scale_std_floor)
        q95 = torch.quantile(returns_abs, 0.95).clamp_min(0.0)
        q95_value = float(q95.item())
        if not math.isfinite(q95_value) or q95_value <= 0.0:
            return float(self._value_scale_std_floor)
        robust = q95_value / 1.645
        if not math.isfinite(robust) or robust <= 0.0:
            robust = self._value_scale_std_floor
        return float(max(robust, self._value_scale_std_floor))

    def _apply_v_range_update(
        self, target_min: float, target_max: float
    ) -> tuple[float, float, float, float, bool]:
        """Update running distributional support with smoothing and limits."""

        target_min = float(target_min)
        target_max = float(target_max)
        old_min = float(self.running_v_min)
        old_max = float(self.running_v_max)

        if target_max <= target_min:
            span = max(abs(target_max - target_min), 1e-6)
            center = 0.5 * (target_max + target_min)
            target_min = center - 0.5 * span
            target_max = center + 0.5 * span

        if not self.v_range_initialized:
            self.running_v_min = target_min
            self.running_v_max = target_max
            self.v_range_initialized = True
            self.policy.update_atoms(self.running_v_min, self.running_v_max)
            return old_min, old_max, self.running_v_min, self.running_v_max, True

        alpha = float(self.v_range_ema_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        ema_min = float((1.0 - alpha) * self.running_v_min + alpha * target_min)
        ema_max = float((1.0 - alpha) * self.running_v_max + alpha * target_max)
        candidate_min = min(ema_min, target_min)
        candidate_max = max(ema_max, target_max)

        candidate_min, candidate_max = self._limit_v_range_step(
            self.running_v_min, self.running_v_max, candidate_min, candidate_max
        )

        allow_shrink = bool(getattr(self, "_allow_v_range_shrink", True))
        old_span = float(old_max - old_min)
        if not allow_shrink and math.isfinite(old_span) and old_span > 0.0:
            candidate_span = float(candidate_max - candidate_min)
            if candidate_span < old_span - 1e-9:
                candidate_min = old_min
                candidate_max = old_max

        if candidate_max <= candidate_min:
            span = max(abs(candidate_max - candidate_min), 1e-6)
            center = 0.5 * (candidate_max + candidate_min)
            candidate_min = center - 0.5 * span
            candidate_max = center + 0.5 * span

        changed = (
            not math.isclose(candidate_min, self.running_v_min, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(candidate_max, self.running_v_max, rel_tol=0.0, abs_tol=1e-12)
        )

        self.running_v_min = float(candidate_min)
        self.running_v_max = float(candidate_max)

        if changed:
            self.policy.update_atoms(self.running_v_min, self.running_v_max)

        return old_min, old_max, self.running_v_min, self.running_v_max, changed

    def _extract_rms_stats(
        self, rms: Optional[RunningMeanStd]
    ) -> Optional[tuple[float, float, float]]:
        if rms is None:
            return None
        count = float(rms.count)
        if not math.isfinite(count) or count <= 1e-3:
            return None
        mean = float(np.asarray(rms.mean).reshape(-1)[0])
        var = float(np.asarray(rms.var).reshape(-1)[0])
        if not math.isfinite(mean):
            mean = 0.0
        if not math.isfinite(var) or var < 0.0:
            var = 0.0
        return mean, var, count

    def _is_value_scale_frame_stable(
        self, ret_abs_p95: float, explained_var: float
    ) -> bool:
        """Check if return statistics look stable enough to update scaling."""

        if not math.isfinite(ret_abs_p95):
            return False

        if self._value_scale_stability_max_abs_p95 is not None:
            if ret_abs_p95 > self._value_scale_stability_max_abs_p95:
                return False

        if self._value_scale_stability_min_ev is not None:
            if not math.isfinite(explained_var):
                return False
            if explained_var < self._value_scale_stability_min_ev:
                return False

        return True

    def _summarize_recent_return_stats(
        self,
        sample_mean: float,
        sample_var: float,
        sample_weight: float,
        *,
        buffer: Optional[deque[tuple[float, float, float]]] = None,
        inplace: bool = True,
    ) -> tuple[float, float, float, deque[tuple[float, float, float]]]:
        sample_var = max(float(sample_var), 0.0)
        sample_weight = max(float(sample_weight), 0.0)
        if buffer is None:
            buffer = self._value_scale_recent_stats
        if self._value_scale_window_updates <= 0:
            return sample_mean, sample_var, sample_weight, buffer

        if inplace:
            target_buffer = buffer
        else:
            target_buffer = deque(
                buffer, maxlen=self._value_scale_window_updates or None
            )

        if sample_weight > 0.0:
            target_buffer.append((sample_mean, sample_var, sample_weight))

        if not target_buffer:
            return sample_mean, sample_var, sample_weight, target_buffer

        total_weight = float(sum(entry[2] for entry in target_buffer))
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            return sample_mean, sample_var, 0.0, target_buffer

        mean_weighted = float(
            sum(entry[0] * entry[2] for entry in target_buffer) / total_weight
        )
        second_weighted = float(
            sum((entry[1] + entry[0] * entry[0]) * entry[2] for entry in target_buffer)
            / total_weight
        )
        var_weighted = max(second_weighted - mean_weighted * mean_weighted, 0.0)
        return mean_weighted, var_weighted, total_weight, target_buffer

    def _apply_return_stats_ema(
        self,
        sample_mean: float,
        sample_var: float,
        sample_weight: float,
        *,
        base_mean: Optional[float] = None,
        base_second: Optional[float] = None,
        base_initialized: Optional[bool] = None,
    ) -> tuple[float, float, float, bool]:
        sample_var = max(float(sample_var), 0.0)
        sample_weight = max(float(sample_weight), 0.0)
        sample_second = sample_var + sample_mean * sample_mean
        if base_mean is None:
            base_mean = float(self._value_scale_stats_mean)
        if base_second is None:
            base_second = float(self._value_scale_stats_second)
        if base_initialized is None:
            base_initialized = bool(self._value_scale_stats_initialized)

        if sample_weight <= 0.0:
            var_existing = max(base_second - base_mean * base_mean, 0.0)
            return base_mean, var_existing, base_second, base_initialized

        if not base_initialized:
            new_mean = float(sample_mean)
            new_second = float(sample_second)
            new_var = max(new_second - new_mean * new_mean, 0.0)
            return new_mean, new_var, new_second, True

        beta = float(self._value_scale_ema_beta)
        one_minus = 1.0 - beta
        new_mean = float(one_minus * base_mean + beta * sample_mean)
        new_second = float(one_minus * base_second + beta * sample_second)
        new_var = max(new_second - new_mean * new_mean, 0.0)
        return new_mean, new_var, new_second, True

    def _finalize_return_stats(self) -> None:
        """Commit accumulated return statistics after an optimisation step."""

        if not hasattr(self, "running_v_min") or not hasattr(self, "running_v_max"):
            clip = float(getattr(self, "ret_clip", 1.0))
            self.running_v_min = -clip
            self.running_v_max = clip
            self.v_range_initialized = False

        base_scale = float(self.value_target_scale)
        base_scale_safe = base_scale if abs(base_scale) > 1e-8 else 1.0
        returns_tensor = torch.as_tensor(
            self.rollout_buffer.returns, device=self.device, dtype=torch.float32
        ).flatten()
        returns_raw_tensor = returns_tensor * float(base_scale_safe)

        warmup_limit = int(getattr(self, "_value_scale_warmup_limit", 3))
        min_samples = int(getattr(self, "_value_scale_min_samples", 256))
        frozen = bool(getattr(self, "_value_scale_frozen", False))
        warmup_active = (not frozen) and (self._value_scale_update_count < warmup_limit)

        if warmup_active and returns_raw_tensor.numel() > 0:
            returns_np = returns_raw_tensor.detach().cpu().numpy().astype(np.float32)
            finite_mask = np.isfinite(returns_np)
            if np.any(finite_mask):
                buffer = self._value_scale_warmup_buffer
                buffer.extend(returns_np[finite_mask].tolist())
                buffer_limit = int(getattr(self, "_value_scale_warmup_buffer_limit", 65536))
                if len(buffer) > buffer_limit:
                    del buffer[:-buffer_limit]

        sample_count = len(self._value_scale_warmup_buffer)
        sample_ready = warmup_active and sample_count >= min_samples

        before_mean = float(self._ret_mean_value)
        before_std = max(float(self._ret_std_value), self._value_scale_std_floor)
        before_scale_effective = float(self._value_target_scale_effective)
        prev_v_min = float(self.running_v_min)
        prev_v_max = float(self.running_v_max)

        prev_v_min_unscaled = prev_v_min * before_std + before_mean
        prev_v_max_unscaled = prev_v_max * before_std + before_mean

        self.logger.record("train/value_scale_mean_before", float(before_mean))
        self.logger.record("train/value_scale_std_before", float(before_std))
        self.logger.record("train/value_target_scale_before", float(before_scale_effective))
        if not self._use_quantile_value:
            self.logger.record("train/value_scale_vmin_before", float(prev_v_min_unscaled))
            self.logger.record("train/value_scale_vmax_before", float(prev_v_max_unscaled))

        update_applied = False
        running_v_min_unscaled = prev_v_min_unscaled
        running_v_max_unscaled = prev_v_max_unscaled
        new_mean = before_mean
        new_std = before_std

        block_samples = 1 if (warmup_active and not sample_ready) else 0
        block_freeze = 1 if (not warmup_active and frozen) else 0
        block_stability = 0

        if self.normalize_returns:
            pending_rms = self._pending_rms
            if warmup_active and pending_rms is not None and returns_raw_tensor.numel() > 0:
                with torch.no_grad():
                    pending_values = (
                        returns_raw_tensor.detach().cpu().numpy().astype(np.float64)
                    )
                    finite_mask = np.isfinite(pending_values)
                    if np.any(finite_mask):
                        pending_rms.update(pending_values[finite_mask])

            rms_ready = (
                warmup_active
                and pending_rms is not None
                and pending_rms.count >= float(min_samples)
            )

            if warmup_active and sample_ready and rms_ready and pending_rms is not None:
                sample_stats = self._extract_rms_stats(pending_rms)
                if sample_stats is not None:
                    sample_mean, sample_var, sample_weight = sample_stats
                    (
                        blended_mean,
                        blended_var,
                        blended_weight,
                        updated_buffer,
                    ) = self._summarize_recent_return_stats(
                        sample_mean,
                        sample_var,
                        sample_weight,
                        inplace=True,
                    )
                    self._value_scale_recent_stats = updated_buffer

                    (
                        target_mean,
                        target_var,
                        target_second,
                        target_initialized,
                    ) = self._apply_return_stats_ema(
                        blended_mean, blended_var, blended_weight
                    )
                    self._value_scale_stats_mean = target_mean
                    self._value_scale_stats_second = target_second
                    self._value_scale_stats_initialized = target_initialized

                    proposed_mean = float(target_mean)
                    proposed_std = max(math.sqrt(max(target_var, 0.0)), self._value_scale_std_floor)
                    new_mean = self._limit_mean_step(
                        before_mean, proposed_mean, proposed_std
                    )
                    new_std = self._limit_std_step(before_std, proposed_std)
                    new_std = max(new_std, self._value_scale_std_floor)

                    self._ret_mean_value = float(new_mean)
                    self._ret_std_value = float(new_std)
                    self._ret_mean_snapshot = float(new_mean)
                    self._ret_std_snapshot = float(new_std)

                    denom = max(
                        self.ret_clip * new_std,
                        self.ret_clip * self._value_scale_std_floor,
                    )
                    self._value_target_scale_effective = float(1.0 / denom)
                    self._value_target_scale_robust = 1.0

                    running_v_min_unscaled = self.running_v_min * new_std + new_mean
                    running_v_max_unscaled = self.running_v_max * new_std + new_mean

                    update_applied = True
                    self._value_scale_update_count += 1
                    if self._value_scale_update_count >= warmup_limit:
                        self._value_scale_frozen = True
                        self._value_scale_warmup_buffer.clear()

                    self.ret_rms.mean[...] = float(new_mean)
                    self.ret_rms.var[...] = float(new_std * new_std)
                    self.ret_rms.count = max(float(self.ret_rms.count), 1.0)

            if not update_applied:
                self._ret_mean_snapshot = float(self._ret_mean_value)
                self._ret_std_snapshot = max(
                    float(self._ret_std_value), self._value_scale_std_floor
                )

            self._pending_ret_mean = float(self._ret_mean_snapshot)
            self._pending_ret_std = float(self._ret_std_snapshot)
            self._pending_rms = None
        else:
            if warmup_active and sample_ready:
                abs_buffer = np.abs(
                    np.asarray(self._value_scale_warmup_buffer, dtype=np.float32)
                )
                finite_mask = np.isfinite(abs_buffer)
                if np.any(finite_mask):
                    robust_scale_value = float(
                        np.nanquantile(abs_buffer[finite_mask], 0.99)
                    )
                else:
                    robust_scale_value = 1.0
                if not math.isfinite(robust_scale_value) or robust_scale_value <= 0.0:
                    robust_scale_value = 1.0
                robust_scale_value = float(max(robust_scale_value, 1e-6))
                self._value_target_scale_robust = robust_scale_value
                effective_scale = float(self.value_target_scale) / robust_scale_value
                effective_scale = float(min(max(effective_scale, 1e-3), 1e3))
                self._value_target_scale_effective = effective_scale
                if self._value_clip_limit_unscaled is not None:
                    self._value_clip_limit_scaled = (
                        self._value_clip_limit_unscaled * self._value_target_scale_effective
                    )

                update_applied = True
                self._value_scale_update_count += 1
                if self._value_scale_update_count >= warmup_limit:
                    self._value_scale_frozen = True
                    self._value_scale_warmup_buffer.clear()

            self._pending_ret_mean = None
            self._pending_ret_std = None
            self._pending_rms = None

        self.logger.record(
            "train/value_scale_update_block_warmup",
            int(warmup_active and not sample_ready),
        )

        self.logger.record("train/value_scale_mean_next", float(new_mean))
        self.logger.record("train/value_scale_std_next", float(new_std))
        if not self._use_quantile_value:
            self.logger.record("train/value_scale_vmin_next", float(running_v_min_unscaled))
            self.logger.record("train/value_scale_vmax_next", float(running_v_max_unscaled))
        self.logger.record("train/value_scale_mean_after", float(new_mean))
        self.logger.record("train/value_scale_std_after", float(new_std))
        if not self._use_quantile_value:
            self.logger.record("train/value_scale_vmin_after", float(running_v_min_unscaled))
            self.logger.record("train/value_scale_vmax_after", float(running_v_max_unscaled))
        self.logger.record(
            "train/value_target_scale_after", float(self._value_target_scale_effective)
        )
        self.logger.record(
            "train/value_scale_update_applied", 1 if update_applied else 0
        )
        self.logger.record("train/value_scale_update_count", float(self._value_scale_update_count))
        self.logger.record("train/value_scale_update_block_samples", float(block_samples))
        self.logger.record("train/value_scale_update_block_freeze", float(block_freeze))
        self.logger.record("train/value_scale_update_block_stability", float(block_stability))

        if self._value_scale_update_count >= warmup_limit:
            self._value_scale_frozen = True
        self.logger.record("train/value_scale_frozen", float(self._value_scale_frozen))

    def __init__(
        self,
        policy: Union[str, Type[RecurrentActorCriticPolicy]],
        env: Union[VecEnv, str],
        cql_alpha: float = 1.0,
        cql_beta: float = 5.0,
        cvar_alpha: float = 0.05,
        cvar_weight: float = 0.5,
        cvar_cap: Optional[float] = None,
        cvar_winsor_pct: float = 0.1,
        cvar_ema_beta: float = 0.9,
        cvar_use_constraint: bool = False,
        cvar_limit: float = -2.0,
        cvar_lambda_lr: float = 1e-2,
        cvar_use_penalty: bool = True,
        cvar_penalty_cap: float = 0.7,
        v_range_ema_alpha: float = 0.01,
        vf_coef_warmup: Optional[float] = None,
        vf_coef_warmup_updates: int = 0,
        vf_bad_explained_scale: float = 1.0,
        vf_bad_explained_floor: float = 0.0,
        bad_explained_patience: int = 2,
        entropy_boost_factor: float = 1.5,
        entropy_boost_cap: Optional[float] = None,
        clip_range_warmup: Optional[float] = None,
        clip_range_warmup_updates: int = 8,
        critic_grad_warmup_updates: int = 0,
        cvar_activation_threshold: float = 0.25,
        cvar_activation_hysteresis: float = 0.05,
        cvar_ramp_updates: int = 4,
        value_target_scale: Union[str, float, None] = 1.0,
        normalize_returns: bool = True,
        ret_clip: float = 10.0,
        enable_kl_diagnostics: bool = True,
        bc_warmup_steps: int = 0,
        bc_decay_steps: int = 0,
        bc_final_coef: Optional[float] = None,
        ent_coef_final: Optional[float] = None,
        ent_coef_min: float = 5e-4,
        ent_coef_decay_steps: int = 0,
        ent_coef_plateau_window: int = 0,
        ent_coef_plateau_tolerance: float = 0.0,
        ent_coef_plateau_min_updates: int = 0,
        target_kl: Optional[float] = None,
        kl_early_stop: bool = True,
        kl_epoch_decay: float = 0.5,
        kl_exceed_stop_fraction: float = 0.25,
        kl_early_stop_use_ema: bool = True,
        kl_ema_updates: int = 10,
        kl_ema_alpha: Optional[float] = None,
        kl_consec_minibatches: int = 0,
        kl_penalty_beta: float = 0.0,
        kl_penalty_beta_min: float = 0.0,
        kl_penalty_beta_max: float = 0.1,
        kl_penalty_pid_kp: float = 0.0,
        kl_penalty_pid_ki: float = 0.0,
        kl_penalty_pid_kd: float = 0.0,
        ppo_clip_range: Optional[float] = None,
        use_torch_compile: bool = False,
        microbatch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        loss_head_weights: Optional[Mapping[str, Union[float, bool]]] = None,
        optimizer_lr_min: Optional[float] = None,
        scheduler_min_lr: Optional[float] = None,
        optimizer_lr_max: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self._last_lstm_states: Optional[RNNStates | tuple[torch.Tensor, ...]] = None
        self._last_rollout_entropy: float = 0.0
        self._last_rollout_entropy_raw: float = 0.0
        self._update_calls: int = 0
        self._global_update_step: int = 0
        self._loss_head_weights: Optional[dict[str, float]] = None
        self._kl_diag = bool(enable_kl_diagnostics)
        self._optimizer_lr_floor_warned = False

        kwargs_local = dict(kwargs)

        optimizer_lr_min_candidate = kwargs_local.pop("optimizer_lr_min", optimizer_lr_min)
        if optimizer_lr_min_candidate is None:
            optimizer_lr_min_value = 1e-5
        else:
            optimizer_lr_min_value = float(optimizer_lr_min_candidate)
            if not math.isfinite(optimizer_lr_min_value) or optimizer_lr_min_value < 0.0:
                raise ValueError("'optimizer_lr_min' must be a non-negative finite value")

        scheduler_min_lr_candidate = kwargs_local.pop("scheduler_min_lr", scheduler_min_lr)
        if scheduler_min_lr_candidate is None:
            scheduler_min_lr_value = float(optimizer_lr_min_value)
        else:
            scheduler_min_lr_value = float(scheduler_min_lr_candidate)
            if not math.isfinite(scheduler_min_lr_value) or scheduler_min_lr_value < 0.0:
                raise ValueError("'scheduler_min_lr' must be a non-negative finite value")

        optimizer_lr_max_candidate = kwargs_local.pop("optimizer_lr_max", optimizer_lr_max)
        if optimizer_lr_max_candidate is None:
            optimizer_lr_max_value = float("inf")
        else:
            optimizer_lr_max_value = float(optimizer_lr_max_candidate)
            if optimizer_lr_max_value <= 0.0:
                raise ValueError("'optimizer_lr_max' must be positive when provided")
            if not math.isfinite(optimizer_lr_max_value):
                optimizer_lr_max_value = float("inf")

        if math.isfinite(optimizer_lr_max_value) and optimizer_lr_max_value < optimizer_lr_min_value:
            optimizer_lr_max_value = optimizer_lr_min_value

        if scheduler_min_lr_value < optimizer_lr_min_value:
            scheduler_min_lr_value = optimizer_lr_min_value
        else:
            optimizer_lr_min_value = scheduler_min_lr_value

        if math.isfinite(optimizer_lr_max_value) and optimizer_lr_max_value < optimizer_lr_min_value:
            optimizer_lr_max_value = optimizer_lr_min_value

        self._optimizer_lr_min = float(optimizer_lr_min_value)
        self._optimizer_lr_max = float(optimizer_lr_max_value)
        self._scheduler_min_lr = float(scheduler_min_lr_value)

        if ppo_clip_range is not None:
            clip_range_candidate = ppo_clip_range
        else:
            clip_range_candidate = kwargs_local.pop("clip_range", 0.05)
        clip_range_value = float(clip_range_candidate)
        if not math.isfinite(clip_range_value) or clip_range_value <= 0.0:
            raise ValueError("'ppo_clip_range' must be a positive finite value")

        clip_range_warmup_candidate = kwargs_local.pop("clip_range_warmup", clip_range_warmup)
        if clip_range_warmup_candidate is None:
            clip_range_warmup_value = max(clip_range_value, 0.2)
        else:
            clip_range_warmup_value = float(clip_range_warmup_candidate)
            if not math.isfinite(clip_range_warmup_value) or clip_range_warmup_value <= 0.0:
                raise ValueError("'clip_range_warmup' must be a positive finite value")
        clip_range_warmup_updates_value = max(
            0, int(kwargs_local.pop("clip_range_warmup_updates", clip_range_warmup_updates))
        )

        vf_coef_target_value = float(kwargs_local.get("vf_coef", kwargs.get("vf_coef", 0.5)))
        vf_coef_warmup_candidate = kwargs_local.pop("vf_coef_warmup", vf_coef_warmup)
        if vf_coef_warmup_candidate is None:
            vf_coef_warmup_value = float(vf_coef_target_value)
        else:
            vf_coef_warmup_value = float(vf_coef_warmup_candidate)
            if not math.isfinite(vf_coef_warmup_value) or vf_coef_warmup_value <= 0.0:
                raise ValueError("'vf_coef_warmup' must be a positive finite value")
        kwargs_local["vf_coef"] = vf_coef_target_value
        vf_coef_warmup_updates_value = max(
            0, int(kwargs_local.pop("vf_coef_warmup_updates", vf_coef_warmup_updates))
        )
        vf_bad_explained_scale_value = float(
            kwargs_local.pop("vf_bad_explained_scale", vf_bad_explained_scale)
        )
        if vf_bad_explained_scale_value < 0.0:
            raise ValueError("'vf_bad_explained_scale' must be non-negative")
        vf_bad_explained_floor_value = float(
            kwargs_local.pop("vf_bad_explained_floor", vf_bad_explained_floor)
        )
        if vf_bad_explained_floor_value < 0.0:
            raise ValueError("'vf_bad_explained_floor' must be non-negative")
        bad_explained_patience_value = max(
            0, int(kwargs_local.pop("bad_explained_patience", bad_explained_patience))
        )

        entropy_boost_factor_value = float(
            kwargs_local.pop("entropy_boost_factor", entropy_boost_factor)
        )
        if entropy_boost_factor_value < 1.0:
            entropy_boost_factor_value = 1.0
        entropy_boost_cap_candidate = kwargs_local.pop("entropy_boost_cap", entropy_boost_cap)

        critic_grad_warmup_updates_value = max(
            0, int(kwargs_local.pop("critic_grad_warmup_updates", critic_grad_warmup_updates))
        )

        cvar_activation_threshold_value = float(
            kwargs_local.pop("cvar_activation_threshold", cvar_activation_threshold)
        )
        if cvar_activation_threshold_value < 0.0:
            raise ValueError("'cvar_activation_threshold' must be non-negative")
        cvar_activation_hysteresis_value = float(
            kwargs_local.pop("cvar_activation_hysteresis", cvar_activation_hysteresis)
        )
        if cvar_activation_hysteresis_value < 0.0:
            raise ValueError("'cvar_activation_hysteresis' must be non-negative")
        cvar_ramp_updates_value = max(
            0, int(kwargs_local.pop("cvar_ramp_updates", cvar_ramp_updates))
        )

        kwargs_local["clip_range"] = clip_range_value

        if target_kl is not None:
            target_kl_candidate = target_kl
        else:
            target_kl_candidate = kwargs_local.pop("target_kl", 0.5)
        target_kl_value: Optional[float]
        if target_kl_candidate is None:
            target_kl_value = None
        else:
            target_kl_value = float(target_kl_candidate)
            if not math.isfinite(target_kl_value) or target_kl_value < 0.0:
                raise ValueError("'target_kl' must be non-negative and finite when provided")
        if target_kl_value is None:
            kwargs_local["target_kl"] = None
        else:
            kwargs_local["target_kl"] = target_kl_value

        self.cql_alpha = float(cql_alpha)
        self.cql_beta = float(cql_beta)
        self.cvar_alpha = float(cvar_alpha)
        if not math.isfinite(self.cvar_alpha) or not (0.0 < self.cvar_alpha <= 1.0):
            raise ValueError("'cvar_alpha' must be a finite probability in the interval (0, 1]")
        self.cvar_weight = float(cvar_weight)
        if cvar_cap is not None and cvar_cap <= 0.0:
            raise ValueError("'cvar_cap' must be positive when provided")
        self.cvar_cap = float(cvar_cap) if cvar_cap is not None else None

        self.cvar_use_constraint = bool(
            kwargs_local.pop("cvar_use_constraint", cvar_use_constraint)
        )
        self.cvar_limit = float(kwargs_local.pop("cvar_limit", cvar_limit))
        self.cvar_lambda_lr = float(kwargs_local.pop("cvar_lambda_lr", cvar_lambda_lr))
        if self.cvar_lambda_lr < 0.0:
            raise ValueError("'cvar_lambda_lr' must be non-negative")
        self.cvar_use_penalty = bool(
            kwargs_local.pop("cvar_use_penalty", cvar_use_penalty)
        )
        self.cvar_penalty_cap = float(
            kwargs_local.pop("cvar_penalty_cap", cvar_penalty_cap)
        )
        if self.cvar_penalty_cap < 0.0:
            raise ValueError("'cvar_penalty_cap' must be non-negative")

        winsor_candidate = float(kwargs_local.pop("cvar_winsor_pct", cvar_winsor_pct))
        self.cvar_winsor_pct = winsor_candidate
        self.cvar_ema_beta = float(kwargs_local.pop("cvar_ema_beta", cvar_ema_beta))
        if not math.isfinite(self.cvar_ema_beta) or not (0.0 <= self.cvar_ema_beta < 1.0):
            raise ValueError("'cvar_ema_beta' must be in [0, 1)")

        self._cvar_empirical_ema: Optional[float] = None
        self._cvar_violation_ema: Optional[float] = None
        self._last_rollout_reward_raw: Optional[np.ndarray] = None
        self._last_rollout_reward_costs: Optional[np.ndarray] = None
        self._last_rollout_clip_bounds: Optional[np.ndarray] = None
        self._last_rollout_clip_hard_caps: Optional[np.ndarray] = None
        self._last_rollout_clip_bounds_min: Optional[float] = None
        self._last_rollout_clip_bounds_median: Optional[float] = None
        self._last_rollout_clip_bounds_max: Optional[float] = None
        self._last_rollout_clip_cap_fraction: Optional[float] = None
        self._reward_robust_clip_fraction: Optional[float] = None

        self.v_range_ema_alpha = float(v_range_ema_alpha)
        if not (0.0 < self.v_range_ema_alpha <= 1.0):
            raise ValueError("'v_range_ema_alpha' must be in (0, 1]")

        self._vf_coef_target = vf_coef_target_value
        self._vf_coef_warmup = vf_coef_warmup_value
        self._vf_coef_warmup_updates = vf_coef_warmup_updates_value
        self._bad_explained_patience = bad_explained_patience_value
        self._bad_explained_counter = 0
        self._last_explained_variance: Optional[float] = None

        self._entropy_boost_factor = entropy_boost_factor_value
        self._entropy_boost_cap_candidate = entropy_boost_cap_candidate

        self._clip_range_base = clip_range_value
        self._clip_range_warmup = max(clip_range_warmup_value, clip_range_value)
        self._clip_range_warmup_updates = clip_range_warmup_updates_value
        self._clip_range_current = self._clip_range_warmup

        self._critic_grad_warmup_updates = critic_grad_warmup_updates_value
        self._critic_grad_blocked = critic_grad_warmup_updates_value > 0
        self._critic_grad_block_logged_state: Optional[bool] = None

        self._cvar_weight_target = float(self.cvar_weight)
        self._cvar_activation_threshold = cvar_activation_threshold_value
        self._cvar_activation_hysteresis = cvar_activation_hysteresis_value
        self._cvar_ramp_updates = cvar_ramp_updates_value
        self._cvar_ramp_progress = 0
        self._current_cvar_weight = 0.0
        self._cvar_lambda = 0.0
        self.cvar_lambda = 0.0

        self.value_target_scale = self._coerce_value_target_scale(value_target_scale)
        normalize_returns_sentinel: object = object()
        normalize_returns_kwarg = kwargs_local.pop(
            "normalize_returns", normalize_returns_sentinel
        )
        if normalize_returns_kwarg is not normalize_returns_sentinel:
            normalize_returns_kwarg_bool = bool(normalize_returns_kwarg)
            normalize_returns_param_bool = bool(normalize_returns)
            if normalize_returns_kwarg_bool != normalize_returns_param_bool:
                raise TypeError(
                    "'normalize_returns' was provided both explicitly and via kwargs; "
                    "pass it only once"
                )
            normalize_returns_value = normalize_returns_kwarg_bool
        else:
            normalize_returns_value = bool(normalize_returns)
        ret_clip_candidate = kwargs_local.pop("ret_clip", ret_clip)
        ret_clip_value = float(ret_clip_candidate)
        if not math.isfinite(ret_clip_value) or ret_clip_value <= 0.0:
            raise ValueError("'ret_clip' must be a positive finite value")
        self.normalize_returns = normalize_returns_value
        self.ret_clip = ret_clip_value
        self.ret_rms = RunningMeanStd(shape=())
        self._ret_mean_value = 0.0
        self._ret_std_value = 1.0
        self._ret_mean_snapshot = 0.0
        self._ret_std_snapshot = 1.0
        self._pending_rms: Optional[RunningMeanStd] = None
        self._pending_ret_mean: Optional[float] = None
        self._pending_ret_std: Optional[float] = None
        value_scale_cfg = kwargs_local.pop("value_scale", None)
        value_scale_ema_beta = None
        value_scale_max_rel_step = None
        value_scale_std_floor = None
        value_scale_window_updates = None
        value_scale_warmup_updates = None
        value_scale_freeze_after = None
        value_scale_range_max_rel_step = None
        value_scale_stability_cfg_raw: Optional[Mapping[str, Any]] = None
        value_scale_stability_patience = None
        if isinstance(value_scale_cfg, Mapping):
            value_scale_ema_beta = value_scale_cfg.get("ema_beta")
            value_scale_max_rel_step = value_scale_cfg.get("max_rel_step")
            value_scale_std_floor = value_scale_cfg.get("std_floor")
            value_scale_window_updates = value_scale_cfg.get("window_updates")
            value_scale_warmup_updates = value_scale_cfg.get("warmup_updates")
            value_scale_freeze_after = value_scale_cfg.get("freeze_after")
            value_scale_range_max_rel_step = value_scale_cfg.get("range_max_rel_step")
            stability_candidate = value_scale_cfg.get("stability")
            if isinstance(stability_candidate, Mapping):
                value_scale_stability_cfg_raw = stability_candidate
            value_scale_stability_patience = value_scale_cfg.get("stability_patience")
        elif value_scale_cfg is not None:
            value_scale_ema_beta = getattr(value_scale_cfg, "ema_beta", None)
            value_scale_max_rel_step = getattr(value_scale_cfg, "max_rel_step", None)
            value_scale_std_floor = getattr(value_scale_cfg, "std_floor", None)
            value_scale_window_updates = getattr(value_scale_cfg, "window_updates", None)
            value_scale_warmup_updates = getattr(value_scale_cfg, "warmup_updates", None)
            value_scale_freeze_after = getattr(value_scale_cfg, "freeze_after", None)
            value_scale_range_max_rel_step = getattr(
                value_scale_cfg, "range_max_rel_step", None
            )
            stability_candidate = getattr(value_scale_cfg, "stability", None)
            if isinstance(stability_candidate, Mapping):
                value_scale_stability_cfg_raw = stability_candidate
            value_scale_stability_patience = getattr(
                value_scale_cfg, "stability_patience", None
            )

        if value_scale_ema_beta is None:
            value_scale_ema_beta = kwargs_local.pop("value_scale_ema_beta", 0.2)
        if value_scale_max_rel_step is None:
            value_scale_max_rel_step = kwargs_local.pop("value_scale_max_rel_step", None)
        else:
            kwargs_local.pop("value_scale_max_rel_step", None)
        if value_scale_max_rel_step is None:
            raise ValueError("'value_scale.max_rel_step' must be provided")
        if value_scale_std_floor is None:
            value_scale_std_floor = kwargs_local.pop("value_scale_std_floor", 1e-2)
        if value_scale_window_updates is None:
            value_scale_window_updates = kwargs_local.pop("value_scale_window_updates", 0)
        if value_scale_warmup_updates is None:
            value_scale_warmup_updates = kwargs_local.pop("value_scale_warmup_updates", 0)
        else:
            kwargs_local.pop("value_scale_warmup_updates", None)
        if value_scale_freeze_after is None:
            value_scale_freeze_after = kwargs_local.pop("value_scale_freeze_after", None)
        else:
            kwargs_local.pop("value_scale_freeze_after", None)
        if value_scale_range_max_rel_step is None:
            value_scale_range_max_rel_step = kwargs_local.pop(
                "value_scale_range_max_rel_step", None
            )
        else:
            kwargs_local.pop("value_scale_range_max_rel_step", None)
        if value_scale_stability_cfg_raw is None:
            stability_raw_alt = kwargs_local.pop("value_scale_stability", None)
            if isinstance(stability_raw_alt, Mapping):
                value_scale_stability_cfg_raw = stability_raw_alt
        else:
            kwargs_local.pop("value_scale_stability", None)
        if value_scale_stability_patience is None:
            value_scale_stability_patience = kwargs_local.pop(
                "value_scale_stability_patience", None
            )
        else:
            kwargs_local.pop("value_scale_stability_patience", None)

        self._value_scale_ema_beta = float(value_scale_ema_beta)
        if not (0.0 < self._value_scale_ema_beta <= 1.0):
            raise ValueError("'value_scale.ema_beta' must be in (0, 1]")
        self._value_scale_max_rel_step = float(value_scale_max_rel_step)
        if self._value_scale_max_rel_step < 0.0:
            raise ValueError("'value_scale.max_rel_step' must be non-negative")
        self._value_scale_std_floor = max(float(value_scale_std_floor), 1e-8)
        self._value_scale_window_updates = int(value_scale_window_updates)
        if self._value_scale_window_updates < 0:
            raise ValueError("'value_scale.window_updates' must be non-negative")
        self._value_scale_recent_stats: deque[tuple[float, float, float]] = deque(
            maxlen=self._value_scale_window_updates or None
        )
        self._value_scale_stats_initialized = False
        self._value_scale_stats_mean = 0.0
        self._value_scale_stats_second = 1.0
        self._value_target_scale_effective = float(self.value_target_scale)
        self._value_target_scale_robust = 1.0
        self._value_scale_warmup_limit = 3
        self._value_scale_min_samples = 256
        self._value_scale_warmup_buffer: list[float] = []
        self._value_scale_warmup_buffer_limit = 65536
        self._value_clip_limit_scaled: Optional[float] = None
        self._value_scale_warmup_updates = int(self._value_scale_warmup_limit)
        self._value_scale_freeze_after = int(self._value_scale_warmup_limit)
        if value_scale_range_max_rel_step is None:
            range_step_value = 0.15
        else:
            range_step_value = float(value_scale_range_max_rel_step)
            if not math.isfinite(range_step_value):
                raise ValueError("'value_scale.range_max_rel_step' must be finite")
            if range_step_value < 0.0:
                raise ValueError("'value_scale.range_max_rel_step' must be non-negative")
        range_step_value = float(max(range_step_value, 0.0))
        range_step_value = min(range_step_value, float(self._value_scale_max_rel_step))
        self._value_scale_range_max_rel_step = float(range_step_value)

        stability_min_ev = None
        stability_max_p95 = None
        stability_patience_value = None
        if value_scale_stability_cfg_raw is not None:
            stability_min_ev_candidate = value_scale_stability_cfg_raw.get(
                "min_explained_variance"
            )
            if stability_min_ev_candidate is None:
                stability_min_ev_candidate = value_scale_stability_cfg_raw.get(
                    "ev_min"
                )
            if stability_min_ev_candidate is not None:
                stability_min_ev = float(stability_min_ev_candidate)
            stability_max_p95_candidate = value_scale_stability_cfg_raw.get(
                "max_abs_p95"
            )
            if stability_max_p95_candidate is None:
                stability_max_p95_candidate = value_scale_stability_cfg_raw.get(
                    "ret_abs_p95_max"
                )
            if stability_max_p95_candidate is not None:
                stability_max_p95 = float(stability_max_p95_candidate)
            stability_patience_candidate = value_scale_stability_cfg_raw.get(
                "patience"
            )
            if stability_patience_candidate is None:
                stability_patience_candidate = value_scale_stability_cfg_raw.get(
                    "consecutive"
                )
            if stability_patience_candidate is not None:
                stability_patience_value = int(stability_patience_candidate)

        if value_scale_stability_patience is not None:
            stability_patience_value = int(value_scale_stability_patience)

        if stability_min_ev is not None and not math.isfinite(stability_min_ev):
            stability_min_ev = None
        if stability_max_p95 is not None and stability_max_p95 <= 0.0:
            stability_max_p95 = None
        self._value_scale_stability_min_ev = stability_min_ev
        self._value_scale_stability_max_abs_p95 = stability_max_p95
        if stability_patience_value is None:
            stability_patience_value = 0
        self._value_scale_stability_patience = max(0, int(stability_patience_value))
        self._value_scale_requires_stability = (
            self._value_scale_stability_min_ev is not None
            or self._value_scale_stability_max_abs_p95 is not None
            or self._value_scale_stability_patience > 0
        )
        self._value_scale_update_count = 0
        self._value_scale_frame_stable = True
        self._value_scale_stable_counter = 0
        self._value_scale_latest_ret_abs_p95 = 0.0
        self._value_scale_frozen = False

        super().__init__(policy=policy, env=env, **kwargs_local)

        self._rebuild_scheduler_if_needed()

        self._use_quantile_value = bool(
            getattr(self.policy, "uses_quantile_value_head", False)
        )
        self._quantile_huber_kappa = float(
            getattr(self.policy, "quantile_huber_kappa", 1.0)
        )

        self._ensure_score_action_space()

        # Stable-Baselines3 lazily initialises the internal logger, but older
        # versions may skip creating ``self._logger`` when ``logger=None`` is
        # passed through the constructor.  Accessing :pyattr:`self.logger`
        # would then raise ``AttributeError``.  Guard against this scenario by
        # ensuring the attribute exists before we start recording metrics.
        if not hasattr(self, "_logger") or self._logger is None:  # pragma: no cover - safety guard
            from stable_baselines3.common.logger import configure

            self._logger = configure()

        # Early debug diagnostics ensure configuration mismatches are visible in logs.
        self.logger.record("debug/vf_coef_target", float(self._vf_coef_target))
        self.logger.record("debug/vf_coef_warmup", float(self._vf_coef_warmup))

        self.logger.record("debug/value_scale_ema_beta", float(self._value_scale_ema_beta))
        self.logger.record(
            "debug/value_scale_max_rel_step", float(self._value_scale_max_rel_step)
        )
        self.logger.record("debug/value_scale_std_floor", float(self._value_scale_std_floor))
        self.logger.record(
            "debug/value_scale_window_updates", float(self._value_scale_window_updates)
        )
        self.logger.record(
            "debug/value_scale_warmup_updates", float(self._value_scale_warmup_updates)
        )
        if self._value_scale_freeze_after is not None:
            self.logger.record(
                "debug/value_scale_freeze_after", float(self._value_scale_freeze_after)
            )
        self.logger.record(
            "debug/value_range_max_rel_step",
            float(self._value_scale_range_max_rel_step),
        )
        if self._value_scale_stability_min_ev is not None:
            self.logger.record(
                "debug/value_scale_stability_min_ev",
                float(self._value_scale_stability_min_ev),
            )
        if self._value_scale_stability_max_abs_p95 is not None:
            self.logger.record(
                "debug/value_scale_stability_max_abs_p95",
                float(self._value_scale_stability_max_abs_p95),
            )
        self.logger.record(
            "debug/value_scale_stability_patience",
            float(self._value_scale_stability_patience),
        )
        self.logger.record("debug/value_scale_ret_count_virtualized", 1.0)

        self.logger.record("debug/patch_tag", 1.0)
        self.logger.record("debug/loaded_from", 1.0)

        self.logger.record("config/optimizer_lr_min", float(self._optimizer_lr_min))
        if math.isfinite(self._optimizer_lr_max):
            self.logger.record("config/optimizer_lr_max", float(self._optimizer_lr_max))
        self.logger.record("config/scheduler_min_lr", float(self._scheduler_min_lr))

        base_lr = float(self.lr_schedule(1.0))
        value_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for name, param in self.policy.named_parameters():
            if not param.requires_grad:
                continue
            target = (
                value_params
                if (
                    "value" in name
                    or "value_net" in name
                    or "critic" in name
                    or "v_head" in name
                )
                else other_params
            )
            target.append(param)

        param_groups: list[dict[str, Any]] = []
        if other_params:
            param_groups.append(
                {
                    "params": other_params,
                    "lr": base_lr,
                    "initial_lr": base_lr,
                    "_lr_scale": 1.0,
                }
            )
        if value_params:
            value_lr_scale = 2.0
            param_groups.append(
                {
                    "params": value_params,
                    "lr": base_lr * value_lr_scale,
                    "initial_lr": base_lr * value_lr_scale,
                    "_lr_scale": value_lr_scale,
                }
            )
        if param_groups:
            self.policy.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=0.0,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            self.logger.record(
                "train/optimizer_lr_groups",
                [float(group.get("lr", 0.0)) for group in param_groups],
            )

        base_logger = getattr(self, "_logger", None)

        self._configure_loss_head_weights(loss_head_weights)

        self._configure_gradient_accumulation(
            microbatch_size=microbatch_size,
            grad_steps=gradient_accumulation_steps,
        )

        self.running_v_min = 0.0
        self.running_v_max = 0.0
        self.v_range_initialized = False

        # --- Value clip limit wiring -----------------------------------------
        # По умолчанию политика выставляет value_clip_limit = max(|v_min|,|v_max|).
        # При normalize_returns=True нам нужен ТОЛЬКО нормализованный клип ±ret_clip.
        # Поэтому полностью отключаем «raw-clip» в этом режиме.
        clip_limit_unscaled = getattr(self.policy, "value_clip_limit", None)
        self._value_clip_limit_unscaled: Optional[float] = None
        self._value_clip_limit_scaled: Optional[float] = None
        if not self.normalize_returns and clip_limit_unscaled is not None:
            clip_limit_unscaled_f = float(clip_limit_unscaled)
            if clip_limit_unscaled_f <= 0.0 or not math.isfinite(clip_limit_unscaled_f):
                raise ValueError(
                    f"Invalid 'value_clip_limit' for distributional value head: {clip_limit_unscaled}"
                )
            self._value_clip_limit_unscaled = clip_limit_unscaled_f
            self._value_clip_limit_scaled = (
                clip_limit_unscaled_f * self._value_target_scale_effective
            )
        # Явно обнуляем лимит в политике (если атрибут есть), чтобы ничего не «воскресало».
        try:
            if self.normalize_returns:
                setattr(self.policy, "value_clip_limit", None)
        except Exception:
            pass
        # Диагностика
        self.logger.record("debug/raw_value_clip_enabled", 0.0 if self.normalize_returns else 1.0)
        self._value_target_raw_outlier_warn_threshold = 0.01
        self._last_raw_outlier_frac = 0.0
        self._allow_v_range_shrink = True

        self.bc_warmup_steps = max(0, int(bc_warmup_steps))
        self.bc_decay_steps = max(0, int(bc_decay_steps))
        self.bc_initial_coef = float(self.cql_alpha)
        self.bc_final_coef = float(bc_final_coef) if bc_final_coef is not None else 0.0
        self._current_bc_coef = float(self.bc_initial_coef)

        ent_coef_min_value = float(ent_coef_min)
        if not math.isfinite(ent_coef_min_value) or ent_coef_min_value < 0.0:
            raise ValueError("'ent_coef_min' must be a non-negative finite value")
        self.ent_coef_min = ent_coef_min_value

        initial_ent_coef_raw = float(self.ent_coef)
        if initial_ent_coef_raw < self.ent_coef_min:
            self.ent_coef = float(self.ent_coef_min)
        self._ent_coef_last_raw = float(initial_ent_coef_raw)
        self._ent_coef_last_clamped = float(self.ent_coef)
        self._ent_coef_last_clamp_applied = bool(
            self._ent_coef_last_clamped > initial_ent_coef_raw + 1e-12
        )

        self.ent_coef_initial = float(self.ent_coef)
        self.ent_coef_final = (
            float(ent_coef_final) if ent_coef_final is not None else float(self.ent_coef_initial)
        )
        if self._entropy_boost_cap_candidate is None:
            self._entropy_boost_cap = float(self.ent_coef_initial * 5.0)
        else:
            cap_value = float(self._entropy_boost_cap_candidate)
            if cap_value <= 0.0 or not math.isfinite(cap_value):
                raise ValueError("'entropy_boost_cap' must be positive when provided")
            self._entropy_boost_cap = cap_value
        self.ent_coef_decay_steps = max(0, int(ent_coef_decay_steps))
        self.entropy_plateau_window = max(0, int(ent_coef_plateau_window))
        self.entropy_plateau_tolerance = float(ent_coef_plateau_tolerance)
        self.entropy_plateau_min_updates = max(0, int(ent_coef_plateau_min_updates))
        self._entropy_window: Optional[deque[tuple[int, float]]] = (
            deque(maxlen=self.entropy_plateau_window) if self.entropy_plateau_window > 0 else None
        )
        self._entropy_plateau = False
        self._entropy_decay_start_update: Optional[int] = None
        self._last_entropy_slope = 0.0

        if self._entropy_window is None and self.ent_coef_decay_steps > 0:
            self._entropy_decay_start_update = 0

        self.lr_scheduler = getattr(self.policy, "optimizer_scheduler", None)

        if use_torch_compile and self.device.type == "cuda":
            self.policy = torch.compile(self.policy, mode="reduce-overhead")

        # --- KL-adaptive training controls ----------------------------------------------------
        self.kl_epoch_decay = float(kl_epoch_decay)
        if not (0.0 < self.kl_epoch_decay <= 1.0):
            raise ValueError("'kl_epoch_decay' must be in (0, 1]")
        self._kl_min_lr = float(self._scheduler_min_lr)
        self._kl_lr_scale = 1.0
        self._base_lr_schedule = self.lr_schedule

        def _scaled_lr_schedule(progress_remaining: float) -> float:
            base_lr = self._base_lr_schedule(progress_remaining)
            return float(max(base_lr, self._kl_min_lr))

        self.lr_schedule = _scaled_lr_schedule
        self.kl_early_stop = bool(kl_early_stop)
        self._kl_early_stop_use_ema = bool(kl_early_stop_use_ema)
        ema_updates_value = max(1, int(kl_ema_updates))
        self._kl_ema_window = int(ema_updates_value)
        if kl_ema_alpha is None:
            self._kl_ema_alpha: Optional[float] = None
        else:
            alpha_value = float(kl_ema_alpha)
            if not math.isfinite(alpha_value) or not (0.0 < alpha_value <= 1.0):
                raise ValueError("'kl_ema_alpha' must be within (0, 1]")
            self._kl_ema_alpha = alpha_value
        self._kl_consec_minibatches = max(0, int(kl_consec_minibatches))
        self._base_n_epochs = max(1, int(self.n_epochs))
        self._kl_epoch_factor = 1.0
        self._kl_epoch_factor_min = 1.0 / float(self._base_n_epochs)
        self._kl_base_param_lrs: list[float] = []
        self._refresh_kl_base_lrs()

        self._kl_exceed_stop_fraction = float(kl_exceed_stop_fraction)
        if not (0.0 <= self._kl_exceed_stop_fraction <= 1.0):
            raise ValueError("'kl_exceed_stop_fraction' must be within [0, 1]")
        # Alias used by logging and update routines.
        self.kl_exceed_stop_fraction = self._kl_exceed_stop_fraction

        beta_min = max(0.0, float(kl_penalty_beta_min))
        beta_max = max(beta_min, float(kl_penalty_beta_max))
        beta_initial = float(kl_penalty_beta)
        if not math.isfinite(beta_initial):
            raise ValueError("'kl_penalty_beta' must be finite")
        beta_initial = min(max(beta_initial, beta_min), beta_max)
        kp = float(kl_penalty_pid_kp)
        if not math.isfinite(kp) or kp < 0.0:
            kp = 0.0
        ki = float(kl_penalty_pid_ki)
        if not math.isfinite(ki) or ki < 0.0:
            ki = 0.0
        kd = float(kl_penalty_pid_kd)
        if not math.isfinite(kd) or kd < 0.0:
            kd = 0.0
        self.kl_beta = beta_initial
        self.kl_penalty_beta_min = beta_min
        self.kl_penalty_beta_max = beta_max
        self.kl_penalty_kp = kp
        self.kl_penalty_ki = ki
        self.kl_penalty_kd = kd
        self._kl_err_int = 0.0
        self._kl_err_prev = 0.0
        self._kl_penalty_error = 0.0
        self._kl_pid_p = 0.0
        self._kl_pid_i = 0.0
        self._kl_pid_d = 0.0

        self._fixed_clip_range = clip_range_value
        self.clip_range = lambda _: self._compute_clip_range_value()
        self.target_kl = target_kl_value

        self.normalize_advantage = True

        atoms = max(1, int(getattr(self.policy, "num_atoms", 1)))
        ce_norm = math.log(float(atoms))
        self._critic_ce_normalizer = ce_norm if ce_norm > 1e-6 else 1.0
        self.logger.record("debug/critic_ce_normalizer", float(self._critic_ce_normalizer))

        policy_block_fn = getattr(self.policy, "set_critic_gradient_blocked", None)
        if callable(policy_block_fn):
            policy_block_fn(self._critic_grad_blocked)

    @property
    def kl_exceed_stop_fraction(self) -> float:
        """Return the configured KL exceed stop fraction."""

        return self._kl_exceed_stop_fraction

    @kl_exceed_stop_fraction.setter
    def kl_exceed_stop_fraction(self, value: float) -> None:
        self._kl_exceed_stop_fraction = float(value)

    def _update_learning_rate(self, optimizer: Optional[torch.optim.Optimizer]) -> None:
        if optimizer is None:
            return

        # Если у политики уже есть внешний шедулер (например, OneCycleLR),
        # не пытаемся перезаписывать lr статическим расписанием SB3.
        external_scheduler = getattr(self.policy, "lr_scheduler", None)
        if external_scheduler is None:
            external_scheduler = getattr(self, "lr_scheduler", None)
        if external_scheduler is not None:
            base_schedule = getattr(self, "_base_lr_schedule", None)
            base_value = None
            if callable(base_schedule):
                try:
                    base_value = float(base_schedule(self._current_progress_remaining))
                except Exception:
                    base_value = None
            if base_value is None:
                groups = getattr(optimizer, "param_groups", [])
                if groups:
                    base_value = float(groups[0].get("lr", 0.0))
            if base_value is not None:
                self.logger.record("train/learning_rate", float(base_value))
            self._enforce_optimizer_lr_bounds(log_values=False, warn_on_floor=True)
            return

        base_lr = float(self.lr_schedule(self._current_progress_remaining))
        self.logger.record("train/learning_rate", base_lr)

        min_lr = float(getattr(self, "_kl_min_lr", 0.0))
        group_lrs: list[float] = []
        for group in optimizer.param_groups:
            scale = float(group.get("_lr_scale", 1.0))
            scaled_lr = base_lr * scale
            if min_lr > 0.0:
                scaled_lr = max(scaled_lr, min_lr)
            group["lr"] = scaled_lr
            group.setdefault("initial_lr", scaled_lr)
            group_lrs.append(scaled_lr)

        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "base_lrs"):
            self.lr_scheduler.base_lrs = list(group_lrs)

        self._enforce_optimizer_lr_bounds(log_values=False, warn_on_floor=True)

    def _compute_clip_range_value(self, update_index: Optional[int] = None) -> float:
        idx = self._update_calls if update_index is None else max(0, int(update_index))
        if self._clip_range_warmup_updates <= 0:
            return float(self._clip_range_base)
        progress = min(1.0, idx / float(max(1, self._clip_range_warmup_updates)))
        warmup = float(self._clip_range_warmup)
        base = float(self._clip_range_base)
        value = warmup + (base - warmup) * progress
        return float(max(value, 1e-6))

    def _compute_vf_coef_value(self, update_index: int) -> float:
        base = float(self._vf_coef_target)
        last_ev = self._last_explained_variance
        if last_ev is None:
            return base
        return base

    def _compute_entropy_boost(self, nominal_ent_coef: float) -> float:
        if self._bad_explained_counter <= max(0, self._bad_explained_patience):
            return float(nominal_ent_coef)
        counter = self._bad_explained_counter - self._bad_explained_patience
        boosted = nominal_ent_coef * (self._entropy_boost_factor ** float(counter))
        return float(min(boosted, self._entropy_boost_cap))

    def _compute_cvar_weight(self) -> float:
        last_ev = self._last_explained_variance
        if last_ev is None:
            self._cvar_ramp_progress = 0
            return 0.0
        if self._value_scale_update_count < self._value_scale_warmup_updates:
            self._cvar_ramp_progress = 0
            return 0.0
        if self._value_scale_requires_stability:
            patience = max(1, self._value_scale_stability_patience)
            if not self._value_scale_frame_stable:
                self._cvar_ramp_progress = 0
                return 0.0
            if self._value_scale_stable_counter < patience:
                self._cvar_ramp_progress = 0
                return 0.0
        threshold = self._cvar_activation_threshold
        hysteresis = self._cvar_activation_hysteresis
        if last_ev + hysteresis < threshold:
            self._cvar_ramp_progress = 0
            return 0.0
        if self._cvar_ramp_updates <= 0:
            self._cvar_ramp_progress = 0
            weight = float(self._cvar_weight_target)
            return float(min(weight, 1.0))
        self._cvar_ramp_progress = min(
            self._cvar_ramp_progress + 1, self._cvar_ramp_updates
        )
        ramp = self._cvar_ramp_progress / float(max(1, self._cvar_ramp_updates))
        weight = float(self._cvar_weight_target * ramp)
        return float(min(weight, 1.0))

    def _update_critic_gradient_block(self, update_index: int) -> None:
        should_block = update_index < self._critic_grad_warmup_updates
        if should_block == self._critic_grad_blocked:
            return

        policy_block_fn = getattr(self.policy, "set_critic_gradient_blocked", None)
        if callable(policy_block_fn):
            policy_block_fn(should_block)

        self._critic_grad_blocked = should_block
        if self._critic_grad_block_logged_state != should_block:
            self._critic_grad_block_logged_state = should_block
            self.logger.record("debug/critic_grad_block_switch", float(should_block))

    def _configure_gradient_accumulation(
        self,
        microbatch_size: Optional[int],
        grad_steps: Optional[int],
    ) -> None:
        batch_size = int(self.batch_size)
        if batch_size <= 0:
            raise ValueError("'batch_size' must be positive to configure gradient accumulation")

        grad_steps_local = None if grad_steps is None else int(grad_steps)
        micro_size_local = None if microbatch_size is None else int(microbatch_size)

        if grad_steps_local is not None and grad_steps_local <= 0:
            raise ValueError("'gradient_accumulation_steps' must be a positive integer")
        if micro_size_local is not None and micro_size_local <= 0:
            raise ValueError("'microbatch_size' must be a positive integer")

        if grad_steps_local is None and micro_size_local is None:
            micro_size_local = batch_size
            grad_steps_local = 1
        elif grad_steps_local is None:
            if batch_size % micro_size_local != 0:
                raise ValueError("'microbatch_size' must evenly divide batch_size")
            grad_steps_local = batch_size // micro_size_local
        elif micro_size_local is None:
            if batch_size % grad_steps_local != 0:
                raise ValueError("'gradient_accumulation_steps' must evenly divide batch_size")
            micro_size_local = batch_size // grad_steps_local
        else:
            if micro_size_local * grad_steps_local != batch_size:
                if batch_size % micro_size_local != 0 or batch_size // micro_size_local != grad_steps_local:
                    raise ValueError(
                        "microbatch_size * gradient_accumulation_steps must equal batch_size"
                    )

        self._microbatch_size = int(micro_size_local)
        self._grad_accumulation_steps = int(grad_steps_local)


    def _configure_loss_head_weights(
        self, weights_cfg: Optional[Mapping[str, Union[float, bool]]]
    ) -> None:
        setter = getattr(self.policy, "set_loss_head_weights", None)
        if weights_cfg is None:
            self._loss_head_weights = None
            if callable(setter):
                setter(None)
            return

        normalized: dict[str, float] = {}
        for head_name, raw_value in weights_cfg.items():
            if raw_value is None:
                continue
            if isinstance(raw_value, bool):
                normalized[head_name] = 1.0 if raw_value else 0.0
                continue
            try:
                normalized[head_name] = float(raw_value)
            except (TypeError, ValueError):
                continue

        if not normalized:
            self._loss_head_weights = None
            if callable(setter):
                setter(None)
            return

        self._loss_head_weights = normalized
        if callable(setter):
            setter(normalized)
        for head_name, weight in normalized.items():
            self.logger.record(f"config/loss_head_weight_{head_name}", float(weight))

    def _rebuild_scheduler_if_needed(self) -> None:
        """Ensure the policy scheduler tracks the current optimizer instance."""

        policy = getattr(self, "policy", None)
        if policy is None:
            return

        optimizer = getattr(policy, "optimizer", None)
        if optimizer is None:
            return

        scheduler = getattr(policy, "lr_scheduler", None)
        if scheduler is None:
            scheduler = getattr(policy, "optimizer_scheduler", None)

        scheduler_fn = getattr(policy, "optimizer_scheduler_fn", None)
        need_rebuild = bool(scheduler_fn is not None) and (
            scheduler is None or getattr(scheduler, "optimizer", None) is not optimizer
        )

        if not need_rebuild:
            if scheduler is not None:
                setattr(self, "lr_scheduler", scheduler)
            return

        policy.optimizer_scheduler = scheduler_fn(optimizer)
        policy.lr_scheduler = policy.optimizer_scheduler
        setattr(self, "lr_scheduler", policy.lr_scheduler)

        if hasattr(policy.lr_scheduler, "base_lrs"):
            policy.lr_scheduler.base_lrs = [
                float(group.get("initial_lr", group.get("lr", 0.0)))
                for group in optimizer.param_groups
            ]


    def _refresh_kl_base_lrs(self) -> None:
        """Cache optimiser base LRs before KL scaling is applied."""

        optimizer = getattr(self.policy, "optimizer", None)
        if optimizer is None:
            self._kl_base_param_lrs = []
            return

        scale = float(getattr(self, "_kl_lr_scale", 1.0))
        if scale <= 0.0:
            scale = 1.0

        base_lrs: list[float] = []
        for group in getattr(optimizer, "param_groups", []):
            current_lr = float(group.get("lr", 0.0))
            base_lr = current_lr / scale if scale != 0.0 else current_lr
            group["_kl_base_lr"] = base_lr
            base_lrs.append(base_lr)

        self._kl_base_param_lrs = base_lrs

    def _apply_epoch_decay(self) -> float:
        """Shrink effective epoch multiplier respecting minimum of one epoch."""

        previous_factor = self._kl_epoch_factor
        proposed_factor = previous_factor * self.kl_epoch_decay
        proposed_factor = min(proposed_factor, 1.0)
        if proposed_factor < self._kl_epoch_factor_min:
            self._kl_epoch_factor = self._kl_epoch_factor_min
        else:
            self._kl_epoch_factor = proposed_factor

        if previous_factor <= 0.0:
            return 1.0
        return self._kl_epoch_factor / previous_factor

    def _kl_integral_limit(self) -> float:
        """Return symmetric bounds for the KL error integrator."""

        if self.kl_penalty_ki > 0.0:
            span = self.kl_penalty_beta_max - self.kl_penalty_beta_min
            limit = span / max(self.kl_penalty_ki, 1e-12)
            if not math.isfinite(limit) or limit <= 0.0:
                return 0.0
            return float(limit)
        return 0.0

    def _adjust_kl_penalty(self, observed_kl: float) -> None:
        """Adapt the KL penalty strength using a PID regulator."""

        if self.target_kl is None or self.target_kl <= 0.0:
            return

        if not math.isfinite(observed_kl):
            return

        error = float(observed_kl) - float(self.target_kl)
        if not math.isfinite(error):
            return

        self._kl_penalty_error = error
        self._kl_err_int += error

        # --- Anti-windup for integral term ------------------------------------
        if self.kl_penalty_ki > 0.0:
            i_cap = (self.kl_penalty_beta_max - self.kl_penalty_beta_min) / max(
                self.kl_penalty_ki, 1e-12
            )
            # symmetric clamp to keep I-term from pushing beta beyond limits
            self._kl_err_int = float(min(max(self._kl_err_int, -i_cap), i_cap))
        else:
            # disable integral accumulation when I-term is off
            self._kl_err_int = 0.0
        p_term = self.kl_penalty_kp * error
        i_term = self.kl_penalty_ki * self._kl_err_int
        d_term = self.kl_penalty_kd * (error - self._kl_err_prev)
        self._kl_err_prev = error

        candidate_beta = self.kl_beta + p_term + i_term + d_term
        if not math.isfinite(candidate_beta):
            candidate_beta = self.kl_penalty_beta_max

        candidate_beta = min(max(candidate_beta, self.kl_penalty_beta_min), self.kl_penalty_beta_max)
        self.kl_beta = candidate_beta
        self._kl_pid_p = p_term
        self._kl_pid_i = i_term
        self._kl_pid_d = d_term

        # Back-calculation: soften integral when beta saturates against error direction.
        sat_min = self.kl_beta <= (self.kl_penalty_beta_min + 1e-12)
        sat_max = self.kl_beta >= (self.kl_penalty_beta_max - 1e-12)
        if (sat_min and error < 0.0) or (sat_max and error > 0.0):
            self._kl_err_int *= 0.9

        self.logger.record("train/kl_penalty_beta", float(self.kl_beta))
        self.logger.record("train/kl_penalty_error", float(self._kl_penalty_error))
        self.logger.record("train/kl_penalty_pid_p", float(self._kl_pid_p))
        self.logger.record("train/kl_penalty_pid_i", float(self._kl_pid_i))
        self.logger.record("train/kl_penalty_pid_d", float(self._kl_pid_d))

    def _handle_kl_divergence(self, approx_kl: float) -> tuple[float, float]:
        """React to KL overshoot by shrinking future epoch budget."""

        self._kl_lr_scale = 1.0
        self._refresh_kl_base_lrs()
        self._enforce_optimizer_lr_bounds(log_values=False, warn_on_floor=True)
        epoch_decay = self._apply_epoch_decay()
        self.logger.record("train/kl_last_exceeded", approx_kl)
        self.logger.record("train/kl_lr_decay_applied", 0.0)
        self.logger.record("train/kl_lr_scale", 1.0)
        self.logger.record("train/kl_epoch_decay_applied", epoch_decay)
        return 0.0, epoch_decay

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return self.policy.parameters(recurse)

    def _update_bc_coef(self) -> float:
        if self.bc_decay_steps <= 0:
            self._current_bc_coef = self.bc_initial_coef
            return self._current_bc_coef

        timesteps_after_warmup = max(0, self.num_timesteps - self.bc_warmup_steps)
        if timesteps_after_warmup <= 0:
            self._current_bc_coef = self.bc_initial_coef
            return self._current_bc_coef

        decay_progress = min(1.0, timesteps_after_warmup / max(1, self.bc_decay_steps))
        self._current_bc_coef = (
            self.bc_initial_coef + (self.bc_final_coef - self.bc_initial_coef) * decay_progress
        )
        return self._current_bc_coef

    def _update_ent_coef(self, update_index: int) -> float:
        if self.ent_coef_decay_steps <= 0:
            raw_value = float(self.ent_coef_final)
        elif self._entropy_decay_start_update is None:
            raw_value = float(self.ent_coef_initial)
        else:
            steps_since_start = max(0, update_index - self._entropy_decay_start_update)
            progress = min(1.0, steps_since_start / float(self.ent_coef_decay_steps))
            raw_value = float(
                self.ent_coef_initial + (self.ent_coef_final - self.ent_coef_initial) * progress
            )

        clamped_value = float(max(raw_value, self.ent_coef_min))
        self._ent_coef_last_raw = float(raw_value)
        self._ent_coef_last_clamped = float(clamped_value)
        self._ent_coef_last_clamp_applied = bool(clamped_value > raw_value + 1e-12)
        self.ent_coef = clamped_value
        return self.ent_coef

    def _maybe_update_entropy_schedule(self, update_index: int, avg_entropy: float) -> None:
        if self._entropy_window is None:
            self._last_entropy_slope = 0.0
            if self._entropy_decay_start_update is None and self.ent_coef_decay_steps > 0:
                self._entropy_decay_start_update = 0
            return

        self._entropy_window.append((update_index, avg_entropy))
        if len(self._entropy_window) < 2:
            self._last_entropy_slope = 0.0
            return

        xs = torch.tensor([item[0] for item in self._entropy_window], dtype=torch.float32)
        ys = torch.tensor([item[1] for item in self._entropy_window], dtype=torch.float32)
        xs_centered = xs - xs.mean()
        ys_centered = ys - ys.mean()
        denom = torch.sum(xs_centered * xs_centered).item()
        if denom <= 1e-8:
            self._last_entropy_slope = 0.0
        else:
            self._last_entropy_slope = float(torch.sum(xs_centered * ys_centered).item() / (denom + 1e-8))

        window_filled = len(self._entropy_window) == self.entropy_plateau_window
        ready_for_decay = (
            self.ent_coef_decay_steps > 0
            and self._entropy_decay_start_update is None
            and window_filled
            and update_index >= self.entropy_plateau_min_updates
        )
        if not ready_for_decay:
            return

        if abs(self._last_entropy_slope) <= self.entropy_plateau_tolerance and not self._entropy_plateau:
            self._entropy_plateau = True
            self._entropy_decay_start_update = update_index

    def _kl_diag_step(self, dist: torch.distributions.Distribution, rollout_data) -> None:
        if not hasattr(self, "logger"):
            return

        with torch.no_grad():
            inner = getattr(dist, "distribution", None)
            if inner is None:
                inner = dist

            mean_new = getattr(inner, "mean", None)
            if callable(mean_new):
                mean_new = mean_new()
            if mean_new is None:
                mean_new = getattr(dist, "mean_actions", None)
                if callable(mean_new):
                    mean_new = mean_new()

            std_new = getattr(inner, "stddev", None)
            if callable(std_new):
                std_new = std_new()
            if std_new is None:
                get_std = getattr(inner, "get_std", None)
                std_new = get_std() if callable(get_std) else None

            scores = rollout_data.actions.to(device=self.device, dtype=torch.float32)
            raw_actions = rollout_data.actions_raw.to(device=self.device, dtype=torch.float32)

            def _coerce(value: Any) -> Optional[torch.Tensor]:
                if value is None:
                    return None
                if torch.is_tensor(value):
                    return value.to(device=scores.device, dtype=scores.dtype)
                return torch.as_tensor(value, device=scores.device, dtype=scores.dtype)

            mean_tensor = _coerce(mean_new)
            std_tensor = _coerce(std_new)
            std_safe: Optional[torch.Tensor]
            if std_tensor is not None:
                std_safe = torch.clamp(std_tensor, min=1e-6)
            else:
                std_safe = None

            def _quantiles(values: Optional[torch.Tensor], probs: Sequence[float]) -> torch.Tensor:
                if values is None:
                    return torch.full(
                        (len(probs),),
                        float("nan"),
                        device=scores.device,
                        dtype=torch.float32,
                    )
                flat = values.reshape(-1).to(dtype=torch.float32)
                finite = flat[torch.isfinite(flat)]
                if finite.numel() == 0:
                    return torch.full(
                        (len(probs),),
                        float("nan"),
                        device=scores.device,
                        dtype=torch.float32,
                    )
                q = torch.tensor(probs, device=scores.device, dtype=torch.float32)
                return torch.quantile(finite, q)

            z_abs: Optional[torch.Tensor] = None
            if mean_tensor is not None and std_safe is not None:
                z = (raw_actions - mean_tensor) / std_safe
                z_abs = z.abs()

            z_stats = _quantiles(z_abs, (0.5, 0.9, 1.0)) if z_abs is not None else _quantiles(None, (0.5, 0.9, 1.0))
            z_values = z_stats.detach().cpu().tolist()
            self.logger.record("diag/z_abs_p50", float(z_values[0]))
            self.logger.record("diag/z_abs_p90", float(z_values[1]))
            self.logger.record("diag/z_abs_max", float(z_values[2]))

            sigma_stats = (
                _quantiles(std_safe, (0.1, 0.5)) if std_safe is not None else _quantiles(None, (0.1, 0.5))
            )
            sigma_values = sigma_stats.detach().cpu().tolist()
            self.logger.record("diag/sigma_new_p10", float(sigma_values[0]))
            self.logger.record("diag/sigma_new_p50", float(sigma_values[1]))

            edge_mask = (scores < 0.02) | (scores > 0.98)
            edge_frac = edge_mask.float().mean() if edge_mask.numel() > 0 else scores.new_tensor(float("nan"))
            self.logger.record("diag/score_edge_frac", float(edge_frac.item()))

            raw_lp_new = self.policy._log_prob_raw_only(dist, raw_actions).reshape(-1)

            old_lp_raw = rollout_data.old_log_prob_raw.reshape(-1)
            kl_raw = old_lp_raw - raw_lp_new
            kl_stats = _quantiles(kl_raw, (0.5, 0.9, 1.0)).detach().cpu().tolist()
            self.logger.record("diag/kl_raw_p50", float(kl_stats[0]))
            self.logger.record("diag/kl_raw_p90", float(kl_stats[1]))
            self.logger.record("diag/kl_raw_max", float(kl_stats[2]))

            finite_mask = torch.isfinite(kl_raw)
            if finite_mask.any():
                exceed_frac = (kl_raw[finite_mask] > 0.1).float().mean()
                self.logger.record("diag/kl_raw_exceed_frac@0.1", float(exceed_frac.item()))
            else:
                self.logger.record("diag/kl_raw_exceed_frac@0.1", float("nan"))

            raw_actions_flat = raw_actions.reshape(-1)
            if raw_actions_flat.numel() > 0:
                mu_old = raw_actions_flat.mean()
                sigma_old = raw_actions_flat.std(unbiased=True)
                if not torch.isfinite(sigma_old):
                    sigma_old = raw_actions_flat.std(unbiased=False)
                if not torch.isfinite(sigma_old):
                    sigma_old = raw_actions_flat.new_tensor(1e-6)
            else:
                mu_old = raw_actions.new_tensor(0.0)
                sigma_old = raw_actions.new_tensor(1e-6)
            sigma_old = torch.clamp(sigma_old, min=1e-6)

            if mean_tensor is not None and std_safe is not None:
                std_ratio = std_safe / sigma_old
                kl_gauss = torch.log(std_ratio) + (
                    (sigma_old**2 + (mu_old - mean_tensor) ** 2) / (2 * std_safe**2)
                ) - 0.5
                kl_gauss_stats = _quantiles(kl_gauss, (0.5, 0.9)).detach().cpu().tolist()
                self.logger.record("diag/kl_gauss_p50", float(kl_gauss_stats[0]))
                self.logger.record("diag/kl_gauss_p90", float(kl_gauss_stats[1]))

                mean_term = ((mu_old - mean_tensor) ** 2) / (2 * std_safe**2)
                var_ratio = (sigma_old**2) / (std_safe**2)
                var_term = (sigma_old**2) / (2 * std_safe**2) - 0.5 - 0.5 * torch.log(var_ratio)
                self.logger.record(
                    "diag/kl_mean_term_p90",
                    float(_quantiles(mean_term, (0.9,)).detach().cpu().numpy()[0]),
                )
                self.logger.record(
                    "diag/kl_var_term_p90",
                    float(_quantiles(var_term, (0.9,)).detach().cpu().numpy()[0]),
                )
            else:
                nan_stats = _quantiles(None, (0.5, 0.9)).detach().cpu().tolist()
                self.logger.record("diag/kl_gauss_p50", float(nan_stats[0]))
                self.logger.record("diag/kl_gauss_p90", float(nan_stats[1]))
                self.logger.record("diag/kl_mean_term_p90", float("nan"))
                self.logger.record("diag/kl_var_term_p90", float("nan"))

    def _record_raw_policy_metrics(
        self,
        avg_policy_entropy_raw: float,
        entropy_raw_count: int,
        kl_raw_sum: float,
        kl_raw_count: int,
        raw_z_clip_count: float = 0.0,
        raw_z_total: int = 0,
    ) -> None:
        if entropy_raw_count > 0:
            self.logger.record("train/policy_entropy_raw", float(avg_policy_entropy_raw))
        if kl_raw_count > 0:
            approx_kl_raw_mean = kl_raw_sum / float(kl_raw_count)
            self.logger.record("train/approx_kl_raw", float(approx_kl_raw_mean))
        if raw_z_total > 0:
            fraction = float(raw_z_clip_count) / float(raw_z_total)
            self.logger.record("train/raw_z_clip_fraction", fraction)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RecurrentRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        if not isinstance(rollout_buffer, RecurrentRolloutBuffer):
            raise TypeError(
                "DistributionalPPO requires a RecurrentRolloutBuffer for recurrent policies"
            )
        assert self._last_obs is not None, "No previous observation was provided"

        if self._last_lstm_states is None:
            init_states = self.policy.recurrent_initial_state
            self._last_lstm_states = self._clone_states_to_device(init_states, self.device)

        self.policy.set_training_mode(False)
        self._ensure_score_action_space()

        vec_normalize_env: Optional[VecNormalize] = None
        for candidate_env in (env, getattr(self, "env", None)):
            if candidate_env is None:
                continue
            if isinstance(candidate_env, VecNormalize):
                vec_normalize_env = candidate_env
            else:
                try:
                    vec_normalize_env = unwrap_vec_normalize(candidate_env)
                except ValueError:
                    vec_normalize_env = None
            if vec_normalize_env is not None:
                break

        if vec_normalize_env is not None and getattr(vec_normalize_env, "norm_reward", False):
            raise AssertionError("VecNormalize reward normalization must be disabled to recover raw ΔPnL.")

        entropy_loss_total = 0.0
        entropy_loss_count = 0
        entropy_raw_sum = 0.0
        entropy_raw_count = 0

        n_steps = 0
        self._activate_return_scale_snapshot()
        rollout_buffer.reset()
        callback.on_rollout_start()

        buffer_size = rollout_buffer.buffer_size
        n_envs = env.num_envs
        reward_raw_buffer = np.zeros((buffer_size, n_envs), dtype=np.float32)
        reward_costs_buffer = np.full((buffer_size, n_envs), np.nan, dtype=np.float32)
        clip_bound_buffer = np.full((buffer_size, n_envs), np.nan, dtype=np.float32)
        clip_cap_buffer = np.full((buffer_size, n_envs), np.nan, dtype=np.float32)
        base_reward_scale = self._resolve_value_scale_safe()

        while n_steps < n_rollout_steps:
            raw_actions_tensor: Optional[torch.Tensor] = None
            old_log_prob_raw_tensor: Optional[torch.Tensor] = None

            with torch.no_grad():
                obs_tensor = self.policy.obs_to_tensor(self._last_obs)[0]
                episode_starts = torch.as_tensor(
                    self._last_episode_starts, dtype=torch.float32, device=self.device
                )
                prev_actor_states = self._extract_actor_states(self._last_lstm_states)
                actions, _, log_probs, self._last_lstm_states = self.policy.forward(
                    obs_tensor, self._last_lstm_states, episode_starts
                )
                if self._use_quantile_value:
                    value_quantiles = self.policy.last_value_quantiles
                else:
                    value_logits = self.policy.last_value_logits

                dist_output = self.policy.get_distribution(
                    obs_tensor,
                    prev_actor_states,
                    episode_starts,
                )
                dist = dist_output[0] if isinstance(dist_output, tuple) else dist_output
                inner_dist = getattr(dist, "distribution", None)
                if inner_dist is not None:
                    entropy_raw_tensor = inner_dist.entropy()
                    if entropy_raw_tensor.ndim > 1:
                        entropy_raw_tensor = entropy_raw_tensor.sum(dim=-1)
                    if torch.isfinite(entropy_raw_tensor).all():
                        entropy_raw_detached = entropy_raw_tensor.to(dtype=torch.float32)
                        entropy_raw_sum += float(entropy_raw_detached.sum().cpu().item())
                        entropy_raw_count += int(entropy_raw_detached.numel())

                if isinstance(self.action_space, gym.spaces.Box):
                    raw_actions_tensor = self.policy.last_raw_actions
                    if raw_actions_tensor is None:
                        raise RuntimeError("Policy did not cache raw actions during rollout collection")
                    raw_actions_tensor = raw_actions_tensor.to(device=self.device)
                    old_log_prob_raw_tensor = self.policy._log_prob_raw_only(
                        dist, raw_actions_tensor
                    ).unsqueeze(-1)
                else:
                    raw_actions_tensor = actions
                    if not isinstance(raw_actions_tensor, torch.Tensor):
                        raw_actions_tensor = torch.as_tensor(
                            raw_actions_tensor, dtype=torch.float32, device=self.device
                        )
                    old_log_prob_raw_tensor = log_probs

            if self._use_quantile_value:
                if value_quantiles is None:
                    raise RuntimeError("Policy did not cache value quantiles during forward pass")
                mean_values_norm = value_quantiles.mean(dim=1, keepdim=True).detach()
            else:
                if value_logits is None:
                    raise RuntimeError("Policy did not cache value logits during forward pass")
                probs = torch.softmax(value_logits, dim=1)
                mean_values_norm = (probs * self.policy.atoms).sum(dim=1, keepdim=True).detach()

            if self.normalize_returns:
                # НЕТ raw-clip: просто де-нормализация и переход в буферную шкалу
                ret_std_tensor = mean_values_norm.new_tensor(self._ret_std_snapshot)
                ret_mu_tensor = mean_values_norm.new_tensor(self._ret_mean_snapshot)
                scalar_values = (mean_values_norm * ret_std_tensor + ret_mu_tensor) / self.value_target_scale
            else:
                scalar_values = mean_values_norm
                if self._value_clip_limit_scaled is not None:
                    scalar_values = torch.clamp(
                        scalar_values,
                        min=-self._value_clip_limit_scaled,
                        max=self._value_clip_limit_scaled,
                    )
                scalar_values_raw = self._to_raw_returns(scalar_values)
                if self._value_clip_limit_unscaled is not None:
                    scalar_values_raw = torch.clamp(
                        scalar_values_raw,
                        min=-self._value_clip_limit_unscaled,
                        max=self._value_clip_limit_unscaled,
                    )
                scalar_values = scalar_values_raw / base_reward_scale  # стабильная шкала буфера (обычно =1)

            actions_np = actions.cpu().numpy()
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions_np, self.action_space.low, self.action_space.high)
            else:
                clipped_actions = actions_np

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            raw_rewards = rewards
            if vec_normalize_env is not None and hasattr(vec_normalize_env, "get_original_reward"):
                raw_rewards = vec_normalize_env.get_original_reward()
            raw_rewards = np.asarray(raw_rewards, dtype=np.float32)
            if raw_rewards.size > 0:
                self.logger.record(
                    "rollout/raw_reward_mean_in_fraction",
                    float(np.mean(raw_rewards)),
                )
                frac_gt_log10 = float(np.mean(raw_rewards > math.log(10.0)))
                self.logger.record("rollout/reward_gt_log10_fraction", frac_gt_log10)

            scaled_rewards = (
                raw_rewards / self.value_target_scale
            )  # не трогаем rollout'ы динамическим скейлом — нормализация применяется только в лоссе критика

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            step_pos = rollout_buffer.pos
            reward_raw_step = np.zeros(n_envs, dtype=np.float32)
            reward_costs_step = np.full(n_envs, np.nan, dtype=np.float32)
            clip_bound_step = np.full(n_envs, np.nan, dtype=np.float32)
            clip_hard_cap_step = np.full(n_envs, np.nan, dtype=np.float32)
            for env_idx, info in enumerate(infos):
                if env_idx < scaled_rewards.size:
                    safe_fallback = float(scaled_rewards[env_idx]) * base_reward_scale
                else:
                    safe_fallback = 0.0
                raw_value = safe_fallback
                clip_value = 0.0
                costs_value: float = float("nan")
                hard_cap_value: float = float("nan")
                if isinstance(info, Mapping):
                    candidate = info.get("reward_used_fraction")
                    if candidate is None:
                        candidate = info.get("reward_raw_fraction")
                    if candidate is not None:
                        try:
                            raw_value = float(candidate)
                        except (TypeError, ValueError):
                            raw_value = safe_fallback
                    clip_candidate = info.get("reward_clip_bound_fraction")
                    if clip_candidate is not None:
                        try:
                            clip_value = float(clip_candidate)
                        except (TypeError, ValueError):
                            clip_value = 0.0
                    hard_cap_candidate = info.get("reward_clip_hard_cap_fraction")
                    if hard_cap_candidate is not None:
                        try:
                            hard_cap_value = float(hard_cap_candidate)
                        except (TypeError, ValueError):
                            hard_cap_value = float("nan")
                    cost_candidate = info.get("reward_costs_fraction")
                    if cost_candidate is not None:
                        try:
                            costs_value = float(cost_candidate)
                        except (TypeError, ValueError):
                            costs_value = float("nan")
                    robust_candidate = info.get("reward_robust_clip_fraction")
                    if robust_candidate is not None:
                        try:
                            robust_value = float(robust_candidate)
                        except (TypeError, ValueError):
                            robust_value = float("nan")
                        if math.isfinite(robust_value) and robust_value > 0.0:
                            if self._reward_robust_clip_fraction is None:
                                self._reward_robust_clip_fraction = float(robust_value)
                if (
                    self._reward_robust_clip_fraction is not None
                    and math.isfinite(self._reward_robust_clip_fraction)
                    and self._reward_robust_clip_fraction > 0.0
                ):
                    # ``raw_value`` is already clipped inside the environment; this
                    # duplicate clip exists purely to keep logged diagnostics within
                    # the robust bound and avoid confusing out-of-range telemetry.
                    raw_value = float(
                        np.clip(
                            raw_value,
                            -self._reward_robust_clip_fraction,
                            self._reward_robust_clip_fraction,
                        )
                    )
                    if clip_value <= 0.0 or not math.isfinite(clip_value):
                        clip_value = float(self._reward_robust_clip_fraction)
                reward_raw_step[env_idx] = float(raw_value)
                reward_costs_step[env_idx] = float(costs_value)
                clip_bound_step[env_idx] = clip_value
                clip_hard_cap_step[env_idx] = hard_cap_value

            reward_raw_buffer[step_pos] = reward_raw_step
            reward_costs_buffer[step_pos] = reward_costs_step
            clip_bound_buffer[step_pos] = clip_bound_step
            clip_cap_buffer[step_pos] = clip_hard_cap_step

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                scaled_rewards,
                self._last_episode_starts,
                scalar_values.squeeze(-1),
                log_probs,
                lstm_states=self._last_lstm_states,
                actions_raw=raw_actions_tensor,
                log_prob_raw=old_log_prob_raw_tensor,
            )

            entropy_loss_total += float(-log_probs.mean().item())
            entropy_loss_count += 1

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            obs_tensor = self.policy.obs_to_tensor(new_obs)[0]
            episode_starts = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            _, _, _, _ = self.policy.forward(obs_tensor, self._last_lstm_states, episode_starts)
            if self._use_quantile_value:
                last_value_quantiles = self.policy.last_value_quantiles
            else:
                last_value_logits = self.policy.last_value_logits

        self._last_rollout_reward_raw = reward_raw_buffer.copy()
        self._last_rollout_reward_costs = reward_costs_buffer.copy()
        self._last_rollout_clip_bounds = clip_bound_buffer.copy()
        self._last_rollout_clip_hard_caps = clip_cap_buffer.copy()

        self._last_rollout_clip_bounds_min = None
        self._last_rollout_clip_bounds_median = None
        self._last_rollout_clip_bounds_max = None
        self._last_rollout_clip_cap_fraction = None
        if self._last_rollout_clip_bounds is not None:
            clip_bounds_np = np.asarray(self._last_rollout_clip_bounds, dtype=np.float32).flatten()
            finite_mask = np.isfinite(clip_bounds_np)
            if np.any(finite_mask):
                finite_bounds = clip_bounds_np[finite_mask]
                self._last_rollout_clip_bounds_min = float(np.min(finite_bounds))
                self._last_rollout_clip_bounds_median = float(np.median(finite_bounds))
                self._last_rollout_clip_bounds_max = float(np.max(finite_bounds))
            if self._last_rollout_clip_hard_caps is not None:
                hard_caps_np = np.asarray(self._last_rollout_clip_hard_caps, dtype=np.float32).flatten()
                mask = np.isfinite(clip_bounds_np) & np.isfinite(hard_caps_np)
                if np.any(mask):
                    hits = np.abs(clip_bounds_np[mask] - hard_caps_np[mask]) <= 1e-6
                    self._last_rollout_clip_cap_fraction = float(np.mean(hits.astype(np.float32)))

        if self._use_quantile_value:
            if last_value_quantiles is None:
                raise RuntimeError("Policy did not cache value quantiles during terminal forward pass")
            last_mean_norm = last_value_quantiles.mean(dim=1)
        else:
            if last_value_logits is None:
                raise RuntimeError("Policy did not cache value logits during terminal forward pass")
            last_probs = torch.softmax(last_value_logits, dim=1)
            last_mean_norm = (last_probs * self.policy.atoms).sum(dim=1)

        if self.normalize_returns:
            # НЕТ raw-clip при терминальном значении
            ret_std_tensor = last_mean_norm.new_tensor(self._ret_std_snapshot)
            ret_mu_tensor = last_mean_norm.new_tensor(self._ret_mean_snapshot)
            last_scalar_values = (last_mean_norm * ret_std_tensor + ret_mu_tensor) / self.value_target_scale
        else:
            last_scalar_scaled = last_mean_norm
            if self._value_clip_limit_scaled is not None:
                last_scalar_scaled = torch.clamp(
                    last_scalar_scaled,
                    min=-self._value_clip_limit_scaled,
                    max=self._value_clip_limit_scaled,
                )
            last_scalar_raw = self._to_raw_returns(last_scalar_scaled)
            if self._value_clip_limit_unscaled is not None:
                last_scalar_raw = torch.clamp(
                    last_scalar_raw,
                    min=-self._value_clip_limit_unscaled,
                    max=self._value_clip_limit_unscaled,
                )
            last_scalar_values = last_scalar_raw / base_reward_scale

        rollout_buffer.compute_returns_and_advantage(last_values=last_scalar_values, dones=dones)
        callback.on_rollout_end()

        if entropy_loss_count > 0:
            self._last_rollout_entropy = entropy_loss_total / float(entropy_loss_count)
            self.logger.record("rollout/policy_entropy", self._last_rollout_entropy)
        else:
            self._last_rollout_entropy = 0.0

        if entropy_raw_count > 0:
            self._last_rollout_entropy_raw = entropy_raw_sum / float(entropy_raw_count)
        else:
            self._last_rollout_entropy_raw = self._last_rollout_entropy

        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._rebuild_scheduler_if_needed()
        self._update_learning_rate(self.policy.optimizer)
        self._refresh_kl_base_lrs()
        self._ensure_score_action_space()

        current_update = self._global_update_step
        # hard-kill any warmup coming from configs/CLI
        self._critic_grad_warmup_updates = 0
        self._critic_grad_blocked = False
        policy_block_fn = getattr(self.policy, "set_critic_gradient_blocked", None)
        if callable(policy_block_fn):
            policy_block_fn(False)
        if self._critic_grad_block_logged_state is not False:
            self._critic_grad_block_logged_state = False
            self.logger.record("debug/critic_grad_block_switch", 0.0)
        # do not re-enable in this step:
        # self._update_critic_gradient_block(current_update)
        self._clip_range_current = self._compute_clip_range_value(current_update)
        clip_range = float(self._clip_range_current)
        self._update_ent_coef(current_update)
        ent_coef_raw_value = float(self._ent_coef_last_raw)
        ent_coef_nominal_value = float(self._ent_coef_last_clamped)
        ent_coef_boosted_value = float(self._compute_entropy_boost(ent_coef_nominal_value))
        ent_coef_eff_value = float(max(ent_coef_boosted_value, self.ent_coef_min))
        ent_coef_autoclamp_flag = bool(
            self._ent_coef_last_clamp_applied or ent_coef_eff_value > ent_coef_boosted_value + 1e-12
        )
        self.ent_coef = ent_coef_eff_value
        vf_coef_effective = self._compute_vf_coef_value(current_update)
        self.vf_coef = vf_coef_effective
        current_cvar_weight_nominal = float(max(self._compute_cvar_weight(), 0.0))
        if not self.cvar_use_penalty:
            current_cvar_weight_nominal = 0.0
        elif math.isfinite(self.cvar_penalty_cap):
            current_cvar_weight_nominal = float(
                min(max(current_cvar_weight_nominal, 0.0), self.cvar_penalty_cap)
            )
        current_cvar_weight_raw = float(current_cvar_weight_nominal)

        self._activate_return_scale_snapshot()

        if current_update < 3:
            self.logger.record(
                f"debug/update_{current_update}_vf_coef_effective", float(vf_coef_effective)
            )
            self.logger.record(
                f"debug/update_{current_update}_critic_ce_normalizer",
                float(self._critic_ce_normalizer),
            )

        returns_tensor = torch.as_tensor(
            self.rollout_buffer.returns, device=self.device, dtype=torch.float32
        ).flatten()

        base_scale = float(self.value_target_scale)
        returns_raw_tensor, base_scale_safe = self._decode_returns_scale_only(returns_tensor)
        cvar_penalty_scale = 1.0 / base_scale_safe  # legacy telemetry only

        # Rollout returns are stored directly in the base fraction scale; decode via the
        # static value_target_scale without reapplying any running mean/std factors.
        returns_decode_path = "scale_only"
        self.logger.record("train/returns_decode_path", returns_decode_path)


        rewards_tensor = torch.as_tensor(
            self.rollout_buffer.rewards, device=self.device, dtype=torch.float32
        ).flatten()

        if self._last_rollout_reward_raw is not None:
            rewards_raw_np = np.asarray(self._last_rollout_reward_raw, dtype=np.float32)
            rewards_raw_tensor = torch.as_tensor(rewards_raw_np, device=self.device, dtype=torch.float32).flatten()
        else:
            rewards_raw_tensor = rewards_tensor * base_scale_safe

        (
            rewards_winsor_tensor,
            cvar_empirical_tensor,
            reward_raw_p50_tensor,
            reward_raw_p95_tensor,
            returns_abs_p95_fraction_tensor,
        ) = self._compute_cvar_statistics(rewards_raw_tensor)

        if returns_raw_tensor.numel() > 0:
            returns_abs_p95_value_tensor = torch.quantile(
                returns_raw_tensor.abs(), 0.95
            ).clamp_min(0.0)
        else:
            returns_abs_p95_value_tensor = returns_raw_tensor.new_tensor(0.0)

        cvar_empirical_value = float(cvar_empirical_tensor.item())
        cvar_offset_value, cvar_scale_value = self._get_cvar_normalization_params()
        cvar_offset_tensor = rewards_raw_tensor.new_tensor(cvar_offset_value)
        cvar_scale_value = max(float(cvar_scale_value), 1e-8)
        cvar_scale_tensor = rewards_raw_tensor.new_tensor(cvar_scale_value)
        cvar_empirical_unit_tensor = (cvar_empirical_tensor - cvar_offset_tensor) / cvar_scale_tensor
        cvar_empirical_unit_value = float(cvar_empirical_unit_tensor.item())
        cvar_limit_raw_value = self._get_cvar_limit_raw()
        cvar_limit_raw_tensor = rewards_raw_tensor.new_tensor(cvar_limit_raw_value)
        cvar_limit_unit_tensor = (cvar_limit_raw_tensor - cvar_offset_tensor) / cvar_scale_tensor
        cvar_limit_unit_value = float(cvar_limit_unit_tensor.item())
        cvar_gap_tensor = cvar_limit_raw_tensor - cvar_empirical_tensor  # >0 if CVaR below limit (raw)
        cvar_gap_value = float(cvar_gap_tensor.item())
        cvar_gap_unit_tensor = cvar_limit_unit_tensor - cvar_empirical_unit_tensor
        cvar_gap_unit_value = float(cvar_gap_unit_tensor.item())
        self._cvar_lambda = self._bounded_dual_update(
            float(self._cvar_lambda), float(self.cvar_lambda_lr), cvar_gap_unit_value
        )
        cvar_violation_unit_tensor = torch.clamp(cvar_gap_unit_tensor.detach(), min=0.0)
        cvar_violation_unit_value = float(cvar_violation_unit_tensor.item())
        cvar_gap_pos_value_raw = self._compute_cvar_violation(cvar_empirical_value)
        if not self.cvar_use_penalty:
            current_cvar_weight_nominal = 0.0
            current_cvar_weight_raw = 0.0
            penalty_active = False
        else:
            (
                current_cvar_weight_nominal,
                current_cvar_weight_raw,
                penalty_active,
            ) = self._resolve_cvar_penalty_state(
                current_cvar_weight_nominal, current_cvar_weight_raw, cvar_violation_unit_value
            )
        current_cvar_weight_scaled = float(current_cvar_weight_raw)
        self._current_cvar_weight = float(current_cvar_weight_scaled)
        cvar_penalty_active_value = 1.0 if penalty_active else 0.0
        cvar_violation_raw = float(cvar_gap_value)      # может быть < 0
        cvar_violation = float(cvar_gap_pos_value_raw)  # всегда >= 0 (клиповано)
        self.cvar_lambda = float(self._cvar_lambda)
        # --- CVaR debug block: не дублируем train/*, оставляем debug/*
        self.logger.record("debug/cvar_violation", float(cvar_violation))
        beta = float(self.cvar_ema_beta)
        if self._cvar_empirical_ema is None:
            self._cvar_empirical_ema = float(cvar_empirical_value)
        else:
            self._cvar_empirical_ema = beta * self._cvar_empirical_ema + (1.0 - beta) * float(cvar_empirical_value)
        if self._cvar_violation_ema is None:
            self._cvar_violation_ema = float(cvar_gap_pos_value_raw)
        else:
            self._cvar_violation_ema = beta * self._cvar_violation_ema + (1.0 - beta) * float(cvar_gap_pos_value_raw)
        lambda_scaled = float(self._cvar_lambda)
        constraint_term_value = float(lambda_scaled * cvar_violation_unit_value)

        reward_raw_p50_value = float(reward_raw_p50_tensor.item())
        reward_raw_p95_value = float(reward_raw_p95_tensor.item())

        reward_costs_fraction_value: Optional[float] = None
        reward_costs_fraction_mean_value: Optional[float] = None
        if self._last_rollout_reward_costs is not None:
            reward_costs_np = np.asarray(self._last_rollout_reward_costs, dtype=np.float32).flatten()
            finite_costs_mask = np.isfinite(reward_costs_np)
            if np.any(finite_costs_mask):
                finite_costs = reward_costs_np[finite_costs_mask]
                reward_costs_fraction_value = float(np.median(finite_costs))
                reward_costs_fraction_mean_value = float(np.mean(finite_costs))

        clip_bound_min_value = self._last_rollout_clip_bounds_min
        clip_bound_median_value = self._last_rollout_clip_bounds_median
        clip_bound_max_value = self._last_rollout_clip_bounds_max
        clip_bound_value = float(clip_bound_median_value) if clip_bound_median_value is not None else 0.0
        clip_bound_cap_frac_value = self._last_rollout_clip_cap_fraction
        clip_bound_cap_frac_logged = (
            float(clip_bound_cap_frac_value)
            if clip_bound_cap_frac_value is not None
            else 0.0
        )

        returns_abs_p95_fraction_value = float(returns_abs_p95_fraction_tensor.item())
        returns_abs_p95_value = float(returns_abs_p95_value_tensor.item())

        self._value_scale_latest_ret_abs_p95 = float(returns_abs_p95_value)

        ret_mu_value = float(self._ret_mean_snapshot)
        ret_std_value = float(self._ret_std_snapshot)
        pending_mean_value = ret_mu_value
        pending_std_value = ret_std_value

        if self.normalize_returns:
            pending_rms = self._pending_rms
            if pending_rms is not None and returns_raw_tensor.numel() > 0:
                with torch.no_grad():
                    pending_rms.update(returns_raw_tensor.detach().cpu().numpy())

            if pending_rms is not None and pending_rms.count > 1e-3:
                sample_stats = self._extract_rms_stats(pending_rms)
                if sample_stats is not None:
                    (
                        blended_mean,
                        blended_var,
                        blended_weight,
                        _,
                    ) = self._summarize_recent_return_stats(
                        sample_stats[0],
                        sample_stats[1],
                        sample_stats[2],
                        inplace=False,
                    )
                    (
                        preview_mean,
                        preview_var,
                        _,
                        _,
                    ) = self._apply_return_stats_ema(
                        blended_mean,
                        blended_var,
                        blended_weight,
                        base_mean=float(self._value_scale_stats_mean),
                        base_second=float(self._value_scale_stats_second),
                        base_initialized=bool(self._value_scale_stats_initialized),
                    )
                    pending_mean_value = float(preview_mean)
                    pending_std_value = max(
                        math.sqrt(max(preview_var, 0.0)),
                        self._value_scale_std_floor,
                    )

            self._pending_ret_mean = float(pending_mean_value)
            self._pending_ret_std = float(pending_std_value)

            self._value_target_scale_robust = 1.0
            denom = max(
                self.ret_clip * ret_std_value,
                self.ret_clip * self._value_scale_std_floor,
            )
            self._value_target_scale_effective = float(1.0 / denom)
            if self._value_clip_limit_unscaled is not None:
                self._value_clip_limit_scaled = None

            target_v_min = -float(self.ret_clip)
            target_v_max = float(self.ret_clip)
            updated_v_min = float(self.running_v_min)
            updated_v_max = float(self.running_v_max)
            if not getattr(self, "_value_scale_frozen", False):
                _, _, updated_v_min, updated_v_max, _ = self._apply_v_range_update(
                    target_v_min, target_v_max
                )

            running_v_min_unscaled = updated_v_min * ret_std_value + ret_mu_value
            running_v_max_unscaled = updated_v_max * ret_std_value + ret_mu_value
        else:
            base_scale = float(self.value_target_scale)
            effective_scale = float(self._value_target_scale_effective)
            robust_scale_value = float(self._value_target_scale_robust)
            if not math.isfinite(effective_scale) or effective_scale <= 0.0:
                effective_scale = float(min(max(base_scale, 1e-3), 1e3))
                self._value_target_scale_effective = effective_scale
            if not math.isfinite(robust_scale_value) or robust_scale_value <= 0.0:
                robust_scale_value = 1.0
                self._value_target_scale_robust = robust_scale_value
            if self._value_clip_limit_unscaled is not None:
                self._value_clip_limit_scaled = (
                    self._value_clip_limit_unscaled * self._value_target_scale_effective
                )

            scaled_returns_tensor = returns_tensor * self._value_target_scale_effective

            if self._value_clip_limit_unscaled is not None:
                min_half_range = float(self._value_clip_limit_scaled)
            else:
                if self._use_quantile_value or self.policy.atoms.numel() == 0:
                    min_half_range = 0.0
                else:
                    with torch.no_grad():
                        min_half_range = float(torch.max(torch.abs(self.policy.atoms)).item())
            if not math.isfinite(min_half_range):
                min_half_range = 0.0

            if scaled_returns_tensor.numel() == 0:
                v_min = -min_half_range
                v_max = min_half_range
            else:
                quantile_bounds = torch.tensor(
                    [0.02, 0.98], device=scaled_returns_tensor.device, dtype=scaled_returns_tensor.dtype
                )
                v_low, v_high = torch.quantile(scaled_returns_tensor, quantile_bounds)
                raw_min = float(torch.min(scaled_returns_tensor).item())
                raw_max = float(torch.max(scaled_returns_tensor).item())
                v_min = float(min(v_low.item(), raw_min))
                v_max = float(max(v_high.item(), raw_max))

            if not math.isfinite(v_min) or not math.isfinite(v_max):
                raise RuntimeError(
                    f"Encountered non-finite return bounds when updating value support: {v_min}, {v_max}"
                )

            if v_max <= v_min:
                center = v_min
                half_range = 0.0
            else:
                center = 0.5 * (v_max + v_min)
                half_range = 0.5 * (v_max - v_min)

            half_range = max(half_range, min_half_range)

            padding = max(1e-6 * self._value_target_scale_effective, half_range * 0.05)

            half_range += padding
            v_min = center - half_range
            v_max = center + half_range

            if v_max <= v_min:
                raise RuntimeError(
                    f"Failed to compute a valid value support range: v_min={v_min}, v_max={v_max}"
                )

            updated_v_min = float(self.running_v_min)
            updated_v_max = float(self.running_v_max)
            if not getattr(self, "_value_scale_frozen", False):
                _, _, updated_v_min, updated_v_max, _ = self._apply_v_range_update(
                    v_min, v_max
                )

            running_v_min_unscaled = (
                updated_v_min / self._value_target_scale_effective
            )
            running_v_max_unscaled = (
                updated_v_max / self._value_target_scale_effective
            )

        ret_mu_tensor = torch.as_tensor(ret_mu_value, device=self.device, dtype=torch.float32)
        ret_std_tensor = torch.as_tensor(ret_std_value, device=self.device, dtype=torch.float32)

        if not self._use_quantile_value:
            self.logger.record("train/v_min", running_v_min_unscaled)
            self.logger.record("train/v_max", running_v_max_unscaled)
            self.logger.record("train/v_min_scaled", self.running_v_min)
            self.logger.record("train/v_max_scaled", self.running_v_max)
        self.logger.record("train/value_target_scale", float(self._value_target_scale_effective))
        self.logger.record("train/value_target_scale_config", float(self.value_target_scale))
        self.logger.record("train/value_target_scale_robust", float(self._value_target_scale_robust))
        self.logger.record(
            "train/value_target_scale[1/fraction]",
            float(self._value_target_scale_effective),
        )
        self.logger.record(
            "train/value_target_scale_config[fraction]",
            float(self.value_target_scale),
        )
        self.logger.record(
            "train/value_target_scale_robust[fraction]",
            float(self._value_target_scale_robust),
        )
        if self._value_clip_limit_unscaled is not None:
            self.logger.record("train/value_clip_limit", float(self._value_clip_limit_unscaled))
        self.logger.record("train/ret_mean", float(ret_mu_value))
        self.logger.record("train/ret_std", float(ret_std_value))
        if self._pending_ret_mean is not None and self._pending_ret_std is not None:
            self.logger.record("train/ret_mean_candidate", float(self._pending_ret_mean))
            self.logger.record("train/ret_std_candidate", float(self._pending_ret_std))
        self.logger.record("train/returns_abs_p95", float(returns_abs_p95_value))

        if not (0.0 < float(self.gamma) <= 1.0):
            raise RuntimeError(f"Invalid discount factor 'gamma': {self.gamma}")
        if not (0.0 <= float(self.gae_lambda) <= 1.0):
            raise RuntimeError(f"Invalid GAE lambda 'gae_lambda': {self.gae_lambda}")

        self.logger.record("train/gamma", float(self.gamma))
        self.logger.record("train/gae_lambda", float(self.gae_lambda))

        bc_coef = self._update_bc_coef()
        self.logger.record("train/policy_bc_coef", bc_coef)

        policy_entropy_sum = 0.0
        policy_entropy_count = 0
        entropy_raw_sum = 0.0
        entropy_raw_count = 0
        kl_raw_sum = 0.0
        kl_raw_count = 0
        raw_z_clip_count = 0.0
        raw_z_total = 0
        approx_kl_divs: list[float] = []
        target_return_batches: list[torch.Tensor] = []
        mean_value_batches: list[torch.Tensor] = []
        last_optimizer_lr: Optional[float] = None
        last_scheduler_lr: Optional[float] = None
        kl_exceed_fraction_latest = 0.0

        policy_loss_value = 0.0
        policy_loss_ppo_value = 0.0
        policy_loss_bc_value = 0.0
        policy_loss_bc_weighted_value = 0.0
        critic_loss_value = 0.0
        cvar_raw_value = 0.0
        cvar_unit_value = 0.0
        cvar_loss_value = 0.0
        cvar_loss_unit_value = 0.0
        cvar_term_raw_value = 0.0
        cvar_term_value = 0.0
        total_loss_value = 0.0
        clamp_below_sum = 0.0
        clamp_above_sum = 0.0
        clamp_weight = 0.0
        clamp_below_raw_sum = 0.0
        clamp_above_raw_sum = 0.0
        clamp_raw_weight = 0.0
        raw_outlier_warn_count = 0
        raw_outlier_frac_max = 0.0

        adv_mean_accum = 0.0
        adv_std_accum = 0.0
        adv_batch_count = 0

        value_logits_final: Optional[torch.Tensor] = None
        value_quantiles_final: Optional[torch.Tensor] = None
        quantile_batches_unscaled: list[torch.Tensor] = []
        quantile_batches_norm: list[torch.Tensor] = []
        value_mse_value = 0.0

        clip_fraction_numer = 0.0
        clip_fraction_denom = 0
        ratio_sum = 0.0
        ratio_sq_sum = 0.0
        ratio_count = 0
        log_prob_sum = 0.0
        log_prob_count = 0
        # KL penalty aggregation (for logging average component)
        kl_penalty_component_total = 0.0
        kl_penalty_component_count = 0
        adv_z_values: list[torch.Tensor] = []
        approx_kl_exceed_count = 0
        minibatches_processed = 0
        approx_kl_smooth_latest = 0.0
        approx_kl_last_exceeded_raw = 0.0
        approx_kl_last_exceeded_smooth = 0.0
        kl_exceed_consec_max_latest = 0
        kl_stop_trigger_value_raw: Optional[float] = None

        base_n_epochs = max(1, int(self.n_epochs))
        if base_n_epochs != self._base_n_epochs:
            self._base_n_epochs = base_n_epochs
            self._kl_epoch_factor_min = 1.0 / float(self._base_n_epochs)
            self._kl_epoch_factor = min(max(self._kl_epoch_factor, self._kl_epoch_factor_min), 1.0)

        effective_n_epochs = max(1, int(round(self._base_n_epochs * self._kl_epoch_factor)))
        kl_early_stop_triggered = False
        epochs_completed = 0
        approx_kl_latest = 0.0




        effective_batch_size = int(self.batch_size)
        if effective_batch_size <= 0:
            raise RuntimeError("PPO batch_size must be positive for training")
        microbatch_size_effective = max(
            1, int(getattr(self, "_microbatch_size", effective_batch_size))
        )
        grad_accum_steps = max(
            1, int(getattr(self, "_grad_accumulation_steps", 1))
        )
        if effective_batch_size % microbatch_size_effective != 0:
            raise RuntimeError(
                "Configured batch_size must be divisible by microbatch_size; adjust n_steps, n_envs, or microbatch_size"
            )

        def _prepare_minibatch_iterator() -> tuple[Optional[Iterable[tuple[Any, ...]]], int]:
            microbatches = list(self.rollout_buffer.get(microbatch_size_effective))
            if not microbatches:
                return None, effective_batch_size
            total_micro = len(microbatches)
            if total_micro % grad_accum_steps != 0:
                raise RuntimeError(
                    "Rollout buffer produced incomplete micro-batch bucket; ensure n_steps * n_envs is divisible by batch_size and microbatch_size"
                )

            def _grouped_microbatches() -> Iterable[tuple[Any, ...]]:
                for start_idx in range(0, total_micro, grad_accum_steps):
                    yield tuple(microbatches[start_idx:start_idx + grad_accum_steps])

            expected_batch = microbatch_size_effective * grad_accum_steps
            return _grouped_microbatches(), expected_batch

        for _ in range(effective_n_epochs):
            minibatch_iterator, expected_batch_size = _prepare_minibatch_iterator()
            if minibatch_iterator is None:
                self.logger.record("warn/empty_rollout_buffer", 1.0)
                break

            epochs_completed += 1
            self.logger.record("train/expected_batch_size", float(expected_batch_size))
            self.logger.record("train/microbatch_size", float(microbatch_size_effective))
            self.logger.record("train/grad_accum_steps", float(grad_accum_steps))

            epoch_minibatches_processed = 0
            epoch_exceed_count = 0
            epoch_consec_run = 0
            epoch_consec_max = 0
            kl_window: Optional[deque[float]] = None
            kl_window_sum = 0.0
            kl_smooth_value: Optional[float] = None

            for microbatch_group in minibatch_iterator:
                microbatch_items = tuple(microbatch_group)
                sample_counts = [int(data.advantages.shape[0]) for data in microbatch_items]
                bucket_target_size = int(sum(sample_counts))
                if bucket_target_size <= 0:
                    self.logger.record("warn/empty_microbatch_group", 1.0)
                    continue
                self.logger.record("train/actual_batch_size", float(bucket_target_size))
                clip_range = float(self._clip_range_current)
                self.policy.optimizer.zero_grad(set_to_none=True)

                bucket_policy_loss_value = 0.0
                bucket_policy_loss_ppo_value = 0.0
                bucket_policy_loss_bc_value = 0.0
                bucket_policy_loss_bc_weighted_value = 0.0
                bucket_critic_loss_value = 0.0
                bucket_cvar_raw_value = 0.0
                bucket_cvar_unit_value = 0.0
                bucket_cvar_loss_value = 0.0
                bucket_cvar_loss_unit_value = 0.0
                bucket_cvar_term_raw_value = 0.0
                bucket_cvar_term_value = 0.0
                bucket_total_loss_value = 0.0
                bucket_value_mse_value = 0.0
                bucket_value_logits_fp32: Optional[torch.Tensor] = None
                bucket_value_quantiles_fp32: Optional[torch.Tensor] = None
                approx_kl_weighted_sum = 0.0
                bucket_sample_count = 0

                for rollout_data, sample_count in zip(microbatch_items, sample_counts):
                    _values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        rollout_data.actions,
                        rollout_data.lstm_states,
                        rollout_data.episode_starts,
                        actions_raw=rollout_data.actions_raw,
                    )

                    advantages = rollout_data.advantages
                    if sample_count <= 0:
                        continue
                    bucket_sample_count += sample_count
                    weight = float(sample_count) / float(bucket_target_size)

                    with torch.no_grad():
                        adv_mean_tensor = advantages.mean()
                        adv_std_tensor = advantages.std(unbiased=False)
                        adv_std_tensor_clamped = torch.clamp(adv_std_tensor, min=1e-8)
                    adv_mean = float(adv_mean_tensor.item())
                    adv_std = float(adv_std_tensor.item())
                    adv_mean_accum += adv_mean
                    adv_std_accum += adv_std
                    adv_batch_count += 1

                    advantages = (advantages - adv_mean_tensor) / adv_std_tensor_clamped
                    with torch.no_grad():
                        adv_z_values.append(advantages.detach().cpu())

                    entropy_tensor = entropy
                    actor_states = self._extract_actor_states(rollout_data.lstm_states)
                    dist_output = self.policy.get_distribution(
                        rollout_data.observations,
                        actor_states,
                        rollout_data.episode_starts,
                    )
                    # Some recurrent policies (including custom ones used in this
                    # project) return auxiliary data such as the updated RNN
                    # states alongside the action distribution. When the method
                    # returns a tuple, the actual distribution object is the
                    # first element, so unwrap it before proceeding.
                    if isinstance(dist_output, tuple):
                        dist = dist_output[0]
                    else:
                        dist = dist_output

                    if self._kl_diag:
                        try:
                            self._kl_diag_step(dist, rollout_data)
                        except Exception:
                            self.logger.record("diag/error", 1.0)

                    if log_prob.shape != rollout_data.old_log_prob.shape:
                        raise RuntimeError(
                            "Log-prob shape mismatch between rollout buffer and training step"
                        )

                    log_ratio = log_prob - rollout_data.old_log_prob
                    ratio = torch.exp(log_ratio)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss_ppo = -torch.min(policy_loss_1, policy_loss_2).mean()

                    with torch.no_grad():
                        ratio_detached = ratio.detach()
                        clip_mask = ratio_detached.sub(1.0).abs() > clip_range
                        clipped = clip_mask.float().mean()
                        self.logger.record("train/clip_fraction_batch", float(clipped.item()))
                        ratio_sum += float(ratio_detached.sum().item())
                        ratio_sq_sum += float((ratio_detached.square()).sum().item())
                        ratio_count += int(ratio_detached.numel())
                        clip_fraction_numer += float(clip_mask.sum().item())
                        clip_fraction_denom += int(clip_mask.numel())

                        log_prob_detached = log_prob.detach()
                        log_prob_sum += float(log_prob_detached.sum().item())
                        log_prob_count += int(log_prob_detached.numel())

                    if bc_coef <= 0.0:
                        policy_loss_bc = advantages.new_zeros(())
                        policy_loss_bc_weighted = policy_loss_bc
                    else:
                        with torch.no_grad():
                            weights = torch.exp(advantages / self.cql_beta)
                            weights = torch.clamp(weights, max=100.0)
                        policy_loss_bc = (-log_prob * weights).mean()
                        policy_loss_bc_weighted = policy_loss_bc * bc_coef

                    policy_loss = policy_loss_ppo + policy_loss_bc_weighted
                    if self.kl_beta > 0.0:
                        kl_penalty_sample = (rollout_data.old_log_prob - log_prob).mean()
                        kl_penalty_component = self.kl_beta * kl_penalty_sample
                        policy_loss = policy_loss + kl_penalty_component
                        kl_penalty_component_total += float(kl_penalty_component.item())
                        kl_penalty_component_count += 1

                    inner_dist = getattr(dist, "distribution", None)
                    if entropy_tensor is None:
                        entropy_fn = getattr(self.policy, "weighted_entropy", None)
                        if callable(entropy_fn):
                            entropy_tensor = entropy_fn(dist)
                        else:
                            entropy_tensor = dist.entropy()

                    with torch.no_grad():
                        if inner_dist is not None:
                            entropy_raw_tensor = inner_dist.entropy()
                            if entropy_raw_tensor.ndim > 1:
                                entropy_raw_tensor = entropy_raw_tensor.sum(dim=-1)
                            if torch.isfinite(entropy_raw_tensor).all():
                                entropy_raw_detached = entropy_raw_tensor.to(dtype=torch.float32)
                                entropy_raw_sum += float(entropy_raw_detached.sum().cpu().item())
                                entropy_raw_count += int(entropy_raw_detached.numel())

                        if (
                            isinstance(self.action_space, gym.spaces.Box)
                            and inner_dist is not None
                        ):
                            eps = float(getattr(self.policy, "_score_clip_eps", 1e-6))
                            raw_actions = rollout_data.actions_raw.to(
                                device=self.device, dtype=torch.float32
                            )
                            mean = getattr(inner_dist, "mean", None)
                            if callable(mean):
                                mean = mean()
                            if mean is None:
                                mean = getattr(dist, "mean_actions", None)
                                if callable(mean):
                                    mean = mean()
                            std = getattr(inner_dist, "stddev", None)
                            if std is None:
                                get_std = getattr(inner_dist, "get_std", None)
                                std = get_std() if callable(get_std) else None
                            z_clip_mask: Optional[torch.Tensor] = None
                            if mean is not None and std is not None:
                                std_safe = torch.where(
                                    std > 0.0,
                                    std,
                                    torch.full_like(std, eps),
                                )
                                z_unclamped = (raw_actions - mean) / std_safe
                                z_clip_mask = z_unclamped.abs() > 8.0
                            if z_clip_mask is not None:
                                raw_z_clip_count += float(z_clip_mask.sum().item())
                                raw_z_total += int(z_clip_mask.numel())
                            log_prob_raw_new = self.policy._log_prob_raw_only(
                                dist, raw_actions
                            ).reshape(-1)
                            old_log_prob_raw = rollout_data.old_log_prob_raw.reshape(-1)
                            approx_kl_raw_tensor = old_log_prob_raw - log_prob_raw_new
                            if torch.isfinite(approx_kl_raw_tensor).all() and approx_kl_raw_tensor.numel() > 0:
                                kl_raw_sum += float(approx_kl_raw_tensor.sum().item())
                                kl_raw_count += int(approx_kl_raw_tensor.numel())

                    if entropy_tensor.ndim > 1:
                        entropy_tensor = entropy_tensor.sum(dim=-1)
                    entropy_loss = -torch.mean(entropy_tensor)

                    entropy_detached = entropy_tensor.detach().to(dtype=torch.float32)
                    policy_entropy_sum += float(entropy_detached.sum().cpu().item())
                    policy_entropy_count += int(entropy_detached.numel())

                    if self._use_quantile_value:
                        value_quantiles = self.policy.last_value_quantiles
                        if value_quantiles is None:
                            raise RuntimeError(
                                "Policy did not cache value quantiles during training forward pass"
                            )
                        value_head_fp32 = value_quantiles.to(dtype=torch.float32)
                        target_distribution: Optional[torch.Tensor] = None
                    else:
                        value_logits = self.policy.last_value_logits
                        if value_logits is None:
                            raise RuntimeError(
                                "Policy did not cache value logits during training forward pass"
                            )
                        value_logits_fp32 = value_logits.to(dtype=torch.float32)
                        value_head_fp32 = value_logits_fp32
                        target_distribution = torch.zeros_like(value_logits_fp32)

                    with torch.no_grad():

                        buffer_returns = rollout_data.returns.to(
                            device=self.device, dtype=torch.float32
                        )
                        # Rollout-returns храним в базе (fraction / value_target_scale):
                        # декодируем БЕЗ μ/σ, используя только базовый масштаб.
                        target_returns_raw, base_scale_safe = self._decode_returns_scale_only(
                            buffer_returns
                        )

                        # НЕТ raw-clip при normalize_returns: полагаемся на нормализованный ±ret_clip
                        if (not self.normalize_returns) and (
                            self._value_clip_limit_unscaled is not None
                        ):
                            target_returns_raw = torch.clamp(
                                target_returns_raw,
                                min=-self._value_clip_limit_unscaled,
                                max=self._value_clip_limit_unscaled,
                            )

                        weight_before_raw = weight
                        raw_weight = float(target_returns_raw.numel())

                        if self.normalize_returns:
                            target_returns_norm_raw = (
                                target_returns_raw - ret_mu_tensor
                            ) / ret_std_tensor
                            target_returns_norm = target_returns_norm_raw.clamp(
                                -self.ret_clip, self.ret_clip
                            )
                        else:
                            target_returns_norm_raw = (
                                (target_returns_raw / float(base_scale_safe))
                                * self._value_target_scale_effective
                            )
                            target_returns_norm = target_returns_norm_raw
                            if self._value_clip_limit_scaled is not None:
                                target_returns_norm = torch.clamp(
                                    target_returns_norm,
                                    min=-self._value_clip_limit_scaled,
                                    max=self._value_clip_limit_scaled,
                                )

                        if self._use_quantile_value:
                            raw_outlier_frac = 0.0
                            self._last_raw_outlier_frac = 0.0
                        else:
                            raw_below_frac, raw_above_frac = self._value_target_outlier_fractions(
                                target_returns_norm_raw,
                                float(self.policy.v_min),
                                float(self.policy.v_max),
                            )
                            self.logger.record("train/value_target_below_frac_raw", raw_below_frac)
                            self.logger.record("train/value_target_above_frac_raw", raw_above_frac)
                            clamp_below_raw_sum += raw_below_frac * raw_weight
                            clamp_above_raw_sum += raw_above_frac * raw_weight
                            clamp_raw_weight += raw_weight
                            raw_outlier_frac = raw_below_frac + raw_above_frac
                            self._last_raw_outlier_frac = float(raw_outlier_frac)
                            raw_outlier_frac_max = max(raw_outlier_frac_max, raw_outlier_frac)
                            if raw_outlier_frac > self._value_target_raw_outlier_warn_threshold:
                                raw_outlier_warn_count += 1
                                self.logger.record(
                                    "warn/value_target_raw_outlier_frac", float(raw_outlier_frac)
                                )

                        weight = weight_before_raw

                        if not self._use_quantile_value:
                            delta_z = (self.policy.v_max - self.policy.v_min) / float(
                                self.policy.num_atoms - 1
                            )
                            clamped_targets = target_returns_norm.clamp(
                                self.policy.v_min, self.policy.v_max
                            )
                            b = (clamped_targets - self.policy.v_min) / (delta_z + 1e-8)
                            lower_bound = b.floor().long().clamp(min=0, max=self.policy.num_atoms - 1)
                            upper_bound = b.ceil().long().clamp(min=0, max=self.policy.num_atoms - 1)

                            same_bounds = lower_bound == upper_bound

                            target_distribution.zero_()
                            lower_prob = (upper_bound.to(torch.float32) - b).clamp(min=0.0)
                            upper_prob = (b - lower_bound.to(torch.float32)).clamp(min=0.0)
                            lower_prob = torch.where(
                                same_bounds,
                                torch.ones_like(lower_prob),
                                lower_prob,
                            )
                            upper_prob = torch.where(
                                same_bounds,
                                torch.zeros_like(upper_prob),
                                upper_prob,
                            )
                            target_distribution.scatter_add_(1, lower_bound.view(-1, 1), lower_prob.view(-1, 1))
                            target_distribution.scatter_add_(1, upper_bound.view(-1, 1), upper_prob.view(-1, 1))

                            normaliser = target_distribution.sum(dim=1, keepdim=True).clamp_min(1e-8)
                            target_distribution = target_distribution / normaliser

                            if torch.any(same_bounds):
                                same_indices = same_bounds.nonzero(as_tuple=False).squeeze(1)
                                if same_indices.numel() > 0:
                                    target_distribution[same_indices] = 0.0
                                    target_distribution[same_indices, lower_bound[same_indices]] = 1.0

                        target_return_batches.append(target_returns_raw.reshape(-1, 1).detach())

                        below_frac = float(
                            (target_returns_norm < self.policy.v_min).float().mean().item()
                        )
                        above_frac = float(
                            (target_returns_norm > self.policy.v_max).float().mean().item()
                        )
                        clamp_below_sum += below_frac * weight
                        clamp_above_sum += above_frac * weight
                        clamp_weight += weight

                    if self._use_quantile_value:
                        quantiles_fp32 = value_head_fp32
                        mean_values_norm = quantiles_fp32.mean(dim=1, keepdim=True)
                        mean_values_unscaled = self._to_raw_returns(mean_values_norm)
                        quantiles_unscaled = self._to_raw_returns(quantiles_fp32)
                        if (not self.normalize_returns) and (
                            self._value_clip_limit_unscaled is not None
                        ):
                            mean_values_unscaled = torch.clamp(
                                mean_values_unscaled,
                                min=-self._value_clip_limit_unscaled,
                                max=self._value_clip_limit_unscaled,
                            )
                            quantiles_unscaled = torch.clamp(
                                quantiles_unscaled,
                                min=-self._value_clip_limit_unscaled,
                                max=self._value_clip_limit_unscaled,
                            )

                        mean_value_batches.append(mean_values_unscaled.detach())
                        quantile_batches_unscaled.append(quantiles_unscaled.detach())
                        quantile_batches_norm.append(quantiles_fp32.detach())

                        mean_values_flat = mean_values_unscaled.view(-1)
                        target_returns_flat = target_returns_raw.view(-1)
                        mse_tensor = F.mse_loss(
                            mean_values_flat,
                            target_returns_flat,
                            reduction="mean",
                        )
                        bucket_value_mse_value += float(mse_tensor.item()) * weight

                        predicted_cvar_norm = self._cvar_from_quantiles(quantiles_fp32)
                        cvar_raw = self._to_raw_returns(predicted_cvar_norm).mean()

                        critic_loss = self._quantile_huber_loss(
                            quantiles_fp32, target_returns_norm
                        )
                    else:
                        pred_probs_fp32 = torch.softmax(value_logits_fp32, dim=1).clamp(min=1e-8, max=1.0)
                        log_predictions = torch.log(pred_probs_fp32)
                        critic_loss = -(target_distribution * log_predictions).sum(dim=1).mean()
                        critic_loss = critic_loss / self._critic_ce_normalizer

                        with torch.no_grad():

                            mean_values_norm = (pred_probs_fp32 * self.policy.atoms).sum(dim=1, keepdim=True)
                            mean_values_unscaled = self._to_raw_returns(mean_values_norm)

                            mean_values_flat = mean_values_unscaled.view(-1)
                            target_returns_flat = target_returns_raw.view(-1)
                            mse_tensor = F.mse_loss(
                                mean_values_flat,
                                target_returns_flat,
                                reduction="mean",
                            )
                            bucket_value_mse_value += float(mse_tensor.item()) * weight

                            if (not self.normalize_returns) and (
                                self._value_clip_limit_unscaled is not None
                            ):
                                mean_values_unscaled = torch.clamp(
                                    mean_values_unscaled,
                                    min=-self._value_clip_limit_unscaled,
                                    max=self._value_clip_limit_unscaled,
                                )
                            mean_value_batches.append(mean_values_unscaled.detach())

                        predicted_cvar = calculate_cvar(
                            pred_probs_fp32, self.policy.atoms, self.cvar_alpha
                        )
                        cvar_raw = self._to_raw_returns(predicted_cvar).mean()

                    cvar_unit_tensor = (cvar_raw - cvar_offset_tensor) / cvar_scale_tensor
                    cvar_loss = -cvar_unit_tensor
                    cvar_loss_raw_tensor = cvar_loss * cvar_scale_tensor
                    cvar_term = current_cvar_weight_scaled * cvar_loss
                    cvar_term_raw_tensor = current_cvar_weight_scaled * cvar_loss_raw_tensor
                    if self.cvar_cap is not None:
                        cvar_term = torch.clamp(cvar_term, min=-self.cvar_cap, max=self.cvar_cap)

                    loss = (
                        policy_loss.to(dtype=torch.float32)
                        + ent_coef_eff_value * entropy_loss.to(dtype=torch.float32)
                        + vf_coef_effective * critic_loss
                        + cvar_term
                    )

                    if self.cvar_use_constraint:
                        loss = loss + loss.new_tensor(lambda_scaled) * cvar_violation_unit_tensor

                    loss_weighted = loss * loss.new_tensor(weight)
                    loss_weighted.backward()

                    bucket_policy_loss_value += float(policy_loss.item()) * weight
                    bucket_policy_loss_ppo_value += float(policy_loss_ppo.item()) * weight
                    bucket_policy_loss_bc_value += float(policy_loss_bc.item()) * weight
                    bucket_policy_loss_bc_weighted_value += float(policy_loss_bc_weighted.item()) * weight
                    bucket_critic_loss_value += float(critic_loss.item()) * weight
                    bucket_cvar_raw_value += float(cvar_raw.item()) * weight
                    bucket_cvar_unit_value += float(cvar_unit_tensor.item()) * weight
                    bucket_cvar_loss_value += float(cvar_loss_raw_tensor.item()) * weight
                    bucket_cvar_loss_unit_value += float(cvar_loss.item()) * weight
                    bucket_cvar_term_raw_value += float(cvar_term_raw_tensor.item()) * weight
                    bucket_cvar_term_value += float(cvar_term.item()) * weight
                    bucket_total_loss_value += float(loss.item()) * weight
                    if self._use_quantile_value:
                        bucket_value_quantiles_fp32 = quantiles_fp32.detach()
                    else:
                        bucket_value_logits_fp32 = value_logits_fp32.detach()

                    approx_kl_component = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    approx_kl_weighted_sum += approx_kl_component * float(sample_count)

                if bucket_sample_count != bucket_target_size:
                    self.logger.record(
                        "warn/microbatch_size_mismatch",
                        float(bucket_sample_count - bucket_target_size),
                    )

                max_grad_norm = (
                    0.5 if self.max_grad_norm is None else float(self.max_grad_norm)
                )
                if max_grad_norm <= 0.0:
                    self.logger.record("warn/max_grad_norm_nonpos", float(max_grad_norm))
                    max_grad_norm = 0.5
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), max_grad_norm
                )
                grad_norm_value = (
                    total_grad_norm.item()
                    if isinstance(total_grad_norm, torch.Tensor)
                    else float(total_grad_norm)
                )
                self.logger.record("train/grad_norm_pre_clip", float(grad_norm_value))
                self.logger.record("train/max_grad_norm_used", float(max_grad_norm))

                post_clip_norm_sq = 0.0
                for param in self.policy.parameters():
                    grad = param.grad
                    if grad is None:
                        continue
                    post_clip_norm_sq += float(grad.detach().to(dtype=torch.float32).pow(2).sum().item())
                post_clip_norm = math.sqrt(post_clip_norm_sq) if post_clip_norm_sq > 0.0 else 0.0
                self.logger.record("train/grad_norm_post_clip", float(post_clip_norm))

                # KL-scale не должен ломать внешний шедулер:
                # - при scale == 1.0 ничего не делаем,
                # - при scale != 1.0 масштабируем ТЕКУЩИЙ LR из оптимайзера.
                scale = float(getattr(self, "_kl_lr_scale", 1.0))
                if scale != 1.0:
                    for group in self.policy.optimizer.param_groups:
                        cur_lr = float(group.get("lr", 0.0))
                        scaled_lr = max(cur_lr * scale, self._kl_min_lr)
                        group["lr"] = scaled_lr
                        if "initial_lr" in group:
                            group["initial_lr"] = scaled_lr

                # Перед шагом оптимизатора удостоверимся, что LR не ниже жёсткого пола.
                self._enforce_optimizer_lr_bounds(log_values=False, warn_on_floor=False)

                self.policy.optimizer.step()

                # Ensure any scheduler is wired to the active optimizer before stepping it.
                self._rebuild_scheduler_if_needed()
                # Шаг шедулера (если он есть) — и только ПОСЛЕ него логируем optimizer_lr
                scheduler = getattr(self.policy, "lr_scheduler", None) or self.lr_scheduler
                last_scheduler_lr: Optional[float] = None
                if scheduler is not None:
                    scheduler.step()
                    get_last_lr = getattr(scheduler, "get_last_lr", None)
                    if callable(get_last_lr):
                        try:
                            scheduler_lrs = get_last_lr()
                        except TypeError:
                            scheduler_lrs = None
                        if scheduler_lrs:
                            last_scheduler_lr = float(scheduler_lrs[0])
                    # Освежаем _kl_base_lr для корректного KL-скейла при внешнем шедулере
                    self._refresh_kl_base_lrs()
                else:
                    self._refresh_kl_base_lrs()

                self._enforce_optimizer_lr_bounds(
                    scheduler_lr=last_scheduler_lr,
                    log_values=True,
                    warn_on_floor=True,
                )

                approx_kl = approx_kl_weighted_sum / float(bucket_sample_count)
                approx_kl_latest = approx_kl
                approx_kl_divs.append(approx_kl)
                minibatches_processed += 1
                epoch_minibatches_processed += 1

                approx_kl_smooth_value = approx_kl
                if self._kl_early_stop_use_ema:
                    if self._kl_ema_alpha is not None:
                        if kl_smooth_value is None:
                            kl_smooth_value = approx_kl
                        else:
                            kl_smooth_value = (
                                self._kl_ema_alpha * approx_kl
                                + (1.0 - self._kl_ema_alpha) * kl_smooth_value
                            )
                        approx_kl_smooth_value = kl_smooth_value
                    else:
                        if kl_window is None:
                            kl_window = deque()
                            kl_window_sum = 0.0
                        kl_window.append(approx_kl)
                        kl_window_sum += float(approx_kl)
                        if len(kl_window) > self._kl_ema_window:
                            removed_value = float(kl_window.popleft())
                            kl_window_sum -= removed_value
                        approx_kl_smooth_value = kl_window_sum / float(len(kl_window))
                        kl_smooth_value = approx_kl_smooth_value
                else:
                    kl_smooth_value = approx_kl
                    approx_kl_smooth_value = approx_kl

                approx_kl_smooth_latest = float(approx_kl_smooth_value)

                exceed_flag = False
                if self.target_kl is not None and self.target_kl > 0.0:
                    exceed_flag = approx_kl_smooth_value > float(self.target_kl)
                    if exceed_flag:
                        approx_kl_exceed_count += 1
                        epoch_exceed_count += 1
                        epoch_consec_run += 1
                        if epoch_consec_run > epoch_consec_max:
                            epoch_consec_max = epoch_consec_run
                        approx_kl_last_exceeded_raw = float(approx_kl)
                        approx_kl_last_exceeded_smooth = float(approx_kl_smooth_value)
                    else:
                        epoch_consec_run = 0
                else:
                    epoch_consec_run = 0

                if (
                    exceed_flag
                    and self.kl_early_stop
                    and self._kl_consec_minibatches > 0
                    and epoch_consec_run >= self._kl_consec_minibatches
                ):
                    kl_early_stop_triggered = True
                    kl_stop_trigger_value_raw = float(approx_kl)
                    break

                policy_loss_value = bucket_policy_loss_value
                policy_loss_ppo_value = bucket_policy_loss_ppo_value
                policy_loss_bc_value = bucket_policy_loss_bc_value
                policy_loss_bc_weighted_value = bucket_policy_loss_bc_weighted_value
                critic_loss_value = bucket_critic_loss_value
                value_mse_value = bucket_value_mse_value
                cvar_raw_value = bucket_cvar_raw_value
                cvar_unit_value = bucket_cvar_unit_value
                cvar_loss_value = bucket_cvar_loss_value
                cvar_loss_unit_value = bucket_cvar_loss_unit_value
                cvar_term_raw_value = bucket_cvar_term_raw_value
                cvar_term_value = bucket_cvar_term_value
                total_loss_value = bucket_total_loss_value

                if bucket_value_logits_fp32 is not None:
                    value_logits_final = bucket_value_logits_fp32

                # Fraction-based KL stop evaluated after completing the epoch

            if kl_early_stop_triggered:
                if epoch_minibatches_processed > 0:
                    epoch_exceed_fraction = float(epoch_exceed_count) / float(epoch_minibatches_processed)
                else:
                    epoch_exceed_fraction = 0.0
                kl_exceed_fraction_latest = epoch_exceed_fraction
                kl_exceed_consec_max_latest = epoch_consec_max
                break

            if epoch_minibatches_processed > 0:
                epoch_exceed_fraction = float(epoch_exceed_count) / float(epoch_minibatches_processed)
            else:
                epoch_exceed_fraction = 0.0
            kl_exceed_fraction_latest = epoch_exceed_fraction
            kl_exceed_consec_max_latest = epoch_consec_max

            if (
                self.kl_early_stop
                and self.target_kl is not None
                and self.target_kl > 0.0
                and self.kl_exceed_stop_fraction > 0.0
                and epoch_exceed_fraction >= self.kl_exceed_stop_fraction
            ):
                kl_early_stop_triggered = True
                kl_stop_trigger_value_raw = (
                    approx_kl_last_exceeded_raw if approx_kl_last_exceeded_raw > 0.0 else float(approx_kl_latest)
                )
                break

        if kl_early_stop_triggered:
            if (
                self.kl_early_stop
                and self.target_kl is not None
                and self.target_kl > 0.0
            ):
                trigger_value = (
                    kl_stop_trigger_value_raw
                    if kl_stop_trigger_value_raw is not None
                    else float(approx_kl_last_exceeded_raw or approx_kl_latest)
                )
                self._handle_kl_divergence(trigger_value)
        self._n_updates += epochs_completed
        self._update_calls += 1
        self._global_update_step += 1

        avg_policy_entropy = (
            policy_entropy_sum / float(policy_entropy_count)
            if policy_entropy_count > 0
            else self._last_rollout_entropy
        )
        avg_policy_entropy_raw = (
            entropy_raw_sum / float(entropy_raw_count)
            if entropy_raw_count > 0
            else self._last_rollout_entropy_raw
        )
        self._maybe_update_entropy_schedule(current_update, avg_policy_entropy)
        self.logger.record("train/policy_entropy", float(avg_policy_entropy))
        self._record_raw_policy_metrics(
            avg_policy_entropy_raw,
            entropy_raw_count,
            kl_raw_sum,
            kl_raw_count,
            raw_z_clip_count,
            raw_z_total,
        )
        if self._use_quantile_value:
            if value_quantiles_final is None:
                cached_quantiles = getattr(self.policy, "last_value_quantiles", None)
                if cached_quantiles is None:
                    cached_quantiles = getattr(self.policy, "_last_value_quantiles", None)
                if cached_quantiles is not None:
                    value_quantiles_final = cached_quantiles.detach().to(dtype=torch.float32)
            if value_quantiles_final is None:
                raise RuntimeError("No value quantiles captured during training loop")
        else:
            if value_logits_final is None:
                cached_logits = getattr(self.policy, "last_value_logits", None)
                if cached_logits is None:
                    cached_logits = getattr(self.policy, "_last_value_logits", None)
                if cached_logits is not None:
                    value_logits_final = cached_logits.detach().to(dtype=torch.float32)
            if value_logits_final is None:
                raise RuntimeError("No value logits captured during training loop")

        if len(target_return_batches) == 0 or len(mean_value_batches) == 0:
            rollout_returns = torch.as_tensor(
                self.rollout_buffer.returns, device=self.device, dtype=torch.float32
            )

            # Final EV/MSE metrics also need fully decoded raw returns
            rollout_returns, _ = self._decode_returns_scale_only(rollout_returns)

            with torch.no_grad():
                if self._use_quantile_value:
                    quantiles_fp32 = value_quantiles_final
                    value_pred_norm = quantiles_fp32.mean(dim=1, keepdim=True)
                    value_pred_raw = self._to_raw_returns(value_pred_norm)
                    quantiles_raw = self._to_raw_returns(quantiles_fp32)
                    if self.normalize_returns:
                        y_pred_tensor = value_pred_raw
                        if not quantile_batches_unscaled:
                            quantile_batches_unscaled.append(quantiles_raw.detach())
                            quantile_batches_norm.append(quantiles_fp32.detach())
                    else:
                        y_pred_tensor = value_pred_raw
                        if self._value_clip_limit_unscaled is not None:
                            y_pred_tensor = torch.clamp(
                                y_pred_tensor,
                                min=-self._value_clip_limit_unscaled,
                                max=self._value_clip_limit_unscaled,
                            )
                        if not quantile_batches_unscaled:
                            quantile_batches_unscaled.append(quantiles_raw.detach())
                            quantile_batches_norm.append(quantiles_fp32.detach())
                else:
                    pred_probs = torch.softmax(value_logits_final, dim=1)
                    value_pred_norm = (pred_probs * self.policy.atoms).sum(dim=1, keepdim=True)
                    value_pred_raw = self._to_raw_returns(value_pred_norm)
                    if self.normalize_returns:
                        y_pred_tensor = value_pred_raw
                    else:
                        y_pred_tensor = value_pred_raw
                        if self._value_clip_limit_unscaled is not None:
                            y_pred_tensor = torch.clamp(
                                y_pred_tensor,
                                min=-self._value_clip_limit_unscaled,
                                max=self._value_clip_limit_unscaled,
                            )

            y_true_tensor = rollout_returns.reshape(-1, 1)

        else:
            y_true_tensor = torch.cat([t.reshape(-1, 1) for t in target_return_batches], dim=0)
            y_pred_tensor = torch.cat([t.reshape(-1, 1) for t in mean_value_batches], dim=0)

        if y_true_tensor.numel() == 0 or y_pred_tensor.numel() == 0:
            explained_var = 0.0
        else:
            if y_true_tensor.shape != y_pred_tensor.shape:
                min_elems = min(y_true_tensor.shape[0], y_pred_tensor.shape[0])
                y_true_tensor = y_true_tensor[:min_elems]
                y_pred_tensor = y_pred_tensor[:min_elems]

            y_true_np = y_true_tensor.flatten().detach().cpu().numpy()
            y_pred_np = y_pred_tensor.flatten().detach().cpu().numpy()
            explained_var = np.nan_to_num(safe_explained_variance(y_true_np, y_pred_np))

            with torch.no_grad():
                value_mse_value = float(
                    F.mse_loss(
                        y_pred_tensor.flatten().to(dtype=torch.float32),
                        y_true_tensor.flatten().to(dtype=torch.float32),
                        reduction="mean",
                    ).item()
                )

        if self._use_quantile_value:
            self._record_quantile_summary(quantile_batches_unscaled, quantile_batches_norm)

        bc_ratio = abs(policy_loss_bc_weighted_value) / (abs(policy_loss_ppo_value) + 1e-8)

        if explained_var < 0.0:
            self._bad_explained_counter += 1
        else:
            self._bad_explained_counter = 0
        self._last_explained_variance = float(explained_var)

        frame_stable = self._is_value_scale_frame_stable(
            self._value_scale_latest_ret_abs_p95, self._last_explained_variance
        )
        self._value_scale_frame_stable = frame_stable
        if frame_stable:
            self._value_scale_stable_counter += 1
        else:
            self._value_scale_stable_counter = 0
        self.logger.record("train/value_scale_frame_stable", float(frame_stable))
        self.logger.record(
            "train/value_scale_stable_consec", float(self._value_scale_stable_counter)
        )
        self.logger.record(
            "train/value_scale_ret_abs_p95", float(self._value_scale_latest_ret_abs_p95)
        )

        self.logger.record("train/policy_loss", policy_loss_value)
        self.logger.record("train/policy_loss_ppo", policy_loss_ppo_value)
        self.logger.record("train/policy_loss_bc", policy_loss_bc_value)
        self.logger.record("train/policy_loss_bc_weighted", policy_loss_bc_weighted_value)
        self.logger.record("train/policy_bc_vs_ppo_ratio", bc_ratio)
        cvar_empirical_ema_value = float(
            self._cvar_empirical_ema if self._cvar_empirical_ema is not None else cvar_empirical_value
        )
        cvar_violation_ema_value = float(
            self._cvar_violation_ema if self._cvar_violation_ema is not None else cvar_violation
        )
        self.logger.record("train/value_ce_loss", critic_loss_value)
        self._record_cvar_logs(
            cvar_raw_value=cvar_raw_value,
            cvar_unit_value=cvar_unit_value,
            cvar_loss_raw_value=cvar_loss_value,
            cvar_loss_unit_value=cvar_loss_unit_value,
            cvar_term_raw_value=cvar_term_raw_value,
            cvar_term_unit_value=cvar_term_value,
            cvar_empirical_value=cvar_empirical_value,
            cvar_empirical_unit_value=cvar_empirical_unit_value,
            cvar_empirical_ema_value=cvar_empirical_ema_value,
            cvar_violation_raw_value=cvar_violation,
            cvar_violation_raw_unclipped_value=cvar_violation_raw,
            cvar_violation_unit_value=cvar_violation_unit_value,
            cvar_violation_ema_value=cvar_violation_ema_value,
            cvar_gap_raw_value=cvar_gap_value,
            cvar_gap_unit_value=cvar_gap_unit_value,
            cvar_penalty_active_value=cvar_penalty_active_value,
            cvar_lambda_value=float(self.cvar_lambda),
            cvar_scale_value=cvar_scale_value,
            cvar_limit_raw_value=cvar_limit_raw_value,
            cvar_limit_unit_value=cvar_limit_unit_value,
            current_cvar_weight_scaled=current_cvar_weight_scaled,
            current_cvar_weight_nominal=current_cvar_weight_nominal,
            current_cvar_weight_raw=current_cvar_weight_raw,
            cvar_penalty_cap_value=float(self.cvar_penalty_cap)
            if self.cvar_penalty_cap is not None
            else 0.0,
        )
        if reward_costs_fraction_value is not None:
            self.logger.record(
                "train/reward_costs_in_fraction", reward_costs_fraction_value
            )
            if reward_costs_fraction_mean_value is not None:
                self.logger.record(
                    "train/reward_costs_mean_in_fraction",
                    reward_costs_fraction_mean_value,
                )
        if (
            self._reward_robust_clip_fraction is not None
            and math.isfinite(self._reward_robust_clip_fraction)
        ):
            self.logger.record(
                "train/reward_robust_clip_in_fraction",
                float(self._reward_robust_clip_fraction),
            )
        self.logger.record("train/reward_raw_p50_in_fraction", reward_raw_p50_value)
        self.logger.record("train/reward_raw_p95_in_fraction", reward_raw_p95_value)
        self.logger.record(
            "train/returns_abs_p95_in_fraction", returns_abs_p95_fraction_value
        )
        self.logger.record("train/returns_abs_p95", returns_abs_p95_value)
        # Backward-compatible aliases without explicit units
        self.logger.record("train/reward_raw_p50", reward_raw_p50_value)
        self.logger.record("train/reward_raw_p95", reward_raw_p95_value)
        self.logger.record(
            "train/reward_clip_bound_in_fraction", float(clip_bound_value)
        )
        self.logger.record(
            "train/reward_clip_bound_is_cap_fraction",
            float(clip_bound_cap_frac_logged),
        )
        if clip_bound_min_value is not None:
            self.logger.record(
                "train/reward_clip_bound_min_in_fraction",
                float(clip_bound_min_value),
            )
        if clip_bound_max_value is not None:
            self.logger.record(
                "train/reward_clip_bound_max_in_fraction",
                float(clip_bound_max_value),
            )
        self.logger.record("debug/cvar_limit", cvar_limit_raw_value)
        if self.cvar_use_constraint:
            self.logger.record("train/cvar_constraint", constraint_term_value)
        if self.cvar_cap is not None:
            self.logger.record("train/cvar_cap", self.cvar_cap)
        if self.cvar_penalty_cap is not None:
            self.logger.record("train/cvar_penalty_cap", float(self.cvar_penalty_cap))
        self.logger.record("train/value_mse", value_mse_value)

        self.logger.record("train/entropy_loss", -avg_policy_entropy)
        self.logger.record("train/policy_entropy_slope", self._last_entropy_slope)
        self.logger.record("train/entropy_plateau", float(self._entropy_plateau))
        decay_start = self._entropy_decay_start_update if self._entropy_decay_start_update is not None else -1
        self.logger.record("train/entropy_decay_start_update", float(decay_start))

        self.logger.record("train/ent_coef", float(ent_coef_raw_value))
        self.logger.record("train/ent_coef_nominal", float(ent_coef_nominal_value))
        self.logger.record("train/ent_coef_eff", float(ent_coef_eff_value))
        self.logger.record("train/ent_coef_min", float(self.ent_coef_min))
        self.logger.record(
            "train/ent_coef_autoclamp",
            1.0 if ent_coef_autoclamp_flag else 0.0,
        )
        self.logger.record("train/vf_coef_effective", float(vf_coef_effective))
        self.logger.record("debug/cvar_penalty_scale", float(cvar_penalty_scale))
        self.logger.record("train/critic_gradient_blocked", float(self._critic_grad_blocked))
        if (not self._use_quantile_value) and clamp_raw_weight > 0.0:
            self.logger.record(
                "train/value_target_below_frac_raw",
                clamp_below_raw_sum / clamp_raw_weight,
            )
            self.logger.record(
                "train/value_target_above_frac_raw",
                clamp_above_raw_sum / clamp_raw_weight,
            )
        if (not self._use_quantile_value) and clamp_weight > 0.0:
            self.logger.record("train/value_target_below_frac", clamp_below_sum / clamp_weight)
            self.logger.record("train/value_target_above_frac", clamp_above_sum / clamp_weight)
        if not self._use_quantile_value:
            self.logger.record(
                "train/value_target_raw_outlier_threshold",
                float(self._value_target_raw_outlier_warn_threshold),
            )
            self.logger.record("train/value_target_raw_outlier_max_frac", raw_outlier_frac_max)
            if raw_outlier_warn_count > 0:
                self.logger.record(
                    "warn/value_target_raw_outlier_batches", float(raw_outlier_warn_count)
                )
        self.logger.record("train/ent_coef_initial", float(self.ent_coef_initial))
        self.logger.record("train/ent_coef_final", float(self.ent_coef_final))

        approx_kl_exceed_frac = (
            float(kl_exceed_fraction_latest)
            if minibatches_processed > 0
            else 0.0
        )
        if len(approx_kl_divs) > 0:
            approx_kl_array = np.asarray(approx_kl_divs, dtype=np.float64)
            approx_kl_mean = float(np.mean(approx_kl_array))
            self.logger.record("train/approx_kl", approx_kl_mean)
            self.logger.record("train/approx_kl_median", float(np.median(approx_kl_array)))
            self.logger.record("train/approx_kl_p90", float(np.quantile(approx_kl_array, 0.9)))
            self.logger.record("train/approx_kl_max", float(np.max(approx_kl_array)))
            if self.target_kl is not None and self.target_kl > 0.0:
                self._adjust_kl_penalty(approx_kl_mean)
        self.logger.record("train/approx_kl_ema", float(approx_kl_smooth_latest))
        if self.target_kl is not None and self.target_kl > 0.0:
            self.logger.record("train/kl_exceed_frac", approx_kl_exceed_frac)
            self.logger.record("train/kl_exceed_stop_fraction", float(self.kl_exceed_stop_fraction))
            self.logger.record("train/kl_last_exceeded", float(approx_kl_last_exceeded_raw))
            self.logger.record("train/kl_exceed_consec_max", float(kl_exceed_consec_max_latest))
            if kl_early_stop_triggered:
                self.logger.record("train/kl_exceed_frac_at_stop", float(kl_exceed_fraction_latest))
        if last_optimizer_lr is not None:
            self.logger.record("train/optimizer_lr", last_optimizer_lr)
        if last_scheduler_lr is not None:
            self.logger.record("train/scheduler_lr", last_scheduler_lr)
        self.logger.record("train/loss", total_loss_value)
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/clip_range", float(self._clip_range_current))
        clip_range_for_log = float(self.clip_range(self._current_progress_remaining))
        self.logger.record("train/clip_range_schedule", clip_range_for_log)
        if ratio_count > 0:
            ratio_mean = ratio_sum / float(ratio_count)
            ratio_var = max(ratio_sq_sum / float(ratio_count) - ratio_mean**2, 0.0)
            ratio_std = math.sqrt(ratio_var)
            self.logger.record("train/ratio_mean", float(ratio_mean))
            self.logger.record("train/ratio_std", float(ratio_std))
        if log_prob_count > 0:
            self.logger.record("train/log_prob_mean", float(log_prob_sum / float(log_prob_count)))
        if adv_batch_count > 0:
            self.logger.record("train/adv_mean", adv_mean_accum / adv_batch_count)
            self.logger.record("train/adv_std", adv_std_accum / adv_batch_count)
        if adv_z_values:
            adv_z_tensor = torch.cat(adv_z_values)
            if adv_z_tensor.numel() > 0:
                adv_z_tensor = adv_z_tensor.to(dtype=torch.float32)
                quantiles = torch.quantile(
                    adv_z_tensor,
                    torch.tensor([0.1, 0.5, 0.9], dtype=adv_z_tensor.dtype, device=adv_z_tensor.device),
                )
                self.logger.record("train/adv_z_p10", float(quantiles[0].item()))
                self.logger.record("train/adv_z_p50", float(quantiles[1].item()))
                self.logger.record("train/adv_z_p90", float(quantiles[2].item()))
        self.logger.record("train/n_epochs_effective", float(effective_n_epochs))
        self.logger.record("train/n_epochs_completed", float(epochs_completed))
        self.logger.record("train/n_minibatches_done", float(minibatches_processed))
        self.logger.record("train/kl_early_stop", int(kl_early_stop_triggered))
        if clip_fraction_denom > 0:
            self.logger.record(
                "train/clip_fraction",
                float(clip_fraction_numer) / float(clip_fraction_denom),
            )
        if self.target_kl is not None and self.target_kl > 0.0:
            self.logger.record("train/target_kl", float(self.target_kl))
            self.logger.record("train/kl_lr_scale", float(self._kl_lr_scale))
            self.logger.record("train/kl_epoch_factor", float(self._kl_epoch_factor))
            self.logger.record("train/kl_lr_decay_applied", 0.0)
            self.logger.record("train/kl_penalty_beta", float(self.kl_beta))
            self.logger.record("train/kl_penalty_error", float(self._kl_penalty_error))
            self.logger.record("train/kl_penalty_pid_p", float(self._kl_pid_p))
            self.logger.record("train/kl_penalty_pid_i", float(self._kl_pid_i))
            self.logger.record("train/kl_penalty_pid_d", float(self._kl_pid_d))
            if kl_penalty_component_count > 0:
                self.logger.record(
                    "train/policy_loss_kl_penalty",
                    float(kl_penalty_component_total) / float(kl_penalty_component_count),
                )

        if y_true_tensor.numel() > 0 and y_pred_tensor.numel() > 0:
            y_true_np = y_true_tensor.flatten().detach().cpu().numpy().astype(np.float64)
            y_pred_np = y_pred_tensor.flatten().detach().cpu().numpy().astype(np.float64)
            self.logger.record("train/value_pred_mean", float(np.mean(y_pred_np)))
            self.logger.record("train/value_pred_std", float(np.std(y_pred_np)))
            self.logger.record("train/target_return_mean", float(np.mean(y_true_np)))
            self.logger.record("train/target_return_std", float(np.std(y_true_np)))
            diff_np = y_pred_np - y_true_np
            self.logger.record("train/value_mae", float(np.mean(np.abs(diff_np))))
            self.logger.record(
                "train/value_rmse", float(math.sqrt(np.mean(np.square(diff_np))))
            )

        self._finalize_return_stats()

    def _serialize_kl_penalty_state(self) -> dict[str, float]:
        return {
            "kl_beta": float(self.kl_beta),
            "kl_err_int": float(self._kl_err_int),
            "kl_err_prev": float(self._kl_err_prev),
            "kl_penalty_error": float(self._kl_penalty_error),
            "kl_pid_p": float(self._kl_pid_p),
            "kl_pid_i": float(self._kl_pid_i),
            "kl_pid_d": float(self._kl_pid_d),
        }

    def _restore_kl_penalty_state(self, state: Optional[Mapping[str, Any]]) -> None:
        if not isinstance(state, Mapping):
            return

        def _safe_float(key: str, default: float) -> float:
            value = state.get(key, default)
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return default
            if not math.isfinite(value_f):
                return default
            return value_f

        beta_value = _safe_float("kl_beta", float(self.kl_beta))
        beta_value = min(max(beta_value, self.kl_penalty_beta_min), self.kl_penalty_beta_max)
        self.kl_beta = beta_value
        self._kl_err_int = _safe_float("kl_err_int", float(self._kl_err_int))
        self._kl_err_prev = _safe_float("kl_err_prev", float(self._kl_err_prev))
        self._kl_penalty_error = _safe_float("kl_penalty_error", float(self._kl_penalty_error))
        self._kl_pid_p = _safe_float("kl_pid_p", float(self._kl_pid_p))
        self._kl_pid_i = _safe_float("kl_pid_i", float(self._kl_pid_i))
        self._kl_pid_d = _safe_float("kl_pid_d", float(self._kl_pid_d))

        integral_limit = self._kl_integral_limit()
        self._kl_err_int = float(np.clip(self._kl_err_int, -integral_limit, integral_limit))

    def get_parameters(self) -> dict[str, dict]:
        params = super().get_parameters()
        params["kl_penalty_state"] = self._serialize_kl_penalty_state()
        return params

    def set_parameters(
        self,
        load_path_or_dict: Union[str, io.BufferedIOBase, Mapping[str, Any]],
        exact_match: bool = True,
        device: Union[torch.device, str] = "auto",
    ) -> None:
        if isinstance(load_path_or_dict, Mapping):
            params: dict[str, Any] = dict(load_path_or_dict)
        else:
            _, params_loaded, _ = load_from_zip_file(
                load_path_or_dict,
                device=device,
                load_data=False,
            )
            params = dict(params_loaded)

        kl_state = params.pop("kl_penalty_state", None)
        super().set_parameters(params, exact_match=exact_match, device=device)
        self._restore_kl_penalty_state(kl_state)

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Union[BaseCallback, Sequence[BaseCallback]]] = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "DistributionalPPO":
        callbacks: list[BaseCallback] = []

        if callback is not None:
            if isinstance(callback, BaseCallback):
                callbacks.append(callback)
            else:
                callbacks.extend(callback)

        callback_for_super: Optional[BaseCallback]
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                warn=True,
            )
            callbacks.append(eval_callback)
            callback_for_super = (
                callbacks[0]
                if len(callbacks) == 1
                else CallbackList(callbacks)
            )
        else:
            if not callbacks:
                callback_for_super = None
            elif len(callbacks) == 1:
                callback_for_super = callbacks[0]
            else:
                callback_for_super = CallbackList(callbacks)

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback_for_super,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[GymEnv] = None,
        device: Union[str, torch.device] = "auto",
        custom_objects: Optional[dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> "DistributionalPPO":
        model = super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )
        if isinstance(model, DistributionalPPO):
            model._ensure_score_action_space()
        return model


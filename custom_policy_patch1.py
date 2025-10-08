# Имя файла: custom_policy_patch.py
# ИЗМЕНЕНИЯ (АРХИТЕКТУРНЫЙ РЕФАКТОРИНГ):
# 1. Устранен конфликт двух механизмов памяти (Трансформер vs GRU).
# 2. Выбран путь "чистой рекуррентности": GRU становится основным и единственным
#    механизмом памяти.
# 3. CustomMlpExtractor радикально упрощен. Теперь это не сложный анализатор
#    окон, а простой MLP-кодировщик признаков ОДНОГО временного шага.
# 4. Все неиспользуемые более классы (Attention, Conv блоки и т.д.) удалены.
# 
# --- ИСПРАВЛЕНИЕ IndexError ---
# Применены точечные исправления к CustomActorCriticPolicy, чтобы она
# корректно работала с новой GRU-архитектурой, не затрагивая остальной код.

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from gymnasium import spaces
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule  # тип коллбэка lr_schedule

from action_proto import ActionType



class CustomMlpExtractor(nn.Module):
    """
    Простой MLP-экстрактор для кодирования признаков одного временного шага.
    Он заменяет сложную трансформер-подобную архитектуру, передавая
    задачу обработки последовательности рекуррентной сети (GRU).
    """
    def __init__(self, feature_dim: int, hidden_dim: int, activation: Type[nn.Module]):
        super().__init__()
        # Определяем размерность для actor и critic
        self.latent_dim_pi = hidden_dim
        self.latent_dim_vf = hidden_dim

        # Простая полносвязная сеть для проекции признаков
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Просто прогоняем признаки через MLP
        return self.input_projection(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return features


class CustomActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        # правильное имя в SB3:
        lr_schedule: Optional[Schedule] = None,
        *args,
        # бэк-компат: если кто-то ещё передаёт старое имя:
        lr_scheduler: Optional[Schedule] = None,
        optimizer_class=None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        arch_params=None,
        optimizer_scheduler_fn: Optional[Callable[[Optimizer], Any]] = None,
        **kwargs,
    ):
        # если нам пришло lr_scheduler (старое имя) — мапим на lr_schedule
        if lr_schedule is None and lr_scheduler is not None:
            lr_schedule = lr_scheduler

        arch_params = arch_params or {}
        hidden_dim = arch_params.get('hidden_dim', 64)
        # SB3 ожидает, что lstm_hidden_size задаёт фактическую размерность скрытого
        # состояния. Если мы не пробросим это значение, политика внутри базового
        # класса создаст LSTM размером по умолчанию (256), а дальнейшие головы,
        # построенные на «hidden_dim», начнут конфликтовать по размерностям.
        kwargs = dict(kwargs)
        exec_mode_override = kwargs.pop("execution_mode", None)
        include_heads_override = kwargs.pop("include_heads", None)
        kwargs.setdefault("lstm_hidden_size", hidden_dim)
        enable_critic_lstm = arch_params.get("enable_critic_lstm")
        if enable_critic_lstm is not None:
            kwargs.setdefault("enable_critic_lstm", enable_critic_lstm)
        self.hidden_dim = hidden_dim

        if optimizer_class is None:
            optimizer_class = torch.optim.Adam

        # super().__init__ вызывает _build, поэтому заранее сохраняем размерность действия
        if isinstance(action_space, spaces.Box):
            self.action_dim = int(np.prod(action_space.shape))
            self._multi_discrete_nvec: Optional[np.ndarray] = None
        elif isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
            self._multi_discrete_nvec = None
        elif isinstance(action_space, spaces.MultiDiscrete):
            # MultiDiscrete actions are modeled via a MultiCategorical distribution
            # whose logits are concatenated for every sub-action.
            self._multi_discrete_nvec = action_space.nvec.astype(np.int64)
            self.action_dim = int(self._multi_discrete_nvec.sum())
            if self._multi_discrete_nvec.size == 4:
                self._multi_discrete_head_names: Tuple[str, ...] = (
                    "price_offset_ticks",
                    "ttl_steps",
                    "type",
                    "volume_frac",
                )
            else:
                self._multi_discrete_head_names = tuple(
                    f"head_{i}" for i in range(int(self._multi_discrete_nvec.size))
                )
            self._multi_head_sizes: Tuple[int, ...] = tuple(
                int(x) for x in self._multi_discrete_nvec.tolist()
            )
            name_to_idx = {name: idx for idx, name in enumerate(self._multi_discrete_head_names)}
            self._price_head_index: Optional[int] = name_to_idx.get("price_offset_ticks")
            self._ttl_head_index: Optional[int] = name_to_idx.get("ttl_steps")
            self._type_head_index: Optional[int] = name_to_idx.get("type")
            self._volume_head_index: Optional[int] = name_to_idx.get("volume_frac")
        else:
            raise NotImplementedError(
                f"Action space {type(action_space)} is not supported by CustomActorCriticPolicy"
            )

        if not hasattr(self, "_multi_head_sizes"):
            self._multi_head_sizes = tuple()
        if not hasattr(self, "_price_head_index"):
            self._price_head_index = None
        if not hasattr(self, "_ttl_head_index"):
            self._ttl_head_index = None
        if not hasattr(self, "_type_head_index"):
            self._type_head_index = None
        if not hasattr(self, "_volume_head_index"):
            self._volume_head_index = None

        try:
            self._bar_market_type_index = int(ActionType.MARKET)
        except Exception:
            self._bar_market_type_index = 0
        self._bar_fixed_price_offset: int = 0
        self._bar_fixed_ttl: int = 0
        self._active_heads_logged: bool = False
        self._include_heads_bool: Optional[Dict[str, bool]] = None
        self._execution_mode = self._coerce_execution_mode(
            arch_params.get("execution_mode") if arch_params else None
        )
        if exec_mode_override is not None and not self._execution_mode:
            self._execution_mode = self._coerce_execution_mode(exec_mode_override)
        include_heads_cfg = arch_params.get("include_heads") if arch_params else None
        if isinstance(include_heads_cfg, Mapping) and not include_heads_cfg:
            include_heads_cfg = None
        self._register_active_heads(include_heads_cfg)
        if include_heads_override is not None:
            self._register_active_heads(include_heads_override)

        self._loss_head_weights_tensor: Optional[torch.Tensor] = None

        act_str = arch_params.get('activation', 'relu').lower()
        if act_str == 'relu':
            self.activation = nn.ReLU
        elif act_str == 'tanh':
            self.activation = nn.Tanh
        elif act_str == 'leakyrelu':
            self.activation = nn.LeakyReLU
        else:
            self.activation = nn.ReLU

        # Параметры n_res_blocks, n_attn_blocks, attn_heads больше не нужны
        
        self.use_memory = True  # Память обеспечивается рекуррентными слоями SB3

        def _coerce_arch_float(value: Optional[float], fallback: float, key: str) -> float:
            if value is None:
                return fallback
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid '{key}' value in policy arch params: {value}") from exc

        def _coerce_arch_int(value: Optional[int], fallback: int, key: str) -> int:
            if value is None:
                return fallback
            if isinstance(value, bool):
                raise ValueError(f"Invalid '{key}' value in policy arch params: {value}")
            if isinstance(value, int):
                return int(value)
            try:
                coerced = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid '{key}' value in policy arch params: {value}") from exc
            return coerced

        self.num_atoms = _coerce_arch_int(arch_params.get("num_atoms"), 51, "num_atoms")
        self.v_min = _coerce_arch_float(arch_params.get("v_min"), -1.0, "v_min")
        self.v_max = _coerce_arch_float(arch_params.get("v_max"), 1.0, "v_max")
        if self.num_atoms < 1:
            raise ValueError(
                f"Invalid 'num_atoms' for distributional value head: {self.num_atoms} (must be >= 1)"
            )
        if self.v_max <= self.v_min:
            raise ValueError(
                f"Invalid value range for distributional value head: v_min={self.v_min}, v_max={self.v_max}"
            )

        clip_limit_cfg = arch_params.get("value_clip_limit")
        if clip_limit_cfg is not None:
            clip_limit = _coerce_arch_float(clip_limit_cfg, 1.0, "value_clip_limit")
        else:
            clip_limit = max(abs(self.v_min), abs(self.v_max))
        if clip_limit <= 0.0:
            raise ValueError(
                f"Invalid 'value_clip_limit' for distributional value head: {clip_limit} (must be > 0)"
            )
        self.value_clip_limit = float(clip_limit)

        self.optimizer_scheduler_fn = optimizer_scheduler_fn

        # dist_head создаётся позже в _build, но атрибут инициализируем заранее,
        # чтобы на него можно было безопасно ссылаться до сборки модели.
        self.dist_head: Optional[nn.Linear] = None
        self._last_value_logits: Optional[torch.Tensor] = None
        self._critic_gradient_blocked: bool = False

        # lr_schedule уже используется базовым классом во время вызова super().__init__.
        # Сохраняем ссылку заранее, чтобы можно было переинициализировать оптимизатор
        # после того, как базовый класс создаст рекуррентные слои (LSTM).
        self._pending_lr_schedule = lr_schedule

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

        # После инициализации базовый класс знает фактическую размерность скрытого
        # состояния (self.lstm_output_dim). Синхронизируем её с кастомным полем,
        # чтобы избежать расхождений при создании голов актёра и критика.
        self.hidden_dim = self.lstm_output_dim

        # буфер с опорой атомов остаётся
        atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.register_buffer("atoms", atoms)

        # совместимость: где-то в старом коде могли обращаться к self.value_net
        self.value_net = self.dist_head

        

        if isinstance(self.action_space, spaces.Box):
            self.unconstrained_log_std = nn.Parameter(torch.zeros(self.action_dim))

        def _zeros_for(module: Optional[nn.Module]) -> Tuple[torch.Tensor, ...]:
            zeros = torch.zeros(self.lstm_hidden_state_shape, device=self.device)
            if isinstance(module, nn.GRU):
                return (zeros.clone(),)
            return (zeros.clone(), torch.zeros_like(zeros))

        pi_initial = _zeros_for(self.lstm_actor)
        if self.lstm_critic is not None:
            vf_initial = _zeros_for(self.lstm_critic)
        else:
            vf_initial = tuple(t.clone() for t in pi_initial)

        self.recurrent_initial_state = RNNStates(pi=pi_initial, vf=vf_initial)

        # После того как LSTM-слои фактически созданы, переинициализируем оптимизатор
        # с нужным набором параметров и (опционально) кастомным планировщиком.
        self._setup_custom_optimizer()

    def _coerce_execution_mode(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip().lower()
        try:
            return str(value).strip().lower()
        except Exception:
            return ""

    @staticmethod
    def _coerce_head_flag(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(numeric):
                return None
            return numeric != 0.0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if not lowered:
                return None
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return bool(value.item())
        return None

    def _resolve_include_heads(
        self, payload: Optional[Mapping[str, Any] | Sequence[Any]]
    ) -> Optional[Dict[str, bool]]:
        if self._multi_discrete_nvec is None or int(self._multi_discrete_nvec.size) == 0:
            return None
        if payload is None:
            return None

        mapping_payload: Optional[Mapping[str, Any]]
        sequence_payload: Optional[Sequence[Any]]
        if isinstance(payload, Mapping):
            mapping_payload = payload
            sequence_payload = None
        elif isinstance(payload, (list, tuple)):
            mapping_payload = None
            sequence_payload = payload
        else:
            return None

        num_heads = int(self._multi_discrete_nvec.size)
        head_names = getattr(self, "_multi_discrete_head_names", tuple())
        resolved: Dict[str, bool] = {}

        for idx in range(num_heads):
            head_name = head_names[idx] if idx < len(head_names) else f"head_{idx}"
            raw_value: Any = None
            if mapping_payload is not None:
                if idx < len(head_names):
                    raw_value = mapping_payload.get(head_names[idx])
                if raw_value is None:
                    raw_value = mapping_payload.get(str(idx))
                if raw_value is None:
                    raw_value = mapping_payload.get(idx)
            if raw_value is None and sequence_payload is not None and idx < len(sequence_payload):
                raw_value = sequence_payload[idx]

            flag = self._coerce_head_flag(raw_value)
            if flag is not None:
                resolved[head_name] = flag

        if not resolved:
            if head_names:
                return {name: True for name in head_names}
            return None

        for idx in range(num_heads):
            head_name = head_names[idx] if idx < len(head_names) else f"head_{idx}"
            if head_name not in resolved:
                resolved[head_name] = True

        return resolved

    def _register_active_heads(
        self, payload: Optional[Mapping[str, Any] | Sequence[Any]]
    ) -> None:
        resolved = self._resolve_include_heads(payload)
        if resolved is None:
            return

        self._include_heads_bool = resolved
        if not self._active_heads_logged:
            active_names = [name for name, enabled in resolved.items() if enabled]
            active_repr = ", ".join(active_names) if active_names else "none"
            print(f"[CustomActorCriticPolicy] Active action heads: {active_repr}")
            self._active_heads_logged = True

        if self._execution_mode == "bar":
            expected = {
                "type": False,
                "price_offset_ticks": False,
                "ttl_steps": False,
                "volume_frac": True,
            }
            canonical = {name: resolved.get(name, True) for name in expected}
            if canonical != expected:
                raise ValueError(
                    "BAR execution mode requires include_heads to exactly disable type, "
                    "price_offset_ticks and ttl_steps while keeping volume_frac enabled"
                )

    def _is_bar_execution_mode(self) -> bool:
        return self._execution_mode == "bar" and self._volume_head_index is not None

    def _extract_volume_logits(self, action_logits: torch.Tensor) -> torch.Tensor:
        if self._volume_head_index is None or not self._multi_head_sizes:
            raise RuntimeError("Volume head is not configured for BAR execution mode")
        start = int(sum(self._multi_head_sizes[: self._volume_head_index]))
        end = start + int(self._multi_head_sizes[self._volume_head_index])
        return action_logits[..., start:end]

    def _bar_action_distribution(self, latent_pi: torch.Tensor) -> torch.distributions.Categorical:
        action_logits = self.action_net(latent_pi)
        volume_logits = self._extract_volume_logits(action_logits)
        return torch.distributions.Categorical(logits=volume_logits)

    def _assemble_bar_actions(self, volume_actions: torch.Tensor) -> torch.Tensor:
        if not self._multi_head_sizes:
            return volume_actions.unsqueeze(-1)

        batch_size = volume_actions.shape[0]
        num_heads = len(self._multi_head_sizes)
        actions = torch.zeros(
            (batch_size, num_heads), device=volume_actions.device, dtype=torch.long
        )
        if self._price_head_index is not None:
            actions[:, self._price_head_index] = self._bar_fixed_price_offset
        if self._ttl_head_index is not None:
            actions[:, self._ttl_head_index] = self._bar_fixed_ttl
        if self._type_head_index is not None:
            actions[:, self._type_head_index] = self._bar_market_type_index
        if self._volume_head_index is not None:
            actions[:, self._volume_head_index] = volume_actions.to(dtype=torch.long)
        return actions

    @torch.no_grad()
    def update_atoms(self, v_min: float, v_max: float) -> None:
        """
        Динамически обновляет диапазон и сами атомы для value-распределения.
        """
        # Проверяем, изменился ли диапазон, чтобы не делать лишнюю работу
        if self.v_min == v_min and self.v_max == v_max:
            return

        self.v_min = v_min
        self.v_max = v_max
        
        # Создаем временный тензор с новыми значениями
        new_atoms = torch.linspace(v_min, v_max, self.num_atoms, device=self.atoms.device)
        
        # Копируем данные "in-place" в существующий буфер.
        # Это сохраняет регистрацию и гарантирует правильное сохранение/загрузку.
        self.atoms.copy_(new_atoms)

    def _build_mlp_extractor(self) -> None:
        # Теперь создается простой MLP экстрактор
        self.mlp_extractor = CustomMlpExtractor(
            feature_dim=self.features_dim, 
            hidden_dim=self.hidden_dim,
            activation=self.activation
        )
    def _build(self, lr_schedule) -> None:
        """
        Создаёт архитектуру сети, используя базовую реализацию SB3, а затем
        заменяет value-голову на дистрибутивную.
        """
        super()._build(lr_schedule)

        # Перестраиваем value-голову на распределение атомов.
        self.dist_head = nn.Linear(self.lstm_output_dim, self.num_atoms)
        self.value_net = self.dist_head

        # Во время вызова _build из базового класса рекуррентные слои ещё не созданы,
        # поэтому откладываем настройку оптимизатора до завершения __init__.
        self._pending_lr_schedule = lr_schedule

    def _setup_custom_optimizer(self) -> None:
        """
        Настраивает оптимизатор и (опционально) планировщик, используя только те
        параметры, которые должны участвовать в обучении кастомной политики.
        """

        lr_schedule = getattr(self, "_pending_lr_schedule", None)
        if lr_schedule is None:
            return

        modules: list[nn.Module] = [self.mlp_extractor]

        lstm_actor = getattr(self, "lstm_actor", None)
        if lstm_actor is not None:
            modules.append(lstm_actor)

        lstm_critic = getattr(self, "lstm_critic", None)
        if lstm_critic is not None and lstm_critic is not lstm_actor:
            modules.append(lstm_critic)

        modules.extend([self.action_net, self.dist_head])

        params: list[nn.Parameter] = []
        for module in modules:
            params.extend(module.parameters())

        if getattr(self, "log_std", None) is not None:
            params.append(self.log_std)
        if hasattr(self, "unconstrained_log_std"):
            params.append(self.unconstrained_log_std)

        self.optimizer = self.optimizer_class(params, lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.optimizer_scheduler_fn is not None:
            self.optimizer_scheduler = self.optimizer_scheduler_fn(self.optimizer)
        else:
            self.optimizer_scheduler = None

        # После настройки оптимизатора ссылка на lr_schedule больше не нужна.
        self._pending_lr_schedule = None
    # --- ИСПРАВЛЕНИЕ: Метод переименован с forward_rnn на _forward_recurrent ---
    def _forward_recurrent(
        self,
        features: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, RNNStates]:
        """
        Обрабатывает последовательность признаков, используя рекуррентные блоки
        актёра и критика из базовой политики.
        Возвращает скрытые состояния и обновлённые RNNStates.
        """
        if self.share_features_extractor:
            pi_features = vf_features = features  # type: ignore[assignment]
        else:
            assert isinstance(features, tuple) and len(features) == 2
            pi_features, vf_features = features

        if self._critic_gradient_blocked:
            if isinstance(vf_features, torch.Tensor):
                vf_features = vf_features.detach()
            else:
                vf_features = tuple(feat.detach() for feat in vf_features)

        latent_pi, new_pi_states = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )

        if self.lstm_critic is not None:
            latent_vf, new_vf_states = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            new_vf_states = tuple(state.detach() for state in new_pi_states)
        else:
            latent_vf = self.critic(vf_features)
            new_vf_states = lstm_states.vf

        return latent_pi, latent_vf, RNNStates(new_pi_states, new_vf_states)

    # Backward compatibility with SB3 `RecurrentActorCriticPolicy` expectations.
    # Some call sites still invoke `forward_rnn`, so keep the public alias alive.
    def forward_rnn(
        self,
        features: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, RNNStates]:
        return self._forward_recurrent(features, lstm_states, episode_starts)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        if self._is_bar_execution_mode():
            return self._bar_action_distribution(latent_pi)
        if isinstance(self.action_space, spaces.Box):
            mean_actions = self.action_net(latent_pi)
            # Smoothly map the unconstrained parameter into the range [-5, 0]
            # torch.tanh returns [-1, 1]; rescale and shift it accordingly.
            log_std = -2.5 + 2.5 * torch.tanh(self.unconstrained_log_std)
            return self.action_dist.proba_distribution(mean_actions, log_std)
        elif isinstance(self.action_space, spaces.Discrete):
            action_logits = self.action_net(latent_pi)
            return self.action_dist.proba_distribution(action_logits)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            action_logits = self.action_net(latent_pi)
            # The underlying MultiCategorical distribution expects a concatenated
            # logits tensor of shape [batch_size, sum(nvec)]. The policy head
            # already produces the required dimensionality.
            return self.action_dist.proba_distribution(action_logits)
        else:
            raise NotImplementedError(f"Action space {type(self.action_space)} not supported")

    def _get_value_logits(self, latent_vf: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты распределения ценностей без их агрегации."""

        value_logits = self.dist_head(latent_vf)  # [B, n_atoms]
        self._last_value_logits = value_logits
        return value_logits

    def _value_from_logits(self, value_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(value_logits, dim=-1)
        return (probs * self.atoms).sum(dim=-1, keepdim=True)

    def _get_value_from_latent(self, latent_vf: torch.Tensor) -> torch.Tensor:
        """
        Переопределяем базовый метод SB3.
        Вместо отдельной линейной головы берём logits → probs → ожидание.
        """
        value_logits = self._get_value_logits(latent_vf)
        return self._value_from_logits(value_logits)

    def _loss_head_weights_for_device(
        self, device: torch.device
    ) -> Optional[torch.Tensor]:
        if self._loss_head_weights_tensor is None:
            return None
        if self._loss_head_weights_tensor.device != device:
            self._loss_head_weights_tensor = self._loss_head_weights_tensor.to(device=device)
        return self._loss_head_weights_tensor

    def set_critic_gradient_blocked(self, blocked: bool) -> None:
        self._critic_gradient_blocked = bool(blocked)

    def set_loss_head_weights(
        self, weights: Optional[Mapping[str, float] | Sequence[float]]
    ) -> None:
        if self._multi_discrete_nvec is None:
            self._loss_head_weights_tensor = None
            self._register_active_heads(weights)
            return
        if weights is None:
            self._loss_head_weights_tensor = None
            self._register_active_heads(weights)
            return

        values: list[float] = []
        num_heads = int(self._multi_discrete_nvec.size)
        if isinstance(weights, Mapping):
            head_names = getattr(self, "_multi_discrete_head_names", tuple())
            for idx in range(num_heads):
                raw_value = None
                if idx < len(head_names):
                    raw_value = weights.get(head_names[idx])
                if raw_value is None:
                    raw_value = weights.get(str(idx)) if isinstance(weights, Mapping) else None
                if raw_value is None:
                    raw_value = weights.get(idx) if isinstance(weights, Mapping) else None
                if raw_value is None:
                    values.append(1.0)
                    continue
                if isinstance(raw_value, bool):
                    values.append(1.0 if raw_value else 0.0)
                    continue
                try:
                    values.append(float(raw_value))
                except (TypeError, ValueError):
                    values.append(1.0)
        else:
            seq = list(weights)
            for idx in range(num_heads):
                if idx < len(seq):
                    try:
                        values.append(float(seq[idx]))
                    except (TypeError, ValueError):
                        values.append(1.0)
                else:
                    values.append(1.0)

        if not values or all(abs(v - 1.0) < 1e-8 for v in values):
            self._loss_head_weights_tensor = None
            return

        device = getattr(self, "device", torch.device("cpu"))
        self._loss_head_weights_tensor = torch.as_tensor(
            values, dtype=torch.float32, device=device
        )
        self._register_active_heads(weights)

    def _iter_multi_heads(
        self, distribution: torch.distributions.Distribution
    ) -> Optional[Sequence[torch.distributions.Distribution]]:
        comps = getattr(distribution, "distributions", None)
        if comps is None:
            inner = getattr(distribution, "distribution", None)
            comps = getattr(inner, "distributions", None)
        return comps if isinstance(comps, (list, tuple)) and len(comps) > 0 else None

    def _weighted_log_prob(
        self, distribution: torch.distributions.Distribution, actions: torch.Tensor
    ) -> torch.Tensor:
        if self._multi_discrete_nvec is None:
            return distribution.log_prob(actions)

        comps = self._iter_multi_heads(distribution)
        if comps is None:
            return distribution.log_prob(actions)

        weights = self._loss_head_weights_for_device(comps[0].logits.device)
        if weights is None or len(comps) != int(weights.shape[0]):
            return distribution.log_prob(actions)

        actions_tensor = torch.as_tensor(actions, device=comps[0].logits.device)
        if actions_tensor.ndim == 1:
            actions_tensor = actions_tensor.view(-1, len(comps))
        elif actions_tensor.ndim > 2:
            actions_tensor = actions_tensor.reshape(actions_tensor.shape[0], len(comps))
        actions_tensor = actions_tensor.to(dtype=torch.long)

        log_prob_sum = None
        for idx, (categorical, weight) in enumerate(zip(comps, weights)):
            if torch.is_tensor(weight):
                weight_value = float(weight.detach().cpu().item())
            else:
                weight_value = float(weight)
            if not math.isfinite(weight_value) or abs(weight_value) < 1e-12:
                continue
            head_log_prob = categorical.log_prob(actions_tensor[:, idx])
            if head_log_prob.ndim > 1:
                head_log_prob = head_log_prob.sum(dim=-1)
            head_log_prob = head_log_prob * weight_value
            log_prob_sum = head_log_prob if log_prob_sum is None else (log_prob_sum + head_log_prob)

        if log_prob_sum is None:
            base = comps[0].logits
            return base.new_zeros(actions_tensor.shape[0])

        return log_prob_sum

    def weighted_entropy(
        self, distribution: torch.distributions.Distribution
    ) -> torch.Tensor:
        if self._multi_discrete_nvec is None:
            return distribution.entropy()

        comps = self._iter_multi_heads(distribution)
        if comps is None:
            return distribution.entropy()

        weights = self._loss_head_weights_for_device(comps[0].logits.device)
        if weights is None or len(comps) != int(weights.shape[0]):
            return distribution.entropy()

        ent_sum = None
        for categorical, weight in zip(comps, weights):
            if torch.is_tensor(weight):
                weight_value = float(weight.detach().cpu().item())
            else:
                weight_value = float(weight)
            if not math.isfinite(weight_value) or abs(weight_value) < 1e-12:
                continue
            head_entropy = categorical.entropy() * weight_value
            ent_sum = head_entropy if ent_sum is None else (ent_sum + head_entropy)

        if ent_sum is None:
            base = comps[0].logits
            return base.new_zeros(base.shape[0])

        return ent_sum

    @property
    def last_value_logits(self) -> Optional[torch.Tensor]:
        return self._last_value_logits

    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: Optional[RNNStates],
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        if lstm_states is None:
            lstm_states = self.recurrent_initial_state

        features = self.extract_features(obs)
        latent_pi, latent_vf, new_states = self._forward_recurrent(features, lstm_states, episode_starts)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self._get_value_from_latent(latent_vf)

        if self._is_bar_execution_mode():
            distribution = self._bar_action_distribution(latent_pi)
            if deterministic:
                volume_actions = torch.argmax(distribution.logits, dim=-1)
            else:
                volume_actions = distribution.sample()
            volume_actions = volume_actions.to(dtype=torch.long)
            log_prob = distribution.log_prob(volume_actions)
            actions = self._assemble_bar_actions(volume_actions)
        else:
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = self._weighted_log_prob(distribution, actions)

        return actions, values, log_prob, new_states

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: Optional[RNNStates],
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lstm_states is None:
            lstm_states = self.recurrent_initial_state

        features = self.extract_features(obs)
        latent_pi, latent_vf, _ = self._forward_recurrent(features, lstm_states, episode_starts)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self._get_value_from_latent(latent_vf)

        if self._is_bar_execution_mode():
            distribution = self._bar_action_distribution(latent_pi)
            actions_tensor = torch.as_tensor(
                actions, device=distribution.logits.device
            )
            num_heads = len(self._multi_head_sizes) if self._multi_head_sizes else 1
            if actions_tensor.ndim == 1:
                actions_tensor = actions_tensor.view(-1, num_heads)
            elif actions_tensor.ndim > 2:
                actions_tensor = actions_tensor.reshape(actions_tensor.shape[0], num_heads)
            volume_actions = actions_tensor[:, self._volume_head_index]  # type: ignore[index]
            volume_actions = volume_actions.to(dtype=torch.long)
            log_prob = distribution.log_prob(volume_actions)
            entropy = distribution.entropy()
        else:
            distribution = self._get_action_dist_from_latent(latent_pi)
            log_prob = self._weighted_log_prob(distribution, actions)
            entropy = self.weighted_entropy(distribution)

        return values, log_prob, entropy

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, ...],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self._get_value_from_latent(latent_vf)

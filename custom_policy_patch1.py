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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type

from torch.optim import Optimizer

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule  # тип коллбэка lr_schedule
from stable_baselines3.common.utils import zip_strict

from utils.model_io import upgrade_quantile_value_state_dict


class QuantileValueHead(nn.Module):
    """Linear value head that predicts fixed equally spaced quantiles."""

    def __init__(self, input_dim: int, num_quantiles: int, huber_kappa: float) -> None:
        super().__init__()
        if num_quantiles <= 0:
            raise ValueError("'num_quantiles' must be positive for QuantileValueHead")
        self.num_quantiles = int(num_quantiles)
        self.huber_kappa = float(huber_kappa)
        if not math.isfinite(self.huber_kappa) or self.huber_kappa <= 0.0:
            raise ValueError("'huber_kappa' must be a positive finite value")
        self.linear = nn.Linear(input_dim, self.num_quantiles)
        taus = torch.linspace(0.0, 1.0, steps=self.num_quantiles + 1, dtype=torch.float32)
        midpoints = 0.5 * (taus[:-1] + taus[1:])
        self.register_buffer("taus", midpoints, persistent=True)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.linear(latent)



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


class _CategoricalAdapter:
    """Minimal adapter exposing SB3's distribution interface for torch Categorical."""

    def __init__(self, logits: torch.Tensor) -> None:
        self._dist = torch.distributions.Categorical(logits=logits)
        # Expose the wrapped distribution via a public attribute so that
        # downstream code does not need to rely on the private ``_dist`` name.
        self.distribution = self._dist

    def sample(self) -> torch.Tensor:
        return self._dist.sample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return torch.argmax(self._dist.logits, dim=-1)
        return self.sample()

    @property
    def logits(self) -> torch.Tensor:
        return self._dist.logits

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dist, name)


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
        kwargs.pop("execution_mode", None)
        kwargs.pop("include_heads", None)
        kwargs.setdefault("lstm_hidden_size", hidden_dim)
        enable_critic_lstm = arch_params.get("enable_critic_lstm")
        if enable_critic_lstm is not None:
            kwargs.setdefault("enable_critic_lstm", enable_critic_lstm)
        self.hidden_dim = hidden_dim

        if optimizer_class is None:
            optimizer_class = torch.optim.Adam

        if not isinstance(action_space, spaces.Box):
            raise NotImplementedError(
                f"Score policy requires Box action space, got {type(action_space)!r}"
            )
        if int(np.prod(action_space.shape)) != 1:
            raise ValueError(
                "Score policy expects a single-dimensional Box action space"
            )
        self.action_dim = 1
        self._multi_discrete_nvec = None
        self._multi_head_sizes: Tuple[int, ...] = tuple()
        self._price_head_index = None
        self._ttl_head_index = None
        self._type_head_index = None
        self._volume_head_index = None

        self._active_heads_logged = False
        self._include_heads_bool: Optional[Dict[str, bool]] = None
        self._execution_mode = "score"
        self._loss_head_weights_tensor: Optional[torch.Tensor] = None
        self._score_clip_eps: float = 5e-3  # используется только как fallback при logit() вне train-path

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

        critic_cfg_raw = arch_params.get("critic") if arch_params else None
        self._critic_cfg: Mapping[str, Any] | None = critic_cfg_raw if isinstance(
            critic_cfg_raw, Mapping
        ) else None
        critic_cfg = self._critic_cfg or {}

        distributional_flag = critic_cfg.get("distributional")
        self._use_quantile_value_head = bool(distributional_flag)

        if self._use_quantile_value_head:
            self.num_quantiles = _coerce_arch_int(
                critic_cfg.get("num_quantiles"), 32, "critic.num_quantiles"
            )
            if self.num_quantiles < 1:
                raise ValueError("'critic.num_quantiles' must be >= 1 when distributional critic is enabled")
            self.quantile_huber_kappa = _coerce_arch_float(
                critic_cfg.get("huber_kappa"), 1.0, "critic.huber_kappa"
            )
        else:
            self.num_quantiles = 0
            self.quantile_huber_kappa = 1.0

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
        self.quantile_head: Optional[QuantileValueHead] = None
        self._value_head_module: Optional[nn.Module] = None
        self._last_value_logits: Optional[torch.Tensor] = None
        self._last_value_quantiles: Optional[torch.Tensor] = None
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

        # ``ActorCriticPolicy`` exposes ``squash_output`` as a read-only property in SB3.
        # Older code mutates this flag, expecting to bypass action squashing.  In the
        # refactor this assignment happens after ``super().__init__`` which now raises
        # ``AttributeError`` because the property has no setter.  Provide a lightweight
        # override with our own setter so legacy code can keep toggling the behaviour.
        self.squash_output = False

        # После инициализации базовый класс знает фактическую размерность скрытого
        # состояния (self.lstm_output_dim). Синхронизируем её с кастомным полем,
        # чтобы избежать расхождений при создании голов актёра и критика.
        self.hidden_dim = self.lstm_output_dim

        # буфер с опорой атомов остаётся
        if self._use_quantile_value_head:
            atoms = torch.empty(0, dtype=torch.float32)
        else:
            atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.register_buffer("atoms", atoms)

        # совместимость: где-то в старом коде могли обращаться к self.value_net
        if not self._use_quantile_value_head:
            self.value_net = self.dist_head

        

        if isinstance(self.action_space, spaces.Box):
            self.unconstrained_log_std = nn.Parameter(torch.zeros(self.action_dim))
            param = getattr(self, "unconstrained_log_std", None)
            assert isinstance(param, torch.nn.Parameter), "missing unconstrained_log_std parameter"
            assert tuple(param.shape) == (self.action_dim,), \
                f"bad log_std shape {tuple(param.shape)} != ({self.action_dim},)"

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
        return False

    def _volume_actions_from_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 1:
            return actions.to(dtype=torch.long)
        if actions.ndim > 2:
            actions = actions.reshape(actions.shape[0], -1)
        idx = self._volume_head_index
        if idx is None:
            idx = actions.shape[-1] - 1
        return actions[:, idx].to(dtype=torch.long)

    @torch.no_grad()
    def update_atoms(self, v_min: float, v_max: float) -> None:
        """
        Динамически обновляет диапазон и сами атомы для value-распределения.
        """
        if self._use_quantile_value_head:
            return
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

        if self._use_quantile_value_head:
            self.quantile_head = QuantileValueHead(
                self.lstm_output_dim, self.num_quantiles, self.quantile_huber_kappa
            )
            self._value_head_module = self.quantile_head
            # In distributional mode with quantiles we intentionally drop the
            # categorical head; downstream code must use ``value_quantiles``
            # instead of ``dist_head``.
            self.dist_head = None
            self.value_net = self.quantile_head.linear
        else:
            self.dist_head = nn.Linear(self.lstm_output_dim, self.num_atoms)
            self._value_head_module = self.dist_head
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

        modules.append(self.action_net)
        if self._value_head_module is not None:
            modules.append(self._value_head_module)

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

    @staticmethod
    def _as_tensor_tuple(states: Any) -> Tuple[torch.Tensor, ...]:
        """Нормализует произвольный контейнер состояний в кортеж тензоров."""

        if isinstance(states, torch.Tensor):
            return (states,)

        if isinstance(states, (list, tuple)):
            tensor_seq: list[torch.Tensor] = []
            for item in states:
                if not isinstance(item, torch.Tensor):
                    raise TypeError(
                        "Компоненты RNN-состояний должны быть torch.Tensor, "
                        f"получено {type(item)!r}"
                    )
                tensor_seq.append(item)
            return tuple(tensor_seq)

        raise TypeError(f"Неподдерживаемый тип контейнера RNN-состояния: {type(states)!r}")

    def _align_state_tuple(
        self,
        states: Tuple[torch.Tensor, ...],
        reference: Tuple[torch.Tensor, ...],
        module: Optional[nn.Module],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Приводит список состояний к формату, ожидаемому рекуррентным модулем.

        Старый код иногда передаёт только скрытое состояние LSTM (без ячейки)
        или вовсе пустой кортеж. Чтобы не падать с ValueError во время оценки,
        добавляем недостающие компоненты, используя доступную форму.
        """

        if isinstance(module, nn.GRU):
            if not states:
                if not reference:
                    raise ValueError("Невозможно восстановить скрытое состояние GRU")
                return (reference[0].clone(),)
            # если пришло больше одного состояния, берём только скрытое
            return (states[0],)

        if isinstance(module, nn.LSTM):
            if not states:
                return tuple(t.clone() for t in reference)
            if len(states) == 1:
                hidden_state = states[0]
                cell_state = torch.zeros_like(hidden_state)
                return (hidden_state, cell_state)
            # обрезаем возможные лишние состояния от устаревших реализаций
            return states[:2]

        return states

    def _coerce_lstm_states(self, lstm_states: RNNStates | Tuple[Any, ...]) -> RNNStates:
        """Приводит состояния LSTM к современному контейнеру RNNStates."""

        if hasattr(lstm_states, "pi") and hasattr(lstm_states, "vf"):
            pi_states = self._align_state_tuple(
                tuple(lstm_states.pi), self.recurrent_initial_state.pi, self.lstm_actor
            )
            vf_states = self._align_state_tuple(
                tuple(lstm_states.vf), self.recurrent_initial_state.vf, self.lstm_critic
            )
            return RNNStates(pi=pi_states, vf=vf_states)

        if not isinstance(lstm_states, (list, tuple)):
            raise TypeError(
                "Ожидается RNNStates или tuple со скрытыми состояниями, "
                f"получено {type(lstm_states)!r}"
            )

        if len(lstm_states) == 2 and all(torch.is_tensor(x) for x in lstm_states):
            # Это плоское (h, c) актёра LSTM, а не (pi_states, vf_states)
            actor_states = self._align_state_tuple(
                tuple(lstm_states), self.recurrent_initial_state.pi, self.lstm_actor
            )
            if self.lstm_critic is None or getattr(self, "shared_lstm", False):
                vf_states = actor_states
            else:
                # отдельный критик: берём его нулевую инициализацию
                vf_states = tuple(t.clone() for t in self.recurrent_initial_state.vf)
            return RNNStates(pi=actor_states, vf=vf_states)

        if (
            len(lstm_states) == 2
            and (self.lstm_critic is not None or not getattr(self, "shared_lstm", False))
        ):
            pi_raw, vf_raw = lstm_states
            pi_states = self._align_state_tuple(
                self._as_tensor_tuple(pi_raw), self.recurrent_initial_state.pi, self.lstm_actor
            )
            vf_states = self._align_state_tuple(
                self._as_tensor_tuple(vf_raw), self.recurrent_initial_state.vf, self.lstm_critic
            )
            return RNNStates(pi=pi_states, vf=vf_states)

        actor_states = self._align_state_tuple(
            self._as_tensor_tuple(lstm_states), self.recurrent_initial_state.pi, self.lstm_actor
        )

        if self.lstm_critic is None or getattr(self, "shared_lstm", False):
            vf_states = self._align_state_tuple(
                actor_states, self.recurrent_initial_state.vf, self.lstm_critic
            )
            return RNNStates(pi=actor_states, vf=vf_states)

        half = len(actor_states) // 2
        if half > 0 and len(actor_states) % 2 == 0:
            pi_states = self._align_state_tuple(
                actor_states[:half], self.recurrent_initial_state.pi, self.lstm_actor
            )
            vf_states = self._align_state_tuple(
                actor_states[half:], self.recurrent_initial_state.vf, self.lstm_critic
            )
            return RNNStates(pi=pi_states, vf=vf_states)

        raise ValueError(
            "Неподдерживаемый формат tuple для отдельных состояний актёра/критика"
        )

    @staticmethod
    def _process_sequence(
        features: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, ...],
        episode_starts: torch.Tensor,
        recurrent_module: nn.Module,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Адаптация sb3_contrib для поддержки как LSTM, так и GRU."""

        if isinstance(features, torch.Tensor):
            target_device = features.device
        else:
            target_device = next(
                (
                    tensor.device
                    for tensor in features
                    if isinstance(tensor, torch.Tensor)
                ),
                episode_starts.device,
            )

        episode_starts = episode_starts.to(target_device)

        if isinstance(recurrent_module, nn.GRU):
            if not lstm_states:
                raise ValueError("GRU ожидает хотя бы одно скрытое состояние")

            hidden_state = lstm_states[0].to(target_device)
            # если вдруг пришли (h, c) от LSTM, просто игнорируем c
            n_seq = hidden_state.shape[1]

            features_sequence = features.reshape((n_seq, -1, recurrent_module.input_size)).swapaxes(0, 1)
            episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

            if torch.all(episode_starts == 0.0):
                output, hidden_state = recurrent_module(features_sequence, hidden_state)
                output = torch.flatten(output.transpose(0, 1), start_dim=0, end_dim=1)
                return output, (hidden_state,)

            outputs: list[torch.Tensor] = []
            for step_features, episode_start in zip_strict(features_sequence, episode_starts):
                hidden_state = (1.0 - episode_start).view(1, n_seq, 1) * hidden_state
                step_output, hidden_state = recurrent_module(step_features.unsqueeze(dim=0), hidden_state)
                outputs.append(step_output)

            output = torch.flatten(torch.cat(outputs).transpose(0, 1), start_dim=0, end_dim=1)
            return output, (hidden_state,)

        if not isinstance(recurrent_module, nn.LSTM):
            raise TypeError(
                f"Неподдерживаемый тип рекуррентного модуля: {type(recurrent_module)!r}"
            )

        if len(lstm_states) != 2:
            raise ValueError("LSTM ожидает кортеж из (hidden_state, cell_state)")

        lstm_hidden = tuple(state.to(target_device) for state in lstm_states[:2])

        return RecurrentActorCriticPolicy._process_sequence(
            features,
            lstm_hidden,
            episode_starts,
            recurrent_module,
        )

    def _forward_recurrent(
        self,
        features: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        lstm_states: RNNStates | Tuple[Any, ...],
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, RNNStates]:
        """
        Обрабатывает последовательность признаков, используя рекуррентные блоки
        актёра и критика из базовой политики.
        Возвращает скрытые состояния и обновлённые RNNStates.
        """
        lstm_states = self._coerce_lstm_states(lstm_states)

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
        mean_actions = self.action_net(latent_pi)
        # σ ∈ [0.2, 1.5] — безопаснее для сигмоидной головы
        sigma_min, sigma_max = 0.2, 1.5
        sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(self.unconstrained_log_std)
        log_std = torch.log(sigma)
        return self.action_dist.proba_distribution(mean_actions, log_std)

    def _get_value_logits(self, latent_vf: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты распределения/квантили ценностей без агрегации."""

        if self._use_quantile_value_head:
            if self.quantile_head is None:
                raise RuntimeError("Quantile value head is not initialised")
            quantiles = self.quantile_head(latent_vf)
            self._last_value_logits = None
            self._last_value_quantiles = quantiles
            return quantiles

        if self.dist_head is None:
            raise RuntimeError("Categorical value head is not initialised")
        value_logits = self.dist_head(latent_vf)  # [B, n_atoms]
        self._last_value_logits = value_logits
        self._last_value_quantiles = None
        return value_logits

    def _value_from_logits(self, value_logits: torch.Tensor) -> torch.Tensor:
        if self._use_quantile_value_head:
            return value_logits.mean(dim=-1, keepdim=True)

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

    def _prepare_score_tensor(self, actions: Any, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
        if tensor.ndim == 0:
            tensor = tensor.view(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.view(-1, 1)
        return tensor

    def _score_to_raw(self, scores: torch.Tensor) -> torch.Tensor:
        # Fallback для evaluate(): безопасный logit только когда нет raw
        clipped = torch.clamp(scores, self._score_clip_eps, 1.0 - self._score_clip_eps)
        return torch.log(clipped) - torch.log1p(-clipped)

    def _log_sigmoid_jacobian_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        # log σ'(x) = log σ(x) + log(1-σ(x)) = -softplus(-x) - softplus(x)
        return -(F.softplus(-raw) + F.softplus(raw))

    def _clamp_by_z(
        self,
        distribution: torch.distributions.Distribution,
        raw: torch.Tensor,
        zmax: float = 8.0,
    ) -> torch.Tensor:
        """Ограничить |z| = |(raw-μ)/σ| ≤ zmax, возвращая скорректированный raw."""
        inner = getattr(distribution, "distribution", None) or distribution
        mean = getattr(inner, "mean", None)
        if mean is None:
            mean = getattr(inner, "mean_actions", None)
        std = getattr(inner, "stddev", None)
        if std is None:
            get_std = getattr(inner, "get_std", None)
            std = get_std() if callable(get_std) else None
        if mean is None or std is None:
            # страховка на случай нестандартного адаптера
            return torch.clamp(raw, -8.0, 8.0)
        z = (raw - mean) / std
        z = torch.clamp(z, -zmax, zmax)
        return mean + std * z

    def _weighted_log_prob(
        self,
        distribution: torch.distributions.Distribution,
        actions: torch.Tensor,
        raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(self.action_space, spaces.Box):
            # Если есть реальный raw (из forward), работаем с ним.
            # В evaluate() raw нет — восстановим через безопасный logit.
            if raw is None:
                scores = self._prepare_score_tensor(actions, self.device)
                if not torch.isfinite(scores).all():
                    raise RuntimeError("Received non-finite score when computing log_prob")
                raw = self._score_to_raw(scores)
            elif not torch.isfinite(raw).all():
                raise RuntimeError("Received non-finite raw action when computing log_prob")
            raw_stable = self._clamp_by_z(distribution, raw, zmax=8.0)
            log_prob_raw = distribution.log_prob(raw_stable)
            log_det = self._log_sigmoid_jacobian_from_raw(raw).sum(dim=-1)
            # p_s(s) = p_raw(raw) / σ'(raw)  ⇒  log p_s = log p_raw - log σ'(raw)
            return log_prob_raw - log_det
        return distribution.log_prob(actions)

    def weighted_entropy(
        self, distribution: torch.distributions.Distribution
    ) -> torch.Tensor:
        entropy = distribution.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)

        inner = getattr(distribution, "distribution", None)
        if inner is None:
            mean = getattr(distribution, "mean_actions", None)
            get_std = getattr(distribution, "get_std", None)
            if mean is not None and callable(get_std):
                std = get_std()
                inner = torch.distributions.Normal(mean, std)

        if inner is None:
            return entropy

        raw = inner.rsample()
        # h(s) = h(raw) + E[log |ds/dr|] = h(raw) + E[log σ'(raw)]
        log_jac = self._log_sigmoid_jacobian_from_raw(raw)
        return entropy + log_jac.sum(dim=-1)

    @property
    def squash_output(self) -> bool:
        if hasattr(self, "_squash_output_override"):
            return bool(self._squash_output_override)
        return super().squash_output

    @squash_output.setter
    def squash_output(self, value: bool) -> None:
        self._squash_output_override = bool(value)

    @property
    def last_value_logits(self) -> Optional[torch.Tensor]:
        return self._last_value_logits

    @property
    def last_value_quantiles(self) -> Optional[torch.Tensor]:
        return self._last_value_quantiles

    @property
    def uses_quantile_value_head(self) -> bool:
        return self._use_quantile_value_head

    @property
    def quantile_levels(self) -> Optional[torch.Tensor]:
        if self.quantile_head is None:
            return None
        return self.quantile_head.taus

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

        distribution = self._get_action_dist_from_latent(latent_pi)
        raw_actions = distribution.get_actions(deterministic=deterministic)
        scores = torch.sigmoid(raw_actions)  # без clamp — работаем с реальным raw
        if not torch.isfinite(scores).all():
            raise RuntimeError("Policy produced non-finite score action")
        log_prob = self._weighted_log_prob(distribution, scores, raw_actions)
        return scores, values, log_prob, new_states

    def _predict(
        self,
        observation: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, RNNStates]:
        """Route predictions through ``forward`` to obtain sigmoid-clipped scores."""
        actions, _, _, new_states = self.forward(
            observation, lstm_states, episode_starts, deterministic=deterministic
        )
        return actions, new_states

    def predict(
        self,
        observation: np.ndarray | Mapping[str, np.ndarray],
        state: RNNStates | tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, RNNStates | tuple[np.ndarray, ...]]:
        """SB3 helper to obtain actions in numpy format with RNN state support."""

        def _to_tensor_tuple(seq: Sequence[Any]) -> tuple[torch.Tensor, ...]:
            tensors: list[torch.Tensor] = []
            for item in seq:
                if isinstance(item, torch.Tensor):
                    tensor = item.to(device=self.device, dtype=torch.float32)
                else:
                    tensor = torch.as_tensor(item, dtype=torch.float32, device=self.device)
                tensors.append(tensor)
            return tuple(tensors)

        training_mode = self.training
        self.set_training_mode(False)
        try:
            obs_tensor, vectorized_env = self.obs_to_tensor(observation)

            if isinstance(obs_tensor, dict):
                first_key = next(iter(obs_tensor.keys()))
                n_envs = obs_tensor[first_key].shape[0]
            else:
                n_envs = obs_tensor.shape[0]

            if state is None:
                lstm_states: RNNStates | tuple[torch.Tensor, ...] = self.recurrent_initial_state
            elif hasattr(state, "pi") and hasattr(state, "vf"):
                lstm_states = RNNStates(pi=_to_tensor_tuple(state.pi), vf=_to_tensor_tuple(state.vf))
            else:
                lstm_states = _to_tensor_tuple(state)

            lstm_states = self._coerce_lstm_states(lstm_states)
            lstm_states = RNNStates(
                pi=tuple(t.to(self.device) for t in lstm_states.pi),
                vf=tuple(t.to(self.device) for t in lstm_states.vf),
            )

            if episode_start is None:
                episode_start_arr = np.zeros(n_envs, dtype=bool)
            else:
                episode_start_arr = np.asarray(episode_start, dtype=bool).reshape(n_envs)

            episode_starts_tensor = torch.as_tensor(
                episode_start_arr.astype(np.float32), device=self.device
            )

            with torch.no_grad():
                actions_tensor, new_states = self._predict(
                    obs_tensor,
                    lstm_states=lstm_states,
                    episode_starts=episode_starts_tensor,
                    deterministic=deterministic,
                )

            if hasattr(new_states, "pi") and hasattr(new_states, "vf"):
                states_out: RNNStates | tuple[np.ndarray, ...] = RNNStates(
                    pi=tuple(state.detach().cpu().numpy() for state in new_states.pi),
                    vf=tuple(state.detach().cpu().numpy() for state in new_states.vf),
                )
            else:
                states_out = tuple(state.detach().cpu().numpy() for state in new_states)

            actions = actions_tensor
            if isinstance(actions, torch.Tensor):
                actions = actions.detach().cpu().numpy()

            if isinstance(self.action_space, spaces.Box):
                if self.squash_output:
                    actions = self.unscale_action(actions)
                else:
                    actions = np.clip(actions, self.action_space.low, self.action_space.high)

            if not vectorized_env:
                actions = actions.squeeze(axis=0)

            return actions, states_out
        finally:
            self.set_training_mode(training_mode)

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

    def value_quantiles(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, ...],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        if not self._use_quantile_value_head:
            return self.predict_values(obs, lstm_states, episode_starts)

        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self._get_value_logits(latent_vf)

    def value_head_metadata(self) -> dict[str, Any]:
        if self._use_quantile_value_head:
            return {"type": "quantile", "num_quantiles": int(self.num_quantiles)}
        return {"type": "categorical", "num_quantiles": None}

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True):
        if self._use_quantile_value_head:
            state_dict = upgrade_quantile_value_state_dict(
                state_dict,
                target_prefix="quantile_head.linear",
                num_quantiles=int(self.num_quantiles),
                fallback_prefixes=("value_net", "dist_head"),
            )
        filtered_state = dict(state_dict)
        removed_action_keys: list[str] = []
        need_relax = False
        if "unconstrained_log_std" not in state_dict:
            need_relax = True
        action_head = getattr(self, "action_net", None)
        if isinstance(action_head, nn.Linear):
            weight_key = "action_net.weight"
            bias_key = "action_net.bias"
            expected_weight_shape = tuple(action_head.weight.shape)
            expected_bias_shape = tuple(action_head.bias.shape)
            if weight_key in filtered_state and tuple(filtered_state[weight_key].shape) != expected_weight_shape:
                removed_action_keys.append(weight_key)
                filtered_state.pop(weight_key, None)
            if bias_key in filtered_state and tuple(filtered_state[bias_key].shape) != expected_bias_shape:
                removed_action_keys.append(bias_key)
                filtered_state.pop(bias_key, None)
        if removed_action_keys:
            need_relax = True
        load_strict = strict and not need_relax
        return super().load_state_dict(filtered_state, strict=load_strict)

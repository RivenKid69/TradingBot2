# Population-Based Training + Adversarial Training Integration

Этот модуль реализует интеграцию Population-Based Training (PBT) и State-Adversarial PPO (SA-PPO) для robust обучения торгового бота.

## Компоненты

### 1. State Perturbation (`state_perturbation.py`)

Генерирует adversarial perturbations на state observations.

**Функции:**

- FGSM (Fast Gradient Sign Method) attack
- PGD (Projected Gradient Descent) attack
- Support L-inf and L2 norms
- State clipping для ограничений

**Пример использования:**

```python
from adversarial import PerturbationConfig, StatePerturbation

config = PerturbationConfig(
    epsilon=0.075,
    attack_steps=3,
    attack_lr=0.03,
    norm_type="linf",
    attack_method="pgd"
)

perturbation = StatePerturbation(config)

# Generate adversarial perturbation
delta = perturbation.generate_perturbation(state, loss_fn)
adversarial_state = state + delta
```

### 2. State-Adversarial PPO (`sa_ppo.py`)

Расширяет PPO с adversarial training для robustness.

**Функции:**

- Mixed clean/adversarial training
- Robust KL regularization
- Adaptive epsilon scheduling
- Attack на policy и value losses

**Пример использования:**

```python
from adversarial import SAPPOConfig, StateAdversarialPPO

config = SAPPOConfig(
    enabled=True,
    adversarial_ratio=0.5,
    robust_kl_coef=0.1,
    warmup_updates=10,
    attack_policy=True,
    attack_value=True
)

sa_ppo = StateAdversarialPPO(config, ppo_model)
sa_ppo.on_training_start()

# During training
loss, info = sa_ppo.compute_adversarial_loss(
    states, actions, advantages, returns, old_log_probs
)
```

### 3. Population-Based Training (`pbt_scheduler.py`)

Управляет популяцией parallel training runs с periodic exploitation и exploration.

**Функции:**

- Population management
- Exploitation strategies (truncation, binary tournament)
- Exploration strategies (perturb, resample, both)
- Hyperparameter mutation
- Checkpoint management

**Пример использования:**

```python
from adversarial import PBTConfig, HyperparamConfig, PBTScheduler

# Define hyperparameters to optimize
hyperparams = [
    HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3),
    HyperparamConfig(name="adversarial_epsilon", min_value=0.01, max_value=0.15),
]

config = PBTConfig(
    population_size=10,
    perturbation_interval=5,
    hyperparams=hyperparams,
    metric_name="mean_reward",
    metric_mode="max"
)

scheduler = PBTScheduler(config)
population = scheduler.initialize_population()

# During training
for member in population:
    # Train for perturbation_interval steps
    train_member(member)

    # Update performance
    performance = evaluate_member(member)
    scheduler.update_performance(member, performance, step, model_state)

    # Exploit and explore
    if scheduler.should_exploit_and_explore(member):
        new_state, new_hyperparams = scheduler.exploit_and_explore(member)
        if new_state is not None:
            load_model_state(new_state)
        update_hyperparams(new_hyperparams)
```

## Конфигурация

См. `configs/config_pbt_adversarial.yaml` для полного примера конфигурации.

Ключевые параметры:

```yaml
# PBT Configuration
pbt:
  enabled: true
  population_size: 8
  perturbation_interval: 10
  metric_name: mean_reward
  metric_mode: max
  hyperparams:
    - name: learning_rate
      min_value: 1.0e-5
      max_value: 5.0e-4
    - name: adversarial_epsilon
      min_value: 0.01
      max_value: 0.15

# Adversarial Training Configuration
adversarial:
  enabled: true
  perturbation:
    epsilon: 0.075
    attack_steps: 3
    attack_method: pgd
  adversarial_ratio: 0.5
  robust_kl_coef: 0.1
  warmup_updates: 10
```

## Исследования и References

**Population-Based Training:**

- [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) (DeepMind 2017)
- [Ray Tune PBT Guide](https://docs.ray.io/en/latest/tune/examples/pbt_guide.html)

**Adversarial Training for RL:**

- [Robust Deep RL against Adversarial Perturbations](https://arxiv.org/abs/2003.08938) (NeurIPS 2020)
- [State-Adversarial PPO](https://github.com/huanzhang12/SA_PPO)

## Тестирование

Запустите тесты:

```bash
# Все тесты
pytest tests/test_state_perturbation.py tests/test_sa_ppo.py tests/test_pbt_scheduler.py -v

# С coverage
pytest tests/ -v --cov=adversarial --cov-report=html --cov-report=term

# Конкретный модуль
pytest tests/test_state_perturbation.py -v
```

## Архитектура

```
adversarial/
├── __init__.py                 # Публичный API
├── state_perturbation.py       # PGD/FGSM attacks
├── sa_ppo.py                   # State-Adversarial PPO
├── pbt_scheduler.py            # Population-Based Training
└── README.md                   # Этот файл

tests/
├── test_state_perturbation.py  # Тесты для perturbation
├── test_sa_ppo.py             # Тесты для SA-PPO
└── test_pbt_scheduler.py      # Тесты для PBT

configs/
└── config_pbt_adversarial.yaml # Конфигурация

```

## Best Practices

1. **Warmup**: Начинайте adversarial training после нескольких updates (warmup_updates)
2. **Adversarial Ratio**: Используйте 0.5 для баланса clean/adversarial samples
3. **Epsilon**: Начните с 0.075 и адаптируйте через scheduling
4. **Population Size**: 8-10 members для хорошего diversity
5. **Perturbation Interval**: 5-10 updates между PBT perturbations
6. **Attack Steps**: 3 PGD steps обычно достаточно

## Performance Tips

1. Используйте `torch.compile` для ускорения (если доступно)
2. Кэшируйте adversarial perturbations где возможно
3. Используйте mixed precision training
4. Профилируйте attack generation time
5. Мониторьте attack success rate и robustness metrics

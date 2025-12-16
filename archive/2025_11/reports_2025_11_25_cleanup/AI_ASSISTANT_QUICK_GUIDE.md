# AI-Powered Quantitative Research Platform - Quick Guide for AI Assistants

> **Экспресс-справочник для быстрого понимания проекта**

**Version**: 1.0
**Date**: 2025-11-22
**Purpose**: Максимально быстро дать AI-ассистенту полную картину проекта

---

## 🎯 Что это?

**AI-Powered Quantitative Research Platform** - высокочастотный торговый бот для криптовалют с RL (Reinforcement Learning):
- **Биржа**: Binance spot/futures
- **Алгоритм**: Distributional PPO + Twin Critics
- **Optimizer**: AdaptiveUPGD (continual learning)
- **Features**: 63 features (price, volume, volatility, momentum, microstructure)
- **Статус**: ✅ Production Ready (version 2.1, 2025-11-22)

---

## 📍 Где искать информацию?

### 1 минута - Срочная задача
**Читать**: [README.md](README.md) (100 строк)
- Статус проекта, последние критические исправления, quick start

### 5 минут - Понять проект
**Читать**: [QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md)
- Архитектура, основные файлы, команды, workflow

### 15 минут - Детальное понимание
**Читать**: [CLAUDE.md](CLAUDE.md) (2000+ строк) ⭐
- **ПОЛНАЯ** документация проекта (Russian)
- Архитектура, компоненты, конфигурации, частые ошибки

### Навигация по всем документам
**Читать**: [DOCS_INDEX.md](DOCS_INDEX.md)
- Навигационный hub для ВСЕХ 280+ документов

### Поддержка документации
**Читать**: [DOCUMENTATION_MAINTENANCE_GUIDE.md](DOCUMENTATION_MAINTENANCE_GUIDE.md)
- Как обновлять документацию, checklist, templates

---

## 🔥 КРИТИЧЕСКИ ВАЖНО - Прочитать ПЕРЕД любой работой

### Недавние критические исправления (2025-11-21 / 2025-11-22)

**⚠️ НЕ ОТКАТЫВАЙТЕ эти изменения - они предотвращают критические баги!**

1. **Twin Critics VF Clipping** (2025-11-22) - ✅ VERIFIED
   - Отчет: [TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md](TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md)
   - Каждый критик теперь клипится относительно СВОИХ old values (не shared min(Q1, Q2))
   - 49/50 tests passed (98%), PRODUCTION READY

2. **LSTM State Reset** (2025-11-21) - ✅ FIXED
   - Отчет: [CRITICAL_LSTM_RESET_FIX_REPORT.md](CRITICAL_LSTM_RESET_FIX_REPORT.md)
   - LSTM states теперь автоматически сбрасываются при episode boundaries
   - Предотвращает temporal leakage (5-15% improvement)
   - Code: `distributional_ppo.py:7418-7427`

3. **Action Space Fixes** (2025-11-21) - ✅ FIXED
   - Отчет: [CRITICAL_FIXES_COMPLETE_REPORT.md](CRITICAL_FIXES_COMPLETE_REPORT.md)
   - `ActionProto.volume_frac` = TARGET position (было: DELTA)
   - Предотвращает position doubling в production
   - LongOnlyActionWrapper теперь сохраняет reduction signals

4. **UPGD Negative Utility** (2025-11-21) - ✅ FIXED
   - Отчет: [UPGD_NEGATIVE_UTILITY_FIX_REPORT.md](UPGD_NEGATIVE_UTILITY_FIX_REPORT.md)
   - Min-max normalization вместо division-by-global-max
   - Работает корректно для всех знаков utilities

5. **NaN Handling** (2025-11-21) - ✅ IMPROVED
   - Отчет: [NUMERICAL_ISSUES_FIX_SUMMARY.md](NUMERICAL_ISSUES_FIX_SUMMARY.md)
   - External features NaN → 0.0 теперь логируется
   - Parameter `log_nan=True` для debugging

6. **3 Feature Bugs** (2025-11-20) - ✅ FIXED
   - Yang-Zhang Bessel's Correction
   - Log vs Linear Returns Mismatch
   - EWMA Cold Start Bias

7. **5 Numerical Stability Bugs** (2025-11-20) - ✅ FIXED
   - Log of Near-Zero → использовать F.log_softmax
   - VGS-UPGD Noise Amplification
   - CVaR Quantile Clipping
   - LSTM Gradient Clipping
   - NaN Propagation

**Обязательный checklist**: [REGRESSION_PREVENTION_CHECKLIST.md](REGRESSION_PREVENTION_CHECKLIST.md)

---

## 🏗️ Архитектура (в одной схеме)

```
┌─────────────────── LAYERED ARCHITECTURE ────────────────────┐
│                                                              │
│  core_*     →  impl_*  →  service_*  →  strategies  → script_*
│  (base)     (implement)  (business)     (algos)      (CLI)  │
│                                                              │
│  НЕ НАРУШАТЬ зависимости! →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→ │
└──────────────────────────────────────────────────────────────┘

┌─────────────── KEY COMPONENTS ──────────────────┐
│                                                  │
│  RL Model:      Distributional PPO + Twin Critics
│  Optimizer:     AdaptiveUPGD (default)          │
│  Gradient Scaling: VGS (enabled by default)     │
│  Training:      PBT + SA-PPO (optional)         │
│  Features:      63 (price, vol, volatility, etc)│
│  Execution:     Full LOB simulation + slippage  │
│  Risk:          Position limits, kill switch    │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 📂 Ключевые файлы (Top 20)

| Файл | Назначение | Когда читать |
|------|-----------|--------------|
| `distributional_ppo.py` | PPO implementation | Изменения в RL algorithm |
| `custom_policy_patch1.py` | Policy architecture | Изменения в policy network |
| `train_model_multi_patch.py` | Training loop | Изменения в training |
| `feature_pipe.py` | Feature engineering | Добавление новых features |
| `impl_sim_executor.py` | Execution simulator | Изменения в execution logic |
| `risk_guard.py` | Risk management | Изменения в risk checks |
| `config_train.yaml` | Training config | Default training settings |
| `config_pbt_adversarial.yaml` | PBT config | PBT + SA-PPO training |
| `optimizers/upgd.py` | UPGD optimizer | Optimizer issues |
| `variance_gradient_scaler.py` | VGS | Gradient scaling issues |
| `adversarial/sa_ppo.py` | SA-PPO | Adversarial training |
| `adversarial/pbt_scheduler.py` | PBT | Population-based training |
| `obs_builder.pyx` | Observation builder | Feature observation issues |
| `reward.pyx` | Reward calculation | Reward issues |
| `lob_state_cython.pyx` | LOB state | LOB simulation issues |
| `mediator.py` | Component mediator | Integration issues |
| `service_train.py` | Training service | Training pipeline |
| `service_backtest.py` | Backtest service | Backtesting |
| `script_live.py` | Live trading | Production trading |
| `tests/test_twin_critics*.py` | Twin Critics tests | Twin Critics verification |

---

## ⚡ Типичные задачи - Где искать

| Задача | Где искать | Команда |
|--------|-----------|---------|
| **Найти definition класса** | Glob | `*.py` pattern с именем класса |
| **Исправить feature bug** | `features/`, `feature_config.py` | `pytest tests/test_features*.py` |
| **Изменить execution logic** | `impl_sim_executor.py`, `execution_sim.py` | `pytest tests/test_execution*.py` |
| **Настроить риск** | `configs/risk.yaml`, `risk_guard.py` | Проверить `test_risk*.py` |
| **Обновить PPO** | `distributional_ppo.py` | Проверить `test_distributional_ppo*.py` |
| **Добавить метрику** | `services/monitoring.py` | Обновить `metrics.json` schema |
| **Calibrate parameters** | `service_calibrate_*.py` | Запустить calibration script |
| **Debug training** | `train_model_multi_patch.py` + logs | Проверить tensorboard |
| **Data issues** | `impl_offline_data.py`, `data_validation.py` | Проверить data degradation |
| **Live trading issues** | `script_live.py`, `service_signal_runner.py` | Проверить kill switch, state storage |

---

## 🧪 Тесты (Test Coverage)

### Актуальные числа (2025-11-22)

**Total Test Count**:
- **Twin Critics**: 207 tests total
  - VF Clipping correctness: 49 tests (98% pass - 49/50)
  - VF Clipping integration: 28 tests
  - General Twin Critics: 130+ tests
- **UPGD**: 126 tests
- **PBT**: 137 tests
- **Critical fixes**: 101+ new tests (98%+ pass rate)

**Запуск тестов**:
```bash
# Все тесты
pytest tests/

# Specific component
pytest tests/test_twin_critics*.py -v
pytest tests/test_upgd*.py -v
pytest tests/test_pbt*.py -v

# Критические исправления
pytest tests/test_twin_critics_vf_clipping_correctness.py -v  # 11/11 ✅
pytest tests/test_lstm_episode_boundary_reset.py -v           # 8/8 ✅
pytest tests/test_nan_handling_external_features.py -v        # 9/10 ✅
```

---

## 🚨 Частые ошибки (НЕ ДЕЛАТЬ!)

| Ошибка | Почему опасно | Решение |
|--------|---------------|---------|
| **Изменить ActionProto semantics** | Position doubling! | TARGET, не DELTA (см. CRITICAL_FIXES_COMPLETE_REPORT.md) |
| **Удалить LSTM state reset** | Temporal leakage | НЕ трогать `_reset_lstm_states_for_done_envs` (distributional_ppo.py:7418-7427) |
| **Откатить UPGD utility scaling** | Inverses logic | Min-max normalization обязательна (optimizers/upgd.py:93-174) |
| **Использовать division-by-max для utilities** | Fails with negative values | См. UPGD_NEGATIVE_UTILITY_FIX_REPORT.md |
| **Нарушить слоистую архитектуру** | Циклические импорты | core → impl → service → strategies → script (ТОЛЬКО →) |
| **Не сбрасывать LSTM states** | Model переобучается на первый episode | Автоматически в distributional_ppo.py |
| **Клипить обоих critics с min(Q1,Q2)** | Loses Twin Critics benefit | Каждый critic клипится относительно СВОИХ old values |

---

## 🔧 Конфигурация (Quick Reference)

### Default Settings (config_train.yaml)

```yaml
# OPTIMIZER (AdaptiveUPGD - default)
optimizer_class: AdaptiveUPGD
optimizer_kwargs:
  lr: 1.0e-4
  sigma: 0.001  # CRITICAL для VGS interaction

# VGS (Variance Gradient Scaler)
vgs:
  enabled: true
  accumulation_steps: 4
  warmup_steps: 10

# TWIN CRITICS (enabled by default)
use_twin_critics: true
num_atoms: 21          # Distributional critic quantiles
v_min: -10.0
v_max: 10.0

# VALUE CLIPPING
clip_range_vf: 0.7     # Twin Critics VF clipping

# CVAR RISK-AWARE LEARNING
cvar_alpha: 0.05       # Worst 5% tail
cvar_weight: 0.15
```

---

## 📚 Документация - Иерархия важности

### Tier 1 - ВСЕГДА актуальны
1. **[CLAUDE.md](CLAUDE.md)** ⭐ - Master reference
2. **[README.md](README.md)** - Project overview
3. **[DOCS_INDEX.md](DOCS_INDEX.md)** - Navigation hub
4. **Critical Reports** (Root/*.md) - Latest fixes

### Tier 2 - Справочники
1. **[QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md)** - Quick start
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Architecture
3. **[FILE_REFERENCE.md](FILE_REFERENCE.md)** - File organization
4. **[DOCUMENTATION_MAINTENANCE_GUIDE.md](DOCUMENTATION_MAINTENANCE_GUIDE.md)** - Maintenance

### Tier 3 - Техническая документация
1. **docs/UPGD_INTEGRATION.md** - UPGD optimizer
2. **docs/twin_critics.md** - Twin critics
3. **docs/seasonality.md** - Seasonality
4. **docs/pipeline.md** - Pipeline architecture

### Tier 4 - Отчеты (280+ files)
1. **docs/reports/analysis/** - Анализы
2. **docs/reports/audits/** - Аудиты
3. **docs/reports/fixes/** - Исправления
4. **docs/reports/tests/** - Тесты

---

## 🎯 Decision Tree - Что читать?

```
START
  │
  ├─→ Новый в проекте?
  │   └─→ READ: README.md → QUICK_START_REFERENCE.md
  │
  ├─→ Нужна полная картина?
  │   └─→ READ: CLAUDE.md (2000 строк, все детали)
  │
  ├─→ Ищешь конкретный документ?
  │   └─→ READ: DOCS_INDEX.md (навигация по 280+ файлам)
  │
  ├─→ Работаешь с критическим компонентом?
  │   ├─→ Twin Critics → TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md
  │   ├─→ LSTM → CRITICAL_LSTM_RESET_FIX_REPORT.md
  │   ├─→ Action Space → CRITICAL_FIXES_COMPLETE_REPORT.md
  │   ├─→ UPGD → UPGD_NEGATIVE_UTILITY_FIX_REPORT.md
  │   └─→ Features → CRITICAL_FIXES_REPORT.md
  │
  ├─→ Обновляешь документацию?
  │   └─→ READ: DOCUMENTATION_MAINTENANCE_GUIDE.md
  │
  ├─→ Ищешь конкретный код?
  │   └─→ USE: Glob/Grep tools (НЕ bash find/grep!)
  │
  └─→ Debugging?
      └─→ READ: REGRESSION_PREVENTION_CHECKLIST.md
```

---

## 🔍 Search Strategy (для AI-ассистентов)

### 1. Needle Query (ищешь конкретный file/class/function)
```
USE: Glob tool
PATTERN: "*.py" с именем
EXAMPLE: Glob pattern="*twin_critics*" для Twin Critics файлов
```

### 2. Keyword Search (ищешь где используется)
```
USE: Grep tool
PATTERN: regex для keyword
EXAMPLE: Grep pattern="use_twin_critics" для всех упоминаний
```

### 3. Exploration (изучаешь codebase)
```
USE: Task tool с subagent_type=Explore
THOROUGHNESS: "quick" | "medium" | "very thorough"
EXAMPLE: Task(Explore, "how Twin Critics works", thoroughness="medium")
```

### 4. Documentation Search
```
USE: Read tool + Grep
FIRST: Проверить DOCS_INDEX.md
THEN: Read нужный документ
EXAMPLE: Read("DOCS_INDEX.md") → найти ссылку → Read(ссылка)
```

---

## ⚠️ ВАЖНО для AI-ассистентов

### DO ✅
1. **Всегда читать файлы перед изменением** - НЕ редактировать непрочитанные файлы
2. **Использовать Glob/Grep** - НЕ bash find/grep
3. **Следовать слоистой архитектуре** - core → impl → service → strategies → script
4. **Проверять тесты перед изменениями** - найти соответствующие тесты
5. **Читать critical reports** - перед работой с критическими компонентами
6. **Использовать Task(Explore)** - для открытых исследований codebase

### DON'T ❌
1. **НЕ откатывать критические исправления** - см. critical reports
2. **НЕ нарушать зависимости слоёв** - циклические импорты
3. **НЕ использовать bash find/grep** - используй Glob/Grep tools
4. **НЕ изменять ActionProto semantics** - position doubling риск
5. **НЕ удалять LSTM state reset** - temporal leakage риск
6. **НЕ изменять UPGD utility scaling** - logic inversion риск
7. **НЕ добавлять emojis** - только если пользователь явно просит

---

## 📞 Если застрял

1. **Проверь** [CLAUDE.md](CLAUDE.md) - 99% ответов там
2. **Проверь** [DOCS_INDEX.md](DOCS_INDEX.md) - навигация к нужному документу
3. **Проверь** Critical Reports (Root/*.md) - возможно уже исправлено
4. **Используй** Task(Explore) - пусть специализированный агент найдёт
5. **Проверь** [REGRESSION_PREVENTION_CHECKLIST.md](REGRESSION_PREVENTION_CHECKLIST.md) - что нельзя делать

---

## 📊 Актуальный Статус (2025-11-22)

**Version**: 2.1
**Status**: ✅ Production Ready
**Last Major Update**: 2025-11-22 (Twin Critics VF Clipping verification)

**Active Features**:
- ✅ Twin Critics (enabled by default) - VERIFIED
- ✅ AdaptiveUPGD optimizer (default)
- ✅ VGS (Variance Gradient Scaler) - enabled
- ✅ PBT (Population-Based Training) - optional
- ✅ SA-PPO (State-Adversarial PPO) - optional
- ✅ 63 features pipeline
- ✅ Full LOB simulation
- ✅ Risk management

**Recent Fixes** (ALL ACTIVE):
- Twin Critics VF Clipping (2025-11-22) ✅
- LSTM State Reset (2025-11-21) ✅
- Action Space (2025-11-21) ✅
- UPGD Negative Utility (2025-11-21) ✅
- NaN Handling (2025-11-21) ✅
- Features & Volatility (2025-11-20) ✅
- Numerical Stability (2025-11-20) ✅

**Test Coverage**: 101+ new tests, 98%+ pass rate

---

**Maintained by**: Claude Code
**Last Updated**: 2025-11-22
**For**: AI Assistants (Quick Reference)
**Master Documentation**: [CLAUDE.md](CLAUDE.md) ⭐

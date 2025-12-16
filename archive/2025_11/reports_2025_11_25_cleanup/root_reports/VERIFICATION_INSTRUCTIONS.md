# Инструкции по Верификации - AI-Powered Quantitative Research Platform

**Последнее обновление**: 2025-11-21
**Версия**: 2.1

## 🔴 КРИТИЧЕСКИЕ ТЕСТЫ (2025-11-21) - ОБЯЗАТЕЛЬНО ЗАПУСТИТЬ!

### Перед любыми изменениями в LSTM, Action Space или Data Pipeline:

**Запустите эти тесты для предотвращения регрессий:**

```bash
# LSTM Episode Boundary Reset (8 tests)
pytest tests/test_lstm_episode_boundary_reset.py -v

# Action Space Fixes (21 tests)
pytest tests/test_critical_action_space_fixes.py -v

# NaN Handling in External Features (10 tests)
pytest tests/test_nan_handling_external_features.py -v

# Data Integrity Tests (from 2025-11-20)
pytest tests/test_stale_bar_temporal_causality.py -v
pytest tests/test_normalization_cross_symbol_contamination.py -v
pytest tests/test_quantile_loss_formula_default.py -v
```

**Ожидаемые результаты:**
- LSTM tests: 8/8 passed
- Action Space tests: 21/21 passed (2 skipped)
- NaN tests: 9/10 passed (1 skipped - Cython)
- Data tests: 18/18 passed

**⚠️ ЕСЛИ ХОТЯ БЫ ОДИН ТЕСТ НЕ ПРОХОДИТ:**
1. **НЕ КОММИТЬТЕ** изменения
2. Прочитайте [REGRESSION_PREVENTION_CHECKLIST.md](REGRESSION_PREVENTION_CHECKLIST.md)
3. Проверьте релевантный fix report:
   - LSTM: [CRITICAL_LSTM_RESET_FIX_REPORT.md](CRITICAL_LSTM_RESET_FIX_REPORT.md)
   - Action Space: [CRITICAL_FIXES_COMPLETE_REPORT.md](CRITICAL_FIXES_COMPLETE_REPORT.md)
   - NaN/Data: [NUMERICAL_ISSUES_FIX_SUMMARY.md](NUMERICAL_ISSUES_FIX_SUMMARY.md)

---

## ⚠️ Проверка Интеграции Технических Индикаторов

### ВАЖНО: Обязательная проверка перед обучением!

После интеграции технических индикаторов **необходимо проверить**, что они действительно передаются в модель.

## Быстрая Проверка

### Шаг 1: Запустите скрипт верификации

```bash
python verify_observation_integration.py
```

### Шаг 2: Проверьте результат

**✅ Успешно - если видите:**
```
✅ ALL CHECKS PASSED!

Technical indicators are correctly integrated into observations.
The model will receive all 56 features including:
  • Market data (price, volumes)
  • Moving averages (sma_5, sma_15)
  • Technical indicators (RSI, MACD, etc.)
  • CVD (cumulative volume delta)
  • GARCH volatility
  • Yang-Zhang volatility
  • Fear & Greed Index
  • Agent state
```

**✗ Проблема - если видите:**
```
⚠️  ISSUES FOUND:
  1. obs_builder not compiled/available - using LEGACY mode
  2. Too few non-zero features: 12/56
```

## Решение Проблем

### Проблема 1: `obs_builder` не импортируется

**Причина**: Cython модули не скомпилированы или скомпилированы для другой версии Python

**Решение**:

```bash
# Проверьте версию Python
python --version

# Перекомпилируйте модули
python setup.py build_ext --inplace

# Проверьте что импорт работает
python -c "from obs_builder import build_observation_vector; print('OK')"
```

### Проблема 2: Слишком мало ненулевых features (< 40)

**Причина**: Используется legacy fallback режим

**Решение**:
1. Убедитесь что `obs_builder` компилируется (см. Проблема 1)
2. Проверьте что `mediator.py` был обновлен (должен содержать `_extract_technical_indicators`)
3. Перезапустите окружение

### Проблема 3: Технические индикаторы отсутствуют в data

**Причина**: Feather файлы не содержат индикаторы

**Решение**:

```bash
# Запустите prepare_and_run.py для создания индикаторов
python prepare_and_run.py
```

## Детальная Проверка

### Проверка 1: Размер observation

```python
import numpy as np
from trading_patchnew import TradingEnv
import pandas as pd

df = pd.read_feather('data/processed/BTCUSDT.feather')
env = TradingEnv(df=df)
obs, info = env.reset()

print(f"Observation shape: {obs.shape}")
# Должно быть: (56,)

print(f"Non-zero count: {np.count_nonzero(obs)}")
# Должно быть: > 40
```

### Проверка 2: Наличие индикаторов в данных

```python
import pandas as pd

df = pd.read_feather('data/processed/BTCUSDT.feather')

indicators = ['sma_5', 'sma_15', 'rsi', 'cvd_24h', 'cvd_168h',
              'yang_zhang_24h', 'yang_zhang_168h', 'garch_12h', 'garch_24h']

for ind in indicators:
    if ind in df.columns:
        print(f"✓ {ind}: present")
    else:
        print(f"✗ {ind}: MISSING")
```

### Проверка 3: Mediator использует obs_builder

```python
from mediator import _HAVE_OBS_BUILDER

if _HAVE_OBS_BUILDER:
    print("✓ Mediator will use obs_builder (NEW MODE)")
else:
    print("✗ Mediator using legacy fallback (OLD MODE)")
```

## Критерии Успеха

Перед началом обучения убедитесь что:

- [ ] `verify_observation_integration.py` проходит без ошибок
- [ ] Observation shape = (56,)
- [ ] Non-zero count > 40
- [ ] `_HAVE_OBS_BUILDER = True`
- [ ] Все технические индикаторы присутствуют в feather файлах
- [ ] Тесты проходят: `python test_technical_indicators_in_obs.py`

## Что Проверить в Логах Обучения

При запуске `train_model_multi_patch.py` проверьте:

```python
# Должно быть в начале обучения:
INFO - Environment created with observation_space: Box(56,)
INFO - obs_builder available: True

# НЕ должно быть:
WARNING - obs_builder failed, falling back to legacy
WARNING - Using legacy observation builder
```

## Быстрый Тест

```bash
# Один скрипт для полной проверки
python << 'EOF'
import sys
import numpy as np
import pandas as pd

# 1. Проверка импортов
try:
    from obs_builder import build_observation_vector
    print("✓ obs_builder OK")
except:
    print("✗ obs_builder FAILED")
    sys.exit(1)

# 2. Проверка mediator
from mediator import _HAVE_OBS_BUILDER
if not _HAVE_OBS_BUILDER:
    print("✗ Mediator not using obs_builder")
    sys.exit(1)
print("✓ Mediator OK")

# 3. Проверка environment
from trading_patchnew import TradingEnv
df = pd.DataFrame({
    'timestamp': [1700000000],
    'open': [50000], 'high': [50100], 'low': [49900], 'close': [50000],
    'volume': [100], 'quote_asset_volume': [5000000],
    'sma_5': [50000], 'sma_15': [50000], 'rsi': [50],
    'cvd_24h': [0.5], 'garch_12h': [0.03], 'yang_zhang_24h': [0.025],
    'fear_greed_value': [50]
})

env = TradingEnv(df=df)
obs, _ = env.reset()

if obs.shape == (56,) and np.count_nonzero(obs) > 20:
    print(f"✓ Environment OK: {obs.shape}, {np.count_nonzero(obs)} non-zero")
    print("\n✅ ALL SYSTEMS GO! Ready for training.")
else:
    print(f"✗ Problem: shape={obs.shape}, non-zero={np.count_nonzero(obs)}")
    sys.exit(1)
EOF
```

## Поддержка

Если проблемы сохраняются:

1. Проверьте версию Python: `python --version`
2. Проверьте установленные пакеты: `pip list | grep -i cython`
3. Проверьте скомпилированные модули: `ls -la *.so`
4. Откройте issue с выводом `verify_observation_integration.py`

## 🧪 Полный Набор Тестов для Верификации

### Критические тесты (ОБЯЗАТЕЛЬНО перед коммитом)
```bash
# Все критические тесты одной командой
pytest tests/test_lstm_episode_boundary_reset.py \
       tests/test_critical_action_space_fixes.py \
       tests/test_nan_handling_external_features.py \
       tests/test_stale_bar_temporal_causality.py \
       tests/test_normalization_cross_symbol_contamination.py \
       tests/test_quantile_loss_formula_default.py -v

# Ожидается: 67/69 passed, 3 skipped
```

### Execution тесты
```bash
pytest tests/test_execution*.py -v
```

### PPO тесты
```bash
pytest tests/test_distributional_ppo*.py -v
```

### UPGD/VGS тесты (если используется)
```bash
pytest tests/test_upgd*.py -v
pytest tests/test_vgs*.py -v
```

### PBT тесты (если используется)
```bash
pytest tests/test_pbt*.py -v
```

### Все тесты
```bash
pytest tests/ -v
```

## 📋 Checklist перед обучением модели

- [ ] Все критические тесты проходят (67/69 passed minimum)
- [ ] LSTM state reset активен (`_reset_lstm_states_for_done_envs` вызывается)
- [ ] Action space semantics = TARGET (не DELTA!)
- [ ] Action space bounds = [-1, 1] везде
- [ ] External features NaN handling настроен (`log_nan=True` для debugging)
- [ ] Observation shape = (56,) или корректное для вашей конфигурации
- [ ] Non-zero features > 40 (для 56D observation)
- [ ] `_HAVE_OBS_BUILDER = True` (если используется Cython)
- [ ] Прочитан [REGRESSION_PREVENTION_CHECKLIST.md](REGRESSION_PREVENTION_CHECKLIST.md)

## 🔍 Диагностика проблем

### Проблема: LSTM value loss не снижается

**Возможная причина**: LSTM states не сбрасываются на episode boundaries

**Решение**:
```bash
# Проверьте что reset работает
pytest tests/test_lstm_episode_boundary_reset.py::test_lstm_states_reset_on_done -v

# Проверьте что метод существует
python -c "from distributional_ppo import DistributionalPPO; print(hasattr(DistributionalPPO, '_reset_lstm_states_for_done_envs'))"
# Должно быть: True
```

### Проблема: Position doubling в production

**Возможная причина**: DELTA semantics вместо TARGET

**Решение**:
```bash
# Проверьте semantics
pytest tests/test_critical_action_space_fixes.py::test_problem2_position_doubling -v

# Проверьте ActionProto contract
grep -n "volume_frac" risk_guard.py
# Должно использовать TARGET semantics
```

### Проблема: External features всегда 0.0

**Возможная причина**: NaN конвертируется в 0.0 молча

**Решение**:
```bash
# Включите NaN logging
pytest tests/test_nan_handling_external_features.py::test_log_nan_parameter -v

# Проверьте mediator
python -c "from mediator import _get_safe_float; print(_get_safe_float.__doc__)"
```

## 📚 Дополнительные Ресурсы

### Критические Исправления
- [CRITICAL_LSTM_RESET_FIX_REPORT.md](CRITICAL_LSTM_RESET_FIX_REPORT.md) - LSTM state reset fix
- [CRITICAL_FIXES_COMPLETE_REPORT.md](CRITICAL_FIXES_COMPLETE_REPORT.md) - Action space fixes
- [NUMERICAL_ISSUES_FIX_SUMMARY.md](NUMERICAL_ISSUES_FIX_SUMMARY.md) - LSTM + NaN comprehensive summary
- [CRITICAL_FIXES_REPORT.md](CRITICAL_FIXES_REPORT.md) - Data & critic bugs (2025-11-20)
- [REGRESSION_PREVENTION_CHECKLIST.md](REGRESSION_PREVENTION_CHECKLIST.md) - **ОБЯЗАТЕЛЬНО к прочтению!**

### Общая Документация
- [OBSERVATION_MAPPING.md](OBSERVATION_MAPPING.md) - Полное описание observation vector
- [test_technical_indicators_in_obs.py](test_technical_indicators_in_obs.py) - Unit тесты
- [mediator.py](mediator.py) - Реализация `_build_observation()`
- [DOCS_INDEX.md](DOCS_INDEX.md) - Навигация по всей документации

---

**Maintained by**: Development Team + Claude Code
**Last Updated**: 2025-11-21
**Version**: 2.1

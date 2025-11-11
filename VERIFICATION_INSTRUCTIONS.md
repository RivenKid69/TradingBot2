# Проверка Интеграции Технических Индикаторов

## ⚠️ ВАЖНО: Обязательная проверка перед обучением!

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

## Ссылки

- [OBSERVATION_MAPPING.md](OBSERVATION_MAPPING.md) - Полное описание observation vector
- [test_technical_indicators_in_obs.py](test_technical_indicators_in_obs.py) - Unit тесты
- [mediator.py](mediator.py) - Реализация `_build_observation()`

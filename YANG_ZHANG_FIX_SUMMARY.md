# Yang-Zhang Volatility: Исправления для автоматической работы

## Обзор

Признак Yang-Zhang волатильности был корректно реализован, но имел проблемы с автоматической интеграцией в пайплайн обучения и онлайн-режим.

## Найденные проблемы

### 1. Отсутствие передачи OHLC в оффлайн-режиме
**Файл:** `feature_pipe.py:761-774`

**Проблема:** Метод `transform_df` не передавал OHLC колонки в `apply_offline_features`, поэтому Yang-Zhang волатильность не рассчитывалась при обучении модели.

**Решение:** Добавлена автоматическая проверка наличия колонок `open`, `high`, `low` в датафрейме и их передача в `apply_offline_features`.

```python
def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
    # Проверяем наличие OHLC колонок
    open_col = "open" if "open" in df.columns else None
    high_col = "high" if "high" in df.columns else None
    low_col = "low" if "low" in df.columns else None

    return apply_offline_features(
        df, spec=self.spec,
        ts_col="ts_ms", symbol_col="symbol", price_col=self.price_col,
        open_col=open_col, high_col=high_col, low_col=low_col,
    )
```

### 2. Отсутствие OHLC данных в prices.parquet
**Файл:** `make_prices_from_klines.py`

**Проблема:** Скрипт создавал только колонки `ts_ms`, `symbol`, `price`, без OHLC данных необходимых для Yang-Zhang.

**Решение:** Добавлен флаг `--include-ohlc` для включения OHLC колонок в выходной файл.

```bash
# РАНЬШЕ (без OHLC):
python make_prices_from_klines.py \
  --in-klines data/klines.parquet \
  --symbol BTCUSDT \
  --out data/prices.parquet

# ТЕПЕРЬ (с OHLC для Yang-Zhang):
python make_prices_from_klines.py \
  --in-klines data/klines.parquet \
  --symbol BTCUSDT \
  --out data/prices.parquet \
  --include-ohlc  # Обязательно для Yang-Zhang!
```

## Что работает автоматически

### ✅ Онлайн-режим (Real-time)
- OHLC данные извлекаются из баров автоматически
- Yang-Zhang волатильность рассчитывается при наличии OHLC в барах
- Не требует дополнительной конфигурации

### ✅ Оффлайн-режим (Training)
- Автоматическая проверка наличия OHLC колонок
- Если OHLC присутствуют → Yang-Zhang рассчитывается
- Если OHLC отсутствуют → Yang-Zhang возвращает NaN

### ✅ Конфигурация
- Настраивается через `configs/legacy_realtime.yaml`:
```yaml
features:
  lookbacks_prices: [5, 15, 60]
  rsi_period: 14
  yang_zhang_windows: [1440, 10080, 43200]  # 24h, 7d, 30d
```

## Инструкции по использованию

### Для обучения модели:

1. **Подготовить данные с OHLC:**
```bash
python make_prices_from_klines.py \
  --in-klines data/klines.parquet \
  --symbol BTCUSDT \
  --out data/prices.parquet \
  --include-ohlc
```

2. **Обучить модель (стандартный процесс):**
```bash
python service_train.py --config configs/config_train.yaml
```

Yang-Zhang волатильность будет автоматически рассчитана, если OHLC колонки присутствуют в данных.

### Для онлайн-режима:

Просто запустите обычным образом:
```bash
python service_signal_runner.py --config configs/legacy_realtime.yaml
```

Yang-Zhang будет рассчитываться автоматически из OHLC баров.

## Тесты

### Базовые тесты функции:
```bash
python test_yang_zhang_simple.py
```

### Интеграционные тесты:
```bash
python test_yang_zhang_integration.py
```

## Проверка результата

### Проверка колонок в данных:
```python
import pandas as pd
df = pd.read_parquet("data/prices.parquet")
print(df.columns)
# Ожидаем: ['ts_ms', 'symbol', 'price', 'open', 'high', 'low', 'close']
```

### Проверка признаков после обучения:
```python
from feature_pipe import FeaturePipe
from transformers import FeatureSpec

spec = FeatureSpec(
    lookbacks_prices=[5, 15, 60],
    rsi_period=14,
    yang_zhang_windows=[1440, 10080, 43200]
)
pipe = FeaturePipe(spec=spec)
feats = pipe.transform_df(df)

# Проверяем наличие Yang-Zhang колонок
assert 'yang_zhang_24h' in feats.columns
assert 'yang_zhang_168h' in feats.columns
assert 'yang_zhang_720h' in feats.columns
```

## Файлы изменены

1. `feature_pipe.py` - добавлена передача OHLC в `transform_df`
2. `make_prices_from_klines.py` - добавлен флаг `--include-ohlc`
3. `docs/yang_zhang_volatility.md` - обновлена документация
4. `test_yang_zhang_integration.py` - добавлены интеграционные тесты

## Обратная совместимость

✅ Все изменения обратно совместимы:
- Если OHLC колонки отсутствуют, Yang-Zhang просто возвращает NaN
- Старый код без OHLC продолжит работать (без Yang-Zhang признаков)
- Добавленный флаг `--include-ohlc` опциональный (по умолчанию False)

## Итого

**Теперь Yang-Zhang волатильность работает автоматически "из коробки":**
1. ✅ В онлайн-режиме - извлекает OHLC из баров
2. ✅ При обучении - использует OHLC из данных (если есть)
3. ✅ Конфигурируется через YAML
4. ✅ Не ломает существующий код

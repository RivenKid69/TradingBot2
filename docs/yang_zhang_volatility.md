# Yang-Zhang Volatility Feature

## Описание

Yang-Zhang волатильность - это наиболее комплексный OHLC-оценщик волатильности, который учитывает:
- Ночную волатильность (overnight volatility)
- Внутридневную волатильность (intraday volatility)
- Волатильность Rogers-Satchell

## Формула

```
σ²_YZ = σ²_o + k·σ²_c + (1-k)·σ²_rs
```

где:
- **σ²_o** - ночная волатильность = (1/(n-1)) Σ(log(O_i/C_{i-1}) - μ_o)²
- **σ²_c** - волатильность open-close = (1/(n-1)) Σ(log(C_i/O_i) - μ_c)²
- **σ²_rs** - Rogers-Satchell = (1/n) Σ[log(H_i/C_i)·log(H_i/O_i) + log(L_i/C_i)·log(L_i/O_i)]
- **k** = 0.34 (эмпирически оптимальный вес)

## Преимущества

1. **Комплексность**: Использует всю информацию OHLC (Open, High, Low, Close)
2. **Точность**: Превосходит традиционные модели GARCH для краткосрочных прогнозов
3. **Научная обоснованность**: Подтверждено множеством исследований (см. SpringerOpen)

## Конфигурация

В файле конфигурации (например, `configs/legacy_realtime.yaml`):

```yaml
features:
  lookbacks_prices: [5, 15, 60]
  rsi_period: 14
  # Yang-Zhang volatility windows (in minutes)
  yang_zhang_windows: [1440, 10080, 43200]  # 24h, 7d, 30d
```

### Параметры

- **yang_zhang_windows**: список окон в минутах для расчета волатильности
  - `1440` = 24 часа (1 день)
  - `10080` = 168 часов (7 дней)
  - `43200` = 720 часов (30 дней)

## Использование

### Онлайн (real-time)

```python
from transformers import FeatureSpec, OnlineFeatureTransformer

spec = FeatureSpec(
    lookbacks_prices=[5, 15, 60],
    rsi_period=14,
    yang_zhang_windows=[1440, 10080, 43200]
)

transformer = OnlineFeatureTransformer(spec)

# Обработка бара с OHLC данными
feats = transformer.update(
    symbol="BTCUSDT",
    ts_ms=1234567890000,
    close=50000.0,
    open_price=49900.0,
    high=50100.0,
    low=49800.0,
)

# Доступ к признакам
print(feats['yang_zhang_24h'])   # Волатильность за 24 часа
print(feats['yang_zhang_168h'])  # Волатильность за 7 дней
print(feats['yang_zhang_720h'])  # Волатильность за 30 дней
```

### Оффлайн (batch processing)

```python
from transformers import FeatureSpec, apply_offline_features
import pandas as pd

# Загрузка данных с OHLC
df = pd.read_parquet("data/klines.parquet")

spec = FeatureSpec(
    lookbacks_prices=[5, 15, 60],
    rsi_period=14,
    yang_zhang_windows=[1440, 10080, 43200]
)

# Расчет признаков
features = apply_offline_features(
    df,
    spec=spec,
    ts_col="ts_ms",
    symbol_col="symbol",
    price_col="close",
    open_col="open",
    high_col="high",
    low_col="low",
)
```

### CLI (make_features.py)

```bash
python make_features.py \
  --in data/klines.parquet \
  --out data/features.parquet \
  --price-col close \
  --open-col open \
  --high-col high \
  --low-col low \
  --yang-zhang-windows "1440,10080,43200"
```

### Подготовка данных для обучения (make_prices_from_klines.py)

**ВАЖНО:** Для работы Yang-Zhang волатильности при обучении необходимо включить OHLC колонки:

```bash
python make_prices_from_klines.py \
  --in-klines data/klines.parquet \
  --symbol BTCUSDT \
  --out data/prices.parquet \
  --include-ohlc  # Обязательный флаг для Yang-Zhang!
```

Без флага `--include-ohlc` будут созданы только колонки `ts_ms`, `symbol`, `price`, и Yang-Zhang волатильность будет иметь значения NaN при обучении.

## Выходные признаки

Для каждого окна создается признак с именем `yang_zhang_{hours}h`:

- `yang_zhang_24h` - волатильность за 24 часа
- `yang_zhang_168h` - волатильность за 7 дней
- `yang_zhang_720h` - волатильность за 30 дней

Значения:
- **Число > 0**: рассчитанная волатильность (стандартное отклонение логарифмических доходностей)
- **NaN**: недостаточно данных для расчета

## Требования к данным

1. **Минимум данных**: Для окна N минут требуется как минимум N баров
2. **OHLC данные**: Необходимы все четыре значения (Open, High, Low, Close)
3. **Последовательность**: Данные должны быть упорядочены по времени
4. **Непрерывность**: Лучшие результаты при минимальных пропусках данных

## Интерпретация

- **Низкая волатильность** (< 0.01): Рынок стабилен, низкая неопределенность
- **Средняя волатильность** (0.01 - 0.05): Нормальные рыночные условия
- **Высокая волатильность** (> 0.05): Повышенный риск, большие ценовые движения

## Примеры значений

Для Bitcoin (BTCUSDT):
- **Спокойный рынок**: ~0.015 - 0.025
- **Нормальный рынок**: ~0.025 - 0.040
- **Волатильный рынок**: ~0.040 - 0.080
- **Экстремальная волатильность**: > 0.080

## Связанные файлы

- `transformers.py` - основная реализация
- `feature_pipe.py` - интеграция с пайплайном
- `make_features.py` - CLI для batch обработки
- `test_yang_zhang_simple.py` - unit тесты

## Ссылки

- Yang, D. and Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices"
- SpringerOpen: "Realized-GARCH models for cryptocurrency volatility forecasting"
- Rogers, L.C.G. and Satchell, S.E. (1991). "Estimating Variance From High, Low and Closing Prices"

## Часто задаваемые вопросы

**Q: Почему волатильность возвращает NaN?**
A: Недостаточно исторических данных. Для окна N минут нужно минимум N баров с полными OHLC данными.

**Q: Можно ли использовать только цены close?**
A: Нет, для Yang-Zhang необходимы все OHLC данные. Без них признак не будет рассчитываться.

**Q: Какие окна лучше использовать?**
A: Зависит от торговой стратегии:
- Для внутридневной торговли: 60-1440 минут (1ч - 1д)
- Для свинг-трейдинга: 1440-10080 минут (1д - 7д)
- Для позиционной торговли: 10080-43200 минут (7д - 30д)

**Q: Как Yang-Zhang соотносится с другими мерами волатильности?**
A: Yang-Zhang более точен чем:
- Простое стандартное отклонение (использует только close)
- Parkinson (использует только high/low)
- Garman-Klass (не учитывает overnight gaps)

**Q: Влияет ли это на производительность?**
A: Незначительно. Расчет Yang-Zhang имеет сложность O(N) для окна размером N. Для типичных окон (1440-43200 минут) overhead минимален.

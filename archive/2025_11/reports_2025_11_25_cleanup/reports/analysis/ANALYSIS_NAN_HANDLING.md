# Анализ и решение проблемы NaN в полосах Боллинджера

## Проблема

В симуляторе массивы `v_bb_low`/`v_bb_up` заполняются NaN до тех пор, пока индикатор Боллинджера не накопит достаточную историю (20 баров). Эти NaN значения попадают в observation vector, нарушая контракт Gym environment.

## Воспроизведение

Проблема воспроизводится в `test_nan_reproduction.py`. Результаты показывают:
- 6 из 10 признаков содержат NaN на ранних барах
- NaN попадают из: RSI (первые 14 баров), MACD (первые 26 баров), Momentum (первые 10 баров), Bollinger Bands (первые 20 баров)

## Источник проблемы

### 1. MarketSimulator.cpp (строки 280-287)
```cpp
if (w_close20.size() == 20) {
    double mean = sum20 / 20.0;
    double var  = std::max(0.0, sum20_sq / 20.0 - mean * mean);
    double sd   = std::sqrt(var);
    v_ma20[i]   = mean;
    v_bb_low[i] = mean - 2.0 * sd;
    v_bb_up[i]  = mean + 2.0 * sd;
}
```
Первые 19 баров остаются NaN, так как векторы инициализируются NaN (строка 44-47).

### 2. MarketSimulator::get_or_nan (строки 367-369)
```cpp
double MarketSimulator::get_or_nan(const std::vector<double>& v, std::size_t i) {
    return (i < v.size()) ? v[i] : NAN;
}
```
Просто передает NaN дальше без преобразования.

### 3. obs_builder.pyx::_clipf (строки 7-12)
```cython
cdef inline float _clipf(double value, double lower, double upper) nogil:
    if value < lower:
        value = lower
    elif value > upper:
        value = upper
    return <float>value
```
НЕ обрабатывает NaN! Сравнения с NaN всегда False, поэтому возвращает NaN без изменений.

### 4. Места в obs_builder.pyx где NaN попадают в observation

#### a) Прямое присваивание (строки 99-112):
```cython
out_features[feature_idx] = rsi14         # NaN первые 14 баров
out_features[feature_idx] = macd          # NaN первые ~26 баров
out_features[feature_idx] = macd_signal   # NaN первые ~35 баров
out_features[feature_idx] = momentum      # NaN первые 10 баров
out_features[feature_idx] = atr           # NaN первые 14 баров
out_features[feature_idx] = cci           # NaN первые 20 баров
out_features[feature_idx] = obv           # Всегда валиден
```

#### b) bb_squeeze (строки 168-170):
```cython
bb_squeeze = tanh((bb_upper - bb_lower) / (price_d + 1e-8))
out_features[feature_idx] = <float>bb_squeeze  # NaN если bb_upper или bb_lower - NaN!
```

#### c) price_momentum (строки 160-162):
```cython
price_momentum = tanh(momentum / (price_d * 0.01 + 1e-8))
out_features[feature_idx] = <float>price_momentum  # NaN если momentum - NaN!
```

#### d) trend_strength (строки 175-177):
```cython
trend_strength = tanh((macd - macd_signal) / (price_d * 0.01 + 1e-8))
out_features[feature_idx] = <float>trend_strength  # NaN если macd или macd_signal - NaN!
```

### 5. Частичное решение (строки 180-195)
Существующий код ПРАВИЛЬНО обрабатывает bb_lower/bb_upper в двух местах:
```cython
bb_valid = not isnan(bb_lower)
if (not bb_valid) or bb_width <= min_bb_width:
    feature_val = 0.5  # Безопасное значение по умолчанию
else:
    feature_val = _clipf((price_d - bb_lower) / (bb_width + 1e-9), -1.0, 2.0)
```

НО это решение применено только к двум признакам (позиция в полосах и ширина полос), а не ко всем.

## Лучшие практики обработки NaN в RL окружениях

### Стратегии из литературы и практики:

1. **Zero-filling**: Простая замена NaN на 0.0
   - Плюсы: быстро, предсказуемо
   - Минусы: может ввести агента в заблуждение (0 != "нет данных")

2. **Осмысленные дефолты**: Использовать значения, которые имеют смысл
   - RSI: 50.0 (нейтральная зона)
   - MACD: 0.0 (нет сигнала)
   - BB position: 0.5 (середина)
   - Плюсы: семантически корректно
   - Минусы: требует знания каждого индикатора

3. **Флаги валидности**: Добавить бинарные флаги "данные доступны"
   - Плюсы: агент явно знает, когда данные недоступны
   - Минусы: увеличивает размерность observation space

4. **Forward-fill**: Использовать последнее известное значение
   - Плюсы: естественно для временных рядов
   - Минусы: на старте истории нет

5. **Комбинированный подход** (РЕКОМЕНДУЕТСЯ):
   - Критичные индикаторы (MA, BB) получают флаги валидности
   - Остальные индикаторы заменяются осмысленными дефолтами
   - ГАРАНТИЯ: НИ ОДНОГО NaN в observation

## Решение

### Принципы:
1. **НИ ОДНОГО NaN** в observation vector - это ЖЕСТКОЕ требование Gym
2. **Семантическая корректность**: дефолты должны соответствовать "нет сигнала"
3. **Минимальные изменения**: не менять размерность observation space
4. **Флаги валидности**: для MA5/MA20 уже есть (строки 87-97)

### Реализация:

#### 1. Безопасная версия _clipf (добавить проверку NaN)
```cython
cdef inline float _clipf(double value, double lower, double upper) nogil:
    # CRITICAL: Handle NaN by returning a safe default (0.0)
    # NaN comparisons are always False, so we must check explicitly
    if isnan(value):
        return 0.0
    if value < lower:
        value = lower
    elif value > upper:
        value = upper
    return <float>value
```

#### 2. Обработка каждого индикатора с NaN:

```cython
# RSI: нейтральная зона 50 означает "нет тренда"
out_features[feature_idx] = rsi14 if not isnan(rsi14) else 50.0

# MACD/signal: 0 означает "нет расхождения"
out_features[feature_idx] = macd if not isnan(macd) else 0.0
out_features[feature_idx] = macd_signal if not isnan(macd_signal) else 0.0

# Momentum: 0 означает "нет движения"
out_features[feature_idx] = momentum if not isnan(momentum) else 0.0

# ATR: небольшая волатильность по умолчанию (price * 0.01)
out_features[feature_idx] = atr if not isnan(atr) else <float>(price_d * 0.01)

# CCI: 0 означает "на среднем уровне"
out_features[feature_idx] = cci if not isnan(cci) else 0.0

# OBV: всегда валиден, но на всякий случай
out_features[feature_idx] = obv if not isnan(obv) else 0.0
```

#### 3. Производные признаки с проверкой валидности BB:

```cython
# bb_squeeze: нужна валидность BB
bb_valid = not isnan(bb_lower)
if bb_valid:
    bb_squeeze = tanh((bb_upper - bb_lower) / (price_d + 1e-8))
else:
    bb_squeeze = 0.0  # нет сжатия по умолчанию
out_features[feature_idx] = <float>bb_squeeze

# price_momentum: нужна валидность momentum
if not isnan(momentum):
    price_momentum = tanh(momentum / (price_d * 0.01 + 1e-8))
else:
    price_momentum = 0.0
out_features[feature_idx] = <float>price_momentum

# trend_strength: нужна валидность MACD
if not isnan(macd) and not isnan(macd_signal):
    trend_strength = tanh((macd - macd_signal) / (price_d * 0.01 + 1e-8))
else:
    trend_strength = 0.0
out_features[feature_idx] = <float>trend_strength
```

## Преимущества решения

1. ✅ **Полное устранение NaN**: гарантировано нет NaN в observation
2. ✅ **Семантическая корректность**: дефолты имеют смысл (нейтральные/нулевые сигналы)
3. ✅ **Обратная совместимость**: не меняется размерность observation space
4. ✅ **Минимальные изменения**: локальные правки в одном файле
5. ✅ **Устойчивость**: даже если появятся новые источники NaN, _clipf их обработает

## Тестирование

1. `test_nan_reproduction.py`: подтверждает проблему
2. `test_obs_builder_nan_handling.py`: проверяет решение (будет создан)
3. Интеграционные тесты: запуск симуляции с первых баров

## Список изменений

- `obs_builder.pyx`:
  - Добавлена проверка NaN в `_clipf`
  - Добавлены проверки для всех индикаторов (rsi, macd, momentum, atr, cci, obv)
  - Добавлены проверки для производных признаков (bb_squeeze, price_momentum, trend_strength)

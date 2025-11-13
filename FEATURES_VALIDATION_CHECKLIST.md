# FEATURES VALIDATION CHECKLIST

**Версия:** 1.0
**Дата:** 2025-11-13
**Назначение:** Чеклист для проверки новых признаков на NaN/Inf безопасность

---

## КОГДА ИСПОЛЬЗОВАТЬ ЭТОТ ЧЕКЛИСТ

- ✅ При добавлении нового признака в observation vector
- ✅ При изменении существующего признака
- ✅ При изменении источника данных для признака
- ✅ При рефакторинге obs_builder.pyx
- ✅ При code review изменений в feature pipeline

---

## CHECKLIST ДЛЯ НОВОГО ПРИЗНАКА

### 1️⃣ ИСТОЧНИК ДАННЫХ

**Вопросы для проверки:**

- [ ] Откуда берется значение признака?
  - [ ] Технический индикатор (MarketSimulator)
  - [ ] Market data (mediator.py)
  - [ ] State переменная (env state)
  - [ ] External feature (norm_cols)
  - [ ] Производный расчет (в obs_builder.pyx)

- [ ] Может ли источник вернуть NaN?
  - [ ] Да → Требуется обработка NaN
  - [ ] Нет → Задокументировать почему

- [ ] Может ли источник вернуть Inf?
  - [ ] Да → Требуется обработка Inf
  - [ ] Нет → Задокументировать почему

- [ ] Может ли источник вернуть None/null?
  - [ ] Да → Требуется проверка на None перед float()
  - [ ] Нет → OK

### 2️⃣ ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ

**Если источник external (не в obs_builder.pyx):**

- [ ] Используется `_get_safe_float()` или `_coerce_finite()`?
  ```python
  # ✅ ПРАВИЛЬНО
  value = self._get_safe_float(row, "column_name", default=0.0)
  value = self._coerce_finite(getattr(state, "attr_name", 0.0), default=0.0)

  # ❌ НЕПРАВИЛЬНО
  value = float(row["column_name"])  # может быть NaN/Inf/None
  ```

- [ ] Указан семантически корректный default?
  - Примеры:
    - RSI → 50.0 (нейтральная зона)
    - MACD → 0.0 (нет дивергенции)
    - Volume → 0.0 (нет объема)
    - Price → НЕ 0.0! (использовать previous price или reasonable fallback)

### 3️⃣ МАТЕМАТИЧЕСКИЕ ОПЕРАЦИИ

**Если признак вычисляется в obs_builder.pyx:**

#### Деление

- [ ] Все деления защищены epsilon добавкой?
  ```cython
  # ✅ ПРАВИЛЬНО
  result = numerator / (denominator + 1e-8)

  # ❌ НЕПРАВИЛЬНО
  result = numerator / denominator  # деление на 0
  ```

- [ ] Если знаменатель может быть 0, есть ли специальная логика?
  ```cython
  # ✅ ПРАВИЛЬНО
  if total_worth <= 1e-8:
      feature_val = 1.0  # специальный случай
  else:
      feature_val = cash / total_worth
  ```

#### Логарифм / Квадратный корень

- [ ] Используется `log1p()` вместо `log()`?
  ```cython
  # ✅ ПРАВИЛЬНО - log1p безопасен для x > -1
  result = log1p(value)

  # ⚠️ ОПАСНО - log требует value > 0
  result = log(value)
  ```

- [ ] Аргумент sqrt всегда >= 0?
  ```cython
  # ✅ ПРАВИЛЬНО
  variance = max(0.0, sum_sq / n - mean * mean)
  std_dev = sqrt(variance)
  ```

#### Tanh / Exp

- [ ] Аргумент tanh проверен на NaN?
  ```cython
  # ✅ ПРАВИЛЬНО
  if not isnan(input_value):
      result = tanh(input_value / scale)
  else:
      result = 0.0

  # ⚠️ ОПАСНО - tanh(NaN) = NaN
  result = tanh(input_value / scale)
  ```

- [ ] Если используются производные переменные в tanh, они обработаны?
  ```cython
  # ❌ НЕПРАВИЛЬНО - vol_proxy VULN-01
  out_features[idx] = atr if not isnan(atr) else default_atr  # обработали
  # ...
  vol_proxy = tanh(log1p(atr / price))  # используем ОРИГИНАЛЬНЫЙ atr!

  # ✅ ПРАВИЛЬНО
  atr_safe = atr if not isnan(atr) else default_atr
  out_features[idx] = atr_safe
  # ...
  vol_proxy = tanh(log1p(atr_safe / price))  # используем обработанный
  ```

### 4️⃣ ОБРАБОТКА NaN В obs_builder.pyx

- [ ] Добавлена явная проверка `isnan()` перед использованием?
  ```cython
  # ✅ ПРАВИЛЬНО
  out_features[feature_idx] = indicator if not isnan(indicator) else default_value
  ```

- [ ] Используется `_clipf()` для финальной защиты?
  ```cython
  # ✅ ПРАВИЛЬНО - _clipf обрабатывает NaN → 0.0
  feature_val = _clipf(computed_value, min_val, max_val)
  out_features[feature_idx] = feature_val
  ```

- [ ] Дефолтное значение семантически корректно?
  - См. таблицу в секции 5

### 5️⃣ ДЕФОЛТНЫЕ ЗНАЧЕНИЯ - СЕМАНТИЧЕСКАЯ КОРРЕКТНОСТЬ

| Тип признака | Дефолт | Обоснование |
|--------------|--------|-------------|
| Trend indicator (RSI, MACD) | 0.0 или нейтральное | Нет тренда |
| RSI | 50.0 | Нейтральная зона (не перекуплено/перепродано) |
| Momentum | 0.0 | Нет движения |
| Volatility (ATR, BB width) | 0.0 или small% от price | Низкая волатильность |
| Volume | 0.0 | Нет объема |
| Position-related | 0.0 или 1.0 (cash) | Зависит от контекста |
| Price | ⚠️ НЕ 0.0! | Использовать prev price или fallback |
| Binary flags | 0.0 или 1.0 | Зависит от логики |
| Normalized [-1, 1] | 0.0 | Нейтральное значение |

**ВАЖНО**: Дефолт должен быть "безопасным" для модели, т.е. не вызывать ложных сигналов.

### 6️⃣ МИНИМАЛЬНЫЙ ПЕРИОД ГОТОВНОСТИ

- [ ] Определен минимальный период для расчета признака?
- [ ] Документирован в комментарии?
  ```cython
  # MACD requires 26 bars for EMA26 + 9 for signal = 35 bars minimum
  out_features[feature_idx] = macd if not isnan(macd) else 0.0
  ```

**Таблица минимальных периодов (для 4h timeframe):**

| Индикатор | Бары | Часы | Дни | Готов на баре |
|-----------|------|------|-----|---------------|
| MA5 | 5 | 20h | 0.8 | >= 4 |
| MA20 | 20 | 80h | 3.3 | >= 19 |
| Bollinger(20) | 20 | 80h | 3.3 | >= 19 |
| ATR(14) | 14 | 56h | 2.3 | >= 13 |
| RSI(14) | 15 | 60h | 2.5 | >= 14 |
| MACD(12,26) | 26 | 104h | 4.3 | >= 25 |
| MACD Signal(9) | 35 | 140h | 5.8 | >= 34 |
| Momentum(10) | 10 | 40h | 1.7 | >= 9 |
| CCI(20) | 20 | 80h | 3.3 | >= 19 |
| OBV | 1 | 4h | 0.2 | >= 0 |

### 7️⃣ ДИАПАЗОН ВАЛИДНЫХ ЗНАЧЕНИЙ

- [ ] Определен expected range для признака?
- [ ] Добавлен clipping если необходимо?
  ```cython
  # Признак должен быть в [0, 1]
  feature_val = _clipf(computed_value, 0.0, 1.0)
  ```

- [ ] Задокументирован expected range в комментарии?
  ```cython
  # Feature range: [-1, 1] via tanh normalization
  # Feature range: [0, 100] for RSI
  # Feature range: unbounded, typical range [-200, 200] for CCI
  ```

### 8️⃣ EDGE CASES

**Обязательные проверки:**

- [ ] Что происходит на первом баре (i=0)?
  - [ ] Все зависимости инициализированы?
  - [ ] Предыдущие значения (prev_price, etc.) доступны?

- [ ] Что происходит при price=0 или очень маленьком?
  - [ ] Деления на price защищены?
  - [ ] Дефолты не зависят от невалидного price?

- [ ] Что происходит при пустых данных?
  - [ ] norm_cols_values.shape[0] == 0?
  - [ ] max_num_tokens == 0?

- [ ] Что происходит при экстремальных значениях?
  - [ ] cash=0, units=0?
  - [ ] bb_width очень маленькая или отрицательная?
  - [ ] Индикаторы выходят за типичные границы?

### 9️⃣ ТЕСТИРОВАНИЕ

- [ ] Создан unit test для признака с NaN входом?
  ```python
  def test_new_feature_nan_handling():
      # Подать NaN на вход
      obs = build_observation_with_nan_input(...)
      # Проверить что output валиден
      assert np.all(np.isfinite(obs))
  ```

- [ ] Создан test для edge cases?
  ```python
  def test_new_feature_edge_cases():
      # price=0
      obs1 = build_observation(price=0.0, ...)
      assert np.all(np.isfinite(obs1))

      # first bar
      obs2 = build_observation(bar_idx=0, ...)
      assert np.all(np.isfinite(obs2))
  ```

- [ ] Проверен на первых 30 барах симуляции?
  ```python
  def test_new_feature_early_bars():
      for i in range(30):
          obs = build_observation(bar_idx=i, ...)
          assert np.all(np.isfinite(obs)), f"NaN/Inf on bar {i}"
  ```

### 🔟 ДОКУМЕНТАЦИЯ

- [ ] Добавлен комментарий в obs_builder.pyx?
  ```cython
  # Feature: price_momentum
  # Source: momentum indicator / MarketSimulator
  # Min period: 10 bars (40h)
  # Default: 0.0 (no momentum)
  # Range: [-1, 1] via tanh
  # NaN handling: returns 0.0 if momentum not ready
  ```

- [ ] Обновлена таблица в OBSERVATION_VECTOR_AUDIT_REPORT.md?

- [ ] Обновлен список признаков в документации проекта?

---

## ПРИМЕРЫ ПРАВИЛЬНОЙ РЕАЛИЗАЦИИ

### Пример 1: Простой технический индикатор

```cython
# Feature: RSI(14)
# Source: MarketSimulator.get_rsi()
# Min period: 15 bars
# Default: 50.0 (neutral - neither overbought nor oversold)
# Range: [0, 100]
# NaN handling: RSI may be NaN for first 14 bars, use neutral 50.0

cdef float rsi_value = rsi14 if not isnan(rsi14) else 50.0
out_features[feature_idx] = rsi_value
feature_idx += 1
```

### Пример 2: Производный признак с делением

```cython
# Feature: cash_fraction
# Source: calculated from cash and total_worth
# Min period: 1 bar (always available)
# Default: 1.0 (100% cash) if portfolio is empty
# Range: [0, 1]
# Edge case: total_worth=0 handled specially

cdef double position_value = units * price_d
cdef double total_worth = cash + position_value
cdef float feature_val

if not isfinite(total_worth) or total_worth <= 1e-8:
    # Portfolio empty or invalid - 100% cash
    feature_val = 1.0
else:
    feature_val = _clipf(cash / total_worth, 0.0, 1.0)

out_features[feature_idx] = feature_val
feature_idx += 1
```

### Пример 3: Нормализованный внешний признак

```cython
# Feature: normalized external columns (21 features)
# Source: mediator._extract_norm_cols() via _get_safe_float
# Min period: varies by column (see FEATURES_VALIDATION_CHECKLIST.md)
# Default: 0.0 (already handled by _get_safe_float)
# Range: [-3, 3] via tanh + clip
# NaN handling: _clipf returns 0.0 for NaN

for i in range(norm_cols_values.shape[0]):
    # Apply tanh normalization, then clip to safe range
    # _clipf handles NaN → 0.0
    feature_val = _clipf(tanh(norm_cols_values[i]), -3.0, 3.0)
    out_features[feature_idx] = feature_val
    feature_idx += 1
```

### Пример 4: Производный признак с tanh

```cython
# Feature: price_momentum
# Source: derived from momentum indicator
# Min period: 10 bars (momentum requirement)
# Default: 0.0 (no momentum signal)
# Range: [-1, 1] via tanh
# NaN handling: check momentum before computing

cdef double price_momentum
if not isnan(momentum):
    # Normalize by 1% of price (typical intraday move)
    price_momentum = tanh(momentum / (price_d * 0.01 + 1e-8))
else:
    price_momentum = 0.0  # momentum not ready yet

out_features[feature_idx] = <float>price_momentum
feature_idx += 1
```

---

## ПРИМЕРЫ НЕПРАВИЛЬНОЙ РЕАЛИЗАЦИИ (ANTI-PATTERNS)

### ❌ Anti-pattern 1: Деление без защиты

```cython
# ❌ НЕПРАВИЛЬНО - деление на 0
cdef float ratio = numerator / denominator

# ✅ ПРАВИЛЬНО
cdef float ratio = numerator / (denominator + 1e-8)
```

### ❌ Anti-pattern 2: NaN не обработан

```cython
# ❌ НЕПРАВИЛЬНО - может быть NaN
out_features[feature_idx] = rsi14

# ✅ ПРАВИЛЬНО
out_features[feature_idx] = rsi14 if not isnan(rsi14) else 50.0
```

### ❌ Anti-pattern 3: tanh от необработанного значения

```cython
# ❌ НЕПРАВИЛЬНО - tanh(NaN) = NaN
result = tanh(indicator_value / scale)

# ✅ ПРАВИЛЬНО
if not isnan(indicator_value):
    result = tanh(indicator_value / scale)
else:
    result = 0.0
```

### ❌ Anti-pattern 4: Использование оригинала после обработки

```cython
# ❌ НЕПРАВИЛЬНО - vol_proxy VULN-01
out_features[idx] = atr if not isnan(atr) else (price * 0.01)
# ...
vol_proxy = tanh(log1p(atr / price))  # используем ОРИГИНАЛЬНЫЙ atr!

# ✅ ПРАВИЛЬНО
atr_safe = atr if not isnan(atr) else (price * 0.01)
out_features[idx] = atr_safe
# ...
vol_proxy = tanh(log1p(atr_safe / price))  # используем обработанный
```

### ❌ Anti-pattern 5: Дефолт зависит от невалидного значения

```cython
# ❌ НЕПРАВИЛЬНО - если price=0, то atr_default=0
atr_default = price * 0.01
out_features[idx] = atr if not isnan(atr) else atr_default

# ✅ ПРАВИЛЬНО - фиксированный дефолт или валидированный price
cdef float price_safe = price if price > 0.0 else 100.0
atr_default = price_safe * 0.01
out_features[idx] = atr if not isnan(atr) else atr_default
```

---

## БЫСТРЫЙ ЧЕКЛИСТ (TL;DR)

При добавлении нового признака, проверь:

1. ✅ Источник защищен от NaN/Inf (`_get_safe_float`, `_coerce_finite`)
2. ✅ Деления имеют `+ 1e-8`
3. ✅ Используется `log1p` вместо `log`
4. ✅ `tanh` применяется к валидным значениям (не NaN)
5. ✅ Дефолт семантически корректен
6. ✅ `_clipf` применен для финальной защиты
7. ✅ Edge cases обработаны (price=0, first bar, empty data)
8. ✅ Созданы unit tests
9. ✅ Добавлены комментарии

---

## КОНТАКТЫ

Вопросы и предложения по улучшению чеклиста:
- Создать issue в репозитории
- Обновить этот документ через PR

**Версия документа:** 1.0
**Последнее обновление:** 2025-11-13

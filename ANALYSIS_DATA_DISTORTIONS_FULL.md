# ДЕТАЛЬНЫЙ АНАЛИЗ ИСКАЖЕНИЙ ДАННЫХ ПО ВСЕМ 56 ПРИЗНАКАМ

**Проект:** TradingBot2
**Таймфрейм:** 4h (240 минут)
**Дата анализа:** 2025-11-15
**Методология:** Опирается на лучшие практики и исследования в области финансовых данных

---

## ОГЛАВЛЕНИЕ

- [Полная структура 56 признаков](#полная-структура-56-признаков)
- [ЭТАП 1: Признаки 1-9](#этап-1-признаки-1-9)
- [ЭТАП 2: Признаки 10-19](#этап-2-признаки-10-19)
- [ЭТАП 3: Признаки 20-28](#этап-3-признаки-20-28)
- [ЭТАП 4: Признаки 29-37](#этап-4-признаки-29-37)
- [ЭТАП 5: Признаки 38-47](#этап-5-признаки-38-47)
- [ЭТАП 6: Признаки 48-56](#этап-6-признаки-48-56)

---

## ПОЛНАЯ СТРУКТУРА 56 ПРИЗНАКОВ

### Архитектура observation vector

```
БЛОК 1: BAR (3) - индексы 0-2
├─ price
├─ log_volume_norm
└─ rel_volume

БЛОК 2-3: MOVING AVERAGES (4) - индексы 3-6
├─ ma5 (sma_1200 = 5 баров × 240 мин)
├─ ma5_valid
├─ ma20 (sma_5040 = 21 бар × 240 мин)
└─ ma20_valid

БЛОК 4: INDICATORS (13) - индексы 7-19
├─ rsi14
├─ macd
├─ macd_signal
├─ momentum
├─ atr
├─ cci
├─ obv
├─ ret_bar
└─ vol_proxy

БЛОК 5: AGENT (6) - индексы 16-21
├─ cash_ratio
├─ position_norm
├─ last_vol_imbalance
├─ last_trade_intensity
├─ last_realized_spread
└─ last_agent_fill_ratio

БЛОК 6: MICROSTRUCTURE (3) - индексы 22-24
├─ price_momentum
├─ bb_squeeze
└─ trend_strength

БЛОК 7: BOLLINGER BANDS (2) - индексы 25-26
├─ bb_position
└─ bb_width

БЛОК 8: METADATA (5) - индексы 27-31
├─ is_high_importance
├─ time_since_event
├─ risk_off_flag
├─ fear_greed_value
└─ fear_greed_indicator

БЛОК 9: EXTERNAL (21) - индексы 32-52
├─ cvd_24h
├─ cvd_7d
├─ yang_zhang_48h
├─ yang_zhang_7d
├─ garch_200h
├─ garch_14d
├─ ret_12h
├─ ret_24h
├─ ret_4h
├─ sma_12000
├─ yang_zhang_30d
├─ parkinson_48h
├─ parkinson_7d
├─ garch_30d
├─ taker_buy_ratio
├─ taker_buy_ratio_sma_24h
├─ taker_buy_ratio_sma_8h
├─ taker_buy_ratio_sma_16h
├─ taker_buy_ratio_momentum_4h
├─ taker_buy_ratio_momentum_8h
└─ taker_buy_ratio_momentum_12h

БЛОК 10: TOKEN_META (2) - индексы 53-54
├─ num_tokens_norm
└─ token_id_norm

БЛОК 11: TOKEN (1) - индекс 55
└─ token_one_hot
```

### Путь создания данных

```
RAW DATA (Binance/Exchange)
    ↓
transformers.py: OnlineFeatureTransformer.update()
    ├─ Вычисление SMA, RSI, Yang-Zhang, Parkinson, GARCH
    ├─ Вычисление CVD, Taker Buy Ratio
    └─ Возврат словаря признаков
    ↓
mediator.py: _extract_norm_cols()
    ├─ Извлечение 21 external признака
    └─ Возврат np.ndarray[21]
    ↓
mediator.py: _build_observation()
    ├─ _extract_market_data() → price, volume
    ├─ _extract_technical_indicators() → ma5, ma20, rsi14, etc.
    ├─ _extract_norm_cols() → 21 external
    └─ build_observation_vector() в obs_builder.pyx
    ↓
obs_builder.pyx: build_observation_vector()
    ├─ Валидация входных данных (P0, P1, P2)
    ├─ Заполнение 56 признаков
    ├─ Применение tanh/clip нормализации
    └─ Возврат observation[56]
    ↓
OBSERVATION VECTOR (готов для модели)
```

---

## ЭТАП 1: Признаки 1-9

> **Охват:** BAR блок + MA блок + начало INDICATORS блока
> **Файлы:** mediator.py:1037-1199, obs_builder.pyx:229-252

### Признак 1: price (индекс 0)

**Определение:** Текущая цена актива

**Путь создания:**
1. **Источник:** `mediator.py:_extract_market_data()` → параметр `mark_price`
2. **Валидация P0 (mediator.py:1052):** `_validate_critical_price(mark_price)`
   - Проверяет: не None, не NaN, не Inf, > 0
   - При ошибке: ValueError с детальным сообщением
3. **Запись в observation (obs_builder.pyx:230):** `out_features[0] = price`

**Потенциальные искажения:**

| Тип искажения | Источник | Защита | Риск |
|--------------|----------|--------|------|
| **NaN** | Пропущенные данные от биржи | ✅ P0: `_validate_critical_price` (mediator.py:961) | КРИТИЧЕСКИЙ блокируется |
| **Inf** | Переполнение вычислений | ✅ P0: `_validate_critical_price` (mediator.py:969) | КРИТИЧЕСКИЙ блокируется |
| **Отрицательная/нулевая цена** | Ошибка данных | ✅ P0: `_validate_critical_price` (mediator.py:978) | КРИТИЧЕСКИЙ блокируется |
| **Выбросы** | Аномальные рыночные условия | ⚠️ Нет защиты | ВЫСОКИЙ |
| **Спайки** | Flash crashes/pumps | ⚠️ Нет защиты | ВЫСОКИЙ |

**Исследования:**
- "Best Practices for Ensuring Financial Data Accuracy" (Paystand): требует положительные конечные цены
- "Investment Model Validation" (CFA Institute): валидация критичных параметров
- "Training ML Models with Financial Data" (EODHD): fail-fast подход

**Рекомендации:**
1. ⚠️ **ДОБАВИТЬ**: Защиту от выбросов (IQR метод или z-score clipping)
2. ⚠️ **ДОБАВИТЬ**: Проверку на спайки (разница с предыдущей ценой > порог)
3. ✅ **СОХРАНИТЬ**: Строгую валидацию P0 (работает корректно)

---

### Признак 2: log_volume_norm (индекс 1)

**Определение:** Нормализованный логарифм объема торгов

**Путь создания:**
1. **Источник данных (mediator.py:1059):** `_get_safe_float(row, "quote_asset_volume", 1.0, min_value=0.0)`
   - Параметры: default=1.0, min_value=0.0
   - Защита: гарантирует quote_volume ≥ 0
2. **Нормализация (mediator.py:1065):**
   ```python
   if quote_volume > 0:
       log_volume_norm = tanh(log1p(quote_volume / 240e6))
   else:
       log_volume_norm = 0.0
   ```
   - Делитель: 240e6 (адаптация для 4h таймфрейма)
   - Функция: tanh(log1p(x)) → диапазон (-1, 1)
3. **Валидация P2 (obs_builder.pyx:570):** `_validate_volume_metric(log_volume_norm)`
   - Проверяет: не NaN, не Inf
4. **Запись (obs_builder.pyx:232):** `out_features[1] = log_volume_norm`

**Потенциальные искажения:**

| Тип искажения | Источник | Защита | Риск |
|--------------|----------|--------|------|
| **NaN из log1p(-1)** | Отрицательный volume | ✅ P0: min_value=0.0 предотвращает | НИЗКИЙ |
| **NaN из вычислений** | Проблемы upstream | ✅ P2: `_validate_volume_metric` | НИЗКИЙ |
| **Нулевой volume** | Малоликвидные периоды | ⚠️ Fallback к 0.0 может скрыть проблему | СРЕДНИЙ |
| **Аномальные спайки** | Манипуляции/wash trading | ⚠️ Нет защиты | СРЕДНИЙ |
| **Неправильный делитель** | Изменение таймфрейма | ⚠️ Hardcoded 240e6 | СРЕДНИЙ |

**Математика:**
```
log1p(x) = ln(1 + x)
tanh(y) = (e^y - e^-y) / (e^y + e^-y)

Для 4h бара:
- Типичный quote_volume: 100M - 500M USD
- Нормализация: x / 240e6 → 0.4 - 2.0
- log1p(0.4-2.0) → 0.34 - 1.10
- tanh(0.34-1.10) → 0.33 - 0.80
```

**Исследования:**
- "Incomplete Data - Machine Learning Trading" (OMSCS): NaN должны обрабатываться явно
- "Defense in Depth" (OWASP): многоуровневая валидация
- "Data Validation Best Practices" (Cube Software): валидация на границах

**Рекомендации:**
1. ⚠️ **ПЕРЕСМОТРЕТЬ**: Fallback к 0.0 при нулевом volume (может маскировать проблемы)
2. ⚠️ **ДОБАВИТЬ**: Детекцию аномальных спайков volume
3. ⚠️ **ПАРАМЕТРИЗОВАТЬ**: Делитель 240e6 через конфиг (для разных таймфреймов)
4. ✅ **СОХРАНИТЬ**: P0/P2 валидацию (работает хорошо)

---

### Признак 3: rel_volume (индекс 2)

**Определение:** Относительный объем торгов

**Путь создания:**
1. **Источник данных (mediator.py:1058):** `_get_safe_float(row, "volume", 1.0, min_value=0.0)`
   - Параметры: default=1.0, min_value=0.0
   - Защита: гарантирует volume ≥ 0
2. **Нормализация (mediator.py:1068):**
   ```python
   if volume > 0:
       rel_volume = tanh(log1p(volume / 24000.0))
   else:
       rel_volume = 0.0
   ```
   - Делитель: 24000.0 (адаптация для 4h таймфрейма)
   - Функция: tanh(log1p(x)) → диапазон (-1, 1)
3. **Валидация P2 (obs_builder.pyx:571):** `_validate_volume_metric(rel_volume)`
4. **Запись (obs_builder.pyx:234):** `out_features[2] = rel_volume`

**Потенциальные искажения:**

| Тип искажения | Источник | Защита | Риск |
|--------------|----------|--------|------|
| **NaN из log1p(-1)** | Отрицательный volume | ✅ P0: min_value=0.0 | НИЗКИЙ |
| **NaN из вычислений** | Проблемы upstream | ✅ P2: `_validate_volume_metric` | НИЗКИЙ |
| **Нулевой volume** | Малоликвидные периоды | ⚠️ Fallback к 0.0 | СРЕДНИЙ |
| **Wash trading** | Искусственное завышение | ⚠️ Нет защиты | ВЫСОКИЙ |
| **Неправильный делитель** | Изменение таймфрейма | ⚠️ Hardcoded 24000.0 | СРЕДНИЙ |

**Математика:**
```
Для 4h бара:
- Типичный volume (BTC): 50 - 500 BTC
- Нормализация: x / 24000.0 → 0.002 - 0.021
- log1p(0.002-0.021) → 0.002 - 0.021
- tanh(0.002-0.021) → 0.002 - 0.021
```

**Исследования:**
- "Defense in Depth" (OWASP): валидация предотвращает NaN propagation
- "Data validation best practices" (Cube Software): проверка на границах
- "Market Manipulation Detection": wash trading влияет на volume метрики

**Рекомендации:**
1. ⚠️ **ДОБАВИТЬ**: Детекцию wash trading (сравнение с historical volume profile)
2. ⚠️ **ПАРАМЕТРИЗОВАТЬ**: Делитель 24000.0 через конфиг
3. ⚠️ **ЛОГИРОВАТЬ**: Случаи нулевого volume для анализа
4. ✅ **СОХРАНИТЬ**: P0/P2 валидацию

---

### Признак 4: ma5 (индекс 3)

**Определение:** 5-периодное скользящее среднее (sma_1200 = 5 баров × 240 мин = 20h)

**Путь создания:**
1. **Генерация (transformers.py:859-867):**
   ```python
   for i, lb in enumerate(self.spec.lookbacks_prices):
       lb_minutes = self.spec._lookbacks_prices_minutes[i]
       if len(seq) >= lb:
           window = seq[-lb:]
           sma = sum(window) / float(lb)
           feats[f"sma_{lb_minutes}"] = float(sma)
   ```
   - Окно: последние 5 баров (1200 минут)
   - Формула: среднее арифметическое

2. **Извлечение (mediator.py:1083):** `_get_safe_float(row, "sma_1200", nan)`
   - Fallback: NaN (индикатор недостаточности данных)

3. **Обработка NaN (obs_builder.pyx:237-240):**
   ```cython
   ma5_valid = not isnan(ma5)
   out_features[3] = ma5 if ma5_valid else 0.0
   out_features[4] = 1.0 if ma5_valid else 0.0  # validity flag
   ```

**Потенциальные искажения:**

| Тип искажения | Источник | Защита | Риск |
|--------------|----------|--------|------|
| **NaN при холодном старте** | Недостаточно баров (< 5) | ✅ Флаг валидности + fallback 0.0 | НИЗКИЙ |
| **Запаздывание (lag)** | Природа SMA | ⚠️ Присуще индикатору | СРЕДНИЙ |
| **Смещение данных (look-ahead)** | Неправильная индексация | ⚠️ Требует проверки | КРИТИЧЕСКИЙ |
| **Точность float32** | Потеря точности при суммировании | ⚠️ Нет защиты | НИЗКИЙ |

**Математика:**
```
SMA_5 = (P_t + P_{t-1} + P_{t-2} + P_{t-3} + P_{t-4}) / 5

где P_i - цена закрытия i-го бара

Lag: ~2.5 бара (половина окна)
```

**Исследования:**
- "Technical Analysis Indicators" (Murphy, 1999): SMA имеет lag = window/2
- "Look-Ahead Bias in Financial ML" (de Prado, 2018): критичная проблема
- "Numerical Stability in Trading Systems" (Kahan summation для больших окон)

**Рекомендации:**
1. ✅ **СОХРАНИТЬ**: Флаг валидности (хорошая практика)
2. ⚠️ **ПРОВЕРИТЬ**: Отсутствие look-ahead bias (индексация seq[-lb:] корректна?)
3. ⚠️ **РАССМОТРЕТЬ**: EMA вместо SMA (меньший lag)
4. ⚠️ **ДОКУМЕНТИРОВАТЬ**: Ожидаемый lag для интерпретации модели

---

### Признак 5: ma5_valid (индекс 4)

**Определение:** Флаг валидности ma5 (1.0 = валидно, 0.0 = невалидно)

**Путь создания:**
1. **Вычисление (obs_builder.pyx:237):** `ma5_valid = not isnan(ma5)`
2. **Запись (obs_builder.pyx:240):** `out_features[4] = 1.0 if ma5_valid else 0.0`

**Потенциальные искажения:**

| Тип искажения | Источник | Защита | Риск |
|--------------|----------|--------|------|
| **Некорректная валидность** | ma5 = 0.0 != NaN | ✅ Использует isnan(), не проверку на 0 | НИЗКИЙ |
| **Временная согласованность** | Флаг не совпадает со значением | ✅ Вычисляется напрямую из ma5 | НИЗКИЙ |

**Исследования:**
- "Handling Missing Indicators" (QuantConnect): флаги валидности - best practice
- "Feature Engineering for ML" (Zheng & Casari): indicator validity как признак

**Рекомендации:**
1. ✅ **СОХРАНИТЬ**: Подход с флагом (корректен и полезен для модели)
2. ✅ **ХОРОШАЯ ПРАКТИКА**: Позволяет модели учиться различать холодный старт

---

### Признак 6: ma20 (индекс 5)

**Определение:** 20-периодное скользящее среднее (sma_5040 = 21 бар × 240 мин = 84h ≈ 3.5 дня)

**Путь создания:**
1. **Генерация (transformers.py:859-867):** Аналогично ma5, окно = 21 бар
2. **Извлечение (mediator.py:1086):** `_get_safe_float(row, "sma_5040", nan)`
3. **Обработка NaN (obs_builder.pyx:243-246):**
   ```cython
   ma20_valid = not isnan(ma20)
   out_features[5] = ma20 if ma20_valid else 0.0
   out_features[6] = 1.0 if ma20_valid else 0.0  # validity flag
   ```

**Потенциальные искажения:**

| Тип искажения | Источник | Защита | Риск |
|--------------|----------|--------|------|
| **NaN при холодном старте** | Недостаточно баров (< 21) | ✅ Флаг валидности + fallback 0.0 | НИЗКИЙ |
| **Большее запаздывание** | Окно 21 бар (lag ~10.5 бар) | ⚠️ Присуще индикатору | СРЕДНИЙ |
| **Смещение данных** | Look-ahead bias | ⚠️ Требует проверки | КРИТИЧЕСКИЙ |
| **Точность суммирования** | float32 для 21 элемента | ⚠️ Потенциальная потеря точности | НИЗКИЙ |

**Математика:**
```
SMA_21 = Σ(P_i) / 21, i = t-20 ... t

Lag: ~10.5 бара = 42 часа
Для 4h таймфрейма это значительный lag
```

**Исследования:**
- "SMA vs EMA Performance" (Murphy): большие окна → больший lag
- "Numerical Stability" (Kahan): суммирование требует осторожности

**Рекомендации:**
1. ⚠️ **РАССМОТРЕТЬ**: EMA вместо SMA (lag ≈ 4.2 бара вместо 10.5)
2. ⚠️ **ПРОВЕРИТЬ**: Корректность индексации (look-ahead bias)
3. ⚠️ **ОПТИМИЗИРОВАТЬ**: Kahan summation для численной стабильности
4. ✅ **СОХРАНИТЬ**: Флаг валидности

---

### Признак 7: ma20_valid (индекс 6)

**Определение:** Флаг валидности ma20 (1.0 = валидно, 0.0 = невалидно)

**Путь создания:**
1. **Вычисление (obs_builder.pyx:243):** `ma20_valid = not isnan(ma20)`
2. **Запись (obs_builder.pyx:246):** `out_features[6] = 1.0 if ma20_valid else 0.0`

**Потенциальные искажения:** Аналогично ma5_valid (см. выше)

**Рекомендации:**
1. ✅ **СОХРАНИТЬ**: Корректная реализация

---

### Признак 8: rsi14 (индекс 7)

**Определение:** 14-периодный RSI (Relative Strength Index) по методу Уайлдера

**Путь создания:**
1. **Генерация (transformers.py:881-900):**
   ```python
   # Wilder's RSI formula with exponential smoothing
   if st["avg_gain"] is not None and st["avg_loss"] is not None:
       avg_gain = float(st["avg_gain"])
       avg_loss = float(st["avg_loss"])

       # Edge cases
       if avg_loss == 0.0 and avg_gain > 0.0:
           feats["rsi"] = 100.0  # Pure uptrend
       elif avg_gain == 0.0 and avg_loss > 0.0:
           feats["rsi"] = 0.0    # Pure downtrend
       elif avg_gain == 0.0 and avg_loss == 0.0:
           feats["rsi"] = 50.0   # No movement
       else:
           rs = avg_gain / avg_loss
           feats["rsi"] = 100.0 - (100.0 / (1.0 + rs))
   else:
       feats["rsi"] = nan
   ```

2. **Обновление avg_gain/avg_loss (transformers.py:792-801):**
   ```python
   delta = price - float(last)
   gain = max(delta, 0.0)
   loss = max(-delta, 0.0)

   if st["avg_gain"] is None:
       st["avg_gain"] = gain
       st["avg_loss"] = loss
   else:
       p = self.spec.rsi_period  # 14
       st["avg_gain"] = ((avg_gain * (p - 1)) + gain) / p
       st["avg_loss"] = ((avg_loss * (p - 1)) + loss) / p
   ```

3. **Извлечение (mediator.py:1087):** `_get_safe_float(row, "rsi", 50.0)`
   - Fallback: 50.0 (нейтральное значение)

4. **Обработка NaN (obs_builder.pyx:251):**
   ```cython
   out_features[7] = rsi14 if not isnan(rsi14) else 50.0
   ```

**Потенциальные искажения:**

| Тип искажения | Источник | Защита | Риск |
|--------------|----------|--------|------|
| **NaN при холодном старте** | Недостаточно данных (< 14 баров) | ✅ Fallback 50.0 (нейтральное) | НИЗКИЙ |
| **Деление на ноль** | avg_loss = 0 | ✅ Обработка edge cases (transformers.py:886-894) | НИЗКИЙ |
| **Смещение инициализации** | Первая avg = первый delta (не среднее) | ⚠️ Wilder's method (общепринято, но неточно в начале) | СРЕДНИЙ |
| **Численная нестабильность** | Малые значения avg_loss | ⚠️ Может привести к экстремальным RS | СРЕДНИЙ |
| **Запаздывание** | EMA сглаживание с периодом 14 | ⚠️ Присуще индикатору | НИЗКИЙ |

**Математика:**
```
Wilder's RSI:
1. avg_gain_t = ((avg_gain_{t-1} × 13) + gain_t) / 14
2. avg_loss_t = ((avg_loss_{t-1} × 13) + loss_t) / 14
3. RS = avg_gain / avg_loss
4. RSI = 100 - (100 / (1 + RS))

Диапазон: [0, 100]
- RSI > 70: overbought
- RSI < 30: oversold
- RSI = 50: neutral

Эквивалент EMA с α = 1/14 ≈ 0.071
```

**Исследования:**
- Wilder, J. Welles (1978): "New Concepts in Technical Trading Systems"
- "RSI Edge Cases" (TradingView): обработка avg_loss = 0
- "Numerical Issues in RSI" (QuantConnect): деление на малые числа

**Рекомендации:**
1. ✅ **СОХРАНИТЬ**: Обработку edge cases (корректная реализация)
2. ⚠️ **ДОБАВИТЬ**: Epsilon в знаменатель для численной стабильности:
   ```python
   rs = avg_gain / (avg_loss + 1e-10)
   ```
3. ⚠️ **ДОКУМЕНТИРОВАТЬ**: Поведение при холодном старте (неточность первых 14 значений)
4. ⚠️ **РАССМОТРЕТЬ**: Альтернативную инициализацию (SMA первых 14 баров вместо первого delta)

---

### Признак 9: macd (индекс 8)

**Определение:** MACD (Moving Average Convergence Divergence) - разница между быстрой и медленной EMA

**Путь создания:**
1. **Вычисление (в MarketSimulator, см. mediator.py:1100-1103):**
   ```python
   if sim is not None and hasattr(sim, "get_macd"):
       macd = float(sim.get_macd(row_idx))
   else:
       macd = 0.0  # Fallback
   ```
   - Источник: MarketSimulator (не из transformers.py)
   - Вероятная формула: MACD = EMA_12 - EMA_26

2. **Обработка NaN (obs_builder.pyx:255):**
   ```cython
   out_features[8] = macd if not isnan(macd) else 0.0
   ```

**Потенциальные искажения:**

| Тип искажения | Источник | Защита | Риск |
|--------------|----------|--------|------|
| **NaN при холодном старте** | Недостаточно данных (< 26 баров для EMA_26) | ✅ Fallback 0.0 | СРЕДНИЙ |
| **Fallback маскирует проблемы** | 0.0 неотличим от "MACD = 0" | ⚠️ Нет флага валидности | ВЫСОКИЙ |
| **Зависимость от внешнего модуля** | MarketSimulator может отсутствовать | ⚠️ Fallback к 0.0 без логирования | ВЫСОКИЙ |
| **Запаздывание** | EMA природа (lag ≈ 13 баров) | ⚠️ Присуще индикатору | СРЕДНИЙ |
| **Неизвестная реализация** | get_macd() не в transformers.py | ⚠️ Требует проверки исходного кода | КРИТИЧЕСКИЙ |

**Математика (стандартная формула):**
```
MACD = EMA_12(price) - EMA_26(price)

где EMA_n:
EMA_t = α × price_t + (1-α) × EMA_{t-1}
α = 2 / (n + 1)

Для MACD:
- α_12 = 2/13 ≈ 0.154
- α_26 = 2/27 ≈ 0.074

Lag EMA_26 ≈ (26-1)/2 ≈ 12.5 баров
```

**Исследования:**
- Appel, Gerald (1979): "The Moving Average Convergence Divergence Trading Method"
- "MACD Calculation" (Investopedia): стандартная формула
- "Handling Missing Technical Indicators" (QuantConnect): флаги валидности

**Рекомендации:**
1. ⚠️ **КРИТИЧНО ПРОВЕРИТЬ**: Реализацию get_macd() в MarketSimulator
2. ⚠️ **ДОБАВИТЬ**: Флаг валидности (как для ma5/ma20)
3. ⚠️ **ЛОГИРОВАТЬ**: Случаи fallback к 0.0 для отладки
4. ⚠️ **РАССМОТРЕТЬ**: Перенос вычисления в transformers.py для прозрачности
5. ⚠️ **АЛЬТЕРНАТИВА**: Использовать NaN вместо 0.0 для честного отображения недоступности

---

## ЭТАП 2: Признаки 10-19

> **СТАТУС:** Ожидает выполнения
> **Команда пользователя:** "идти дальше"

---

## ЭТАП 3: Признаки 20-28

> **СТАТУС:** Ожидает выполнения
> **Команда пользователя:** "идти дальше"

---

## ЭТАП 4: Признаки 29-37

> **СТАТУС:** Ожидает выполнения
> **Команда пользователя:** "идти дальше"

---

## ЭТАП 5: Признаки 38-47

> **СТАТУС:** Ожидает выполнения
> **Команда пользователя:** "идти дальше"

---

## ЭТАП 6: Признаки 48-56

> **СТАТУС:** Ожидает выполнения
> **Команда пользователя:** "идти дальше"

---

## МЕТОДОЛОГИЯ АНАЛИЗА

Для каждого признака анализируется:

1. **Определение**: Что измеряет признак
2. **Путь создания**: Полная цепочка от raw data до observation
3. **Потенциальные искажения**: Таблица с типами, источниками, защитой и рисками
4. **Математика**: Формулы и вычисления
5. **Исследования**: Ссылки на best practices
6. **Рекомендации**: Конкретные действия с приоритетами

### Уровни риска:
- **КРИТИЧЕСКИЙ**: Может вызвать сбой системы или серьезное искажение
- **ВЫСОКИЙ**: Значительное влияние на качество данных
- **СРЕДНИЙ**: Умеренное влияние, требует внимания
- **НИЗКИЙ**: Минимальное влияние, хорошо контролируется

### Приоритеты рекомендаций:
- ✅ **СОХРАНИТЬ**: Работает корректно, не требует изменений
- ⚠️ **ДОБАВИТЬ**: Новая защита/функциональность
- ⚠️ **ПЕРЕСМОТРЕТЬ**: Требует изменения подхода
- ⚠️ **ПРОВЕРИТЬ**: Требует дополнительного исследования
- ⚠️ **КРИТИЧНО**: Требует немедленного внимания

---

**КОНЕЦ ЭТАПА 1**

Ждите команды пользователя "идти дальше" для продолжения анализа признаков 10-19.

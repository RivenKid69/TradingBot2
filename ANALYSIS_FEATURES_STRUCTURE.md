# МАКСИМАЛЬНО ГЛУБОКИЙ АНАЛИЗ ВСЕХ 56 ПРИЗНАКОВ
## Миграция 1h -> 4h таймфрейм

### СТРУКТУРА 56 ПРИЗНАКОВ

#### 1. Bar-level features (3 признака)
- `price` - текущая цена (индекс 0)
- `log_volume_norm` - нормализованный логарифм объема (индекс 1)  
- `rel_volume` - относительный объем (индекс 2)

#### 2. MA indicators (4 признака) - ВНИМАНИЕ!
- `ma5` - SMA 5 баров (индекс 3)
- `ma5_valid` - флаг валидности ma5 (индекс 4)
- `ma20` - SMA 20 баров (индекс 5)
- `ma20_valid` - флаг валидности ma20 (индекс 6)

#### 3. Technical indicators (6 признаков)
- `rsi14` (индекс 7)
- `macd` (индекс 8)
- `macd_signal` (индекс 9)
- `momentum` (индекс 10)
- `atr` (индекс 11)
- `cci` (индекс 12)
- `obv` (индекс 13)

#### 4. Derived features (2 признака)
- `ret_bar` - таnh((price - prev_price) / prev_price) (индекс 14)
- `vol_proxy` - tanh(log1p(atr / price)) (индекс 15)

#### 5. Agent state features (6 признаков)
- `cash_ratio` - clip(cash / total_worth, 0, 1) (индекс 16)
- `position_ratio` - tanh(position_value / total_worth) (индекс 17)
- `vol_imbalance` - tanh(last_vol_imbalance) (индекс 18)
- `trade_intensity` - tanh(last_trade_intensity) (индекс 19)
- `realized_spread` - clip(last_realized_spread, -0.1, 0.1) (индекс 20)
- `fill_ratio` (индекс 21)

#### 6. Microstructure replacement (3 признака) - ЗАМЕНА!
- `price_momentum` - tanh(momentum / (price * 0.01)) (индекс 22)
- `bb_squeeze` - tanh((bb_upper - bb_lower) / price) (индекс 23)
- `trend_strength` - tanh((macd - macd_signal) / (price * 0.01)) (индекс 24)

#### 7. BB context (2 признака)
- `bb_position` - (price - bb_lower) / bb_width (индекс 25)
- `bb_width` - (bb_upper - bb_lower) / price (индекс 26)

#### 8. Event metadata (5 признаков)
- `is_high_importance` (индекс 27)
- `time_since_event` - tanh(time_since_event / 6.0) (индекс 28) - КРИТИЧНО!
- `risk_off_flag` (индекс 29)
- `fear_greed_value` - clip(fear_greed_value / 100.0, -3, 3) (индекс 30)
- `fear_greed_indicator` (индекс 31)

#### 9. External norm_cols (21 признак) - КРИТИЧНО!
[Индексы 32-52]
1. `cvd_24h` [0]
2. `cvd_7d` [1]
3. `yang_zhang_48h` [2] - 12 баров
4. `yang_zhang_7d` [3]
5. `garch_200h` [4] - 50 баров, МИНИМУМ для GARCH
6. `garch_14d` [5]
7. `ret_12h` [6] - 3 бара
8. `ret_24h` [7] - 6 баров
9. `ret_4h` [8] - 1 бар
10. `sma_12000` [9] - 50 баров
11. `yang_zhang_30d` [10]
12. `parkinson_48h` [11]
13. `parkinson_7d` [12]
14. `garch_30d` [13]
15. `taker_buy_ratio` [14]
16. `taker_buy_ratio_sma_24h` [15]
17. `taker_buy_ratio_sma_8h` [16]
18. `taker_buy_ratio_sma_16h` [17]
19. `taker_buy_ratio_momentum_4h` [18]
20. `taker_buy_ratio_momentum_8h` [19]
21. `taker_buy_ratio_momentum_12h` [20]

#### 10. Token metadata (2 признака)
- `num_tokens_norm` (индекс 53)
- `token_id_norm` (индекс 54)

#### 11. Token one-hot (1 признак)
- `token_one_hot` (индекс 55)

---

### КРИТИЧЕСКИЕ НАЙДЕННЫЕ ПРОБЛЕМЫ

#### ПРОБЛЕМА #1: РАЗНОЕ КОЛИЧЕСТВО ВНЕШНИХ ПРИЗНАКОВ
**Файл:** mediator.py, строки 1014-1057
**Тип:** НЕСООТВЕТСТВИЕ
**Статус:** НАЙДЕНО И ЗАФИКСИРОВАНО

```python
# mediator.py - _extract_norm_cols возвращает 21 признак (строка 1023)
norm_cols = np.zeros(21, dtype=np.float32)
```

**НАЙДЕНО**: Feature config говорит 21, mediator говорит 21, но есть несоответствие в именах!

#### ПРОБЛЕМА #2: НЕПРАВИЛЬНАЯ НОРМАЛИЗАЦИЯ time_since_event
**Файл:** obs_builder.pyx, строка 204
**Тип:** КРИТИЧЕСКАЯ ОШИБКА
**Статус:** ТРЕБУЕТ ПРОВЕРКИ

```cython
# obs_builder.pyx строка 204 (НЕПРАВИЛЬНО для 4h!)
out_features[feature_idx] = <float>tanh(time_since_event / 6.0)
```

**ПРОБЛЕМА**: Делитель 6.0 используется, но:
- Для 1h интервала: 24 (24 часа / 1 час = 24 бара)
- Для 4h интервала: 6 (24 часа / 4 часа = 6 баров) - ПРАВИЛЬНО!

**СТАТУС**: ✓ ПРАВИЛЬНО (уже исправлено)

#### ПРОБЛЕМА #3: НЕПРАВИЛЬНАЯ НОРМАЛИЗАЦИЯ ОБЪЕМОВ  
**Файл:** mediator.py, строки 935-948
**Тип:** НЕСООТВЕТСТВИЕ ДЕЛИТЕЛЕЙ
**Статус:** КРИТИЧЕСКАЯ

```python
# Строка 942: для log_volume_norm используется 240e6
log_volume_norm = float(np.tanh(np.log1p(quote_volume / 240e6)))

# Строка 948: для rel_volume используется 24000
rel_volume = float(np.tanh(np.log1p(volume / 24000.0)))
```

**ПРОВЕРКА в config_4h_timeframe.py**:
- VOLUME_NORM_DIVISOR = 240e6 ✓ (совпадает)
- REL_VOLUME_DIVISOR = 24000 ✓ (совпадает)

**СТАТУС**: ✓ СОГЛАСОВАНО

#### ПРОБЛЕМА #4: НЕСООТВЕТСТВИЕ ИМЕН SMA В РАЗНЫХ МЕСТАХ
**Файл:** 
  - mediator.py, строка 962: `sma_1200` (для ma5)
  - mediator.py, строка 965: `sma_5040` (для ma20)
  - transformers.py, строка 606: `sma_240`, `sma_720`, `sma_1200`, `sma_1440`, `sma_5040`, `sma_10080`, `sma_12000`

**ПРОБЛЕМА**: Имена SMA в ПОЛНОМ наборе (transformers) отличаются от имен в mediator!

**КРИТИЧЕСКИЙ АНАЛИЗ**:
1. mediator.py использует ТОЛЬКО 2 SMA:
   - sma_1200 (5 баров = 1200 минут = 20 часов)
   - sma_5040 (21 бар = 5040 минут = 84 часа = 3.5 дня)

2. transformers.py генерирует ВСЕ 7 SMA:
   - sma_240, sma_720, sma_1200, sma_1440, sma_5040, sma_10080, sma_12000

3. mediator.py НЕ ВЫТЯГИВАЕТ некоторые SMA!

**СТАТУС**: ❌ ПРОБЛЕМА НАЙДЕНА

#### ПРОБЛЕМА #5: НЕПРАВИЛЬНЫЕ ОКНА GARCH
**Файл:** mediator.py, строки 1031-1033
**Тип:** НЕСООТВЕТСТВИЕ
**Статус:** КРИТИЧЕСКАЯ

```python
norm_cols[4] = self._get_safe_float(row, "garch_200h", 0.0)   # 50 баров
norm_cols[5] = self._get_safe_float(row, "garch_14d", 0.0)    # 84 бара
```

**ПРОВЕРКА в test_4h_integration_fixes.py, строки 66-69**:
```python
# Для 4h интервала: 12000, 20160, 43200 минут
# После конвертации: 50, 84, 180 баров
expected_bars = [50, 84, 180]
```

**СТАТУС**: ✓ ПРАВИЛЬНО

#### ПРОБЛЕМА #6: НЕСООТВЕТСТВИЕ ОКОН TAKER BUY RATIO
**Файл:** 
  - config_4h_timeframe.py, строки 154-163
  - mediator.py, строки 1047-1051

**АНАЛИЗ**:

config_4h_timeframe.py:
```python
TAKER_BUY_RATIO_SMA_WINDOWS = [2, 4, 6]  # бары: 8h, 16h, 24h
TAKER_BUY_RATIO_MOMENTUM_WINDOWS = [1, 2, 3, 6]  # бары: 4h, 8h, 12h, 24h
```

mediator.py (строки 1047-1051):
```python
norm_cols[16] = self._get_safe_float(row, "taker_buy_ratio_sma_8h", 0.0)     # 2 бара
norm_cols[17] = self._get_safe_float(row, "taker_buy_ratio_sma_16h", 0.0)   # 4 бара
norm_cols[18] = self._get_safe_float(row, "taker_buy_ratio_momentum_4h", 0.0)  # 1 бар
norm_cols[19] = self._get_safe_float(row, "taker_buy_ratio_momentum_8h", 0.0)  # 2 бара
norm_cols[20] = self._get_safe_float(row, "taker_buy_ratio_momentum_12h", 0.0) # 3 бара
```

**СТАТУС**: ✓ СОГЛАСОВАНО

#### ПРОБЛЕМА #7: ОТСУТСТВИЕ MOMENTUM_24h В TAKER BUY RATIO
**Файл:** mediator.py, строки 1047-1051
**Тип:** НЕПОЛНОТА
**Статус:** КРИТИЧЕСКАЯ

**НАЙДЕНО**: 
- transformers.py генерирует 4 моментума: 4h, 8h, 12h, 24h (240, 480, 720, 1440 минут)
- mediator.py вытягивает ТОЛЬКО 3: 4h, 8h, 12h (отсутствует 24h!)

**КРИТИЧЕСКИЙ АНАЛИЗ**:
1. transformers.py строка 413:
```python
self.taker_buy_ratio_momentum = [4 * 60, 8 * 60, 12 * 60, 24 * 60]  # 4h, 8h, 12h, 24h
```

2. mediator.py отсутствует `taker_buy_ratio_momentum_24h` в норм_колс!

**СТАТУС**: ❌ КРИТИЧЕСКАЯ ПРОБЛЕМА

#### ПРОБЛЕМА #8: НЕСООТВЕТСТВИЕ МЕЖДУ ОНЛАЙН И ОФФЛАЙН ИМЕН SMA
**Файл:** transformers.py, строка 606 vs строка 781
**Тип:** НЕСООТВЕТСТВИЕ
**Статус:** КРИТИЧЕСКАЯ

**ОНЛАЙН** (строка 606):
```python
feats[f"sma_{lb_minutes}"] = float(sma)  # sma_240, sma_720, ...
```

**ОФФЛАЙН** (строка 781):
```python
base_cols += [f"sma_{x}" for x in spec.lookbacks_prices]  # sma_1, sma_3, ...
```

**ПРОБЛЕМА**: Онлайн использует МИНУТЫ (sma_240), оффлайн использует БАРЫ (sma_1)!

**СТАТУС**: ❌ КРИТИЧЕСКАЯ ПРОБЛЕМА

#### ПРОБЛЕМА #9: НЕСООТВЕТСТВИЕ RETURNS ИМЕН
**Файл:** transformers.py, строка 609
**Тип:** НЕПРАВИЛЬНОЕ ФОРМАТИРОВАНИЕ
**Статус:** ТРЕБУЕТ ПРОВЕРКИ

```python
ret_name = f"ret_{_format_window_name(lb_minutes)}"
```

**ПРОВЕРКА _format_window_name**:
- 240 → "4h" ✓
- 720 → "12h" ✓
- 1200 → "20h" ✓
- 1440 → "24h" ✓
- 5040 → "84h" ✓ (ВАЖНО: 5040 = 84 часа, не кратно дню)
- 10080 → "7d" ✓
- 12000 → "200h" ✓

**СТАТУС**: ✓ ПРАВИЛЬНО

#### ПРОБЛЕМА #10: НЕПРАВИЛЬНЫЕ ДЕЛИТЕЛИ НОРМАЛИЗАЦИИ RETURNED В MEDIATOR
**Файл:** mediator.py, строки 1034-1039
**Тип:** НЕСООТВЕТСТВИЕ
**Статус:** ПРОВЕРИТЬ

```python
norm_cols[6] = self._get_safe_float(row, "ret_12h", 0.0)   # ret за 3 бара
norm_cols[7] = self._get_safe_float(row, "ret_24h", 0.0)   # ret за 6 баров
norm_cols[8] = self._get_safe_float(row, "ret_4h", 0.0)    # ret за 1 бар
```

**ПРОБЛЕМА**: Это просто извлекаемые признаки, не создаваемые в mediator.
Они должны быть созданы в transformers.py!

**СТАТУС**: ✓ ПРАВИЛЬНО

#### ПРОБЛЕМА #11: ДВОЙНАЯ НОРМАЛИЗАЦИЯ В OBS_BUILDER
**Файл:** obs_builder.pyx, строки 223-227
**Тип:** ДВОЙНАЯ НОРМАЛИЗАЦИЯ
**Статус:** НАЙДЕНО И ПРОВЕРЕНО

```cython
# External normalized columns (уже нормализованы в mediator!)
for i in range(norm_cols_values.shape[0]):
    feature_val = _clipf(tanh(norm_cols_values[i]), -3.0, 3.0)  # ДВОЙНОЙ tanh!
    out_features[feature_idx] = feature_val
```

**ПРОБЛЕМА**: norm_cols в mediator НЕ нормализованы (это сырые значения),
но в obs_builder применяется tanh + clip!

**АНАЛИЗ**:
- verify_56_features.py, строки 99-102:
```python
assert norm_cols[0] > 10.0, \
    f"❌ norm_cols[0]={norm_cols[0]}, похоже что tanh уже применен (должно быть ~1000)"
```

Это подтверждает: норм_колс НЕ нормализованы в mediator!

**СТАТУС**: ✓ ПРАВИЛЬНО (tanh применяется в obs_builder)

#### ПРОБЛЕМА #12: НЕСООТВЕТСТВИЕ МЕЖДУ CONFIG И РЕАЛЬНЫМ КОДОМ
**Файл:** config_4h_timeframe.py vs transformers.py
**Тип:** НЕСООТВЕТСТВИЕ ДЕФОЛТОВ
**Статус:** НАЙДЕНО

**config_4h_timeframe.py ПРЕДЛАГАЕТ** (строка 355):
```python
COMBINED_LOOKBACKS = [1, 3, 5, 6, 21, 42, 50]  # бары
# Конвертируются в: [240, 720, 1200, 1440, 5040, 10080, 12000] минут
```

**transformers.py ИСПОЛЬЗУЕТ** (строка 329):
```python
self.lookbacks_prices = [240, 720, 1200, 1440, 5040, 10080, 12000]
```

**СТАТУС**: ✓ СОГЛАСОВАНО

#### ПРОБЛЕМА #13: ОТСУТСТВИЕ GARCH_200H В ОФФЛАЙН ГЕНЕРАЦИИ
**Файл:** make_features.py, строка 36
**Тип:** НЕСООТВЕТСТВИЕ ДЕФОЛТОВ
**Статус:** КРИТИЧЕСКАЯ

```python
p.add_argument("--garch-windows", default="12000,20160,43200", 
    help="Окна GARCH(1,1) волатильности в минутах для 4h 
    (по умолчанию 12000,20160,43200 = 8d,14d,30d)")
```

**ПРОБЛЕМА**:
- Комментарий говорит "8d,14d,30d"
- На самом деле: 12000 = 200h, 20160 = 14d, 43200 = 30d
- Комментарий НЕПРАВИЛЬНЫЙ! 12000 минут = 200 часов, а не 8 дней!

**ПРАВИЛЬНО ДОЛЖНО БЫТЬ**:
```python
help="...200h (50 баров),14d,30d..."
```

**СТАТУС**: ❌ ОШИБКА В КОММЕНТАРИИ


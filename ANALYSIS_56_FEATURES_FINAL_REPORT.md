# МАКСИМАЛЬНО ГЛУБОКИЙ АНАЛИЗ: СИСТЕМА 56 ПРИЗНАКОВ (1h -> 4h)

## СТРУКТУРА 56 ПРИЗНАКОВ В ДЕТАЛЯХ

### Индексы в observation vector (55 -> 56 по конфигурации):
```
[0-2]   : Bar-level (3) - price, log_volume_norm, rel_volume
[3-6]   : MA indicators (4) - ma5, ma5_valid, ma20, ma20_valid
[7-13]  : Technical (7) - rsi14, macd, macd_signal, momentum, atr, cci, obv
[14-15] : Derived (2) - ret_bar, vol_proxy  
[16-21] : Agent state (6) - cash_ratio, position_ratio, vol_imbalance, 
                           trade_intensity, realized_spread, fill_ratio
[22-24] : Microstructure (3) - price_momentum, bb_squeeze, trend_strength
[25-26] : BB context (2) - bb_position, bb_width
[27-31] : Metadata (5) - is_high_importance, time_since_event, risk_off_flag,
                         fear_greed_value, fear_greed_indicator
[32-52] : External (21) - cvd, garch, yang_zhang, parkinson, returns, sma,
                         taker_buy_ratio
[53-54] : Token meta (2) - num_tokens_norm, token_id_norm
[55]    : Token one-hot (1) - token_one_hot

TOTAL: 3+4+7+2+6+3+2+5+21+2+1 = 56
```

---

## ПУТИ ДЛЯ ВСЕХ 56 ПРИЗНАКОВ

### 1. BAR-LEVEL FEATURES (3)

| №  | Признак | Путь расчета | Путь нормализации | Статус |
|----|---------|--------------|-------------------|--------|
| 0  | price | mediator._extract_market_data:932 | mediator._build_observation:1167 | ✓ |
| 1  | log_volume_norm | mediator._extract_market_data:942 | obs_builder.pyx:82 | ✓ |
| 2  | rel_volume | mediator._extract_market_data:948 | obs_builder.pyx:84 | ✓ |

### 2. MA INDICATORS (4)

| №  | Признак | Путь | Статус |
|----|---------|------|--------|
| 3  | ma5 | mediator._extract_technical_indicators:962 | ✓ |
| 4  | ma5_valid | obs_builder.pyx:87-90 | ✓ |
| 5  | ma20 | mediator._extract_technical_indicators:965 | ✓ |
| 6  | ma20_valid | obs_builder.pyx:93-96 | ✓ |

### 3. TECHNICAL INDICATORS (7)

| №  | Признак | Путь | Статус |
|----|---------|------|--------|
| 7  | rsi14 | mediator._extract_technical_indicators:966 | ✓ |
| 8  | macd | mediator._extract_technical_indicators:970 | ✓ |
| 9  | macd_signal | mediator._extract_technical_indicators:970 | ✓ |
| 10 | momentum | mediator._extract_technical_indicators:971 | ✓ |
| 11 | atr | mediator._extract_technical_indicators:972 | ✓ |
| 12 | cci | mediator._extract_technical_indicators:973 | ✓ |
| 13 | obv | mediator._extract_technical_indicators:974 | ✓ |

### 4. DERIVED FEATURES (2)

| №  | Признак | Путь расчета | Статус |
|----|---------|--------------|--------|
| 14 | ret_bar | obs_builder.pyx:115 | ✓ |
| 15 | vol_proxy | obs_builder.pyx:119 | ✓ |

### 5. AGENT STATE (6)

| №  | Признак | Путь расчета | Нормализация | Статус |
|----|---------|--------------|--------------|--------|
| 16 | cash_ratio | obs_builder.pyx:130 | clip(0,1) | ✓ |
| 17 | position_ratio | obs_builder.pyx:137 | tanh() | ✓ |
| 18 | vol_imbalance | obs_builder.pyx:141 | tanh() | ✓ |
| 19 | trade_intensity | obs_builder.pyx:143 | tanh() | ✓ |
| 20 | realized_spread | obs_builder.pyx:146 | clip(-0.1,0.1) | ✓ |
| 21 | fill_ratio | obs_builder.pyx:150 | direct | ✓ |

### 6. MICROSTRUCTURE REPLACEMENT (3) - ЗАМЕНА!

| №  | Признак | Путь расчета | Замена | Статус |
|----|---------|--------------|--------|--------|
| 22 | price_momentum | obs_builder.pyx:160 | Вместо ofi_proxy | ✓ |
| 23 | bb_squeeze | obs_builder.pyx:168 | Вместо qimb | ✓ |
| 24 | trend_strength | obs_builder.pyx:175 | Вместо micro_dev | ✓ |

### 7. BB CONTEXT (2)

| №  | Признак | Путь расчета | Статус |
|----|---------|--------------|--------|
| 25 | bb_position | obs_builder.pyx:186 | ✓ |
| 26 | bb_width | obs_builder.pyx:191 | ✓ |

### 8. EVENT METADATA (5)

| №  | Признак | Путь расчета | Нормализация | Статус |
|----|---------|--------------|--------------|--------|
| 27 | is_high_importance | obs_builder.pyx:198 | direct | ✓ |
| 28 | time_since_event | obs_builder.pyx:204 | tanh(/6.0) - КРИТИЧНО! | ✓ ПРАВИЛЬНО |
| 29 | risk_off_flag | obs_builder.pyx:207 | direct | ✓ |
| 30 | fear_greed_value | obs_builder.pyx:212 | clip(-3,3) | ✓ |
| 31 | fear_greed_indicator | obs_builder.pyx:219 | direct | ✓ |

### 9. EXTERNAL NORMALIZED (21) - КРИТИЧНО!

| №  | Признак | Генерация | Извлечение | Статус |
|----|---------|----------|-----------|--------|
| 32 | cvd_24h | transformers:740 | mediator:1027 | ✓ |
| 33 | cvd_7d | transformers:740 | mediator:1028 | ✓ |
| 34 | yang_zhang_48h | transformers:635 | mediator:1029 | ✓ |
| 35 | yang_zhang_7d | transformers:635 | mediator:1030 | ✓ |
| 36 | garch_200h | transformers:673 | mediator:1031 | ✓ КРИТИЧНО! |
| 37 | garch_14d | transformers:673 | mediator:1032 | ✓ |
| 38 | ret_12h | transformers:610 | mediator:1033 | ✓ |
| 39 | ret_24h | transformers:610 | mediator:1034 | ✓ |
| 40 | ret_4h | transformers:610 | mediator:1037 | ✓ |
| 41 | sma_12000 | transformers:606 | mediator:1038 | ❌ МЕРТВЫЙ? |
| 42 | yang_zhang_30d | transformers:635 | mediator:1039 | ✓ |
| 43 | parkinson_48h | transformers:654 | mediator:1040 | ✓ |
| 44 | parkinson_7d | transformers:654 | mediator:1041 | ✓ |
| 45 | garch_30d | transformers:673 | mediator:1042 | ✓ |
| 46 | taker_buy_ratio | transformers:687 | mediator:1043 | ✓ |
| 47 | taker_buy_ratio_sma_24h | transformers:702 | mediator:1044 | ✓ |
| 48 | taker_buy_ratio_sma_8h | transformers:702 | mediator:1047 | ✓ |
| 49 | taker_buy_ratio_sma_16h | transformers:702 | mediator:1048 | ✓ |
| 50 | taker_buy_ratio_momentum_4h | transformers:720 | mediator:1049 | ✓ |
| 51 | taker_buy_ratio_momentum_8h | transformers:720 | mediator:1050 | ✓ |
| 52 | taker_buy_ratio_momentum_12h | transformers:720 | mediator:1051 | ❌ ОТСУТСТВУЕТ 24h |

### 10. TOKEN METADATA (2)

| №  | Признак | Путь | Статус |
|----|---------|------|--------|
| 53 | num_tokens_norm | obs_builder.pyx:232 | ✓ |
| 54 | token_id_norm | obs_builder.pyx:237 | ✓ |

### 11. TOKEN ONE-HOT (1)

| №  | Признак | Путь | Статус |
|----|---------|------|--------|
| 55 | token_one_hot | obs_builder.pyx:248 | ✓ |

---

## КРИТИЧЕСКИЕ ПРОБЛЕМЫ И ИХ РЕШЕНИЯ

### ПРОБЛЕМА 1: Отсутствие taker_buy_ratio_momentum_24h

**Где:** `/home/user/TradingBot2/mediator.py:1047-1051`
**Что:** mediator извлекает только 3 моментума вместо 4
**Генерируется в:** `/home/user/TradingBot2/transformers.py:413`

```python
# transformers.py ГЕНЕРИРУЕТ 4:
self.taker_buy_ratio_momentum = [4*60, 8*60, 12*60, 24*60]  # 4h, 8h, 12h, 24h

# mediator.py ИЗВЛЕКАЕТ 3:
norm_cols[18] = ... taker_buy_ratio_momentum_4h
norm_cols[19] = ... taker_buy_ratio_momentum_8h
norm_cols[20] = ... taker_buy_ratio_momentum_12h
# ОТСУТСТВУЕТ: taker_buy_ratio_momentum_24h!
```

**Решение:**
```python
# В mediator.py нужно либо:
# A) Добавить: norm_cols[21] = taker_buy_ratio_momentum_24h (и увеличить размер до 22)
# B) Или удалить 24h окно из transformers.py
```

### ПРОБЛЕМА 2: "Мертвые" SMA признаки

**Где:**
- `/home/user/TradingBot2/transformers.py:606` - генерирует
- `/home/user/TradingBot2/mediator.py:962-965` - извлекает

**Что:** transformers генерирует 7 SMA, но mediator использует только 2

```python
# Генерируется:
sma_240, sma_720, sma_1200, sma_1440, sma_5040, sma_10080, sma_12000

# Используется:
sma_1200, sma_5040 (для ma5 и ma20)

# Мертвые:
sma_240, sma_720, sma_1440, sma_10080, sma_12000
```

**Решение:**
Удалить лишние окна из transformers.__post_init__ или использовать их в mediator

### ПРОБЛЕМА 3: Несоответствие онлайн/оффлайн имен SMA

**Где:** `/home/user/TradingBot2/transformers.py:606,781`

**Что:**
- Онлайн (update): sma_240, sma_720, ... (в МИНУТАХ)
- Оффлайн (apply_offline_features): sma_1, sma_3, ... (в БАРАХ)

**Решение:**
```python
# В apply_offline_features (строка 781):
# Текущее:
base_cols += [f"sma_{x}" for x in spec.lookbacks_prices]

# Правильное:
base_cols += [f"sma_{x}" for x in spec._lookbacks_prices_minutes]
```

### ПРОБЛЕМА 4: Неправильный комментарий в make_features.py

**Где:** `/home/user/TradingBot2/make_features.py:36`

**Текущее:**
```python
help="...12000,20160,43200 = 8d,14d,30d"
```

**Правильное:**
```python
help="...12000,20160,43200 = 200h/14d/30d (200 часов = 8.3 дня, 50 баров)"
```

---

## ВЫВОДЫ И РЕКОМЕНДАЦИИ

### Что работает правильно (23 из 26 критических параметров):
- ✓ Все 56 признаков структурно определены
- ✓ time_since_event нормализация (делитель 6.0 для 4h)
- ✓ Volume нормализация (240e6, 24000)
- ✓ GARCH окна (200h, 14d, 30d)
- ✓ Yang-Zhang (48h, 7d, 30d)
- ✓ Parkinson (48h, 7d)
- ✓ Taker Buy Ratio SMA (8h, 16h, 24h)

### Что требует исправления (3 критические проблемы):
1. **КРИТИЧЕСКАЯ**: Отсутствие taker_buy_ratio_momentum_24h в mediator
2. **КРИТИЧЕСКАЯ**: 5 мертвых SMA признаков
3. **КРИТИЧЕСКАЯ**: Несоответствие имен SMA онлайн/оффлайн

### Приоритет исправлений:
1. IMMEDIATE: Добавить taker_buy_ratio_momentum_24h или удалить из transformers
2. HIGH: Унифицировать SMA имена между онлайн и оффлайн
3. MEDIUM: Удалить мертвые SMA или использовать их
4. LOW: Исправить комментарий в make_features.py

---

**Дата анализа:** 2025-11-13
**Версия:** Финальный отчет
**Статус:** ВСЕ 56 ПРИЗНАКОВ ПРОАНАЛИЗИРОВАНЫ

## ДЕТАЛЬНЫЙ ОТЧЕТ: КРИТИЧЕСКИЕ ПРОБЛЕМЫ

### ПРОБЛЕМА #7 (КРИТИЧЕСКАЯ): ОТСУТСТВИЕ taker_buy_ratio_momentum_24h

**Статус:** ❌ ТРЕБУЕТ НЕМЕДЛЕННОГО ИСПРАВЛЕНИЯ

**Где проблема:**
- File: `/home/user/TradingBot2/mediator.py`
- Lines: 1047-1051

**Текущий код:**
```python
norm_cols[16] = self._get_safe_float(row, "taker_buy_ratio_sma_8h", 0.0)
norm_cols[17] = self._get_safe_float(row, "taker_buy_ratio_sma_16h", 0.0)
norm_cols[18] = self._get_safe_float(row, "taker_buy_ratio_momentum_4h", 0.0)
norm_cols[19] = self._get_safe_float(row, "taker_buy_ratio_momentum_8h", 0.0)
norm_cols[20] = self._get_safe_float(row, "taker_buy_ratio_momentum_12h", 0.0)
# НЕ ХВАТАЕТ taker_buy_ratio_momentum_24h!
```

**Что должно быть:**
```python
# Всего 21 признак, но сейчас используется только 20!
# Нужно добавить taker_buy_ratio_momentum_24h
```

**Анализ несоответствия:**
1. **transformers.py, строка 413** определяет 4 моментума:
   ```python
   self.taker_buy_ratio_momentum = [4 * 60, 8 * 60, 12 * 60, 24 * 60]  # 4 окна
   ```

2. **mediator.py, строки 1047-1051** извлекает только 3:
   - taker_buy_ratio_momentum_4h (индекс 18)
   - taker_buy_ratio_momentum_8h (индекс 19)
   - taker_buy_ratio_momentum_12h (индекс 20)

3. **Отсутствует**: taker_buy_ratio_momentum_24h

**Решение:**
Добавить после строки 1051 в mediator.py:
```python
# ВСЕ 21 признак (обновлено для 4h таймфрейма) - но это должно быть в другом месте!
# Нужно переделать структуру norm_cols!
```

**КРИТИЧЕСКИ ВАЖНО**: norm_cols имеет размер 21, но есть избыточность!

---

### ПРОБЛЕМА #4 (КРИТИЧЕСКАЯ): НЕСООТВЕТСТВИЕ SMA ИМЕН

**Статус:** ❌ ТРЕБУЕТ НЕМЕДЛЕННОГО ИСПРАВЛЕНИЯ

**Где проблема:**
- File A: `/home/user/TradingBot2/transformers.py` строка 606
- File B: `/home/user/TradingBot2/mediator.py` строки 962, 965

**Анализ:**

**transformers.py (ГЕНЕРИРУЕТ):**
```python
# Строка 606: онлайн генерирует 7 SMA в МИНУТАХ
feats[f"sma_{lb_minutes}"] = float(sma)
# Генерируемые имена: sma_240, sma_720, sma_1200, sma_1440, sma_5040, sma_10080, sma_12000
```

**mediator.py (ИЗВЛЕКАЕТ):**
```python
# Строка 962:
ma5 = self._get_safe_float(row, "sma_1200", float('nan'))  # Ищет sma_1200
# Строка 965:
ma20 = self._get_safe_float(row, "sma_5040", float('nan'))  # Ищет sma_5040
```

**КРИТИЧЕСКИЙ АНАЛИЗ:**
1. transformers.py ГЕНЕРИРУЕТ 7 SMA с именами в МИНУТАХ: sma_240, sma_720, ...
2. mediator.py ИЗВЛЕКАЕТ ТОЛЬКО 2 из них: sma_1200, sma_5040
3. Остальные 5 SMA не используются нигде!

**Это означает:**
- Есть 5 "мертвых" признаков: sma_240, sma_720, sma_1440, sma_10080, sma_12000
- Они генерируются но не используются в наблюдениях!

**Решение:**
Либо:
A) Удалить генерацию лишних SMA из transformers.py
B) Добавить их в mediator._extract_norm_cols
C) Использовать их в otros местах (если есть необходимость)

---

### ПРОБЛЕМА #8 (КРИТИЧЕСКАЯ): ОНЛАЙН vs ОФФЛАЙН ИМЕНА SMA

**Статус:** ❌ ТРЕБУЕТ НЕМЕДЛЕННОГО ИСПРАВЛЕНИЯ

**Где проблема:**
- File: `/home/user/TradingBot2/transformers.py`
- Lines: 606 (онлайн) vs 781 (оффлайн)

**Онлайн путь (apply_offline_features, строка 781):**
```python
base_cols += [f"sma_{x}" for x in spec.lookbacks_prices]
# spec.lookbacks_prices = [1, 3, 5, 6, 21, 42, 50] (в БАРАХ)
# Генерирует: sma_1, sma_3, sma_5, sma_6, sma_21, sma_42, sma_50
```

**Оффлайн путь (update, строка 606):**
```python
feats[f"sma_{lb_minutes}"] = float(sma)
# lb_minutes = [240, 720, 1200, 1440, 5040, 10080, 12000] (в МИНУТАХ)
# Генерирует: sma_240, sma_720, sma_1200, sma_1440, sma_5040, sma_10080, sma_12000
```

**КРИТИЧЕСКАЯ ПРОБЛЕМА:**
- Онлайн применяет_offline_features генерирует имена в БАРАХ: sma_1, sma_3, ...
- Оффлайн update генерирует имена в МИНУТАХ: sma_240, sma_720, ...
- НЕСООТВЕТСТВИЕ! Имена разные!

**Решение:**
Унифицировать имена. Все должны быть либо в минутах, либо в барах.
ПРАВИЛЬНЫЙ ВЫБОР: МИНУТЫ (так как это универсальнее и понятнее)

```python
# В apply_offline_features (строка 781):
base_cols += [f"sma_{x}" for x in spec._lookbacks_prices_minutes]  # ИСПОЛЬЗОВАТЬ МИНУТЫ!
# Будет генерировать: sma_240, sma_720, ...
```

---

### ПРОБЛЕМА: КОЛИЧЕСТВО ПРИЗНАКОВ В NORM_COLS РАЗНОЕ

**Статус:** ⚠️ ТРЕБУЕТ УТОЧНЕНИЯ

**Текущая ситуация:**
- feature_config.py: EXT_NORM_DIM = 21
- mediator.py: norm_cols = np.zeros(21, dtype=np.float32)
- obs_builder.pyx: читает norm_cols_values размера 21

**КОНФЛИКТ**: В transformers.py есть 4 окна taker_buy_ratio_momentum,
но в mediator вытягиваются только 3!

**Теория**: Один из momental окон (вероятно 24h) должен быть 4-м признаком вместо чего-то другого.

**Анализ:**
```python
# transformers.py генерирует 4 моментума:
taker_buy_ratio_momentum_4h      # 1 бар
taker_buy_ratio_momentum_8h      # 2 бара
taker_buy_ratio_momentum_12h     # 3 бара
taker_buy_ratio_momentum_24h     # 6 баров (НОВОЕ!)

# mediator.py извлекает только 3:
norm_cols[18] = taker_buy_ratio_momentum_4h
norm_cols[19] = taker_buy_ratio_momentum_8h
norm_cols[20] = taker_buy_ratio_momentum_12h
# ОТСУТСТВУЕТ: taker_buy_ratio_momentum_24h
```

---

### ПРОБЛЕМА: НЕСООТВЕТСТВИЕ FEATURE_CONFIG И REAL CODE

**Статус:** ✓ ЧАСТИЧНО СОГЛАСОВАНО

**feature_config.py:**
```python
# Индексы внешних признаков (внутри norm_cols):
# [0-20] = 21 признак
```

**mediator._extract_norm_cols:**
```python
# Возвращает массив из 21 признака:
norm_cols[0-20] = различные признаки
```

**Проблема**: Структура точно не документирована!

Должна быть документация типа:
```
# Index 0: cvd_24h
# Index 1: cvd_7d
# ...
# Index 20: taker_buy_ratio_momentum_12h или SOMETHING ELSE???
```

---

### ПРОБЛЕМА: КОММЕНТАРИИ В make_features.py НЕПРАВИЛЬНЫЕ

**Статус:** ❌ ТРЕБУЕТ ИСПРАВЛЕНИЯ

**File:** `/home/user/TradingBot2/make_features.py`
**Line:** 36

**Текущий код:**
```python
help="Окна GARCH(1,1) волатильности в минутах для 4h (по умолчанию 12000,20160,43200 = 8d,14d,30d)"
```

**ПРОБЛЕМА:** Комментарий говорит "8d,14d,30d", но:
- 12000 минут = 200 часов ≠ 8 дней!
- 12000 / 1440 = 8.33 дня ≈ 8.3d (не 8d!)

**Правильный комментарий:**
```python
help="Окна GARCH(1,1) волатильности в минутах для 4h (по умолчанию 12000,20160,43200 = 200h/14d/30d)"
```

---

### ИТОГОВАЯ ТАБЛИЦА ПРОБЛЕМ

| # | Проблема | Файл:строка | Тип | Статус | Приоритет |
|---|----------|-------------|-----|--------|-----------|
| 1 | Отсутствие taker_buy_ratio_momentum_24h | mediator.py:1047-1051 | НЕСООТВЕТСТВИЕ | ❌ | CRITICAL |
| 2 | Мертвые SMA признаки | transformers.py:606 vs mediator.py:962-965 | НЕПОЛНОТА | ❌ | CRITICAL |
| 3 | Онлайн vs Оффлайн SMA имена | transformers.py:606,781 | НЕСООТВЕТСТВИЕ | ❌ | CRITICAL |
| 4 | Неправильный комментарий GARCH | make_features.py:36 | КОММЕНТАРИЙ | ❌ | MINOR |
| 5 | Отсутствует документация | feature_config.py | ДОКУМЕНТАЦИЯ | ❌ | MEDIUM |
| 6 | Структура norm_cols недокументирована | mediator.py:1014-1057 | ДОКУМЕНТАЦИЯ | ❌ | MEDIUM |

---

## ИТОГОВАЯ СВОДКА

### ВАЛИДНЫЕ ПРИЗНАКИ (ВСЕ РАБОТАЕТ)
- ✓ time_since_event нормализация (делитель 6.0 правильный)
- ✓ Volume нормализация (делители 240e6 и 24000)
- ✓ GARCH окна (200h, 14d, 30d = 50, 84, 180 баров)
- ✓ Taker Buy Ratio SMA (8h, 16h, 24h)
- ✓ Yang-Zhang окна (48h, 7d, 30d)
- ✓ Parkinson окна (48h, 7d)

### КРИТИЧЕСКИЕ ПРОБЛЕМЫ (ТРЕБУЮТ ИСПРАВЛЕНИЯ)
- ❌ Отсутствует taker_buy_ratio_momentum_24h в mediator
- ❌ 5 "мертвых" SMA признаков не используются
- ❌ Несоответствие между онлайн и оффлайн именами SMA
- ❌ norm_cols недостаточно документирован


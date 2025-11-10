# Полный список всех признаков в TradingBot2

## Откуда взялось число 57?

### Расчёт старой (ОШИБОЧНОЙ) конфигурации:

**1. feature_config.py (старая версия с MAX_NUM_TOKENS=16):**

Блоки в make_layout():
- bar: 3 фичи (price, log_volume_norm, rel_volume)
- derived: 2 фичи (ret_1h, vol_proxy)
- indicators: 13 фич (ma5 + valid, ma20 + valid, rsi, macd, macd_signal, momentum, atr, cci, obv)
- microstructure: 3 фичи (ofi_proxy, qimb, micro_dev)
- agent: 6 фич (cash_ratio, position_ratio, vol_imbalance, trade_intensity, realized_spread, agent_fill_ratio)
- metadata: 2 фичи (is_high_importance, time_since_event) ← **НЕДОУЧТЕНО!**
- external: 8 фич (norm_cols для cvd, garch, yang_zhang, returns)
- token: 16 фич (MAX_NUM_TOKENS=16) ← **ПРОБЛЕМА!**

**ИТОГО: 3 + 2 + 13 + 3 + 6 + 2 + 8 + 16 = 53**

**2. trading_patchnew.py добавлял загадочный +4:**

```python
observation_space = spaces.Box(
    low=-np.inf, high=np.inf, shape=(N_FEATURES + 4,), dtype=np.float32
)
```

**53 + 4 = 57**

Откуда +4? Вероятно legacy код для: units, cash, signal_pos, log_ret_prev
**НО эти поля уже были в других блоках!**

**3. Что РЕАЛЬНО заполнялось obs_builder:**

С параметрами:
- max_num_tokens = 1 (НЕ 16!)
- norm_cols = np.zeros(8)

Блоки:
- Bar: 3
- MA5: 2
- MA20: 2
- Technical: 7
- Derived: 2
- Agent: 6
- Microstructure: 3
- Bollinger: 2
- Event metadata: 3 (is_high_importance, time_since_event, risk_off_flag)
- Fear & Greed: 2 (fear_greed_value, fear_greed_indicator)
- norm_cols (external): 8
- Token metadata: 2 (num_tokens_norm, token_id_norm)
- Token one-hot: 1 (max_num_tokens=1)

**ИТОГО: 3+2+2+7+2+6+3+2+3+2+8+2+1 = 43**

**4. НЕСООТВЕТСТВИЕ:**

- ❌ Объявлено: observation_space.shape = (57,)
- ❌ Заполнено: 43 позиции
- ❌ Пустых: 14 позиций (всегда нули!)

---

## Полный список всех признаков в коде

### 📊 ПРИЗНАКИ из prepare_and_run.py (apply_offline_features)

**Создаваемые в prepare_and_run.py (~30 признаков):**

#### 1. SMA (Simple Moving Averages)
- `sma_5` (5-периодная)
- `sma_15` (15-периодная)
- `sma_60` (60-периодная)

#### 2. Returns (Логарифмические доходности)
- `ret_5m` (5-минутная)
- `ret_15m` (15-минутная)
- `ret_60m` (60-минутная)

#### 3. RSI (Relative Strength Index)
- `rsi` (14-периодный по Wilder)

#### 4. Yang-Zhang Volatility
- `yang_zhang_24h` (24-часовая волатильность)
- `yang_zhang_168h` (168-часовая / недельная)
- `yang_zhang_720h` (720-часовая / месячная)

#### 5. Parkinson Volatility
- `parkinson_24h` (24-часовая)
- `parkinson_168h` (168-часовая)

#### 6. GARCH Volatility
- `garch_500m` (500-минутная)
- `garch_12h` (12-часовая)
- `garch_24h` (24-часовая)

#### 7. Taker Buy Ratio (базовый)
- `taker_buy_ratio` (соотношение покупок агрессора)

#### 8. Taker Buy Ratio SMA
- `taker_buy_ratio_sma_6h` (6-часовое среднее)
- `taker_buy_ratio_sma_12h` (12-часовое)
- `taker_buy_ratio_sma_24h` (24-часовое)

#### 9. Taker Buy Ratio Momentum
- `taker_buy_ratio_momentum_1h` (1-часовой моментум)
- `taker_buy_ratio_momentum_6h` (6-часовой)
- `taker_buy_ratio_momentum_12h` (12-часовой)

#### 10. CVD (Cumulative Volume Delta)
- `cvd_24h` (24-часовая кумулятивная дельта)
- `cvd_168h` (168-часовая / недельная)

---

### 📈 ПРИЗНАКИ в OBSERVATION VECTOR (obs_builder.pyx) - 43 позиции

#### Позиции 0-2: Bar-Level (3)
- 0: `price`
- 1: `log_volume_norm`
- 2: `rel_volume`

#### Позиции 3-6: Moving Averages (4)
- 3: `ma5` (из sma_5)
- 4: `ma5_valid`
- 5: `ma20` (из sma_15)
- 6: `ma20_valid`

#### Позиции 7-13: Technical Indicators (7)
- 7: `rsi14` (из rsi)
- 8: `macd`
- 9: `macd_signal`
- 10: `momentum`
- 11: `atr`
- 12: `cci`
- 13: `obv`

#### Позиции 14-15: Derived (2)
- 14: `ret_1h` (вычисленный)
- 15: `vol_proxy` (вычисленный)

#### Позиции 16-21: Agent State (6)
- 16: `cash_ratio`
- 17: `position_ratio`
- 18: `vol_imbalance`
- 19: `trade_intensity`
- 20: `realized_spread`
- 21: `agent_fill_ratio`

#### Позиции 22-24: Microstructure (3)
- 22: `ofi_proxy`
- 23: `qimb`
- 24: `micro_dev`

#### Позиции 25-26: Bollinger Bands (2)
- 25: `bb_position`
- 26: `bb_width`

#### Позиции 27-29: Event Metadata (3)
- 27: `is_high_importance`
- 28: `time_since_event`
- 29: `risk_off_flag`

#### Позиции 30-31: Fear & Greed (2)
- 30: `fear_greed_value`
- 31: `fear_greed_indicator`

#### Позиции 32-39: External (norm_cols) - ГЛАВНЫЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ! (8)
- 32: **`cvd_24h`** ← из prepare_and_run.py
- 33: **`cvd_168h`** ← из prepare_and_run.py
- 34: **`yang_zhang_24h`** ← из prepare_and_run.py
- 35: **`yang_zhang_168h`** ← из prepare_and_run.py
- 36: **`garch_12h`** ← из prepare_and_run.py
- 37: **`garch_24h`** ← из prepare_and_run.py
- 38: **`ret_15m`** ← из prepare_and_run.py
- 39: **`ret_60m`** ← из prepare_and_run.py

#### Позиции 40-41: Token Metadata (2)
- 40: `num_tokens_norm`
- 41: `token_id_norm`

#### Позиция 42: Token One-Hot (1)
- 42: `token_0`

---

## ❌ Какие признаки НЕ используются?

### НЕ ПОПАЛИ В OBSERVATION (но создаются в prepare_and_run.py):

1. **`sma_60`** — создаётся, но не используется (используются только sma_5 и sma_15)

2. **`ret_5m`** — создаётся, но не используется (используются только ret_15m и ret_60m)

3. **`yang_zhang_720h`** — создаётся, но не используется (используются только yang_zhang_24h и yang_zhang_168h)

4. **`parkinson_24h`, `parkinson_168h`** — создаются, но не используются

5. **`garch_500m`** — создаётся, но не используется (используются только garch_12h и garch_24h)

6. **Taker Buy Ratio и все его производные** — создаются, но не используются:
   - `taker_buy_ratio`
   - `taker_buy_ratio_sma_6h`, `taker_buy_ratio_sma_12h`, `taker_buy_ratio_sma_24h`
   - `taker_buy_ratio_momentum_1h`, `taker_buy_ratio_momentum_6h`, `taker_buy_ratio_momentum_12h`

**ИТОГО НЕ ИСПОЛЬЗУЕТСЯ: ~13 признаков**

---

## 📊 Итоговая статистика

- **Создаётся** в prepare_and_run.py: **~30 признаков**
- **Используется** в observation: **43 позиции**
  - из них из prepare_and_run.py: **~12 признаков**
  - остальные: вычисляемые + состояние агента + metadata
- **НЕ используется**: **~13 признаков** (создаются, но игнорируются)

### ✅ ИСПОЛЬЗУЕМЫЕ признаки из prepare_and_run.py:
- `sma_5`, `sma_15`
- `rsi`
- `cvd_24h`, `cvd_168h`
- `yang_zhang_24h`, `yang_zhang_168h`
- `garch_12h`, `garch_24h`
- `ret_15m`, `ret_60m`
- `fear_greed_value` (если есть)

---

## 🎯 Ответ на вопрос "Откуда 57?"

**57 = 53 (feature_config с MAX_NUM_TOKENS=16) + 4 (загадочный legacy +4)**

**НО реально заполнялось только 43 позиции!**

### Причины несоответствия:
1. `MAX_NUM_TOKENS` был 16 в feature_config, но использовался только 1
2. `metadata` был размер 2 в feature_config, но заполнялось 5 полей
3. Загадочный +4 в observation_space
4. Отсутствие token_meta блока в feature_config

### После исправления:
✅ **Всё стало 43 — идеальное соответствие между объявлением и реализацией!**

---

## 🔍 Возможные улучшения

Если хочешь использовать больше признаков из prepare_and_run.py, можно:

1. **Добавить taker_buy_ratio** признаки в norm_cols (расширить с 8 до 16)
2. **Добавить parkinson volatility** в norm_cols
3. **Использовать sma_60** и ret_5m
4. **Добавить yang_zhang_720h** (месячная волатильность)

Но текущие 43 признака уже включают **самые важные** индикаторы для алготрейдинга!

---

**Last Updated**: 2025-01-10
**Status**: Актуальная информация после исправления размера observation с 57 на 43

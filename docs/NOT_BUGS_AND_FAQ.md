# Closed Questions & Not-Bugs Reference

> These are code patterns that LOOK like bugs under static analysis but are CORRECT and intentional, plus previously-investigated questions that are settled.
>
> Before "fixing" something in these areas, read the relevant entry here first — these are settled questions, not open defects.

---

## ✅ FAQ: Закрытые вопросы (НЕ ПЕРЕОТКРЫВАТЬ!)

Эти вопросы были тщательно проанализированы. Подробности: archive/2025_11/reports_2025_11_25_cleanup/root_reports/CONCEPTUAL_ANALYSIS_REPORT_2025_11_24.md

| Вопрос | Ответ |
|--------|-------|
| "Look-ahead bias в индикаторах?" | ✅ **Исправлено 2025-11-23**. Все фичи сдвинуты. |
| "VGS недооценивает variance в N раз?" | ⚠️ **By design**. Var[mean(g)] валиден, работает в production. |
| "-10.0 bankruptcy penalty слишком резкий?" | ✅ **Стандартная практика RL**. Potential shaping даёт smooth gradient. |
| "_last_signal_position двойное присваивание?" | ⚠️ **Удалено 2025-11-25**. Было избыточно, но не баг (значения идентичны). |
| "Первые 2 steps в CLOSE_TO_OPEN reward=0?" | ⚠️ **By design**. Delayed execution: reward × prev_signal_pos, где prev=0 для первых шагов. |
| "signal_only terminated всегда False?" | ⚠️ **By design**. В signal_only нет капитала в риске, банкротство не имеет смысла. |
| "ActionProto double mapping в LongOnlyActionWrapper?" | ⚠️ **НЕ баг**. API контракт: input [-1,1] → output [0,1]. Если передать [0,1] - нарушение контракта. |
| "adaptive_upgd.py grad_norm_ema=1.0 warmup?" | ⚠️ **НЕ баг**. Default `instant_noise_scale=True` bypasses EMA. См. #28. |
| "info[signal_pos] разная семантика?" | ⚠️ **By design**. signal_only: prev (для reward), normal: next (после execution). См. #7. |
| "mediator norm_cols_validity=True?" | ⚠️ **НЕ баг**. Начальное значение полностью перезаписывается в цикле. См. #29. |
| "mediator empty observation silent fail?" | ⚠️ **НЕ баг**. Defensive check для edge cases без observation_space. |
| "mediator race condition signal_pos?" | ⚠️ **НЕ баг**. Single-threaded архитектура, нет параллелизма. |
| "risk_guard асимметричный buffer?" | ⚠️ **By design**. Buffer только на увеличение позиции (корректный risk mgmt). См. #30. |
| "ops_kill_switch cooldown reset при init?" | ⚠️ **НЕ баг**. _last_ts=0.0 = "reset в epoch". Логика корректна. См. #31. |
| "RSI valid на 1 бар раньше (off-by-one)?" | ⚠️ **НЕ баг**. RSI-14 valid на bar 14 (после 14 price changes). Timing корректен. См. #32. |
| "obs_builder vol_proxy=0.01 constant warmup?" | ⚠️ **By design**. 1% price fallback лучше чем NaN или 0. См. #33. |
| "obs_builder FG=50 vs missing неразличимы?" | ✅ **Исправлено 2025-11-26**. Теперь `_get_safe_float_with_validity()` различает. |
| "policy sigma range [0.2,1.5] не адаптируется?" | ⚠️ **НЕ баг**. Standard PPO range для continuous actions. См. #35. |
| "CVaR weight_start=0.5 совпадение?" | ⚠️ **НЕ баг**. Математически корректно: граница = midpoint. См. #3. |
| "features_pipeline constant на shifted data?" | ⚠️ **НЕ баг**. nanstd игнорирует NaN, для типичных datasets работает. См. #36. |
| "mediator step_idx=current не next?" | ⚠️ **Minor**. info для logging, не для agent. Семантика "обработали row X". |
| "Twin Critics logging memory leak?" | ⚠️ **НЕ баг**. Accumulators reset at line 12288 after logging. См. #45. |
| "ddof=1 vs ddof=0 в advantage normalization?" | ⚠️ **Minor inconsistency**. SB3 uses ddof=0, difference <0.1% for n>1000. См. #46. |
| "VGS race condition в PBT?" | ⚠️ **НЕ issue**. Separate workers, unique checkpoint files, Python GIL. См. #47. |
| "CVaR ~16% approximation error?" | ⚠️ **Documented limitation**. Trade-off: speed vs accuracy. N=51 gives ~5% error. |
| "Winsorization [1%,99%] insufficient for crypto?" | ⚠️ **Configurable**. Can adjust in features_pipeline.py:181. |
| "tanh в potential shaping нарушает Ng theorem?" | ⚠️ **НЕ баг**. Ng et al. (1999) разрешает ЛЮБУЮ функцию Φ(s). tanh(net_worth) валиден. |
| "gap_filled look-ahead bias?" | ⚠️ **НЕ баг**. Feature shifting (shift(1)) применяется ПОСЛЕ вычисления. См. features_pipeline.py:441-442. |
| "Earnings unbounded future window?" | ⚠️ **Документация**. Пользователь обязан гарантировать актуальность earnings calendar. Не code bug. |
| "γ не синхронизирован между env и model?" | ⚠️ **Documented**. [PLATFORM_REFERENCE.md](PLATFORM_REFERENCE.md): "reward.gamma == model.params.gamma (оба = 0.99)". Конфигурационная ответственность пользователя. |
| "3 уровня reward clipping создают non-monotonic value?" | ⚠️ **НЕ баг**. Разные клипы: (1) ratio→log safety, (2) final bounds. Служат разным целям. См. #59. |
| "Long-only reward=0 при pos=0 асимметричен?" | ⚠️ **By design**. `reward = log(ratio) × position`. При pos=0 агент не участвовал → reward=0 корректен. |
| "L2 ADV не учитывает intraday seasonality?" | ⚠️ **By design**. L2 simple/fast; L2+ has `tod_curve`. См. #54. |
| "L2 нет temp/perm impact separation?" | ⚠️ **By design**. L2=√participation; L3 has AlmgrenChriss/Gatheral. См. #55. |
| "L2 spread статичен?" | ⚠️ **By design**. L2+ has vol_regime_multipliers. См. #56. |
| "L2 limit fills детерминистичны?" | ⚠️ **By design**. L2=binary; L3 has QueueReactiveModel. См. #57. |
| "whale_threshold не масштабируется по ADV?" | ⚠️ **Configurable**. Threshold = participation ratio (уже normalized). Config profiles exist. См. #58. |

---

## 🔬 НЕ БАГИ: Корректные паттерны кода (НЕ "ИСПРАВЛЯТЬ"!)

> **ВАЖНО**: Следующие паттерны кода ВЫГЛЯДЯТ как ошибки при статическом анализе, но являются **корректными и намеренными**. НЕ пытайтесь их "исправить"!

### 1. Episode Starts Off-by-One (distributional_ppo.py:8314, 8347)

```python
# Строка 8314: добавляем _last_episode_starts в буфер
rollout_buffer.add(..., self._last_episode_starts, ...)

# Строка 8347: обновляем ПОСЛЕ добавления
self._last_episode_starts = dones
```

**Почему это НЕ баг**: Это стандартный паттерн Stable-Baselines3. `_last_episode_starts` хранит `dones` от **предыдущего** шага. При вычислении GAE (строка 280) используется `episode_starts[step+1]` -- это означает "был ли шаг step терминальным". Сдвиг на 1 **намеренный** и семантически корректный.

**Референс**: SB3 `OnPolicyAlgorithm.collect_rollouts()`, PPO paper (Schulman et al., 2017)

---

### 2. VGS применяется ПЕРЕД grad clipping (distributional_ppo.py:11664-11676)

```python
# Строка 11664: VGS масштабирует градиенты
vgs_scaling_factor = self._variance_gradient_scaler.scale_gradients()

# Строка 11676: Потом clipping
total_grad_norm = torch.nn.utils.clip_grad_norm_(...)
```

**Почему это НЕ баг**: VGS **уменьшает** градиенты (scaling_factor < 1.0, см. variance_gradient_scaler.py:446). Порядок корректен:

1. VGS снижает variance высокошумных градиентов
2. clip_grad_norm защищает от оставшихся выбросов

**Референс**: variance_gradient_scaler.py docstring, Adam optimizer design

---

### 3. CVaR Interpolation Weight = 0.5 (distributional_ppo.py:3726-3728)

```python
tau_i_prev = (alpha_idx - 0.5) / num_quantiles  # центр предыдущего интервала
tau_i = (alpha_idx + 0.5) / num_quantiles        # центр текущего интервала
interval_start = alpha_idx / num_quantiles       # граница между ними
weight_start = (interval_start - tau_i_prev) / (tau_i - tau_i_prev)  # = 0.5
```

**Почему это НЕ баг**: `interval_start` (граница квантильного интервала) находится **ровно посередине** между центрами соседних интервалов `tau_i_prev` и `tau_i`. Вес 0.5 -- это математически корректная линейная интерполяция.

**Математика**: `weight = (α_idx/N - (α_idx-0.5)/N) / ((α_idx+0.5)/N - (α_idx-0.5)/N) = 0.5/N / (1/N) = 0.5`

---

### 4. LSTM Init State Index 0 (distributional_ppo.py:2217)

```python
state_tensor[:, env_idx, ...] = init_tensor[:, 0, ...].detach().to(...)
```

**Почему это НЕ баг**: `recurrent_initial_state` инициализируется **нулями** для всех environments (custom_policy_patch1.py:492). Все init states идентичны, поэтому `init_tensor[:, 0, ...]` безопасен.

**Референс**: custom_policy_patch1.py:491-503 -- `torch.zeros(self.lstm_hidden_state_shape, ...)`

---

### 5. Twin Critics Loss Averaging БЕЗ VF Clipping (distributional_ppo.py:11073)

```python
# Когда VF clipping ВЫКЛЮЧЕН:
critic_loss_unclipped_per_sample = (loss_critic_1 + loss_critic_2) / 2.0
```

**Почему это НЕ баг**: Без VF clipping нет необходимости в `max(clipped, unclipped)`. Простое усреднение losses двух critics корректно. Когда VF clipping **включён**, используется правильная логика (строки 11168-11170):

```python
loss_c1_final = torch.max(loss_c1_unclipped, loss_c1_clipped)
loss_c2_final = torch.max(loss_c2_unclipped, loss_c2_clipped)
critic_loss = torch.mean((loss_c1_final + loss_c2_final) / 2.0)
```

---

### 6. close_orig vs_close_shifted маркеры (features_pipeline.py, trading_patchnew.py)

```python
# features_pipeline.py:329-331 -- пропускает shift если close_orig есть
if "close_orig" in frame.columns:
    shifted_frames.append(frame)
    continue

# trading_patchnew.py:305-307 -- проверяет close_orig ПЕРВЫМ
if "close_orig" in self.df.columns:
    self._close_actual = self.df["close_orig"].copy()
elif "close" in self.df.columns and "_close_shifted" not in self.df.columns:
    # Shift применяется только здесь
```

**Почему это НЕ баг**: Проверка `close_orig` идёт **раньше** проверки `_close_shifted`. Если данные пришли с `close_orig` (уже сдвинуты), shift НЕ применяется повторно. Два маркера имеют разную семантику:

- `close_orig` -- оригинальная цена ДО shift (для анализа)
- `_close_shifted` -- флаг что shift уже применён

---

### 7. info["signal_pos_next"] vs info["signal_pos_requested"] (trading_patchnew.py:2194-2204)

```python
if self._reward_signal_only:
    info["signal_pos_next"] = float(next_signal_pos)      # ACTUAL position after step
    info["signal_pos_requested"] = float(agent_signal_pos)  # Agent's INTENTION
else:
    info["signal_pos_next"] = float(next_signal_pos)
    info["signal_pos_requested"] = float(agent_signal_pos)
```

**Почему это корректно** (исправлено 2025-11-25):

1. В CLOSE_TO_OPEN режиме: `next_signal_pos ≠ agent_signal_pos` из-за 1-bar delay
2. `signal_pos_next` показывает **фактическую** позицию после шага (используется для reward)
3. `signal_pos_requested` показывает **намерение** агента (для debugging/анализа)
4. **До фикса**: `signal_pos_next = agent_signal_pos` → вводило в заблуждение при отладке

**Тесты**: `tests/test_signal_pos_next_close_to_open_consistency.py` (8 тестов)

---

### 8. Advantage Normalization с ddof=1 (distributional_ppo.py:8442)

```python
adv_std = float(np.std(advantages_flat, ddof=1))
# ...
normalized_advantages = (adv - adv_mean) / (adv_std + EPSILON)
```

**Почему это НЕ баг**:

1. `ddof=1` для несмещённой оценки дисперсии (Bessel's correction)
2. Если `n_samples == 1`, `std` будет `NaN`
3. Код защищён проверкой на строках 8444-8445: `if not np.isfinite(adv_std): skip`
4. `EPSILON = 1e-8` защищает от деления на ноль

---

### 9. Policy Adaptive Activation (custom_policy_patch1.py:491-497, 1301-1314)

```python
# __init__: определяем тип активации по action_space
action_low = float(self.action_space.low.flat[0])
self._use_tanh_activation = action_low < 0.0

# _apply_action_activation: выбираем sigmoid или tanh
if getattr(self, "_use_tanh_activation", False):
    return torch.tanh(raw)
else:
    return torch.sigmoid(raw)
```

**Почему это НЕ баг**: Это **КРИТИЧЕСКИЙ FIX** (2025-11-25):

1. `LongOnlyActionWrapper` устанавливает `action_space = [-1, 1]`
2. Policy детектирует это и использует `tanh` (выход [-1, 1])
3. Wrapper маппит [-1, 1] → [0, 1] для TradingEnv
4. БЕЗ этого фикса: sigmoid [0,1] → mapping → [0.5, 1.0] -- **минимум 50% позиции!**

**Тесты**: `tests/test_long_only_action_space_fix.py` (26 тестов)

---

### 10. step() Observation from NEXT Row (trading_patchnew.py:1007-1037, mediator.py:1724-1739)

```python
# Вычисляем индекс СЛЕДУЮЩЕЙ строки для observation
obs_row_idx = min(next_idx, len(self.df) - 1)
next_row = self.df.iloc[obs_row_idx]
obs = self._mediator._build_observation(row=next_row, state=state, mark_price=next_mark_price)
```

**Почему это КОРРЕКТНО** (исправлено 2025-11-25):

1. **Gymnasium семантика**: `step(a)` возвращает `(s_{t+1}, r_t, ...)` -- observation **после** действия
2. До фикса: reset() и step()#1 возвращали obs из одной строки (row[0]) -- дубликат!
3. После фикса: reset() → row[0], step()#1 → row[1], step()#2 → row[2]
4. Terminal case: при next_idx >= len(df), используется последняя доступная строка

**Влияние бага на training**:

- Sample efficiency: ~1% loss (1 бесполезный transition на эпизод)
- LSTM: первые два hidden state обновления от идентичного входа
- Первый step reward: всегда 0 (log(price[0]/price[0])=0)

**Тесты**: `tests/test_step_observation_next_row.py` (6 тестов)

---

### 11. CLOSE_TO_OPEN + SIGNAL_ONLY Delayed Position (trading_patchnew.py:1725-1756)

```python
if self.decision_mode == DecisionTiming.CLOSE_TO_OPEN:
    # Всегда уважаем 1-bar delay для signal position
    next_signal_pos = executed_signal_pos  # от delayed proto
else:
    next_signal_pos = agent_signal_pos if self._reward_signal_only else executed_signal_pos
```

**Почему это КОРРЕКТНО** (исправлено 2025-11-25):

1. **CLOSE_TO_OPEN семантика**: действие агента исполняется на **следующем** баре
2. До фикса: в SIGNAL_ONLY позиция обновлялась мгновенно → look-ahead bias
3. После фикса: даже в SIGNAL_ONLY режиме позиция задерживается на 1 бар
4. Reward = log(price_change) × position → позиция должна соответствовать реальному timing'у

**Влияние бага на training**:

- Training Sharpe: inflated на ~10-30% vs reality
- Look-ahead bias: reward за позицию, которой ещё нет
- Training/Live gap: увеличен из-за нереалистичных rewards

**Тесты**: `tests/test_close_to_open_signal_only_timing.py` (5 тестов)

---

### 12. Первые 2 step'а в CLOSE_TO_OPEN имеют reward ≈ 0 (trading_patchnew.py:1997-2015)

```python
# reward = log(price_ratio) × prev_signal_pos
# Step #1: prev_signal_pos = 0 (initial) → reward = 0
# Step #2: prev_signal_pos = 0 (delayed HOLD) → reward = 0
# Step #3+: prev_signal_pos = executed_action → reward ≠ 0
reward_raw_fraction = math.log(ratio_clipped) * prev_signal_pos
```

**Почему это BY DESIGN (НЕ баг)**:

1. **Физика delayed execution**: в CLOSE_TO_OPEN действие исполняется на **следующем** баре
2. При reset() устанавливается `_pending_action = HOLD(0.0)` -- первое действие
3. Step #1: prev_pos = 0 (initial), action = HOLD(0.0) → reward × 0 = 0
4. Step #2: prev_pos = 0 (от HOLD), action = A1 → reward × 0 = 0
5. Step #3: prev_pos = A1, reward × A1 ≠ 0

**Семантика**: Reward отражает позицию, которая **РЕАЛЬНО была** во время движения цены, а не намерение агента. Это корректно для реалистичного trading simulation.

**Влияние на training**:

- Короткие эпизоды (< 5 баров) получают мало ненулевых rewards
- ~2/N долевая потеря sample efficiency для N-bar эпизодов
- Это **НЕ влияет на качество обучения** -- агент учится правильной семантике

**Не пытайтесь "исправить"** -- это сломает корректность симуляции!

---

### 13. В signal_only режиме terminated всегда False (trading_patchnew.py:1067-1086)

```python
# is_bankrupt устанавливается ТОЛЬКО в mediator.step()
# В signal_only режиме mediator.step() НЕ вызывается
terminated = bool(getattr(state, "is_bankrupt", False))  # всегда False
```

**Почему это BY DESIGN (НЕ баг)**:

1. **Signal_only режим**: агент учится генерировать сигналы без реального execution
2. Нет реальных позиций → нет реального capital at risk → нет банкротства
3. Reward = log(price_change) × signal_position -- чисто сигнальный training
4. Эпизоды заканчиваются через **truncation** (`max_steps`), НЕ termination

**Альтернатива**: Добавить "виртуальное банкротство"?

- Это усложнит семантику без реальной пользы
- Сигнальный режим не симулирует капитал -- банкротство не имеет смысла
- Если нужна проверка drawdown → используйте real execution mode

**Не пытайтесь добавить виртуальное банкротство** -- это нарушит принцип signal_only!

---

### 14. ActionProto "double mapping" в LongOnlyActionWrapper (wrappers/action_space.py:120-147)

```python
# API контракт: INPUT [-1, 1] → OUTPUT [0, 1]
mapped = self._map_to_long_only(action.volume_frac)  # (x+1)/2
# -1.0 → 0.0, 0.0 → 0.5, 1.0 → 1.0
```

**Почему это НЕ баг (API CONTRACT)**:

| Input ([-1,1]) | Output ([0,1]) | Позиция |
|----------------|----------------|---------|
| -1.0 | 0.0 | Exit to cash |
| -0.5 | 0.25 | 25% long |
| 0.0 | 0.5 | 50% long |
| 0.5 | 0.75 | 75% long |
| 1.0 | 1.0 | 100% long |

**ЧАСТАЯ ОШИБКА**: передача `ActionProto(volume_frac=0.5)` с ожиданием "50% позиции"

- 0.5 в [-1,1] маппится в 0.75 в [0,1] -- это **75%**, не 50%!
- Для 50% позиции передавайте `volume_frac=0.0`

**Почему wrapper всегда применяет маппинг**:

- Wrapper НЕ ЗНАЕТ семантику входящего ActionProto
- Он ВСЕГДА преобразует [-1,1] → [0,1] согласно API
- Если вам нужно передать [0,1] напрямую -- НЕ используйте LongOnlyActionWrapper

**Тесты**: `tests/test_long_only_action_space_fix.py::test_action_proto_transformation`

---

### 15. signal_pos в observation = next_signal_pos (trading_patchnew.py:1829-1837)

```python
# FIX (2025-11-26): Set mediator signal_pos to next_signal_pos for observation
if self._reward_signal_only:
    try:
        setattr(
            self._mediator,
            "_last_signal_position",
            float(next_signal_pos),  # FIX: was prev_signal_pos_for_reward
        )
    except Exception:
        pass
```

**Почему это КОРРЕКТНО** (исправлено 2025-11-26):

1. **Gymnasium семантика**: `step(action)` возвращает `s_{t+1}` -- состояние **ПОСЛЕ** действия
2. Observation содержит market data из `next_row` (время t+1)
3. signal_pos в observation должен быть `next_signal_pos` (позиция после step, время t+1)
4. **До фикса**: market data t+1, signal_pos t → temporal mismatch!
5. **После фикса**: market data t+1, signal_pos t+1 → согласованы

**Reward НЕ затронут**:

- Reward = `log(price_change) × prev_signal_pos_for_reward`
- Reward использует позицию, которая **РЕАЛЬНО была** во время price change
- Это корректно и не изменилось

**Влияние бага на training**:

- MDP violation: observation не отражало результат действия
- LSTM confusion: hidden state обновлялся с несогласованным входом
- Sample inefficiency: agent не видел эффект своих действий в obs

**Тесты**: `tests/test_signal_pos_observation_consistency.py` (10 тестов)

---

### 16. Limit Order Maker Fill Logic (execution_sim.py:11420-11448)

```python
elif best_ask is not None and price_q < best_ask:
    filled_price = float(price_q)
    liquidity_role = "maker"
    if (intrabar_fill_price is not None
        and intrabar_fill_price <= limit_price_value + tolerance):
        maker_fill = True
        filled = True
    else:
        filled = False  # ← НЕ заполняется если цена не достигла лимита!
```

**Почему это НЕ баг**: BUY LIMIT с ценой НИЖЕ best_ask НЕ заполняется мгновенно. Заполнение происходит ТОЛЬКО если `intrabar_fill_price` (low бара) достигает лимитной цены. Это корректная симуляция maker orders.

---

### 17. Fee Computed on Filled Price (execution_sim.py:3507-3526)

```python
trade_notional = filled_price * qty_total  # filled_price includes slippage
fee = self._compute_trade_fee(price=filled_price, ...)  # Fee от actual fill price
```

**Почему это НЕ баг (НЕ double-counting)**:

- **Slippage**: разница между expected и actual price (market impact)
- **Fee**: процент от actual fill price (биржевая комиссия)

На реальной бирже комиссия взимается от **фактической цены исполнения**. Это корректное поведение.

---

### 18. VGS _param_ids не сохраняется в state_dict (variance_gradient_scaler.py:136)

```python
self._param_ids: Dict[int, int] = {}  # UNUSED - legacy placeholder
```

**Почему это НЕ баг**: `_param_ids` **НИГДЕ НЕ ИСПОЛЬЗУЕТСЯ**! Поиск `_param_ids[` по коду даёт 0 результатов. VGS работает через `enumerate(self._parameters)` напрямую. Это мёртвый/placeholder код.

---

### 19. UPGDW global_max_util = -inf (optimizers/upgdw.py:106)

```python
global_max_util = torch.tensor(-torch.inf, device="cpu")
# В первом проходе обновляется если есть gradients
# Во втором проходе используется для scaled_utility
```

**Почему это НЕ баг**: Если `global_max_util` остаётся `-inf`, это означает что ВСЕ параметры имели `grad=None` в первом проходе. Но тогда они ТАКЖЕ будут пропущены во втором проходе (`if p.grad is None: continue`). Деление на `-inf` не произойдёт.

---

### 20. CVaR tail_mass = max(alpha, mass * (full_mass + frac)) (distributional_ppo.py:3696)

```python
tail_mass = max(alpha, mass * (full_mass + frac))
# Для α=0.95, N=20: tail_mass = max(0.95, 0.05*19) = 0.95 ✓
```

**Почему это НЕ баг**: Формула **математически корректна**. `max()` защищает от underestimate из-за дискретизации квантилей. Результат всегда ≥ alpha.

---

### 21. CVaR alpha_idx_float < 0 → Extrapolation (distributional_ppo.py:3650-3678)

```python
if alpha_idx_float < 0.0:
    # EXTRAPOLATION CASE: handles negative alpha_idx_float
    # This branch executes BEFORE floor() could give -1
```

**Почему это НЕ баг**: Отрицательный `alpha_idx_float` (для α < tau_0) обрабатывается **отдельным branch** через экстраполяцию. Negative indexing `q[:, -1]` **НИКОГДА не достигается**.

---

### 22. Rolling Window Drawdown Peak (risk_guard.py:99-133)

```python
peak = max(max(self._peak_nw_window, default=nw), nw)
# _peak_nw_window is a deque with maxlen=dd_window
```

**Почему это НЕ баг (BY DESIGN)**: Peak вычисляется в пределах **СКОЛЬЗЯЩЕГО ОКНА** (`dd_window` баров). Это **намеренное** поведение для "recent drawdown" метрики. После заполнения окна peak может уменьшиться -- это корректно.

Для глобального drawdown: `dd_window: 999999` в configs/risk.yaml.

---

### 23. Kill Switch Crash Recovery (services/ops_kill_switch.py:123-156)

```python
def _trip() -> None:
    _tripped = True  # 1. In-memory first
    try:
        atomic_write_with_retry(_flag_path, "1", ...)  # 2. Flag file
    except Exception:
        pass  # OK - _save_state provides backup
    _save_state()  # 3. ALWAYS runs
```

**Почему это НЕ баг**: Crash recovery обеспечивается **дублированием**:

- Если flag write упал → state содержит `tripped=True`
- Если _save_state упал → flag file существует
- При старте проверяются ОБА

I/O внутри lock -- trade-off для consistency, не race condition.

---

### 24. All Features Shifted Together (features_pipeline.py:339-353)

```python
for col in cols_to_shift:
    frame_copy[col] = frame_copy[col].shift(1)
```

**Почему это НЕ баг (НЕТ temporal mismatch)**: SMA, Return, RSI и **ВСЕ** features сдвигаются на 1 период **ОДНОВРЕМЕННО**. После shift они все представляют данные на момент t-1. Temporal alignment сохраняется.

---

### 25. Winsorization Prevents Unbounded Z-scores (features_pipeline.py:588-607)

```python
if "winsorize_bounds" in ms:
    lower, upper = ms["winsorize_bounds"]
    v = np.clip(v, lower, upper)  # Clipping BEFORE z-score!
z = (v - ms["mean"]) / ms["std"]
```

**Почему это НЕ баг**: Winsorization bounds из training применяются **ДО** вычисления z-score. Flash crash: raw=70 → clipped=95 → z=-1.0 (не -6.0!). Экстремальные 50+ sigma z-scores предотвращены.

---

### 26. row_idx для Reward, obs_row_idx для Observation (trading_patchnew.py:2017-2036)

```python
reward_price_curr = self._resolve_reward_price(row_idx, row)  # Current step
# ... while observation uses next_row (obs_row_idx = next_idx)
```

**Почему это НЕ баг (GYMNASIUM SEMANTICS)**:

- `step(action)` returns `(s_{t+1}, r_t, ...)` по стандарту Gymnasium
- `s_{t+1}`: observation из next_row (будущее состояние)
- `r_t`: reward за текущий переход (текущие цены)

Это **корректная MDP семантика**, не temporal mismatch!

---

### 27. GRU vs LSTM Different Paths (custom_policy_patch1.py:972-1012)

```python
if isinstance(recurrent_module, nn.GRU):
    # Handle locally with explicit reshape
    episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
    ...
else:  # LSTM
    # Delegate to base class _process_sequence
    return RecurrentActorCriticPolicy._process_sequence(...)
```

**Почему это НЕ баг (BY DESIGN)**:

- GRU проще (одно hidden state) → обрабатывается локально
- LSTM сложнее (h, c states) → делегируется в базовый класс sb3_contrib
- `_process_sequence` внутри делает тот же reshape для episode_starts
- Оба пути корректно обрабатывают episode boundaries

---

### 28. AdaptiveUPGD grad_norm_ema=1.0 при инициализации (adaptive_upgd.py:159)

```python
if group["adaptive_noise"]:
    state["grad_norm_ema"] = 1.0  # Neutral starting point
```

**Почему это НЕ баг**:

1. **Default mode bypasses EMA**: `instant_noise_scale=True` (default) использует `current_grad_norm` напрямую
2. Строки 215-219: `if group["instant_noise_scale"]: grad_norm_for_noise = current_grad_norm`
3. EMA используется ТОЛЬКО для legacy mode и diagnostics
4. Для legacy mode (`instant_noise_scale=False`) применяется bias correction (строка 224-225)

**Fix уже применён** (2025-11-26): `instant_noise_scale=True` по умолчанию для VGS совместимости.

---

### 29. mediator norm_cols_validity=True (mediator.py:1272)

```python
norm_cols_validity = np.ones(21, dtype=bool)  # Assume valid by default
# Далее ВСЕ 21 элемент перезаписываются:
norm_cols_values[0], norm_cols_validity[0] = self._get_safe_float_with_validity(row, "cvd_24h", 0.0)
# ... (строки 1276-1301)
norm_cols_values[20], norm_cols_validity[20] = self._get_safe_float_with_validity(...)
```

**Почему это НЕ баг**: Начальное значение `np.ones(21)` **полностью перезаписывается** в цикле (строки 1276-1301). Каждый из 21 элементов явно получает значение от `_get_safe_float_with_validity()`. Начальное значение нерелевантно.

---

### 30. risk_guard.py асимметричный buffer (risk_guard.py:668-671)

```python
if exposure_delta > self._EPS:
    buffered_delta = notional_delta * buffer_mult  # Buffer ТОЛЬКО на increase
else:
    buffered_delta = notional_delta  # Без buffer на decrease
```

**Почему это BY DESIGN (корректный risk management)**:

- **Position INCREASE** → нужен safety margin (slippage, fees, market impact)
- **Position DECREASE** → риск уменьшается, дополнительный buffer не нужен
- Это стандартная практика: консервативность при открытии, не при закрытии позиций

---

### 31. ops_kill_switch _last_ts=0.0 при инициализации (ops_kill_switch.py:28, 112-114)

```python
_last_ts: Dict[str, float] = {"rest": 0.0, "ws": 0.0, ...}  # Line 28

def _maybe_reset_all(now: float) -> None:
    for k in list(_counters.keys()):
        if now - _last_ts[k] > _reset_cooldown_sec:  # При now > 60: True
            _counters[k] = 0
            _last_ts[k] = now
```

**Почему это НЕ баг**:

1. `_last_ts[k] = 0.0` означает "последний reset в Unix epoch"
2. При первом вызове `record_error()` в time > 60s: counter сбрасывается до 0, затем инкрементируется до 1
3. При вызове в time < 60s: counter просто инкрементируется до 1
4. Оба сценария дают корректный результат (counter = 1)

---

### 32. RSI timing: valid на bar 14 (transformers.py:959-968)

```python
st["gain_history"].append(gain)
st["loss_history"].append(loss)

if st["avg_gain"] is None or st["avg_loss"] is None:
    if len(st["gain_history"]) == self.spec.rsi_period:  # == 14
        st["avg_gain"] = sum(st["gain_history"]) / float(self.spec.rsi_period)
        st["avg_loss"] = sum(st["loss_history"]) / float(self.spec.rsi_period)
```

**Почему это НЕ баг (timing корректен)**:

| Bar | Action | len(gain_history) | RSI valid? |
|-----|--------|-------------------|------------|
| 0 | last_close = price0 | 0 | ❌ |
| 1 | delta = p1-p0, append | 1 | ❌ |
| ... | ... | ... | ❌ |
| 14 | delta = p14-p13, append | 14 | ✅ SMA computed |

**RSI-14** требует 14 price changes → доступен после 15 prices (bars 0-14). Bar 14 -- корректный момент.

**Референс**: Wilder (1978), "New Concepts in Technical Trading Systems"

---

### 33. obs_builder vol_proxy=0.01 во время ATR warmup (obs_builder.pyx:389-396)

```cython
if atr_valid:
    vol_proxy = tanh(log1p(atr / (price_d + 1e-8)))
else:
    atr_fallback = price_d * 0.01  # 1% of price
    vol_proxy = tanh(log1p(atr_fallback / (price_d + 1e-8)))
```

**Почему это BY DESIGN (trade-off)**:

| Вариант | vol_proxy | Проблема |
|---------|-----------|----------|
| NaN | NaN | Observation crash, NaN propagation |
| 0.0 | 0.0 | Model видит "нулевая волатильность" -- неверно! |
| **1% price** | ~0.01 | Разумная аппроксимация типичного ATR |

Типичный ATR для crypto: 1-3% от цены. Fallback 1% -- консервативная оценка.

---

### 34. obs_builder FG=50 vs missing РАЗЛИЧИМЫ (obs_builder.pyx:590-600)

```cython
if has_fear_greed:
    feature_val = _clipf(fear_greed_value / 100.0, -3.0, 3.0)  # FG=50 → 0.5
    indicator = 1.0  # FLAG: present
else:
    feature_val = 0.0
    indicator = 0.0  # FLAG: missing
```

**Почему это НЕ баг**:

| Сценарий | feature_val | indicator | Различимы? |
|----------|-------------|-----------|------------|
| FG = 50 | 0.5 | **1.0** | ✅ |
| FG missing | 0.0 | **0.0** | ✅ |

Indicator flag (второй элемент пары) **полностью различает** реальные данные от отсутствующих.

---

### 35. Policy sigma range [0.2, 1.5] (custom_policy_patch1.py:1088-1091)

```python
sigma_min, sigma_max = 0.2, 1.5
sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(self.unconstrained_log_std)
```

**Почему это НЕ баг (standard PPO practice)**:

- **σ = 0.2**: near-deterministic actions (exploitation phase)
- **σ = 1.5**: high exploration
- Работает для обоих: tanh [-1,1] и sigmoid [0,1] выходов
- Большое σ естественно приводит к saturated actions (bounds)

**Референс**: Schulman et al. (2017) PPO, OpenAI Baselines defaults

---

### 36. features_pipeline constant detection на shifted data (features_pipeline.py:396-410)

```python
m = float(np.nanmean(v_clean))  # Ignores NaN
s = float(np.nanstd(v_clean, ddof=0))  # Ignores NaN
is_constant = (not np.isfinite(s)) or (s == 0.0)
```

**Почему это НЕ баг (practical for typical datasets)**:

1. `nanmean`/`nanstd` **игнорируют NaN** при вычислении
2. Shifted data имеет NaN только в первых ~20 rows
3. Типичный training dataset: 10,000+ rows
4. Первые 20 NaN rows составляют < 0.2% -- negligible impact
5. Statistics корректно вычисляются на valid portion

**Edge case**: Если dataset < 100 rows, могут быть issues. Но training datasets всегда >>1000 rows.

---

### 37. mark_for_obs passed but "recomputed" inside _signal_only_step (trading_patchnew.py:1868-1879, 1040)

```python
# Caller (step method):
mark_for_obs = self._resolve_reward_price(row_idx, row)  # current row
result = self._signal_only_step(..., float(mark_for_obs), ...)

# Inside _signal_only_step:
next_mark_price = self._resolve_reward_price(obs_row_idx, next_row)  # NEXT row (different!)
```

**Почему это НЕ баг**:

1. `mark_price` (from caller) используется для **текущего** net_worth (line 979)
2. `next_mark_price` вычисляется для **следующей** строки (Gymnasium semantics: obs = s_{t+1})
3. Это **разные rows** с разными ценами -- повторное вычисление НЕОБХОДИМО
4. `mark_price` также используется как fallback (line 1042) если next invalid

---

### 38. ratio_clipped not clipped in signal_only mode (trading_patchnew.py:2126-2129)

```python
# Signal-only mode:
ratio_clipped = float(ratio_price)  # No np.clip() call!

# Non-signal_only mode:
ratio_clipped = float(np.clip(ratio_price, ratio_clip_floor, ratio_clip_ceiling))
```

**Почему это BY DESIGN (НЕ баг)**:

1. Variable named "ratio_clipped" for **API consistency** -- info dict always has this key
2. In signal_only: ratio is **sanitized** (NaN→1.0) but not bounds-clipped
3. Signal-only mode doesn't simulate extreme price moves -- clipping unnecessary
4. Comment added to code explaining this design decision

---

### 39. Empty action array returned without mapping (wrappers/action_space.py:108-110)

```python
if isinstance(action, np.ndarray):
    if action.size == 0:
        return action  # Returns empty array as-is
```

**Почему это НЕ баг (корректное поведение)**:

1. Empty array contains **nothing to map** -- no elements to transform
2. Mapping formula `(arr + 1.0) / 2.0` on empty array would still produce empty array
3. Early return preserves type and is more efficient
4. This is standard defensive programming for edge cases

---

### 40. _log_sigmoid_jacobian_from_raw misleading name (custom_policy_patch1.py:1350-1353)

```python
def _log_sigmoid_jacobian_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
    # DEPRECATED: Use _log_activation_jacobian instead
    # Kept for backwards compatibility
    return self._log_activation_jacobian(raw)
```

**Почему это НЕ баг**:

1. Method is **explicitly marked DEPRECATED** in comment
2. Delegates to correctly-named `_log_activation_jacobian`
3. Kept for **backwards compatibility** -- external code may reference it
4. Will be removed in future major version

---

### 41. 4 samples for entropy estimation (custom_policy_patch1.py:1420-1433)

```python
samples = 4
entropy_accum: Optional[torch.Tensor] = None
for _ in range(samples):
    raw_sample = rsample_fn()
    ...
entropy_estimate = -(entropy_accum / float(samples))
```

**Почему это НЕ проблема**:

1. Monte Carlo entropy variance scales as O(1/n) -- 4 samples gives ~25% relative error
2. **ent_coef = 0.001** (from configs) -- entropy contributes tiny fraction to loss
3. Impact on total loss: `0.001 × entropy × (1 ± 0.25)` ≈ negligible
4. Increasing to 16 samples would 4x compute for <0.1% loss improvement
5. Trade-off: speed vs accuracy -- current choice prioritizes training throughput

---

### 42. No handling for reduction with spaces/case (distributional_ppo.py:3495-3496)

```python
if reduction not in ("none", "mean", "sum"):
    raise ValueError(f"Invalid reduction mode: {reduction}")
```

**Почему это НЕ баг (стандартный API design)**:

1. Follows **PyTorch convention** -- exact string matching, no normalization
2. `torch.nn.functional.mse_loss(reduction="Mean")` also raises error
3. Case sensitivity is **intentional** for API strictness
4. Adding `.lower().strip()` would hide caller bugs and violate principle of least surprise

---

### 43. Redundant isfinite(bb_width) check (obs_builder.pyx:550-559)

```python
if (not bb_valid) or bb_width <= min_bb_width:
    feature_val = 0.5
else:
    if not isfinite(bb_width):  # "Redundant" check
        feature_val = 0.5
    else:
        feature_val = _clipf(...)
```

**Почему это НЕ баг (defense-in-depth)**:

1. `bb_valid` checks **indicator computed** -- not that bb_width is finite
2. Edge case: bb_valid=True but bb_width=inf from overflow in upstream calc
3. Comment in code explicitly says "Additional safety" -- **intentional redundancy**
4. Cost: one `isfinite()` check; Benefit: guaranteed NaN-free output
5. Defense-in-depth is **best practice** for numerical code

---

### 44. ma20 variable is actually 21-bar MA (mediator.py:1199-1201)

```python
# HISTORICAL NAMING: Variable named "ma20" for feature schema compatibility
# Actual value is 21-bar SMA (sma_5040 = 21 bars × 240 min)
ma20 = self._get_safe_float(row, "sma_5040", float('nan'))
```

**Почему это BY DESIGN (НЕ баг)**:

1. Variable name is **legacy** from feature schema (feature_config.py)
2. Renaming would break:
   - Feature parity checks
   - Trained models expecting this feature order
   - Audit scripts and documentation
3. Comment added to code explaining the naming
4. Underlying value (21-bar SMA) is **correct** -- only name is historical artifact

---

### 45. Twin Critics Logging Accumulators (distributional_ppo.py:11088-11094, 12288-12290)

```python
# Accumulation during training:
self._twin_critic_1_loss_sum += float(loss_critic_1.mean().item()) * weight

# Reset after logging:
self._twin_critic_1_loss_sum = 0.0
self._twin_critic_2_loss_sum = 0.0
self._twin_critic_loss_count = 0
```

**Почему это НЕ memory leak**:

1. Accumulators are **RESET** at line 12288-12290 after logging
2. Reset happens at end of each train() iteration
3. Float values can't overflow in practice (values << 1e308)
4. This is standard accumulate-then-log pattern

---

### 46. Advantage Normalization ddof=1 (distributional_ppo.py:8454)

```python
adv_std = float(np.std(advantages_flat, ddof=1))  # Sample std with Bessel correction
```

**Почему это minor inconsistency (НЕ баг)**:

1. SB3 uses `ddof=0` (population std), our code uses `ddof=1` (sample std)
2. Difference: factor √(n/(n-1)) ≈ 1.0005 for n=10000
3. For typical batch sizes (n>1000): difference < 0.1%
4. Both approaches are valid -- this is a philosophical difference
5. ddof=1 gives unbiased estimate, ddof=0 is more common in RL

**Референс**: Bessel's correction, SB3 `on_policy_algorithm.py`

---

### 47. VGS State in PBT Checkpoints (adversarial/pbt_scheduler.py:340-455)

```python
# Each worker saves to unique file:
checkpoint_path = f"member_{member.member_id}_step_{step}.pt"
torch.save(checkpoint_to_save, checkpoint_path)

# VGS state is serialized atomically:
has_vgs = 'vgs_state' in checkpoint_data
```

**Почему это НЕ race condition**:

1. Each PBT worker has **its own model and VGS instance**
2. Checkpoints are saved to **unique files** per worker
3. torch.save/load are atomic at OS level
4. Python GIL prevents concurrent access to live objects
5. VGS state_dict is serialized **before** save (no concurrent modification)

---

### 48. CVaR Approximation Error ~16% for N=21 (distributional_ppo.py:3612-3615)

```python
# Note on Accuracy:
#     - Perfect for linear distributions (0% error)
#     - ~5-18% approximation error for standard normal (decreases with N)
#     - N=21 (default): ~16% error
```

**Почему это documented trade-off (НЕ баг)**:

1. **Already documented** in code with accuracy notes
2. Numerical integration over discrete quantiles has inherent error
3. Error decreases with N: N=51 gives ~5%, N=101 gives ~2%
4. Trade-off: more quantiles = more accurate but slower training
5. For risk-critical applications: increase `num_quantiles` to 51+

**Референс**: Dabney et al. (2018) "IQN", quantile regression theory

---

### 49. Winsorization Percentiles [1%, 99%] (features_pipeline.py:181)

```python
winsorize_percentiles: Tuple[float, float] = (1.0, 99.0)
```

**Почему это configurable (НЕ issue)**:

1. Default [1%, 99%] clips 2% of extreme values
2. For crypto with fat tails: can adjust to [0.5%, 99.5%] or [0.1%, 99.9%]
3. This is a **configurable parameter**, not hardcoded limitation
4. Winsorization bounds are computed from training data and stored
5. Inference applies same bounds for consistency

---

### 50. obs_builder.pyx boundscheck=False (obs_builder.pyx:1)

```cython
# cython: boundscheck=False, wraparound=False
```

**Почему это BY DESIGN (performance trade-off)**:

1. `boundscheck=False` is a **deliberate Cython optimization** for critical path
2. The `build_observation_vector` Python wrapper validates all inputs before calling C version
3. Array size is determined by `compute_n_features()` which ensures consistency with observation_space
4. If mismatch occurs, it's a configuration error caught during testing
5. Re-enabling bounds checking would add ~15-20% overhead to observation building
6. Defense layers: P0 (mediator validation) → P1 (wrapper validation) → C function

**Referenced in**: 2025-11-26 bug investigation (Issue #2 - concluded NOT A BUG)

---

### 51. Slippage Model Uses Mid-Price (execution_sim.py:5901-5910)

```python
cost_fraction = float(expected_bps) / 1e4
if side_key == "BUY":
    candidate = mid_val * (1.0 + cost_fraction)
```

**Почему это НЕ проблема (already has market impact model)**:

1. Slippage module уже включает **market impact term**: `k * sqrt(participation_ratio)` (impl_slippage.py:2342)
2. Это стиль **Almgren-Chriss** square-root impact model
3. `participation_ratio = order_notional / ADV` учитывает размер ордера
4. Mid-price -- только reference point; фактический slippage включает:
   - Half spread (`half_spread`)
   - Market impact (`k_effective * sqrt(participation_ratio)`)
   - Volatility adjustments
   - Tail shock для extreme conditions
5. Для полного LOB simulation нужен external LOB -- это documented design choice

**Референс**: Almgren & Chriss (2001), impl_slippage.py:2290-2354

---

### 52. Latency Clamping Warnings Configurable (execution_sim.py:7110-7126)

```python
if ratio > 1.0 and self._intrabar_log_warnings:  # Configurable!
    logger.warning("intrabar latency %.0f ms exceeds timeframe %.0f ms ...")
    # Throttled to avoid log spam
if ratio > 1.0:
    ratio = 1.0  # Clamped to end of bar
```

**Почему это НЕ "silent" clamping**:

1. Warning **IS** logged when `_intrabar_log_warnings=True`
2. Default `False` для performance (production не нуждается в verbose logging)
3. Throttling предотвращает log spam
4. Configurable через `execution.intrabar.log_warnings: true`
5. Clamping at 100% -- корректное поведение (исполнение в конце бара)

**Референс**: execution_sim.py:2555, 2598-2604

---

### 53. No LOB Depth Tracking (execution_sim.py:11414-11424, docstring)

```python
# Из docstring модуля (execution_sim.py:14-16):
# 3) Работать как с внешним LOB (если он передан), так и без него (простая модель):
#    - Для LIMIT без LOB исполняем только если есть abs_price
```

**Почему это BY DESIGN (not a bug)**:

1. **Documented design choice**: модуль работает с/без external LOB
2. Full LOB simulation = significant computational overhead
3. Queue position tracking добавит complexity без proportional benefit
4. Для backtesting стратегий простая модель достаточна
5. Production с крупными объёмами: используйте external LOB adapter
6. Market impact через `participation_ratio` уже покрывает основной эффект

**Референс**: execution_sim.py:4-23 (module docstring), standard backtesting practice

---

### 54. L2 ADV Ignores Intraday Seasonality (execution_providers.py:2867-2870)

```python
if market.adv is not None and market.adv > 0:
    ref_price = market.get_mid_price() or bar.typical_price
    order_notional = order.get_notional(ref_price)
    return order_notional / market.adv  # No TOD adjustment
```

**Почему это BY DESIGN (L2 vs L2+ trade-off)**:

1. L2 (`StatisticalSlippageProvider`) is intentionally **simple and fast** for rapid backtesting
2. L2+ (`CryptoParametricSlippageProvider`) has `tod_curve` at lines 785-792 with Asia/EU/US session factors (0.70-1.15)
3. L2+ applies TOD adjustment to slippage, effectively capturing intraday effects
4. Adding TOD to L2 would require `hour_utc` parameter breaking backward compatibility
5. Users requiring accurate intraday cost estimation should use L2+ or L3

**Fidelity Level Selection**:

- **L2**: Quick backtests, strategy screening (±30-50% cost error acceptable)
- **L2+**: Production cost estimation (TOD, imbalance, funding, whale detection)
- **L3**: HFT research, queue position tracking, fill probability models

**Референс**: ITG (2012) "Global Cost Review", Kyle (1985)

---

### 55. L2 No Permanent vs Temporary Impact Separation (impl_slippage.py:2342-2349)

```python
impact_term = k_effective * math.sqrt(participation_ratio)  # √participation = temporary
base_cost = half_spread + impact_term  # Single-term model
```

**Почему это BY DESIGN (L2 vs L3 trade-off)**:

1. L2 uses **simplified Almgren-Chriss**: `k * √participation` -- temporary impact only
2. L3 has full separation in `lob/market_impact.py`:
   - `AlmgrenChrissModel`: `temp = η * σ * (Q/V)^0.5`, `perm = γ * (Q/V)`
   - `GatheralModel`: transient impact with power-law decay `G(t) = (1 + t/τ)^(-β)`
3. For bar-level simulation, temp/perm distinction matters less (impact reverts within bar)
4. For HFT simulation, use L3 with proper impact decay modeling

**Референс**: Almgren & Chriss (2001), Gatheral (2010)

---

### 56. L2 Spread Model Static (execution_providers.py:514-518)

```python
spread = market.get_spread_bps()
if spread is None or not math.isfinite(spread) or spread < 0:
    half_spread = self.spread_bps / 2.0  # Default fallback
```

**Почему это BY DESIGN**:

1. L2 uses market spread if available in `MarketState.get_spread_bps()`
2. L2+ adds volatility-based adjustments via `vol_regime_multipliers` (0.8-1.5x)
3. L2+ has order book `imbalance_penalty_max` (up to 30% extra cost)
4. Dynamic spread widening is implemented in L2+, not L2

**Референс**: Cont et al. (2014) "Price Impact of Order Book Events"

---

### 57. L2 Limit Order Fills Deterministic (execution_sim.py:11750-11755)

```python
if intrabar_fill_price is not None and intrabar_fill_price <= limit_price_value + tolerance:
    maker_fill = True
    filled = True  # Binary: filled or not
```

**Почему это BY DESIGN (L2 vs L3 trade-off)**:

1. L2 uses **binary fill logic**: price touches limit → filled
2. L3 has probabilistic models in `lob/fill_probability.py`:
   - `PoissonFillModel`: `P(fill in T) = 1 - exp(-λT / position)`
   - `QueueReactiveModel`: `λ_i = f(q_i, spread, volatility, imbalance)`
   - `QueueValueModel`: Value = P(fill) × spread/2 - adverse_selection
3. Queue position tracking in `lob/queue_tracker.py` with MBP/MBO estimation
4. L2 is 100-1000x faster than L3 for backtesting

**Референс**: Huang et al. (2015) Queue-Reactive Model, Moallemi & Yuan (2017)

---

### 58. Whale Threshold 1% Not ADV-Scaled (execution_providers.py:798)

```python
whale_threshold: float = 0.01  # 1% of ADV
```

**Почему это CONFIGURABLE (not a bug)**:

1. Threshold is **participation ratio** (order/ADV), already normalized by ADV
2. 1% default is reasonable: $100M order on $10B ADV is whale behavior
3. For low-ADV altcoins: use `CryptoParametricConfig(whale_threshold=0.005)` (0.5%)
4. For stablecoin pairs: use profile `from_profile("stablecoin")` with lower threshold
5. Configuration profiles exist: `default`, `conservative`, `aggressive`, `altcoin`, `stablecoin`

**Usage**:

```python
# For low-liquidity altcoins
config = CryptoParametricConfig(whale_threshold=0.005)  # 0.5%
provider = CryptoParametricSlippageProvider(config=config)

# Or use built-in profile
provider = CryptoParametricSlippageProvider.from_profile("altcoin")
```

---

### 59. Reward Clipping is NOT Stacked (trading_patchnew.py:2201, 2345)

```python
# Line 2201: Numerical safety BEFORE log()
ratio_clipped = np.clip(ratio, 1e-10, 1e10)

# Line 2345: Final reward bounds (policy requirement)
reward = float(np.clip(reward_before_clip, -clip_for_clamp, clip_for_clamp))
```

**Почему это НЕ создаёт non-monotonic value function**:

1. **First clip** (line 2201): Protects against numerical overflow in `log(ratio)`
   - Without this, ratio=0 → log(0)=-inf → NaN propagation
   - Clipping to [1e-10, 1e10] is defensive programming, not reward shaping

2. **Second clip** (line 2345): Bounds the final reward for policy stability
   - RL policies need bounded rewards for numerical stability
   - `clip_for_clamp` is typically large (e.g., 10.0), rarely triggered

3. **Different code paths**: `reward.pyx` has separate `_clamp` for non-signal-only mode
   - These are independent code paths, not stacked operations

**Value function remains monotonic** because:

- Both clips are defensive (rarely triggered in normal operation)
- First clip applies BEFORE log → preserves log's monotonicity
- Second clip applies AFTER all computations → bounds extreme outliers only

**Референс**: Standard numerical programming practice, Schulman et al. (2017) PPO

---

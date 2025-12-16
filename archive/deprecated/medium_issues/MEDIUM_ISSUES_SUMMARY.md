# КРАТКОЕ РЕЗЮМЕ MEDIUM PRIORITY ПРОБЛЕМ
## AI-Powered Quantitative Research Platform Mathematical Audit

**Всего:** 14 проблем | **Impact:** 4-6/10 каждая

---

## MEDIUM #1: Return Fallback 0.0 Instead of NaN
**Файл:** Feature calculation (obs_builder.pyx, features/)
**Проблема:** При невозможности вычислить return (first bar, invalid price), возвращается `0.0` вместо `NaN`
**Эффект:** `0.0` выглядит как "нет изменения цены", но на самом деле означает "нет данных" - модель не может различить
**Решение:**
```python
if price_prev <= 0 or price_current <= 0:
    return np.nan  # Вместо 0.0
```
**Impact:** 4/10 - влияет на первый bar каждого episode, создает semantic ambiguity

---

## MEDIUM #2: Parkinson Volatility Uses valid_bars Instead of n
**Файл:** Volatility estimators (features/)
**Проблема:** Формула Parkinson использует `valid_bars` (количество валидных данных) вместо `n` (размер окна) в знаменателе
**Академическая формула:** `σ_P = √(Σ[ln(H/L)]² / (4·n·ln2))`
**Текущая реализация:** `σ_P = √(Σ[ln(H/L)]² / (4·valid_bars·ln2))`
**Вопрос:** Это intentional (statistically correct for missing data) или ошибка?
**Решение:** Документировать выбор OR использовать `n` с data completeness check
**Impact:** 5/10 - разница 4-12% при missing data, влияет на volatility estimates

---

## MEDIUM #3: No Outlier Detection for Returns
**Файл:** Feature calculation, data preprocessing
**Проблема:** Нет фильтрации экстремальных returns (flash crashes -50%, fat-finger errors)
**Эффект:** Один outlier может сдвинуть mean на 5x и раздуть std на 50%+ → нормализация искажена
**Решение:** Winsorization (1/99 percentile) или Z-score filtering (|z| < 3):
```python
returns = np.clip(returns, np.percentile(returns, 1), np.percentile(returns, 99))
```
**Impact:** 6/10 - contaminated statistics, model learns on anomalies

---

## MEDIUM #4: Zero Std Fallback to 1.0
**Файл:** Feature normalization (features_pipeline.py)
**Проблема:** Когда feature имеет zero variance, используется `std = 1.0` как fallback
**Текущий код:**
```python
if s < 1e-8:
    s = 1.0
normalized = (value - mean) / s
```
**Эффект:** Constant features НЕ нормализуются к 0 если mean != constant (edge case с NaN)
**Решение:** Explicit zero для constant features:
```python
if s < 1e-8:
    return np.zeros_like(values)
```
**Impact:** 3/10 - редкий edge case, но лучше обработать явно

---

## MEDIUM #5: Lookahead Bias в Close Price Shifting
**Файл:** features_pipeline.py:163-164, 213-214
**Проблема:** `shift(1)` применяется к `close` в двух местах (fit и transform_df) → риск double-shifting
**Эффект:** Если данные уже shifted на входе, может произойти triple shift → потеря данных
**Решение:** Добавить флаг `_close_shifted`, shift только один раз:
```python
if not self._close_shifted:
    df["close"] = df["close"].shift(1)
    self._close_shifted = True
```
**Impact:** 5/10 - зависит от data flow, может быть невидимым

---

## MEDIUM #6: Unrealistic Data Degradation Patterns
**Файл:** data_validation.py, impl_offline_data.py
**Проблема:** Degradation simulation использует IID probabilities (independent drops/stale) - реальные сети имеют correlated failures
**Отсутствует:**
- Burst failures (кластеризованные сбои)
- Recovery lag (queue flush после dropout)
- Timestamp jitter
**Решение:** Markov chain для коррелированных failures:
```python
if self.state == 'DEGRADED':
    if random() < 0.6:  # 60% остаться в degraded
        self.state = 'DEGRADED'
```
**Impact:** 5/10 - может вызвать overfitting к specific degradation pattern

---

## MEDIUM #7: Double Turnover Penalty
**Файл:** reward.pyx:141-154
**Проблема:** Система применяет ДВА penalties на trading:
1. Transaction costs: `taker_fee + spread + impact` (~0.12%)
2. Turnover penalty: `turnover_penalty_coef * notional` (~0.05%)
**Итого:** ~0.17% за сделку
**Вопрос:** Это intentional double penalty чтобы discourage overtrading, или redundant?
**Решение:** Документировать design choice:
```python
# Intentional double penalty:
# 1. Transaction costs = real market costs
# 2. Turnover penalty = behavioral regularization
```
**Impact:** 4/10 - влияет на trading frequency, может быть by design

---

## MEDIUM #8: Event Reward Logic
**Файл:** reward.pyx:59-77
**Проблема:** Все `ClosedReason` кроме NONE, BANKRUPTCY, и TP получают `-loss_penalty`
**Эффект:** Даже нейтральные closes (timeout, manual stop) наказываются как losses
**Текущий код:**
```python
if closed_reason == ClosedReason.STATIC_TP_LONG or closed_reason == ClosedReason.STATIC_TP_SHORT:
    return profit_bonus
return -loss_penalty  # ← Всё остальное!
```
**Решение:** Добавить TIMEOUT case:
```python
if closed_reason == ClosedReason.TIMEOUT:
    return 0.0  # Neutral
```
**Impact:** 4/10 - может быть слишком punitive для neutral closes

---

## MEDIUM #9: Hard-coded Reward Clip
**Файл:** reward.pyx:163
**Проблема:** Reward clipping hardcoded вместо чтения из config
**Текущий код:**
```python
reward = _clamp(reward, -10.0, 10.0)  # ← Hardcoded!
```
**Config имеет:** `reward_cap: 10.0`
**Решение:** Pass as parameter:
```python
def compute_reward_view(..., reward_cap=10.0):
    reward = _clamp(reward, -reward_cap, reward_cap)
```
**Impact:** 3/10 - minor, но лучше читать из config

---

## MEDIUM #10: BB Position Asymmetric Clipping
**Файл:** obs_builder.pyx:498
**Проблема:** Bollinger Bands position clipped к `[-1.0, 2.0]` вместо стандартного `[0, 1]`
**Текущий код:**
```python
feature_val = _clipf((price_d - bb_lower) / (bb_width + 1e-9), -1.0, 2.0)
```
**Стандартная формула:** `(price - lower) / (upper - lower)` → `[0, 1]`
**Вопрос:** Intentional (capture extreme bullish moves) или ошибка?
**Решение:** Документировать OR стандартизировать к `[-1, 1]` или `[0, 1]`
**Impact:** 3/10 - unusual но может быть by design

---

## MEDIUM #11: BB Squeeze Normalization
**Файл:** obs_builder.pyx:419-462
**Проблема:** BB squeeze нормализуется по-другому чем другие indicators
**Текущий код:**
```python
bb_squeeze = tanh((bb_upper - bb_lower) / (price_d + 1e-8))  # Делит на price
price_momentum = tanh(momentum / (price_d * 0.01 + 1e-8))    # Делит на 1% price
```
**Эффект:** BB squeeze использует full price для scaling, другие используют 1% price
**Вопрос:** Intentional (BB width typically 1-5% of price) или inconsistency?
**Решение:** Документировать distinction:
```python
# BB width is typically 1-5% of price, so normalize by full price
# Other indicators represent % moves, so normalize by 1% price
```
**Impact:** 3/10 - design choice, но нужна документация

---

## MEDIUM #12: Bankruptcy State Ambiguity
**Файл:** obs_builder.pyx:384-389
**Проблема:** Когда `total_worth ≈ 0`, cash ratio показывает "100% cash" вместо bankruptcy
**Текущий код:**
```python
if total_worth <= 1e-8:
    feature_val = 1.0  # ← Выглядит как "all cash"
else:
    feature_val = cash / total_worth
```
**Эффект:** Bankruptcy (negative equity) и "zero equity + all cash" выглядят одинаково
**Решение:** Добавить explicit bankruptcy flag:
```python
if total_worth < -1e-8:  # Negative equity
    cash_ratio = 0.0
    is_bankrupt = 1.0
elif total_worth <= 1e-8:  # Near-zero
    cash_ratio = 1.0
    is_bankrupt = 0.0
```
**Impact:** 2/10 - очень редкий edge case

---

## MEDIUM #13: Checkpoint Integrity Validation Missing
**Файл:** Model checkpointing (distributional_ppo.py, variance_gradient_scaler.py)
**Проблема:** Нет validation что loaded checkpoint не corrupted
**Отсутствует:**
- Version check
- Checksum/hash
- State dict completeness validation
**Решение:** Add validation:
```python
def save_checkpoint(state, path):
    import hashlib
    state['__version__'] = '1.0'
    state['__checksum__'] = hashlib.sha256(pickle.dumps(state)).hexdigest()
    torch.save(state, path)

def load_checkpoint(path):
    state = torch.load(path)
    assert state.get('__version__') == '1.0', "Version mismatch!"
    # Verify checksum
```
**Impact:** 6/10 - важно для production reliability

---

## MEDIUM #14: Entropy NaN/Inf Validation Missing
**Файл:** distributional_ppo.py (entropy loss calculation)
**Проблема:** Entropy loss не проверяет NaN/Inf values от `distribution.entropy()`
**Текущий код:**
```python
entropy = distribution.entropy()
entropy_loss = -entropy.mean()  # ← No validation!
```
**Решение:** Add validation:
```python
entropy = distribution.entropy()
if not torch.all(torch.isfinite(entropy)):
    self.logger.record("warn/entropy_invalid", 1.0)
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
entropy_loss = -entropy.mean()
```
**Impact:** 5/10 - может предотвратить rare training crashes

---

## SUMMARY TABLE

| Issue | Impact | Effort | Priority | Type |
|-------|--------|--------|----------|------|
| #1 Return Fallback | 4/10 | Low | Quick Win | Data Quality |
| #2 Parkinson Vol | 5/10 | Low | Document | Statistical |
| #3 Outlier Detection | 6/10 | Medium | Important | Robustness |
| #4 Zero Std | 3/10 | Low | Quick Win | Edge Case |
| #5 Lookahead Bias | 5/10 | Medium | Important | Data Leakage |
| #6 Degradation | 5/10 | High | Future | Infrastructure |
| #7 Double Penalty | 4/10 | Low | Document | Design |
| #8 Event Reward | 4/10 | Low | Quick Win | Logic |
| #9 Hard-coded Clip | 3/10 | Low | Quick Win | Config |
| #10 BB Position | 3/10 | Low | Document | Design |
| #11 BB Squeeze | 3/10 | Low | Document | Design |
| #12 Bankruptcy | 2/10 | Low | Edge Case | Rare |
| #13 Checkpoint | 6/10 | Medium | Important | Production |
| #14 Entropy | 5/10 | Low | Quick Win | Robustness |

---

## QUICK WINS (High Impact / Low Effort)

**Исправить за 1-2 часа:**
1. #1: Return → NaN для missing data
2. #4: Zero std → explicit zeros
3. #8: Event reward → add TIMEOUT case
4. #9: Reward clip → read from config
5. #14: Entropy → add NaN validation

**Итого:** 5 quick wins, ~2 hours

---

## DOCUMENTATION NEEDED

**Добавить комментарии/docstrings:**
- #2: Parkinson vol denominator choice
- #7: Double turnover penalty rationale
- #10: BB position asymmetric clipping
- #11: BB squeeze normalization scale

**Итого:** 4 doc updates, ~30 min

---

## ВАЖНЫЕ УЛУЧШЕНИЯ (Средний effort)

**Исправить в следующем спринте:**
- #3: Outlier detection (Winsorization)
- #5: Lookahead bias validation
- #13: Checkpoint integrity validation

**Итого:** 3 improvements, ~1 day

---

## ИТОГО MEDIUM ISSUES

**Total:** 14 проблем
**Average Impact:** 4.2/10
**Quick Wins:** 5 (можно за 2 часа)
**Documentation:** 4 (можно за 30 мин)
**Important Improvements:** 3 (нужен 1 день)
**Future Work:** 2 (оставить на потом)

**Рекомендуемый порядок:**
1. Week 1: Quick Wins (#1, #4, #8, #9, #14)
2. Week 1: Documentation (#2, #7, #10, #11)
3. Week 2: Important (#3, #5, #13)
4. Future: Infrastructure (#6, #12)

**Total effort для завершения:** ~2-3 дня работы

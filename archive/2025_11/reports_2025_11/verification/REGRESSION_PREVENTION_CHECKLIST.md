# Regression Prevention Checklist
## AI-Powered Quantitative Research Platform - Предотвращение повторения критических проблем

**Дата создания**: 2025-11-21
**Версия**: 1.0
**Статус**: Обязательно к применению ✅

---

## 🎯 Цель документа

Этот чек-лист должен использоваться **ВСЕМИ** разработчиками и AI-ассистентами перед:
1. Изменением критической логики (PPO, LSTM, features, execution)
2. Рефакторингом существующего кода
3. Добавлением новых возможностей к core компонентам
4. Code review критических изменений

**⚠️ ВАЖНО**: Игнорирование этого чек-листа может привести к регрессиям, которые уже были исправлены!

---

## 📋 ОБЯЗАТЕЛЬНЫЙ ЧЕК-ЛИСТ ПЕРЕД ИЗМЕНЕНИЯМИ

### ✅ 1. LSTM State Management

**Перед изменением `distributional_ppo.py`, особенно методов с LSTM:**

- [ ] **Проверить**: LSTM states сбрасываются на episode boundaries (`done=True`)
- [ ] **Проверить**: Вызов `_reset_lstm_states_for_done_envs()` присутствует в rollout loop
- [ ] **Проверить**: Вызов происходит ПОСЛЕ обновления `self._last_episode_starts`
- [ ] **Запустить**: `pytest tests/test_lstm_episode_boundary_reset.py -v`
- [ ] **Проверить**: Все 8 тестов проходят

**Критический код (НЕ УДАЛЯТЬ!):**
```python
# distributional_ppo.py:7418-7427
self._last_episode_starts = dones

if np.any(dones):
    init_states = self.policy.recurrent_initial_state
    init_states_on_device = self._clone_states_to_device(init_states, self.device)
    self._last_lstm_states = self._reset_lstm_states_for_done_envs(
        self._last_lstm_states, dones, init_states_on_device
    )
```

**Симптомы регрессии:**
- Value loss не снижается
- Explained variance низкая (<0.5)
- Model overfits на первые episodes
- Странное поведение при смене длины episodes

**См.**: [CRITICAL_LSTM_RESET_FIX_REPORT.md](CRITICAL_LSTM_RESET_FIX_REPORT.md)

---

### ✅ 2. NaN Handling в External Features

**Перед изменением `mediator.py`, `obs_builder.pyx`, feature extraction:**

- [ ] **Проверить**: NaN values конвертируются в default (обычно 0.0)
- [ ] **Проверить**: `_clipf()` в obs_builder.pyx содержит `if isnan(value): return 0.0`
- [ ] **Проверить**: `_get_safe_float()` возвращает default при non-finite values
- [ ] **Добавить**: `log_nan=True` для debugging если нужно отслеживать NaN
- [ ] **Запустить**: `pytest tests/test_nan_handling_external_features.py -v`
- [ ] **Проверить**: 9+ тестов проходят (1 может быть skipped - Cython)

**Критический код (НЕ ИЗМЕНЯТЬ логику!):**
```python
# mediator.py:1045-1052
if not math.isfinite(result):
    if log_nan:
        logger.warning(
            f"Feature '{col}' has non-finite value ({result}), "
            f"using default={default}. Model cannot distinguish "
            f"missing data from zero values."
        )
    return default
```

**Известное ограничение (design decision):**
- NaN → 0.0 создает semantic ambiguity (missing data выглядит как zero)
- Future fix: validity flags для external features (требует retrain)

**Симптомы регрессии:**
- NaN propagation в observations
- Training crashes с "NaN loss"
- External features содержат NaN в tensorboard

**См.**: [NUMERICAL_ISSUES_FIX_SUMMARY.md](NUMERICAL_ISSUES_FIX_SUMMARY.md) - Issue #2

---

### ✅ 3. Action Space Semantics

**Перед изменением action space, wrappers, risk_guard:**

- [ ] **Проверить**: `ActionProto.volume_frac` означает **TARGET position**, НЕ DELTA
- [ ] **Проверить**: Action space bounds = `[-1, 1]` (не [0, 1])
- [ ] **Проверить**: LongOnlyWrapper использует **mapping**, не clipping
- [ ] **Запустить**: `pytest tests/test_critical_action_space_fixes.py -v`
- [ ] **Проверить**: Все тесты проходят

**Критические паттерны:**
```python
# ✅ ПРАВИЛЬНО: TARGET semantics
next_units = volume_frac * max_position

# ❌ НЕПРАВИЛЬНО: DELTA semantics (удвоение позиции!)
next_units = current_units + volume_frac * max_position

# ✅ ПРАВИЛЬНО: Mapping для long-only
mapped_action = (action + 1.0) / 2.0  # [-1,1] → [0,1]

# ❌ НЕПРАВИЛЬНО: Clipping (теряет reduction сигналы)
clipped_action = max(0, action)
```

**Симптомы регрессии:**
- Position doubling в live trading
- Policy не может reduce positions
- Unexpected long-only behavior

**См.**: [CRITICAL_FIXES_COMPLETE_REPORT.md](CRITICAL_FIXES_COMPLETE_REPORT.md)

---

### ✅ 4. Numerical Stability

**Перед изменением математических вычислений (variance, loss accumulation):**

- [ ] **Проверить**: Epsilon guards на division (`denominator + 1e-8`)
- [ ] **Проверить**: Explained variance использует epsilon guards
- [ ] **Проверить**: Нет катастрофической cancellation в `(x - mean)²`
- [ ] **Рассмотреть**: Использование float64 для критических вычислений
- [ ] **Рассмотреть**: Kahan summation для длинных accumulations

**Best practices:**
```python
# ✅ ПРАВИЛЬНО: Epsilon guard
std = np.std(values) + 1e-8
normalized = (values - mean) / std

# ❌ НЕПРАВИЛЬНО: Division by zero risk
normalized = (values - mean) / np.std(values)

# ✅ ПРАВИЛЬНО: Epsilon guard в explained variance
denom = max(denom_raw, 1e-12)
if denom_raw <= 0.0 or not math.isfinite(denom_raw):
    return float("nan")
```

**Симптомы регрессии:**
- Division by zero errors
- NaN в explained variance
- Численная нестабильность

**См.**: [NUMERICAL_ISSUES_FIX_SUMMARY.md](NUMERICAL_ISSUES_FIX_SUMMARY.md) - Issue #5, #6

---

### ✅ 5. Test Coverage

**Перед коммитом изменений:**

- [ ] **Запустить**: Все существующие тесты
- [ ] **Добавить**: Новые тесты для новой функциональности
- [ ] **Проверить**: Регрессионные тесты проходят
- [ ] **Проверить**: Test coverage не снизился

**Критические test suites:**
```bash
# LSTM state reset
pytest tests/test_lstm_episode_boundary_reset.py -v

# NaN handling
pytest tests/test_nan_handling_external_features.py -v

# Action space
pytest tests/test_critical_action_space_fixes.py -v

# Distributional PPO
pytest tests/test_distributional_ppo*.py -v

# Все тесты
pytest tests/ -v --tb=short
```

**Minimum passing criteria:**
- ✅ Все regression tests проходят
- ✅ No new warnings о deprecated functions
- ✅ Test coverage ≥ текущему уровню

---

## 🔍 CODE REVIEW CHECKLIST

### Для reviewer (человека или AI):

#### LSTM-Related Changes
- [ ] LSTM state reset вызывается на episode boundaries
- [ ] Нет изменений, которые могут пропустить reset
- [ ] Тесты `test_lstm_episode_boundary_reset.py` проходят

#### Feature Engineering Changes
- [ ] NaN values обрабатываются корректно
- [ ] Нет silent NaN → 0.0 без документации
- [ ] Тесты `test_nan_handling_external_features.py` проходят

#### Action Space Changes
- [ ] Семантика TARGET/DELTA соблюдена
- [ ] Action bounds = [-1, 1]
- [ ] LongOnlyWrapper использует mapping
- [ ] Тесты `test_critical_action_space_fixes.py` проходят

#### Numerical Stability
- [ ] Epsilon guards на всех division operations
- [ ] Нет катастрофической cancellation
- [ ] Float precision адекватна (float32 vs float64)

#### Documentation
- [ ] Критические изменения задокументированы
- [ ] CLAUDE.md обновлен если нужно
- [ ] Добавлены комментарии для non-obvious решений

---

## 🚨 RED FLAGS - Немедленно остановиться!

### Если вы видите эти изменения БЕЗ явного обоснования:

1. **Удаление вызова `_reset_lstm_states_for_done_envs()`**
   - ⛔ STOP! Это критическая регрессия
   - 📖 См.: [CRITICAL_LSTM_RESET_FIX_REPORT.md](CRITICAL_LSTM_RESET_FIX_REPORT.md)

2. **Изменение NaN handling в `_clipf()` или `_get_safe_float()`**
   - ⛔ STOP! Убедитесь что понимаете design decision
   - 📖 См.: obs_builder.pyx:14-29 для обоснования

3. **Изменение ActionProto.volume_frac семантики**
   - ⛔ STOP! TARGET → DELTA приведет к position doubling
   - 📖 См.: [CRITICAL_FIXES_COMPLETE_REPORT.md](CRITICAL_FIXES_COMPLETE_REPORT.md)

4. **Удаление epsilon guards (`+ 1e-8`, `+ 1e-12`)**
   - ⛔ STOP! Риск division by zero
   - 📖 См.: distributional_ppo.py:255-258

5. **Изменение action space bounds [−1,1] → [0,1]**
   - ⛔ STOP! Потеря short/reduction capability
   - 📖 См.: CLAUDE.md - Критические правила

---

## 📚 Обязательное чтение перед критическими изменениями

### Документы для изучения:

1. **[CLAUDE.md](CLAUDE.md)** - Главная документация проекта
   - Раздел: "Критические правила (НЕ НАРУШАТЬ!)"
   - Раздел: "КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ - ОБЯЗАТЕЛЬНО К ПРОЧТЕНИЮ"

2. **[NUMERICAL_ISSUES_FIX_SUMMARY.md](NUMERICAL_ISSUES_FIX_SUMMARY.md)**
   - Issue #4: LSTM State Reset
   - Issue #2: NaN Handling
   - Другие numerical issues

3. **[CRITICAL_LSTM_RESET_FIX_REPORT.md](CRITICAL_LSTM_RESET_FIX_REPORT.md)**
   - Полная документация LSTM fix
   - Academic references
   - Expected impact

4. **[CRITICAL_FIXES_COMPLETE_REPORT.md](CRITICAL_FIXES_COMPLETE_REPORT.md)**
   - Action space fixes
   - Position doubling prevention

---

## 🧪 Regression Test Suite

### Запуск всех регрессионных тестов:

```bash
# Comprehensive regression test suite
pytest tests/test_lstm_episode_boundary_reset.py \
       tests/test_nan_handling_external_features.py \
       tests/test_critical_action_space_fixes.py \
       tests/test_distributional_ppo*.py \
       -v --tb=short

# Expected results:
# - 35+ tests passing
# - 0-1 tests skipped (Cython modules ok)
# - 0 tests failed ❌ (если есть failures - не коммитить!)
```

### Continuous Integration:

**Добавить в CI pipeline:**
```yaml
# .github/workflows/regression_tests.yml
name: Regression Tests

on: [push, pull_request]

jobs:
  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run regression tests
        run: |
          pytest tests/test_lstm_episode_boundary_reset.py -v
          pytest tests/test_nan_handling_external_features.py -v
          pytest tests/test_critical_action_space_fixes.py -v
      - name: Fail on regression
        if: failure()
        run: exit 1
```

---

## 🎓 Learning from Past Mistakes

### История критических проблем:

| Дата | Проблема | Impact | Урок |
|------|----------|--------|------|
| 2025-11-21 | LSTM states not reset | 5-15% accuracy loss | Всегда сбрасывать LSTM на boundaries |
| 2025-11-21 | NaN silent conversion | Semantic ambiguity | Explicit logging для debugging |
| 2025-11-21 | Position doubling | Critical in live | TARGET semantics строго |
| 2025-11-20 | Quantile loss inverted | 10% worse performance | Unit tests для loss functions |
| 2025-11-20 | Cross-symbol contamination | Multi-symbol bias | Per-symbol normalization |

### Общие темы (patterns):

1. **Temporal leakage** - Состояние не сбрасывается → contamination
2. **Silent failures** - NaN/0.0 conversions без warnings
3. **Semantic ambiguity** - Один value означает разные вещи
4. **Test coverage gaps** - Критическая логика без тестов

### Правила предотвращения:

1. **Explicit > Implicit** - Явно документировать design decisions
2. **Test Everything** - Особенно edge cases и boundaries
3. **Log Ambiguities** - Если есть semantic ambiguity, добавить logging
4. **Review History** - Проверить was this fixed before?

---

## 📞 Контакты и Эскалация

### Если нашли регрессию:

1. **Немедленно**: Откатить изменения если в production
2. **Создать**: Bug report с reference на этот checklist
3. **Добавить**: Regression test для предотвращения повторения
4. **Обновить**: Этот checklist если нужно

### Если не уверены:

1. **Проверить**: Документацию (CLAUDE.md, fix reports)
2. **Запустить**: Regression tests
3. **Спросить**: Team lead или senior developer
4. **НЕ**: Коммитить пока не уверены

---

## ✅ Final Checklist Before Commit

**Я подтверждаю что:**

- [ ] Прочитал этот checklist полностью
- [ ] Запустил все regression tests (все проходят)
- [ ] Проверил критические patterns (LSTM, NaN, action space)
- [ ] Добавил тесты для новой функциональности
- [ ] Обновил документацию если нужно
- [ ] Нет red flags в моих изменениях
- [ ] Code review пройден (или self-review для малых изменений)

**Commit message включает** (если релевантно):
- `fix:` для bugfixes
- `test:` для новых тестов
- `docs:` для документации
- Reference на issue/report если есть

---

**Дата последнего обновления**: 2025-11-21
**Версия**: 1.0
**Следующий review**: Каждые 3 месяца или после критических fixes

**Этот документ является живым** - обновляйте его при обнаружении новых регрессий!

---

**End of Checklist**

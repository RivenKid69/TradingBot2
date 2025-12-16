# ACTION SPACE - CRITICAL REFERENCE GUIDE
## ⚠️ ОБЯЗАТЕЛЬНО К ПРОЧТЕНИЮ ПЕРЕД ИЗМЕНЕНИЕМ ACTION SPACE ЛОГИКИ

**Дата последнего обновления**: 2025-11-21
**Статус**: PRODUCTION CRITICAL - НЕ ИЗМЕНЯТЬ БЕЗ ПОНИМАНИЯ

---

## 🚨 КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ

**ТРИ критических бага были исправлены 2025-11-21. НЕ ОТКАТЫВАЙТЕ эти исправления!**

Если вы:
- Изменяете `ActionProto.volume_frac` семантику
- Модифицируете `LongOnlyActionWrapper`
- Трогаете `risk_guard.py` position logic
- Меняете action space bounds

**ОСТАНОВИТЕСЬ И ПРОЧИТАЙТЕ ЭТО ПЕРВЫМ!**

---

## 📋 Основной контракт (НЕ НАРУШАТЬ)

### ActionProto.volume_frac Semantics

```python
# ✅ ПРАВИЛЬНАЯ семантика (с 2025-11-21):
volume_frac ∈ [-1.0, 1.0]  # TARGET position as fraction of max_position

Интерпретация:
  > 0: LONG target  (e.g., 0.5 = target 50% long)
  < 0: SHORT target (e.g., -0.5 = target 50% short)
  = 0: FLAT (no position, 100% cash)

КРИТИЧЕСКИ ВАЖНО: Это TARGET (желаемое состояние), НЕ DELTA (изменение)!
```

### Execution Layer Responsibility

```python
# Execution layer ДОЛЖЕН вычислять delta:
current_position = state.units
target_position = volume_frac * max_position
delta = target_position - current_position  # ← Execution вычисляет delta

if delta > 0:
    side = "BUY"
    quantity = abs(delta)
elif delta < 0:
    side = "SELL"
    quantity = abs(delta)
else:
    # No action needed
    pass
```

---

## 🔴 КРИТИЧЕСКИЕ БАГИ (ИСПРАВЛЕНЫ - НЕ ПОВТОРЯТЬ!)

### Bug #1: Sign Convention Mismatch (FIXED 2025-11-21)

**Что было (НЕПРАВИЛЬНО)**:
```python
# ❌ BUGGY CODE (до 2025-11-21):
class LongOnlyActionWrapper:
    def action(self, action):
        # Negative actions clipped to 0.0 → SIGNAL LOSS!
        return np.clip(action, 0.0, 1.0)  # ❌ Теряет reduction сигналы
```

**Что стало (ПРАВИЛЬНО)**:
```python
# ✅ FIXED CODE (с 2025-11-21):
class LongOnlyActionWrapper:
    def action(self, action):
        # Linear mapping preserves information
        mapped = (action + 1.0) / 2.0  # ✅ [-1,1] → [0,1]
        return np.clip(mapped, 0.0, 1.0)
```

**Почему это важно**:
- Policy может выражать "reduce position" через negative outputs
- Long-only означает "no shorts", НЕ "no reductions"

---

### Bug #2: Position Doubling (FIXED 2025-11-21) - **САМЫЙ КРИТИЧНЫЙ**

**Что было (НЕПРАВИЛЬНО)**:
```python
# ❌ BUGGY CODE (до 2025-11-21):
def on_action_proposed(self, state, proto):
    # DELTA interpretation → POSITION DOUBLING!
    delta_units = proto.volume_frac * max_position
    next_units = state.units + delta_units  # ❌ Adds to current!

    # Example: current=50, volume_frac=0.5, max=100
    # Bug: next = 50 + 50 = 100 (DOUBLES on repeat!)
```

**Что стало (ПРАВИЛЬНО)**:
```python
# ✅ FIXED CODE (с 2025-11-21):
def on_action_proposed(self, state, proto):
    # TARGET interpretation → NO DOUBLING
    target_units = proto.volume_frac * max_position
    next_units = target_units  # ✅ Direct target, not adding!

    # Example: current=50, volume_frac=0.5, max=100
    # Correct: next = 50 (maintains position)
```

**Почему это критично**:
- Повторные одинаковые actions НЕ должны накапливаться
- В live trading это привело бы к 2x leverage violations!
- Risk guard ДОЛЖЕН видеть TARGET, не DELTA

---

### Bug #3: Action Space Range Mismatch (FIXED 2025-11-21)

**Что было (НЕПРАВИЛЬНО)**:
```python
# ❌ BUGGY CODE (до 2025-11-21):
# В разных местах разные bounds!

# action_proto.py:
# Contract: volume_frac ∈ [-1, 1]

# trading_patchnew.py:
if scalar < 0.0 or scalar > 1.0:
    scalar = np.clip(scalar, 0.0, 1.0)  # ❌ Clips to [0,1]!

# risk_guard.py:
# Expects [-1, 1]  # ❌ Mismatch!
```

**Что стало (ПРАВИЛЬНО)**:
```python
# ✅ FIXED CODE (с 2025-11-21):
# Унифицировано [-1, 1] ВЕЗДЕ

# action_proto.py:
# Contract: volume_frac ∈ [-1, 1] ✅

# trading_patchnew.py:
if scalar < -1.0 or scalar > 1.0:
    scalar = np.clip(scalar, -1.0, 1.0)  # ✅ [-1,1]

# risk_guard.py:
# Expects [-1, 1] ✅ Consistent
```

**Почему это важно**:
- Architectural consistency
- Все компоненты должны иметь одинаковое понимание
- Избегает silent bugs

---

## ✅ Правильные паттерны (СЛЕДОВАТЬ)

### Pattern 1: Interpreting volume_frac

```python
# ✅ ПРАВИЛЬНО - TARGET semantics
def calculate_target_position(volume_frac, max_position):
    """Calculate TARGET position from volume_frac."""
    return volume_frac * max_position

# ❌ НЕПРАВИЛЬНО - DELTA semantics (УСТАРЕЛО!)
def calculate_position_change(volume_frac, max_position, current):
    """DO NOT USE - causes position doubling!"""
    delta = volume_frac * max_position
    return current + delta  # ❌ WRONG!
```

### Pattern 2: Long-Only Transformation

```python
# ✅ ПРАВИЛЬНО - Linear mapping
def map_to_long_only(action):
    """Map [-1, 1] to [0, 1] preserving information."""
    return (action + 1.0) / 2.0

# Example:
# -1.0 → 0.0 (full exit)
# -0.5 → 0.25 (25% long)
#  0.0 → 0.5 (50% long)
#  0.5 → 0.75 (75% long)
#  1.0 → 1.0 (100% long)

# ❌ НЕПРАВИЛЬНО - Simple clipping (УСТАРЕЛО!)
def clip_to_long_only(action):
    """DO NOT USE - loses reduction signals!"""
    return max(0.0, action)  # ❌ All negatives → 0!
```

### Pattern 3: Bounds Enforcement

```python
# ✅ ПРАВИЛЬНО - Uniform [-1, 1]
def validate_action(action):
    """Enforce [-1, 1] bounds uniformly."""
    return np.clip(action, -1.0, 1.0)

# ❌ НЕПРАВИЛЬНО - Mixed bounds (УСТАРЕЛО!)
def validate_action_wrong(action):
    """DO NOT USE - inconsistent with contract!"""
    return np.clip(action, 0.0, 1.0)  # ❌ Wrong range!
```

---

## 🧪 Обязательные тесты перед изменениями

**ВСЕГДА запускайте эти тесты после изменения action space логики**:

```bash
# Полный test suite для action space
pytest tests/test_critical_action_space_fixes.py -v

# Критические regression тесты:
pytest tests/test_critical_action_space_fixes.py::TestTargetPositionSemantics::test_risk_guard_prevent_position_doubling -v
pytest tests/test_critical_action_space_fixes.py::TestIntegrationSemantics::test_repeated_actions_no_accumulation -v
pytest tests/test_critical_action_space_fixes.py::TestLongOnlyWrapperFix::test_negative_to_reduction_mapping -v
```

**Ожидаемый результат**: 21/21 passed, 2 skipped

**Если тесты падают** - ВЫ СЛОМАЛИ КРИТИЧЕСКУЮ ЛОГИКУ! Откатите изменения.

---

## 📚 Дополнительная документация

- **Полный анализ**: [CRITICAL_ACTION_SPACE_ISSUES_ANALYSIS.md](../CRITICAL_ACTION_SPACE_ISSUES_ANALYSIS.md)
- **Отчёт о исправлениях**: [CRITICAL_FIXES_COMPLETE_REPORT.md](../CRITICAL_FIXES_COMPLETE_REPORT.md)
- **Тесты**: [tests/test_critical_action_space_fixes.py](../tests/test_critical_action_space_fixes.py)

---

## 🔍 Checklist перед изменениями

Перед изменением action space логики, ответьте на эти вопросы:

- [ ] Я прочитал [CRITICAL_FIXES_COMPLETE_REPORT.md](../CRITICAL_FIXES_COMPLETE_REPORT.md)?
- [ ] Я понимаю разницу между TARGET и DELTA semantics?
- [ ] Я знаю почему position doubling критичен?
- [ ] Я понимаю почему long-only нужен mapping, не clipping?
- [ ] Я запустил `pytest tests/test_critical_action_space_fixes.py` перед изменениями?
- [ ] Я знаю что делать если тесты упадут после моих изменений?

**Если хотя бы на один вопрос "НЕТ" - НЕ ИЗМЕНЯЙТЕ КОД!**

---

## 🆘 Что делать если что-то сломалось

### Признаки position doubling бага:

- Position растёт экспоненциально при одинаковых actions
- Position violations в risk_guard при нормальных actions
- Leverage в 2x+ при target 1x

**Решение**: Проверьте используется ли TARGET semantics (не DELTA!)

### Признаки signal loss в long-only:

- Policy всегда держит position, никогда не выходит
- Max drawdown аномально высокий
- Policy не реагирует на stop signals

**Решение**: Проверьте используется ли mapping (не clipping!)

### Признаки action space mismatch:

- Silent bugs при граничных values
- Inconsistent behavior между компонентами
- Unexpected action clipping

**Решение**: Унифицируйте bounds к [-1, 1] везде

---

**Последнее обновление**: 2025-11-21
**Ответственный**: System Architecture Team
**Критичность**: PRODUCTION CRITICAL
**Статус**: ✅ FIXES VERIFIED AND TESTED

**НЕ ИЗМЕНЯЙТЕ БЕЗ REVIEW И ПОНИМАНИЯ ПОСЛЕДСТВИЙ!**

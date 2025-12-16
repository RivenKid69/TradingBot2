# AI-Powered Quantitative Research Platform Documentation Maintenance Guide

> **Руководство по поддержке и актуализации документации проекта**

**Версия**: 1.0
**Дата**: 2025-11-22
**Статус**: ✅ Active

---

## 🎯 Назначение

Этот документ описывает:
- Как организована документация проекта
- Как правильно обновлять документацию
- Checklist для предотвращения устаревания
- Связи между документами

---

## 📁 Структура документации

### 1. Корневая документация (Root Level)

#### Главные документы (ВСЕГДА актуальные)
- **[README.md](README.md)** - Обзор проекта, quick start, статус
- **[CLAUDE.md](CLAUDE.md)** - **Полная документация** (Russian, master reference)
- **[DOCS_INDEX.md](DOCS_INDEX.md)** - Навигационный hub для всей документации
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Архитектура системы
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Руководство для contributors
- **[CHANGELOG.md](CHANGELOG.md)** - История изменений

#### Quick References
- **[QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md)** - Быстрый старт
- **[FILE_REFERENCE.md](FILE_REFERENCE.md)** - Справочник файлов
- **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)** - Инструкции по сборке
- **[VERIFICATION_INSTRUCTIONS.md](VERIFICATION_INSTRUCTIONS.md)** - Инструкции по верификации

### 2. Критические отчеты (Root Level) - ⭐ ПРИОРИТЕТ

**Правило**: Критические исправления ВСЕГДА документируются в корне

#### Action Space Fixes (2025-11-21)
- **[CRITICAL_FIXES_COMPLETE_REPORT.md](CRITICAL_FIXES_COMPLETE_REPORT.md)** - 3 action space bugs
- **[CRITICAL_ACTION_SPACE_ISSUES_ANALYSIS.md](CRITICAL_ACTION_SPACE_ISSUES_ANALYSIS.md)** - Детальный анализ

#### LSTM & NaN Fixes (2025-11-21)
- **[NUMERICAL_ISSUES_FIX_SUMMARY.md](NUMERICAL_ISSUES_FIX_SUMMARY.md)** - LSTM + NaN comprehensive summary
- **[CRITICAL_LSTM_RESET_FIX_REPORT.md](CRITICAL_LSTM_RESET_FIX_REPORT.md)** - LSTM state reset fix
- **[FIX_REPORT_NAN_HANDLING_2025_11_21.md](FIX_REPORT_NAN_HANDLING_2025_11_21.md)** - NaN handling fix
- **[FINAL_FIX_SUMMARY_2025_11_21.md](FINAL_FIX_SUMMARY_2025_11_21.md)** - Final comprehensive report

#### Twin Critics (2025-11-21, 2025-11-22)
- **[TWIN_CRITICS_GAE_FIX_REPORT.md](TWIN_CRITICS_GAE_FIX_REPORT.md)** - GAE computation fix
- **[TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md](TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md)** - ⭐ VF Clipping verification (LATEST)
- **[TWIN_CRITICS_VF_CLIPPING_COMPLETE_REPORT.md](TWIN_CRITICS_VF_CLIPPING_COMPLETE_REPORT.md)** - Complete implementation
- **[TWIN_CRITICS_VF_CLIPPING_FIX_REPORT.md](TWIN_CRITICS_VF_CLIPPING_FIX_REPORT.md)** - Fix report
- **[TWIN_CRITICS_VF_ALL_MODES_IMPLEMENTATION.md](TWIN_CRITICS_VF_ALL_MODES_IMPLEMENTATION.md)** - All VF modes
- **[BUG_ANALYSIS_TWIN_CRITICS_VF_CLIPPING.md](BUG_ANALYSIS_TWIN_CRITICS_VF_CLIPPING.md)** - Bug analysis
- **[FIX_DESIGN_TWIN_CRITICS_VF_CLIPPING.md](FIX_DESIGN_TWIN_CRITICS_VF_CLIPPING.md)** - Fix design
- **[TWIN_CRITICS_VF_CLIPPING_QUICKSTART.md](TWIN_CRITICS_VF_CLIPPING_QUICKSTART.md)** - Quick start
- **[TWIN_CRITICS_VF_CLIPPING_STATUS.md](TWIN_CRITICS_VF_CLIPPING_STATUS.md)** - Status tracking

#### UPGD Optimizer (2025-11-21)
- **[UPGD_NEGATIVE_UTILITY_FIX_REPORT.md](UPGD_NEGATIVE_UTILITY_FIX_REPORT.md)** - Negative utility scaling fix

#### Feature & Volatility Fixes (2025-11-20)
- **[CRITICAL_FIXES_REPORT.md](CRITICAL_FIXES_REPORT.md)** - 3 feature engineering bugs
- **[CRITICAL_FIXES_5_REPORT.md](CRITICAL_FIXES_5_REPORT.md)** - 5 numerical stability bugs

#### Regression Prevention
- **[REGRESSION_PREVENTION_CHECKLIST.md](REGRESSION_PREVENTION_CHECKLIST.md)** - ⭐ Обязательный checklist

#### Integration Reports
- **[INTEGRATION_SUCCESS_REPORT.md](INTEGRATION_SUCCESS_REPORT.md)** - ✅ Integration success

### 3. Техническая документация (docs/)

#### docs/ - Основные темы
- **docs/UPGD_INTEGRATION.md** - UPGD optimizer integration
- **docs/twin_critics.md** - Twin critics architecture
- **docs/seasonality*.md** - Seasonality framework (10 files)
- **docs/pipeline.md** - Decision pipeline
- **docs/bar_execution.md** - Bar execution mode
- **docs/large_orders.md** - Large order execution
- **docs/parallel.md** - Parallel environments
- **docs/eval.md** - Model evaluation

#### docs/reports/ - Отчеты по категориям

##### docs/reports/analysis/
- Анализы данных, алгоритмов, системы
- Примеры: ANALYSIS_DATA_DISTORTIONS_FULL.md, PARKINSON_ANALYSIS_MATHEMATICAL.md

##### docs/reports/audits/
- Аудиты компонентов, документации, интеграций
- Примеры: DOCUMENTATION_AUDIT_2025-11-11.md, FEATURE_AUDIT_REPORT.md

##### docs/reports/bugs/
- Исторические bug reports
- Примеры: BUG_REPORT_RSI_NAN.md, PARKINSON_ERROR_CORRECTION.md

##### docs/reports/features/
- Feature mappings, анализы, документация
- Примеры: FEATURE_MAPPING_63.md, GARCH_FEATURE.md

##### docs/reports/fixes/
- Детальные отчеты о исправлениях (не критических)
- Примеры: DISTRIBUTIONAL_VF_CLIPPING_FIX.md, QUANTILE_LOSS_FIX.md

##### docs/reports/integration/
- Миграции, интеграции, PBT/Adversarial
- Примеры: MIGRATION_GUIDE_62_TO_63.md, PBT_ADVERSARIAL_INTEGRATION_REPORT.md

##### docs/reports/tests/
- Тестирование, верификация, coverage
- Примеры: TEST_COVERAGE_REPORT.md, COMPREHENSIVE_TEST_VALIDATION_REPORT.md

##### docs/reports/self_review/
- Само-аудиты, критические ревью
- Примеры: SELF_REVIEW_CRITICAL_BUGS_FOUND.md

##### docs/reports/summaries/
- Общие summaries, changes
- Примеры: CHANGES_SUMMARY.md

##### docs/reports/upgd_vgs/
- UPGD и VGS специфичные отчеты
- Примеры: UPGD_VGS_FIX_DESIGN.md, VGS_DEEP_ANALYSIS_REPORT.md

##### docs/reports/twin_critics/
- Twin Critics отчеты (исторические)
- Примеры: TWIN_CRITICS_COMPREHENSIVE_AUDIT_REPORT.md

#### docs/archive/ - Устаревшие документы
- **docs/archive/documentation_meta/** - Старые индексы
- **docs/archive/uncategorized/** - Несортированные устаревшие файлы

---

## 📋 Правила именования

### Критические отчеты (Root Level)

**Шаблон**: `<TYPE>_<COMPONENT>_<SUBJECT>_<STATUS>.md`

**Примеры**:
- `CRITICAL_FIXES_COMPLETE_REPORT.md` - Complete report о critical fixes
- `TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md` - Verification report
- `NUMERICAL_ISSUES_FIX_SUMMARY.md` - Summary report

**Правило**:
- Критические исправления → **КРИТИЧЕСКИЙ_ТИП** в начале (CRITICAL_, NUMERICAL_, TWIN_CRITICS_)
- Дата в имени файла НЕ нужна (дата в метаданных документа)
- Status в конце: `_REPORT`, `_SUMMARY`, `_VERIFICATION`, `_FIX`, `_STATUS`

### Отчеты в docs/reports/

**Правило**: Имена в UPPER_CASE с категорией

**Примеры**:
- `docs/reports/analysis/PARKINSON_ANALYSIS_MATHEMATICAL.md`
- `docs/reports/fixes/QUANTILE_LOSS_FIX.md`
- `docs/reports/audits/DOCUMENTATION_AUDIT_2025-11-11.md`

### Техническая документация (docs/)

**Правило**: Имена в lowercase, snake_case предпочтительнее

**Примеры**:
- `docs/seasonality.md`
- `docs/twin_critics.md`
- `docs/bar_execution.md`

---

## 🔄 Как обновлять документацию

### При внесении критического исправления

1. **Создать отчет в корне**:
   ```
   CRITICAL_<COMPONENT>_<SUBJECT>_REPORT.md
   ```

2. **Обновить CLAUDE.md**:
   - Добавить в раздел "⚠️ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ"
   - Обновить таблицу "Частые ошибки и их решения"
   - Добавить в "Production Checklist"
   - Обновить "Test Coverage" numbers

3. **Обновить DOCS_INDEX.md**:
   - Добавить в раздел "🔥 CRITICAL - READ FIRST"
   - Обновить test coverage stats

4. **Обновить README.md**:
   - Раздел "✅ Последние Критические Исправления"
   - Обновить "Test Coverage" line

5. **Обновить CHANGELOG.md**:
   - Добавить запись с датой и описанием

### При добавлении новой возможности

1. **Создать техническую документацию**:
   ```
   docs/<feature_name>.md
   ```

2. **Создать интеграционный отчет** (если требуется):
   ```
   docs/reports/integration/<FEATURE>_INTEGRATION_REPORT.md
   ```

3. **Обновить CLAUDE.md**:
   - Раздел "🚀 Продвинутые возможности"
   - Примеры конфигурации
   - CLI команды

4. **Обновить DOCS_INDEX.md**:
   - Соответствующий раздел

5. **Обновить config templates**:
   - Добавить примеры в YAML configs

### При обновлении тестов

1. **Запустить тесты и собрать статистику**:
   ```bash
   cd tests
   python -m pytest test_twin_critics*.py --collect-only 2>/dev/null | grep "<Function" | wc -l
   python -m pytest test_upgd*.py --collect-only 2>/dev/null | grep "<Function" | wc -l
   python -m pytest test_pbt*.py --collect-only 2>/dev/null | grep "<Function" | wc -l
   ```

2. **Обновить числа в документации**:
   - CLAUDE.md (раздел "Test Coverage")
   - DOCS_INDEX.md (раздел "Test Coverage")
   - README.md (line 24)

3. **Создать test report** (если много изменений):
   ```
   docs/reports/tests/<COMPONENT>_TEST_REPORT.md
   ```

### При refactoring/переименовании

1. **Обновить все ссылки**:
   - Использовать search&replace для всех `*.md` файлов
   - Проверить broken links

2. **Создать migration guide**:
   ```
   docs/reports/integration/MIGRATION_<OLD>_TO_<NEW>.md
   ```

3. **Обновить FILE_REFERENCE.md**

---

## ✅ Checklist перед commit

### Для критических изменений

- [ ] Создан отчет в корне с правильным именем
- [ ] Обновлен CLAUDE.md (5 мест)
- [ ] Обновлен DOCS_INDEX.md (раздел CRITICAL)
- [ ] Обновлен README.md (статус, coverage)
- [ ] Обновлен CHANGELOG.md
- [ ] Созданы/обновлены тесты
- [ ] Обновлены test coverage numbers
- [ ] Проверены все ссылки

### Для обычных изменений

- [ ] Создана/обновлена техническая документация
- [ ] Обновлен DOCS_INDEX.md
- [ ] Обновлены config examples (если нужно)
- [ ] Обновлены CLI examples (если нужно)
- [ ] Проверены ссылки в CLAUDE.md

### Для обновлений документации

- [ ] Проверена актуальность всех дат
- [ ] Обновлены "Last Updated" timestamps
- [ ] Проверены все внутренние ссылки
- [ ] Проверена consistency между документами
- [ ] Обновлен DOCS_INDEX.md (если добавлены новые файлы)

---

## 🔗 Связи между документами

### Master Reference Chain (Обязательная актуальность)

```
CLAUDE.md (Master)
  ↓ references
DOCS_INDEX.md (Navigation)
  ↓ references
README.md (Overview)
  ↓ references
Critical Reports (Root/*.md)
  ↓ references
Technical Docs (docs/*.md)
  ↓ references
Reports (docs/reports/*/*.md)
```

### Ключевые связи

**CLAUDE.md** (Master Documentation):
- Ссылается на: Все critical reports, docs/UPGD_INTEGRATION.md, docs/twin_critics.md
- Используется: Разработчиками, AI assistants для понимания проекта

**DOCS_INDEX.md** (Navigation Hub):
- Ссылается на: ВСЕ документы проекта
- Используется: Для навигации, поиска документов

**README.md** (Project Overview):
- Ссылается на: CLAUDE.md, DOCS_INDEX.md, critical reports
- Используется: Новыми пользователями, GitHub visitors

**Critical Reports** (Root/*.md):
- Ссылаются на: Code locations, test files, related reports
- Используются: При debugging, regression prevention, training

---

## 📊 Метрики документации

### Текущие числа (2025-11-22)

**Test Coverage**:
- Twin Critics: 207 total tests (49 VF Clipping correctness tests)
- UPGD: 126 tests
- PBT: 137 tests
- Critical fixes total: 101+ tests (98%+ pass rate)

**Documentation Files**:
- Root level: 30+ files (including critical reports)
- docs/ directory: 50+ technical documents
- docs/reports/: 200+ analysis/audit/fix reports
- Total: 280+ markdown files

**Critical Reports** (Root):
- Action Space: 2 reports
- LSTM & NaN: 4 reports
- Twin Critics: 9 reports
- UPGD: 1 report
- Feature/Volatility: 2 reports
- Integration: 1 report
- Regression Prevention: 1 report

---

## 🎯 Приоритеты поддержки

### Tier 1 - ВСЕГДА актуальны (обновлять немедленно)
1. CLAUDE.md
2. README.md
3. DOCS_INDEX.md
4. Critical Reports (Root/*.md)
5. REGRESSION_PREVENTION_CHECKLIST.md

### Tier 2 - Актуализировать при изменениях
1. ARCHITECTURE.md
2. QUICK_START_REFERENCE.md
3. docs/UPGD_INTEGRATION.md
4. docs/twin_critics.md
5. Config templates (configs/*.yaml)

### Tier 3 - Периодические обновления
1. FILE_REFERENCE.md
2. docs/reports/tests/*.md (test reports)
3. docs/reports/integration/*.md
4. CHANGELOG.md (append-only)

### Tier 4 - Архивные (только для reference)
1. docs/archive/**/*.md
2. docs/reports/bugs/*.md (исторические)
3. docs/reports/audits/*.md (старые аудиты)

---

## 🔍 Как проверить актуальность

### Автоматические проверки

```bash
# 1. Проверить существование всех критических отчетов
for file in CRITICAL_FIXES_COMPLETE_REPORT.md \
            NUMERICAL_ISSUES_FIX_SUMMARY.md \
            UPGD_NEGATIVE_UTILITY_FIX_REPORT.md \
            CRITICAL_LSTM_RESET_FIX_REPORT.md \
            TWIN_CRITICS_GAE_FIX_REPORT.md \
            TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md; do
  if [ ! -f "$file" ]; then
    echo "MISSING: $file"
  fi
done

# 2. Проверить test coverage
cd tests
echo "Twin Critics tests:"
python -m pytest test_twin_critics*.py --collect-only 2>/dev/null | grep "<Function" | wc -l
echo "UPGD tests:"
python -m pytest test_upgd*.py --collect-only 2>/dev/null | grep "<Function" | wc -l
echo "PBT tests:"
python -m pytest test_pbt*.py --collect-only 2>/dev/null | grep "<Function" | wc -l

# 3. Проверить broken links (требует tools)
# markdown-link-check *.md
```

### Ручные проверки

1. **Проверить даты**:
   - Все "Last Updated" соответствуют недавним изменениям?
   - Даты в CLAUDE.md соответствуют git log?

2. **Проверить test numbers**:
   - Запустить pytest --collect-only
   - Сравнить с numbers в CLAUDE.md

3. **Проверить consistency**:
   - CLAUDE.md статус === README.md статус?
   - DOCS_INDEX.md links === существующие files?

4. **Проверить актуальность примеров**:
   - CLI commands работают?
   - Config examples соответствуют actual configs?

---

## 🚨 Частые ошибки

### 1. Забыли обновить test coverage numbers
**Решение**: Всегда запускать `pytest --collect-only` и обновлять numbers в:
- CLAUDE.md
- DOCS_INDEX.md
- README.md

### 2. Broken links после переименования
**Решение**: Search&replace во всех `.md` файлах:
```bash
grep -r "old_filename.md" *.md docs/*.md docs/**/*.md
```

### 3. Duplicate information в разных файлах
**Решение**: Использовать single source of truth:
- CLAUDE.md - master reference
- Другие файлы - ссылаются на CLAUDE.md

### 4. Устаревшие даты "Last Updated"
**Решение**: Автоматизировать через git hooks или CI

### 5. Критический отчет создан в docs/reports/ вместо root
**Решение**: Critical reports ВСЕГДА в root, остальные в docs/reports/

---

## 📝 Templates

### Template: Critical Report

```markdown
# <Component> <Subject> Fix Report

**Date**: YYYY-MM-DD
**Status**: ✅ FIXED | 🔄 IN PROGRESS | ⚠️ PARTIAL
**Criticality**: CRITICAL | HIGH | MEDIUM | LOW

---

## 🔍 Problem Description

[Describe the bug/issue]

**Impact**:
- [What was affected]
- [How severe]

**Root Cause**:
- [Why it happened]

---

## ✅ Solution

**Changes Made**:
1. [File: line numbers] - [What changed]
2. [File: line numbers] - [What changed]

**Verification**:
- Tests created: X tests
- Tests passed: X/X (XX%)

---

## ⚠️ Action Required

- [ ] Models trained before YYYY-MM-DD → **RETRAIN RECOMMENDED/REQUIRED**
- [ ] Check [specific component]
- [ ] Run [specific tests]

---

## 📊 Test Coverage

**New Tests**:
- `tests/test_<component>_<subject>.py` - X tests

**Results**:
- X/X passed (XX%)

---

## 🔗 Related

- [Related Report 1]
- [Related Report 2]
- [Code: file.py:line]

---

**Last Updated**: YYYY-MM-DD
```

### Template: Technical Documentation

```markdown
# <Feature Name>

**Status**: ✅ Production Ready | 🔄 Beta | ⚠️ Experimental
**Version**: X.Y
**Last Updated**: YYYY-MM-DD

---

## Overview

[1-2 paragraph description]

---

## Configuration

```yaml
# Example configuration
feature:
  enabled: true
  param1: value1
  param2: value2
```

---

## Usage

```python
# Example code
```

```bash
# Example CLI
```

---

## API Reference

[Functions, classes, parameters]

---

## Examples

[Practical examples]

---

## Troubleshooting

[Common issues and solutions]

---

## Related Documentation

- [Related Doc 1]
- [Related Doc 2]
```

---

## 📅 Maintenance Schedule

### Daily (при наличии изменений)
- Обновление CHANGELOG.md при commit

### Weekly
- Проверка broken links
- Обновление test coverage numbers

### Monthly
- Аудит CLAUDE.md accuracy
- Проверка consistency между документами
- Архивирование устаревших отчетов

### Quarterly
- Полный аудит всей документации
- Reorganization если нужна
- Update templates

---

## 🎓 Best Practices

1. **One Source of Truth**: CLAUDE.md - master reference, остальные ссылаются
2. **Consistent Naming**: Следовать templates, не изобретать новые схемы
3. **Critical First**: Критические изменения в root, остальные в docs/
4. **Link Everything**: Использовать relative links, не hardcode paths
5. **Date Everything**: Всегда указывать "Last Updated"
6. **Test Numbers**: Всегда проверять перед commit
7. **Search Before Create**: Может документ уже существует?
8. **Archive Old**: Не удалять, переносить в docs/archive/

---

## 📞 Questions?

- Проверьте [DOCS_INDEX.md](DOCS_INDEX.md) - возможно документ уже существует
- Проверьте [CLAUDE.md](CLAUDE.md) - master reference
- Создайте issue если нашли inconsistency

---

**Maintained by**: Claude Code
**Last Updated**: 2025-11-22
**Version**: 1.0

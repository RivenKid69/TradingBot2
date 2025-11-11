# TradingBot2 - Индекс документации

Этот документ является справочником по всем аналитическим документам, созданным для проекта TradingBot2.

---

## Обзор документов

### 1. **PROJECT_STRUCTURE_ANALYSIS.md** (Главный документ)
**Размер**: ~14,000 слов
**Содержание**:
- Полный обзор проекта (что это, характеристики)
- Основная архитектура (слойная архитектура)
- Подробное описание всех компонентов:
  - Core Layer
  - Implementation Layer
  - Service Layer
  - Features & Observations
  - Training & RL Models
  - Execution & Simulation
  - Risk Management
  - Data & Configuration
  - Monitoring & Logging
- Структура директорий
- Главные entry points (точки входа)
- Поток данных (обучение, бэктестирование, инфоренс)
- Основные функции и возможности
- YAML конфигурация
- Технологические стеки
- Ключевые концепции и паттерны
- Процесс разработки и развертывания
- Основные файлы проекта (ТОП 20)
- Метрики и эффективность

**Для кого**: Для деталь ного понимания проекта, разработчиков, архитекторов

---

### 2. **ARCHITECTURE_DIAGRAM.md** (Визуальные диаграммы)
**Размер**: ~5,000 слов
**Содержание**:
- Общая архитектура системы (ASCII диаграмма)
- Поток обучения (Training Flow)
- Поток инфоренса (Inference Flow)
- Архитектура компонентов для симуляции и бэктестирования
- Архитектура признаков (56D Vector)
- Поток конфигурации (Configuration Flow)
- Интеграция Cython модулей (оптимизация)
- Summary: Главные компоненты

**Для кого**: Для быстрого визуального понимания, менеджеры, аналитики

---

### 3. **QUICK_START_REFERENCE.md** (Краткий справочник)
**Размер**: ~4,000 слов
**Содержание**:
- Что это такое (1 параграф)
- Главные файлы (меню) с примерами команд:
  - Обучение модели
  - Бэктестирование
  - Лайв-торговля
  - Расчет метрик
  - Загрузка данных
  - Полный цикл
- Архитектура слоев (3 строки кода)
- Таблица основных компонентов
- Признаки (56D Vector) - визуальная структура
- YAML конфигурация (примеры)
- Директории (структура дерева)
- Потоки обучения и инфоренса (простые диаграммы)
- Важные метрики
- Cython модули
- Типичные команды с параметрами
- Процесс разработки (6 шагов)
- Логирование и результаты
- ТОП-5 файлов для чтения
- Полезные ссылки
- Типичные ошибки и решения
- Типовой workflow (bash скрипт)

**Для кого**: Для быстрого старта, новых разработчиков, чит-листы

---

## Существующая документация в проекте

### Официальная документация
- **README.md** - основной README с примерами
- **ARCHITECTURE.md** - архитектура слоев (официальная)
- **CONTRIBUTING.md** - руководство по вкладу
- **CHANGELOG.md** - история изменений

### Специализированная документация
- **CODEBASE_STRUCTURE_ANALYSIS.md** - структура кодовой базы
- **FULL_FEATURES_LIST.md** - полный список признаков (57 vs 56)
- **FEATURE_MAPPING_56.md** - маппинг 56 признаков
- **OBSERVATION_MAPPING.md** - маппинг наблюдений
- **FILE_REFERENCE.md** - справка по файлам
- **DATASET_FIX_README.md** - исправления датасета

### Документация по компонентам
- **docs/bar_execution.md** - баровый режим исполнения
- **docs/pipeline.md** - decision pipeline архитектура
- **docs/seasonality.md** - сезонность ликвидности
- **docs/moving_average.md** - утилита скользящего среднего
- **docs/universe.md** - управление символами
- **docs/permissions.md** - права доступа и владение файлами

### Отчеты и анализы
- **TRAINING_PIPELINE_ANALYSIS.md** - анализ pipeline обучения
- **TRAINING_METRICS_ANALYSIS.md** - анализ метрик обучения
- **FEATURE_AUDIT_REPORT.md** - аудит признаков
- **SIZE_ANALYSIS.md** - анализ размеров компонентов
- **VERIFICATION_REPORT.md** - отчет верификации
- **DOCUMENTATION_AUDIT_2025-11-11.md** - аудит документации

### Специальные отчеты
- **ANALYSIS_4H_TIMEFRAME.md** - анализ 4h timeframe
- **GARCH_FEATURE.md** - GARCH признак
- **YANG_ZHANG_FIX_SUMMARY.md** - исправление Yang-Zhang
- **METRICS_QUICK_REFERENCE.txt** - быстрая справка по метрикам
- **METRICS_FIXES_SUMMARY.md** - сводка исправлений метрик

---

## Рекомендуемый порядок чтения

### Для новых разработчиков (День 1)
1. README.md (проект на одной странице)
2. QUICK_START_REFERENCE.md (быстрый старт)
3. ARCHITECTURE_DIAGRAM.md (визуальное понимание)

### Для углубленного изучения (День 2-3)
1. ARCHITECTURE.md (официальная архитектура)
2. PROJECT_STRUCTURE_ANALYSIS.md (полный анализ)
3. docs/pipeline.md (decision pipeline)

### Для специализированных задач
- **Обучение модели**: TRAINING_PIPELINE_ANALYSIS.md, FULL_FEATURES_LIST.md
- **Исполнение ордеров**: docs/bar_execution.md, QUICK_START_REFERENCE.md
- **Риск-менеджмент**: PROJECT_STRUCTURE_ANALYSIS.md (раздел Risk Management)
- **Признаки**: FEATURE_MAPPING_56.md, OBSERVATION_MAPPING.md
- **Данные**: docs/universe.md, DATASET_FIX_README.md

---

## Ключевые цифры проекта

| Метрика | Значение |
|---------|----------|
| Python файлов | ~410 |
| Строк кода | ~117,000 |
| Слоев архитектуры | 5 (Scripts, Strategies, Services, Impl, Core) |
| Основных компонентов | 50+ |
| Размер признаков | 56D |
| Cython модулей | 7+ |
| Тестов | 150+ |
| Конфигов YAML | 15+ |
| Сервисов | 7 основных |
| Реализаций исполнителей | 3+ (Bar, Sim, REST) |

---

## Главные компоненты (краткая справка)

| Компонент | Файл | Строк кода | Назначение |
|-----------|------|-----------|-----------|
| ML Модель | distributional_ppo.py | 454KB | Distributional PPO с CVaR |
| Entry Point Обучения | train_model_multi_patch.py | 220KB | Главный скрипт обучения |
| Исполнитель Сигналов | service_signal_runner.py | 386KB | Главный исполнитель |
| Симулятор | execution_sim.py | 562KB | Микро-симулятор ордеров |
| Признаки | feature_pipe.py | 35KB | Pipeline признаков |
| Проскальзывание | impl_slippage.py | 96KB | Модель slippage |
| LOB State | lob_state_cython.pyx | 57KB | Состояние книги заявок |
| Бэктестирование | service_backtest.py | 76KB | Сервис бэктеста |
| Задержки | impl_latency.py | 44KB | Модель latency |
| Конфигурация | core_config.py | 48KB | Загрузка YAML конфигов |

---

## Точки входа (Entry Points)

```bash
# ГЛАВНЫЕ ТОЧКИ ВХОДА

# 1. Обучение
python train_model_multi_patch.py --config configs/config_train.yaml

# 2. Бэктестирование
python script_backtest.py --config configs/config_sim.yaml

# 3. Инфоренс/Лайв
python script_live.py --config configs/config_live.yaml

# 4. Метрики
python script_eval.py --config configs/config_eval.yaml

# 5. Загрузка данных
python ingest_orchestrator.py --symbols BTCUSDT,ETHUSDT --interval 1m

# 6. Полный цикл
python scripts/run_full_cycle.py --symbols BTCUSDT,ETHUSDT --interval 1m,5m,15m
```

---

## Архитектурные слои

```
┌─────────────────────────────────────────┐
│ SCRIPTS (train_model_multi_patch.py)   │ <- Точки входа
├─────────────────────────────────────────┤
│ SERVICES (service_*.py)                 │ <- Бизнес-логика
├─────────────────────────────────────────┤
│ IMPLEMENTATIONS (impl_*.py)             │ <- Реализации
├─────────────────────────────────────────┤
│ CORE (core_*.py)                        │ <- Базовые модели
└─────────────────────────────────────────┘
```

---

## Процесс обучения (Training Pipeline)

```
Binance API
    ↓
Загрузка свечей (1m, 5m, 15m, 1h)
    ↓
Feature engineering (SMA, RSI, MACD, Yang-Zhang, CVD, GARCH)
    ↓
Подготовка датасета (train/val/test splits)
    ↓
ОБУЧЕНИЕ (Distributional PPO + Optuna HPO)
    ↓
Сохранение модели (artifacts/)
```

---

## Процесс инфоренса (Inference Pipeline)

```
Live (WebSocket) ИЛИ Historical (CSV/Parquet)
    ↓
Расчет 56D признаков (obs_builder.pyx)
    ↓
Инференс модели (Distributional PPO)
    ↓
Преобразование в OrderIntent
    ↓
Risk Guards (проверка позиций, drawdown)
    ↓
Исполнение (BarExecutor ИЛИ REST API)
    ↓
Логирование сделок
```

---

## Структура признаков (56D)

```
Bar (3)           │ Derived (2)    │ Indicators (13)
Microstructure(3) │ Agent (6)      │ Metadata (5)
External (21)     │ Token (3)      │

ИТОГО: 56 признаков
```

Описание каждого блока в QUICK_START_REFERENCE.md и PROJECT_STRUCTURE_ANALYSIS.md

---

## Метрики для оценки

- **Sharpe Ratio** - риск-скорректированная доходность
- **Sortino Ratio** - downside volatility adjusted
- **Maximum Drawdown** - максимальная просадка
- **Win Rate** - % прибыльных сделок
- **PnL** - прибыль/убыток
- **CVaR** - Conditional Value at Risk
- **Cumulative Return** - общий возврат

---

## Быстрые ссылки на документы

### Внутри проекта
- [PROJECT_STRUCTURE_ANALYSIS.md](/home/user/TradingBot2/PROJECT_STRUCTURE_ANALYSIS.md)
- [ARCHITECTURE_DIAGRAM.md](/home/user/TradingBot2/ARCHITECTURE_DIAGRAM.md)
- [QUICK_START_REFERENCE.md](/home/user/TradingBot2/QUICK_START_REFERENCE.md)
- [README.md](/home/user/TradingBot2/README.md)
- [ARCHITECTURE.md](/home/user/TradingBot2/ARCHITECTURE.md)

### Документация
- [docs/bar_execution.md](/home/user/TradingBot2/docs/bar_execution.md)
- [docs/pipeline.md](/home/user/TradingBot2/docs/pipeline.md)
- [docs/seasonality.md](/home/user/TradingBot2/docs/seasonality.md)

---

## Как использовать эту документацию

1. **Если вы новичок в проекте**:
   - Начните с QUICK_START_REFERENCE.md
   - Посмотрите ARCHITECTURE_DIAGRAM.md
   - Прочитайте ARCHITECTURE.md

2. **Если вам нужна детальная информация**:
   - Читайте PROJECT_STRUCTURE_ANALYSIS.md
   - Изучайте FULL_FEATURES_LIST.md
   - Смотрите CODEBASE_STRUCTURE_ANALYSIS.md

3. **Если вы ищете конкретный компонент**:
   - Используйте оглавление в PROJECT_STRUCTURE_ANALYSIS.md
   - Обратитесь к FILE_REFERENCE.md
   - Посмотрите INDEX в каждом документе

4. **Для процессов и workflows**:
   - Training: TRAINING_PIPELINE_ANALYSIS.md
   - Backtesting: docs/bar_execution.md + QUICK_START_REFERENCE.md
   - Live Trading: service_signal_runner.py (в коде) + docs/

---

## Лучшие практики чтения

### Метод 1: "Быстрый старт" (1 час)
1. QUICK_START_REFERENCE.md (30 мин)
2. ARCHITECTURE_DIAGRAM.md (30 мин)

### Метод 2: "Углубленное изучение" (4-6 часов)
1. README.md (20 мин)
2. QUICK_START_REFERENCE.md (30 мин)
3. ARCHITECTURE.md (40 мин)
4. PROJECT_STRUCTURE_ANALYSIS.md (2+ часа)
5. ARCHITECTURE_DIAGRAM.md (30 мин)
6. docs/pipeline.md (20 мин)

### Метод 3: "Специализированное путешествие"
- Обучение: FULL_FEATURES_LIST.md → TRAINING_PIPELINE_ANALYSIS.md → train_model_multi_patch.py
- Исполнение: QUICK_START_REFERENCE.md → docs/bar_execution.md → service_signal_runner.py
- Признаки: FEATURE_MAPPING_56.md → OBSERVATION_MAPPING.md → feature_pipe.py

---

## История создания документов

| Документ | Дата | Статус | Размер |
|----------|------|--------|--------|
| PROJECT_STRUCTURE_ANALYSIS.md | 2025-11-11 | Новый | ~14K слов |
| ARCHITECTURE_DIAGRAM.md | 2025-11-11 | Новый | ~5K слов |
| QUICK_START_REFERENCE.md | 2025-11-11 | Новый | ~4K слов |
| DOCUMENTATION_INDEX.md | 2025-11-11 | Новый | ~3K слов |

---

## Версионирование документации

- **Версия**: 1.0
- **Дата**: 2025-11-11
- **Ветка**: claude/update-documentation-011CV2SMDZCkyAyYPfB2ewet
- **Статус**: Актуальные

---

## Контрибьютинг в документацию

Если вы обновляете код проекта, пожалуйста обновите и документацию:

1. Обновите QUICK_START_REFERENCE.md для изменений в entry points
2. Обновите PROJECT_STRUCTURE_ANALYSIS.md для архитектурных изменений
3. Обновите ARCHITECTURE_DIAGRAM.md для потоков данных
4. Проверьте последовательность и консистентность всех документов

---

**Последнее обновление**: 2025-11-11 16:43 UTC
**Автор анализа**: Claude Code Agent
**Проект**: TradingBot2


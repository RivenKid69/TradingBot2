# Проверка технических документов на актуальность - Детальный отчет

**Дата проверки:** 2025-11-11  
**Проверено документов:** 11  
**Найдено проблем:** 5

---

## КРИТИЧЕСКИЕ ПРОБЛЕМЫ (2)

### 1. ❌ permissions.md - Отсутствует .github/workflows/

**Документ:** `/home/user/TradingBot2/docs/permissions.md`  
**Проблема:** Документ ссылается на `.github/workflows/` для CI/CD workflows, но директория не существует.

**Строка в документе:**
```markdown
- Continuous integration workflows in `.github/workflows/` — owned by the **DevOps team**.
```

**Проверка:**
```bash
ls -la /home/user/TradingBot2/.github/
# Результат: директория не найдена
```

**Влияние:** Средне - документация описывает структуру, которая не реализована в проекте.  
**Рекомендация:**
- Либо создать `.github/workflows/` директорию и добавить CI/CD конфигурацию
- Либо обновить документацию, если CI/CD управляется иначе

---

### 2. ❌ permissions.md - Отсутствует CODEOWNERS

**Документ:** `/home/user/TradingBot2/docs/permissions.md`  
**Проблема:** Документ ссылается на `CODEOWNERS` файл для управления правами доступа, но файл не существует.

**Строка в документе:**
```markdown
Ownership is enforced via the repository's `CODEOWNERS` file.
```

**Проверка:**
```bash
ls -la /home/user/TradingBot2/CODEOWNERS
# Результат: файл не найден
```

**Влияние:** Средне - документация предполагает наличие механизма управления ownership, который не реализован.  
**Рекомендация:**
- Создать `.github/CODEOWNERS` файл (стандартное местоположение GitHub) или
- Обновить документацию, если управление правами доступа реализовано другим способом

---

## ПРЕДУПРЕЖДЕНИЯ (3)

### 3. ⚠️ large_orders.md - Неправильные пути к тестам

**Документ:** `/home/user/TradingBot2/docs/large_orders.md`  
**Проблема:** Документ ссылается на тесты в директории `tests/`, но они находятся в корне проекта.

**Строка в документе (линия 62):**
```markdown
Unit tests should be implemented in `tests/test_execution_determinism.py` 
and `tests/test_executor_threshold.py`
```

**Фактическое местоположение:**
- ✓ `/home/user/TradingBot2/test_execution_determinism.py` (существует)
- ✓ `/home/user/TradingBot2/test_executor_threshold.py` (существует)

**Влияние:** Низко - тесты существуют, просто в другом месте.  
**Рекомендация:**
- Переместить тесты в `tests/` директорию или
- Обновить документацию на правильные пути

---

### 4. ⚠️ data_degradation.md - Неправильное имя класса

**Документ:** `/home/user/TradingBot2/docs/data_degradation.md`  
**Проблема:** Документация упоминает публичный класс `LatencyQueue`, а в коде это приватный класс `_LatencyQueue`.

**Строка в документе (линия 57-58):**
```markdown
Components log degradation statistics on shutdown. Look for messages
`OfflineCSVBarSource degradation`, `BinanceWS degradation` or
`LatencyQueue degradation` in the INFO log level
```

**Фактическое имя класса:**
- Класс: `_LatencyQueue` в `/home/user/TradingBot2/execution_sim.py` (строка 50)
- Лог-сообщение содержит строку "LatencyQueue degradation" (строка 1022)

**Влияние:** Низко - это просто лог-сообщение, содержит корректный текст для поиска.  
**Рекомендация:**
- Обновить документацию для указания, что это приватный класс, или
- Сделать класс публичным, если он часть публичного API

---

### 5. ⚠️ dynamic_spread.md - Отсутствуют рекомендуемые директории

**Документ:** `/home/user/TradingBot2/docs/dynamic_spread.md`  
**Проблема:** Документация предлагает хранить данные в директориях, которые не созданы.

**Строки в документе:**
- Линия 48: `--data data/seasonality_source/latest.parquet`
- Линия 75-76: "Keep the raw dataset under `data/seasonality_source/`"

**Фактическое состояние:**
- `/home/user/TradingBot2/data/seasonality_source/` - НЕ СУЩЕСТВУЕТ
- `/home/user/TradingBot2/data/slippage/` - НЕ СУЩЕСТВУЕТ

**Влияние:** Низко - это рекомендации для пользователей, директории создаются при необходимости.  
**Проверка:**
```bash
ls -la /home/user/TradingBot2/data/ | grep -E "seasonality|slippage"
# Результат: директории не найдены
```

**Рекомендация:**
- Это нормально для рекомендаций. Директории должны быть созданы пользователем при первом использовании
- Можно добавить инструкции по созданию директорий в документацию

---

## УСПЕШНО ПРОВЕРЕНО ✓

### Документы без проблем:

1. ✅ **pipeline.md**
   - Все упомянутые классы найдены: `PipelineConfig`, `PipelineStageConfig`
   - Все drop reasons верны: INCOMPLETE_BAR, MAINTENANCE, WINDOW и др.
   - Пути файлов корректны: `data/universe/symbols.json`

2. ✅ **bar_execution.md**
   - Все классы существуют: `SpotSignalEnvelope`, `SpotSignalEconomics`, `PortfolioState`
   - Все схемы найдены: `spot_signal_envelope.schema.json` и др.
   - Функция `decide_spot_trade` найдена
   - Конфигурационные параметры корректны: `execution.portfolio.equity_usd`, `execution.bar_price`, `execution.min_rebalance_step`

3. ✅ **universe.md**
   - Функция `get_symbols` найдена в `services/universe.py`
   - Путь к файлу символов корректен: `data/universe/symbols.json`
   - Функция `core_config.load_config` существует
   - CLI использование документировано правильно

4. ✅ **parallel.md**
   - Класс `SharedMemoryVecEnv` найден в `shared_memory_vec_env.py`
   - Импорты в примерах кода правильны
   - Примеры воспроизводимы

5. ✅ **eval.md**
   - `script_eval.py` поддерживает `--all-profiles` флаг (строка 27)
   - Конфигурационное поле `all_profiles` поддерживается (строка 129)
   - `service_eval` функция `from_config` существует

6. ✅ **data_degradation.md**
   - `DataDegradationConfig` найдена в `config.py`
   - Классы `BinanceWS` и `EventBus` существуют
   - `get_symbols` функция доступна
   - Пример кода синтаксически правильный

7. ✅ **no_trade.md**
   - `NoTradeConfig` найдена в `no_trade_config.py`
   - `compute_no_trade_mask` функция существует
   - `TradingEnv` класс найден
   - `configs/no_trade.yaml` файл существует
   - `state/no_trade_state.json` путь корректен
   - `apply_no_trade_mask.py` файл существует
   - CLI `no-trade-mask` определен в `setup.py`

8. ✅ **dynamic_spread.md**
   - `DynamicSpreadConfig` найдена в `slippage.py`
   - `scripts/build_spread_seasonality.py` существует
   - Все параметры скрипта документированы

9. ✅ **moving_average.md**
   - `simple_moving_average` функция найдена в `utils/moving_average.py`
   - Экспортирована в `utils/__init__.py`
   - Документация двуязычна (EN/RU)

---

## СТАТИСТИКА ПРОВЕРКИ

| Категория | Результат |
|-----------|-----------|
| Критические проблемы | 2 |
| Предупреждения | 3 |
| Успешно проверено | 9 |
| Всего проверено | 11 |
| Процент актуальности | 82% |

---

## РЕКОМЕНДАЦИИ ПО ПРИОРИТИЗАЦИИ

### Срочно (P1):
1. Создать `.github/workflows/` или обновить `permissions.md`
2. Создать `.github/CODEOWNERS` или обновить `permissions.md`

### Среднее значение (P2):
3. Переместить тесты в `tests/` или обновить `large_orders.md`
4. Обновить `data_degradation.md` для ясности о `_LatencyQueue`

### Низкое значение (P3):
5. Добавить инструкции по созданию рекомендуемых директорий в `dynamic_spread.md`

---

## ВЫВОДЫ

Большинство документации актуальна и соответствует кодовой базе. Основные проблемы сосредоточены на:
- Отсутствии структуры управления репозиторием (.github/CODEOWNERS)
- Несогласованности расположения файлов (тесты в корне вместо tests/)
- Использовании приватных деталей реализации в публичной документации

Рекомендуется выполнить обновления в порядке приоритизации.

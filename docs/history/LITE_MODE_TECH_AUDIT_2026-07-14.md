# RivenQuant Desktop — повторный полный технический аудит Lite Mode

Дата проверки: 2026-07-14  
Проверенная версия интерфейса: 2.6.0  
Git-снимок: `4c8fa99`, рабочее дерево содержит незакоммиченные изменения  
Область: Lite UI, source backend, собранный `riven-backend.exe`, локальный CCEA paper-контур, фоновые jobs, состояние и интерфейс Windows.

> Это внутренний технический аудит, а не независимая сертификация. Результаты относятся к локальному состоянию проекта на дату проверки.

## Итог

Lite Mode пока нельзя считать функционально завершённым или безопасно готовым к использованию. Найдено **24 актуальных дефекта**:

- **7 P0** — блокеры релиза, риск ложного чувства безопасности или нарушения целостности исследования;
- **10 P1** — ключевой функционал сломан, зависает или показывает выдуманные данные;
- **5 P2** — существенные логические и UX-несостыковки;
- **2 P3** — менее критичные, но реальные проблемы качества интерфейса.

Главный вывод: нормальный CCEA paper-контур работает, но вокруг него Lite UI содержит несколько независимых слоёв демонстрационной телеметрии и локального состояния, которые выглядят как результаты реальных расчётов. Кроме того, фактическая цепочка Data Manager и обучение из собранного EXE не проходят end-to-end.

## Сводный реестр

| ID | Приоритет | Область | Проблема |
|---|---:|---|---|
| L2-001 | P0 | Emergency Halt | Без CCEA или при ошибке соединения API выдумывает ликвидацию и возвращает `success` |
| L2-002 | P0 | Quick Start | «Готово к торгам» и результаты этапов строятся из `localStorage` и захардкоженных чисел |
| L2-003 | P0 | Data Manager | Фактическая Lite-цепочка ломается на No-Trade, Splits и Training Table |
| L2-004 | P0 | Data integrity | Lite принудительно использует `decision_delay_ms=50`, хотя безопасный минимум проекта — 8000 мс |
| L2-005 | P0 | Packaged EXE | Обучение RL в собранном `riven-backend.exe` падает до начала обучения |
| L2-006 | P0 | Risk limits | «Сохранить лимиты» игнорирует почти все поля, но сообщает, что лимиты применены |
| L2-007 | P0 | Gas Guard | Интерфейс заявляет активную автоматическую блокировку DEX, которой в Lite нет |
| L2-008 | P1 | Portfolio | VaR, PDT, SPAN и Greeks являются формулами/константами интерфейса, а не расчётами риска |
| L2-009 | P1 | Telemetry | PSI, slippage, WebSocket timing и `HFT_NODE_OK` показывают ложное здоровье пустой системы |
| L2-010 | P1 | Quant Lab | Training и Backtest не завершают UI-состояние после падения job |
| L2-011 | P1 | Multi-asset | Quant Lab остаётся на equity-настройках при Forex/Futures/Crypto/Options |
| L2-012 | P1 | Agent Manager | Применение калибровок/порогов указывает на несуществующие конфиги для 4 из 5 активов |
| L2-013 | P1 | Futures | Historical Sandbox для futures использует live-конфиг |
| L2-014 | P1 | Risk audit | Пустой или недоступный лог превращается в «Все лимиты в норме» |
| L2-015 | P1 | Dead controls | Две доступные Lite-кнопки вызывают несуществующие JavaScript-функции |
| L2-016 | P1 | Source runtime | При отдельном `RIVEN_DATA_DIR` фоновые source-jobs ищут код внутри data-каталога и падают |
| L2-017 | P1 | Regression tests | Зелёные тесты не исполняют реальные packaged jobs и не проверяют параметры Lite UI |
| L2-018 | P2 | Data UX | Диапазон импорта по умолчанию заканчивается 2024-12-31 |
| L2-019 | P2 | Adapter model | «Active Adapter (Broker)» смешивает брокера исполнения и поставщика данных |
| L2-020 | P2 | Demo history | Реалистичная демо-история выглядит как текущие сделки; источник отмечен только общим badge |
| L2-021 | P2 | Strategy registry | «Реестр стратегий» существует только в `localStorage` WebView |
| L2-022 | P2 | State sync | Asset/adapter UI подтверждает успех без проверки HTTP-статуса backend |
| L2-023 | P3 | Navigation | Прокрутка основной области переносится между модулями |
| L2-024 | P3 | Copilot UX | Стартовое сообщение преувеличивает возможности rule-based помощника |

## Детальные результаты

### L2-001 — Emergency Halt выдумывает успешную ликвидацию

**Приоритет: P0.**

Когда CCEA supervisor запущен, Emergency Halt работает правильно: реальная локальная paper-позиция была закрыта, Agent перешёл в pause, kill switch включился, после reset состояние восстановилось.

Но если CCEA выключен, `app.py:5422-5762` переходит в legacy fallback. Для каждого класса активов там захардкожены позиции, отменённые ордера и финансовый результат. Воспроизведённый ответ для equity:

```json
{
  "status": "success",
  "execution_mode": "mock_simulated",
  "orders_cancelled": 2,
  "positions_liquidated": [
    {"symbol": "SPY", "qty": 200, "price": 512.4},
    {"symbol": "AAPL", "qty": 100, "price": 178.5}
  ]
}
```

Ещё хуже: исключение реального адаптера обрабатывается как `error_fallback_simulated`, после чего API всё равно возвращает `status: success` и фиктивную позицию `LIQ_MOCK` (`app.py:5710-5762`). Это прямо противоречит комментарию `Never fabricate a liquidation report` на `app.py:5401-5404`.

**Критерий исправления:** без авторитетного execution-контура halt должен включить локальную блокировку и вернуть явный `partial/failed/unavailable`, не заявляя ни об одном отменённом или закрытом ордере. Ошибка брокера не может превращаться в success.

### L2-002 — Quick Start показывает выдуманную готовность

**Приоритет: P0.**

`renderPipelineStatusDetails()` (`index.html:13655-14070`) берёт номер этапа из `localStorage` (`rivenquant_workflow_progress_v3`) и на его основании показывает `ГОТОВО К ТОРГАМ`. Существование и актуальность артефактов при повторном запуске не перепроверяются.

После завершения этапа интерфейс не читает фактические отчёты, а выводит константы:

- `12,450 свечей`, `0 NaNs`, OHLC «Соблюдены», PII «Чисто»;
- `0 NaN строк (100% OK)`;
- Sharpe `1.84`, PnL `+24.5%`, `64` прогона оптимизации;
- сходимость обучения `Успешно (OK)`;
- успешный backtest с latency/slippage;
- сохранённые risk-метрики и график;
- запущенный forward test и активный Conformal Risk Guard.

Эти значения не связаны с backend job output. После удаления файлов или смены runtime браузерный прогресс остаётся.

**Критерий исправления:** каждый badge и показатель должен иметь источник, timestamp, artifact hash/job id. При отсутствии/устаревании результата показывать `Нет данных` или `Требуется повторный запуск`; готовность вычислять backend readiness-проверкой.

### L2-003 — Data Manager не проходит собственную цепочку

**Приоритет: P0.**

Проверка выполнена на собранном EXE с настоящим локальным `prices.parquet`:

- `run_features` — успешно, создано 49 строк;
- `run_targets` — успешно;
- `run_notrade` — HTTP 400 `Invalid job name`, потому что backend знает `run_no_trade`, а UI отправляет `run_notrade` (`index.html:22707-22716`, `app.py:7654`);
- `run_splits` — падает с требованием `--data`/`--config`; UI отправляет `n_splits` и `train_size_pct`, которые backend вообще не читает (`index.html:22719-22728`, `app.py:7661-7672`);
- `run_training_table` — падает с `KeyError: "['price'] not in index"`, потому что Lite создаёт/использует колонку `close`, но не передаёт `price_col=close` (`index.html:22731-22740`, `build_training_table.py:26-27`).

Тот же контракт `run_splits` затрагивает walk-forward в Quant Lab: UI передаёт `data`, но `app.py` не добавляет `--data` в команду.

**Критерий исправления:** один типизированный контракт UI→API→worker для каждого шага и packaged E2E, который нажимает реальные Lite-кнопки и проверяет созданные артефакты.

### L2-004 — Lite отключает безопасный LeakGuard default

**Приоритет: P0.**

`build_training_table.py` имеет безопасный default `decision_delay_ms=8000` и прямо предупреждает о forward-looking bias при меньшем значении. Lite Data Manager всегда отправляет `50` мс (`index.html:22736`). В packaged-прогоне worker напечатал:

```text
WARNING: decision_delay_ms=50 is below recommended minimum of 8000ms!
Insufficient delay may create forward-looking bias in training.
```

При этом Quick Start позже способен показать LeakGuard как чистый/пройденный. Тест `tests/test_forward_looking_bias_fix.py` проходит 13/13, потому что проверяет default worker, а не переопределение из Lite UI.

**Критерий исправления:** Lite не должен молча ослаблять 8000 мс. Меньшее значение — только как явно опасный expert override с блокирующим предупреждением и записью в manifest.

### L2-005 — RL training сломан в packaged EXE

**Приоритет: P0.**

Фактический `run_train` в `dist/riven-backend.exe` падает до запуска обучения:

```text
FileNotFoundError: ...\stable_baselines3\version.txt
[PYI-13352:ERROR] Failed to execute script 'desktop_backend'
```

PyInstaller включил Python-модули, но не data-файл `stable_baselines3/version.txt`. Проверка allowlist в `test_frozen_worker_dispatches_all_lite_job_modules` доказывает только перевод команды в worker, а не импорт и исполнение SB3.

**Критерий исправления:** packaged smoke должен реально импортировать SB3 и выполнить короткое обучение до появления валидного model artifact с exit code 0.

### L2-006 — кнопка сохранения риск-лимитов сохраняет только концентрацию

**Приоритет: P0.**

Форма предлагает:

- дневной лимит убытка;
- максимальную просадку;
- плечо;
- концентрацию;
- PDT, SPAN и Greeks switches.

Но `saveRiskLimits()` (`index.html:24373-24409`) меняет в YAML только `max_total_exposure_pct`, рассчитанный из концентрации. Остальные считанные значения не используются, checkbox-состояния даже не читаются. После этого UI сообщает: `Новые лимиты рисков сохранены и применены`.

`fetchRiskLimitsDefaults()` также восстанавливает только концентрацию, поэтому «Сбросить на заводские» не сбрасывает всю форму.

**Критерий исправления:** backend-схема risk policy, атомарное сохранение всех полей, валидация, read-back и подтверждение, что Agent применил конкретную версию policy. Нельзя писать «применено» после одной строковой замены YAML.

### L2-007 — Gas Guard является интерфейсной заглушкой

**Приоритет: P0.**

В Web3-вкладке написано `Gas Guard Active` и заявлено, что программа автоматически заблокирует DEX-ордера. На деле `updateLiteportGasAlarmLabel()` (`index.html:24341-24347`) только меняет подпись ползунка. Значение не сохраняется, не отправляется backend и не проверяется execution engine. DEX execution flow в Lite отсутствует.

**Критерий исправления:** либо удалить/пометить как `Not implemented`, либо реализовать Agent-side pre-trade guard с реальным gas oracle, fail-closed поведением, policy persistence и тестом блокировки ордера.

### L2-008 — Portfolio Risk рисует фиктивные показатели

**Приоритет: P1.**

`updateLiveAuditDetails()` (`index.html:24300-24331`) не вызывает риск-сервис:

- VaR = `NLV * 0.0245`;
- equity PDT всегда `PASSED`;
- futures SPAN всегда `STABLE (12.4k)`;
- options Greeks всегда `NEUTRAL (Δ=-0.12)`.

На пустом options-портфеле интерфейс показал NLV `$100,001`, VaR `$2,450.02` и `NEUTRAL (Δ=-0.12)` при отсутствии позиций.

**Критерий исправления:** реальные расчёты с методологией, valuation timestamp и source badge; при пустом/невалидном портфеле — `N/A`, а не зелёный результат.

### L2-009 — системная телеметрия изображает здоровье пустой системы

**Приоритет: P1.**

Найдено несколько независимых ложных defaults:

- при отсутствии `models/drift_report.json` backend подставляет PSI `0.045`, feature `f_volatility`, worst `0.072`, status `stable` (`app.py:209-213`);
- Analytics подставляет slippage `0.8 BPS` и оценивает его как удовлетворительный без сделок (`index.html:22905-22941`);
- поле «последний WS сигнал» показывает время NTP sync, если WebSocket данных нет (`index.html:22895-22903`);
- sidebar постоянно показывает `HFT_NODE_OK`, даже когда UI одновременно показывает `idle_node`, WS OFF, broker OFF и signaler stopped (`index.html:322`);
- Data Manager при `No data` показывает `PSI 0.000 Stable` и «Защита от утечки данных активна».

**Критерий исправления:** отсутствие измерения должно быть `N/A/Нет данных`, не «stable». Статус ноды должен агрегироваться из реальных health checks и иметь degraded/offline состояния.

### L2-010 — Quant Lab зависает после завершившейся ошибки

**Приоритет: P1.**

Воспроизведено в браузере:

- Training запускает только polling лога и вообще не проверяет `/api/job/status` (`index.html:16369-16382`). После падения worker кнопка Start остаётся disabled, Stop — enabled, текст навсегда «Запуск обучения…»;
- Backtest проверяет status, но при failed делает `return` без `stopQuantLabBacktest()` (`index.html:16534-16542`). Статус остаётся «В процессе»;
- Strategy Coder восстанавливает кнопки после ошибки, но его заголовок результата остаётся «Выполняется расчет…» (`resetCoderBacktestUI()`, `index.html:17288-17296`).

**Критерий исправления:** единая конечная машина `idle/running/succeeded/failed/cancelled`, один polling helper и обязательный cleanup в `finally` для всех Lite jobs.

### L2-011 — Quant Lab не следует активному классу актива

**Приоритет: P1.**

При активном `options (theta_data)` Quant Lab продолжил показывать:

- `strategies.momentum:MomentumStrategy`;
- `models/ppo_agent.zip`;
- `data/stocks/SPY_features.parquet`;
- календарь NYSE и equity ADV-параметры.

В отличие от Data Manager и Agent Manager, для Quant Lab нет asset-aware default updater; значения фиксированы в HTML и затем сохраняются глобально в `localStorage`, без ключа asset (`index.html:1100-1108`, `16250-16320`). Это позволяет незаметно запускать equity pipeline в options/forex/futures-контексте.

**Критерий исправления:** отдельная валидируемая конфигурация на asset class, автоматическое переключение путей/стратегии/календаря и блокировка несовместимых комбинаций.

### L2-012 — Agent Manager применяет результаты в несуществующие файлы

**Приоритет: P1.**

`getActiveSandboxConfigPath()` (`index.html:23709-23715`) возвращает:

- `configs/config_stocks.yaml`;
- `configs/config_forex.yaml`;
- `configs/config_futures.yaml`;
- `configs/config_options.yaml`.

Все четыре файла отсутствуют. Существует только fallback `configs/sandbox.yaml` для crypto. Поэтому «Применить издержки к конфигу» и «Сохранить пороги» гарантированно получают 404 для 4 из 5 классов активов.

**Критерий исправления:** использовать канонические существующие конфиги из backend mapping и заранее проверять capability/path; packaged E2E для каждого asset class.

### L2-013 — futures sandbox использует live-конфиг

**Приоритет: P1.**

Frontend и backend сопоставляют futures `sandbox` с `configs/config_live_futures.yaml` (`index.html:13037`, `app.py:752`). Документация упоминает `config_backtest_futures.yaml`, но такого файла нет.

Historical backtest не должен молча наследовать live execution semantics.

**Критерий исправления:** выделенный backtest/sandbox config для futures и contract test, подтверждающий отсутствие live broker/endpoints в историческом запуске.

### L2-014 — пустой риск-лог трактуется как доказательство нормы

**Приоритет: P1.**

`syncRiskLogs()` (`index.html:24559-24574`) при пустом логе **или HTTP-ошибке** пишет:

```text
Событий безопасности за текущую сессию не зафиксировано.
> Все лимиты в норме.
```

Отсутствие данных не доказывает норму. Особенно опасно рядом с фиктивными VaR/Greeks и неработающими настройками лимитов.

**Критерий исправления:** различать `пустой подтверждённый журнал`, `журнал недоступен`, `risk engine не запущен`, `нет policy`; «в норме» только после актуального backend check.

### L2-015 — две Lite-кнопки вызывают undefined handlers

**Приоритет: P1.**

Статический поиск всех ссылок нашёл две реальные Lite-пустышки:

- крестик консоли Quick Workflow вызывает `hideWorkflowConsole()` (`index.html:543`), определения нет;
- drift CTA «Применить и начать» вызывает `autoPopulateAndStartTraining()` (`index.html:958`), определения нет.

Обе кнопки вызывают `ReferenceError` при достижении соответствующего состояния.

**Критерий исправления:** реализовать действия или удалить controls; добавить DOM-test, проверяющий разрешение каждого доступного `onclick` handler.

### L2-016 — source runtime с отдельным data-каталогом ломает jobs

**Приоритет: P1.**

Чистый source backend с `RIVEN_DATA_DIR=.run/audit2-source-20260714` запускает subprocess из data-каталога. В результате:

```text
can't open file '...\.run\audit2-source-20260714\scripts\download_options_data.py'
ModuleNotFoundError: No module named 'app'
```

Это затрагивает Data Manager и backtest. Обычный dev-запуск из корня маскирует дефект; packaged dispatcher имеет другой путь.

**Критерий исправления:** разделить `code_root` и `data_root`, всегда строить source-команды абсолютными путями к коду, а output — относительно data root.

### L2-017 — текущие regression-тесты создают ложный зелёный статус

**Приоритет: P1.**

Пройдено:

- `tests/test_lite_mode_audit_closure.py` + `tests/test_desktop_e2e.py`: **11 passed**;
- `tests/test_forward_looking_bias_fix.py`: **13 passed**;
- `tools/check_mvp_honesty.py`: все проверки pass.

Но:

- frozen-worker test проверяет только наличие модуля и перевод command line, не запускает его;
- E2E проверяет CCEA lifecycle/persistence, но не Data/Train/QuantLab buttons;
- forward-looking test проверяет default 8000, не Lite override 50;
- honesty tool проверяет несколько API flags, но не UI-константы, risk cards и workflow readiness.

Поэтому утверждение предыдущего отчёта «все 22 пункта закрыты» и packaged pipeline «проверен» не покрывает фактический пользовательский путь. LITE-001 и safety-часть LITE-003 должны считаться **переоткрытыми частично** до реального UI-driven packaged E2E.

**Критерий исправления:** тесты должны запускать собранный EXE, взаимодействовать с реальными Lite controls и проверять exit code, artifact schema, UI terminal state и отсутствие выдуманных success-данных.

### L2-018 — импорт по умолчанию заканчивается в 2024 году

**Приоритет: P2.**

На 2026-07-14 Data Manager предлагает `2023-01-01 — 2024-12-31` (`index.html:1452-1456`). Новый пользователь без изменения формы получает заведомо устаревший датасет без предупреждения.

**Критерий исправления:** end date = текущая дата/последняя доступная сессия, плюс явный stale-data warning.

### L2-019 — adapter называется broker, хотя им не является

**Приоритет: P2.**

Селектор подписан `Active Adapter (Broker)`, но содержит Yahoo, Polygon, Theta Data и Dukascopy. После выбора `theta_data` реальный CCEA execution broker остаётся `sim_paper`, а UI-контекст выглядит как будто брокер сменился.

**Критерий исправления:** раздельные поля `Market data provider` и `Execution broker`, с отдельными connection/capability badges.

### L2-020 — демо-история слишком похожа на текущую торговлю

**Приоритет: P2.**

В чистом runtime `/api/trades` честно возвращает `simulated: true`, `data_source: demo_mock`, а UI показывает общий amber badge. Однако таблица сразу заполнена десятью правдоподобными сделками с agent names, order IDs, fees и slippage; внутри таблицы нет колонки/метки источника. Пользователь легко читает их как текущую историю счёта.

**Критерий исправления:** пустое состояние по умолчанию или явный `DEMO` на каждой строке/в заголовке таблицы; demo summary не смешивать с account summary.

### L2-021 — «реестр стратегий» не является устойчивым реестром

**Приоритет: P2.**

Quant Lab сохраняет карточки «реестра» только в `localStorage` (`index.html:16681+`). Очистка WebView/site data удаляет их; backend model registry, artifact hash, версия и связь с файлами отсутствуют.

**Критерий исправления:** переименовать в «локальные закладки» либо сохранять в backend registry с immutable metadata и проверкой существования артефактов.

### L2-022 — UI подтверждает asset/adapter до ответа backend

**Приоритет: P2.**

`selectAsset()` и `onAdapterChanged()` сначала меняют глобальное состояние и интерфейс, затем выполняют `fetch`, но не проверяют `res.ok` (`index.html:12931-12992`). Даже HTTP 400/500 приводит к success-toast, если сеть как таковая доступна.

Backend корректно отклоняет несовместимую asset/adapter пару HTTP 400, но UI не умеет откатить состояние.

**Критерий исправления:** validate/commit pattern: применить UI только после 2xx, при ошибке восстановить подтверждённое server state и показать detail.

### L2-023 — прокрутка переносится между разделами

**Приоритет: P3.**

`switchModule()` скрывает/показывает workspaces, но не сбрасывает `scrollTop` основной прокручиваемой области (`index.html:14550+`). При переходе из длинного Data/History экрана Portfolio может открыться посередине, скрывая заголовок, sync и Emergency Halt controls.

**Критерий исправления:** после смены модуля прокрутить main container к началу либо хранить отдельную позицию на модуль и явно восстанавливать её.

### L2-024 — стартовое сообщение Copilot преувеличивает возможности

**Приоритет: P3.**

Первое сообщение заявляет: «вычислительный контур полностью проиндексирован» и «прямой доступ ко всем 22 модулям» (`index.html:11125`). На деле `/api/copilot` — deterministic `rule_based_advisory` (`app.py:8464-8470`). После первого ответа UI уже честно добавляет пометку «правила/шаблоны», но стартовый экран её не показывает.

**Критерий исправления:** с первого сообщения назвать это локальным командным помощником; показывать список реально доступных команд/capabilities, а не общий claim о полном доступе.

## Что подтверждено рабочим

- Tauri dev-приложение запускается и создаёт настоящее Windows-окно.
- Root `index.html` внутри собранного EXE совпадает с текущим source UI по SHA-256.
- Все 8 основных Lite-разделов открываются; проверены Data Manager, 5 вкладок Quant Lab, 3 вкладки Agent Manager, Analytics, Portfolio, History и Strategy Coder.
- Все 5 asset classes и 13 заявленных adapter combinations переключаются; backend отклоняет недопустимые пары HTTP 400.
- Data Manager реально создаёт `features.parquet` и `targets.parquet` в packaged runtime.
- Manual Strategy: syntax validation не пишет файл; явное сохранение создаёт strategy и params.
- MetaMask без provider честно остаётся `DISCONNECTED`; WalletConnect disabled; Safe/MPC не показывают ложного подключения.
- CCEA paper order, позиция, mark/NAV, Emergency Halt и reset работают в нормальном CCEA-пути.
- `/api/trades` и `/api/portfolio/holdings` маркируют fallback как demo/simulated.
- Source и frozen backend возвращают согласованные основные status/CCEA/portfolio/trades/telemetry API.

## Ограничение нативной UI-проверки

Нативное Tauri-окно успешно запустилось, но инструмент автоматизации Windows не смог получить screenshot/window state из-за `GetCursorPos: Access denied (0x80070005)`. Поэтому детальный интерактивный прогон выполнен через тот же локальный HTML/backend во встроенном браузерном контуре. Это ограничение среды аудита, а не найденный дефект приложения.

## Обязательный порядок исправлений

1. Удалить все fabricated success-пути Emergency Halt и сделать fallback fail-closed.
2. Перевести Quick Start и Portfolio/Risk на backend evidence; убрать все hardcoded success-метрики.
3. Исправить Data Manager contracts (`run_no_trade`, splits, `price_col`) и вернуть безопасный delay 8000.
4. Починить PyInstaller SB3 data files и выполнить реальное короткое packaged обучение.
5. Сделать risk limits и Gas Guard реальными Agent-side policies либо честно отключить.
6. Унифицировать lifecycle всех jobs и asset-aware конфигурации Quant/Agent Manager.
7. Добавить UI-driven packaged E2E на каждую кнопку и каждый asset class.
8. После этого закрыть P2/P3 UX и naming issues.

## Критерий готовности Lite Mode

Lite можно считать готовым только когда чистый установленный desktop проходит один полный сценарий:

`выбор asset/provider → ingest → audit → features → targets → no-trade → splits → training table → короткое training → backtest → evaluation → paper order → portfolio/risk → emergency halt → restart/persistence`

Для каждого этапа должны проверяться одновременно:

- настоящий exit code 0;
- ожидаемый артефакт и его schema/hash;
- корректное terminal state UI;
- отсутствие mock/default метрик без явного `DEMO`;
- одинаковое поведение source и packaged EXE;
- fail-closed при недоступном broker/CCEA/risk engine.

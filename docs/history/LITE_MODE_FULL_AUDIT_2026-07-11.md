# RivenQuant Desktop — полный реестр проблем Lite Mode

Дата исходной проверки: 2026-07-11  
Дата закрытия remediation: 2026-07-14  
Версия интерфейса: 2.6.0  
Область проверки: собранное Windows-приложение, исходный backend, Lite UI, CCEA paper-контур и формы адаптеров.

## Итоговый статус

Все 22 пункта исходного аудита закрыты и получили regression-проверку. Lite Mode теперь использует исполняемые операции либо явно отключённые/неподдерживаемые элементы без ложного сообщения об успехе.

Упакованный `riven-backend.exe` проверен как самостоятельный worker: он создаёт `features.parquet`, `targets.parquet`, `training_table.parquet`, запускает backtest и сохраняет отчёт. CCEA paper-контур проверен через shutdown/restart с сохранением marks, NAV, unrealized и day P&L. Этот статус относится к перечисленным ниже 22 дефектам и не является заявлением о полном аудите всей платформы.

## Покрытие проверки

- 8 основных Lite-экранов.
- 104 кнопки и действия внутри Lite.
- 96 полей, переключателей и списков.
- 18 внутренних вкладок.
- 5 классов активов: equity, forex, futures, crypto, options.
- 13 сочетаний брокеров и поставщиков данных.
- 16 типов фоновых задач.
- Исходный backend и упакованный `riven-backend.exe`.
- Реальное Windows-окно Tauri.
- Paper trade, EOD, shutdown и повторный запуск.

## Сводка

| ID | Приоритет | Тип | Проблема | Статус |
|---|---:|---|---|---|
| LITE-001 | P0 | Desktop runtime | Фоновые Lite-задачи не запускаются в упакованном EXE | Закрыто 2026-07-14 |
| LITE-002 | P0 | Broker | IB сообщает об успешном несуществующем подключении | Закрыто 2026-07-14 |
| LITE-003 | P0 | Safety | Lite Emergency Halt перезаписан одноимённой Pro-функцией | Закрыто 2026-07-14 |
| LITE-004 | P0 | Data integrity | Paper NAV и mark позиции меняются после перезапуска | Закрыто 2026-07-14 |
| LITE-005 | P1 | JavaScript | `fetchStatus()` падает каждые 5 секунд | Закрыто 2026-07-14 |
| LITE-006 | P1 | Пустышка | Auto-Heal/ffill сообщает успех без файла и обработки | Закрыто 2026-07-14 |
| LITE-007 | P1 | Пустышка | MetaMask и WalletConnect имитируют подключение | Закрыто 2026-07-14 |
| LITE-008 | P1 | Пустышка | Safe SDK и MPC Vault только показывают success-toast | Закрыто 2026-07-14 |
| LITE-009 | P1 | Пустышка | Синхронизация сделок ничего не синхронизирует | Закрыто 2026-07-14 |
| LITE-010 | P1 | Пустышка | Закрытие демо-позиции не меняет портфель | Закрыто 2026-07-14 |
| LITE-011 | P1 | Credentials | Адаптеры без ключей получают ошибку при сохранении | Закрыто 2026-07-14 |
| LITE-012 | P1 | UX/state | Выбор адаптера сбрасывается при выборе актива | Закрыто 2026-07-14 |
| LITE-013 | P1 | Data UI | `READY` отображается одновременно с 0% и отсутствием данных | Закрыто 2026-07-14 |
| LITE-014 | P1 | Copilot | `/start`, `/backtest`, `/pipeline` не работают в EXE | Закрыто 2026-07-14 |
| LITE-015 | P1 | Copilot/Safety | `/start` использует legacy live path вне CCEA | Закрыто 2026-07-14 |
| LITE-016 | P2 | EOD UX | Результат EOD мгновенно перезаписывается | Закрыто 2026-07-14 |
| LITE-017 | P2 | Strategy Coder | «Проверить синтаксис» сохраняет стратегию на диск | Закрыто 2026-07-14 |
| LITE-018 | P2 | Пустышка | Микрофон Copilot не включает голосовой ввод | Закрыто 2026-07-14 |
| LITE-019 | P2 | Пустышка | Скрепка Copilot не открывает выбор файла | Закрыто 2026-07-14 |
| LITE-020 | P2 | Honesty UI | Глобальный demo-badge остаётся от другого экрана | Закрыто 2026-07-14 |
| LITE-021 | P2 | Code quality | Дублирующиеся глобальные обработчики Lite/Pro | Закрыто 2026-07-14 |
| LITE-022 | P2 | Browser/dev | Standalone UI предполагает backend только на порту 8002 | Закрыто 2026-07-14 |

## Реализованное закрытие и доказательства

| ID | Что изменено | Проверка результата |
|---|---|---|
| LITE-001 | Добавлен allowlisted worker-dispatcher для PyInstaller, durable job-state с настоящим exit code и startup grace для Windows PID. | Packaged EXE создал features/targets/training table; успешные и падающие workers вернули корректные terminal states. |
| LITE-002 | Connector считается подключённым только после фактического backend handshake; IB/OANDA дополнительно проверяют реальное состояние. | Попытка IB на `127.0.0.1:1` отклонена, активный `sim_paper` не заменён; broker regression suite пройден. |
| LITE-003 | Lite/Pro handlers разделены; Lite Halt вызывает CCEA Agent, паузит его, отменяет реальные заявки и закрывает фактические позиции без mock-отчёта. | E2E открыл paper ETH-позицию, Halt увидел 1 и оставил 0; обработчики уникальны. |
| LITE-004 | Marks сохраняются в SQLite P&L ledger и восстанавливаются вместе с позициями. | Два запуска backend: mark 55 000, NAV 100 500, unrealized 500 и day P&L 0 сохранились точно. |
| LITE-005 | Все необязательные DOM-узлы в `fetchStatus()` защищены null-check. | Периодический browser poll прошёл без новых console errors. |
| LITE-006 | Реализован атомарный Parquet repair: backup, sort/dedup, замена infinity, ограниченный group ffill и точные счётчики. | Без файла API возвращает 404 и кнопка disabled; unit test оставляет длинный gap незаполненным и помечает partial. |
| LITE-007 | MetaMask использует `window.ethereum`, chain switch и RPC balance; WalletConnect отключён без SDK. | В среде без провайдера нет фиксированного адреса/баланса и нет статуса CONNECTED. |
| LITE-008 | Safe требует адрес и проверяет bytecode; MPC controls явно disabled как не настроенные. | UI не заявляет о подключении или сохранении при отсутствии интеграции. |
| LITE-009 | Paper sync читает Agent Books; live sync требует capability активного connector. | Paper E2E вернул `source=agent_books` и фактическое число сделок; unsupported live получает 501. |
| LITE-010 | Портфель и закрытие идут через CCEA broker; demo-fallback нельзя закрыть. | Fake SPY вернул 404; реальная paper BTC-позиция закрылась до quantity 0. |
| LITE-011 | Yahoo, Dukascopy и Theta Data завершают настройку локально без пустого Vault-запроса. | Browser-проверка показала «Готово» без ошибки credentials. |
| LITE-012 | Выбор adapter хранится отдельно для каждого asset class. | Polygon сохранился после перехода Equity → Crypto → Equity. |
| LITE-013 | Badge вычисляется из фактических артефактов: Ready / Incomplete / No data. | В чистом runtime показаны No data, 0% и три `Not generated yet`. |
| LITE-014 | Copilot jobs используют тот же packaged worker/status contract. | `/backtest` и pipeline jobs в EXE либо создают результат с exit 0, либо честно возвращают failed + exit code. |
| LITE-015 | `/start` и `/stop` переведены на локальный CCEA lifecycle с явным PAPER/LIVE и broker. | API-проверка перевела Agent PAUSED → RUNNING в `paper/sim_paper`. |
| LITE-016 | EOD результат вынесен в toast до следующего poll. | Browser показал точный EOD NAV/day P&L, после чего обычный P&L продолжил обновляться. |
| LITE-017 | AST/compile validate отделён от save; параметры пишутся отдельным endpoint. | SHA файла стратегии до/после validate не изменился, `written=false`. |
| LITE-018 | Подключён реальный Web Speech API; без поддержки control сообщает unsupported. | Вызов не показывает ложное «активирован» без `SpeechRecognition`. |
| LITE-019 | Реализован file picker для разрешённых текстовых config-файлов до 1 MB. | Выбранное содержимое действительно загружается в Copilot textarea. |
| LITE-020 | Demo badge привязан к source module и очищается при навигации. | Pro Data-QA badge скрыт на Lite Data. |
| LITE-021 | Удалены дубли; emergency handlers имеют отдельные Lite/Pro имена. | Статический regression test подтверждает ровно одно объявление каждого handler. |
| LITE-022 | API base использует инъекцию Tauri или same-origin; 8002 остаётся только legacy fallback. | Полный browser/E2E прогон выполнен на случайных портах без ручной перенастройки. |

---

## Подробные проблемы

Ниже сохранены исходные формулировки и сценарии воспроизведения от 2026-07-11 как историческая база regression-проверок. Актуальный статус и способ закрытия каждого пункта указаны в таблицах выше.

### LITE-001 — фоновые задачи не запускаются в упакованном EXE

**Приоритет:** P0  
**Затронуто:** Data Manager, Quant Lab, Agent Manager, Strategy Coder, Copilot.

Backend формирует команды через `sys.executable`. В исходном режиме это `python.exe`, но внутри PyInstaller-сборки это `riven-backend.exe`. Приложение пытается запустить команду вида:

```text
riven-backend.exe make_features.py --in ... --out ...
```

Реальный результат в собранном приложении:

```text
usage: riven-backend.exe [-h] [--host HOST] [--port PORT]
riven-backend.exe: error: unrecognized arguments:
make_features.py --in data/definitely_missing.parquet ...
```

API сначала возвращает PID и создаёт впечатление успешного запуска, затем дочерний процесс завершается.

Затронутые действия:

- загрузка исторических данных для всех пяти классов активов;
- расчёт features;
- генерация targets;
- no-trade mask;
- walk-forward splits;
- training table;
- PSI;
- probability calibration;
- cost/slippage calibration;
- threshold tuner;
- parity check;
- обучение PPO;
- backtest и evaluate;
- команды Copilot `/start`, `/backtest`, `/pipeline`.

Код: `app.py`, обработчик `/api/run_job`, построение команд через `sys.executable` и `start_background()`.

**Ожидается:** packaged backend должен использовать встроенный dispatcher/worker либо поставляемый Python runtime, а статус задачи должен отражать её exit code.

### LITE-002 — ложное успешное подключение Interactive Brokers

**Приоритет:** P0

Проверочный сценарий:

1. Указать IB host `127.0.0.1`.
2. Указать порт `1`, на котором TWS/Gateway отсутствует.
3. Нажать сохранение/подключение.

Фактический результат:

- API вернул `ok: true`, `connected: true`;
- активный paper broker был заменён на IB;
- CCEA status стал показывать подключённый IB.

Причина: `_IBBackend` игнорирует отрицательный результат `IBOrderExecutionAdapter.connect()`, а `DelegatingConnector.connect()` считает наличие объекта backend достаточным доказательством подключения.

Код:

- `packages/agent/broker/adapters/ib.py`;
- `packages/agent/broker/adapters/_delegating.py`;
- `ccea/desktop_supervisor.py::connect_live_broker()`.

**Ожидается:** проверять реальное `isConnected()`/heartbeat/account request до замены paper broker.

### LITE-003 — Emergency Halt Lite вызывает Pro-логику

**Приоритет:** P0

В `index.html` дважды объявлены глобальные функции:

```text
triggerEmergencyHalt()
resetEmergencyHalt()
```

Lite-реализация вызывает `/api/panic_halt` и `/api/panic_reset`. Позднее объявленная Pro-реализация перезаписывает её и использует `/api/killswitch/trigger`, `/api/killswitch/reset`, `confirm()`, `prompt()` и скрытое Pro-поле `killswitch-reset-token`.

Следствие: кнопки Portfolio Lite выполняют не тот сценарий, который описывает Lite UI.

Код: `index.html`, объявления примерно около строк 24078/24094 и 25559/25588.

**Ожидается:** отдельные имена `triggerLiteEmergencyHalt` и `triggerProEmergencyHalt`, единый согласованный API-контракт.

### LITE-004 — paper NAV и рыночная цена теряются после перезапуска

**Приоритет:** P0

Сценарий:

1. Paper BUY 0.1 BTC по 50 000.
2. Mark price 55 000.
3. До перезапуска NAV = 100 500, unrealized P&L = +500.
4. Сделать EOD.
5. Перезапустить desktop backend.

После перезапуска:

- позиция и средняя цена сохранились;
- mark сбросился с 55 000 на 50 000;
- NAV стал 100 000;
- unrealized P&L стал 0;
- Day P&L стал -500;
- last close NAV остался 100 500.

Причина: `SimBroker.restore_state()` восстанавливает cash и positions, но не последнюю рыночную цену/marks.

Код: `packages/agent/broker/adapters/sim.py`, `ccea/desktop_supervisor.py`.

**Ожидается:** marks должны сохраняться в durable state или восстанавливаться из P&L ledger.

### LITE-005 — периодическая ошибка `fetchStatus()`

**Приоритет:** P1

Каждые пять секунд браузерная консоль получает:

```text
TypeError: Cannot read properties of null (reading 'classList')
at fetchStatus (...:14845)
```

Код без проверки обращается к отсутствующему элементу `badge-exec`:

```javascript
document.getElementById('badge-exec').classList.add('hidden');
```

Из-за исключения код ниже этой строки, включая часть workflow telemetry, не обновляется.

Код: `index.html::fetchStatus()`.

**Ожидается:** null-check либо восстановление элемента `badge-exec`.

### LITE-006 — Auto-Heal/ffill является пустышкой

**Приоритет:** P1

Функция `triggerLiteAutoHeal()` не отправляет запрос backend и не читает/изменяет файл. Она только записывает заранее подготовленный текст в консоль и показывает success-toast.

Проверено при полном отсутствии `data/prices.parquet`. UI сообщил:

```text
Очистка пропущенных значений методом Forward-Fill...
Health Score = 100%. Ошибок нет.
Авто-исправление пропусков успешно выполнено!
```

При этом Data Manager продолжил показывать 0% и `Not generated yet`.

Код: `index.html::triggerLiteAutoHeal()`.

**Ожидается:** реальный backend job с количеством исправленных значений; кнопка должна быть disabled без входного файла.

### LITE-007 — MetaMask и WalletConnect имитируют подключение

**Приоритет:** P1

`connectWeb3Provider()` не обращается к `window.ethereum`, WalletConnect SDK или RPC. Через 800 мс функция устанавливает фиксированный адрес:

```text
0x89205A1244444C3F88000A1A225F4C7b11124444
```

После нажатия без установленного/подключённого кошелька UI показал:

- `CONNECTED`;
- 1.458 ETH;
- 24,800 USDC;
- сообщение об успешном подключении.

Демо-маркер рядом с Web3-состоянием отсутствует.

Код: `index.html::connectWeb3Provider()`, `updateWeb3UI()`.

**Ожидается:** реальный provider handshake либо явная маркировка `DEMO / SIMULATED` и отключённые кнопки.

### LITE-008 — Safe SDK и MPC Vault являются пустышками

**Приоритет:** P1

- `initializeSafeConnection()` не использует Safe SDK и всегда показывает успех через таймер.
- `saveMPCConfiguration()` ничего не сохраняет, но сообщает о записи в `LOCAL_VAULT.md`.

Код: `index.html::initializeSafeConnection()`, `saveMPCConfiguration()`.

**Ожидается:** реальная интеграция или честный demo-state без заявления о сохранении.

### LITE-009 — синхронизация истории сделок является пустышкой

**Приоритет:** P1

Endpoint `/api/trades/sync` безусловно возвращает:

```json
{"status":"success","detail":"Trades synchronized from active broker/wallet."}
```

Ни брокер, ни кошелёк не опрашиваются, состояние не меняется. UI показывает успешную синхронизацию.

Код: `app.py::api_trades_sync()`, `index.html::syncLiteHistoryTrades()`.

### LITE-010 — закрытие демонстрационной позиции сообщает ложный успех

**Приоритет:** P1

Без реальных брокерских ключей `/api/portfolio/holdings` возвращает demo-позиции SPY/AAPL. Нажатие Close возвращает:

```text
Mock close position submitted for SPY
```

После обновления SPY остаётся в портфеле. UI может сообщить, что позиция закрыта.

**Ожидается:** demo-позиции не должны иметь рабочую кнопку Close либо demo state должен изменяться локально и явно оставаться simulated.

### LITE-011 — сохранение адаптеров без ключей завершается ошибкой

**Приоритет:** P1

Для Yahoo, Dukascopy и Theta Data модальное окно сообщает, что дополнительные API-ключи не нужны. Однако кнопка «Сохранить ключи» остаётся активной и отправляет пустой `credentials` в `/api/ccea/store_credentials`.

Backend корректно отвечает `no credentials supplied`, а UI показывает ошибку.

Затронуто:

- Yahoo;
- Dukascopy;
- Theta Data.

**Ожидается:** скрыть/переименовать кнопку либо считать локальный адаптер готовым без обращения к Vault.

### LITE-012 — выбранный адаптер может сбрасываться

**Приоритет:** P1

`onAdapterChanged()` сохраняет выбранный adapter, но `selectAsset()` сразу присваивает первый adapter из `adaptersByAsset[assetClass]`.

Пример: пользователь выбирает Polygon, затем нажимает US Equities — активным снова становится Alpaca.

Код: `index.html::selectAsset()`, `onAdapterChanged()`.

### LITE-013 — Data Manager показывает противоречивый статус

**Приоритет:** P1

В packaged desktop при отсутствии всех трёх файлов корректно отображаются:

- Health = 0%;
- `prices.parquet — Not generated yet`;
- `features.parquet — Not generated yet`;
- `training_table.parquet — Not generated yet`.

Одновременно верхний badge остаётся `READY`.

Причина: `checkDataFiles()` обновляет score/status/ring, но не обновляет `litedata-audit-badge`.

### LITE-014 — команды Copilot не работают в packaged desktop

**Приоритет:** P1

Команды `/start`, `/backtest` и `/pipeline` запускают subprocess через `sys.executable`, поэтому в PyInstaller повторяют дефект LITE-001. Copilot может сообщить PID/успех до фактического падения процесса.

### LITE-015 — Copilot `/start` использует legacy live path

**Приоритет:** P1

Команда `/start` запускает:

```text
script_live.py --config configs/config_live.yaml
```

Она не использует desktop CCEA supervisor/Agent Vault flow и не добавляет явный `--paper`. При наличии реальных legacy credentials это не соответствует заявленной desktop-архитектуре.

Код: `app.py::api_copilot()`.

**Ожидается:** Copilot должен вызывать CCEA lifecycle request и явно показывать paper/live режим.

### LITE-016 — EOD-сообщение сразу исчезает

**Приоритет:** P2

`cceaEodClose()` записывает результат EOD в `ccea-pnl`, затем вызывает `pollCcea()`. Poll немедленно перезаписывает текст обычным Equity/PnL status. Пользователь не успевает увидеть результат операции.

**Ожидается:** отдельный toast или EOD status field.

### LITE-017 — «Проверить синтаксис» сохраняет файл

**Приоритет:** P2

`coderCheckSyntax()` просто вызывает `coderSaveStrategy()`. Таким образом, проверка синтаксиса имеет побочный эффект: компилирует и записывает стратегию/параметры на диск.

То же относится к «Применить параметры» через `coderSaveParamsOnly()`.

Код: `index.html::coderCheckSyntax()`, `coderSaveParamsOnly()`.

**Ожидается:** отдельный validate endpoint без записи; сохранение только по явной кнопке.

### LITE-018 — микрофон Copilot является пустышкой

**Приоритет:** P2

Кнопка микрофона только вызывает:

```javascript
showActionToast('Микрофон активирован')
```

SpeechRecognition, разрешение микрофона и запись не запускаются.

### LITE-019 — прикрепление файла Copilot является пустышкой

**Приоритет:** P2

Кнопка со скрепкой только показывает сообщение «Прикрепите конфигурационный файл». File picker и upload отсутствуют.

### LITE-020 — глобальный demo-badge протекает между экранами

**Приоритет:** P2

Badge `showSimBadge()` является одним глобальным элементом. После перехода между экранами он может продолжать показывать статус предыдущего запроса, например:

- `Firm Risk: representative exposures` на Lite Home/Data Manager;
- `Data-QA: representative feeds` при отсутствии data-файлов.

**Ожидается:** badge должен принадлежать конкретному экрану/источнику и очищаться при смене модуля.

### LITE-021 — дублирующиеся глобальные функции

**Приоритет:** P2

Найдены повторные объявления Lite-handler names:

- `triggerEmergencyHalt`;
- `resetEmergencyHalt`;
- `startQuantLabBacktest`;
- `stopQuantLabBacktest`;
- `closeLiteHistoryAuditor`.

Emergency-функции имеют разную семантику и приводят к реальному конфликту. Остальные дубли увеличивают риск скрытого изменения поведения при дальнейшей разработке.

### LITE-022 — standalone UI рассчитан на порт 8002

**Приоритет:** P2

Если `window.RIVEN_API_BASE` не инъецирован Tauri и UI открыт с произвольного локального порта, `getApiBase()` направляет запросы на `http://127.0.0.1:8002`.

Это не ломает штатный Tauri flow, но мешает локальному browser/E2E тестированию на случайном порту и создаёт ложные `Failed to fetch`.

Код: `index.html::getApiBase()`.

---

## Что подтверждено рабочим

- Запуск desktop-окна и splash screen.
- Загрузка локальных Tailwind, Chart.js, FontAwesome, Monaco и шрифтов.
- Переключение Lite/Pro.
- Все восемь основных Lite-разделов.
- Data Manager: четыре вкладки.
- Quant Lab: пять вкладок.
- Analytics: три вкладки.
- Agent Manager: три вкладки.
- Portfolio: три вкладки.
- Все пять классов активов.
- Все 13 допустимых сочетаний asset/adapter проходят server validation.
- Недопустимые asset/adapter возвращают HTTP 400.
- Все формы credentials корректно перестраиваются по adapter.
- Polygon и Deribit credentials записываются в CCEA Vault без возврата секрета в ответе.
- Неверные Alpaca/OANDA/Binance credentials не отключают paper broker.
- CCEA paper order исполняется и попадает в ledger.
- Ledger/broker reconciliation и integrity check проходят до перезапуска.
- Graceful backend shutdown освобождает SQLite-файлы.
- `/api/portfolio/holdings` и `/api/trades` честно помечают fallback data как simulated/demo.
- В исходном Python-режиме `/api/run_job` правильно строит команды для 16 проверенных задач.

## Проверенные формы адаптеров

| Asset | Adapter | Поля |
|---|---|---|
| Equity | Alpaca | API key, secret |
| Equity | Polygon | API key |
| Equity | Yahoo | ключи не требуются |
| Forex | OANDA | token, account ID |
| Forex | Dukascopy | ключи не требуются |
| Futures | IB | host, port, client ID |
| Futures | Binance Futures | API key, secret |
| Crypto | Binance | API key, secret |
| Crypto | Deribit | client ID, client secret |
| Options | IB | host, port, client ID |
| Options | Theta Data | UI считает, что ключи не требуются |
| Options | Deribit | client ID, client secret |
| Options | Polygon | API key |

## Рекомендуемый порядок исправления

1. Исправить packaged job runner (LITE-001).
2. Исправить реальную проверку IB connection (LITE-002).
3. Разделить Lite/Pro Emergency Halt handlers (LITE-003).
4. Сохранять mark prices и исправить EOD/restart accounting (LITE-004).
5. Устранить периодическую ошибку `fetchStatus()` (LITE-005).
6. Убрать ложные success-сообщения Auto-Heal/Web3/Safe/MPC/Sync/Close либо реализовать операции.
7. Исправить credentials UX, adapter selection и Data `READY` badge.
8. Разделить Strategy Coder validate/save.
9. Привести Copilot к CCEA lifecycle и убрать неработающие mic/attachment controls.
10. Добавить packaged desktop E2E на каждую группу Lite jobs, проверяющий не только PID, но и exit code/артефакт.

## Критерии готовности после исправления

- Любая кнопка запуска задачи завершается реальным результатом либо понятной ошибкой.
- API не считает PID доказательством успеха.
- В packaged EXE хотя бы один полный pipeline создаёт data/features/targets/training table и backtest report.
- Невозможное broker connection никогда не заменяет paper broker.
- После перезапуска NAV, marks, unrealized P&L и day P&L остаются согласованными.
- Никакая demo-функция не использует слова `CONNECTED`, `saved`, `synchronized`, `closed` или `success`, если операция не произошла.
- На каждом demo/simulated экране присутствует локальный и актуальный маркер.
- Browser console не содержит повторяющихся ошибок.
- Для каждой Lite-кнопки есть regression test либо явная маркировка `Not implemented`/disabled.

## Финальная верификация 2026-07-14

- `pytest tests/ccea/phase5/test_broker_protocol.py tests/test_ib_adapters.py tests/test_lite_mode_audit_closure.py tests/test_pnl_ledger.py` — **159 passed, 2 skipped** (`ib_insync` отсутствует; реальное отрицательное подключение отдельно проверено fail-closed).
- `pytest tests/test_forward_looking_bias_fix.py` — **13 passed**; training-table default использует безопасный `decision_delay_ms=8000`.
- `pytest tests/test_desktop_e2e.py` на исходном backend — **passed**.
- Тот же `test_desktop_e2e.py` с `RIVEN_DESKTOP_BACKEND_EXE=dist/riven-backend.exe` — **passed** на двух последовательных запусках.
- PyInstaller profile `research` — сборка успешна; sidecar скопирован в `desktop/src-tauri/binaries/`.
- Packaged pipeline — `features`, `targets`, `training_table` и `logs/sandbox_reports.csv` созданы; backtest записал **3485 строк**, все jobs завершились `exit_code=0`.
- Все inline JavaScript-блоки `index.html` компилируются через `new Function()`.
- Browser smoke на `127.0.0.1:62020` — 8 Lite-разделов, CCEA PAPER, No-data state и честный Emergency Halt проверены интерактивно.
- `cargo check --manifest-path desktop/src-tauri/Cargo.toml` — **успешно**.

## Примечание

Это внутренний технический отчёт. Он не является независимым аудитом или сертификацией. Все результаты относятся к локальной сборке и состоянию проекта на дату проверки.

# Binance Spot order execution (P0-C)

> **Статус:** ✅ ЗАКРЫТО 2026-07-16 · закрывает P0-C / §3.4 из
> [PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md](history/PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md).
> Тесты: `tests/test_binance_spot_execution.py` (17) + live smoke.

## Проблема (что было)

В проекте есть **два** пути живого исполнения:

1. **CCEA Agent-зона** (`packages/agent/broker/adapters/binance.py::BinanceConnector`)
   — уже поддерживала spot (`futures=False`), подключалась через
   `desktop_supervisor.connect_live_broker` / `/api/ccea/connect_broker`.
2. **Adapters registry** (`adapters/registry.py::create_order_execution_adapter`)
   — используется MVP-эндпоинтами `app.py`: panic-halt, holdings, close.

Для spot-вендора `ExchangeVendor.BINANCE` в registry были зарегистрированы
`MARKET_DATA` / `FEE` / `TRADING_HOURS` / `EXCHANGE_INFO` и все `FUTURES_*`, но
**НЕ было** `ORDER_EXECUTION` (в отличие от futures, у которого есть
`futures_order_execution.py`). Поэтому:

- `create_order_execution_adapter(ExchangeVendor.BINANCE, …)` бросал
  `ValueError: No adapter registered for binance/order_execution`;
- `/api/portfolio/holdings?asset=crypto` и `/api/portfolio/close` для crypto
  ловили это в `broker_error`;
- `/api/panic_halt` для crypto был захардкожен как **fail-closed**: «Для Binance
  spot в этой сборке нет order-execution адаптера» — kill switch срабатывал, но
  позиции НЕ закрывались.

Итог: crypto-spot live/panic был невозможен через MVP-путь.

## Решение

### 1. Новый адаптер `adapters/binance/order_execution.py`

`BinanceOrderExecutionAdapter(OrderExecutionAdapter)` — spot-исполнение по
Binance Spot REST API, по образцу `futures_order_execution.py` (тот же
HMAC-SHA256 signing, транспорт `RestBudgetSession`):

| Метод | Endpoint | Примечание |
|-------|----------|------------|
| `submit_order(order)` | POST `/api/v3/order` | MARKET/LIMIT из `core_models.Order` |
| `submit_spot_order(...)` | POST `/api/v3/order` | + STOP_LOSS/TAKE_PROFIT/*_LIMIT |
| `cancel_order(...)` | DELETE `/api/v3/order` | symbol обязателен (Binance) |
| `get_order_status(...)` | GET `/api/v3/order` | → `ExecReport` |
| `get_open_orders(symbol?)` | GET `/api/v3/openOrders` | |
| `cancel_all_orders(symbol?)` | DELETE `/api/v3/openOrders` | без symbol — перебор по открытым |
| `get_positions(symbols?)` | GET `/api/v3/account` | балансы → синтетические позиции |
| `get_account_info()` | GET `/api/v3/account` | quote-asset cash |
| `get_last_price(symbol)` | GET `/api/v3/ticker/price` | unsigned |

**Spot vs futures — намеренные отличия:**

- Нет leverage / margin / positionSide / reduceOnly (long-only cash-рынок).
- **«Позиции» = балансы**: ненулевой баланс base-актива превращается в
  синтетическую long-`Position` с ключом `{asset}{quote_asset}` (quote по
  умолчанию `USDT`, настраивается), чтобы panic/close могли его **market-SELL**.
  Quote/stable-активы (`USDT/USDC/BUSD/FDUSD/TUSD/DAI/USD`) исключаются — это
  кэш, а не позиция.
- `avg_entry_price = 0` (unknown): spot-account-endpoint **не отдаёт** cost
  basis. Мы сообщаем это честно (`meta.cost_basis_available = False`), никогда
  не выдумываем цену входа.
- `close_position(symbol)` = market-SELL всего base-баланса (спот только long).

### 2. Регистрация (`adapters/binance/__init__.py`)

`AdapterType.ORDER_EXECUTION` зарегистрирован для `ExchangeVendor.BINANCE` и
`ExchangeVendor.BINANCE_US` (последний — с `spot_url=https://api.binance.us`).
Futures-фабрика не затронута (проверено тестом на отсутствие регрессии).

### 3. Проводка MVP (`app.py`)

- **panic-halt**: убран fail-closed short-circuit для crypto; добавлена реальная
  ветка `elif asset_class == "crypto"` → `create_order_execution_adapter(
  ExchangeVendor.BINANCE)` → `cancel_all_orders()` + `_close_via_market_orders()`
  (spot get_positions отдаёт пары → market-SELL для флэттенинга).
- **holdings / close**: уже вызывали `create_order_execution_adapter(
  ExchangeVendor.BINANCE)` — теперь адаптер существует, и они работают.
- `/api/adapters/test_connection` принимает `binance_us` / `binance_futures`.

### 4. MVP UI (`index.html`)

Селектор `#vault-cred-broker` теперь честно различает **Binance Spot (Crypto)**
(`binance`) и **Binance Futures (Crypto)** (`binance_futures`) — раньше `binance`
был ошибочно подписан «Binance Futures», хотя connect-путь по умолчанию идёт в
spot.

## Проверка

- `tests/test_binance_spot_execution.py` (17): регистрация (BINANCE + BINANCE_US,
  без регрессии futures), submit market/limit, error-code → failure, stop-limit
  через `submit_spot_order`, cancel требует symbol, cancel_all перебор,
  positions из балансов (маппинг в пары, кастомный quote, фильтр по symbol),
  close market-SELL + no-op когда flat, order-status → ExecReport, account cash;
  **+ интеграционный тест `test_panic_halt_crypto_spot_flattens`** — реальный
  FastAPI-путь `/api/panic_halt` для crypto флэттенит spot через registry-адаптер
  (broker застаблен, без сети).
- Live smoke: registry резолвит `BinanceOrderExecutionAdapter`;
  `/api/adapters/test_connection` принимает `binance`/`binance_futures`;
  `/api/portfolio/holdings?asset=crypto` доходит до адаптера без
  «No adapter registered».

## Известное ограничение (legacy fallback)

MVP-путь panic/holdings в `app.py` — это **sandbox-oriented legacy fallback**;
авторитетный live-путь — CCEA Agent (`BinanceConnector`, обрабатывает ошибки
через `OrderResult.success`). В fallback read-методы адаптера (`get_positions`/
`get_open_orders`) при ошибке аутентификации возвращают пусто (как и futures-
адаптер) — это консистентно с существующим поведением, но означает, что неверные
ключи выглядят как «нет позиций». Для реальной торговли используйте CCEA Agent.

## Файлы

- `adapters/binance/order_execution.py` — новый spot-адаптер.
- `adapters/binance/__init__.py` — регистрация ORDER_EXECUTION (BINANCE/BINANCE_US).
- `app.py` — panic crypto-ветка + `test_connection` вендоры.
- `index.html` — spot/futures в `#vault-cred-broker`.
- `tests/test_binance_spot_execution.py` — 17 тестов.
- `tests/test_lite_mode_audit_2026_07_14.py` — снят lock на старое fail-closed поведение.

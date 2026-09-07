# Dukascopy forex adapter (закрытие stub-гэпа)

> **Статус:** ✅ ЗАКРЫТО 2026-07-16 · UI предлагал Dukascopy для forex, но за ним
> была 43-строчная Phase-0 заглушка. Тесты: `tests/test_dukascopy_adapter.py`
> (13) + реальный сетевой смоук (4995 тиков → бары; 14 баров через ingest-CLI).

## Проблема

`adapters/dukascopy/__init__.py` был пустышкой (`__all__ = []`, ни одного
адаптера, ни регистрации), хотя:

- UI (`adaptersByAsset.forex = ['oanda', 'dukascopy']`) предлагал Dukascopy;
- в Data Manager есть провайдер «Dukascopy (Public ticks)» с обещанием
  «высокоточные тиковые данные без необходимости авторизации»;
- registry lazy-мапил `DUKASCOPY → adapters.dukascopy`, но там нечего было
  регистрировать → `create_market_data_adapter("dukascopy")` не давал ничего.

## Решение: реальный публичный bi5 tick-feed

Реализован **бесплатный публичный историко-тиковый фид** Dukascopy (основа
`duka`/`dukascopy-node`/`dukascopy-python`), **без авторизации** — ровно то, что
обещает UI. Кредентальный JForex API (live-торговля) намеренно НЕ используется:
это data-only адаптер (как Yahoo/Polygon), поэтому регистрируется только
`MARKET_DATA` — фейкового order-execution нет.

### Формат данных

Dukascopy публикует тик-историю почасовыми LZMA-сжатыми `.bi5`-файлами:

```
https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
```

**Готча (знаменитая):** месяц в пути **0-индексный** (январь = `00`). Каждая
запись — 20 байт, big-endian `>IIIff`:

| поле | тип | смысл |
|------|-----|-------|
| ms   | uint32 | смещение мс от начала часа |
| ask  | uint32 | ask в «пунктах» (целое) |
| bid  | uint32 | bid в «пунктах» |
| ask_vol | float32 | объём ask (млн) |
| bid_vol | float32 | объём bid (млн) |

Цены — целые «пункты», масштаб 10^decimals на инструмент (большинство FX = 1e5,
JPY-пары и металлы = 1e3; переопределяется `config["point_values"]`).

### Реализация (`adapters/dukascopy/market_data.py`)

- `DukascopyMarketDataAdapter(MarketDataAdapter)`; транспорт — **plain
  `requests`** (не `RestBudgetSession`: он JSON-ориентирован и декодирует байты
  в текст, ломая bi5 — этот баг поймал реальный смоук).
- `get_bars(symbol, tf, start_ts, end_ts, limit)`: качает все почасовые файлы
  диапазона (кап `max_hours`, дефолт 720), декодит тики, агрегирует в OHLCV
  ресемплом по mid-цене (pandas), сохраняет bid/ask-каналы OHLC;
  `volume_base` = число тиков (честный прокси), `volume_quote` = спред в пунктах.
- `get_latest_bar`, `get_tick` (последний доступный тик), `stream_bars`/
  `stream_ticks` — polling-генераторы (честно near-real-time: bi5 финализируется
  почасово, это НЕ sub-second; для low-latency — OANDA/IB).
- Выходные — `core_models.Bar`/`Tick` (те же поля, что у OANDA-адаптера).
- Выходные — 404/выходные/праздники → пусто, без падений.

### Проводка

1. Registry: `MARKET_DATA` для `ExchangeVendor.DUKASCOPY` (`__init__.py`).
2. **Ingest-путь** (`scripts/download_forex_data.py`): добавлен `--provider
   {oanda|dukascopy}` + `download_pair_dukascopy` (та же выходная схема, что у
   OANDA — downstream provider-agnostic) + диспетчер `_download_pair`. `app.py`
   forex-ветка теперь передаёт `--provider` (раньше читала, но не передавала).
3. **Premium-data** (`services/premium_data.py`): dukascopy добавлен в
   entitlement-матрицу (forex, keyless, ticks=history) → карточка «Интрадей-фиды»
   и `/api/data/premium/*` умеют качать минутки Dukascopy.
4. UI: провайдер уже был в списке forex + добавлен в premium-vendor select;
   `/api/adapters/{status,test_connection}` знают dukascopy.

## Проверка

- `tests/test_dukascopy_adapter.py` (13): регистрация, 0-индексный месяц в URL,
  масштаб цен + override, декод bi5 (синтетический LZMA), агрегация OHLCV +
  bid/ask, лимит, выходные→пусто, плохой tf→ошибка, get_tick, dukascopy в
  premium-матрице.
- **Реальный сетевой смоук**: скачан настоящий bi5 (23 394 байта) → 4995 тиков
  EURUSD за час, реалистичные цены/спред; ingest-CLI
  `--provider dukascopy` сохранил 14 реальных 1h-баров в parquet со штатной
  схемой. Смоук поймал и починил реальный баг (бинарный download через
  RestBudgetSession возвращал `str`).

## Файлы

- `adapters/dukascopy/market_data.py` — новый адаптер.
- `adapters/dukascopy/__init__.py` — регистрация MARKET_DATA.
- `scripts/download_forex_data.py` — `--provider`, `download_pair_dukascopy`, `_download_pair`.
- `app.py` — forex-ingest `--provider`, adapters status/test_connection.
- `services/premium_data.py` — dukascopy в матрице; `index.html` — premium-vendor select.
- `tests/test_dukascopy_adapter.py`.

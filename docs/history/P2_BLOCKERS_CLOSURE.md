# P2-блокеры (зрелость и масштаб) — ЗАКРЫТИЕ

Закрытие семи P2-блокеров из gap-анализа. Все — с тестами; #4 и #7 оказались уже
построены в cross-sectional работе (Part B/C) и верифицированы. Окружение: системный
Python + `PYTHONPATH=.venv/Lib/site-packages`.

---

## P2-1 — Расширение каталога сигналов + COT/economic calendar ✅

**Было:** ~26 сигналов; нет residual momentum / seasonality / sentiment; COT/calendar не в пайплайне.
**Стало:** **32 фактора**; alt-data; COT и economic calendar подключены как enrichers.

| Артефакт | Содержимое |
|---|---|
| [signals/common_signals.py](../../signals/common_signals.py) | ResidualMomentum, Seasonality (month-of-year, PIT), Sentiment (alt-data BYO), Week52High, IdiosyncraticVol, COTPositioning |
| [loaders/altdata_enrich.py](../../loaders/altdata_enrich.py) | `COTEnricher` (as-of + publish-lag → PIT), `EconCalendarEnricher` (high-impact-soon флаг; использует `data/forex/calendar/economic_calendar.parquet`) |
| тесты | [test_common_signals.py](../../tests/test_common_signals.py) (8), [test_altdata_enrich.py](../../tests/test_altdata_enrich.py) (4) |

Вплетено: `build_signal_library` (COMMON_SIGNAL_KINDS) + `build_enrichers` (ALTDATA_ENRICHERS).

## P2-2 — Feature Store (версии на уровне фичи + online-кэш) ✅

**Было:** per-run / file-level версионирование, нет переиспользования/inference-кэша.
**Стало:** ключ `(name, asof, content_hash)`; версия бампится только при изменении контента;
as-of чтение (PIT); online-кэш с TTL для inference; materialize нескольких фич.

| Артефакт | Тесты |
|---|---|
| [service_feature_store.py](../../service_feature_store.py) | [test_feature_store.py](../../tests/test_feature_store.py) (7) |

## P2-3 — FIX-протокол + Smart Order Routing ✅

**Было:** только базовый OrderRouter; нет FIX, нет мульти-venue.
**Стало:** самодостаточный **FIX 4.4** (корректные BodyLength/CheckSum) + SOR с water-filling сплитом.

| Артефакт | Содержимое | Тесты |
|---|---|---|
| [fix_protocol.py](../../packages/agent/execution/fix_protocol.py) | encode/parse/verify, NewOrderSingle/Cancel/ExecutionReport, FixSession | [test_fix_and_sor.py](../../tests/test_fix_and_sor.py) (8) |
| [smart_order_router.py](../../packages/agent/execution/smart_order_router.py) | Venue (fee/latency/liquidity/impact), мульти-venue split по маржинальной стоимости | |

## P2-4 — Cross-asset единый портфель (уже построено: Stage C1) ✅

`service_cross_asset.py`: **валютная нормализация** `[r_base=(1+r_local)(1+r_fx)−1]`, **joint Σ**
(стек base-доходностей → Ledoit-Wolf, PSD), **класс-risk-parity** + **общий vol-target**.
API `POST /api/xs/cross_asset`, UI «Unified Cross-Asset». Тесты `test_xs_cross_asset.py` (8) — зелёные.

## P2-5 — TS-база (ClickHouse/Timescale) вместо плоского parquet ✅

**Было:** плоский parquet.
**Стало:** абстракция с бэкендами; партиционирование по символу для 10³ символов.

| Артефакт | Содержимое | Тесты |
|---|---|---|
| [services/tsdb.py](../../services/tsdb.py) | `ParquetTSBackend` (partition-by-symbol, time-range/column pushdown), `ClickHouseTSBackend`, `TimescaleTSBackend` (DI-драйвер, graceful fallback), `TimeSeriesStore` фасад, `make_backend` | [test_tsdb.py](../../tests/test_tsdb.py) (6) |

## P2-6 — Автоматизация (drift-ретрейн, авто-TCA, e2e CI GDPR/DORA) ✅

| Артефакт | Содержимое | Тесты |
|---|---|---|
| [services/automation/drift_retrain.py](../../services/automation/drift_retrain.py) | `DriftRetrainScheduler` (PSI-порог + cooldown → триггер ретрейна) | [test_automation.py](../../tests/test_automation.py) (6) |
| [services/automation/tca_reporter.py](../../services/automation/tca_reporter.py) | `TCAReporter` (implementation shortfall, slippage, by-venue/symbol/side, markdown) | |
| [tests/test_gdpr_dora_e2e.py](../../tests/test_gdpr_dora_e2e.py) | end-to-end: GDPR export/delete + DORA concentration/incident | (4) |

## P2-7 — Options как отдельный greeks-оптимизатор (уже построено: Stage B5) ✅

`service_options_portfolio.py`: `OptionsPortfolioConstructor` — greeks-нейтральная конструкция
(delta/vega/gamma) через null-space проекцию (НЕ MVO). API `POST /api/xs/options/construct`,
UI greeks-конструктор. Тесты `test_xs_options.py` (11) — зелёные.

---

## Сводка проверок

| Блок | Тесты | Статус |
|---|---|---|
| P2-1 сигналы + COT/calendar | 12 | ✅ |
| P2-2 feature store | 7 | ✅ |
| P2-3 FIX + SOR | 8 | ✅ |
| P2-4 cross-asset (C1) | 8 (существ.) | ✅ |
| P2-5 TS-DB | 6 | ✅ |
| P2-6 автоматизация | 10 | ✅ |
| P2-7 options greeks (B5) | 11 (существ.) | ✅ |
| Регрессия (signals/data/xs) | 77 | ✅ зелёные |

**Всего новых P2-тестов: 43** (+ 19 существующих для C1/B5) = 62 зелёных.

**Команды проверки:**

```
PYTHONPATH=.venv/Lib/site-packages python -m pytest tests/test_common_signals.py tests/test_altdata_enrich.py tests/test_feature_store.py tests/test_fix_and_sor.py tests/test_tsdb.py tests/test_automation.py tests/test_gdpr_dora_e2e.py -q
```

---

## MVP-UI (вынесено в интерфейс, browser-verified)

Новые P2-фичи (#1,#2,#3,#5,#6) вынесены панелью **«🏛️ Институциональный масштаб (P2)»**
в pro-backtest → вкладка Cross-Sectional. #4 (cross-asset) и #7 (options) уже были в UI (C1/B5).

| Что | Эндпойнт | UI |
|---|---|---|
| Smart Order Routing (мульти-venue split) + FIX 4.4 preview | `POST /api/exec/route` | карточка «SOR + FIX» (кнопка Route $3M) |
| Каталог сигналов (33 фактора по классам) | `GET /api/xs/signal_catalog` | карточка «Каталог сигналов» |
| Автоматизация: drift-ретрейн + TS-DB backend | `GET /api/automation/status` | карточка «Автоматизация» |
| Feature Store (фичи + версии) | `GET /api/features/store` | карточка «Feature Store» |

Проверено в браузере: каталог = «33 факторов (common 6)»; SOR = сплит на 4 venue (181 bps) +
**FIX 4.4 🔏 valid**; drift-ретрейн = «НУЖЕН» (по реальному `models/drift_report.json`);
TS-DB backend = ParquetTSBackend; feature store = честный пустой state. Ошибок от P2-кода нет.

# P0-блокеры готовности к показу про-квантам/фондам — ЗАКРЫТИЕ

Документ фиксирует закрытие четырёх P0-блокеров из gap-анализа. Все пункты —
с воспроизводимыми артефактами, тестами и проверками. Окружение: системный Python

- `PYTHONPATH=.venv/Lib/site-packages` (как в проекте).

---

## P0-1 — Реальный бэктест на живых данных (вместо синтетики) ✅

**Было:** весь edge — на synthetic данных (движок давал фиктивный Sharpe > 5).
**Стало:** реальный cross-sectional крипто-бэктест на исторических дневных барах
**Binance** (public API, без ключей), `pit_quality=true`.

| Артефакт | Назначение |
|---|---|
| [configs/config_xs_crypto_real.yaml](configs/config_xs_crypto_real.yaml) | `source: free, vendor: binance`, 14 символов USDT |
| [tools/xs_crypto_real_sweep.py](tools/xs_crypto_real_sweep.py) | Pre-registered sweep (8 вариантов), все репортятся |
| [reports/XS_CRYPTO_REAL_TRUST_REPORT.md](reports/XS_CRYPTO_REAL_TRUST_REPORT.md) | Опубликованный Trust-Report |

- **Данные:** 14 символов × 1000 баров, период **2023-09-19 → 2026-06-14** (~2.7 года).
- **Честный итог:** лучший вариант (mom-only, monthly, risk-parity) — сырой Sharpe 3.17,
  но **Deflated Sharpe 0.32** (после поправки на множественное тестирование) → вердикт
  `likely_overfit`. **Анти-оверфит контур реально отвергает то, что синтетика принимала.**
- **Воспроизведение:** `PYTHONPATH=.venv/Lib/site-packages python tools/xs_crypto_real_sweep.py`

---

## P0-2 — PIT-данные для equity (настоящий point-in-time) ✅

**Было:** фундаментал — free-снимок (`pit_quality=none`, look-ahead), value/quality
сигналы закомментированы как «нечестные»; index-membership деградировал в static.
**Стало:** настоящий **бесплатный PIT-фундаментал из SEC EDGAR** + загрузчик истории
членства в индексе. Данные Sharadar/Compustat — это покупка, но для US equity бесплатный
EDGAR уже даёт **подлинный PIT** (та же `fundamentals_path`-семантика — drop-in).

### Фундаментал (SEC EDGAR XBRL companyfacts)

| Артефакт | Назначение |
|---|---|
| [services/edgar_fundamentals.py](services/edgar_fundamentals.py) | PIT-загрузчик: `publish_ts = дата подачи (filed)` → честный as-of |
| [scripts/download_edgar_fundamentals.py](scripts/download_edgar_fundamentals.py) | Скачивание в parquet-кэш |
| [loaders/equity_enrich.py](loaders/equity_enrich.py) | enricher `edgar_fundamentals` (pit=true) в реестре |
| [tests/test_edgar_fundamentals.py](tests/test_edgar_fundamentals.py) | 5 тестов (включая **as-of no-look-ahead**) |

- **Скачано реально:** 659 записей, 10 имён (AAPL…PG), **2009-07-22 → 2026-05-20**.
- **EPS/BVPS/FCF/ROE** на дату подачи каждого 10-K/10-Q (per-share деривации).

### История index-membership (survivorship-free)

| Артефакт | Назначение |
|---|---|
| [services/index_membership_loader.py](services/index_membership_loader.py) | Загрузчик changes-файла → PIT `IndexMembershipUniverse` |
| [data/universe/sp500_membership_demo.csv](data/universe/sp500_membership_demo.csv) | Demo-bootstrap (реальный якорь: TSLA добавлен 2020-12-21) |
| [tests/test_index_membership.py](tests/test_index_membership.py) | 5 тестов (PIT add/remove, survivorship-free, wiring) |

`build_universe` теперь использует `universe.type: index_membership` + `membership_path`.

### Реальный equity-бэктест на честном PIT

| Артефакт | Назначение |
|---|---|
| [configs/config_xs_equity_real.yaml](configs/config_xs_equity_real.yaml) | Yahoo цены + EDGAR фундаментал, value/quality **включены** |
| [reports/XS_EQUITY_REAL_TRUST_REPORT.md](reports/XS_EQUITY_REAL_TRUST_REPORT.md) | Опубликованный Trust-Report |

- **Data-Trust вердикт: `ok`** — ВСЕ колонки сигналов (earnings/book_value/fcf/roe)
  имеют **`pit_quality=true, vendor=sec_edgar`**. Бэктест backtest-safe.
- **Результат (market-neutral long-short, 129 недель):** Sharpe **1.92**, доходность +20%,
  MaxDD −5.5%, PSR 0.92, Deflated Sharpe 0.53.

---

## P0-3 — Честность MVP (мок под видом реального устранён) ✅

**Было:** ряд эндпойнтов отдавал mock как реальные данные (вкл. фейковое compliance-evidence).
**Стало:** demo/simulated помечается флагами, evidence-эндпойнты не синтезируют фейк,
фронт показывает бейдж **🟡 SIMULATED**.

| Эндпойнт | Фикс |
|---|---|
| `/api/trades` | dict с `simulated`/`data_source`; на фронте — бейдж |
| `/api/risk/summary` | убран хардкод `SAFE`; честный leak-status + clock из реального источника |
| `/api/ai-act/explain/{id}` | **404** вместо синтеза фейкового решения (рег. риск) |
| `/api/compliance/best-execution`, `/api/dora/concentration-risk` | флаг `demo: true` + disclaimer |
| `/api/portfolio/holdings` | флаг `simulated` (нет брокерских ключей) |
| `/api/copilot` | помечен `engine: rule_based_advisory` (не агент) |

- **Фронт:** [index.html](index.html) — `showSimBadge`/`flagFromPayload`, бейдж на trades/holdings.
- **Проверка:** [tools/check_mvp_honesty.py](tools/check_mvp_honesty.py) — **9/9 проверок PASS**
  (через FastAPI TestClient).

---

## P0-4 — Experiment tracking + Model registry (MLflow-подобный) ✅

**Было:** прогоны без трекинга, нет реестра моделей/версий/lineage/подписи — нельзя
защитить выбор модели перед LP/регулятором.
**Стало:** лёгкий файловый бэкенд: прогоны, метрики, **lineage (модель→данные→конфиг→git)**,
версии, стадии, **rollback**, **Ed25519-подпись артефактов**.

| Артефакт | Назначение |
|---|---|
| [core_experiment.py](core_experiment.py) | Контракты (RunRecord, ModelVersion, Lineage, ArtifactRef) |
| [service_experiment_tracking.py](service_experiment_tracking.py) | Tracker + Registry + ArtifactSigner (Ed25519/HMAC) |
| [tools/experiment_cli.py](tools/experiment_cli.py) | CLI (experiments/runs/models/promote/rollback/verify) |
| [tests/test_experiment_tracking.py](tests/test_experiment_tracking.py) | **14 тестов** (подпись, tamper, rollback, lineage) |
| app.py | 11 REST-эндпойнтов `/api/experiments/*`, `/api/models/*` |

- **Интеграция с реальными бэктестами:** sweep'ы P0-1/P0-2 логируют прогон с lineage
  (git-commit, config-hash, data-range) и регистрируют подписанный артефакт:
  - `xs_crypto_alpha` v1 — Ed25519, signature_valid: true.
  - `xs_equity_alpha` v1 — Ed25519, signature_valid: true.
- **Проверка API:** transition (с архивацией предыдущего production), rollback,
  signature-verify — все зелёные.

---

## Сводка проверок

| Блок | Тесты/проверки | Статус |
|---|---|---|
| P0-1 crypto real backtest | реальный прогон, 133 ребаланса, Trust-Report | ✅ |
| P0-2 EDGAR PIT fundamentals | 5 тестов + реальная загрузка + verdict=ok | ✅ |
| P0-2 index membership | 5 тестов | ✅ |
| P0-3 MVP honesty | 9/9 (TestClient) | ✅ |
| P0-4 tracking/registry | 14 тестов + 8 API-проверок | ✅ |
| Регрессия xs-движка | 57 существующих xs-тестов | ✅ зелёные |

**Команды воспроизведения** (из корня репо, с `SEC_EDGAR_USER_AGENT="You you@mail"`):

```
PYTHONPATH=.venv/Lib/site-packages python tools/xs_crypto_real_sweep.py
PYTHONPATH=.venv/Lib/site-packages python scripts/download_edgar_fundamentals.py --symbols AAPL MSFT GOOGL AMZN NVDA META JPM XOM JNJ PG --out data/fundamentals_edgar/edgar_pit.parquet
PYTHONPATH=.venv/Lib/site-packages python tools/xs_equity_real_report.py
PYTHONPATH=.venv/Lib/site-packages python -m pytest tests/test_experiment_tracking.py tests/test_edgar_fundamentals.py tests/test_index_membership.py -q
SEASONALITY_API_TOKEN=mock_token PYTHONPATH=.venv/Lib/site-packages python tools/check_mvp_honesty.py
```

---

## MVP-UI (вынесено в интерфейс, browser-verified)

P0 закрыт не только в коде, но и в MVP (`index.html` + REST в `app.py`):

| Что | Где в UI | Эндпойнты |
|---|---|---|
| **Панель MLOps** (эксперименты, прогоны с lineage, реестр моделей, promote/rollback/verify, Ed25519-подпись) | Сайдбар **«3b. MLOps & Model Registry»** | `/api/experiments*`, `/api/models*` (11 шт.) |
| **Реальные бэктесты** (Crypto/Equity/EDGAR — запуск в фоне + лог + Trust-Report) | pro-backtest → вкладка **Cross-Sectional**, карточка «▶ Реальные бэктесты» | `POST /api/xs/real/run` |
| **Честность: бейдж 🟡 SIMULATED** | trades + holdings (lite/pro) | флаги `simulated`/`data_source` в ответах |
| **Честность: compliance demo-баннер** | pro-compliance (MiFID/DORA) | флаг `demo:true` |
| **Честность: copilot rule-based лейбл** | lite/pro copilot-чат | `engine: rule_based_advisory` |

Проверка фронта (DOM/state в браузере): панель MLOps рендерит реальные данные
(`xs_crypto_alpha` v1 🔏 valid, git-lineage); кнопка EDGAR отрабатывает end-to-end
(DONE exit 0, лог стримит реальный вывод); бейджи simulated/demo показываются.
Скриншоты этой страницы таймаутят (известно), верификация — через DOM/state eval.

# Полный гэп-анализ платформы: заявлено vs реально vs что доделать

**Дата:** 2026-07-15 · **Метод:** code-grounded проверка (Grep/Read/pytest/импорты) поверх двух предыдущих аудитов
([PRO_PIPELINE_GAP_ANALYSIS.md](PRO_PIPELINE_GAP_ANALYSIS.md) 2026-06-15, [docs/LITE_MODE_AUDIT_CLOSURE_2026-07-15.md](docs/LITE_MODE_AUDIT_CLOSURE_2026-07-15.md)) ·
**Статус:** внутренний инженерный анализ, не сертификация. Каждое утверждение проверяемо по указанному файлу/тесту.

---

## 0. TL;DR

Платформа — это **настоящий, большой, преимущественно рабочий код** (21 675 собираемых тестов, реальный CI на GitHub Actions
ubuntu+windows с ruff/black/guardrails/coverage), а не витрина. Слабости трёх типов:

1. **«Проводка» (wiring):** самые сильные движки (оптимизатор, FIX/SOR, CPCV/PBO, market-abuse, model registry)
   достижимы из Python/тестов/REST-по-запросу, но не включены в регулярный live-контур. Это ~60% оставшейся работы.
2. **Данные и вендоры:** матрица адаптеров неполна и местами бита (см. §3); нет планировщика регулярных задач —
   всё запускается руками/через REST.
3. **Продуктовая зрелость десктопа:** нет автообновления, лицензирования, ручного ордер-тикета, push-алертов.

После закрытия Lite-аудита (24 дефекта + 34 ревью-находки, 2026-07-15) «вранья» в интерфейсе практически не осталось:
всё нереальное либо удалено, либо помечено DEMO/SIMULATED. Оставшийся разрыв — не честность, а полнота.

---

## 1. Заявлено vs реально — карта по доменам

Легенда: ✅ работает и проверено · 🟡 есть, но не доделано/не подключено · 🔴 сломано · ⬜ отсутствует.

| Домен | Заявлено (claude.md / доки) | Фактически |
|---|---|---|
| **RL-обучение** (Distributional PPO, Twin Critics, UPGD, VGS, PBT, SA-PPO, conformal) | ✅ Production, сотни тестов | ✅ Реально: движок настоящий, тесты зелёные, тренировка работает даже внутри packaged EXE (проверено 2026-07-15). 🟡 PBT/SA-PPO/conformal — опциональные ветки конфига, не дефолтный путь; ✅ GPU-обучение закрыто 2026-07-16 (`--device`+детекция, §5.22); распределённое (DDP/Ray) — нет. |
| **Бэктест/симуляция исполнения** (L2/L2+/L3, LOB, TCA-калибровка) | ✅ Production, 5 asset-классов | ✅ Движок реальный (execution_sim, lob/, параметрические TCA 84+86 тестов). 🟡 L3 не питается реальными книгами (P2 №13 прошлого аудита); реальные бэктесты (P0-1) есть для crypto/equity, для futures/forex/options — конфиги без «real trust report». |
| **Cross-sectional платформа** (signals→Σ→μ→w*→execution) | 🚧 в claude.md, но roadmap «ВЕСЬ ПЛАН ЗАВЕРШЁН» | ✅ Все стадии A1–A13, B1–B5, C1 реализованы; 32 сигнальных класса в `signals/` — подтверждено. 🟡 Нет регулярного запуска (rebalance по расписанию); XS→Agent мост есть (`live_factory`), но боевой путь — paper. |
| **Live-трейдинг / CCEA** | ✅ Production, «CCEA implemented» | ✅ Paper-контур в десктопе реально работает (order→firewall→journal→fill→PnL, E2E-тесты). 🟡 Live-брокеры: Alpaca/OANDA/Binance-futures/Deribit имеют ORDER_EXECUTION; ✅ Binance **spot** execution зарегистрирован (P0-C, 2026-07-16); ✅ `configs/agent.yaml` создан + `--dry-run` проходит (P0-D, 2026-07-16); ✅ Ed25519-гейт подписи модели на пути активации артефакта демона, fail-closed для LIVE (P0-E, 2026-07-16). |
| **Адаптеры** («Multi-Exchange ✅») | Binance/Alpaca/Polygon/Yahoo/OANDA/IB/Deribit/Theta/Dukascopy | См. матрицу §2. 🔴 theta_data — битый импорт; 🔴 dukascopy — заглушка 43 строки; 🟡 IB options не зарегистрированы в registry; 🟡 polygon — только market data (UI предлагает его для options). |
| **Риск-менеджмент** | ✅ гварды по всем классам, kill switch, pre-trade VaR | ✅ Kill switch, asset-гварды, pre-trade VaR/сценарии — код+тесты реальные. ✅ **Enforcement-разрыв ЗАКРЫТ 2026-07-16 (P0-B):** `lite_limits` (daily loss, max DD, leverage, concentration) теперь применяются двухуровнево — pre-trade RiskChecker + intra-day circuit breaker с auto-halt. `service_signal_runner.py` по-прежнему читает `max_total_notional`/`exposure`; CCEA-путь читает всё через `services/live_risk_limits.py`. См. [docs/RISK_LIMIT_ENFORCEMENT.md](docs/RISK_LIMIT_ENFORCEMENT.md). |
| **MLOps** (experiment tracking, Ed25519 registry, drift-retrain) | ✅ P0-4/P2 закрыты | ✅ Registry подписывает/проверяет (`service_experiment_tracking.py`), drift-движок с closed-loop есть. 🟡 Подпись **не проверяется при загрузке модели в live** (agentd не вызывает verify); ⬜ нет планировщика — drift-retrain только по REST-запросу. |
| **Учёт/комплаенс** (P&L ledger, blotter, hash-chain, MAR) | ✅ PP-1…PP-5 закрыты | ✅ Код и 70 тестов есть, on_fill подключён к CCEA-супервизору. 🟡 Live-глубина: EOD NAV — ручная кнопка; multi-currency NAV/corp actions на live-позиции — нет (P2 №14); налоговые лоты FIFO — нет (P2 №18). 🔴 `services.compliance` — битый импорт (см. §3). |
| **Desktop-приложение** | Tauri + sidecar, NSIS, «MVP 1:1» | ✅ Работает, packaged research-EXE собирается и обучает (2026-07-15), sidecar обновлён. ⬜ Нет автообновлений (tauri.conf без updater), лицензирования/активации, crash-reporting; 🟡 часть Pro-панелей — seeded-PRNG demo (честно бейджится): Regime, Attribution, Tearsheet, Consistency/Capacity. |
| **Тесты/CI** | «654+ файлов, 14 000+ функций» | ✅ Фактически **больше**: 747 файлов, 21 675 собираемых тестов; CI: build+test на ubuntu/windows, ruff, black, CCEA guardrails, coverage. 🟡 Часть тестов скипается без Cython-модулей; заявленные в claude.md точечные числа1 местами устарели (в меньшую сторону — реальность лучше). |
| **Данные** (PIT EDGAR, corp actions, universe, QC, feature store, TS-DB) | ✅ P0-2/P2 закрыты | ✅ Загрузчики stock/forex/options/EDGAR/calendar работают (Lite E2E проходит), data-QC + failover есть. 🟡 Feature store/TS-DB — сервисы существуют, но train/live их не обязаны использовать; corp actions лежат отдельно от live-позиций; ✅ минутки/тики закрыты 2026-07-16 (`services/premium_data.py`, §5.25); L2-глубины нет. |

---

## 2. Фактическая матрица адаптеров (registry, проверено 2026-07-15)

| Vendor | Market data | Streaming (WS) | Order execution | Прочее | Статус |
|---|---|---|---|---|---|
| Alpaca | ✅ | ✅ | ✅ (equity, paper/live) | fees/hours/info | Рабочий |
| Binance | ✅ spot | ✅ | ✅ spot / ✅ futures | fees/hours/info, futures MD | Spot-execution закрыт (P0-C, 2026-07-16) |
| OANDA | ✅ | — (polling) | ✅ (forex practice/live) | fees/hours/info | Рабочий |
| IB | ✅ futures | через ib_insync | ✅ futures (TWS/Gateway) | options.py есть, **не зарегистрирован** | Futures ок, options не подключены |
| Deribit | ✅ options | ✅ | ✅ options | inverse margin | Рабочий |
| Polygon | ✅ | ✅ | ⬜ | — | Только данные |
| Yahoo | ✅ | ⬜ | ⬜ | corp actions, earnings | Только данные |
| Theta Data | 🔴 битый импорт (`Bar` из adapters.models) | — | ⬜ | — | **Сломан** |
| Dukascopy | 🔴 заглушка (43 строки `__init__`) | ⬜ | ⬜ | — | **Не реализован** |

UI (`adaptersByAsset` в index.html) предлагает комбинации options→theta_data/polygon и forex→dukascopy,
за которыми нет рабочего бэкенда — backend честно вернёт ошибку, но выбор не должен предлагаться.

---

## 3. Сломано прямо сейчас (конкретика)

1. ✅ **ЗАКРЫТО 2026-07-16**: `packages.shared.models` импортируется — `TimeFrame`→`core_models`, `OrderSide`/`PositionSide`→`core_futures`, `MarketSnapshot`/`ExecutionMode`/`RiskLevel`/`ChangeClass` экспортированы из contracts. См. [docs/P0_ADE_CLOSURE_2026-07-16.md](docs/P0_ADE_CLOSURE_2026-07-16.md).
2. ✅ **ЗАКРЫТО 2026-07-16**: `services.compliance` импортируется — архив mifid (Financial-Entity, вне ICT-build) деградирует gracefully (`ARCHIVE_AVAILABLE=False`), CORE+INTEGRATION остаются.
3. ✅ **ЗАКРЫТО 2026-07-16**: `adapters.theta_data` импортируется — `Bar` берётся из `core_models` (как во всех адаптерах).
4. ✅ **ЗАКРЫТО 2026-07-16**: Binance **spot** ORDER_EXECUTION зарегистрирован (`adapters/binance/order_execution.py`) — live/panic для crypto-spot работают через registry-путь. См. [docs/CRYPTO_SPOT_EXECUTION.md](docs/CRYPTO_SPOT_EXECUTION.md).
5. ✅ **ЗАКРЫТО 2026-07-16**: `configs/agent.yaml` создан + `python -m packages.agent.daemon --config … --dry-run` проходит; команда запуска в claude.md исправлена. См. [docs/P0_ADE_CLOSURE_2026-07-16.md](docs/P0_ADE_CLOSURE_2026-07-16.md).
6. ✅ **ЗАКРЫТО 2026-07-16**: `lite_limits` теперь enforced — pre-trade RiskChecker (leverage/drawdown/daily-loss/concentration) + intra-day `LiveRiskMonitor` circuit breaker (auto-halt при пробое дневного убытка/просадки). `/api/risk/limits` возвращает `applied_to_agent: true` при живом Agent и перезагружает лимиты без рестарта. См. [docs/RISK_LIMIT_ENFORCEMENT.md](docs/RISK_LIMIT_ENFORCEMENT.md).

---

## 4. Есть, но не подключено / не закончено (wiring gap)

Из прошлого аудита осталось открытым всё P2 + добавления этой проверки:

7. ✅ **ЗАКРЫТО 2026-07-15/16**: Ed25519-подпись моделей проверяется при загрузке в live — `services/model_signature_gate.py` (enforce/warn/off, fail-closed до pickle-десериализации), проводка в `service_rl_inference` **и в самом демоне** (`RunController._verify_model_signature`, P0-E), REST `/api/models/verify_for_live` + `/api/agent/daemon/config`, 23 теста. См. [docs/MODEL_SIGNATURE_AND_REBALANCE.md](docs/MODEL_SIGNATURE_AND_REBALANCE.md), [docs/P0_ADE_CLOSURE_2026-07-16.md](docs/P0_ADE_CLOSURE_2026-07-16.md).
8. 🟡 Drift-retrain / авто-TCA — только по REST-вызову; **нет планировщика** (cron/scheduler-демона) ни в source, ни в десктопе.
9. ✅ **ЗАКРЫТО 2026-07-15** (полностью, включая live): регулярный XS-rebalance-раннер — `service_xs_rebalance.py` (веса → гардрейлы: turnover-cap/no-trade-band/концентрация → Intents → CCEA Agent OMS → журнал решений), планировщик job `xs_rebalance`, REST `/api/xs/rebalance/{run,last}`. **Авто-торговля на LIVE-брокере** открывается через CCEA operator-approval: `packages/agent/approval/live_trading_authorization.py` (мандат привязан к хешу конфига + брокеру + потолку лимитов + TTL + бюджету, hash-chained аудит, двухшаговая церемония `/api/ccea/live_trading/*`, авто-revoke при halt/смене брокера). Live-ордера идут через настоящий `LiveExecutionEngine`. 43 теста + live smoke. См. [docs/MODEL_SIGNATURE_AND_REBALANCE.md](docs/MODEL_SIGNATURE_AND_REBALANCE.md).
10. 🟡 L3-симулятор не питается реальными L2/L3-книгами (P2 №13).
11. 🟡 Multi-currency base-NAV, FX P&L, corp actions на live-позиции (P2 №14).
12. 🟡 Model-promotion gate + champion/challenger + замыкание drift→auto-retrain (P2 №16).
13. 🟡 Cardinality/round-lot/no-trade-band ограничения оптимизатора; полный 16-сценарный SPAN (P2 №17).
14. 🟡 Drop-copy, NBBO-роутинг на live MD, TCA impact-decomposition, FIFO/tax-lot учёт (P2 №18).
15. 🟡 GDPR/DORA: data_export/data_deletion есть, но DSAR/consent/breach/BCP не заведены в product runtime (P2 №19).
16. 🟡 PBT/SA-PPO/conformal — работают, но вне дефолтного пути обучения; нет пресетов «включи одним флагом» в Lite/Pro UI.
17. 🟡 Feature store и TS-DB-абстракция не используются обязательным образом train/live-контуром (декоративны без потребителя).
18. 🟡 IB options adapter написан (`adapters/ib/options.py`), но не зарегистрирован в registry.
19. 🟡 EOD NAV — ручная кнопка; нет автоматического EOD-процесса (расписание + отчёт).
20. 🟡 Pro-панели на seeded-PRNG (Regime, Attribution, Tearsheet, Consistency, Capacity) — честно помечены DEMO, но не заменены реальными расчётами, хотя данные для большинства есть (metrics.json, XS attribution).

---

## 5. Чего логически не хватает практикам

### 5.1 Квант-исследователь (ежедневная работа)
21. ⬜ **Планировщик задач** (обновление данных → пересчёт фичей → retrain → отчёт) — сейчас всё руками.
22. 🟡 **GPU ЗАКРЫТ 2026-07-16**: `services/hardware.py` (честная детекция: torch-сборка/CUDA/VRAM/nvidia-smi + причина и install-hint) + `train_model_multi_patch.py --device auto|cpu|cuda` → `DistributionalPPO(device=…)` + `run_train.params.device` из UI + `/api/hardware/gpu` + GPU-чип в Quant Lab. Распределённое (DDP/Ray) — остаётся ⬜. См. [docs/QUANT_GAPS_P2HIM_CLOSURE.md](docs/QUANT_GAPS_P2HIM_CLOSURE.md)
23. ⬜ Гиперпараметрический поиск как сервис (optuna в excludes сборки; PBT есть, но не как «tune-кнопка»).
24. ✅ **ЗАКРЫТО 2026-07-16**: экран сравнения экспериментов — `/api/experiments/{exp}/compare` (union params/metrics + differs) + reproducibility-бандл `/runs/{id}/bundle` (run+истории метрик+registry-ссылки+среда) + UI в Pro MLOps (чекбоксы→сравнение, лучшее зелёным, отличия жёлтым, «только отличия»). См. [docs/QUANT_GAPS_P2HIM_CLOSURE.md](docs/QUANT_GAPS_P2HIM_CLOSURE.md)
25. 🟡 **Минутки/тики ЗАКРЫТЫ 2026-07-16**: `services/premium_data.py` (честная entitlement-матрица binance/polygon/alpaca/oanda; минутки через оконную пагинацию `get_bars` → `data/minute/*` в схеме download_stock_data + sha256-манифест; **настоящий тиковый бэкфилл** Binance aggTrades с fromId-пагинацией → `data/ticks/*`) + CLI + `/api/data/premium/*` + карточка в Data Manager; реальный сетевой смоук (120 баров, 1834 тика). L2-глубина/borrow/delistings — остаются ⬜. См. [docs/QUANT_GAPS_P2HIM_CLOSURE.md](docs/QUANT_GAPS_P2HIM_CLOSURE.md)
26. ⬜ Воспроизводимость прогона одним артефактом: манифест «данные(hash)+конфиг+seed+код(sha)» создаётся частями (dataset_versioning, experiment tracking), но не единым бандлом.

### 5.2 Трейдер / PM
27. ✅ **ЗАКРЫТО 2026-07-16**: ручной ордер-тикет (market/limit/stop/stop-limit, TIF, reduce-only) через настоящий Agent OMS + панель рабочих ордеров с отменой — `submit_manual_order`/`open_orders`/`cancel_order` в supervisor, REST `/api/ccea/order/*`, UI-карточки в Lite Portfolio. 25 тестов + live smoke. См. [docs/MANUAL_ORDER_TICKET.md](docs/MANUAL_ORDER_TICKET.md).
28. ✅ **ЗАКРЫТО 2026-07-16**: частичное закрытие позиции из UI (кнопка «½» + `POST /api/portfolio/close {symbol, quantity}`) через тот же OMS-путь.
29. ⬜ Алерты «на телефон»: движок telegram/webhook есть (`services/alerts.py`) и используется runner-ом, но нет UI-настройки каналов и правил (PnL-порог, маржа, дисконнект).
30. ⬜ Multi-account / суб-счета (везде один счёт).
31. ⬜ Pre-market чеклист одним экраном (календарь сессий/праздников per-asset, статус фидов, маржа, gap-риски) — куски есть (session router, cme_calendar), сводного нет.
32. ⬜ Отчёт клиенту/себе за период (PDF/HTML: PnL, сделки, издержки, риск) — tear-sheet есть в XS-движке, не выведен как «отчёт за период по счёту».

### 5.3 Ops / эксплуатация
33. ⬜ Автообновление десктопа (Tauri updater не сконфигурирован) и канал доставки обновлений.
34. ⬜ Лицензирование/активация продукта.
35. ⬜ Crash-reporting/телеметрия ошибок (Sentry-класс) — сейчас только локальные логи.
36. ⬜ Бэкапы state/ledger (hash-chain журналы есть, автоматического бэкапа/восстановления нет).
37. 🟡 deploy/docker+helm существуют, но не проверены как рабочий путь (образы не собираются в CI).
38. ⬜ Prometheus/OTel-метрики процесса (healthchecks /health /ready /live есть; экспорта метрик нет, OpenTelemetry «not installed»).
39. ⬜ Ротация/ретенция логов (logs/ растёт бесконечно).
40. ⬜ Миграции схем состояния (state/*.json, ledger) между версиями приложения.

---

## 6. Полный приоритизированный список работ

### P0 — блокирует реальную работу с деньгами
| # | Работа | Ссылка |
|---|---|---|
| P0-A | ✅ **ЗАКРЫТО 2026-07-16**: `packages.shared.models` (TimeFrame→core_models, OrderSide/PositionSide→core_futures, +MarketSnapshot/ExecutionMode/RiskLevel/ChangeClass в contracts), `adapters.theta_data` (Bar→core_models), `services.compliance` (graceful degrade архива mifid — ICT-build, `ARCHIVE_AVAILABLE`). См. [docs/P0_ADE_CLOSURE_2026-07-16.md](docs/P0_ADE_CLOSURE_2026-07-16.md) | §3.1–3 |
| P0-B | ✅ **ЗАКРЫТО 2026-07-16**: двухуровневый enforcement `lite_limits` — pre-trade RiskChecker (leverage/drawdown/daily-loss/concentration из формы) + intra-day `LiveRiskMonitor` circuit breaker (day-loss / max-DD → auto-halt kill switch + флэттенинг). `services/live_risk_limits.py`, `packages/agent/policy/risk_checker.py`, проводка в `ccea/desktop_supervisor.py`, REST `/api/risk/enforcement`, Lite-карточка «Применение лимитов (live)»; 16 тестов + live smoke. См. [docs/RISK_LIMIT_ENFORCEMENT.md](docs/RISK_LIMIT_ENFORCEMENT.md) | §3.6 |
| P0-C | ✅ **ЗАКРЫТО 2026-07-16**: `adapters/binance/order_execution.py` (`BinanceOrderExecutionAdapter`, spot `/api/v3/*`, HMAC/RestBudgetSession по образцу futures) зарегистрирован для `BINANCE`+`BINANCE_US`; panic/holdings/close для crypto теперь исполняются через registry-путь (балансы→синтетические пары→market-SELL для флэттенинга); UI различает Binance Spot/Futures. 17 тестов + live smoke. См. [docs/CRYPTO_SPOT_EXECUTION.md](docs/CRYPTO_SPOT_EXECUTION.md) | §3.4 |
| P0-D | ✅ **ЗАКРЫТО 2026-07-16**: `configs/agent.yaml` (полная схема + safe defaults) + фикс `build_daemon_config` (stale `DegradedModeConfig` поля → реальные, Decimal-пороги kill-switch) + smoke `--dry-run/--dump-config` (6 тестов) + исправлена команда запуска в claude.md (`python -m packages.agent.daemon`). См. [docs/P0_ADE_CLOSURE_2026-07-16.md](docs/P0_ADE_CLOSURE_2026-07-16.md) | §3.5 |
| P0-E | ✅ **ЗАКРЫТО 2026-07-16**: Ed25519-гейт подписи модели теперь на пути активации артефакта в самом демоне (`RunController._verify_model_signature` → `packages/agent/daemon/model_gate.py`, тот же `verify_model_artifact`, что у RL-загрузчика), fail-closed для LIVE ДО десериализации pickle. MVP: `/api/agent/daemon/config` + карточка в Pro Security. 10 тестов. См. [docs/P0_ADE_CLOSURE_2026-07-16.md](docs/P0_ADE_CLOSURE_2026-07-16.md) | §4.7 |
| P0-F | ✅ **ЗАКРЫТО 2026-07-15**: планировщик `services/scheduler.py` + `configs/scheduler.yaml` + `/api/scheduler/*` + UI-карточка (anacron catch-up, fail-closed пайплайны, ретраи+алерты, CCEA-гейт торговых задач; 24 теста + live smoke). См. [docs/SCHEDULER.md](docs/SCHEDULER.md) | §4.8, §5.21 |

### P1 — нужно для продукта, которым пользуются каждый день
| # | Работа | Ссылка |
|---|---|---|
| P1-A | ✅ **ЗАКРЫТО 2026-07-16**: ручной ордер-тикет (limit/stop/TIF/reduce-only) + частичное закрытие + рабочие ордера через CCEA Agent OMS. См. [docs/MANUAL_ORDER_TICKET.md](docs/MANUAL_ORDER_TICKET.md) | §5.27–28 |
| P1-B | UI-настройка алертов (telegram/webhook) + правила (PnL, маржа, дисконнект, kill switch) | §5.29 |
| P1-C | XS rebalance-раннер по расписанию: weights→Intents→Agent + журнал решений + kill-критерии | §4.9 |
| P1-D | Автоматический EOD-процесс: NAV, сверка позиций с брокером, отчёт за день | §4.19, §5.32 |
| P1-E | Убрать из UI вендоров без бэкенда (theta_data/dukascopy/polygon-options) или реализовать их | §2 |
| P1-F | Зарегистрировать IB options в registry + тест create_adapter на каждую пару UI | §4.18 |
| P1-G | Автообновление десктопа (Tauri updater + подпись) | §5.33 |
| P1-H | Замена 5 demo-панелей Pro на реальные данные (Regime — из drift/vol; Attribution/Tearsheet — из XS-движка) | §4.20 |
| P1-I | Real-backtest trust-report для futures/forex/options (по образцу P0-1 crypto/equity) | §1 бэктест |
| P1-J | Model-promotion gate + champion/challenger (замкнуть drift→retrain→promote) | §4.12 |
| P1-K | Sentry-класс crash-reporting + ротация логов | §5.35, §5.39 |

### P2 — институциональная зрелость (наследуется из PRO_PIPELINE §P2 + новое)
| # | Работа |
|---|---|
| P2-A | Реальные L2/L3-книги в `lob/` (данные глубины) |
| P2-B | Multi-currency NAV + FX P&L + corp actions на live-позиции |
| P2-C | Regime-conditioned лимиты; research-attribution |
| P2-D | Cardinality/round-lot/no-trade-band; полный 16-сценарный SPAN |
| P2-E | Drop-copy, NBBO-роутинг, TCA impact-decomposition, FIFO/tax-lots |
| P2-F | GDPR/DORA (DSAR/consent/breach/BCP) в product runtime |
| P2-G | Multi-account/суб-счета |
| P2-H | 🟡 **GPU ЗАКРЫТ 2026-07-16** (`services/hardware.py` + `--device` + UI, см. §5.22); распределённое обучение + optuna-tune — остаются |
| P2-I | ✅ **ЗАКРЫТО 2026-07-16**: экран сравнения экспериментов + reproducibility-бандл (см. §5.24) |
| P2-J | Prometheus/OTel-метрики; docker/helm в CI; бэкапы+миграции state |
| P2-K | Лицензирование/активация |
| P2-L | Feature store/TS-DB как обязательный путь данных (или удалить) |
| P2-M | 🟡 **Минутки/тики ЗАКРЫТЫ 2026-07-16** (`services/premium_data.py` + Binance aggTrades тик-бэкфилл + entitlement-матрица + UI, см. §5.25); borrow/delistings — остаются |

### P3 — полировка
| # | Работа |
|---|---|
| P3-A | Pre-market чеклист-экран (сессии, фиды, маржа) |
| P3-B | Периодический отчёт клиенту (PDF) из tear-sheet |
| P3-C | Обновить точечные числа тестов в claude.md (реальность лучше заявленного) |
| P3-D | UI-конфиг PBT/SA-PPO/conformal пресетами |

---

## 7. Что подтверждено работающим (чтобы список выше не читался как приговор)

- RL-стек end-to-end, включая packaged EXE (обучение → артефакт → load-back), 2026-07-15.
- Lite-цепочка данных целиком subprocess-воркерами (E2E-тест), LeakGuard ≥ 8000 мс.
- CCEA paper-контур: ордер → firewall → журнал → fill → PnL → halt → restart/persistence.
- Брокеры: Alpaca (equity), OANDA (forex), Binance futures, Deribit options — market data + исполнение.
- Оптимизатор (MVO/BL/robust/multi-period/Kelly), BARRA-риск-модель, CPCV/PBO/DSR, 32 сигнала, XS-вертикали B1–B5+C1.
- Books-and-records: P&L ledger, blotter, cash GL, hash-chain, instrument master, MAR-детекторы (+70 тестов).
- Honesty-контур MVP: всё нереальное помечено DEMO/SIMULATED, fail-closed Emergency Halt (аудит 2026-07-14 закрыт полностью).
- CI: GitHub Actions (ubuntu+windows), ruff, black, CCEA guardrails, coverage; 21 675 тестов собираются.

---

1 Примечание: точечные числа тестов в claude.md (например «654+ файлов») занижены — фактически 747 файлов / 21 675 тестов;
критичные для честности числа лучше генерировать, а не хардкодить (P3-C).

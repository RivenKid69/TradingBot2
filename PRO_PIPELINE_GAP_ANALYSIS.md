# Pro-Quant Pipeline Gap Analysis (RivenQuant / CustodiaCloud)

> **Назначение.** Сверка полного рабочего пайплайна институционального квант-фонда
> (data → research → portfolio → risk → execution → post-trade → compliance) с тем,
> что **реально реализовано в коде** этого репозитория. Каждое утверждение
> code-grounded (cite file:line в подробных секциях). Метрики покрытия — экспертная
> оценка по результатам 7 параллельных аудитов кода (2026-06-15).
>
> **НЕ маркетинг.** Это внутренний инженерный документ. Статусы «есть/частично/нет»
> отражают состояние кода, а не сертификацию.

---

## 0. Главный вывод (TL;DR)

Движковая часть платформы **уже институционального уровня по реализации**: оптимизатор
(MVO/min-var/risk-parity/Black-Litterman/Kelly/robust/multi-period), BARRA-style риск-модель
(+ Ledoit-Wolf, PCA, EWMA), cross-sectional backtest с purge/embargo, CPCV+PBO+Deflated Sharpe,
OMS с идемпотентностью/журналом/parent-child, TCA, FIX 4.4 session engine, SOR, market-abuse
детекторы, Ed25519-подписанный model registry, AES-GCM vault — **всё это настоящий код, не заглушки.**

**Системная слабость — НЕ отсутствие функционала, а «проводка» (wiring) и источники данных:**
самый сильный код часто (1) достижим только из Python API/тестов, но не из YAML-конфига или
live-пути, либо (2) питается mock-данными в MVP-эндпоинтах. Этот паттерн повторяется в 5 из 7
доменов. Поэтому путь к «100% покрытия про-потребностей» — это **на ~60% работа по интеграции
уже написанного** и на ~40% — добитие реально отсутствующих кусков (firm-wide риск-агрегатор,
live P&L ledger, instrument master, IS-исполнитель).

### Scorecard покрытия

| # | Домен пайплайна | Реализация движка | Live-проводка | Итоговое покрытие |
|---|------------------|:---:|:---:|:---:|
| 1 | Data & reference | ⭐⭐⭐⭐ | ⭐⭐⭐ | **~70%** |
| 2 | Research / alpha / backtest | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **~85%** |
| 3 | Portfolio construction & risk-model | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **~80%** |
| 4 | Risk management | ⭐⭐⭐⭐ | ⭐⭐⭐ | **~70%** |
| 5 | Execution / OMS / EMS / TCA | ⭐⭐⭐⭐ | ⭐⭐⭐ | **~70%** |
| 6 | Post-trade / accounting / attribution | ⭐⭐⭐ (sim) | ⭐⭐ (live) | **~55%** |
| 7 | Compliance / governance / MLOps | ⭐⭐⭐⭐ | ⭐⭐ | **~60%** |

---

## 1. DATA & REFERENCE LAYER — ~70%

**✅ Есть:** мульти-венью адаптеры L1+бары+тики RT/историч. (Binance/Alpaca/Polygon/Yahoo/IB/OANDA/Deribit),
corporate actions (splits/divs/M&A типы), continuous-futures roll (back-adjust, честный `pit_quality`),
PIT-фундаментал через SEC EDGAR (`services/edgar_fundamentals.py`), survivorship-free index membership
(`services/index_membership_loader.py`, `services/survivorship.py`), PIT alt-data COT+календарь
(`loaders/altdata_enrich.py`), feature store с content-hash версиями + online-кэш (`service_feature_store.py`),
TSDB-абстракция ClickHouse/Timescale/parquet (`services/tsdb.py`), trading-календари с DST/half-days,
data-trust gate блокирующий бэктест на `pit_quality=none`.

**⚠️ Частично:** market-data глубина — **только L1** из живых фидов (L2/L3 есть только как `lob/` симулятор,
не как ingestion реальных книг; единственный реальный `get_order_book` — Deribit options);
`data_validation.py` — жёсткий fail-fast валидатор (nulls/inf/OHLC), но **без spike/outlier/staleness
детекции и без cross-source сверки**; дивидендная корректировка off по умолчанию; alt-data только COT+календарь.

**❌ Нет:**
- **Instrument master / symbology** (FIGI/CUSIP/ISIN/SEDOL/OCC mapping) — нет вообще. Символы = сырые тикеры вендора.
- **Vendor failover / cross-source reconciliation** market-data.
- **Staleness/spike/outlier watchdog** на ingestion.
- **Restatement-aware bitemporal store** фундаментала (10-K/A supersede).

**→ Что доработать:**
1. `services/instrument_master.py` — канонический `InstrumentId` по FIGI (OpenFIGI бесплатен) + cross-ref. **Крупнейший gap.**
2. Реальный L2/L3 ingestion (`stream_depth` для Binance/IB) → кормить `lob/` реальными книгами.
3. `services/data_quality.py` — session-aware gaps, MAD/z-spike, staleness heartbeat, cross-vendor дивергенция → в Data-Trust report.
4. `MarketDataRouter` поверх `adapters/registry.py` с primary→secondary failover + health/circuit-breaker.

---

## 2. RESEARCH / ALPHA / BACKTEST — ~85% (сильнейший домен)

**✅ Есть:** каталог из **32 факторных сигналов** (crypto/equity/futures/forex/options/common — все с реальным
`compute_panel`), комбинирование (Equal/IC-weighted/Ridge/GBM, neutralize по OLS), IC/IR/ic_decay/turnover
диагностика (`impl_signal_diagnostics.py`), **PIT-корректный XS backtest** с purge+embargo (`service_xs_backtest.py`,
`splits.py`), **CPCV + PBO(CSCV) + Deflated Sharpe + PSR + min-TRL** (`research/cv_overfitting.py`, реально и строго),
capacity-анализ (Almgren-Chriss, `impl_capacity.py`), experiment tracking с Ed25519-подписью + git-lineage
(`service_experiment_tracking.py`, dirty-tree блокирует промоушн), feature-parity online/offline (1e-9),
RL-as-signal (CVaR-utility × conformal-confidence, `impl_rl_signal.py`). **Синтетика честно огорожена баннером.**

**⚠️ Частично:** event-driven и cross-sectional — **два разных движка** (RL-путь `service_backtest.py` vs XS
`service_xs_backtest.py`), не один; equity-сигналы по умолчанию `pit_quality='approx'` (EDGAR PIT есть, но не
подключён в `signals/equity_signals.py`); CPCV/PBO **есть, но не включены** в стандартные real-data sweeps
(в `trust_report` PBO считается только если передать матрицу путей).

**❌ Нет:**
- **Block/stationary bootstrap CI** на Sharpe/DD (Politis-Romano).
- **Monte-Carlo path simulation** для распределения max-DD.
- **Regime-conditioned attribution** на research-слое (Sharpe по VIX/trend-режимам; режимы есть только в training-env).
- Долгосрочный **signal-decay/crowding мониторинг** как стоящий отчёт.

**→ Что доработать:**
1. Прокинуть CPCV/PBO в `tools/xs_*_real_*.py` (построить `(T×N)` OOS-матрицу → в `trust_report`). **Дёшево, превращает строгость в отчётную строгость.**
2. `research/bootstrap.py` — block-bootstrap CI + p-value в trust report.
3. Regime-conditioned срез XS-доходностей (переиспользовать существующий regime-labeller).
4. Подключить EDGAR PIT в equity-сигналы (убрать `approx` по умолчанию).

---

## 3. PORTFOLIO CONSTRUCTION & RISK-MODEL — ~80%

**✅ Есть (всё реальная математика, не заглушки):** в `service_optimizer.py` — MVO, min-var, max-sharpe,
true ERC risk-parity, Black-Litterman (полный P/Q/Ω/τ), Kelly, vol-target, **robust (box + ellipsoidal)**,
**multi-period (Gârleanu–Pedersen)**, **tcost-aware objective** (κ·L1+quad + per-name √-impact Almgren-Chriss,
решается SLSQP). Ограничения: long/short, gross/net, box, turnover, **sector caps, factor caps** (в SLSQP и cvxpy).
Риск-модель `service_risk_model.py`: BARRA-fundamental Σ=BFBᵀ+D, statistical (Ledoit-Wolf + PCA), EWMA factor-cov,
nearest-PSD conditioning. Cross-asset unified (`service_cross_asset.py`), options-greeks-neutral (`service_options_portfolio.py`),
отдельный pre-trade constraint engine с GICS (`services/portfolio_constraints.py`).

**⚠️ Частично — главный структурный gap «config wiring»:** `build_optimizer()` прокидывает из YAML только
`gross_max/net_target/long_only/max_position/max_turnover`. Из конфига **НЕЛЬЗЯ включить** уже написанные
sector_caps, factor_caps, robust, black_litterman views, multi-period — **~40% мощи оптимизатора «тёмная» из YAML**.
EWMA только на factor-уровне (asset-Σ — плоское окно). Sector caps — gross, не signed. Нет hard ADV-participation cap.

**❌ Нет:** beta-neutral как first-class constraint (βᵀw=0), cardinality / round-lot / no-trade-band,
country/region caps (есть только в compliance-конфигах), per-name spread/borrow/short-rebate в objective,
альтернативные shrinkage-таргеты (constant-correlation, OAS), DCC-GARCH.

**→ Что доработать:**
1. **Расширить `OptimizerCfg` и `build_optimizer()`** — поля sector_caps/factor_caps/robust/bl_views/multi_period. **Высочайший ROI при низком усилии** — оживляет уже написанное.
2. `beta_neutral: bool` (βᵀw=0 из market-beta колонки) + `max_participation` (|Δw|·NAV ≤ frac·ADV) — обе тривиальны (ADV уже течёт в tcost).
3. `cov_method`+`ewma_halflife` в `RiskCfg` для asset-Σ пути; +1 альт. shrinkage-таргет.
4. Cardinality (top-N эвристика + re-opt) и no-trade-band; country_map/country_caps (реюз sector-механики).

---

## 4. RISK MANAGEMENT — ~70%

**✅ Есть:** portfolio-level pre-trade **VaR/CVaR (parametric+historical), stress-grid, factor-exposure
limits** — и это **реально блокирует отправку** ордеров (`service_pretrade_risk.py` → `service_xs_live.py:200-216`
`pretrade_check(strict=True)` → `sent=False`). Drawdown/notional/bankruptcy guards (`risk_guard.py`),
intraday factor-monitor, мульти-канальный alerting с эскалацией L1→L5 + dedup + persistence
(`services/core/alerting.py`), ops kill-switch с crash-recovery, CME Rule 80B circuit breakers, SPAN-маржа,
crypto-futures tiered leverage/ADL/funding guards, forex stop-out levels, stock PDT/Reg-T/Rule-201,
conformal CVaR bounds → position scaling, drift PSI/KS/Wasserstein.

**⚠️ Частично:** Monte-Carlo VaR **нет** (только parametric+historical); stress = только синтетические шоки
(нет именованных 2008/2020 сценариев, нет liquidity-stress); режимы есть, но **не заведены в live risk-gates**
(только training); SPAN упрощён (1 worst-scenario вместо 16-массива, опционы выкл); `PortfolioRiskSummary`
у unified-futures **декларирует VaR-поля, но не считает их**.

**❌ Нет (ключевой институциональный пробел):**
- **Firm-wide / иерархический (strategy→desk→firm) лимит-фреймворк.** Каждый домен (XS-gate, futures, forex, stock)
  проверяет свой срез — **нет общего агрегатора** в единый consolidated VaR/CVaR + gross/net/sector/factor.
- **Firm-wide real-time VaR-агрегация** по всем классам одновременно.
- **Risk attribution: marginal / component / incremental VaR** — нет вообще.
- **Reverse stress testing**, liquidity-adjusted VaR (ADV-горизонт ликвидации), cross-asset margin netting.

**→ Что доработать:**
1. **`service_firm_risk.py` — firm-wide агрегатор** (позиции из всех книг → общее exposure/returns → consolidated VaR/CVaR против иерархических лимитов). **Главный gap домена.**
2. Monte-Carlo VaR в `PreTradeRiskAnalyzer` (`np.random.multivariate_normal` + t-dist).
3. Component/marginal VaR (∂VaR/∂wᵢ·wᵢ) — атрибуция риска на позиции/факторы.
4. Библиотека именованных историч. сценариев (GFC/COVID/flash-crash/CHF/2022-rates) + reverse stress + liquidity-stress.
5. Завести `market_regimes.json` в live risk-limits (ужесточать VaR в STRONG_TREND/ILLIQUID).
6. Global kill-switch по firm P&L/drawdown (сейчас только по connectivity/error).

---

## 5. EXECUTION / OMS / EMS / TCA — ~70%

**✅ Есть (live-wired):** OMS — state-machine, parent/child (`child_executor.py`), **идемпотентность**
(детерминированный client_order_id + journal dedup + recovery), durable order journal (log-before-submit),
partial fills (notional-weighted avg). EMS — TWAP/VWAP/POV slicing weight-delta (`service_xs_execution.py`),
clock-driven child release + cancel-replace стрэгглеров. SOR — venue-selection + water-filling по marginal cost
(`smart_order_router.py`). FIX **4.4 session engine** (logon/heartbeat/seqnum/resend/gapfill/TLS, `fix_session.py`).
Broker REST-адаптеры Alpaca/IB/OANDA (submit/cancel/replace/bracket). **TCA** — pre-trade estimate + post-trade
arrival/VWAP slippage + IS + by-venue (`tca_compliance.py`, `tca_reporter.py`). Resilience (retry/CB/poll/reconcile),
rate-limit/hard-caps, allocation (VWAP block + give-up/CMTA) + T+1/T+2 ClearingEngine.

**⚠️ Частично:** **IS как live-алго — нет** (TCA его рекомендует, AC optimal-horizon считается, но
исполнитель только TWAP/VWAP/POV); engine-level **cancel/amend нет** (только straggler-cancel; FIX 35=G отсутствует);
**SOR не подключён к live-пути** (`/api/exec/route` = синтетические venue + FIX-preview строка; `live_factory`
шлёт одному брокеру); FIX не сертифицирован ни с одним реальным контрагентом (все live-адаптеры REST/SDK);
TCA impact-decomposition (permanent/temporary/reversion) — **поля есть, не заполняются**.

**❌ Нет:** drop-copy session, live MD-session интегрированная с OMS/SOR (NBBO-routing), FIX 35=G amend,
iceberg/liquidity-seeking как **live** алго (есть только в `lob/` симуляции), price-collar / fat-finger
(только size/loss-капы, нет price-band), авто-routing real fills в block allocation.

**→ Что доработать:**
1. **Подключить SOR + venue-analysis в live submission** (`build_live_stack` принимает `SmartOrderRouter`, `child_executor` зовёт `route_live`+`dispatch`). Сейчас SOR — dead code для live-ордеров.
2. **`ISExecutor`** на базе `AlmgrenChrissModel.compute_optimal_execution_time` (front-loaded schedule).
3. **Engine-level `cancel_order`/`replace_order` + FIX 35=G**.
4. **Price-collar / fat-finger gate** в `engine.execute` (limit within X% of mid, notional ≤ Y×ADV).
5. Заполнить TCA permanent/temporary/reversion (reversion-окна + AC-модель).
6. Drop-copy consumer + production `LiquidityProvider` на `adapters/*/market_data.py`.

---

## 6. POST-TRADE / ACCOUNTING / ATTRIBUTION — ~55% (слабейший по live-готовности)

**✅ Есть:** в **симуляторе** — realized/unrealized split + average-cost lot accounting (`execution_sim.py:8603`),
MtM (bid/ask/mid), fees+funding в equity, P&L regression-тест. Reconciliation vs broker
(`packages/agent/reconciliation/reconciler.py`, `services/position_sync.py` — auto-reconcile, halt/flatten),
cash/account reconcile. **Attribution** — exact factor P&L (`r=Bf+u`), signal attribution, **Brinson-Hood-Beebower**
(`service_attribution.py`). Analytics — Sharpe/Sortino/Calmar/IR + benchmark-relative, **LP-grade HTML tear-sheet**
(`service_tearsheet.py`, GIPS gross-vs-net, Trust Report). Settlement T+1/T+2 holiday-aware, block allocation,
corporate-action на ценовые серии. **MiFIR Art.25 audit trail с hash-chain** (`audit_trail_writer.py`), durable order journal.

**⚠️ Частично:** lot accounting **только average-cost** (нет FIFO/tax-lot; в SimBroker cost_basis вообще
перезаписывается последним fill — баг paper-брокера); tear-sheet **только из XS-бэктеста**, не из live/blotter;
clearing status = константа `"allocated"`; GIPS — только gross/net, без composite/dispersion.

**❌ Нет (live):**
- **Live realized/unrealized P&L ledger в Agent** — `PortfolioState` хранит только equity для pre-trade risk; equity подаётся извне. Нет дневного/EOD close, нет NAV-snapshot.
- **Immutable trade blotter** (trade-economics: gross/net/fees/financing/settlement-ref) как official record + **cash ledger / treasury** (нет вообще).
- **Multi-currency / base-currency NAV + FX P&L** (account currency хардкод USD; нет price-vs-currency декомпозиции).
- **Corporate actions на LIVE позиции** (split share-adjust, dividend cash на pay-date) — есть только на историч. серии.
- **Break-resolution workflow** (aging/investigation/resolved state machine), **settlement tracking** (pending→settled/failed).

**→ Что доработать:**
1. **Live P&L engine в Agent** — потребляет `FillHandler` fills → per-symbol avg-cost/FIFO, realized на close, unrealized vs live-marks, fees+financing, day/EOD close. Переиспользовать инвентарную логику `execution_sim.py:8603`. **Главный gap.**
2. **Immutable trade blotter + cash ledger** (append-only trade-economics + GL для fees/interest/financing/dividends).
3. **Multi-currency base-NAV + FX P&L** декомпозиция.
4. Применять corporate actions к live-позициям (share-adjust + dividend cash booking).
5. Settlement/clearing lifecycle (pending→settled/failed) + break-resolution state machine.
6. FIFO/specific-lot опция + фикс cost-basis в `sim.py`.
7. Tear-sheet из **live** realized P&L (после #1/#2).

---

## 7. COMPLIANCE / GOVERNANCE / MLOps — ~60%

**✅ Есть (реальный код, не markdown):** market-abuse детекторы spoofing/layering/wash/marking-close
(`services/algo_integration/market_abuse.py`), **OTR/order-to-trade RTS 6/9 — wired и кормится live-потоком**
(`otr_monitor.py`, `app.py:3960`), best-execution MiFID Art.27 (`best_execution.py`), **model registry с
Ed25519-подписью + versioning/staging/rollback + git-lineage** (`service_experiment_tracking.py`), drift PSI/KS/Wasserstein,
**AES-256-GCM vault + PBKDF2-100k** (`local_vault.py`), retention 7-лет с legal-hold (`retention_service.py`),
RBAC default-deny + MFA + break-glass, tamper-evident hash-chains, EU AI Act (12/21 модулей с реальной логикой).

**⚠️ Частично:** **market-abuse — ОРФАН** (импортируется только в тестах, не кормится live-потоком — детекторы
ничего не наблюдают в проде); best-execution питается **mock-данными** (`app.py:468-485`); TCAReporter не
инстанцирован; **drift→retrain detection-only** (`run_closed_loop` без вызывающего, не запускает обучение);
order journal **мутабельный** (in-place UPDATE, без hash-chain/HMAC, несмотря на «audit» в docstring);
API audit-middleware auto-persist = заглушка; GDPR-эндпоинты работают на **mock-репах**; model-cards = статические
шаблоны (хардкод Sharpe 1.2).

**❌ Нет (как live-enforced):** model-promotion approval workflow (текст в `proToast`, нет backend);
champion/challenger (хардкод в `index.html`, нет traffic-split); WORM/object-lock; incident-mgmt/DR/BCP
(код **заархивирован**, unwired); governance DSAR/consent/breach достижимы только из CCEA Cloud + тестов, не из `app.py`.

**→ Что доработать:**
1. **Завести market-abuse surveillance в live order/fill путь** (instantiate `MarketAbuseMonitor`, `record_order/trade` из `app.py:3960` + agent engine, alerts → bus). **Главный surveillance gap.**
2. **Tamper-evident order journal** — prev_hash + HMAC (ключ из vault), append-only status-events, `previous_hash` колонка в DB.
3. **Заменить mock-источники реальными** (best-ex fill stream, GDPR на реальный datastore, TCAReporter на реальный trade-log) — или честно метить `simulated=True`.
4. **Model-promotion approval gate** (approver + sign-off + two-person) + `drift_retrain.run_closed_loop` реально зовёт `train_model_multi_patch.py`.
5. Завести governance + DR-слой в runtime (эндпоинты/шедулеры) либо явно scope «Cloud-only».
6. Привязать model-cards к реальным `ModelVersion`/`RunRecord`.

---

## 8. Консолидированный roadmap (приоритизированный)

### P0 — без этого «pro-fund на 100%» не закрывается (реально отсутствующее ядро)
1. ✅ **СДЕЛАНО (2026-06-15).** **Firm-wide иерархический risk-агрегатор** (`service_firm_risk.py`) — consolidated
   VaR/CVaR (parametric+historical) + strategy→desk→firm лимиты + Euler component/marginal/incremental VaR +
   diversification benefit. Academic grounding: Artzner et al. (1999, coherence/субаддитивность), Rockafellar–Uryasev
   (2000/2002, CVaR), Tasche/Litterman (Euler allocation). REST: `/api/firm_risk/aggregate` (real posted),
   `/api/firm_risk/demo` (представительный, флаг simulated, втягивает live agent-book). MVP-карточка «Firm-Wide Risk»
   в Pro-дашборде. Тесты: `tests/test_firm_risk.py` (16). *(Домен 4)*
2. ✅ **СДЕЛАНО (2026-06-15).** **Live realized/unrealized P&L ledger в Agent**
   (`packages/agent/accounting/pnl_ledger.py`) — average-cost + FIFO, realized/unrealized/fees/financing, day-P&L,
   EOD NAV-snapshot, SQLite-персист + crash-recovery, `to_portfolio_state()` (engine берёт equity из ledger, а не
   извне). Wired: FillHandler `on_fill`→ledger (через `ledger_fill_callback`), `build_live_stack(pnl_ledger=...)`,
   CCEA-супервизор (paper-путь через настоящий FillHandler), SimBroker cost-basis fix. REST:
   `/api/agent/pnl/{status,nav_history,eod_close}` + `pnl_ledger` в `/api/ccea/status`. MVP: P&L-блок в CCEA-карточках
   (Home+Pro) + кнопка EOD NAV. Тесты: `tests/test_pnl_ledger.py` (15). *(Домен 6)*
3. ✅ **СДЕЛАНО (2026-06-15).** **Instrument master / symbology** (`services/instrument_master.py`) — канонический
   FIGI-keyed `InstrumentRecord` + cross-ref CUSIP/ISIN/SEDOL/OCC/тикер; валидаторы check-digit (ISO 6166 ISIN,
   ANSI X9.6 CUSIP, SEDOL, OMG FIGI), OCC 21-char parse/build, offline-seed + опц. OpenFIGI. REST:
   `/api/instruments/{resolve,search,list,occ_parse}`. Wired: blotter аннотирует каждую сделку FIGI; глобальный
   `get_default_master()` в app.py. MVP: lookup-виджет в карточке «Books & Records». Тесты:
   `tests/test_instrument_master.py` (19). *(Домен 1)*
4. ✅ **СДЕЛАНО (2026-06-15).** **Market-abuse surveillance + tamper-evident journal в live-путь.**
   Surveillance: глобальный `MarketAbuseMonitor` (был orphan) подключён в order-approval (`api_approve`) +
   `record_execution` (record_order/record_trade), и в CCEA-агент через `BooksAndRecords.on_order/on_fill` (живой
   fill-поток). Tamper-evident journal: append-only hash-chained `order_audit` (HMAC keyed из vault,
   Schneier–Kelsey/RFC 2104) + `verify_audit_chain()`. Hash-chain helper: `packages/agent/audit/hash_chain.py`.
   REST: `/api/surveillance/market_abuse`, `/api/agent/journal/integrity`. MVP: surveillance + integrity-бейджи в
   карточке «Books & Records». Тесты: `tests/test_journal_tamper_evident.py` (4) + surveillance в books-тестах. *(Домен 7)*
5. ✅ **СДЕЛАНО (2026-06-15).** **Immutable trade blotter + cash ledger** (books-and-records).
   `packages/agent/accounting/blotter.py`: `TradeBlotter` (append-only, hash-chained, полная экономика сделки + FIGI +
   settlement T+N) + `CashLedger` (append-only GL, hash-chained, running balance, типы TRADE/FEE/FINANCING/...).
   Фасад `packages/agent/accounting/books.py` (`BooksAndRecords`) связывает PnLLedger+blotter+cash+instrument-master+
   surveillance, единый `on_fill()`; cash GL сверяется с P&L cash. Wired в CCEA-супервизор (FillHandler→books).
   REST: `/api/agent/{blotter,cash_ledger}`. MVP: blotter+cash таблицы в карточке «Books & Records». Тесты:
   `tests/test_books_and_records.py` (16). *(Домен 6)*

### P1 — «оживить уже написанное» (высокий ROI, в основном wiring) — ✅ ВСЕ СДЕЛАНЫ (2026-06-15)
6. ✅ **СДЕЛАНО.** `OptimizerCfg`/`build_optimizer()` — sector/factor caps, robust (box/ellipsoidal), BL-views, multi-period
   (Gârleanu–Pedersen) + **beta-neutral** из YAML; также в `/api/xs/optimize`. `service_xs_pipeline.py`,
   `service_optimizer.py` (`MultiPeriodOptimizer.solve`). Тесты: `test_optimizer_config_p1.py` (7). *(Домен 3)*
7. ✅ **СДЕЛАНО.** SOR подключён в live submission — `routed_broker_submit` + `BrokerLiquidityProvider` в
   `build_live_stack`; `/api/exec/route?dispatch=true` реально диспатчит child-ордера по (paper) venues. Pro-карточка
   «Execution & Data-QA». `live_factory.py`, `smart_order_router.py`. Тесты: `test_sor_live_p1.py` (4). *(Домен 5)*
8. ✅ **СДЕЛАНО.** CPCV/PBO в sweeps (`tools/xs_crypto_real_sweep.py` строит T×N OOS-матрицу → PBO) + block-bootstrap
   CI (Politis–Romano) в `trust_report`. `research/bootstrap.py`. Тесты: `test_bootstrap_pbo_p1.py` (8). *(Домен 2)*
9. ✅ **СДЕЛАНО.** Monte-Carlo VaR (Gaussian/Student-t) + Euler component/marginal/incremental VaR + именованные
   историч. сценарии (2008/2020/2010/2015/2018/2022) в `PreTradeRiskAnalyzer`; отдаётся через `/api/xs/pretrade_risk`.
   `service_pretrade_risk.py`. Тесты: `test_pretrade_risk_p1.py` (7). *(Домен 4)*
10. ✅ **СДЕЛАНО.** Live IS/Almgren-Chriss executor (front-loaded `_is_profile`) в `RebalanceScheduler` +
    engine-level `cancel_order`/`replace_order` + FIX **35=G** (`order_cancel_replace_request`) + **price-collar/
    fat-finger gate** (`PriceCollarConfig`). Wired в CCEA-супервизор + `/api/xs/execution_plan?algo=IS`.
    Тесты: `test_execution_p1.py` (11). *(Домен 5)*
11. ✅ **СДЕЛАНО.** `services/market_data_quality.py` — robust spike (MAD/Hampel), staleness, frozen, session-aware
    gap, OHLC QC + cross-vendor reconcile + `MarketDataRouter` (primary→secondary failover + circuit-breaker).
    REST `/api/data_quality/{check,demo}` + Pro-карточка. Тесты: `test_market_data_quality_p1.py` (13). *(Домен 1)*
    *(beta-neutral и ADV-participation cap из бывшего #12 включены в #6/#10.)*

### P2 — зрелость / полнота
13. Real L2/L3 depth ingestion → кормить `lob/` реальными книгами. *(Домен 1)*
14. Multi-currency base-NAV + FX P&L + corporate actions на live-позиции. *(Домен 6)*
15. Regime-conditioned risk-limits + research-attribution. *(Домены 2,4)*
16. Model-promotion approval gate + champion/challenger + drift→auto-retrain замыкание. *(Домен 7)*
17. Cardinality/round-lot/no-trade-band + country caps; full 16-scenario SPAN. *(Домены 3,4)*
18. Drop-copy + live MD-session NBBO-routing; TCA impact-decomposition; FIFO/tax-lot. *(Домены 5,6)*
19. Завести governance/DR (DSAR/consent/breach/BCP) в product runtime. *(Домен 7)*

---

## 9. Сквозная тема: «wiring gap»

Повторяющийся паттерн (5 из 7 доменов): **сильный, протестированный движок существует, но не подключён
к live-пути или питается mock-данными.** Примеры: SOR/FIX/IS-алго (execution), robust/BL/multi-period
оптимизатор (portfolio), market-abuse/GDPR/DR (compliance), CPCV/PBO (research). Это **хорошая новость**:
большая часть пути к 100% — не R&D, а интеграция уже оплаченного кода. Рекомендация — приоритезировать
P1-блок «оживления» параллельно с P0-ядром, и **честно метить `simulated=True`** все surface-ы, которые
пока на mock-данных, до момента их реального подключения (часть уже метит — OTR; best-ex и GDPR — нет).

---

**Дата:** 2026-06-15 · **Метод:** 7 параллельных code-grounded аудитов (Grep/Read) · **Статус:** внутренний инженерный анализ, не сертификация.

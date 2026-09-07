# Cross-Sectional Platform — Поэтапный план реализации (Build Roadmap)

**Назначение:** исполняемый по этапам план. Пользователь выдаёт один Stage → агент выполняет его **полностью**
(код + тесты + интеграция + UI-крючки), затем следующий. Итерационно доводим до 100%.

**Архитектурная цель:** перевести продукт в парадигму **«десятки сигналов → риск-модель → портфельная
оптимизация по всему юниверсу»**, на одном asset-agnostic движке, с вертикалями под все 5 классов активов.
RL (Distributional PPO) сохраняется как **один из сигналов**.

> Связанный дизайн-док (архитектура, формулы, контракты): `CROSS_SECTIONAL_PLATFORM_DESIGN.md`.

---

## 0. Принципы (соблюдать в каждом Stage)

1. **Аддитивность.** Только новые файлы + новый `mode: cross_sectional`. Существующий single-instrument /
   RL-MVP не трогаем функционально. Все текущие тесты остаются зелёными. Никаких breaking changes.
2. **Соло-разработчик.** Каждый Stage самодостаточен и завершаем за одну сессию: новые файлы + тесты +
   точка интеграции. В конце Stage — рабочий, протестированный инкремент.
3. **Данные — забота пользователя; функционал — наша.** Каждая вертикаль поставляется с:
   - **бесплатным data-адаптером** (где есть бесплатный источник), и
   - **BYO-data слотом** (parquet/CSV/коннектор премиум-данных).
   Мы продаём инструмент. Сможет ли пользователь раскрыть его на 100% — зависит от его данных. Главное —
   функционал существует и корректен.
4. **Честность про free-data.** Где бесплатные данные ограничены (нет настоящего PIT-фундаментала,
   survivorship-bias в бесплатных юниверсах) — это **явно помечается в коде/UI** + есть BYO-слот для
   корректных данных. Движок строится с правильной PIT-архитектурой независимо от качества free-источника.
5. **Слои.** Соблюдаем `core_ → impl_ → service_ → strategies → script_`. Зависимости только слева направо.
6. **CCEA.** Целевые веса = Intents. Cloud отдаёт target-веса, ордера делает Agent. Не нарушаем.
7. **Каждый Stage заканчивается:** новые `pytest tests/test_xs_*.py` зелёные + краткая запись в этом файле
   «Stage N: DONE» с перечнем добавленного.

---

## 1. Как это встраивается в текущий MVP (контракт интеграции)

| Текущий компонент MVP | Как используется в новом слое | Меняем? |
|---|---|---|
| `core_contracts.py`, слой `core_` | Добавляем `core_portfolio.py` (новые Protocols) | +файл |
| `data_loader_multi_asset.py` | Источник сырья для Panel | reuse |
| `services/survivorship.py` (`UniverseSnapshot`) | PIT-юниверс | reuse |
| `services/portfolio_constraints.py` (`FactorTiltValidator`, `RebalanceEngine`, `PortfolioConstraintManager`) | ограничения + доводка target | reuse |
| `services/corporate_actions.py` | total-return / split-adjust | reuse |
| `execution_providers*.py`, `lob/market_impact.py` | tcost-модель для оптимизатора + capacity | reuse |
| `distributional_ppo.py`, `service_conformal.py` | RL → `RLAlphaSignal` (один сигнал) | +адаптер |
| `services/unified_futures_risk.py` (`PortfolioRiskManager`) | портфельные риск-гарды live | reuse/extend |
| `impl_greeks_vectorized.py`, `impl_pricing.py` | options-портфель (греческие) | reuse |
| `impl_cme_rollover.py` | continuous futures серии | reuse |
| `adapters/*` (binance, yahoo, oanda, alpaca, polygon, deribit, ib) | free/BYO data-адаптеры | reuse |
| `app.py` (FastAPI, ~130 endpoints) | +endpoints `/api/xs/*` | +роуты |
| `index.html` Lite «ИИ-пайплайн» | +ветка «Cross-sectional портфель» | +ветка |
| `index.html` Pro (`pro-research/model-lab/backtest/risk`) | +под-экраны Signal Lab / Optimizer / Trust / Attribution | +экраны |
| Compliance (MiFID/DORA/AI-Act) | attribution-report как evidence; model-card μ-модели | reuse |

**Новый режим включается флагом** `mode: cross_sectional` в YAML и переключателем в UI. По умолчанию — текущее
поведение MVP. Никакой регрессии.

---

## 2. Карта этапов и зависимостей

```
PART A — ДВИЖОК (asset-agnostic, строим один раз для всех 5)
  A1  Scaffolding + core contracts + Panel
  A2  Data sources (free adapters + BYO) + total-return/corporate actions
  A3  Universe (PIT) layer
  A4  Signal framework + cross-sectional transforms + IC-диагностика
  A5  Risk model (Σ: factor model + Ledoit-Wolf)
  A6  Alpha model (μ: combination) + RL-as-signal
  A7  Portfolio optimizer (w*: MVO/max-Sharpe/risk-parity/min-var)
  A8  Cross-sectional backtest engine
  A9  Backtest validation / Trust Report (Deflated Sharpe, PBO, purged CV, capacity)
  A10 Attribution engine
  A11 Live execution (weights→Intents→Agent) + portfolio risk guards
  A12 Backend API (/api/xs/*) + script entrypoints + config schema
  A13 UI (Lite ветка + Pro экраны)

PART B — ВЕРТИКАЛИ (плагины на движок; каждая = сигналы+факторы+юниверс+free-data+config+UI-пресет)
  B1  Crypto      (зависит от A1–A9; полная — после A13)
  B2  Equity      (-//-)
  B3  Futures     (+ impl_continuous_futures)
  B4  Forex
  B5  Options     (ОТДЕЛЬНЫЙ greeks-оптимизатор)

PART C — КРОСС-ASSET (опционально, последним)
  C1  Unified cross-asset portfolio (один оптимизатор по всем классам, нормализация валют, vol-target)
```

Зависимости: A1→A2→A3→A4→A5,A6→A7→A8→A9→A10→A11→A12→A13. Вертикали B* требуют A1–A9 для бэктеста и A11–A13
для live+UI. C1 требует все B (кроме опционально options).

---

# PART A — ДВИЖОК

### Stage A1 — Scaffolding, core-контракты, Panel API

- **Цель:** базовые типы и Panel; включить `mode: cross_sectional` как no-op ветку.
- **Новые файлы:** `core_portfolio.py` (Protocols: `UniverseProvider`, `Signal`, `AlphaModel`, `RiskModel`,
  `PortfolioConstructor`, `CrossSectionalStrategy`; типы `Panel`, `TargetWeights`, `RebalanceEvent`);
  `impl_panel.py` (`PanelBuilder`: сборка MultiIndex `(ts_ms, symbol)` из `data_loader_multi_asset`,
  выравнивание по календарю, as-of join хелперы); `xs/__init__.py` (пакет нового слоя, опционально).
- **Интеграция:** `core_config.py` — добавить `mode` поле (default = текущее). Никакой логики ещё не включаем.
- **Free/BYO:** Panel принимает любой DataFrame-источник (free или BYO) — агностично.
- **Тесты:** `tests/test_xs_panel.py` — сборка panel, выравнивание, отсутствие NaN-дыр, MultiIndex корректен.
- **Acceptance:** Panel строится из существующих parquet; контракты импортируются; текущие тесты зелёные.

### Stage A2 — Data sources (free + BYO) + total-return

- **Цель:** единый слой источников котировок/доходностей с бесплатными коннекторами и BYO.
- **Новые файлы:** `impl_data_sources.py` (`PriceSource` Protocol; `FreePriceSource` поверх `adapters/yahoo`,
  `adapters/binance` public; `ParquetPriceSource` = BYO; `FundamentalsSource` Protocol + `FreeFundamentals`
  (yfinance) + `ParquetFundamentals` BYO с PIT-лагом publish_ts); total-return/split-adjust через
  `services/corporate_actions.py`).
- **Free/BYO:** явный флаг `pit_quality: {true|approx|none}` в метаданных источника; UI/лог это показывает.
- **Тесты:** `tests/test_xs_data_sources.py` — free-адаптер отдаёт бары; BYO parquet грузится; total-return
  пересчёт корректен; PIT-лаг применяется (fundamental доступен только после publish_ts).
- **Acceptance:** одинаковый интерфейс для free и BYO; leakage-guard на фундаментал работает.

### Stage A3 — Universe (PIT) layer

- **Цель:** survivorship-free состав юниверса на дату.
- **Новые файлы:** `impl_universe.py` (`UniverseProvider` поверх `services/survivorship.UniverseSnapshot` +
  `DelistingTracker`; `StaticUniverse` (список из конфига) + `IndexMembershipUniverse` (история состава, BYO);
  ликвидностный фильтр по ADV).
- **Free/BYO:** free = статический/текущий список (помечается как survivorship-biased); BYO = история membership.
- **Тесты:** `tests/test_xs_universe.py` — `constituents(asof)` отдаёт PIT-состав; делистнутые тогда-активные
  тикеры присутствуют в прошлом; ликвидностный фильтр режет неликвид.
- **Acceptance:** бэктест сможет итерироваться по историческому составу; honest-флаг survivorship проставлен.

### Stage A4 — Signal framework + cross-sectional transforms + IC

- **Цель:** каркас сигналов и поперечные преобразования (без asset-конкретики — она в Part B).
- **Новые файлы:** `impl_cross_sectional.py` (`rank`, `zscore`, `winsorize`, `neutralize(by=[sector,beta,size])`
  через регрессию, `decay`/half-life); `service_signals.py` (`SignalLibrary` реестр; базовый `Signal` ABC;
  пайплайн `compute → transform → store`); `impl_signal_diagnostics.py` (IC, IC-decay, quantile spread,
  turnover, авто-корреляция сигналов).
- **Интеграция:** общий с `transformers.py` (переиспользовать индикаторные расчёты, но в panel-режиме).
- **Тесты:** `tests/test_xs_signals.py` — transforms математически верны (zscore mean≈0/std≈1, neutralize даёт
  нулевую корреляцию с фактором); IC считается; пустой/одно-имя юниверс не падает.
- **Acceptance:** можно зарегистрировать сигнал-заглушку и получить нормализованный + IC-отчёт.

### Stage A5 — Risk model (Σ)

- **Цель:** факторная ковариация активов.
- **Новые файлы:** `service_risk_model.py` (`FactorRiskModel`: exposures B через
  `portfolio_constraints.FactorTiltValidator.set_factor_loadings`, factor cov F (EWMA/Ledoit-Wolf),
  specific risk D, `cov() = B F Bᵀ + diag(D)`; `StatRiskModel`: Ledoit-Wolf напрямую как baseline;
  PCA-статфакторы); гарантия PSD (shrinkage).
- **Тесты:** `tests/test_xs_risk_model.py` — Σ симметрична и PSD; экспозиции согласованы; на синтетике факторов
  восстанавливается известная ковариация.
- **Acceptance:** для N имён отдаётся стабильная Σ + экспозиции для оптимизатора и attribution.

### Stage A6 — Alpha model (μ) + RL-as-signal

- **Цель:** комбинация сигналов в ожидаемую доходность; RL как сигнал.
- **Новые файлы:** `service_alpha.py` (`AlphaModel`: `EqualWeightAlpha`, `ICWeightedAlpha`, `RidgeAlpha`,
  опц. `GBMAlpha`; rolling-fit, purge-aware); `impl_rl_signal.py` (`RLAlphaSignal` — обёртка над
  `distributional_ppo` выходом: value/quantiles → ожидаемая полезность; вес через `service_conformal`).
- **Интеграция:** не трогаем обучение RL; только читаем выход через адаптер (RL-режим MVP сохранён).
- **Тесты:** `tests/test_xs_alpha.py` — IC-weighted повышает вес сигналам с высоким IC; ridge регуляризует;
  RLAlphaSignal отдаёт Series по символам с conformal-весом.
- **Acceptance:** из набора сигналов получаем μ; RL участвует измеримо (ненулевой IC-вклад).

### Stage A7 — Portfolio optimizer (w*)

- **Цель:** ядро оптимизации.
- **Новые файлы:** `service_optimizer.py` (`PortfolioOptimizer`: режимы `mean_variance`, `max_sharpe`,
  `risk_parity`, `min_variance`, `equal_weight`, `black_litterman`; solver = cvxpy/osqp; **аналитический
  fallback** без неравенств для окружений без солвера; ограничения из `PortfolioConstraintManager`:
  gross/net, position/sector/factor-tilt, turnover; tcost через `execution_providers`).
- **Интеграция:** результат доводится `RebalanceEngine.rebalance_to_target` + `_enforce_limits`.
- **Тесты:** `tests/test_xs_optimizer.py` — аналитические кейсы (equal-weight при μ=const,Σ=I; tangency);
  ограничения не нарушаются; turnover-cap соблюдён; fallback работает без cvxpy.
- **Acceptance:** μ+Σ+constraints → исполнимый вектор w*.

### Stage A8 — Cross-sectional backtest engine

- **Цель:** бэктест поверх Panel (не per-instrument env).
- **Новые файлы:** `service_xs_backtest.py` (цикл по датам ребаланса: universe→signals→μ→Σ→w*→trade-list→
  costs→apply returns→equity/exposures/turnover; журнал по датам); `core_xs_results.py` (структуры
  результатов: equity curve, weights history, exposures, trades).
- **Интеграция:** walk-forward из `make_walkforward_splits.py`; косты из `execution_providers`.
- **Тесты:** `tests/test_xs_backtest.py` — end-to-end на синтетике; нет look-ahead (leakage-probe);
  метрики (Sharpe/maxDD/turnover) считаются; результат детерминирован при seed.
- **Acceptance:** прогон cross-sectional long-short на синтетическом юниверсе даёт корректную equity-кривую.

### Stage A9 — Backtest validation / Trust Report

- **Цель:** анти-оверфит и доверие (P0-ценность).
- **Новые файлы:** `service_backtest_validation.py` (Deflated Sharpe Ratio, PBO через combinatorial purged CV,
  purged & embargoed K-fold, multiple-testing haircut, IS/OOS деградация); `impl_capacity.py` (capacity-кривая
  через impact-модели `lob/market_impact.py`: AUM vs Sharpe-деградация).
- **Тесты:** `tests/test_xs_validation.py` — DSR/PBO на известных распределениях; purge убирает граничный лик;
  capacity монотонно ухудшается с ростом размера.
- **Acceptance:** по любому бэктесту генерируется «Trust Report» (JSON) с DSR, PBO, capacity.

### Stage A10 — Attribution engine

- **Цель:** разложение P&L и риска.
- **Новые файлы:** `service_attribution.py` (factor P&L attribution = factor_return×exposure + specific;
  сигнальная attribution; Brinson allocation/selection; tear-sheet JSON + экспорт-структура для PDF).
- **Интеграция:** evidence-экспорт для compliance (MiFID/AI-Act).
- **Тесты:** `tests/test_xs_attribution.py` — сумма факторных вкладов + specific = полный P&L (с точностью eps).
- **Acceptance:** для прогона отдаётся attribution-отчёт; tie-out с total P&L.

### Stage A11 — Live execution (weights→Intents→Agent) + portfolio risk guards

- **Цель:** ребаланс в live через CCEA.
- **Новые файлы:** `service_xs_live.py` (на дату ребаланса: target_weights → ноционалы → набор Intents →
  передача Agent; reconciliation через `services/position_sync`); `service_xs_portfolio_risk.py`
  (portfolio-level guards поверх `unified_futures_risk.PortfolioRiskManager`: gross/net, factor-exposure,
  концентрация; pre-trade проверка trade-list).
- **Интеграция:** CCEA — Cloud отдаёт веса (Intent), Agent делает ордера; не нарушаем границу.
- **Тесты:** `tests/test_xs_live.py` — веса→Intents корректно; pre-trade guard блокирует нарушение лимитов;
  reconciliation ловит расхождение (моки адаптеров).
- **Acceptance:** симулированный ребаланс портфеля проходит через Intent-слой и риск-гарды.

### Stage A12 — Backend API + script entrypoints + config schema

- **Цель:** сетевой и CLI доступ к движку.
- **Новые файлы:** `script_xs_backtest.py`, `script_xs_live.py` (CLI entrypoints, слой `script_`);
  `configs/config_xs_template.yaml` (схема из дизайн-дока).
- **Изменяем (аддитивно):** `app.py` — endpoints `/api/xs/signals`, `/api/xs/risk_model`, `/api/xs/optimize`,
  `/api/xs/backtest`, `/api/xs/trust_report`, `/api/xs/attribution`, `/api/xs/live/rebalance`,
  `/api/xs/universe`, `/api/xs/config` (валидация Pydantic).
- **Тесты:** `tests/test_xs_api.py` — endpoints отвечают; конфиг валидируется; CLI бэктест отрабатывает на
  free-данных (smoke).
- **Acceptance:** `python script_xs_backtest.py --config configs/config_xs_template.yaml` даёт Trust Report.

### Stage A13 — UI (Lite ветка + Pro экраны)

- **Цель:** вывести функционал в интерфейс, аддитивно к текущему.
- **Изменяем (аддитивно):** `index.html` —
  - Lite «ИИ-пайплайн»: новая ветка **«Cross-sectional портфель»** (выбор юниверса, чекбоксы сигналов,
    режим оптимизатора, лимиты gross/net/turnover, кнопка → бэктест + Trust Report).
  - Pro: `pro-research`→**Signal Lab** (IC/decay/quantile); `pro-model-lab`→**Alpha Model** + **Risk Model**
    (Σ heatmap, факторы) + новый **Portfolio Constructor** (efficient frontier, constraints);
    `pro-backtest`→**Trust Report** (DSR/PBO/capacity); `pro-risk`→factor exposures; новый **Attribution** экран.
  - Telemetry: добавить gross/net/turnover/factor-exposures.
- **Free/BYO:** в UI — переключатель источника данных (free / загрузить свой) + honest-бейдж PIT-качества.
- **Тесты:** smoke на новые endpoints из UI (ручной чек-лист в этом файле).
- **Acceptance:** из Lite одной кнопкой прогоняется cross-sectional бэктест; Pro показывает Signal Lab/Trust/Attribution.

---

# PART B — ВЕРТИКАЛИ (плагины на готовый движок)

> Каждая вертикаль = `SignalLibrary` под класс + факторы `RiskModel` + `UniverseProvider` + free-data адаптер +
> config-пресет + UI-пресет + тесты. Движок (A) не меняется — только плагины.

### Stage B1 — Crypto (рекомендуемый первый)

- **Новые файлы:** `signals/crypto_signals.py` (momentum 30/90d, short-term reversal, **funding-carry**,
  **basis** spot-perp, size/mcap, опц. on-chain-заглушка с BYO-слотом); `risk/crypto_factors.py`
  (BTC-beta, sector L1/DeFi/..., size).
- **Free data:** `adapters/binance` public (бары, funding) — бесплатно и без ключей для истории.
- **Config:** `configs/config_xs_crypto.yaml` (юниверс топ-N по ADV, weekly/daily ребаланс, market-neutral).
- **UI:** пресет в Lite + крипто-факторы в Pro Risk Model.
- **Тесты:** `tests/test_xs_crypto.py` — сигналы считаются на free-данных; бэктест long-short проходит.
- **Acceptance:** полный cross-sectional крипто-контур на бесплатных данных, end-to-end + Trust Report.

### Stage B2 — Equity (US)

- **Новые файлы:** `signals/equity_signals.py` (momentum 12-1, value E/P,B/P,FCF-yield, quality ROE/accruals,
  low-vol, size); `risk/equity_factors.py` (market, size, value, momentum, sector — Barra-lite).
- **Free data:** `adapters/yahoo`/yfinance (цены + базовый фундаментал). **Honest-note:** free-фундаментал НЕ
  настоящий PIT и юниверс survivorship-biased → бейдж `pit_quality: approx`; BYO-слот для Shardar/Compustat и т.п.
- **Config:** `configs/config_xs_equity.yaml` (SP500/NASDAQ100 список, weekly, beta+sector-neutral).
- **UI:** пресет; honest-баннер про качество free-данных.
- **Тесты:** `tests/test_xs_equity.py` — сигналы/факторы на free-данных; PIT-лаг фундаментала применён.
- **Acceptance:** equity market-neutral контур работает; ограничения free-данных явно задокументированы в UI.

### Stage B3 — Futures (CME / continuous)

- **Новые файлы:** `impl_continuous_futures.py` (back-adjusted непрерывные серии поверх `impl_cme_rollover`);
  `signals/futures_signals.py` (trend 50/100/200, carry/roll-yield, value, vol-target); `risk/futures_factors.py`
  (asset-class факторы: equity-index/rates/energy/metals/ag/FX).
- **Free data:** `adapters/yahoo`/stooq continuous-прокси (бесплатные continuous-серии) + BYO для точных.
- **Config:** `configs/config_xs_futures.yaml` (диверсифицированный CTA-портфель, risk-parity, vol-target).
- **UI:** пресет; continuous-contract индикатор.
- **Тесты:** `tests/test_xs_futures.py` — back-adjust корректен (нет скачка на roll); CTA-бэктест проходит.
- **Acceptance:** диверсифицированный фьючерсный risk-parity контур end-to-end.

### Stage B4 — Forex (G10/EM)

- **Новые файлы:** `signals/forex_signals.py` (carry, momentum/trend, value/PPP, terms-of-trade);
  `risk/forex_factors.py` (USD-beta, carry, value).
- **Free data:** `adapters/oanda` practice / бесплатные FX-серии + BYO.
- **Config:** `configs/config_xs_forex.yaml` (G10 пары, carry+momentum, USD-neutral опция).
- **UI:** пресет.
- **Тесты:** `tests/test_xs_forex.py` — carry/value сигналы; малый юниверс не ломает оптимизатор.
- **Acceptance:** FX cross-sectional carry/momentum контур end-to-end.

### Stage B5 — Options (ОТДЕЛЬНЫЙ greeks-оптимизатор)

- **Цель:** опционы — другая машинерия: портфель экспозиций по греческим, не directional веса.
- **Новые файлы:** `service_options_portfolio.py` (`OptionsPortfolioConstructor`: vol-risk-premium,
  skew, dispersion, term-structure сигналы → структуры; **greeks-neutral** ограничения delta/vega/gamma;
  оптимизация в пространстве греческих); `signals/options_signals.py`; `risk/options_factors.py` (vol-факторы).
- **Интеграция:** reuse `impl_pricing.py`, `impl_greeks_vectorized.py`, `service_conformal`.
- **Free data:** ограниченно — yfinance option chains (EOD) + BYO (Deribit free для крипто-опционов через
  `adapters/deribit`).
- **Config:** `configs/config_xs_options.yaml`.
- **UI:** отдельный под-экран (греческие, vol-surface), помеченный как options-режим.
- **Тесты:** `tests/test_xs_options.py` — greeks-neutral соблюдается; vol-premium сигнал считается.
- **Acceptance:** vol-harvesting/dispersion портфель с контролем греческих, end-to-end на доступных данных.

---

# PART C — КРОСС-ASSET (последним, опционально)

### Stage C1 — Unified cross-asset portfolio

- **Цель:** один оптимизатор по equity+futures+FX+crypto одновременно (true multi-asset).
- **Новые файлы:** `service_cross_asset.py` (унифицированный риск-агрегатор, нормализация валют в базовую,
  общий vol-target, кросс-asset ковариация; верхний risk-parity между классами).
- **Интеграция:** поверх всех B-вертикалей и `unified_futures_risk`.
- **Тесты:** `tests/test_xs_cross_asset.py` — агрегированная Σ PSD; валютная нормализация корректна.
- **Acceptance:** единый портфель из ≥2 классов с общим риск-таргетом.

---

## 3. Прогресс (агент отмечает по мере выполнения)

- [x] A1  — core contracts + Panel ✅ DONE (15/15 tests). Добавлено: `core_portfolio.py`
      (Protocols UniverseProvider/Signal/AlphaModel/RiskModel/PortfolioConstructor/CrossSectionalStrategy,
      типы Panel/TargetWeights/CovMatrix, RebalanceEvent, MODE_* константы, validate_panel/cross_section
      хелперы); `impl_panel.py` (PanelBuilder.from_frames/from_long/from_data_loader, normalize_ts_ms
      [sec/ms/us/ns/datetime], union/intersection align + as-of ffill, **asof_join PIT с publish_lag**,
      add_forward_returns); `core_config.CommonRunConfig.mode` (аддитивно, default=single_instrument);
      `tests/test_xs_panel.py`. Регрессий нет (106 config-тестов зелёные; единственный фейл —
      pre-existing Windows-path баг в `test_microvm_isolation`, не связан).
- [x] A2  — data sources (free + BYO) ✅ DONE (11/11 tests). Добавлено: `impl_data_sources.py` —
      `DataSourceMeta` (флаг **pit_quality** true/approx/none для UI/логов); `PriceSource`/`FundamentalsSource`
      Protocols; `AdapterPriceSource` (free, поверх адаптеров через registry, DI-адаптер, мягкая деградация),
      `ParquetPriceSource` (BYO); `ParquetFundamentals` (BYO, PIT-true), `FreeFundamentals` (yfinance snapshot,
      PIT-none, honest-warning); `bars_to_frame`, **`total_return_index`/`add_total_return`** (детерминированный
      total-return: сплиты+дивиденды, без внешних данных), `apply_corporate_actions_if_available` (мягкое
      делегирование в services/corporate_actions), `build_price_panel` (интеграция с A1). Только новые файлы →
      нулевой риск регрессии. PIT leakage-guard подтверждён через asof_join.
- [x] A3  — universe (PIT) ✅ DONE (7/7 tests; cumulative A1–A3 = 33/33). Добавлено: `impl_universe.py` —
      `ms_to_date`/`date_to_ms` (мост ms↔date); `StaticUniverse` (free, honest **survivorship_biased=True**);
      `IndexMembershipUniverse` (BYO, survivorship-free, поверх `survivorship.UniverseSnapshot`+`DelistingTracker`,
      `from_baseline(...)` конструктор, PIT-реконструкция состава на дату, делистнутые-позже тикеры присутствуют
      в прошлом); `ADVLiquidityFilter` (trailing dollar-volume ADV из Panel, сохраняет honest-флаг базы). Все —
      реализации `core_portfolio.UniverseProvider`. Только новые файлы (+reuse services/survivorship) → нулевой
      риск регрессии.
- [x] A4  — signals + transforms + IC ✅ DONE (13/13; cumulative A1–A4 = 46/46). Добавлено:
      `impl_cross_sectional.py` (`rank`/`zscore`/`winsorize`/`neutralize` [OLS-остаток, dummies для
      категориальных]/`decay`; `apply_cs`, `run_pipeline` с факторной панелью); `service_signals.py`
      (`BaseSignal` ABC + `compute_panel`/`compute`; примитивы `ColumnSignal`/`MomentumSignal`/`FunctionSignal`;
      `SignalSpec`, `SignalLibrary` → панель сигналов с трансформами+нейтрализацией); `impl_signal_diagnostics.py`
      (`information_coefficient` [Spearman без scipy, IC/IR/hit_rate], `ic_decay`, `quantile_spread`, `turnover`,
      `signal_autocorr`, `signal_report`). Устойчиво к пустому/одно-имённому юниверсу. Только новые файлы →
      нулевой риск регрессии. Asset-конкретные библиотеки сигналов — в Part B.
- [x] A5  — risk model Σ ✅ DONE (8/8; cumulative A1–A5 = 54/54). Добавлено: `service_risk_model.py` —
      `FactorRiskModel` (BARRA-style: B через интеграцию с `FactorTiltValidator`, факторные доходности
      кросс-секционной регрессией, F=cov [sample/Ledoit-Wolf/EWMA], D=specific risk, **Σ=BFBᵀ+diag(D)**);
      `StatRiskModel` (Ledoit-Wolf shrinkage + PCA-статфакторы, baseline); `ledoit_wolf_identity` (LW-2004,
      closed-form δ), `nearest_psd` (eigenvalue-clipping), `to_wide_returns` (panel/Series/wide). Σ гарантированно
      симметрична и PSD. Подтверждено: на синтетике без шума точно восстанавливает известную ковариацию (factor
      returns + Σ). Только новые файлы → нулевой риск регрессии.
- [x] A6  — alpha μ + RL-as-signal ✅ DONE (8/8; cumulative A1–A6 = 62/62). Добавлено: `service_alpha.py`
      (`BaseAlphaModel` + `predict_panel`; `EqualWeightAlpha`, `ICWeightedAlpha` [вес ∝ знаковый IC],
      `RidgeAlpha` [L2-регуляризация], `GBMAlpha` [опц., lazy sklearn]); `impl_rl_signal.py` — **`RLAlphaSignal`**
      (RL-выход → один сигнал среди многих; utility×conformal-confidence; `expected_utility_from_quantiles`
      [mean/CVaR], `conformal_confidence_from_widths`; DI-дружелюбно, обучение RL не трогаем). Подтверждено:
      IC-weighted повышает вес предсказательному сигналу, ridge регуляризует, RL имеет измеримый IC в
      SignalLibrary. Только новые файлы → нулевой риск регрессии.
- [x] A7  — optimizer w* ✅ DONE (12/12; cumulative A1–A7 = 74/74). Добавлено: `service_optimizer.py` —
      `PortfolioOptimizer` (`core_portfolio.PortfolioConstructor`): режимы equal_weight / min_variance /
      mean_variance / max_sharpe(tangency) / risk_parity(ERC, sqrt-damping) / black_litterman; **аналитическое
      решение + итеративная проекция** на жёсткие ограничения (gross/net/box/turnover/long-only) — работает
      **без cvxpy** (cvxpy/osqp отсутствуют в окружении; guarded cvxpy-ветка для будущего); `OptimizerConstraints`
      (мост к portfolio_constraints), `black_litterman_mu`, `_solve_psd`/`_risk_parity`. Подтверждено: точные
      аналитические кейсы (inverse-var, tangency∝Σ⁻¹μ, ERC∝1/σ, market-neutral) + все hard-лимиты не нарушаются.
      Только новые файлы → нулевой риск регрессии.
- [x] A8  — cross-sectional backtest ✅ DONE (5/5; cumulative A1–A8 = 79/79). Добавлено: `core_xs_results.py`
      (`XSBacktestResult`, `compute_metrics`: Sharpe/maxDD/total-return/turnover/hit-rate); `service_xs_backtest.py`
      — `CrossSectionalBacktest` (цикл по ребалансам: universe(t)→signals→**alpha обучен строго на прошлом,
      target реализован до t (purge/embargo)**→Σ на трейлинг-окне≤t→optimizer w*→costs→realized P&L→equity;
      walk-forward expanding window; линейные косты bps + опц. cost_fn для execution_providers). Подтверждено:
      end-to-end oracle-сигнал → растущая equity (market-neutral net≈0), детерминизм, и **rigorous
      leakage-probe** (панели, совпадающие до t*, дают идентичные веса на t*). Только новые файлы → нулевой риск
      регрессии.
- [x] A9  — validation / Trust Report ✅ DONE (10/10; cumulative A1–A9 = 89/89). **Первая «продаваемая» веха.**
      Добавлено: `service_backtest_validation.py` — **Deflated Sharpe Ratio** + Probabilistic Sharpe (Bailey & LdP,
      skew/kurt/N-trials), **PBO** через combinatorial symmetric CV (Bailey et al. 2015), **purged & embargoed
      K-fold** (LdP), `sharpe_haircut` (multiple-testing), `is_oos_degradation`, **`trust_report`** (JSON:
      DSR/PBO/capacity/verdict); `impl_capacity.py` — `capacity_curve` (AUM→Sharpe через √participation impact,
      Almgren-Chriss; опц. lob.market_impact), `capacity_from_result`. Подтверждено: DSR убывает с числом
      испытаний, PBO≈0 для реального скилла и ≈0.5 для шума, purge исключает граничный лик, capacity монотонно
      деградирует. Только новые файлы → нулевой риск регрессии.
- [x] A10 — attribution ✅ DONE (5/5; cumulative A1–A10 = 94/94). Добавлено: `service_attribution.py` —
      **`factor_attribution`** (точное разложение wᵀr = Σ(wᵀB)·f + wᵀu = факторные вклады + specific, tie-out
      к полному P&L), `signal_attribution` (автономный long-short P&L по сигналу + Sharpe), `brinson_attribution`
      (allocation/selection/interaction, tie-out к active return), `tear_sheet` (JSON для LP/compliance evidence).
      Подтверждено: tie-out факторов+specific = total с точностью 1e-9, Brinson tie-out, JSON-сериализация.
      Только новые файлы → нулевой риск регрессии.
- [x] A11 — live execution + portfolio risk ✅ DONE (7/7; cumulative A1–A11 = 101/101). Добавлено:
      `service_xs_portfolio_risk.py` (`PortfolioRiskGuard` + `PortfolioRiskLimits`: gross/net, концентрация по
      имени/сектору, факторные экспозиции Bᵀw, turnover; pre-trade `check` → `RiskDecision`; опц. unified manager);
      `service_xs_live.py` — `CrossSectionalLiveRunner` (w* → **Intents** [target_weight+target_notional, БЕЗ
      side/qty/price], idempotency-ключ; `rebalance`: reconcile→guard→build→send Agent; `reconcile` ловит дрейф).
      **CCEA-граница строго соблюдена**: Cloud отдаёт только Intents, ордера делает Agent (guard `_FORBIDDEN_FIELDS`).
      Подтверждено: notional=w×equity, нет order-полей, guard блокирует нарушения, reconciliation ловит
      расхождение, blocked-ребаланс ничего не отправляет. Только новые файлы → нулевой риск регрессии.
- [x] A12 — API + scripts + config ✅ DONE (9/9; cumulative A1–A12 = 110/110). Добавлено: `service_xs_pipeline.py`
      (Pydantic `XSConfig` + builders + `run_backtest`/`latest_target_weights` + `load_panel` [synthetic/parquet/free]);
      `xs_api.py` (изолированный APIRouter `/api/xs/*`: config/universe/optimize/risk_model/trust_report/attribution/
      backtest/signals/live·rebalance); `script_xs_backtest.py` + `script_xs_live.py` (CLI); `configs/config_xs_template.yaml`
      (synthetic — работает без данных). Аддитивно в `app.py`: guarded `register_xs_routes(api)` (не ломает MVP).
      Также добавлен `alpha_refit_every` в движок A8 (perf+реализм: бэктест шаблона 91s→10s, default=1 не меняет A8).
      Acceptance: `python script_xs_backtest.py --config configs/config_xs_template.yaml` → Trust Report. Роутер
      протестирован изолированно (TestClient), app.py парсится.
- [x] A13 — UI ✅ DONE (cumulative A1–A13 = 112/112). **PART A (движок) ЗАВЕРШЁН.** Интеграция **тесно в Pro-режим**
      MVP (не отдельно): 5 native под-вкладок «✦ Cross-Sectional» в `index.html`, каждая в своём профильном модуле —
      **Model Lab** (каноническая конфигурация стратегии: data/PIT-бейдж + сигналы + alpha + Σ risk model + Portfolio
      Constructor → целевые веса), **Research** (Signal Lab: IC/IR/spread/turnover), **Backtest** (Trust Report:
      DSR/PSR/PBO/capacity/verdict), **Risk** (pre-trade portfolio guard: gross/net/violations), **OMS** (Rebalance →
      Intents, CCEA). Все вкладки читают общий `xs-*` конфиг из Model Lab (через DOM). Бэкенд: `+/api/xs/weights`
      (latest target weights). Каждая правка — 3 точки (массив switcher + nav-кнопка + панель), всё div-balanced
      (проверено). Standalone-консоль удалена (`/api/xs/ui` убран). Тесты: endpoints + Pro-интеграция в index.html.
- [x] B1  — crypto vertical ✅ DONE (7/7; cumulative 119/119). **Первый полный вертикальный эталон.** Добавлено:
      `signals/crypto_signals.py` (CryptoMomentum, ShortTermReversal, **FundingCarry**, **Basis** spot-perp, **Size**
      mcap, OnChain — все graceful к отсутствующим колонкам = BYO-слот); `xs_risk/crypto_factors.py` (**btc_beta**,
      size, sector one-hot → exposures B + `build_crypto_risk_model`); pipeline: крипто-kinds в `build_signal_library`
      + `risk.type='crypto_factor'` (FactorRiskModel из панели) + sectors/mcaps/btc_symbol в XSConfig;
      `configs/config_xs_crypto.yaml` (крипто-сигналы + crypto-factor Σ + market-neutral + weekly); UI: крипто-сигналы
      (funding/basis/size, BYO) добавлены в Cross-Sectional Lab. Acceptance: end-to-end на synthetic + free-binance
      путь готов. **GOTCHA:** пакет назван `xs_risk/` (НЕ `risk/` — коллизия с существующим top-level `risk.py`
      [RiskManager/RiskConfig]). Для будущих вертикалей `risk/...` из роадмапа = `xs_risk/...`.
- [x] **B2  — equity vertical (US): DONE** — `signals/equity_signals.py` (EquityMomentum 12-1, EarningsYield E/P,
      BookToPrice B/P, FCFYield, ReturnOnEquity ROE, Accruals, LowVolatility, EquitySize — все graceful к
      отсутствующим колонкам = BYO·PIT-слот; yield-сигналы берут готовую ep/bp/fcf_yield колонку ИЛИ
      фундаментал/цена); `xs_risk/equity_factors.py` (**market_beta** к индексу/равновзвеш.прокси, size, value,
      momentum, sector one-hot → exposures B [Barra-lite] + `build_equity_risk_model`); pipeline: equity-kinds в
      `build_signal_library` (`_build_equity`) + `risk.type='equity_factor'` (`build_equity_factor_risk`,
      диспетчер `_build_run_risk_model`) + values/market_symbol/momentum_lookback + equity-колонки в SignalCfg;
      `configs/config_xs_equity.yaml` (ценовые сигналы по умолчанию [честны], фундаментальные закомментированы =
      BYO·PIT; beta+sector факторная Σ; market-neutral; weekly). **Honest-data:** free-фундаментал (yfinance) =
      снимок, НЕ PIT; free-списки SP500/NDX survivorship-biased → `pit_quality=approx` + honest-баннер в UI.
      UI: селектор «Класс актива» (crypto|equity) + equity-чекбоксы (momentum/low-vol честны; E/P,B/P,ROE,size =
      BYO·PIT) + honest-баннер + опция risk «Factor (Barra-lite)» (→ crypto_factor/equity_factor по классу);
      `xsAssetChange()` тогглит группы/символы, `xsCfg()` стал asset-class-aware (periods 252/365, cost 3/5 bps).
      Factor-Σ деградирует gracefully без sectors/mcaps (только market_beta+momentum). Тесты: `tests/test_xs_equity.py`
      **11/11**; суммарно **130/130 xs** + регресс `test_risk_seasonality` 4/4 = **134 зелёных**, без регрессий.
      CLI smoke: equity preset → 52 ребаланса, market-neutral, Trust Report (verdict honest на synthetic-шуме).
- [x] **B3  — futures vertical (CME/continuous): DONE** — `impl_continuous_futures.py` (back-adjusted непрерывные
      серии: `back_adjust` [ratio сохраняет доходности / diff сохраняет уровни], `roll_events_from_overlap`,
      `stitch_contracts` [сегменты `[own_roll_ts, next_roll_ts)` БЕЗ перекрытия → нет двойной корректировки],
      `build_continuous_panel` BYO, `synthetic_continuous_frames` для demo; `ContinuousMeta` с pit_quality/method/
      n_rolls = continuous-contract индикатор); `signals/futures_signals.py` (Trend [TS-momentum 50/100/200, опц.
      vol-normalize], Carry [roll-yield из готовой колонки ИЛИ front/back контрактов = BYO], FuturesValue
      [long-horizon mean-reversion], RealizedVolInv [vol-target сайзинг]); `xs_risk/futures_factors.py` (market_beta
      к basket'у + vol-фактор + asset-class one-hot [equity_index/rates/energy/metals/fx/ag] → FactorRiskModel);
      pipeline `_build_futures` + `risk.type='futures_factor'` (`build_futures_factor_risk`, dispatcher) +
      futures-колонки/asset_classes в SignalCfg/XSConfig; `load_panel` для futures-synthetic роутится через
      `synthetic_continuous_frames` (демонстрирует continuous-модуль в контуре); `configs/config_xs_futures.yaml`
      (диверсиф. CTA: trend 50/100/200 vol-adj + value, carry закомм.=BYO; asset-class факторная Σ; mean-variance
      [μ(trend)→направление, Σ(vol/asset-class)→vol-target сайзинг]; market-neutral; weekly; cost 1.5bps).
      **Honest-data:** free continuous (yahoo ES=F/stooq) = back-adjusted прокси непрозрачным методом →
      `pit_quality=approx`; точные roll-accurate = BYO контракты+расписание. UI: класс «Futures (CTA)» в селекторе
      + futures-чекбоксы (trend/value честны; carry=BYO) + continuous-contract индикатор (sky-баннер) + risk
      «Factor» → futures_factor; `xsAssetChange()`/`xsCfg()` стали 3-классовыми (XS_DEFAULT_SYMS, periods 252,
      cost 1.5bps). Factor-Σ деградирует gracefully без asset_classes (market_beta+vol). Тесты:
      `tests/test_xs_futures.py` **13/13** (incl. back-adjust «нет скачка на роллах» ratio+diff); суммарно
      **143 xs** + регресс `test_risk_seasonality` 4/4 = **147 зелёных**, без регрессий. CLI smoke: futures preset →
      92 ребаланса, market-neutral, Trust Report (verdict honest на synthetic). **Баг пойман+исправлен:**
      перекрытие контрактов в `stitch_contracts` давало двойную ratio-корректировку → сегменты сделаны
      непересекающимися (граница = ключ gap'а).
- [x] **B4  — forex vertical (G10/EM): DONE** — `signals/forex_signals.py` (FXCarry [дифференциал ставок: готовая
      `rate_diff`/`carry` ИЛИ `rate_base−rate_quote` = BYO], FXMomentum [трендовый FX-курс], FXValue [готовая
      `ppp`/`reer_gap` колонка ИЛИ прокси −long-return mean-reversion], TermsOfTrade [BYO сырьевые валюты]); kind'ы
      с префиксом **`fx_`** чтобы НЕ коллидировать с futures (`carry`/`trend`/`value`); `xs_risk/forex_factors.py`
      (**usd_beta** к доллару [USD-индекс/равновзвеш.прокси] + carry + value — МИНИМУМ непрерывных факторов, без
      раздувания one-hot [мал юниверс]; opt bloc one-hot → FactorRiskModel; деградирует до usd_beta-only без
      carries/values); pipeline `_build_forex` + `risk.type='forex_factor'` (`build_forex_factor_risk`, dispatcher) +
      forex-колонки (rate_*/ppp/reer/terms) в SignalCfg + carries/blocs/usd_symbol в XSConfig;
      `configs/config_xs_forex.yaml` (G10 9 пар: momentum 90/180 + value-PPP прокси [честны], carry/ToT закомм.=BYO;
      usd_beta+carry+value Σ; mean-variance; **USD-neutral net=0**; weekly; cost 1.0bps). **Honest-data:** carry =
      ставки (BYO/free-loader), value/PPP = ppp/reer колонка (BYO, иначе mean-reversion прокси); free = OANDA
      practice. UI: класс «Forex (G10)» + forex-чекбоксы (momentum/value честны; carry/ToT=BYO) + emerald honest-note
      (малый юниверс/USD-neutral) + risk «Factor» → forex_factor; `xsAssetChange()`/`xsCfg()` стали 4-классовыми
      (XS_DEFAULT_SYMS+forex, vendor=oanda, cost 1.0bps, periods 252). **Малый юниверс не ломает оптимизатор**
      (проверено 3-4 пары + factor-Σ → веса без ошибок). Тесты: `tests/test_xs_forex.py` **9/9** (incl.
      small-universe acceptance); суммарно **152 xs** + регресс `test_risk_seasonality` 4/4 = **156 зелёных**, без
      регрессий. CLI smoke: forex preset → 68 ребалансов, USD-neutral, Trust Report.
- [x] **B5  — options vertical (ОТДЕЛЬНЫЙ greeks-оптимизатор): DONE** — опционы = портфель ЭКСПОЗИЦИЙ по
      греческим, НЕ directional веса. `service_options_portfolio.py` (**OptionsPortfolioConstructor**: альфа по ногам
      → **null-space проекция** `w = w0 − pinv(G)(G w0)` ⇒ точная greeks-нейтральность по выбранным грекам
      [delta/gamma/vega/theta/rho]; numpy-only, без cvxpy/scipy; gross-scale + клип по ногам + ре-проекция;
      остаточные греки в отчёте; `OptionLeg`/`GreeksNeutralConstraints`/`OptionsPortfolio`; `synthetic_option_book`
      сетка страйк×экспирация call+put; reuse `impl_greeks_vectorized.compute_all_greeks_batch`); `signals/
      options_signals.py` (VolRiskPremium [IV−RV], Skew [put−call IV], Dispersion, TermStructure [front−back IV] —
      все graceful=BYO; опционные данные платные); `xs_risk/options_factors.py` (vol-факторный риск-вью:
      `vol_level_beta` к VIX/DVOL + skew + term → FactorRiskModel, на изменениях IV); pipeline `_build_options` +
      options-колонки (iv/rv/vrp/skew/dispersion/term) в SignalCfg; **API `POST /api/xs/options/construct`**
      (demo:true → синтетический бук, или legs[]); `configs/config_xs_options.yaml` (greeks-neutral spec +
      vol_factors). UI: класс «Options (greeks)» + options-чекбоксы (VRP/skew/dispersion/term, BYO) + **отдельная
      greeks-neutral мини-панель** (нейтрализуй Δ/ν/Γ + gross/max-pos + кнопка «Build greeks-neutral book» →
      `xsOptionsConstruct()` хитит API, рендерит остаточные греки/edge) + fuchsia honest-note; xsAssetChange()/xsCfg()
      стали 5-классовыми. Тесты: `tests/test_xs_options.py` **11/11** (delta/vega/gamma-нейтральность residual≈0,
      too-few-legs→0, max_position диверсификация, сигналы+graceful, vol-факторы, pipeline, API endpoint); суммарно
      **163 xs** + регресс `test_risk_seasonality` 4/4 = **167 зелёных**, без регрессий. Smoke: 42-ногий бук,
      delta/gamma/vega residual=0.0, theta/rho ретейнятся (харвест), edge(α·w)=0.07.

**PART B (вертикали) ЗАВЕРШЕНА:** B1 crypto, B2 equity, B3 futures, B4 forex, B5 options — все 5 классов активов
покрыты (4 directional `μ→Σ→w*` + 1 greeks-space). Каждый: сигналы (graceful BYO) + факторная риск-модель +
пресет + UI-класс + тесты. Движок (Part A) + 5 вертикалей = полный про-парадигма «десятки сигналов → риск-модель →
портфель» на всех активах, на бесплатных/synthetic данных с honest pit_quality и BYO-слотами.

- [x] **C1  — unified cross-asset portfolio: DONE** — `service_cross_asset.py`: единый портфель поверх directional
      вертикалей (crypto+equity+futures+forex; options отдельны как greeks). Слои: (1) **валютная нормализация**
      `normalize_returns_to_base` [r_base=(1+r_local)(1+r_fx)−1]; (2) **joint Σ** `build_cross_asset_cov` [стек всех
      base-доходностей → StatRiskModel Ledoit-Wolf → симметрична+PSD; **honest позиционный fallback** если ts классов
      не пересекаются [разные эпохи/сессии free-данных]]; (3) **верхний класс-risk-parity** [a_c ∝ 1/vol_c] + внутри
      класса веса вертикали → combined w=Σ_c a_c·w_c; (4) **общий vol-target** [масштаб к целевой годовой vol].
      `AssetClassBlock`/`CrossAssetResult`; `block_from_xs_config` (прогон вертикали → веса+доходности) +
      `combine_from_configs` (high-level); **API `POST /api/xs/cross_asset`** (demo:true → 4 синт. класса).
      UI: карта «Unified Cross-Asset (C1)» в Lab (target vol + класс-веса risk_parity/equal + кнопка «Combine
      cross-asset» → `xsCrossAsset()` рендерит класс-аллокации/достигнутую vol/топ-позиции). Тесты:
      `tests/test_xs_cross_asset.py` **8/8** (валютная нормализация формула, joint Σ PSD+симметрия, vol-target
      достигнут, risk-parity favors low-vol класс, equal-weighting, block_from_config+combine, API endpoint);
      суммарно **171 xs** + регресс `test_risk_seasonality` 4/4 = **175 зелёных**, без регрессий. Smoke: 4-class
      unified → 16 имён, joint Σ PSD+symmetric, класс-alloc Σ=1, achieved vol=target=0.10 точно, gross≈1/net≈0.

---

## 🎉 ВЕСЬ ПЛАН ЗАВЕРШЁН (Part A + Part B + C1)

**Part A** (движок A1–A13) + **Part B** (5 вертикалей B1–B5: crypto/equity/futures/forex directional + options greeks)

- **C1** (unified cross-asset) = полная про-парадигма **«десятки сигналов → риск-модель → портфельная оптимизация
по всему юниверсу»** на ВСЕХ 5 классах активов + кросс-asset объединение. Всё аддитивно (новый `mode:
cross_sectional`), на бесплатных/synthetic данных с honest `pit_quality` + BYO-слотами, тесно интегрировано в Pro-режим
MVP (5-классовый Cross-Sectional Lab + greeks-конструктор + cross-asset карта + API `/api/xs/*`). **175 xs-тестов
зелёных, ноль регрессий.** Закрыт разрыв №1 (single-instrument RL → институциональная cross-sectional парадигма;
RL остаётся одним сигналом `RLAlphaSignal`).

> **Pro-readiness P0/P1 (поверх движка) — ЗАКРЫТО (2026-06-14).** Реальные бэктесты (Binance/Yahoo+EDGAR PIT)
> вместо синтетики; честность MVP (бейджи simulated/demo, 404 вместо фейк-evidence); MLOps tracking+registry
> (lineage, Ed25519); глобальная API-auth (P0). Live pre-trade VaR/CVaR/стресс; execution TWAP/VWAP/POV (нарезка
> w*−w₀); tcost-в-objective + Kelly/vol-target; авто-recovery (retry/circuit-breaker/reconcile); RL-as-signal
> завершён (P1). Всё в коде + MVP-UI. Записи: **[P0_BLOCKERS_CLOSURE.md](P0_BLOCKERS_CLOSURE.md)**,
> **[P1_BLOCKERS_CLOSURE.md](P1_BLOCKERS_CLOSURE.md)**, эндпойнты — **[MVP_DOCUMENTATION.md](MVP_DOCUMENTATION.md) §7**.

---

## 4. Как работаем

1. Пользователь пишет: «делай Stage A1» (или любой).
2. Агент выполняет Stage **целиком**: новые файлы по слоям, аддитивная интеграция, тесты, прогон `pytest`,
   обновление чек-листа выше («Stage X: DONE» + что добавлено).
3. Не ломаем существующее: перед завершением — прогон затронутых текущих тестов (зелёные).
4. Переходим к следующему Stage по готовности.

**Рекомендуемый порядок исполнения:** A1→A13 (движок), затем B1 (crypto, free-данные) как первый полный
вертикальный эталон, далее B2–B5, затем при желании C1.

---

# Part D — Live Data & RL Integration (план)

> **Цель:** замкнуть три «опциональных» направления в полноценный про-пайплайн: (1) **end-to-end free-data
> лоадеры** (binance/yahoo/oanda и т.д.) с обогащением панели реальными колонками, которые «оживляют» BYO-сигналы;
> (2) **реальный PIT-фундаментал** по классам (честный backtest value/quality); (3) **RLAlphaSignal в пресетах**
> (обученная Distributional-PPO политика как ОДИН измеримый сигнал среди многих, training НЕ трогаем).
>
> **Как это делают про кванты (и почему именно так):** реальный data-слой профи = `universe (PIT) → raw prices
> (multi-source, total-return) → enrichment (funding/fundamentals/rates/IV) joined POINT-IN-TIME с publish-lag →
> data-quality/PIT-валидация → сигналы → модели(вкл. ML/RL) как сигналы → Σ → optimizer → backtest → live`. То, что
> отличает профи от любителя — **honest PIT-дисциплина** и **data-quality gate** (не «look-ahead через свежий
> фундаментал/снапшот»). Всё ниже строится ровно по этой цепочке и тесно ложится в наш Pro-режим (Cross-Sectional
> Lab + `/api/xs/*` + honest `pit_quality` бейджи + BYO-слоты).
>
> **Принципы Part D (как и Part A-C):** всё **аддитивно** (новые файлы/лоадеры, существующее не ломаем);
> **honest-data** (каждая колонка несёт `pit_quality` + провенанс; снапшот/прокси громко помечаются, не выдаются за
> PIT); **BYO везде** (free даёт что может, точное — через BYO parquet); **кэш + rate-limit** (free-тиры лимитированы
> → parquet-кэш, backoff); **CCEA-граница** (публичные free-данные тянем на research/Cloud-стороне; приватные ключи/
> live-фиды остаются в Agent; RL-инференс по подписанному артефакту). **Каждый Stage:** новые файлы по слоям +
> тесты (DI, без сети/без обучения) + прогон `pytest` + аддитивная интеграция в `load_panel`/`build_signal_library`/
> `/api/xs` + UI-хуки в Cross-Sectional Lab + обновление чек-листа.

## Архитектурный фундамент (общий для D0-D7)

**Что есть сейчас (точки интеграции):**

- `impl_data_sources.py`: `AdapterPriceSource` (free OHLCV через `adapters.registry`), `ParquetPriceSource`/
  `ParquetFundamentals` (BYO, PIT-true), `FreeFundamentals` (yfinance снапшот, PIT-none), `total_return_index`,
  `DataSourceMeta(pit_quality)`. **Сейчас free-путь тянет ТОЛЬКО OHLCV** (`bars_to_frame`) → все non-price сигналы
  (funding/basis/mcap, фундаментал, rate_diff, iv) на free пусты (BYO).
- `service_xs_pipeline.load_panel(source='free')` → `AdapterPriceSource(vendor)` → `build_price_panel`.
- `impl_panel.PanelBuilder.asof_join(..., publish_lag_ms=...)` — готовый **PIT-join** (есть, но не подключён к free).
- Реальные free-обогащения УЖЕ в адаптерах: `adapters/binance/futures_market_data.get_funding_rate_history`,
  `adapters/yahoo/corporate_actions.get_dividends`, `adapters/yahoo/earnings.get_earnings_history/calendar`,
  `adapters/deribit/options` (IV/greeks), `adapters/oanda` (FX bars).
- `impl_rl_signal.RLAlphaSignal` (DI: utility/confidence как Series/callable) + `expected_utility_from_quantiles` +
  `conformal_confidence_from_widths` — **готов**, не хватает инференс-адаптера (checkpoint→utility) и pipeline-kind.

### Stage D0 — Unified data-assembly layer (фундамент)

- **Цель:** один оркестратор `price source + enrichment sources → собранная панель` с провенансом и `pit_quality`
  по КАЖДОЙ колонке, кэшем и honest data-quality отчётом. База для D1-D5.
- **Новые файлы:** `service_xs_data.py` (`DataAssembler`: `assemble(symbols, timeframe, price_source,
  enrichers=[...]) -> (Panel, DataQualityReport)`; каждый enricher = `Enricher` protocol `enrich(panel)->panel` +
  `meta: DataSourceMeta`; PIT-join обогащений через `PanelBuilder.asof_join(publish_lag_ms)`); `impl_data_cache.py`
  (parquet-кэш `data/cache/xs/<vendor>/<symbol>_<tf>.parquet` + TTL/atomic write; защита от free rate-limit);
  `core_xs_data.py` (`ColumnProvenance`, `DataQualityReport`: coverage %, gaps, staleness, per-column pit_quality).
- **Интеграция:** `load_panel` получает ветку `source='free'` через `DataAssembler` (обратносовместимо: без enrichers
  = текущее поведение OHLCV); новое поле `XSConfig.data.enrich: [..]` (список обогащений) + `cache: true`.
- **UI:** в Lab — **Data Quality панель** (бейджи источника/`pit_quality` по колонкам, coverage, последний бар,
  кнопка «refresh data» с индикатором кэша); honest-цвета (true=зелёный/approx=янтарь/none=красный).
- **Тесты:** `tests/test_xs_data_assembler.py` — assemble с фейковым price+enricher (DI, без сети); PIT-join не
  тянет будущее; кэш hit/miss; DataQualityReport корректен.
- **Acceptance:** собранная панель с провенансом + honest data-quality, кэш работает, обратная совместимость.

### Stage D1 — Crypto free end-to-end (Binance)

- **Цель:** crypto-сигналы funding_carry/basis/size «оживают» на бесплатных Binance-данных.
- **Новые файлы:** `loaders/crypto_enrich.py` (`FundingEnricher` поверх `binance/futures_market_data.
  get_funding_rate_history` → колонка `funding_rate`, asof-join к барам с publish-lag; `BasisEnricher` spot-vs-perp
  → `basis`; `MarketCapEnricher` — free coingecko/статичная карта → `mcap`, honest `pit_quality=approx`).
- **Интеграция:** `enrich: [funding, basis, mcap]` в crypto-пресете; `config_xs_crypto.yaml` получает `data.source:
  free, vendor: binance, enrich: [...]` вариант (synthetic остаётся дефолтом).
- **UI:** бейдж «free: binance (funding live, mcap approx)»; чекбоксы funding/basis/size перестают быть «пустыми BYO»
  когда выбран free+enrich.
- **Тесты:** `tests/test_xs_crypto_freedata.py` — фейковый funding-history → колонка появляется, signs корректны,
  PIT-lag применён; signal оживает (не NaN).
- **Acceptance:** полный crypto-контур на 100% бесплатных данных (prices+funding), end-to-end backtest + Trust Report.

### Stage D2 — Equity free + РЕАЛЬНЫЙ PIT-фундаментал

- **Цель:** честный backtest value/quality (E/P, B/P, ROE…) — ядро «реального PIT-фундаментала».
- **Новые файлы:** `loaders/equity_enrich.py` (`TotalReturnEnricher` — reuse `total_return_index` поверх yahoo
  dividends/splits → total-return цены; `PITFundamentalsEnricher` — обёртка `ParquetFundamentals` + `asof_join
  (publish_lag_ms)` → колонки earnings/book_value/fcf/roe с честным publish-lag = **PIT-true**; `EarningsEnricher`
  — yahoo earnings calendar → `has_earnings_soon`/event-флаги).
- **Honest PIT-история (про-grade):** (а) **BYO PIT-фундаментал** (Sharadar/Compustat parquet с `publish_ts`) →
  `pit_quality=true`, backtest честный; (б) **free yfinance fundamentals = СНАПШОТ** → `pit_quality=none`,
  только live-screening, **громкий баннер** «не backtest-safe» (уже есть `FreeFundamentals`); (в) free-юниверс
  survivorship-biased → честный флаг.
- **Интеграция:** `enrich: [total_return, pit_fundamentals(parquet_path), earnings]`; equity value/quality сигналы
  раскомментируются в пресете при наличии PIT-источника.
- **UI:** per-column PIT-бейдж (fundamental колонки красные если снапшот, зелёные если BYO PIT); honest-баннер
  «free фундаментал = снапшот, для backtest подайте BYO PIT».
- **Тесты:** `tests/test_xs_equity_pit.py` — asof-join фундаментала НЕ использует publish_ts из будущего (анти-
  look-ahead, ключевой про-тест); total-return корректен (реинвест дивидендов); snapshot помечен pit_quality=none.
- **Acceptance:** equity value/quality честно backtest-able на BYO PIT; free-ограничения явны в UI/логах.

### Stage D3 — Forex free (OANDA) + дифференциалы ставок

- **Новые файлы:** `loaders/forex_enrich.py` (`OandaPriceSource`-обёртка через `adapters/oanda`; `RateDiffEnricher`
  — free/BYO короткие ставки по валютам → `rate_base`/`rate_quote`/`rate_diff`; PPP/reer = BYO honest).
- **Интеграция:** `enrich: [rate_diff]` в `config_xs_forex.yaml` free-варианте; fx_carry оживает.
- **UI:** бейдж «free: oanda (rates approx/BYO)».
- **Тесты:** `tests/test_xs_forex_freedata.py` — rate_diff собирается, carry оживает, малый юниверс не ломается.
- **Acceptance:** FX carry+momentum на бесплатных/практик-данных end-to-end.

### Stage D4 — Futures continuous free + roll-accurate

- **Новые файлы:** `loaders/futures_enrich.py` (`ContinuousProxySource` — yahoo/stooq `ES=F…` уже-back-adjusted
  прокси, honest `pit_quality=approx`; `RollAccurateAssembler` — BYO контракты → `impl_continuous_futures.
  build_continuous_panel(method=ratio|diff)`; `CarryEnricher` front/back → `carry`/`roll_yield`).
- **Интеграция:** `config_xs_futures.yaml` free/BYO варианты; continuous-meta (метод/n_rolls) → UI индикатор реальный.
- **Тесты:** `tests/test_xs_futures_freedata.py` — back-adjust из BYO контрактов без скачка; carry из front/back.
- **Acceptance:** диверсиф. CTA на continuous-прокси (free) или roll-accurate (BYO) end-to-end.

### Stage D5 — Options free (Deribit/yfinance EOD) + IV-поверхность

- **Новые файлы:** `loaders/options_enrich.py` (`DeribitIVEnricher` — free крипто-опционы (IV/greeks/DVOL) →
  `iv`/`realized_vol`/`skew`/`term_slope`; `YFinanceChainEnricher` — EOD US chains, honest ограниченно;
  `OptionsBookLoader` — chain → список `OptionLeg` для greeks-конструктора с реальными IV).
- **Интеграция:** VRP/skew/term сигналы оживают; `/api/xs/options/construct` принимает реальный бук (не только demo).
- **Тесты:** `tests/test_xs_options_freedata.py` — IV-колонки собираются; бук из chain → greeks-нейтральный портфель.
- **Acceptance:** options-vol сигналы + greeks-конструктор на free Deribit/EOD данных.

### Stage D6 — RLAlphaSignal в пресетах (headline)

- **Цель:** обученная Distributional-PPO политика = ОДИН измеримый сигнал (IC рядом с классикой). **Training НЕ
  трогаем** — только инференс по checkpoint.
- **Новые файлы:** `service_rl_inference.py` (`RLInferenceAdapter`: загрузка checkpoint Distributional-PPO →
  прогон политики/критика по cross-section панели → `utility` панель [value-head ИЛИ
  `expected_utility_from_quantiles(cvar_alpha)`] + опц. `confidence` из `service_conformal` ширин интервалов;
  безопасно/ленивая загрузка torch; DI: можно подать stub-policy для тестов); pipeline-kind `rl_alpha`
  (`_build_rl`: создаёт `RLAlphaSignal.from_value_panel/from_quantiles` из адаптера по `cfg.rl.checkpoint`).
- **Интеграция:** новый блок `XSConfig.rl: {checkpoint, utility: value|cvar, cvar_alpha, conformal: {...},
  confidence: bool}`; `kind: rl_alpha` работает в ЛЮБОМ вертикальном пресете (crypto/equity/futures/forex);
  без checkpoint → graceful (сигнал нейтрален/отсутствует, honest-note). **CCEA:** артефакт подписан, инференс не
  торгует (выдаёт сигнал → μ → веса как у остальных).
- **UI:** в Lab — тумблер «RL-alpha (Distributional-PPO)» + пикер артефакта (checkpoint path) + выбор utility
  (value/CVaR); **IC `rl_alpha` показывается в Signal-диагностике рядом с факторами** (весь смысл — измеримый
  RL-edge); honest-бейдж «требует обученный артефакт (BYO/CCEA-signed)».
- **Тесты:** `tests/test_xs_rl_signal.py` — со stub-policy (без torch/без обучения): rl_alpha регистрируется,
  utility×confidence считается, IC измерим; graceful без checkpoint; интеграция в backtest одного пресета.
- **Acceptance:** RL как сигнал в реальном пресете, IC измерим, ноль изменений в обучении, graceful-fallback.

### Stage D7 — Data-Quality & PIT-validation gate (про-grade финал)

- **Цель:** «Data Trust» как параллель «Backtest Trust Report» — то, что отличает институционал.
- **Новые файлы:** `service_data_quality.py` (`pit_leak_scan` — ни одна backtest-колонка не использует publish_ts из
  будущего; `coverage_report`; `staleness`; агрегатор `survivorship`/`pit_quality` по сигналам через провенанс;
  `data_trust_report(panel, provenance) -> JSON` с verdict).
- **Интеграция:** pre-backtest **gate**: если backtested-сигнал зависит от `pit_quality=none` колонки → громкий
  warning/verdict в Trust Report (нельзя «случайно» получить look-ahead через снапшот); endpoint `/api/xs/
  data_trust`.
- **UI:** **Data Trust панель** в Lab (рядом с Backtest Trust) + lineage `signal → колонки → pit_quality`.
- **Тесты:** `tests/test_xs_data_quality.py` — PIT-leak ловится; verdict деградирует при none-колонках в backtest.
- **Acceptance:** honest data-quality gate; пользователь ВИДИТ, что сигнал честен (PIT) или только live-screening.

## Прогресс Part D

- [x] **D0 — unified data-assembly layer: DONE** — `core_xs_data.py` (`ColumnProvenance` [column/source/vendor/
      pit_quality/free], `DataQualityReport` [coverage по колонкам, per-symbol coverage, first/last ts, staleness,
      survivorship, worst_pit, **verdict ok|warn|poor**]); `impl_data_cache.py` (`ParquetCache` — атомарный parquet-
      кэш `data/cache/xs/<vendor>/<sym>_<tf>.parquet`, TTL по mtime, best-effort, инъекция now_ms для тестов;
      защита free rate-limit); `service_xs_data.py` (`DataAssembler`: price source [+кэш] → enrichers →
      `(Panel, DataQualityReport)`; `Enricher` Protocol + `FunctionEnricher`/`ColumnMapEnricher` [статич. карта→
      колонка, pit=approx]/**`AsofEnricher`** [PIT-обогатитель: long(publish_ts,symbol,vals)→`PanelBuilder.asof_join`
      с publish_lag, анти-look-ahead]; `build_quality_report`). Интеграция: `XSConfig.data.enrich[]`/`cache`/
      `cache_ttl_ms`; `load_panel(source='free')` → `assemble_free` (DataAssembler с реестром `build_enrichers`
      [D0: 'mcap' из cfg.mcaps; D1-D5 расширят]); `data_quality_for_config` (любой источник: free через assembler,
      synthetic→pit=none honest, parquet→pit=true); **API `POST /api/xs/data_quality`**. UI: карта «Data Quality
      (D0)» в Lab (кнопка «Проверить данные» → `xsDataQuality()` рендерит verdict + per-column 🟢/🟡/🔴 pit_quality +
      coverage + warnings). Обратная совместимость: free без enrichers = прежний OHLCV; free-сбой адаптера → graceful
      (assembler ловит, отчёт «poor», не падает). Тесты: `tests/test_xs_data_assembler.py` **10/10** (provenance,
      ColumnMap/Function/Asof enrichers, **PIT-safety: asof не тянет будущее + publish_lag**, cache hit/TTL,
      assembler-uses-cache, coverage/verdict, synthetic report, API); суммарно **185 xs** (+10) зелёных, без
      регрессий. Примечание: реальный free binance-fetch имеет pre-existing баг адаптера (`get_klines use_futures`)
      — чинится в D1; D0 деградирует honestly.
- [x] **D1 — crypto free end-to-end (Binance): DONE** — `loaders/` пакет (НЕ коллидирует) + `loaders/crypto_enrich.py`:
      **`FundingEnricher`** (AsofEnricher над binance `get_funding_rate_history` → `FundingPayment(timestamp_ms,
      funding_rate)` → колонка `funding_rate`, **PIT-true** [funding наблюдаем в fundingTime, publish_lag=0]);
      **`BasisEnricher`** (perp-close провайдер → asof-join → `basis = perp_close/spot_close − 1`, PIT-true, дропает
      perp_close); **`MarketCapEnricher`** (snapshot `pit=approx` ИЛИ history_fn `pit=true`). Все провайдеры DI
      (дефолт = реальный Binance-адаптер, тесты = фейки без сети; сбои → graceful NaN). Реестр: `build_enrichers`
      теперь строит crypto-обогатители (`CRYPTO_ENRICHERS=funding/basis/mcap`, mcap пропускается без cfg.mcaps).
      **Баг адаптера исправлен** (из D0): `adapters/binance/market_data.py` звал `client.get_klines(use_futures=…)`,
      а `BinancePublicClient.get_klines` принимает `market="spot"|"futures"` → заменил на `market=`. Пресет
      `config_xs_crypto.yaml`: `cache: true` + закомм. `enrich: [funding, basis, mcap]` + сигналы funding/basis/size
      «оживают» на `source: free`. UI: `xsCfg` авто-выводит `data.enrich` из выбранных crypto-чекбоксов (funding→
      funding, basis→basis, size→mcap); Data Quality панель (D0) покажет funding/basis=🟢true, mcap=🟡approx.
      Тесты: `tests/test_xs_crypto_freedata.py` **10/10** (FundingEnricher PIT + множественные сеттлменты, Basis,
      mcap snapshot/history, **FundingCarry-сигнал оживает** на собранной панели + провенанс true, реестр, **адаптер-
      регрессия** get_klines(market=) без use_futures); адаптерные тесты 130 passed (фикс безопасен); суммарно
      **195 xs** (+10) зелёных, без регрессий. Acceptance: crypto-контур на 100% бесплатных данных (prices+funding,
      PIT-корректно).
- [x] **D2 — equity free + РЕАЛЬНЫЙ PIT-фундаментал: DONE** — `loaders/equity_enrich.py`: **`PITFundamentalsEnricher`**
      (AsofEnricher над источником фундаментала `get_fundamentals(symbols,fields)→long[publish_ts,symbol,vals]` →
      колонки earnings/book_value/fcf/roe **as-of с publish-lag = анти-look-ahead**; pit_quality НАСЛЕДУЕТСЯ от
      источника: BYO `ParquetFundamentals`→**true** [backtest честный], free `FreeFundamentals` снимок→**none**
      [НЕ backtest-safe, live-screening, громко помечается]); `make_pit_fundamentals_enricher` (parquet_path→true,
      иначе free→none); **`TotalReturnEnricher`** (реинвест дивидендов+сплиты через детерминированный
      `total_return_index` → `tr_close`, pit=approx [free yahoo div]); **`EarningsEnricher`** (`has_earnings_soon`
      флаг по анонс. календарю, pit=approx). Live-провайдеры исправлены под реальные классы (`YahooCorporateActions
      Adapter.get_dividends`→`Dividend.ex_date/amount`, `ex_date`→ms через normalize_ts_ms; `YahooEarningsAdapter`);
      все DI + graceful. Реестр: `build_enrichers` теперь маршрутизирует crypto→equity (`EQUITY_ENRICHERS=
      total_return/pit_fundamentals/earnings`). XSConfig: `fundamentals_path`(BYO PIT)/`fundamentals_fields`/
      `fundamentals_publish_lag_days`. Пресет `config_xs_equity.yaml`: `cache` + закомм. `enrich:[total_return,
      pit_fundamentals]` + BYO `fundamentals_path` слот + lag=1д. UI: `xsCfg` авто-выводит `enrich:[pit_fundamentals]`
      при выбранных E/P,B/P,ROE; Data Quality панель + существующий honest-баннер показывают фундаментал 🔴none
      (free снимок) / 🟢true (BYO PIT). Тесты: `tests/test_xs_equity_pit.py` **8/8** (**PIT анти-look-ahead: до
      publish_ts — NaN + publish_lag** [ключевой про-тест], free снимок→pit=none, total-return реинвест дивиденда,
      earnings-окно, equity value-сигнал оживает + провенанс true, реестр equity, data-quality). Гоча: actions/dates
      провайдеры возвращают ts в **мс** (как панель); live-дефолты конвертят ex_date→мс. Суммарно **203 xs** (+8)
      зелёных, без регрессий. Acceptance: value/quality честно backtest-able на BYO PIT; free=снимок явно помечен.
- [x] **D3 — forex free (OANDA) + rate-diff: DONE** — `loaders/forex_enrich.py`: `parse_pair` (EURUSD/EUR_USD/
      EUR/USD → base,quote); **`RateDiffEnricher`** (дифференциал ставок → `rate_base`/`rate_quote`/`rate_diff=
      base−quote`; два режима: **static snapshot** карта `{currency:rate}` [G10 policy rates] → `pit=approx`, либо
      **history** `history_fn(currencies)→long[publish_ts,currency,rate]` → as-of join с publish-lag → `pit=true`
      PIT); `oanda_price_source` (тонкая обёртка OANDA practice, нужны OANDA_API_KEY/ACCOUNT_ID; цены — стандартный
      free-путь D0). PPP/reer/terms = BYO honest. Реестр: `build_enrichers` маршрутизирует …→forex (`FOREX_ENRICHERS=
      rate_diff`; пропуск без `policy_rates`). XSConfig: `policy_rates: {currency: rate}`. Пресет `config_xs_forex.
      yaml`: `cache` + закомм. `enrich:[rate_diff]` + `policy_rates` G10 (иллюстративные) + fx_carry «оживает» note.
      UI: `xsCfg` авто-выводит `enrich:[rate_diff]` + self-contained `policy_rates` G10 при выбранном fx_carry. Тесты:
      `tests/test_xs_forex_freedata.py` **6/6** (parse_pair, static rate_diff + знаки [EUR−USD, USD−JPY], missing-
      currency→NaN, **history PIT [NaN до publish_ts]**, fx_carry оживает на малом юниверсе + провенанс, реестр с/без
      policy_rates). Суммарно **209 xs** (+6) зелёных, без регрессий. Acceptance: FX carry на бесплатных/практик-
      данных (rate-diff из policy_rates snapshot=approx или BYO history=PIT); малый юниверс не ломается.
- [x] **D4 — futures continuous free + roll-accurate: DONE** — `loaders/futures_enrich.py`: **`ContinuousProxySource`**
      (PriceSource поверх yahoo: транслирует CME-символы ES/NQ/CL → yahoo-тикеры ES=F/NQ=F/CL=F [`DEFAULT_CME_YAHOO_MAP`]
      и обратно; УЖЕ back-adjusted continuous прокси непрозрачным методом → **`pit_quality='approx'`** honest);
      **`build_roll_accurate_panel`** (BYO контракты → точная back-adjusted серия через `impl_continuous_futures.
      build_continuous_panel(method=ratio|diff)` → `pit='true'` + continuous-meta метод/n_rolls); **`CarryEnricher`**
      (front/back контракты → `front`/`back`/`carry=(front−back)/back`/`roll_yield`, as-of PIT-true; BYO-провайдер —
      free прокси не несёт contango). Интеграция: `_price_source_for(cfg)` в assemble_free [futures free →
      ContinuousProxySource, иначе AdapterPriceSource]; реестр `build_enrichers`…→futures (`FUTURES_ENRICHERS=carry`,
      carry=BYO-only → honest skip без провайдера). Пресет `config_xs_futures.yaml`: `cache` + free continuous-прокси
      / roll-accurate BYO заметки. UI: существующий continuous-contract индикатор (B3) + Data Quality панель покажут
      proxy `pit=approx`. Тесты: `tests/test_xs_futures_freedata.py` **7/7** (ES→ES=F трансляция+возврат символа+
      pit=approx, default-map покрывает majors, **roll-accurate без скачка на роллe + meta pit=true/n_rolls**,
      CarryEnricher PIT + graceful empty, Carry-сигнал оживает, `_price_source_for` роутинг futures→proxy). Синт.
      futures-пресет не затронут (synthetic path unchanged). Суммарно **216 xs** (+7) зелёных, без регрессий.
      Acceptance: CTA на continuous-прокси (free, approx) или roll-accurate (BYO, PIT-true) end-to-end.
- [x] **D5 — options free (Deribit/yfinance EOD) + IV: DONE** — `loaders/options_enrich.py`: **`IVSummaryEnricher`**
      (AsofEnricher: iv/skew/term_slope по андерлаю as-of publish-lag) → **`DeribitIVEnricher`** (free крипто-опционы,
      history→`pit=approx`) / **`YFinanceChainEnricher`** (EOD US chains СНИМОК→`pit=none` honest, не backtest-safe);
      **`RealizedVolEnricher`** (annualized realized vol из close → `realized_vol`, **PIT-true**; нужна VRP=IV−RV);
      **`OptionsBookLoader.chain_to_legs`** (option chain → `List[OptionLeg]` с реальными IV для greeks-конструктора;
      поддержка `time_to_expiry` или `expiry_days`). Интеграция: реестр `build_enrichers`…→options (`OPTIONS_
      ENRICHERS=iv/realized_vol`; iv-вендор из `cfg.iv_vendor` deribit|yfinance); **`/api/xs/options/construct`
      принимает реальный `chain`+`spot`** (не только demo) → OptionsBookLoader → greeks-нейтральный портфель.
      XSConfig: `iv_vendor`. Пресет `config_xs_options.yaml`: `iv_vendor` + закомм. `enrich:[iv,realized_vol]`. UI:
      `xsCfg` авто-выводит `enrich:[iv,realized_vol]` под выбранные VRP/skew/term; Data Quality покажет iv 🟡approx
      (deribit)/🔴none (yfinance снимок), realized_vol 🟢true. Тесты: `tests/test_xs_options_freedata.py` **8/8**
      (IV as-of PIT, yfinance=none, realized_vol из close, **VRP/skew/term оживают** на собранной панели, **chain→legs
      →greeks-нейтральный портфель** residual≈0, expiry_days, **API chain→construct**, реестр). Суммарно **224 xs**
      (+8) зелёных, без регрессий. Acceptance: options-vol сигналы + greeks-конструктор на free Deribit/EOD данных.
- [x] **D6 — RLAlphaSignal в пресетах (RL-as-signal): DONE** — `service_rl_inference.py`: **`RLInferenceAdapter`**
      (читает ВЫХОД обученной Distributional-PPO политики по cross-section панели → `utility_panel` [value-head
      `utility='value'` ИЛИ нижние квантили `utility='cvar'` через `expected_utility_from_quantiles`] + опц.
      `confidence_panel` [conformal-ширины через `conformal_confidence_from_widths`] → `build_signal()` собирает
      готовый `impl_rl_signal.RLAlphaSignal` [utility × confidence]). **ОБУЧЕНИЕ НЕ ТРОГАЕМ** — только инференс. DI:
      `value_fn`/`quantiles_fn`/`obs_fn`/`widths_fn` или `model_loader(checkpoint)`; torch ленив и только в
      пользовательском загрузчике. **Честно:** дефолтный загрузчик = no-op (универсального нет — нужна модельно-
      специфичная obs-схема), без артефакта → `available()=False` → сигнал нейтрален (NaN), пайплайн пропускает
      (graceful). CCEA: инференс НЕ создаёт ордера (сигнал→μ→веса как у всех); артефакт BYO/CCEA-signed. Интеграция:
      `RLCfg{checkpoint,utility,cvar_alpha,confidence,conf_baseline_width}` + `XSConfig.rl`; **kind `rl_alpha`**
      (`_build_rl`) работает в ЛЮБОМ вертикальном пресете. UI: тумблер «RL-alpha (Distributional-PPO)» + пикер
      артефакта (checkpoint) + выбор utility (value/CVaR) + honest-бейдж «нужен обученный артефакт»; `xsCfg` добавляет
      rl_alpha сигнал + `rl` блок → **IC `rl_alpha` виден в Signal-диагностике рядом с факторами** (весь смысл).
      Тесты: `tests/test_xs_rl_signal.py` **8/8** (без torch/без обучения, stub-policy DI): value/cvar utility,
      **conformal confidence шринкует**, graceful без checkpoint/loader, **IC измерим** (signal_report), pipeline kind
      нейтрален без артефакта, **RL-сигнал в бэктесте РЯДОМ С КЛАССИКОЙ** (momentum+rl_alpha). Суммарно **232 xs**
      (+8) зелёных, без регрессий. Acceptance: RL как измеримый сигнал в реальном пресете, ноль изменений в обучении,
      graceful-fallback.
- [x] **D7 — data-quality & PIT-validation gate (Data Trust): DONE** — `service_data_quality.py`: **`pit_leak_scan`**
      (строгий анти-look-ahead верификатор: значение колонки не появляется в панели РАНЬШЕ первой публикации
      `value_ts >= min(publish_ts)`; список нарушений); **`signal_columns`** (lineage: какие колонки панели читает
      сигнал, по `*_col`/`*_column` атрибутам); **`data_trust_report`** (поверх D0 `DataQualityReport`: per-signal
      lineage [columns → worst_pit → backtest_safe], **PIT-violations** [backtested-сигнал зависит от
      `pit_quality=none` колонки], **trust_verdict** trusted|caution|untrusted). Интеграция: `_panel_with_provenance`/
      `_provenance_for` рефактор; **`data_trust_for_config`**; **pre-backtest gate** — `run_backtest` несёт
      `data_trust` (verdict+violations+lineage) и громко warning при none-зависимостях; API **`POST /api/xs/
      data_trust`**. UI: кнопка «Data Trust (PIT)» в Data Quality карте → `xsDataTrust()` рендерит verdict +
      lineage `сигнал → колонки → 🟢/🟡/🔴 pit` + violations. Тесты: `tests/test_xs_data_quality.py` **8/8**
      (pit_leak_scan чистый/ловит look-ahead, lineage, verdict trusted [all PIT-true]/caution [approx]/**untrusted
      [сигнал на none-колонке]**, run_backtest несёт data_trust, API). Суммарно **240 xs** (+8) зелёных, без
      регрессий. Acceptance: honest data-quality gate — пользователь ВИДИТ, честен ли сигнал (PIT) или только
      live-screening.

---

## 🎉 PART D ЗАВЕРШЕНА (D0–D7)

**D0** unified data-assembly (DataAssembler + parquet-кэш + DataQualityReport/провенанс) · **D1** crypto free
(funding/basis/mcap + фикс binance get_klines) · **D2** equity free + **РЕАЛЬНЫЙ PIT-фундаментал** (BYO PIT +
total-return + honest snapshot) · **D3** forex free (rate-diff) · **D4** futures continuous-прокси + roll-accurate ·
**D5** options free (IV + book loader) · **D6** **RL-as-signal** (Distributional-PPO как измеримый сигнал, training
не трогаем) · **D7** **Data-Trust gate** (PIT-leak + lineage + verdict). Единый паттерн: `loaders/<asset>_enrich.py`
(AsofEnricher PIT / ColumnMap / Function) → реестр `build_enrichers` → UI auto-derive `data.enrich` → Data Quality/
Trust панели с per-column `pit_quality`. Каждый сигнал оживает на бесплатных данных, честно помеченных; RL — один
сигнал среди многих; PIT-дисциплина видна и проверяема. **240 xs-тестов зелёных, ноль регрессий.** Три «опциональных»
направления (free-data лоадеры, RLAlphaSignal в пресет, реальный PIT-фундаментал) закрыты полностью.

**Рекомендуемый порядок:** **D0** (фундамент — без него остальное висит в воздухе) → **D1** (crypto, самый
богатый free-источник, быстрый эталон) → **D6** (RL-сигнал — headline, нужен только D0+любой пресет, даёт максимум
ценности рано) → **D2** (equity PIT — самый «про» по дисциплине данных) → **D3/D4/D5** (forex/futures/options
обогащения) → **D7** (data-quality gate — закрывающий про-штрих). Можно брать D1/D6 параллельно после D0.

**Маппинг на 3 исходных пункта:** free-data лоадеры = **D0-D5**; RLAlphaSignal в пресет = **D6**; реальный
PIT-фундаментал = **D2** (+ gate **D7**).

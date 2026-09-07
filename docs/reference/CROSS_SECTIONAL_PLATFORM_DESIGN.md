# Cross-Sectional Quant Platform — Дизайн и план внедрения

**Статус:** Draft v1.0 для ревью архитектуры
**Цель:** Перевести продукт из парадигмы «один RL-агент на инструмент» в институциональную парадигму
**«десятки сигналов → риск-модель → портфельная оптимизация по всему юниверсу».**

> Это не добавление фичи. Это смена ядра: что такое «стратегия» в продукте.
> RL (Distributional PPO) при этом **не выбрасывается** — он становится одним из источников альфы (`AlphaSignal`),
> наряду с классическими факторами, а не финальным генератором ордеров.

---

## 0. Сдвиг парадигмы: было → стало

### Было (single-instrument)

```
features_pipeline (per-symbol) → RL agent.decide() → ActionProto.volume_frac (target pos per symbol)
                               → Order(s) → execution → risk_guard (post-hoc)
```

Каждый инструмент торгуется своим агентом изолированно. Нет понятия «портфель как объект оптимизации»,
нет совместной риск-модели, нет cross-sectional нормализации сигналов.

### Стало (cross-sectional)

```
                ┌────────────────────────────────────────────────────────────────────┐
   PIT Universe │  на каждую дату ребаланса t:                                          │
  (survivorship)│                                                                       │
        │       │   Panel(features)  →  SignalLibrary  →  μ (alpha model, expected ret) │
        ▼       │       [N×F]            [N×K signals]        [N×1]                      │
  ┌──────────┐  │                            │                    │                     │
  │ Universe │──┼──► RiskModel: factor exposures B [N×P], factor cov F [P×P],            │
  │ Snapshot │  │              specific risk D [N] → Σ = BFBᵀ + D  [N×N]                 │
  └──────────┘  │                                          │                            │
                │   Optimizer:  w* = argmax  μᵀw − λ·wᵀΣw − tcost(w−w₀)                  │
                │              s.t. constraints (portfolio_constraints.py)               │
                │                          │                                            │
                │   Target weights w* [N×1]  →  Intents (CCEA)  →  Agent → Orders        │
                │                          │                                            │
                │   Attribution: P&L и риск по факторам/сигналам                         │
                └────────────────────────────────────────────────────────────────────┘
```

Единица работы — **вектор целевых весов по всему юниверсу** на дату ребаланса.
RL-агент — это `μ`-вклад одного сигнала, а не торговый контур.

---

## 1. Что переиспользуем vs что строим (карта на существующий код)

| Слой | Уже есть (переиспользуем) | Нужно построить |
|------|---------------------------|-----------------|
| **PIT-юниверс** | `services/survivorship.py`: `UniverseSnapshot`, `DelistingTracker`, `get_constituents(asof)` | Адаптер `UniverseProvider` поверх них; загрузка index-membership истории |
| **Данные/фичи** | `features_pipeline.py` (per-symbol TS), `data_loader_multi_asset.py`, корпдействия `services/corporate_actions.py` | **Panel API** (symbol×time×feature), PIT-fundamentals store, cross-sectional transforms |
| **Сигналы (alpha)** | отдельные индикаторы в `transformers.py` | `SignalLibrary`, `Signal` Protocol, cross-sectional normalize/neutralize, `SignalStore` |
| **Alpha-комбинация (μ)** | — | `AlphaModel` (IC-weighted / ridge / ML); **RL как сигнал** |
| **Риск-модель (Σ)** | `FactorTiltValidator` (exposures B), `services/sector_momentum.py` | `RiskModel`: factor cov F, specific risk D, Ledoit-Wolf shrinkage, Σ |
| **Оптимизатор (w*)** | `RebalanceEngine.rebalance_to_target`, `_enforce_limits` | `PortfolioOptimizer`: MVO/max-Sharpe/risk-parity/BL, turnover- и tcost-aware |
| **Ограничения** | `portfolio_constraints.py`: `PositionLimit`, `SectorExposure`, `FactorTiltLimit`, `PortfolioConstraintManager` | Связать оптимизатор с constraint manager (как hard/soft constraints) |
| **Косты исполнения** | `execution_providers*.py` (L2/L3 TCA), `lob/market_impact.py` | `tcost(w−w₀)` адаптер в оптимизатор + capacity-отчёт |
| **Бэктест** | `make_walkforward_splits.py`, `diag_val_split.py` | **Cross-sectional backtest engine** + purged/combinatorial CV + anti-overfit метрики |
| **Исполнение live** | CCEA Agent, `services/position_sync.py`, OMS | Маппинг `w*` → набор Intents → rebalance trade-list |
| **Портфельный риск live** | `services/unified_futures_risk.py::PortfolioRiskManager`, `risk_guard.py` | Factor-exposure guards на уровне портфеля |
| **Compliance** | MiFID/DORA/AI-Act слой | Attribution-report как evidence; model-governance для μ-модели |
| **RL-ядро** | `distributional_ppo.py`, conformal | Рефактор: выход агента = `AlphaSignal` (μ-вклад + uncertainty), не ордер |

**Вывод:** ~50% портфельного низа и весь execution/risk/compliance backbone уже есть.
Строим **середину**: Signals → μ → Σ → w*, плюс cross-sectional backtest.

---

## 2. Новые core-контракты (Protocols)

Добавить в `core_contracts.py` (или новый `core_portfolio.py`), слой `core_`:

```python
# core_portfolio.py
from typing import Protocol, Mapping, Sequence
import pandas as pd
import numpy as np

# --- Panel: индекс (timestamp, symbol), колонки = features/signals ---
Panel = pd.DataFrame  # MultiIndex[(ts, symbol)] × columns

class UniverseProvider(Protocol):
    def constituents(self, asof_ms: int) -> Sequence[str]: ...   # PIT, без survivorship bias
    def is_tradable(self, symbol: str, asof_ms: int) -> bool: ...

class Signal(Protocol):
    """Cross-sectional сигнал: на дату t отдаёт вектор по символам."""
    name: str
    def compute(self, panel: Panel, asof_ms: int) -> pd.Series: ...   # index=symbol, value=raw signal
    # нормализация/нейтрализация делается отдельным CrossSectionalTransform

class AlphaModel(Protocol):
    """Комбинирует сигналы → ожидаемая доходность μ по юниверсу."""
    def fit(self, signals: Panel, forward_returns: Panel) -> None: ...
    def predict(self, signals_t: pd.DataFrame) -> pd.Series: ...      # μ: index=symbol

class RiskModel(Protocol):
    """Факторная риск-модель → ковариация активов Σ."""
    def fit(self, returns: Panel) -> None: ...
    def exposures(self, asof_ms: int) -> pd.DataFrame: ...            # B: [N×P]
    def factor_cov(self, asof_ms: int) -> pd.DataFrame: ...           # F: [P×P]
    def specific_var(self, asof_ms: int) -> pd.Series: ...            # D: [N]
    def cov(self, asof_ms: int) -> pd.DataFrame: ...                  # Σ = BFBᵀ + diag(D)

class PortfolioConstructor(Protocol):
    """μ + Σ + constraints + текущие веса → целевые веса w*."""
    def solve(self, mu: pd.Series, cov: pd.DataFrame,
              current_w: pd.Series, constraints, tcost_model) -> pd.Series: ...

class CrossSectionalStrategy(Protocol):
    """Полная cross-sectional стратегия = universe+signals+alpha+risk+optimizer."""
    def target_weights(self, asof_ms: int) -> pd.Series: ...
```

Ключевая идея: **`CrossSectionalStrategy.target_weights()` возвращает Series весов** — это и есть «Intent-вектор» CCEA. Существующий per-symbol `SignalPolicy` остаётся для legacy/single-instrument режима (обратная совместимость).

---

## 3. Слой данных: Panel + Point-in-Time

### 3.1 Panel API

Новый модуль `impl_panel.py` (`impl_`):

- Сборка `Panel` (MultiIndex `(ts_ms, symbol)`) из `data_loader_multi_asset.py`.
- Выравнивание по календарю (trading sessions из `services/*calendar*`).
- As-of join фундаментальных данных (PIT) и корпдействий.
- **Гарантия PIT:** для каждой строки доступны только данные с `publish_ts <= ts`. Лаг публикаций фундаментала (например, отчётность +1–3 дня) — обязателен.

### 3.2 Survivorship-free universe

Обёртка `impl_universe.py` над `services/survivorship.py::UniverseSnapshot`:

- `constituents(asof)` отдаёт состав индекса **на дату**, включая делистнутые тогда-активные тикеры.
- Бэктест итерируется по историческому составу, а не по сегодняшнему.

### 3.3 Corporate actions / total return

- Использовать `services/corporate_actions.py`; цены → total-return серии (reinvest dividends, split-adjust).
- Для фьючерсов — back-adjusted continuous contracts (новый `impl_continuous_futures.py`, переиспользуя `impl_cme_rollover.py`).

**Acceptance:** тест «leakage probe» — модель, обученная только на будущих данных, не должна давать положительный OOS edge (sanity), и наоборот — отсутствие look-ahead подтверждается `tools/check_feature_parity.py`-аналогом для panel.

---

## 4. Signal / Alpha layer

### 4.1 SignalLibrary (`service_signals.py`)

Каталог cross-sectional сигналов, каждый реализует `Signal`:

- **Momentum**: 12-1m, 6-1m, residual momentum.
- **Value** (equity): E/P, B/P, FCF yield (из PIT-fundamentals).
- **Quality**: ROE, accruals, gross profitability.
- **Low-vol / beta**.
- **Carry** (FX/futures), **basis/funding** (crypto).
- **Reversal** (short-term).
- **RL-signal** (см. 4.3).

### 4.2 Cross-sectional transforms (`impl_cross_sectional.py`)

На каждую дату t по всему юниверсу:

- `rank` / `zscore` / `winsorize`.
- **Neutralization**: регрессия сигнала на (sector, beta, size) → остаток (нейтрализованный сигнал).
- `decay`/half-life сглаживание.
Это устраняет «случайные» экспозиции и делает сигналы сопоставимыми — то, чего сейчас нет.

### 4.3 RL как сигнал (рефактор без выбрасывания)

`distributional_ppo` сейчас выдаёт `volume_frac` (target pos). Рефактор:

- Обернуть выход политики в `RLAlphaSignal.compute(panel,t) -> pd.Series`:
  - значение сигнала = ожидаемый ретёрн/полезность позиции (можно из value-head/quantiles),
  - **conformal uncertainty** (`service_conformal.py`) → вес/шринк сигнала.
- Агент обучается per-symbol как и сейчас, но его выход — **вход в AlphaModel**, не финальное действие.
- Это сохраняет всю вашу RL-инвестицию и делает её транспарентной (один сигнал среди многих, с измеримым IC).

### 4.4 AlphaModel — комбинация в μ (`service_alpha.py`)

Методы (выбираемые в конфиге):

- **IC-weighted**: вес сигнала ∝ его rolling Information Coefficient.
- **Ridge/Elastic-net** регрессия сигналов на forward returns (с регуляризацией — анти-оверфит).
- **ML meta-model** (gradient boosting) — опционально.
Выход: `μ` (ожидаемые доходности по символам) на дату t.
Метрики качества сигналов: IC, IC-decay, quantile spread, turnover, factor-collinearity.

---

## 5. Risk model — Σ (`service_risk_model.py`)

Факторная модель ковариации (то, чего критически нет):

- **Exposures B**: переиспользовать `FactorTiltValidator.set_factor_loadings`; добавить статистические факторы (PCA) + фундаментальные (sector/size/style) + (для crypto) BTC-beta.
- **Factor covariance F**: EWMA или Ledoit-Wolf shrinkage.
- **Specific risk D**: дисперсия остатков по активу.
- **Σ = B F Bᵀ + diag(D)** — положительно определённая (shrinkage гарантирует).
- Опционально: статистическая Σ через Ledoit-Wolf напрямую (baseline).

**Зачем именно факторная:** для N=500–3000 сэмпловая ковариация вырождена; факторная даёт стабильную Σ, экспозиции для optimizer и attribution.

---

## 6. Portfolio optimizer — w* (`service_optimizer.py`)

Ядро (через `cvxpy`/`osqp` или аналитика для частных случаев):

```
maximize   μᵀw − λ · wᵀΣw − κ · tcost(w − w₀)
subject to:
   constraints из portfolio_constraints.py:
     • Σ|wᵢ| ≤ gross_max,  Σwᵢ = net_target  (e.g. 0 для market-neutral)
     • |wᵢ| ≤ position_limit  (PositionLimit)
     • |sector exposure| ≤ cap  (SectorExposure)
     • |Bᵀw|_factor ≤ tilt_cap  (FactorTiltLimit)  ← beta-neutral, etc.
     • turnover ≤ max_turnover
```

Режимы (config): `mean_variance`, `max_sharpe`, `risk_parity`, `min_variance`, `black_litterman`, `equal_weight` (baseline).

- `tcost(·)` — из ваших L2/L3 impact-моделей (Almgren-Chriss) → оптимизатор учитывает реальные косты.
- Связь с `RebalanceEngine`: optimizer ПРОИЗВОДИТ target, `RebalanceEngine.rebalance_to_target` + `_enforce_limits` доводит до исполнимого набора.

**Acceptance:** на синтетике optimizer воспроизводит известные решения (equal-weight при μ=const,Σ=I; tangency-portfolio в аналитическом случае).

---

## 7. Cross-sectional backtest engine (`service_xs_backtest.py`)

Новый бэктест поверх Panel (НЕ per-instrument env):

1. Для каждой даты ребаланса t (из walk-forward splits):
   - `universe = UniverseProvider.constituents(t)` (PIT)
   - `signals = SignalLibrary.compute(panel, t)` → transforms
   - `μ = AlphaModel.predict(...)`, `Σ = RiskModel.cov(t)`
   - `w* = PortfolioConstructor.solve(μ, Σ, w₀, constraints, tcost)`
   - trade-list = w* − w₀; косты через `execution_providers`
   - применить ретёрны t→t+1, обновить equity/exposures.
2. **Anti-overfit слой** (`service_backtest_validation.py`):
   - **Deflated Sharpe Ratio** (Bailey & López de Prado) — поправка на число испытаний.
   - **PBO** (Probability of Backtest Overfitting) через combinatorial purged CV.
   - **Purged & embargoed K-fold** (устранение лика на границах).
   - Multiple-testing haircut, OOS/IS деградация.
3. **Capacity report**: с учётом impact-кривых — на каком AUM Sharpe деградирует.

**Это P0 для доверия профи** — выводится в UI как «Backtest Trust Report».

---

## 8. Attribution (`service_attribution.py`)

- **P&L attribution**: разложение доходности по факторам (factor return × exposure) + specific.
- **Сигнальная attribution**: вклад каждого `Signal` в реализованный P&L.
- **Brinson** (allocation vs selection) для секторов.
- Экспорт tear-sheet (PDF) + JSON-evidence для compliance/LP.

---

## 9. Исполнение: w* → Intents → Orders (CCEA)

- `w*` (целевые веса) × equity = целевые ноционалы по символам = **набор Intents** (target exposure — ровно то, что CCEA разрешает передавать из Cloud в Agent).
- Agent локально превращает `(target − current)` в trade-list, прогоняет через local risk firewall, режет на slices (execution algos `5.9`), отправляет ордера.
- Reconciliation (`position_sync`) сверяет фактический портфель с целевым.
- **Соответствие CCEA сохраняется**: Cloud отдаёт target-веса (Intent), не ордера.

---

## 10. Риск и compliance на уровне портфеля

- Portfolio-level guards: factor-exposure limits, gross/net, концентрация — поверх `unified_futures_risk::PortfolioRiskManager` и `portfolio_constraints`.
- Pre-trade: проверка, что rebalance trade-list не нарушает лимиты ДО отправки.
- Compliance evidence: attribution-report + model-card μ-модели (AI-Act), reproducibility-lineage.

---

## 11. UI (Lite & Pro)

**Lite (MVP, push-button):**

- В «ИИ-пайплайн» добавить ветку **«Cross-sectional портфель»**: выбрать юниверс (индекс/список), набор сигналов (чекбоксы пресетов), режим оптимизатора, лимиты (gross/net/turnover) — одна кнопка → бэктест с Trust-Report.
- Telemetry: добавить gross/net exposure, factor exposures, turnover.

**Pro (раскрыть существующие вкладки):**

- `pro-research` → Signal Lab (IC, decay, quantile spreads по каждому сигналу).
- `pro-model-lab` → Alpha Model (комбинация, веса сигналов) + Risk Model (факторы, Σ heatmap).
- новый под-раздел **Portfolio Constructor** (optimizer settings, constraints, efficient frontier).
- `pro-backtest` → Trust Report (Deflated Sharpe, PBO, capacity).
- `pro-risk` → factor exposures, scenario/stress.
- новый **Attribution** экран.

---

## 12. Конфиг (YAML)

```yaml
# configs/config_xs_equity.yaml
mode: cross_sectional
asset_class: equity
universe:
  source: index_membership          # PIT через survivorship.UniverseSnapshot
  index: SP500
  min_adv_usd: 5_000_000            # ликвидностный фильтр
rebalance:
  frequency: weekly                 # daily/weekly/monthly
  calendar: us_equity
signals:
  - {name: momentum_12_1, transform: [winsorize, zscore], neutralize: [sector, beta]}
  - {name: value_ep,      transform: [rank]}
  - {name: quality_roe,   transform: [zscore], neutralize: [sector]}
  - {name: rl_alpha,      weight_by: conformal_confidence}   # RL как сигнал
alpha_model:
  method: ic_weighted               # ic_weighted | ridge | gbm
  ic_lookback: 252
risk_model:
  type: factor                      # factor | ledoit_wolf
  factors: [market, size, value, momentum, sector]
  cov_estimator: ledoit_wolf
optimizer:
  objective: mean_variance          # max_sharpe | risk_parity | min_variance | black_litterman
  risk_aversion: 5.0
  constraints:
    gross_max: 2.0                  # 2x leverage
    net_target: 0.0                 # market-neutral
    max_position: 0.03
    max_sector: 0.10
    beta_neutral: true
    max_turnover: 0.20
  tcost_model: almgren_chriss
backtest:
  walk_forward: {train: 756, test: 252, step: 63}
  validation: {deflated_sharpe: true, pbo: true, purge: 5, embargo: 2}
```

---

## 13. Фазовый план внедрения

> Принцип: каждая фаза — самостоятельно ценный и тестируемый инкремент. Single-instrument режим не ломается (обратная совместимость).

### Phase 0 — Фундамент данных (2–3 нед)

- `core_portfolio.py` контракты; `impl_panel.py` Panel API; `impl_universe.py` поверх survivorship.
- PIT-fundamentals store + as-of join; total-return/back-adjusted серии.
- **Deliverable:** Panel по SP500 с PIT-гарантией; leakage-probe тест зелёный.
- **Acceptance:** `pytest tests/test_panel_pit.py` (новый), нет look-ahead.

### Phase 1 — Signals + cross-sectional transforms (2 нед)

- `service_signals.py` (5–8 базовых сигналов), `impl_cross_sectional.py` (rank/zscore/neutralize).
- IC/decay диагностика на каждый сигнал.
- **Deliverable:** Signal Lab данные; отчёт IC по сигналам.

### Phase 2 — Risk model Σ (2 нед)

- `service_risk_model.py`: factor exposures (reuse FactorTiltValidator), F, D, Σ = BFBᵀ+D, Ledoit-Wolf.
- **Acceptance:** Σ положительно определена; экспозиции согласованы с `portfolio_constraints`.

### Phase 3 — Optimizer w* (2–3 нед)

- `service_optimizer.py` (cvxpy): MVO + constraints из `portfolio_constraints` + tcost.
- Связка с `RebalanceEngine`.
- **Acceptance:** аналитические тест-кейсы воспроизводятся; constraints не нарушаются.

### Phase 4 — Cross-sectional backtest + Trust Report (3 нед)

- `service_xs_backtest.py` + `service_backtest_validation.py` (Deflated Sharpe, PBO, purged CV) + capacity report.
- **Deliverable:** end-to-end бэктест cross-sectional equity long-short с Trust-Report.
- **Это веха «доверия профи».**

### Phase 5 — AlphaModel + RL-as-signal (2–3 нед)

- `service_alpha.py` (IC-weighted/ridge); рефактор `distributional_ppo` выхода в `RLAlphaSignal`.
- **Acceptance:** RL-сигнал имеет измеримый IC, входит в μ; legacy RL-режим сохранён.

### Phase 6 — Live execution + Attribution + UI (3–4 нед)

- `w*` → Intents → CCEA Agent rebalance; portfolio-level guards; `service_attribution.py`.
- UI: Signal Lab, Portfolio Constructor, Trust Report, Attribution; Lite cross-sectional ветка.
- **Deliverable:** живой ребаланс market-neutral портфеля через Agent + attribution для LP.

**Итого ориентир:** ~16–20 недель до полноценного cross-sectional контура equity; futures/FX/crypto подключаются переиспользованием (carry/basis сигналы + соответствующие риск-факторы).

---

## 14. Обратная совместимость и риски

- **Совместимость:** single-instrument `SignalPolicy`/RL-env остаётся как `mode: single_instrument`. Cross-sectional — параллельный `mode: cross_sectional`. Общий execution/risk/compliance backbone.
- **Риск 1 — optimizer-зависимость (cvxpy):** инкапсулировать за `PortfolioConstructor`, дать аналитический fallback (closed-form max-Sharpe без неравенств) для окружений без солвера.
- **Риск 2 — данные:** PIT-fundamentals и index-membership — самый дорогой кусок данных. Начать с equity (доступно), crypto работает и без фундаментала (price/funding/on-chain).
- **Риск 3 — overfit μ-модели:** обязательный purged CV + deflated Sharpe; число сигналов под контролем (multiple testing).
- **Риск 4 — позиционная развилка:** cross-sectional требует широких данных (equity) — это меняет приоритет дата-партнёрств. Зафиксировать первый вертикальный рынок (рекомендация: **US equity market-neutral** как референс, затем crypto cross-sectional).

---

## 15. Связанные существующие модули (быстрый индекс)

| Новый модуль | Опирается на |
|---|---|
| `impl_universe.py` | `services/survivorship.py` |
| `impl_panel.py` | `data_loader_multi_asset.py`, `services/corporate_actions.py`, `services/*calendar*` |
| `service_risk_model.py` | `services/portfolio_constraints.py::FactorTiltValidator`, `services/sector_momentum.py` |
| `service_optimizer.py` | `services/portfolio_constraints.py::{RebalanceEngine, PortfolioConstraintManager}` |
| `service_xs_backtest.py` | `make_walkforward_splits.py`, `execution_providers*.py`, `lob/market_impact.py` |
| `service_alpha.py` | `distributional_ppo.py`, `service_conformal.py` |
| Execution | CCEA Agent, `services/position_sync.py`, `services/unified_futures_risk.py` |

---

## 16. Применимость к 5 классам активов

**Принцип: одно ядро, плагины на класс.** Core (Panel → Signals → μ → Σ → optimizer → w*) — asset-agnostic.
На каждый класс меняются только три плагина: `SignalLibrary`, факторы `RiskModel`, asset-specific constraints.
Поэтому «сразу на всех 5» возможно **архитектурно**, но рекомендуется **последовательный rollout** (данные — длинный шест).

| Класс | Юниверс | Сигналы (плагин) | Факторы риска | Особенности / сложность |
|---|---|---|---|---|
| **Equity** | 500–3000 | momentum, value(E/P,B/P), quality(ROE), low-vol, size | market, size, value, momentum, sector | Нужны **PIT-fundamentals + index-membership**. Эталон парадигмы. ★★★ данные |
| **Crypto** | 50–200 | momentum, reversal, carry(funding), basis, on-chain, size | BTC-beta, sector(L1/DeFi/...), size | Без фундаментала. Данные доступны. ★ данные — **лучший быстрый старт** |
| **Futures (CME)** | 30–60 | trend, carry, value, vol-target | asset-class факторы (equity/rates/energy/...) | Нужны **back-adjusted continuous** серии. Это классика CTA. ★★ данные |
| **Forex** | 10–28 | carry, momentum, value(PPP), terms-of-trade | USD-beta, carry, value | Малый юниверс → оптимизатор проще (часто signal-weighted basket). ★ данные |
| **Options** | — | **vol-risk-premium, skew, dispersion, term-structure** | **vol-факторы** | ⚠️ **НЕ та же машинерия.** Портфель = экспозиции по **греческим**, не directional веса. Нужен отдельный вариант optimizer'а (vega/gamma/delta-neutral structures). Делать **последним и отдельно.** |

**Важное различие, которое надо зафиксировать:**

- **(A) 5 отдельных cross-sectional стратегий** (каждая внутри своего класса) — то, что описано выше. Переиспользует одно ядро. **Достижимо.**
- **(B) ОДИН кросс-asset портфель** (equity+futures+FX+crypto в одном оптимизаторе, единый risk-parity) — это **отдельная, более сложная** возможность: нужен унифицированный кросс-asset risk-model, нормализация валют, общий vol-target. **Это отдельная Phase 7+, не «бесплатно».**

**Рекомендация по последовательности (data-first):**
`crypto → equity → futures → FX → options(отдельный optimizer) → [Phase 7] кросс-asset (B)`.
Crypto первым — данные доступны без фундаментала, контур валидируется быстрее всего; затем equity как флагман.

---

## 17. Насколько это ложится на текущий MVP/код (честная оценка)

**Итоговый вердикт: хороший структурный fit ≈ 7–8/10 для equity/crypto/futures/FX.**
Реальный блокер — **данные, а не архитектура.** Options требует отдельного варианта оптимизатора.

### Ложится хорошо (низкое трение)

- **Слоистая архитектура** (`core_/impl_/service_/strategies/script_`): новые модули — это новые файлы на правильных слоях, без переписывания. План уже размечен по слоям.
- **CCEA Intent = target weights.** Идеальный fit: вектор весов — это набор Intent'ов. Архитектура не нарушается.
- **~50% компонентов готовы** (проверено в коде): `survivorship.UniverseSnapshot` (PIT-юниверс), `portfolio_constraints.{FactorTiltValidator, RebalanceEngine, PortfolioConstraintManager}`, `execution_providers*` (tcost/impact), `unified_futures_risk.PortfolioRiskManager`, `conformal` (вес сигнала).
- **Backend (FastAPI, ~130 endpoints)** — добавление endpoints для signals/optimizer/attribution рутинно.
- **Адаптеры на 5 классов уже есть** — плумбинг данных по классам существует.

### Трение (честные точки сопротивления)

1. **Per-symbol предположение глубоко зашито.** `features_pipeline`, RL-env (`TradingEnv`), `mediator`, `SignalPolicy.decide(features, ctx)` (ctx = ОДИН символ) — всё single-symbol. Cross-sectional — это **параллельный новый путь** (`mode: cross_sectional`), не модификация. Цена — поддержка **двух парадигм** одновременно (не переписывание, но +сложность).
2. **Слой данных — главный реальный пробел.** Сейчас данные — per-symbol OHLCV parquet. Нужны: выровненный panel, **PIT-fundamentals** (equity), **история index-membership** (equity), **back-adjusted continuous** (futures). `survivorship.py` даёт МОДЕЛЬ юниверса, но не сами ДАННЫЕ (membership/fundamentals). Это data-acquisition, не код — и это длинный шест.
3. **Нет солвера (cvxpy/osqp).** Новая зависимость + нужен аналитический fallback. Минорно.
4. **`index.html` — 2.4MB / 32k строк, один файл.** Новые экраны (Signal Lab, Portfolio Constructor, Trust Report, Attribution) — реальная фронтовая работа в неудобном файле. Backend-first смягчает (UI может отставать).
5. **RL-рефактор (RL-as-signal)** трогает обработку выхода `distributional_ppo` — изолируется за адаптером `RLAlphaSignal`, но требует аккуратности с conformal.
6. **Options** не ложится на тот же optimizer — нужен отдельный greeks-based вариант.

### Что это значит практически

- Архитектурно «всё на 5 классах» — **да**, потому что ядро общее. Но это значит **×5 на данные и ×5 на валидацию** — поэтому строим ядро один раз, классы зажигаем по очереди.
- Самый дешёвый и быстрый первый контур — **crypto cross-sectional** (нет фундаментала, данные есть).
- Single-instrument режим **не ломается** — это отдельный `mode`.

---

**Резюме:** строим середину пайплайна (Signals → μ → Σ → w*) + cross-sectional backtest с anti-overfit,
переиспользуя уже готовые PIT-юниверс, constraint-слой, execution/risk/compliance и RL-ядро (как сигнал).
Ядро asset-agnostic → все 5 классов на одном движке, но rollout последовательный (data-first: crypto→equity→futures→FX→options).
Options — отдельный greeks-optimizer. Кросс-asset единый портфель — отдельная Phase 7.
Первая веха ценности — **Phase 4 (Trust Report)**; первая монетизируемая веха — **Phase 6 (live market-neutral + attribution)**.

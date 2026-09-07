# P1-блокеры (production-операции фонда) — ЗАКРЫТИЕ

Закрытие пяти P1-блокеров из gap-анализа. Все — с тестами, вплетены в cross-sectional
live-путь. Окружение: системный Python + `PYTHONPATH=.venv/Lib/site-packages`.
Примечание: `cvxpy` в окружении НЕТ → tcost-aware решение реализовано на `scipy.optimize`
(цель — tcost в objective, а не способ решения).

---

## P1-1 — Live портфельный риск: VaR/CVaR/стресс/сценарии ✅

**Было:** риск только backtest-time + conformal-bounds.
**Стало:** pre-trade VaR/CVaR/стресс ПЕРЕД отправкой rebalance + real-time мониторинг
факторных экспозиций.

| Артефакт | Содержимое |
|---|---|
| [service_pretrade_risk.py](../../service_pretrade_risk.py) | `PreTradeRiskAnalyzer` (Gaussian + historical VaR/CVaR), `scenario_grid` (шок −10%, vol×1.5, сдвиг корреляций, кризис-combo), `factor_exposures`, `pretrade_check` (gate); `FactorExposureMonitor` (внутридневной) |
| [tests/test_pretrade_risk.py](../../tests/test_pretrade_risk.py) | 6 тестов |

Вплетено в `CrossSectionalLiveRunner.rebalance`: VaR/сценарии считаются и **блокируют**
ребаланс при нарушении лимитов; отчёт идёт в `RebalanceResult.risk_report`.

## P1-2 — Multi-period tcost-aware оптимизация + Kelly/vol-target ✅

**Было:** tcost учитывался post-hoc (turnover — hard-constraint), сайзинга нет.
**Стало:** `tcost(w−w₀)` В целевой функции (scipy SLSQP): `λ·wᵀΣw − μᵀw + κ·tcost`;
сайзинг vol-target / фракционный Kelly.

| Артефакт | Содержимое |
|---|---|
| [service_optimizer.py](../../service_optimizer.py) | `TCostModel`, `SizingConfig`, `kelly_weights`, `_solve_scipy` (tcost+constraints в objective), `_apply_sizing` |
| [tests/test_optimizer_tcost.py](../../tests/test_optimizer_tcost.py) | 5 тестов (tcost↓turnover, factor-cap в solve, vol-target попадает в цель, Kelly) |

Конфиг: `optimizer.tcost_aware/tcost_linear/tcost_quad`, `sizing/target_vol/kelly_fraction`.
**Дефолт `tcost_aware=False` → аналитический путь без изменений** (sharpe −2.3726, регрессии нет).

## P1-3 — Execution-алго: TWAP/VWAP/POV нарезка w*−w₀ ✅

**Было:** только market/bar-fill.
**Стало:** портфельный scheduler: target/current веса → trade-list → impact-aware
child-slices (TWAP/VWAP/POV) поверх AC √participation. Нарезка снижает импакт ~`1/√N`.

| Артефакт | Содержимое |
|---|---|
| [service_xs_execution.py](../../service_xs_execution.py) | `RebalanceScheduler.build_plan` → `RebalancePlan` (slices + оценка импакта) |
| [tests/test_xs_execution.py](../../tests/test_xs_execution.py) | 6 тестов (нарезка↓cost, POV, VWAP U-профиль, trade-list) |

Single-order алго уже были в `execution_algos.py`; здесь — портфельный слой. Вплетено в
`rebalance` → `RebalanceResult.execution_plan`.

## P1-4 — Авто-recovery исполнения ✅

**Было:** journal-based sequence recovery, но без backoff/circuit-breaker/auto-poll/reconcile.
**Стало:** устойчивость к сбоям брокера (Agent-зона).

| Артефакт | Содержимое |
|---|---|
| [packages/agent/execution/resilience.py](../../packages/agent/execution/resilience.py) | `RetryPolicy` (exp backoff+jitter), `CircuitBreaker` (CLOSED/OPEN/HALF_OPEN), `ResilientExecutor` (retry под breaker), `OrderStatusPoller` (авто-poll статусов/филлов), `StartupReconciler` (сверка ордеров+позиций при старте) |
| [tests/test_execution_resilience.py](../../tests/test_execution_resilience.py) | 10 тестов (время инъектируется → детерминизм без реального сна) |

## P1-5 — RL-as-signal (завершение) ✅

**Было:** легаси `volume_frac` как финальное действие.
**Стало (уже было построено в Stage A6/D6, верифицировано):** выход Distributional-PPO
обёрнут в `RLAlphaSignal` (μ-вклад из value/квантилей + conformal-uncertainty); настоящий
SB3-загрузчик чекпоинта; **в cross-sectional пути `volume_frac` отсутствует** (grep пуст).

| Артефакт | Содержимое |
|---|---|
| [impl_rl_signal.py](../../impl_rl_signal.py), [service_rl_inference.py](../../service_rl_inference.py) | `RLAlphaSignal`, `RLInferenceAdapter`, `make_sb3_distributional_loader` |
| [tests/test_xs_rl_signal.py](../../tests/test_xs_rl_signal.py), [tests/test_xs_rl_inference_e2e.py](../../tests/test_xs_rl_inference_e2e.py) | 15 тестов (проходят) |

Подключено в пайплайн: `signals: [{kind: rl_alpha}]` + `rl:` конфиг. Без чекпоинта — нейтрально (NaN).

---

## Сводка проверок

| Блок | Тесты | Статус |
|---|---|---|
| P1-1 pre-trade риск | 6 + 3 интеграции | ✅ |
| P1-2 tcost+sizing | 5 | ✅ |
| P1-3 execution scheduler | 6 | ✅ |
| P1-4 auto-recovery | 10 | ✅ |
| P1-5 RL-as-signal | 15 (существ.) | ✅ |
| Интеграция в live-путь | tests/test_xs_live_p1.py | ✅ |
| Регрессия xs-движка | 77 тестов | ✅ зелёные |
| Дефолтный путь (tcost off) | sharpe −2.3726 без изменений | ✅ |

**Всего новых P1-тестов: 30** (+15 RL существующих) = 45 зелёных.

Вплетено в `CrossSectionalLiveRunner.rebalance`: limit-guard → **pre-trade VaR/стресс** →
intents → **execution-plan (slices)**. Оптимизатор tcost/sizing — через `OptimizerCfg`.
Resilience — Agent-зона библиотека, готовая к подключению в daemon.

---

## MVP-UI (вынесено в интерфейс, browser-verified)

| Что | Где в UI | Эндпойнты |
|---|---|---|
| **Live-риск + Execution-план** (VaR/CVaR, стресс-сценарии, TWAP/VWAP/POV slices, оценка издержек) | pro-backtest → вкладка **Cross-Sectional**, карточка «🛡️ Live-риск + Execution-план» | `POST /api/xs/real/analyze`, `POST /api/xs/pretrade_risk`, `POST /api/xs/execution_plan` |
| **Тумблеры оптимизатора** (tcost в objective + linear bps, сайзинг vol-target/Kelly + target σ, RL-сигнал) — **функциональные** (меняют веса/риск/издержки) | та же карточка | override в `/api/xs/real/analyze` |
| **Индикатор авто-recovery** (circuit-breaker: ARMED/CLOSED, retry-политика, порог) | **pro-risk** (бейдж рядом с «Состояние») + строка в P1-панели | `GET /api/agent/recovery/status` |

Проверено в браузере: бейдж pro-risk = «ARMED (CLOSED)»; включение tcost меняет результат
(VaR 1.06%→1.83%, издержки 20.8→57.8 bps); execution-план рендерит slices и оценку импакта.
Нюанс UX: анализ с включённым tcost считается до ~1 мин (полный tcost-aware бэктест) —
в спиннере есть подсказка.

### Новые/изменённые файлы (код)

| Файл | Назначение |
|---|---|
| `service_pretrade_risk.py` | VaR/CVaR/стресс/сценарии + FactorExposureMonitor |
| `service_optimizer.py` | tcost в objective (scipy), Kelly/vol-target sizing |
| `service_xs_execution.py` | RebalanceScheduler (TWAP/VWAP/POV slices) |
| `packages/agent/execution/resilience.py` | retry/circuit-breaker/poller/reconciler |
| `service_xs_live.py`, `service_xs_pipeline.py`, `xs_api.py`, `app.py`, `index.html` | вплетение + endpoints + UI |

**Команды проверки:**

```
PYTHONPATH=.venv/Lib/site-packages python -m pytest tests/test_pretrade_risk.py tests/test_optimizer_tcost.py tests/test_xs_execution.py tests/test_execution_resilience.py tests/test_xs_live_p1.py -q
```

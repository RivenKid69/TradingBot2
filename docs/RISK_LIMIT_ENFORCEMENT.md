# Enforcement риск-лимитов в live-контуре (P0-B)

> **Статус:** ✅ ЗАКРЫТО 2026-07-16 · закрывает P0-B / §3.6 из
> [PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md](history/PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md).
> Тесты: `tests/test_live_risk_enforcement.py` (16) + live smoke на CCEA-сервере.

## Проблема (что было)

Форма риск-лимитов Lite сохраняла в `configs/risk.yaml` блок `lite_limits`
(дневной лимит убытка, макс. просадка, плечо, концентрация, PDT/SPAN/Greeks-
тумблеры). Но **никакой рантайм-код их не применял** — `app.py` только
отображал и записывал значения. Из всей формы в торговом контуре реально
работали лишь `max_total_notional` / `max_total_exposure_pct` /
`exposure_buffer_frac` (читает `service_signal_runner.py`).

Опасность: трейдер выставлял «дневной стоп −$1000» и «плечо 2×», думал, что
защищён, — а стоп-лосса по убытку не существовало. Endpoint честно возвращал
`applied_to_agent: false`, но и «следующий RUN» лимиты бы не подхватил —
потребителя просто не было.

## Решение: двухуровневая защита (как на профессиональных десках)

Институциональный риск-менеджмент разделяет два независимых контура. Мы
реализовали оба и связали с пользовательскими `lite_limits`.

### 1. Pre-trade gate (перед каждым ордером)

`LiveExecutionEngine` уже вызывал `RiskChecker.check()` на каждый Intent, но
supervisor строил движки с **дефолтным** RiskChecker (хардкод-лимиты) — лимиты
пользователя игнорировались, а проверок плеча и просадки не было вовсе.

Расширили `packages/agent/policy/risk_checker.py`:

| Проверка | Логика | Пропуск |
|----------|--------|---------|
| `LEVERAGE` | `(gross_exposure + \|notional\|) / equity > max_leverage` → блок; warn при 0.9× | на exit-ордерах (снижают риск) |
| `MAX_DRAWDOWN` | `(peak − equity)/peak > max_drawdown_pct` → блок нового риска | на exit-ордерах |

`services/live_risk_limits.py::build_risk_checker()` мапит `lite_limits` в
kwargs RiskChecker: `daily_loss_limit_usd → max_daily_loss`,
`max_concentration_pct → max_concentration_pct/100`, `max_leverage`,
`max_drawdown_pct/100`, `max_position_size = max_leverage × equity`.
Незаданные лимиты остаются `None` → проверка присутствует, но всегда проходит
(обратная совместимость сохранена).

### 2. Intra-day monitor / circuit breaker (account-level стоп-лосс)

Pre-trade недостаточно: **убыток растёт от движения рынка без новых ордеров**.
`LiveRiskMonitor` (`services/live_risk_limits.py`) питается snapshot'ом
P&L-леджера Agent'а после каждого booked-fill и на периодической оценке:

- отслеживает **trailing peak equity** (high-water mark), долговечный —
  переживает рестарт в пределах дня (`state/live_risk_peak.json`);
- при `day_pnl ≤ −daily_loss_limit_usd` **или** `drawdown ≥ max_drawdown_pct`
  триггерит `halt_callback` **один раз** (идемпотентно, без повторных
  срабатываний до явного reset);
- halt в supervisor = `ops_kill_switch._trip()` + `emergency_halt()` (отмена
  рабочих ордеров, флэттенинг) + отзыв live-мандатов CCEA;
- `reset_day()` на EOD (peak = текущий equity, снятие breach-флага);
- `reset_breach()` при ручном panic-reset.

## Проводка (wiring)

`ccea/desktop_supervisor.py`:

- `start()` поднимает `LiveRiskMonitor(halt_callback=self._on_risk_breach, …)`.
- `_build_user_risk_checker()` = `build_risk_checker(load_live_risk_limits(), equity=…)`;
  и paper-, и live-движок строятся с ним + `on_fill` оборачивается
  `_risk_wrapped_on_fill()` (после каждого fill → `_evaluate_risk()`).
- `reload_risk_limits()` (вызывается при сохранении формы) сбрасывает движки,
  чтобы пересобрать RiskChecker с новыми лимитами — применяется без рестарта.
- `eod_close()` → `_risk_monitor.reset_day()`.
- `status()` включает `risk_enforcement`.

## REST API

| Endpoint | Назначение |
|----------|------------|
| `GET /api/risk/enforcement` | текущий статус: `armed` / `breached` / `no_limits` / `agent_offline`, usage-проценты (daily loss / drawdown / leverage), флаг `kill_switch_tripped` |
| `POST /api/risk/limits` | сохраняет `lite_limits` в `risk.yaml`, вызывает `reload_risk_limits()`, честно возвращает `applied_to_agent` + `enforcement` |
| `POST /api/panic_reset` | снимает kill switch + `reset_risk_breach()` → снова `armed` |

## MVP UI

Карточка **«Применение лимитов (live)»** в обзоре Lite-портфеля
(`index.html`, id `risk-enforce-badge` / `risk-enforce-body`):

- бейдж `ARMED` / `BREACHED` / нет лимитов;
- прогресс-бары использования дневного убытка / просадки / плеча к лимиту;
- баннер circuit breaker при пробое;
- обновляется в `loadLitePortfolio` и по интервалу 5с;
- тост сохранения различает «🛡️ применены live» vs «💾 сохранены» по
  `applied_to_agent`.

## Проверка (live smoke, 2026-07-16)

Прогон на реальном CCEA-сервере (порт 8127):

```
1) до лимитов        → status=no_limits, enforced=False
2) сохранить лимиты  → applied_to_agent=True (daily $1000, leverage 2×, DD 15%, conc 50%)
3) после сохранения  → status=armed, enforced=True
4) pre-trade блок    → ордер 5 BTC ($250k) отклонён HTTP 400 «Concentration 255% > 50%»
5) circuit breaker   → убыток −$1500 (движение mark) → status=breached,
                        breaches=['daily_loss'], kill_switch=True,
                        следующий ордер отклонён «Kill switch активен»
6) panic_reset       → kill_switch=False, status=armed (снова вооружён)
```

Полный цикл `armed → pre-trade block → intra-day breach → auto-halt →
reset → armed` подтверждён.

## Файлы

- `packages/agent/policy/risk_checker.py` — `LEVERAGE` / `MAX_DRAWDOWN` checks.
- `services/live_risk_limits.py` — loader, `build_risk_checker`, `LiveRiskMonitor`.
- `ccea/desktop_supervisor.py` — проводка pre-trade + monitor + halt.
- `app.py` — `/api/risk/enforcement`, обновлённые `/api/risk/limits`, `/api/panic_reset`.
- `index.html` — карточка «Применение лимитов (live)».
- `tests/test_live_risk_enforcement.py` — 16 тестов (pre-trade RiskChecker,
  loader/builder, LiveRiskMonitor breach/idempotent/peak-durability/reset,
  REST-offline). Supervisor-интеграция — через live smoke: два реальных
  control-plane в одном pytest-процессе конфликтуют на общем SQLAlchemy-стейте.

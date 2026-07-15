# Планировщик регулярных задач

**Файлы:** `services/scheduler.py` (ядро) · `configs/scheduler.yaml` (каталог задач) ·
проводка и действия — `app.py` (`_build_scheduler_actions`, `/api/scheduler/*`) ·
тесты — `tests/test_scheduler_service.py` (24) · UI — карточка «Планировщик задач» в Lite Data Manager.

Закрывает P0-F из [PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md](../PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md):
до него обновление данных, drift-retrain и EOD-переоценка запускались только руками.

---

## Принципы (почему именно так)

| Решение | Обоснование |
|---|---|
| **Anacron-семантика, не cron** | Desktop-машина не работает 24/7. Пропущенный из-за выключенного приложения запуск навёрстывается при старте, но только внутри `catch_up_grace_sec` (устаревший слот честно пропускается — данные за «вчера в 3 ночи» бэкапить в 19:00 уже поздно и/или бессмысленно). Проверено вживую 2026-07-15. |
| **Fail-closed пайплайны** | Составная задача (features → targets → … → training table) останавливается на первом упавшем шаге. Ни один шаг не работает на выходе битого предыдущего. |
| **LeakGuard-пол не ослабляем** | `research_nightly` принудительно поднимает `decision_delay_ms` до ≥ 8000 мс, что бы ни было в params (тест `test_research_pipeline_enforces_leakguard_floor`). |
| **CCEA: торговые задачи не автостартуют** | `trading_impacting: true` ⇒ автозапуск только при `enabled: true` **и** env `RIVEN_ALLOW_SCHEDULED_TRADING=1` (двойной opt-in); ручной запуск — только с явным подтверждением человека (`confirm_trading=true`, UI показывает confirm-диалог). Иначе — журналируемый `skipped`, не ошибка. |
| **Один воркер исполнения** | Глобальная сериализация: обучение и инжест никогда не идут одновременно (диск/CPU/API-квоты). |
| **Ретраи с exp backoff + алерт** | `max_retries`, `retry_backoff_sec` (×2 на каждую попытку). После исчерпания — алерт в Telegram/webhook (`services/alerts.py`, секция `alerts:` в scheduler.yaml). |
| **Терминальный статус — только по exit code** | Воркеры запускаются той же машинерией, что `/api/run_job` (pid-файл + durable status); успех = `state=succeeded, exit=0`, таймаут убивает процесс и фиксируется отдельным статусом. |
| **Долговечность** | Состояние — `state/scheduler_state.json` (atomic write), журнал — `logs/scheduler_runs.jsonl`. Enable/disable из UI переживает рестарт. |
| **Skip ≠ failure** | Невыполненный precondition (нет данных, нет конфига, CCEA выключен) — честный `skipped` с причиной, без алертов и ретраев. |

## Каталог задач (defaults)

| id | Действие | Расписание (UTC) | Вкл. по умолчанию | Примечание |
|---|---|---|---|---|
| `data_refresh` | ingest + контроль результата (файл обновился, строк > 0) | 00:15 | ❌ | требует настроенного `configs/ingest.yaml` |
| `research_nightly` | features→targets→no-trade→splits→training table | 01:00 | ❌ | включать после стабильного data_refresh |
| `drift_check` | run_psi → PSI-решение → рекомендация или auto-retrain | 06:00 | ✅ (auto_retrain=false) | cooldown ретрейна durable (`state/drift_retrain_state.json`) |
| `eod_close_report` | CCEA eod_close + снапшот + отчёт `reports/daily/DATE.{json,md}` | 21:15 | ✅ | без CCEA — честный отчёт «NAV не фиксировался» |
| `tca_weekly` | TCA по сделкам с arrival price → `reports/tca/` | Вс 08:00 | ❌ | без arrival price — честный skip |
| `state_backup` | zip `state/` + `logs/*.jsonl` + `configs/*.yaml`, ретенция | 03:00 | ✅ | |
| `log_rotation` | gzip-архив `logs/*.log` старше N дней | 03:30 | ✅ | |
| `xs_rebalance` | веса → гардрейлы → Intents → CCEA Agent OMS | 13:45 торговые дни | ❌, trading_impacting | **реальный ребаланс** (`service_xs_rebalance`); turnover-cap/no-trade-band/подпись RL-модели; см. [MODEL_SIGNATURE_AND_REBALANCE.md](MODEL_SIGNATURE_AND_REBALANCE.md) |

## REST

- `GET /api/scheduler/status` — задачи, расписание, последние статусы, next_run.
- `GET /api/scheduler/runs?limit=50` — журнал запусков (новые сверху).
- `POST /api/scheduler/job/{id}/enable` `{enabled: bool}` — переживает рестарт.
- `POST /api/scheduler/job/{id}/run` `{confirm_trading?: bool}` — ручной запуск; для торговых задач без подтверждения — 409.

Все под глобальной auth-middleware. При `RIVEN_ENABLE_SCHEDULER=0` эндпоинты отвечают 503.

## Окружение

| Переменная | Default | Смысл |
|---|---|---|
| `RIVEN_ENABLE_SCHEDULER` | `1` | Выключить планировщик целиком |
| `RIVEN_ALLOW_SCHEDULED_TRADING` | (нет) | Двойной opt-in для автозапуска trading-impacting задач |
| `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`, `ALERT_WEBHOOK_URL` | (нет) | Каналы алертов (`alerts.channel` в scheduler.yaml) |

Под pytest автостарт молчит намеренно (иначе каждый тестовый импорт `app` запускал бы catch-up задачи в рабочей копии);
тесты создают собственные экземпляры `SchedulerService`.

## Что осталось на следующие итерации

- Праздничные календари per-exchange для `market_days_only` (сейчас — только Сб/Вс; движки календарей в проекте есть: `services/cme_calendar.py`, session router).
- ✅ Боевое наполнение `xs_rebalance` (P1-C) сделано — см. [MODEL_SIGNATURE_AND_REBALANCE.md](MODEL_SIGNATURE_AND_REBALANCE.md). Осталось: авто-ребаланс на live-брокере (сейчас fail-closed на paper).
- Зависимости между задачами по артефактам (сейчас — последовательность шагов внутри задачи + разнесённые времена).

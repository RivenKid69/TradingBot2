# Квант-гэпы: DAG-пайплайн, GPU, сравнение экспериментов, интрадей-фиды

> **Статус:** ✅ ЗАКРЫТО 2026-07-16 · закрывает §5.21+/22/24/25 (P2-H/I/M) из
> [PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md](PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md).
> Тесты: `tests/test_quant_gaps_p2him.py` (25) + регрессия (scheduler/tracking/audit —
> 105 passed) + live smoke + **реальный сетевой смоук** (Binance: 120 минуток,
> 1834 тика скачаны в parquet).

## 1. Research-пайплайн как DAG (`services/research_pipeline.py`)

Планировщик (P0-F) отвечает за «когда»; квант-гэп — «что и в каком порядке».
Раньше ночной research-цикл был захардкоженной линейной цепочкой внутри
scheduler-action. Теперь — легковесный DAG-оркестратор (паттерн
Airflow/Dagster/Prefect, без внешних демонов):

- **Декларативные YAML-спеки** `configs/pipelines/*.yaml`: шаги = worker из
  единого реестра джобов приложения + params + `depends_on` + retries + timeout.
  Валидация: уникальные id, известные зависимости, ацикличность.
- **Топологическое выполнение** (Kahn, сортированное ready-множество —
  детерминированный порядок).
- **Fail-closed**: шаг с упавшей зависимостью получает статус `blocked` и НЕ
  выполняется.
- **Долговечный журнал** `state/pipeline_runs/<run_id>.json` (пишется после
  каждого шага) → **resume** докатывает упавший прогон, не повторяя
  succeeded-шаги. Cancel между шагами.
- **LeakGuard-пол в движке**: `decision_delay_ms` клампится к ≥ 8000 мс —
  пользовательский YAML не может ослабить защиту от утечки.
- Референс-спека `configs/pipelines/research_nightly.yaml`: features →
  targets → no-trade → splits; training_table — параллельная ветка от features.
- Scheduler-action **`pipeline.run`** (params.pipeline=<имя>) — DAG по
  расписанию; `pipeline.research_nightly` остаётся как legacy-линейный пресет.

REST: `GET /api/pipeline/list`, `POST /api/pipeline/run` (фоновый тред,
+`resume_run_id`), `GET /api/pipeline/status`, `POST /api/pipeline/cancel`.
UI: карточка **«Research-пайплайн (DAG)»** рядом с планировщиком (шаги со
статусами, Запустить/Resume).

## 2. GPU-обучение (`services/hardware.py`, P2-H)

SB3 умеет CUDA сам, но не было ни детекции, ни контроля, ни честности —
дефолтная установка ставит CPU-torch, и обучение молча шло на CPU.

- `gpu_status()`: какой torch собран (`torch.version.cuda`), доступна ли CUDA,
  устройства (имя/VRAM/capability), GPU-инвентарь через nvidia-smi (виден даже
  при CPU-torch), **причина** недоступности и **подсказка установки**
  (`pip install -e ".[gpu]"`).
- `resolve_device(requested)`: auto→cuda при наличии; запрошенный cuda без CUDA
  **честно деградирует в CPU** (research fail-open) с причиной в логе/UI.
- Проводка: `train_model_multi_patch.py --device auto|cpu|cuda[:N]` → резолв →
  env-мост `RIVEN_TRAIN_DEVICE_EFFECTIVE` → `DistributionalPPO(device=…)`
  (тот же паттерн, что MARKET_REGIMES_JSON). Джоб `run_train` принимает
  `params.device` из UI.
- `quick_benchmark()` — matmul CPU vs GPU по запросу (`?bench=1`).

REST: `GET /api/hardware/gpu`. UI: селект устройства + честный GPU-чип в
Quant Lab → Обучение («GPU недоступна — torch собран без CUDA…»).

## 3. Экран сравнения экспериментов (P2-I)

Tracking/registry были, экрана сравнения не было.

- `GET /api/experiments/{exp}/compare?runs=a,b,…` — union параметров и метрик
  по прогонам, флаг `differs` на строку (json-нормализация значений).
- `GET /api/experiments/{exp}/runs/{id}/bundle` — **reproducibility-бандл**:
  run record + полные истории метрик + ссылки registry (модель/версия/stage/
  sha256/подписан ли) + среда (python/torch/git sha).
- UI (Pro → MLOps): чекбоксы в списке прогонов → «Сравнить выбранные (2+)» →
  таблица: метрики бок-о-бок (лучшее значение зелёным по карте направлений
  sharpe↑/loss↓/…), параметры с подсветкой отличий, тумблер «только отличия»;
  кнопка бандла у каждого прогона.

## 4. Интрадей-фиды: минутки/тики (`services/premium_data.py`, P2-M)

Адаптеры уже умели минутные бары — не было единой точки, честной
entitlement-матрицы, layout'а и UI.

- **Матрица вендоров** (`vendor_status()`): binance (минутки+**тиковая история**
  бесплатно), polygon (минутки, платный план, ключ), alpaca (минутки, IEX
  бесплатно / SIP платно), oanda (минутки). `ready` = адаптер importable **и**
  ключи заданы (placeholder-ключи не считаются). Тики честно `unavailable`
  там, где адаптер не умеет историю.
- **Минутки** `download_minute_bars()`: оконная пагинация поверх `get_bars`,
  дедуп/сортировка, схема **как у `scripts/download_stock_data.py`**
  (`timestamp(sec)/open/high/low/close/volume/symbol`) →
  `data/minute/{vendor}/{SYMBOL}_{tf}.parquet` + `.manifest.json`
  (vendor/диапазон/rows/sha256) для QC и lineage. Объём берётся из
  канонического `Bar.volume_base` (fallback `.volume`/`.volume_quote`).
- **Тики** `download_binance_agg_trades()`: настоящий исторический бэкфилл
  через публичный `/api/v3/aggTrades` с канонической fromId-пагинацией →
  `data/ticks/binance/{SYMBOL}_ticks.parquet`
  (ts_ms/price/qty/agg_id/is_buyer_maker/symbol).
- CLI `scripts/download_premium_data.py {vendors|bars|ticks}`.

REST: `GET /api/data/premium/vendors`, `POST /api/data/premium/download`
(фоновый джоб через канонический CLI, 409 при идущей закачке),
`GET /api/data/premium/download/status` (+хвост лога). UI: карточка
**«Интрадей-фиды (минутки/тики)»** в Data Manager: матрица
READY/НУЖНЫ КЛЮЧИ/НЕТ АДАПТЕРА, форма закачки, прогресс.

## Проверка

- `tests/test_quant_gaps_p2him.py` — 25 тестов: hardware (честность/резолв/env/
  REST/CLI-флаг/джоб-проводка), premium (матрица/схема+манифест/дедуп/ошибки/
  fromId-пагинация/REST-валидация), compare (diff/404/400/bundle),
  DAG (валидация/топология/blocked fail-closed/resume/LeakGuard-кламп/REST/
  scheduler-action).
- Регрессия: scheduler + experiment tracking + lite audit = 105 passed.
- **Реальный сетевой смоук** (Binance, без ключей): 120 минутных баров и
  1834 aggTrades-тика скачаны в parquet с корректной схемой; смоук поймал и
  починил реальный баг (`Bar.volume_base` vs `.volume`).
- Live smoke REST: gpu/vendors/pipeline list/404.

## Файлы

- `services/research_pipeline.py`, `configs/pipelines/research_nightly.yaml`
- `services/hardware.py`; `train_model_multi_patch.py` (`--device`+env-мост+ctor)
- `services/premium_data.py`, `scripts/download_premium_data.py`
- `app.py` — REST (pipeline/hardware/premium/compare/bundle) + `pipeline.run`
  action + `run_train` device
- `index.html` — 4 UI-поверхности (DAG-карточка, GPU-чип+селект, MLOps-сравнение,
  интрадей-фиды)
- `tests/test_quant_gaps_p2him.py`

## Остаётся открытым (честно)

- §5.23 optuna-tune как сервис (PBT есть; optuna в excludes сборки).
- Распределённое обучение (DDP/Ray) — device=cuda закрыт, мульти-нода нет.
- L2-глубина/borrow/delistings из §5.25 — отдельные ветки P2-M.

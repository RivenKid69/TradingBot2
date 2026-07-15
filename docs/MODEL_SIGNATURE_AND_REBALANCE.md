# Гейт подписи моделей + регулярный XS-ребаланс

Два связанных закрытия из [PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md](../PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md):
**§4.7** (подпись моделей не проверялась при загрузке в live) и **§4.9 / P1-C**
(движок XS выдавал веса, но не было регулярного пути «веса → ордера»).

---

## 1. Ed25519-гейт модельных артефактов (§4.7)

**Файлы:** `services/model_signature_gate.py` · проводка — `service_rl_inference.py`
(`make_sb3_distributional_loader`) · REST `GET /api/models/verify_for_live` ·
тесты `tests/test_model_signature_gate.py` (13).

### Зачем это security-контрол, а не бюрократия

SB3-чекпоинт (`.zip`) десериализуется через **pickle**. Загрузка неподписанного
или подменённого файла = **исполнение произвольного кода** в процессе, у которого
есть доступ к брокерским ключам. Реестр (`service_experiment_tracking.ArtifactSigner`)
подписывал артефакты Ed25519, но ни одна точка загрузки подпись не проверяла —
защита была декоративной. CCEA design doc прямо требует «Artifact Signature
Verification: REQUIRED».

### Как работает

Перед десериализацией (и до импорта torch) загрузчик вызывает
`verify_model_artifact(path, live=...)`. Гейт:

1. считает sha256 файла;
2. ищет запись в реестре **по digest, а не по пути** (пользователь может грузить
   исходный файл, а не копию из реестра — отпечаток тот же);
3. проверяет Ed25519-подпись реестровой записи;
4. при `RIVEN_REQUIRE_PRODUCTION_MODEL=1` дополнительно требует `stage=production`.

### Политики (`RIVEN_MODEL_SIGNATURE_POLICY` или аргумент)

| Политика | Поведение | Дефолт |
|---|---|---|
| `enforce` | любой провал → `ModelSignatureError` (fail-closed) | **live-контексты** |
| `warn` | вердикт логируется, загрузка продолжается | research/backtest |
| `off` | проверка пропускается (только явным решением оператора) | — |

Важно: в enforce-режиме провал в торговом пути поднимает исключение, а **не**
возвращает «нейтральный сигнал» — контур обязан остановиться, а не молча
торговать без модели.

### Регистрация модели, чтобы она прошла гейт

```python
from service_experiment_tracking import get_registry
get_registry().register("crypto_alpha", artifact_path="models/ppo_agent.zip")
# для production-требования:
get_registry().transition("crypto_alpha", 1, "production")
```

Проверить артефакт без загрузки: `GET /api/models/verify_for_live?path=...` →
вердикт с полем `ok` (пройдёт ли enforce-гейт) и `effective_live_policy`.

---

## 2. Регулярный XS-ребаланс (§4.9 / P1-C)

**Файлы:** `service_xs_rebalance.py` (планнер + runner) · `submit_rebalance_order`
в `ccea/desktop_supervisor.py` · планировщик — job `xs_rebalance` в
`configs/scheduler.yaml` (`trade.xs_rebalance` в `app.py`) · REST
`/api/xs/rebalance/{run,last}` · тесты `tests/test_xs_rebalance.py` (23).

### Поток

```
целевые веса (service_xs_pipeline.latest_target_weights)
   → гардрейлы (plan_rebalance, чистая функция)
   → Intents по одному (supervisor.submit_rebalance_order)
   → реальный Agent OMS: policy firewall → hash-chain журнал → fill
      → P&L ledger + immutable blotter + cash GL + MAR surveillance
   → журнал решения (logs/xs_rebalance/<ts>.json + last.json)
```

### Гардрейлы одного ребаланса (`RebalanceLimits`)

| Параметр | Смысл |
|---|---|
| `max_position_weight` (0.20) | клип целевого веса на инструмент (концентрация) |
| `drift_band_bps` (25) | \|Δw\| ниже порога — позицию не трогаем (no-trade band → меньше издержек) |
| `min_trade_notional` (25) | мелкие дельты не торгуем |
| `max_turnover` (0.25) | Σ\|Δnotional\| ≤ 25% equity; при превышении — **пропорциональный скейлинг** всех дельт (сохраняет направление сдвига; портфель сходится к цели за несколько периодов) |
| `max_orders` (50) | верхняя граница, отброшенные — в журнал |

Дополнительно: позиции без цены — честный skip (не торгуем вслепую); позиция,
пересекающая ноль (лонг→шорт), режется на `close_leg` + `open_leg` (иначе
OMS-намерение неоднозначно); ордера сортируются «сначала продажи» (высвобождают кэш).

### Гейты (fail-closed, до любых расчётов)

1. kill switch активен → `blocked`;
2. нет конфига / нет CCEA Agent → `blocked`;
3. `paper_only=true`, а брокер live → `blocked` (v1 исполняет **только** на
   sim_paper; авто-ребаланс на живом брокере сознательно запрещён до отдельного
   включения через CCEA approval);
4. RL-модель в конфиге не прошла подпись → `blocked`.

### Статусы записи решения

`blocked` (гейт не пройден, ничего не считалось), `dry_run` (план построен, ордера
не слались), `noop` (все дельты в бэнде), `ok` / `partial` / `failed` (по факту OMS).

### Запуск

- **По расписанию:** job `xs_rebalance` в `configs/scheduler.yaml` — торговая
  задача, двойной opt-in (`enabled: true` + env `RIVEN_ALLOW_SCHEDULED_TRADING=1`),
  плюс `params.config` = путь к XS-конфигу. `dry_run: true` в params строит план
  без отправки.
- **Вручную:** `POST /api/xs/rebalance/run` `{config, dry_run, confirm_trading, ...}`.
  `dry_run=true` по умолчанию (безопасно); реальная отправка — только с
  `confirm_trading=true` (иначе 409). Последнее решение — `GET /api/xs/rebalance/last`.

### Проверено вживую (2026-07-15)

На реальном сервере с CCEA paper-контуром: dry-run построил план из 8 ордеров
(turnover raw 1.0 → scale 0.25, sells-first), confirmed-прогон отправил 8/8 через
Agent OMS → 8 позиций в портфеле Agent'а + 8 записей в блоттере + журнал решения;
`confirm_trading=false` при `dry_run=false` вернул 409; гейт подписи вернул честный
вердикт с `effective_live_policy=enforce`.

## Что осталось на следующие итерации

- Авто-ребаланс на **live-брокере** (сейчас fail-closed на paper) — через отдельный
  CCEA approval-процесс.
- Champion/challenger + автоматический promote (`RIVEN_REQUIRE_PRODUCTION_MODEL`
  уже даёт stage-gate; полное замыкание drift→retrain→promote — P2 №16).

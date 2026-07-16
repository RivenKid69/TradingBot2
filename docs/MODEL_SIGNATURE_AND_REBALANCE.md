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
3. `paper_only=true`, а брокер live → `blocked`;
4. брокер live и **нет действующего мандата авторизации** (или конфиг
   изменился / бюджет исчерпан / превышен потолок) → `blocked` (см. §3);
5. RL-модель в конфиге не прошла подпись → `blocked`.

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

---

## 3. Авторизация авто-торговли на LIVE-брокере (CCEA operator approval)

**Файлы:** `packages/agent/approval/live_trading_authorization.py` (Agent-зона) ·
проводка — `ccea/desktop_supervisor.py` (`grant/revoke_live_trading`,
`live_trading_status`, `submit_rebalance_order(allow_live=...)`, `_ensure_live_engine`) ·
гейт в `service_xs_rebalance.run_rebalance` · REST
`/api/ccea/live_trading/{request,grant,revoke,status}` · тесты
`tests/test_live_trading_authorization.py` (20) + live-ветки в `tests/test_xs_rebalance.py`.

### Зачем отдельный контур авторизации

Автоматическая отправка ордеров на **реальный счёт** — самый чувствительный
контур платформы. Он открывается ТОЛЬКО через явную локальную авторизацию
оператора, спроектированную по образцу реального algo-governance (MiFID II RTS 6:
«material change to an algorithm requires re-authorisation») и prime-brokerage
pre-trade mandates.

### Модель мандата

| Свойство | Смысл |
|---|---|
| **Human-in-the-loop, локально** | выдаёт только оператор Agent-зоны; **Cloud выдать не может** (CCEA-принцип) |
| **Привязка к хешу конфига** | мандат действителен только для того конфига (sha256 канонизированного YAML), который оператор видел; любое изменение → мандат невалиден → снова fail-closed |
| **Привязка к брокеру** | мандат для `binance` не открывает `oanda` |
| **Потолок лимитов** | max turnover / notional-на-ребаланс / orders-на-ребаланс; рантайм-лимиты ребаланса прижимаются к потолку (строже — можно, слабее — нет) |
| **TTL** | мандаты истекают (жёсткий максимум — неделя) |
| **Бюджет** | опционально: макс. суммарный нотионал и/или число ребалансов; при исчерпании мандат закрывается |
| **Revoke** | в любой момент; авто-снятие при emergency halt и смене брокера |
| **Hard-caps** | оператор не может превысить жёсткие пределы (turnover ≤ 1.0, notional/ребаланс ≤ $5M, TTL ≤ 7 дней) даже вручную |
| **Долговечность + tamper-evidence** | состояние — JSON (atomic); каждое событие (GRANT/CONSUME/REVOKE/EXPIRE/REJECT/SUPERSEDE) — в keyed hash-chain аудит Agent-зоны; переживает рестарт |

### Двухшаговая церемония

1. `POST /api/ccea/live_trading/request` — сервер выдаёт **одноразовый
   confirmation_token** + резюме мандата (config hash, брокер, потолки, TTL,
   бюджет) + предупреждение «это разрешит АВТОМАТИЧЕСКУЮ отправку ордеров на
   РЕАЛЬНЫЙ счёт». Оператор читает и осознанно подтверждает.
2. `POST /api/ccea/live_trading/grant` `{request_id, confirmation_token}` —
   оператор возвращает токен; только тогда выдаётся мандат. Токен одноразовый
   (anti-replay: после успеха заявка сжигается); неверный ввод не сжигает заявку
   (можно повторить в пределах 10-мин TTL; brute-force исключён энтропией токена).

Управление: `POST /api/ccea/live_trading/revoke` (`auth_id` или все),
`GET /api/ccea/live_trading/status` (активные мандаты + валидность аудита).

### Как ребаланс использует мандат

При live-брокере `run_rebalance`: **precheck** (наличие мандата + совпадение
хеша конфига + не-исчерпанность бюджета → потолок) → прижать рантайм-лимиты к
потолку → построить план → **финальная проверка** точных чисел плана (turnover,
notional, orders) против потолка/бюджета → отправка с `allow_live=True` через тот
же OMS-стек (firewall/journal/collar/books) на live-движке → **consume** (учёт
использования). Любой провал на любом шаге → `blocked`, ноль ордеров.

Live-ордера идут через `_ensure_live_engine` — настоящий `LiveExecutionEngine`,
привязанный к live-коннектору (тот же firewall/hash-chain журнал/price-collar,
что и paper). Live-брокер исполняет асинхронно: если моментального fill'а нет,
ордер остаётся `submitted` и реконсилируется штатным путём Agent'а — мы **не
выдумываем fill**.

### Проверено вживую (2026-07-15)

Церемония на реальном сервере: request (config_hash, ceiling) → grant с неверным
токеном 409 → grant с верным токеном → активный мандат (binance, TTL 3600s,
max_rebalances=3) → replay того же токена 404 (anti-replay) → status
(active=1, **audit_valid=True**) → revoke → status (active=0). Paper-ребаланс на
том же сервере не затронут (`authorization: null`, sim_paper мандата не требует).

## Что осталось на следующие итерации

- Live fill-reconciliation loop для десктопа (сейчас live-fill приходит через
  штатный путь Agent'а; полноценный polling-цикл в супервизоре — отдельная задача).
- Champion/challenger + автоматический promote (`RIVEN_REQUIRE_PRODUCTION_MODEL`
  уже даёт stage-gate; полное замыкание drift→retrain→promote — P2 №16).

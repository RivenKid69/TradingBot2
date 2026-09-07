# Gas Guard · Fireblocks MPC · EIP-6963 multi-wallet (закрытие «disabled» фич)

> **Статус:** ✅ ЗАКРЫТО 2026-07-16 · три honestly-disabled/NOT-IMPLEMENTED
> фичи Lite Portfolio заменены на РЕАЛЬНЫЕ реализации.
> Тесты: `tests/test_web3_custody.py` (18) + реальный сетевой смоук (живой газ
> ETH/Base/ARB) + live REST. Регрессия lite-audit (64 passed).

Аудит Lite-режима нашёл три фичи, которые честно говорили «не реализовано»/
«недоступно» и были disabled. Закрыли каждую настоящей реализацией (не фейком).

## 1. Gas Guard — реальный on-chain gas oracle + порог (`services/web3/gas_oracle.py`)

Было: слайдер disabled, баннер «Gas Guard — NOT IMPLEMENTED: порог не
сохраняется и не применяется».

Стало:

- **Реальный gas oracle** — `eth_gasPrice` с публичного JSON-RPC узла
  (publicnode.com, без ключей) для ethereum/base/arbitrum/optimism/polygon.
  Вызов делает backend → браузерный CSP/offline не мешает.
- **Долговечный порог** — `GasGuardConfig` (enabled + threshold_gwei + chain)
  в `state/web3_gas_guard.json`.
- **Применение** — `evaluate()` даёт вердикт armed/breached против живого газа;
  `preflight()` — pre-trade gate: `allow=False` → on-chain транзакцию слать
  нельзя. **Fail-closed**: guard включён, но цена газа недоступна → не шлём;
  fail-open только когда guard выключен.
- REST: `GET /api/web3/gas` (живой вердикт), `POST /api/web3/gas_guard`
  (сохранить порог), `GET /api/web3/gas_guard/preflight` (gate).
- UI: слайдер+чекбокс включён, живой газ, ARMED/BREACHED-бейдж, реальная причина.

Честная граница: в Lite нет авто-DEX-исполнения, поэтому guard применяется как
pre-flight перед любой on-chain транзакцией (ручной/агентской), а не глушит
несуществующий поток ордеров — но порог реально сохраняется и реально
оценивается против живого газа. Смоук: threshold 0.1 < живой газ 0.19 Gwei →
breached, preflight `allow=False`.

## 2. Fireblocks MPC — реальный API-клиент (`services/custody/fireblocks_client.py`)

Было: панель disabled, `saveMPCConfiguration` — честный no-op.

Стало: **настоящий клиент Fireblocks REST API** с корректной аутентификацией
(это не заглушка — рабочий connector, нужен institutional-аккаунт Fireblocks):

- **RS256-JWT**, подписанный RSA-ключом пользователя, с полным набором claim'ов
  Fireblocks: `uri` (путь+query), `nonce`, `iat`, `exp` (< ~55с), `sub`=apiKey,
  `bodyHash`=sha256(body). Заголовки `X-API-Key` + `Authorization: Bearer`.
  Подпись проверена против публичного ключа в тесте.
- `test_connection()` → реальный `GET /v1/vault/accounts_paged?limit=1`
  (валидирует креды); `list_vault_accounts` / `get_vault_account`.
- **Безопасность**: приватный RSA-ключ — главный секрет — НИКОГДА не копируется
  в наше хранилище; конфиг ссылается на ключ по ПУТИ (паттерн Fireblocks SDK).
- Честность: без кред → «не настроен»; неверные креды → реальная ошибка API
  (401), не 500.
- REST: `GET /api/custody/fireblocks/status`, `POST …/connect` (валидирует +
  персистит путь при успехе), `GET …/vaults`.
- UI: включённые inputs (API Key, Vault ID, путь к ключу, sandbox-чекбокс) +
  реальная кнопка «Подключить Fireblocks MPC» + честный результат.

## 3. WalletConnect → EIP-6963 мультикошелёк (frontend)

Было: кнопка disabled, «WalletConnect SDK не входит в offline Lite build».

Стало: **реальный EIP-6963 multi-injected-provider discovery** — современный
стандарт, которым дискаверят все инжектированные кошельки (MetaMask, Rabby,
Coinbase, Brave, Frame, Trust…) БЕЗ внешнего SDK, чисто через браузерные
события `eip6963:requestProvider`/`eip6963:announceProvider`. Работает offline.
Кнопка «Подключить кошелёк (EIP-6963)» дискаверит провайдеры, показывает пикер
при нескольких, коннектится через выбранный (тот же eth_requestAccounts +
чтение балансов из сети).

Честная граница: мобильный WalletConnect-relay (QR через wss-relay) требует
онлайн-SDK и вне scope offline-сборки — но реальная потребность «подключить
кошелёк, отличный от дефолтного инжектированного» закрыта настоящим EIP-6963.

## Проверка

- `tests/test_web3_custody.py` (18): gas guard (armed/breached/disabled/
  fail-closed-preflight/persist/REST/валидация), Fireblocks (honest-not-
  configured, реальный RS256-JWT против публичного ключа, парсинг vaults, 401,
  ключ-не-копируется, REST), UI-markup (нет «NOT IMPLEMENTED»/«недоступен»,
  есть реальные функции + EIP-6963).
- Реальный сетевой смоук: живой газ ETH 0.26 / Base 0.006 / ARB 0.02 Gwei;
  breach→preflight-block.
- lite-audit регрессия: обновлён L2-007 (Gas Guard больше не stub) — 64 passed.

## Файлы

- `services/web3/gas_oracle.py`, `services/web3/__init__.py`
- `services/custody/fireblocks_client.py`, `services/custody/__init__.py`
- `app.py` — REST `/api/web3/*`, `/api/custody/fireblocks/*`
- `index.html` — Gas Guard live-панель, Fireblocks connect-форма, EIP-6963 JS
- `tests/test_web3_custody.py`; `.gitignore` — runtime config

## 4. Fireblocks отправка транзакций / withdrawal flow (2026-07-16)

Раньше был реализован только read (vaults/balances) + валидация кред. Теперь —
полный **withdrawal/transfer flow** по реальному Fireblocks API, с профессио-
нальными safeguard'ами для движения реальных денег.

Клиент (`services/custody/fireblocks_client.py`):

- `estimate_fee` → `POST /v1/transactions/estimate_fee` (реальная оценка);
- `create_transaction` → `POST /v1/transactions` (MPC co-signing — на стороне
  Fireblocks по TAP-политике vault'а; мы аутентифицируем вызов JWT);
- `get_transaction` / `get_transaction_by_external_id` — статусы;
- `validate_transfer` (EVM-адрес regex, amount>0 строкой, тип назначения) +
  `build_transfer_payload` (**amount строкой, никогда float**; `externalTxId`
  для идемпотентности; source/destination явными типами).

Оркестрация (REST, `app.py`) — best practices движения денег:

- **Two-step ceremony** (как CCEA live-trading): `POST …/withdraw/preview`
  (валидация + реальная estimate_fee + Gas Guard preflight + одноразовый
  `confirmation_token` + `externalTxId`) → `POST …/withdraw/submit` (тот же
  токен + `confirm=true`).
- **Идемпотентность**: `externalTxId` генерится в preview и несётся в submit —
  Fireblocks дедупит; токен одноразовый (anti-replay: удаляется при submit),
  TTL 10 мин.
- **Gas Guard gate**: для EVM-ассета (известная сеть) на submit — **свежий**
  `preflight` (газ мог подскочить между preview и submit); breach → **409,
  отправка блокируется** (fail-closed). Неизвестный asset → gas guard N/A
  (честно, без выдумок).
- **Долговечный журнал** `state/fireblocks_withdrawals.jsonl`: submitted /
  blocked_gas / failed.
- `GET …/tx/{id}` — реальный статус (SUBMITTED→…→COMPLETED);
  `GET …/withdrawals` — журнал.

UI (Fireblocks-панель): форма отправки (asset/amount/source vault/назначение) с
**двухшаговым** «1. Предпросмотр + оценка» → «2. Подтвердить и отправить»
(нативный confirm перед движением реальных средств), статус Gas Guard, журнал.

Проверка (в `tests/test_web3_custody.py`, всего 26): validate/build (string-
amount, идемпотентность), estimate/create на верных endpoint'ах, полный
two-step flow + **create вызван ровно один раз** + anti-replay токена, Gas Guard
блокирует submit (409), honest-not-configured, UI-markup + явный confirm.

## Остаётся честно вне scope

- Мобильный WalletConnect-relay (нужен онлайн-SDK) — EIP-6963 закрывает
  desktop/extension-кошельки.
- Fireblocks: whitelisted-адреса/contract-calls/webhook-подтверждения статусов —
  реализован core-flow (estimate → two-step submit → status/journal); расширения
  при спросе.

# Ручной ордер-тикет трейдера + частичное закрытие

Закрывает §5.27–28 из [PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md](history/PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md):
трейдеру не хватало ручного ордер-тикета (limit/stop из UI) и частичного закрытия позиции.

**Файлы:** `ccea/desktop_supervisor.py` (`submit_manual_order`, `close_position(quantity)`,
`open_orders`, `cancel_order`) · REST `/api/ccea/order/{submit,cancel}`, `/api/ccea/open_orders`,
`/api/portfolio/close` (partial) · UI-карточки «Ордер-тикет» + «Рабочие ордера» в Lite Portfolio ·
тесты `tests/test_manual_order_ticket.py` (25).

---

## Что появилось

| Возможность | Как |
|---|---|
| **Ордер-тикет** | market / limit / stop / stop-limit, сторона BUY/SELL, количество, Time-in-Force (GTC/DAY/IOC/FOK), reduce-only |
| **Частичное закрытие** | кнопка «½» в таблице позиций (prompt на количество) или `POST /api/portfolio/close {symbol, quantity}` |
| **Рабочие ордера** | панель неисполненных ордеров активного брокера с кнопкой Cancel |
| **Всё через настоящий OMS** | OrderIntent → policy firewall → hash-chain журнал → fill → P&L ledger + blotter + cash GL + MAR surveillance |

## Поток и проверки

`submit_manual_order` строит `OrderIntent` с правильным `intent_type` (market/limit/stop entry
или CLOSE_POSITION для reduce-only) и гонит через тот же движок, что paper_trade/rebalance
(paper — на `_ensure_paper_engine`, live — на `_ensure_live_engine`).

Валидация до отправки:

- некорректная сторона / тип / количество ≤ 0 → отказ;
- limit/stop-limit требуют положительную `limit_price`; stop/stop-limit — `stop_price`;
- недопустимый TIF → отказ;
- **reduce-only**: сторона обязана уменьшать позицию, а количество не может её перевернуть
  (ужимается до размера позиции);
- **live-брокер**: наращивание экспозиции требует мандата авторизации
  ([MODEL_SIGNATURE_AND_REBALANCE.md](MODEL_SIGNATURE_AND_REBALANCE.md) §3); reduce-only
  разрешён без мандата (снижение риска);
- kill switch активен → REST-эндпоинт отвечает 400;
- отправка требует `confirm=true` (UI показывает confirm-диалог) — 409 без него.

Ответ содержит `state`: `filled` (paper-брокер исполняет мгновенно; limit/stop — по своей цене,
это симуляция) или `submitted` (live-брокер исполняет асинхронно — fill реконсилируется штатным
путём Agent'а, выдуманных fill'ов нет).

## REST

- `POST /api/ccea/order/submit` `{symbol, side, order_type, quantity, limit_price?, stop_price?, time_in_force?, reduce_only?, confirm}`
- `POST /api/ccea/order/cancel` `{client_order_id}`
- `GET /api/ccea/open_orders?symbol=…`
- `POST /api/portfolio/close` `{symbol, quantity?}` — `quantity` опционально (None = закрыть целиком)

## Потокобезопасность журнала

Ручные ордера, ребаланс и fill приходят на разных потоках (uvicorn threadpool для sync-эндпоинтов).
`packages/agent/reconciliation/journal.py` теперь открывает SQLite с `check_same_thread=False` и
сериализует запись в append-only hash-chain локом — это латентный баг, который проявился бы и в
реальном многопоточном uvicorn, не только в тестах.

## Проверено вживую (2026-07-16)

На реальном CCEA paper-сервере: confirm-гейт (409 без подтверждения); limit-sell прошёл;
валидация (limit без цены → 400); после установки котировки market-buy 0.2 → filled;
частичное закрытие 0.1 → остаток корректный; reduce-only sell 99 ужат до размера позиции и
закрыл её целиком. Юнит-тесты гоняют настоящий supervisor (SimBroker + реальный OMS), не мок.

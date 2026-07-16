# Cross-Sectional Crypto — Trust Report (РЕАЛЬНЫЕ данные)

> **Это НЕ синтетика.** Бэктест выполнен на реальных исторических дневных барах Binance (public API, без ключей). Цены крипты — point-in-time true (наблюдаемые), поэтому Trust-Report **backtest-safe**.

## Данные

- **Источник:** binance (public, no keys)
- **Класс:** crypto, таймфрейм 1d, pit_quality=true (observed prices)
- **Юниверс (14):** BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, BCHUSDT, AVAXUSDT, DOTUSDT, TRXUSDT, ATOMUSDT
- **Период:** 2023-09-19 → 2026-06-14  (14000 строк панели, 14 символов)

## Методология

- **Движок:** cross-sectional (Panel -> signals -> alpha(IC-weighted) -> risk(crypto_factor, Ledoit-Wolf) -> MVO -> walk-forward)
- **Вариантов перебрано (n_trials):** 8 — pre-registered сетка
- **Выбор лучшего:** best by Deflated Sharpe (multiple-testing-adjusted)
- **Анти-оверфит:** probabilistic_sharpe, deflated_sharpe (n_trials-adjusted)
- **Честность:** pre-registered grid; ALL variants reported; crypto prices are PIT-true.

## Все варианты (pre-registered grid)

| Вариант | Ребаланс | Объектив | Перио­дов | Sharpe | Ann.Ret | MaxDD | Turnover | PSR | **Deflated SR** | Вердикт |
|---|---|---|---|---|---|---|---|---|---|---|
| mom+rev, weekly | 7d | mean_variance | 133 | -2.3726 | -1.0354 | -0.4914 | 0.9244 | 0.0564 | **0.0007** | likely_overfit |
| mom only, weekly | 7d | mean_variance | 133 | -2.4849 | -1.0308 | -0.4616 | 0.535 | 0.0592 | **0.001** | likely_overfit |
| mom only, biweekly | 14d | mean_variance | 66 | -0.4787 | -0.3954 | -0.3586 | 0.6362 | 0.4191 | **0.0477** | likely_overfit |
| mom only, monthly | 30d | mean_variance | 31 | -2.2814 | -1.9419 | -0.3269 | 0.9261 | 0.2448 | **0.0136** | likely_overfit |
| mom-long, biweekly | 14d | mean_variance | 66 | -1.9329 | -1.3922 | -0.3993 | 0.5165 | 0.2013 | **0.0101** | likely_overfit |
| mom-long, monthly | 30d | mean_variance | 31 | -4.1707 | -4.2146 | -0.4111 | 0.7487 | 0.0726 | **0.0007** | likely_overfit |
| mom only, monthly, rp ⭐ | 30d | risk_parity | 31 | 3.1692 | 1.3434 | -0.0712 | 0.081 | 0.7933 | **0.3177** | likely_overfit |
| mom-long, monthly, rp | 30d | risk_parity | 31 | 3.1692 | 1.3434 | -0.0712 | 0.081 | 0.7933 | **0.3177** | likely_overfit |

## Вывод

- **Лучший вариант:** `mom only, monthly, rp` — Sharpe **3.1692**, Deflated Sharpe **0.3177**, вердикт **likely_overfit**.
- ⚠️ **Честный результат:** после поправки на multiple-testing ни один из наивных вариантов не показывает устойчивого edge (Deflated SR < 0.95). Это **подтверждает работоспособность анти-оверфит контура** — он отвергает слабые стратегии, тогда как на synthetic тот же движок давал фиктивный Sharpe > 5. Для реального edge нужны более содержательные сигналы/данные (funding/basis/on-chain), а не подгонка параметров.

---
_Сгенерировано `tools/xs_crypto_real_sweep.py`. Воспроизводимо: `PYTHONPATH=.venv/Lib/site-packages python tools/xs_crypto_real_sweep.py`._
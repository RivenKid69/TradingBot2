# Cross-Sectional Equity — Trust Report (РЕАЛЬНЫЕ данные, честный PIT)

> **Честный point-in-time.** Цены — Yahoo (free). Фундаментал — SEC EDGAR XBRL (free), где `publish_ts = дата подачи отчёта в SEC`. Поэтому value/quality сигналы (E/P, B/P, ROE) бэктестятся **без look-ahead** — это отличает институционал от любителя, который берёт «снимок сейчас» (survivorship + look-ahead).

## Данные

- **Цены:** yahoo (free); **Фундаментал:** SEC EDGAR XBRL (free, PIT=filing date)
- **Юниверс (10):** AAPL, MSFT, GOOGL, AMZN, NVDA, META, JPM, XOM, JNJ, PG
- **Период:** 2023-08-21 → 2026-06-12 (10 символов)
- **Data-Trust вердикт:** **ok**

## PIT-провенанс колонок

| Колонка | pit_quality | Источник |
|---|---|---|
| open | **true** | yahoo |
| high | **true** | yahoo |
| low | **true** | yahoo |
| close | **true** | yahoo |
| volume | **true** | yahoo |
| earnings | **true** | sec_edgar |
| book_value | **true** | sec_edgar |
| fcf | **true** | sec_edgar |
| roe | **true** | sec_edgar |

## Результат бэктеста (market-neutral long-short)

| Метрика | Значение |
|---|---|
| Периодов (недель) | 129 |
| Sharpe (annual) | **1.924** |
| Total return | 0.203 |
| Max drawdown | -0.055 |
| Avg turnover | 0.169 |
| Probabilistic Sharpe | 0.918 |
| **Deflated Sharpe** (n_trials=6) | **0.531** |
| Вердикт | weak |

## Вывод

- Все колонки сигналов имеют **pit_quality=true** (включая EDGAR-фундаментал) → бэктест **backtest-safe**, value/quality честны.
- Платные Sharadar/Compustat подключаются через тот же `fundamentals_path` parquet (drop-in, шире покрытие/история) — но для US equity бесплатный EDGAR уже даёт **подлинный PIT**, так что данные покупать не обязательно.
- Survivorship-free юниверс: `universe.type: index_membership` + `membership_path` (см. `services/index_membership_loader.py`).

---
_Воспроизводимо: `python tools/xs_equity_real_report.py`._

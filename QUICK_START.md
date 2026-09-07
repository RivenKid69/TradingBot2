# Quick Start Guide

> **5-минутный старт** для foundation multi-asset workflows (equities-first; FX/futures/digital assets — optional expansion)
>
> **CCEA Architecture**: Этот проект использует [Cloud-Controlled Execution Architecture](docs/CCEA_OVERVIEW.md) — Cloud (research/simulation) и Agent (live execution) строго разделены.

---

## ⚠️ Важно: CCEA Architecture

Перед началом работы ознакомьтесь с ключевыми принципами архитектуры:

| Зона | Что делает | Что НЕ делает |
|------|------------|---------------|
| **Cloud** | Research, backtesting, simulation, monitoring, lifecycle management | Хранение ключей, генерация ордеров, доступ к trading API |
| **Agent** | Live execution, хранение секретов, риск-контроли, создание ордеров | Работа без согласия пользователя |

**Ключевые гарантии:**

- Cloud **НИКОГДА** не хранит broker API keys
- Cloud **НИКОГДА** не генерирует и не передаёт ордера
- Все торговые операции происходят **ТОЛЬКО** в Agent локально

---

## Содержание

1. [Установка](#установка)
2. [Crypto (BTC/ETH на Binance)](#1-crypto-btceth-на-binance--optional)
3. [US Equities (акции на Alpaca)](#2-us-equities-акции-на-alpaca)
4. [Forex (валюты на OANDA)](#3-forex-валюты-на-oanda)
5. [Futures (фьючерсы)](#4-futures-фьючерсы)
6. [Live Execution (CCEA)](#live-execution-ccea-architecture)
7. [Что дальше?](#что-дальше)

---

## Установка

### Шаг 1: Клонирование и зависимости

```bash
# Клонировать репозиторий
git clone <repo-url>
cd AI-Powered-Quantitative-Research-Platform

# Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или: .venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt
```

### Шаг 2: Проверка установки

```bash
# Запустить диагностику
python scripts/doctor.py

# Ожидаемый вывод:
# ✅ Python 3.12+
# ✅ PyTorch installed
# ✅ Core modules importable
# ✅ Ready for training!
```

---

## 1. Crypto (BTC/ETH на Binance) — optional

> **Стратегия**: Momentum на 4H таймфрейме
> **Особенности**: 24/7 торговля, maker/taker комиссии, Long-only

### Quick Start (3 команды)

```bash
# 1. Скачать тестовые данные (BTCUSDT 2024)
python scripts/prepare_training_data.py --preset crypto_starter

# 2. Запустить бэктест на готовом конфиге
python script_backtest.py --config configs/quickstart/crypto_momentum.yaml

# 3. Обучить модель (10 минут на GPU)
python train_model_multi_patch.py --config configs/quickstart/crypto_momentum.yaml
```

### Результаты бэктеста

После команды 2 вы увидите сводку метрик симуляции/бэктеста (пример: период, волатильность, max drawdown, transaction costs, slippage assumptions). Эти результаты не являются обещанием будущих результатов.

### Файлы результатов

```
artifacts/
├── crypto_momentum_backtest.html    # Интерактивный отчёт
├── crypto_momentum_trades.csv       # Все сделки
└── crypto_momentum_equity.png       # Equity curve
```

### Настройка под себя

```yaml
# configs/quickstart/crypto_momentum.yaml - ключевые параметры

# Символы для торговли
data:
  symbols: ["BTCUSDT", "ETHUSDT"]  # Добавьте другие пары
  timeframe: "4h"                   # 1h, 4h, 1d

# Риск-менеджмент
risk:
  max_position_pct: 0.3    # Макс 30% капитала в позиции
  stop_loss_pct: 0.05      # 5% стоп-лосс

# Обучение
model:
  params:
    n_steps: 2048          # Увеличьте для большего контекста
    learning_rate: 1.0e-4  # Уменьшите для стабильности
```

---

## 2. US Equities (акции на Alpaca)

> **Стратегия**: Mean-reversion на акциях S&P 500
> **Особенности**: NYSE часы (9:30-16:00 ET), без комиссий, регуляторные сборы

### Настройка Alpaca (1 раз)

```bash
# Получите API ключи на https://app.alpaca.markets/
export ALPACA_API_KEY="your_key"
export ALPACA_API_SECRET="your_secret"
```

### Quick Start (3 команды)

```bash
# 1. Скачать данные (SPY, AAPL, MSFT за 2024)
python scripts/download_stock_data.py \
    --symbols SPY AAPL MSFT GOOGL NVDA \
    --start 2023-01-01 --timeframe 4h

# 2. Запустить бэктест
python script_backtest.py --config configs/quickstart/equity_swing.yaml

# 3. Обучить модель
python train_model_multi_patch.py --config configs/quickstart/equity_swing.yaml
```

### Результаты бэктеста

```
═══════════════════════════════════════════════
BACKTEST RESULTS: equity_swing
═══════════════════════════════════════════════
Period:          2023-01-01 to 2024-12-01
Total Return:    +28.7%
Sharpe Ratio:    1.42
Max Drawdown:    -8.9%
Win Rate:        52.1%
Benchmark (SPY): +21.3%
Alpha:           +7.4%
═══════════════════════════════════════════════
```

### Настройка под себя

```yaml
# configs/quickstart/equity_swing.yaml - ключевые параметры

# Символы
data:
  symbols: ["SPY", "AAPL", "MSFT"]  # S&P 500 компоненты
  timeframe: "4h"
  filter_trading_hours: true        # Только NYSE часы

# Бенчмарк
benchmark:
  symbol: "SPY"
  enabled: true

# Risk guards (US Equity специфичные)
risk:
  margin_type: "reg_t"             # Regulation T margin
  pattern_day_trader: false         # PDT правило (если < $25K)
```

---

## 3. Forex (валюты на OANDA)

> **Стратегия**: Carry + Momentum на мажорах
> **Особенности**: 24/5 торговля, spread-only, OTC рынок

### Настройка OANDA (1 раз)

```bash
# Создайте demo аккаунт на https://www.oanda.com/
export OANDA_API_KEY="your_api_key"
export OANDA_ACCOUNT_ID="your_account_id"
```

### Quick Start (3 команды)

```bash
# 1. Скачать данные (EUR/USD, GBP/USD за 2024)
python scripts/download_forex_data.py \
    --pairs EURUSD GBPUSD USDJPY \
    --start 2023-01-01 --timeframe 4h

# 2. Запустить бэктест
python script_backtest.py --config configs/quickstart/forex_carry.yaml

# 3. Обучить модель
python train_model_multi_patch.py --config configs/quickstart/forex_carry.yaml
```

### Результаты бэктеста

```
═══════════════════════════════════════════════
BACKTEST RESULTS: forex_carry
═══════════════════════════════════════════════
Period:          2023-01-01 to 2024-12-01
Total Return:    +18.2%
Sharpe Ratio:    1.31
Max Drawdown:    -6.7%
Win Rate:        48.9%
Pips Profit:     +1,842
Swap Income:     +$2,340
═══════════════════════════════════════════════
```

### Forex-специфичные настройки

```yaml
# configs/quickstart/forex_carry.yaml

# Валютные пары
data:
  pairs: ["EUR_USD", "GBP_USD", "USD_JPY"]
  timeframe: "4h"

# Session-aware execution
execution:
  session_aware: true
  best_sessions: ["london", "new_york"]  # Избегаем Sydney

# Leverage (осторожно!)
leverage:
  max: 30                    # ESMA лимит для EU
  default: 10                # Консервативно

# Swap rates
fees:
  include_swaps: true
  swap_data_source: "oanda"
```

---

## 4. Futures (фьючерсы)

### 4A. Crypto Perpetuals (Binance Futures)

> **Стратегия**: Funding rate arbitrage + Momentum
> **Особенности**: 24/7, funding каждые 8h, до 125x leverage

```bash
# 1. Настроить API
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"

# 2. Скачать данные с funding rates
python scripts/download_funding_history.py --symbols BTCUSDT ETHUSDT --days 365

# 3. Бэктест
python script_backtest.py --config configs/quickstart/crypto_perp.yaml

# 4. Обучение
python train_model_multi_patch.py --config configs/quickstart/crypto_perp.yaml
```

### 4B. CME Futures (через Interactive Brokers)

> **Стратегия**: Equity index momentum (ES, NQ)
> **Особенности**: Globex часы, SPAN margin, daily settlement

```bash
# 1. Запустить TWS/Gateway (порт 7497 для paper)
# См. https://interactivebrokers.github.io/tws-api/

# 2. Скачать данные
python scripts/download_cme_data.py --symbols ES NQ GC --days 365

# 3. Бэктест
python script_backtest.py --config configs/quickstart/cme_index.yaml

# 4. Обучение
python train_model_multi_patch.py --config configs/quickstart/cme_index.yaml
```

### Futures Results

```
═══════════════════════════════════════════════
BACKTEST RESULTS: crypto_perp (BTCUSDT Perpetual)
═══════════════════════════════════════════════
Period:          2024-01-01 to 2024-12-01
Total Return:    +67.4% (10x leverage)
Sharpe Ratio:    2.12
Max Drawdown:    -18.3%
Funding Income:  +$8,234
Liquidation Risk: 0 events
═══════════════════════════════════════════════

═══════════════════════════════════════════════
BACKTEST RESULTS: cme_index (ES E-mini S&P 500)
═══════════════════════════════════════════════
Period:          2024-01-01 to 2024-12-01
Total Return:    +31.2%
Sharpe Ratio:    1.67
Max Drawdown:    -9.1%
Settlement P&L:  +$45,230
SPAN Margin Avg: $12,400
═══════════════════════════════════════════════
```

---

## Сравнительная таблица

| Метрика | Crypto | US Equity | Forex | Crypto Perp | CME Futures |
|---------|--------|-----------|-------|-------------|-------------|
| **Часы торговли** | 24/7 | 9:30-16:00 ET | 24/5 | 24/7 | Globex 23/5 |
| **Комиссии** | 2-4 bps | $0 + regulatory | Spread only | 2-4 bps | $1-2/contract |
| **Leverage** | 1x (spot) | 1-4x | 30-50x | 1-125x | SPAN-based |
| **Min capital** | $100 | $100 | $100 | $100 | $5,000+ |
| **Сложность** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Live Execution (CCEA Architecture)

> **Ключевой принцип**: Live execution происходит **ТОЛЬКО** через customer-controlled Agent. Cloud управляет lifecycle (start/stop/deploy), но не выполняет ордера и не хранит broker credentials.

### Продуктовые режимы

| Режим | Описание | Cloud | Agent |
|-------|----------|-------|-------|
| **Cloud + BYO Agent (B2B)** | Research/sim/monitoring + customer-controlled execution via Agent | Research, Sim, Monitoring | Опционально (для live execution) |
| **Enterprise on-prem/VPC** | Размещение в инфраструктуре клиента | Self-hosted | HSM/KMS, air-gapped |

### Шаг 1: Установка Agent

```bash
# Agent устанавливается на ВАШЕЙ машине (BYO host)
# Credentials остаются ТОЛЬКО у вас - никогда не отправляются в Cloud

# Установка Agent daemon
pip install -e packages/agent/

# Просмотр справки
python -m packages.agent.daemon.agentd --help
```

### Шаг 2: Настройка Local Vault (секреты)

```bash
# Все API ключи хранятся ТОЛЬКО локально в Local Vault
# Cloud никогда не получает доступ к вашим секретам

# Настройка credentials (интерактивно)
python -m packages.agent.vault.setup

# Или через environment variables (ваш выбор)
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

**Важно**:

- Секреты хранятся в OS keychain или зашифрованном файле
- Телеметрия автоматически редактируется перед отправкой в Cloud
- Логи никогда не содержат секретов

### Шаг 3: Запуск Agent

```bash
# Запуск Agent daemon
python -m packages.agent.daemon.agentd --config configs/agent.yaml

# Agent будет:
# - Подключаться к Cloud для получения lifecycle commands
# - Загружать подписанные артефакты стратегий
# - Запрашивать локальное подтверждение для trading-impacting изменений
# - Выполнять торговлю ЛОКАЛЬНО с вашими credentials
```

### Шаг 4: Deploy стратегии через Cloud

```bash
# Cloud отправляет REQUEST_START_RUN (НЕ ордер!)
# Agent показывает diff и запрашивает локальное подтверждение

# Пример deploy через CLI
python -m packages.cloud.cli deploy \
    --strategy my_momentum_v2 \
    --agent agent-001 \
    --mode LIVE
```

### Trading-Impacting vs Non-Impacting

| Категория | Примеры | Требует Approval |
|-----------|---------|------------------|
| **Trading-Impacting** | Новая версия стратегии, PAPER→LIVE, риск-лимиты, universe инструментов | ✅ Да (локально) |
| **Non-Impacting** | Log level, telemetry verbosity, UI параметры | ❌ Нет |

### Safe Defaults (нельзя отключить)

- **Redaction**: ON (обязательно) — секреты не уходят в Cloud
- **Local Approval**: REQUIRED для trading-impacting изменений
- **Artifact Signature**: REQUIRED — Agent проверяет подпись
- **RAW Telemetry**: OFF (только enterprise opt-in)

### Development/Testing Only

```bash
# Для локального тестирования БЕЗ полной CCEA инфраструктуры
# НЕ для production!
python script_live.py --config configs/config_live.yaml --dry-run
```

**Документация CCEA:**

- [CCEA Overview](docs/CCEA_OVERVIEW.md) — полный обзор архитектуры
- [Agent Installation](docs/agent/INSTALLATION.md) — установка Agent
- [Local Vault](docs/agent/LOCAL_VAULT.md) — управление секретами
- [Risk Controls](docs/agent/RISK_CONTROLS.md) — локальные hard caps

---

## Что дальше?

### 1. Изучите Reference Pipelines

Готовые стратегии с полной документацией:

| Pipeline | Asset Class | Стратегия | Документация |
|----------|-------------|-----------|--------------|
| `crypto_intraday_momentum` | Digital assets (optional) | Trend-following на 4H | [Futures overview](docs/futures/overview.md) |
| `equity_swing_reversion` | Equities (MVP) | Mean-reversion на дневках | [Stock trading guide](docs/STOCK_TRADING_GUIDE.md) |
| `forex_carry_momentum` | FX (optional) | Carry + Momentum | [FX integration plan](docs/FOREX_INTEGRATION_PLAN.md) |
| `futures_basis_trading` | Futures (optional) | Basis + funding | [Futures integration plan](docs/FUTURES_INTEGRATION_PLAN.md) |

### 2. Настройте Risk Management

```bash
# Проверьте и настройте риск-параметры
python tools/check_risk_config.py --config your_config.yaml
```

### 3. Запустите Live Execution (CCEA Architecture)

**Production (через Agent):**

```bash
# 1. Установите и настройте Agent (см. раздел "Live Trading (CCEA)")
python -m packages.agent.daemon.agentd --config configs/agent.yaml

# 2. Deploy стратегию через Cloud control plane
# Cloud отправляет REQUEST_START_RUN → Agent запрашивает локальное подтверждение
```

**Development/Testing Only:**

```bash
# Только для локального тестирования БЕЗ CCEA инфраструктуры
# Crypto (Binance Testnet)
python script_live.py --config configs/quickstart/crypto_momentum.yaml --paper --dry-run

# Equity (Alpaca Paper)
python script_live.py --config configs/quickstart/equity_swing.yaml --paper --dry-run

# Forex (OANDA Practice)
python script_live.py --config configs/quickstart/forex_carry.yaml --paper --dry-run
```

> **Важно**: `script_live.py` предназначен только для разработки. Production live trading **ВСЕГДА** через Agent.

### 4. Мониторинг

```bash
# Запустить dashboard
python scripts/dashboard.py --port 8080
# Откройте http://localhost:8080
```

---

## Troubleshooting

### Частые проблемы

| Проблема | Решение |
|----------|---------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `API key invalid` | Проверьте environment variables |
| `No data found` | Запустите `scripts/prepare_training_data.py` |
| `CUDA out of memory` | Уменьшите `batch_size` или используйте CPU |
| `Backtest shows 0 trades` | Проверьте `decision_timing` и `no_trade` windows |

### Получить помощь

```bash
# Полная диагностика
python scripts/doctor.py --verbose

# Проверить конкретный конфиг
python scripts/doctor.py --config your_config.yaml
```

---

## FAQ

**Q: Какой asset class выбрать для начала?**
A: Crypto — самый простой для старта (24/7, нет регуляторных ограничений). US Equity — для классического инвестирования.

**Q: Сколько данных нужно для обучения?**
A: Минимум 1 год для 4H таймфрейма (~2,200 баров). Рекомендуется 2-3 года.

**Q: Можно ли торговать несколько asset classes одновременно?**
A: Да, но каждый asset class требует отдельной модели. Cross-asset allocation — продвинутая тема.

**Q: Какой hardware нужен?**
A: Минимум: 8GB RAM, 4 CPU cores. Рекомендуется: 16GB RAM, NVIDIA GPU (RTX 3060+).

---

**Версия**: 1.0.0 | **Дата**: 2025-12-04

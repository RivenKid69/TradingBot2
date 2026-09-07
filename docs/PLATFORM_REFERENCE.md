# Platform Reference (Multi-Asset Capabilities)

> Consolidated platform reference: architecture, APIs and pipeline behaviour.
> Detailed reference material. Read the section for the subsystem you are working on.
>
> **Authoritative per-domain docs** (prefer these where they exist):
>
> - Futures: `docs/futures/` + `docs/FUTURES_INTEGRATION_PLAN.md`
> - Options: `docs/options/` + `docs/OPTIONS_INTEGRATION_PLAN.md`
> - L3 LOB: `docs/l3_simulator/` + `docs/L3_MIGRATION_GUIDE.md`
> - Forex: `docs/FOREX_INTEGRATION_PLAN.md` + `docs/FOREX_INTEGRATION_QUICK_REF.md`
> - Stocks: `docs/STOCK_TRADING_GUIDE.md`

---

## 📈 Multi-Exchange Support (Phase 2)

### Поддерживаемые биржи

| Биржа | Тип | Статус | Адаптеры |
|-------|-----|--------|----------|
| **Binance** | Digital assets (Spot/Futures; optional) | ✅ Implemented | MarketData, Fee, TradingHours, ExchangeInfo |
| **Alpaca** | US Equities (equities-first) | ✅ Implemented | MarketData (REST + WebSocket), Fee, TradingHours, ExchangeInfo, OrderExecution |
| **Polygon** | US Equities (Data) | ✅ Implemented | MarketData, TradingHours, ExchangeInfo |
| **Yahoo** | Indices/Macro | ✅ Implemented | MarketData (VIX, DXY, Treasury), CorporateActions, Earnings |
| **OANDA** | FX (OTC) | ✅ Implemented | MarketData, Fee, TradingHours, ExchangeInfo, OrderExecution |
| **Interactive Brokers** | Futures (CME examples) | ✅ Implemented | MarketData, OrderExecution, ExchangeInfo (via TWS API) |

### Архитектура адаптеров

```
adapters/
├── base.py           # Абстрактные базовые классы
├── models.py         # Exchange-agnostic модели данных
├── registry.py       # Фабрика + регистрация адаптеров
├── config.py         # Pydantic конфигурация
├── websocket_base.py # Production-grade async WebSocket wrapper
├── binance/          # Binance реализация (crypto)
│   ├── market_data.py
│   ├── fees.py
│   ├── trading_hours.py
│   └── exchange_info.py
├── alpaca/           # Alpaca реализация (stocks)
│   ├── market_data.py  # REST + WebSocket streaming (sync/async)
│   ├── fees.py
│   ├── trading_hours.py
│   ├── exchange_info.py
│   └── order_execution.py
├── polygon/          # Polygon.io реализация (stocks data)
│   ├── market_data.py
│   ├── trading_hours.py
│   └── exchange_info.py
├── yahoo/            # Yahoo Finance реализация (indices/macro)
│   ├── market_data.py      # VIX, DXY, Treasury yields
│   ├── corporate_actions.py # Dividends, splits
│   └── earnings.py          # Earnings calendar
├── oanda/            # OANDA реализация (forex OTC)
│   ├── market_data.py      # FX pairs real-time quotes
│   ├── fees.py             # Spread-based fees (no commission)
│   ├── trading_hours.py    # Sun 5pm - Fri 5pm ET sessions
│   ├── exchange_info.py    # Currency pair specifications
│   └── order_execution.py  # OTC dealer execution
└── ib/               # Interactive Brokers реализация (CME futures)
    ├── market_data.py      # Historical bars, real-time quotes (via TWS API)
    ├── order_execution.py  # Market/limit/bracket orders, margin queries
    └── exchange_info.py    # Contract specifications (ES, NQ, GC, etc.)
```

### Использование

```python
# Через Registry
from adapters.registry import create_market_data_adapter, create_fee_adapter

# Crypto
binance_md = create_market_data_adapter("binance")
binance_fees = create_fee_adapter("binance")

# Stocks
alpaca_md = create_market_data_adapter("alpaca", {
    "api_key": "...",
    "api_secret": "...",
    "feed": "iex",
})

# Indices/VIX (Yahoo Finance)
yahoo_md = create_market_data_adapter("yahoo")
vix_bars = yahoo_md.get_bars("^VIX", "1d", limit=365)
dxy_bars = yahoo_md.get_bars("DX-Y.NYB", "1d", limit=365)

# Alpaca Real-time Streaming (sync)
for bar in alpaca_md.stream_bars(["AAPL", "MSFT"], 60000):
    print(f"Bar: {bar.symbol} @ {bar.close}")

# Alpaca Real-time Streaming (async - for live trading)
async for bar in alpaca_md.stream_bars_async(["AAPL", "MSFT"]):
    await process_bar(bar)

# Через Config
from adapters.config import ExchangeConfig

config = ExchangeConfig.from_yaml("configs/exchange.yaml")
adapter = config.create_market_data_adapter()
```

### Конфигурация

**configs/exchange.yaml** -- главный файл конфигурации биржи:

```yaml
vendor: "alpaca"  # или "binance"
market_type: "EQUITY"  # или "CRYPTO_SPOT"

alpaca:
  api_key: "${ALPACA_API_KEY}"
  api_secret: "${ALPACA_API_SECRET}"
  paper: true
  feed: "iex"
  extended_hours: false
```

**configs/config_live_alpaca.yaml** -- live trading для Alpaca

### Ключевые отличия Crypto vs Stocks

| Аспект | Crypto (Binance) | Stocks (Alpaca) |
|--------|------------------|-----------------|
| **Часы торговли** | 24/7 | NYSE 9:30-16:00 ET + extended |
| **Комиссии** | % от notional (maker/taker) | $0 (+ regulatory на продажу) |
| **Минимальный лот** | По фильтрам биржи | 1 share (или fractional) |
| **Tick size** | Varies by symbol | $0.01 |
| **Short selling** | Через futures | Shortable flag per symbol |
| **Latency** | ~100-500ms | ~50-200ms |

### Команды для Alpaca

```bash
# Получить universe акций
python scripts/fetch_alpaca_universe.py --popular

# Live trading (paper)
python script_live.py --config configs/config_live_alpaca.yaml

# Запустить тесты адаптеров
pytest tests/test_alpaca_adapters.py -v
```

### Требования

```bash
pip install alpaca-py  # Alpaca SDK
```

### Environment Variables

```bash
# Alpaca
ALPACA_API_KEY=...
ALPACA_API_SECRET=...

# Binance (существующие)
BINANCE_API_KEY=...
BINANCE_API_SECRET=...

# Polygon.io (альтернативный data provider)
POLYGON_API_KEY=...
```

---

## 📊 Stock Training & Backtest (Phase 3)

### Обзор

Phase 3 добавляет полную поддержку акций в training и backtest pipeline:

1. **Multi-Asset Data Loader** (`data_loader_multi_asset.py`)
   - Унифицированная загрузка данных для crypto и stocks
   - Фильтрация по trading hours для US equities
   - Поддержка нескольких data vendors (Alpaca, Polygon)

2. **Polygon Data Provider** (`adapters/polygon/`)
   - Альтернативный источник рыночных данных
   - Historical bars и real-time streaming
   - US market holidays и trading hours

3. **WebSocket Wrapper** (`adapters/websocket_base.py`)
   - Production-grade async WebSocket с auto-reconnect
   - Exponential backoff и heartbeat monitoring
   - Rate limiting и message buffering

### Поддерживаемые символы

**Tech Stocks:**

- AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA

**Index ETFs:**

- SPY (S&P 500), QQQ (Nasdaq 100), IWM (Russell 2000)

**Precious Metals ETFs:**

- GLD (SPDR Gold Trust, $60B AUM)
- IAU (iShares Gold Trust)
- SGOL (Aberdeen Physical Gold)
- SLV (iShares Silver Trust)

### Скачивание данных

```bash
# Скачать все поддерживаемые символы (3 года истории)
python scripts/download_stock_data.py \
    --symbols AAPL MSFT GOOGL AMZN NVDA META TSLA SPY QQQ IWM GLD IAU SGOL SLV \
    --start 2020-01-01 --timeframe 1h --resample 4h

# Только precious metals
python scripts/download_stock_data.py \
    --symbols GLD IAU SGOL SLV \
    --start 2020-01-01 --timeframe 1h --resample 4h

# Популярные tech stocks
python scripts/download_stock_data.py --popular --start 2020-01-01
```

Данные сохраняются в: `data/raw_stocks/*.parquet`

### Stock Training Configuration

```yaml
# configs/config_train_stocks.yaml
mode: train
asset_class: equity
data_vendor: alpaca  # или polygon

data:
  timeframe: "4h"
  filter_trading_hours: true
  include_extended_hours: false
  paths:
    - "data/raw_stocks/*.parquet"
    - "data/stocks/*.parquet"

env:
  session:
    calendar: us_equity
    extended_hours: false
```

### Stock Backtest Configuration

```yaml
# configs/config_backtest_stocks.yaml
mode: backtest
asset_class: equity

fees:
  structure: flat
  maker_bps: 0.0
  taker_bps: 0.0
  regulatory:
    enabled: true
    sec_fee_per_million: 27.80
    taf_fee_per_share: 0.000166
```

### Ключевые особенности Stock Trading

| Аспект | Crypto (Binance) | Stocks (Alpaca/Polygon) |
|--------|------------------|-------------------------|
| **Часы торговли** | 24/7 | NYSE 9:30-16:00 ET |
| **Extended hours** | N/A | 4:00-9:30, 16:00-20:00 ET |
| **Комиссии** | % от notional | $0 + regulatory fees |
| **Min trade** | LOT_SIZE filter | 1 share (fractional OK) |
| **Holidays** | Нет | US market holidays |

### Использование Multi-Asset Loader

```python
from data_loader_multi_asset import (
    load_multi_asset_data,
    load_from_adapter,
    AssetClass,
    DataVendor,
)

# Загрузка из файлов
frames, obs_shapes = load_multi_asset_data(
    paths=["data/stocks/*.parquet"],
    asset_class="equity",
    timeframe="4h",
    filter_trading_hours=True,
)

# Загрузка через адаптер
frames, obs_shapes = load_from_adapter(
    vendor="polygon",
    symbols=["AAPL", "MSFT", "GOOGL"],
    timeframe="1h",
    start_date="2024-01-01",
    end_date="2024-12-31",
)
```

### Gold-Specific Features (опционально)

Для улучшения модели на precious metals можно добавить макро-индикаторы:

| Feature | Источник | Корреляция с золотом |
|---------|----------|----------------------|
| DXY (Dollar Index) | Yahoo (`DX-Y.NYB`) | Обратная (сильная) |
| Real Yields (TIPS) | FRED (`DFII10`) | Обратная |
| Gold/Silver Ratio | Расчёт (`GLD/SLV`) | Mean-reverts (60-80) |
| VIX | Yahoo (`^VIX`) | Положительная (fear) |

```bash
# Скачать VIX для fear indicator
python scripts/download_stock_data.py --symbols ^VIX --start 2020-01-01
```

### Требования

```bash
pip install polygon-api-client  # Polygon.io
pip install alpaca-py           # Alpaca
```

---

## 🔄 Execution Providers (Phase 4)

### Обзор

Phase 4 добавляет абстракцию execution providers для унифицированной симуляции исполнения crypto и акций.

**Файл**: `execution_providers.py` (~1800 строк)

### Архитектура

```
┌──────────────────────────────────────────────────────────────┐
│                    Protocols (Interfaces)                     │
├──────────────────┬──────────────────┬────────────────────────┤
│ SlippageProvider │  FillProvider    │     FeeProvider        │
└────────┬─────────┴────────┬─────────┴──────────┬─────────────┘
         │                  │                    │
┌────────▼─────────────────▼───────────────────▼───────────────┐
│                  L2 Implementations (Production)              │
├─────────────────────┬──────────────────┬─────────────────────┤
│StatisticalSlippage  │ OHLCVFillProvider│ CryptoFeeProvider   │
│ (√participation)    │ (bar-based fills)│ EquityFeeProvider   │
└─────────────────────┴──────────────────┴─────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────┐
│               L2ExecutionProvider (Combined)                  │
│    - Auto-selects crypto/equity defaults                     │
│    - Pre-trade cost estimation                               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│            L2+ CryptoParametricSlippageProvider               │
│    - 6 slippage factors (research-backed)                    │
│    - Volatility regime detection                             │
│    - Adaptive impact coefficient                             │
│    - Whale detection & TWAP adjustment                       │
└──────────────────────────────────────────────────────────────┘
```

### Уровни точности (Fidelity Levels)

| Level | Модель | Статус | Описание |
|-------|--------|--------|----------|
| **L1** | Constant | N/A | Фиксированный spread/fee (не реализован) |
| **L2** | Statistical | ✅ Production | √participation impact (Almgren-Chriss) |
| **L2+** | Parametric TCA | ✅ Production | 6-factor crypto model (see below) |
| **L3** | LOB | ✅ Production | Full order book simulation |

### Ключевые классы

| Класс | Назначение |
|-------|------------|
| `MarketState` | Snapshot рынка (bid/ask/spread/adv) |
| `Order` | Ордер для исполнения |
| `Fill` | Результат исполнения |
| `BarData` | OHLCV данные бара |
| `StatisticalSlippageProvider` | √participation slippage модель |
| `CryptoParametricSlippageProvider` | L2+ Smart parametric TCA (6 факторов) |
| `CryptoParametricConfig` | Конфигурация для parametric TCA |
| `VolatilityRegime` | Enum: LOW/NORMAL/HIGH волатильность |
| `OHLCVFillProvider` | Fill logic на основе bar range |
| `CryptoFeeProvider` | Maker/taker комиссии (Binance) |
| `EquityFeeProvider` | Regulatory fees (SEC/TAF) |
| `L2ExecutionProvider` | Комбинированный провайдер |

### Различия Crypto vs Equity

| Параметр | Crypto | Equity |
|----------|--------|--------|
| Default spread | 5 bps | 2 bps |
| Impact coef | 0.1 | 0.05 |
| Fee structure | Maker 2bps / Taker 4bps | $0 + SEC/TAF on sells |
| SEC fee | N/A | ~$0.0000278/$ |
| TAF fee | N/A | ~$0.000166/share (max $8.30) |

### Использование

```python
from execution_providers import (
    create_execution_provider,
    AssetClass,
    Order,
    MarketState,
    BarData,
)

# Создание провайдера для акций
provider = create_execution_provider(AssetClass.EQUITY)

# Исполнение ордера
fill = provider.execute(
    Order(symbol="AAPL", side="BUY", qty=100, order_type="MARKET"),
    MarketState(timestamp=now, bid=150.0, ask=150.02, adv=10_000_000),
    BarData(open=150.0, high=151.0, low=149.0, close=150.5, volume=100000),
)

# Результат
print(f"Price: {fill.price}, Fee: {fill.fee}, Slippage: {fill.slippage_bps} bps")
```

### Factory Functions

```python
# Создание отдельных провайдеров
slippage = create_slippage_provider("L2", AssetClass.EQUITY)
fees = create_fee_provider(AssetClass.CRYPTO)
fill = create_fill_provider("L2", AssetClass.CRYPTO, slippage, fees)

# Комбинированный провайдер
provider = create_execution_provider(AssetClass.EQUITY, level="L2")
```

### Backward Compatibility

```python
from execution_providers import wrap_legacy_slippage_config, wrap_legacy_fees_model

# Обёртки для существующих конфигов
slippage = wrap_legacy_slippage_config(existing_slippage_config)
fees = wrap_legacy_fees_model(existing_fees_model)
```

### Slippage Model (Almgren-Chriss)

```
slippage_bps = half_spread + k * sqrt(participation) * vol_scale * 10000
```

Где:

- `half_spread` -- половина спреда из MarketState
- `k` -- impact coefficient (0.1 для crypto, 0.05 для equity)
- `participation` -- order_notional / ADV
- `vol_scale` -- volatility adjustment factor

### Limit Order Fill Logic

```
1. Check immediate execution (crossing spread):
   - BUY LIMIT >= ask → TAKER fill at ask
   - SELL LIMIT <= bid → TAKER fill at bid

2. Check passive fill (bar range):
   - BUY LIMIT: fills if bar_low <= limit_price → MAKER
   - SELL LIMIT: fills if bar_high >= limit_price → MAKER
```

### Тестирование

```bash
# Все тесты execution providers
pytest tests/test_execution_providers.py -v

# Интеграционные тесты
pytest tests/test_execution_providers.py::TestIntegration -v
```

**Покрытие**: 95 тестов (at documentation time; verify via CI) + 84 теста parametric TCA

### Референсы

- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Kyle (1985): "Continuous Auctions and Insider Trading"
- Cont (2001): "Empirical Properties of Asset Returns"
- Cont, Kukanov, Stoikov (2014): "The Price Impact of Order Book Events"
- Cartea, Jaimungal, Penalva (2015): "Algorithmic and HF Trading"

---

## 📊 Crypto Parametric TCA (L2+)

### Обзор

Smart parametric Transaction Cost Analysis model для криптовалютных рынков. Расширяет базовую √participation модель (Almgren-Chriss) с 6 crypto-специфичными факторами.

**Статус**: ✅ Tested and operational | **Тесты**: 84 (at documentation time; verify via CI)

### Формула Total Slippage

```
slippage = half_spread
    × (1 + k × √participation)      # Almgren-Chriss impact
    × vol_regime_mult               # Volatility regime (Cont 2001)
    × (1 + imbalance_penalty)       # Order book imbalance (Cont et al. 2014)
    × funding_stress                # Funding rate stress (perp-specific)
    × (1 / tod_factor)              # Time-of-day liquidity curve
    × correlation_decay             # BTC correlation decay (altcoins)
    × asymmetric_adjustment         # Panic selling premium
```

### 6 Slippage Factors

| Factor | Формула | Референс |
|--------|---------|----------|
| **√Participation** | `k × √(Q/ADV)` | Almgren-Chriss (2001) |
| **Volatility Regime** | Percentile-based LOW/NORMAL/HIGH | Cont (2001) |
| **Order Book Imbalance** | `(bid - ask) / (bid + ask)` | Cont et al. (2014) |
| **Funding Rate Stress** | `1 + \|funding\| × sensitivity` | Empirical (Binance) |
| **Time-of-Day** | 24-hour liquidity curve (Asia/EU/US) | Binance research |
| **BTC Correlation Decay** | `1 + (1 - corr) × decay_factor` | Empirical (altcoins) |

### Smart Features

| Feature | Описание |
|---------|----------|
| **Regime Detection** | Автоматическое определение LOW/NORMAL/HIGH volatility |
| **Adaptive Impact** | Коэффициент k адаптируется по trailing fill quality |
| **Asymmetric Slippage** | Продажи в downtrend стоят дороже (panic liquidity) |
| **Whale Detection** | Большие ордеры (Q/ADV > 1%) получают TWAP-adjusted model |

### Использование

```python
from execution_providers import (
    CryptoParametricSlippageProvider,
    CryptoParametricConfig,
    Order,
    MarketState,
)

# 1. Базовое использование (defaults)
provider = CryptoParametricSlippageProvider()

# 2. С кастомной конфигурацией
config = CryptoParametricConfig(
    impact_coef_base=0.12,
    spread_bps=6.0,
    whale_threshold=0.02,
)
provider = CryptoParametricSlippageProvider(config=config)

# 3. Из профиля
provider = CryptoParametricSlippageProvider.from_profile("altcoin")
# Профили: "default", "conservative", "aggressive", "altcoin", "stablecoin"

# 4. Вычисление slippage
slippage_bps = provider.compute_slippage_bps(
    order=Order("ETHUSDT", "BUY", 10.0, "MARKET"),
    market=MarketState(timestamp=0, bid=2000.0, ask=2001.0, adv=50_000_000),
    participation_ratio=0.005,
    funding_rate=0.0003,       # Slightly positive
    btc_correlation=0.85,      # High correlation
    hour_utc=14,               # EU session
    recent_returns=[-0.01, 0.005, -0.008],  # For regime detection
)

# 5. Pre-trade cost estimation
estimate = provider.estimate_impact_cost(
    notional=1_000_000,
    adv=500_000_000,
    side="BUY",
    hour_utc=16,
)
print(f"Impact: {estimate['impact_bps']:.2f} bps")
print(f"Cost: ${estimate['impact_cost']:.2f}")
print(f"Recommendation: {estimate['recommendation']}")
```

### Конфигурация (CryptoParametricConfig)

| Параметр | Default | Описание |
|----------|---------|----------|
| `impact_coef_base` | 0.10 | Base k coefficient |
| `impact_coef_range` | (0.05, 0.15) | Adaptive k bounds |
| `spread_bps` | 5.0 | Default spread (if market unavailable) |
| `vol_regime_multipliers` | {low: 0.8, normal: 1.0, high: 1.5} | Regime scaling |
| `vol_lookback_periods` | 20 | Periods for regime detection |
| `vol_regime_thresholds` | (25.0, 75.0) | Percentiles for LOW/HIGH |
| `imbalance_penalty_max` | 0.3 | Max imbalance penalty (30%) |
| `funding_stress_sensitivity` | 10.0 | Funding rate multiplier |
| `tod_curve` | {0-23: factors} | 24-hour liquidity curve |
| `btc_correlation_decay_factor` | 0.5 | Altcoin decay factor |
| `whale_threshold` | 0.01 | 1% ADV = whale |
| `whale_twap_adjustment` | 0.7 | TWAP adjustment |
| `asymmetric_sell_premium` | 0.2 | 20% panic selling premium |
| `downtrend_threshold` | -0.02 | -2% = downtrend |
| `min_slippage_bps` | 1.0 | Floor |
| `max_slippage_bps` | 500.0 | Cap |

### Профили

| Профиль | impact_coef | spread_bps | Применение |
|---------|-------------|------------|------------|
| `default` | 0.10 | 5.0 | BTC/ETH majors |
| `conservative` | 0.12 | 6.0 | Safer estimates |
| `aggressive` | 0.08 | 4.0 | Tighter estimates |
| `altcoin` | 0.15 | 10.0 | Low-cap altcoins |
| `stablecoin` | 0.05 | 1.0 | USDT/USDC pairs |

### Time-of-Day Curve (Default)

| Session | Часы (UTC) | Factor | Описание |
|---------|------------|--------|----------|
| Asia | 00:00-08:00 | 0.70-0.90 | Lower liquidity |
| EU | 08:00-16:00 | 0.95-1.10 | Increasing liquidity |
| US/EU overlap | 14:00-18:00 | 1.10-1.15 | Peak liquidity |
| US | 18:00-24:00 | 0.85-1.05 | Declining liquidity |

### Adaptive Learning

```python
# После каждого fill обновляем модель
predicted = provider.compute_slippage_bps(order, market, participation)
# ... execution happens ...
actual = (fill_price - expected_price) / expected_price * 10000

provider.update_fill_quality(predicted, actual)
# k coefficient автоматически адаптируется
```

### Тестирование

```bash
# Все тесты parametric TCA
pytest tests/test_crypto_parametric_tca.py -v

# По категориям
pytest tests/test_crypto_parametric_tca.py::TestVolatilityRegime -v
pytest tests/test_crypto_parametric_tca.py::TestWhaleDetection -v
pytest tests/test_crypto_parametric_tca.py::TestAdaptiveImpact -v
```

**Покрытие**: 84 теста (at documentation time; verify via CI)

---

## 📈 Equity Parametric TCA (L2+)

### Обзор

Smart parametric Transaction Cost Analysis model для US equities. Расширяет базовую √participation модель (Almgren-Chriss) с equity-специфичными факторами.

**Статус**: ✅ Tested and operational | **Тесты**: 86 (at documentation time; verify via CI)

### Формула Total Slippage

```
slippage = half_spread
    × (1 + k × √participation)      # Almgren-Chriss impact
    × volatility_regime_mult        # Volatility regime (Hasbrouck 2007)
    × market_cap_mult               # Market cap tier (Kissell 2013)
    × (1 + beta_stress)             # Systematic risk adjustment
    × intraday_factor               # U-curve liquidity (ITG 2012)
    × auction_factor                # Opening/closing auction proximity
    × (1 + short_penalty)           # Short squeeze risk
    × event_mult                    # Earnings/news events
    × (1 + sector_penalty)          # Sector rotation
    × imbalance_factor              # Order book imbalance
```

### 9 Slippage Factors

| Factor | Формула | Референс |
|--------|---------|----------|
| **√Participation** | `k × √(Q/ADV)`, k ∈ [0.03, 0.08] | Almgren-Chriss (2001) |
| **Market Cap Tier** | mega=0.7, large=1.0, mid=1.3, small=1.8, micro=2.5 | Kissell (2013) |
| **Intraday U-Curve** | open=1.5 → midday=1.0 → close=1.3 | ITG (2012) |
| **Auction Proximity** | `1 + 0.3 × exp(-minutes/10)` | NYSE/NASDAQ mechanics |
| **Beta Stress** | `1 + \|β-1\| × SPY_move × 0.1` | Systematic risk |
| **Short Interest** | `log1p(ratio/threshold) × max_penalty` | GME-style squeeze |
| **Events** | Earnings=2.5×, News=1.5× | Event-driven volatility |
| **Sector Rotation** | Penalty when sector ETF down >1% | Cross-asset signal |
| **Volatility Regime** | LOW=0.85, NORMAL=1.0, HIGH=1.4 | Hasbrouck (2007) |

### Smart Features

| Feature | Описание |
|---------|----------|
| **Market Cap Auto-Detection** | Классификация MEGA/LARGE/MID/SMALL/MICRO по market cap |
| **Trading Session Detection** | PRE_MARKET, OPEN_AUCTION, REGULAR, CLOSE_AUCTION, AFTER_HOURS, CLOSED |
| **Adaptive Impact** | Коэффициент k адаптируется по trailing fill quality |
| **Auction Detector** | Экспоненциальный decay вблизи 9:30/16:00 ET |
| **Earnings Calendar** | Автоматическое определение T-1 to T+1 earnings window |
| **Cross-Asset Signal** | SPY volatility spike → все акции получают penalty |
| **Sector Rotation** | XLF/XLK/XLV down >1% → соответствующие акции получают penalty |

### Market Cap Tiers

| Tier | Threshold | Multiplier | Примеры |
|------|-----------|------------|---------|
| **MEGA** | >$200B | 0.7 | AAPL, MSFT, GOOGL |
| **LARGE** | $10B-$200B | 1.0 | Most S&P 500 |
| **MID** | $2B-$10B | 1.3 | Mid-cap stocks |
| **SMALL** | $300M-$2B | 1.8 | Regional banks |
| **MICRO** | <$300M | 2.5 | Penny stocks |

### Intraday U-Curve (US Eastern Time)

| Session | Часы (ET) | Factor | Описание |
|---------|-----------|--------|----------|
| Pre-market | 4:00-9:30 | 2.0-2.5 | Very low liquidity |
| Open auction | 9:30-10:00 | ~1.5 | High volume, wide spreads |
| Morning | 10:00-12:00 | 1.1-1.2 | Improving liquidity |
| Midday | 12:00-14:00 | **1.0** | Peak liquidity (best execution) |
| Afternoon | 14:00-15:00 | 1.05-1.1 | Still good |
| Pre-close | 15:00-16:00 | ~1.3 | Rising activity |
| After-hours | 16:00-20:00 | 2.0-2.5 | Low liquidity |

### Использование

```python
from execution_providers import (
    EquityParametricSlippageProvider,
    EquityParametricConfig,
    MarketCapTier,
    TradingSession,
    Order,
    MarketState,
    AssetClass,
)

# 1. Базовое использование (defaults)
provider = EquityParametricSlippageProvider()

# 2. С кастомной конфигурацией
config = EquityParametricConfig(
    impact_coef_base=0.06,
    spread_bps=2.5,
    market_cap_multipliers={"mega": 0.6, "large": 1.0, ...},
)
provider = EquityParametricSlippageProvider(config=config)

# 3. Из профиля
provider = EquityParametricSlippageProvider.from_profile("large_cap")
# Профили: "default", "conservative", "aggressive", "slow_internet", "large_cap", "small_cap"

# 4. Вычисление slippage с полным набором параметров
slippage_bps = provider.compute_slippage_bps(
    order=Order("AAPL", "BUY", 1000, "MARKET", asset_class=AssetClass.EQUITY),
    market=MarketState(timestamp=0, bid=175.0, ask=175.02, adv=80_000_000),
    participation_ratio=0.002,
    market_cap=2.8e12,           # $2.8T (MEGA cap)
    beta=1.2,                    # Stock beta vs SPY
    time_et=12,                  # 12:00 ET (midday - best liquidity)
    spy_return_today=-0.015,     # SPY down 1.5%
    short_interest_ratio=3.0,    # 3 days to cover
    has_earnings_soon=False,
    sector="technology",
    sector_etf_return=-0.02,     # XLK down 2%
)

# 5. Pre-trade cost estimation с рекомендациями
estimate = provider.estimate_impact_cost(
    notional=1_000_000,
    adv=50_000_000,
    market_cap=50e9,
    beta=1.3,
    time_et=14,
    has_earnings_soon=True,
)
print(f"Impact: {estimate['impact_bps']:.2f} bps")
print(f"Cost: ${estimate['impact_cost']:.2f}")
print(f"Market Cap Tier: {estimate['market_cap_tier']}")
print(f"Trading Session: {estimate['trading_session']}")
print(f"Recommendation: {estimate['recommendation']}")
```

### Конфигурация (EquityParametricConfig)

| Параметр | Default | Описание |
|----------|---------|----------|
| `impact_coef_base` | 0.05 | Base k coefficient (lower than crypto!) |
| `impact_coef_range` | (0.03, 0.08) | Adaptive k bounds |
| `spread_bps` | 2.0 | Default spread (tighter than crypto) |
| `market_cap_multipliers` | {mega: 0.7, ..., micro: 2.5} | Tier multipliers |
| `market_cap_thresholds` | {mega: 200e9, large: 10e9, ...} | USD thresholds |
| `intraday_curve` | {hour: factor} | 24-hour liquidity curve (ET) |
| `auction_decay_minutes` | 10.0 | Exponential decay parameter |
| `auction_premium` | 0.3 | Max 30% auction premium |
| `vol_regime_multipliers` | {low: 0.85, normal: 1.0, high: 1.4} | Volatility scaling |
| `beta_stress_sensitivity` | 0.1 | 10% per unit beta deviation × SPY move |
| `short_interest_max_penalty` | 0.3 | Max 30% short squeeze penalty |
| `short_interest_threshold` | 5.0 | 5 days to cover threshold |
| `earnings_event_multiplier` | 2.5 | 2.5× during earnings |
| `news_event_multiplier` | 1.5 | 1.5× during news |
| `sector_penalty_threshold` | -0.01 | -1% sector ETF return triggers penalty |
| `sector_penalty_max` | 0.15 | Max 15% sector penalty |
| `min_slippage_bps` | 0.5 | Floor (lower than crypto) |
| `max_slippage_bps` | 200.0 | Cap (lower than crypto) |

### Профили

| Профиль | impact_coef | spread_bps | min_bps | Применение |
|---------|-------------|------------|---------|------------|
| `default` | 0.05 | 2.0 | 0.5 | Standard institutional |
| `conservative` | 0.07 | 3.0 | 1.0 | Safer estimates |
| `aggressive` | 0.04 | 1.5 | 0.3 | Tighter estimates |
| `slow_internet` | 0.06 | 4.0 | 1.5 | Wider spreads / slower fills |
| `large_cap` | 0.04 | 1.5 | 0.3 | MEGA/LARGE caps |
| `small_cap` | 0.08 | 5.0 | 2.0 | SMALL/MICRO caps |

### Сравнение Crypto vs Equity TCA

| Параметр | Crypto | Equity |
|----------|--------|--------|
| Base k coefficient | 0.10 | 0.05 |
| Default spread | 5.0 bps | 2.0 bps |
| Max slippage | 500 bps | 200 bps |
| Time-of-day | 24h UTC curve | US Eastern U-curve |
| Special factors | Funding rate, BTC correlation | Beta stress, earnings, sector rotation |
| Market structure | 24/7 trading | 9:30-16:00 ET + extended |

### Тестирование

```bash
# Все тесты equity parametric TCA
pytest tests/test_equity_parametric_tca.py -v

# По категориям
pytest tests/test_equity_parametric_tca.py::TestMarketCapTierClassification -v
pytest tests/test_equity_parametric_tca.py::TestIntradayUCurve -v
pytest tests/test_equity_parametric_tca.py::TestAuctionProximityFactor -v
pytest tests/test_equity_parametric_tca.py::TestBetaStress -v
pytest tests/test_equity_parametric_tca.py::TestShortSqueeze -v
pytest tests/test_equity_parametric_tca.py::TestEarningsWindow -v
pytest tests/test_equity_parametric_tca.py::TestSectorRotation -v
pytest tests/test_equity_parametric_tca.py::TestL2Integration -v
```

**Покрытие**: 86 тестов (at documentation time; verify via CI)

### Референсы

- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Kissell & Glantz (2013): "Optimal Trading Strategies"
- Hasbrouck (2007): "Empirical Market Microstructure"
- Kyle (1985): "Continuous Auctions and Insider Trading"
- ITG (2012): "Global Cost Review" -- intraday patterns
- Cont, Kukanov, Stoikov (2014): "Price Impact of Order Book Events"
- Pagano & Schwartz (2003): "Opening and Closing Auctions"

---

## 📊 Stock Features & Risk Management (Phase 5)

### Обзор

Phase 5 добавляет stock-специфичные features и risk guards, параллельно crypto Fear & Greed индексу.

**Файлы**:

- `stock_features.py` -- VIX integration, market regime, relative strength
- `services/stock_risk_guards.py` -- Margin, short sale, corporate actions guards
- `services/universe_stocks.py` -- Stock universe management with TTL caching

### Stock Features (`stock_features.py`)

| Feature | Описание | Источник |
|---------|----------|----------|
| **VIX Value** | Market volatility (fear gauge) | Yahoo `^VIX` |
| **VIX Regime** | LOW (<12), NORMAL (12-20), ELEVATED (20-30), EXTREME (>30) | CBOE thresholds |
| **Market Regime** | BULL/SIDEWAYS/BEAR based on SPY + VIX | SMA crossover + VIX |
| **RS vs SPY (20d)** | 20-day relative strength vs S&P 500 | Levy (1967) |
| **RS vs SPY (50d)** | 50-day relative strength vs S&P 500 | Moskowitz et al. (2012) |
| **RS vs QQQ (20d)** | 20-day relative strength vs Nasdaq 100 | Momentum proxy |
| **Sector Momentum** | Sector rotation signal | XLK, XLF, XLV ETF returns |

**Использование**:

```python
from stock_features import (
    StockFeatures,
    BenchmarkData,
    calculate_vix_regime,
    calculate_market_regime,
    calculate_relative_strength,
    VIXRegime,
    MarketRegime,
)

# Calculate VIX regime
vix_normalized, regime = calculate_vix_regime(vix_value=25.0)
# regime = VIXRegime.ELEVATED

# Calculate market regime
market_regime = calculate_market_regime(
    spy_prices=spy_close_list,
    vix_value=25.0,
)
# market_regime = MarketRegime.SIDEWAYS

# Calculate relative strength
rs_20d = calculate_relative_strength(
    stock_prices=stock_close_list,
    benchmark_prices=spy_close_list,
    window=20,
)
```

### Stock Risk Guards (`services/stock_risk_guards.py`)

| Guard | Правило | Описание |
|-------|---------|----------|
| **MarginGuard** | Reg T | 50% initial, 25% maintenance margin |
| **ShortSaleGuard** | Rule 201 | Uptick rule при -10% drop |
| **CorporateActionsHandler** | SEC | Dividends, splits, ex-dates |

**Margin Call Types**:

- `FEDERAL` -- Below Reg T initial margin (new positions)
- `MAINTENANCE` -- Below 25% maintenance margin
- `HOUSE` -- Broker's stricter requirements

**Short Sale Restrictions**:

- `UPTICK_RULE` -- Rule 201 (short only on uptick)
- `HTB` -- Hard-to-borrow (may not be available)
- `RESTRICTED` -- Exchange restricted
- `NOT_SHORTABLE` -- Cannot be shorted

**Использование**:

```python
from services.stock_risk_guards import (
    MarginGuard,
    ShortSaleGuard,
    MarginCallType,
    ShortSaleRestriction,
)

# Margin check
margin_guard = MarginGuard()
result = margin_guard.check_margin_requirement(
    position_value=100000,
    account_equity=60000,
    is_new_position=True,
)
# result.margin_call_type = MarginCallType.NONE if OK

# Short sale check
short_guard = ShortSaleGuard()
restriction = short_guard.check_short_restriction(
    symbol="GME",
    price_change_pct=-0.12,  # -12% drop
)
# restriction = ShortSaleRestriction.UPTICK_RULE
```

### Benchmark Temporal Alignment (Fix 2025-11-29)

**Проблема**: VIX/SPY/QQQ данные использовали positional index вместо timestamp merge → look-ahead bias.

**Решение**: `pd.merge_asof(direction="backward")` для корректного temporal alignment.

```python
# stock_features.py:_align_benchmark_by_timestamp()
aligned = pd.merge_asof(
    stock_df,
    benchmark_df,
    on="timestamp",
    direction="backward",  # Use last available benchmark value
    suffixes=("", "_benchmark"),
)
```

### Тестирование

```bash
# Stock features tests
pytest tests/test_stock_features.py -v

# Stock risk guards tests
pytest tests/test_stock_risk_guards.py -v

# Benchmark alignment tests
pytest tests/test_benchmark_temporal_alignment.py -v
```

### Референсы

- CBOE VIX White Paper (2003): VIX as fear gauge
- Lo, A.W. (2004): "The Adaptive Markets Hypothesis"
- Moskowitz, T.J. et al. (2012): "Time series momentum"
- Levy, R. (1967): "Relative Strength as a Criterion for Investment Selection"
- Reg T (Federal Reserve): Initial/maintenance margin requirements
- SEC Rule 201: Short sale circuit breaker

---

## 🔴 Live Trading Improvements (Phase 9)

### Обзор

Phase 9 добавляет полную поддержку live trading для акций через Alpaca:

1. **Unified Live Script** (`script_live.py`)
   - Единый entry point для crypto и stocks
   - Автоматическое определение asset class
   - CLI аргументы для переключения режимов

2. **Position Synchronization** (`services/position_sync.py`)
   - Синхронизация локального состояния с биржей
   - Background polling с настраиваемым интервалом
   - Автоматическое обнаружение и обработка расхождений

3. **Advanced Order Management** (`adapters/alpaca/order_execution.py`)
   - Bracket orders (take-profit + stop-loss)
   - OCO (One-Cancels-Other) orders
   - Order replacement (cancel + new)
   - Order history и wait-for-fill

4. **Extended Hours Trading** (`services/session_router.py`)
   - Session detection (pre-market, regular, after-hours)
   - Session-aware order routing
   - Spread adjustment для extended hours

### Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                      script_live.py                              │
│  - CLI: --asset-class, --extended-hours, --paper/--live         │
│  - Auto-detection: detect_asset_class()                         │
│  - Defaults: apply_asset_class_defaults()                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│  Crypto (Binance)   │       │  Equity (Alpaca)    │
│  - 24/7 trading     │       │  - Market hours     │
│  - GTC orders       │       │  - DAY orders       │
│  - 5 bps slippage   │       │  - 2 bps slippage   │
└─────────────────────┘       └─────────┬───────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────┐
          ▼                             ▼                         ▼
┌─────────────────┐         ┌─────────────────┐       ┌─────────────────┐
│ Position Sync   │         │ Order Execution │       │ Session Router  │
│ - Reconcile     │         │ - Bracket orders│       │ - Pre-market    │
│ - Background    │         │ - OCO orders    │       │ - Regular       │
│ - Callbacks     │         │ - Replace order │       │ - After-hours   │
└─────────────────┘         └─────────────────┘       └─────────────────┘
```

### Asset Class Detection

```python
# Приоритет определения asset class:
# 1. Explicit: --asset-class equity
# 2. Vendor: vendor=alpaca → equity
# 3. Market type: market_type=EQUITY → equity
# 4. Default: crypto (backward compatible)

def detect_asset_class(cfg_dict: Dict[str, Any]) -> str:
    # Priority 1: Explicit
    if "asset_class" in cfg_dict:
        return cfg_dict["asset_class"]

    # Priority 2: Vendor mapping
    vendor = cfg_dict.get("vendor", "").lower()
    if vendor in ("alpaca", "polygon"):
        return "equity"
    if vendor == "binance":
        return "crypto"

    # Priority 3: Market type
    market_type = cfg_dict.get("market_type", "").upper()
    if market_type in ("EQUITY", "STOCK"):
        return "equity"

    # Default: crypto
    return "crypto"
```

### Asset Class Defaults

| Параметр | Crypto | Equity |
|----------|--------|--------|
| `slippage_bps` | 5.0 | 2.0 |
| `limit_offset_bps` | 10.0 | 5.0 |
| `tif` | GTC | DAY |
| `extended_hours` | False | False |
| `default_vendor` | binance | alpaca |

### Position Synchronization

```python
from services.position_sync import (
    PositionSynchronizer,
    SyncConfig,
    reconcile_alpaca_state,
)

# Конфигурация
config = SyncConfig(
    sync_interval_sec=30.0,       # Интервал polling
    position_tolerance=0.01,      # 1% tolerance
    auto_reconcile=True,          # Автоматическая коррекция
    max_reconcile_qty=1000.0,     # Максимальный объём коррекции
)

# Создание synchronizer
sync = PositionSynchronizer(
    position_provider=alpaca_adapter,
    local_state_getter=get_local_positions,
    config=config,
    on_discrepancy=handle_discrepancy,
    on_sync_complete=on_sync,
)

# Запуск background sync
sync.start_background_sync()
```

### Bracket Orders (Alpaca)

```python
from adapters.alpaca.order_execution import (
    AlpacaOrderExecutionAdapter,
    BracketOrderConfig,
)

adapter = AlpacaOrderExecutionAdapter(api_key, api_secret, paper=True)

# Bracket order: entry + take-profit + stop-loss
config = BracketOrderConfig(
    symbol="AAPL",
    side=Side.BUY,
    qty=100,
    entry_price=150.0,           # Optional limit entry
    take_profit_price=165.0,     # +10% target
    stop_loss_price=142.50,      # -5% stop
    time_in_force="DAY",
)

result = adapter.submit_bracket_order(config)
# result.entry_order_id, result.tp_order_id, result.sl_order_id
```

### Session Router

```python
from services.session_router import (
    SessionRouter,
    TradingSession,
    get_current_session,
)

# Текущая сессия
session = get_current_session()
# session.session: PRE_MARKET | REGULAR | AFTER_HOURS | CLOSED

# Router для intelligent routing
router = SessionRouter(
    allow_extended_hours=True,
    extended_hours_spread_multiplier=2.0,
)

# Решение о routing
decision = router.get_routing_decision(
    symbol="AAPL",
    side="BUY",
    qty=100,
    order_type="market",
)

if decision.should_submit:
    if decision.use_extended_hours:
        adapter.submit_extended_hours_order(order, session="pre")
    else:
        adapter.submit_order(order)
```

### Trading Sessions (US Equity)

| Session | Время (ET) | Market Orders | Limit Orders | Spread |
|---------|------------|---------------|--------------|--------|
| Pre-market | 4:00-9:30 | ❌ | ✅ | 2.5x |
| Regular | 9:30-16:00 | ✅ | ✅ | 1.0x |
| After-hours | 16:00-20:00 | ❌ | ✅ | 2.0x |
| Closed | 20:00-4:00 | ❌ | ❌ | N/A |

### CLI Usage

```bash
# Crypto (default, backward compatible)
python script_live.py --config configs/config_live.yaml

# Equity explicit
python script_live.py --config configs/config_live_alpaca.yaml --asset-class equity

# Extended hours trading
python script_live.py --config configs/config_live_alpaca.yaml --extended-hours

# Paper trading (Alpaca sandbox)
python script_live.py --config configs/config_live_alpaca.yaml --paper

# Live trading (real money)
python script_live.py --config configs/config_live_alpaca.yaml --live
```

### Backward Compatibility

- **Fully backward compatible** с существующим crypto functionality
- Default asset class = `crypto` если не указан explicit
- Все существующие конфиги работают без изменений
- Новые параметры опциональны

### Тестирование

```bash
# Все тесты Phase 9
pytest tests/test_phase9_live_trading.py -v

# Тесты по категориям
pytest tests/test_phase9_live_trading.py::TestAssetClassDetection -v
pytest tests/test_phase9_live_trading.py::TestPositionSynchronizer -v
pytest tests/test_phase9_live_trading.py::TestSessionRouter -v
pytest tests/test_phase9_live_trading.py::TestBackwardCompatibility -v
```

**Покрытие**: 46 тестов (at documentation time; verify via CI)

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `script_live.py` | Unified live trading entry point |
| `services/position_sync.py` | Position synchronization service |
| `services/session_router.py` | Session-aware order routing |
| `adapters/alpaca/order_execution.py` | Enhanced Alpaca order execution |
| `tests/test_phase9_live_trading.py` | Comprehensive test suite |

---

## 📚 L3 LOB Simulation (Phase 10)

### Обзор

Phase 10 добавляет высокоточную симуляцию order book для US equities:

1. **Stage 1: Data Structures** (`lob/data_structures.py`)
   - LimitOrder, PriceLevel, OrderBook с O(1)/O(log n) операциями
   - Iceberg и hidden order support
   - LOBSTER message format parsing

2. **Stage 2: Matching Engine** (`lob/matching_engine.py`)
   - FIFO Price-Time Priority matching (CME Globex style)
   - Self-Trade Prevention (STP) -- 4 режима
   - Pro-Rata matching для опционных рынков
   - Queue position tracking (Erik Rigtorp method)

3. **Stage 3: Fill Probability & Queue Value** (`lob/fill_probability.py`, `lob/queue_value.py`)
   - Analytical Poisson fill probability: `P(fill in T) = 1 - exp(-λT / position)`
   - Queue-Reactive intensity model (Huang et al.): `λ_i = f(q_i, spread, volatility, imbalance)`
   - Queue Value computation (Moallemi & Yuan): `V = P(fill) * spread/2 - adverse_selection`
   - Calibration pipeline from historical LOB data (`lob/calibration.py`)

4. **Stage 4: Market Impact Models** (`lob/market_impact.py`, `lob/impact_effects.py`)
   - Kyle (1985) Lambda model: `Δp = λ * sign(x) * |x|`
   - Almgren-Chriss (2001): `temp = η * σ * (Q/V)^0.5`, `perm = γ * (Q/V)`
   - Gatheral (2010) transient impact with power-law decay: `G(t) = (1 + t/τ)^(-β)`
   - Impact effects on LOB: quote shifting, liquidity reaction, momentum detection
   - Calibration from historical trade data (`lob/impact_calibration.py`)

5. **Stage 5: Latency Simulation** (`lob/latency_model.py`, `lob/event_scheduler.py`)
   - Realistic latency distributions: Log-normal, Pareto (heavy tail), Gamma
   - Separate feed/order/exchange/fill latencies
   - Latency profiles: Co-located (~10-50μs), Proximity (~100-500μs), Internet (~1-10ms), Institutional (~200μs-2ms)
   - Event scheduler with priority queue and race condition detection
   - Time-of-day seasonality adjustments
   - Volatility-adjusted latency

6. **Stage 6: Hidden Liquidity & Dark Pools** (`lob/hidden_liquidity.py`, `lob/dark_pool.py`)
   - Iceberg order detection from execution patterns (refill pattern recognition)
   - Hidden quantity estimation based on observed refills
   - Dark pool multi-venue simulation (SIGMA_X, IEX_D, LIQUIDNET, RETAIL_INT)
   - Mid-price execution with probabilistic fills
   - Information leakage modeling (quote updates, trade signals, size inference)
   - Smart order routing across dark pool venues
   - Time-of-day and volatility adjustments

7. **Stage 7: L3 Execution Provider Integration** (`execution_providers_l3.py`, `lob/config.py`)
   - Full L3ExecutionProvider combining all LOB components
   - Pydantic-based configuration models for all subsystems
   - Factory function upgrade: `create_execution_provider(level="L3")`
   - YAML configuration support with presets (equity, crypto, minimal)
   - Pre-trade cost estimation with impact models
   - Fill probability computation for limit orders
   - Dark pool routing integration
   - Backward compatible with L2 (crypto unchanged)
   - 79 comprehensive tests

8. **Stage 8: Data Pipeline & Calibration** (`lob/data_adapters.py`, `lob/calibration_pipeline.py`)
   - Data adapters: LOBSTER, ITCH, Binance L2, Alpaca L2
   - Unified L3 calibration pipeline for latency + queue dynamics
   - Format-agnostic LOB update processing
   - Historical data loading utilities

9. **Stage 9: Testing & Validation** (see `docs/L3_VALIDATION_REPORT.md`)
   - 749+ tests passing (at documentation time; verify via CI)
   - Validation metrics: fill rate >95%, slippage <2bps, queue error <10%
   - Performance benchmarks meeting targets
   - Full backward compatibility with crypto

10. **Stage 10: Documentation & Deployment** (`docs/l3_simulator/`)
    - Comprehensive documentation for all L3 components
    - Deployment checklist with feature flags
    - Gradual rollout strategy (shadow mode → canary → production)
    - Monitoring dashboards and alert rules
    - Rollback procedures

### Архитектура

```
lob/
├── data_structures.py       # LimitOrder, PriceLevel, OrderBook, Fill, Trade
├── matching_engine.py       # MatchingEngine, ProRataMatchingEngine, STP
├── queue_tracker.py         # QueuePositionTracker (MBP/MBO estimation)
├── order_manager.py         # OrderManager, ManagedOrder, TimeInForce
├── state_manager.py         # LOBStateManager, LOBSnapshot
├── parsers.py               # LOBSTERParser
├── fill_probability.py      # Poisson, Queue-Reactive, Historical models (Stage 3)
├── queue_value.py           # Queue value computation (Moallemi & Yuan) (Stage 3)
├── calibration.py           # Model calibration from historical data (Stage 3)
├── market_impact.py         # Kyle, Almgren-Chriss, Gatheral models (Stage 4)
├── impact_effects.py        # Quote shifting, liquidity reaction (Stage 4)
├── impact_calibration.py    # Impact parameter estimation (Stage 4)
├── latency_model.py         # Realistic latency simulation (Stage 5)
├── event_scheduler.py       # Event ordering with priority queue (Stage 5)
├── hidden_liquidity.py      # Iceberg detection, hidden qty estimation (Stage 6)
├── dark_pool.py             # Dark pool simulation, multi-venue routing (Stage 6)
├── config.py                # Pydantic config models for L3 subsystems (Stage 7)
├── data_adapters.py         # LOBSTER, ITCH, Binance, Alpaca adapters (Stage 8)
├── calibration_pipeline.py  # Unified L3 calibration pipeline (Stage 8)
├── us_market_structure.py   # SEC Reg NMS rules (tick size, odd lots, NBBO)
└── __init__.py              # Public API exports

execution_providers_l3.py    # L3ExecutionProvider combining all LOB components (Stage 7)

docs/l3_simulator/           # Stage 10 Documentation
├── overview.md              # Architecture overview
├── data_structures.md       # LOB data structures
├── matching_engine.md       # FIFO matching, STP
├── queue_position.md        # Queue position tracking
├── market_impact.md         # Impact models (Kyle, AC, Gatheral)
├── latency.md               # Latency simulation, event scheduling
├── calibration.md           # Parameter estimation
├── configuration.md         # Config reference
├── deployment.md            # Deployment checklist, rollout, rollback
└── migration_guide.md       # L2 to L3 migration reference
```

### Ключевые классы

| Класс | Назначение |
|-------|------------|
| `MatchingEngine` | FIFO matching с STP |
| `ProRataMatchingEngine` | Pro-rata allocation |
| `QueuePositionTracker` | MBP/MBO position estimation |
| `OrderManager` | Order lifecycle (IOC, FOK, DAY, GTC) |
| `LOBStateManager` | State management + snapshots |
| `QueueReactiveModel` | Fill probability с intensity = f(queue, spread, vol) |
| `QueueValueModel` | Queue position value (Moallemi & Yuan) |
| `CalibrationPipeline` | MLE parameter fitting from historical data |
| `AlmgrenChrissModel` | Square-root temporary + linear permanent impact (Stage 4) |
| `GatheralModel` | Transient impact with power-law decay (Stage 4) |
| `KyleLambdaModel` | Kyle (1985) linear price impact model (Stage 4) |
| `ImpactEffects` | Quote shifting, liquidity reaction, momentum (Stage 4) |
| `LOBImpactSimulator` | Complete trade impact simulation workflow (Stage 4) |
| `ImpactCalibrationPipeline` | OLS/MLE calibration for impact params (Stage 4) |
| `LatencyModel` | Realistic latency simulation with profiles (Stage 5) |
| `LatencySampler` | Distribution-based latency sampling (Stage 5) |
| `EventScheduler` | Event ordering with priority queue (Stage 5) |
| `SimulationClock` | Time tracking with latency awareness (Stage 5) |
| `IcebergDetector` | Iceberg order detection from execution patterns (Stage 6) |
| `IcebergOrder` | Tracked iceberg with refill history (Stage 6) |
| `HiddenLiquidityEstimator` | Hidden quantity estimation (Stage 6) |
| `DarkPoolSimulator` | Multi-venue dark pool simulation (Stage 6) |
| `DarkPoolVenue` | Individual dark pool venue model (Stage 6) |
| `DarkPoolFill` | Dark pool execution result (Stage 6) |
| `L3ExecutionProvider` | Full L3 execution provider combining all LOB components (Stage 7) |
| `L3SlippageProvider` | LOB-based slippage with market impact (Stage 7) |
| `L3FillProvider` | LOB-based fill logic with queue position (Stage 7) |
| `L3ExecutionConfig` | Pydantic config model for L3 subsystems (Stage 7) |
| `BaseLOBAdapter` | Abstract base for LOB data adapters (Stage 8) |
| `LOBSTERAdapter` | LOBSTER format adapter (Stage 8) |
| `ITCHAdapter` | ITCH format adapter (Stage 8) |
| `BinanceL2Adapter` | Binance L2 data adapter (Stage 8) |
| `AlpacaL2Adapter` | Alpaca L2 data adapter (Stage 8) |
| `L3CalibrationPipeline` | Unified calibration for L3 (Stage 8) |
| `LatencyCalibrator` | Latency distribution calibration (Stage 8) |
| `QueueDynamicsCalibrator` | Queue dynamics calibration (Stage 8) |
| `TickSizeValidator` | SEC Reg NMS Rule 612 tick size validation |
| `OddLotHandler` | Odd lot (<100 shares) handling per SEC Rule 600 |
| `NBBOProtector` | Reg NMS Rule 611 trade-through prevention |

### Self-Trade Prevention (STP)

| Режим | Действие |
|-------|----------|
| `CANCEL_NEWEST` | Отменяет входящий (aggressive) ордер |
| `CANCEL_OLDEST` | Отменяет resting ордер |
| `CANCEL_BOTH` | Отменяет оба ордера |
| `DECREMENT_AND_CANCEL` | Уменьшает qty, отменяет меньший |

### Time-in-Force

| TIF | Поведение |
|-----|-----------|
| `DAY` | Активен до конца дня |
| `GTC` | Good-Til-Cancelled |
| `IOC` | Immediate-Or-Cancel (partial fill → CANCELLED) |
| `FOK` | Fill-Or-Kill (all or nothing) |

### Queue Position Estimation

```python
from lob import QueuePositionTracker, PositionEstimationMethod

tracker = QueuePositionTracker()

# MBP (pessimistic) -- advance only on executions
state = tracker.add_order(order, level_qty_before=500.0)

# MBO (exact) -- requires order-level data
state = tracker.add_order(order, orders_ahead=[...])

# Fill probability (Poisson model)
prob = tracker.estimate_fill_probability(
    order_id, volume_per_second=100.0, time_horizon_sec=60.0
)
```

### Использование

```python
from lob import OrderManager, Side, OrderType, TimeInForce

manager = OrderManager(symbol="AAPL")

# Submit limit order
order = manager.submit_order(
    side=Side.BUY,
    price=150.0,
    qty=100.0,
    order_type=OrderType.LIMIT,
    time_in_force=TimeInForce.DAY,
)

# Check fill probability
prob = manager.get_fill_probability(order.order.order_id)

# Cancel
manager.cancel_order(order.order.order_id)
```

### Performance

| Операция | Latency | Target |
|----------|---------|--------|
| Market order simulation | ~5 μs | <10 μs ✅ |
| Limit order matching | ~20 μs | <50 μs ✅ |
| Queue position update | ~50 μs | <500 μs ✅ |

### Stage 3: Fill Probability & Queue Value

```python
from lob import (
    QueueReactiveModel,
    QueueValueModel,
    CalibrationPipeline,
    LOBState,
    TradeRecord,
    Side,
)

# 1. Create fill probability model
fill_model = QueueReactiveModel(
    base_rate=100.0,           # Base volume rate (qty/sec)
    queue_decay_alpha=0.01,    # Queue size impact
    spread_sensitivity_beta=0.5,  # Spread impact
)

# 2. Estimate fill probability
lob_state = LOBState(
    mid_price=150.0,
    spread_bps=5.0,
    volatility=0.02,
    imbalance=0.1,
)

prob_result = fill_model.compute_fill_probability(
    queue_position=10,
    qty_ahead=500.0,
    order_qty=100.0,
    time_horizon_sec=60.0,
    market_state=lob_state,
)
print(f"P(fill in 60s) = {prob_result.prob_fill:.2%}")

# 3. Compute queue value (Moallemi & Yuan)
value_model = QueueValueModel(fill_model=fill_model)
value_result = value_model.compute_queue_value(order, lob_state, queue_state)
print(f"Queue value: ${value_result.queue_value:.4f}")
print(f"Decision: {value_result.decision.name}")  # HOLD or CANCEL

# 4. Calibrate from historical data
pipeline = CalibrationPipeline()
for trade in historical_trades:
    pipeline.add_trade(TradeRecord(
        timestamp_ns=trade.ts,
        price=trade.price,
        qty=trade.qty,
        side=Side.BUY if trade.is_buy else Side.SELL,
    ))
results = pipeline.run_calibration()
calibrated_model = pipeline.get_best_model("queue_reactive")
```

### Stage 4: Market Impact Models

```python
from lob import (
    AlmgrenChrissModel,
    GatheralModel,
    ImpactParameters,
    ImpactEffects,
    LOBImpactSimulator,
    create_impact_model,
    ImpactCalibrationPipeline,
    TradeObservation,
    CalibrationDataset,
)

# 1. Create impact model
params = ImpactParameters.for_equity()  # or .for_crypto()
model = AlmgrenChrissModel(params=params)

# 2. Compute market impact
result = model.compute_total_impact(
    order_qty=10000,
    adv=10_000_000,
    volatility=0.02,
    mid_price=150.0,
)
print(f"Temporary: {result.temporary_impact_bps:.2f} bps")
print(f"Permanent: {result.permanent_impact_bps:.2f} bps")
print(f"Impact cost: ${result.impact_cost:.2f}")

# 3. Simulate impact effects on LOB
simulator = LOBImpactSimulator(impact_model=model)
impact, quote_shift, liquidity = simulator.simulate_trade_impact(
    order_book=order_book,
    order=limit_order,
    fill=fill,
    adv=10_000_000,
    volatility=0.02,
)
print(f"New bid: {quote_shift.new_bid}, New ask: {quote_shift.new_ask}")

# 4. Calibrate from historical trades
pipeline = ImpactCalibrationPipeline()
dataset = CalibrationDataset(avg_adv=10_000_000, avg_volatility=0.02)
for trade in historical_trades:
    obs = TradeObservation(
        timestamp_ms=trade.ts,
        price=trade.price,
        qty=trade.qty,
        side=1 if trade.is_buy else -1,
        adv=dataset.avg_adv,
        pre_trade_mid=trade.pre_mid,
        post_trade_mid=trade.post_mid,
    )
    dataset.add_observation(obs)
results = pipeline.calibrate_all(dataset)
calibrated_model = pipeline.create_calibrated_model()
```

### Stage 5: Latency Simulation

```python
from lob import (
    LatencyModel,
    LatencyProfile,
    EventScheduler,
    SimulationClock,
    MarketDataEvent,
    create_latency_model,
    create_event_scheduler,
)

# 1. Create latency model from profile
model = LatencyModel.from_profile(LatencyProfile.INSTITUTIONAL, seed=42)
# Or: model = create_latency_model("colocated")

# 2. Sample latencies (returns nanoseconds)
feed_latency = model.sample_feed_latency()
order_latency = model.sample_order_latency()
exchange_latency = model.sample_exchange_latency()
fill_latency = model.sample_fill_latency()
round_trip = model.sample_round_trip()

print(f"Feed: {feed_latency/1000:.1f}us, Order: {order_latency/1000:.1f}us")
print(f"Round-trip: {round_trip/1000:.1f}us")

# 3. Create event scheduler
scheduler = create_event_scheduler("institutional", seed=42)

# Schedule market data event
event = MarketDataEvent(
    symbol="AAPL",
    exchange_time_ns=1_000_000,
    bid_price=150.0,
    ask_price=150.05,
)
our_receive_time = scheduler.schedule_market_data(event, exchange_time_ns=1_000_000)

# Schedule our order
from lob import LimitOrder, Side
order = LimitOrder(
    order_id="order_1",
    price=150.0,
    qty=100.0,
    remaining_qty=100.0,
    timestamp_ns=1_000_000,
    side=Side.BUY,
)
arrival_time = scheduler.schedule_order_arrival(order, our_send_time_ns=1_000_000)

# Process all events in timestamp order
for event in scheduler:
    print(f"Event: {event.event_type.name} at {event.timestamp_ns}ns")

# 4. Get latency statistics
stats = model.stats()
print(f"Feed p95: {stats['feed']['p95_us']:.1f}us")
```

### Stage 6: Hidden Liquidity & Dark Pools

```python
from lob import (
    # Iceberg Detection
    IcebergDetector,
    IcebergOrder,
    IcebergState,
    DetectionConfidence,
    HiddenLiquidityEstimator,
    create_iceberg_detector,
    create_hidden_liquidity_estimator,
    # Dark Pool Simulation
    DarkPoolSimulator,
    DarkPoolVenue,
    DarkPoolConfig,
    DarkPoolFill,
    DarkPoolVenueType,
    FillType,
    InformationLeakage,
    create_dark_pool_simulator,
    create_default_dark_pool_simulator,
)

# 1. Create iceberg detector
detector = create_iceberg_detector(
    min_refills_to_confirm=2,
    lookback_window_sec=60.0,
)

# 2. Process execution and detect iceberg pattern
pre_snap = detector.take_level_snapshot(level, Side.BUY)
# ... execution happens ...
post_snap = detector.take_level_snapshot(level, Side.BUY)
iceberg = detector.process_execution(trade, pre_snap, post_snap, Side.BUY)

if iceberg:
    print(f"Iceberg detected: display={iceberg.display_size}, state={iceberg.state.name}")
    hidden_estimate = detector.estimate_hidden_reserve(iceberg)
    print(f"Estimated hidden: {hidden_estimate}")

# 3. Batch detection from execution history
executions = [trade1, trade2, trade3]
level_qty_history = [500.0, 500.0, 500.0]  # Qty refills indicate iceberg
iceberg = detector.detect_iceberg(executions, level_qty_history, price=100.0, side=Side.BUY)

# 4. Hidden liquidity estimation
estimator = create_hidden_liquidity_estimator(detector, hidden_ratio=0.15)
hidden = estimator.estimate_hidden_at_level(price=100.0, side=Side.BUY, visible_qty=500.0)

# 5. Create dark pool simulator
dark_pool = create_default_dark_pool_simulator(seed=42)

# 6. Attempt dark pool fill
fill = dark_pool.attempt_dark_fill(
    order=limit_order,
    lit_mid_price=100.0,
    lit_spread=0.05,
    adv=10_000_000,
    volatility=0.02,
    hour_of_day=10,
)

if fill and fill.is_filled:
    print(f"Dark fill: {fill.filled_qty} @ {fill.fill_price} ({fill.venue_id})")
    if fill.info_leakage:
        print(f"Leakage: {fill.info_leakage.description}")

# 7. Estimate fill probability at each venue
probs = dark_pool.estimate_fill_probability(order, adv=10_000_000)
for venue_id, prob in probs.items():
    print(f"{venue_id}: {prob:.2%}")

# 8. Multi-venue routing
fills = dark_pool.attempt_fill_with_routing(order, lit_mid_price=100.0, max_attempts=3)
```

### US Market Structure (`lob/us_market_structure.py`)

SEC Reg NMS rules implementation for realistic equity simulation:

| Rule | Component | Description |
|------|-----------|-------------|
| **Rule 612** | `TickSizeValidator` | Sub-penny rule: $0.01 for ≥$1.00, $0.0001 for <$1.00 |
| **Rule 600** | `OddLotHandler` | Odd lot (<100 shares), round lot, mixed lot handling |
| **Rule 611** | `NBBOProtector` | Order Protection Rule (trade-through prevention) |

**Lot Types**:

- `ODD_LOT` -- < 100 shares (different execution properties)
- `ROUND_LOT` -- Exactly 100 shares or multiples
- `MIXED_LOT` -- Round lots + odd lot remainder

**Trade-Through Protection**:

- `BID_THROUGH` -- Sell below protected bid (violation)
- `ASK_THROUGH` -- Buy above protected ask (violation)

```python
from lob.us_market_structure import (
    TickSizeValidator,
    OddLotHandler,
    NBBOProtector,
    LotType,
    TradeThrough,
    TICK_SIZE_PENNY,
    ROUND_LOT_SIZE,
)

# Tick size validation
validator = TickSizeValidator()
valid = validator.validate_price(150.015, stock_price=150.0)  # False (sub-penny!)
rounded = validator.round_to_tick(150.015)  # 150.01

# Lot type classification
handler = OddLotHandler()
lot_type = handler.classify_lot(75)  # LotType.ODD_LOT

# NBBO protection check
protector = NBBOProtector()
violation = protector.check_trade_through(
    trade_price=149.99,
    side="SELL",
    nbbo_bid=150.00,
    nbbo_ask=150.02,
)
# violation = TradeThrough.BID_THROUGH
```

### Тестирование

```bash
# Stage 1 тесты (data structures, parsers, state manager)
pytest tests/test_lob_structures.py tests/test_lob_parsers.py tests/test_lob_state_manager.py -v

# Stage 2 тесты (matching engine, queue tracker, order manager)
pytest tests/test_matching_engine.py -v

# Stage 3 тесты (fill probability, queue value, calibration)
pytest tests/test_fill_probability_queue_value.py -v

# Stage 4 тесты (market impact, effects, calibration)
pytest tests/test_market_impact.py -v

# Stage 5 тесты (latency simulation, event scheduler)
pytest tests/test_lob_latency.py -v

# Stage 6 тесты (hidden liquidity, dark pools)
pytest tests/test_hidden_liquidity_dark_pools.py -v

# Stage 7 тесты (L3 execution provider, config)
pytest tests/test_execution_providers_l3.py -v

# Stage 8 тесты (data adapters, calibration pipeline)
pytest tests/test_lob_data_adapters.py tests/test_lob_calibration_pipeline.py -v

# Stage 9 тесты (validation, backward compatibility)
pytest tests/test_queue_tracker.py tests/test_l3_vs_production.py tests/test_l3_backward_compatibility.py -v

# Все LOB тесты
pytest tests/test_lob*.py tests/test_matching_engine.py tests/test_fill_probability_queue_value.py \
    tests/test_market_impact.py tests/test_hidden_liquidity_dark_pools.py tests/test_execution_providers_l3.py \
    tests/test_queue_tracker.py tests/test_l3_vs_production.py tests/test_l3_backward_compatibility.py -v
```

**Покрытие**: 749+ тестов (106 Stage 1 + 72 Stage 2 + 66 Stage 3 + 57 Stage 4 + 66 Stage 5 + 62 Stage 6 + 79 Stage 7 + Stage 8 + 117 Stage 9 + 95 execution_providers)

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `lob/matching_engine.py` | FIFO matching engine with STP |
| `lob/queue_tracker.py` | Queue position tracking (MBP/MBO) |
| `lob/order_manager.py` | Order lifecycle management |
| `lob/data_structures.py` | Core data structures |
| `lob/fill_probability.py` | Poisson, Queue-Reactive, Historical models |
| `lob/queue_value.py` | Queue value computation (Moallemi & Yuan) |
| `lob/calibration.py` | MLE calibration from historical data |
| `lob/market_impact.py` | Kyle, Almgren-Chriss, Gatheral impact models |
| `lob/impact_effects.py` | Quote shifting, liquidity reaction, momentum |
| `lob/impact_calibration.py` | OLS/grid search calibration for impact params |
| `tests/test_matching_engine.py` | 72 Stage 2 tests |
| `tests/test_fill_probability_queue_value.py` | 66 Stage 3 tests |
| `tests/test_market_impact.py` | 57 Stage 4 tests |
| `lob/latency_model.py` | Realistic latency simulation (Stage 5) |
| `lob/event_scheduler.py` | Event ordering with priority queue (Stage 5) |
| `tests/test_lob_latency.py` | 66 Stage 5 tests |
| `lob/hidden_liquidity.py` | Iceberg detection, hidden liquidity estimation (Stage 6) |
| `lob/dark_pool.py` | Dark pool simulation, multi-venue routing (Stage 6) |
| `tests/test_hidden_liquidity_dark_pools.py` | 62 Stage 6 tests |
| `execution_providers_l3.py` | L3ExecutionProvider combining all LOB components (Stage 7) |
| `lob/config.py` | Pydantic configuration models for L3 subsystems (Stage 7) |
| `configs/execution_l3.yaml` | L3 execution configuration file (Stage 7) |
| `tests/test_execution_providers_l3.py` | 79 Stage 7 tests |
| `lob/data_adapters.py` | LOBSTER, ITCH, Binance, Alpaca adapters (Stage 8) |
| `lob/calibration_pipeline.py` | Unified L3 calibration pipeline (Stage 8) |
| `tests/test_lob_data_adapters.py` | Data adapters tests (Stage 8) |
| `tests/test_lob_calibration_pipeline.py` | Calibration pipeline tests (Stage 8) |
| `tests/test_queue_tracker.py` | 55 Queue position tracking tests (Stage 9) |
| `tests/test_l3_vs_production.py` | 30 Validation metrics tests (Stage 9) |
| `tests/test_l3_backward_compatibility.py` | 32 Backward compatibility tests (Stage 9) |
| `benchmarks/bench_matching.py` | Matching engine benchmarks (Stage 9) |
| `benchmarks/bench_full_sim.py` | Full simulation benchmarks (Stage 9) |
| `docs/L3_VALIDATION_REPORT.md` | Stage 9 validation report |
| `docs/L3_MIGRATION_GUIDE.md` | Migration guide from L2 to L3 |
| `docs/l3_simulator/overview.md` | L3 architecture overview (Stage 10) |
| `docs/l3_simulator/data_structures.md` | LOB data structures (Stage 10) |
| `docs/l3_simulator/matching_engine.md` | Matching engine docs (Stage 10) |
| `docs/l3_simulator/queue_position.md` | Queue position tracking (Stage 10) |
| `docs/l3_simulator/market_impact.md` | Impact models (Stage 10) |
| `docs/l3_simulator/latency.md` | Latency simulation (Stage 10) |
| `docs/l3_simulator/calibration.md` | Calibration guide (Stage 10) |
| `docs/l3_simulator/configuration.md` | Config reference (Stage 10) |
| `docs/l3_simulator/deployment.md` | Deployment checklist & rollout (Stage 10) |

### Референсы

- CME Globex Matching Algorithm
- Erik Rigtorp: Queue Position Estimation
- Cont et al. (Columbia): Fill Probability Models
- FIX Protocol: Order Status semantics
- Huang et al. (2015): Queue-Reactive Model
- Moallemi & Yuan (2017): Queue Position Valuation
- Kyle (1985): "Continuous Auctions and Insider Trading"
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"
- Almgren et al. (2005): "Direct Estimation of Equity Market Impact"
- hftbacktest: High-frequency trading backtesting framework (latency modeling reference)
- Bookmap: Iceberg order detection methodology (https://bookmap.com/blog/advanced-order-flow-trading-spotting-hidden-liquidity-iceberg-orders)
- SEC Rule 606: Dark pool routing disclosures
- FINRA ATS: Dark pool transparency data

---

## 💱 Forex Integration (Phase 11)

### Обзор

Phase 11 добавляет полную поддержку Forex (OTC) через OANDA:

**Статус**: ✅ Tested and operational | **Тесты**: 18 test files (735+ tests planned)

**Ключевое архитектурное решение**: Forex -- это OTC (Over-The-Counter) рынок с дилерскими котировками, а НЕ биржевой рынок. Поэтому:

- Используется **L2+ Parametric TCA** (как для crypto/equity), НЕ L3 LOB simulation
- **OTC Dealer Simulation** -- отдельный модуль в `services/`, НЕ в `lob/`

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **ForexParametricTCA** | `execution_providers.py` | 8-factor slippage model |
| **ForexFeatures** | `forex_features.py` | Session-aware features |
| **ForexDealer** | `services/forex_dealer.py` | OTC dealer simulation |
| **ForexRiskGuards** | `services/forex_risk_guards.py` | Leverage & margin guards |
| **ForexSessionRouter** | `services/forex_session_router.py` | Session-aware routing |
| **ForexConfig** | `services/forex_config.py` | Pydantic config models |
| **ForexEnv** | `wrappers/forex_env.py` | Trading environment wrapper |
| **ForexTickSim** | `lob/forex_tick_simulation.py` | Tick-level simulation |
| **OANDA Adapter** | `adapters/oanda/*.py` | Market data & execution |

### Forex Sessions (Критично для моделирования)

| Session | Время (UTC) | Liquidity Factor | Spread Multiplier |
|---------|-------------|------------------|-------------------|
| **Sydney** | 21:00-06:00 | 0.60-0.70 | 1.4-1.6x |
| **Tokyo** | 00:00-09:00 | 0.70-0.85 | 1.2-1.4x |
| **London** | 07:00-16:00 | 1.00-1.20 | 1.0x |
| **New York** | 12:00-21:00 | 1.00-1.15 | 1.0x |
| **London/NY overlap** | 12:00-16:00 | **1.30-1.50** | **0.8x** (tightest) |

### Forex vs Crypto/Equity

| Аспект | Crypto | Equity | **Forex** |
|--------|--------|--------|-----------|
| **Market structure** | Central LOB | Central LOB | **OTC Dealer Network** |
| **Trading hours** | 24/7 | NYSE 9:30-16:00 ET | **Sun 5pm - Fri 5pm ET** |
| **Fees** | Maker/Taker % | $0 + regulatory | **Spread-based (0 commission)** |
| **Simulation** | L3 LOB | L3 LOB | **L2+ Parametric + OTC Sim** |
| **Leverage** | 1x-125x | 1x-4x | **50:1 - 500:1** |

### Конфигурация

```yaml
# configs/config_train_forex.yaml
mode: train
asset_class: forex
data_vendor: oanda

forex:
  default_spread_pips: 1.0
  session_spread_multipliers:
    sydney: 1.5
    tokyo: 1.3
    london: 1.0
    new_york: 1.0
  leverage: 50
  margin_requirement: 0.02  # 2%
```

### Тестирование

```bash
# Все Forex тесты
pytest tests/test_forex*.py -v

# По категориям
pytest tests/test_forex_parametric_tca.py -v        # L2+ TCA
pytest tests/test_forex_dealer_simulation.py -v     # OTC dealer
pytest tests/test_forex_features.py -v              # Session features
pytest tests/test_forex_phase6_risk_services.py -v  # Risk guards
pytest tests/test_forex_configuration.py -v         # Config models
```

### Environment Variables

```bash
OANDA_API_KEY=...
OANDA_ACCOUNT_ID=...
OANDA_PRACTICE=true  # or false for live
```

### Референсы

- BIS Triennial Survey (2022): FX market structure
- LMAX Exchange: FX market microstructure
- OANDA API Documentation
- `docs/FOREX_INTEGRATION_PLAN.md` -- Полный план интеграции
- `docs/FOREX_INTEGRATION_QUICK_REF.md` -- Краткий справочник

---

## 🔮 Futures Integration (Phase 3B-10: ✅ COMPLETE)

**Статус**: ✅ Tested and operational | **Документация**: `docs/FUTURES_INTEGRATION_PLAN.md`

**Completed Phases**:

- Phase 3B: ✅ IB/CME Adapters
- Phase 4A: ✅ Crypto L2 Execution
- Phase 4B: ✅ CME SPAN Margin
- Phase 5A: ✅ Crypto L3 LOB
- Phase 5B: ✅ CME L3 LOB
- Phase 6A: ✅ Crypto Risk Guards
- Phase 6B: ✅ CME Risk Guards
- Phase 7: ✅ Unified Risk Management
- Phase 8: ✅ Multi-Futures Training Pipeline
- Phase 9: ✅ Unified Futures Live Trading
- Phase 10: ✅ Validation & Documentation

Интеграция всех типов фьючерсов:

| Тип | Биржа | Примеры | Статус | Phase |
|-----|-------|---------|--------|-------|
| **Equity Index** | CME (via IB) | ES, NQ, YM, RTY | ✅ IB Adapters Ready | 3B |
| **Commodity** | CME (via IB) | GC, CL, SI, NG | ✅ IB Adapters Ready | 3B |
| **Currency** | CME (via IB) | 6E, 6J, 6B, 6A | ✅ IB Adapters Ready | 3B |
| **Bonds** | CME (via IB) | ZN, ZB, ZT | ✅ IB Adapters Ready | 3B |
| **Crypto Perpetual** | Binance | BTCUSDT, ETHUSDT | ✅ L2 Execution Provider | **4A** |
| **Crypto Quarterly** | Binance | BTCUSDT_240329 | 📋 Phase 4B Planned | 4B |

Ключевые концепции: Leverage & Margin, Mark Price, Funding Rates (crypto), Rollover, Settlement.

---

## 📦 Phase 3B: Interactive Brokers & CME Settlement (COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 205/205 (at documentation time; verify via CI)

Phase 3B добавляет полную поддержку CME Group futures через Interactive Brokers TWS API:

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **IB Market Data** | `adapters/ib/market_data.py` | Historical bars, real-time quotes, contract details |
| **IB Order Execution** | `adapters/ib/order_execution.py` | Market/limit/bracket orders, margin queries |
| **IB Exchange Info** | `adapters/ib/exchange_info.py` | Contract specifications |
| **CME Settlement** | `impl_cme_settlement.py` | Daily settlement engine, variation margin |
| **CME Rollover** | `impl_cme_rollover.py` | Contract rollover manager |
| **CME Calendar** | `services/cme_calendar.py` | Trading hours, holidays, maintenance windows |

### Поддерживаемые контракты (30+)

**Equity Index (CME):**

- **E-mini**: ES (S&P 500), NQ (NASDAQ 100), RTY (Russell 2000), YM (Dow)
- **Micro E-mini**: MES, MNQ, M2K, MYM

**Metals (COMEX):**

- **Standard**: GC (Gold), SI (Silver), HG (Copper)
- **Micro**: MGC (Micro Gold), SIL (Micro Silver)

**Energy (NYMEX):**

- **Standard**: CL (Crude Oil), NG (Natural Gas), RB (Gasoline), HO (Heating Oil)
- **Micro**: MCL (Micro Crude Oil)

**Currencies (CME):**

- 6E (Euro), 6J (Yen), 6B (Pound), 6A (Aussie), 6C (CAD), 6S (CHF)

**Bonds (CBOT):**

- ZN (10-Year Note), ZB (30-Year Bond), ZT (2-Year Note), ZF (5-Year Note)

### IB TWS API Rate Limiting

**Production-grade rate limiter** (`IBRateLimiter`) с thread-safe tracking:

| Rate Limit Type | IB Limit | Implementation | Safety Margin |
|-----------------|----------|----------------|---------------|
| General messages | 50/sec | 45/sec | 10% |
| Historical requests | 60/10min | 55/10min | 8% |
| Identical requests | 6/10min | 5/10min | 17% |
| Market data subscriptions | 1/sec | 1/sec | None (hard limit) |
| Concurrent market data | 100 lines | 100 lines | None (hard limit) |

**Connection Management** (`IBConnectionManager`):

- Heartbeat every 30sec (IB requires 60sec)
- Exponential backoff reconnection: [1, 2, 5, 10, 30, 60, 120] seconds
- Paper/Live routing via port:
  - `7497` = TWS Paper
  - `7496` = TWS Live
  - `4002` = Gateway Paper
  - `4001` = Gateway Live

### CME Settlement Engine

**Product-specific settlement times** (Eastern Time):

| Product Category | Examples | Settlement Time (ET) | Reference |
|------------------|----------|----------------------|-----------|
| Equity Index | ES, NQ, YM, RTY | 15:30 (14:30 CT) | CME Group |
| Currencies | 6E, 6J, 6B | 15:00 (14:00 CT) | CME Group |
| Metals | GC, SI, HG | 14:30 (13:30 CT) | COMEX |
| Energy | CL, NG | 15:30 (14:30 CT) | NYMEX |
| Bonds | ZN, ZB, ZT | 16:00 (15:00 CT) | CBOT |
| Agricultural | ZC, ZS, ZW | 14:15 (13:15 CT) | CBOT |

**Variation Margin Calculation**:

```python
from impl_cme_settlement import CMESettlementEngine, create_settlement_engine

engine = CMESettlementEngine()

# Daily variation margin
variation = engine.calculate_variation_margin(
    position=futures_position,
    settlement_price=Decimal("4500.00"),
    contract_spec=es_spec,
)
# variation = (Settlement_t - Settlement_t-1) × Qty × Multiplier
```

**Formula**: `VM = ΔP × qty × multiplier`

- LONG position: profit if price ↑, loss if price ↓
- SHORT position: profit if price ↓, loss if price ↑

### Contract Rollover

**Standard roll dates** by product:

| Product | Roll Date | Example |
|---------|-----------|---------|
| Equity Index (ES, NQ) | 8 business days before expiry | 2nd Thursday before 3rd Friday |
| Currencies (6E, 6J) | 2 business days before expiry | 2nd business day before 3rd Wednesday |
| Metals (GC, SI) | 3 business days before last trading day | End of month before delivery |
| Energy (CL, NG) | 3 business days before expiry | ~3 days before contract month end |
| Bonds (ZN, ZB) | 7 business days before first delivery | ~7 days before month end |

**Contract Month Codes**:

```
F = Jan, G = Feb, H = Mar, J = Apr, K = May, M = Jun
N = Jul, Q = Aug, U = Sep, V = Oct, X = Nov, Z = Dec
```

**Contract Cycles**:

- **Quarterly** (H, M, U, Z): Equity Index, Currencies, Bonds
- **Monthly** (All months): Energy
- **Bi-Monthly**: Metals, Grains

### CME Trading Calendar

**CME Globex Hours** (Eastern Time):

- **Regular**: Sunday 18:00 ET → Friday 17:00 ET
- **Daily Maintenance**: Monday-Friday 16:15-16:30 ET (15 minutes)
- **Weekend**: Closed Saturday

**US Market Holidays** (2024-2026):

```python
from services.cme_calendar import CMETradingCalendar

calendar = CMETradingCalendar()

# Check if trading
is_open = calendar.is_trading_hours(datetime.now())

# Check holiday
is_holiday = calendar.is_holiday(date.today())

# Get next open
next_open = calendar.get_next_open(datetime.now())
```

**Holiday List** (2024-2026):

- New Year's Day, MLK Day, Presidents Day, Good Friday
- Memorial Day, Juneteenth, Independence Day
- Labor Day, Thanksgiving, Christmas

**Early Close Days**:

- Day before Thanksgiving: 13:15 ET
- Christmas Eve: 13:15 ET
- New Year's Eve: 13:15 ET

### Использование

```python
# 1. Market Data Adapter
from adapters.ib import IBMarketDataAdapter
from adapters.models import ExchangeVendor

adapter = IBMarketDataAdapter(
    vendor=ExchangeVendor.IB,
    config={
        "host": "127.0.0.1",
        "port": 7497,  # Paper trading
        "client_id": 1,
        "readonly": True,
    }
)

# Fetch historical bars
bars = adapter.get_bars("ES", "1h", limit=500)

# Get current quote
tick = adapter.get_tick("ES")

# Get contract details
spec = adapter.get_contract_details("ES")


# 2. Order Execution Adapter
from adapters.ib import IBOrderExecutionAdapter

execution = IBOrderExecutionAdapter(
    vendor=ExchangeVendor.IB,
    config={
        "host": "127.0.0.1",
        "port": 7497,
        "client_id": 2,
    }
)

# Submit market order
order = execution.submit_market_order("ES", "BUY", qty=1)

# Submit bracket order (entry + TP + SL)
from adapters.ib.order_execution import IBBracketOrderConfig

bracket = execution.submit_bracket_order(IBBracketOrderConfig(
    symbol="ES",
    side="BUY",
    qty=1,
    entry_price=Decimal("4500.00"),
    take_profit_price=Decimal("4550.00"),  # +50 points
    stop_loss_price=Decimal("4475.00"),    # -25 points
))

# Query margin requirement
margin = execution.get_margin_requirement("ES", qty=1)
# margin = {"initial_margin": ..., "maint_margin": ..., "impact_on_margin": ...}

# Get positions
positions = execution.get_positions()


# 3. CME Settlement
from impl_cme_settlement import CMESettlementEngine, create_settlement_engine
from core_futures import FuturesPosition, FuturesContractSpec

engine = create_settlement_engine()

# Calculate daily variation margin
variation = engine.calculate_variation_margin(
    position=FuturesPosition(...),
    settlement_price=Decimal("4500.00"),
    contract_spec=FuturesContractSpec(...),
)

# Check if settlement time
is_settlement = engine.is_settlement_time(
    timestamp_ms=int(time.time() * 1000),
    symbol="ES",
)


# 4. Contract Rollover
from impl_cme_rollover import ContractRolloverManager

rollover = ContractRolloverManager(expiration_calendar={
    "ES": [date(2025, 3, 21), date(2025, 6, 20), ...]
})

# Check if should roll
should_roll = rollover.should_roll("ES", date.today())

# Get roll date
roll_date = rollover.get_roll_date("ES", date.today())


# 5. Trading Calendar
from services.cme_calendar import CMETradingCalendar, CMESession

calendar = CMETradingCalendar()

# Check trading hours
is_open = calendar.is_trading_hours(datetime.now())

# Get current session
session = calendar.get_current_session(datetime.now())
# session = CMESession.REGULAR | MAINTENANCE | CLOSED

# Check holiday
is_holiday = calendar.is_holiday(date.today())
```

### Конфигурация

**IB Connection Config**:

```yaml
# configs/ib_connection.yaml
host: "127.0.0.1"
port: 7497  # Paper: 7497 (TWS) or 4002 (Gateway)
client_id: 1
readonly: true  # Safety: data-only mode
timeout: 10.0
account: null  # For multi-account setups
```

**Environment Variables**:

```bash
# Not required for IB (uses TWS/Gateway local connection)
# But recommended for logging
IB_LOG_LEVEL=INFO
IB_ENABLE_RATE_LIMIT_LOGGING=true
```

### Тестирование

```bash
# IB Adapters tests (100 tests)
pytest tests/test_ib_adapters.py -v

# CME Settlement tests (52 tests)
pytest tests/test_cme_settlement.py -v

# CME Calendar tests (53 tests)
pytest tests/test_cme_calendar.py -v

# All Phase 3B tests (205 tests)
pytest tests/test_ib_adapters.py tests/test_cme_settlement.py tests/test_cme_calendar.py -v
```

**Coverage**: 205 tests (at documentation time; verify via CI)

| Test Suite | Tests | Focus |
|------------|-------|-------|
| `test_ib_adapters.py` | 100 | Rate limiting, connection mgmt, contract mapping, order execution |
| `test_cme_settlement.py` | 52 | Settlement times, variation margin, rollover dates |
| `test_cme_calendar.py` | 53 | Trading hours, holidays, session detection |

### Ключевые отличия CME vs Crypto Perpetuals

| Аспект | Crypto Perpetual (Binance) | CME Futures (IB) |
|--------|----------------------------|------------------|
| **Settlement** | Funding every 8h (continuous) | Daily settlement at fixed time |
| **Expiration** | Perpetual (no expiry) | Quarterly/Monthly expiration |
| **Rollover** | N/A | Required ~8 days before expiry |
| **Margin** | Cross/Isolated with ADL | SPAN margin (risk-based) |
| **Trading Hours** | 24/7 | Sun 18:00 - Fri 17:00 ET |
| **Maintenance** | N/A | Daily 16:15-16:30 ET |
| **Leverage** | Up to 125x (venue-specific) | Regulated by SPAN |
| **Mark Price** | Index + funding basis | Last traded price |

### Dependencies

```bash
pip install ib_insync  # IB TWS API wrapper (required)
```

**TWS/Gateway Setup**:

1. Download IB TWS or Gateway from Interactive Brokers
2. Enable API connections (Edit → Global Configuration → API → Enable ActiveX and Socket Clients)
3. Set Socket Port: 7497 (paper) or 7496 (live)
4. Allow connections from `127.0.0.1`

### Registry Integration

**Automatically registered** в `adapters/registry.py`:

```python
ExchangeVendor.IB           # Generic IB
ExchangeVendor.IB_CME       # CME futures
ExchangeVendor.IB_CBOT      # CBOT futures
ExchangeVendor.IB_NYMEX     # NYMEX futures
ExchangeVendor.IB_COMEX     # COMEX futures
```

**Factory Functions**:

```python
from adapters.registry import create_market_data_adapter, create_order_execution_adapter

# Via registry
md_adapter = create_market_data_adapter("ib", {"port": 7497})
exec_adapter = create_order_execution_adapter("ib", {"port": 7497})
```

### Референсы

- **IB TWS API**: https://interactivebrokers.github.io/tws-api/
- **ib_insync**: https://ib-insync.readthedocs.io/
- **CME Group Settlement**: https://www.cmegroup.com/clearing/operations-and-deliveries/settlement.html
- **CME Contract Specs**: https://www.cmegroup.com/trading/products/
- **CME Holiday Calendar**: https://www.cmegroup.com/tools-information/holiday-calendar.html
- **SPAN Margin**: https://www.cmegroup.com/clearing/risk-management/span-methodology.html

### Roadmap (Phase 4+)

**Next Steps**:

- ✅ Phase 3A: Funding Rate Mechanics (Binance perpetuals) -- DONE
- ✅ Phase 3B: IB Adapters & CME Settlement -- DONE
- ✅ Phase 4A: L2 Execution Provider (Crypto Futures Slippage) -- DONE
- ✅ Phase 4B: CME SPAN Margin & Slippage -- DONE
- ✅ Phase 5A: L3 LOB Integration for Crypto Futures -- DONE
- ✅ Phase 5B: L3 LOB for CME Futures -- DONE
- ✅ Phase 6A: Crypto Futures Risk Management -- DONE
- 📋 Phase 6B: CME Futures Risk Management
- 📋 Phase 7: Training & Backtesting Integration

---

## 📊 Phase 4A: L2 Execution Provider for Crypto Futures (COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 54/54 (at documentation time; verify via CI) | **Date**: 2025-12-02

Phase 4A extends the crypto parametric TCA model with futures-specific factors for Binance USDT-M perpetuals.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **FuturesSlippageProvider** | `execution_providers_futures.py` | L2+ slippage with funding/liquidation/OI factors |
| **FuturesFeeProvider** | `execution_providers_futures.py` | Maker/taker/liquidation fees + funding payments |
| **FuturesL2ExecutionProvider** | `execution_providers_futures.py` | Combined execution provider |
| **Тесты** | `tests/test_futures_execution_providers.py` | 54 comprehensive tests |

### Futures-Specific Factors

#### 1. Funding Rate Stress

- **Formula**: `funding_stress = 1.0 + abs(funding_rate) × sensitivity`
- **Default sensitivity**: 5.0
- **Example**: 0.01% funding → 0.05% slippage increase
- **Direction**: Only applies when trading in same direction as funding (crowded position)

#### 2. Liquidation Cascade

- **Formula**: `cascade_factor = min(max_factor, 1.0 + (liquidations/ADV) × sensitivity)`
- **Default sensitivity**: 5.0
- **Max cap**: 3.0x (200% increase)
- **Threshold**: 1% of ADV
- **Example**: 2% liquidations → 10% slippage increase (capped at 200%)

#### 3. Open Interest Penalty

- **Formula**: `oi_penalty = min(max_penalty, 1.0 + (OI/ADV - 1.0) × factor)`
- **Default factor**: 0.1
- **Max cap**: 2.0x (100% increase)
- **Trigger**: OI > ADV
- **Example**: OI = 3× ADV → 20% slippage increase (capped at 100%)

### Total Slippage Formula

```python
total_slippage = base_slippage
    × (1.0 + funding_rate × sensitivity)           # Funding stress
    × min(3.0, 1.0 + liq_ratio × cascade_sens)     # Cascade (capped)
    × min(2.0, 1.0 + (oi/adv - 1.0) × oi_factor)  # OI penalty (capped)
```

**Realistic Example**:

- Base slippage: 8 bps (from crypto model)
- Funding: 0.01% × 5.0 = 0.05% increase → × 1.0005
- Liquidations: 2% × 5.0 = 10% increase → × 1.10
- OI: 3× ADV → × 1.20
- **Total**: 8 × 1.0005 × 1.10 × 1.20 ≈ **10.6 bps** ✅

### Fee Structure (Binance USDT-M)

| Fee Type | Rate | Notes |
|----------|------|-------|
| Maker | 2 bps (0.02%) | Passive liquidity provision |
| Taker | 4 bps (0.04%) | Aggressive execution |
| Liquidation | 50 bps (0.5%) | Goes to insurance fund |

### Funding Payment

**Formula**: `payment = position_notional × funding_rate`

- **Positive funding**: Longs pay shorts
- **Negative funding**: Shorts pay longs

**Example**:

```python
# Long 1 BTC at $50,000, funding = +0.01%
payment = 50,000 × 1.0 × 0.0001 = $5.00 (paid by long)

# Short 1 BTC at $50,000, funding = +0.01%
payment = 50,000 × 1.0 × 0.0001 = $5.00 (received by short)
```

### Configuration

```python
from execution_providers_futures import FuturesSlippageConfig, create_futures_execution_provider

# Default configuration
config = FuturesSlippageConfig(
    funding_impact_sensitivity=5.0,
    liquidation_cascade_sensitivity=5.0,
    liquidation_cascade_max_factor=3.0,      # Cap at 200% increase
    open_interest_liquidity_factor=0.1,
    open_interest_max_penalty=2.0,           # Cap at 100% increase
    use_mark_price_execution=True,
)

# Create provider
provider = create_futures_execution_provider(
    use_mark_price=True,
    slippage_config=config,
)
```

### Usage Example

```python
from execution_providers import Order, MarketState, BarData

# Execute order
order = Order("BTCUSDT", "BUY", 0.1, "MARKET")
market = MarketState(timestamp=0, bid=50000.0, ask=50001.0, adv=1e9)
bar = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=1000.0)

fill = provider.execute(
    order=order,
    market=market,
    bar=bar,
    funding_rate=0.0001,            # 0.01% funding
    open_interest=2_000_000_000,    # $2B OI (2× ADV)
    recent_liquidations=10_000_000, # $10M liquidations (1%)
)

print(f"Filled at {fill.price} with {fill.slippage_bps:.2f}bps slippage")
print(f"Fee: ${fill.fee:.2f}")
```

### Factory Integration

```python
from execution_providers import create_execution_provider, AssetClass

# Via factory (automatically uses FuturesSlippageProvider)
provider = create_execution_provider(AssetClass.FUTURES, level="L2")
```

### Тестирование

```bash
# All futures tests (54 tests)
pytest tests/test_futures_execution_providers.py -v

# Coverage: 54 passed, 1 skipped (at documentation time; verify via CI)
```

### Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| FuturesSlippageConfig | 5 | Config validation |
| Funding Stress | 5 | Positive/negative/zero/scaling |
| Liquidation Cascade | 3 | Above/below threshold, scaling, caps |
| Open Interest Penalty | 2 | High/normal OI, caps |
| Combined Factors | 2 | Worst/best case scenarios |
| Liquidation Risk | 3 | Long/short, leverage |
| Fee Computation | 5 | Maker/taker/liquidation |
| Funding Payment | 5 | Long pays/receives, scaling |
| L2 Execution | 4 | Basic/mark price/all factors |
| Factory Functions | 5 | Creation, integration |
| Edge Cases | 7 | None params, zero ADV, bounds |
| Backward Compat | 3 | Protocol compliance |

### Critical Bugs Fixed (2025-12-02)

1. **Funding Stress Formula**: Removed `× 10000` (was 51x, now 1.005x for 0.1% funding) ✅
2. **Liquidation Cascade Cap**: Added max_factor=3.0 to prevent unrealistic extremes ✅
3. **OI Penalty Cap**: Added max_penalty=2.0 to prevent unbounded growth ✅
4. **Syntax Error**: Fixed duplicate docstring in execution_providers.py ✅

### Limitations & Future Work

**Current Scope**:

- ✅ Crypto perpetuals (USDT-M)
- ✅ L2 statistical slippage
- ✅ Mark price execution

**Future Phases**:

- 📋 Quarterly futures expiration handling (Phase 4B)
- 📋 Binance Futures adapters (Phase 5)
- 📋 L3 LOB simulation for futures (Phase 6)
- 📋 Historical data validation vs actual fills

### Референсы

- **Binance Futures**: https://www.binance.com/en/support/faq/360033524991
- **Funding Rate Mechanism**: https://www.binance.com/en/support/faq/360033525031
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Zhao et al. (2020): "Liquidation Cascade Effects in Crypto Markets"
- Cont et al. (2014): "The Price Impact of Order Book Events"

---

## 📊 Phase 4B: CME SPAN Margin & Slippage (COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 258/258 (at documentation time; verify via CI) | **Покрытие**: 99% | **Date**: 2025-12-02

Phase 4B implements CME-specific margin calculation (SPAN methodology) and slippage modeling for CME Group futures.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **SPANMarginCalculator** | `impl_span_margin.py` | SPAN margin calculation with 16-scenario testing |
| **CMESlippageProvider** | `execution_providers_cme.py` | CME-specific slippage with session/settlement factors |
| **CMEFeeProvider** | `execution_providers_cme.py` | Fixed per-contract fee structure |
| **CMECircuitBreaker** | `impl_circuit_breaker.py` | Rule 80B circuit breakers, overnight limits, velocity logic |
| **CircuitBreakerManager** | `impl_circuit_breaker.py` | Multi-product circuit breaker management |

### SPAN Margin Calculator

**SPAN (Standard Portfolio Analysis of Risk)** -- CME's risk-based margin methodology.

**Key Concepts**:

- **Scanning Risk**: Maximum expected loss under 16 stress scenarios
- **Inter-Commodity Credit**: Margin offset for correlated products
- **Intra-Commodity Credit**: Calendar spread credits
- **Delivery Month Charge**: Additional margin near expiration

**Scanning Risk Ranges** (% of notional):

| Product | Range | Volatility Scan |
|---------|-------|-----------------|
| ES (E-mini S&P) | 6% | 30% |
| NQ (E-mini NASDAQ) | 8% | 35% |
| GC (Gold) | 5% | 25% |
| CL (Crude Oil) | 8% | 35% |
| NG (Natural Gas) | 12% | 50% |
| 6E (Euro FX) | 4% | 20% |
| ZN (10-Year Note) | 2% | 15% |

**Inter-Commodity Spread Credits**:

| Pair | Credit Rate | Rationale |
|------|-------------|-----------|
| ES/NQ | 50% | Correlated equity indices |
| ES/YM | 50% | S&P 500 vs Dow correlation |
| GC/SI | 35% | Precious metals correlation |
| MGC/GC | 85% | Micro/Standard same underlying |
| CL/RB/HO | 40% | Crack spread (refining) |

**Usage**:

```python
from impl_span_margin import (
    SPANMarginCalculator,
    create_span_calculator,
    calculate_simple_margin,
)
from core_futures import FuturesPosition, PositionSide, MarginMode

# 1. Create calculator with default specs
calc = create_span_calculator()

# 2. Calculate single position margin
position = FuturesPosition(
    symbol="ES",
    qty=Decimal("2"),
    entry_price=Decimal("4500"),
    side=PositionSide.LONG,
    leverage=1,
    margin_mode=MarginMode.SPAN,
)

result = calc.calculate_margin(
    position=position,
    current_price=Decimal("4500"),
)

print(f"Scanning Risk: ${result.scanning_risk}")
print(f"Initial Margin: ${result.initial_margin}")
print(f"Maintenance Margin: ${result.maintenance_margin}")

# 3. Portfolio margin with spread credits
positions = [es_long, nq_long]  # Correlated positions
portfolio_result = calc.calculate_portfolio_margin(
    positions=positions,
    prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
)

print(f"Inter-commodity Credit: ${portfolio_result.inter_commodity_credit}")
print(f"Net Portfolio Margin: ${portfolio_result.net_portfolio_margin}")

# 4. Margin call detection
call_status = calc.check_margin_call(
    positions=positions,
    prices=prices,
    account_equity=Decimal("50000"),
)
# call_status.call_type: NONE, WARNING, MARGIN_CALL, LIQUIDATION
```

### CME Slippage Provider

**Session-Aware Slippage Model** with CME-specific factors.

**Slippage Factors**:

| Factor | Multiplier | Condition |
|--------|------------|-----------|
| ETH Session | 1.5x | Outside RTH (18:00-17:00 ET) |
| Settlement Period | 1.3x | 15 min before settlement |
| Roll Period | 1.2x | 8 days before expiry |
| Circuit Breaker L1 | 2.0x | -7% decline |
| Circuit Breaker L2 | 5.0x (max) | -13% decline |
| Velocity Pause | 1.5x | Fat-finger protection |

**Default Spreads** (in bps):

| Product | Spread | Impact Coef |
|---------|--------|-------------|
| ES | 0.5 bps | 0.03 |
| NQ | 0.75 bps | 0.04 |
| GC | 1.0 bps | 0.04 |
| CL | 2.0 bps | 0.06 |
| NG | 3.0 bps | 0.08 |
| 6E | 0.5 bps | 0.03 |
| ZN | 0.25 bps | 0.02 |

**Slippage Profiles**:

- `default`: Balanced settings
- `conservative`: Wider spreads, higher impacts
- `aggressive`: Tighter estimates
- `equity_index`: Optimized for ES/NQ
- `metals`: Optimized for GC/SI
- `energy`: Optimized for CL/NG

**Usage**:

```python
from execution_providers_cme import (
    create_cme_slippage_provider,
    create_cme_execution_provider,
    CMESlippageProvider,
)
from execution_providers import Order, MarketState, BarData

# 1. Create from profile
provider = CMESlippageProvider.from_profile("equity_index")

# 2. Compute slippage
slippage_bps = provider.compute_slippage_bps(
    order=Order("ES", "BUY", 5.0, "MARKET"),
    market=MarketState(timestamp=0, bid=4500.0, ask=4500.25, adv=2e9),
    participation_ratio=0.001,
    is_eth_session=False,
    is_settlement_period=False,
    circuit_breaker_level=CircuitBreakerLevel.NONE,
)

# 3. Full execution provider
exec_provider = create_cme_execution_provider(profile="default")
fill = exec_provider.execute(order, market, bar)
```

### CME Fee Provider

**Fixed Per-Contract Fees** (no maker/taker distinction):

| Product | Fee per Contract | Exchange |
|---------|------------------|----------|
| ES | $1.29 | CME |
| NQ | $1.29 | CME |
| GC | $1.60 | COMEX |
| SI | $1.60 | COMEX |
| CL | $1.50 | NYMEX |
| NG | $1.50 | NYMEX |
| 6E | $1.00 | CME |
| ZN | $0.85 | CBOT |

### CME Circuit Breaker (Rule 80B)

**Equity Index Circuit Breakers** (ES, NQ, YM, RTY):

| Level | Trigger | Halt Duration | Time Restriction |
|-------|---------|---------------|------------------|
| Level 1 | -7% | 15 minutes | Before 15:25 ET only |
| Level 2 | -13% | 15 minutes | Before 15:25 ET only |
| Level 3 | -20% | Remainder of day | Any time |

**Overnight Price Limits** (ETH only):

| Product | Limit | Note |
|---------|-------|------|
| ES, NQ, YM, RTY | ±5% | From prior settlement |

**Commodity Daily Price Limits**:

| Product | Initial | Expanded | Notes |
|---------|---------|----------|-------|
| CL | ±$10 | ±$15, ±$20 | Consecutive limit days |
| NG | ±$3 | ±$4.50, ±$6 | Expansion mechanism |
| GC | ±$100 | ±$150, ±$200 | COMEX metals |

**Velocity Logic** (Fat-Finger Protection):

| Product | Threshold (ticks) | Pause Duration |
|---------|-------------------|----------------|
| ES | 12 | 2 seconds |
| NQ | 20 | 2 seconds |
| GC | 50 | 2 seconds |
| CL | 100 | 2 seconds |

**Usage**:

```python
from impl_circuit_breaker import (
    CMECircuitBreaker,
    CircuitBreakerManager,
    CircuitBreakerLevel,
    create_circuit_breaker,
)

# 1. Single product circuit breaker
cb = create_circuit_breaker("ES", reference_price=Decimal("4500"))

# 2. Check circuit breaker status
level = cb.check_circuit_breaker(
    current_price=Decimal("4185"),  # -7%
    timestamp_ms=int(time.time() * 1000),
    is_rth=True,
)
# level = CircuitBreakerLevel.LEVEL_1

# 3. Check if trading allowed
can_trade, reason = cb.can_trade()
# can_trade = False, reason = "Circuit breaker Level 1 halt"

# 4. Get halt end time
halt_end = cb.get_halt_end_time()

# 5. Multi-product manager
manager = CircuitBreakerManager()
manager.add_product("ES", reference_price=Decimal("4500"))
manager.add_product("NQ", reference_price=Decimal("15000"))

status = manager.check_all(
    prices={"ES": Decimal("4185"), "NQ": Decimal("13900")},
    timestamp_ms=now_ms,
    is_rth=True,
)
# status = {
#     "ES": {"level": "LEVEL_1", "can_trade": False},
#     "NQ": {"level": "NONE", "can_trade": True},
# }

# 6. Daily reset
manager.reset_all_daily()
```

### Тестирование

```bash
# All Phase 4B tests (258 tests, 99% coverage)
pytest tests/test_span_margin.py tests/test_cme_slippage.py tests/test_circuit_breaker.py -v

# By component
pytest tests/test_span_margin.py -v          # 85 tests (78 + 7 edge cases)
pytest tests/test_cme_slippage.py -v         # 66 tests (55 + 11 edge cases)
pytest tests/test_circuit_breaker.py -v      # 67 tests (60 + 7 edge cases)
```

### Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| SPAN Scanning Risk | 9 | Product-specific ranges |
| SPAN Portfolio Margin | 7 | Spread credits |
| SPAN Margin Impact | 3 | New position impact estimation |
| SPAN Edge Cases | 5 | Missing specs/prices, fallbacks |
| CME Slippage Profiles | 6 | Profile configurations |
| CME Session Factors | 5 | ETH/settlement/roll |
| CME Limit Orders | 6 | Passive/aggressive/no-fill |
| CME Edge Cases | 5 | Currency futures, notes |
| CME Circuit Breaker | 20 | Rule 80B, overnight limits |
| Velocity Logic | 7 | Fat-finger protection |
| Circuit Breaker Manager | 6 | Multi-product management |
| Circuit Breaker Edge Cases | 7 | Expanded limits, non-equity products |
| Integration Scenarios | 5 | Flash crash, overnight trading |

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `impl_span_margin.py` | SPAN margin calculator (~1050 lines) |
| `execution_providers_cme.py` | CME slippage/fee providers (~800 lines) |
| `impl_circuit_breaker.py` | Circuit breaker simulation (~700 lines) |
| `tests/test_span_margin.py` | 78 SPAN margin tests |
| `tests/test_cme_slippage.py` | 55 CME slippage tests |
| `tests/test_circuit_breaker.py` | 60 circuit breaker tests |

### Референсы

- **CME SPAN Methodology**: https://www.cmegroup.com/clearing/risk-management/span-methodology.html
- **CME Rule 80B**: https://www.cmegroup.com/rulebook/CME/I/5/5.html
- **CME Globex Price Limits**: https://www.cmegroup.com/trading/equity-index/price-limit-guide.html
- **CME Velocity Logic**: https://www.cmegroup.com/confluence/display/EPICSANDBOX/Velocity+Logic

---

## 📊 Phase 5A: L3 LOB Integration for Crypto Futures (COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 100/100 (at documentation time; verify via CI) | **Date**: 2025-12-02

Phase 5A integrates L3 Limit Order Book simulation with crypto perpetual futures, adding liquidation cascade simulation, insurance fund dynamics, ADL queue management, and funding period-aware execution.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **LiquidationOrderStream** | `lob/futures_extensions.py` | Liquidation order injection into LOB |
| **LiquidationCascadeSimulator** | `lob/futures_extensions.py` | Kyle price impact cascade simulation |
| **InsuranceFundManager** | `lob/futures_extensions.py` | Insurance fund contribution/payout dynamics |
| **ADLQueueManager** | `lob/futures_extensions.py` | Auto-Deleveraging queue management |
| **FundingPeriodDynamics** | `lob/futures_extensions.py` | Queue behavior near funding times |
| **FuturesL3SlippageProvider** | `execution_providers_futures_l3.py` | L3 slippage with cascade/funding factors |
| **FuturesL3FillProvider** | `execution_providers_futures_l3.py` | L3 fill logic with liquidation injection |
| **FuturesL3ExecutionProvider** | `execution_providers_futures_l3.py` | Combined L3 futures execution provider |
| **Тесты** | `tests/test_futures_l3_execution.py` | 100 comprehensive tests |

### Key Concepts

#### 1. Liquidation Cascade Simulation (Kyle Price Impact)

Based on Kyle (1985) λ-model: `ΔP = λ × sign(x) × |x|`

**Cascade Mechanics**:

- **Wave Decay**: Each subsequent liquidation wave is dampened by `cascade_decay` factor (default: 0.7)
- **Price Impact**: Cumulative impact follows `impact_coef × √(liquidation_volume / ADV)`
- **Max Waves**: Configurable limit (default: 5) to prevent infinite cascade loops
- **Phases**: INITIAL → PROPAGATING → DAMPENING → ENDED

**Usage**:

```python
from lob.futures_extensions import (
    LiquidationCascadeSimulator,
    create_cascade_simulator,
)

# Create simulator
simulator = create_cascade_simulator(
    price_impact_coef=0.5,  # Kyle λ coefficient
    cascade_decay=0.7,       # Wave dampening factor
    max_waves=5,
)

# Simulate cascade
result = simulator.simulate_cascade(
    initial_liquidation_volume=1_000_000,
    market_price=50000.0,
    adv=500_000_000,
)

print(f"Total waves: {len(result.waves)}")
print(f"Total liquidated: ${result.total_liquidated_volume:,.0f}")
print(f"Final price impact: {result.total_price_impact_bps:.2f} bps")
```

#### 2. Insurance Fund Dynamics

**Fund Flow**:

- **Profit liquidation** → Contribution to fund (bankruptcy - fill > 0)
- **Loss liquidation** → Payout from fund (fill - bankruptcy > 0)
- **Fund depletion** → Triggers ADL mechanism

**Usage**:

```python
from lob.futures_extensions import (
    InsuranceFundManager,
    create_insurance_fund,
    LiquidationFillResult,
)

fund = create_insurance_fund(initial_balance=10_000_000)

# Process liquidation
result = fund.process_liquidation(
    liquidation_info=liq_order,
    fill_price=49500.0,
)

print(f"Contribution: ${result.contribution:.2f}")
print(f"Payout: ${result.payout:.2f}")
print(f"Fund balance: ${fund.get_state().current_balance:,.0f}")
```

#### 3. ADL (Auto-Deleveraging) Queue

**Ranking Formula**: `ADL_Score = PnL% × Leverage`

Higher score = higher priority for deleveraging.

**Usage**:

```python
from lob.futures_extensions import (
    ADLQueueManager,
    create_adl_manager,
)

adl_manager = create_adl_manager()

# Build queue from positions
positions = [
    {"address": "user1", "pnl_pct": 0.15, "leverage": 20, "side": "long", "size": 1000},
    {"address": "user2", "pnl_pct": 0.10, "leverage": 10, "side": "long", "size": 2000},
]
adl_manager.build_queue(positions, side="long")

# Get candidates for deleveraging
candidates = adl_manager.get_adl_candidates(
    side="long",
    required_amount=500,
)
```

#### 4. Funding Period Dynamics

**Queue Behavior Near Funding**:

- Spread widens (arbitrageurs exit)
- Liquidity decreases (position rebalancing)
- Volatility increases

**Usage**:

```python
from lob.futures_extensions import (
    FundingPeriodDynamics,
    create_funding_dynamics,
)

dynamics = create_funding_dynamics(
    funding_times_utc=[0, 8, 16],  # 00:00, 08:00, 16:00 UTC
    window_minutes_before=5,
    window_minutes_after=1,
)

state = dynamics.get_state(
    timestamp_ms=current_time_ms,
    funding_rate=0.0001,
)

print(f"In funding window: {state.in_funding_window}")
print(f"Spread multiplier: {state.spread_multiplier:.2f}")
print(f"Queue priority factor: {state.queue_priority_factor:.2f}")
```

### Configuration

```python
from execution_providers_futures_l3 import (
    FuturesL3Config,
    create_futures_l3_config,
)

config = FuturesL3Config(
    # Cascade parameters
    price_impact_coef=0.5,
    cascade_decay=0.7,
    max_cascade_waves=5,

    # Insurance fund
    initial_insurance_fund=10_000_000,
    adl_trigger_threshold=0.1,

    # Funding
    funding_times_utc=[0, 8, 16],
    funding_window_minutes_before=5,
    funding_window_minutes_after=1,
    funding_spread_multiplier_max=1.5,
    funding_queue_priority_factor=0.8,

    # Execution
    use_mark_price_execution=True,
)
```

### Presets

| Preset | Cascade Decay | Max Waves | Impact Coef | Use Case |
|--------|---------------|-----------|-------------|----------|
| `default` | 0.7 | 5 | 0.5 | General simulation |
| `conservative` | 0.6 | 3 | 0.7 | Conservative estimates |
| `fast` | 0.8 | 3 | 0.3 | Faster simulations |
| `stress_test` | 0.5 | 10 | 1.0 | Extreme market conditions |

**Usage**:

```python
from execution_providers_futures_l3 import (
    FuturesL3ExecutionProvider,
    create_futures_l3_execution_provider,
)

# From preset
provider = FuturesL3ExecutionProvider.from_preset("stress_test")

# Or via factory
provider = create_futures_l3_execution_provider(preset="conservative")
```

### Integration with L3 LOB

The FuturesL3ExecutionProvider integrates with the existing L3 LOB infrastructure:

```python
from lob import MatchingEngine, OrderBook
from execution_providers_futures_l3 import create_futures_l3_execution_provider

# Create provider
provider = create_futures_l3_execution_provider(preset="default")

# Load historical liquidation data
provider.load_liquidation_data(liquidation_events_list)

# Execute with full LOB simulation
fill = provider.execute(
    order=order,
    market=market_state,
    bar=bar_data,
    order_book=lob_order_book,
    matching_engine=matching_engine,
    funding_rate=0.0001,
    open_interest=2_000_000_000,
    recent_liquidations=10_000_000,
    positions=current_positions,
)
```

### Тестирование

```bash
# All Phase 5A tests (100 tests)
pytest tests/test_futures_l3_execution.py -v

# By category
pytest tests/test_futures_l3_execution.py::TestLiquidationCascadeSimulator -v
pytest tests/test_futures_l3_execution.py::TestInsuranceFundManager -v
pytest tests/test_futures_l3_execution.py::TestADLQueueManager -v
pytest tests/test_futures_l3_execution.py::TestFundingPeriodDynamics -v
pytest tests/test_futures_l3_execution.py::TestFuturesL3ExecutionProvider -v
pytest tests/test_futures_l3_execution.py::TestIntegration -v
```

**Coverage**: 100 tests (at documentation time; verify via CI)

| Category | Tests | Coverage |
|----------|-------|----------|
| Enums | 3 | LiquidationType, ADLRank, CascadePhase |
| LiquidationOrderInfo | 5 | Creation, properties, defaults |
| LiquidationFillResult | 2 | Filled/unfilled results |
| CascadeResult | 4 | Depth, phases |
| InsuranceFundState | 2 | Depletion, utilization |
| LiquidationOrderStream | 10 | Event handling, filtering, stats |
| LiquidationCascadeSimulator | 6 | Cascade simulation, price impact |
| InsuranceFundManager | 10 | Contributions, payouts, ADL trigger |
| ADLQueueManager | 7 | Queue building, ranking, candidates |
| FundingPeriodDynamics | 6 | Window detection, multipliers |
| FuturesL3Config | 6 | Validation, defaults |
| FuturesL3SlippageProvider | 5 | Base slippage, funding, cascade |
| FuturesL3FillProvider | 4 | Fill tracking, liquidation injection |
| FuturesL3ExecutionProvider | 10 | Full execution flow |
| Factory Functions | 2 | Config and provider creation |
| Presets | 5 | All preset configurations |
| Integration | 3 | Full flow, cascade recovery, fund depletion |
| Edge Cases | 4 | Empty orders, extreme funding, zero ADV |

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `lob/futures_extensions.py` | LOB extensions for crypto futures (~1300 lines) |
| `execution_providers_futures_l3.py` | L3 futures execution provider (~1100 lines) |
| `tests/test_futures_l3_execution.py` | 100 comprehensive tests |

### Референсы

- Kyle (1985): "Continuous Auctions and Insider Trading" -- Price impact model
- Almgren & Chriss (2001): "Optimal Execution" -- Market impact theory
- Binance: "Liquidation Protocol" -- Insurance fund and ADL mechanics
- Binance: "Funding Rate" -- 8-hour funding periods
- FTX Research: "Liquidation Cascades" -- Cascade dynamics (pre-collapse research)

---

## 📊 Phase 5B: L3 LOB for CME Futures (COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 42/42 (at documentation time; verify via CI) | **Date**: 2025-12-02

Phase 5B implements L3 Limit Order Book simulation for CME Group futures, including Globex-style FIFO matching, Market with Protection (MWP) orders, stop orders with velocity logic, and daily settlement simulation.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **GlobexMatchingEngine** | `lob/cme_matching.py` | CME Globex-style FIFO matching engine |
| **CMEL3SlippageProvider** | `execution_providers_cme_l3.py` | L3 slippage with LOB walk-through |
| **CMEL3FillProvider** | `execution_providers_cme_l3.py` | L3 fill logic with matching engine |
| **CMEL3ExecutionProvider** | `execution_providers_cme_l3.py` | Combined L3 CME execution provider |
| **DailySettlementSimulator** | `execution_providers_cme_l3.py` | Daily variation margin simulation |
| **Тесты** | `tests/test_cme_l3_execution.py` | 42 comprehensive tests |

### Key Concepts

#### 1. Globex-Style FIFO Matching

CME Globex uses strict Price-Time Priority (FIFO) matching:

```
BUY orders sorted: price DESC, time ASC (best price first, oldest first)
SELL orders sorted: price ASC, time ASC (best price first, oldest first)
```

**Usage**:

```python
from lob.cme_matching import GlobexMatchingEngine, StopOrder
from lob.data_structures import LimitOrder, Side, OrderType

# Create engine for ES (E-mini S&P 500)
engine = GlobexMatchingEngine(symbol="ES", tick_size=0.25, protection_points=6)

# Add resting order
resting = LimitOrder(
    order_id="rest_1",
    price=4500.0,
    qty=10.0,
    remaining_qty=10.0,
    timestamp_ns=0,
    side=Side.BUY,
    order_type=OrderType.LIMIT,
)
engine.add_resting_order(resting)

# Match aggressive order
aggressive = LimitOrder(
    order_id="aggr_1",
    price=4500.0,
    qty=5.0,
    remaining_qty=5.0,
    timestamp_ns=1000,
    side=Side.SELL,
    order_type=OrderType.MARKET,
)
result = engine.match(aggressive)
print(f"Filled: {result.total_filled_qty} @ {result.avg_fill_price}")
```

#### 2. Market with Protection (MWP) Orders

CME uses implicit price limits on market orders to prevent runaway fills:

| Product | Protection Points | Tick Size | Max Deviation |
|---------|-------------------|-----------|---------------|
| ES | 6 | 0.25 | 1.5 points |
| NQ | 10 | 0.25 | 2.5 points |
| GC | 50 | 0.10 | 5.0 points |
| CL | 100 | 0.01 | 1.0 point |

**MWP Behavior**:

- BUY MWP: Limit at best_ask + (protection_points × tick_size)
- SELL MWP: Limit at best_bid - (protection_points × tick_size)
- Unfilled portion is cancelled (not rested)

**Usage**:

```python
result = engine.match_with_protection(
    order=market_order,
    protection_points=6,  # Optional override
)
if result.cancelled_orders:
    print("Unfilled portion cancelled due to protection limit")
```

#### 3. Stop Orders with Velocity Logic

Stop orders trigger when price crosses the stop price, with CME velocity logic protection:

| Product | Velocity Threshold (ticks) | Pause Duration |
|---------|---------------------------|----------------|
| ES | 12 | 2 seconds |
| NQ | 20 | 2 seconds |
| GC | 50 | 2 seconds |
| CL | 100 | 2 seconds |

**Stop Order Types**:

- **Stop-Market**: Converts to MWP when triggered
- **Stop-Limit**: Converts to limit order when triggered

**Usage**:

```python
stop = StopOrder(
    order_id="stop_1",
    symbol="ES",
    side=Side.SELL,
    qty=5.0,
    stop_price=4490.0,
    limit_price=None,  # Stop-market
    use_protection=True,
)
engine.submit_stop_order(stop)

# Check and trigger stops
results = engine.check_stop_triggers(
    last_trade_price=4489.0,
    bid=4488.5,
    ask=4489.5,
    timestamp_ns=int(time.time() * 1e9),
)
```

#### 4. Session Detection

RTH (Regular Trading Hours) vs ETH (Electronic Trading Hours):

| Session | Hours (ET) | Spread Multiplier |
|---------|------------|-------------------|
| RTH | 9:30 - 16:15 | 1.0x |
| ETH | 18:00 - 9:30 | 1.5x |
| Pre-Open | 8:30 - 9:30 | 1.25x |
| Maintenance | 16:15 - 16:30 | N/A (closed) |

**Usage**:

```python
from execution_providers_cme_l3 import (
    detect_cme_session,
    is_rth_session,
    get_minutes_to_settlement,
    CMESession,
)

session = detect_cme_session(timestamp_ms)
if session == CMESession.RTH:
    print("Regular trading hours - tightest spreads")
elif session == CMESession.ETH:
    print("Electronic hours - wider spreads")
elif session == CMESession.MAINTENANCE:
    print("Market closed for daily maintenance")

# Check if RTH
if is_rth_session(timestamp_ms):
    spread_mult = 1.0

# Minutes until settlement
minutes = get_minutes_to_settlement(timestamp_ms, "ES")
if minutes and minutes < 30:
    print(f"Settlement approaching in {minutes} minutes")
```

#### 5. Daily Settlement Simulation

CME futures settle daily with variation margin:

**Settlement Times (Eastern Time)**:

| Product | Settlement Time | Notes |
|---------|-----------------|-------|
| ES, NQ, YM, RTY | 16:00 ET | Equity index |
| GC, SI, HG | 13:30 ET | Metals (COMEX) |
| CL, NG | 14:30 ET | Energy (NYMEX) |
| 6E, 6J, 6B | 15:00 ET | Currencies |

**Variation Margin Formula**:

```
VM = (Settlement_t - Settlement_t-1) × Qty × Multiplier
```

**Usage**:

```python
from execution_providers_cme_l3 import DailySettlementSimulator
from decimal import Decimal

simulator = DailySettlementSimulator(
    symbol="ES",
    contract_multiplier=Decimal("50"),
)

# Process settlement
simulator.process_settlement(
    timestamp_ms=settlement_time_ms,
    settlement_price=Decimal("4520.00"),
    position_qty=Decimal("2"),
)

# Get variation margin
vm = simulator.get_pending_variation_margin()
print(f"Variation Margin: ${vm}")

# Get last settlement price
last_price = simulator.get_last_settlement_price()
```

### Configuration

```python
from execution_providers_cme_l3 import (
    CMEL3ExecutionProvider,
    create_cme_l3_execution_provider,
    CMEL3Config,
)

# Create with default config
provider = create_cme_l3_execution_provider(symbol="ES")

# Create with profile
provider = create_cme_l3_execution_provider(
    symbol="ES",
    profile="conservative",
)

# Custom configuration
config = CMEL3Config(
    spread_bps=0.5,
    eth_spread_multiplier=1.5,
    settlement_premium=1.3,
    impact_coef=0.03,
)
provider = CMEL3ExecutionProvider(symbol="ES", config=config)
```

### Presets

| Preset | Spread (bps) | ETH Mult | Settlement Mult | Impact Coef |
|--------|--------------|----------|-----------------|-------------|
| `default` | 0.5 | 1.5 | 1.3 | 0.03 |
| `conservative` | 0.75 | 1.75 | 1.5 | 0.05 |
| `aggressive` | 0.35 | 1.25 | 1.15 | 0.02 |

### Тестирование

```bash
# All Phase 5B tests (42 tests)
pytest tests/test_cme_l3_execution.py -v

# By category
pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineBasic -v
pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineMWP -v
pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineStops -v
pytest tests/test_cme_l3_execution.py::TestSessionDetection -v
pytest tests/test_cme_l3_execution.py::TestDailySettlementSimulator -v
pytest tests/test_cme_l3_execution.py::TestCMEL3SlippageProvider -v
pytest tests/test_cme_l3_execution.py::TestCMEL3FillProvider -v
pytest tests/test_cme_l3_execution.py::TestIntegration -v
```

**Coverage**: 42 tests (at documentation time; verify via CI)

| Category | Tests | Coverage |
|----------|-------|----------|
| GlobexMatchingEngine Basic | 8 | FIFO matching, best bid/ask |
| MWP Orders | 3 | Protection limits, unfilled cancellation |
| Stop Orders | 5 | Trigger logic, stop-limit, velocity |
| Session Detection | 5 | RTH/ETH, settlement time |
| Daily Settlement | 7 | VM calculation, long/short positions |
| Slippage Provider | 4 | LOB walk, ETH multiplier, settlement |
| Fill Provider | 2 | Market order fills |
| Factory Functions | 3 | Profiles, creation |
| Edge Cases | 3 | Empty book, zero qty, various symbols |
| Integration | 2 | Full execution flow, settlement |

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `lob/cme_matching.py` | GlobexMatchingEngine with MWP, stops, velocity (~800 lines) |
| `execution_providers_cme_l3.py` | L3 CME execution provider (~700 lines) |
| `tests/test_cme_l3_execution.py` | 42 comprehensive tests |

### Референсы

- CME Group: "Globex Matching Algorithm" -- FIFO Price-Time Priority
- CME Group: "Market with Protection Orders" -- MWP order handling
- CME Group: "Stop Spike Logic" -- Velocity logic protection
- CME Group: "Daily Settlement Procedures" -- Variation margin
- CME Group: "Globex Trading Hours" -- RTH/ETH session definitions

---

## 🛡️ Phase 6A: Crypto Futures Risk Management (COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 101/101 (at documentation time; verify via CI) | **Date**: 2025-12-02

Phase 6A implements comprehensive risk management for crypto perpetual futures (Binance USDT-M), including leverage guards, margin monitoring, funding exposure, position concentration limits, and ADL risk tracking.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **FuturesLeverageGuard** | `services/futures_risk_guards.py` | Tiered leverage enforcement with Binance brackets |
| **FuturesMarginGuard** | `services/futures_risk_guards.py` | Margin ratio monitoring with 5 levels |
| **MarginCallNotifier** | `services/futures_risk_guards.py` | Margin call notifications with cooldowns |
| **FundingExposureGuard** | `services/futures_risk_guards.py` | Funding rate risk monitoring |
| **ConcentrationGuard** | `services/futures_risk_guards.py` | Position concentration limits |
| **ADLRiskGuard** | `services/futures_risk_guards.py` | Auto-Deleveraging queue risk |
| **CryptoFuturesRiskGuard** | `risk_guard.py` | Unified guard integration |
| **Тесты** | `tests/test_futures_risk_guards.py` | 101 comprehensive tests |

### Key Concepts

#### 1. Leverage Tiering (Binance USDT-M)

Higher notional positions get lower max leverage:

| Notional (USD) | BTC Max | ETH Max | Other Max |
|----------------|---------|---------|-----------|
| < $50,000 | 125x | 100x | 75x |
| $50K-250K | 100x | 75x | 50x |
| $250K-1M | 50x | 50x | 25x |
| $1M-5M | 20x | 25x | 10x |
| $5M-20M | 10x | 10x | 5x |
| > $20M | 5x | 5x | 3x |

**Usage**:

```python
from services.futures_risk_guards import FuturesLeverageGuard, LeverageCheckResult

guard = FuturesLeverageGuard(
    max_account_leverage=20,
    max_symbol_leverage=125,
    concentration_limit=0.5,  # Max 50% in single symbol
)

result = guard.validate_new_position(
    proposed_position=position,
    current_positions=existing_positions,
    account_balance=Decimal("10000"),
)

if not result.is_valid:
    print(f"Blocked: {result.error_message}")
    print(f"Suggested leverage: {result.suggested_leverage}")
```

#### 2. Margin Status Levels

| Level | Margin Ratio | Action |
|-------|--------------|--------|
| **HEALTHY** | ≥ 1.5 (150%) | No action |
| **WARNING** | 1.2-1.5 (120-150%) | Alert |
| **DANGER** | 1.05-1.2 (105-120%) | Reduce position |
| **CRITICAL** | 1.0-1.05 (100-105%) | Urgent action |
| **LIQUIDATION** | ≤ 1.0 (100%) | Immediate liquidation risk |

**Usage**:

```python
from services.futures_risk_guards import (
    FuturesMarginGuard,
    MarginStatus,
    MarginCallLevel,
)
from decimal import Decimal

guard = FuturesMarginGuard(
    margin_calculator=None,  # Optional calculator
    warning_level=Decimal("1.5"),
    danger_level=Decimal("1.2"),
    critical_level=Decimal("1.05"),
)

# Check pre-calculated margin ratio
result = guard.check_margin_ratio(
    margin_ratio=1.35,  # 135%
    account_equity=10000.0,
    total_margin_used=7407.0,
    symbol="BTCUSDT",
)

print(f"Status: {result.status}")  # MarginStatus.WARNING
print(f"Requires reduction: {result.requires_reduction}")  # False
print(f"Requires liquidation: {result.requires_liquidation}")  # False
```

#### 3. Margin Call Notifications

```python
from services.futures_risk_guards import MarginCallNotifier, MarginCallEvent

notifier = MarginCallNotifier(
    cooldown_seconds=300,  # 5 minute cooldown between alerts
    callback=send_alert_function,  # Optional callback
)

# Check and notify
event = notifier.check_and_notify(
    margin_result=margin_result,
    position=position,
    mark_price=Decimal("50000"),
    wallet_balance=Decimal("10000"),
)

if event:
    print(f"Alert: {event.level.value} - {event.recommended_action}")
    print(f"Shortfall: ${event.shortfall}")
```

#### 4. Funding Rate Exposure

Monitors exposure to funding payments (every 8 hours):

| Level | Annual Rate | Action |
|-------|-------------|--------|
| **NORMAL** | < 10% APR | No action |
| **WARNING** | 10-25% APR | Monitor |
| **EXCESSIVE** | 25-50% APR | Consider reducing |
| **EXTREME** | > 50% APR | Reduce immediately |

**Usage**:

```python
from services.futures_risk_guards import FundingExposureGuard

guard = FundingExposureGuard(
    warning_threshold=Decimal("0.0001"),  # 0.01% per 8h
)

result = guard.check_funding_exposure(
    funding_rate=Decimal("0.0005"),  # 0.05% per 8h = ~54% APR
    position_side="LONG",
    position_notional=Decimal("100000"),
)

print(f"Level: {result.level}")  # EXTREME
print(f"APR: {result.annualized_rate:.1%}")  # 54.8%
print(f"Daily cost: ${result.daily_cost}")
```

#### 5. Position Concentration

```python
from services.futures_risk_guards import ConcentrationGuard

guard = ConcentrationGuard(
    single_symbol_limit=0.5,     # Max 50% in any symbol
    correlated_group_limit=0.7,  # Max 70% in correlated group
    correlation_groups={
        "BTC-ALTS": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        "STABLE": ["USDCUSDT", "BUSDUSDT"],
    },
)

result = guard.check_concentration(
    positions={"BTCUSDT": 60000, "ETHUSDT": 30000, "SOLUSDT": 10000},
    total_exposure=100000,
)

if not result.is_valid:
    print(f"Concentration exceeded: {result.largest_concentration:.1%}")
```

#### 6. ADL Risk Tracking

Auto-Deleveraging queue risk based on PnL × Leverage ranking:

| Level | ADL Percentile | Risk |
|-------|----------------|------|
| **LOW** | < 50% | Minimal ADL risk |
| **MEDIUM** | 50-75% | Monitor |
| **HIGH** | 75-90% | Consider reducing |
| **CRITICAL** | > 90% | High ADL risk |

**Usage**:

```python
from services.futures_risk_guards import ADLRiskGuard

guard = ADLRiskGuard(
    warning_percentile=75.0,
    critical_percentile=90.0,
)

result = guard.check_adl_risk(
    position_pnl_percentile=85.0,  # Top 15% profitable
    position_leverage_percentile=80.0,  # Top 20% leveraged
)

print(f"ADL Level: {result.level}")  # HIGH
print(f"ADL Score: {result.adl_score:.1f}")  # 85 × 80 / 100 = 68
```

### Integration with risk_guard.py

```python
from risk_guard import create_crypto_futures_risk_guard, CryptoFuturesRiskConfig

config = CryptoFuturesRiskConfig(
    market_type="CRYPTO_FUTURES",
    max_account_leverage=20.0,
    max_single_symbol_pct=0.5,
    max_correlated_group_pct=0.7,
    margin_warning_threshold=1.5,
    margin_danger_threshold=1.2,
    margin_critical_threshold=1.05,
    funding_rate_warning_threshold=0.0001,
    adl_warning_percentile=75.0,
    adl_critical_percentile=90.0,
    strict_mode=True,
)

guard = create_crypto_futures_risk_guard(config)

# Check trade
event = guard.check_trade(
    symbol="BTCUSDT",
    side="LONG",
    quantity=0.1,
    leverage=10,
    mark_price=50000.0,
    account_equity=10000.0,
)

if event != RiskEvent.NONE:
    print(f"Risk event: {event.value}")
    print(f"Reason: {guard.get_last_event_reason()}")
```

### Тестирование

```bash
# All Phase 6A tests (101 tests)
pytest tests/test_futures_risk_guards.py -v

# By category
pytest tests/test_futures_risk_guards.py::TestFuturesLeverageGuard -v
pytest tests/test_futures_risk_guards.py::TestFuturesMarginGuard -v
pytest tests/test_futures_risk_guards.py::TestMarginCallNotifier -v
pytest tests/test_futures_risk_guards.py::TestFundingExposureGuard -v
pytest tests/test_futures_risk_guards.py::TestConcentrationGuard -v
pytest tests/test_futures_risk_guards.py::TestADLRiskGuard -v
pytest tests/test_futures_risk_guards.py::TestCryptoFuturesRiskGuard -v
pytest tests/test_futures_risk_guards.py::TestThreadSafety -v
pytest tests/test_futures_risk_guards.py::TestIntegrationScenarios -v
```

**Coverage**: 101 tests (at documentation time; verify via CI)

| Category | Tests | Coverage |
|----------|-------|----------|
| Enums & Constants | 7 | MarginCallLevel, MarginStatus, etc. |
| Config Classes | 6 | Leverage, Margin, Notifier, etc. |
| LeverageCheckResult | 2 | Valid/invalid results |
| MarginCheckResult | 2 | Healthy/danger results |
| MarginCallEvent | 4 | Creation, urgency, escalation |
| FuturesLeverageGuard | 8 | Validation, max position |
| FuturesMarginGuard | 7 | All margin levels |
| MarginCallNotifier | 7 | Notifications, cooldowns |
| FundingExposureGuard | 8 | All funding levels |
| ConcentrationGuard | 6 | Single/correlated limits |
| ADLRiskGuard | 5 | All ADL levels |
| CryptoFuturesRiskGuard | 4 | Integration tests |
| Factory Functions | 4 | Creation, spot handling |
| RiskEvent Integration | 7 | All event types |
| Edge Cases | 6 | Zero values, extremes |
| Thread Safety | 2 | Concurrent access |
| Integration Scenarios | 4 | Full workflows |
| Risk Summary | 2 | Summary generation |

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `services/futures_risk_guards.py` | All futures risk guard implementations (~1200 lines) |
| `risk_guard.py` | CryptoFuturesRiskGuard integration (~200 lines added) |
| `tests/test_futures_risk_guards.py` | 101 comprehensive tests |

### Референсы

- Binance: "Leverage and Margin of USDⓈ-M Futures"
- Binance: "Auto-Deleveraging (ADL)"
- Binance: "Funding Rate History"
- Binance: "Liquidation Protocol"
- Risk management best practices for derivatives trading

---

## 🛡️ Phase 6B: CME Futures Risk Management (COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 130/130 (at documentation time; verify via CI) | **Покрытие**: 98% | **Date**: 2025-12-02

Phase 6B implements comprehensive risk management for CME Group futures (via Interactive Brokers), including SPAN margin monitoring, position limits, circuit breaker awareness, settlement risk management, and contract rollover guards.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **SPANMarginGuard** | `services/cme_risk_guards.py` | SPAN margin monitoring with 4 levels |
| **CMEPositionLimitGuard** | `services/cme_risk_guards.py` | CME speculative limits & accountability levels |
| **CircuitBreakerAwareGuard** | `services/cme_risk_guards.py` | Rule 80B circuit breaker integration |
| **SettlementRiskGuard** | `services/cme_risk_guards.py` | Daily settlement risk management |
| **RolloverGuard** | `services/cme_risk_guards.py` | Contract expiration & rollover tracking |
| **CMEFuturesRiskGuard** | `services/cme_risk_guards.py` | Unified guard combining all CME guards |
| **Тесты** | `tests/test_cme_risk_guards.py` | 130 comprehensive tests |

### Key Concepts

#### 1. SPAN Margin Status Levels

| Level | Margin Ratio | Action |
|-------|--------------|--------|
| **HEALTHY** | ≥ 1.5 (150%) | No action |
| **WARNING** | 1.2-1.5 (120-150%) | Alert |
| **DANGER** | 1.05-1.2 (105-120%) | Reduce position |
| **CRITICAL** | 1.0-1.05 (100-105%) | Urgent action |
| **LIQUIDATION** | ≤ 1.0 (100%) | Immediate liquidation risk |

#### 2. CME Position Limits (Speculative)

| Product | Speculative Limit | Accountability Level |
|---------|-------------------|---------------------|
| ES | 50,000 | 20,000 |
| NQ | 40,000 | 15,000 |
| YM | 25,000 | 10,000 |
| RTY | 20,000 | 5,000 |
| GC | 6,000 | 3,000 |
| CL | 10,000 | 5,000 |
| 6E | 10,000 | 5,000 |
| ZN | 150,000 | 50,000 |

#### 3. Circuit Breaker Levels (Rule 80B)

| Level | Trigger | RTH Halt | ETH Action |
|-------|---------|----------|------------|
| Level 1 | -7% | 15 min | Monitoring |
| Level 2 | -13% | 15 min | Restrict trading |
| Level 3 | -20% | Day halt | Block all trading |

#### 4. Settlement Risk Levels

| Level | Minutes to Settlement | Action |
|-------|----------------------|--------|
| **NORMAL** | > warn_minutes | Normal trading |
| **APPROACHING** | warn - critical | Alert, monitor VM |
| **IMMINENT** | critical - block | Prepare for settlement |
| **SETTLEMENT** | < block_minutes | Block new positions |

#### 5. Rollover Risk Levels

| Level | Days to Roll | Action |
|-------|--------------|--------|
| **SAFE** | > warn_days | Normal trading |
| **MONITORING** | warn - critical | Monitor spreads |
| **APPROACHING** | critical - block | Prepare roll trades |
| **IMMINENT** | 0 - block | Execute rollover |
| **EXPIRED** | < 0 | Force close only |

### Usage

```python
from services.cme_risk_guards import (
    CMEFuturesRiskGuard,
    SPANMarginGuard,
    CMEPositionLimitGuard,
    CircuitBreakerAwareGuard,
    SettlementRiskGuard,
    RolloverGuard,
    RiskEvent,
)
from decimal import Decimal

# 1. Unified Risk Guard
guard = CMEFuturesRiskGuard(strict_mode=True)
guard.add_symbol_to_monitor("ES", Decimal("4500"))

event = guard.check_trade(
    symbol="ES",
    side="LONG",
    quantity=5,
    account_equity=Decimal("500000"),
    positions=current_positions,
    prices={"ES": Decimal("4500")},
    contract_specs=specs,
    timestamp_ms=int(time.time() * 1000),
)

if event != RiskEvent.NONE:
    print(f"Risk event: {event.value}")
    print(f"Details: {guard.get_last_event_details()}")

# 2. SPAN Margin Guard
margin_guard = SPANMarginGuard()
margin_result = margin_guard.check_margin(
    account_equity=Decimal("500000"),
    positions=positions,
    prices=prices,
    contract_specs=specs,
)
print(f"Margin Status: {margin_result.status}")
print(f"Margin Ratio: {margin_result.margin_ratio}")

# 3. Position Limit Guard
limit_guard = CMEPositionLimitGuard()
limit_result = limit_guard.check_position_limit("ES", 45000)
print(f"Within Limit: {limit_result.is_within_limit}")
print(f"Utilization: {limit_result.utilization_pct}%")

# 4. Circuit Breaker Aware Guard
cb_guard = CircuitBreakerAwareGuard()
cb_guard.add_symbol("ES", Decimal("4500"))
cb_result = cb_guard.check_trading_allowed(
    symbol="ES",
    current_price=Decimal("4185"),  # -7%
    timestamp_ms=now_ms,
    is_rth=True,
)
print(f"Can Trade: {cb_result.can_trade}")
print(f"CB Level: {cb_result.circuit_breaker_level}")

# 5. Settlement Risk Guard
settle_guard = SettlementRiskGuard()
settle_result = settle_guard.check_settlement_risk(
    symbol="ES",
    timestamp_ms=now_ms,
)
print(f"Settlement Risk: {settle_result.risk_level}")
print(f"Minutes to Settlement: {settle_result.minutes_to_settlement}")

# 6. Rollover Guard
roll_guard = RolloverGuard()
roll_guard.set_expiration_calendar("ES", [date(2025, 3, 21)])
roll_result = roll_guard.check_rollover_risk("ES", date.today())
print(f"Rollover Risk: {roll_result.risk_level}")
print(f"Days to Roll: {roll_result.days_to_roll}")
```

### Risk Event Types

| Event | Trigger | Strict Mode |
|-------|---------|-------------|
| `NONE` | All checks pass | - |
| `MARGIN_WARNING` | Margin ratio < warning | Strict only |
| `MARGIN_DANGER` | Margin ratio < danger | Always |
| `MARGIN_CRITICAL` | Margin ratio < critical | Always |
| `MARGIN_LIQUIDATION` | Margin ratio ≤ 1.0 | Always |
| `POSITION_LIMIT_EXCEEDED` | Over speculative limit | Always |
| `POSITION_ACCOUNTABILITY` | Over accountability | Strict only |
| `CIRCUIT_BREAKER_L1` | -7% decline | Always |
| `CIRCUIT_BREAKER_L2` | -13% decline | Always |
| `CIRCUIT_BREAKER_L3` | -20% decline | Always |
| `VELOCITY_PAUSE` | Rapid price movement | Always |
| `SETTLEMENT_APPROACHING` | < warn_minutes | Strict only |
| `SETTLEMENT_IMMINENT` | < critical_minutes | Always |
| `ROLLOVER_WARNING` | < warn_days | Strict only |
| `ROLLOVER_IMMINENT` | < block_days | Always |
| `ROLLOVER_REQUIRED` | Contract expired | Always |

### Тестирование

```bash
# All Phase 6B tests (130 tests, 98% coverage)
pytest tests/test_cme_risk_guards.py -v

# By component
pytest tests/test_cme_risk_guards.py::TestSPANMarginGuard -v
pytest tests/test_cme_risk_guards.py::TestCMEPositionLimitGuard -v
pytest tests/test_cme_risk_guards.py::TestCircuitBreakerAwareGuard -v
pytest tests/test_cme_risk_guards.py::TestSettlementRiskGuard -v
pytest tests/test_cme_risk_guards.py::TestRolloverGuard -v
pytest tests/test_cme_risk_guards.py::TestCMEFuturesRiskGuard -v
```

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `services/cme_risk_guards.py` | All CME risk guard implementations (~1850 lines) |
| `tests/test_cme_risk_guards.py` | 130 comprehensive tests |

### Configuration

```python
from services.cme_risk_guards import (
    SPANMarginGuardConfig,
    CMEPositionLimitGuardConfig,
    CircuitBreakerGuardConfig,
    SettlementRiskGuardConfig,
    RolloverGuardConfig,
)

# SPAN Margin Config
margin_config = SPANMarginGuardConfig(
    warning_ratio=Decimal("1.5"),
    danger_ratio=Decimal("1.2"),
    critical_ratio=Decimal("1.05"),
)

# Circuit Breaker Config
cb_config = CircuitBreakerGuardConfig(
    prevent_trades_on_halt=True,
    pre_cb_warning_pct=Decimal("-0.05"),
)

# Settlement Risk Config
settle_config = SettlementRiskGuardConfig(
    warn_minutes_before=60,
    critical_minutes_before=30,
    block_new_positions_minutes=15,
)

# Rollover Config
roll_config = RolloverGuardConfig(
    warn_days_before=8,
    critical_days_before=3,
    block_new_positions_days=1,
)
```

### Референсы

- CME Group: "Position Limits and Accountability Levels"
- CME Group: "SPAN Margin Methodology"
- CME Group: "Rule 80B - Circuit Breakers"
- CME Group: "Daily Settlement Procedures"
- CME Group: "Contract Specifications and Expiration"

---

## 🛡️ Phase 7: Unified Futures Risk Management (COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 116/116 (at documentation time; verify via CI) | **Покрытие**: 98% | **Date**: 2025-12-02

Phase 7 unifies crypto futures and CME futures risk management into a single interface with automatic asset type detection, portfolio-level risk aggregation, and cross-asset correlation handling.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **UnifiedFuturesRiskGuard** | `services/unified_futures_risk.py` | Main unified guard with auto-delegation |
| **AssetType** | `services/unified_futures_risk.py` | Enum for asset classification |
| **UnifiedRiskConfig** | `services/unified_futures_risk.py` | Pydantic config combining crypto/CME settings |
| **UnifiedRiskEvent** | `services/unified_futures_risk.py` | Unified risk events across asset types |
| **UnifiedMarginResult** | `services/unified_futures_risk.py` | Unified margin check results |
| **PortfolioRiskManager** | `services/unified_futures_risk.py` | Portfolio-level risk aggregation |
| **Тесты** | `tests/test_unified_futures_risk.py` | 116 comprehensive tests |
| **Config** | `configs/unified_futures_risk.yaml` | YAML configuration with profiles |

### Key Concepts

#### 1. Asset Type Detection

Automatic detection from symbol patterns:

| Pattern | Asset Type | Examples |
|---------|------------|----------|
| `*USDT`, `*BUSD` | CRYPTO_PERPETUAL | BTCUSDT, ETHBUSD |
| `*_YYMMDD` | CRYPTO_QUARTERLY | BTCUSDT_240329 |
| `ES`, `NQ`, `YM`, `RTY` | CME_EQUITY_INDEX | ES, NQ, MES, MNQ |
| `GC`, `SI`, `HG`, `MGC` | CME_METAL | Gold, Silver, Copper |
| `CL`, `NG`, `RB`, `HO` | CME_ENERGY | Crude, NatGas |
| `6E`, `6J`, `6B`, `6A` | CME_CURRENCY | Euro, Yen, Pound |
| `ZN`, `ZB`, `ZT`, `ZF` | CME_BOND | 10Y, 30Y notes |
| Other | UNKNOWN | Fallback |

#### 2. Automatic Guard Delegation

```python
from services.unified_futures_risk import UnifiedFuturesRiskGuard

guard = UnifiedFuturesRiskGuard()

# Crypto symbols → Crypto guards
event = guard.check_trade("BTCUSDT", "BUY", 0.1, ...)  # Uses crypto guards

# CME symbols → CME guards
event = guard.check_trade("ES", "BUY", 5, ...)  # Uses CME guards
```

#### 3. Unified Risk Events

| Event | Description | Crypto | CME |
|-------|-------------|--------|-----|
| `MARGIN_WARNING` | Approaching margin limit | ✅ | ✅ |
| `MARGIN_DANGER` | Low margin ratio | ✅ | ✅ |
| `MARGIN_CRITICAL` | Critical margin | ✅ | ✅ |
| `MARGIN_LIQUIDATION` | Liquidation risk | ✅ | ✅ |
| `LEVERAGE_EXCEEDED` | Over leverage limit | ✅ | - |
| `CONCENTRATION_EXCEEDED` | Position too large | ✅ | - |
| `FUNDING_WARNING` | High funding rate | ✅ | - |
| `FUNDING_EXCESSIVE` | Extreme funding | ✅ | - |
| `ADL_WARNING` | ADL queue risk | ✅ | - |
| `ADL_CRITICAL` | High ADL risk | ✅ | - |
| `CIRCUIT_BREAKER_L1` | -7% decline | - | ✅ |
| `CIRCUIT_BREAKER_L2` | -13% decline | - | ✅ |
| `CIRCUIT_BREAKER_L3` | -20% decline | - | ✅ |
| `VELOCITY_PAUSE` | Rapid price move | - | ✅ |
| `POSITION_LIMIT_EXCEEDED` | Over spec limit | - | ✅ |
| `SETTLEMENT_APPROACHING` | Near settlement | - | ✅ |
| `ROLLOVER_WARNING` | Near expiry | - | ✅ |

### Usage

```python
from services.unified_futures_risk import (
    UnifiedFuturesRiskGuard,
    UnifiedRiskConfig,
    CryptoRiskConfig,
    CMERiskConfig,
    PortfolioRiskConfig,
    create_unified_risk_guard,
    load_config_from_yaml,
)
from decimal import Decimal

# 1. Create with defaults
guard = UnifiedFuturesRiskGuard()

# 2. Create from YAML config
config = load_config_from_yaml("configs/unified_futures_risk.yaml")
guard = create_unified_risk_guard(config)

# 3. Create with custom config
config = UnifiedRiskConfig(
    crypto=CryptoRiskConfig(
        max_account_leverage=20.0,
        max_symbol_leverage=125.0,
        margin_warning_threshold=1.5,
        margin_danger_threshold=1.2,
        margin_critical_threshold=1.05,
        max_single_symbol_pct=0.5,
    ),
    cme=CMERiskConfig(
        margin_warning_ratio=1.5,
        margin_danger_ratio=1.2,
        margin_critical_ratio=1.05,
        enforce_speculative_limits=True,
        prevent_trades_on_halt=True,
    ),
    portfolio=PortfolioRiskConfig(
        enable_correlation_tracking=True,
        correlation_lookback_days=30,
    ),
)
guard = UnifiedFuturesRiskGuard(config=config)

# 4. Check trade (auto-delegates based on symbol)
event = guard.check_trade(
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.5,
    leverage=10,
    account_equity=Decimal("50000"),
    mark_price=Decimal("45000"),
    funding_rate=Decimal("0.0001"),
)

if event != UnifiedRiskEvent.NONE:
    print(f"Risk event: {event.value}")
    print(f"Details: {guard.get_last_event_details()}")

# 5. Check margin (crypto)
margin_result = guard.check_margin(
    symbol="ETHUSDT",
    account_equity=Decimal("100000"),
    positions=crypto_positions,
    mark_prices={"ETHUSDT": Decimal("3000")},
)
print(f"Status: {margin_result.status}")
print(f"Margin Ratio: {margin_result.margin_ratio}")

# 6. Check margin (CME)
margin_result = guard.check_margin(
    symbol="ES",
    account_equity=Decimal("500000"),
    positions=cme_positions,
    prices={"ES": Decimal("4500")},
    contract_specs=es_spec,
)
print(f"Status: {margin_result.status}")
print(f"Available Margin: ${margin_result.available_margin}")

# 7. Get asset type
asset_type = guard.get_asset_type("BTCUSDT")  # CRYPTO_PERPETUAL
asset_type = guard.get_asset_type("ES")       # CME_EQUITY_INDEX
asset_type = guard.get_asset_type("GC")       # CME_METAL

# 8. Portfolio-level risk (cross-asset)
portfolio_result = guard.check_portfolio_risk(
    all_positions={"BTCUSDT": pos1, "ES": pos2, "GC": pos3},
    account_equity=Decimal("1000000"),
)
print(f"Total Margin Used: ${portfolio_result.total_margin_used}")
print(f"Cross-Asset Correlation: {portfolio_result.correlation_warning}")
```

### Configuration (YAML)

```yaml
# configs/unified_futures_risk.yaml
crypto:
  max_account_leverage: 20.0
  max_symbol_leverage: 125.0
  margin_warning_threshold: 1.5
  margin_danger_threshold: 1.2
  margin_critical_threshold: 1.05
  max_single_symbol_pct: 0.5
  max_correlated_group_pct: 0.7
  funding_warning_threshold: 0.0001
  funding_excessive_threshold: 0.0003
  adl_warning_percentile: 75.0
  adl_critical_percentile: 90.0
  strict_mode: true

cme:
  margin_warning_ratio: 1.5
  margin_danger_ratio: 1.2
  margin_critical_ratio: 1.05
  prevent_trades_on_halt: true
  pre_cb_warning_pct: -0.05
  settlement_warn_minutes: 60
  settlement_critical_minutes: 30
  rollover_warn_days: 8
  rollover_critical_days: 3
  enforce_speculative_limits: true
  strict_mode: true

portfolio:
  enable_correlation_tracking: true
  correlation_lookback_days: 30
  correlation_spike_threshold: 0.8
  aggregate_margin_across_types: true

profiles:
  conservative:
    crypto:
      max_account_leverage: 10.0
      margin_warning_threshold: 2.0
    cme:
      margin_warning_ratio: 2.0
  aggressive:
    crypto:
      max_account_leverage: 50.0
      margin_warning_threshold: 1.2
    cme:
      margin_warning_ratio: 1.2
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  UnifiedFuturesRiskGuard                        │
│  - Asset type detection                                         │
│  - Automatic guard delegation                                   │
│  - Unified event translation                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────┐
│  Crypto Guards      │               │  CME Guards         │
│  ├─ LeverageGuard   │               │  ├─ SPANMarginGuard │
│  ├─ MarginGuard     │               │  ├─ PositionLimits  │
│  ├─ ConcentrationG  │               │  ├─ CircuitBreaker  │
│  ├─ FundingGuard    │               │  ├─ SettlementRisk  │
│  └─ ADLRiskGuard    │               │  └─ RolloverGuard   │
└─────────────────────┘               └─────────────────────┘
          │                                       │
          └───────────────────┬───────────────────┘
                              ▼
                 ┌─────────────────────┐
                 │  PortfolioRiskMgr   │
                 │  - Cross-asset      │
                 │  - Correlation      │
                 │  - Aggregation      │
                 └─────────────────────┘
```

### Тестирование

```bash
# All Phase 7 tests (116 tests)
pytest tests/test_unified_futures_risk.py -v

# By category
pytest tests/test_unified_futures_risk.py::TestAssetType -v
pytest tests/test_unified_futures_risk.py::TestUnifiedRiskEvent -v
pytest tests/test_unified_futures_risk.py::TestUnifiedMarginResult -v
pytest tests/test_unified_futures_risk.py::TestUnifiedRiskConfig -v
pytest tests/test_unified_futures_risk.py::TestUnifiedFuturesRiskGuard -v
pytest tests/test_unified_futures_risk.py::TestPortfolioRiskManager -v
pytest tests/test_unified_futures_risk.py::TestFactoryFunctions -v
pytest tests/test_unified_futures_risk.py::TestIntegration -v

# Regression tests (Phase 6A + 6B)
pytest tests/test_futures_risk_guards.py tests/test_cme_risk_guards.py -v  # 231 tests
```

**Coverage**: 116 Phase 7 tests + 231 regression tests = 347 total tests (at documentation time; verify via CI)

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `services/unified_futures_risk.py` | Unified risk management (~900 lines) |
| `configs/unified_futures_risk.yaml` | Configuration with profiles |
| `tests/test_unified_futures_risk.py` | 116 comprehensive tests |

### Референсы

- Phase 6A: Crypto Futures Risk Guards
- Phase 6B: CME Futures Risk Guards
- Portfolio theory: Markowitz (1952) mean-variance optimization
- Risk aggregation: Basel III framework concepts

---

## 🔴 Phase 9: Unified Futures Live Trading (COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 81/81 (at documentation time; verify via CI) | **Date**: 2025-12-02

Phase 9 implements unified live trading infrastructure for futures, including position synchronization, margin monitoring, funding rate tracking, and a coordinated live runner.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **FuturesLiveRunner** | `services/futures_live_runner.py` | Main live trading coordinator |
| **FuturesPositionSynchronizer** | `services/futures_position_sync.py` | Position sync with exchange |
| **FuturesMarginMonitor** | `services/futures_margin_monitor.py` | Real-time margin monitoring |
| **FuturesFundingTracker** | `services/futures_funding_tracker.py` | Funding rate tracking & predictions |
| **Live Config** | `configs/config_live_futures.yaml` | Live trading configuration |
| **Tests** | `tests/test_futures_live_trading.py` | 81 comprehensive tests |

### Key Concepts

#### 1. Position Synchronization

Real-time position sync between local state and exchange:

```python
from services.futures_position_sync import (
    FuturesPositionSynchronizer,
    FuturesSyncConfig,
    FuturesSyncEventType,
)

config = FuturesSyncConfig(
    exchange=Exchange.BINANCE,
    futures_type=FuturesType.CRYPTO_PERPETUAL,
    sync_interval_sec=10.0,       # Sync every 10 seconds
    qty_tolerance_pct=0.001,      # 0.1% tolerance
    auto_reconcile=False,         # Manual reconciliation
)

sync = FuturesPositionSynchronizer(
    position_provider=position_provider,
    account_provider=account_provider,
    local_state_getter=get_local_positions,
    config=config,
    on_event=handle_sync_event,
)

# Start background sync
await sync.start_async()

# Or sync once
events = await sync.sync_once()
for event in events:
    if event.event_type == FuturesSyncEventType.QTY_MISMATCH:
        print(f"Position mismatch: {event.symbol}")
```

#### 2. Sync Event Types

| Event Type | Description |
|------------|-------------|
| `POSITION_OPENED` | New position detected on exchange |
| `POSITION_CLOSED` | Position closed on exchange |
| `POSITION_MODIFIED` | Position size changed |
| `QTY_MISMATCH` | Local vs exchange quantity differs |
| `LEVERAGE_MISMATCH` | Leverage setting differs |
| `LIQUIDATION_DETECTED` | Position liquidated |
| `ADL_DETECTED` | Auto-deleveraging occurred |
| `FUNDING_RECEIVED` | Funding payment received |
| `FUNDING_PAID` | Funding payment made |
| `SETTLEMENT_OCCURRED` | Daily settlement (CME) |
| `MARGIN_CALL` | Margin call triggered |
| `MARGIN_RATIO_LOW` | Margin ratio below threshold |

#### 3. Margin Monitoring

Real-time margin ratio tracking with alerts:

```python
from services.futures_margin_monitor import (
    FuturesMarginMonitor,
    MarginMonitorConfig,
    MarginStatus,
)

config = MarginMonitorConfig(
    check_interval_sec=5.0,
    warning_ratio=1.5,    # 150%
    danger_ratio=1.2,     # 120%
    critical_ratio=1.05,  # 105%
)

monitor = FuturesMarginMonitor(
    account_provider=account_provider,
    position_provider=position_provider,
    config=config,
    on_status_change=handle_margin_alert,
)

# Check current status
status = await monitor.check_margin()
print(f"Margin ratio: {status.margin_ratio:.2f}")
print(f"Status: {status.status}")  # HEALTHY, WARNING, DANGER, CRITICAL
```

#### 4. Funding Rate Tracking

Historical tracking and prediction for crypto perpetuals:

```python
from services.futures_funding_tracker import (
    FuturesFundingTracker,
    FundingTrackerConfig,
    FundingRateInfo,
)

config = FundingTrackerConfig(
    data_dir="data/futures",
    prediction_method="ewma",    # last, avg, ewma
    cache_ttl_sec=300,
)

tracker = FuturesFundingTracker(
    funding_provider=funding_provider,
    config=config,
)

# Get current funding info
info = await tracker.get_funding_info("BTCUSDT")
print(f"Current rate: {info.funding_rate:.4%}")
print(f"Next funding: {info.next_funding_time}")
print(f"Predicted rate: {info.predicted_rate:.4%}")

# Get funding statistics
stats = tracker.get_funding_stats("BTCUSDT", lookback_days=30)
print(f"Avg rate: {stats.avg_rate:.4%}")
print(f"Annualized: {stats.annualized_rate:.2%}")
```

#### 5. Live Runner

Coordinates all components for unified live trading:

```python
from services.futures_live_runner import (
    FuturesLiveRunner,
    FuturesLiveConfig,
    create_futures_live_runner,
)

# Load from YAML
config = FuturesLiveConfig.from_yaml("configs/config_live_futures.yaml")

# Create runner
runner = create_futures_live_runner(config)

# Start live trading
await runner.start()

# Runner coordinates:
# - Position sync (every 5-10 sec)
# - Margin monitoring (every 5 sec)
# - Funding tracking (every 60 sec)
# - Signal generation (main loop)
# - Order execution
# - Risk management
```

### Configuration

```yaml
# configs/config_live_futures.yaml
futures_type: "CRYPTO_PERPETUAL"
exchange: "binance"
symbols:
  - "BTCUSDT"
  - "ETHUSDT"

paper_trading: true

# Timing
main_loop_interval_sec: 1.0
position_sync_interval_sec: 5.0
margin_check_interval_sec: 10.0
funding_check_interval_sec: 60.0

# Feature flags
enable_position_sync: true
enable_margin_monitoring: true
enable_funding_tracking: true
enable_adl_monitoring: true

# Risk settings
strict_mode: true
max_leverage: 10
max_position_value: 100000
max_total_exposure: 500000

# Margin thresholds
margin:
  warning_ratio: 1.5
  danger_ratio: 1.2
  critical_ratio: 1.1
  alert_cooldown_sec: 300

# Position sync settings
position_sync:
  interval_sec: 5.0
  tolerance: 0.01
  auto_reconcile: false

# Funding tracking
funding:
  data_dir: "data/futures"
  prediction_method: "ewma"
  cache_ttl_sec: 300
```

### ADL Risk Levels

| Level | Description | Action |
|-------|-------------|--------|
| `SAFE` | Low ADL risk | Normal trading |
| `WARNING` | Moderate ADL risk | Monitor closely |
| `DANGER` | High ADL risk | Consider reducing |
| `CRITICAL` | Imminent ADL risk | Reduce immediately |

### Тестирование

```bash
# All Phase 9 tests (81 tests)
pytest tests/test_futures_live_trading.py -v

# By category
pytest tests/test_futures_live_trading.py::TestFuturesLiveConfig -v
pytest tests/test_futures_live_trading.py::TestFuturesSyncConfig -v
pytest tests/test_futures_live_trading.py::TestFuturesSyncEventType -v
pytest tests/test_futures_live_trading.py::TestFuturesPositionSynchronizer -v
pytest tests/test_futures_live_trading.py::TestFuturesMarginMonitor -v
pytest tests/test_futures_live_trading.py::TestFuturesFundingTracker -v
pytest tests/test_futures_live_trading.py::TestFuturesLiveRunner -v
```

**Coverage**: 81 tests (at documentation time; verify via CI)

| Category | Tests | Coverage |
|----------|-------|----------|
| FuturesLiveConfig | 10 | Config loading, validation, defaults |
| FuturesSyncConfig | 6 | Sync config defaults, custom values |
| FuturesSyncEventType | 8 | All event types |
| FundingRateInfo | 4 | Funding rate data model |
| MarginStatus | 5 | Margin status levels |
| ADLRiskLevel | 4 | ADL risk classification |
| FuturesPositionSynchronizer | 15 | Position sync workflow |
| FuturesMarginMonitor | 10 | Margin monitoring |
| FuturesFundingTracker | 8 | Funding tracking & prediction |
| FuturesLiveRunner | 7 | Live runner coordination |
| Integration | 4 | End-to-end scenarios |

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `services/futures_live_runner.py` | Main live trading coordinator (~500 lines) |
| `services/futures_position_sync.py` | Position synchronization (~600 lines) |
| `services/futures_margin_monitor.py` | Margin monitoring (~400 lines) |
| `services/futures_funding_tracker.py` | Funding rate tracking (~450 lines) |
| `configs/config_live_futures.yaml` | Live trading configuration |
| `tests/test_futures_live_trading.py` | 81 comprehensive tests |

### Референсы

- Phase 8: Multi-Futures Training Pipeline (prerequisite)
- Phase 6A/6B: Crypto/CME Risk Guards (integrated)
- Phase 7: Unified Risk Management (integrated)
- Binance Futures API: Position, Account, Funding Rate endpoints
- CME Group: Daily settlement procedures

---

## 📋 Phase 10: Validation & Documentation (COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 171/171 (at documentation time; verify via CI) | **Date**: 2025-12-02

Phase 10 completes the Futures Integration project with comprehensive validation testing, backward compatibility verification, performance benchmarks, and documentation.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **Validation Tests** | `tests/test_futures_validation.py` | 125 comprehensive validation tests |
| **Backward Compatibility** | `tests/test_futures_backward_compatibility.py` | 46 passed, 20 skipped compatibility tests |
| **Performance Benchmarks** | `benchmarks/bench_futures_simulation.py` | Performance measurement suite |
| **Integration Report** | `FUTURES_INTEGRATION_REPORT.md` | Project completion report |
| **Documentation Suite** | `docs/futures/*.md` | 8 documentation files |

### Validation Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| Core Models | 15 | FuturesPosition, ContractSpec, MarginMode |
| Margin Calculations | 20 | Tiered margin, SPAN, liquidation price |
| Funding Rates | 12 | Rate calculation, payment simulation |
| Slippage Models | 18 | Crypto L2, CME L2, cascade effects |
| Risk Guards | 15 | Leverage, margin, concentration, ADL |
| L3 LOB Simulation | 15 | Fill probability, impact models, matching |
| Cross-Component | 10 | Full trade cycle, data flow |
| Validation Metrics | 5 | Fill rate, slippage, funding accuracy |

### Backward Compatibility Categories

| Category | Tests | Status |
|----------|-------|--------|
| Crypto Spot | 10 | ✅ All pass |
| US Equity | 10 | ✅ All pass (some skipped) |
| Forex (OANDA) | 8 | ✅ All pass |
| L3 LOB | 8 | ✅ All pass |
| Risk Management | 4 | ✅ All pass |
| Trading Env | 4 | ✅ All pass |
| Adapters | 6 | ✅ All pass |
| Features Pipeline | 4 | ✅ All pass (some skipped) |
| Model Training | 4 | ✅ All pass |
| Configuration | 4 | ✅ All pass |

### Validation Metrics Achieved

> **Note**: Metrics below are from internal testing at documentation time. Verify current performance via test suite execution.

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Fill Rate (L2) | > 95% | 98.5% | ✅ |
| Fill Rate (L3) | > 90% | 94.2% | ✅ |
| Slippage Error | < 3 bps | 1.8 bps | ✅ |
| Funding Rate Accuracy | > 99% | 99.7% | ✅ |
| Liquidation Timing | < 1 bar | 0.2 bars | ✅ |
| Margin Calculation Error | < 0.1% | 0.02% | ✅ |

### Performance Benchmarks

| Operation | Target | Achieved | Status |
|-----------|--------|----------|--------|
| L2 Crypto Slippage | < 100 μs | 45 μs | ✅ |
| L2 CME Slippage | < 100 μs | 52 μs | ✅ |
| L3 Matching | < 500 μs | 180 μs | ✅ |
| Tiered Margin Calc | < 50 μs | 18 μs | ✅ |
| SPAN Margin Calc | < 100 μs | 75 μs | ✅ |
| Funding Rate Calc | < 10 μs | 3 μs | ✅ |
| Liquidation Price | < 50 μs | 22 μs | ✅ |
| Risk Guard Check | < 50 μs | 28 μs | ✅ |

### Documentation Suite

| File | Description |
|------|-------------|
| `docs/futures/overview.md` | Architecture overview |
| `docs/futures/api_reference.md` | API reference |
| `docs/futures/configuration.md` | Configuration guide |
| `docs/futures/margin_calculation.md` | Margin calculation |
| `docs/futures/funding_rates.md` | Funding rates |
| `docs/futures/liquidation.md` | Liquidation engine |
| `docs/futures/deployment.md` | Deployment guide |
| `docs/futures/migration_guide.md` | Migration guide |

### Тестирование

```bash
# All Phase 10 tests
pytest tests/test_futures_validation.py tests/test_futures_backward_compatibility.py -v

# Validation tests only (125 tests)
pytest tests/test_futures_validation.py -v

# Backward compatibility only (66 tests)
pytest tests/test_futures_backward_compatibility.py -v

# Run benchmarks
python benchmarks/bench_futures_simulation.py
```

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `tests/test_futures_validation.py` | 125 validation tests |
| `tests/test_futures_backward_compatibility.py` | 66 backward compatibility tests |
| `benchmarks/bench_futures_simulation.py` | Performance benchmark suite |
| `FUTURES_INTEGRATION_REPORT.md` | Integration completion report |
| `docs/futures/*.md` | 8 documentation files |

### Референсы

- Phase 3B-9: All preceding futures integration phases
- Binance Futures API: Reference for crypto perpetual simulation
- CME Group: SPAN methodology, Rule 80B, trading hours
- Kyle (1985): Price impact model
- Almgren & Chriss (2001): Optimal execution

---

## 📈 Options Integration (Phase 1: COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 240/240 (at documentation time; verify via CI) | **Date**: 2025-12-03

Phase 1 implements core options pricing, Greeks computation, IV solving, and exercise probability analysis.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **Black-Scholes Pricing** | `impl_pricing.py` | BSM with continuous dividends |
| **Binomial Trees** | `impl_pricing.py` | Leisen-Reimer, CRR |
| **Jump Diffusion** | `impl_pricing.py` | Merton model with Poisson jumps |
| **Variance Swap** | `impl_pricing.py` | Strike calculation via replication |
| **Discrete Dividends** | `impl_pricing.py` | Escrowed, piecewise-lognormal |
| **Greeks (12)** | `impl_greeks_vectorized.py` | Delta, Gamma, Theta, Vega, Rho, Vanna, Volga, Charm, Speed, Color, Zomma, Ultima |
| **Batch Greeks** | `impl_greeks_vectorized.py` | Vectorized NumPy computation |
| **IV Solver** | `impl_iv_calculation.py` | Hybrid Newton-Raphson/Brent/bisection |
| **Exercise Probability** | `impl_exercise_probability.py` | LSMC, Barone-Adesi-Whaley |
| **Core Models** | `core_options.py` | OptionsContractSpec, GreeksResult, IVResult |

### Key Concepts

#### 1. Black-Scholes-Merton with Dividends

```python
from impl_pricing import black_scholes_price

# Call option price
price = black_scholes_price(
    spot=100.0,
    strike=100.0,
    time_to_expiry=0.25,  # 3 months
    rate=0.05,            # 5% risk-free rate
    dividend_yield=0.02,  # 2% continuous dividend
    volatility=0.20,      # 20% volatility
    is_call=True,
)
```

#### 2. All 12 Greeks

| Greek | Symbol | Definition | Order |
|-------|--------|------------|-------|
| Delta | Δ | ∂V/∂S | 1st |
| Gamma | Γ | ∂²V/∂S² | 2nd |
| Theta | Θ | ∂V/∂t | 1st |
| Vega | ν | ∂V/∂σ | 1st |
| Rho | ρ | ∂V/∂r | 1st |
| Vanna | | ∂²V/∂S∂σ | 2nd |
| Volga | | ∂²V/∂σ² | 2nd |
| Charm | | ∂²V/∂S∂t | 2nd |
| Speed | | ∂³V/∂S³ | 3rd |
| Color | | ∂³V/∂S²∂t | 3rd |
| Zomma | | ∂³V/∂S²∂σ | 3rd |
| Ultima | | ∂³V/∂σ³ | 3rd |

```python
from impl_greeks_vectorized import compute_all_greeks

greeks = compute_all_greeks(
    spot=100.0, strike=100.0, time_to_expiry=0.25,
    rate=0.05, dividend_yield=0.0, volatility=0.20, is_call=True,
)
print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.6f}")
print(f"Vega: {greeks.vega:.4f}")
```

#### 3. Vectorized Batch Greeks

```python
from impl_greeks_vectorized import compute_all_greeks_batch
import numpy as np

# Portfolio of 1000 options
n = 1000
result = compute_all_greeks_batch(
    spot=np.full(n, 100.0),
    strike=np.linspace(80, 120, n),
    time_to_expiry=np.full(n, 0.25),
    rate=np.full(n, 0.05),
    dividend_yield=np.full(n, 0.0),
    volatility=np.full(n, 0.20),
    is_call=np.ones(n, dtype=bool),
)
# result.delta, result.gamma, etc. are all numpy arrays of shape (n,)
```

#### 4. Implied Volatility Solver

```python
from impl_iv_calculation import calculate_iv

result = calculate_iv(
    spot=100.0, strike=100.0, time_to_expiry=0.25,
    rate=0.05, dividend_yield=0.0,
    market_price=5.5, is_call=True,
)
if result.converged:
    print(f"IV: {result.implied_volatility:.2%}")
```

#### 5. American Options (Barone-Adesi-Whaley)

```python
from impl_exercise_probability import barone_adesi_whaley

american_price, early_exercise_premium = barone_adesi_whaley(
    spot=100.0, strike=100.0, time_to_expiry=0.25,
    rate=0.05, dividend_yield=0.0, volatility=0.20, is_call=False,
)
```

#### 6. Longstaff-Schwartz Monte Carlo

```python
from impl_exercise_probability import (
    longstaff_schwartz_price,
    compute_exercise_probability,
)

# LSMC price for American put
price = longstaff_schwartz_price(
    spot=100.0, strike=100.0, time_to_expiry=0.25,
    rate=0.05, dividend_yield=0.0, volatility=0.20,
    is_call=False, n_paths=50000, n_steps=50, seed=42,
)

# Exercise probability at each timestep
probs = compute_exercise_probability(
    spot=100.0, strike=100.0, time_to_expiry=0.25,
    rate=0.05, dividend_yield=0.0, volatility=0.20,
    is_call=False, n_paths=10000, n_steps=50,
)
```

#### 7. Variance Swap Strike

Based on Demeterfi et al. (1999) replication formula:

```python
from impl_pricing import variance_swap_strike, compute_variance_swap_value
import numpy as np

# Strike grid for replication
strikes = np.array([70, 80, 90, 100, 110, 120, 130])
call_prices = np.array([...])  # From market
put_prices = np.array([...])   # From market

strike = variance_swap_strike(
    call_prices=call_prices,
    put_prices=put_prices,
    call_strikes=strikes,
    put_strikes=strikes,
    forward=100.0,
    rate=0.05,
    time_to_expiry=0.25,
)
# strike ≈ σ² (fair variance)
```

### Pricing Methods

| Method | Function | Use Case |
|--------|----------|----------|
| Black-Scholes | `black_scholes_price()` | European options |
| Leisen-Reimer | `binomial_tree_lr()` | American options, smooth convergence |
| Cox-Ross-Rubinstein | `binomial_tree_crr()` | American options, education |
| Merton Jump-Diffusion | `merton_jump_diffusion_price()` | Fat tails, jumps |
| LSMC | `longstaff_schwartz_price()` | Path-dependent, American |
| BAW | `barone_adesi_whaley()` | Fast American approximation |

### Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| Scalar Greeks | 24 | All 12 Greeks, edge cases |
| Vectorized Greeks | 19 | Batch, consistency, performance |
| Black-Scholes | 17 | Put-call parity, bounds |
| Binomial | 20 | Convergence, American exercise |
| Jump Diffusion | 11 | Merton, calibration |
| Variance Swap | 10 | Strike, value, scaling |
| IV Solver | 23 | Round-trip, edge cases |
| LSMC | 15 | Accuracy, variance reduction |
| Exercise Probability | 13 | BAW, early exercise |
| Contract Specs | 15 | Enums, dataclasses |
| Edge Cases | 6 | Zero/extreme inputs |
| Performance | 2 | Batch throughput |
| Integration | 3 | Full workflow |

### Тестирование

```bash
# All Phase 1 Options tests (240 tests)
pytest tests/test_options_core.py -v

# By category
pytest tests/test_options_core.py::TestScalarGreeks -v
pytest tests/test_options_core.py::TestVectorizedGreeks -v
pytest tests/test_options_core.py::TestBlackScholesPricing -v
pytest tests/test_options_core.py::TestBinomialPricing -v
pytest tests/test_options_core.py::TestIVCalculation -v
pytest tests/test_options_core.py::TestLongstaffSchwartzMC -v
pytest tests/test_options_core.py::TestVarianceSwap -v
```

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `impl_pricing.py` | BSM, binomial, jump diffusion, variance swap (~800 lines) |
| `impl_greeks_vectorized.py` | 12 Greeks with vectorized batch (~600 lines) |
| `impl_iv_calculation.py` | Hybrid IV solver (~400 lines) |
| `impl_exercise_probability.py` | LSMC, BAW, exercise analysis (~700 lines) |
| `core_options.py` | Core models (OptionsContractSpec, GreeksResult) |
| `tests/test_options_core.py` | 240 comprehensive tests |

### Референсы

- Black & Scholes (1973): "The Pricing of Options and Corporate Liabilities"
- Merton (1973): "Theory of Rational Option Pricing"
- Leisen & Reimer (1996): "Binomial Models for Option Valuation"
- Longstaff & Schwartz (2001): "Valuing American Options by Simulation"
- Barone-Adesi & Whaley (1987): "Efficient Analytic Approximation"
- Demeterfi et al. (1999): "A Guide to Volatility and Variance Swaps"

---

## 📈 Options Integration (Phase 2: COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 160 (159 pass, 1 skip) | **Date**: 2025-12-03

Phase 2 implements exchange adapters for options data and execution via IB TWS API and Polygon.io.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **IB Options Market Data** | `adapters/ib/options.py` | Option chain, quotes, Greeks streaming |
| **IB Options Execution** | `adapters/ib/options.py` | Single-leg orders, margin queries |
| **IB Rate Limiter** | `adapters/ib/options_rate_limiter.py` | Priority queue with LRU caching |
| **Polygon Options** | `adapters/polygon/options.py` | Historical options data (2018+) |
| **Core Options Models** | `core_options.py` | OptionsContractSpec, GreeksResult, IVResult |

### OCC Options Symbology

Standard format: `SYMBOL(6) + YYMMDD + C/P + STRIKE(8)` (21 chars total)

| Component | Format | Example |
|-----------|--------|---------|
| Symbol | 6 chars, right-padded | `AAPL` |
| Expiry | YYMMDD | `241220` |
| Type | C or P | `C` |
| Strike | 8 digits (strike × 1000) | `00200000` ($200) |

**Full example**: `AAPL  241220C00200000` = AAPL Dec 20 2024 $200 Call

**Polygon format**: `O:AAPL241220C00200000`

```python
from adapters.ib.options import create_occ_symbol, parse_occ_symbol
from datetime import date
from decimal import Decimal

# Create OCC symbol (option_type is "C" or "P" string, NOT boolean)
occ = create_occ_symbol(
    underlying="AAPL",
    expiration=date(2024, 12, 20),
    option_type="C",  # "C" for call, "P" for put
    strike=Decimal("200"),
)
# → "AAPL  241220C00200000" (21 chars)

# Parse OCC symbol
parsed = parse_occ_symbol("AAPL  241220C00200000")
# → {"symbol": "AAPL", "expiration": date(2024,12,20), "option_type": "C", "strike": Decimal("200")}

# Polygon ticker conversion
from adapters.polygon.options import polygon_ticker_to_occ, occ_to_polygon_ticker

occ = polygon_ticker_to_occ("O:AAPL241220C00200000")
# → "AAPL  241220C00200000"

polygon = occ_to_polygon_ticker("AAPL  241220C00200000")
# → "O:AAPL241220C00200000"
```

### IB Rate Limits

| Limit Type | IB Limit | Implementation |
|------------|----------|----------------|
| Option chains | 10/min | 8/min (safety margin) |
| Option quotes | 100/sec | 80/sec (safety margin) |
| Order submissions | 50/sec | 40/sec (safety margin) |
| Concurrent subscriptions | 100 lines | Tracked by manager |

**Request Priorities** (lower = higher priority):

- 0: Order execution (highest)
- 1: Risk/margin queries
- 2: Front-month quotes
- 3: Active series
- 4: Background requests
- 9: Backfill (lowest)

```python
from adapters.ib.options_rate_limiter import (
    IBOptionsRateLimitManager,
    OptionsChainCache,
    RequestPriority,
)
from datetime import date

# Create rate limit manager
manager = IBOptionsRateLimitManager(
    chain_limit_per_min=8,    # 10 IB limit with safety
    quote_limit_per_sec=80,   # 100 IB limit with safety
    order_limit_per_sec=40,   # 50 IB limit with safety
)

# Request chain with priority (uses cache if available)
def callback(chain):
    print(f"Got chain with {len(chain)} contracts")

queued = manager.request_chain(
    underlying="AAPL",
    expiration=date(2024, 12, 20),
    callback=callback,
    priority=RequestPriority.FRONT_MONTH,
)
# Returns True if queued, False if served from cache

# Check subscription count (property, not method)
count = manager.subscription_count  # int

# Create standalone cache
cache = OptionsChainCache(
    max_chains=100,           # LRU eviction when exceeded
    default_ttl_sec=300.0,    # 5-min default TTL
    front_month_ttl_sec=60.0, # 1-min for front month
)

# Cache operations
cache.put(underlying="AAPL", expiration=date(2024, 12, 20), chain=contracts)
cached = cache.get(underlying="AAPL", expiration=date(2024, 12, 20))
cache.invalidate(underlying="AAPL", expiration=date(2024, 12, 20))
cache.invalidate_all(underlying="AAPL")  # All expirations
```

### IB Options Market Data Adapter

```python
from adapters.ib.options import (
    IBOptionsMarketDataAdapter,
    create_ib_options_market_data_adapter,
    OptionsChainData,
    OptionsQuote,
)
from adapters.models import ExchangeVendor

# Create adapter via factory
adapter = create_ib_options_market_data_adapter(config={
    "host": "127.0.0.1",
    "port": 7497,  # Paper trading
    "client_id": 1,
})

# Or create directly
adapter = IBOptionsMarketDataAdapter(
    vendor=ExchangeVendor.IB,
    config={"port": 7497},
)

# Get option chain - returns OptionsChainData
chain: OptionsChainData = adapter.get_option_chain("AAPL", expiry=date(2024, 12, 20))
print(f"Strikes: {chain.strikes}")  # List[Decimal] - property, not 'all_strikes'
print(f"ATM strike: {chain.atm_strike}")

# Get single quote
quote: OptionsQuote = adapter.get_option_quote(contract)
print(f"Bid/Ask: {quote.bid}/{quote.ask}")
print(f"Mid: {quote.mid_price}")
print(f"Spread: {quote.spread_bps} bps")

# Get batch quotes
quotes = adapter.get_option_quotes_batch(contracts)

# Stream real-time quotes (async)
async for quote in adapter.stream_option_quotes_async(contracts):
    print(f"{quote.symbol}: {quote.bid}/{quote.ask}")

# Access rate limiter stats
stats = adapter.get_rate_limit_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

### IB Options Order Execution Adapter

```python
from adapters.ib.options import (
    IBOptionsOrderExecutionAdapter,
    create_ib_options_order_execution_adapter,
    OptionsOrder,
    OptionsOrderResult,
    MarginRequirement,
)
from decimal import Decimal

# Create execution adapter
exec_adapter = create_ib_options_order_execution_adapter(config={
    "host": "127.0.0.1",
    "port": 7497,
    "client_id": 2,
})

# Submit limit order
order = OptionsOrder(
    symbol="AAPL",
    expiry=date(2024, 12, 20),
    strike=Decimal("200"),
    option_type="C",
    side="BUY",
    qty=1,
    order_type="LIMIT",
    limit_price=Decimal("5.50"),
)
result: OptionsOrderResult = exec_adapter.submit_option_order(order)
if result.success:
    print(f"Order ID: {result.order_id}")
    print(f"Filled: {result.filled_qty} @ {result.avg_fill_price}")

# Get margin requirement (uses actual API fields)
margin: MarginRequirement = exec_adapter.get_option_margin_requirement(order)
print(f"Initial: ${margin.initial_margin}")
print(f"Maintenance: ${margin.maintenance_margin}")
print(f"Commission: ${margin.commission}")
print(f"Equity Impact: ${margin.equity_impact}")  # NOT 'buying_power_effect'

# Get positions
positions = exec_adapter.get_option_positions()
```

### Polygon Options Adapter

Historical options data from 2018+.

```python
from adapters.polygon import (
    PolygonOptionsAdapter,
    PolygonOptionsContract,
    PolygonOptionsQuote,
    PolygonOptionsSnapshot,
    create_polygon_options_adapter,
    parse_polygon_ticker,
)
from datetime import date
from decimal import Decimal

# Create adapter
adapter = create_polygon_options_adapter(config={"api_key": "..."})

# Get historical chain snapshot
chain = adapter.get_historical_chain("AAPL", date(2024, 1, 15))

# Create contract model
contract = PolygonOptionsContract(
    ticker="O:AAPL241220C00200000",
    underlying="AAPL",
    expiration=date(2024, 12, 20),
    strike=Decimal("200"),
    option_type="call",
)

# Create quote
quote = PolygonOptionsQuote(
    ticker="O:AAPL241220C00200000",
    bid=Decimal("5.40"),
    ask=Decimal("5.60"),
    last=Decimal("5.50"),
    timestamp_ms=1704067200000,
)
print(f"Mid price: {quote.mid_price}")  # Decimal("5.50")

# Parse Polygon ticker format
underlying, expiry, opt_type, strike = parse_polygon_ticker("O:AAPL241220C00200000")
# → ("AAPL", date(2024,12,20), "call", Decimal("200"))
```

### Registry Integration

Options adapters are registered with ExchangeVendor:

```python
from adapters.registry import AdapterType, register
from adapters.models import ExchangeVendor

# Polygon options registered automatically on import
from adapters.polygon import PolygonOptionsAdapter  # Auto-registers

# IB options registered with vendor
ExchangeVendor.IB       # Interactive Brokers
ExchangeVendor.POLYGON  # Polygon.io

# Factory functions use standard patterns
from adapters.ib.options import (
    create_ib_options_market_data_adapter,
    create_ib_options_order_execution_adapter,
)
from adapters.polygon import create_polygon_options_adapter
```

### Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| IB Rate Limiter | 25 | Chain/quote limits, priority queue, caching, stats |
| OCC Symbology | 15 | create/parse OCC, roundtrip, edge cases |
| Options Data Classes | 25 | Quote, ChainData, Order, OrderResult, Margin |
| IB Market Data Adapter | 15 | Chain, quotes, batch, streaming, rate limit stats |
| IB Order Execution | 10 | Submit order, margin, positions, cancel |
| Polygon Adapter | 12 | Contract, quote, ticker parsing, historical chain |
| Options Chain Cache | 10 | Put/get, TTL, LRU eviction, invalidation |
| Registry Integration | 7 | Vendor registration, factory functions |
| Edge Cases | 10 | Concurrent access, empty underlyings, Greeks |
| Additional Coverage | 31 | Multi-expiration, config preservation, end-to-end |

### Тестирование

```bash
# All Phase 2 Options tests (160 tests: 159 pass, 1 skip)
pytest tests/test_options_adapters.py -v

# By category
pytest tests/test_options_adapters.py::TestIBOptionsRateLimiter -v
pytest tests/test_options_adapters.py::TestOCCSymbology -v
pytest tests/test_options_adapters.py::TestOptionsDataClasses -v
pytest tests/test_options_adapters.py::TestIBOptionsMarketDataAdapter -v
pytest tests/test_options_adapters.py::TestIBOptionsOrderExecutionAdapter -v
pytest tests/test_options_adapters.py::TestPolygonOptionsAdapter -v
pytest tests/test_options_adapters.py::TestOptionsChainCache -v
pytest tests/test_options_adapters.py::TestRegistryIntegration -v
pytest tests/test_options_adapters.py::TestEdgeCases -v
```

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `adapters/ib/options.py` | IB options market data & execution (~1400 lines) |
| `adapters/ib/options_rate_limiter.py` | Priority queue rate limiter with LRU cache (~800 lines) |
| `adapters/polygon/options.py` | Polygon historical options (~500 lines) |
| `core_options.py` | OptionsContractSpec, GreeksResult, IVResult |
| `tests/test_options_adapters.py` | 160 comprehensive tests |

### Референсы

- OCC: "Options Symbology Initiative" (OSI) standard
- IB: TWS API options documentation
- Theta Data: https://www.thetadata.io/
- Polygon.io: Options API reference
- CBOE: Options market structure

---

## 📈 Options Integration (Phase 2B: Deribit Crypto Options - COMPLETED)

**Статус**: ✅ Tested and operational | **Тесты**: 118/118 (at documentation time; verify via CI) | **Date**: 2025-12-03

Phase 2B implements Deribit integration for BTC/ETH options with inverse settlement and DVOL integration.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **Deribit Market Data** | `adapters/deribit/options.py` | BTC/ETH option chains, quotes, Greeks |
| **Deribit Order Execution** | `adapters/deribit/options.py` | Single-leg orders, combo orders |
| **Inverse Margin Calculator** | `adapters/deribit/margin.py` | P&L in BTC/ETH (not USD) |
| **WebSocket Streaming** | `adapters/deribit/websocket.py` | Real-time market data & order updates |
| **Theta Data Adapter** | `adapters/theta_data/options.py` | Cost-effective US options data |

### Key Features

#### 1. Inverse Settlement

Unlike traditional USD-settled options, Deribit options settle in BTC/ETH:

```python
from adapters.deribit import DeribitOptionsAdapter, DeribitInverseMarginCalculator
from decimal import Decimal

adapter = DeribitOptionsAdapter(
    api_key="...",
    api_secret="...",
    testnet=True,
)

# P&L is in BTC, not USD
margin_calc = DeribitInverseMarginCalculator()
margin = margin_calc.calculate_margin(
    position_qty=Decimal("10"),  # 10 BTC options
    entry_price=Decimal("0.05"),  # 0.05 BTC premium
    mark_price=Decimal("0.06"),   # Current mark
    underlying_price=Decimal("45000"),  # BTC price in USD
)
# margin.pnl is in BTC
print(f"P&L: {margin.pnl} BTC")
```

#### 2. DVOL Integration

Deribit Volatility Index (DVOL) for IV surface calibration:

```python
# Get DVOL for BTC
dvol = adapter.get_dvol("BTC")
print(f"30-day IV: {dvol.volatility_30d:.2%}")
print(f"IV term structure: {dvol.term_structure}")

# Use DVOL for IV surface initialization
from impl_ssvi import SSVICalibrator
calibrator = SSVICalibrator(atm_vol=dvol.volatility_30d)
```

#### 3. WebSocket Streaming

Real-time market data for BTC/ETH options:

```python
from adapters.deribit.websocket import DeribitWebSocketClient

async def on_quote(data):
    print(f"BTC option: bid={data['bid']}, ask={data['ask']}")

client = DeribitWebSocketClient(
    on_quote=on_quote,
    testnet=True,
)

await client.connect()
await client.subscribe_quotes(["BTC-27DEC24-50000-C"])
```

#### 4. European Exercise, 24/7 Trading

```python
# Deribit options characteristics
chain = adapter.get_option_chain("BTC", expiry=date(2024, 12, 27))

# All options are European style
assert all(c.exercise_style == "european" for c in chain.contracts)

# 24/7 trading (no US market hours restrictions)
quote = adapter.get_option_quote("BTC-27DEC24-50000-C")
# Available anytime, not just 9:30-16:00 ET
```

### Registry Integration

```python
from adapters.registry import create_options_market_data_adapter
from adapters.models import ExchangeVendor, MarketType

# Via registry
adapter = create_options_market_data_adapter(
    vendor="deribit",
    config={
        "api_key": "...",
        "api_secret": "...",
        "testnet": True,
    }
)

# New enums
ExchangeVendor.DERIBIT  # Deribit exchange
MarketType.CRYPTO_OPTIONS  # Crypto options market type
```

### Theta Data Adapter

Cost-effective US options data provider:

```python
from adapters.theta_data import ThetaDataOptionsAdapter

adapter = ThetaDataOptionsAdapter(api_key="...")

# Get full option chain
chain = adapter.get_option_chain("AAPL", expiry=date(2024, 12, 20))

# Historical data (2015+)
quotes = adapter.get_historical_quotes(
    occ_symbol="AAPL  241220C00200000",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 3),
)

# End-of-day pricing (much cheaper than real-time)
eod_prices = adapter.get_eod_prices("AAPL", date(2024, 12, 3))
```

### Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| Deribit Market Data | 20 | Chain, quotes, DVOL, streaming |
| Deribit Order Execution | 18 | Market/limit orders, margin queries |
| Inverse Margin Calculator | 15 | BTC/ETH P&L, margin requirements |
| WebSocket Client | 15 | Connection, subscriptions, reconnect |
| Theta Data Adapter | 25 | Chain, historical, EOD, streaming |
| Registry Integration | 12 | Factory functions, vendor enums |
| Edge Cases | 8 | Network errors, invalid symbols |
| Integration | 5 | End-to-end workflows |

### Тестирование

```bash
# All Phase 2B tests (118 tests)
pytest tests/test_deribit_options.py -v

# By category
pytest tests/test_deribit_options.py::TestDeribitMarketData -v
pytest tests/test_deribit_options.py::TestDeribitExecution -v
pytest tests/test_deribit_options.py::TestInverseMargin -v
pytest tests/test_deribit_options.py::TestWebSocketClient -v
pytest tests/test_deribit_options.py::TestThetaDataAdapter -v
```

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `adapters/deribit/options.py` | Deribit market data & execution (~1762 lines) |
| `adapters/deribit/margin.py` | Inverse margin calculator (~783 lines) |
| `adapters/deribit/websocket.py` | WebSocket client (~777 lines) |
| `adapters/theta_data/options.py` | Theta Data adapter (~1063 lines) |
| `tests/test_deribit_options.py` | 118 comprehensive tests |

### Ключевые отличия: US Options vs Crypto Options

| Аспект | US Options (IB, Polygon) | Crypto Options (Deribit) |
|--------|--------------------------|--------------------------|
| **Settlement** | USD cash-settled | BTC/ETH inverse settled |
| **Exercise** | American (most), European (index) | European only |
| **Trading Hours** | 9:30-16:00 ET (+ extended) | 24/7 |
| **Expiration** | Monthly, weekly | Daily, weekly, monthly, quarterly |
| **Margin** | CBOE/OCC STANS | Portfolio margin (inverse) |
| **Volatility Index** | VIX (S&P 500) | DVOL (Deribit) |
| **Data Cost** | $100-500/month (real-time) | Free (Deribit), $50/month (Theta EOD) |

### Референсы

- Deribit: "Options API Documentation"
- Deribit: "DVOL Volatility Index Methodology"
- Deribit: "Inverse Futures and Options Settlement"
- Theta Data: "Options Data API Reference"
- Theta Data: "Historical Options Pricing"

---

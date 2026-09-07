# 📊 FOREX INTEGRATION PLAN - L2+ Parametric TCA with OTC Dealer Simulation

## Executive Summary

Комплексный план интеграции FX в CustodiaCloud с параметрической TCA моделью (L2+) и OTC dealer simulation для повышения реалистичности симуляции. План не меняет CCEA boundary и не меняет equities-first коммерческий фокус (FX — расширение по validated demand).

**Ключевое архитектурное решение**: Forex -- это OTC (Over-The-Counter) рынок с дилерскими котировками, а НЕ биржевой рынок с центральным order book. Поэтому:

- Используется **L2+ Parametric TCA** (как для crypto/equity), НЕ L3 LOB simulation
- **OTC Dealer Simulation** -- отдельный модуль в `services/`, НЕ в `lob/`
- Концепция "L3" неприменима напрямую к OTC рынкам

**Ключевой принцип тестирования**: Zero Regression Policy

- **Mandatory regression gates** после каждой фазы
- **Isolation tests** для предотвращения cross-contamination
- **Backward compatibility tests** для сохранения API контрактов
- **CI/CD pipeline** с автоматическими проверками

**Статус**: 📋 Plan v2.1 (Testing & Regression Hardened)
**Дата**: 2025-11-29
**Estimated LOC**: 6,000
**Estimated Tests**: 735 (включая 110 regression/isolation/backward compat)
**Timeline**: 17 недель (1 инженер)

---

## 📏 Units Convention (ВАЖНО!)

Для избежания путаницы, все единицы измерения стандартизированы:

| Величина | Единица | Пример | Конверсия |
|----------|---------|--------|-----------|
| **Spread** | Pips | EUR/USD: 1.0 pip | 1 pip = 0.0001 (JPY: 0.01) |
| **Slippage** | Pips | 0.5 pips | 1 pip ≈ 1 bps для EUR/USD |
| **ADV** | USD equivalent | $500B | Через quote currency |
| **Impact coefficient** | Dimensionless | 0.03 | k × √participation |
| **Participation** | Fraction | 0.001 = 0.1% | Order / Session ADV |
| **Interest rates** | Annual % | 5.25% | Fed Funds Rate |
| **Swap points** | Pips/day | -0.5 pips/day | Long/Short asymmetric |

**Pip Sizes по валютам**:

- Standard (4 decimals): EUR, GBP, AUD, NZD, CHF, CAD → 0.0001
- JPY pairs (2 decimals): USD/JPY, EUR/JPY, GBP/JPY → 0.01

---

## 🎯 Ключевые требования

### Функциональные

1. **Полный pipeline** от загрузки данных до live trading
2. **L2+ parametric TCA** с OTC dealer simulation (95%+ реализм)
3. **Параллельность** с crypto/equity (никаких регрессий)
4. **Гибкая конфигурация** через YAML
5. **100% покрытие тестами** нового функционала + регрессионные тесты

### Архитектурные

1. Использование существующего adapter registry pattern
2. `ForexParametricSlippageProvider` в `execution_providers.py` (НЕ отдельный файл)
3. OTC dealer simulation в `services/forex_dealer.py` (НЕ в `lob/`)
4. Интеграция с `features_pipeline.py`
5. Интеграция с risk management system
6. Поддержка PBT/SA-PPO training

---

## 📐 Forex Market Microstructure Analysis

### Ключевые отличия от Crypto/Equity

| Аспект | Crypto | Equity | **Forex** |
|--------|--------|--------|-----------|
| **Market structure** | Central LOB | Central LOB | **OTC Dealer Network** |
| **Simulation approach** | L3 LOB possible | L3 LOB possible | **L2+ Parametric + OTC Sim** |
| **Trading hours** | 24/7 | NYSE 9:30-16:00 ET | **Sun 5pm - Fri 5pm ET** |
| **Fees** | Maker/Taker % | $0 + regulatory | **Spread-based (0 commission)** |
| **Order book** | Real LOB | Real LOB | **Dealer quotes (no FIFO)** |
| **Liquidity** | Varies by coin | Market cap based | **Session-dependent** |
| **Leverage** | 1x-125x | 1x-4x (margin) | **50:1 - 500:1** |
| **Tick size** | Varies | $0.01 | **Pip (0.0001/0.01 for JPY)** |
| **Settlement** | T+0 | T+2 | **T+2 (spot), T+0 (rolling)** |
| **Execution model** | Exchange matching | Exchange matching | **Last-look dealer discretion** |

### Forex Sessions (Критично для моделирования)

| Session | Время (UTC) | Время (ET) | Liquidity Factor | Major Pairs | Spread Multiplier |
|---------|-------------|------------|------------------|-------------|-------------------|
| **Sydney** | 21:00-06:00 | 4pm-1am | 0.60-0.70 | AUD, NZD | 1.4-1.6x |
| **Tokyo** | 00:00-09:00 | 7pm-4am | 0.70-0.85 | JPY crosses | 1.2-1.4x |
| **London** | 07:00-16:00 | 2am-11am | **1.00-1.20** | EUR, GBP | 1.0x |
| **New York** | 12:00-21:00 | 7am-4pm | **1.00-1.15** | USD pairs | 1.0x |
| **London/NY overlap** | 12:00-16:00 | 7am-11am | **1.30-1.50** | ALL MAJORS | **0.8x** (tightest) |
| **Tokyo/London overlap** | 07:00-09:00 | 2am-4am | 0.85-0.95 | EUR/JPY | 1.1x |
| **Weekend gap** | Fri 21:00 - Sun 21:00 | Closed | 0.00 | N/A | N/A |

### Currency Pair Classification & Spread Profiles

| Category | Examples | Retail Spread | Institutional Spread | Daily Range | ADV (BIS 2022) |
|----------|----------|---------------|---------------------|-------------|----------------|
| **Majors** | EUR/USD, USD/JPY, GBP/USD | 1.0-1.5 pips | 0.1-0.3 pips | 50-100 pips | $500B+ |
| **Minors** | EUR/GBP, EUR/CHF, GBP/CHF | 1.5-3.0 pips | 0.3-0.8 pips | 40-80 pips | $50-100B |
| **Crosses** | EUR/JPY, GBP/JPY, AUD/JPY | 2.0-5.0 pips | 0.5-1.5 pips | 70-150 pips | $20-50B |
| **Exotics** | USD/TRY, USD/ZAR, USD/MXN | 15-80 pips | 5-30 pips | 200-500 pips | $1-10B |

**Spread Profiles** (для конфигурации):

```yaml
spread_profiles:
  institutional:
    EUR_USD: 0.3
    GBP_USD: 0.5
    USD_JPY: 0.3
  retail:
    EUR_USD: 1.2
    GBP_USD: 1.5
    USD_JPY: 1.2
  conservative:
    EUR_USD: 1.8
    GBP_USD: 2.2
    USD_JPY: 1.8
```

---

## 🏗️ Architecture Design

### Layer Integration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FOREX INTEGRATION                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                   │
│  │   CRYPTO    │   │   EQUITY    │   │   FOREX     │                   │
│  │  (Binance)  │   │  (Alpaca)   │   │  (OANDA)    │   ← NEW           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                   │
│         │                 │                 │                           │
│  ┌──────┴─────────────────┴─────────────────┴──────┐                   │
│  │              Adapter Registry (adapters/)        │                   │
│  │  + ExchangeVendor.OANDA, .IG, .DUKASCOPY        │   ← UPDATE        │
│  └──────┬─────────────────┬─────────────────┬──────┘                   │
│         │                 │                 │                           │
│  ┌──────┴─────────────────┴─────────────────┴──────┐                   │
│  │     Unified execution_providers.py (L2+)        │                   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌────────────────┐            │
│  │  │CryptoParametric│ │EquityParametric│ │ForexParametric │ ← NEW     │
│  │  │SlippageProvider│ │SlippageProvider│ │SlippageProvider│           │
│  │  └───────────────┘ └───────────────┘ └────────────────┘            │
│  └──────────────────────────────────────────────────┘                   │
│                             │                                           │
│  ┌──────────────────────────┴──────────────────────┐                   │
│  │         OTC Dealer Simulation (services/)        │   ← NEW          │
│  │  ┌─────────────────────────────────────────────┐│                   │
│  │  │  ForexDealerSimulator (NOT in lob/)         ││                   │
│  │  │  - Multi-dealer quote aggregation           ││                   │
│  │  │  - Last-look rejection simulation           ││                   │
│  │  │  - NO queue position (OTC, not exchange)    ││                   │
│  │  └─────────────────────────────────────────────┘│                   │
│  └──────────────────────────────────────────────────┘                   │
│                             │                                           │
│  ┌──────────────────────────┴──────────────────────┐                   │
│  │      Feature Pipeline (features_pipeline.py)     │                   │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │                   │
│  │  │ FG Index │  │VIX/RS SPY│  │ Carry/IR/DXY  │ │   ← NEW           │
│  │  │ (crypto) │  │ (equity) │  │   (forex)     │ │                   │
│  │  └──────────┘  └──────────┘  └───────────────┘ │                   │
│  └──────────────────────────────────────────────────┘                   │
│                             │                                           │
│  ┌──────────────────────────┴──────────────────────┐                   │
│  │              Services Layer                       │                   │
│  │  ┌───────────────────┐  ┌────────────────────┐  │                   │
│  │  │forex_position_sync│  │forex_session_router│  │   ← NEW           │
│  │  └───────────────────┘  └────────────────────┘  │                   │
│  └──────────────────────────────────────────────────┘                   │
│                             │                                           │
│  ┌──────────────────────────┴──────────────────────┐                   │
│  │              Risk Management System              │                   │
│  │  - Position limits                               │                   │
│  │  - Drawdown guards                               │                   │
│  │  - Leverage monitoring (NEW: Forex 50:1-500:1)  │   ← UPDATE        │
│  │  - Margin call simulation                        │   ← NEW           │
│  │  - Swap/rollover costs                           │   ← NEW           │
│  │  - Kill switch                                   │                   │
│  └──────────────────────────────────────────────────┘                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### New Files Structure

```
AI-Powered-Quantitative-Research-Platform/
├── adapters/
│   ├── models.py                          # UPDATE: +OANDA, +IG, +DUKASCOPY
│   ├── registry.py                        # UPDATE: +lazy loading for oanda
│   └── oanda/                             # NEW: OANDA adapter package
│       ├── __init__.py
│       ├── market_data.py                 # Historical + streaming data
│       ├── fees.py                        # Spread-based fee model
│       ├── trading_hours.py               # Forex sessions calendar (DST-aware)
│       ├── exchange_info.py               # Currency pairs info
│       └── order_execution.py             # Order placement API
│
├── execution_providers.py                 # UPDATE: +ForexParametricSlippageProvider
│                                          # (НЕ отдельный файл!)
│
├── forex_features.py                      # NEW: Forex-specific features
│
├── services/
│   ├── forex_dealer.py                    # NEW: OTC dealer simulation (NOT in lob/)
│   ├── forex_risk_guards.py               # NEW: Margin/leverage guards
│   ├── forex_position_sync.py             # NEW: Position sync with OANDA
│   └── forex_session_router.py            # NEW: Session-aware order routing
│
├── scripts/
│   ├── download_forex_data.py             # NEW: Data downloader
│   ├── download_swap_rates.py             # NEW: Swap rates from OANDA/FRED
│   └── download_economic_calendar.py      # NEW: Economic calendar events
│
├── data/
│   └── forex/
│       ├── rates/                         # Central bank rates (FRED)
│       ├── calendar/                      # Economic calendar events
│       └── swaps/                         # Historical swap rates
│
├── configs/
│   ├── config_train_forex.yaml            # NEW
│   ├── config_backtest_forex.yaml         # NEW
│   ├── config_live_oanda.yaml             # NEW
│   └── forex_defaults.yaml                # NEW: Forex-specific defaults
│
├── features_pipeline.py                   # UPDATE: +forex feature integration
│
└── tests/
    ├── test_oanda_adapters.py             # NEW: 130 tests
    ├── test_forex_execution_providers.py  # NEW: 100 tests
    ├── test_forex_features.py             # NEW: 70 tests
    ├── test_forex_dealer_simulation.py    # NEW: 80 tests
    ├── test_forex_risk_guards.py          # NEW: 60 tests
    ├── test_forex_position_sync.py        # NEW: 40 tests
    ├── test_forex_session_router.py       # NEW: 30 tests
    ├── test_forex_integration.py          # NEW: 50 tests
    ├── test_forex_properties.py           # NEW: Property-based tests (40)
    └── test_forex_stress.py               # NEW: Stress tests (20)
```

---

## 📅 Implementation Phases

### Phase 0: Foundation & Research (Week 1)

**Цель**: Исследование и подготовка инфраструктуры

#### 0.1 Market Research

- [ ] Изучить OANDA v20 API documentation
- [ ] Изучить альтернативы (IG Markets, Dukascopy, FXCM)
- [ ] Собрать reference spreads по currency pairs (BIS Triennial Survey 2022)
- [ ] Проанализировать session liquidity patterns
- [ ] Изучить last-look mechanics (Oomen 2017)
- [ ] Изучить OANDA API rate limits (120 requests/sec)

#### 0.2 Architecture Design

- [ ] Финализировать file structure
- [ ] Определить interfaces для forex adapters (следуя Alpaca pattern)
- [ ] Спланировать backward compatibility checks
- [ ] Определить data sources:
  - **Price data**: OANDA v20 API (primary), Dukascopy (tick data backup)
  - **Interest rates**: FRED API (Federal Funds, ECB, BOJ, BOE rates)
  - **Economic calendar**: OANDA Labs Calendar API + ForexFactory (backup)
  - **Swap rates**: OANDA API `/v3/accounts/{id}/instruments` (financing field)

#### 0.3 Test Infrastructure

- [ ] Создать test fixtures для forex data
- [ ] Настроить mock OANDA API для тестов (VCR pattern)
- [ ] Добавить forex в CI/CD pipeline
- [ ] Setup property-based testing с Hypothesis

**Deliverables**:

- Research document
- Finalized architecture diagram
- Test infrastructure
- Data source mapping

**Tests**: ~25 (infrastructure + mock setup)

---

### Phase 1: Core Enums & Models (Week 2)

**Цель**: Расширение базовых моделей для поддержки Forex

#### 1.1 Update adapters/models.py

```python
# =========================
# ExchangeVendor UPDATES
# =========================
class ExchangeVendor(str, Enum):
    """Supported exchange vendors."""
    # Existing
    BINANCE = "binance"
    BINANCE_US = "binance_us"
    ALPACA = "alpaca"
    POLYGON = "polygon"
    YAHOO = "yahoo"
    # NEW: Forex vendors
    OANDA = "oanda"              # Primary forex broker
    IG = "ig"                    # Alternative (IG Markets)
    DUKASCOPY = "dukascopy"      # Historical tick data
    UNKNOWN = "unknown"

    @property
    def market_type(self) -> MarketType:
        """Default market type for vendor."""
        if self in (ExchangeVendor.BINANCE, ExchangeVendor.BINANCE_US):
            return MarketType.CRYPTO_SPOT
        elif self == ExchangeVendor.ALPACA:
            return MarketType.EQUITY
        elif self in (ExchangeVendor.OANDA, ExchangeVendor.IG, ExchangeVendor.DUKASCOPY):
            return MarketType.FOREX  # NEW
        return MarketType.CRYPTO_SPOT


# =========================
# MarketType.FOREX уже существует! ✓
# =========================
# class MarketType(str, Enum):
#     ...
#     FOREX = "FOREX"  # Already exists at line 37


# =========================
# NEW: Forex Sessions Enum
# =========================
class ForexSessionType(str, Enum):
    """Forex trading session type."""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    TOKYO_LONDON_OVERLAP = "tokyo_london_overlap"
    WEEKEND = "weekend"
    OFF_HOURS = "off_hours"


# =========================
# NEW: Forex Session Dataclass
# =========================
@dataclass(frozen=True)
class ForexSessionWindow:
    """Forex session trading window."""
    session_type: ForexSessionType
    start_hour_utc: int
    end_hour_utc: int
    liquidity_factor: float
    spread_multiplier: float
    days_of_week: Tuple[int, ...] = (0, 1, 2, 3, 4)  # Mon-Fri


FOREX_SESSION_WINDOWS: List[ForexSessionWindow] = [
    ForexSessionWindow(ForexSessionType.SYDNEY, 21, 6, 0.65, 1.5, (0, 1, 2, 3, 4, 6)),
    ForexSessionWindow(ForexSessionType.TOKYO, 0, 9, 0.75, 1.3, (0, 1, 2, 3, 4)),
    ForexSessionWindow(ForexSessionType.LONDON, 7, 16, 1.10, 1.0, (0, 1, 2, 3, 4)),
    ForexSessionWindow(ForexSessionType.NEW_YORK, 12, 21, 1.05, 1.0, (0, 1, 2, 3, 4)),
    ForexSessionWindow(ForexSessionType.LONDON_NY_OVERLAP, 12, 16, 1.35, 0.8, (0, 1, 2, 3, 4)),
]
```

#### 1.2 Update execution_providers.py (CRITICAL!)

**ВАЖНО**: `ForexParametricSlippageProvider` добавляется в СУЩЕСТВУЮЩИЙ `execution_providers.py`, НЕ в отдельный файл!

```python
# =========================
# AssetClass UPDATE (line ~58)
# =========================
class AssetClass(enum.Enum):
    """Asset class enumeration for execution providers."""
    CRYPTO = "crypto"
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"  # NEW


# =========================
# Factory function UPDATE
# =========================
def create_slippage_provider(
    level: str,
    asset_class: AssetClass,
    config: Optional[Dict[str, Any]] = None,
) -> SlippageProvider:
    """Create slippage provider for asset class."""
    if level == "L2+":
        if asset_class == AssetClass.CRYPTO:
            return CryptoParametricSlippageProvider(
                config=CryptoParametricConfig(**(config or {}))
            )
        elif asset_class == AssetClass.EQUITY:
            return EquityParametricSlippageProvider(
                config=EquityParametricConfig(**(config or {}))
            )
        elif asset_class == AssetClass.FOREX:  # NEW
            return ForexParametricSlippageProvider(
                config=ForexParametricConfig(**(config or {}))
            )
    # ... rest of factory
```

#### 1.3 Update adapters/registry.py

```python
# Lazy loading modules (line ~165)
self._lazy_modules: Dict[ExchangeVendor, str] = {
    ExchangeVendor.BINANCE: "adapters.binance",
    ExchangeVendor.BINANCE_US: "adapters.binance",
    ExchangeVendor.ALPACA: "adapters.alpaca",
    ExchangeVendor.YAHOO: "adapters.yahoo",
    ExchangeVendor.OANDA: "adapters.oanda",        # NEW
    ExchangeVendor.IG: "adapters.ig",              # NEW (future)
    ExchangeVendor.DUKASCOPY: "adapters.dukascopy", # NEW (future)
}
```

#### 1.4 Verify data_loader_multi_asset.py

```python
# Already has FOREX! ✓ (line 62)
class AssetClass(str, Enum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    FOREX = "forex"  # ✓ Already exists

# Add OANDA to DataVendor
class DataVendor(str, Enum):
    BINANCE = "binance"
    ALPACA = "alpaca"
    POLYGON = "polygon"
    OANDA = "oanda"      # NEW
    CSV = "csv"
    FEATHER = "feather"
    PARQUET = "parquet"
```

**Deliverables**:

- Updated enums in `adapters/models.py`
- Updated `AssetClass` in `execution_providers.py`
- Updated registry lazy loading
- Verified `data_loader_multi_asset.py` compatibility
- Unit tests for new enums

**Tests**: ~35

---

### Phase 2: OANDA Adapter Implementation (Weeks 3-4)

**Цель**: Полная реализация OANDA adapter по паттерну Alpaca

#### 2.1 adapters/oanda/**init**.py

```python
"""
OANDA Forex Adapter Package

Supports:
- OANDA v20 REST API
- Streaming prices via WebSocket
- Historical candles (M1 to M)
- Order execution with last-look handling
- Swap rate queries

API Rate Limits:
- 120 requests per second (streaming is separate)
- Max 5000 candles per request
- Burst: 200 requests allowed

Environment Variables:
- OANDA_API_KEY: API access token
- OANDA_ACCOUNT_ID: Trading account ID
- OANDA_PRACTICE: "true" for demo, "false" for live
"""
from .market_data import OandaMarketDataAdapter
from .fees import OandaFeeAdapter
from .trading_hours import OandaTradingHoursAdapter
from .exchange_info import OandaExchangeInfoAdapter
from .order_execution import OandaOrderExecutionAdapter

__all__ = [
    "OandaMarketDataAdapter",
    "OandaFeeAdapter",
    "OandaTradingHoursAdapter",
    "OandaExchangeInfoAdapter",
    "OandaOrderExecutionAdapter",
]
```

#### 2.2 adapters/oanda/market_data.py (~450 LOC)

```python
"""
OANDA Market Data Adapter

Features:
- Historical candles with bid/ask prices
- Real-time streaming via WebSocket
- Auto-reconnect with exponential backoff
- Rate limiting (120 req/s)

Timeframes:
- S5, S10, S15, S30: Seconds
- M1, M2, M4, M5, M10, M15, M30: Minutes
- H1, H2, H3, H4, H6, H8, H12: Hours
- D, W, M: Day, Week, Month
"""
import asyncio
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor
from core_models import Bar, Tick

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for OANDA API."""

    def __init__(self, rate: float = 120.0, burst: int = 200):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self.rate
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class OandaMarketDataAdapter(MarketDataAdapter):
    """OANDA v20 API market data adapter."""

    # OANDA API endpoints
    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"
    STREAM_PRACTICE_URL = "https://stream-fxpractice.oanda.com"
    STREAM_LIVE_URL = "https://stream-fxtrade.oanda.com"

    # Timeframe mapping
    TIMEFRAME_MAP = {
        "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
        "1h": "H1", "4h": "H4", "1d": "D", "1w": "W",
        "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
        "H1": "H1", "H4": "H4", "D": "D", "W": "W",
    }

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.OANDA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._config = config or {}
        self._api_key = self._config.get("api_key") or os.getenv("OANDA_API_KEY")
        self._account_id = self._config.get("account_id") or os.getenv("OANDA_ACCOUNT_ID")
        self._practice = self._config.get("practice", True)

        if self._practice:
            self._base_url = self.PRACTICE_URL
            self._stream_url = self.STREAM_PRACTICE_URL
        else:
            self._base_url = self.LIVE_URL
            self._stream_url = self.STREAM_LIVE_URL

        self._rate_limiter = RateLimiter()
        self._session = None

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Bar]:
        """
        Fetch historical candles from OANDA.

        Args:
            symbol: Currency pair (e.g., "EUR_USD", "GBP/JPY" -> normalized to "GBP_JPY")
            timeframe: Bar timeframe (e.g., "1h", "4h", "1d", "H4")
            limit: Maximum bars (max 5000 per request)
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds

        Returns:
            List of Bar objects with bid/ask/mid prices
        """
        instrument = self._normalize_symbol(symbol)
        granularity = self.TIMEFRAME_MAP.get(timeframe.lower(), timeframe.upper())

        # Build request
        params = {
            "granularity": granularity,
            "count": min(limit, 5000),
            "price": "MBA",  # Mid, Bid, Ask
        }

        if start_ts:
            params["from"] = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc).isoformat()
        if end_ts:
            params["to"] = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc).isoformat()

        # ... API call implementation
        return []  # Placeholder

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to OANDA format (EUR_USD)."""
        return symbol.replace("/", "_").upper()

    async def stream_prices_async(
        self,
        symbols: List[str],
    ) -> AsyncIterator[Tick]:
        """
        Stream real-time prices via WebSocket.

        Yields:
            Tick objects with bid/ask prices
        """
        # ... WebSocket implementation
        pass
```

#### 2.3 adapters/oanda/trading_hours.py (~300 LOC)

```python
"""
Forex Trading Hours Adapter with DST Awareness

Forex Market Hours:
- Opens: Sunday 5:00 PM ET (21:00 UTC winter, 20:00 UTC summer)
- Closes: Friday 5:00 PM ET
- Daily rollover: 5:00 PM ET

DST Handling:
- US DST: Second Sunday March to First Sunday November
- Rollover time shifts between 21:00 and 22:00 UTC
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from adapters.base import TradingHoursAdapter
from adapters.models import ExchangeVendor, ForexSessionType, ForexSessionWindow

logger = logging.getLogger(__name__)


class OandaTradingHoursAdapter(TradingHoursAdapter):
    """Forex trading hours adapter with DST awareness."""

    # US Eastern timezone
    ET = ZoneInfo("America/New_York")

    # Rollover time in ET (constant regardless of DST)
    ROLLOVER_HOUR_ET = 17  # 5:00 PM ET

    def is_market_open(
        self,
        ts: int,
        *,
        session_type: Optional[str] = None,
    ) -> bool:
        """
        Check if forex market is open.

        Args:
            ts: Unix timestamp in milliseconds
            session_type: Optional session filter

        Returns:
            True if market is open
        """
        dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        dt_et = dt_utc.astimezone(self.ET)
        weekday = dt_et.weekday()
        hour_et = dt_et.hour

        # Closed Saturday all day
        if weekday == 5:  # Saturday
            return False

        # Sunday: opens at 5pm ET
        if weekday == 6:  # Sunday
            return hour_et >= self.ROLLOVER_HOUR_ET

        # Friday: closes at 5pm ET
        if weekday == 4:  # Friday
            return hour_et < self.ROLLOVER_HOUR_ET

        # Mon-Thu: 24 hours
        return True

    def get_current_session(self, ts: int) -> Tuple[ForexSessionType, float, float]:
        """
        Determine current forex session.

        Returns:
            (session_type, liquidity_factor, spread_multiplier)
        """
        dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        hour_utc = dt_utc.hour
        weekday = dt_utc.weekday()

        # Weekend
        if weekday == 5 or (weekday == 6 and hour_utc < 21):
            return (ForexSessionType.WEEKEND, 0.0, float('inf'))

        # Check overlaps first (they take priority)
        if 12 <= hour_utc < 16:
            return (ForexSessionType.LONDON_NY_OVERLAP, 1.35, 0.8)
        if 7 <= hour_utc < 9:
            return (ForexSessionType.TOKYO_LONDON_OVERLAP, 0.90, 1.1)

        # Individual sessions
        if 21 <= hour_utc or hour_utc < 6:
            return (ForexSessionType.SYDNEY, 0.65, 1.5)
        if 0 <= hour_utc < 9:
            return (ForexSessionType.TOKYO, 0.75, 1.3)
        if 7 <= hour_utc < 16:
            return (ForexSessionType.LONDON, 1.10, 1.0)
        if 12 <= hour_utc < 21:
            return (ForexSessionType.NEW_YORK, 1.05, 1.0)

        return (ForexSessionType.OFF_HOURS, 0.50, 2.0)

    def get_rollover_time_utc(self, date: datetime) -> datetime:
        """
        Get rollover time in UTC for a given date (DST-aware).

        Returns:
            Rollover datetime in UTC
        """
        # Create 5pm ET for the date
        dt_et = date.replace(
            hour=self.ROLLOVER_HOUR_ET,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=self.ET
        )
        return dt_et.astimezone(timezone.utc)
```

#### 2.4 - 2.6: Other Adapters (fees.py, exchange_info.py, order_execution.py)

*Similar patterns to Alpaca adapters - see Phase 2 in original plan*

**Deliverables**:

- Complete OANDA adapter package (5 files)
- Rate limiting implementation
- DST-aware trading hours
- Registration in adapter registry
- Environment variable support
- Async streaming support

**Tests**: ~130 (including rate limit tests, DST edge cases)

---

### Phase 3: ForexParametricSlippageProvider (L2+) (Weeks 5-6)

**Цель**: Research-backed parametric TCA model для Forex

**ВАЖНО**: Добавляется в СУЩЕСТВУЮЩИЙ `execution_providers.py`, следуя паттерну `CryptoParametricSlippageProvider` и `EquityParametricSlippageProvider`.

#### 3.1 Добавить в execution_providers.py (~550 LOC addition)

```python
"""
Forex Parametric TCA Model (L2+)

Extends Almgren-Chriss √participation with 8 forex-specific factors:
1. √Participation impact (base)
2. Session liquidity (Sydney/Tokyo/London/NY/overlaps)
3. Spread regime (tight/normal/wide)
4. Interest rate differential (carry)
5. Volatility regime (ATR-based)
6. News event proximity
7. DXY correlation (for non-USD pairs)
8. Pair type multiplier (major/minor/cross/exotic)

References:
- Lyons (2001): "The Microstructure Approach to Exchange Rates"
- Evans & Lyons (2002): "Order Flow and Exchange Rate Dynamics"
- Berger et al. (2008): "The Development of the Global FX Market"
- King, Osler, Rime (2012): "Foreign Exchange Market Structure"
- BIS (2022): Triennial Central Bank Survey of FX Markets
- Chaboud et al. (2014): "Rise of the Machines" (Algorithmic FX Trading)
"""

class ForexSession(enum.Enum):
    """Forex trading session."""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    TOKYO_LONDON_OVERLAP = "tokyo_london_overlap"
    OFF_HOURS = "off_hours"
    WEEKEND = "weekend"


class PairType(enum.Enum):
    """Currency pair classification."""
    MAJOR = "major"      # G7 pairs: EUR/USD, USD/JPY, GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD
    MINOR = "minor"      # Cross without USD: EUR/GBP, EUR/CHF, GBP/CHF
    CROSS = "cross"      # JPY crosses: EUR/JPY, GBP/JPY, AUD/JPY
    EXOTIC = "exotic"    # EM currencies: USD/TRY, USD/ZAR, USD/MXN, USD/PLN


@dataclass
class ForexParametricConfig:
    """
    Configuration for ForexParametricSlippageProvider.

    All spreads and slippage values are in PIPS (not bps).
    1 pip = 0.0001 for most pairs, 0.01 for JPY pairs.
    """
    # Base impact coefficient (Almgren-Chriss style)
    # Lower than crypto (0.10) and equity (0.05) due to higher FX liquidity
    impact_coef_base: float = 0.03
    impact_coef_range: Tuple[float, float] = (0.02, 0.05)

    # Default spreads by pair type (pips) - RETAIL profile
    # Override with spread_profile for institutional
    default_spreads_pips: Dict[str, float] = field(default_factory=lambda: {
        "major": 1.2,
        "minor": 2.0,
        "cross": 3.0,
        "exotic": 25.0,
    })

    # Spread profiles for different client types
    spread_profiles: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "institutional": {"major": 0.3, "minor": 0.8, "cross": 1.5, "exotic": 8.0},
        "retail": {"major": 1.2, "minor": 2.0, "cross": 3.0, "exotic": 25.0},
        "conservative": {"major": 1.8, "minor": 3.0, "cross": 4.5, "exotic": 40.0},
    })

    # Session liquidity factors (1.0 = normal liquidity)
    session_liquidity: Dict[str, float] = field(default_factory=lambda: {
        "sydney": 0.65,
        "tokyo": 0.75,
        "london": 1.10,
        "new_york": 1.05,
        "london_ny_overlap": 1.35,
        "tokyo_london_overlap": 0.90,
        "off_hours": 0.50,
        "weekend": 0.0,  # Market closed
    })

    # Interest rate differential sensitivity
    # Positive carry = tighter spreads (more market makers)
    carry_sensitivity: float = 0.03  # 3% slippage adjustment per 1% rate differential

    # DXY correlation decay for non-USD pairs
    # Low correlation with DXY = less USD liquidity spillover
    dxy_correlation_decay: float = 0.25

    # News event impact multipliers
    news_event_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "nfp": 3.0,           # Non-Farm Payrolls (huge impact)
        "fomc": 2.5,          # Fed decisions
        "ecb": 2.0,           # ECB decisions
        "boe": 1.8,           # Bank of England
        "boj": 1.8,           # Bank of Japan
        "rba": 1.5,           # Reserve Bank of Australia
        "cpi": 1.8,           # Inflation data
        "gdp": 1.5,           # GDP releases
        "pmi": 1.3,           # PMI data
        "retail_sales": 1.2,
        "other": 1.1,
    })

    # Pair type slippage multipliers
    pair_type_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "major": 1.0,
        "minor": 1.4,
        "cross": 1.8,
        "exotic": 3.5,
    })

    # Volatility regime multipliers
    vol_regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.80,      # VIX-equivalent < 10
        "normal": 1.00,   # VIX-equivalent 10-20
        "high": 1.50,     # VIX-equivalent 20-30
        "extreme": 2.50,  # VIX-equivalent > 30 (flash crash, crisis)
    })

    # Bounds (in pips)
    min_slippage_pips: float = 0.05
    max_slippage_pips: float = 150.0  # For exotic pairs during news

    # Adaptive coefficient learning rate
    adaptive_learning_rate: float = 0.1


class ForexParametricSlippageProvider:
    """
    L2+: Smart parametric TCA model for forex markets.

    Formula:
        slippage_pips = half_spread_pips
            × (1 + k × √participation)          # Almgren-Chriss impact
            × (1 / session_liquidity_factor)    # Session adjustment
            × volatility_regime_mult            # Vol regime
            × (1 + carry_adjustment)            # Interest rate differential
            × dxy_correlation_factor            # USD liquidity spillover
            × news_event_factor                 # Economic calendar
            × pair_type_multiplier              # Major/Minor/Exotic

    Key differences from Crypto/Equity:
    - All values in PIPS, not basis points
    - Session-based liquidity (overlaps are best)
    - Carry trade considerations
    - No central LOB (OTC dealer market)
    - Last-look rejection handled separately
    """

    # Currency pair classifications
    MAJORS = frozenset({
        "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF",
        "AUD_USD", "USD_CAD", "NZD_USD"
    })
    MINORS = frozenset({
        "EUR_GBP", "EUR_CHF", "GBP_CHF", "EUR_AUD",
        "EUR_CAD", "EUR_NZD", "GBP_AUD", "GBP_CAD"
    })
    EXOTICS = frozenset({
        "USD_TRY", "USD_ZAR", "USD_MXN", "USD_PLN",
        "USD_HUF", "USD_CZK", "USD_SGD", "USD_HKD",
        "USD_NOK", "USD_SEK", "USD_DKK"
    })

    def __init__(
        self,
        config: Optional[ForexParametricConfig] = None,
        spread_profile: str = "retail",
    ) -> None:
        self.config = config or ForexParametricConfig()
        self.spread_profile = spread_profile
        self._adaptive_k: float = self.config.impact_coef_base
        self._fill_quality_history: List[Tuple[float, float]] = []

    @classmethod
    def from_profile(cls, profile: str) -> "ForexParametricSlippageProvider":
        """
        Create provider from named profile.

        Profiles:
        - "retail": Standard retail spreads (default)
        - "institutional": Tight institutional spreads
        - "conservative": Wider spreads for safety margin
        - "major_only": Optimized for major pairs
        - "exotic": Wider spreads for exotic pairs
        """
        profiles = {
            "retail": ForexParametricConfig(),
            "institutional": ForexParametricConfig(
                impact_coef_base=0.02,
                min_slippage_pips=0.02,
            ),
            "conservative": ForexParametricConfig(
                impact_coef_base=0.04,
                min_slippage_pips=0.1,
                max_slippage_pips=200.0,
            ),
        }
        config = profiles.get(profile, ForexParametricConfig())
        return cls(config=config, spread_profile=profile)

    def compute_slippage_pips(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        *,
        session: Optional[ForexSession] = None,
        pair_type: Optional[PairType] = None,
        interest_rate_diff: Optional[float] = None,
        dxy_correlation: Optional[float] = None,
        upcoming_news: Optional[str] = None,
        recent_returns: Optional[Sequence[float]] = None,
    ) -> float:
        """
        Compute expected slippage in pips.

        Args:
            order: Order to execute
            market: Current market state
            participation_ratio: Order size / estimated session volume
            session: Current forex session
            pair_type: Currency pair classification
            interest_rate_diff: Base rate - Quote rate (annual %)
            dxy_correlation: Correlation with Dollar Index (-1 to 1)
            upcoming_news: Type of upcoming economic event
            recent_returns: Recent returns for volatility regime detection

        Returns:
            Expected slippage in pips (always positive)
        """
        # 1. Determine pair type and get base spread
        pair_type = pair_type or self._classify_pair(order.symbol)
        spreads = self.config.spread_profiles.get(
            self.spread_profile,
            self.config.default_spreads_pips
        )
        half_spread = spreads.get(pair_type.value, 1.5) / 2.0

        # Override with market spread if available (convert bps to pips)
        if market.spread_bps is not None and math.isfinite(market.spread_bps):
            # 1 pip ≈ 1 bps for most pairs (rough approximation)
            half_spread = market.spread_bps / 10.0 / 2.0

        # 2. √Participation impact (Almgren-Chriss)
        participation = max(1e-12, abs(participation_ratio))
        impact = self._adaptive_k * math.sqrt(participation)

        # 3. Session liquidity adjustment
        session = session or self._detect_session(market.timestamp)
        session_factor = self.config.session_liquidity.get(session.value, 1.0)

        # Invert: low liquidity = higher slippage
        session_adjustment = 1.0 / max(0.3, session_factor) if session_factor > 0 else 10.0

        # 4. Volatility regime
        vol_regime = self._detect_volatility_regime(recent_returns)
        vol_mult = self.config.vol_regime_multipliers.get(vol_regime, 1.0)

        # 5. Carry adjustment
        carry_factor = 1.0
        if interest_rate_diff is not None:
            # Positive carry (long high-yield) = more liquidity = lower slippage
            carry_factor = 1.0 - interest_rate_diff * self.config.carry_sensitivity
            carry_factor = max(0.7, min(1.4, carry_factor))

        # 6. DXY correlation (for non-USD pairs only)
        dxy_factor = 1.0
        if dxy_correlation is not None and not self._has_usd(order.symbol):
            # Lower correlation with USD = less liquidity spillover
            dxy_factor = 1.0 + (1.0 - abs(dxy_correlation)) * self.config.dxy_correlation_decay

        # 7. News event impact
        news_factor = 1.0
        if upcoming_news is not None:
            news_factor = self.config.news_event_multipliers.get(
                upcoming_news.lower(), 1.0
            )

        # 8. Pair type multiplier
        pair_mult = self.config.pair_type_multipliers.get(pair_type.value, 1.0)

        # Combine all factors
        total_slippage = (
            half_spread
            * (1.0 + impact)  # Impact term is already dimensionless
            * session_adjustment
            * vol_mult
            * carry_factor
            * dxy_factor
            * news_factor
            * pair_mult
        )

        # Apply bounds
        return max(
            self.config.min_slippage_pips,
            min(self.config.max_slippage_pips, total_slippage)
        )

    def _classify_pair(self, symbol: str) -> PairType:
        """Classify currency pair."""
        norm = symbol.replace("/", "_").upper()
        if norm in self.MAJORS:
            return PairType.MAJOR
        if norm in self.MINORS:
            return PairType.MINOR
        if norm in self.EXOTICS:
            return PairType.EXOTIC
        return PairType.CROSS

    def _has_usd(self, symbol: str) -> bool:
        """Check if pair contains USD."""
        norm = symbol.replace("/", "_").upper()
        return "USD" in norm

    def _detect_session(self, timestamp: int) -> ForexSession:
        """Detect current forex session from timestamp."""
        dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        hour = dt.hour
        weekday = dt.weekday()

        # Weekend check
        if weekday == 5 or (weekday == 6 and hour < 21):
            return ForexSession.WEEKEND

        # Check overlaps first (priority)
        if 12 <= hour < 16:
            return ForexSession.LONDON_NY_OVERLAP
        if 7 <= hour < 9:
            return ForexSession.TOKYO_LONDON_OVERLAP

        # Individual sessions
        if 21 <= hour or hour < 6:
            return ForexSession.SYDNEY
        if 0 <= hour < 9:
            return ForexSession.TOKYO
        if 7 <= hour < 16:
            return ForexSession.LONDON
        if 12 <= hour < 21:
            return ForexSession.NEW_YORK

        return ForexSession.OFF_HOURS

    def _detect_volatility_regime(
        self,
        recent_returns: Optional[Sequence[float]],
    ) -> str:
        """Detect volatility regime from recent returns."""
        if recent_returns is None or len(recent_returns) < 5:
            return "normal"

        std = float(np.std(recent_returns))
        annualized_vol = std * math.sqrt(252 * 6)  # Assuming 4h bars

        if annualized_vol < 0.05:
            return "low"
        elif annualized_vol < 0.10:
            return "normal"
        elif annualized_vol < 0.20:
            return "high"
        else:
            return "extreme"

    def update_fill_quality(self, predicted: float, actual: float) -> None:
        """
        Update adaptive impact coefficient based on fill quality.

        Called after each fill to calibrate the model.
        """
        self._fill_quality_history.append((predicted, actual))

        # Keep last 100 fills
        if len(self._fill_quality_history) > 100:
            self._fill_quality_history = self._fill_quality_history[-100:]

        # Adjust k based on prediction error
        if len(self._fill_quality_history) >= 10:
            errors = [(a - p) / max(0.1, p) for p, a in self._fill_quality_history[-10:]]
            mean_error = sum(errors) / len(errors)

            # If we're consistently under/over predicting, adjust k
            k_min, k_max = self.config.impact_coef_range
            adjustment = 1.0 + mean_error * self.config.adaptive_learning_rate
            self._adaptive_k = max(k_min, min(k_max, self._adaptive_k * adjustment))
```

**Deliverables**:

- `ForexParametricSlippageProvider` added to `execution_providers.py`
- `ForexParametricConfig` with all 8 factors
- `ForexFeeProvider` (returns 0, cost is in spread)
- Factory function updates
- Spread profiles: institutional, retail, conservative
- Adaptive coefficient learning

**Tests**: ~100

---

### Phase 4: Forex Features Pipeline (Weeks 7-8)

**Цель**: Forex-специфичные features + integration с существующим pipeline

#### 4.1 forex_features.py (~450 LOC)

```python
"""
Forex-specific features for ML training.

Features parallel to crypto Fear&Greed and equity VIX:
1. Interest Rate Differential (Carry) - analogous to funding rate
2. Relative Strength vs DXY - analogous to RS vs SPY
3. Session indicators (one-hot)
4. Spread regime
5. COT positioning (Commitments of Traders)
6. Economic calendar proximity
7. Cross-currency momentum
8. Implied volatility (FX VIX equivalent)

Data Sources:
- Interest rates: FRED API (Federal Reserve Economic Data)
- DXY: Yahoo Finance (DX-Y.NYB)
- COT: CFTC weekly reports
- Economic calendar: OANDA Labs API / ForexFactory
- Implied vol: OANDA streaming quotes

References:
- Brunnermeier et al. (2008): "Carry Trades and Currency Crashes"
- Lustig & Verdelhan (2007): "The Cross Section of Foreign Currency Risk Premia"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ForexFeatureConfig:
    """Configuration for forex feature calculation."""

    # FRED series IDs for central bank rates
    RATE_SERIES: Dict[str, str] = {
        "USD": "FEDFUNDS",      # Federal Funds Rate
        "EUR": "ECBDFR",        # ECB Deposit Facility Rate
        "GBP": "IUDSOIA",       # BOE Official Bank Rate
        "JPY": "IRSTCI01JPM156N", # BOJ Policy Rate
        "CHF": "IRSTCI01CHM156N", # SNB Policy Rate
        "AUD": "RBATCTR",       # RBA Cash Rate
        "CAD": "IRSTCB01CAM156N", # BOC Policy Rate
        "NZD": "RBATCTR",       # RBNZ Official Cash Rate (approx)
    }

    # Economic calendar data source
    CALENDAR_SOURCE: str = "oanda_labs"  # or "forexfactory", "investing_com"

    # COT data URL
    COT_URL: str = "https://www.cftc.gov/dea/newcot/"


@dataclass
class ForexFeatures:
    """Container for forex-specific features."""

    # Interest rate differential (carry)
    base_rate: float = 0.0
    quote_rate: float = 0.0
    rate_differential: float = 0.0  # Annual %
    rate_differential_norm: float = 0.0  # Normalized [-1, 1]
    carry_valid: bool = False

    # DXY relative strength
    dxy_value: float = 100.0
    dxy_return_1d: float = 0.0
    dxy_return_5d: float = 0.0
    rs_vs_dxy_20d: float = 0.0
    dxy_valid: bool = False

    # Session indicators (one-hot)
    is_sydney: bool = False
    is_tokyo: bool = False
    is_london: bool = False
    is_new_york: bool = False
    is_overlap: bool = False
    session_liquidity: float = 1.0

    # Spread dynamics
    spread_pips: float = 1.0
    spread_zscore: float = 0.0
    spread_regime: str = "normal"  # "tight", "normal", "wide"
    spread_valid: bool = False

    # COT positioning (weekly)
    cot_net_long_pct: float = 0.5  # Normalized [0, 1]
    cot_change_1w: float = 0.0
    cot_valid: bool = False

    # Economic calendar
    hours_to_next_event: float = 999.0
    next_event_impact: float = 0.0  # 0-3 scale
    is_news_window: bool = False

    # Volatility
    realized_vol_5d: float = 0.0
    realized_vol_20d: float = 0.0
    vol_ratio: float = 1.0  # 5d / 20d
    vol_valid: bool = False


def calculate_carry_features(
    base_currency: str,
    quote_currency: str,
    rates_df: pd.DataFrame,
    timestamp: int,
) -> Tuple[float, float, float, bool]:
    """
    Calculate interest rate differential features.

    Args:
        base_currency: Base currency code (e.g., "EUR")
        quote_currency: Quote currency code (e.g., "USD")
        rates_df: DataFrame with rate columns
        timestamp: Current timestamp (ms)

    Returns:
        (base_rate, quote_rate, differential, valid)
    """
    base_col = f"{base_currency}_RATE"
    quote_col = f"{quote_currency}_RATE"

    if base_col not in rates_df.columns or quote_col not in rates_df.columns:
        return (0.0, 0.0, 0.0, False)

    # Get most recent rate before timestamp
    ts_dt = pd.Timestamp(timestamp, unit='ms', tz='UTC')
    mask = rates_df.index <= ts_dt

    if not mask.any():
        return (0.0, 0.0, 0.0, False)

    latest = rates_df.loc[mask].iloc[-1]
    base_rate = float(latest.get(base_col, 0.0))
    quote_rate = float(latest.get(quote_col, 0.0))

    return (base_rate, quote_rate, base_rate - quote_rate, True)
```

#### 4.2 Update features_pipeline.py Integration

```python
# Add to features_pipeline.py

def _add_forex_features(
    df: pd.DataFrame,
    symbol: str,
    forex_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Add forex-specific features to DataFrame.

    Args:
        df: Price DataFrame with OHLCV
        symbol: Currency pair (e.g., "EUR_USD")
        forex_data: Optional dict with:
            - "rates": Interest rates DataFrame
            - "dxy": DXY prices DataFrame
            - "calendar": Economic calendar DataFrame
            - "cot": COT positioning DataFrame

    Returns:
        DataFrame with added forex features
    """
    from forex_features import calculate_carry_features

    # Parse currencies
    base, quote = symbol.split("_")

    # Add carry features
    if forex_data and "rates" in forex_data:
        rates_df = forex_data["rates"]
        df["carry_diff"] = df["timestamp"].apply(
            lambda ts: calculate_carry_features(base, quote, rates_df, ts)[2]
        )
    else:
        df["carry_diff"] = 0.0

    # Add session indicators
    df["session"] = df["timestamp"].apply(_detect_forex_session)
    df["is_overlap"] = df["session"].isin(["london_ny_overlap", "tokyo_london_overlap"])

    # ... more features

    return df
```

#### 4.3 Economic Calendar Integration

```python
class EconomicCalendar:
    """
    Economic events calendar for forex trading.

    Data Sources:
    - Primary: OANDA Labs Calendar API
    - Backup: ForexFactory scraping
    - Alternative: Investing.com API (unofficial)

    High-Impact Events (by currency):
    - USD: NFP, FOMC, CPI, GDP, ISM PMI
    - EUR: ECB, German CPI, Eurozone GDP
    - GBP: BOE, UK CPI, UK GDP
    - JPY: BOJ, Tankan, Japan CPI
    """

    def __init__(self, source: str = "oanda_labs"):
        self.source = source
        self._cache: Dict[str, List[Dict]] = {}
        self._last_refresh: Optional[datetime] = None

    def get_upcoming_events(
        self,
        currencies: List[str],
        hours_ahead: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get upcoming high-impact events."""
        # Implementation depends on source
        pass

    def hours_to_next_high_impact(
        self,
        currency: str,
        current_ts: int,
    ) -> Tuple[float, str, int]:
        """
        Get hours until next high-impact event.

        Returns:
            (hours, event_name, impact_level)
        """
        pass
```

#### 4.4 Swap Rates Data Source

```python
class SwapRatesProvider:
    """
    Provides swap/financing rates for forex positions.

    Data Sources:
    - OANDA: /v3/accounts/{id}/instruments (financing field)
    - Historical: Cache locally for backtesting

    Swap Calculation:
    - Long swap: (Base rate - Quote rate - Markup) / 365
    - Short swap: (Quote rate - Base rate - Markup) / 365
    - Wednesday: 3x swap (weekend rollover)
    """

    OANDA_INSTRUMENTS_ENDPOINT = "/v3/accounts/{account_id}/instruments"

    def get_current_swaps(
        self,
        symbol: str,
    ) -> Tuple[float, float]:
        """
        Get current swap rates from OANDA.

        Returns:
            (long_swap_pips, short_swap_pips)
        """
        pass

    def get_historical_swaps(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load historical swap rates from cache."""
        pass
```

**Deliverables**:

- `forex_features.py` with full feature set
- `features_pipeline.py` integration
- Economic calendar integration with multiple sources
- Swap rates provider
- COT data loader (weekly CFTC reports)

**Tests**: ~80 (including data source mocking)

---

### Phase 5: OTC Dealer Quote Simulation (Weeks 9-10)

**ПЕРЕИМЕНОВАНО**: Ранее называлось "L3 Forex Dealer Simulation"

**ВАЖНО**: Это НЕ L3 LOB simulation! Forex -- OTC рынок без центрального order book. Модуль помещается в `services/`, НЕ в `lob/`.

**Цель**: High-fidelity OTC dealer quote simulation для 95% реализма

#### 5.1 services/forex_dealer.py (~550 LOC)

```python
"""
OTC Forex Dealer Quote Simulation

Unlike exchange LOBs (used for crypto/equity L3), forex is OTC with dealer quotes.
This module simulates the dealer market structure:

1. Multi-dealer quote aggregation (like ECN)
2. Last-look rejection simulation
3. Quote flickering (rapid updates)
4. Size-dependent spread widening
5. Latency arbitrage protection
6. Session-dependent liquidity

Key Differences from LOB Simulation:
- NO queue position (no FIFO matching)
- NO price-time priority
- Dealer discretion (last-look)
- Indicative vs firm quotes
- Request-for-quote (RFQ) model for large sizes

References:
- Oomen (2017): "Last Look" in FX - Journal of Financial Markets
- Hasbrouck & Saar (2013): "Low-latency trading"
- King, Osler, Rime (2012): "Foreign Exchange Market Structure"
- BIS (2022): Triennial Survey - FX Market Structure
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QuoteType(Enum):
    """Type of dealer quote."""
    FIRM = "firm"          # Executable quote
    INDICATIVE = "indicative"  # May be rejected
    LAST_LOOK = "last_look"    # Subject to last-look window


class RejectReason(Enum):
    """Reason for trade rejection."""
    NONE = "none"
    LAST_LOOK_ADVERSE = "last_look_adverse"
    SIZE_EXCEEDED = "size_exceeded"
    QUOTE_EXPIRED = "quote_expired"
    PRICE_MOVED = "price_moved"
    LATENCY_ARBITRAGE = "latency_arbitrage"
    DEALER_DISCRETION = "dealer_discretion"


@dataclass
class DealerProfile:
    """Individual dealer characteristics."""
    dealer_id: str
    spread_factor: float = 1.0      # Multiplier on base spread
    max_size_usd: float = 5_000_000  # Max single trade
    last_look_window_ms: int = 200   # Last-look window
    base_reject_prob: float = 0.05   # Base rejection probability
    adverse_threshold_pips: float = 0.3  # Adverse move threshold
    latency_ms: float = 50.0         # Quote latency
    is_primary: bool = False         # Primary liquidity provider


@dataclass
class DealerQuote:
    """Single dealer quote."""
    dealer_id: str
    bid: float
    ask: float
    bid_size_usd: float
    ask_size_usd: float
    timestamp_ns: int
    quote_type: QuoteType
    valid_for_ms: int
    last_look_ms: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pips(self) -> float:
        # Assuming 4-decimal pair (adjust for JPY)
        return (self.ask - self.bid) * 10000


@dataclass
class AggregatedQuote:
    """Best bid/ask across all dealers (ECN-style)."""
    best_bid: float
    best_ask: float
    total_bid_size: float
    total_ask_size: float
    dealer_quotes: List[DealerQuote]
    timestamp_ns: int

    @property
    def spread_pips(self) -> float:
        return (self.best_ask - self.best_bid) * 10000


@dataclass
class ExecutionResult:
    """Result of execution attempt."""
    filled: bool
    fill_price: Optional[float] = None
    fill_qty: Optional[float] = None
    dealer_id: Optional[str] = None
    latency_ns: int = 0
    reject_reason: RejectReason = RejectReason.NONE
    slippage_pips: float = 0.0
    last_look_passed: bool = True


@dataclass
class ForexDealerConfig:
    """Configuration for dealer simulation."""
    num_dealers: int = 5
    base_spread_pips: float = 1.0
    last_look_enabled: bool = True
    size_impact_threshold_usd: float = 1_000_000
    size_impact_factor: float = 0.5  # Spread widening per $1M
    session_spread_adjustment: bool = True
    latency_variance: float = 0.3  # Coefficient of variation


class ForexDealerSimulator:
    """
    Simulates multi-dealer OTC forex market.

    This is fundamentally different from LOB simulation:
    - Dealers provide quotes (no order book depth)
    - Last-look gives dealers rejection rights
    - No queue position (not FIFO)
    - Price improvement possible
    """

    def __init__(
        self,
        config: Optional[ForexDealerConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or ForexDealerConfig()
        self._rng = np.random.default_rng(seed)
        self._dealers = self._create_dealers()

    def _create_dealers(self) -> List[DealerProfile]:
        """Create heterogeneous dealer pool."""
        dealers = []
        for i in range(self.config.num_dealers):
            # First dealer is primary LP with tighter spreads
            is_primary = (i == 0)
            dealers.append(DealerProfile(
                dealer_id=f"dealer_{i}",
                spread_factor=0.8 if is_primary else 1.0 + self._rng.uniform(-0.15, 0.25),
                max_size_usd=10_000_000 if is_primary else self._rng.uniform(2e6, 8e6),
                last_look_window_ms=int(150 if is_primary else self._rng.uniform(100, 300)),
                base_reject_prob=0.02 if is_primary else self._rng.uniform(0.03, 0.12),
                adverse_threshold_pips=0.2 if is_primary else self._rng.uniform(0.2, 0.5),
                latency_ms=30 if is_primary else self._rng.uniform(40, 120),
                is_primary=is_primary,
            ))
        return dealers

    def get_aggregated_quote(
        self,
        symbol: str,
        mid_price: float,
        session_factor: float = 1.0,
        order_size_usd: float = 100_000,
    ) -> AggregatedQuote:
        """
        Get aggregated quote from all dealers.

        Args:
            symbol: Currency pair
            mid_price: Current mid price
            session_factor: Session liquidity factor (0.5 to 1.5)
            order_size_usd: Indicative order size for spread adjustment

        Returns:
            AggregatedQuote with best bid/ask
        """
        timestamp_ns = time.time_ns()
        quotes = []

        pip_size = 0.01 if "JPY" in symbol else 0.0001
        base_spread = self.config.base_spread_pips * pip_size

        for dealer in self._dealers:
            # Session-adjusted spread
            spread = base_spread * dealer.spread_factor / max(0.5, session_factor)

            # Size impact
            if order_size_usd > self.config.size_impact_threshold_usd:
                excess = order_size_usd - self.config.size_impact_threshold_usd
                size_widening = (excess / 1_000_000) * self.config.size_impact_factor * pip_size
                spread += size_widening

            # Add noise
            spread *= (1 + self._rng.uniform(-0.1, 0.1))

            half_spread = spread / 2
            bid = mid_price - half_spread
            ask = mid_price + half_spread

            quotes.append(DealerQuote(
                dealer_id=dealer.dealer_id,
                bid=bid,
                ask=ask,
                bid_size_usd=dealer.max_size_usd * self._rng.uniform(0.5, 1.0),
                ask_size_usd=dealer.max_size_usd * self._rng.uniform(0.5, 1.0),
                timestamp_ns=timestamp_ns,
                quote_type=QuoteType.LAST_LOOK if self.config.last_look_enabled else QuoteType.FIRM,
                valid_for_ms=200,
                last_look_ms=dealer.last_look_window_ms,
            ))

        # Aggregate
        best_bid = max(q.bid for q in quotes)
        best_ask = min(q.ask for q in quotes)
        total_bid = sum(q.bid_size_usd for q in quotes if q.bid == best_bid)
        total_ask = sum(q.ask_size_usd for q in quotes if q.ask == best_ask)

        return AggregatedQuote(
            best_bid=best_bid,
            best_ask=best_ask,
            total_bid_size=total_bid,
            total_ask_size=total_ask,
            dealer_quotes=quotes,
            timestamp_ns=timestamp_ns,
        )

    def attempt_execution(
        self,
        is_buy: bool,
        size_usd: float,
        quote: AggregatedQuote,
        current_mid: float,
    ) -> ExecutionResult:
        """
        Attempt to execute against dealer quotes.

        Simulates:
        1. Dealer selection (best price)
        2. Size check
        3. Latency
        4. Last-look decision

        Args:
            is_buy: True for buy, False for sell
            size_usd: Order size in USD
            quote: Current aggregated quote
            current_mid: Current mid price (for last-look check)

        Returns:
            ExecutionResult
        """
        # Sort dealers by price (best first)
        if is_buy:
            sorted_quotes = sorted(quote.dealer_quotes, key=lambda q: q.ask)
            reference_price = quote.best_ask
        else:
            sorted_quotes = sorted(quote.dealer_quotes, key=lambda q: -q.bid)
            reference_price = quote.best_bid

        for dealer_quote in sorted_quotes:
            dealer = self._get_dealer(dealer_quote.dealer_id)

            # Size check
            available = dealer_quote.ask_size_usd if is_buy else dealer_quote.bid_size_usd
            if size_usd > available:
                continue

            # Simulate latency
            latency_ms = dealer.latency_ms * (1 + self._rng.normal(0, self.config.latency_variance))
            latency_ns = int(max(10, latency_ms) * 1_000_000)

            # Last-look check
            if self.config.last_look_enabled:
                quote_mid = dealer_quote.mid
                price_move_pips = abs(current_mid - quote_mid) * 10000

                # Adverse selection check
                is_adverse = (
                    (is_buy and current_mid > quote_mid) or
                    (not is_buy and current_mid < quote_mid)
                )

                if is_adverse and price_move_pips > dealer.adverse_threshold_pips:
                    if self._rng.random() < 0.7:  # 70% reject on adverse
                        continue  # Rejected, try next dealer

                # Random rejection
                if self._rng.random() < dealer.base_reject_prob:
                    continue

            # Success!
            fill_price = dealer_quote.ask if is_buy else dealer_quote.bid
            slippage_pips = abs(fill_price - reference_price) * 10000

            return ExecutionResult(
                filled=True,
                fill_price=fill_price,
                fill_qty=size_usd,
                dealer_id=dealer_quote.dealer_id,
                latency_ns=latency_ns,
                slippage_pips=slippage_pips,
                last_look_passed=True,
            )

        # All dealers rejected
        return ExecutionResult(
            filled=False,
            reject_reason=RejectReason.DEALER_DISCRETION,
        )

    def _get_dealer(self, dealer_id: str) -> DealerProfile:
        """Get dealer by ID."""
        for d in self._dealers:
            if d.dealer_id == dealer_id:
                return d
        raise ValueError(f"Unknown dealer: {dealer_id}")
```

**Deliverables**:

- `services/forex_dealer.py` (NOT in `lob/`)
- `ForexDealerSimulator` with multi-dealer quotes
- Last-look simulation
- Size-dependent spread widening
- Integration with `ForexParametricSlippageProvider`

**Tests**: ~85

---

### Phase 6: Forex Risk Management & Services (Weeks 11-12)

**Цель**: Forex-специфичный risk management + position sync + session routing

#### 6.1 services/forex_risk_guards.py (~400 LOC)

*Содержимое как в оригинальном плане, с добавлением:*

```python
# Add swap cost tracking
class SwapCostTracker:
    """
    Track cumulative swap costs for positions.

    Data Source: OANDA API or historical cache
    """

    def calculate_daily_swap(
        self,
        symbol: str,
        position_units: float,
        is_long: bool,
        current_price: float,
        day_of_week: int,  # 0=Mon, 6=Sun
    ) -> float:
        """
        Calculate daily swap cost/credit.

        Note: Wednesday = 3x swap (weekend rollover)
        """
        multiplier = 3 if day_of_week == 2 else 1  # Wednesday
        # ... implementation
```

#### 6.2 services/forex_position_sync.py (~300 LOC) -- NEW

```python
"""
Forex Position Synchronization Service

Syncs local position state with OANDA account.
Similar to services/position_sync.py for Alpaca.

Features:
- Background polling (configurable interval)
- Discrepancy detection and alerting
- Automatic reconciliation (optional)
- Swap cost tracking
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ForexPosition:
    """Forex position from OANDA."""
    symbol: str
    units: float  # Positive = long, Negative = short
    average_price: float
    unrealized_pnl: float
    margin_used: float
    financing: float  # Accumulated swap


@dataclass
class SyncConfig:
    """Position sync configuration."""
    sync_interval_sec: float = 30.0
    position_tolerance_pct: float = 0.01  # 1%
    auto_reconcile: bool = False
    max_reconcile_units: float = 100_000
    alert_on_discrepancy: bool = True


class ForexPositionSynchronizer:
    """
    Synchronizes local state with OANDA positions.
    """

    def __init__(
        self,
        oanda_adapter,  # OandaOrderExecutionAdapter
        local_state_getter: Callable[[], Dict[str, float]],
        config: Optional[SyncConfig] = None,
        on_discrepancy: Optional[Callable] = None,
    ):
        self.oanda = oanda_adapter
        self.get_local_state = local_state_getter
        self.config = config or SyncConfig()
        self.on_discrepancy = on_discrepancy
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def sync_once(self) -> List[str]:
        """
        Perform single sync check.

        Returns:
            List of symbols with discrepancies
        """
        discrepancies = []

        # Get OANDA positions
        oanda_positions = await self.oanda.get_positions_async()
        oanda_map = {p.symbol: p.units for p in oanda_positions}

        # Get local positions
        local_map = self.get_local_state()

        # Compare
        all_symbols = set(oanda_map.keys()) | set(local_map.keys())

        for symbol in all_symbols:
            oanda_units = oanda_map.get(symbol, 0.0)
            local_units = local_map.get(symbol, 0.0)

            if abs(oanda_units) < 1 and abs(local_units) < 1:
                continue  # Both effectively zero

            diff_pct = abs(oanda_units - local_units) / max(abs(oanda_units), abs(local_units), 1)

            if diff_pct > self.config.position_tolerance_pct:
                discrepancies.append(symbol)
                logger.warning(
                    f"Position discrepancy: {symbol} "
                    f"OANDA={oanda_units:.2f} Local={local_units:.2f}"
                )

                if self.on_discrepancy:
                    self.on_discrepancy(symbol, oanda_units, local_units)

        return discrepancies

    def start_background_sync(self) -> None:
        """Start background sync task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._sync_loop())

    def stop_background_sync(self) -> None:
        """Stop background sync."""
        self._running = False
        if self._task:
            self._task.cancel()
```

#### 6.3 services/forex_session_router.py (~250 LOC) -- NEW

```python
"""
Forex Session-Aware Order Routing

Routes orders based on current session and liquidity.
Similar to services/session_router.py for Alpaca extended hours.

Features:
- Session detection with DST awareness
- Spread adjustment recommendations
- Optimal execution window suggestions
- Rollover time avoidance
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from adapters.models import ForexSessionType

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Order routing decision."""
    should_submit: bool
    session: ForexSessionType
    liquidity_factor: float
    spread_multiplier: float
    recommended_delay_sec: Optional[float] = None
    reason: str = ""


class ForexSessionRouter:
    """
    Session-aware order routing for forex.
    """

    ET = ZoneInfo("America/New_York")
    ROLLOVER_HOUR_ET = 17  # 5pm ET
    ROLLOVER_KEEPOUT_MINUTES = 30

    def __init__(
        self,
        avoid_rollover: bool = True,
        min_liquidity_factor: float = 0.5,
    ):
        self.avoid_rollover = avoid_rollover
        self.min_liquidity_factor = min_liquidity_factor

    def get_routing_decision(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        timestamp_ms: Optional[int] = None,
    ) -> RoutingDecision:
        """
        Get routing decision for order.

        Args:
            symbol: Currency pair
            side: "BUY" or "SELL"
            size_usd: Order size
            timestamp_ms: Optional timestamp (default: now)

        Returns:
            RoutingDecision
        """
        now_et = datetime.now(self.ET)

        # Check rollover window
        if self.avoid_rollover:
            minutes_to_rollover = self._minutes_to_rollover(now_et)
            if abs(minutes_to_rollover) < self.ROLLOVER_KEEPOUT_MINUTES:
                return RoutingDecision(
                    should_submit=False,
                    session=ForexSessionType.OFF_HOURS,
                    liquidity_factor=0.3,
                    spread_multiplier=3.0,
                    recommended_delay_sec=float(self.ROLLOVER_KEEPOUT_MINUTES * 60),
                    reason="Near rollover time (5pm ET)",
                )

        # Get current session
        session, liq_factor, spread_mult = self._detect_session(now_et)

        # Check minimum liquidity
        if liq_factor < self.min_liquidity_factor:
            return RoutingDecision(
                should_submit=False,
                session=session,
                liquidity_factor=liq_factor,
                spread_multiplier=spread_mult,
                reason=f"Low liquidity session: {session.value}",
            )

        return RoutingDecision(
            should_submit=True,
            session=session,
            liquidity_factor=liq_factor,
            spread_multiplier=spread_mult,
        )

    def _minutes_to_rollover(self, dt_et: datetime) -> float:
        """Minutes until next rollover (negative if just passed)."""
        rollover_today = dt_et.replace(hour=self.ROLLOVER_HOUR_ET, minute=0, second=0)
        diff = (rollover_today - dt_et).total_seconds() / 60
        return diff

    def _detect_session(self, dt_et: datetime) -> Tuple[ForexSessionType, float, float]:
        """Detect session from ET datetime."""
        # Convert to UTC for session logic
        dt_utc = dt_et.astimezone(ZoneInfo("UTC"))
        hour_utc = dt_utc.hour

        # Session detection logic (same as trading_hours.py)
        # ... implementation
        return (ForexSessionType.LONDON, 1.1, 1.0)
```

**Deliverables**:

- `services/forex_risk_guards.py` -- margin, leverage, swap tracking
- `services/forex_position_sync.py` -- position synchronization
- `services/forex_session_router.py` -- session-aware routing
- Integration with existing risk system

**Tests**: ~100

---

### Phase 7: Data Pipeline & Downloaders (Weeks 13-14)

**Цель**: Полный data pipeline для Forex

#### 7.1 scripts/download_forex_data.py

*Как в оригинальном плане*

#### 7.2 scripts/download_swap_rates.py -- NEW

```python
"""
Download historical swap rates for backtesting.

Sources:
- OANDA API: /v3/accounts/{id}/instruments (financing field)
- Cache locally for historical backtest

Usage:
    python scripts/download_swap_rates.py \
        --pairs EUR_USD GBP_USD USD_JPY \
        --start 2020-01-01 \
        --output data/forex/swaps/
"""
```

#### 7.3 scripts/download_economic_calendar.py -- NEW

```python
"""
Download economic calendar events.

Sources:
- Primary: OANDA Labs Calendar API
- Backup: ForexFactory (scraping)

Usage:
    python scripts/download_economic_calendar.py \
        --currencies USD EUR GBP JPY \
        --start 2020-01-01 \
        --output data/forex/calendar/
"""
```

**Deliverables**:

- `scripts/download_forex_data.py`
- `scripts/download_swap_rates.py`
- `scripts/download_economic_calendar.py`
- `scripts/download_interest_rates.py` (FRED)
- Data loader integration

**Tests**: ~50

---

### Phase 8: Configuration System (Week 14)

*Как в оригинальном плане с добавлениями:*

#### 8.1 configs/forex_defaults.yaml -- Updated

```yaml
# =============================================================================
# FOREX DEFAULTS CONFIGURATION v2.0
# =============================================================================

forex:
  asset_class: forex
  data_vendor: oanda
  market: spot

  # Trading hours (DST-aware)
  session:
    calendar: forex_24x5
    weekend_filter: true
    rollover_time_et: 17  # 5pm ET (constant regardless of DST)
    rollover_keepout_minutes: 30
    dst_aware: true

  # Fees: Spread-based
  fees:
    structure: spread_only
    maker_bps: 0.0
    taker_bps: 0.0
    swap_enabled: true
    swap_data_source: oanda  # or "cache"

  # Slippage: L2+ Parametric model
  slippage:
    level: "L2+"
    provider: ForexParametricSlippageProvider  # In execution_providers.py
    profile: retail  # or "institutional", "conservative"
    impact_coef_base: 0.03
    session_adjustment: true
    news_adjustment: true

  # OTC Dealer Simulation (NOT L3 LOB!)
  dealer_simulation:
    enabled: true
    provider: ForexDealerSimulator  # In services/forex_dealer.py
    num_dealers: 5
    last_look_enabled: true

  # Leverage
  leverage:
    max_leverage: 50.0
    default_leverage: 30.0
    margin_warning: 1.20
    margin_call: 1.00
    stop_out: 0.50

  # Position Sync
  position_sync:
    enabled: true
    interval_sec: 30.0
    auto_reconcile: false

  # Data sources
  data_sources:
    price_data: oanda
    interest_rates: fred  # FRED API
    economic_calendar: oanda_labs
    swap_rates: oanda
    dxy: yahoo  # DX-Y.NYB

# API Rate Limits
rate_limits:
  oanda:
    requests_per_second: 120
    burst: 200
```

**Tests**: ~35

---

### Phase 9: Training & Backtest Integration (Week 15)

*Как в оригинальном плане*

**Tests**: ~45

---

### Phase 10: Testing & Validation (Weeks 16-17)

**Цель**: Comprehensive testing + property-based + stress tests + **REGRESSION PREVENTION**

> **КРИТИЧЕСКИ ВАЖНО**: Forex интеграция НЕ ДОЛЖНА нарушить существующий функционал crypto и equity. Каждая фаза включает обязательные регрессионные проверки.

#### 10.1 Unit Tests (Forex-specific)

```
tests/
├── test_oanda_adapters.py              # 130 tests
├── test_forex_execution_providers.py   # 100 tests
├── test_forex_features.py              # 70 tests
├── test_forex_dealer_simulation.py     # 85 tests
├── test_forex_risk_guards.py           # 60 tests
├── test_forex_position_sync.py         # 40 tests
├── test_forex_session_router.py        # 30 tests
├── test_forex_integration.py           # 50 tests
├── test_forex_properties.py            # 40 tests (Property-based)
├── test_forex_stress.py                # 20 tests (Stress scenarios)
├── test_forex_regression.py            # 45 tests (NEW: Regression suite)
├── test_forex_isolation.py             # 35 tests (NEW: Isolation verification)
└── test_forex_backward_compat.py       # 30 tests (NEW: Backward compatibility)
```

#### 10.2 Regression Testing Suite (КРИТИЧЕСКИ ВАЖНО!)

```python
# tests/test_forex_regression.py
"""
Regression Test Suite for Forex Integration.

PURPOSE: Verify that adding Forex does NOT break existing crypto/equity functionality.
RUN: After EVERY phase completion and before merge to main.

Test Categories:
1. Crypto execution providers unchanged
2. Equity execution providers unchanged
3. Adapter registry backward compatible
4. Feature pipeline produces identical outputs
5. Risk guards behavior unchanged
6. Training pipeline produces consistent results
"""
import pytest
import numpy as np
from typing import Dict, Any

# =============================================================================
# BASELINE SNAPSHOTS (captured before Forex integration)
# =============================================================================

CRYPTO_SLIPPAGE_BASELINE = {
    # Snapshot of CryptoParametricSlippageProvider outputs
    # Captured with fixed seed for reproducibility
    "BTCUSDT_participation_0.001": 2.45,  # pips
    "ETHUSDT_participation_0.005": 5.12,
    "BTCUSDT_high_vol": 8.73,
}

EQUITY_FEE_BASELINE = {
    # Snapshot of equity fee calculations (if EquityParametric exists)
    "AAPL_100_shares_sell": 0.0278,  # SEC fee
    "SPY_1000_shares_sell": 0.278,
}

FEATURE_PIPELINE_BASELINE = {
    # Hash of feature vector for known input
    "crypto_btc_features_hash": "a1b2c3d4...",
    "equity_aapl_features_hash": "e5f6g7h8...",
}


class TestCryptoRegressionSuite:
    """Ensure crypto functionality unchanged after Forex integration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup crypto test environment."""
        from execution_providers import (
            CryptoParametricSlippageProvider,
            CryptoParametricConfig,
            AssetClass,
        )
        self.provider = CryptoParametricSlippageProvider()

    def test_crypto_slippage_unchanged(self):
        """Crypto slippage calculations must match baseline."""
        # Test with fixed parameters
        slippage = self.provider.compute_slippage_bps(
            order=self._create_btc_order(),
            market=self._create_btc_market(),
            participation_ratio=0.001,
        )
        assert abs(slippage - CRYPTO_SLIPPAGE_BASELINE["BTCUSDT_participation_0.001"]) < 0.01

    def test_crypto_profiles_exist(self):
        """All crypto profiles must still exist."""
        profiles = ["default", "conservative", "aggressive", "altcoin", "stablecoin"]
        for profile in profiles:
            provider = CryptoParametricSlippageProvider.from_profile(profile)
            assert provider is not None

    def test_crypto_asset_class_unchanged(self):
        """AssetClass.CRYPTO must still work identically."""
        from execution_providers import AssetClass, create_execution_provider
        provider = create_execution_provider(AssetClass.CRYPTO)
        assert provider is not None

    def test_crypto_config_defaults_unchanged(self):
        """CryptoParametricConfig defaults must not change."""
        from execution_providers import CryptoParametricConfig
        config = CryptoParametricConfig()
        assert config.impact_coef_base == 0.10
        assert config.spread_bps == 5.0
        assert config.whale_threshold == 0.01

    def test_binance_adapter_unchanged(self):
        """Binance adapter must work identically."""
        from adapters.registry import create_market_data_adapter
        adapter = create_market_data_adapter("binance")
        assert adapter is not None


class TestEquityRegressionSuite:
    """Ensure equity functionality unchanged after Forex integration."""

    def test_equity_asset_class_unchanged(self):
        """AssetClass.EQUITY must still work identically."""
        from execution_providers import AssetClass, create_execution_provider
        provider = create_execution_provider(AssetClass.EQUITY)
        assert provider is not None

    def test_alpaca_adapter_unchanged(self):
        """Alpaca adapter must work identically."""
        from adapters.registry import create_market_data_adapter
        adapter = create_market_data_adapter("alpaca")
        assert adapter is not None

    def test_equity_fee_provider_unchanged(self):
        """Equity fee calculations must match baseline."""
        from execution_providers import EquityFeeProvider
        provider = EquityFeeProvider()
        # SEC fee calculation unchanged
        fee = provider.compute_fee(
            price=150.0,
            qty=100,
            side="SELL",
            liquidity_role="taker",
        )
        assert abs(fee - EQUITY_FEE_BASELINE["AAPL_100_shares_sell"]) < 0.001


class TestAdapterRegistryRegression:
    """Ensure adapter registry backward compatible."""

    def test_existing_vendors_unchanged(self):
        """All existing vendors must still be registered."""
        from adapters.models import ExchangeVendor
        existing = ["binance", "binance_us", "alpaca", "polygon", "yahoo"]
        for vendor in existing:
            assert hasattr(ExchangeVendor, vendor.upper())

    def test_lazy_loading_unchanged(self):
        """Lazy loading for existing adapters must work."""
        from adapters.registry import AdapterRegistry
        registry = AdapterRegistry()
        # These should not raise
        registry.get_market_data_adapter("binance")
        registry.get_market_data_adapter("alpaca")

    def test_forex_addition_isolated(self):
        """Adding OANDA must not affect other vendors."""
        from adapters.models import ExchangeVendor
        # OANDA added
        assert hasattr(ExchangeVendor, "OANDA")
        # Others unchanged
        assert ExchangeVendor.BINANCE.value == "binance"
        assert ExchangeVendor.ALPACA.value == "alpaca"


class TestFeaturePipelineRegression:
    """Ensure feature pipeline produces identical outputs."""

    def test_crypto_features_unchanged(self):
        """Crypto feature vector must be identical."""
        from features_pipeline import compute_features
        # Test with known input data
        features = compute_features(self._load_btc_test_data(), asset_class="crypto")
        # Compare hash or specific values
        assert len(features.columns) >= 63  # Original feature count

    def test_equity_features_unchanged(self):
        """Equity feature vector must be identical."""
        from features_pipeline import compute_features
        features = compute_features(self._load_aapl_test_data(), asset_class="equity")
        assert "vix_regime" in features.columns or "rs_vs_spy_20d" in features.columns

    def test_forex_features_isolated(self):
        """Forex features must not affect crypto/equity."""
        from features_pipeline import compute_features
        # Forex-specific features only appear for forex asset class
        crypto_features = compute_features(self._load_btc_test_data(), asset_class="crypto")
        assert "carry_diff" not in crypto_features.columns
        assert "session_liquidity" not in crypto_features.columns


class TestRiskGuardsRegression:
    """Ensure risk guards behavior unchanged."""

    def test_crypto_risk_limits_unchanged(self):
        """Crypto risk limits must be identical."""
        from risk_guard import RiskGuard
        guard = RiskGuard.from_config("configs/risk.yaml")
        # Verify limits unchanged

    def test_equity_risk_guards_unchanged(self):
        """Equity risk guards (margin, short) must work identically."""
        from services.stock_risk_guards import MarginGuard, ShortSaleGuard
        margin = MarginGuard()
        short = ShortSaleGuard()
        # Verify behavior unchanged


# =============================================================================
# PHASE-SPECIFIC REGRESSION GATES
# =============================================================================

class TestPhaseRegressionGates:
    """
    Regression gates to run after each phase.

    Usage:
        pytest tests/test_forex_regression.py::TestPhaseRegressionGates -v
    """

    @pytest.mark.phase1
    def test_phase1_gate(self):
        """Phase 1 (Enums): No regression in existing enums."""
        from adapters.models import MarketType, ExchangeVendor
        from execution_providers import AssetClass

        # MarketType unchanged
        assert MarketType.CRYPTO_SPOT.value == "CRYPTO_SPOT"
        assert MarketType.EQUITY.value == "EQUITY"
        # FOREX added but others unchanged
        assert MarketType.FOREX.value == "FOREX"

    @pytest.mark.phase2
    def test_phase2_gate(self):
        """Phase 2 (OANDA): Existing adapters unaffected."""
        from adapters.registry import AdapterRegistry
        registry = AdapterRegistry()
        # All existing adapters must work
        assert registry.get_market_data_adapter("binance") is not None
        assert registry.get_market_data_adapter("alpaca") is not None

    @pytest.mark.phase3
    def test_phase3_gate(self):
        """Phase 3 (L2+): Existing slippage providers unchanged."""
        from execution_providers import (
            CryptoParametricSlippageProvider,
            create_slippage_provider,
            AssetClass,
        )
        # Crypto provider still works
        crypto = create_slippage_provider("L2+", AssetClass.CRYPTO)
        assert isinstance(crypto, CryptoParametricSlippageProvider)

    @pytest.mark.phase4
    def test_phase4_gate(self):
        """Phase 4 (Features): Pipeline backward compatible."""
        # Feature count for crypto/equity unchanged
        pass

    @pytest.mark.phase5
    def test_phase5_gate(self):
        """Phase 5 (OTC Sim): services/ folder structure intact."""
        import importlib
        # Existing services must import
        importlib.import_module("services.position_sync")
        importlib.import_module("services.session_router")
        importlib.import_module("services.stock_risk_guards")

    @pytest.mark.phase6
    def test_phase6_gate(self):
        """Phase 6 (Risk): Existing risk guards unchanged."""
        from services.stock_risk_guards import MarginGuard
        guard = MarginGuard()
        # Verify default behavior

    @pytest.mark.full
    def test_full_regression_suite(self):
        """Run complete regression suite."""
        # This is a meta-test that ensures all above pass
        pass
```

#### 10.3 Isolation Tests (NEW!)

```python
# tests/test_forex_isolation.py
"""
Isolation Tests for Forex Integration.

PURPOSE: Verify that Forex code is properly isolated and does not
         introduce unwanted dependencies or side effects.

Key Isolation Properties:
1. Forex imports do not pull in crypto/equity code unnecessarily
2. Forex configuration does not affect other asset classes
3. Forex errors do not propagate to other pipelines
4. Forex can be disabled without affecting crypto/equity
"""
import pytest
import sys
from unittest.mock import patch, MagicMock


class TestImportIsolation:
    """Verify import isolation between asset classes."""

    def test_forex_adapter_no_binance_import(self):
        """Importing OANDA adapter must not import Binance."""
        # Clear any cached imports
        modules_before = set(sys.modules.keys())

        # Import OANDA
        from adapters.oanda import OandaMarketDataAdapter

        modules_after = set(sys.modules.keys())
        new_modules = modules_after - modules_before

        # Binance should not be imported
        assert not any("binance" in m.lower() for m in new_modules)

    def test_forex_adapter_no_alpaca_import(self):
        """Importing OANDA adapter must not import Alpaca."""
        modules_before = set(sys.modules.keys())
        from adapters.oanda import OandaMarketDataAdapter
        modules_after = set(sys.modules.keys())
        new_modules = modules_after - modules_before
        assert not any("alpaca" in m.lower() for m in new_modules)

    def test_crypto_adapter_no_forex_import(self):
        """Importing Binance adapter must not import Forex."""
        modules_before = set(sys.modules.keys())
        from adapters.binance import BinanceMarketDataAdapter
        modules_after = set(sys.modules.keys())
        new_modules = modules_after - modules_before
        assert not any("oanda" in m.lower() for m in new_modules)
        assert not any("forex" in m.lower() for m in new_modules)


class TestConfigurationIsolation:
    """Verify configuration isolation between asset classes."""

    def test_forex_config_no_crypto_pollution(self):
        """Forex config must not affect crypto defaults."""
        from execution_providers import CryptoParametricConfig, ForexParametricConfig

        # Create forex config with custom values
        forex_cfg = ForexParametricConfig(impact_coef_base=0.03)

        # Crypto config must still have its defaults
        crypto_cfg = CryptoParametricConfig()
        assert crypto_cfg.impact_coef_base == 0.10  # Unchanged

    def test_asset_class_defaults_independent(self):
        """Each asset class must have independent defaults."""
        from execution_providers import create_execution_provider, AssetClass

        crypto = create_execution_provider(AssetClass.CRYPTO)
        equity = create_execution_provider(AssetClass.EQUITY)
        forex = create_execution_provider(AssetClass.FOREX)

        # All must be different instances/types
        assert type(crypto) != type(forex)
        assert type(equity) != type(forex)


class TestErrorIsolation:
    """Verify error isolation - Forex errors don't break crypto/equity."""

    def test_forex_adapter_failure_isolated(self):
        """Forex adapter failure must not affect crypto trading."""
        from adapters.registry import AdapterRegistry

        registry = AdapterRegistry()

        # Simulate OANDA failure
        with patch("adapters.oanda.OandaMarketDataAdapter") as mock:
            mock.side_effect = ConnectionError("OANDA unavailable")

            # Crypto must still work
            crypto_adapter = registry.get_market_data_adapter("binance")
            assert crypto_adapter is not None

    def test_forex_slippage_error_isolated(self):
        """Forex slippage error must not affect crypto slippage."""
        from execution_providers import (
            CryptoParametricSlippageProvider,
            ForexParametricSlippageProvider,
        )

        # Even if forex provider raises
        forex = ForexParametricSlippageProvider()
        crypto = CryptoParametricSlippageProvider()

        # Crypto must work independently
        result = crypto.compute_slippage_bps(
            order=self._create_order(),
            market=self._create_market(),
            participation_ratio=0.001,
        )
        assert result > 0


class TestDisableIsolation:
    """Verify Forex can be disabled without side effects."""

    def test_forex_disabled_crypto_works(self):
        """With Forex disabled, crypto pipeline must work normally."""
        # Simulate forex disabled via config
        with patch.dict("os.environ", {"FOREX_ENABLED": "false"}):
            from execution_providers import create_execution_provider, AssetClass

            # Crypto still works
            crypto = create_execution_provider(AssetClass.CRYPTO)
            assert crypto is not None

    def test_forex_disabled_no_import_errors(self):
        """With Forex disabled, no import errors must occur."""
        # Remove OANDA from available adapters
        with patch("adapters.registry.AdapterRegistry._lazy_modules",
                   {"binance": "adapters.binance", "alpaca": "adapters.alpaca"}):
            from adapters.registry import AdapterRegistry
            registry = AdapterRegistry()

            # Crypto and equity still work
            assert registry.get_market_data_adapter("binance") is not None
```

#### 10.4 Backward Compatibility Tests (NEW!)

```python
# tests/test_forex_backward_compat.py
"""
Backward Compatibility Tests.

PURPOSE: Ensure existing code that uses crypto/equity APIs continues to work
         without any changes after Forex integration.

Scenarios:
1. Existing scripts run unchanged
2. Existing configs load correctly
3. Existing trained models can be loaded
4. Existing API contracts preserved
"""
import pytest
import yaml


class TestAPIContractPreservation:
    """Verify public API contracts are preserved."""

    def test_execution_providers_api_unchanged(self):
        """Public API of execution_providers must be unchanged."""
        import execution_providers as ep

        # All existing exports must exist
        required_exports = [
            "AssetClass",
            "Order",
            "MarketState",
            "Fill",
            "BarData",
            "SlippageProvider",
            "FeeProvider",
            "FillProvider",
            "CryptoParametricSlippageProvider",
            "CryptoFeeProvider",
            "EquityFeeProvider",
            "StatisticalSlippageProvider",
            "OHLCVFillProvider",
            "L2ExecutionProvider",
            "create_execution_provider",
            "create_slippage_provider",
            "create_fee_provider",
        ]

        for name in required_exports:
            assert hasattr(ep, name), f"Missing export: {name}"

    def test_adapter_models_api_unchanged(self):
        """Public API of adapters.models must be unchanged."""
        from adapters import models

        required = [
            "MarketType",
            "ExchangeVendor",
            "FeeStructure",
            "SessionType",
            "ExchangeRule",
        ]

        for name in required:
            assert hasattr(models, name), f"Missing: {name}"


class TestExistingConfigsLoad:
    """Verify existing config files still load correctly."""

    @pytest.mark.parametrize("config_path", [
        "configs/config_train.yaml",
        "configs/config_sim.yaml",
        "configs/config_live.yaml",
        "configs/config_eval.yaml",
        "configs/config_train_stocks.yaml",
        "configs/config_backtest_stocks.yaml",
        "configs/config_live_alpaca.yaml",
    ])
    def test_existing_config_loads(self, config_path):
        """Existing config must load without errors."""
        import os
        if not os.path.exists(config_path):
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert config is not None

    def test_asset_class_detection_backward_compat(self):
        """Asset class detection must work for existing configs."""
        # Config without explicit asset_class (legacy)
        legacy_config = {"vendor": "binance"}

        from script_live import detect_asset_class
        result = detect_asset_class(legacy_config)
        assert result == "crypto"  # Default behavior preserved

        # Config with alpaca vendor
        alpaca_config = {"vendor": "alpaca"}
        result = detect_asset_class(alpaca_config)
        assert result == "equity"


class TestExistingScriptsUnchanged:
    """Verify existing scripts work without modification."""

    def test_script_backtest_syntax_valid(self):
        """script_backtest.py must be syntactically valid."""
        import py_compile
        py_compile.compile("script_backtest.py", doraise=True)

    def test_script_live_syntax_valid(self):
        """script_live.py must be syntactically valid."""
        import py_compile
        py_compile.compile("script_live.py", doraise=True)

    def test_train_model_syntax_valid(self):
        """train_model_multi_patch.py must be syntactically valid."""
        import py_compile
        py_compile.compile("train_model_multi_patch.py", doraise=True)


class TestModelCompatibility:
    """Verify trained model compatibility."""

    def test_crypto_model_loads_after_forex(self):
        """Crypto models trained before Forex must still load."""
        # This tests that adding ForexParametric doesn't break
        # torch.load() for existing models
        pass

    def test_observation_space_unchanged(self):
        """Observation space dimensions must be unchanged for crypto/equity."""
        # Forex features must not increase crypto observation space
        pass
```

#### 10.5 Property-Based Tests (Hypothesis)

```python
# tests/test_forex_properties.py
"""
Property-based tests using Hypothesis.

Properties tested:
1. Slippage monotonicity: More participation → more slippage
2. Session ordering: Overlap liquidity > individual session
3. Spread bounds: Always within configured min/max
4. Pair classification: All pairs classified correctly
5. DST handling: Rollover time consistent regardless of date
"""
from hypothesis import given, strategies as st, assume
import pytest

from execution_providers import ForexParametricSlippageProvider, ForexParametricConfig


@given(
    participation=st.floats(0.0, 0.1, allow_nan=False),
    participation2=st.floats(0.0, 0.1, allow_nan=False),
)
def test_slippage_monotonic_in_participation(participation, participation2):
    """Higher participation should result in equal or higher slippage."""
    assume(participation <= participation2)

    provider = ForexParametricSlippageProvider()
    # ... create orders with same parameters except participation

    slip1 = provider.compute_slippage_pips(order, market, participation)
    slip2 = provider.compute_slippage_pips(order, market, participation2)

    assert slip1 <= slip2, f"Slippage not monotonic: {slip1} > {slip2}"


@given(symbol=st.sampled_from(["EUR_USD", "GBP_JPY", "USD_TRY", "EUR_GBP"]))
def test_pair_classification_deterministic(symbol):
    """Pair classification should be deterministic."""
    provider = ForexParametricSlippageProvider()

    result1 = provider._classify_pair(symbol)
    result2 = provider._classify_pair(symbol)

    assert result1 == result2


@given(
    hour=st.integers(0, 23),
    minute=st.integers(0, 59),
)
def test_session_detection_complete(hour, minute):
    """Every timestamp should map to exactly one session."""
    from services.forex_session_router import ForexSessionRouter
    from adapters.models import ForexSessionType

    router = ForexSessionRouter()
    # Create timestamp for the hour/minute
    # ... verify session is not None and is valid enum
```

#### 10.6 Stress Tests

```python
# tests/test_forex_stress.py
"""
Stress tests for forex simulation.

Scenarios:
1. CHF flash crash (Jan 2015): 30% move in minutes
2. NFP release: Extreme volatility spike
3. Weekend gap: Large gap on Sunday open
4. API rate limit: 120 requests/sec exceeded
5. All dealers reject: No fill scenario
"""
import pytest

class TestForexStressScenarios:

    def test_chf_flash_crash_scenario(self):
        """
        Simulate CHF flash crash conditions.

        Jan 15, 2015: SNB removed EUR/CHF floor
        - EUR/CHF dropped 30% in minutes
        - Spreads widened to 50+ pips
        - Many dealers stopped quoting
        """
        provider = ForexParametricSlippageProvider()

        # Extreme volatility regime
        returns = [-0.05, -0.08, -0.12, -0.06, -0.03]  # Large negative

        slippage = provider.compute_slippage_pips(
            order=Order("EUR_CHF", "SELL", 100000, "MARKET"),
            market=MarketState(timestamp=0, bid=1.0, ask=1.02, spread_bps=200),
            participation_ratio=0.01,
            recent_returns=returns,
        )

        # Should hit max slippage bound
        assert slippage == provider.config.max_slippage_pips

    def test_nfp_release_spike(self):
        """Simulate Non-Farm Payrolls release conditions."""
        provider = ForexParametricSlippageProvider()

        slippage = provider.compute_slippage_pips(
            order=Order("EUR_USD", "BUY", 1000000, "MARKET"),
            market=MarketState(timestamp=0, bid=1.1000, ask=1.1005),
            participation_ratio=0.005,
            upcoming_news="nfp",
        )

        # NFP multiplier should apply (3.0x)
        assert slippage > provider.config.min_slippage_pips * 2

    def test_all_dealers_reject(self):
        """Test scenario where all dealers reject via last-look."""
        from services.forex_dealer import ForexDealerSimulator, ForexDealerConfig

        # Configure high rejection probability
        config = ForexDealerConfig(
            num_dealers=3,
            last_look_enabled=True,
        )

        sim = ForexDealerSimulator(config=config, seed=42)

        # ... simulate adverse price move and verify rejection handling
```

#### 10.7 Validation Metrics

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Spread accuracy | ±15% vs OANDA live | Compare simulated vs actual |
| Session liquidity | ±25% vs historical | Backtest ADV comparison |
| Fill rate (majors) | >98% | Paper trading validation |
| Fill rate (exotics) | >90% | Paper trading validation |
| Last-look reject rate | 5-12% | Industry benchmarks |
| Slippage estimate | ±2 pips (majors) | Paper trading |
| Position sync accuracy | 100% | Unit tests |
| **Backward compatibility** | **100% crypto/equity** | **Regression suite (mandatory)** |
| **Isolation verification** | **100%** | **Isolation tests** |
| **API contract preservation** | **100%** | **Backward compat tests** |

#### 10.8 Continuous Regression Protocol (ОБЯЗАТЕЛЬНО!)

**Каждая фаза ДОЛЖНА пройти регрессионный гейт перед merge:**

```bash
# Phase Gate Protocol - выполнять после КАЖДОЙ фазы

# 1. Run existing test suite (MUST PASS 100%)
pytest tests/ --ignore=tests/test_forex*.py -v --tb=short
# Expected: All existing tests pass

# 2. Run phase-specific regression gate
pytest tests/test_forex_regression.py::TestPhaseRegressionGates -v -m phaseN
# Replace N with current phase number

# 3. Run isolation tests (after Phase 2+)
pytest tests/test_forex_isolation.py -v

# 4. Run backward compatibility tests (after Phase 3+)
pytest tests/test_forex_backward_compat.py -v

# 5. Run full forex test suite
pytest tests/test_forex*.py -v

# MERGE CRITERIA:
# - Steps 1-4 MUST pass with 0 failures
# - Step 5 coverage must be >=90% for new code
```

**CI/CD Integration (GitHub Actions):**

```yaml
# .github/workflows/forex-regression.yml
name: Forex Regression Gate

on:
  pull_request:
    branches: [main, feature/forex-integration]
    paths:
      - 'adapters/**'
      - 'execution_providers.py'
      - 'services/forex_*.py'
      - 'forex_features.py'

jobs:
  regression-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run existing tests (regression check)
        run: |
          pytest tests/ --ignore=tests/test_forex*.py \
            -v --tb=short --junitxml=regression.xml
          # This MUST pass 100%

      - name: Run isolation tests
        run: pytest tests/test_forex_isolation.py -v

      - name: Run backward compatibility tests
        run: pytest tests/test_forex_backward_compat.py -v

      - name: Run forex-specific tests
        run: pytest tests/test_forex*.py -v --cov=. --cov-report=xml

      - name: Check coverage threshold
        run: |
          coverage report --fail-under=90 \
            --include="adapters/oanda/*,execution_providers.py,forex_features.py"
```

**Deliverables**:

- **735+ tests total** (увеличено на 115 для regression/isolation/backward compat)
- Property-based test suite (40 tests)
- Stress test scenarios (20 tests)
- **Regression test suite (45 tests) -- NEW**
- **Isolation test suite (35 tests) -- NEW**
- **Backward compatibility tests (30 tests) -- NEW**
- Validation report
- CI/CD pipeline configuration

---

## 📊 Summary

### Phase Overview (Updated v2.1)

| Phase | Description | Duration | LOC | Tests | Regression Gate |
|-------|-------------|----------|-----|-------|-----------------|
| 0 | Foundation & Research | 1 week | 100 | 25 | N/A |
| 1 | Core Enums & Models | 1 week | 250 | 35 | `pytest -m phase1` |
| 2 | OANDA Adapter | 2 weeks | 1,500 | 130 | `pytest -m phase2` |
| 3 | ForexParametricSlippage (L2+) | 2 weeks | 600 | 100 | `pytest -m phase3` |
| 4 | Forex Features | 2 weeks | 500 | 80 | `pytest -m phase4` |
| 5 | OTC Dealer Simulation | 2 weeks | 600 | 85 | `pytest -m phase5` |
| 6 | Risk Management + Services | 2 weeks | 900 | 100 | `pytest -m phase6` |
| 7 | Data Pipeline | 1 week | 450 | 50 | `pytest -m phase7` |
| 8 | Configuration | 1 week | 350 | 35 | `pytest -m phase8` |
| 9 | Training Integration | 1 week | 350 | 45 | `pytest -m phase9` |
| 10 | Testing & Validation | **2 weeks** | 400 | **735 (total)** | `pytest -m full` |
| **TOTAL** | | **17 weeks** | **6,000** | **735** | |

### Test Distribution Summary

| Category | Test Count | Purpose |
|----------|------------|---------|
| Forex Unit Tests | 625 | Forex-specific functionality |
| **Regression Tests** | **45** | Crypto/equity unchanged |
| **Isolation Tests** | **35** | No cross-contamination |
| **Backward Compat Tests** | **30** | API contracts preserved |
| **TOTAL** | **735** | |

### Risk Assessment (Updated v2.1)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OANDA API changes | Low | Medium | Abstract API layer, version pinning |
| OANDA rate limits (120/s) | Medium | Low | Rate limiter with exponential backoff |
| Forex data quality | Medium | High | Multiple data sources, validation |
| DST handling bugs | Medium | Medium | Extensive timezone tests, ZoneInfo |
| Session detection bugs | Medium | Medium | Property-based testing |
| **Crypto regression** | **Low** | **Critical** | **Mandatory regression gate per phase** |
| **Equity regression** | **Low** | **Critical** | **Mandatory regression gate per phase** |
| **Import pollution** | **Low** | **High** | **Isolation tests, lazy loading** |
| **API contract break** | **Low** | **Critical** | **Backward compat tests** |
| OTC simulation accuracy | Medium | Medium | Validate vs paper trading |
| Last-look calibration | Medium | Medium | Adjustable parameters, A/B testing |

### Regression Prevention Checklist (Per Phase)

```
□ All existing tests pass (pytest tests/ --ignore=test_forex*)
□ Phase regression gate passes (pytest -m phaseN)
□ Isolation tests pass (after Phase 2)
□ Backward compat tests pass (after Phase 3)
□ No new imports in crypto/equity code paths
□ No changes to existing API signatures
□ Code review confirms isolation
```

### Success Criteria

1. **Functional**
   - [ ] Full pipeline from data download to live trading
   - [ ] All 7 major pairs fully supported
   - [ ] Session-aware execution with DST handling
   - [ ] Position synchronization operational

2. **Performance**
   - [ ] Slippage estimate accuracy ±15%
   - [ ] Fill rate >98% for major pairs
   - [ ] API rate limit compliance

3. **Quality**
   - [ ] **735+ tests passing**
   - [ ] Property-based tests coverage
   - [ ] Stress scenarios handled
   - [ ] Code coverage >90% for new code
   - [ ] Documentation complete

4. **Regression Prevention (КРИТИЧНО!)**
   - [ ] **100% existing crypto tests pass** (no regressions)
   - [ ] **100% existing equity tests pass** (no regressions)
   - [ ] **All isolation tests pass** (no cross-contamination)
   - [ ] **All backward compat tests pass** (API unchanged)
   - [ ] **Phase gates passed** for all 10 phases
   - [ ] **CI/CD regression pipeline operational**
   - [ ] Code review confirms no changes to crypto/equity code paths

---

## 📚 References

### Academic

- Lyons, R. (2001): "The Microstructure Approach to Exchange Rates", MIT Press
- Evans, M. & Lyons, R. (2002): "Order Flow and Exchange Rate Dynamics", Journal of Political Economy
- Berger, D. et al. (2008): "The Development of the Global FX Market", BIS Quarterly Review
- **King, M., Osler, C., Rime, D. (2012)**: "Foreign Exchange Market Structure, Players, and Evolution"
- Oomen, R. (2017): "Last Look", Journal of Financial Markets
- **Chaboud, A. et al. (2014)**: "Rise of the Machines: Algorithmic Trading in the FX Market"
- Hasbrouck, J. & Saar, G. (2013): "Low-latency Trading", Journal of Financial Markets
- Brunnermeier, M. et al. (2008): "Carry Trades and Currency Crashes"
- **Almgren, R. & Chriss, N. (2001)**: "Optimal Execution of Portfolio Transactions"

### Industry

- **BIS (2022)**: Triennial Central Bank Survey of Foreign Exchange Markets
- OANDA v20 API Documentation: https://developer.oanda.com/rest-live-v20/
- CFTC Commitments of Traders Reports
- ForexFactory Economic Calendar

---

## 🔜 Next Steps

1. **Approve plan v2.0** - Review and finalize
2. **Set up OANDA demo account** - Practice account for development
3. **Create branch** - `feature/forex-integration`
4. **Phase 0** - Start research and infrastructure setup
5. **Weekly sync** - Progress review and adjustment

---

**Author**: Sultan Suyunbayev
**Version**: 2.1 (Testing & Regression Hardened)
**Last Updated**: 2025-11-29
**Reviewer Notes**:

**v2.1 Changes (Regression & Testing Focus):**

- Added comprehensive regression test suite (45 tests)
- Added isolation test suite (35 tests)
- Added backward compatibility tests (30 tests)
- Added phase-specific regression gates
- Added CI/CD pipeline configuration (GitHub Actions)
- Added continuous regression protocol
- Updated test count: 620 → 735
- Updated timeline: 16 → 17 weeks
- Added regression prevention checklist
- Added regression-specific risks to risk assessment
- Added Section 4 to Success Criteria (Regression Prevention)

**v2.0 Changes:**

- Fixed L3 terminology (OTC ≠ LOB)
- Added missing services (position_sync, session_router)
- Fixed file locations (execution_providers.py, services/)
- Added property-based and stress testing
- Extended timeline to 16 weeks
- Added data sources for calendar/swaps
- Added research references

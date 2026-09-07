# ADR-001: Unified Multi-Asset Futures Architecture

## Status

**PROPOSED** | Date: 2025-11-30 | Author: Sultan Suyunbayev

---

## Context

The trading platform currently supports:

- Crypto spot (Binance)
- US Equities (Alpaca, Polygon)
- Forex (OANDA)

We need to add support for futures trading across multiple asset classes:

- **Crypto Perpetuals** - Binance USDT-M perpetual futures
- **Crypto Quarterly** - Binance quarterly delivery futures
- **Equity Index Futures** - CME ES, NQ, YM, RTY
- **Commodity Futures** - CME GC, CL, SI, NG
- **Currency Futures** - CME 6E, 6J, 6B

The key challenge is creating a unified architecture that:

1. Handles diverse settlement mechanisms (perpetual funding vs daily settlement)
2. Supports different margin systems (tiered vs SPAN)
3. Manages contract expiration and rollover
4. Maintains consistent API across exchanges (Binance vs IB/CME)

---

## Decision

### 1. Unified Futures Model Hierarchy

We will create a **three-tier model hierarchy** that abstracts futures-specific concepts:

```
FuturesContractBase (abstract)
├── PerpetualContract
│   └── Binance USDT-M perpetuals
├── DeliveryContract
│   ├── Binance quarterly
│   └── CME all contracts
└── ContinuousContract
    └── Auto-rolled continuous front month
```

**Rationale**: This hierarchy captures the fundamental difference between perpetual (no expiry, funding-based) and delivery (expiration, physical/cash settlement) contracts while providing a common interface.

### 2. Margin System Abstraction

We will implement a **pluggable margin calculator** interface:

```python
class MarginCalculator(Protocol):
    def initial_margin(self, position: Position) -> Decimal: ...
    def maintenance_margin(self, position: Position) -> Decimal: ...
    def liquidation_price(self, position: Position) -> Decimal: ...

class TieredMarginCalculator(MarginCalculator):
    """Binance-style tiered leverage brackets"""

class SPANMarginCalculator(MarginCalculator):
    """CME SPAN portfolio margin"""
```

**Rationale**: Binance uses simple tiered brackets while CME uses SPAN portfolio margining. A protocol-based approach allows both to coexist.

### 3. Settlement Handler Pattern

We will use a **Strategy pattern** for settlement:

```python
class SettlementHandler(Protocol):
    def process_settlement(self, position: Position, market: MarketState) -> SettlementResult: ...

class FundingSettlement(SettlementHandler):
    """8-hour funding rate settlement (Binance perps)"""

class DailySettlement(SettlementHandler):
    """Daily cash settlement (CME futures)"""

class PhysicalDelivery(SettlementHandler):
    """Physical delivery at expiration"""
```

**Rationale**: Different futures types have fundamentally different settlement mechanisms. The Strategy pattern allows clean separation of these concerns.

### 4. Exchange Adapter Extension

We will extend the existing adapter pattern with futures-specific interfaces:

```
adapters/
├── base.py                    # Existing base classes
├── models.py                  # Extended MarketType enum
├── binance/
│   ├── market_data.py        # Existing (use_futures flag)
│   ├── futures_market_data.py # NEW: Futures-specific endpoints
│   ├── futures_exchange_info.py # NEW: Contract specs
│   └── futures_order_execution.py # NEW: Futures orders
└── interactive_brokers/       # NEW: IB adapter for CME
    ├── __init__.py
    ├── market_data.py
    ├── order_execution.py
    ├── contract_resolver.py   # Symbol → IB Contract
    └── margin_calculator.py   # SPAN approximation
```

**Rationale**: Separate futures adapters avoid polluting existing spot/equity code while maintaining the established adapter pattern.

### 5. Rollover Management

We will implement **automatic rollover detection and handling**:

```python
class RolloverManager:
    def detect_rollover_date(self, symbol: str) -> datetime:
        """Detect optimal rollover date based on volume/OI crossover"""

    def construct_continuous_series(self, symbol: str, method: RolloverMethod) -> pd.DataFrame:
        """Build continuous contract with specified adjustment method"""

class RolloverMethod(Enum):
    RATIO_ADJUST = "ratio"      # Multiplicative adjustment
    DIFFERENCE_ADJUST = "diff"  # Additive adjustment
    UNADJUSTED = "raw"          # No adjustment (gaps)
```

**Rationale**: Futures require continuous price series for backtesting. Multiple adjustment methods support different use cases (returns-based vs price-based analysis).

### 6. L2+ Parametric TCA for Futures

We will create a **FuturesParametricSlippageProvider** following existing L2+ patterns:

```python
class FuturesParametricSlippageProvider(SlippageProvider):
    """L2+ smart parametric TCA for futures markets"""

    # Slippage factors:
    # 1. √participation (Almgren-Chriss)
    # 2. Funding rate stress (perps only)
    # 3. Open interest imbalance
    # 4. Roll period premium (near expiry)
    # 5. Time-of-day liquidity (CME sessions)
    # 6. Volatility regime
```

**Rationale**: Futures have unique cost factors (funding, roll premium) that don't exist in spot markets. A dedicated provider captures these effects.

### 7. Feature Engineering Extension

We will add **futures-specific features** to the feature pipeline:

```python
FUTURES_FEATURES = [
    # Perpetual-specific
    "funding_rate",           # Current 8h funding
    "funding_rate_ma_24h",    # 24h moving average
    "funding_rate_zscore",    # Standardized funding

    # Common futures
    "basis",                  # Futures - spot spread
    "basis_annualized",       # Annualized basis (carry)
    "open_interest",          # Total OI
    "oi_change_1h",           # OI rate of change
    "long_short_ratio",       # Sentiment

    # CME-specific
    "commitment_of_traders",  # COT positioning (weekly)
    "roll_days_remaining",    # Days to rollover
]
```

**Rationale**: These features capture futures-specific alpha signals that don't exist in spot markets.

### 8. Risk Guard Extension

We will create **FuturesRiskGuard** extending ForexRiskGuard pattern:

```python
class FuturesRiskGuard:
    def check_margin_requirement(self, position, price, leverage) -> MarginStatus: ...
    def check_liquidation_proximity(self, position, mark_price) -> LiquidationRisk: ...
    def check_funding_exposure(self, positions, funding_rates) -> FundingRisk: ...
    def check_adl_risk(self, position, adl_quantile) -> ADLRisk: ...
    def check_expiration_proximity(self, position, expiry_date) -> ExpirationRisk: ...
```

**Rationale**: Futures have unique risks (liquidation, funding, ADL, expiration) requiring dedicated guards.

---

## Consequences

### Positive

1. **Unified Interface**: Single API for all futures types simplifies strategy development
2. **Code Reuse**: Extends existing patterns (adapters, TCA providers, risk guards)
3. **Flexibility**: Protocol-based design allows easy addition of new exchanges
4. **Maintainability**: Clear separation of exchange-specific vs common logic
5. **Testability**: Each component can be tested independently

### Negative

1. **Complexity**: Multiple abstraction layers add cognitive overhead
2. **Performance**: Protocol dispatch slightly slower than direct calls
3. **Learning Curve**: Developers must understand the hierarchy
4. **IB Dependency**: CME access requires Interactive Brokers infrastructure

### Neutral

1. **Migration**: Existing code unchanged, new features are additive
2. **Configuration**: Additional YAML config files required
3. **Testing**: Significant new test coverage needed

---

## Alternatives Considered

### Alternative 1: Separate Codebases

**Description**: Create independent implementations for each futures type.

**Rejected Because**:

- Massive code duplication
- Inconsistent APIs
- Maintenance nightmare

### Alternative 2: Single Monolithic Futures Module

**Description**: One large module handling all futures types with conditionals.

**Rejected Because**:

- Violates Single Responsibility Principle
- Difficult to test
- Hard to extend

### Alternative 3: External Futures Library

**Description**: Use third-party library like `zipline-reloaded` or `backtrader`.

**Rejected Because**:

- Different architectural patterns
- Integration complexity
- Loss of control over execution simulation

---

## Implementation Plan

### Phase 1: Crypto Perpetuals (Weeks 1-2)

1. Extend `MarketType` enum
2. Create `PerpetualContract` model
3. Create `TieredMarginCalculator`
4. Create `FundingSettlement` handler
5. Extend Binance adapter for futures

### Phase 2: L2+ Execution (Week 3)

1. Create `FuturesParametricSlippageProvider`
2. Add funding rate impact factor
3. Add open interest imbalance factor
4. Create execution simulation tests

### Phase 3: Features & Risk (Week 4)

1. Add futures features to pipeline
2. Create `FuturesRiskGuard`
3. Add funding exposure tracking
4. Create risk guard tests

### Phase 4: CME Integration (Weeks 5-6)

1. Create IB adapter
2. Create `SPANMarginCalculator` (simplified)
3. Create `DailySettlement` handler
4. Add CME trading hours

### Phase 5: Rollover & Continuous (Week 7)

1. Create `RolloverManager`
2. Implement volume-based detection
3. Create continuous contract construction
4. Validate against historical data

### Phase 6: Integration & Testing (Week 8)

1. End-to-end integration tests
2. Paper trading validation
3. Documentation
4. Performance optimization

---

## Technical Specifications

### 1. Contract Model

```python
@dataclass
class FuturesContractBase:
    symbol: str
    underlying: str
    exchange: ExchangeVendor
    market_type: MarketType
    contract_size: Decimal
    tick_size: Decimal
    tick_value: Decimal
    currency: str
    trading_hours: TradingHours

@dataclass
class PerpetualContract(FuturesContractBase):
    funding_interval_hours: int = 8
    max_leverage: int = 125

@dataclass
class DeliveryContract(FuturesContractBase):
    expiration_date: datetime
    last_trading_date: datetime
    settlement_type: SettlementType
    first_notice_date: Optional[datetime] = None
```

### 2. Position Model

```python
@dataclass
class FuturesPosition:
    contract: FuturesContractBase
    size: Decimal  # Positive = long, negative = short
    entry_price: Decimal
    mark_price: Decimal
    leverage: int
    margin_type: MarginType  # CROSS or ISOLATED
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    liquidation_price: Optional[Decimal]

    @property
    def notional_value(self) -> Decimal:
        return abs(self.size) * self.contract.contract_size * self.mark_price
```

### 3. Configuration Schema

```yaml
# configs/futures.yaml
futures:
  crypto_perpetual:
    default_leverage: 10
    max_leverage: 125
    margin_type: cross
    funding_check_enabled: true

  equity_index:
    margin_type: span
    initial_margin_pct: 5.0
    maintenance_margin_pct: 4.5

  rollover:
    method: ratio_adjust
    detection: volume_crossover
    days_before_expiry: 5
```

---

## References

1. Binance Futures API Documentation
2. Interactive Brokers TWS API Guide
3. CME SPAN Margin Methodology
4. Almgren-Chriss (2001) - Market Impact Model
5. Existing codebase patterns:
   - `adapters/models.py`
   - `services/forex_risk_guards.py`
   - `execution_providers.py`

---

## Approval

| Role | Name | Date | Decision |
|------|------|------|----------|
| Architect | | | |
| Tech Lead | | | |
| Product Owner | | | |

---

**Document End**

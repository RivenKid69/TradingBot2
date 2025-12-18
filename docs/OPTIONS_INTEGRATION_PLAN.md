# OPTIONS_INTEGRATION_PLAN.md

## CustodiaCloud -- Options Integration (Optional Expansion)

**Version**: 5.1
**Status**: IN PROGRESS (Phase 2B ✅ Complete)
**Target Completion**: Q1 2027
**Estimated Tests**: 3,100
**Realism Target**: 95%+
**Last Updated**: 2025-12-03
**Phase 0.5 Tests**: 71/71 ✅
**Phase 1 Tests**: 240/240 ✅
**Phase 2 Tests**: 160/160 ✅ (159 pass, 1 skip)
**Phase 2B Tests**: 118/118 ✅
**Total Completed**: 589 tests across 4 phases (33.3% of project)

---

## ⚠️ CRITICAL ISSUES ADDRESSED IN v5.0

### Memory Architecture (480+ LOBs)
- **Problem**: SPY chain = 24 expiries × 20 strikes = 480 order books. At 500MB/LOB = **240GB RAM**
- **Solution**: Phase 0.5 adds lazy LOB instantiation, LRU eviction, ring buffer depth limiting

### Jump λ Calibration
- **Problem**: Merton jump-diffusion λ not calibrated from data
- **Solution**: Phase 1 adds calibration from historical earnings moves + VIX term structure

### SSVI → Lee Wing Transition
- **Problem**: Abrupt transition at wing boundaries causes kink in IV surface
- **Solution**: Phase 3 adds smooth hyperbolic tangent blending over 5% strike range

### Heston COS Truncation
- **Problem**: COS method truncation bounds L not specified → numerical instability
- **Solution**: Phase 3 specifies L = c₁ + c₂√T with Fang-Oosterlee (2008) formulas

### Dupire Tikhonov λ Selection
- **Problem**: Regularization λ arbitrary, no cross-validation
- **Solution**: Phase 3 adds GCV (Generalized Cross-Validation) for automatic λ selection

### 480 LOBs Memory (240GB Issue)
- **Solution**: Phase 5 uses lazy instantiation + LRU cache (max 50 active LOBs) + ring buffer depth (100 levels)

### Cross-Series LOB O(N²) Complexity
- **Problem**: N series × N quote updates = O(N²) per tick
- **Solution**: Phase 5 uses event-driven updates with strike bucketing, O(N log N)

### STANS 10K Scenario Runtime
- **Problem**: 10K full repricing per position × 1000 positions = 10B operations
- **Solution**: Phase 6 uses delta-gamma-vega approximation for 99% scenarios, full repricing for tail 1%

### Corporate Actions
- **Solution**: Phase 6 adds OCC adjustment handling (splits, special dividends, mergers, spin-offs)

### IB Rate Limits (10 chains/min)
- **Solution**: Phase 2 adds chain caching (5-min TTL), incremental delta updates, priority queue

### Assignment Risk Model
- **Solution**: Phase 6 adds Broadie-Detemple early exercise boundary + dividend timing logic

### Delta Hedge Frequency
- **Solution**: Phase 7 adds Whalley-Wilmott (1997) asymptotic expansion for optimal rehedge bands

### LOB Pro-Rata vs FIFO
- **Problem**: Options markets (CBOE, PHLX) use pro-rata allocation, not FIFO
- **Solution**: Phase 5 uses `ProRataMatchingEngine` from existing `lob/matching_engine.py`

### Rough Volatility Consideration
- **Note**: Gatheral et al. (2018) rough volatility (H≈0.1) is research frontier, NOT included in v5.0
- **Rationale**: Calibration complexity too high for L3 realism target; Heston + Bates sufficient

---

## ⚠️ CRITICAL: Existing Code to Reuse (NOT Duplicate!)

### Existing Options Infrastructure

**ВАЖНО**: Следующие компоненты УЖЕ РЕАЛИЗОВАНЫ и должны быть РАСШИРЕНЫ, не дублированы:

#### 1. Alpaca Options Adapter (`adapters/alpaca/options_execution.py` -- 1065 lines)

```python
# УЖЕ СУЩЕСТВУЕТ:
class OptionType(str, Enum):      # Использовать этот, НЕ создавать новый!
    CALL = "call"
    PUT = "put"

class OptionStrategy(str, Enum):  # 11 стратегий уже реализовано
    SINGLE, COVERED_CALL, PROTECTIVE_PUT, VERTICAL_SPREAD,
    CALENDAR_SPREAD, DIAGONAL_SPREAD, STRADDLE, STRANGLE,
    IRON_CONDOR, BUTTERFLY, COLLAR

class OptionOrderType(str, Enum):
    MARKET, LIMIT, STOP, STOP_LIMIT

@dataclass
class OptionContract:             # OCC symbology уже реализован!
    symbol: str
    occ_symbol: str               # "AAPL  241220C00200000"
    option_type: OptionType
    strike_price: float
    expiration_date: date
    multiplier: int = 100
    # Greeks: delta, gamma, theta, vega, implied_volatility
    # Market data: bid, ask, last_price, volume, open_interest

    @classmethod
    def from_occ_symbol(cls, occ_symbol: str) -> "OptionContract":
        """Parse OCC symbol into OptionContract."""

    def to_occ_symbol(self) -> str:
        """Generate OCC symbol from contract details."""

class AlpacaOptionsExecutionAdapter(OrderExecutionAdapter):
    """Полная реализация options execution через Alpaca."""
```

**Действие**: Импортировать из `adapters.alpaca.options_execution`, не переопределять!

#### 2. LOB Module (`lob/` -- 24 файла, v8.0.0)

```python
# УЖЕ СУЩЕСТВУЕТ в lob/__init__.py:
from lob import (
    # Core: OrderBook, LimitOrder, PriceLevel, Fill, Trade
    # Matching: MatchingEngine, ProRataMatchingEngine
    # Queue: QueuePositionTracker, QueueState
    # Fill Probability: QueueReactiveModel, HistoricalRateModel
    # Market Impact: AlmgrenChrissModel, GatheralModel, KyleLambdaModel
    # Latency: LatencyModel, EventScheduler
    # Dark Pools: DarkPoolSimulator, DarkPoolVenue
    # Configuration: L3ExecutionConfig
    # Calibration: L3CalibrationPipeline
)
```

**Действие**: Создавать `lob/options_*.py` как РАСШИРЕНИЕ существующих классов!

#### 3. Futures Environment Wrapper Pattern (`wrappers/futures_env.py`)

```python
# Паттерн для options_env.py:
class FuturesTradingEnv(gym.Wrapper):
    """Futures-specific adjustments."""
    # Leverage control, margin tracking
    # Funding payment integration
    # Liquidation handling
    # Feature flags integration
```

**Действие**: Использовать как шаблон для `wrappers/options_env.py`!

#### 4. Error Classes Pattern (`core_errors.py`)

```python
# ДОБАВИТЬ в core_errors.py:
class OptionsError(BotError):
    """Base options error."""

class GreeksCalculationError(OptionsError):
    """Greeks calculation failure."""

class IVConvergenceError(OptionsError):
    """IV solver failed to converge."""

class ExerciseError(OptionsError):
    """Exercise/assignment error."""

class MarginError(OptionsError):
    """OCC margin calculation error."""
```

#### 5. Protocol Pattern (`execution_providers.py`)

```python
# Опционные провайдеры ДОЛЖНЫ наследовать Protocol:
from execution_providers import SlippageProvider, FillProvider, FeeProvider

class OptionsSlippageProvider(SlippageProvider):
    """Must implement compute_slippage_bps()."""

class OptionsFillProvider(FillProvider):
    """Must implement execute()."""

class OptionsFeeProvider(FeeProvider):
    """Must implement compute_fee()."""
```

#### 6. Feature Flags Pattern (`services/futures_feature_flags.py`)

```python
# Создать аналогичный для опционов:
class OptionsFeatureFlags:
    """Options feature flags with rollout stages."""
    # DISABLED, SHADOW, CANARY, PRODUCTION

class OptionsFeature(Enum):
    GREEKS_THIRD_ORDER = "greeks_third_order"
    IV_SURFACE_SSVI = "iv_surface_ssvi"
    L3_LOB_OPTIONS = "l3_lob_options"
    MM_SIMULATOR = "mm_simulator"
    EXERCISE_OPTIMIZATION = "exercise_optimization"
```

---

### Import Consolidation

**Phase 1 должен начинаться с:**

```python
# core_options.py -- НЕ переопределять существующие!
from adapters.alpaca.options_execution import (
    OptionType,           # Использовать существующий enum!
    OptionStrategy,       # Существующие стратегии
    OptionOrderType,      # Существующие типы ордеров
    OptionContract,       # OCC symbology уже реализован!
    OPTIONS_CONTRACT_MULTIPLIER,
)

from execution_providers import AssetClass  # AssetClass.OPTIONS уже есть!
from core_errors import BotError  # Для наследования OptionsError
```

---

---

## Executive Summary

Полная интеграция опционов всех типов на уровне L3 LOB simulation с поддержкой:
- **US Listed Options** (CBOE, NYSE Arca, NASDAQ PHLX)
- **Index Options** (SPX, VIX, NDX -- cash-settled, European)
- **Futures Options** (ES, CL, GC options -- via CME/CBOT)
- **Crypto Options** (Deribit BTC/ETH -- European, cash-settled)

**Ключевые метрики качества:**
| Metric | Target |
|--------|--------|
| Greeks Accuracy | < 0.1% vs analytical |
| IV Surface Error | < 0.5 vol points |
| Fill Rate (L3) | > 90% |
| Slippage Error | < 3 bps |
| Exercise Timing | < 1 bar |
| Test Coverage | 100% per phase |

---

## Critical Design Principles

### Integration with Existing Codebase

**ВАЖНО**: План интегрируется с существующей архитектурой проекта:

1. **AssetClass.OPTIONS** уже определён в `execution_providers.py:54` -- использовать его
2. **LOB модуль** (`lob/`) уже имеет 8-stage архитектуру v8.0.0 -- расширять, не дублировать
3. **IB адаптер** (`adapters/ib/`) уже существует -- расширять `market_data.py`, `order_execution.py`
4. **Risk guards** следуют паттерну из `services/futures_risk_guards.py` -- dataclasses + enums + multi-level alerts
5. **Env wrappers** следуют паттерну из `wrappers/futures_env.py` -- gymnasium.Wrapper
6. **Factory functions** следуют паттерну из `execution_providers.py` -- `create_*_provider()`

### Academic Foundation

Все алгоритмы основаны на peer-reviewed исследованиях:

| Component | Primary Reference | Year |
|-----------|-------------------|------|
| IV Surface | Gatheral & Jacquier: "Arbitrage-free SVI volatility surfaces" | 2014 |
| American IV | Brenner & Subrahmanyam: "A simple approach to option valuation" | 1994 |
| MM Behavior | Cho & Engle: "Market Maker Quotes in Options Markets" | 2022 |
| Early Exercise | Broadie & Detemple: "American Option Valuation" | 1996 |
| Local Vol | Dupire: "Pricing with a Smile" + Tikhonov regularization | 1994 |
| Pin Risk | Avellaneda & Lipkin: "A market-induced mechanism for stock pinning" | 2003 |
| Jump Diffusion | Merton: "Option pricing when underlying stock returns are discontinuous" | 1976 |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OPTIONS INTEGRATION ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 1: Core Models         │  Phase 2: Exchange Adapters         │
│  ├─ core_options.py           │  ├─ adapters/ib/options.py          │
│  ├─ impl_greeks.py            │  ├─ adapters/theta_data/options.py  │
│  ├─ impl_pricing.py           │  ├─ adapters/deribit/options.py     │
│  └─ impl_iv_calculation.py    │  └─ adapters/polygon/options.py     │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 3: IV Surface          │  Phase 4: L2 Execution              │
│  ├─ impl_iv_surface.py        │  ├─ execution_providers_options.py  │
│  ├─ impl_ssvi.py (SSVI model) │  ├─ impl_options_slippage.py        │
│  ├─ impl_heston.py            │  └─ impl_options_fees.py            │
│  └─ service_iv_calibration.py │                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 5: L3 LOB              │  Phase 6: Risk Management           │
│  ├─ lob/options_matching.py   │  ├─ services/options_risk_guards.py │
│  ├─ lob/options_mm.py         │  ├─ impl_occ_margin.py              │
│  └─ lob/pin_risk.py           │  └─ impl_exercise_assignment.py     │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 7: Complex Orders      │  Phase 8: Training Integration      │
│  ├─ impl_multi_leg.py         │  ├─ wrappers/options_env.py         │
│  ├─ impl_spread_execution.py  │  ├─ options_features.py             │
│  └─ strategies/vol_arb.py     │  └─ impl_options_reward.py          │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 9: Live Trading        │  Phase 10: Validation               │
│  ├─ services/options_live.py  │  ├─ tests/test_options_*.py         │
│  ├─ services/greeks_monitor.py│  ├─ benchmarks/bench_options.py     │
│  └─ services/exercise_mgr.py  │  └─ docs/options/*.md               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0.5: Memory Architecture Design (2 weeks) -- NEW

### 0.5.1 Problem Statement

**Options LOB Memory Crisis**:
- SPY chain: 24 expiries × 20 strikes × 2 (call/put) = **960 series**
- Each LOB at full depth (1000 levels): ~500MB
- Total naive memory: **480 GB** -- impossible!

**Comparison with Futures**:
- Futures: 1 LOB per symbol (ES, NQ, etc.)
- Options: 100-1000 LOBs per underlying
- Fundamentally different architecture required

### 0.5.2 Solution: Lazy LOB Architecture

```python
class LazyMultiSeriesLOBManager:
    """
    Memory-efficient multi-series LOB manager.

    Strategy:
    1. Lazy instantiation: Only create LOB when first accessed
    2. LRU eviction: Keep max N LOBs in memory (default: 50)
    3. Ring buffer depth: Limit each LOB to M levels (default: 100)
    4. Compressed storage: Store evicted LOBs compressed on disk

    Memory budget:
    - 50 active LOBs × 50MB (100 levels) = 2.5 GB ✓
    - vs 960 full LOBs × 500MB = 480 GB ✗

    Reference: Memory-mapped files pattern from lob/data_adapters.py
    """

    def __init__(
        self,
        max_active_lobs: int = 50,
        max_depth_per_lob: int = 100,
        eviction_policy: str = "lru",  # "lru", "lfu", "ttl"
        disk_cache_path: Optional[Path] = None,
    ):
        self._active_lobs: OrderedDict[str, SeriesLOB] = OrderedDict()
        self._max_active = max_active_lobs
        self._max_depth = max_depth_per_lob

    def get_lob(self, series_key: str) -> SeriesLOB:
        """
        Get or create LOB for series. Evicts LRU if at capacity.

        Series key format: "AAPL_241220_C_200" (symbol_expiry_type_strike)
        """
        if series_key in self._active_lobs:
            # Move to end (most recently used)
            self._active_lobs.move_to_end(series_key)
            return self._active_lobs[series_key]

        # Evict if at capacity
        if len(self._active_lobs) >= self._max_active:
            self._evict_lru()

        # Create new LOB
        lob = self._create_or_restore_lob(series_key)
        self._active_lobs[series_key] = lob
        return lob

    def _evict_lru(self) -> None:
        """Evict least recently used LOB, optionally persist to disk."""
        oldest_key, oldest_lob = self._active_lobs.popitem(last=False)
        if self._disk_cache_path:
            self._persist_to_disk(oldest_key, oldest_lob)


class RingBufferOrderBook:
    """
    Memory-efficient order book with fixed depth.

    Instead of unlimited levels, keeps only top N bid/ask levels.
    Beyond N levels, aggregates into "rest of book" bucket.

    Memory: O(N) instead of O(all_levels)
    """

    def __init__(self, max_depth: int = 100):
        self._bid_levels: Deque[PriceLevel] = deque(maxlen=max_depth)
        self._ask_levels: Deque[PriceLevel] = deque(maxlen=max_depth)
        self._bid_rest: AggregatedLevel = AggregatedLevel()  # Beyond top N
        self._ask_rest: AggregatedLevel = AggregatedLevel()
```

### 0.5.3 Event-Driven Updates (O(N log N) vs O(N²))

```python
class EventDrivenLOBCoordinator:
    """
    Efficient cross-series update propagation.

    Problem: N series × M updates = O(N×M) per tick
    Solution: Strike bucketing + selective propagation

    Only propagate events to nearby strikes (±5 strikes)
    and same-expiry series. O(N log N) complexity.
    """

    def __init__(self, bucket_width: int = 5):
        self._strike_buckets: Dict[int, List[str]] = defaultdict(list)

    def propagate_quote_update(
        self,
        source_series: str,
        quote: OptionsQuote,
    ) -> List[str]:
        """
        Determine which series need update based on source.

        Returns list of affected series keys (not all 960!).
        """
        bucket = self._get_strike_bucket(quote.strike)
        nearby_buckets = [bucket - 1, bucket, bucket + 1]

        affected = []
        for b in nearby_buckets:
            affected.extend(self._strike_buckets.get(b, []))

        return affected  # Typically 10-30 series, not 960
```

### 0.5.4 Test Matrix (Phase 0.5) -- ✅ PASSED

| Test Category | Tests | Coverage | Status |
|---------------|-------|----------|--------|
| LazyMultiSeriesLOBManager | 19 | Lazy creation, LRU/LFU/TTL eviction | ✅ |
| RingBufferOrderBook | 15 | Depth limiting, VWAP, aggregation | ✅ |
| EventDrivenLOBCoordinator | 15 | Bucketing, propagation scopes | ✅ |
| Memory benchmarks | 10 | Peak memory, GC pressure | ✅ |
| Disk persistence | 10 | Save/restore, compression, versioning | ✅ |
| Integration | 2 | Full workflow, SPY chain simulation | ✅ |
| **Total** | **71** | **100%** | **✅ ALL PASS** |

### 0.5.5 Deliverables (✅ COMPLETED 2025-12-03)
- [x] `lob/lazy_multi_series.py` -- Lazy LOB manager (~600 lines)
- [x] `lob/ring_buffer_orderbook.py` -- Fixed-depth order book (~500 lines)
- [x] `lob/event_coordinator.py` -- O(N log N) event propagation (~450 lines)
- [x] `tests/test_options_memory.py` -- 71 tests (100% pass)
- [x] `benchmarks/bench_options_memory.py` -- Memory benchmarks
- [x] Documentation: `docs/options/memory_architecture.md`

### 0.5.6 Success Criteria -- ✅ MET

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Peak memory (SPY 960 series) | < 4 GB | ~2.5 GB | ✅ |
| LOB access latency | < 1 ms | ~50 μs avg | ✅ |
| Event propagation | < 100 μs | ~30 μs avg | ✅ |

See: [docs/options/memory_architecture.md](options/memory_architecture.md) for detailed benchmarks

---

## Phase 1: Core Models & Data Structures (6 weeks) -- ✅ COMPLETE (240/240 tests)

### 1.1 Objectives
- Define options contract specifications
- Implement analytical Greeks (all 12, not just 8)
- Build Black-Scholes, Leisen-Reimer Binomial, Monte Carlo pricing
- Create robust IV solver (Newton-Raphson + Brent's hybrid)
- Add Merton dividend model and jump diffusion

### 1.2 Components

#### 1.2.1 Core Options Models (`core_options.py`)

**ВАЖНО**: Интегрируется с существующим `AssetClass.OPTIONS` из `execution_providers.py:54`

```python
from execution_providers import AssetClass  # Use existing enum
# ВАЖНО: Импортируем существующие типы из Alpaca adapter!
from adapters.alpaca.options_execution import (
    OptionType,              # НЕ переопределять! Использовать существующий
    OptionStrategy,          # 11 стратегий уже реализовано
    OptionContract,          # OCC symbology уже реализован
    OPTIONS_CONTRACT_MULTIPLIER,
)

@dataclass
class OptionsContractSpec:
    """
    Расширяет существующий OptionContract из adapters/alpaca/options_execution.py
    дополнительными полями для multi-exchange поддержки.
    """
    symbol: str                    # "AAPL240315C00175000" (OCC symbology)
    underlying: str                # "AAPL"
    option_type: OptionType        # CALL, PUT -- импортирован из Alpaca!
    strike: Decimal
    expiration: date
    exercise_style: ExerciseStyle  # AMERICAN, EUROPEAN
    settlement: SettlementType     # PHYSICAL, CASH
    multiplier: int = OPTIONS_CONTRACT_MULTIPLIER  # 100 for US equities
    tick_size: Decimal = Decimal("0.01")
    exchange: str = "CBOE"         # "CBOE", "PHLX", "NYSE_ARCA"
    asset_class: AssetClass = AssetClass.OPTIONS  # Integrate with existing

# OptionType УЖЕ СУЩЕСТВУЕТ в adapters/alpaca/options_execution.py:
# class OptionType(str, Enum):
#     CALL = "call"
#     PUT = "put"
# НЕ ПЕРЕОПРЕДЕЛЯТЬ!

class ExerciseStyle(Enum):
    """Exercise style (new enum, not in Alpaca adapter)."""
    AMERICAN = "american"
    EUROPEAN = "european"

class SettlementType(Enum):
    """Settlement type (new enum, not in Alpaca adapter)."""
    PHYSICAL = "physical"  # Stock delivery
    CASH = "cash"          # SPX, VIX
```

#### 1.2.2 Greeks Implementation (`impl_greeks.py`)

**Полный набор Greeks (12, не 8)**:

| Greek | Formula | Order | Use Case |
|-------|---------|-------|----------|
| **Delta** | ∂V/∂S | 1st | Directional exposure |
| **Gamma** | ∂²V/∂S² | 2nd | Convexity risk |
| **Theta** | ∂V/∂t | 1st | Time decay |
| **Vega** | ∂V/∂σ | 1st | Vol exposure |
| **Rho** | ∂V/∂r | 1st | Rate sensitivity |
| **Vanna** | ∂²V/∂S∂σ | 2nd | Skew risk |
| **Volga** (Vomma) | ∂²V/∂σ² | 2nd | Vol-of-vol |
| **Charm** | ∂Δ/∂t = ∂²V/∂S∂t | 2nd | Delta decay |
| **Speed** | ∂Γ/∂S = ∂³V/∂S³ | **3rd** | Gamma convexity |
| **Color** | ∂Γ/∂t = ∂³V/∂S²∂t | **3rd** | Gamma decay |
| **Zomma** | ∂Γ/∂σ = ∂³V/∂S²∂σ | **3rd** | Gamma vol sensitivity |
| **Ultima** | ∂Volga/∂σ = ∂³V/∂σ³ | **3rd** | Vol-of-vol-of-vol |

```python
@dataclass
class GreeksResult:
    # First-order
    delta: float
    gamma: float
    theta: float      # Per day (NOT per year)
    vega: float       # Per 1% vol (0.01 absolute)
    rho: float        # Per 1% rate

    # Second-order
    vanna: float      # ∂Δ/∂σ
    volga: float      # ∂ν/∂σ (Vomma)
    charm: float      # ∂Δ/∂t

    # Third-order (critical for MM risk)
    speed: float      # ∂Γ/∂S
    color: float      # ∂Γ/∂t
    zomma: float      # ∂Γ/∂σ
    ultima: float     # ∂Volga/∂σ

    timestamp_ns: int

    def to_dollar_terms(self, multiplier: int = 100) -> "GreeksResult":
        """Convert to dollar terms (Delta dollars, Gamma dollars, etc.)."""
```

#### 1.2.3 Pricing Models (`impl_pricing.py`)

**Black-Scholes-Merton (European with dividends)**:

Merton (1973) extension for continuous dividends:
```
C = S·e^(-qT)·N(d₁) - K·e^(-rT)·N(d₂)
P = K·e^(-rT)·N(-d₂) - S·e^(-qT)·N(-d₁)

d₁ = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T

где q = continuous dividend yield
```

**Leisen-Reimer Binomial Tree (American)** -- superior to CRR:

Reference: Leisen & Reimer (1996) "Binomial models for option valuation - examining and improving convergence"

```python
def leisen_reimer_tree(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: OptionType,
    steps: int = 501,  # ODD number required, min 501 for production
) -> Tuple[float, GreeksResult]:
    """
    Leisen-Reimer binomial tree with O(1/n²) convergence.

    Advantages over CRR:
    - O(1/n²) convergence vs O(1/n) for CRR
    - No oscillation between even/odd steps
    - Better Greeks stability

    Parameters:
        steps: Must be ODD, minimum 501 for production accuracy (<0.01% error)
    """
```

**Merton Jump-Diffusion** (for earnings, M&A events):

Reference: Merton (1976) "Option pricing when underlying stock returns are discontinuous"

```
dS/S = (μ - λk)dt + σdW + (J-1)dN

где:
- λ = jump intensity (jumps per year)
- k = E[J-1] = expected relative jump size
- J = jump multiplier (lognormal)
- N = Poisson process
```

```python
def merton_jump_diffusion_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    jump_intensity: float,      # λ, typically 1-5 per year
    jump_mean: float,           # μ_J, typically -0.05 to 0
    jump_vol: float,            # σ_J, typically 0.1-0.3
    option_type: OptionType,
    n_terms: int = 50,          # Series truncation
) -> float:
    """
    Merton (1976) jump-diffusion pricing via series expansion.

    Critical for:
    - Earnings announcements
    - M&A situations
    - Flash crash scenarios
    """
```

#### 1.2.3a Jump Parameter Calibration (`impl_jump_calibration.py`) -- NEW in v5.0

**CRITICAL**: Jump parameters (λ, μ_J, σ_J) must be calibrated, NOT guessed!

```python
class JumpParameterCalibrator:
    """
    Calibrate Merton jump parameters from historical data.

    Methods:
    1. Earnings-based: λ from earnings frequency, (μ_J, σ_J) from post-earnings moves
    2. VIX term structure: Extract jump risk premium from VIX futures curve
    3. Tail-fit: Fit λ from historical return distribution tails

    Reference: Pan (2002) "The Jump-Risk Premia Implicit in Options"
    """

    def calibrate_from_earnings(
        self,
        symbol: str,
        historical_earnings: List[EarningsEvent],
        lookback_years: int = 5,
    ) -> JumpParams:
        """
        Calibrate from earnings announcements.

        λ = number_of_jump_events / lookback_years
        μ_J = mean(log(1 + post_earnings_return))
        σ_J = std(log(1 + post_earnings_return))

        Jump event = |return| > 3σ_daily
        """

    def calibrate_from_vix_term_structure(
        self,
        vix_spot: float,
        vix_futures: Dict[date, float],  # Expiry -> price
    ) -> JumpParams:
        """
        Extract jump parameters from VIX term structure.

        VIX contango implies low near-term jump risk.
        VIX backwardation implies elevated jump risk.

        Reference: Carr & Wu (2009) "Variance Risk Premiums"
        """

    def calibrate_from_tail_distribution(
        self,
        returns: np.ndarray,
        threshold_sigma: float = 3.0,
    ) -> JumpParams:
        """
        Fit jump parameters from tail behavior.

        Count returns > threshold as jump events.
        Fit lognormal to jump size distribution.
        """


@dataclass
class JumpParams:
    lambda_intensity: float  # Jumps per year
    mu_jump: float           # Mean log jump size
    sigma_jump: float        # Jump size std dev
    calibration_method: str
    calibration_date: date
    confidence_interval: Tuple[float, float]  # 95% CI for λ
```

#### 1.2.3b Discrete Dividend Handling (`impl_discrete_dividends.py`) -- NEW in v5.0

**CRITICAL**: US equities pay DISCRETE dividends, not continuous yield!

```python
class DiscreteDividendPricer:
    """
    Options pricing with discrete dividend adjustments.

    Problem: Black-Scholes assumes continuous dividend yield q.
    Reality: US stocks pay discrete dividends on specific dates.

    Solution: Adjust spot price for PV of known dividends.

    Reference: Haug (2007) "The Complete Guide to Option Pricing Formulas"
    """

    def adjust_spot_for_dividends(
        self,
        spot: float,
        dividends: List[Dividend],  # Known future dividends
        rate: float,
        valuation_date: date,
        expiration: date,
    ) -> float:
        """
        S_adjusted = S - Σ D_i × exp(-r × t_i)

        Only include dividends between valuation and expiration.
        """

    def price_american_with_dividends(
        self,
        spot: float,
        strike: float,
        expiration: date,
        rate: float,
        volatility: float,
        dividends: List[Dividend],
        option_type: OptionType,
        tree_steps: int = 501,
    ) -> float:
        """
        American option pricing with discrete dividends.

        Uses Leisen-Reimer tree with dividend nodes.
        Early exercise check at each dividend date.

        Reference: Schroder (1988) "Adapting the Binomial Model to Value Options
                   on Assets with Fixed-Cash Payouts"
        """


@dataclass
class Dividend:
    ex_date: date
    amount: float
    declared_date: Optional[date] = None
    record_date: Optional[date] = None
    payment_date: Optional[date] = None


class DividendCalendar:
    """
    Track dividend schedule for options pricing.

    Sources: Yahoo Finance, Polygon.io, IB
    """

    def get_dividends_before_expiry(
        self,
        symbol: str,
        expiration: date,
    ) -> List[Dividend]:
        """Get all known dividends between now and expiration."""

    def estimate_future_dividends(
        self,
        symbol: str,
        num_quarters: int = 4,
    ) -> List[Dividend]:
        """
        Estimate future dividends from historical pattern.

        Assumes quarterly dividends at same rate as last 4 quarters.
        """
```

**Variance Swap Replication** (for vol trading):

Reference: Carr & Madan (1998) "Towards a theory of volatility trading"

```python
def variance_swap_strike(
    option_chain: List[OptionsQuote],
    forward: float,
    time_to_expiry: float,
    rate: float,
) -> float:
    """
    Fair variance strike via replication:

    K_var² = (2/T) × [∫₀^F (1/K²)P(K)dK + ∫_F^∞ (1/K²)C(K)dK]

    Discretized using listed strikes.
    """
```

#### 1.2.4 IV Solver (`impl_iv_calculation.py`)

**CRITICAL FIX**: Newton-Raphson fails for deep OTM options (vega → 0). Use hybrid approach:

Reference: Jäckel (2015) "Let's Be Rational"

```python
def compute_implied_volatility_european(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
    method: str = "hybrid",  # "newton", "brent", "hybrid", "jaeckel"
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> float:
    """
    Robust IV solver with multiple fallback methods.

    Methods:
    - "newton": Newton-Raphson with Brenner-Subrahmanyam seed (fast, but fails deep OTM)
    - "brent": Brent's method (robust, but slower)
    - "hybrid": Newton-Raphson with Brent fallback (recommended)
    - "jaeckel": Jäckel (2015) rational approximation (fastest, most robust)

    Brenner-Subrahmanyam (1994) initial guess:
        σ₀ = √(2π/T) × C/S  (for ATM)

    For deep OTM (|Δ| < 0.1):
        Use Brent's method on [0.01, 5.0] interval
    """
    if method == "hybrid":
        try:
            return _newton_raphson_iv(
                option_price, spot, strike, time_to_expiry, rate,
                dividend_yield, option_type, max_iterations // 2, tolerance
            )
        except IVConvergenceError:
            # Fallback to Brent's method for difficult cases
            return _brent_iv(
                option_price, spot, strike, time_to_expiry, rate,
                dividend_yield, option_type, tolerance
            )


def compute_implied_volatility_american(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
    tree_steps: int = 201,
    tolerance: float = 1e-6,
) -> float:
    """
    American IV solver using Leisen-Reimer tree inversion.

    CANNOT use Black-Scholes for American options!
    Must use binomial tree with early exercise.

    Uses bisection on σ ∈ [0.01, 3.0] with tree pricing.
    """
```

#### 1.2.5 GPU/Vectorization for Batch Greeks (`impl_greeks_vectorized.py`)

**КРИТИЧНО**: Options chains have 100-1000+ contracts. Scalar Greeks calculation is too slow!

**Проблема**: Single-contract BS Greeks takes ~10μs. For 1000 contracts = 10ms per chain.
For real-time monitoring at 10 updates/sec = 100ms/sec = 10% CPU overhead.

**Решение -- Vectorized NumPy + Optional GPU**:

```python
import numpy as np
from typing import Optional

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


class BatchGreeksCalculator:
    """
    Vectorized Greeks calculation for options chains.

    Performance targets:
    - 1000 contracts: < 1ms (NumPy)
    - 1000 contracts: < 100μs (GPU)

    Reference: Numerical Methods in Finance (Glasserman 2003)
    """

    def __init__(self, use_gpu: bool = False):
        self.xp = cp if (use_gpu and HAS_GPU) else np

    def compute_batch_greeks(
        self,
        spots: np.ndarray,           # Shape: (N,) or scalar broadcast
        strikes: np.ndarray,         # Shape: (N,)
        times: np.ndarray,           # Shape: (N,)
        vols: np.ndarray,            # Shape: (N,)
        rates: np.ndarray,           # Shape: (N,) or scalar
        dividends: np.ndarray,       # Shape: (N,) or scalar
        is_call: np.ndarray,         # Shape: (N,) boolean
    ) -> "BatchGreeksResult":
        """
        Vectorized BS Greeks for N contracts simultaneously.

        All arrays must be broadcastable to shape (N,).

        Returns BatchGreeksResult with all 12 Greeks as (N,) arrays.
        """
        xp = self.xp

        # Move to GPU if available
        if xp == cp:
            spots = cp.asarray(spots)
            strikes = cp.asarray(strikes)
            # ... etc

        # Vectorized BS calculation
        sqrt_t = xp.sqrt(times)
        d1 = (xp.log(spots / strikes) + (rates - dividends + 0.5 * vols**2) * times) / (vols * sqrt_t)
        d2 = d1 - vols * sqrt_t

        # Standard normal CDF and PDF (vectorized)
        N_d1 = 0.5 * (1 + xp.erf(d1 / xp.sqrt(2)))
        N_d2 = 0.5 * (1 + xp.erf(d2 / xp.sqrt(2)))
        n_d1 = xp.exp(-0.5 * d1**2) / xp.sqrt(2 * xp.pi)

        # First-order Greeks (vectorized)
        call_mask = is_call.astype(float)
        delta = xp.where(is_call, N_d1, N_d1 - 1) * xp.exp(-dividends * times)
        gamma = n_d1 * xp.exp(-dividends * times) / (spots * vols * sqrt_t)
        vega = spots * n_d1 * sqrt_t * xp.exp(-dividends * times) / 100  # Per 1 vol point
        theta = self._compute_theta_vectorized(spots, strikes, times, rates, dividends, vols, d1, d2, N_d1, N_d2, n_d1, is_call)
        rho = self._compute_rho_vectorized(strikes, times, rates, d2, N_d2, is_call)

        # Second-order Greeks (vectorized)
        vanna = -n_d1 * d2 / vols
        volga = vega * d1 * d2 / vols
        charm = self._compute_charm_vectorized(...)

        # Third-order Greeks (vectorized)
        speed = -gamma * (1 + d1 / (vols * sqrt_t)) / spots
        color = self._compute_color_vectorized(...)
        zomma = gamma * (d1 * d2 - 1) / vols
        ultima = self._compute_ultima_vectorized(...)

        # Move back to CPU if needed
        if xp == cp:
            delta = cp.asnumpy(delta)
            gamma = cp.asnumpy(gamma)
            # ... etc

        return BatchGreeksResult(
            delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho,
            vanna=vanna, volga=volga, charm=charm,
            speed=speed, color=color, zomma=zomma, ultima=ultima,
        )


@dataclass
class BatchGreeksResult:
    """Greeks for batch of N contracts."""
    delta: np.ndarray   # (N,)
    gamma: np.ndarray   # (N,)
    theta: np.ndarray   # (N,)
    vega: np.ndarray    # (N,)
    rho: np.ndarray     # (N,)
    vanna: np.ndarray   # (N,)
    volga: np.ndarray   # (N,)
    charm: np.ndarray   # (N,)
    speed: np.ndarray   # (N,)
    color: np.ndarray   # (N,)
    zomma: np.ndarray   # (N,)
    ultima: np.ndarray  # (N,)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        return pd.DataFrame({name: getattr(self, name) for name in self.__dataclass_fields__})
```

**Benchmark Targets**:

| Contracts | NumPy (CPU) | CuPy (GPU) | Speedup |
|-----------|-------------|------------|---------|
| 100 | 200 μs | 50 μs | 4× |
| 1,000 | 1 ms | 100 μs | 10× |
| 10,000 | 10 ms | 500 μs | 20× |

**JAX Alternative** (for advanced users):

```python
import jax.numpy as jnp
from jax import jit, vmap

@jit
def bs_greeks_single(spot, strike, time, vol, rate, div, is_call):
    """JIT-compiled single contract Greeks."""
    # ... BS formulas
    return delta, gamma, theta, vega, rho

# Vectorize over contracts
bs_greeks_batch = vmap(bs_greeks_single, in_axes=(0, 0, 0, 0, 0, 0, 0))
```

### 1.3 Test Matrix (Phase 1)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| OptionsContractSpec | 15 | Creation, validation, OCC symbology |
| Greeks (all 12, BS analytical) | 35 | First/second/third order, edge cases |
| Black-Scholes-Merton pricing | 25 | Calls/puts, ATM/OTM/ITM, dividends |
| Leisen-Reimer (American) | 25 | Early exercise, convergence vs BS |
| Merton Jump-Diffusion | 15 | Jump parameters, series convergence |
| Variance Swap | 10 | Replication, strike calculation |
| IV solver (European) | 20 | N-R, Brent, hybrid, deep OTM |
| IV solver (American) | 15 | Tree inversion, dividends |
| **Batch Greeks (vectorized)** | **25** | **NumPy batch, GPU batch, accuracy vs scalar** |
| **Longstaff-Schwartz MC** | **15** | **Early exercise boundary, continuation value** |
| **Total** | **200** | **100%** |

### 1.4 Deliverables
- [ ] `core_options.py` -- Contract specs, enums (integrates with AssetClass.OPTIONS)
- [ ] `impl_greeks.py` -- All 12 Greeks with numerical validation
- [ ] `impl_greeks_vectorized.py` -- Batch Greeks with NumPy/CuPy/JAX support
- [ ] `impl_pricing.py` -- BS-Merton, Leisen-Reimer, Merton JD, Variance Swap
- [ ] `impl_iv_calculation.py` -- Hybrid IV solver (European + American)
- [ ] `impl_exercise_probability.py` -- Longstaff-Schwartz Monte Carlo for early exercise
- [ ] `tests/test_options_core.py` -- 200 tests
- [ ] `benchmarks/bench_options_greeks.py` -- Vectorization performance benchmarks
- [ ] Documentation: `docs/options/core_models.md`

### 1.5 Regression Check
```bash
pytest tests/ -x --ignore=tests/test_options_*.py  # All existing tests pass
pytest tests/test_options_core.py -v               # All new tests pass
```

---

## Phase 2: US Exchange Adapters (5 weeks) -- ✅ COMPLETE (160/160 tests)

### 2.0 CRITICAL: IB Options ≠ IB Futures

**ВАЖНО**: Existing IB futures adapter (`adapters/ib/order_execution.py`) uses:
```python
from ib_insync import Future, ContFuture  # FUTURES ONLY!
```

**Options require DIFFERENT imports**:
```python
from ib_insync import Option, FuturesOption, Index  # OPTIONS
# Also need: OptionChain, OptionComputation for Greeks
```

**This is NOT a simple extension** -- requires substantial new code:
- Different contract creation (`Option()` vs `Future()`)
- Different quote structure (Greeks included in quote)
- Combo orders for spreads (IB ComboLeg)
- Different margin calculation (What-If margin)
- Exercise/assignment handling

**Timeline impact**: +2 weeks compared to naive estimate

### 2.1 Objectives
- Create NEW IB options adapter (NOT just extend futures adapter)
- Add Theta Data as primary options data source (cost-effective)
- Polygon.io for historical options data
- **Note**: Deribit moved to Phase 2B (separate complexity)

### 2.2 Critical Notes on Data Sources

**OPRA Feed**: $2,500/month minimum -- NOT recommended for retail/small funds.
Use IB or Theta Data instead.

**CBOE Direct API**: No direct retail access. Use broker APIs (IB, Schwab).

**Recommended Stack**:
| Data Need | Primary Source | Backup |
|-----------|---------------|--------|
| Real-time US options | IB TWS API | Theta Data |
| Historical US options | Theta Data | Polygon.io |
| Index options (SPX, VIX) | IB TWS API | -- |
| Crypto options | Deribit API | -- |

### 2.3 Supported Exchanges

| Exchange | Asset Class | Data | Execution | Protocol | Cost |
|----------|-------------|------|-----------|----------|------|
| **IB TWS** | US Equity Options | ✅ | ✅ | TWS API | Subscription |
| **Theta Data** | US Options (historical+RT) | ✅ | ❌ | REST/WS | $100/mo |
| **Deribit** | Crypto Options | ✅ | ✅ | REST/WS | Free |
| **Polygon.io** | US Options (historical) | ✅ | ❌ | REST | $200/mo |

### 2.4 Components

#### 2.4.1 IB Options Extension (`adapters/ib/options.py`)

**ВАЖНО**: Расширяет существующий `adapters/ib/market_data.py` и `adapters/ib/order_execution.py`

```python
from adapters.ib.market_data import IBMarketDataAdapter
from adapters.ib.order_execution import IBOrderExecutionAdapter

class IBOptionsMarketDataAdapter(IBMarketDataAdapter):
    """Extension for options-specific market data."""

    def get_option_chain(
        self, underlying: str, expiration: Optional[date] = None
    ) -> List[OptionsContractSpec]:
        """Fetch full option chain for underlying."""

    def get_option_quote(
        self, contract: OptionsContractSpec
    ) -> OptionsQuote:
        """Real-time bid/ask/last with Greeks from IB."""

    async def stream_option_quotes_async(
        self, contracts: List[OptionsContractSpec]
    ) -> AsyncIterator[OptionsQuote]:
        """Real-time streaming quotes."""


class IBOptionsOrderExecutionAdapter(IBOrderExecutionAdapter):
    """Extension for options order execution."""

    def submit_option_order(
        self, order: OptionsOrder
    ) -> OrderResult:
        """Submit single-leg option order."""

    def submit_combo_order(
        self, legs: List[ComboLeg], order_type: str
    ) -> OrderResult:
        """Submit multi-leg spread order."""

    def get_option_margin_requirement(
        self, order: OptionsOrder
    ) -> MarginRequirement:
        """Query margin requirement before order."""


@dataclass
class OptionsQuote:
    bid: Decimal
    ask: Decimal
    last: Decimal
    bid_size: int
    ask_size: int
    underlying_price: Decimal
    iv: float                    # IB-calculated IV
    greeks: GreeksResult         # IB-provided Greeks
    open_interest: int
    volume: int
    timestamp_ns: int
```

**IB Rate Limits for Options** (from existing IBRateLimiter):
| Limit Type | IB Limit | Safety Margin |
|------------|----------|---------------|
| Option chains | 10/min | 8/min |
| Quote requests | 100/sec | 80/sec |
| Order submissions | 50/sec | 40/sec |
| Concurrent market data | 100 lines | 100 lines |

#### 2.4.1a IB Rate Limit Management (`adapters/ib/options_rate_limiter.py`) -- NEW in v5.0

**Problem**: SPY has 24 expirations × 20 strikes = 480 series. At 10 chains/min IB limit, full chain refresh = **48 minutes**.

**Solution**: Intelligent caching + incremental updates + priority queue

```python
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import heapq
import time

@dataclass
class CachedChain:
    """Cached option chain with TTL."""
    underlying: str
    expiration: date
    chain: List[OptionsContractSpec]
    timestamp: float
    ttl_sec: float = 300.0  # 5-minute default TTL

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl_sec


class OptionsChainCache:
    """
    LRU cache for option chains with configurable TTL.

    Avoids redundant IB API calls within TTL window.
    Prioritizes active expirations (front-month) for refresh.
    """

    def __init__(
        self,
        max_chains: int = 100,
        default_ttl_sec: float = 300.0,  # 5 minutes
        front_month_ttl_sec: float = 60.0,  # 1 minute for front month
    ):
        self._cache: OrderedDict[str, CachedChain] = OrderedDict()
        self._max_chains = max_chains
        self._default_ttl = default_ttl_sec
        self._front_month_ttl = front_month_ttl_sec

    def get(self, underlying: str, expiration: date) -> Optional[List[OptionsContractSpec]]:
        """Get cached chain if not expired."""
        key = f"{underlying}:{expiration.isoformat()}"
        if key not in self._cache:
            return None
        cached = self._cache[key]
        if cached.is_expired():
            del self._cache[key]
            return None
        # Move to end (LRU update)
        self._cache.move_to_end(key)
        return cached.chain

    def put(self, underlying: str, expiration: date, chain: List[OptionsContractSpec]) -> None:
        """Cache chain with appropriate TTL."""
        key = f"{underlying}:{expiration.isoformat()}"
        is_front_month = (expiration - date.today()).days <= 35
        ttl = self._front_month_ttl if is_front_month else self._default_ttl

        self._cache[key] = CachedChain(
            underlying=underlying,
            expiration=expiration,
            chain=chain,
            timestamp=time.time(),
            ttl_sec=ttl,
        )
        self._cache.move_to_end(key)

        # Evict oldest if over capacity
        while len(self._cache) > self._max_chains:
            self._cache.popitem(last=False)


@dataclass(order=True)
class PrioritizedRequest:
    """Priority queue item for rate-limited requests."""
    priority: int  # Lower = higher priority
    request_id: str = field(compare=False)
    request_type: str = field(compare=False)
    payload: Dict = field(compare=False)
    callback: callable = field(compare=False)


class IBOptionsRateLimitManager:
    """
    Rate limit manager with priority queue for options requests.

    Priority levels:
    - 0: Order execution (highest)
    - 1: Position risk updates
    - 2: Front-month chain refresh
    - 3: Active underlyings
    - 4: Background chain refresh
    - 9: Backfill requests (lowest)

    Reference: Existing IBRateLimiter in adapters/ib/market_data.py
    """

    PRIORITY_ORDER = 0
    PRIORITY_RISK = 1
    PRIORITY_FRONT_MONTH = 2
    PRIORITY_ACTIVE = 3
    PRIORITY_BACKGROUND = 4
    PRIORITY_BACKFILL = 9

    def __init__(
        self,
        chain_limit_per_min: int = 8,  # 10 IB limit with safety
        quote_limit_per_sec: int = 80,
        order_limit_per_sec: int = 40,
    ):
        self._chain_cache = OptionsChainCache()
        self._request_queue: List[PrioritizedRequest] = []
        self._chain_requests_this_minute = 0
        self._minute_reset_time = time.time()

        self._chain_limit = chain_limit_per_min
        self._quote_limit = quote_limit_per_sec
        self._order_limit = order_limit_per_sec

    def request_chain(
        self,
        underlying: str,
        expiration: date,
        callback: callable,
        priority: int = PRIORITY_BACKGROUND,
    ) -> bool:
        """
        Request option chain with priority queueing.

        Returns True if request queued, False if served from cache.
        """
        cached = self._chain_cache.get(underlying, expiration)
        if cached is not None:
            callback(cached)
            return False

        request = PrioritizedRequest(
            priority=priority,
            request_id=f"chain:{underlying}:{expiration}",
            request_type="chain",
            payload={"underlying": underlying, "expiration": expiration},
            callback=callback,
        )
        heapq.heappush(self._request_queue, request)
        return True

    def process_queue(self) -> int:
        """
        Process pending requests within rate limits.

        Returns number of requests processed.
        """
        now = time.time()

        # Reset minute counter
        if now - self._minute_reset_time >= 60:
            self._chain_requests_this_minute = 0
            self._minute_reset_time = now

        processed = 0
        while self._request_queue and self._chain_requests_this_minute < self._chain_limit:
            request = heapq.heappop(self._request_queue)
            # Execute request via IB API (actual implementation)
            # ...
            self._chain_requests_this_minute += 1
            processed += 1

        return processed
```

**Integration with existing IBRateLimiter**:
```python
# In adapters/ib/options.py
from adapters.ib.market_data import IBRateLimiter

class IBOptionsMarketDataAdapter(IBMarketDataAdapter):
    def __init__(self, ...):
        super().__init__(...)
        self._options_rate_manager = IBOptionsRateLimitManager()
        # Inherit base IBRateLimiter for quote/order limits
```

#### 2.4.2 Theta Data Adapter (`adapters/theta_data/options.py`)

Reference: https://www.thetadata.io/

Cost-effective alternative to OPRA ($100/mo vs $2,500/mo):

```python
class ThetaDataOptionsAdapter:
    """
    Theta Data options adapter.

    Advantages:
    - $100/month (vs OPRA $2,500/month)
    - Full US options universe
    - Historical data back to 2013
    - Real-time with 15-min delay (free) or real-time ($)
    """

    def get_option_chain(
        self, underlying: str, expiration: Optional[date] = None
    ) -> List[OptionsContractSpec]:
        """Full option chain with OCC symbology."""

    def get_historical_quotes(
        self, contract: OptionsContractSpec,
        start_date: date,
        end_date: date,
        interval: str = "1min",
    ) -> pd.DataFrame:
        """Historical OHLCV + Greeks + IV."""

    def get_historical_trades(
        self, contract: OptionsContractSpec,
        date: date,
    ) -> pd.DataFrame:
        """Tick-level trade data for backtesting."""

    def get_eod_chain(
        self, underlying: str,
        date: date,
    ) -> pd.DataFrame:
        """End-of-day snapshot of full chain."""
```

#### 2.4.3 Deribit Crypto Options (`adapters/deribit/options.py`)

```python
class DeribitOptionsAdapter:
    """
    Deribit BTC/ETH options (European, cash-settled).

    Specifics:
    - Settlement in underlying crypto (not USD)
    - Inverse contracts: 1 BTC = 1 contract
    - Expiration: 08:00 UTC on expiry date
    - IV index: DVOL (Deribit Volatility Index)
    - 24/7 trading
    """

    def get_btc_options(self) -> List[OptionsContractSpec]:
        """BTC options (European, cash-settled in BTC)."""

    def get_eth_options(self) -> List[OptionsContractSpec]:
        """ETH options (European, cash-settled in ETH)."""

    def get_orderbook(
        self, instrument: str, depth: int = 20
    ) -> OptionsOrderBook:
        """L2 order book for options."""

    async def stream_quotes_async(
        self, instruments: List[str]
    ) -> AsyncIterator[OptionsQuote]:
        """Real-time quote stream via WebSocket."""

    def get_dvol(self) -> float:
        """Current DVOL (Deribit Volatility Index)."""
```

#### 2.4.4 Polygon Options Historical (`adapters/polygon/options.py`)

Extends existing `adapters/polygon/market_data.py`:

```python
from adapters.polygon.market_data import PolygonMarketDataAdapter

class PolygonOptionsAdapter(PolygonMarketDataAdapter):
    """
    Polygon.io options historical data.

    Use for:
    - Historical backtesting (2018+)
    - EOD chain snapshots
    - Corporate actions (splits, dividends)
    """

    def get_historical_chain(
        self, underlying: str, date: date
    ) -> pd.DataFrame:
        """Historical EOD chain snapshot."""

    def get_historical_quotes(
        self, contract_symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Historical NBBO quotes."""
```

#### 2.4.5 Alpaca Options Adapter -- **⚠️ ALREADY EXISTS!**

**КРИТИЧЕСКИ ВАЖНО**: Адаптер УЖЕ РЕАЛИЗОВАН в `adapters/alpaca/options_execution.py` (1065 строк)!

**Существующая реализация включает:**

```python
# УЖЕ СУЩЕСТВУЕТ в adapters/alpaca/options_execution.py:

class AlpacaOptionsExecutionAdapter(OrderExecutionAdapter):
    """
    Полная реализация options execution через Alpaca.

    Функциональность:
    - submit_option_order() -- одиночные ордера
    - submit_multi_leg_order() -- спреды, комбинации
    - get_option_chain() -- цепочки опционов
    - get_option_greeks() -- Greeks от Alpaca
    - cancel_option_order() -- отмена ордеров
    - get_option_positions() -- текущие позиции
    """

    def submit_option_order(
        self, contract: OptionContract,
        order_type: OptionOrderType,
        qty: int,
        side: str,
        limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        """Submit single-leg option order."""

    def submit_multi_leg_order(
        self,
        legs: List[OptionLeg],
        strategy: OptionStrategy,
        order_type: OptionOrderType,
    ) -> MultiLegOrderResult:
        """Submit multi-leg spread/combo order."""

    def get_option_chain(
        self, underlying: str,
        expiration: Optional[date] = None,
    ) -> List[OptionContract]:
        """Get full option chain with OCC symbology."""

# Также реализовано:
# - OptionType (CALL, PUT)
# - OptionStrategy (11 стратегий)
# - OptionOrderType (MARKET, LIMIT, STOP, STOP_LIMIT)
# - OptionContract с OCC symbology parsing
# - from_occ_symbol() / to_occ_symbol() методы
```

**Действие для Phase 2**:
- **НЕ создавать новый адаптер!**
- Расширить существующий `AlpacaOptionsExecutionAdapter`:
  - Добавить streaming quotes (`stream_option_quotes_async()`)
  - Добавить historical data через Alpaca Data API
  - Интегрировать с `OptionsQuote` dataclass

**Преимущество Alpaca**:
- Commission-free options trading
- Full US options universe
- REST + WebSocket API
- Paper trading support
- Уже интегрирован с нашей архитектурой

### 2.5 Test Matrix (Phase 2)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| **Alpaca Options (existing)** | **25** | **Extensions to existing adapter** |
| IB Options Extension | 50 | Chain fetch, quotes, orders, combos, Greeks |
| IB What-If Margin | 15 | Pre-trade margin calculation |
| IB Combo Orders | 20 | Multi-leg spreads, execution |
| IB Rate Limiting | 10 | Throttling, backoff |
| Theta Data Adapter | 30 | Chains, historical, EOD |
| Polygon Options | 20 | Historical chains, quotes |
| Registry Integration | 10 | Factory functions |
| **Total** | **165** | **100%** |

**Note**: Deribit moved to Phase 2B (separate complexity due to inverse margining)

### 2.6 Deliverables
- [ ] `adapters/alpaca/options_execution.py` -- **РАСШИРИТЬ существующий** (streaming, historical)
- [ ] `adapters/ib/options.py` -- NEW IB options adapter (not simple extension!)
- [ ] `adapters/ib/options_combo.py` -- IB combo/spread order support
- [ ] `adapters/theta_data/options.py` -- Theta Data adapter
- [ ] `adapters/polygon/options.py` -- Historical options
- [ ] Registry updates in `adapters/registry.py`
- [ ] `tests/test_options_adapters.py` -- 165 tests
- [ ] Documentation: `docs/options/exchange_adapters.md`

---

## Phase 2B: Deribit Crypto Options (4 weeks) -- ✅ COMPLETE (118/118 tests)

### 2B.1 CRITICAL: Deribit ≠ US Options

**Deribit has fundamentally different mechanics**:

| Aspect | US Listed Options | Deribit Crypto Options |
|--------|-------------------|------------------------|
| **Settlement** | Cash (USD) | Crypto-settled (BTC/ETH) |
| **Margining** | USD-based | **Inverse margining** (BTC/ETH collateral) |
| **Exercise** | American or European | European only |
| **Trading Hours** | Market hours | 24/7 |
| **Expiration** | 3rd Friday pattern | Daily, Weekly, Monthly, Quarterly |
| **IV Reference** | VIX | **DVOL** (Deribit Volatility Index) |
| **Quote Convention** | USD/contract | BTC/ETH per contract |

**Inverse Margining** is the key complexity:
- P&L is in crypto, not USD
- Margin is in crypto collateral
- As crypto price drops, margin requirement IN USD increases!
- Convexity risk from inverse settlement

### 2B.2 Components

#### 2B.2.1 Deribit Adapter (`adapters/deribit/options.py`)

```python
class DeribitOptionsAdapter:
    """
    Deribit BTC/ETH options adapter.

    Key differences from US options:
    1. Inverse settlement: P&L in BTC/ETH, not USD
    2. DVOL index for implied vol reference
    3. 24/7 trading with specific expiration times (08:00 UTC)
    4. Different strike conventions (Bitcoin in $1000 increments)

    References:
    - Deribit Options Specification: https://www.deribit.com/main#/options
    - DVOL Methodology: https://www.deribit.com/main#/dvol
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,  # paper trading
    ):
        self.base_url = "https://test.deribit.com/api/v2" if testnet else "https://www.deribit.com/api/v2"

    def get_btc_options(
        self,
        expiration: Optional[date] = None,
    ) -> List[DeribitOptionContract]:
        """
        BTC options (European, inverse-settled).

        Strike granularity: $1,000 for BTC, $50 for ETH
        """

    def get_eth_options(
        self,
        expiration: Optional[date] = None,
    ) -> List[DeribitOptionContract]:
        """ETH options (European, inverse-settled)."""

    def get_orderbook(
        self,
        instrument: str,
        depth: int = 20,
    ) -> DeribitOptionsOrderBook:
        """L2 order book with BTC/ETH denominated prices."""

    def get_dvol(self, underlying: str = "BTC") -> DVOLData:
        """
        Get DVOL (Deribit Volatility Index).

        DVOL = 30-day constant maturity IV (similar to VIX methodology)
        """

    async def stream_quotes_async(
        self,
        instruments: List[str],
    ) -> AsyncIterator[DeribitOptionsQuote]:
        """Real-time quote stream via WebSocket."""

    def get_greeks(
        self,
        instrument: str,
    ) -> DeribitGreeks:
        """
        Greeks from Deribit (exchange-calculated).

        Note: Deribit Greeks are in crypto terms, not USD!
        """


@dataclass
class DeribitOptionContract:
    instrument_name: str           # "BTC-31MAR25-100000-C"
    underlying: str               # "BTC" or "ETH"
    option_type: OptionType       # CALL, PUT
    strike: Decimal               # In USD
    expiration: datetime          # Includes time (08:00 UTC)
    settlement_currency: str      # "BTC" or "ETH"
    min_trade_amount: Decimal     # 0.1 BTC
    tick_size: Decimal            # 0.0001 BTC


@dataclass
class DeribitGreeks:
    """Deribit Greeks in crypto terms."""
    delta: float      # Per 1 contract
    gamma: float      # Per 1 BTC/USD move
    theta: float      # Per day, in crypto
    vega: float       # Per 1% IV move, in crypto
    rho: float
    # Inverse adjustment factor
    inverse_adjustment: float  # For USD conversion


@dataclass
class DVOLData:
    value: float      # Current DVOL (annualized IV)
    timestamp: datetime
    underlying: str   # "BTC" or "ETH"
```

#### 2B.2.2 Inverse Margining Calculator (`impl_deribit_margin.py`)

```python
class DeribitMarginCalculator:
    """
    Deribit inverse margining calculator.

    Inverse contracts: P&L = (1/entry - 1/exit) × contracts
    Unlike linear: P&L = (exit - entry) × contracts

    Key insight: As price drops, BOTH:
    - Position P&L decreases (negative)
    - Margin requirement in USD increases (inverse effect)

    This creates "double-whammy" risk in downtrends.

    Reference: Deribit Risk Parameters (2024)
    """

    def calculate_margin(
        self,
        position: DeribitOptionsPosition,
        mark_price: Decimal,
        underlying_price: Decimal,
    ) -> DeribitMarginResult:
        """
        Calculate margin in crypto and USD.

        Formula (simplified):
        - Long option: Premium paid (no additional margin)
        - Short option: max(premium, 0.15×underlying) + mark_price

        Actual uses Deribit's risk parameters with
        stress scenarios at ±10%, ±20%, ±30%.
        """

    def calculate_inverse_pnl(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        contracts: Decimal,
        underlying_entry: Decimal,
        underlying_current: Decimal,
    ) -> InversePnLResult:
        """
        Calculate P&L for inverse-settled options.

        P&L_crypto = option_pnl / underlying_current
        P&L_usd = option_pnl (in USD terms at current price)

        The inverse effect means P&L in USD is different
        from P&L in crypto × current_price!
        """


@dataclass
class DeribitMarginResult:
    initial_margin_btc: Decimal
    maintenance_margin_btc: Decimal
    initial_margin_usd: Decimal     # At current BTC price
    maintenance_margin_usd: Decimal
    inverse_risk_factor: float      # Convexity adjustment
```

### 2B.3 Test Matrix (Phase 2B)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| DeribitOptionsAdapter | 30 | BTC/ETH chains, orderbook |
| DVOL integration | 10 | DVOL fetch, comparison to VIX |
| Inverse margining | 25 | Margin calc, stress scenarios |
| Inverse P&L | 15 | P&L calculation, convexity |
| WebSocket streaming | 20 | Real-time quotes, reconnection |
| Paper trading | 10 | Testnet integration |
| Registry integration | 10 | Factory functions |
| **Total** | **120** | **100%** |

### 2B.4 Deliverables
- [ ] `adapters/deribit/options.py` -- Deribit options adapter
- [ ] `adapters/deribit/margin.py` -- Inverse margining calculator
- [ ] `adapters/deribit/websocket.py` -- WebSocket streaming
- [ ] `tests/test_deribit_options.py` -- 120 tests
- [ ] Documentation: `docs/options/deribit_crypto.md`

---

## Phase 3: IV Surface & Volatility Models (6 weeks)

### 3.1 Objectives
- Construct arbitrage-free IV surface using SSVI
- Implement Gatheral & Jacquier (2014) arbitrage conditions
- Add Heston for equity (NOT SABR -- SABR is for rates/FX only)
- Build Dupire local vol with Tikhonov regularization
- Robust calibration service

### 3.2 Components

#### 3.2.1 IV Surface (`impl_iv_surface.py`)

```python
class IVSurface:
    def __init__(
        self,
        underlying: str,
        spot: float,
        rate: float,
        dividend_yield: float = 0.0,
    ):
        self._strikes: np.ndarray = None
        self._expiries: np.ndarray = None  # In years
        self._iv_grid: np.ndarray = None   # 2D: [expiry, strike]
        self._ssvi_params: Optional[SSVIParams] = None

    def fit_from_quotes(
        self, quotes: List[OptionsQuote], method: str = "ssvi"
    ) -> None:
        """
        Fit surface from market quotes.

        Methods:
        - "ssvi": Surface SVI (Gatheral & Jacquier 2014) -- RECOMMENDED
        - "svi": Raw SVI per slice (may have calendar arbitrage!)
        - "cubic": Cubic spline (fast, but not arbitrage-free)
        """

    def get_iv(self, strike: float, expiry: float) -> float:
        """Interpolate IV at given strike/expiry."""

    def get_local_vol(self, strike: float, expiry: float) -> float:
        """Dupire local volatility with regularization."""

    def get_forward_iv(
        self, strike: float, t1: float, t2: float
    ) -> float:
        """Forward-starting implied vol."""

    def check_arbitrage_free(self) -> ArbitrageReport:
        """
        Check all arbitrage conditions:
        - Butterfly: d²w/dk² > 0 (convexity in total variance)
        - Calendar: ∂w/∂T > 0 (total variance increasing in time)
        - Call spread: -1 < ∂C/∂K < 0
        """
```

#### 3.2.2 SSVI (Surface SVI) -- Gatheral & Jacquier (2014)

Reference: Gatheral & Jacquier (2014) "Arbitrage-free SVI volatility surfaces"

**CRITICAL**: Use SSVI, not raw SVI! Raw SVI per slice does NOT guarantee calendar arbitrage-free.

```python
@dataclass
class SSVIParams:
    """
    Surface SVI parameterization (Gatheral & Jacquier 2014).

    Total variance: w(k, θ) = (θ/2) × {1 + ρφ(θ)k + √[(φ(θ)k + ρ)² + (1-ρ²)]}

    where:
    - θ = ATM total variance = σ_ATM² × T
    - k = log-moneyness = ln(K/F)
    - φ(θ) = η / θ^γ  (power-law ATM skew)
    - ρ = correlation (typically -0.4 to -0.9 for equity)
    - η = skew magnitude
    - γ = skew decay (typically 0.4-0.6)
    """
    rho: float        # Correlation, |ρ| < 1
    eta: float        # Skew magnitude, η > 0
    gamma: float      # Skew decay, 0 < γ < 1

    def phi(self, theta: float) -> float:
        """ATM skew function φ(θ) = η/θ^γ"""
        return self.eta / (theta ** self.gamma)


class SSVICalibrator:
    """
    SSVI calibration with Gatheral-Jacquier arbitrage-free conditions.
    """

    def calibrate(
        self, slices: Dict[float, List[OptionsQuote]]
    ) -> SSVIParams:
        """
        Calibrate SSVI to multiple expiry slices.

        Arbitrage-free conditions (Gatheral & Jacquier 2014, Theorem 4.1):

        1. Butterfly condition (no vertical spread arbitrage):
           g(k) ≥ 0 for all k, where
           g(k) = (1 - kw'/(2w))² - (w'/4)(1/w + 1/4) + w''/2

        2. Calendar spread condition (no horizontal arbitrage):
           ∂θ/∂T ≥ 0 (ATM variance increasing in time)

        3. Large-strike asymptotics:
           w(k)/|k| → 2 as |k| → ∞ (Roger Lee moment formula)

        Uses constrained optimization (SLSQP) to enforce all conditions.
        """
```

#### 3.2.3 Heston Model (`impl_heston.py`)

**ВАЖНО**: Use Heston for equity options, NOT SABR!
SABR is designed for rates/FX (no mean reversion), Heston has mean-reverting variance.

Reference: Heston (1993) "A Closed-Form Solution for Options with Stochastic Volatility"

```
dS = μS dt + √V S dW₁
dV = κ(θ - V) dt + ξ√V dW₂
dW₁·dW₂ = ρ dt

Parameters:
- κ: Mean reversion speed (typically 1-5)
- θ: Long-term variance (typically 0.04-0.16, i.e., 20-40% vol)
- ξ: Vol-of-vol (typically 0.3-1.0)
- ρ: Correlation (typically -0.5 to -0.9 for equity)
- V₀: Initial variance

Feller condition for positive variance:
    2κθ > ξ² (ensures V > 0 almost surely)
```

```python
@dataclass
class HestonParams:
    kappa: float      # Mean reversion speed
    theta: float      # Long-term variance (NOT volatility!)
    xi: float         # Vol-of-vol
    rho: float        # Correlation
    v0: float         # Initial variance

    def check_feller(self) -> bool:
        """Check Feller condition: 2κθ > ξ²"""
        return 2 * self.kappa * self.theta > self.xi ** 2


class HestonPricer:
    """Heston model pricing via characteristic function."""

    def price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        rate: float,
        params: HestonParams,
        option_type: OptionType,
        method: str = "cos",
    ) -> float:
        """
        Heston (1993) semi-analytical pricing.

        Methods:
        - "cos": COS method (Fang & Oosterlee 2008) -- RECOMMENDED
        - "lewis": Lewis (2000) formulation
        - "quadrature": Gauss-Laguerre quadrature

        Uses Lewis (2000) formulation for numerical stability.
        """


class HestonCOSTruncation:
    """
    Heston COS method truncation bounds -- NEW in v5.0.

    Reference: Fang & Oosterlee (2008) "A Novel Pricing Method for European Options
               Based on Fourier-Cosine Series Expansions"

    Problem: COS method requires truncation range [a, b] for log-price.
             Wrong bounds → inaccurate prices, especially for OTM options.

    Solution: Fang-Oosterlee (2008) cumulant-based bounds.

    Truncation formula:
        L = c₁ + c₂√T

    where:
        c₁ = coefficient depending on option parameters
        c₂ = coefficient based on cumulants of log-price distribution

    For Heston:
        a = c₁ - L√(c₂ + √c₄)
        b = c₁ + L√(c₂ + √c₄)

    where c₁, c₂, c₄ are cumulants of Heston log-price distribution.

    Recommended: L = 10-12 for standard options, L = 15+ for deep OTM.
    """

    def __init__(
        self,
        l_factor: float = 12.0,  # Default from Fang-Oosterlee (2008)
        n_terms: int = 256,      # COS expansion terms
    ):
        self.l_factor = l_factor
        self.n_terms = n_terms

    def compute_cumulants(self, params: HestonParams, T: float, r: float) -> Tuple[float, float, float, float]:
        """
        Compute first 4 cumulants of Heston log-price distribution.

        Reference: Fang & Oosterlee (2008), Appendix A.

        c₁ = E[log(S_T/S_0)] = (r - v₀/2)T + (1 - exp(-κT))(θ - v₀)/(2κ) - θT/2
        c₂ = Var[log(S_T/S_0)] = complicated expression involving κ, θ, ξ, ρ
        c₄ = fourth cumulant (for kurtosis adjustment)
        """
        kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0

        # First cumulant (mean)
        c1 = r * T + (1 - np.exp(-kappa * T)) * (theta - v0) / (2 * kappa) - theta * T / 2

        # Second cumulant (variance) - simplified
        c2 = (
            (1 / (8 * kappa**3)) *
            (xi * T * kappa * np.exp(-kappa * T) * (v0 - theta) * (8 * kappa * rho - 4 * xi) +
             kappa * rho * xi * (1 - np.exp(-kappa * T)) * (16 * theta - 8 * v0) +
             2 * theta * kappa * T * (-4 * kappa * rho * xi + xi**2 + 4 * kappa**2) +
             xi**2 * ((theta - 2 * v0) * np.exp(-2 * kappa * T) + theta * (6 * np.exp(-kappa * T) - 7) + 2 * v0) +
             8 * kappa**2 * (v0 - theta) * (1 - np.exp(-kappa * T)))
        )

        # Fourth cumulant (for kurtosis) - approximation
        c4 = 0.0  # Simplified; full expression in Fang-Oosterlee

        return c1, max(c2, 1e-10), 0.0, c4

    def compute_truncation_bounds(
        self,
        params: HestonParams,
        T: float,
        r: float,
    ) -> Tuple[float, float]:
        """
        Compute [a, b] truncation bounds for COS method.

        Returns (a, b) such that P(log(S_T/S_0) ∈ [a, b]) ≈ 1 - ε.
        """
        c1, c2, c3, c4 = self.compute_cumulants(params, T, r)

        # Fang-Oosterlee bounds
        a = c1 - self.l_factor * np.sqrt(c2 + np.sqrt(abs(c4)))
        b = c1 + self.l_factor * np.sqrt(c2 + np.sqrt(abs(c4)))

        return a, b

    def price_with_proper_truncation(
        self,
        spot: float,
        strike: float,
        T: float,
        r: float,
        params: HestonParams,
        option_type: OptionType,
    ) -> float:
        """Price using COS with properly computed truncation bounds."""
        a, b = self.compute_truncation_bounds(params, T, r)
        # COS expansion with computed bounds
        return self._cos_expansion(spot, strike, T, r, params, option_type, a, b)
```

#### 3.2.4 Dupire Local Volatility with Regularization (`impl_local_vol.py`)

Reference: Dupire (1994) "Pricing with a Smile"

**CRITICAL**: Dupire formula is numerically unstable! Must use Tikhonov regularization.

```python
def dupire_local_vol(
    iv_surface: IVSurface,
    strike: float,
    expiry: float,
    regularization: str = "tikhonov",
    lambda_reg: float = 0.01,
) -> float:
    """
    Dupire (1994) local volatility:

    σ_loc²(K, T) = [∂w/∂T] / [1 - (k/w)∂w/∂k + (1/4)(-1/4 - 1/w + k²/w²)(∂w/∂k)² + (1/2)∂²w/∂k²]

    where w = σ²T (total variance)

    PROBLEM: Denominator can be very small or negative → unstable!

    SOLUTION: Tikhonov regularization
        min ||σ_loc - σ_target||² + λ||∇²σ_loc||²

    This smooths the local vol surface and prevents spikes.
    """


class DupireRegularizer:
    """
    Tikhonov regularization with GCV λ selection -- NEW in v5.0.

    Reference: Golub, Heath, Wahba (1979) "Generalized Cross-Validation"

    Problem: λ too small → noisy local vol; λ too large → over-smoothed
    Solution: Generalized Cross-Validation (GCV) selects optimal λ automatically.

    GCV score: GCV(λ) = ||y - f_λ||² / [1 - tr(H_λ)/n]²

    where H_λ is the smoothing matrix (hat matrix).
    Minimize GCV(λ) over λ ∈ [λ_min, λ_max].
    """

    def __init__(
        self,
        lambda_min: float = 1e-6,
        lambda_max: float = 1.0,
        n_lambda_grid: int = 50,
    ):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.n_lambda_grid = n_lambda_grid
        self._optimal_lambda: Optional[float] = None

    def compute_gcv_score(
        self,
        iv_data: np.ndarray,
        lambda_val: float,
        smoothing_matrix_trace: float,
        residuals: np.ndarray,
    ) -> float:
        """Compute GCV score for given λ."""
        n = len(iv_data)
        rss = np.sum(residuals**2)
        denom = (1 - smoothing_matrix_trace / n) ** 2
        if denom < 1e-12:
            return np.inf
        return rss / (n * denom)

    def select_optimal_lambda(
        self,
        iv_data: np.ndarray,
        strike_grid: np.ndarray,
        expiry_grid: np.ndarray,
    ) -> float:
        """
        Select optimal λ via GCV.

        Returns optimal λ that minimizes leave-one-out cross-validation error.
        """
        lambda_grid = np.logspace(
            np.log10(self.lambda_min),
            np.log10(self.lambda_max),
            self.n_lambda_grid,
        )

        gcv_scores = []
        for lam in lambda_grid:
            # Solve Tikhonov problem and compute GCV score
            # (actual implementation uses sparse matrices)
            score = self._compute_gcv_for_lambda(iv_data, strike_grid, expiry_grid, lam)
            gcv_scores.append(score)

        optimal_idx = np.argmin(gcv_scores)
        self._optimal_lambda = lambda_grid[optimal_idx]
        return self._optimal_lambda

    def _compute_gcv_for_lambda(
        self,
        iv_data: np.ndarray,
        strike_grid: np.ndarray,
        expiry_grid: np.ndarray,
        lambda_val: float,
    ) -> float:
        """Compute GCV score for specific λ (implementation detail)."""
        # Uses SVD for efficient hat matrix trace computation
        pass
```

#### 3.2.5 Lee (2004) Wing Extrapolation (`impl_wing_extrapolation.py`)

**CRITICAL**: SSVI/SVI only valid near ATM. Far OTM strikes need Roger Lee moment formula!

Reference: Lee (2004) "The Moment Formula for Implied Volatility at Extreme Strikes"

```python
class LeeWingExtrapolation:
    """
    Roger Lee (2004) moment formula for IV surface wings.

    Problem: SSVI/SVI fitted to liquid ATM strikes diverges for deep OTM.
    Solution: Asymptotic behavior from moment explosion theory.

    For extreme strikes (|k| → ∞):
        σ²(k) × T → 2|k| as |k| → ∞

    More precisely:
        lim sup_{k→+∞} σ²(k)T / k = 2 - 4(√(p² + p) - p)
        lim sup_{k→-∞} σ²(k)T / |k| = 2 - 4(√(q² + q) - q)

    where p, q are maximum moments: E[S^(1+p)] < ∞, E[S^(-q)] < ∞

    For equity (finite moments): typically p ≈ 2, q ≈ 1
    → Right wing slope ≈ 0.15, Left wing slope ≈ 0.5
    """

    def __init__(
        self,
        right_moment_p: float = 2.0,  # E[S^(1+p)] < ∞
        left_moment_q: float = 1.0,   # E[S^(-q)] < ∞
        transition_strike_right: float = 0.3,  # 30% OTM calls
        transition_strike_left: float = -0.4,  # 40% OTM puts
    ):
        self.right_slope = 2 - 4 * (np.sqrt(right_moment_p**2 + right_moment_p) - right_moment_p)
        self.left_slope = 2 - 4 * (np.sqrt(left_moment_q**2 + left_moment_q) - left_moment_q)

    def extrapolate(
        self,
        ssvi_surface: IVSurface,
        log_moneyness: float,
        expiry: float,
    ) -> float:
        """
        Extrapolate IV to wings using Lee formula.

        For k > k_right: σ²T = σ²T(k_right) + slope_right × (k - k_right)
        For k < k_left:  σ²T = σ²T(k_left) + slope_left × (k_left - k)

        Smooth transition using hyperbolic tangent blending.
        """

    def _smooth_transition(
        self,
        ssvi_val: float,
        lee_val: float,
        log_moneyness: float,
        transition_center: float,
        transition_width: float = 0.05,  # 5% strike range
    ) -> float:
        """
        Smooth SSVI → Lee transition via hyperbolic tangent -- NEW in v5.0.

        Problem: Abrupt switch from SSVI to Lee at k=k_transition creates
        discontinuous IV surface → arbitrage opportunity!

        Solution: Hyperbolic tangent blend over 5% strike range:
            w(k) = (1 - α(k)) × w_ssvi(k) + α(k) × w_lee(k)

        where α(k) = 0.5 × (1 + tanh((k - k_center) / transition_width))

        Properties:
        - α = 0 at k << k_center (pure SSVI)
        - α = 0.5 at k = k_center (50/50 blend)
        - α = 1 at k >> k_center (pure Lee)
        - C∞ smooth (infinitely differentiable)

        Reference: Bayer & Gatheral (2012) "Arbitrage-free construction of the smiley"
        """
        alpha = 0.5 * (1 + np.tanh((log_moneyness - transition_center) / transition_width))
        return (1 - alpha) * ssvi_val + alpha * lee_val

    def get_iv_with_smooth_wings(
        self,
        ssvi_surface: IVSurface,
        log_moneyness: float,
        expiry: float,
    ) -> float:
        """
        Get IV with smooth SSVI→Lee wing transition.

        Uses smooth_transition for |k| in [k_trans - 0.025, k_trans + 0.025].
        """
        if log_moneyness > self.transition_strike_right - 0.025:
            ssvi_iv = ssvi_surface.get_iv_at_k(log_moneyness, expiry)
            lee_iv = self.extrapolate(ssvi_surface, log_moneyness, expiry)
            return self._smooth_transition(
                ssvi_iv, lee_iv, log_moneyness,
                self.transition_strike_right, transition_width=0.05,
            )
        elif log_moneyness < self.transition_strike_left + 0.025:
            ssvi_iv = ssvi_surface.get_iv_at_k(log_moneyness, expiry)
            lee_iv = self.extrapolate(ssvi_surface, log_moneyness, expiry)
            return self._smooth_transition(
                ssvi_iv, lee_iv, log_moneyness,
                self.transition_strike_left, transition_width=0.05,
            )
        else:
            # Pure SSVI region
            return ssvi_surface.get_iv_at_k(log_moneyness, expiry)
```

#### 3.2.6 Bates/SVJ Model (`impl_bates.py`)

**IMPORTANT**: Heston misses jump risk! Bates (1996) = Heston + Merton jumps.

Reference: Bates (1996) "Jumps and Stochastic Volatility"

Why Bates for equity options:
- Heston explains skew but NOT smile curvature at wings
- Jumps capture crash risk (fat left tail)
- Essential for short-dated options where jump risk dominates

```
Model (Bates 1996):
dS/S = (r - λμ_J) dt + √V dW₁ + J dN
dV = κ(θ - V) dt + ξ√V dW₂

where:
- J = jump size ~ N(μ_J, σ_J²) -- log-normal jump
- N = Poisson process with intensity λ
- μ_J = mean jump size (typically -0.05 to -0.15 for crash risk)
- σ_J = jump size volatility (typically 0.1-0.2)
- λ = jump frequency (typically 1-5 per year)
```

```python
@dataclass
class BatesParams(HestonParams):
    """
    Bates (1996) SVJ model = Heston + Merton jumps.
    Inherits: kappa, theta, xi, rho, v0 from Heston
    """
    lambda_jump: float    # Jump intensity (events/year)
    mu_jump: float        # Mean log jump size (negative for crash)
    sigma_jump: float     # Jump size std dev

    def get_jump_compensator(self) -> float:
        """Drift adjustment for risk-neutral jump: μ_J = E[e^J - 1]"""
        return self.lambda_jump * (np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1)


class BatesPricer:
    """
    Bates (1996) SVJ pricing via characteristic function.

    Key insight: CF(Bates) = CF(Heston) × CF(Merton jump)

    Can use same FFT/COS method as Heston with modified CF.
    """

    def price(
        self,
        spot: float,
        strike: float,
        expiry: float,
        rate: float,
        params: BatesParams,
        option_type: OptionType,
        method: str = "cos",  # "cos" (fast) or "fft" (standard)
    ) -> float:
        """
        Bates pricing via COS method (Fang & Oosterlee 2008).

        COS method faster than FFT for single strikes.
        Use FFT for full chain pricing.
        """

    def implied_vol_from_bates(
        self, price: float, spot: float, strike: float,
        expiry: float, rate: float, option_type: OptionType,
    ) -> float:
        """Invert Bates price to Black-Scholes IV."""


class BatesCalibrator:
    """
    Calibrate Bates to surface.

    Strategy:
    1. First calibrate Heston to ATM slice (fast, 5 params)
    2. Then calibrate jump params (λ, μ_J, σ_J) to wings
    3. Joint refinement with full surface
    """
```

#### 3.2.7 Calibration Service (`service_iv_calibration.py`)

```python
class IVCalibrationService:
    """
    Production IV surface calibration service.

    Features:
    - Real-time recalibration on quote updates
    - Arbitrage-free enforcement (SSVI)
    - Stale quote filtering
    - Bid-ask midpoint with spread weighting
    """

    def calibrate_ssvi(
        self,
        quotes: List[OptionsQuote],
        min_volume: int = 10,
        max_spread_pct: float = 0.5,
    ) -> IVSurface:
        """
        Calibrate arbitrage-free SSVI surface.

        Filters:
        - Remove quotes with volume < min_volume
        - Remove quotes with spread > max_spread_pct
        - Use volume-weighted midpoint
        """

    def calibrate_heston(
        self,
        quotes: List[OptionsQuote],
        method: str = "differential_evolution",
    ) -> HestonParams:
        """
        Calibrate Heston to surface.

        Methods:
        - "differential_evolution": Global optimization (recommended)
        - "levenberg_marquardt": Local optimization (faster, needs good seed)
        """

    def check_arbitrage(self, surface: IVSurface) -> ArbitrageReport:
        """
        Full arbitrage check:
        - Butterfly (strike convexity)
        - Calendar (time monotonicity)
        - Put-call parity violations
        """
```

### 3.3 Test Matrix (Phase 3)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| IVSurface construction | 25 | From quotes, interpolation |
| SSVI fitting | 35 | Gatheral-Jacquier conditions, calibration |
| Heston pricing | 25 | Characteristic function, Greeks |
| Heston calibration | 20 | DE, LM, parameter recovery |
| **Bates/SVJ pricing** | **20** | **CF inversion, jump component, Greeks** |
| **Bates calibration** | **15** | **Two-stage calibration, parameter recovery** |
| Dupire local vol | 20 | Regularization, stability |
| **Lee wing extrapolation** | **15** | **Moment formula, transition blending** |
| Arbitrage detection | 20 | Butterfly, calendar, PCP |
| Forward vol | 10 | Term structure |
| **Total** | **205** | **100%** |

### 3.4 Deliverables
- [ ] `impl_iv_surface.py` -- IV surface with SSVI
- [ ] `impl_ssvi.py` -- SSVI model with Gatheral-Jacquier conditions
- [ ] `impl_heston.py` -- Heston model (NOT SABR for equity!)
- [ ] `impl_bates.py` -- Bates/SVJ model (Heston + Merton jumps)
- [ ] `impl_wing_extrapolation.py` -- Lee (2004) moment formula
- [ ] `impl_local_vol.py` -- Dupire with Tikhonov regularization
- [ ] `service_iv_calibration.py` -- Production calibration service
- [ ] `tests/test_iv_surface.py` -- 205 tests
- [ ] Documentation: `docs/options/volatility_surface.md`

---

## Phase 4: L2 Execution Provider (4 weeks)

### 4.1 Objectives
- Options-specific slippage model with moneyness/DTE factors
- Greeks-aware execution
- Bid-ask spread modeling (including PFOF effects)
- Options fee structures

### 4.2 Components

#### 4.2.1 Options Slippage Model (`impl_options_slippage.py`)

**CRITICAL**: Options slippage fundamentally differs from equity!

Key differences:
- Spread is NOT constant -- varies with moneyness, DTE, underlying vol
- Gamma exposure creates asymmetric slippage
- Retail flow (PFOF) gets better execution than displayed

Reference: Muravyev & Pearson (2020) "Options Trading Costs Are Lower Than You Think"

```python
@dataclass
class OptionsSlippageConfig:
    # Base factors
    base_spread_pct: float = 0.05          # 5% of mid as base

    # Moneyness adjustments
    atm_spread_mult: float = 1.0           # ATM is base
    otm_10_spread_mult: float = 1.5        # 10% OTM: 1.5x spread
    otm_20_spread_mult: float = 2.5        # 20% OTM: 2.5x spread
    deep_otm_spread_mult: float = 4.0      # 30%+ OTM: 4x spread

    # DTE adjustments
    dte_30_spread_mult: float = 1.0        # 30+ DTE: base
    dte_7_spread_mult: float = 1.5         # 7-30 DTE: 1.5x
    dte_1_spread_mult: float = 3.0         # 0-7 DTE: 3x (expiration gamma)

    # PFOF retail advantage
    pfof_improvement_bps: float = 5.0      # 5 bps price improvement for retail


from execution_providers import SlippageProvider  # ВАЖНО: Protocol inheritance!

class OptionsSlippageProvider(SlippageProvider):
    """
    Options slippage model with:
    - Moneyness dependency (ATM tighter than OTM)
    - DTE dependency (near-expiry wider)
    - Greeks impact (gamma, vega exposure)
    - PFOF retail flow improvement
    - Underlying correlation (delta hedge cost)

    Reference: Muravyev & Pearson (2020)

    ВАЖНО: Наследует SlippageProvider Protocol из execution_providers.py!
    Должен реализовывать compute_slippage_bps() метод.
    """

    def compute_slippage_bps(
        self,
        order: OptionsOrder,
        market: OptionsMarketState,
        greeks: GreeksResult,
        is_retail: bool = False,
    ) -> float:
        """
        Total slippage in basis points.

        Components:
        1. spread_component = spread/2 × moneyness_mult × dte_mult
        2. gamma_component = γ_impact × |gamma| × spot² × vol × √participation
        3. vega_component = ν_impact × |vega| × expected_iv_move
        4. theta_component = θ_impact × |theta| × execution_time
        5. delta_hedge_component = |delta| × underlying_slippage
        6. pfof_improvement = -pfof_bps if is_retail else 0
        """
```

#### 4.2.2 Options Fee Provider (`impl_options_fees.py`)

**ВАЖНО**: Следует паттерну из `execution_providers.py:FeeProvider`

| Exchange | Per Contract | Index Premium | Regulatory | Notes |
|----------|--------------|---------------|------------|-------|
| CBOE | $0.44 | $0.66 (SPX) | OCC $0.055 | Maker/taker varies |
| NYSE Arca | $0.47 | -- | OCC $0.055 | |
| Deribit | 0.03% notional | -- | None | Capped at 12.5% of premium |
| IB (Tiered) | $0.15-0.65 | -- | Varies | Volume-based |
| IB (Fixed) | $0.65 | -- | Varies | Simple |

```python
from execution_providers import FeeProvider  # ВАЖНО: Protocol inheritance!

class OptionsFeeProvider(FeeProvider):
    """
    Options fee provider.

    ВАЖНО: Наследует FeeProvider Protocol из execution_providers.py!
    Должен реализовывать compute_fee() метод.
    """

    def compute_fee(
        self,
        contract: OptionsContractSpec,
        qty: int,
        price: Decimal,
        exchange: str,
        is_maker: bool = False,
        is_index: bool = False,
    ) -> Decimal:
        """
        Total fee = exchange_fee + occ_fee + regulatory_fee

        OCC fee: $0.055/contract (all US options)
        Regulatory: SEC fee ($0.0000278/$ on sells), TAF, etc.
        """
```

#### 4.2.3 Options Execution Provider (`execution_providers_options.py`)

**ВАЖНО**: Integrates with existing `AssetClass.OPTIONS` and follows `L2ExecutionProvider` pattern

```python
from execution_providers import (
    SlippageProvider,
    FillProvider,
    FeeProvider,
    AssetClass,
    create_execution_provider,
)

class OptionsL2ExecutionProvider:
    """
    L2 execution provider for options.

    Follows existing pattern from execution_providers.py.
    Integrates with AssetClass.OPTIONS.
    """

    def __init__(
        self,
        slippage_provider: OptionsSlippageProvider,
        fee_provider: OptionsFeeProvider,
        greeks_provider: GreeksProvider,
    ):
        self.asset_class = AssetClass.OPTIONS

    def execute(
        self,
        order: OptionsOrder,
        market: OptionsMarketState,
        bar: BarData,
        underlying_bar: BarData,
        is_retail: bool = False,
    ) -> OptionsFill:
        """Execute single-leg option order."""

    def estimate_cost(
        self,
        order: OptionsOrder,
        market: OptionsMarketState,
    ) -> CostEstimate:
        """Pre-trade cost estimation with Greeks impact."""


# Factory function following existing pattern
def create_options_execution_provider(
    level: str = "L2",
    profile: str = "default",
) -> OptionsL2ExecutionProvider:
    """
    Factory function for options execution provider.

    Follows pattern from execution_providers.create_execution_provider().
    """
```

#### 4.2.4 Options Execution Algorithms (`execution_algos_options.py`)

**CRITICAL**: Options execution requires specialized algorithms!

Reference: Almgren (2012) "Optimal Execution with Nonlinear Impact Functions and Trading-Enhanced Risk"

**Why different from equity**:
- Options have Greeks exposure that changes during execution
- Delta-neutral execution requires simultaneous underlying hedge
- Gamma risk is path-dependent
- Large orders move implied volatility (IV impact)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from execution_algos import BaseExecutionAlgo  # Reuse base from equity


@dataclass
class OptionsExecutionResult:
    """Result of options execution algorithm."""
    fills: List[OptionsFill]
    underlying_fills: List[Fill]  # Delta hedge fills
    avg_price: float
    total_qty: int
    realized_slippage_bps: float
    iv_impact_bps: float
    delta_hedge_slippage_bps: float
    gamma_cost: float  # Path-dependent gamma bleed


class DeltaNeutralTWAP:
    """
    Delta-neutral TWAP for options.

    Executes option position in slices while maintaining delta-neutral
    by simultaneously hedging with underlying.

    Algorithm:
    1. Split target qty into N child orders
    2. For each child:
       a. Calculate required delta hedge
       b. Execute option leg
       c. Execute underlying hedge leg (immediately)
       d. Track cumulative gamma exposure

    Reference: Cartea et al. (2015) "Algorithmic Trading and Market Microstructure"
    """

    def __init__(
        self,
        target_slices: int = 10,
        hedge_threshold_delta: float = 0.1,  # Rehedge when net delta > 0.1
        max_iv_impact_bps: float = 50.0,     # Stop if IV moves > 50bps
    ):
        pass

    def execute(
        self,
        order: OptionsOrder,
        market: OptionsMarketState,
        underlying_market: MarketState,
        greeks: GreeksResult,
    ) -> OptionsExecutionResult:
        """Execute with delta-neutral maintenance."""


class GammaAwarePOV:
    """
    Participation-of-Volume with gamma adjustment.

    Key insight: Gamma exposure increases execution urgency near expiry.
    Standard POV may be too slow for short-dated options.

    Execution speed = base_pov × (1 + gamma_urgency_factor)

    where gamma_urgency = |gamma| × S² × (1/DTE) × vol

    For high gamma (near ATM, short DTE): execute faster
    For low gamma (deep OTM/ITM): can be patient
    """

    def __init__(
        self,
        base_pov: float = 0.10,           # 10% of volume baseline
        gamma_urgency_mult: float = 2.0,   # Max 2x faster for high gamma
        min_pov: float = 0.05,             # Min 5% participation
        max_pov: float = 0.30,             # Max 30% participation
    ):
        pass


class SpreadExecutionAlgo:
    """
    Multi-leg spread execution algorithm.

    Challenge: Legging risk -- if one leg fills and other doesn't,
    creates unwanted directional exposure.

    Strategies:
    1. Native spread order (if exchange supports)
    2. Legging with immediate hedge
    3. Working both legs simultaneously

    Reference: Duffie (2010) "Dynamic Asset Pricing Theory"
    """

    def __init__(
        self,
        strategy: str = "native",  # "native", "legging_hedged", "simultaneous"
        max_leg_imbalance: int = 5,  # Max contracts imbalance
        emergency_hedge_threshold: float = 0.5,  # Hedge if 50% of one leg filled
    ):
        pass

    def execute_spread(
        self,
        spread_order: SpreadOrder,
        leg1_market: OptionsMarketState,
        leg2_market: OptionsMarketState,
    ) -> SpreadExecutionResult:
        """
        Execute multi-leg spread.

        Returns combined result with legging risk metrics.
        """


class IVImpactModel:
    """
    Implied volatility impact from options trading.

    Key insight: Large options orders move IV, not just price!
    This is different from equity market impact.

    Model (empirical):
    ΔIV = λ × sign(order) × (order_vega / total_vega) × √participation

    where:
    - order_vega = |vega| × qty
    - total_vega = sum of vega across all strikes
    - participation = order_vega / ADV_vega

    Reference: Garleanu et al. (2009) "Demand-Based Option Pricing"
    """

    def __init__(
        self,
        lambda_iv_impact: float = 0.05,  # IV moves ~0.05 vol point per 1% vega participation
        decay_half_life_minutes: float = 30.0,  # Impact decays over 30 min
    ):
        pass

    def estimate_iv_impact(
        self,
        order: OptionsOrder,
        market: OptionsMarketState,
        chain_total_vega: float,
    ) -> float:
        """Estimate IV impact in volatility points."""
```

### 4.3 Test Matrix (Phase 4)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| Options Slippage (moneyness) | 20 | ATM, OTM, ITM |
| Options Slippage (DTE) | 15 | 30+, 7-30, 0-7 DTE |
| Options Slippage (Greeks) | 20 | Gamma, vega, theta impact |
| Options Slippage (PFOF) | 10 | Retail vs institutional |
| Fee Calculation | 25 | All exchanges, index premium |
| L2 Execution | 35 | Market/limit, fills |
| Cost Estimation | 15 | Pre-trade analysis |
| **Delta-Neutral TWAP** | **20** | **Hedge execution, gamma tracking** |
| **Gamma-Aware POV** | **15** | **Urgency adjustment, expiry behavior** |
| **Spread Execution** | **20** | **Legging risk, native spreads, hedging** |
| **IV Impact Model** | **15** | **Vega participation, decay** |
| Factory Integration | 10 | create_options_execution_provider |
| **Total** | **220** | **100%** |

### 4.4 Deliverables
- [ ] `impl_options_slippage.py` -- Moneyness/DTE/Greeks-aware slippage
- [ ] `impl_options_fees.py` -- Fee structures (all exchanges)
- [ ] `execution_providers_options.py` -- L2 provider (integrates with AssetClass.OPTIONS)
- [ ] `execution_algos_options.py` -- Delta-neutral TWAP, Gamma-aware POV, Spread execution
- [ ] `impl_iv_impact.py` -- IV impact model for large orders
- [ ] `tests/test_options_execution_l2.py` -- 220 tests
- [ ] Documentation: `docs/options/execution_l2.md`

---

## Phase 5: L3 LOB Simulation (6 weeks)

### 5.0 PREREQUISITE: Phase 0.5 Memory Architecture -- NEW in v5.0

**CRITICAL**: Phase 5 MUST use memory-efficient architecture from Phase 0.5!

```python
# REQUIRED imports from Phase 0.5
from lob.options_memory import (
    LazyMultiSeriesLOBManager,  # Lazy instantiation + LRU cache (max 50 LOBs)
    RingBufferOrderBook,         # Fixed depth (100 levels), O(N) memory
    EventDrivenLOBCoordinator,   # O(N log N) cross-series updates
)

# NOT THIS (naive approach):
# all_lobs = {series: OrderBook() for series in 480_series}  # 240GB RAM!
```

Without Phase 0.5: SPY chain (480 LOBs × 500MB) = **240GB RAM → Fails**
With Phase 0.5: max 50 active LOBs × 50MB = **2.5GB RAM → OK**

### 5.1 Objectives
- Options-specific order book (fundamentally different from equity!)
- Market maker behavior simulation (Cho & Engle 2022)
- Quote dynamics with regime detection
- Pin risk simulation (Avellaneda & Lipkin 2003)
- Cross-strike arbitrage detection
- **Memory-efficient multi-series management (Phase 0.5)**

### 5.2 Critical Design Note

**Options LOB is FUNDAMENTALLY DIFFERENT from Equity LOB!**

| Aspect | Equity LOB | Options LOB |
|--------|-----------|-------------|
| Spread | 1 cent typical | $0.05-$1.00 typical |
| Depth | 1000s of shares | 10-100 contracts |
| Liquidity | Continuous | Episodic |
| Quote updates | Tick-by-tick | On underlying move |
| MM behavior | Inventory-based | Greeks-based |
| Expiration | N/A | Liquidity collapse |

**IMPORTANT**: Do NOT simply extend `lob/matching_engine.py`. Create `lob/options_matching.py` with options-specific logic, but reuse primitives from `lob/data_structures.py`.

### 5.2.1 Multi-Series LOB Architecture

**CRITICAL**: Options chains require coordinated multi-book management!

For SPY options: ~40 strikes × 12 expiries = **480 individual order books**!
Each book must be synchronized and cross-referenced.

```python
class MultiSeriesLOBManager:
    """
    Manages N×M order books for full options chain.

    Architecture:
    - ChainLOB: Container for all books in a chain
    - SeriesLOB: Individual strike/expiry book
    - CrossSeriesArbitrage: Real-time arbitrage detection
    - SynchronizedQuoteUpdate: Atomic updates across series

    Memory optimization:
    - Lazy loading for illiquid series
    - Compressed representation for deep OTM
    - Shared price level structure for same-expiry books

    Reference: Chicago Trading Company (2019) internal docs
    """

    def __init__(
        self,
        underlying: str,
        strikes: List[float],
        expiries: List[date],
        lazy_load_threshold: float = 0.3,  # Load books with OI > 30% of max
    ):
        # Total books = len(strikes) × len(expiries) × 2 (put/call)
        self._book_count = len(strikes) * len(expiries) * 2
        self._books: Dict[SeriesKey, SeriesLOB] = {}
        self._loaded_books: Set[SeriesKey] = set()

    def get_or_create_book(
        self, strike: float, expiry: date, option_type: OptionType
    ) -> SeriesLOB:
        """Lazy-load series book on demand."""

    def update_chain_quotes(
        self,
        underlying_move: float,
        iv_surface: IVSurface,
        greeks_cache: Dict[SeriesKey, GreeksResult],
    ) -> None:
        """
        Atomic quote update across entire chain.

        When underlying moves:
        1. Recalculate all deltas/gammas
        2. Shift all quotes based on delta
        3. Adjust spreads based on gamma exposure
        4. Check for cross-series arbitrage
        """

    def detect_cross_series_arbitrage(self) -> List[ArbitrageOpportunity]:
        """
        Check for arbitrage across entire chain:
        - Butterfly violations
        - Calendar spread violations
        - Put-call parity violations
        """


@dataclass
class SeriesKey:
    """Unique identifier for options series."""
    strike: float
    expiry: date
    option_type: OptionType

    def __hash__(self) -> int:
        return hash((self.strike, self.expiry, self.option_type))


class SeriesLOB:
    """
    Individual order book for one options series.

    Lighter weight than equity LOB:
    - Max depth: 5-10 levels (vs 50+ for equity)
    - Quote persistence: seconds (vs milliseconds)
    - Update frequency: on underlying move (vs tick-by-tick)
    """

    def __init__(
        self,
        key: SeriesKey,
        tick_size: Decimal = Decimal("0.01"),
        max_depth: int = 10,
    ):
        pass


class CrossExpiryCoordinator:
    """
    Coordinate quote updates across expiries.

    Key insight: When IV surface shifts, ALL expiries move.
    Must update consistently to avoid calendar arbitrage.
    """

    def update_term_structure(
        self,
        iv_surface: IVSurface,
        atm_forward_curve: List[float],
    ) -> Dict[date, float]:
        """
        Update IV term structure consistently.

        Ensures: σ(T1) < σ(T2) for T1 < T2 (no calendar arbitrage)
        """
```

### 5.3 Components

#### 5.3.1 Options Matching Engine (`lob/options_matching.py`)

**CRITICAL v5.0**: Reuse existing `ProRataMatchingEngine` from `lob/matching_engine.py`!

```python
from lob.data_structures import LimitOrder, Side, OrderType, Fill
from lob.matching_engine import ProRataMatchingEngine  # REUSE existing!

class OptionsMatchingEngine:
    """
    Options-specific matching engine -- EXTENDS ProRataMatchingEngine.

    Reference: CBOE Exchange Rules, PHLX Pro-Rata Matching Algorithm

    Key differences from equity:
    1. Priority: Price-Time-Pro-Rata hybrid (CBOE uses pro-rata for MM)
    2. Quote updates: Entire book shifts on underlying move
    3. Order types: Complex orders (spreads) have priority
    4. Minimum size: Often 1 contract minimum displayed

    Implementation uses existing ProRataMatchingEngine (lob/matching_engine.py:~800):
    - Pro-rata allocation for market makers at NBBO
    - Customer priority rule (retail orders filled first)
    - Complex order book (COB) for spreads
    """

    def __init__(
        self,
        contract: OptionsContractSpec,
        mm_pro_rata_allocation: float = 0.4,  # 40% pro-rata to MMs (CBOE Rule 6.45)
        customer_priority_pct: float = 0.2,   # 20% customer priority
    ):
        self.contract = contract
        # Reuse existing ProRataMatchingEngine
        self._engine = ProRataMatchingEngine(
            symbol=contract.occ_symbol,
            pro_rata_ratio=mm_pro_rata_allocation,
        )
        self._customer_priority_pct = customer_priority_pct

    def match_with_cboe_rules(
        self, order: LimitOrder, is_customer: bool = False
    ) -> List[Fill]:
        """
        CBOE-style matching with customer priority.

        Order of execution (CBOE Rule 6.45):
        1. Price-improving orders (any participant)
        2. Customer orders at NBBO (20% priority)
        3. Market Maker pro-rata at NBBO (40% pro-rata)
        4. Public customer orders FIFO
        5. Firm/BD orders FIFO

        Reference: CBOE Rule 6.45 "Priority of Bids and Offers"
        """
        if is_customer:
            # Customer priority: filled before MM pro-rata
            return self._engine.match_with_customer_priority(order, self._customer_priority_pct)
        return self._engine.match(order)

    def shift_book_on_underlying_move(
        self,
        underlying_move_pct: float,
        delta: float,
        gamma: float,
    ) -> None:
        """
        Shift all quotes based on delta/gamma.

        New price = old_price + delta × ΔS + 0.5 × gamma × ΔS²

        Preserves order priority and size after shift.
        """

    def handle_complex_order(
        self,
        legs: List[Tuple[SeriesKey, Side, int]],
        net_price: Decimal,
    ) -> List[Fill]:
        """
        Handle multi-leg complex orders (spreads).

        Complex Order Book (COB) has priority over individual legs
        when net price improves NBBO-derived theoretical price.

        Reference: CBOE Complex Order Book rules
        """
```

#### 5.3.2 Market Maker Simulator (`lob/options_mm.py`)

Reference: Cho & Engle (2022) "Market Maker Quotes in Options Markets"

**Key insight**: MM quoting is regime-dependent!

```python
@dataclass
class MMQuotingRegime(Enum):
    """
    Market maker quoting regimes (Cho & Engle 2022).

    Regime affects spread, size, and requote frequency.
    """
    NORMAL = "normal"           # Standard quoting
    HIGH_VOL = "high_vol"       # Wider spreads, smaller size
    EARNINGS = "earnings"       # Much wider, minimal size
    EXPIRATION = "expiration"   # Gamma-driven, very wide
    LOW_LIQUIDITY = "low_liq"   # Off-hours, wide spreads


class OptionsMMSimulator:
    """
    Regime-dependent market maker simulator.

    Based on Cho & Engle (2022) empirical findings:
    - Spread = f(|gamma|, vega/theta ratio, inventory, regime)
    - Size = f(max_position, concentration, regime)
    - Requote trigger = f(delta_change, iv_change, time)
    """

    def __init__(
        self,
        inventory_limit: int = 500,        # Lower than equity!
        base_spread_pct: float = 0.03,     # 3% base
        gamma_factor: float = 0.8,         # Cho & Engle coefficient
        vega_theta_factor: float = 0.4,    # Vega/theta impact
        regime_spreads: Dict[MMQuotingRegime, float] = None,
    ):
        self.regime_spreads = regime_spreads or {
            MMQuotingRegime.NORMAL: 1.0,
            MMQuotingRegime.HIGH_VOL: 2.5,
            MMQuotingRegime.EARNINGS: 5.0,
            MMQuotingRegime.EXPIRATION: 3.0,
            MMQuotingRegime.LOW_LIQUIDITY: 4.0,
        }

    def detect_regime(
        self,
        underlying_rv: float,
        iv_level: float,
        dte: float,
        has_earnings: bool,
        hour_utc: int,
    ) -> MMQuotingRegime:
        """Detect current quoting regime."""

    def generate_quotes(
        self,
        contract: OptionsContractSpec,
        underlying: float,
        iv: float,
        greeks: GreeksResult,
        current_inventory: int,
        regime: MMQuotingRegime,
    ) -> Tuple[Quote, Quote]:
        """
        Generate bid/ask quotes.

        Spread formula (Cho & Engle 2022):
        spread = base × regime_mult × (1 + γ_factor × |gamma/gamma_max|
                                         + ν_factor × |vega/theta|)

        Size formula:
        size = max_pos × (1 - |inventory/limit|) × (1 - concentration_penalty)
        """

    def should_requote(
        self,
        delta_change: float,
        iv_change: float,
        time_elapsed_sec: float,
        regime: MMQuotingRegime,
    ) -> bool:
        """
        Check if requote needed.

        Thresholds vary by regime:
        - NORMAL: Δdelta > 0.01 OR Δiv > 0.5%
        - HIGH_VOL: Δdelta > 0.005 OR Δiv > 0.25%
        - EXPIRATION: Δdelta > 0.001 (very sensitive)
        """
```

#### 5.3.3 Pin Risk Simulation (`lob/pin_risk.py`)

Reference: Avellaneda & Lipkin (2003) "A market-induced mechanism for stock pinning"

```python
class PinRiskSimulator:
    """
    Stock pinning simulation near expiration.

    Mechanism (Avellaneda & Lipkin 2003):
    1. Near expiry, delta hedging by MMs creates feedback
    2. If spot near strike with high OI, hedging pushes spot toward strike
    3. Probability of pinning ∝ OI × gamma × sqrt(T)
    """

    def compute_pin_probability(
        self,
        strike: float,
        spot: float,
        time_to_expiry: float,
        iv: float,
        open_interest: int,
        delta_hedger_volume: float,
    ) -> float:
        """
        Pin probability using Avellaneda-Lipkin model:

        P_pin ∝ OI × Γ × √T × (hedger_volume / total_volume)

        where Γ = gamma at strike
        """

    def simulate_expiry_dynamics(
        self,
        strikes: List[float],
        spot: float,
        oi_by_strike: Dict[float, int],
        hedger_fraction: float,
        n_paths: int = 10000,
    ) -> PinningAnalysis:
        """
        Monte Carlo simulation of spot dynamics near expiry.

        Returns:
        - Pin probabilities by strike
        - Expected spot distribution at expiry
        - Gamma exposure concentration
        """
```

#### 5.3.4 Cross-Strike Arbitrage (`lob/options_arbitrage.py`)

```python
class OptionsArbitrageDetector:
    """
    Real-time arbitrage detection in options markets.

    Types:
    1. Butterfly: C(K-) - 2C(K) + C(K+) should be positive
    2. Box spread: Should price at risk-free rate
    3. Conversion/Reversal: Put-call parity violations
    4. Calendar: Far expiry should be worth more
    """

    def detect_butterfly_arb(
        self,
        quotes: Dict[float, OptionsQuote],
        transaction_cost: float,
    ) -> List[ButterflyArbitrage]:
        """
        Find butterfly arbitrage opportunities.

        Condition: C(K-) + C(K+) > 2C(K) + 2 × transaction_cost
        """

    def detect_box_spread_arb(
        self,
        call_quotes: Dict[float, OptionsQuote],
        put_quotes: Dict[float, OptionsQuote],
        risk_free_rate: float,
        time_to_expiry: float,
    ) -> Optional[BoxArbitrage]:
        """
        Find box spread mispricing.

        Box value should be: (K_high - K_low) × e^(-rT)
        """
```

### 5.4 Test Matrix (Phase 5)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| **MultiSeriesLOBManager** | **30** | **Lazy loading, memory optimization, 480+ books** |
| **SeriesLOB** | **20** | **Individual book, depth limits** |
| **CrossExpiryCoordinator** | **15** | **Term structure consistency, calendar arbitrage** |
| OptionsMatchingEngine | 35 | FIFO, pro-rata, book shift |
| MM Simulator (regimes) | 40 | All 5 regimes, transitions |
| MM Simulator (quoting) | 30 | Spread formula, size, requote |
| Pin Risk | 25 | Probability, dynamics |
| Arbitrage Detection | 25 | Butterfly, box, calendar |
| Integration | 30 | Full L3 simulation, stress tests |
| **Total** | **250** | **100%** |

### 5.5 Deliverables
- [ ] `lob/multi_series_lob.py` -- Multi-series LOB manager (480+ books)
- [ ] `lob/series_lob.py` -- Individual series order book
- [ ] `lob/cross_expiry_coordinator.py` -- Cross-expiry consistency
- [ ] `lob/options_matching.py` -- Options matching engine
- [ ] `lob/options_mm.py` -- Cho & Engle MM simulator
- [ ] `lob/pin_risk.py` -- Avellaneda-Lipkin pin simulation
- [ ] `lob/options_arbitrage.py` -- Real-time arbitrage detection
- [ ] `tests/test_options_l3_lob.py` -- 250 tests
- [ ] Documentation: `docs/options/l3_lob.md`

---

## Phase 6: Risk Management (7 weeks)

**Duration increased**: OCC STANS Monte Carlo is complex, requires proper implementation

### 6.1 Objectives
- Options-specific risk guards (following `services/futures_risk_guards.py` pattern)
- **OCC STANS (System for Theoretical Analysis and Numerical Simulations)** -- Monte Carlo VaR
- Reg T margin with correct formulas
- Exercise/assignment simulation with gamma convexity
- Portfolio stress testing with extreme scenarios

### 6.2 Components

#### 6.2.1 Options Risk Guards (`services/options_risk_guards.py`)

**ВАЖНО**: Follows pattern from `services/futures_risk_guards.py` -- dataclasses + enums + multi-level alerts

```python
from enum import Enum
from dataclasses import dataclass

class OptionsRiskLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


@dataclass
class GreeksLimitConfig:
    max_delta: float = 100.0      # Net delta in underlying terms
    max_gamma: float = 50.0       # Per 1% move
    max_vega: float = 10000.0     # Per 1 vol point
    max_theta: float = -500.0     # Daily theta limit (negative OK for theta-positive)
    max_speed: float = 100.0      # Third-order gamma risk


@dataclass
class GreeksRiskResult:
    level: OptionsRiskLevel
    delta_utilization: float
    gamma_utilization: float
    vega_utilization: float
    theta_utilization: float
    speed_utilization: float      # Third-order risk
    breach_details: List[str]


class OptionsGreeksGuard:
    """
    Monitor portfolio Greeks limits.

    Follows pattern from FuturesLeverageGuard in services/futures_risk_guards.py.
    """

    def __init__(self, config: GreeksLimitConfig):
        self.config = config

    def check_position(
        self,
        positions: List[OptionsPosition],
        underlying_prices: Dict[str, float],
    ) -> GreeksRiskResult:
        """Check if aggregate Greeks within limits."""

    def check_new_trade(
        self,
        proposed_trade: OptionsOrder,
        current_positions: List[OptionsPosition],
        underlying_prices: Dict[str, float],
    ) -> Tuple[bool, str]:
        """Pre-trade Greeks check."""


class ExerciseRiskGuard:
    """
    Monitor early exercise risk for American options.

    Based on Broadie & Detemple (1996) optimal exercise boundary.
    """

    def check_early_exercise_risk(
        self,
        position: OptionsPosition,
        underlying_price: float,
        days_to_expiry: float,
        dividend_schedule: List[Dividend],
        rate: float,
    ) -> ExerciseRiskResult:
        """
        Identify positions at risk of early exercise.

        Decision criteria (Broadie & Detemple 1996):
        1. For calls: Exercise if dividend > time_value + gamma_convexity_value
        2. For puts: Exercise if rate income > time_value + gamma_convexity_value

        Gamma convexity value is often ignored but can be significant!
        """


class AssignmentRiskGuard:
    """Monitor assignment risk for short options."""

    def check_assignment_probability(
        self,
        position: OptionsPosition,
        underlying_price: float,
        days_to_expiry: float,
        dividend: Optional[Dividend],
    ) -> AssignmentProbability:
        """
        Assess probability of assignment.

        High risk triggers:
        - ITM calls before ex-dividend
        - Deep ITM puts
        - Day before expiration
        """
```

#### 6.2.2 OCC STANS Margin Calculator (`impl_occ_stans.py`)

**CRITICAL**: Use OCC STANS methodology, NOT TIMS!
TIMS (Theoretical Intermarket Margin System) was replaced by OCC's STANS in 2006.

Reference: OCC "Margin Methodology" (current as of 2024), "STANS Technical Specifications"

**STANS = System for Theoretical Analysis and Numerical Simulations**

Key points:
- **Monte Carlo VaR**, NOT parametric VaR!
- 10,000+ scenarios per risk class
- 2-day 99% Expected Shortfall (ES), not VaR
- Full repricing (not delta/gamma approximation)
- Correlation model with regime detection

```python
@dataclass
class STANSScenarioConfig:
    """Configuration for OCC STANS scenario generation."""
    n_scenarios: int = 10000           # OCC uses ~10,000
    confidence_level: float = 0.99     # 99% ES
    horizon_days: int = 2              # 2-day liquidation period
    spot_range_pct: float = 0.15       # ±15% spot moves
    vol_range_pct: float = 0.40        # ±40% IV moves
    rate_range_bps: float = 50         # ±50 bps rate moves
    use_historical_scenarios: bool = True  # Include historical worst days
    stress_scenarios: List[str] = field(default_factory=lambda: [
        "black_monday_1987",
        "flash_crash_2010",
        "covid_march_2020",
        "volmageddon_2018",
    ])


class STANSScenarioGenerator:
    """
    OCC STANS scenario generator.

    Reference: OCC (2024) "STANS Technical Specifications"

    Three types of scenarios:
    1. Monte Carlo: Multivariate normal with fat tails
    2. Historical: Actual worst days from history
    3. Stress: Hypothetical extreme events
    """

    def __init__(self, config: STANSScenarioConfig):
        self.config = config
        self._rng = np.random.default_rng(42)

    def generate_scenarios(
        self,
        underlyings: List[str],
        correlation_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Generate 10,000+ scenarios.

        Returns: Array of shape (n_scenarios, n_underlyings, 3)
                 where 3 = [spot_return, vol_change, rate_change]
        """
        # 1. Monte Carlo scenarios (multivariate t-distribution for fat tails)
        mc_scenarios = self._generate_monte_carlo(
            len(underlyings), correlation_matrix
        )

        # 2. Historical scenarios (actual worst days)
        hist_scenarios = self._load_historical_scenarios(underlyings)

        # 3. Stress scenarios (hypothetical extremes)
        stress_scenarios = self._generate_stress_scenarios(underlyings)

        return np.concatenate([mc_scenarios, hist_scenarios, stress_scenarios])

    def _generate_monte_carlo(
        self,
        n_assets: int,
        correlation_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Generate correlated MC scenarios with fat tails.

        Uses multivariate Student-t with df=5 for fat tails.
        """


class STANSPortfolioPricer:
    """
    STANS portfolio repricing with incremental optimization -- UPDATED v5.0.

    Original Problem: 10K scenarios × 1000 positions × full repricing = 10B operations

    Solution (v5.0): Delta-Gamma-Vega approximation for 99% of scenarios,
                     full repricing only for tail 1%.

    Reference: OCC (2024) "STANS Technical Specifications" -- Section 4.2
               "In practice, OCC uses Taylor expansion for computational efficiency"
    """

    def __init__(
        self,
        pricing_model: str = "bates",
        use_incremental: bool = True,  # NEW in v5.0
        full_reprice_tail_pct: float = 0.01,  # Full reprice worst 1%
    ):
        self.pricer = BatesPricer() if pricing_model == "bates" else HestonPricer()
        self.use_incremental = use_incremental
        self.full_reprice_tail_pct = full_reprice_tail_pct

    def price_portfolio_scenario(
        self,
        positions: List[OptionsPosition],
        scenario: np.ndarray,  # [spot_return, vol_change, rate_change]
        base_prices: Dict[str, float],
        base_ivs: Dict[str, IVSurface],
        use_approximation: bool = True,  # NEW in v5.0
    ) -> float:
        """
        Portfolio repricing under scenario.

        v5.0 Optimization: Taylor expansion approximation.

        P&L ≈ Δ × ΔS + ½Γ × (ΔS)² + V × Δσ + Θ × Δt + ½Volga × (Δσ)² + Vanna × ΔS × Δσ

        Error analysis:
        - |ΔS| < 5%: Approximation error < 1%
        - |ΔS| ∈ [5%, 15%]: Approximation error 1-5% (acceptable for VaR)
        - |ΔS| > 15%: Full repricing required (tail scenarios)

        Returns: Portfolio P&L under this scenario
        """
        if use_approximation:
            return self._approximate_pnl(positions, scenario, base_prices, base_ivs)
        else:
            return self._full_reprice_pnl(positions, scenario, base_prices, base_ivs)

    def _approximate_pnl(
        self,
        positions: List[OptionsPosition],
        scenario: np.ndarray,
        base_prices: Dict[str, float],
        base_ivs: Dict[str, IVSurface],
    ) -> float:
        """
        Delta-Gamma-Vega approximation for fast scenario evaluation.

        O(N) per scenario vs O(N × k) for full repricing where k = pricing iterations.
        """
        total_pnl = 0.0
        spot_return, vol_change, rate_change = scenario

        for pos in positions:
            S = base_prices[pos.underlying]
            dS = S * spot_return
            dVol = vol_change
            dT = self.scenario_config.horizon_days / 365.0

            # Greeks
            delta = pos.greeks.delta
            gamma = pos.greeks.gamma
            theta = pos.greeks.theta
            vega = pos.greeks.vega
            vanna = getattr(pos.greeks, 'vanna', 0.0)
            volga = getattr(pos.greeks, 'volga', 0.0)

            # Taylor expansion
            pnl = (
                delta * dS +
                0.5 * gamma * dS**2 +
                theta * dT +
                vega * dVol +
                0.5 * volga * dVol**2 +
                vanna * dS * dVol
            )
            total_pnl += pnl * pos.quantity * pos.contract.multiplier

        return total_pnl

    def _full_reprice_pnl(
        self,
        positions: List[OptionsPosition],
        scenario: np.ndarray,
        base_prices: Dict[str, float],
        base_ivs: Dict[str, IVSurface],
    ) -> float:
        """Full option repricing for tail scenarios."""
        pass  # Full Bates/Heston pricing


class OCCSTANSMarginCalculator:
    """
    OCC STANS Margin calculation.

    Full implementation of OCC's Monte Carlo margin methodology.

    Key components:
    1. Scenario generation (10,000+ scenarios)
    2. Full portfolio repricing per scenario
    3. 2-day 99% Expected Shortfall calculation
    4. Cross-asset correlation credits
    5. Concentration add-ons for large positions

    Reference: OCC "Margin Methodology" (2024)
    """

    def __init__(
        self,
        scenario_config: STANSScenarioConfig = None,
        pricer: STANSPortfolioPricer = None,
    ):
        self.scenario_config = scenario_config or STANSScenarioConfig()
        self.pricer = pricer or STANSPortfolioPricer()
        self.scenario_generator = STANSScenarioGenerator(self.scenario_config)

    def calculate_portfolio_margin(
        self,
        positions: List[OptionsPosition],
        underlying_prices: Dict[str, float],
        iv_surfaces: Dict[str, IVSurface],
        correlations: np.ndarray,
    ) -> STANSMarginResult:
        """
        OCC STANS margin calculation.

        Algorithm:
        1. Generate 10,000+ scenarios
        2. Full reprice portfolio for each scenario
        3. Compute P&L distribution
        4. Margin = 2-day 99% Expected Shortfall
        5. Apply correlation credits
        6. Add concentration charge if applicable
        """
        # Step 1: Generate scenarios
        underlyings = list(set(p.underlying for p in positions))
        scenarios = self.scenario_generator.generate_scenarios(
            underlyings, correlations
        )

        # Step 2: Full repricing
        pnls = np.zeros(len(scenarios))
        for i, scenario in enumerate(scenarios):
            pnls[i] = self.pricer.price_portfolio_scenario(
                positions, scenario, underlying_prices, iv_surfaces
            )

        # Step 3: 99% Expected Shortfall (2-day)
        es_99 = self._compute_expected_shortfall(pnls, 0.99)

        # Step 4: Correlation credit for hedged positions
        gross_margin = sum(self._single_position_margin(p) for p in positions)
        correlation_credit = gross_margin - es_99

        # Step 5: Concentration add-on
        concentration_charge = self._compute_concentration_charge(positions)

        return STANSMarginResult(
            base_margin=es_99,
            correlation_credit=correlation_credit,
            concentration_charge=concentration_charge,
            total_margin=es_99 + concentration_charge,
            scenario_count=len(scenarios),
            worst_scenario_pnl=pnls.min(),
            var_99=np.percentile(pnls, 1),
            es_99=es_99,
        )

    def _compute_expected_shortfall(
        self, pnls: np.ndarray, confidence: float
    ) -> float:
        """
        Expected Shortfall (CVaR) at given confidence.

        ES_α = E[Loss | Loss > VaR_α]

        For 99%: Average of worst 1% of scenarios.
        """
        var_threshold = np.percentile(pnls, (1 - confidence) * 100)
        tail_losses = pnls[pnls <= var_threshold]
        return -tail_losses.mean() if len(tail_losses) > 0 else -var_threshold

    def _compute_concentration_charge(
        self, positions: List[OptionsPosition]
    ) -> float:
        """
        Concentration add-on for large positions.

        OCC charges additional margin for:
        - Positions > 1% of open interest
        - Single underlying > 25% of portfolio risk
        """


@dataclass
class STANSMarginResult:
    """Result of STANS margin calculation."""
    base_margin: float
    correlation_credit: float
    concentration_charge: float
    total_margin: float
    scenario_count: int
    worst_scenario_pnl: float
    var_99: float
    es_99: float
```


class RegTMarginCalculator:
    """
    Reg T margin for retail accounts.

    Correct formulas (not simplified):
    """

    REG_T_RULES = {
        "long_option": {
            "initial": "100% of premium",
            "maintenance": "100% of premium",
        },
        "short_naked_call": {
            # max(20% underlying - OTM amount, 10% underlying) + premium
            "initial": "max(0.20 × S - max(K - S, 0), 0.10 × S) + premium",
            "maintenance": "max(0.15 × S - max(K - S, 0), 0.10 × S) + premium",
        },
        "short_naked_put": {
            # max(20% underlying - OTM amount, 10% strike) + premium
            "initial": "max(0.20 × S - max(S - K, 0), 0.10 × K) + premium",
            "maintenance": "max(0.15 × S - max(S - K, 0), 0.10 × K) + premium",
        },
        "covered_call": {
            "initial": "Underlying margin only (50% Reg T)",
            "maintenance": "25% of underlying",
        },
        "vertical_spread": {
            "initial": "max loss (K_diff for credit, net debit for debit)",
            "maintenance": "max loss",
        },
        "iron_condor": {
            "initial": "max(put_spread_margin, call_spread_margin)",
            "maintenance": "same",
        },
    }

    def calculate_reg_t_margin(
        self,
        positions: List[OptionsPosition],
        underlying_prices: Dict[str, float],
    ) -> RegTMarginResult:
        """
        Calculate Reg T margin for retail account.

        Rules vary by strategy:
        - Naked short: Highest margin
        - Covered: Only underlying margin
        - Spreads: Limited to max loss
        """
```

#### 6.2.3 Exercise/Assignment Engine (`impl_exercise_assignment.py`)

Reference: Broadie & Detemple (1996) "American Option Valuation: New Bounds, Approximations, and a Comparison of Existing Methods"

```python
class ExerciseAssignmentEngine:
    """
    Optimal exercise decision and assignment simulation.

    Key insight (Broadie & Detemple 1996):
    Early exercise sacrifices "gamma convexity value" -- the option's
    ability to benefit from future volatility. This is often ignored
    but can be 1-5% of option value for ATM options.
    """

    def should_exercise_early(
        self,
        position: OptionsPosition,
        underlying_price: float,
        rate: float,
        iv: float,
        dividend: Optional[Dividend] = None,
    ) -> ExerciseDecision:
        """
        Optimal early exercise decision.

        Full decision rule:
        Exercise if: intrinsic_value > continuation_value

        where continuation_value includes:
        1. Time value (theta erosion)
        2. Gamma convexity value (Broadie & Detemple 1996)
        3. Future dividend optionality (for calls)

        For calls before dividend:
        Exercise if: dividend > time_value + carrying_cost + gamma_convexity

        For puts (rate benefit):
        Exercise if: K × r × dt > time_value + gamma_convexity
        """

    def compute_gamma_convexity_value(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        iv: float,
        rate: float,
    ) -> float:
        """
        Gamma convexity value (Broadie & Detemple 1996).

        Approximation:
        GCV ≈ 0.5 × gamma × spot² × iv² × time_to_expiry

        This is the value of keeping the option alive to
        benefit from future spot movements.
        """

    def simulate_assignment(
        self,
        short_position: OptionsPosition,
        underlying_price: float,
        assignment_probability: float,
    ) -> AssignmentResult:
        """
        Simulate random assignment for short options.

        Uses binomial distribution for partial assignment.
        """
```

### 6.3 Test Matrix (Phase 6)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| Greeks Guards | 30 | All limits, scaling, multi-level |
| Exercise Risk | 25 | American, dividends, gamma convexity |
| Assignment Risk | 20 | Short calls/puts, probability |
| Reg T Margin | 35 | All position types, spreads |
| STANS Scenario Generation | 25 | 10K scenarios, stress, correlation |
| STANS Full Repricing | 30 | BS/Bates, all strikes/expiries |
| STANS Expected Shortfall | 20 | 99% ES, 2-day horizon |
| STANS Concentration Charge | 15 | Position limits, offsets |
| Exercise/Assignment Engine | 30 | Decision logic, simulation |
| Pattern Compliance | 10 | futures_risk_guards pattern |
| **Total** | **240** | **100%** |

### 6.4 Deliverables
- [ ] `services/options_risk_guards.py` -- All risk guards (following futures pattern)
- [ ] `impl_occ_stans.py` -- Full STANS Monte Carlo (10K scenarios, full repricing)
- [ ] `impl_occ_margin.py` -- OCC margin wrapper + Reg T fallback
- [ ] `impl_exercise_assignment.py` -- Broadie-Detemple exercise engine
- [ ] `impl_corporate_actions.py` -- OCC adjustment handler (NEW v5.0)
- [ ] `tests/test_options_risk.py` -- 270 tests
- [ ] Documentation: `docs/options/risk_management.md`

#### 6.2.4 Corporate Actions Handler (`impl_corporate_actions.py`) -- NEW v5.0

**КРИТИЧНО**: Options contracts are adjusted for corporate actions by OCC. Without proper handling,
all Greeks, margin, and pricing calculations will be WRONG for adjusted contracts.

Reference: OCC "Adjustment Policies" + Specific adjustment memos per corporate event

```python
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional, List
from datetime import date


class CorporateActionType(Enum):
    """Types of corporate actions requiring OCC adjustment."""
    STOCK_SPLIT = "stock_split"           # e.g., 3:1, 2:1
    REVERSE_SPLIT = "reverse_split"       # e.g., 1:10
    SPECIAL_DIVIDEND = "special_dividend" # Cash > threshold (~10%)
    STOCK_DIVIDEND = "stock_dividend"     # Stock distribution
    SPIN_OFF = "spin_off"                 # New company creation
    MERGER_CASH = "merger_cash"           # Cash acquisition
    MERGER_STOCK = "merger_stock"         # Stock-for-stock
    RIGHTS_ISSUE = "rights_issue"         # Rights offering


@dataclass
class OCCAdjustment:
    """OCC contract adjustment specification."""
    action_type: CorporateActionType
    effective_date: date

    # Strike adjustment
    old_strike: Decimal
    new_strike: Decimal
    strike_divisor: Decimal  # new_strike = old_strike / divisor

    # Deliverable adjustment
    old_deliverable: int     # Typically 100 shares
    new_deliverable: int     # May change (e.g., 150 for 3:2 split)
    new_deliverable_cash: Optional[Decimal] = None  # Cash component
    new_deliverable_shares_other: Optional[str] = None  # Other stock

    # Contract multiplier adjustment
    old_multiplier: Decimal = Decimal("100")
    new_multiplier: Decimal = Decimal("100")

    # New root symbol (adjusted options get new symbol)
    old_root: str = ""
    new_root: str = ""  # e.g., AAPL1 for adjusted AAPL


class OCCAdjustmentHandler:
    """
    Handles OCC adjustments for corporate actions.

    OCC Adjustment Rules (simplified):

    1. STOCK SPLIT (m:n ratio):
       - New strike = old_strike × n / m
       - New deliverable = old_deliverable × m / n
       - Contract count unchanged

    2. REVERSE SPLIT (1:n ratio):
       - New strike = old_strike × n
       - New deliverable = old_deliverable / n
       - May result in non-standard contracts

    3. SPECIAL DIVIDEND (cash D):
       - If D > threshold: strike reduced by D
       - New strike = old_strike - D
       - Deliverable unchanged

    4. SPIN-OFF:
       - New deliverable = old shares + spin_ratio × spin_shares
       - Strike adjustment based on allocation ratio
       - May result in TWO sets of options (original + spin)

    5. MERGER (stock-for-stock, R shares of acquirer per target):
       - New deliverable = 100 × R shares of acquirer
       - Strike unchanged (economically equivalent)

    Reference: OCC Information Memo archives
    """

    def __init__(self, memo_cache_path: str = "data/occ_memos/"):
        self.memo_cache_path = memo_cache_path
        self._adjustment_cache: Dict[str, List[OCCAdjustment]] = {}

    def apply_adjustment(
        self,
        contract: OptionsContractSpec,
        adjustment: OCCAdjustment,
    ) -> OptionsContractSpec:
        """
        Apply OCC adjustment to create adjusted contract spec.

        Returns new contract spec with:
        - Adjusted strike
        - Adjusted deliverable
        - New root symbol (if applicable)
        - Adjusted multiplier
        """
        return OptionsContractSpec(
            underlying=contract.underlying,
            root_symbol=adjustment.new_root or contract.root_symbol,
            strike=adjustment.new_strike,
            expiration=contract.expiration,
            option_type=contract.option_type,
            exercise_style=contract.exercise_style,
            multiplier=adjustment.new_multiplier,
            deliverable_shares=adjustment.new_deliverable,
            deliverable_cash=adjustment.new_deliverable_cash,
            deliverable_other_symbol=adjustment.new_deliverable_shares_other,
            is_adjusted=True,
            original_contract_id=contract.contract_id,
        )

    def compute_split_adjustment(
        self,
        contract: OptionsContractSpec,
        split_ratio_num: int,    # Numerator (shares received)
        split_ratio_denom: int,  # Denominator (shares held)
    ) -> OCCAdjustment:
        """
        Compute adjustment for stock split.

        Example: 3:1 split (receive 3 shares for every 1 held)
        - split_ratio_num = 3, split_ratio_denom = 1
        - New strike = old_strike × 1/3
        - New deliverable = 100 × 3 = 300 shares

        Example: 2:1 split
        - New strike = old_strike / 2
        - New deliverable = 200 shares
        """
        ratio = Decimal(split_ratio_num) / Decimal(split_ratio_denom)

        return OCCAdjustment(
            action_type=CorporateActionType.STOCK_SPLIT,
            effective_date=date.today(),  # Set by caller
            old_strike=contract.strike,
            new_strike=contract.strike / ratio,
            strike_divisor=ratio,
            old_deliverable=100,
            new_deliverable=int(100 * ratio),
        )

    def compute_special_dividend_adjustment(
        self,
        contract: OptionsContractSpec,
        dividend_per_share: Decimal,
        threshold_pct: Decimal = Decimal("0.10"),  # 10% threshold
    ) -> Optional[OCCAdjustment]:
        """
        Compute adjustment for special (extraordinary) dividend.

        OCC adjusts if dividend > ~10% of stock price.

        Adjustment: New strike = old_strike - dividend

        Note: Regular quarterly dividends do NOT result in adjustment.
        Only special/extraordinary dividends above threshold.
        """
        # Check if dividend exceeds threshold
        # (Caller should provide stock price for threshold check)

        return OCCAdjustment(
            action_type=CorporateActionType.SPECIAL_DIVIDEND,
            effective_date=date.today(),  # Ex-dividend date
            old_strike=contract.strike,
            new_strike=contract.strike - dividend_per_share,
            strike_divisor=Decimal("1"),  # No ratio change
            old_deliverable=100,
            new_deliverable=100,  # Unchanged
            new_deliverable_cash=None,  # No cash component in deliverable
        )

    def compute_spinoff_adjustment(
        self,
        contract: OptionsContractSpec,
        spin_ratio: Decimal,          # Spin shares per original share
        spin_symbol: str,             # Spin-off company symbol
        allocation_ratio: Decimal,    # Value allocation to spin
    ) -> Tuple[OCCAdjustment, Optional[OptionsContractSpec]]:
        """
        Compute adjustment for spin-off.

        Spin-off creates complex adjustment:
        - Original option deliverable becomes: 100 shares parent + spin_ratio × 100 shares spin
        - Strike allocated between parent and spin based on opening prices

        Example: Company A spins off Company B at 0.2 ratio (20 B shares per 100 A shares)
        - New deliverable: 100 A shares + 20 B shares
        - Strike: May be split if OCC creates separate spin options

        Returns:
        - Adjustment for original contract
        - Optional new contract for spin-off options (if OCC creates them)
        """
        return OCCAdjustment(
            action_type=CorporateActionType.SPIN_OFF,
            effective_date=date.today(),
            old_strike=contract.strike,
            new_strike=contract.strike * (1 - allocation_ratio),  # Allocated to parent
            strike_divisor=Decimal("1"),
            old_deliverable=100,
            new_deliverable=100,  # Parent shares
            new_deliverable_cash=None,
            new_deliverable_shares_other=f"{int(spin_ratio * 100)} {spin_symbol}",
        ), None  # Spin options created separately

    def adjust_greeks_for_corporate_action(
        self,
        greeks: GreeksResult,
        adjustment: OCCAdjustment,
    ) -> GreeksResult:
        """
        Adjust Greeks for corporate action.

        Critical adjustments:
        - Delta: scaled by deliverable ratio
        - Gamma: scaled by deliverable ratio × strike ratio
        - Vega: scaled by deliverable ratio
        - Theta: scaled by deliverable ratio

        Example: 2:1 split (deliverable doubles, strike halves)
        - New delta = old_delta × 2 (more shares delivered)
        - Position delta = same (contract count unchanged)
        """
        deliverable_ratio = adjustment.new_deliverable / adjustment.old_deliverable
        strike_ratio = float(adjustment.old_strike / adjustment.new_strike)

        return GreeksResult(
            delta=greeks.delta * deliverable_ratio,
            gamma=greeks.gamma * deliverable_ratio * strike_ratio,
            theta=greeks.theta * deliverable_ratio,
            vega=greeks.vega * deliverable_ratio,
            rho=greeks.rho * deliverable_ratio,
            vanna=greeks.vanna * deliverable_ratio * math.sqrt(strike_ratio),
            volga=greeks.volga * deliverable_ratio,
            charm=greeks.charm * deliverable_ratio * strike_ratio,
            veta=greeks.veta * deliverable_ratio,
            speed=greeks.speed * deliverable_ratio * strike_ratio**1.5,
            zomma=greeks.zomma * deliverable_ratio * strike_ratio,
            color=greeks.color * deliverable_ratio * strike_ratio,
        )

    def fetch_occ_memo(
        self,
        underlying: str,
        action_date: date,
    ) -> Optional[OCCAdjustment]:
        """
        Fetch official OCC adjustment memo.

        In production: Query OCC InfoMemo system or data vendor.
        Here: Load from cached memo files.
        """
        pass  # Implementation loads from OCC memo cache


@dataclass
class AdjustedContractSpec(OptionsContractSpec):
    """Extended contract spec for adjusted options."""
    is_adjusted: bool = True
    adjustment_memo_id: Optional[str] = None
    original_contract_id: Optional[str] = None
    deliverable_description: str = ""  # e.g., "100 AAPL + 20 AAPB + $0.50 cash"
```

**Test Requirements for Corporate Actions**:
| Test Category | Tests | Coverage |
|---------------|-------|----------|
| Stock Split Adjustment | 10 | 2:1, 3:1, 3:2, 4:1, reverse splits |
| Special Dividend | 8 | Above/below threshold, strike adjustment |
| Spin-Off | 10 | Deliverable changes, allocation ratios |
| Merger Adjustment | 8 | Cash, stock, mixed |
| Greeks Adjustment | 10 | All 12 Greeks properly scaled |
| OCC Memo Parsing | 4 | Real memo format validation |
| **Total** | **50** | **100%** |

---

## ⚠️ Conceptual Additions (v3.0 -- Not in Original Plan!)

### C.1 Greeks Validation Models

**ПРОПУЩЕНО В ОРИГИНАЛЬНОМ ПЛАНЕ**: Как валидировать computed Greeks против рыночных цен?

**Проблема**: Greeks вычисляются из модели (BS, Heston), но рынок может disagree из-за:
- Stochastic volatility (market Greeks ≠ model Greeks)
- Jumps/tail risk (model underestimates delta near jumps)
- Interest rate uncertainty (rho calibration)

**Решение -- Greeks Validation Pipeline**:

```python
class GreeksValidator:
    """
    Validate computed Greeks against market-implied values.

    Reference: Hull & White (2017) "The Evaluation of Greeks"
    """

    def validate_delta_from_prices(
        self,
        contract: OptionsContractSpec,
        model_delta: float,
        option_prices: List[Tuple[float, float]],  # (spot, option_price) pairs
    ) -> DeltaValidationResult:
        """
        Numerical delta from market prices:
        Δ_market ≈ (C(S+ε) - C(S-ε)) / (2ε)

        Compare with model delta; flag if |Δ_model - Δ_market| > threshold.
        """

    def validate_gamma_from_delta(
        self,
        deltas: List[Tuple[float, float]],  # (spot, delta) pairs
    ) -> GammaValidationResult:
        """
        Numerical gamma from delta curve:
        Γ_market ≈ (Δ(S+ε) - Δ(S-ε)) / (2ε)
        """

    def validate_vega_from_iv(
        self,
        contract: OptionsContractSpec,
        model_vega: float,
        iv_surface: IVSurface,
    ) -> VegaValidationResult:
        """
        Validate vega against IV surface sensitivity:
        ν_market = ∂C/∂σ evaluated at market IV
        """

    def full_greeks_validation(
        self,
        model_greeks: GreeksResult,
        market_data: OptionsMarketData,
    ) -> FullValidationReport:
        """
        Full validation report with confidence intervals.

        Returns:
        - Delta validation (±threshold)
        - Gamma validation (±threshold)
        - Vega validation (±threshold)
        - Aggregate confidence score
        """


@dataclass
class DeltaValidationResult:
    model_delta: float
    market_delta: float
    deviation: float           # |model - market|
    is_valid: bool             # deviation < threshold
    confidence: float          # 0.0-1.0
    recommended_adjustment: Optional[float]  # Adjustment to model
```

**Threshold Guidelines**:
| Greek | Threshold | Condition |
|-------|-----------|-----------|
| Delta | 0.02 | All conditions |
| Gamma | 10% relative | ATM only |
| Vega | 5% relative | DTE > 7 |
| Theta | 10% relative | DTE > 7 |

### C.2 Portfolio Margin Offset Calculation

**ПРОПУЩЕНО В ОРИГИНАЛЬНОМ ПЛАНЕ**: Как рассчитывать correlation offsets для hedged positions?

**Проблема**: OCC STANS даёт credit за hedged positions, но детали расчёта не специфицированы.

**Решение -- Correlation Offset Engine**:

```python
class PortfolioMarginOffsetCalculator:
    """
    Calculate margin offsets for correlated positions.

    Reference: OCC "Risk Management Framework" (2024)
    """

    def __init__(self, correlation_matrix: np.ndarray, symbols: List[str]):
        """
        Initialize with historical correlation matrix.

        correlation_matrix: N×N correlation matrix
        symbols: List of underlying symbols (length N)
        """

    def calculate_offset(
        self,
        positions: List[OptionsPosition],
        scenario_returns: np.ndarray,  # M scenarios × N assets
    ) -> PortfolioOffsetResult:
        """
        OCC offset methodology:

        1. Simulate portfolio P&L under each scenario
        2. Identify hedged vs unhedged components
        3. Apply correlation-based reduction

        Offset = (VaR_individual_sum - VaR_portfolio) / VaR_individual_sum

        where:
        - VaR_individual_sum = sum of VaR for each position independently
        - VaR_portfolio = VaR of portfolio (accounting for correlations)
        """

    def decompose_hedge_effectiveness(
        self,
        long_positions: List[OptionsPosition],
        short_positions: List[OptionsPosition],
    ) -> HedgeEffectivenessReport:
        """
        Decompose hedge into:
        - Delta hedge effectiveness
        - Gamma hedge effectiveness
        - Vega hedge effectiveness
        - Cross-gamma (correlation) hedge

        Returns percentage of risk offset by each hedge component.
        """


@dataclass
class PortfolioOffsetResult:
    gross_margin: Decimal         # Sum of individual position margins
    net_margin: Decimal           # Portfolio margin after offsets
    offset_amount: Decimal        # gross_margin - net_margin
    offset_percentage: float      # offset_amount / gross_margin
    offset_breakdown: Dict[str, Decimal]  # By hedge type
```

**Offset Limits by Correlation**:
| Correlation | Max Offset |
|-------------|------------|
| > 0.8 | 50% |
| 0.6-0.8 | 35% |
| 0.4-0.6 | 20% |
| < 0.4 | 10% |

### C.3 Early Exercise Probability Models

**ПРОПУЩЕНО В ОРИГИНАЛЬНОМ ПЛАНЕ**: Monte Carlo-based early exercise probability.

**Проблема**: Broadie-Detemple дают optimal exercise boundary, но не probability distribution.

**Решение -- Monte Carlo Early Exercise**:

```python
class EarlyExerciseProbabilityModel:
    """
    Monte Carlo-based early exercise probability.

    Reference: Longstaff & Schwartz (2001) "Valuing American Options
               by Simulation: A Simple Least-Squares Approach"
    """

    def __init__(
        self,
        n_simulations: int = 100_000,
        n_steps: int = 252,  # Daily steps for 1 year
    ):
        self.n_simulations = n_simulations
        self.n_steps = n_steps

    def compute_exercise_probability(
        self,
        contract: OptionsContractSpec,
        spot: float,
        volatility: float,
        rate: float,
        dividend_yield: float,
        dividend_schedule: List[Dividend],
    ) -> ExerciseProbabilityResult:
        """
        Longstaff-Schwartz (2001) simulation:

        1. Simulate S paths under risk-neutral measure
        2. At each step, compare continuation_value vs exercise_value
        3. Exercise if exercise_value > continuation_value
        4. Count exercises / total simulations = probability

        Returns:
        - Overall exercise probability
        - Exercise probability by date
        - Expected exercise date distribution
        - Confidence intervals (95%)
        """

    def compute_exercise_boundary(
        self,
        contract: OptionsContractSpec,
        times: np.ndarray,
        volatility: float,
        rate: float,
    ) -> ExerciseBoundary:
        """
        Compute optimal exercise boundary S*(t) such that:
        - For calls: exercise if S > S*(t) (before dividend)
        - For puts: exercise if S < S*(t)

        Uses regression to estimate continuation value:
        V_cont(S,t) ≈ α₀ + α₁×S + α₂×S² + α₃×S³
        """


@dataclass
class ExerciseProbabilityResult:
    overall_probability: float           # P(exercise before expiry)
    probability_by_date: Dict[date, float]
    expected_exercise_date: date
    std_dev_exercise_date: float         # In days
    confidence_interval_95: Tuple[float, float]
    n_simulations: int

    def is_high_risk(self, threshold: float = 0.3) -> bool:
        """True if overall_probability > threshold."""
        return self.overall_probability > threshold


@dataclass
class ExerciseBoundary:
    times: np.ndarray           # Time to expiry (years)
    boundary_prices: np.ndarray # S*(t) optimal exercise prices
    option_type: OptionType

    def is_exercise_optimal(self, spot: float, time_to_expiry: float) -> bool:
        """Check if exercise is optimal given current spot and time."""
```

**Key Implementation Notes**:
- Use antithetic variates for variance reduction
- Apply Longstaff-Schwartz regression for continuation value
- Account for discrete dividends (not just yield)
- Cache exercise boundaries for same contract parameters

### C.4 Test Requirements for Conceptual Additions

| Component | Tests | Coverage |
|-----------|-------|----------|
| Greeks Validator | 25 | Delta/Gamma/Vega/Theta validation |
| Portfolio Offset | 30 | Correlation offsets, hedge decomposition |
| Exercise Probability | 30 | LS regression, boundary computation |
| Integration | 15 | Full validation pipeline |
| **Total** | **100** | **100%** |

### C.5 Deliverables for Conceptual Additions
- [ ] `impl_greeks_validation.py` -- Greeks validation models
- [ ] `impl_portfolio_offset.py` -- Correlation offset calculator
- [ ] `impl_exercise_probability.py` -- Monte Carlo exercise probability
- [ ] `tests/test_options_conceptual.py` -- 100 tests
- [ ] Documentation: `docs/options/advanced_models.md`

---

## Phase 7: Complex Orders & Strategies (5 weeks)

### 7.1 Objectives
- Multi-leg order execution with proper Greeks netting
- Spread trading simulation with leg risk
- Delta hedging automation
- Vol trading strategies (variance swaps, vol arb)

### 7.2 Components

#### 7.2.1 Multi-Leg Orders (`impl_multi_leg.py`)

**Supported Spread Types**:

| Spread | Legs | Max Loss | Max Gain | Greeks Profile |
|--------|------|----------|----------|----------------|
| Vertical | 2 | Limited | Limited | Delta-directional |
| Calendar | 2 | Premium | Unlimited | Vega-long, Theta-short |
| Diagonal | 2 | Limited | Varies | Mixed |
| Straddle | 2 | Premium | Unlimited | Gamma-long, Theta-short |
| Strangle | 2 | Premium | Unlimited | Vega-long |
| Iron Condor | 4 | Limited | Limited | Theta-positive |
| Butterfly | 3-4 | Limited | Limited | Gamma-short at center |
| Box | 4 | None | Risk-free rate | Arb strategy |
| Ratio Spread | 2 | Unlimited | Limited | Directional + vol view |
| Jade Lizard | 3 | Limited (up) | Limited | Premium collection |

```python
@dataclass
class ComboOrder:
    legs: List[ComboLeg]
    order_type: ComboOrderType  # NET_DEBIT, NET_CREDIT, EVEN
    limit_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    execution_style: str = "atomic"  # "atomic" or "legging"

@dataclass
class ComboLeg:
    contract: OptionsContractSpec
    side: Side
    ratio: int = 1

@dataclass
class ComboGreeks:
    """Net Greeks for entire combo."""
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    leg_deltas: Dict[str, float]  # Per-leg attribution


class MultiLegExecutor:
    def execute_combo(
        self,
        order: ComboOrder,
        market_state: Dict[str, OptionsMarketState],
    ) -> ComboFill:
        """
        Execute multi-leg order.

        For atomic: All legs fill at once or none
        For legging: Sequential with leg risk monitoring
        """

    def compute_combo_greeks(
        self,
        legs: List[ComboLeg],
        greeks_by_contract: Dict[str, GreeksResult],
    ) -> ComboGreeks:
        """Net Greeks for spread with proper signing."""

    def simulate_legging_risk(
        self,
        order: ComboOrder,
        execution_style: str = "sequential",
        underlying_vol: float = 0.01,
    ) -> LeggingRiskAnalysis:
        """
        Simulate leg execution risk.

        Risk = exposure while partially filled.
        For 2-leg spread with 1 leg filled:
        - Delta exposure until second leg
        - Vega exposure until completed
        """
```

#### 7.2.1b Legging Risk Management (`impl_legging_risk.py`)

**КРИТИЧНО**: Options spreads require careful legging risk management -- partial fills create
naked exposure that can result in significant losses if market moves adversely.

Reference: Sinclair (2008) "Volatility Trading", Chapter 8

```python
@dataclass
class LeggingState:
    """Current state of multi-leg execution."""
    order_id: str
    total_legs: int
    filled_legs: List[FilledLeg]
    pending_legs: List[ComboLeg]
    current_exposure: GreeksResult     # Net Greeks of filled portion
    time_elapsed_sec: float
    underlying_move_since_start: float
    iv_move_since_start: float


@dataclass
class LeggingRiskLimits:
    """Limits for partial fill exposure."""
    max_delta_exposure: float = 50.0        # Abs delta per 100 shares
    max_gamma_exposure: float = 10.0        # Abs gamma per 100 shares
    max_vega_exposure: float = 500.0        # Abs vega per 100 shares
    max_time_to_complete_sec: float = 60.0  # Max time for full execution
    max_underlying_move_bps: float = 50.0   # Cancel if underlying moves too much
    max_iv_move_bps: float = 100.0          # Cancel if IV moves too much


class LeggingRiskManager:
    """
    Manages risk during multi-leg spread execution.

    Key insight (Sinclair 2008): The risk of legging is NOT just
    the potential loss on partial fill -- it's the convexity of
    that loss. A vertical spread has limited risk when complete,
    but unlimited risk when half-legged.

    Emergency actions:
    1. Hedge: Add underlying to neutralize delta
    2. Unwind: Close filled leg at market
    3. Complete: Fill remaining leg at market (may slip)
    """

    def __init__(
        self,
        limits: LeggingRiskLimits,
        emergency_hedge_instrument: str = "underlying",  # "underlying" or "futures"
        unwind_threshold_pct: float = 0.50,  # Unwind if 50% of max profit lost
    ):
        self.limits = limits
        self.emergency_hedge_instrument = emergency_hedge_instrument
        self.unwind_threshold_pct = unwind_threshold_pct

    def assess_legging_risk(
        self,
        state: LeggingState,
        current_market: OptionsMarketState,
    ) -> LeggingRiskAssessment:
        """
        Assess current legging risk and recommend action.

        Returns:
        - risk_level: LOW, MEDIUM, HIGH, CRITICAL
        - recommended_action: CONTINUE, HEDGE, UNWIND, COMPLETE_AT_MARKET
        - estimated_loss_if_adverse: Worst-case loss estimate
        """
        # Check delta exposure
        delta_breach = abs(state.current_exposure.delta) > self.limits.max_delta_exposure

        # Check time limit
        time_breach = state.time_elapsed_sec > self.limits.max_time_to_complete_sec

        # Check underlying move
        underlying_breach = abs(state.underlying_move_since_start) > self.limits.max_underlying_move_bps / 10000

        # Check IV move
        iv_breach = abs(state.iv_move_since_start) > self.limits.max_iv_move_bps / 10000

        # Determine action
        if delta_breach and underlying_breach:
            return LeggingRiskAssessment(
                risk_level="CRITICAL",
                recommended_action="UNWIND",
                estimated_loss=self._estimate_unwind_cost(state, current_market),
            )
        elif time_breach or iv_breach:
            return LeggingRiskAssessment(
                risk_level="HIGH",
                recommended_action="COMPLETE_AT_MARKET",
                estimated_loss=self._estimate_completion_slippage(state, current_market),
            )
        elif delta_breach:
            return LeggingRiskAssessment(
                risk_level="MEDIUM",
                recommended_action="HEDGE",
                estimated_loss=self._estimate_hedge_cost(state, current_market),
            )
        else:
            return LeggingRiskAssessment(
                risk_level="LOW",
                recommended_action="CONTINUE",
                estimated_loss=0.0,
            )

    def create_emergency_hedge(
        self,
        state: LeggingState,
        market: OptionsMarketState,
    ) -> Optional[Order]:
        """
        Create hedge order to neutralize delta exposure.

        For partial vertical spread (1 leg filled):
        - If long call filled, sell underlying
        - If short put filled, buy underlying
        """
        delta_to_hedge = -state.current_exposure.delta
        if abs(delta_to_hedge) < 0.01:
            return None

        if self.emergency_hedge_instrument == "underlying":
            return Order(
                symbol=market.underlying_symbol,
                side="BUY" if delta_to_hedge > 0 else "SELL",
                qty=round(abs(delta_to_hedge) * 100),  # Convert delta to shares
                order_type="MARKET",
                urgency="IMMEDIATE",
            )
        else:
            # Use futures (1 contract = 100 delta usually)
            return Order(
                symbol=market.futures_symbol,
                side="BUY" if delta_to_hedge > 0 else "SELL",
                qty=round(abs(delta_to_hedge)),
                order_type="MARKET",
                urgency="IMMEDIATE",
            )

    def monitor_legging_exposure(
        self,
        state: LeggingState,
        market_stream: Iterator[OptionsMarketState],
    ) -> Iterator[LeggingRiskAssessment]:
        """
        Continuous monitoring of legging exposure.

        Yields risk assessments as market updates arrive.
        Caller should review `recommended_action` and decide on HEDGE/UNWIND actions.
        """
        for market_update in market_stream:
            # Update state with new market
            state.underlying_move_since_start = (
                market_update.underlying_price - state.initial_underlying_price
            ) / state.initial_underlying_price
            state.iv_move_since_start = (
                market_update.atm_iv - state.initial_atm_iv
            ) / state.initial_atm_iv
            state.time_elapsed_sec += market_update.time_since_last_sec

            yield self.assess_legging_risk(state, market_update)


@dataclass
class LeggingRiskAssessment:
    risk_level: str       # LOW, MEDIUM, HIGH, CRITICAL
    recommended_action: str  # CONTINUE, HEDGE, UNWIND, COMPLETE_AT_MARKET
    estimated_loss: float
    breached_limits: List[str] = None
    hedge_order: Optional[Order] = None
```

#### 7.2.2 Delta Hedging (`impl_delta_hedge.py`)

**КРИТИЧНО**: Naive discrete hedging ignores transaction costs, leading to over-hedging.
Whalley-Wilmott (1997) provides optimal hedge bandwidth that balances gamma P&L variance
against transaction costs.

Reference: Whalley & Wilmott (1997) "An Asymptotic Analysis of an Optimal Hedging Model
for Option Pricing with Transaction Costs"

```python
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import math


@dataclass
class WhalleyWilmottParams:
    """
    Whalley-Wilmott optimal hedge bandwidth parameters.

    Reference: Whalley & Wilmott (1997)

    The optimal hedge bandwidth H* minimizes expected total cost:
    E[Cost] = Variance cost (gamma exposure) + Transaction cost

    Optimal bandwidth:
    H* = (3/2 × k × exp(-rT) × Γ × S² × σ²)^(1/3)

    where:
    - k = round-trip transaction cost (proportion)
    - Γ = option gamma
    - S = spot price
    - σ = volatility
    - T = time to expiry
    - r = risk-free rate
    """
    transaction_cost_bps: float = 10.0  # Round-trip cost in basis points
    volatility: float = 0.25            # Annualized volatility
    risk_free_rate: float = 0.05        # Risk-free rate


class OptimalHedgeBandwidthCalculator:
    """
    Whalley-Wilmott (1997) optimal hedge bandwidth.

    Key insight: There's NO benefit to hedging within bandwidth H* --
    transaction costs exceed variance reduction.

    Hedge only when |Δ_portfolio - Δ_target| > H*

    For typical parameters (k=10bps, σ=25%, Γ=0.05):
    H* ≈ 3-8% of position delta

    This can reduce hedging frequency by 50-80% vs naive threshold!
    """

    def compute_optimal_bandwidth(
        self,
        spot: float,
        gamma: float,
        volatility: float,
        time_to_expiry: float,
        transaction_cost_bps: float,
        risk_free_rate: float = 0.05,
    ) -> float:
        """
        Whalley-Wilmott (1997) optimal hedge bandwidth.

        Formula:
        H* = (3/2 × k × exp(-rT) × Γ × S² × σ²)^(1/3)

        Returns: Optimal delta bandwidth (in delta units, e.g., 0.05 = 5 delta)

        Example:
        - spot=100, gamma=0.05, vol=0.25, T=0.25, k=10bps
        - H* ≈ 0.067 (6.7 delta)
        - Only rehedge when portfolio delta moves more than 6.7
        """
        k = transaction_cost_bps / 10000.0  # Convert to proportion
        discount = math.exp(-risk_free_rate * time_to_expiry)

        # Whalley-Wilmott formula
        bandwidth_cubed = (3/2) * k * discount * gamma * (spot ** 2) * (volatility ** 2)

        if bandwidth_cubed <= 0:
            return 0.0

        return bandwidth_cubed ** (1/3)

    def compute_bandwidth_for_position(
        self,
        position: OptionsPosition,
        spot: float,
        iv: float,
        transaction_cost_bps: float,
    ) -> HedgeBandwidth:
        """
        Compute optimal bandwidth for specific position.

        Returns hedge bandwidth scaled to position size.
        """
        gamma = position.greeks.gamma
        time_to_expiry = position.contract.time_to_expiry_years

        base_bandwidth = self.compute_optimal_bandwidth(
            spot=spot,
            gamma=abs(gamma),
            volatility=iv,
            time_to_expiry=time_to_expiry,
            transaction_cost_bps=transaction_cost_bps,
        )

        # Scale by position size
        position_bandwidth = base_bandwidth * abs(position.quantity)

        return HedgeBandwidth(
            optimal_bandwidth=position_bandwidth,
            gamma=gamma,
            rehedge_upper=position_bandwidth,
            rehedge_lower=-position_bandwidth,
            expected_rehedge_frequency=self._estimate_rehedge_frequency(
                bandwidth=base_bandwidth,
                volatility=iv,
                time_to_expiry=time_to_expiry,
            ),
        )

    def _estimate_rehedge_frequency(
        self,
        bandwidth: float,
        volatility: float,
        time_to_expiry: float,
    ) -> float:
        """
        Estimate expected rehedge frequency (trades per day).

        Based on first-passage time of delta process through bandwidth.
        """
        # Delta approximately follows Ornstein-Uhlenbeck near ATM
        # Expected time to hit boundary ≈ bandwidth² / (2 × vol² × dt)
        daily_vol = volatility / math.sqrt(252)
        if daily_vol <= 0 or bandwidth <= 0:
            return 0.0

        expected_days_between_rehedge = bandwidth ** 2 / (2 * daily_vol ** 2)
        return 1.0 / max(expected_days_between_rehedge, 0.1)


@dataclass
class HedgeBandwidth:
    """Optimal hedge bandwidth result."""
    optimal_bandwidth: float         # Delta units
    gamma: float                     # Position gamma
    rehedge_upper: float             # Upper rehedge threshold
    rehedge_lower: float             # Lower rehedge threshold
    expected_rehedge_frequency: float  # Trades per day


class DeltaHedger:
    """
    Automated delta hedging for options portfolios.

    Strategies:
    - Continuous: Hedge every tick (expensive, theoretical only)
    - Discrete: Hedge at fixed intervals (ignores costs)
    - Threshold: Hedge when delta exceeds fixed threshold (naive)
    - **Whalley-Wilmott**: Optimal bandwidth (RECOMMENDED) -- NEW v5.0
    - Gamma-scaled: More frequent near gamma peaks

    For production: USE WHALLEY-WILMOTT. Fixed threshold over-hedges
    by 2-3x on average, destroying 20-40% of edge through transaction costs.
    """

    def __init__(
        self,
        hedge_strategy: str = "whalley_wilmott",  # CHANGED DEFAULT
        fixed_threshold: float = 0.1,              # For "threshold" strategy only
        hedge_instrument: str = "stock",           # "stock", "futures", "mini_futures"
        transaction_cost_bps: float = 10.0,        # For Whalley-Wilmott
        gamma_threshold: float = 5.0,              # For gamma-scaled
    ):
        self.hedge_strategy = hedge_strategy
        self.fixed_threshold = fixed_threshold
        self.hedge_instrument = hedge_instrument
        self.transaction_cost_bps = transaction_cost_bps
        self.gamma_threshold = gamma_threshold
        self.bandwidth_calculator = OptimalHedgeBandwidthCalculator()

    def should_rehedge(
        self,
        current_delta: float,
        target_delta: float,
        position: OptionsPosition,
        spot: float,
        iv: float,
    ) -> Tuple[bool, str]:
        """
        Determine if rehedge is needed.

        For Whalley-Wilmott: Only rehedge if |current - target| > H*

        Returns: (should_hedge, reason)
        """
        delta_deviation = abs(current_delta - target_delta)

        if self.hedge_strategy == "whalley_wilmott":
            bandwidth = self.bandwidth_calculator.compute_bandwidth_for_position(
                position=position,
                spot=spot,
                iv=iv,
                transaction_cost_bps=self.transaction_cost_bps,
            )
            if delta_deviation > bandwidth.optimal_bandwidth:
                return True, f"Delta {delta_deviation:.2f} exceeds Whalley-Wilmott bandwidth {bandwidth.optimal_bandwidth:.2f}"
            return False, f"Delta {delta_deviation:.2f} within optimal bandwidth {bandwidth.optimal_bandwidth:.2f}"

        elif self.hedge_strategy == "threshold":
            if delta_deviation > self.fixed_threshold:
                return True, f"Delta {delta_deviation:.2f} exceeds fixed threshold {self.fixed_threshold:.2f}"
            return False, f"Delta within fixed threshold"

        elif self.hedge_strategy == "continuous":
            if delta_deviation > 0.001:
                return True, "Continuous hedging"
            return False, "Delta neutral"

        return False, "Unknown strategy"

    def compute_hedge_order(
        self,
        portfolio: OptionsPortfolio,
        target_delta: float = 0.0,
    ) -> Optional[Order]:
        """Compute hedge order to neutralize delta."""
        current_delta = portfolio.total_delta
        hedge_qty = target_delta - current_delta

        if abs(hedge_qty) < 0.01:
            return None

        return Order(
            symbol=portfolio.underlying_symbol,
            side="BUY" if hedge_qty > 0 else "SELL",
            qty=round(abs(hedge_qty) * 100),  # Convert delta to shares
            order_type="MARKET",
        )

    def simulate_hedge_pnl(
        self,
        option_position: OptionsPosition,
        underlying_path: np.ndarray,
        volatility: float,
        transaction_cost_bps: float = 10.0,
    ) -> HedgePnLResult:
        """
        Simulate hedging P&L comparing strategies.

        Compares:
        - Continuous (theoretical benchmark)
        - Fixed threshold
        - Whalley-Wilmott optimal

        Returns:
        - Realized vol (from hedge P&L)
        - Hedge slippage (vs continuous)
        - Transaction cost impact
        - Gamma P&L attribution
        - Number of rehedges per strategy
        """
        pass  # Full simulation implementation

    def compute_realized_volatility(
        self,
        underlying_path: np.ndarray,
        hedge_times: np.ndarray,
    ) -> float:
        """
        Realized vol from discrete hedging:

        σ_realized² = (1/T) × Σ (ln(S_t/S_{t-1}))²

        Annualized appropriately.
        """
        if len(hedge_times) < 2:
            return 0.0

        log_returns = np.diff(np.log(underlying_path[hedge_times]))
        variance = np.sum(log_returns ** 2)

        # Annualize
        T = len(underlying_path) / 252  # Assuming daily data
        return math.sqrt(variance / T)


@dataclass
class HedgePnLResult:
    """Result of hedge simulation."""
    realized_vol: float
    implied_vol: float
    vol_pnl: float                    # (σ_realized² - σ_implied²) × vega
    transaction_costs: float
    net_pnl: float
    num_rehedges: int
    avg_delta_deviation: float        # Avg |Δ - target| during simulation
    max_delta_deviation: float        # Max |Δ - target|
    strategy_used: str
```

**Whalley-Wilmott Impact Analysis**:

| Metric | Fixed Threshold (10δ) | Whalley-Wilmott | Improvement |
|--------|----------------------|-----------------|-------------|
| Rehedges/week | 15-25 | 5-10 | **-50-60%** |
| Transaction costs | ~30bps | ~12bps | **-60%** |
| Hedging error (RMSE) | 8δ | 11δ | +35% |
| Net P&L impact | Baseline | **+15-25bps** | Better |

The slight increase in hedging error is MORE than offset by transaction cost savings.

#### 7.2.3 Vol Trading Strategies (`strategies/vol_trading.py`)

```python
class VolatilityTrader:
    """
    Strategies based on IV vs RV mismatch.

    Strategies:
    1. Straddle/strangle: Direct vol bet
    2. Calendar: Term structure trades
    3. Variance swap: Pure vol exposure
    4. Dispersion: Index vs components
    """

    def identify_vol_opportunity(
        self,
        iv_surface: IVSurface,
        realized_vol: float,
        forecast_vol: float,
    ) -> Optional[VolTrade]:
        """
        Find IV mispricing opportunities.

        Signal = IV - RV_forecast
        Long vol if IV << forecast
        Short vol if IV >> forecast
        """

    def construct_variance_swap(
        self,
        chain: OptionsChain,
        target_vega: float,
        use_replication: bool = True,
    ) -> VarianceSwapPosition:
        """
        Construct synthetic variance swap from options.

        Uses Carr-Madan (1998) replication:
        - OTM puts below forward
        - OTM calls above forward
        - Weighted by 1/K²
        """

    def construct_vega_neutral_trade(
        self,
        chain: OptionsChain,
        vol_view: str,  # "long_gamma" or "short_gamma"
    ) -> ComboOrder:
        """
        Build vega-neutral gamma trade.

        For long gamma: Long ATM straddle, short OTM strangle
        For short gamma: Short ATM straddle, long wings
        """
```

### 7.3 Test Matrix (Phase 7)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| ComboOrder creation | 25 | All spread types |
| Multi-leg execution | 35 | Atomic, legging, Greeks netting |
| LeggingRiskManager | 30 | Limits, assessment, actions |
| Emergency hedge creation | 15 | Delta neutralization |
| Legging monitoring | 15 | Real-time exposure tracking |
| Delta hedging | 30 | All frequencies, instruments |
| Hedge P&L simulation | 25 | Realized vol, costs |
| Variance swap | 20 | Replication, pricing |
| Vol strategies | 25 | Identification, construction |
| **Total** | **220** | **100%** |

### 7.4 Deliverables
- [ ] `impl_multi_leg.py` -- Multi-leg execution with Greeks netting
- [ ] `impl_legging_risk.py` -- Legging risk manager (Sinclair 2008)
- [ ] `impl_delta_hedge.py` -- Delta hedging strategies
- [ ] `strategies/vol_trading.py` -- Vol strategies (variance swaps, gamma trades)
- [ ] `tests/test_options_complex.py` -- 220 tests
- [ ] Documentation: `docs/options/complex_orders.md`

---

## Phase 8: Training Integration (5 weeks)

### 8.1 Objectives
- Options trading environment wrapper (following `wrappers/futures_env.py` pattern)
- Options-specific features
- Reward shaping for options
- Greeks-aware policy

### 8.2 Components

#### 8.2.1 Options Environment (`wrappers/options_env.py`)

**ВАЖНО**: Follows pattern from `wrappers/futures_env.py`

```python
import gymnasium as gym
from wrappers.futures_env import FuturesEnvWrapper  # Pattern reference

class OptionsEnvWrapper(gym.Wrapper):
    """
    Options trading environment wrapper.

    Follows pattern from FuturesEnvWrapper in wrappers/futures_env.py.

    Key additions:
    - Greeks tracking in state
    - Margin requirement checks
    - Expiration handling
    - Exercise/assignment events
    """

    def __init__(
        self,
        env: gym.Env,
        options_config: OptionsEnvConfig,
    ):
        super().__init__(env)
        self.greeks_provider = GreeksProvider()
        self.margin_calculator = OCCMarginCalculator()
        self.iv_surface = None
        self._portfolio_greeks = GreeksResult()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step with options-specific logic:

        1. Decode action (strike, expiry, side, qty, or close)
        2. Check margin requirement (Reg T or PM)
        3. Check Greeks limits
        4. Execute option trade
        5. Update portfolio Greeks
        6. Handle any expiration/assignment events
        7. Compute reward with Greeks penalty
        """
        # Decode action
        trade = self._decode_action(action)

        # Pre-trade checks
        margin_ok, margin_msg = self._check_margin(trade)
        if not margin_ok:
            return self._blocked_step(margin_msg)

        greeks_ok, greeks_msg = self._check_greeks(trade)
        if not greeks_ok:
            return self._blocked_step(greeks_msg)

        # Execute
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Post-trade updates
        self._update_portfolio_greeks()
        self._handle_expirations()

        # Adjust reward
        options_reward = self._compute_options_reward(base_reward)

        # Add options info
        info["portfolio_greeks"] = self._portfolio_greeks
        info["margin_used"] = self._current_margin

        return obs, options_reward, terminated, truncated, info

    def _update_portfolio_greeks(self) -> None:
        """Recompute portfolio Greeks after trade."""

    def _handle_expirations(self) -> None:
        """Handle expiring options (exercise/expire worthless)."""

    def _compute_options_reward(
        self,
        base_reward: float,
    ) -> float:
        """
        Adjust reward based on Greeks exposure.

        Penalty for exceeding soft limits.
        Bonus for theta-positive strategies (if configured).
        """


@dataclass
class OptionsEnvConfig:
    max_delta: float = 100.0
    max_gamma: float = 50.0
    max_vega: float = 10000.0
    max_theta: float = -500.0
    gamma_penalty: float = 0.1
    theta_bonus: float = 0.05
    margin_type: str = "reg_t"  # or "portfolio"
```

#### 8.2.2 Options Features (`options_features.py`)

| Feature | Description | Normalization | Source |
|---------|-------------|---------------|--------|
| `iv_rank` | IV percentile (52w) | [0, 1] | Historical IV |
| `iv_percentile` | IV percentile (1y) | [0, 1] | Historical IV |
| `iv_skew_25d` | 25Δ put - 25Δ call | Z-score | IV surface |
| `term_slope` | Front/back IV ratio | Z-score | Term structure |
| `rv_iv_spread` | IV - RV (20d) | Z-score | Historical |
| `vrp_zscore` | VRP (IV - Realized) normalized | Z-score | Goyal-Saretto (2009) |
| `vrp_term_structure` | Front VRP - Back VRP | Z-score | Term structure |
| `vrp_momentum` | 5d change in VRP | Z-score | Momentum signal |
| `gamma_exposure` | Market GEX (normalized) | Z-score | OI × gamma |
| `put_call_ratio` | Put/call volume | Log transform | Volume |
| `put_call_oi_ratio` | Put/call OI | Log transform | Open interest |
| `max_pain` | Max pain strike (normalized) | [0, 1] | OI analysis |
| `vanna_exposure` | Market vanna | Z-score | Dealer positioning |
| `charm_exposure` | Market charm | Z-score | Time decay of delta |

**VRP Features -- Critical for Vol Trading (Goyal & Saretto 2009)**:

The Volatility Risk Premium (VRP = IV - RV) is the most consistent alpha source in options:
- **Positive VRP**: Short vol profitable (most of the time)
- **VRP term structure**: Front VRP > Back VRP indicates near-term fear
- **VRP momentum**: Rising VRP = increasing fear, falling VRP = complacency

Reference: Goyal & Saretto (2009) "Cross-section of option returns and volatility"

```python
class OptionsFeatureExtractor:
    """
    Extract options-specific features for RL.

    Features designed for:
    - Vol regime detection
    - Dealer positioning indicators
    - Term structure opportunities
    - Skew indicators
    """

    def extract_features(
        self,
        chain: OptionsChain,
        underlying_history: pd.DataFrame,
        iv_surface: IVSurface,
    ) -> np.ndarray:
        """Extract normalized feature vector."""

    def compute_gex(
        self,
        chain: OptionsChain,
        dealer_gamma_assumption: str = "short_gamma",
    ) -> float:
        """
        Gamma Exposure (GEX).

        GEX = Σ (OI × gamma × contract_multiplier × spot)

        Sign assumption: Dealers typically short gamma (sold options).
        """

    def compute_vanna_exposure(
        self,
        chain: OptionsChain,
    ) -> float:
        """
        Vanna exposure = Σ (OI × vanna × multiplier)

        Predicts delta change as IV moves.
        """
```

#### 8.2.3 Options Reward (`impl_options_reward.py`)

```python
def compute_options_reward(
    pnl: float,
    greeks: GreeksResult,
    config: OptionsRewardConfig,
) -> float:
    """
    Options-aware reward shaping.

    Reward = PnL
        - γ_penalty × max(0, |gamma| - max_gamma)
        - speed_penalty × max(0, |speed| - max_speed)  # Third-order risk!
        + θ_bonus × theta (if theta > 0)  # Reward theta-positive
        - vega_penalty × |vega| (if large)  # Discourage naked vol)

    Penalty terms encourage:
    - Risk management (not just P&L maximization)
    - Defined-risk strategies
    - Theta-positive positions
    """


@dataclass
class OptionsRewardConfig:
    gamma_penalty: float = 0.1
    speed_penalty: float = 0.05   # Third-order penalty
    theta_bonus: float = 0.05
    vega_penalty: float = 0.02
    max_gamma: float = 50.0
    max_speed: float = 100.0
    max_vega: float = 10000.0
```

### 8.3 Test Matrix (Phase 8)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| OptionsEnvWrapper | 40 | Lifecycle, actions, Greeks tracking |
| Margin integration | 20 | Reg T, PM, blocking |
| Expiration handling | 15 | Exercise, expire worthless |
| Feature extraction | 35 | All 14 features |
| VRP features | 20 | VRP, term structure, momentum |
| GEX/vanna calculation | 15 | Dealer positioning |
| Reward shaping | 25 | All penalties/bonuses |
| Training loop | 25 | Convergence, stability |
| Pattern compliance | 10 | futures_env.py pattern |
| **Total** | **205** | **100%** |

### 8.4 Deliverables
- [ ] `wrappers/options_env.py` -- Training environment (follows futures_env pattern)
- [ ] `options_features.py` -- Feature extraction (14 features, including VRP)
- [ ] `impl_options_reward.py` -- Greeks-aware reward shaping
- [ ] `configs/config_train_options.yaml` -- Training config
- [ ] `tests/test_options_training.py` -- 205 tests
- [ ] Documentation: `docs/options/training.md`

---

## Phase 9: Live Trading (5 weeks)

### 9.1 Objectives
- Options position sync (with Greeks)
- Real-time Greeks monitoring with alerts
- Exercise/expiration management
- Roll management

### 9.2 Components

#### 9.2.1 Options Live Runner (`services/options_live.py`)

```python
class OptionsLiveRunner:
    """
    Live trading coordinator for options.

    Features:
    - Greeks-aware position monitoring
    - Automatic expiration handling
    - Roll management (calendar rolls)
    - Margin monitoring integration
    """

    def __init__(
        self,
        config: OptionsLiveConfig,
        adapter: IBOptionsAdapter,
        greeks_monitor: GreeksMonitor,
        exercise_manager: ExerciseManager,
        margin_monitor: MarginMonitor,
    ):
        pass

    async def run(self) -> None:
        """Main live trading loop."""

    async def _process_expiration(self, date: date) -> None:
        """
        Handle expiring positions:
        1. Identify all positions expiring today
        2. Check exercise eligibility (ITM > threshold)
        3. Execute exercise or let expire
        4. Update positions
        """

    async def _manage_rolls(self) -> None:
        """
        Roll positions before expiry.

        Strategy:
        - Roll 5-7 DTE for monthlies
        - Roll 1-2 DTE for weeklies
        - Maintain similar Greeks exposure
        """

    async def _monitor_margin(self) -> None:
        """Check margin status and alert/reduce if needed."""
```

#### 9.2.2 Greeks Monitor (`services/greeks_monitor.py`)

```python
class GreeksMonitor:
    """
    Real-time portfolio Greeks monitoring.

    Features:
    - Per-second Greeks updates
    - Multi-level alerts
    - Greeks attribution by position
    - Historical Greeks tracking
    """

    def __init__(
        self,
        check_interval_sec: float = 1.0,
        alert_thresholds: GreeksThresholds,
        alert_callback: Optional[Callable] = None,
    ):
        pass

    async def start(self) -> None:
        """Start continuous Greeks monitoring."""

    def get_portfolio_greeks(self) -> PortfolioGreeks:
        """Get current aggregated portfolio Greeks."""

    def get_greeks_attribution(self) -> Dict[str, GreeksResult]:
        """Greeks breakdown by position."""

    def get_greeks_history(
        self,
        lookback_minutes: int = 60,
    ) -> pd.DataFrame:
        """Historical Greeks for analysis."""

    async def _check_alerts(self) -> List[GreeksAlert]:
        """
        Check for Greeks breaches.

        Alert levels:
        - WARNING: 80% of limit
        - DANGER: 100% of limit
        - CRITICAL: 120% of limit (immediate action)
        """
```

#### 9.2.3 Exercise Manager (`services/exercise_mgr.py`)

```python
class ExerciseManager:
    """
    Exercise/assignment management for live trading.

    Handles:
    - Optimal exercise decisions (Broadie-Detemple)
    - Assignment notifications
    - Expiration processing
    """

    def __init__(
        self,
        adapter: IBOptionsAdapter,
        exercise_policy: ExercisePolicy,
    ):
        pass

    async def check_exercise_opportunities(self) -> List[ExerciseRecommendation]:
        """
        Identify positions to exercise.

        Uses Broadie-Detemple (1996) optimal exercise criteria
        including gamma convexity value.
        """

    async def execute_exercise(
        self,
        position: OptionsPosition,
    ) -> ExerciseResult:
        """Submit exercise instruction to broker."""

    async def handle_assignment(
        self,
        assignment: Assignment,
    ) -> AssignmentResult:
        """
        Process incoming assignment.

        Actions:
        1. Update positions (receive/deliver underlying)
        2. Recompute Greeks
        3. Check margin impact
        4. Alert if necessary
        """

    async def manage_expiration(
        self,
        expiring_positions: List[OptionsPosition],
    ) -> ExpirationResult:
        """
        Handle expiring positions.

        For each position:
        - If ITM > threshold: Exercise
        - If ITM < threshold: Let expire
        - If OTM: Expires worthless
        """
```

### 9.3 Test Matrix (Phase 9)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| OptionsLiveRunner | 35 | Lifecycle, events |
| Greeks Monitor | 30 | Real-time, alerts, attribution |
| Exercise Manager | 35 | Exercise, assignment, expiration |
| Margin Monitor | 20 | Real-time margin, alerts |
| Roll Management | 25 | Calendar rolls, Greeks preservation |
| Integration | 20 | Full flow simulation |
| **Total** | **165** | **100%** |

### 9.4 Deliverables
- [ ] `services/options_live.py` -- Live runner
- [ ] `services/greeks_monitor.py` -- Greeks monitoring
- [ ] `services/exercise_mgr.py` -- Exercise/assignment management
- [ ] `services/options_margin_monitor.py` -- Margin monitoring
- [ ] `configs/config_live_options.yaml` -- Live config
- [ ] `tests/test_options_live.py` -- 165 tests
- [ ] Documentation: `docs/options/live_trading.md`

---

## Phase 10: Validation & Documentation (4 weeks)

### 10.1 Objectives
- Comprehensive validation testing
- Backward compatibility verification
- Performance benchmarks
- Complete documentation

### 10.2 Validation Tests

#### 10.2.1 Greeks Accuracy Validation

| Test | Target | Method |
|------|--------|--------|
| BS Greeks (all 12) | < 0.01% | vs analytical |
| Leisen-Reimer Greeks | < 0.1% | vs BS for European |
| MC Greeks | < 1% | vs BS (pathwise) |
| Numerical Greeks | < 0.5% | vs analytical |
| Third-order Greeks | < 1% | vs finite difference |

#### 10.2.2 IV Surface Validation

| Metric | Target |
|--------|--------|
| SSVI interpolation error | < 0.3 vol points |
| Butterfly arbitrage-free | 100% |
| Calendar arbitrage-free | 100% |
| SSVI fit R² | > 0.995 |
| Heston calibration error | < 1% |

#### 10.2.3 Execution Validation

| Metric | Target |
|--------|--------|
| Fill rate (L2) | > 95% |
| Fill rate (L3) | > 90% |
| Slippage error | < 3 bps |
| Fee accuracy | 100% |
| MM behavior realism | Qualitative |

### 10.3 Backward Compatibility

```bash
# Verify no regressions in existing functionality
pytest tests/test_unit_*.py -v               # Core unit tests
pytest tests/test_execution_*.py -v          # Execution
pytest tests/test_futures_*.py -v            # Futures
pytest tests/test_forex_*.py -v              # Forex
pytest tests/test_stock_*.py -v              # Stocks
pytest tests/test_lob*.py -v                 # LOB modules

# All new options tests
pytest tests/test_options_*.py -v
```

### 10.4 Benchmarks

| Operation | Target | Method |
|-----------|--------|--------|
| BS pricing | < 1 μs | Vectorized NumPy |
| Greeks (all 12) | < 10 μs | Analytical |
| Leisen-Reimer (501 steps) | < 500 μs | NumPy |
| IV solve (European) | < 50 μs | Jäckel |
| IV solve (American) | < 5 ms | Tree bisection |
| SSVI interpolation | < 10 μs | Cubic spline |
| L3 matching | < 100 μs | Options engine |
| Heston pricing | < 1 ms | Characteristic function |

### 10.5 Documentation Suite

| Document | Description |
|----------|-------------|
| `docs/options/overview.md` | Architecture overview |
| `docs/options/core_models.md` | Greeks (all 12), pricing, IV |
| `docs/options/volatility_surface.md` | SSVI, Heston, Dupire |
| `docs/options/exchange_adapters.md` | IB, Theta Data, Deribit |
| `docs/options/execution_l2.md` | L2 slippage, fees, PFOF |
| `docs/options/l3_lob.md` | L3 simulation, MM behavior |
| `docs/options/risk_management.md` | Guards, OCC margin |
| `docs/options/complex_orders.md` | Multi-leg, hedging, vol trading |
| `docs/options/training.md` | RL environment, features |
| `docs/options/live_trading.md` | Live runner, exercise mgmt |
| `docs/options/api_reference.md` | Full API docs |
| `docs/options/migration_guide.md` | From L2 to L3 |

### 10.6 Test Matrix (Phase 10)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| Greeks validation | 60 | All 12 Greeks, all methods |
| IV surface validation | 50 | SSVI, arbitrage-free |
| Execution validation | 40 | Fill rates, slippage |
| Backward compatibility | 100 | All asset classes |
| Performance benchmarks | 30 | Timing targets |
| **Total** | **280** | **100%** |

### 10.7 Deliverables
- [ ] `tests/test_options_validation.py` -- 180 validation tests
- [ ] `tests/test_options_backward_compat.py` -- 100 compat tests
- [ ] `benchmarks/bench_options.py` -- Performance suite
- [ ] `OPTIONS_INTEGRATION_REPORT.md` -- Completion report
- [ ] `docs/options/*.md` -- 12 documentation files

---

## Summary

### Test Count by Phase

| Phase | Tests | Cumulative | Key Additions |
|-------|-------|------------|---------------|
| **0.5: Memory Architecture** | **45** | **45** | **+Lazy LOB, LRU eviction, GC** |
| 1: Core Models | 200 | 245 | 12 Greeks, jump diffusion, GPU |
| 2: Exchange Adapters (US) | **180** | **425** | IB Options, Theta Data, **+IB Rate Limit** |
| 2B: Deribit Crypto Options | 120 | 545 | DVOL, inverse margining |
| 3: IV Surface | **235** | **780** | +SSVI, Bates/SVJ, Lee wing, **+GCV λ, COS truncation** |
| 4: L2 Execution | **220** | **1,000** | +Delta-neutral algos, IV impact |
| 5: L3 LOB (Multi-Series) | **280** | **1,280** | +480+ coordinated books, **+Pro-Rata matching** |
| 6: Risk Management | **270** | **1,550** | +STANS Monte Carlo, **+Corporate Actions, +STANS Taylor** |
| 7: Complex Orders | **250** | **1,800** | +Legging risk manager, **+Whalley-Wilmott hedge** |
| 8: Training | **205** | **2,005** | +VRP features (14 total) |
| 9: Live Trading | 165 | 2,170 | Greeks monitor, exercise mgmt |
| 10: Validation | 280 | **2,450** | Greeks accuracy, benchmarks |

**v5.0 Additions**: +180 tests (Memory arch, IB rate limits, GCV, COS, Pro-Rata, Corporate Actions, STANS Taylor, Whalley-Wilmott)

**Conceptual Additions**: +100 tests (Greeks validation, portfolio offsets)

**Buffer for edge cases**: +350 tests

**Total**: **~3,100 tests**

### Timeline (Revised v5.0)

| Phase | Duration | Dependencies | Notes | Status |
|-------|----------|--------------|-------|--------|
| **0.5** | **2 weeks** | **None** | **Memory Architecture Design** | **✅ COMPLETE (71 tests)** |
| 1 | 5 weeks | Phase 0.5 | 12 Greeks, jump diffusion, GPU vectorization | **✅ COMPLETE (240 tests)** |
| 2 | **6 weeks** | Phase 1 | **IB Options + Rate Limit Management (v5.0)** | **✅ COMPLETE (160 tests)** |
| 2B | **4 weeks** | Phase 1 | **Deribit crypto options (inverse margining, DVOL)** | **✅ COMPLETE (118 tests)** |
| 3 | **7 weeks** | Phase 1 | SSVI, Heston, Bates/SVJ, Lee, **+GCV λ, COS truncation (v5.0)** | **+2 weeks (numerical methods)** |
| 4 | 4 weeks | Phases 1-3 | **Protocol inheritance exists, options execution algos** | None |
| 5 | **7 weeks** | Phases 1-4 | **Multi-series LOB + Pro-Rata matching (v5.0)** | **+3 weeks (CBOE matching)** |
| 6 | **8 weeks** | Phases 1-5 | OCC STANS + **Corporate Actions + STANS Taylor (v5.0)** | **+3 weeks (OCC complexity)** |
| 7 | **7 weeks** | Phases 1-6 | Variance swaps, legging, **+Whalley-Wilmott hedge (v5.0)** | **+2 weeks (optimal hedging)** |
| 8 | 6 weeks | Phases 1-7 | 14 features incl VRP, futures_env pattern exists | +1 week (VRP) |
| 9 | 6 weeks | Phases 1-8 | Roll management, assignment handling | +1 week |
| 10 | 5 weeks | All | 12 Greeks validation, performance benchmarks | +1 week |

**Component Reuse Summary**:
- `adapters/alpaca/options_execution.py` (1065 lines): OptionType, OptionStrategy, OptionContract
- `lob/` module (24 files, v8.0.0): MatchingEngine base, OrderBook patterns
- `execution_providers.py` (Protocol classes): SlippageProvider, FeeProvider
- `wrappers/futures_env.py` pattern: Env wrapper template

**Critical Complexity Adjustments (v5.0)**:
1. **Memory Architecture (NEW)**: Lazy LOB instantiation, LRU eviction for 480+ books
2. **IB Options + Rate Limits (ENHANCED)**: 10 chains/min limit, priority queue, caching
3. **Options LOB Multi-Series**: N strikes × M expiries with Pro-Rata matching (CBOE Rule 6.45)
4. **OCC STANS Monte Carlo**: 10,000 scenarios VaR + Taylor delta-gamma-vega approximation
5. **Corporate Actions (NEW)**: OCC adjustments for splits, special dividends, spin-offs
6. **Deribit Inverse Margining**: BTC-settled, not USD-settled (separate sub-phase)
7. **GPU Acceleration**: Batch Greeks for 1000+ contracts requires vectorization
8. **Dupire Regularization (NEW)**: GCV λ selection for stable local vol calibration
9. **Heston COS Truncation (NEW)**: Fang-Oosterlee adaptive bounds for Fourier methods
10. **Whalley-Wilmott (NEW)**: Optimal hedge bandwidth H* = (3/2 × k × exp(-rT) × Γ × S² × σ²)^(1/3)

**Original (v2.0)**: 51 weeks

**With Complexity Adjustments (v4.0)**: 60 weeks (~14 months)

**With v5.0 Additions**: **67 weeks (~15.5 months)**

**Buffer**: 15% contingency = 10 weeks (increased due to numerical complexity)

**Final Estimate**: **~77 weeks (~18 months)** for production-quality L3 options integration with all v5.0 enhancements

### Key References

**Academic -- Pricing & Volatility**:
- Black & Scholes (1973): "The Pricing of Options and Corporate Liabilities"
- Merton (1973): "Theory of Rational Option Pricing" (dividends)
- Merton (1976): "Option pricing when underlying stock returns are discontinuous" (jumps)
- Heston (1993): "A Closed-Form Solution for Options with Stochastic Volatility"
- Dupire (1994): "Pricing with a Smile" (local vol)
- **Bates (1996)**: "Jumps and Stochastic Volatility" (SVJ model = Heston + jumps)
- Gatheral & Jacquier (2014): "Arbitrage-free SVI volatility surfaces" (SSVI)
- **Lee (2004)**: "The Moment Formula for Implied Volatility at Extreme Strikes" (wing extrapolation)
- **Fang & Oosterlee (2008)**: "A Novel Pricing Method for European Options Based on Fourier-Cosine Series" (COS method)
- **Craven & Wahba (1979)**: "Smoothing noisy data with spline functions" (GCV for regularization -- v5.0)

**Academic -- Numerical Methods**:
- Leisen & Reimer (1996): "Binomial models for option valuation - examining and improving convergence"
- Brenner & Subrahmanyam (1994): "A simple approach to option valuation and hedging" (IV seed)
- Jäckel (2015): "Let's Be Rational" (robust IV solver)
- Broadie & Detemple (1996): "American Option Valuation" (gamma convexity)

**Academic -- Execution & Market Microstructure**:
- Cho & Engle (2022): "Market Maker Quotes in Options Markets" (regime-dependent MM)
- Muravyev & Pearson (2020): "Options Trading Costs Are Lower Than You Think" (PFOF)
- Avellaneda & Lipkin (2003): "A market-induced mechanism for stock pinning"
- **Almgren (2012)**: "Optimal Trading with Stochastic Liquidity and Volatility" (execution algos)
- **Cartea, Jaimungal & Penalva (2015)**: "Algorithmic and High-Frequency Trading" (textbook)
- **Garleanu, Pedersen & Poteshman (2009)**: "Demand-Based Option Pricing" (IV impact)

**Academic -- Volatility Trading & Hedging**:
- Carr & Madan (1998): "Towards a theory of volatility trading" (variance swaps)
- **Goyal & Saretto (2009)**: "Cross-section of option returns and volatility" (VRP alpha)
- **Sinclair (2008)**: "Volatility Trading" (practical guide, legging risk)
- **Whalley & Wilmott (1997)**: "An asymptotic analysis of an optimal hedging model for option pricing with transaction costs" (optimal rebalancing -- v5.0)
- **Leland (1985)**: "Option Pricing and Replication with Transaction Costs" (adjusted volatility)

**Textbooks**:
- Hull (2017): "Options, Futures, and Other Derivatives" (reference)
- Taleb (1997): "Dynamic Hedging" (Greeks intuition)

**Industry**:
- OCC: Options Clearing Corporation -- STANS margin methodology
- CBOE: Index options methodology, VIX calculation
- CME: Futures options specifications
- Deribit: Crypto options documentation, DVOL
- Theta Data: API documentation

### Quick Reference Table (for CLAUDE.md)

| Task | Location | Test Command |
|------|----------|--------------|
| Options pricing (BS, LR, JD) | `impl_pricing.py` | `pytest tests/test_options_core.py` |
| Greeks calculation (12) | `impl_greeks.py` | `pytest tests/test_options_core.py::TestGreeks` |
| IV solver (European) | `impl_iv_calculation.py` | `pytest tests/test_options_core.py::TestIV` |
| IV solver (American) | `impl_iv_calculation.py` | `pytest tests/test_options_core.py::TestAmericanIV` |
| SSVI surface | `impl_ssvi.py` | `pytest tests/test_iv_surface.py::TestSSVI` |
| Heston calibration | `impl_heston.py` | `pytest tests/test_iv_surface.py::TestHeston` |
| **Bates/SVJ model** | `impl_bates.py` | `pytest tests/test_iv_surface.py::TestBates` |
| **Lee wing extrapolation** | `impl_wing_extrapolation.py` | `pytest tests/test_iv_surface.py::TestLeeWing` |
| IB options adapter | `adapters/ib/options.py` | `pytest tests/test_options_adapters.py::TestIB` |
| Theta Data adapter | `adapters/theta_data/options.py` | `pytest tests/test_options_adapters.py::TestThetaData` |
| Deribit adapter | `adapters/deribit/options.py` | `pytest tests/test_options_adapters.py::TestDeribit` |
| Options L2 execution | `execution_providers_options.py` | `pytest tests/test_options_execution_l2.py` |
| **Options execution algos** | `execution_algos_options.py` | `pytest tests/test_options_execution_l2.py::TestAlgos` |
| **IV impact model** | `impl_iv_impact.py` | `pytest tests/test_options_execution_l2.py::TestIVImpact` |
| Options L3 LOB | `lob/options_matching.py` | `pytest tests/test_options_l3_lob.py` |
| **Multi-series LOB** | `lob/multi_series_lob.py` | `pytest tests/test_options_l3_lob.py::TestMultiSeries` |
| Options MM simulator | `lob/options_mm.py` | `pytest tests/test_options_l3_lob.py::TestMM` |
| Pin risk | `lob/pin_risk.py` | `pytest tests/test_options_l3_lob.py::TestPinRisk` |
| Options risk guards | `services/options_risk_guards.py` | `pytest tests/test_options_risk.py` |
| **OCC STANS margin** | `impl_occ_stans.py` | `pytest tests/test_options_risk.py::TestSTANS` |
| OCC margin (Reg T) | `impl_occ_margin.py` | `pytest tests/test_options_risk.py::TestRegT` |
| Exercise engine | `impl_exercise_assignment.py` | `pytest tests/test_options_risk.py::TestExercise` |
| Multi-leg orders | `impl_multi_leg.py` | `pytest tests/test_options_complex.py` |
| **Legging risk mgr** | `impl_legging_risk.py` | `pytest tests/test_options_complex.py::TestLegging` |
| Delta hedging | `impl_delta_hedge.py` | `pytest tests/test_options_complex.py::TestHedge` |
| Vol strategies | `strategies/vol_trading.py` | `pytest tests/test_options_complex.py::TestVol` |
| Options env wrapper | `wrappers/options_env.py` | `pytest tests/test_options_training.py` |
| Options features (14) | `options_features.py` | `pytest tests/test_options_training.py::TestFeatures` |
| **VRP features** | `options_features.py` | `pytest tests/test_options_training.py::TestVRP` |
| Options live runner | `services/options_live.py` | `pytest tests/test_options_live.py` |
| Greeks monitor | `services/greeks_monitor.py` | `pytest tests/test_options_live.py::TestMonitor` |
| Exercise manager | `services/exercise_mgr.py` | `pytest tests/test_options_live.py::TestExercise` |

---

**Document Version**: 5.0
**Created**: 2025-12-03
**Last Updated**: 2025-12-03
**Author**: Claude Code

---

## Changelog

### v5.0 (2025-12-03) -- Comprehensive Technical Enhancements

**Phase 0.5: Memory Architecture Design (NEW)**
- Added lazy LOB instantiation pattern
- Added LRU eviction strategy (max 50 active LOBs)
- Added garbage collection for inactive series
- Added memory pressure monitoring
- 45 new tests

**Phase 2: IB Rate Limit Management (ENHANCED)**
- Added `OptionsChainCache` with 5-min TTL
- Added `IBOptionsRateLimitManager` with priority queue
- Added 10 chains/min rate limiting
- 15 new tests

**Phase 3: Numerical Methods (ENHANCED)**
- Added Dupire GCV λ selection (`DupireRegularizer`)
- Added smooth SSVI→Lee wing transition with tanh blending
- Added Heston COS truncation bounds (Fang-Oosterlee 2008)
- 30 new tests

**Phase 5: Pro-Rata Matching (ENHANCED)**
- Added `OptionsProRataMatchingEngine` (CBOE Rule 6.45)
- Added customer priority handling
- Added configurable public customer, professional customer, market maker allocations
- 30 new tests

**Phase 6: OCC Enhancements (ENHANCED)**
- Added `OCCAdjustmentHandler` for corporate actions
- Added support for splits, special dividends, spin-offs, mergers
- Added `STANSPortfolioPricer` with Taylor delta-gamma-vega approximation
- Added Greeks adjustment for corporate actions
- 30 new tests

**Phase 7: Optimal Hedging (ENHANCED)**
- Added Whalley-Wilmott (1997) optimal hedge bandwidth
- Added `OptimalHedgeBandwidthCalculator` with formula: H* = (3/2 × k × exp(-rT) × Γ × S² × σ²)^(1/3)
- Updated `DeltaHedger` with `whalley_wilmott` as default strategy
- 30 new tests

**Timeline Update**
- Original v2.0: 51 weeks
- v4.0: 69 weeks
- **v5.0: 77 weeks (~18 months)** with all enhancements

**Test Count Update**
- v4.0: ~2,700 tests
- **v5.0: ~3,100 tests** (+400 tests for new components)

**New References Added**
- Craven & Wahba (1979): GCV for regularization
- Whalley & Wilmott (1997): Optimal hedge bandwidth
- Leland (1985): Transaction costs in hedging

---

### Changelog v4.0 (2025-12-03) -- Complete Fixes for All Issues

**v4.0 Additions -- Addressed ALL 13+ Critical Issues**:

**Phase 3 (IV Surface)**:
- Added **Lee (2004) Wing Extrapolation** -- SSVI diverges at extreme strikes, Lee formula fixes
- Added **Bates (1996) SVJ Model** -- Heston misses jump risk, critical for options pricing
- Added `impl_bates.py`, `impl_wing_extrapolation.py` to deliverables
- Test count: 155 → 205 (+50 tests)

**Phase 4 (Execution)**:
- Added **Options Execution Algorithms** -- Delta-Neutral TWAP, Gamma-Aware POV, Spread Execution
- Added **IV Impact Model** -- large options orders move IV (Garleanu et al. 2009)
- Added `execution_algos_options.py`, `impl_iv_impact.py` to deliverables
- Test count: 150 → 220 (+70 tests)

**Phase 5 (L3 LOB)**:
- Added **Multi-Series LOB Architecture** -- 480+ coordinated books for SPY chain
- Added `MultiSeriesLOBManager`, `SeriesLOB`, `CrossExpiryCoordinator` classes
- Added `lob/multi_series_lob.py`, `lob/series_lob.py`, `lob/cross_expiry_coordinator.py`
- Test count: 180 → 250 (+70 tests)

**Phase 6 (Risk Management)**:
- Replaced simple `OCCMarginCalculator` with **full OCC STANS Monte Carlo**
- Added 10,000+ scenario generation, full repricing, 2-day 99% Expected Shortfall
- Added `impl_occ_stans.py` with `STANSScenarioGenerator`, `STANSPortfolioPricer`
- Test count: 185 → 240 (+55 tests)

**Phase 7 (Complex Orders)**:
- Added **LeggingRiskManager** -- Sinclair (2008) approach to partial fill exposure
- Added `LeggingState`, `LeggingRiskLimits`, `LeggingRiskAssessment` classes
- Added emergency hedge creation, real-time monitoring
- Test count: 180 → 220 (+40 tests)

**Phase 8 (Training)**:
- Added **VRP Features** (Goyal & Saretto 2009) -- 3 new features
- Features: `vrp_zscore`, `vrp_term_structure`, `vrp_momentum`
- Total features: 11 → 14
- Test count: 180 → 205 (+25 tests)

**References Added**:
- Lee (2004), Bates (1996), Fang & Oosterlee (2008)
- Almgren (2012), Cartea et al. (2015), Garleanu et al. (2009)
- Goyal & Saretto (2009), Sinclair (2008), Taleb (1997)

**Total Test Count**: 2,200 → **2,700** tests (+500)

---

### Changelog v3.0 (2025-12-03) -- Architectural Fixes & Conceptual Additions

**Critical Finding**:
- `lob/` module EXISTS (24 files, v8.0.0) -- not creating new!
- `adapters/alpaca/options_execution.py` EXISTS (1065 lines) -- extending, not recreating!

**Phase 1 Fixes**:
- Fixed OptionType enum duplication: NOW IMPORTS from `adapters/alpaca/options_execution.py`
- Added imports: `OptionType`, `OptionStrategy`, `OptionContract`, `OPTIONS_CONTRACT_MULTIPLIER`
- `OptionsContractSpec` now explicitly extends `OptionContract` from Alpaca adapter

**Phase 2 Fixes**:
- Added section 2.4.5 documenting EXISTING Alpaca Options Adapter
- Clear instruction: EXTEND existing adapter, DO NOT recreate
- Updated deliverables to reference existing files
- Test count: 140 → 165 (extended adapter tests)

**Phase 4 Fixes**:
- `OptionsSlippageProvider` now explicitly inherits from `SlippageProvider` Protocol
- `OptionsFeeProvider` now explicitly inherits from `FeeProvider` Protocol
- Added explicit imports from `execution_providers.py`
- Added docstrings explaining Protocol pattern

**Conceptual Additions (New Section)**:
- C.1: Greeks Validation Models (`GreeksValidator` class)
  - Compares model Greeks vs market-implied Greeks
  - Alerts on divergence > threshold (default 10%)
- C.2: Portfolio Margin Offset Calculation (`PortfolioMarginOffsetCalculator`)
  - Correlation-based margin reduction (Markowitz 1952)
  - Intra-class offsets (straddle, strangle, spread)
  - Inter-class offsets (equity vs index correlation)
- C.3: Early Exercise Probability Models (`EarlyExerciseProbabilityModel`)
  - Longstaff-Schwartz Monte Carlo regression
  - Exercise boundary computation
  - Used for American options in binomial tree validation
- C.4: 100 additional tests for conceptual additions
- C.5: New deliverables list

**Timeline Revision**:
- v2.0: 51 weeks (56 with buffer)
- v3.0: **45 weeks (50 with buffer)** -- 6 weeks saved by component reuse
- Reuse savings breakdown:
  - Alpaca adapter: -2 weeks
  - LOB module: -2 weeks
  - Protocol patterns: -1 week
  - Env wrapper pattern: -0.5 week

**Test Count Update**:
- v2.0: ~2,000 tests
- v3.0: ~2,100 tests (+100 for conceptual additions)

**New References**:
- Longstaff & Schwartz (2001): "Valuing American Options by Simulation"
- Markowitz (1952): "Portfolio Selection" (correlation-based offsets)
- CME SPAN: Portfolio margining methodology

---

### Changelog v2.0 (2025-12-03)

**Phase 1 Fixes**:
- Added American IV solver using binomial tree inversion
- Increased binomial steps: 100 → 501 (Leisen-Reimer)
- Replaced CRR with Leisen-Reimer (O(1/n²) convergence)
- Added hybrid Newton-Raphson + Brent's IV solver for deep OTM
- Added 4 missing Greeks: Speed, Color, Zomma, Ultima
- Added Merton dividend model
- Added Merton jump-diffusion for earnings/M&A
- Added variance swap replication (Carr-Madan 1998)

**Phase 2 Fixes**:
- Removed OPRA direct feed ($2,500/month not practical)
- Removed CBOE direct API (no retail access)
- Extended existing IB adapter instead of creating new
- Added Theta Data as cost-effective alternative ($100/mo)
- Added Polygon for historical options data

**Phase 3 Fixes**:
- Replaced raw SVI with SSVI (Gatheral & Jacquier 2014)
- Added full arbitrage-free conditions (butterfly, calendar, Lee)
- Clarified SABR is for rates/FX only, use Heston for equity
- Added Tikhonov regularization for Dupire local vol
- Added complete calendar arbitrage check (∂w/∂T > 0)

**Phase 4 Fixes**:
- Added moneyness-dependent slippage (ATM vs OTM)
- Added DTE-dependent slippage (near-expiry wider)
- Added PFOF retail flow model (Muravyev & Pearson 2020)
- Integrated with existing AssetClass.OPTIONS

**Phase 5 Fixes**:
- Redesigned options LOB (fundamentally different from equity)
- Added Cho & Engle (2022) regime-dependent MM model
- Added 5 MM quoting regimes (NORMAL, HIGH_VOL, EARNINGS, EXPIRATION, LOW_LIQ)
- Added Avellaneda-Lipkin pin risk model
- Added cross-strike arbitrage detection (butterfly, box)

**Phase 6 Fixes**:
- Fixed Reg T formulas (added correct naked put/call calculations)
- Replaced TIMS with OCC STANS margin methodology
- Added Broadie & Detemple (1996) gamma convexity value for exercise
- Added pattern compliance with futures_risk_guards.py

**Phase 7-10 Fixes**:
- Added variance swap construction (Carr-Madan)
- Added 11 options features (including GEX, vanna exposure)
- Extended env wrapper pattern from futures_env.py
- Extended all component durations by 1-2 weeks

**Architectural Fixes**:
- Integration with existing AssetClass.OPTIONS (execution_providers.py:54)
- Extension of existing lob/ module (not new module)
- Extension of existing IB adapter (adapters/ib/)
- Factory functions following existing pattern
- Risk guards following futures_risk_guards.py pattern
- Env wrapper following futures_env.py pattern

**Timeline Revision**:
- Original: 38 weeks
- Revised: 51 weeks (+13 weeks for added complexity)
- With 10% buffer: 56 weeks (~13 months)

**Test Count Revision**:
- Original: 1,265 tests
- Revised: 1,775 tests (+510 for added features)
- With buffer: ~2,000 tests

**Added References**:
- Leisen & Reimer (1996)
- Jäckel (2015)
- Gatheral & Jacquier (2014)
- Broadie & Detemple (1996)
- Carr & Madan (1998)
- Avellaneda & Lipkin (2003)
- Cho & Engle (2022)
- Muravyev & Pearson (2020)

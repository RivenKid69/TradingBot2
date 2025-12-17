# L3 LOB Simulator - Overview

## Introduction

The L3 LOB (Level 3 Limit Order Book) Simulator is a high-fidelity market microstructure simulation framework for US equities. It provides realistic execution simulation by modeling the full order book dynamics, queue positions, market impact, latency, and hidden liquidity.

**Version**: 10.0 (Stage 10 - Documentation & Deployment)
**Status**: Tested and operational
**Tests**: 749+ passing

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         L3 Execution Provider                                │
│                     (execution_providers_l3.py)                              │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ L3 Slippage   │    │ L3 Fill       │    │ L3 Fee        │
│ Provider      │    │ Provider      │    │ Provider      │
└───────┬───────┘    └───────┬───────┘    └───────────────┘
        │                    │
        ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LOB Module (lob/)                                  │
├──────────────┬──────────────┬──────────────┬──────────────┬────────────────┤
│ Data         │ Matching     │ Fill Prob &  │ Market       │ Latency &      │
│ Structures   │ Engine       │ Queue Value  │ Impact       │ Events         │
├──────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│ LimitOrder   │ FIFO Match   │ Poisson      │ Kyle         │ LatencyModel   │
│ PriceLevel   │ Pro-Rata     │ Queue-React  │ Almgren-     │ EventScheduler │
│ OrderBook    │ STP (4 modes)│ Historical   │ Chriss       │ SimClock       │
│ Fill/Trade   │ Queue Track  │ Moallemi     │ Gatheral     │                │
└──────────────┴──────────────┴──────────────┴──────────────┴────────────────┘
        │                                              │
        ▼                                              ▼
┌──────────────────────────┐              ┌───────────────────────────────────┐
│ Hidden Liquidity         │              │ Data Adapters & Calibration       │
├──────────────────────────┤              ├───────────────────────────────────┤
│ Iceberg Detection        │              │ LOBSTER Parser                    │
│ Hidden Qty Estimation    │              │ ITCH Adapter                      │
│ Dark Pool Simulation     │              │ Binance L2 Adapter                │
│ Multi-Venue Routing      │              │ Alpaca L2 Adapter                 │
└──────────────────────────┘              │ L3 Calibration Pipeline           │
                                          └───────────────────────────────────┘
```

## Stages

| Stage | Name | Description | Status |
|-------|------|-------------|--------|
| 1 | Data Structures | Core LOB data structures, parsers | ✅ Complete |
| 2 | Matching Engine | FIFO matching, STP, queue tracking | ✅ Complete |
| 3 | Fill Probability | Poisson, Queue-Reactive, queue value | ✅ Complete |
| 4 | Market Impact | Kyle, Almgren-Chriss, Gatheral models | ✅ Complete |
| 5 | Latency Simulation | Realistic latency profiles, event scheduling | ✅ Complete |
| 6 | Hidden Liquidity | Iceberg detection, dark pool simulation | ✅ Complete |
| 7 | L3 Provider | Integration with execution_providers.py | ✅ Complete |
| 8 | Data Pipeline | Data adapters, unified calibration | ✅ Complete |
| 9 | Testing | Validation metrics, backward compatibility | ✅ Complete |
| 10 | Documentation | This documentation, deployment guide | ✅ Complete |

## Key Features

### 1. High-Fidelity Execution Simulation
- FIFO Price-Time Priority matching (CME Globex style)
- Pro-Rata matching for options markets
- Self-Trade Prevention (4 modes: CANCEL_NEWEST, CANCEL_OLDEST, CANCEL_BOTH, DECREMENT)
- Queue position tracking with MBO/MBP estimation

### 2. Fill Probability Models
- **Poisson Model**: Analytical fill probability based on arrival rates
- **Queue-Reactive Model**: Intensity-based model (Huang et al.)
- **Historical Rate Model**: Calibrated from historical execution data
- **Queue Value**: Moallemi & Yuan methodology for optimal order placement

### 3. Market Impact Models
- **Kyle Lambda**: Linear price impact (Kyle, 1985)
- **Almgren-Chriss**: Square-root temporary + linear permanent impact
- **Gatheral**: Transient impact with power-law decay

### 4. Latency Simulation
- Four profiles: Co-located, Proximity, Retail, Institutional
- Configurable distributions: Log-normal, Pareto, Gamma
- Time-of-day and volatility adjustments
- Event scheduling with race condition detection

### 5. Hidden Liquidity
- Iceberg order detection from execution patterns
- Hidden quantity estimation
- Multi-venue dark pool simulation
- Information leakage modeling

## File Structure

```
lob/
├── __init__.py              # Public API exports (552 lines)
├── data_structures.py       # Core data structures (1,151 lines)
├── parsers.py               # LOBSTER message parsing (689 lines)
├── state_manager.py         # State management (821 lines)
├── matching_engine.py       # FIFO matching (844 lines)
├── queue_tracker.py         # Queue position tracking (766 lines)
├── order_manager.py         # Order lifecycle (1,013 lines)
├── fill_probability.py      # Fill probability models (1,120 lines)
├── queue_value.py           # Queue value computation (844 lines)
├── calibration.py           # Model calibration (1,129 lines)
├── market_impact.py         # Impact models (1,150 lines)
├── impact_effects.py        # Impact effects on LOB (832 lines)
├── impact_calibration.py    # Impact calibration (1,059 lines)
├── latency_model.py         # Latency simulation (997 lines)
├── event_scheduler.py       # Event scheduling (970 lines)
├── hidden_liquidity.py      # Iceberg detection (1,037 lines)
├── dark_pool.py             # Dark pool simulation (1,130 lines)
├── config.py                # Pydantic config (765 lines)
├── data_adapters.py         # Data format adapters (1,224 lines)
└── calibration_pipeline.py  # Unified calibration (1,182 lines)

execution_providers_l3.py    # L3 Execution Provider (1,328 lines)

Total: ~19,000+ lines of LOB code
```

## Quick Start

### Basic Usage

```python
from execution_providers import create_execution_provider, AssetClass
from lob.config import L3ExecutionConfig

# Create L3 provider with default equity config
provider = create_execution_provider(
    AssetClass.EQUITY,
    level="L3",
)

# Execute an order
fill = provider.execute(order, market_state, bar_data)

# Get fill probability for limit order
prob = provider.get_fill_probability(order, market_state, bar_data)

# Pre-trade cost estimation
costs = provider.estimate_execution_cost(
    notional=100_000,
    adv=10_000_000,
    side="BUY",
    volatility=0.02,
)
```

### With Custom Configuration

```python
from lob.config import L3ExecutionConfig, create_l3_config

# Load from YAML
config = L3ExecutionConfig.from_yaml("configs/execution_l3.yaml")

# Or use presets
config = create_l3_config("equity")  # Full equity simulation
config = create_l3_config("minimal")  # Fast, matching only

# Customize
config.latency.enabled = True
config.market_impact.model = "almgren_chriss"
config.dark_pools.enabled = False

provider = create_execution_provider(
    AssetClass.EQUITY,
    level="L3",
    config=config,
)
```

## Performance

| Configuration | Throughput | Use Case |
|--------------|------------|----------|
| L2 (Statistical) | >50,000 ord/sec | Fast backtests, crypto |
| L3 minimal | >15,000 ord/sec | Basic LOB simulation |
| L3 full | >1,000 ord/sec | Full equity simulation |

## Backward Compatibility

- **Crypto unchanged**: `create_execution_provider(AssetClass.CRYPTO)` uses L2
- **L2 available**: `create_execution_provider(AssetClass.EQUITY, level="L2")`
- **Gradual migration**: Feature flags for incremental adoption

## Related Documentation

- [Data Structures](data_structures.md) - Core LOB structures
- [Matching Engine](matching_engine.md) - FIFO matching details
- [Queue Position](queue_position.md) - Position tracking
- [Market Impact](market_impact.md) - Impact models
- [Latency](latency.md) - Latency simulation
- [Calibration](calibration.md) - Parameter estimation
- [Configuration](configuration.md) - Config reference
- [Migration Guide](../L3_MIGRATION_GUIDE.md) - L2 to L3 migration

## References

- Kyle (1985): "Continuous Auctions and Insider Trading"
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"
- Moallemi & Yuan (2017): "Relative Value of Queue Position"
- Huang et al. (2015): "Queue-Reactive Model"
- CME Globex Matching Algorithm
- Erik Rigtorp: Queue Position Estimation

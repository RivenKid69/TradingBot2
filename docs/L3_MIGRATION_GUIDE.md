# L3 Execution Provider Migration Guide

## Overview

This guide covers the migration from L2 (Statistical) to L3 (LOB Simulation) execution providers for US equity simulation. The L3 provider offers higher-fidelity execution simulation with full order book modeling.

**Important**: Crypto functionality remains unchanged and continues to use L2 providers by default.

## Key Differences

| Feature | L2 (Statistical) | L3 (LOB Simulation) |
|---------|------------------|---------------------|
| Slippage Model | √participation impact | Full LOB + Market Impact |
| Fill Logic | OHLCV bar-based | Matching engine + Queue position |
| Queue Position | Not tracked | FIFO/Pro-Rata simulation |
| Market Impact | Almgren-Chriss (simple) | Kyle/Almgren-Chriss/Gatheral |
| Latency | Not simulated | Realistic profiles |
| Dark Pools | Not supported | Multi-venue simulation |
| Hidden Liquidity | Not detected | Iceberg detection |

## Migration Steps

### 1. Update Factory Function Calls

**Before (L2)**:
```python
from execution_providers import create_execution_provider, AssetClass

provider = create_execution_provider(AssetClass.EQUITY)
```

**After (L3)**:
```python
from execution_providers import create_execution_provider, AssetClass

# Option 1: Default L3 config
provider = create_execution_provider(AssetClass.EQUITY, level="L3")

# Option 2: Custom L3 config
from lob.config import L3ExecutionConfig

config = L3ExecutionConfig.for_equity()
config.latency.enabled = True
config.market_impact.model = "almgren_chriss"

provider = create_execution_provider(AssetClass.EQUITY, level="L3", config=config)
```

### 2. Configuration Options

#### From Python
```python
from lob.config import L3ExecutionConfig, create_l3_config

# Preset configurations
config = L3ExecutionConfig.for_equity()      # Full equity simulation
config = L3ExecutionConfig.for_crypto()      # Crypto settings
config = L3ExecutionConfig.minimal()         # Matching engine only (fastest)

# Factory with presets
config = create_l3_config("equity")
config = create_l3_config("crypto")
config = create_l3_config("minimal")

# Custom configuration
config = L3ExecutionConfig(
    enabled=True,
    latency=LatencyConfig(
        enabled=True,
        profile="institutional",
    ),
    fill_probability=FillProbabilityConfig(
        enabled=True,
        model="queue_reactive",
    ),
    market_impact=MarketImpactConfig(
        enabled=True,
        model="almgren_chriss",
        eta=0.05,
        gamma=0.03,
    ),
    dark_pools=DarkPoolsConfig(
        enabled=True,
        max_dark_fill_pct=0.5,
    ),
)
```

#### From YAML
```python
from lob.config import L3ExecutionConfig

# Load from file
config = L3ExecutionConfig.from_yaml("configs/execution_l3.yaml")

# Save to file
config.to_yaml("my_config.yaml")
```

### 3. YAML Configuration File

Create or modify `configs/execution_l3.yaml`:

```yaml
enabled: true

# LOB Data Source
lob_data:
  source: internal
  max_depth: 20

# Latency Simulation
latency:
  enabled: true
  profile: institutional  # colocated, proximity, retail, institutional

# Fill Probability
fill_probability:
  enabled: true
  model: queue_reactive
  base_rate: 100.0

# Market Impact
market_impact:
  enabled: true
  model: almgren_chriss
  eta: 0.05
  gamma: 0.03

# Dark Pools
dark_pools:
  enabled: true
  max_dark_fill_pct: 0.5
```

### 4. Backtest Script Updates

**Before**:
```python
# script_backtest.py
provider = create_execution_provider(AssetClass.EQUITY)
```

**After**:
```python
# script_backtest.py
from lob.config import L3ExecutionConfig

# Load L3 config if specified
if config.get("execution", {}).get("level") == "L3":
    l3_config_path = config.get("execution", {}).get("l3_config_path")
    if l3_config_path:
        l3_config = L3ExecutionConfig.from_yaml(l3_config_path)
    else:
        l3_config = L3ExecutionConfig.for_equity()

    provider = create_execution_provider(
        AssetClass.EQUITY,
        level="L3",
        config=l3_config,
    )
else:
    provider = create_execution_provider(AssetClass.EQUITY, level="L2")
```

### 5. Main Config File Update

Add to your main YAML config:

```yaml
execution:
  level: L3  # or L2 for statistical
  l3_config_path: configs/execution_l3.yaml  # optional
```

## API Reference

### L3ExecutionProvider

```python
class L3ExecutionProvider:
    def __init__(
        self,
        asset_class: AssetClass,
        config: Optional[L3ExecutionConfig] = None,
        slippage_provider: Optional[SlippageProvider] = None,
        fee_provider: Optional[FeeProvider] = None,
        **kwargs,
    ):
        """
        Initialize L3 execution provider.

        Args:
            asset_class: CRYPTO or EQUITY
            config: L3 configuration (uses defaults if None)
            slippage_provider: Custom slippage provider
            fee_provider: Custom fee provider
        """

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """Execute an order with L3 simulation."""

    def estimate_execution_cost(
        self,
        notional: float,
        adv: float,
        side: str,
        volatility: float = 0.02,
    ) -> Dict[str, float]:
        """
        Pre-trade cost estimation.

        Returns:
            {
                "slippage_bps": float,
                "impact_bps": float,
                "fee_bps": float,
                "total_bps": float,
            }
        """

    def get_fill_probability(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        time_horizon_sec: float = 60.0,
    ) -> Optional[FillProbabilityResult]:
        """Get fill probability for a limit order."""
```

### Factory Functions

```python
# Create L3 execution provider
provider = create_execution_provider(
    asset_class=AssetClass.EQUITY,
    level="L3",
    config=L3ExecutionConfig.for_equity(),
)

# Create L3 slippage provider
slippage = create_slippage_provider("L3", AssetClass.EQUITY)

# Create L3 fill provider
fill = create_fill_provider("L3", AssetClass.EQUITY, slippage_provider, fee_provider)
```

## Backward Compatibility

### Crypto Unchanged
Crypto continues to use L2 by default:
```python
# This still works exactly as before
provider = create_execution_provider(AssetClass.CRYPTO)  # L2 by default
```

### L2 Still Available
L2 is fully functional:
```python
# Explicit L2 for equity
provider = create_execution_provider(AssetClass.EQUITY, level="L2")
```

### Gradual Migration
You can migrate incrementally:
```python
# Development: use L3 for more realistic testing
if env == "development":
    provider = create_execution_provider(AssetClass.EQUITY, level="L3")
else:
    provider = create_execution_provider(AssetClass.EQUITY, level="L2")
```

## Performance Considerations

| Configuration | Throughput | Use Case |
|--------------|------------|----------|
| L2 | >50,000/sec | Fast backtests, crypto |
| L3 minimal | >10,000/sec | Basic LOB simulation |
| L3 full | >1,000/sec | Full equity simulation |

### Optimizing L3 Performance

```python
# Fastest L3 config (matching engine only)
config = L3ExecutionConfig.minimal()

# Medium (disable dark pools and latency)
config = L3ExecutionConfig.for_equity()
config.dark_pools.enabled = False
config.latency.enabled = False
config.hidden_liquidity.enabled = False

# Full fidelity (slowest but most realistic)
config = L3ExecutionConfig.for_equity()
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'execution_providers_l3'**
   - Ensure all files are in place after update
   - Check Python path

2. **Config validation errors**
   - Use Pydantic validation: `config = L3ExecutionConfig.model_validate(dict_config)`
   - Check enum values match expected strings

3. **Unexpected fills in dark pools**
   - Disable dark pools: `config.dark_pools.enabled = False`
   - Check `max_dark_fill_pct` setting

4. **Slow performance**
   - Use `L3ExecutionConfig.minimal()` for speed
   - Disable unused features (latency, dark pools, hidden liquidity)

### Debug Mode

```python
import logging
logging.getLogger("execution_providers_l3").setLevel(logging.DEBUG)
```

## Version History

- **v7.0.0** (2025-11-28): Initial L3 execution provider release
  - Full LOB simulation for equities
  - Market impact models (Kyle, Almgren-Chriss, Gatheral)
  - Latency simulation profiles
  - Dark pool multi-venue simulation
  - Hidden liquidity detection
  - YAML configuration support
  - 79 new tests

## References

- [LOB Module Documentation](../lob/__init__.py)
- [Execution Providers](../execution_providers.py)
- [L3 Configuration](../configs/execution_l3.yaml)
- [Market Impact Models](l3_simulator/market_impact.md)
- [claude.md Phase 10 Section](../claude.md#-l3-lob-simulation-phase-10)

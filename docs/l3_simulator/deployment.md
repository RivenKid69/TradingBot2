# L3 LOB Simulator - Deployment Guide

## Overview

This guide covers the deployment of the L3 LOB simulator, including feature flags, gradual rollout strategies, monitoring, and rollback procedures.

## Deployment Checklist

### Pre-Deployment

- [ ] **Code Review**: All L3 code reviewed and approved
- [ ] **Test Coverage**: 749+ tests passing (100% pass rate)
- [ ] **Documentation**: All docs complete in `docs/l3_simulator/`
- [ ] **Config Files**: `configs/execution_l3.yaml` validated
- [ ] **Backward Compatibility**: Crypto path verified unchanged
- [ ] **Performance Benchmarks**: Throughput meets targets

### Configuration

- [ ] **Feature Flag**: `use_l3_simulator: true` in config
- [ ] **L3 Config Path**: `l3_config_path` specified
- [ ] **Monitoring**: Dashboards configured
- [ ] **Alerts**: Alert rules deployed
- [ ] **Logging**: Debug logging ready to enable

### Rollout

- [ ] **Shadow Mode**: L3 running in parallel without affecting trades
- [ ] **Canary**: 1-5% of traffic using L3
- [ ] **Gradual Expansion**: 25% → 50% → 100%
- [ ] **Validation**: Metrics match expected values

### Post-Deployment

- [ ] **Health Checks**: All systems green
- [ ] **Metric Validation**: Slippage, fill rate, latency within bounds
- [ ] **User Feedback**: No unexpected issues reported
- [ ] **Documentation Update**: Any findings documented

## Feature Flags

### Master Switch

```yaml
# configs/config_backtest.yaml
execution:
  level: L3      # L2 or L3
  use_l3_simulator: true  # Feature flag

  # L3-specific config
  l3_config_path: configs/execution_l3.yaml
```

### Component-Level Flags

```yaml
# configs/execution_l3.yaml
enabled: true  # Master L3 switch

latency:
  enabled: true  # Latency simulation on/off

fill_probability:
  enabled: true  # Fill probability models on/off

market_impact:
  enabled: true  # Market impact models on/off

hidden_liquidity:
  enabled: true  # Hidden liquidity detection on/off

dark_pools:
  enabled: true  # Dark pool simulation on/off
```

### Environment-Based Flags

```python
import os

# Environment variable override
USE_L3 = os.getenv("USE_L3_SIMULATOR", "false").lower() == "true"

if USE_L3:
    provider = create_execution_provider(AssetClass.EQUITY, level="L3")
else:
    provider = create_execution_provider(AssetClass.EQUITY, level="L2")
```

### Percentage-Based Rollout

```python
import hashlib

def should_use_l3(symbol: str, rollout_pct: float = 0.0) -> bool:
    """Deterministic percentage-based rollout."""
    hash_val = int(hashlib.md5(symbol.encode()).hexdigest(), 16)
    return (hash_val % 100) < (rollout_pct * 100)

# 10% rollout
if should_use_l3(symbol, rollout_pct=0.10):
    provider = create_execution_provider(AssetClass.EQUITY, level="L3")
```

## Gradual Rollout Strategy

### Phase 1: Shadow Mode (Week 1)

Run L3 in parallel without affecting actual execution.

```python
# Shadow mode implementation
class ShadowL3Provider:
    def __init__(self):
        self.l2_provider = create_execution_provider(AssetClass.EQUITY, level="L2")
        self.l3_provider = create_execution_provider(AssetClass.EQUITY, level="L3")

    def execute(self, order, market_state, bar_data):
        # L2 is authoritative
        l2_fill = self.l2_provider.execute(order, market_state, bar_data)

        # L3 runs in shadow (logged but not used)
        try:
            l3_fill = self.l3_provider.execute(order, market_state, bar_data)
            self._compare_and_log(l2_fill, l3_fill)
        except Exception as e:
            logger.error(f"L3 shadow error: {e}")

        return l2_fill  # Always return L2 result
```

**Validation Metrics:**

- L3 vs L2 slippage correlation
- L3 vs L2 fill rate comparison
- L3 latency overhead
- L3 error rate

### Phase 2: Canary (Week 2)

5% of symbols use L3.

```yaml
# configs/l3_rollout.yaml
rollout:
  enabled: true
  percentage: 0.05  # 5%
  canary_symbols:
    - "AAPL"
    - "MSFT"
    - "GOOGL"
```

**Validation:**

- Compare metrics between L2 and L3 symbols
- Monitor for anomalies
- User feedback collection

### Phase 3: Expansion (Week 3-4)

Gradual increase: 25% → 50% → 100%

```python
# Rollout schedule
ROLLOUT_SCHEDULE = {
    "2025-12-01": 0.25,
    "2025-12-08": 0.50,
    "2025-12-15": 0.75,
    "2025-12-22": 1.00,
}
```

## Monitoring

### Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Fill Rate | >95% | <90% |
| Slippage RMSE | <2 bps | >5 bps |
| Latency p95 | <1ms | >5ms |
| Error Rate | <0.01% | >0.1% |
| Throughput | >1000/sec | <500/sec |

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# L3 execution metrics
l3_executions_total = Counter(
    "l3_executions_total",
    "Total L3 executions",
    ["symbol", "side", "result"]
)

l3_slippage_bps = Histogram(
    "l3_slippage_bps",
    "L3 slippage in basis points",
    ["symbol"],
    buckets=[0.5, 1, 2, 5, 10, 20, 50]
)

l3_latency_seconds = Histogram(
    "l3_latency_seconds",
    "L3 execution latency",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01]
)

l3_enabled = Gauge(
    "l3_enabled",
    "L3 simulator enabled (1=yes, 0=no)"
)
```

### Grafana Dashboard

Example dashboard panels:

1. **Execution Summary**
   - Total executions (L2 vs L3)
   - Success rate
   - Error rate

2. **Slippage Comparison**
   - L2 slippage distribution
   - L3 slippage distribution
   - Correlation

3. **Latency**
   - p50, p95, p99 latency
   - Time series trend

4. **Fill Probability Accuracy**
   - Predicted vs actual fill rate
   - Model confidence

### Alert Rules

```yaml
# alerts/l3_alerts.yaml
groups:
  - name: l3_simulator
    rules:
      - alert: L3HighErrorRate
        expr: rate(l3_executions_total{result="error"}[5m]) / rate(l3_executions_total[5m]) > 0.001
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "L3 error rate high"

      - alert: L3HighSlippage
        expr: histogram_quantile(0.95, l3_slippage_bps) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "L3 slippage p95 above threshold"

      - alert: L3HighLatency
        expr: histogram_quantile(0.95, l3_latency_seconds) > 0.005
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "L3 latency p95 above 5ms"
```

## Rollback Procedure

### Immediate Rollback

```python
# Emergency rollback script
import os

def emergency_rollback():
    """Immediately disable L3 simulator."""

    # 1. Set environment variable
    os.environ["USE_L3_SIMULATOR"] = "false"

    # 2. Update config file
    with open("configs/config_backtest.yaml", "r+") as f:
        config = yaml.safe_load(f)
        config["execution"]["level"] = "L2"
        f.seek(0)
        yaml.dump(config, f)
        f.truncate()

    # 3. Signal running processes
    # Send SIGHUP to reload config

    logger.warning("L3 EMERGENCY ROLLBACK EXECUTED")
```

### Graceful Rollback

1. **Reduce Rollout**: Set `rollout_percentage: 0.0`
2. **Monitor**: Wait for in-flight executions
3. **Disable**: Set `enabled: false`
4. **Verify**: Check all traffic on L2
5. **Investigate**: Analyze logs and metrics

### Rollback Triggers

Automatic rollback if:

- Error rate > 1% for 5 minutes
- Slippage p95 > 10 bps for 10 minutes
- Latency p95 > 10ms for 5 minutes
- Any critical exception in L3 code

## Health Checks

### Startup Check

```python
def l3_health_check() -> bool:
    """Verify L3 components are healthy."""
    try:
        # Test configuration loading
        config = L3ExecutionConfig.from_yaml("configs/execution_l3.yaml")

        # Test provider creation
        provider = create_execution_provider(
            AssetClass.EQUITY,
            level="L3",
            config=config,
        )

        # Test basic execution
        test_order = Order(...)
        test_market = MarketState(...)
        test_bar = BarData(...)

        # Should not raise
        result = provider.execute(test_order, test_market, test_bar)

        return True
    except Exception as e:
        logger.error(f"L3 health check failed: {e}")
        return False
```

### Continuous Health Check

```python
# Run periodically
@scheduler.task("interval", minutes=5)
def periodic_l3_check():
    if not l3_health_check():
        alert("L3 health check failed")
        if AUTO_ROLLBACK_ENABLED:
            emergency_rollback()
```

## Logging

### Debug Logging

```python
import logging

# Enable L3 debug logging
logging.getLogger("execution_providers_l3").setLevel(logging.DEBUG)
logging.getLogger("lob").setLevel(logging.DEBUG)

# Or via environment
os.environ["L3_LOG_LEVEL"] = "DEBUG"
```

### Structured Logging

```python
import structlog

logger = structlog.get_logger("l3")

logger.info(
    "l3_execution",
    symbol=order.symbol,
    side=order.side,
    qty=order.qty,
    fill_price=fill.price if fill else None,
    slippage_bps=slippage_bps,
    latency_ms=latency_ms,
    model_type="almgren_chriss",
)
```

## Performance Tuning

### Memory Optimization

```python
# Limit tracked orders
config.queue_tracking.max_tracked_orders = 5000  # Default 10000

# Reduce dark pool history
config.dark_pools.max_history_size = 500  # Default 1000
```

### Latency Optimization

```python
# Use minimal config for high throughput
config = create_l3_config("minimal")

# Disable unused features
config.latency.enabled = False
config.dark_pools.enabled = False
config.hidden_liquidity.enabled = False
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Import error | Missing dependency | `pip install sortedcontainers` |
| Config validation | Invalid YAML | Check schema |
| High latency | All features enabled | Use minimal config |
| Memory growth | Queue tracking | Reduce `max_tracked_orders` |

### Debug Commands

```bash
# Run L3 tests
pytest tests/test_execution_providers_l3.py -v

# Check L3 import
python -c "from execution_providers import create_execution_provider; print('OK')"

# Validate config
python -c "from lob.config import L3ExecutionConfig; L3ExecutionConfig.from_yaml('configs/execution_l3.yaml')"
```

## Related Documentation

- [Overview](overview.md) - Architecture overview
- [Configuration](configuration.md) - Config reference
- [Migration Guide](../L3_MIGRATION_GUIDE.md) - L2 to L3 migration

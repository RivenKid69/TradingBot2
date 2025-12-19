# Futures Deployment Guide

## Overview

This guide covers deployment of futures trading infrastructure, including feature rollout strategy, monitoring, and rollback procedures.

---

## Pre-Deployment Checklist

### 1. Code Validation

- [ ] All tests passing (verify via CI run logs with commit SHA)
- [ ] Validation metrics meet targets
- [ ] Backward compatibility confirmed
- [ ] Performance benchmarks acceptable

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run futures-specific tests
pytest tests/test_futures_*.py -v

# Run validation tests
pytest tests/test_futures_validation.py -v

# Run backward compatibility tests
pytest tests/test_futures_backward_compatibility.py -v

# Run benchmarks
python benchmarks/bench_futures_simulation.py --iterations 1000
```

### 2. Configuration Review

- [ ] Risk limits configured appropriately
- [ ] Margin thresholds set conservatively
- [ ] Feature flags configured for rollout stage
- [ ] Alert thresholds set

### 3. Infrastructure

- [ ] Sufficient compute resources
- [ ] Network connectivity to exchanges
- [ ] Monitoring dashboards configured
- [ ] Alert channels set up (Slack, PagerDuty)

### 4. Data

- [ ] Historical data available
- [ ] Funding rate history loaded
- [ ] Contract specifications updated

---

## Feature Rollout Strategy

### Rollout Stages

| Stage | Description | Risk Level |
|-------|-------------|------------|
| **DISABLED** | Feature off | None |
| **SHADOW** | Running but not used | Low |
| **CANARY** | Limited symbols | Medium |
| **PRODUCTION** | Full deployment | High |

### Stage 1: Shadow Mode

Run the new system alongside existing without affecting trading.

```yaml
# configs/feature_flags_futures.yaml
features:
  unified_risk_guard:
    enabled: true
    stage: "shadow"
    shadow_log_results: true
```

**Monitor for 1-2 weeks:**
- No crashes or errors
- Results match expectations
- Performance within targets

### Stage 2: Canary Deployment

Enable for select symbols.

```yaml
features:
  unified_risk_guard:
    enabled: true
    stage: "canary"
    canary_symbols:
      - "BTCUSDT"  # Most liquid, well-understood
```

**Monitor for 1 week:**
- Trading behavior as expected
- No unexpected liquidations
- Margin calculations accurate

### Stage 3: Gradual Rollout

Expand to more symbols.

```yaml
features:
  unified_risk_guard:
    enabled: true
    stage: "canary"
    canary_symbols:
      - "BTCUSDT"
      - "ETHUSDT"
      - "BNBUSDT"
```

### Stage 4: Production

Full deployment.

```yaml
features:
  unified_risk_guard:
    enabled: true
    stage: "production"
```

---

## Deployment Commands

### Paper Trading Deployment

```bash
# Deploy to paper trading environment
python script_live.py \
    --config configs/config_live_futures.yaml \
    --paper \
    --symbols BTCUSDT ETHUSDT

# With shadow mode
python script_live.py \
    --config configs/config_live_futures.yaml \
    --paper \
    --feature-stage shadow
```

### Live Trading Deployment

```bash
# Deploy to live (use with caution!)
python script_live.py \
    --config configs/config_live_futures.yaml \
    --live \
    --symbols BTCUSDT \
    --risk-profile conservative

# With canary mode
python script_live.py \
    --config configs/config_live_futures.yaml \
    --live \
    --feature-stage canary \
    --canary-symbols BTCUSDT
```

---

## Monitoring

### Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Fill Rate | > 95% | < 90% |
| Slippage | < 5 bps | > 10 bps |
| Margin Ratio | > 1.5 | < 1.2 |
| Funding Accuracy | > 99% | < 95% |
| Latency (L2) | < 100 μs | > 500 μs |
| Latency (L3) | < 500 μs | > 2 ms |

### Dashboard Setup

```yaml
# Grafana dashboard panels
panels:
  - name: "Margin Ratio"
    query: "futures_margin_ratio{symbol=~'.*'}"
    thresholds:
      - {value: 1.5, color: "green"}
      - {value: 1.2, color: "yellow"}
      - {value: 1.05, color: "red"}

  - name: "Funding Rate"
    query: "futures_funding_rate{symbol=~'.*'}"
    thresholds:
      - {value: 0.0001, color: "green"}
      - {value: 0.0003, color: "yellow"}
      - {value: 0.0005, color: "red"}

  - name: "Position Sync Errors"
    query: "rate(futures_sync_errors_total[5m])"
    alert: "> 0"

  - name: "Liquidation Count"
    query: "increase(futures_liquidations_total[1h])"
    alert: "> 0"
```

### Alert Rules

```yaml
# Prometheus alert rules
groups:
  - name: futures_alerts
    rules:
      - alert: FuturesMarginCritical
        expr: futures_margin_ratio < 1.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Critical margin ratio on {{ $labels.symbol }}"

      - alert: FuturesLiquidationImminent
        expr: futures_margin_ratio < 1.02
        for: 30s
        labels:
          severity: page
        annotations:
          summary: "Liquidation imminent on {{ $labels.symbol }}"

      - alert: FuturesSyncError
        expr: rate(futures_sync_errors_total[5m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Position sync errors detected"

      - alert: FuturesHighSlippage
        expr: futures_slippage_bps > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High slippage on {{ $labels.symbol }}"
```

### Log Monitoring

```bash
# Watch for errors
tail -f logs/futures.log | grep -E "(ERROR|CRITICAL|LIQUIDATION)"

# Watch position sync
tail -f logs/futures.log | grep "position_sync"

# Watch margin status
tail -f logs/futures.log | grep "margin_status"
```

---

## Rollback Procedures

### Quick Rollback (Feature Flag)

```yaml
# Disable feature immediately
features:
  unified_risk_guard:
    enabled: false  # Quick disable
```

### Command-Line Rollback

```bash
# Kill running process
pkill -f "script_live.py.*futures"

# Start with previous config
python script_live.py \
    --config configs/config_live_spot.yaml \
    --paper
```

### Full Rollback

1. **Stop all futures processes**

```bash
./scripts/stop_futures.sh
```

2. **Revert to previous version**

```bash
git checkout <previous_tag> -- \
    execution_providers_futures.py \
    services/futures_risk_guards.py \
    services/unified_futures_risk.py
```

3. **Restart spot-only trading**

```bash
python script_live.py --config configs/config_live.yaml
```

### Rollback Triggers

| Condition | Action |
|-----------|--------|
| Unexpected liquidation | Pause, investigate |
| Position sync errors > 5/min | Rollback to shadow |
| Slippage > 20 bps consistently | Rollback to shadow |
| Multiple margin calls | Pause, reduce leverage |
| System crash | Full rollback |

---

## Post-Deployment

### Day 1-7: Intensive Monitoring

- Check dashboards every 30 minutes
- Review all trades manually
- Compare with shadow results (if available)
- Document any issues

### Day 7-14: Standard Monitoring

- Check dashboards every 2 hours
- Review daily summaries
- Adjust thresholds if needed

### Day 14+: Production Mode

- Normal monitoring cadence
- Weekly performance review
- Monthly configuration review

---

## Disaster Recovery

### Scenarios

| Scenario | Response |
|----------|----------|
| Exchange API down | Queue orders, retry |
| Network partition | Fail-safe (close positions) |
| Database corruption | Restore from backup |
| Invalid margin state | Sync from exchange |
| Runaway liquidation | Kill switch → manual review |

### Kill Switch

```python
from services.ops_kill_switch import trip_kill_switch

# Emergency stop all trading
trip_kill_switch(reason="Manual intervention - futures issue")
```

### Recovery Steps

1. **Assess situation**
   - Check exchange status
   - Check position states
   - Check account balance

2. **Sync positions**
   ```bash
   python scripts/sync_positions.py --force
   ```

3. **Verify state**
   ```bash
   python scripts/verify_positions.py
   ```

4. **Resume trading**
   ```bash
   ./scripts/reset_kill_switch.sh
   python script_live.py --config configs/config_live_futures.yaml
   ```

---

## Environment-Specific Notes

### Development

- Use paper trading only
- Enable verbose logging
- All feature flags in shadow mode

### Staging

- Use paper trading with real market data
- Test full feature set
- Canary mode for new features

### Production

- Live trading enabled
- Conservative risk limits
- Feature flags in production mode
- Full monitoring

---

## Security Considerations

### API Keys

- Store in environment variables, not config files
- Use separate keys for paper/live
- Rotate keys periodically

### Network

- Use VPN for exchange connections
- Whitelist IP addresses where possible
- Monitor for unusual API activity

### Access Control

- Limit who can deploy to production
- Require approval for live deployments
- Audit all configuration changes

---

## References

- Internal: [Configuration Guide](configuration.md)
- Internal: [Risk Management](../FUTURES_INTEGRATION_PLAN.md)
- Binance API: Rate Limits and Best Practices
- CME Group: Production Connectivity Guidelines

# Futures Migration Guide

## Overview

This guide covers migrating existing crypto spot, equity, and forex strategies to futures trading. It includes code changes, configuration updates, and best practices for a smooth transition.

---

## Migration Paths

### From Crypto Spot to Crypto Perpetual Futures

**Complexity**: Medium | **Estimated Effort**: 2-4 hours

#### Key Differences

| Aspect | Crypto Spot | Crypto Perpetual |
|--------|-------------|------------------|
| **Leverage** | 1x only | 1x-125x |
| **Position** | Long only | Long/Short |
| **Fees** | Maker/Taker | Maker/Taker + Funding |
| **Settlement** | Immediate | Continuous (perpetual) |
| **Margin** | N/A | Isolated/Cross |
| **Liquidation** | N/A | Yes |

#### Step 1: Update Configuration

```yaml
# Before: configs/config_train.yaml
mode: train
asset_class: crypto

# After: configs/config_train_futures.yaml
mode: train
asset_class: futures
futures_type: "CRYPTO_PERPETUAL"

env:
  futures:
    enabled: true
    max_leverage: 10
    margin_mode: "isolated"
```

#### Step 2: Update Data Loading

```python
# Before
from data_loader_multi_asset import load_multi_asset_data

frames, obs_shapes = load_multi_asset_data(
    paths=["data/crypto/*.parquet"],
    asset_class="crypto",
)

# After
from data_loader_multi_asset import load_multi_asset_data

frames, obs_shapes = load_multi_asset_data(
    paths=["data/futures/*.parquet"],
    asset_class="futures",
    futures_type="CRYPTO_PERPETUAL",
)
```

#### Step 3: Update Execution Provider

```python
# Before
from execution_providers import create_execution_provider, AssetClass

provider = create_execution_provider(AssetClass.CRYPTO)

# After
from execution_providers_futures import create_futures_execution_provider

provider = create_futures_execution_provider(
    use_mark_price=True,
    max_leverage=10,
)
```

#### Step 4: Add Risk Guards

```python
# Add futures-specific risk guards
from services.futures_risk_guards import (
    FuturesLeverageGuard,
    FuturesMarginGuard,
    FundingExposureGuard,
)

leverage_guard = FuturesLeverageGuard(max_account_leverage=10)
margin_guard = FuturesMarginGuard()
funding_guard = FundingExposureGuard()
```

#### Step 5: Handle Funding Rates

```python
# Funding rate tracking (new)
from services.futures_funding_tracker import FuturesFundingTracker

tracker = FuturesFundingTracker(funding_provider)
info = await tracker.get_funding_info("BTCUSDT")
print(f"Next funding: {info.next_funding_time}")
```

---

### From Equity to CME Futures

**Complexity**: High | **Estimated Effort**: 4-8 hours

#### Key Differences

| Aspect | US Equity | CME Futures |
|--------|-----------|-------------|
| **Trading Hours** | NYSE 9:30-16:00 ET | Sun 18:00 - Fri 17:00 ET |
| **Fees** | $0 + SEC/TAF | Per-contract ($1.29 ES) |
| **Margin** | Reg T (50%/25%) | SPAN (risk-based) |
| **Settlement** | T+2 | Daily variation margin |
| **Expiration** | N/A | Quarterly |
| **Circuit Breakers** | NYSE halts | CME Rule 80B |

#### Step 1: Update Configuration

```yaml
# Before: configs/config_train_stocks.yaml
mode: train
asset_class: equity
data_vendor: alpaca

# After: configs/config_train_futures.yaml
mode: train
asset_class: futures
futures_type: "CME_EQUITY_INDEX"
data_vendor: ib

symbols:
  - "ES"
  - "NQ"
```

#### Step 2: Replace Alpaca Adapter with IB

```python
# Before
from adapters.alpaca import AlpacaMarketDataAdapter

adapter = AlpacaMarketDataAdapter(api_key, api_secret)

# After
from adapters.ib import IBMarketDataAdapter

adapter = IBMarketDataAdapter({
    "host": "127.0.0.1",
    "port": 7497,  # Paper trading
    "client_id": 1,
})
```

#### Step 3: Use SPAN Margin Calculator

```python
# Before: Reg T margin (simple)
initial_margin = position_value * 0.5
maintenance_margin = position_value * 0.25

# After: SPAN margin
from impl_span_margin import create_span_calculator

calc = create_span_calculator()
result = calc.calculate_margin(position, current_price)
initial_margin = result.initial_margin
maintenance_margin = result.maintenance_margin
```

#### Step 4: Add CME-Specific Risk Guards

```python
from services.cme_risk_guards import (
    SPANMarginGuard,
    CMEPositionLimitGuard,
    CircuitBreakerAwareGuard,
    SettlementRiskGuard,
    RolloverGuard,
)

# Unified CME risk guard
from services.cme_risk_guards import CMEFuturesRiskGuard

guard = CMEFuturesRiskGuard(strict_mode=True)
guard.add_symbol_to_monitor("ES", reference_price)
```

#### Step 5: Handle Contract Rollover

```python
from impl_cme_rollover import ContractRolloverManager

rollover = ContractRolloverManager()
rollover.set_expiration_calendar("ES", expiration_dates)

if rollover.should_roll("ES", date.today()):
    # Execute rollover
    days_to_roll = rollover.get_days_to_roll("ES", date.today())
    print(f"Rollover in {days_to_roll} days")
```

#### Step 6: Daily Settlement

```python
from impl_cme_settlement import CMESettlementEngine

engine = CMESettlementEngine()

# Calculate daily variation margin
vm = engine.calculate_variation_margin(
    position=position,
    settlement_price=settlement_price,
    contract_spec=spec,
)
```

---

### From Forex to Currency Futures

**Complexity**: Medium | **Estimated Effort**: 3-5 hours

#### Key Differences

| Aspect | OTC Forex | CME Currency Futures |
|--------|-----------|----------------------|
| **Venue** | Dealer network | CME Globex |
| **Fees** | Spread-based | Per-contract ($1.00) |
| **Lot Size** | Flexible | Standardized (125,000) |
| **Leverage** | Up to 500:1 | SPAN-based |
| **Settlement** | Rolling | Quarterly |
| **Trading Hours** | 24/5 OTC | CME Globex hours |

#### Step 1: Update Configuration

```yaml
# Before: configs/config_train_forex.yaml
mode: train
asset_class: forex
data_vendor: oanda

symbols:
  - "EUR_USD"
  - "USD_JPY"

# After: configs/config_train_futures.yaml
mode: train
asset_class: futures
futures_type: "CME_CURRENCY"
data_vendor: ib

symbols:
  - "6E"  # Euro FX
  - "6J"  # Japanese Yen
```

#### Step 2: Symbol Mapping

```python
# Forex to Futures symbol mapping
FOREX_TO_CME = {
    "EUR_USD": "6E",
    "USD_JPY": "6J",
    "GBP_USD": "6B",
    "AUD_USD": "6A",
    "USD_CAD": "6C",
    "USD_CHF": "6S",
}

# Convert position sizes
# Forex: position in units
# CME: position in contracts (125,000 base currency each)
contracts = forex_units / 125_000
```

#### Step 3: Fee Structure Change

```python
# Before: Spread-based (forex)
spread_cost = qty * spread_pips * pip_value

# After: Per-contract (CME)
fee = qty_contracts * 1.00  # $1.00 per contract for 6E
```

---

## Configuration Migration

### Environment Config Changes

```yaml
# Add to existing config
env:
  futures:
    enabled: true
    max_leverage: 10
    margin_mode: "isolated"
    funding_rate_enabled: true
    simulate_liquidation: true

# Risk configuration
risk:
  futures:
    margin_warning_threshold: 1.5
    margin_danger_threshold: 1.2
    max_single_symbol_pct: 0.5
```

### Feature Flags for Gradual Migration

```yaml
# configs/feature_flags_futures.yaml
features:
  unified_risk_guard:
    enabled: true
    stage: "shadow"  # Start in shadow mode
    shadow_log_results: true
```

---

## Code Migration Patterns

### Pattern 1: Adding Leverage Support

```python
# Before: Spot trading
class SpotPosition:
    def __init__(self, symbol: str, qty: Decimal, entry_price: Decimal):
        self.symbol = symbol
        self.qty = qty
        self.entry_price = entry_price
        self.notional = qty * entry_price

# After: Futures trading
class FuturesPosition:
    def __init__(
        self,
        symbol: str,
        qty: Decimal,
        entry_price: Decimal,
        leverage: int = 1,
        margin_mode: MarginMode = MarginMode.ISOLATED,
    ):
        self.symbol = symbol
        self.qty = qty
        self.entry_price = entry_price
        self.leverage = leverage
        self.margin_mode = margin_mode
        self.notional = qty * entry_price
        self.margin = self.notional / leverage
```

### Pattern 2: Handling Short Positions

```python
# Before: Long-only
def calculate_pnl(position, current_price):
    return (current_price - position.entry_price) * position.qty

# After: Long/Short support
def calculate_pnl(position, current_price):
    price_diff = current_price - position.entry_price
    if position.side == PositionSide.SHORT:
        price_diff = -price_diff
    return price_diff * abs(position.qty)
```

### Pattern 3: Adding Funding Rate Impact

```python
# Before: Simple reward
reward = log_return * position_fraction

# After: Reward with funding impact
funding_payment = position_notional * funding_rate
funding_impact = funding_payment / account_equity
reward = log_return * position_fraction - funding_penalty * funding_impact
```

### Pattern 4: Margin Calculation

```python
# Before: No margin concept
available_capital = account_balance

# After: Margin-based
def get_available_margin(account_equity, positions, margin_mode):
    if margin_mode == MarginMode.CROSS:
        used_margin = sum(p.margin for p in positions)
        return account_equity - used_margin
    else:  # ISOLATED
        # Each position has its own margin
        return account_equity - sum(p.isolated_margin for p in positions)
```

---

## Testing Migration

### Test File Updates

```python
# Before: Spot trading tests
def test_spot_execution():
    provider = create_execution_provider(AssetClass.CRYPTO)
    fill = provider.execute(order, market, bar)
    assert fill.filled

# After: Add futures tests (keep existing)
def test_spot_execution():
    # Keep existing test unchanged
    provider = create_execution_provider(AssetClass.CRYPTO)
    fill = provider.execute(order, market, bar)
    assert fill.filled

def test_futures_execution():
    # Add new test
    provider = create_futures_execution_provider()
    fill = provider.execute(
        order, market, bar,
        funding_rate=0.0001,
        open_interest=1_000_000_000,
    )
    assert fill.filled
```

### Backward Compatibility Tests

```bash
# Run backward compatibility suite
pytest tests/test_futures_backward_compatibility.py -v

# Ensure existing tests pass
pytest tests/test_execution_providers.py -v
pytest tests/test_crypto_parametric_tca.py -v
pytest tests/test_equity_parametric_tca.py -v
```

---

## Common Migration Issues

### Issue 1: Missing Funding Rate Data

**Symptom**: `KeyError: 'funding_rate'` during training

**Solution**:

```python
# Provide default funding rate if missing
funding_rate = bar_data.get("funding_rate", 0.0)
```

### Issue 2: Leverage Bracket Violations

**Symptom**: `ValueError: Leverage 50 exceeds max 20 for notional $1,000,000`

**Solution**:

```python
# Use leverage guard to get max allowed
guard = FuturesLeverageGuard()
max_leverage = guard.get_max_leverage_for_notional(
    symbol="BTCUSDT",
    notional=1_000_000,
)
```

### Issue 3: Missing Contract Specifications

**Symptom**: `KeyError: 'ES'` in SPAN calculation

**Solution**:

```python
# Ensure contract specs are loaded
from impl_span_margin import load_contract_specs, create_span_calculator

specs = load_contract_specs("configs/cme_contracts.yaml")
calc = create_span_calculator(contract_specs=specs)
```

### Issue 4: Settlement Time Mismatch

**Symptom**: Orders rejected near settlement

**Solution**:

```python
# Check settlement time before trading
from services.cme_risk_guards import SettlementRiskGuard

guard = SettlementRiskGuard()
result = guard.check_settlement_risk("ES", timestamp_ms)
if result.risk_level == SettlementRiskLevel.IMMINENT:
    # Wait for settlement to complete
    pass
```

---

## Rollback Plan

### If Migration Fails

1. **Disable futures features**:

```yaml
features:
  unified_risk_guard:
    enabled: false
```

2. **Revert to spot-only config**:

```yaml
asset_class: crypto  # Not futures
```

3. **Restore previous model checkpoint**:

```bash
python script_live.py --checkpoint models/spot_model_v1.pt
```

### Partial Rollback

```yaml
# Keep futures for some symbols, spot for others
features:
  unified_risk_guard:
    enabled: true
    stage: "canary"
    canary_symbols:
      - "BTCUSDT"  # Futures only for BTC
```

---

## Migration Checklist

### Pre-Migration

- [ ] Backup current configuration
- [ ] Backup trained models
- [ ] Document current performance metrics
- [ ] Review futures-specific documentation

### During Migration

- [ ] Update configuration files
- [ ] Update data loading code
- [ ] Add futures risk guards
- [ ] Update execution providers
- [ ] Add margin monitoring
- [ ] Add funding rate tracking

### Post-Migration

- [ ] Run backward compatibility tests
- [ ] Run futures validation tests
- [ ] Compare performance metrics
- [ ] Monitor live paper trading
- [ ] Gradual rollout (shadow → canary → production)

### Validation

- [ ] All existing tests pass
- [ ] New futures tests pass
- [ ] Performance benchmarks meet targets
- [ ] No memory leaks in extended runs
- [ ] Error handling for edge cases

---

## References

- Internal: [Configuration Guide](configuration.md)
- Internal: [Risk Management](deployment.md)
- Internal: [Margin Calculation](margin_calculation.md)
- Internal: [Funding Rates](funding_rates.md)
- Binance: "Futures Trading Guide"
- CME Group: "Getting Started with Futures"

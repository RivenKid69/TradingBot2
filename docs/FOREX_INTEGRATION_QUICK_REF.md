# Forex Integration - Quick Reference Card

## At a Glance

| Metric | Value |
|--------|-------|
| **Total LOC** | ~4,500 |
| **Total Tests** | 430+ |
| **Timeline** | 12 weeks |
| **Risk Level** | Low-Medium |

## Key Files to Create

```
adapters/oanda/
├── __init__.py              # Package init
├── market_data.py           # ~400 LOC
├── fees.py                  # ~150 LOC
├── trading_hours.py         # ~250 LOC
├── exchange_info.py         # ~200 LOC
└── order_execution.py       # ~350 LOC

execution_providers_forex.py # ~600 LOC (L2+ TCA model)

lob/forex_dealer.py          # ~500 LOC (L3 dealer simulation)

forex_features.py            # ~400 LOC

services/forex_risk_guards.py # ~350 LOC

scripts/download_forex_data.py # ~300 LOC

configs/
├── forex_defaults.yaml
├── config_train_forex.yaml
├── config_backtest_forex.yaml
└── config_live_oanda.yaml
```

## Files to Modify

| File | Changes |
|------|---------|
| `adapters/models.py` | Add `OANDA` to ExchangeVendor, FOREX_SESSIONS |
| `adapters/registry.py` | Add OANDA to lazy_modules |
| `execution_providers.py` | Add `FOREX` to AssetClass (if not exists) |
| `data_loader_multi_asset.py` | Add forex data loading |
| `script_live.py` | Add forex to asset class detection |
| `asset_class_defaults.yaml` | Add forex section |

## Forex-Specific Factors (L2+ TCA Model)

8-factor parametric model:

1. √Participation (Almgren-Chriss)
2. Session liquidity (Tokyo/London/NY)
3. Spread regime
4. Interest rate differential (carry)
5. Volatility regime
6. News event proximity
7. DXY correlation
8. Pair type multiplier

## Session Liquidity Multipliers

| Session | Factor |
|---------|--------|
| Sydney | 0.65 |
| Tokyo | 0.75 |
| London | 1.10 |
| New York | 1.05 |
| **London/NY overlap** | **1.30** |

## Forex vs Other Asset Classes

| Aspect | Crypto | Equity | Forex |
|--------|--------|--------|-------|
| Hours | 24/7 | 9:30-16:00 ET | Sun 5pm-Fri 4pm ET |
| Fees | Maker/Taker % | $0 + regulatory | Spread only |
| Slippage k | 0.10 | 0.05 | **0.03** |
| Default spread | 5 bps | 2 bps | **1 pip** |
| Leverage | 1x-125x | 1x-4x | **50:1-500:1** |
| Market structure | Central LOB | Central LOB | **OTC Dealer** |

## Environment Variables

```bash
OANDA_API_KEY=...
OANDA_ACCOUNT_ID=...
OANDA_PRACTICE=true  # or false for live
```

## CLI Commands

```bash
# Download data
python scripts/download_forex_data.py \
    --pairs EUR_USD GBP_USD USD_JPY \
    --start 2020-01-01 \
    --timeframe H4

# Training
python train_model_multi_patch.py \
    --config configs/config_train_forex.yaml

# Backtest
python script_backtest.py \
    --config configs/config_backtest_forex.yaml

# Live trading (paper)
python script_live.py \
    --config configs/config_live_oanda.yaml \
    --paper
```

## Test Commands

```bash
# Run all forex tests
pytest tests/test_oanda*.py tests/test_forex*.py -v

# Run specific category
pytest tests/test_forex_execution_providers.py -v

# Check backward compatibility
pytest tests/test_forex_integration.py::TestBackwardCompatibility -v
```

## Phase Checklist

- [ ] Phase 0: Foundation & Research
- [ ] Phase 1: Core Enums & Models
- [ ] Phase 2: OANDA Adapter
- [ ] Phase 3: ForexParametricSlippage (L2+)
- [ ] Phase 4: Forex Features
- [ ] Phase 5: L3 Dealer Simulation
- [ ] Phase 6: Risk Management
- [ ] Phase 7: Data Pipeline
- [ ] Phase 8: Configuration
- [ ] Phase 9: Training Integration
- [ ] Phase 10: Testing & Validation

## Key References

- Lyons (2001): "The Microstructure Approach to Exchange Rates"
- Evans & Lyons (2002): "Order Flow and Exchange Rate Dynamics"
- Oomen (2017): "Last Look" in FX
- OANDA v20 API: https://developer.oanda.com/rest-live-v20/

## Contact

Full plan: [FOREX_INTEGRATION_PLAN.md](./FOREX_INTEGRATION_PLAN.md)

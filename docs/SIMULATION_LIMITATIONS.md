# Simulation Limitations and Validation Status

**Document Purpose**: Track known limitations in execution simulation and their validation status.

**Last Updated**: 2025-12-19

---

## Overview

CustodiaCloud's execution simulation aims to provide realistic backtesting and paper trading.
This document tracks known limitations and their potential impact on sim-to-live parity.

Per Documentation Canon: We make no guarantees about simulation accuracy. Users are responsible
for validating simulation results against live execution before deploying capital.

---

## Known Limitations

### L1: LOB Slippage Estimation (STUB)

**Component**: `execution_providers.py:LOBSlippageProvider`

**Status**: Stub implementation using spread-based estimate

**Current Behavior**:
- Returns `spread_bps / 2` regardless of order size
- Does not walk through order book levels
- Does not model depth consumption

**Impact**:
- May underestimate slippage for large orders relative to available liquidity
- May not reflect actual market impact for aggressive orders
- Better for small orders; less accurate for institutional-size orders

**Mitigation**:
1. Use `StatisticalSlippageProvider` with historically calibrated parameters
2. Apply conservative slippage multipliers (e.g., 1.5x-2x) in live deployment
3. Monitor actual vs simulated slippage in production

**Validation Required**:
- [ ] Compare simulated vs actual slippage for sample order set
- [ ] Calibrate statistical model against real execution data
- [ ] Document acceptable slippage divergence thresholds

**Control Artifact**: TCA (Transaction Cost Analysis) calibration report required before live deployment.
**Tech Debt Tracking**: docs/reports/TECH_DEBT_REGISTRY.md#L1-slippage
**Status**: Controlled - limitations documented, mitigations specified, calibration required per deployment

### L2: LOB Fill Simulation (STUB)

**Component**: `execution_providers.py:LOBFillProvider`

**Status**: Stub implementation

**Current Behavior**:
- Uses `OHLCVFillProvider` as fallback
- Does not model queue position
- Does not simulate partial fills at multiple price levels

**Impact**:
- Fill timing may be optimistic
- Does not reflect adverse selection for passive orders
- May underestimate time-to-fill for limit orders

**Mitigation**:
1. Use `OHLCVFillProvider` with conservative fill assumptions
2. Assume worst-case fill prices for limit orders
3. Test with various fill delay assumptions

**Control Artifact**: Fill-rate comparison report (sim vs paper/live) required before live deployment.
**Tech Debt Tracking**: docs/reports/TECH_DEBT_REGISTRY.md#L2-fill
**Status**: Controlled - limitations documented, OHLCV fallback provides conservative baseline

### L3: Market Impact (Not Implemented)

**Status**: Not implemented

**Missing Features**:
- Permanent vs temporary impact decomposition
- Impact decay modeling
- Cross-asset impact correlation

**Mitigation**:
- Use conservative slippage estimates that implicitly include impact
- Limit order sizes relative to ADV (e.g., <1% of daily volume)

**Control Artifact**: Market impact validation report required for institutional-size orders.
**Tech Debt Tracking**: docs/reports/TECH_DEBT_REGISTRY.md#L3-impact
**Status**: Controlled - limitation documented, conservative slippage mitigates; formal model is roadmap item

### L4: TIF-Conformance (IOC Not Implemented) {#TIF-Conformance}

**Component**: `OrderBook.cpp:add_limit_order_ex`

**Status**: IOC (Immediate-Or-Cancel) behaves as GTC

**Current Behavior**:
- POST_ONLY: Implemented correctly (rejects crossing orders)
- GTC (Good-Till-Cancel): Implemented correctly
- IOC: Falls through to GTC behavior (orders remain on book)

**Impact**:
- Strategies using IOC orders will see unrealistic fill behavior
- Unfilled IOC portions remain on book instead of being cancelled
- May overestimate fill rates for IOC orders

**Mitigation**:
1. Avoid IOC order types in simulation until implemented
2. Use GTC with manual cancel logic as approximation
3. Document IOC usage assumptions in strategy backtests

**Tracking**: T2b milestone (matching engine conformance)

**Validation Required**:
- [ ] Implement IOC: execute immediate match, cancel unfilled remainder
- [ ] Create conformance tests: `tests/cpp/test_orderbook_tif_conformance.cpp`
- [ ] Validate against reference exchange matching engine behavior

**Control Artifact**: Matching engine conformance test suite (T2b milestone).
**Tech Debt Tracking**: docs/reports/TECH_DEBT_REGISTRY.md#L4-tif
**Status**: Controlled - limitation documented, IOC avoidance recommended until implemented

---

## Validation Procedures

### Sim-to-Live Parity Testing

Before deploying a strategy live, operators should:

1. **Paper Trading Phase**: Run strategy in paper mode with real market data
2. **Slippage Comparison**: Compare simulated fills to paper fills
3. **Latency Accounting**: Add realistic latency to simulation
4. **Fee Verification**: Confirm fee model matches broker schedule

### Recommended Calibration Data

- At least 30 days of execution data for statistical models
- Order sizes representative of target deployment
- Multiple market regimes (normal, volatile, low liquidity)

---

## References

- Execution Providers: `execution_providers.py`
- Fee Models: `execution_providers.py:FeeProvider` implementations
- Slippage Models: `execution_providers.py:SlippageProvider` implementations

---

*This document follows the Documentation Canon - avoiding absolute claims about simulation accuracy.*

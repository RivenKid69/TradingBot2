# Simulation Limitations and Validation Status

**Document Purpose**: Track known limitations in execution simulation and their validation status.

**Last Updated**: 2025-12-21

---

## Overview

CustodiaCloud's execution simulation aims to provide realistic backtesting and paper trading.
This document tracks known limitations and their potential impact on sim-to-live parity.

Per Documentation Canon: We make no guarantees about simulation accuracy. Users are responsible
for validating simulation results against live execution before deploying capital.

### Pre-Production Status and Client Responsibility

Per CCEA Design Doc Section 5.1: "Live Intent is created only on Agent (in strategy runtime)."
Cloud provides research and simulation tools; live execution validation is a client-side responsibility.

**Important distinctions**:

| Aspect | Platform Responsibility | Client Responsibility |
|--------|------------------------|----------------------|
| Simulation models | Provide documented models with honest limitation disclosure | Validate models against their execution data before live deployment |
| Calibration | Provide calibration interfaces and guidance | Perform calibration with their broker/execution data |
| Slippage/fill accuracy | Document known limitations and mitigations | Apply conservative multipliers and validate thresholds |
| Live deployment decision | Provide tools and risk controls | Own the decision to deploy capital |

**Tech Debt Tracking**: `docs/reports/TECH_DEBT_REGISTRY.md#sim-live-validation-framework`

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

**Deployment-Time Validation** (per-client responsibility):

Operators deploying live strategies must complete these validation steps before capital deployment. These are deployment-time requirements, not platform pre-release gates, per Documentation Canon Section 4.3 (no performance promises).

| Step | Description | Owner |
|------|-------------|-------|
| Slippage comparison | Compare simulated vs actual slippage for representative order set | Client ops team |
| Model calibration | Calibrate statistical slippage model against real execution data | Client quant team |
| Threshold documentation | Document acceptable slippage divergence thresholds for strategy | Client risk team |

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

### L3: Market Impact (Implemented)

**Status**: Implemented in `lob/market_impact.py`

**Implemented Models**:
- **KyleLambdaModel**: Kyle (1985) linear price impact with configurable permanent/temporary split
- **AlmgrenChrissModel**: Almgren-Chriss (2001) square-root model with permanent/temporary decomposition
- **GatheralModel**: Gatheral (2010) transient impact with power-law decay
- **CompositeImpactModel**: Weighted ensemble of multiple models
- **ImpactTracker**: Cumulative impact state tracking with decay

**Available Features**:
- Permanent vs temporary impact decomposition (compute_temporary_impact, compute_permanent_impact)
- Impact decay modeling (exponential, power-law, linear decay types)
- Configurable parameters per asset class (ImpactParameters.for_equity(), for_crypto())
- Optimal execution time computation (AlmgrenChrissModel.compute_optimal_execution_time)

**Integration Status**:
- Models available for L3 providers via `create_impact_model()` factory
- Per-client calibration required before production use

**Remaining Gaps**:
- Cross-asset impact correlation (not yet implemented)
- Live calibration workflow documentation

**Deployment-Time Validation** (per-client responsibility):

| Step | Description | Owner |
|------|-------------|-------|
| Model selection | Choose appropriate impact model for asset class | Client quant team |
| Parameter calibration | Calibrate eta/gamma/tau parameters against execution data | Client quant team |
| Validation report | Document calibrated parameters and validation results | Client risk team |

**Control Artifact**: Calibrated impact model parameters with validation report required before live deployment.
**Tech Debt Tracking**: docs/reports/TECH_DEBT_REGISTRY.md#L3-impact
**Status**: Implemented - models available; per-client calibration required

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

**Implementation Roadmap** (T2b milestone):

| Task | Status | Notes |
|------|--------|-------|
| IOC implementation | Planned (T2b) | Execute immediate match, cancel unfilled remainder |
| Conformance test stub | Done | `tests/cpp/test_orderbook_tif_conformance.cpp` |
| Exchange validation | Planned (T2b) | Validate against reference exchange matching engine behavior |

**Control Artifact**: `tests/cpp/test_orderbook_tif_conformance.cpp` (stub with GTEST_SKIP; T2b milestone).
**Tech Debt Tracking**: docs/reports/TECH_DEBT_REGISTRY.md#L4-tif
**Status**: Controlled - limitation documented, IOC avoidance recommended until T2b implementation

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

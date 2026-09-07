# Binance Spot Integration Conformance Requirements

**Version**: 1.0
**Date**: 2025-12-20
**Status**: Control Artifact
**Tech Debt Reference**: `docs/reports/TECH_DEBT_REGISTRY.md#arch-binance-spot-stub`

---

## Purpose

This document defines conformance requirements for Binance Spot private API integration.
Per CCEA Architecture (Design Doc Section 4.2), live broker connectors operate exclusively
in the Agent environment, not in Cloud.

---

## Current Implementation Status

### Cloud-Side (`adapters/binance_spot_private.py`)

| Function | Status | Behavior |
|----------|--------|----------|
| `get_account_info()` | Implemented | Returns account snapshot |
| `place_order()` | **Stub** | Raises `NotImplementedError` |
| `cancel_order()` | **Stub** | Raises `NotImplementedError` |
| `reconcile_state()` | Implemented | Compares local vs remote state |

**Design Rationale**:

- Stubs are intentional fail-closed implementations per CCEA Design Doc
- Cloud code MUST NOT execute live orders (Section 0.2, 0.4)
- Attempting to call these functions produces explicit, auditable errors
- Prevents accidental live trading from Cloud environment

### Agent-Side (Required for Live Trading)

Live order execution requires Agent deployment with:

1. Broker connector configured in Agent environment
2. Secrets stored in Local Vault (Agent-side only)
3. Local approval for TRADING_IMPACTING operations
4. Risk Manager enforcement active

---

## Conformance Test Requirements

### Phase 1: Sandbox/Paper Trading (Pre-Production)

Before any live deployment, the following tests MUST pass:

1. **Authentication Test**
   - [ ] API key authentication successful
   - [ ] Signature generation verified against Binance test endpoint
   - [ ] Error handling for invalid credentials

2. **Order Lifecycle Test (Sandbox)**
   - [ ] Place limit order (BUY)
   - [ ] Place limit order (SELL)
   - [ ] Cancel order
   - [ ] Verify order status transitions

3. **Reconciliation Test**
   - [ ] Local state matches exchange state
   - [ ] Missing orders detected
   - [ ] Position discrepancies detected

4. **Error Handling Test**
   - [ ] Rate limit handling (HTTP 429)
   - [ ] Server error recovery (HTTP 5xx)
   - [ ] Network timeout recovery

### Phase 2: Paper Trading Validation

After sandbox tests pass:

1. Execute paper trades for minimum 30 days
2. Compare simulated vs paper execution metrics
3. Document fill rate divergence

### Phase 3: Live Deployment (Agent-Only)

Agent implementation checklist:

- [ ] Secrets stored in Local Vault (never in Cloud)
- [ ] Order execution isolated in Agent process
- [ ] Kill switch configured and tested
- [ ] Reconciliation active with drift alerts
- [ ] Telemetry redaction verified

---

## Validation Report Template

Upon completing conformance testing, generate a report containing:

```yaml
conformance_report:
  date: YYYY-MM-DD
  environment: sandbox | paper | live
  adapter_version: X.Y.Z
  tests_passed: N/M
  test_results:
    authentication: pass | fail
    order_place: pass | fail | skipped
    order_cancel: pass | fail | skipped
    reconciliation: pass | fail
    error_handling: pass | fail
  notes: |
    Free-form notes about any issues or observations
  approved_by: [Name]
  approval_date: YYYY-MM-DD
```

---

## References

- CCEA Architecture: `archive/root_files/Design Doc CCEA Cloud.txt`
- Documentation Canon: `docs/DOCUMENTATION_CANON_DESIGN.md`
- Tech Debt Registry: `docs/reports/TECH_DEBT_REGISTRY.md#arch-binance-spot-stub`

---

**Document Control**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-20 | Initial conformance requirements |

*This document follows the Documentation Canon - no absolute claims about integration completeness.*

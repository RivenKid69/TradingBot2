# Fill-Rate Validation Report Template

**Document Purpose**: Template for client-side sim-to-live fill rate validation.

**Per CCEA Design Doc Section 5.1**: Live Intent is created only on Agent. Validation is a client-side deployment responsibility.

**Per Documentation Canon Section 4.3**: No performance guarantees. Validation is per-deployment.

---

## Report Metadata

| Field | Value |
|-------|-------|
| **Report Date** | `<FILL: YYYY-MM-DD>` |
| **Client/Deployment ID** | `<FILL: identifier>` |
| **Strategy ID** | `<FILL: strategy name/version>` |
| **Asset Class** | `<FILL: equities/options/futures/forex/crypto>` |
| **Execution Venue** | `<FILL: broker/exchange>` |
| **Validation Period** | `<FILL: start date to end date>` |
| **Prepared By** | `<FILL: analyst name/team>` |
| **Reviewed By** | `<FILL: risk officer name>` |

---

## 1. Executive Summary

`<FILL: 2-3 sentence summary of fill-rate comparison and recommendation>`

---

## 2. Data Collection

### 2.1 Order Sample

| Metric | Value |
|--------|-------|
| Total limit orders analyzed | `<FILL>` |
| Total market orders analyzed | `<FILL>` |
| Order types tested | `<FILL: GTC/IOC/POST_ONLY/etc>` |
| Size range | `<FILL: min-max>` |
| Validation period | `<FILL>` |

### 2.2 Data Sources

- Simulated fills: `<FILL: source (e.g., CustodiaCloud OHLCVFillProvider)>`
- Paper/Live fills: `<FILL: source (e.g., broker execution reports)>`
- Market data: `<FILL: source, granularity, latency>`

---

## 3. Fill Rate Comparison

### 3.1 Overall Fill Rates

| Order Type | Simulated Fill % | Actual Fill % | Difference |
|------------|------------------|---------------|------------|
| Limit orders (passive) | `<FILL>` | `<FILL>` | `<FILL>` |
| Limit orders (aggressive) | `<FILL>` | `<FILL>` | `<FILL>` |
| Market orders | `<FILL>` | `<FILL>` | `<FILL>` |

### 3.2 Time-to-Fill Analysis (Limit Orders)

| Metric | Simulated (seconds) | Actual (seconds) | Difference |
|--------|---------------------|------------------|------------|
| Mean time-to-fill | `<FILL>` | `<FILL>` | `<FILL>` |
| Median time-to-fill | `<FILL>` | `<FILL>` | `<FILL>` |
| 90th percentile | `<FILL>` | `<FILL>` | `<FILL>` |

### 3.3 Partial Fill Analysis

| Metric | Simulated | Actual | Difference |
|--------|-----------|--------|------------|
| Partial fill rate | `<FILL: %>` | `<FILL: %>` | `<FILL>` |
| Mean fill ratio | `<FILL>` | `<FILL>` | `<FILL>` |

### 3.4 By Order Size Bucket

| Size Bucket | Sim Fill % | Actual Fill % | Delta |
|-------------|------------|---------------|-------|
| Small | `<FILL>` | `<FILL>` | `<FILL>` |
| Medium | `<FILL>` | `<FILL>` | `<FILL>` |
| Large | `<FILL>` | `<FILL>` | `<FILL>` |

---

## 4. Queue Position Analysis

**Note**: CustodiaCloud LOBFillProvider uses OHLCV fallback and does not model queue position.
This section documents expected divergence from queue-aware fill models.

### 4.1 Queue Position Impact Assessment

| Scenario | Simulated Behavior | Expected Live Behavior | Gap Severity |
|----------|-------------------|----------------------|--------------|
| Joining deep queue | Fills based on price touch | May not fill | `<FILL>` |
| Front of queue | Same as deep queue | Fills first | `<FILL>` |
| Adverse selection | Not modeled | Lower fill on fading prices | `<FILL>` |

### 4.2 Mitigation Strategy

`<FILL: How does the client account for queue position limitations?>`

---

## 5. Known Limitations and Mitigations

### 5.1 Current Simulation Model Limitations

| Limitation | Impact | Mitigation Applied |
|------------|--------|-------------------|
| OHLCV fallback (no L3) | Optimistic fill timing | `<FILL>` |
| No queue position | Over-estimates passive fills | `<FILL>` |
| No adverse selection | Over-estimates fill rate | `<FILL>` |

Reference: `docs/SIMULATION_LIMITATIONS.md#L2`

### 5.2 IOC Orders (L4-tif)

**IMPORTANT**: IOC orders currently behave as GTC in simulation.

| Aspect | Current Behavior | Expected Behavior | Mitigation |
|--------|-----------------|-------------------|------------|
| Unfilled portion | Remains on book | Cancelled | `<FILL: e.g., avoid IOC in sim>` |

Reference: `docs/SIMULATION_LIMITATIONS.md#L4`

---

## 6. Risk Assessment

### 6.1 Sim-to-Live Gap Analysis

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Over-optimistic fill rate | `<FILL>` | `<FILL>` | `<FILL>` |
| Under-estimated time-to-fill | `<FILL>` | `<FILL>` | `<FILL>` |
| Partial fill handling | `<FILL>` | `<FILL>` | `<FILL>` |

### 6.2 Acceptable Divergence Thresholds

| Metric | Threshold | Action if Exceeded |
|--------|-----------|-------------------|
| Fill rate difference | `<FILL: e.g., +/- 10%>` | `<FILL>` |
| Time-to-fill divergence | `<FILL>` | `<FILL>` |

---

## 7. Conclusion and Approval

### 7.1 Recommendation

- [ ] **APPROVED** for live deployment with documented limitations
- [ ] **CONDITIONAL** - requires additional validation: `<FILL>`
- [ ] **NOT APPROVED** - reason: `<FILL>`

### 7.2 Conditions for Live Deployment

1. `<FILL: condition 1, e.g., conservative position sizing>`
2. `<FILL: condition 2, e.g., avoid IOC orders until L4-tif implemented>`
3. `<FILL: condition 3>`

### 7.3 Signatures

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Quant Analyst | `<FILL>` | `<FILL>` | _________ |
| Risk Officer | `<FILL>` | `<FILL>` | _________ |
| Operations Lead | `<FILL>` | `<FILL>` | _________ |

---

## Appendix A: Test Order Specifications

`<FILL: Details of orders used for validation - sizes, prices, timing>`

---

## Appendix B: Raw Data Reference

- Simulation logs: `<FILL: path/reference>`
- Broker execution reports: `<FILL: path/reference>`
- Comparison analysis: `<FILL: path/reference>`

---

*Template Version: 1.0 | Created: 2025-12-22 | Tech Debt Ref: L2-fill, L4-tif*

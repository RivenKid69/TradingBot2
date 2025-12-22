# TCA (Transaction Cost Analysis) Calibration Report Template

**Document Purpose**: Template for client-side sim-to-live slippage calibration validation.

**Per CCEA Design Doc Section 5.1**: Live Intent is created only on Agent. Calibration is a client-side deployment responsibility.

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

`<FILL: 2-3 sentence summary of calibration results and recommendation>`

---

## 2. Data Collection

### 2.1 Order Sample

| Metric | Value |
|--------|-------|
| Total orders analyzed | `<FILL>` |
| Order types | `<FILL: market/limit/IOC/etc>` |
| Size range | `<FILL: min-max shares/contracts>` |
| Date range | `<FILL>` |

### 2.2 Data Sources

- Simulated fills: `<FILL: source (e.g., CustodiaCloud backtest engine)>`
- Actual fills: `<FILL: source (e.g., broker execution reports)>`
- Market data: `<FILL: source and latency characteristics>`

---

## 3. Slippage Comparison

### 3.1 Summary Statistics

| Metric | Simulated (bps) | Actual (bps) | Difference (bps) |
|--------|-----------------|--------------|------------------|
| Mean slippage | `<FILL>` | `<FILL>` | `<FILL>` |
| Median slippage | `<FILL>` | `<FILL>` | `<FILL>` |
| 95th percentile | `<FILL>` | `<FILL>` | `<FILL>` |
| 99th percentile | `<FILL>` | `<FILL>` | `<FILL>` |
| Max slippage | `<FILL>` | `<FILL>` | `<FILL>` |

### 3.2 By Order Size Bucket

| Size Bucket | Order Count | Sim Mean (bps) | Actual Mean (bps) | Delta |
|-------------|-------------|----------------|-------------------|-------|
| Small (<1000 shares) | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| Medium (1000-10000) | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| Large (>10000) | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

### 3.3 By Market Regime

| Regime | Order Count | Sim Mean (bps) | Actual Mean (bps) | Delta |
|--------|-------------|----------------|-------------------|-------|
| Normal volatility | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| High volatility | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| Low liquidity | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

---

## 4. Model Calibration

### 4.1 Selected Slippage Model

- Model type: `<FILL: StatisticalSlippageProvider / LOBSlippageProvider / custom>`
- Configuration: `<FILL: parameters used>`

### 4.2 Calibrated Parameters

| Parameter | Value | Calibration Method |
|-----------|-------|-------------------|
| `<param1>` | `<FILL>` | `<FILL: regression/historical mean/etc>` |
| `<param2>` | `<FILL>` | `<FILL>` |

### 4.3 Conservative Multiplier

Based on calibration results, recommended slippage multiplier: `<FILL: e.g., 1.5x>`

Rationale: `<FILL: why this multiplier is appropriate>`

---

## 5. Risk Assessment

### 5.1 Sim-to-Live Gap Analysis

| Risk Category | Severity | Mitigation |
|---------------|----------|------------|
| Slippage underestimation for large orders | `<FILL: Low/Medium/High>` | `<FILL>` |
| Adverse selection not modeled | `<FILL>` | `<FILL>` |
| Latency not accounted | `<FILL>` | `<FILL>` |

### 5.2 Acceptable Divergence Thresholds

| Metric | Threshold | Action if Exceeded |
|--------|-----------|-------------------|
| Mean slippage divergence | `<FILL: e.g., +/- 3 bps>` | `<FILL: pause/alert/review>` |
| 95th percentile divergence | `<FILL>` | `<FILL>` |

---

## 6. Conclusion and Approval

### 6.1 Recommendation

- [ ] **APPROVED** for live deployment with calibrated parameters
- [ ] **CONDITIONAL** - requires additional validation: `<FILL>`
- [ ] **NOT APPROVED** - reason: `<FILL>`

### 6.2 Signatures

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Quant Analyst | `<FILL>` | `<FILL>` | _________ |
| Risk Officer | `<FILL>` | `<FILL>` | _________ |
| Operations Lead | `<FILL>` | `<FILL>` | _________ |

---

## Appendix A: Validation Methodology

`<FILL: Describe statistical tests, confidence intervals, sample size justification>`

---

## Appendix B: Raw Data Reference

- Simulation data location: `<FILL: path/reference>`
- Broker execution reports: `<FILL: path/reference>`
- Analysis scripts: `<FILL: path/reference>`

---

*Template Version: 1.0 | Created: 2025-12-22 | Tech Debt Ref: L1-slippage*

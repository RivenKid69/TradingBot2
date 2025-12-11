# DORA Proportionality Assessment

## Phase 0: Proportionality Assessment Report

**Document Version**: 1.0
**Assessment Date**: [TO BE COMPLETED]
**Regulation Reference**: Regulation (EU) 2022/2554 (DORA)
**Applicable Articles**: Articles 2, 3(22), 4, 16

---

## 1. Executive Summary

This document records the DORA proportionality assessment for the AI-Powered Quantitative Research Platform. The assessment determines:

1. Whether DORA applies to the entity (Article 2 scope verification)
2. Which business functions are "critical or important" (Article 3(22))
3. What DORA regime applies (full, simplified, or microenterprise exemptions)
4. How requirements should be implemented proportionately (Article 4)

### Assessment Status

| Item | Status | Notes |
|------|--------|-------|
| Scope Verification | ⏳ Pending | Article 2 check required |
| Function Classification | ⏳ Pending | Article 3(22) assessment required |
| Proportionality Determination | ⏳ Pending | Articles 4, 16 assessment required |
| Legal Review | ⏳ Pending | Required before implementation |
| NCA Confirmation | ⏳ Not Started | If required |

---

## 2. DORA Scope Verification (Article 2)

### 2.1 Entity Type Determination

Per DORA Article 2(1), the regulation applies to 21 types of financial entities.

**For algorithmic trading platforms, the most likely classifications are:**

| Entity Type | Article | Description | Applicability |
|-------------|---------|-------------|---------------|
| Investment Firm | 2(1)(e) | MiFID II authorized investment firm | **Most Likely** |
| Crypto-Asset Service Provider | 2(1)(f) | MiCA authorized CASP | If crypto trading |
| Trading Venue | 2(1)(i) | If operating own venue | Unlikely |
| AIFM | 2(1)(k) | If managing AIFs | If applicable |

### 2.2 Authorization Check

| Field | Value |
|-------|-------|
| Legal Name | [TO BE COMPLETED] |
| LEI | [TO BE COMPLETED] |
| Authorization Type | [TO BE COMPLETED] |
| Authorizing NCA | [TO BE COMPLETED] |
| Member State | [TO BE COMPLETED] |
| Authorization Reference | [TO BE COMPLETED] |

### 2.3 Scope Result

**DORA Applies**: [ ] Yes / [ ] No / [ ] Unclear - Legal Review Required

**Article Reference**: [TO BE COMPLETED]

**Notes**: [TO BE COMPLETED]

---

## 3. Critical/Important Function Classification (Article 3(22))

### 3.1 Definition

Per Article 3(22), a function is "critical or important" if its disruption would materially impair:

1. **Financial performance** of the financial entity, OR
2. **Soundness or continuity** of its services and activities, OR
3. **Compliance** with authorization conditions or regulatory obligations

### 3.2 Platform Functions Assessment

| Function | Category | Financial Impact | Service Impact | Compliance Impact | Classification |
|----------|----------|------------------|----------------|-------------------|----------------|
| Order Execution | Trading | ✅ Yes | ✅ Yes | ✅ Yes | **CRITICAL** |
| Market Data | Data | ✅ Yes | ✅ Yes | ❌ No | **CRITICAL** |
| Risk Monitoring | Risk | ✅ Yes | ✅ Yes | ✅ Yes | **CRITICAL** |
| Kill Switch | Risk | ❌ No | ✅ Yes | ✅ Yes | **CRITICAL** |
| Regulatory Reporting | Compliance | ❌ No | ❌ No | ✅ Yes | **IMPORTANT** |
| Audit Trail | Compliance | ❌ No | ❌ No | ✅ Yes | **IMPORTANT** |
| User Authentication | Security | ❌ No | ✅ Yes | ✅ Yes | **IMPORTANT** |
| Position Reconciliation | Operations | ✅ Yes | ✅ Yes | ❌ No | **CRITICAL** |
| Backtesting | Research | ❌ No | ❌ No | ❌ No | Standard |
| Model Training | Research | ❌ No | ❌ No | ❌ No | Standard |

### 3.3 Third-Party Providers Supporting Critical Functions

| Provider | Services | Critical Function Support | Substitutability |
|----------|----------|--------------------------|------------------|
| Binance | Market Data, Execution | ✅ Yes | Medium |
| Alpaca | Market Data, Execution | ✅ Yes | Medium |
| Polygon | Market Data | ✅ Yes | Low (alternatives exist) |
| OANDA | Forex Execution | ✅ Yes | Medium |
| Interactive Brokers | Multi-asset Execution | ✅ Yes | High difficulty |
| Deribit | Crypto Options | ✅ Yes | High difficulty |

---

## 4. Proportionality Assessment (Articles 4, 16)

### 4.1 Entity Size Classification

Per EU Recommendation 2003/361 (SME Definition):

| Category | Employees | Turnover OR Balance Sheet | Status |
|----------|-----------|---------------------------|--------|
| Microenterprise | < 10 | < €2M | [CHECK] |
| Small Enterprise | < 50 | < €10M | [CHECK] |
| Medium Enterprise | < 250 | < €50M (turnover) / < €43M (balance) | [CHECK] |
| Large Enterprise | ≥ 250 | ≥ €50M | [CHECK] |

**Entity Metrics**:
- Employee Count: [TO BE COMPLETED]
- Annual Turnover (EUR): [TO BE COMPLETED]
- Balance Sheet (EUR): [TO BE COMPLETED]

**Size Classification**: [TO BE DETERMINED]

### 4.2 Article 16 Simplified Framework Check

| Exemption Category | Article | Applicable? | Evidence |
|-------------------|---------|-------------|----------|
| Small/non-interconnected investment firm | 16(1)(a) | [ ] | |
| PSD2 exempted payment institution | 16(1)(b) | [ ] | |
| CRD exempted institution | 16(1)(c) | [ ] | |
| EMD2 exempted e-money institution | 16(1)(d) | [ ] | |
| Small IORP | 16(1)(e) | [ ] | |

**Qualifies for Simplified Framework**: [ ] Yes / [ ] No

### 4.3 Microenterprise Exemptions

If entity qualifies as microenterprise, the following exemptions apply:

| Exemption | Article | Description | Applicable? |
|-----------|---------|-------------|-------------|
| No TPP Strategy | 28(2) | No third-party ICT risk strategy required | [CHECK] |
| Simplified Risk Mgmt | 6(6) | Simplified ICT risk management framework | [CHECK] |
| No Recurring Incidents | CDR 2024/1772 Art. 11(3) | No recurring incident assessment | [CHECK] |
| No Training Requirement | 5(4) | No mandatory management body ICT training | [CHECK] |

### 4.4 Determined Regime

Based on the assessment:

**Applicable Regime**: [ ] FULL / [ ] SIMPLIFIED / [ ] MICROENTERPRISE

**Rationale**: [TO BE COMPLETED]

---

## 5. Proportionality Implementation Guidance

### 5.1 Article 4 Proportionality Factors

| Factor | Assessment | Implementation Impact |
|--------|------------|----------------------|
| Size | [small/medium/large] | [Impact description] |
| Risk Profile | [low/medium/high] | [Impact description] |
| Nature | [entity type] | [Impact description] |
| Scale | [scope of operations] | [Impact description] |
| Complexity | [operational complexity] | [Impact description] |

### 5.2 Requirements by Regime

#### Full Regime Requirements
- Complete ICT risk management framework (Articles 5-15)
- Full third-party risk strategy (Article 28)
- Comprehensive testing program (Articles 24-25)
- TLPT if designated (Article 26)

#### Simplified Regime Requirements
- Article 16 simplified framework (replaces Articles 5-15)
- Basic third-party risk management
- Simplified testing requirements

#### Microenterprise Exemptions
- No third-party ICT risk strategy (Article 28(2))
- Simplified ICT risk management (Article 6(6))
- No recurring incident assessment (CDR 2024/1772 Art. 11(3))
- No management body training (Article 5(4))

---

## 6. Key Dates and Deadlines

| Deadline | Requirement | Status |
|----------|-------------|--------|
| 17 Jan 2025 | DORA Application Date | ⚠️ PASSED |
| 31 Mar 2025 | Reference Date for Register of Information | ⏳ Upcoming |
| 30 Apr 2025 | Register of Information submission (via NCA) | ⏳ Upcoming |
| Ongoing | Major incident reporting capability | 🔴 Required NOW |
| Jan 2026 | First annual ICT risk assessment review | ⏳ |

---

## 7. Recommended Actions

### Immediate Actions (Within 2 Weeks)
1. [ ] Complete entity identification (legal name, LEI)
2. [ ] Document size metrics with evidence
3. [ ] Determine entity type and authorization status
4. [ ] Assess Article 16 exemption eligibility
5. [ ] Complete this proportionality assessment

### Short-Term Actions (Within 1 Month)
1. [ ] Obtain legal review of classification
2. [ ] Begin Register of Information preparation
3. [ ] Implement major incident reporting capability
4. [ ] Document critical/important function classification

### Medium-Term Actions (Within 3 Months)
1. [ ] Complete Register of Information
2. [ ] Submit to NCA per deadline
3. [ ] Implement proportionate ICT risk framework
4. [ ] Establish third-party risk management process

---

## 8. Supporting Documentation

| Document | Location | Status |
|----------|----------|--------|
| Entity Classification Config | `services/archive/dora_financial_entity/configs/entity_classification.yaml` | Archived (FE-specific) |
| NCA Identification Config | `services/archive/dora_financial_entity/configs/nca_identification.yaml` | Archived (FE-specific) |
| Proportionality Config | `services/archive/dora_financial_entity/configs/proportionality_assessment.yaml` | Archived (FE-specific) |
| Financial Statements | [TO BE ADDED] | Required |
| Authorization Documents | [TO BE ADDED] | Required |

> **Note:** These configs are for Financial Entity (FE) compliance. As an ICT Provider,
> we use `configs/dora/` for our operational configurations.

---

## 9. Review and Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Assessor | | | |
| Legal Reviewer | | | |
| Compliance Officer | | | |
| Management Approval | | | |

---

## 10. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | [DATE] | System | Initial assessment template |

---

## Appendix A: DORA Article References

- **Article 2** (Scope): https://www.digital-operational-resilience-act.com/Article_2.html
- **Article 3** (Definitions): https://www.dora-info.eu/dora/article-3/
- **Article 4** (Proportionality): https://www.digital-operational-resilience-act.com/Article_4.html
- **Article 16** (Simplified Framework): https://www.digital-operational-resilience-act.com/Article_16.html
- **EU SME Definition**: https://ec.europa.eu/growth/smes/sme-definition_en

## Appendix B: Related Platform Modules

**Archived Financial Entity Modules** (for building FE compliance tools):
- `services/archive/dora_financial_entity/scope_verification.py` - Article 2 scope verification
- `services/archive/dora_financial_entity/function_classification.py` - Article 3(22) function classification
- `services/archive/dora_financial_entity/proportionality.py` - Articles 4, 16 proportionality assessment

> **ICT Provider Note:** As an ICT Third-Party Provider (Art. 30), these FE-specific modules
> are archived. Our active DORA modules are in `services/dora_integration/`.

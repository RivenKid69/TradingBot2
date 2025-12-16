# Copyright Compliance Policy

**Document ID**: CCP-2025-001
**Version**: 1.0
**Date**: 2025-12-10
**Regulation Reference**: EU AI Act Article 53(1)(c), DSM Directive 2019/790 Article 4

---

## 1. Policy Statement

In accordance with Article 53(1)(c) of Regulation (EU) 2024/1689 (EU AI Act), this policy establishes our commitment to compliance with Union law on copyright and related rights in the context of AI model training.

**Article 53(1)(c) Requirement:**
> "put in place a policy to comply with Union law on copyright and related rights, and in particular to identify and comply with, including through state-of-the-art technologies, a reservation of rights expressed pursuant to Article 4(3) of Directive (EU) 2019/790"

## 2. Scope

This policy applies to all training data used for:

- Reinforcement learning model training (Distributional PPO)
- Feature engineering and preprocessing pipelines
- Backtesting and validation datasets
- Adversarial scenario generation
- Model fine-tuning and updates

## 3. Training Data Categories

### 3.1 Market Data (Primary)

| Category | Copyright Status | Legal Rationale |
|----------|-----------------|-----------------|
| OHLCV Price Data | Not copyrightable | Factual data without creative expression |
| Order Book Data | Not copyrightable | Raw market microstructure data |
| Volume Statistics | Not copyrightable | Aggregated factual statistics |
| Trade Execution Data | Not copyrightable | Factual transaction records |

**Legal Basis**: Price, volume, and trading data constitute factual information not subject to copyright protection under EU law. Per the Database Directive 96/9/EC, factual data compilations may have sui generis database rights, but individual data points are not copyrightable.

### 3.2 Licensed Data

| Provider | License Type | Usage Rights | Verification Date |
|----------|--------------|--------------|-------------------|
| Polygon.io | Commercial API License | ML training permitted | 2024-12-01 |
| Alpha Vantage | API Terms of Service | Research and commercial use | 2024-12-01 |
| Binance | API Terms of Service | Historical data access | 2024-12-01 |
| OANDA | API License | Forex data for analysis | 2024-12-01 |

All licensed data sources are reviewed annually to ensure continued compliance with license terms.

### 3.3 Synthetic Data

Generated data for model robustness:

- Adversarial market scenarios (SA-PPO framework)
- Stress test scenarios (flash crash, gap events)
- Distribution shift simulations
- Edge case generation

**Copyright Status**: Owned by us; no third-party rights applicable.

### 3.4 Computed Features

Technical indicators and derived features:

- Mathematical formulas (RSI, MACD, Bollinger Bands)
- Statistical measures (Z-scores, rolling statistics)
- Custom feature engineering outputs

**Copyright Status**: Mathematical algorithms and formulas are not copyrightable. Computed outputs from factual data inherit no copyright.

## 4. Opt-Out Compliance (Article 4(3) DSM Directive)

### 4.1 Mechanisms Monitored

We implement state-of-the-art technologies to identify opt-out reservations:

#### 4.1.1 robots.txt Directives

Monitored directives include:
```
User-agent: GPTBot
User-agent: AI-Training
User-agent: CCBot
User-agent: Google-Extended
User-agent: anthropic-ai
User-agent: Claude-Web
```

#### 4.1.2 TDMRep Protocol

Per W3C TDMRep specification:

- **HTTP Headers**: `TDM-Reservation: 1`
- **HTML Meta Tags**: `<meta name="tdm-reservation" content="1">`
- **tdmrep.json**: Machine-readable policy files

#### 4.1.3 ai.txt Standard

Emerging standard for AI training opt-out:
```
# ai.txt
User-agent: *
Disallow-AI-Training: /
```

#### 4.1.4 Direct Communications

- Email notices to: copyright@[company].com
- Postal correspondence
- Legal notices via authorized representatives

### 4.2 Opt-Out Compliance Process

```
┌─────────────────────────────┐
│   Data Source Identified    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Check robots.txt / TDMRep  │
│  / ai.txt / Direct Notice   │
└──────────────┬──────────────┘
               │
               ▼
        ┌──────────────┐
        │ Opt-out      │───Yes───▶ Exclude from Training
        │ Found?       │          Log Decision & Evidence
        └──────┬───────┘
               │ No
               ▼
┌─────────────────────────────┐
│  Record Compliance Check    │
│  Store Evidence Hash        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Proceed with Training     │
└─────────────────────────────┘
```

### 4.3 Check Frequency

| Source Type | Check Frequency |
|-------------|-----------------|
| Web sources | Before each ingestion |
| API sources | Quarterly review |
| Licensed data | Annual license review |
| Direct notices | Upon receipt |

## 5. Training Data Sources Registry

### 5.1 Current Sources

| Source ID | Name | Type | Copyright Status | Last Verified |
|-----------|------|------|------------------|---------------|
| binance_ohlcv | Binance OHLCV Data | Public Market Data | Not Applicable | 2024-12-01 |
| polygon_stocks | Polygon.io Stock Data | Licensed Data | Licensed | 2024-12-01 |
| alpha_vantage | Alpha Vantage Data | Licensed Data | Licensed | 2024-12-01 |
| internal_synthetic | Synthetic Scenarios | Synthetic | Not Applicable | 2024-12-01 |
| technical_indicators | Computed Indicators | Proprietary | Not Applicable | 2024-12-01 |

### 5.2 Compliance Status

| Metric | Value |
|--------|-------|
| Total Sources | 5 |
| Opt-Out Checked | 5 (100%) |
| Licensed Sources | 2 |
| Public Domain / Not Applicable | 3 |
| Pending Review | 0 |

## 6. Rights Holder Requests

### 6.1 Request Types

Rights holders may submit:

1. **Information Requests**: Inquire about use of their content
2. **Opt-Out Notices**: Request exclusion from future training
3. **Removal Requests**: Request removal from existing datasets (where technically feasible)
4. **General Inquiries**: Questions about our copyright practices

### 6.2 Contact Information

**Email**: copyright@[company].com
**Postal Address**: [Company Address]
**Response Time**: 30 business days

### 6.3 Request Process

1. Request received and logged
2. Initial acknowledgment within 5 business days
3. Investigation and verification
4. Response with action taken within 30 business days
5. Implementation of any required changes

## 7. Record Keeping

We maintain comprehensive records of:

| Record Type | Retention Period | Storage |
|-------------|------------------|---------|
| Training data sources | System lifecycle + 10 years | Secure database |
| Opt-out checks | System lifecycle + 10 years | Audit logs |
| License agreements | Agreement term + 10 years | Legal archive |
| Rights holder requests | 10 years from resolution | Compliance database |
| Evidence hashes | System lifecycle + 10 years | Immutable storage |

## 8. Compliance Verification

### 8.1 Internal Audits

| Audit Type | Frequency | Scope |
|------------|-----------|-------|
| Source inventory review | Quarterly | All registered sources |
| Opt-out mechanism check | Semi-annual | Monitoring systems |
| License compliance | Annual | All licensed sources |
| Full compliance audit | Annual | Entire copyright program |

### 8.2 Documentation

Each audit produces:
- Findings report
- Remediation plan (if needed)
- Updated compliance status
- Sign-off by compliance officer

## 9. Updates and Changes

This policy is reviewed and updated:

- **Annually**: Scheduled compliance review
- **As needed**: Upon changes in EU copyright law
- **Upon request**: Following GPAI Code of Practice updates
- **When required**: Upon significant changes to training data sources

## 10. Related Documents

- [TRAINING_DATA_SUMMARY.md](TRAINING_DATA_SUMMARY.md) - Article 53(1)(d) summary
- [TERMS_OF_SERVICE.md](../legal/TERMS_OF_SERVICE.md) - Terms of Service with AI disclosure
- [EU_AI_ACT_INTEGRATION_PLAN.md](EU_AI_ACT_INTEGRATION_PLAN.md) - Overall compliance plan

---

## Appendix A: Legal References

| Reference | Description |
|-----------|-------------|
| EU AI Act Article 53(1)(c) | GPAI copyright compliance requirement |
| DSM Directive 2019/790 Article 4 | Text and Data Mining exception |
| DSM Directive 2019/790 Article 4(3) | Opt-out mechanism |
| Database Directive 96/9/EC | Sui generis database rights |
| TDMRep W3C Specification | Technical opt-out standard |

## Appendix B: Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-10 | Initial release | Compliance Team |

---

*This policy is provided in accordance with Article 53(1)(c) of Regulation (EU) 2024/1689 (EU AI Act).*

**Last Updated**: 2025-12-16
**Next Review**: 2026-06-10

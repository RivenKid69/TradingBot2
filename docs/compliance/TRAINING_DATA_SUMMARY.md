# Training Data Summary

**Model**: Distributional PPO Trading Model
**Version**: 4.0
**Summary Date**: 2025-12-10
**Regulation Reference**: EU AI Act Article 53(1)(d)

---

## 1. Overview

This document provides a sufficiently detailed summary of the content used for training our general-purpose AI model, as required by Article 53(1)(d) of Regulation (EU) 2024/1689 (EU AI Act).

| Metric | Value |
|--------|-------|
| Total Training Samples | 340,000,000 |
| Total Data Size | 60 GB |
| Training Period | 2010-01 to 2024-12 |
| Number of Datasets | 5 |
| Asset Classes Covered | Crypto, Equities, Forex |

## 2. Data Categories

| Category | Dataset Count | Description |
|----------|---------------|-------------|
| Market Data | 3 | OHLCV price and volume data |
| Technical Indicators | 1 | Computed analytical features |
| Synthetic Data | 1 | Generated adversarial scenarios |

## 3. Datasets Used

### 3.1 Binance Spot OHLCV

Cryptocurrency spot market OHLCV (Open, High, Low, Close, Volume) data for major trading pairs.

| Property | Value |
|----------|-------|
| Category | Market Data |
| Provider | Binance API |
| Time Range | 2017-01-01 to 2024-12-01 |
| Size | 50,000,000 rows (15 GB) |
| Geographic Coverage | Global |
| Update Frequency | 1-minute bars |
| Data Format | Tabular CSV/Parquet |
| Quality Level | High |
| Personal Data | No |

**Assets Covered**: BTC/USDT, ETH/USDT, BNB/USDT, Major altcoins

**Preprocessing Steps**:

- Outlier detection and removal (>5 sigma)
- Gap filling using forward-fill (max 5 bars)
- Volume normalization per asset
- Z-score normalization for model input

---

### 3.2 US Equity Data

US stock market data including OHLCV, trades, and quotes.

| Property | Value |
|----------|-------|
| Category | Market Data |
| Provider | Polygon.io |
| Time Range | 2010-01-01 to 2024-12-01 |
| Size | 100,000,000 rows (25 GB) |
| Geographic Coverage | United States |
| Update Frequency | 1-minute bars |
| Data Format | Tabular CSV/Parquet |
| Quality Level | High |
| Personal Data | No |

**Assets Covered**: S&P 500 constituents, Russell 2000 constituents

**Preprocessing Steps**:

- Corporate actions adjustment (splits, dividends)
- Exchange code normalization
- Timestamp alignment to market hours
- Duplicate removal

---

### 3.3 Forex Major Pairs

Foreign exchange data for major currency pairs.

| Property | Value |
|----------|-------|
| Category | Market Data |
| Provider | OANDA / Alpha Vantage |
| Time Range | 2010-01-01 to 2024-12-01 |
| Size | 30,000,000 rows (8 GB) |
| Geographic Coverage | Global |
| Update Frequency | 1-minute bars |
| Data Format | Tabular |
| Quality Level | High |
| Personal Data | No |

**Assets Covered**: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD

**Preprocessing Steps**:

- Weekend gap handling
- Spread validation
- Tick-to-bar aggregation

---

### 3.4 Technical Indicators

Computed technical analysis features derived from price and volume data.

| Property | Value |
|----------|-------|
| Category | Technical Indicators |
| Provider | Internal computation |
| Time Range | 2010-01-01 to 2024-12-01 |
| Size | 150,000,000 rows (10 GB) |
| Geographic Coverage | Global |
| Update Frequency | Computed from OHLCV |
| Data Format | Tabular feature matrices |
| Quality Level | High |
| Personal Data | No |

**Features Computed**:

- Price momentum indicators (RSI, MACD, Stochastic)
- Volatility measures (ATR, Bollinger Bands, historical volatility)
- Volume analysis (OBV, VWAP, volume profiles)
- Trend indicators (moving averages, ADX)
- Market microstructure features

**Preprocessing Steps**:

- Rolling window calculation
- Z-score normalization
- Winsorization at 1st/99th percentile
- NaN handling (forward-fill then backfill)

---

### 3.5 Adversarial Scenarios

Synthetically generated adversarial market scenarios for model robustness training.

| Property | Value |
|----------|-------|
| Category | Synthetic Data |
| Provider | Internal generation (SA-PPO) |
| Time Range | 2024-01-01 to 2024-12-01 |
| Size | 10,000,000 rows (2 GB) |
| Geographic Coverage | N/A (synthetic) |
| Update Frequency | Generated per training run |
| Data Format | Numpy arrays |
| Quality Level | High |
| Personal Data | No |

**Scenario Types**:

- Flash crash simulations
- Liquidity crisis scenarios
- High volatility regimes
- Trend reversals
- Black swan events

**Preprocessing Steps**:

- Scenario validation against statistical bounds
- Distribution alignment with real market moments
- Extreme event calibration

---

## 4. Data Quality Measures

We apply the following quality measures to all training data:

1. **Outlier Detection**: Statistical and ML-based outlier detection and removal
2. **Completeness Checks**: >99.5% data availability required for dataset inclusion
3. **Temporal Consistency**: Validation to prevent look-ahead bias
4. **Cross-Source Reconciliation**: Price verification across overlapping sources
5. **Distribution Monitoring**: Continuous monitoring for data drift
6. **Pipeline Testing**: Unit tests for data pipeline integrity
7. **Manual Review**: Human review of edge cases and anomalies

## 5. Bias Mitigation

We implement the following bias mitigation strategies:

| Bias Type | Mitigation Strategy |
|-----------|---------------------|
| Temporal Bias | Sampling across market regimes (bull/bear/sideways) |
| Asset Class Bias | Balanced representation across asset types |
| Survivorship Bias | Include delisted/bankrupt assets where available |
| Look-ahead Bias | Strict point-in-time feature engineering |
| Selection Bias | Stratified sampling across time periods |
| Geographic Bias | Include diverse global markets |

**Regular Audits**: Statistical tests (KS, Chi-square) performed quarterly.

## 6. Data Collection Methodology

Data is collected through:

1. **Licensed API Connections**: Direct feeds from data providers
2. **Exchange APIs**: Public exchange data endpoints
3. **Internal Computation**: Derived features and indicators
4. **Algorithmic Generation**: Synthetic scenario creation

All data collection:

- Respects API rate limits
- Complies with terms of service
- Follows data protection regulations
- Maintains audit trails

## 7. Labeling Methodology

**N/A - Reinforcement Learning Model**

This model uses reinforcement learning and does not require traditional labeled data. The learning signal derives from:

1. **Trading Rewards**: Profit and loss (P&L), risk-adjusted returns
2. **Risk Penalties**: Drawdown, volatility, concentration risk
3. **Constraint Satisfaction**: Position limits, exposure constraints

No human labeling or annotation is performed.

## 8. Personal Data Statement

**No personal data is used for training.**

All training data consists exclusively of:

- Aggregated market statistics
- Price and volume information
- Computed technical indicators
- Synthetically generated scenarios

The model does not process, store, or learn from any personally identifiable information (PII). This includes:

- No user trading history
- No account information
- No demographic data
- No behavioral data tied to individuals

## 9. Copyright Compliance

For detailed copyright compliance information, see [COPYRIGHT_POLICY.md](COPYRIGHT_POLICY.md).

**Summary**:

- All data sources are documented and reviewed
- Opt-out mechanisms are monitored per DSM Directive Article 4(3)
- Licensed data complies with provider terms
- Market data is factual and not subject to copyright

## 10. Data Updates

This summary is updated:

| Trigger | Action |
|---------|--------|
| Major model version | Full summary revision |
| New data source added | Append to dataset list |
| Significant data change | Update relevant sections |
| Annual compliance review | Full review and update |

---

## Appendix A: Data Source Summary Table

| Dataset | Category | Rows | Size | Provider | Period |
|---------|----------|------|------|----------|--------|
| Binance Spot OHLCV | Market Data | 50M | 15 GB | Binance | 2017-2024 |
| US Equity Data | Market Data | 100M | 25 GB | Polygon.io | 2010-2024 |
| Forex Major Pairs | Market Data | 30M | 8 GB | OANDA/AV | 2010-2024 |
| Technical Indicators | Technical | 150M | 10 GB | Internal | 2010-2024 |
| Adversarial Scenarios | Synthetic | 10M | 2 GB | Internal | 2024 |

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-10 | Initial release for EU AI Act compliance |

---

*This summary is provided in accordance with Article 53(1)(d) of Regulation (EU) 2024/1689 (EU AI Act).*

*For questions regarding this summary, please contact: compliance@[company].com*

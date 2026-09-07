# Test Data Collection Plan

## Phase 0 Deliverable: Data Collection Strategy

**Version**: 1.0
**Date**: 2025-11-30
**Status**: COMPLETE

---

## 1. Overview

This document outlines the strategy for collecting historical futures data for development, testing, and validation of the futures integration.

### Data Categories

| Category | Source | Priority | Volume |
|----------|--------|----------|--------|
| Crypto Perpetual OHLCV | Binance API | P0 | ~2GB |
| Crypto Perpetual Funding | Binance API | P0 | ~100MB |
| Crypto Mark Price | Binance API | P0 | ~500MB |
| CME Index Futures | IB TWS / Polygon | P1 | ~1GB |
| CME Commodity Futures | IB TWS / Polygon | P1 | ~500MB |
| CME Currency Futures | IB TWS / Polygon | P2 | ~300MB |

---

## 2. Crypto Perpetual Data (Binance)

### 2.1 Symbols

**Tier 1 (Primary)**:

- `BTCUSDT` - Bitcoin perpetual
- `ETHUSDT` - Ethereum perpetual

**Tier 2 (Secondary)**:

- `BNBUSDT` - BNB perpetual
- `SOLUSDT` - Solana perpetual
- `XRPUSDT` - XRP perpetual

**Tier 3 (Validation)**:

- `AVAXUSDT`, `DOGEUSDT`, `MATICUSDT`

### 2.2 Data Types & Endpoints

| Data Type | Endpoint | Timeframes | Date Range |
|-----------|----------|------------|------------|
| OHLCV | `/fapi/v1/klines` | 1m, 5m, 15m, 1h, 4h, 1d | 2020-01-01 to present |
| Mark Price | `/fapi/v1/markPriceKlines` | 5m, 1h, 4h | 2020-01-01 to present |
| Funding Rate | `/fapi/v1/fundingRate` | 8h intervals | 2020-01-01 to present |
| Open Interest | `/futures/data/openInterestHist` | 5m, 1h, 4h | 2020-01-01 to present |

### 2.3 Collection Scripts

**Existing Script**: `ingest_funding_mark.py`

- Already implements funding rate and mark price collection
- Handles pagination and rate limiting
- Output: Parquet files

**New Script Required**: `scripts/download_futures_data.py`

```python
#!/usr/bin/env python
"""
Download Binance Futures historical data.

Usage:
    python scripts/download_futures_data.py --symbols BTCUSDT ETHUSDT \\
        --start 2020-01-01 --timeframe 1h --data-types ohlcv funding mark oi

Output:
    data/raw_futures/crypto/BTCUSDT_1h_ohlcv.parquet
    data/raw_futures/crypto/BTCUSDT_funding.parquet
    data/raw_futures/crypto/BTCUSDT_1h_mark.parquet
    data/raw_futures/crypto/BTCUSDT_1h_oi.parquet
"""
```

### 2.4 Data Schema

**OHLCV DataFrame**:

```python
columns = [
    "timestamp",      # datetime64[ns, UTC]
    "open",           # float64
    "high",           # float64
    "low",            # float64
    "close",          # float64
    "volume",         # float64 (base asset volume)
    "quote_volume",   # float64 (quote asset volume)
    "trades",         # int64 (number of trades)
    "taker_buy_base", # float64
    "taker_buy_quote",# float64
]
```

**Funding Rate DataFrame**:

```python
columns = [
    "timestamp",      # datetime64[ns, UTC] (funding time)
    "funding_rate",   # float64 (e.g., 0.0001 = 0.01%)
    "mark_price",     # float64 (mark price at funding)
]
```

**Mark Price DataFrame**:

```python
columns = [
    "timestamp",      # datetime64[ns, UTC]
    "open",           # float64
    "high",           # float64
    "low",            # float64
    "close",          # float64
]
```

**Open Interest DataFrame**:

```python
columns = [
    "timestamp",      # datetime64[ns, UTC]
    "sum_open_interest",       # float64 (contracts)
    "sum_open_interest_value", # float64 (USDT value)
]
```

### 2.5 Storage Structure

```
data/
└── raw_futures/
    └── crypto/
        ├── BTCUSDT/
        │   ├── ohlcv_1h.parquet
        │   ├── ohlcv_4h.parquet
        │   ├── funding.parquet
        │   ├── mark_1h.parquet
        │   └── oi_1h.parquet
        ├── ETHUSDT/
        │   └── ...
        └── manifest.json  # Data inventory
```

### 2.6 Rate Limits & Pagination

| Endpoint | Weight | Limit | Max per Request |
|----------|--------|-------|-----------------|
| `/fapi/v1/klines` | 5 | 2400/min | 1500 bars |
| `/fapi/v1/fundingRate` | 1 | 2400/min | 1000 records |
| `/fapi/v1/markPriceKlines` | 1 | 2400/min | 1500 bars |
| `/futures/data/openInterestHist` | 1 | 2400/min | 500 records |

**Recommended Collection Rate**: 10 requests/second with retry logic

---

## 3. CME Futures Data (IB/Polygon)

### 3.1 Symbols

**Equity Index Futures**:

| Symbol | Name | Exchange | Multiplier |
|--------|------|----------|------------|
| ES | E-mini S&P 500 | CME | $50 |
| NQ | E-mini Nasdaq 100 | CME | $20 |
| YM | E-mini Dow | CBOT | $5 |
| RTY | E-mini Russell 2000 | CME | $50 |

**Commodity Futures**:

| Symbol | Name | Exchange | Multiplier |
|--------|------|----------|------------|
| GC | Gold | COMEX | 100 oz |
| CL | Crude Oil WTI | NYMEX | 1000 bbl |
| SI | Silver | COMEX | 5000 oz |
| NG | Natural Gas | NYMEX | 10000 MMBtu |

**Currency Futures**:

| Symbol | Name | Exchange | Multiplier |
|--------|------|----------|------------|
| 6E | Euro FX | CME | 125000 EUR |
| 6J | Japanese Yen | CME | 12500000 JPY |
| 6B | British Pound | CME | 62500 GBP |

### 3.2 Data Sources

**Primary Source**: Interactive Brokers TWS API

- Requires IB account
- Access via `ib_insync` library
- 1 year historical data (more with subscription)

**Secondary Source**: Polygon.io

- Requires API key (paid subscription for futures)
- Extensive historical coverage
- REST API access

**Tertiary Source**: Public Archives

- CME DataMine (expensive)
- Quandl/Nasdaq Data Link
- Yahoo Finance (limited futures support)

### 3.3 Data Types

| Data Type | Source | Timeframes | Notes |
|-----------|--------|------------|-------|
| OHLCV | IB/Polygon | 1m, 5m, 15m, 1h, 1d | Primary price data |
| Volume | IB/Polygon | 5m, 1h | Trading volume |
| Open Interest | CME reports | Daily | End-of-day only |
| Settlement | CME reports | Daily | Official settlement price |
| COT | CFTC | Weekly | Commitment of Traders |

### 3.4 Continuous Contract Construction

**Challenge**: CME futures expire quarterly; need continuous series for backtesting.

**Methods**:

1. **Ratio Adjustment** (Recommended for returns-based analysis)

```python
# On rollover day
adjustment_ratio = new_contract_price / old_contract_price
adjusted_prices = old_prices * adjustment_ratio
```

2. **Difference Adjustment** (For price-level analysis)

```python
# On rollover day
adjustment_diff = new_contract_price - old_contract_price
adjusted_prices = old_prices + adjustment_diff
```

3. **Unadjusted** (Raw contract prices with gaps)

```python
# Simply concatenate contracts
# Warning: gaps at rollover dates
```

**Rollover Detection**:

```python
def detect_rollover_date(front_volume, back_volume):
    """
    Detect rollover when back month volume exceeds front month.
    Typically 5-8 trading days before expiration.
    """
    return back_volume > front_volume
```

### 3.5 Collection Script

**New Script Required**: `scripts/download_cme_futures.py`

```python
#!/usr/bin/env python
"""
Download CME futures data via IB TWS API.

Prerequisites:
    - IB account with market data subscription
    - TWS or IB Gateway running
    - ib_insync installed

Usage:
    python scripts/download_cme_futures.py --symbols ES NQ GC CL \\
        --start 2020-01-01 --timeframe 1h \\
        --continuous --adjustment ratio

Output:
    data/raw_futures/cme/ES_continuous_1h.parquet
    data/raw_futures/cme/ES_202403_1h.parquet  # Individual contracts
"""
```

### 3.6 Storage Structure

```
data/
└── raw_futures/
    └── cme/
        ├── ES/
        │   ├── continuous_1h.parquet      # Adjusted continuous
        │   ├── contracts/
        │   │   ├── ESH24_1h.parquet       # Mar 2024
        │   │   ├── ESM24_1h.parquet       # Jun 2024
        │   │   └── ...
        │   └── rollover_dates.json
        ├── NQ/
        │   └── ...
        ├── GC/
        │   └── ...
        └── manifest.json
```

---

## 4. Data Validation

### 4.1 Quality Checks

```python
def validate_futures_data(df: pd.DataFrame) -> ValidationResult:
    checks = [
        # Completeness
        ("no_missing_timestamps", check_timestamp_continuity),
        ("no_null_ohlc", check_ohlc_completeness),

        # Consistency
        ("high_gte_low", lambda df: (df["high"] >= df["low"]).all()),
        ("high_gte_open", lambda df: (df["high"] >= df["open"]).all()),
        ("high_gte_close", lambda df: (df["high"] >= df["close"]).all()),
        ("low_lte_open", lambda df: (df["low"] <= df["open"]).all()),
        ("low_lte_close", lambda df: (df["low"] <= df["close"]).all()),

        # Volume
        ("positive_volume", lambda df: (df["volume"] >= 0).all()),

        # Funding rate specific
        ("funding_in_range", lambda df: (df["funding_rate"].abs() < 0.01).all()),
    ]
    return run_validation(df, checks)
```

### 4.2 Data Statistics Report

For each collected dataset, generate:

```python
stats = {
    "symbol": "BTCUSDT",
    "start_date": "2020-01-01",
    "end_date": "2024-11-30",
    "total_bars": 43200,
    "missing_bars": 12,
    "missing_pct": 0.03,
    "avg_daily_volume": 50_000_000_000,
    "avg_funding_rate": 0.0001,
    "max_funding_rate": 0.0075,
    "min_funding_rate": -0.0050,
}
```

---

## 5. Test Data Subsets

### 5.1 Unit Test Data

Small datasets for fast unit tests:

```
data/test_fixtures/futures/
├── btcusdt_1h_100bars.parquet    # 100 bars for quick tests
├── ethusdt_funding_30days.parquet # 90 funding samples
├── es_continuous_1week.parquet    # 1 week CME data
└── rollover_sample.parquet        # Rollover scenario
```

### 5.2 Integration Test Data

Medium datasets for integration tests:

```
data/test_fixtures/futures/
├── crypto_perp_1month.parquet     # Full month, multiple symbols
├── cme_index_1month.parquet       # ES + NQ, 1 month
└── extreme_events.parquet         # Flash crashes, high funding
```

### 5.3 Backtest Validation Data

Full datasets for backtest validation:

```
data/raw_futures/
├── crypto/   # Full history 2020-present
└── cme/      # Full history 2020-present
```

---

## 6. Collection Schedule

### 6.1 Initial Collection (Phase 0)

| Task | Duration | Size |
|------|----------|------|
| BTCUSDT OHLCV (1h, 4h) | 2 hours | ~500MB |
| BTCUSDT Funding | 30 min | ~50MB |
| ETHUSDT all data | 2 hours | ~400MB |
| Test fixtures creation | 1 hour | ~10MB |

### 6.2 Ongoing Collection

| Task | Frequency | Notes |
|------|-----------|-------|
| Crypto OHLCV update | Daily | Incremental append |
| Funding rate update | 8 hours | Match funding times |
| Open interest update | Daily | End of day |
| CME data update | Daily | After settlement |

---

## 7. Implementation Checklist

### 7.1 Scripts to Create

- [ ] `scripts/download_futures_data.py` - Crypto futures collection
- [ ] `scripts/download_cme_futures.py` - CME futures via IB
- [ ] `scripts/build_continuous_contracts.py` - Rollover adjustment
- [ ] `scripts/validate_futures_data.py` - Data quality checks
- [ ] `scripts/create_test_fixtures.py` - Test data extraction

### 7.2 Data to Collect

**Phase 0 (Immediate)**:

- [ ] BTCUSDT OHLCV 1h (2020-present)
- [ ] BTCUSDT funding rate (2020-present)
- [ ] ETHUSDT OHLCV 1h (2020-present)
- [ ] ETHUSDT funding rate (2020-present)
- [ ] Unit test fixtures

**Phase 1 (Before Integration)**:

- [ ] All Tier 1/2 crypto perpetuals
- [ ] Open interest data
- [ ] Mark price klines

**Phase 2 (CME Integration)**:

- [ ] ES continuous contract
- [ ] NQ continuous contract
- [ ] GC continuous contract
- [ ] CL continuous contract

---

## 8. Dependencies

### 8.1 Python Packages

```bash
pip install ib_insync          # IB TWS API wrapper
pip install polygon-api-client # Polygon.io (optional)
pip install pyarrow            # Parquet support (already installed)
pip install tqdm               # Progress bars
```

### 8.2 External Services

| Service | Purpose | Required |
|---------|---------|----------|
| Binance API | Crypto futures data | Yes (free) |
| IB TWS/Gateway | CME futures data | For CME only |
| Polygon.io | Alternative CME source | Optional |

### 8.3 Infrastructure

| Component | Purpose |
|-----------|---------|
| ~5GB disk space | Raw data storage |
| Reliable internet | Data download |
| TWS/Gateway running | IB access (if using) |

---

## 9. Validation Queries

After data collection, run these validation queries:

```python
# 1. Check data coverage
def check_coverage(df, expected_start, expected_end):
    actual_start = df["timestamp"].min()
    actual_end = df["timestamp"].max()
    return actual_start <= expected_start and actual_end >= expected_end

# 2. Check for gaps
def find_gaps(df, expected_interval):
    diffs = df["timestamp"].diff()
    gaps = diffs[diffs > expected_interval * 1.5]
    return gaps

# 3. Check funding alignment
def check_funding_alignment(df):
    hours = df["timestamp"].dt.hour
    return hours.isin([0, 8, 16]).all()

# 4. Check OHLC validity
def check_ohlc_validity(df):
    valid = (
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    )
    return valid.all()
```

---

## 10. Next Steps

1. **Immediate**: Create `scripts/download_futures_data.py` for Binance
2. **Week 1**: Collect BTCUSDT and ETHUSDT data
3. **Week 2**: Create test fixtures and validation scripts
4. **Phase 1**: Expand to all Tier 1/2 symbols
5. **Phase 2**: Set up IB connection and collect CME data

---

**Document End**

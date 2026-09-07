# -*- coding: utf-8 -*-
"""
scripts/download_interest_rates.py
Download central bank interest rates from FRED (Federal Reserve Economic Data).

Interest rates are critical for forex trading:
1. Interest rate differentials drive carry trade profitability
2. Rate expectations affect currency valuations
3. Required for swap rate estimation in backtesting

Data Source: FRED (Federal Reserve Bank of St. Louis)
API: https://fred.stlouisfed.org/docs/api/fred/

Features:
- All major central bank policy rates
- Daily/monthly frequency options
- Historical data back to 1990+
- Automatic gap filling (forward fill)
- Rate differential calculation

Supported Currencies:
- USD: Federal Funds Rate (FEDFUNDS)
- EUR: ECB Deposit Facility Rate (ECBDFR)
- GBP: BOE Official Bank Rate (IUDSOIA)
- JPY: BOJ Policy Rate (IRSTCI01JPM156N)
- CHF: SNB Policy Rate (IRSTCI01CHM156N)
- AUD: RBA Cash Rate (RBATCTR)
- CAD: BOC Policy Rate (IRSTCB01CAM156N)
- NZD: RBNZ Official Cash Rate (RBNZCTR)

Usage:
    # Download all major rates
    python scripts/download_interest_rates.py --all --start 2010-01-01

    # Download specific currencies
    python scripts/download_interest_rates.py --currencies USD EUR GBP --start 2015-01-01

    # Download with API key (higher rate limits)
    python scripts/download_interest_rates.py --all --api-key YOUR_FRED_KEY

Output:
    data/forex/rates/USD_rates.parquet
    data/forex/rates/EUR_rates.parquet
    data/forex/rates/interest_rate_differentials.parquet

Columns:
    - date: Date
    - rate: Interest rate (annual %)
    - currency: Currency code
    - series_id: FRED series ID
    - source: "fred"

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =========================
# Configuration
# =========================


@dataclass
class RateDownloadConfig:
    """Configuration for interest rate download."""

    # FRED API key (optional but recommended)
    api_key: Optional[str] = None

    # Currencies to download
    currencies: List[str] = field(default_factory=lambda: ["USD", "EUR", "GBP", "JPY"])

    # Date range
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lookback_years: int = 15  # Default: 15 years

    # Output
    output_dir: str = "data/forex/rates"
    output_format: str = "parquet"

    # Processing
    fill_gaps: bool = True  # Forward fill missing values
    frequency: str = "daily"  # "daily" or "monthly"

    # Skip existing
    skip_existing: bool = True


# =========================
# FRED Series Mapping
# =========================

# Central bank policy rates from FRED
# Source: https://fred.stlouisfed.org/
RATE_SERIES: Dict[str, Dict[str, Any]] = {
    "USD": {
        "series_id": "FEDFUNDS",
        "name": "Federal Funds Effective Rate",
        "frequency": "monthly",
        "description": "US Federal Reserve target rate",
    },
    "EUR": {
        "series_id": "ECBDFR",
        "name": "ECB Deposit Facility Rate",
        "frequency": "monthly",
        "description": "European Central Bank deposit rate",
    },
    "GBP": {
        "series_id": "IUDSOIA",
        "name": "BOE Official Bank Rate",
        "frequency": "daily",
        "description": "Bank of England base rate",
    },
    "JPY": {
        "series_id": "IRSTCI01JPM156N",
        "name": "BOJ Policy Rate",
        "frequency": "monthly",
        "description": "Bank of Japan short-term policy rate",
    },
    "CHF": {
        "series_id": "IRSTCI01CHM156N",
        "name": "SNB Policy Rate",
        "frequency": "monthly",
        "description": "Swiss National Bank policy rate",
    },
    "AUD": {
        "series_id": "RBATCTR",
        "name": "RBA Cash Rate",
        "frequency": "monthly",
        "description": "Reserve Bank of Australia cash rate target",
    },
    "CAD": {
        "series_id": "IRSTCB01CAM156N",
        "name": "BOC Policy Rate",
        "frequency": "monthly",
        "description": "Bank of Canada policy interest rate",
    },
    "NZD": {
        "series_id": "RBNZCTR",
        "name": "RBNZ Official Cash Rate",
        "frequency": "daily",
        "description": "Reserve Bank of New Zealand cash rate",
    },
}

# Alternative series (backup if primary fails)
RATE_SERIES_ALT: Dict[str, str] = {
    "USD": "DFF",  # Daily Federal Funds Rate
    "EUR": "ECBMRRFR",  # ECB Main Refinancing Rate
    "GBP": "BOERUKM",  # UK Bank Rate (monthly)
}


# =========================
# FRED API Functions
# =========================


def fetch_fred_series(
    series_id: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch a series from FRED API.

    Args:
        series_id: FRED series identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: Optional FRED API key

    Returns:
        DataFrame with date and value columns, or None if failed
    """
    try:
        import requests

        api_key = api_key or os.environ.get("FRED_API_KEY")

        # FRED API endpoint
        base_url = "https://api.stlouisfed.org/fred/series/observations"

        params = {
            "series_id": series_id,
            "observation_start": start_date,
            "observation_end": end_date,
            "file_type": "json",
        }

        if api_key:
            params["api_key"] = api_key
        else:
            # Without API key, FRED has strict rate limits
            logger.warning("No FRED API key provided - rate limits apply")

        response = requests.get(base_url, params=params, timeout=60)

        if response.status_code == 429:
            logger.warning("FRED rate limit hit, waiting 60s...")
            time.sleep(60)
            response = requests.get(base_url, params=params, timeout=60)

        if response.status_code != 200:
            logger.error(f"FRED API error {response.status_code}: {response.text[:200]}")
            return None

        data = response.json()
        observations = data.get("observations", [])

        if not observations:
            logger.warning(f"No data returned for {series_id}")
            return None

        # Parse observations
        records = []
        for obs in observations:
            value = obs.get("value", ".")
            if value == "." or value is None:
                continue  # Missing value

            try:
                records.append(
                    {
                        "date": obs["date"],
                        "rate": float(value),
                    }
                )
            except (ValueError, KeyError):
                continue

        if not records:
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        return df

    except ImportError:
        logger.error("requests library required: pip install requests")
        return None
    except Exception as e:
        logger.error(f"Error fetching {series_id}: {e}")
        return None


def fetch_fred_series_pandas(
    series_id: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch FRED series using pandas_datareader (alternative method).

    Args:
        series_id: FRED series identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with date and rate columns, or None if failed
    """
    try:
        import pandas_datareader as pdr

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        df = pdr.get_data_fred(series_id, start=start, end=end)

        if df.empty:
            return None

        df = df.reset_index()
        df.columns = ["date", "rate"]

        return df

    except ImportError:
        logger.debug("pandas_datareader not available, using API directly")
        return None
    except Exception as e:
        logger.debug(f"pandas_datareader failed: {e}")
        return None


# =========================
# Download Functions
# =========================


def download_rate_for_currency(
    currency: str,
    config: RateDownloadConfig,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    Download interest rate data for a single currency.

    Args:
        currency: Currency code (e.g., "USD")
        config: Download configuration

    Returns:
        (currency, dataframe, error_message)
    """
    currency = currency.upper()

    if currency not in RATE_SERIES:
        return currency, None, f"Unknown currency: {currency}"

    series_info = RATE_SERIES[currency]
    series_id = series_info["series_id"]

    # Date range
    end_dt = datetime.now()
    if config.end_date:
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")

    if config.start_date:
        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")
    else:
        start_dt = end_dt - timedelta(days=365 * config.lookback_years)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    logger.info(f"Downloading {currency} rates ({series_id}): {start_str} to {end_str}")

    # Try pandas_datareader first (simpler, no API key needed)
    df = fetch_fred_series_pandas(series_id, start_str, end_str)

    if df is None:
        # Fall back to API
        df = fetch_fred_series(series_id, start_str, end_str, config.api_key)

    if df is None:
        # Try alternative series
        alt_series = RATE_SERIES_ALT.get(currency)
        if alt_series:
            logger.info(f"Trying alternative series {alt_series}")
            df = fetch_fred_series(alt_series, start_str, end_str, config.api_key)

    if df is None:
        return currency, None, f"No data available for {currency}"

    # Add metadata
    df["currency"] = currency
    df["series_id"] = series_id
    df["source"] = "fred"

    # Fill gaps if requested
    if config.fill_gaps:
        df = _fill_rate_gaps(df, start_dt, end_dt)

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"Downloaded {len(df)} records for {currency}")
    return currency, df, None


def _fill_rate_gaps(
    df: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Fill gaps in rate data using forward fill.

    Central bank rates are announced discretely but apply continuously,
    so forward fill is appropriate.
    """
    if df.empty:
        return df

    # Create complete date range
    date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")

    # Reindex to full date range
    df_indexed = df.set_index("date")
    df_reindexed = df_indexed.reindex(date_range)

    # Forward fill rates
    df_reindexed["rate"] = df_reindexed["rate"].ffill()

    # Fill metadata columns
    for col in ["currency", "series_id", "source"]:
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].ffill().bfill()

    # Reset index
    df_filled = df_reindexed.reset_index()
    df_filled = df_filled.rename(columns={"index": "date"})

    # Drop rows with no rate data
    df_filled = df_filled.dropna(subset=["rate"])

    return df_filled


# =========================
# Differential Calculation
# =========================


def calculate_rate_differentials(
    rate_dataframes: Dict[str, pd.DataFrame],
    base_currency: str = "USD",
) -> pd.DataFrame:
    """
    Calculate interest rate differentials between currency pairs.

    Rate differential = Base rate - Quote rate

    Positive differential means carry trade profit for long base currency.

    Args:
        rate_dataframes: Dict of currency -> rate DataFrame
        base_currency: Reference currency (default: USD)

    Returns:
        DataFrame with rate differentials for all pairs
    """
    if base_currency not in rate_dataframes:
        logger.warning(f"Base currency {base_currency} not in rate data")
        return pd.DataFrame()

    base_df = (
        rate_dataframes[base_currency]
        .set_index("date")[["rate"]]
        .rename(columns={"rate": f"{base_currency}_rate"})
    )

    differentials = []

    for currency, df in rate_dataframes.items():
        if currency == base_currency:
            continue

        # Merge with base rate
        curr_df = df.set_index("date")[["rate"]].rename(columns={"rate": f"{currency}_rate"})

        merged = base_df.join(curr_df, how="outer").ffill()

        # Calculate differential
        pair = f"{base_currency}_{currency}"
        merged[f"{pair}_diff"] = merged[f"{base_currency}_rate"] - merged[f"{currency}_rate"]

        differentials.append(merged[[f"{pair}_diff"]])

    if not differentials:
        return pd.DataFrame()

    result = pd.concat(differentials, axis=1)
    result = result.reset_index()

    return result


# =========================
# File I/O
# =========================


def save_rates(
    df: pd.DataFrame,
    currency: str,
    config: RateDownloadConfig,
) -> str:
    """Save rate DataFrame to file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output_format == "parquet":
        filepath = output_dir / f"{currency}_rates.parquet"
        df.to_parquet(filepath, index=False)
    else:
        filepath = output_dir / f"{currency}_rates.csv"
        df.to_csv(filepath, index=False)

    return str(filepath)


def save_differentials(
    df: pd.DataFrame,
    config: RateDownloadConfig,
) -> str:
    """Save rate differentials to file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output_format == "parquet":
        filepath = output_dir / "interest_rate_differentials.parquet"
        df.to_parquet(filepath, index=False)
    else:
        filepath = output_dir / "interest_rate_differentials.csv"
        df.to_csv(filepath, index=False)

    return str(filepath)


# =========================
# Main Runner
# =========================


def download_all_rates(config: RateDownloadConfig) -> Dict[str, Any]:
    """
    Download interest rates for all configured currencies.

    Args:
        config: Download configuration

    Returns:
        Summary dict
    """
    currencies = config.currencies if config.currencies else list(RATE_SERIES.keys())

    logger.info(f"Downloading rates for {len(currencies)} currencies")

    # Check existing files
    output_dir = Path(config.output_dir)
    if config.skip_existing:
        existing = set()
        for f in output_dir.glob("*_rates.*"):
            existing.add(f.stem.replace("_rates", ""))
        currencies = [c for c in currencies if c not in existing]

    if not currencies:
        logger.info("All rate files exist, skipping download")
        return {"success": [], "failed": [], "skipped": True}

    success = []
    failed = []
    rate_dfs = {}

    for currency in currencies:
        currency, df, error = download_rate_for_currency(currency, config)

        if error:
            logger.error(f"Failed for {currency}: {error}")
            failed.append((currency, error))
        elif df is not None and not df.empty:
            filepath = save_rates(df, currency, config)
            logger.info(f"Saved {currency} rates to {filepath}")
            success.append(currency)
            rate_dfs[currency] = df

        # Rate limiting for FRED API
        time.sleep(1.0)

    # Calculate differentials if we have multiple currencies
    if len(rate_dfs) >= 2:
        logger.info("Calculating interest rate differentials")
        diff_df = calculate_rate_differentials(rate_dfs)
        if not diff_df.empty:
            diff_path = save_differentials(diff_df, config)
            logger.info(f"Saved differentials to {diff_path}")

    summary = {
        "success": success,
        "failed": failed,
        "total_downloaded": len(success),
        "total_failed": len(failed),
    }

    logger.info(f"Rate download complete: {len(success)} success, {len(failed)} failed")
    return summary


# =========================
# CLI
# =========================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download central bank interest rates from FRED",
    )

    parser.add_argument(
        "--currencies",
        nargs="+",
        help="Currency codes (e.g., USD EUR GBP)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all supported currencies",
    )
    parser.add_argument(
        "--start",
        dest="start_date",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=15,
        help="Lookback period in years if start not specified (default: 15)",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default="data/forex/rates",
        help="Output directory",
    )
    parser.add_argument(
        "--api-key",
        help="FRED API key (or set FRED_API_KEY env var)",
    )
    parser.add_argument(
        "--no-fill",
        action="store_true",
        help="Don't forward-fill missing values",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if files exist",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Determine currencies
    if args.all:
        currencies = list(RATE_SERIES.keys())
    elif args.currencies:
        currencies = [c.upper() for c in args.currencies]
    else:
        currencies = ["USD", "EUR", "GBP", "JPY"]

    config = RateDownloadConfig(
        api_key=args.api_key,
        currencies=currencies,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_years=args.lookback_years,
        output_dir=args.output_dir,
        fill_gaps=not args.no_fill,
        skip_existing=not args.force,
    )

    try:
        summary = download_all_rates(config)
        return 0 if not summary.get("failed") else 1
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

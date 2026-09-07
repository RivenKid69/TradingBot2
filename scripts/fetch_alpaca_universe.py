#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/fetch_alpaca_universe.py
Fetch and save Alpaca tradable US equity symbols.

This script downloads the list of tradable stocks from Alpaca
and saves them in the same format as the Binance universe.

Usage:
    python scripts/fetch_alpaca_universe.py

    # With custom output
    python scripts/fetch_alpaca_universe.py --output data/universe/alpaca_symbols.json

    # Filter by exchange
    python scripts/fetch_alpaca_universe.py --exchange NYSE

    # Include only fractionable stocks
    python scripts/fetch_alpaca_universe.py --fractionable

Environment Variables:
    ALPACA_API_KEY: Alpaca API key
    ALPACA_API_SECRET: Alpaca API secret

Requirements:
    pip install alpaca-py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_alpaca_assets(
    api_key: str,
    api_secret: str,
    paper: bool = True,
    exchange_filter: Optional[str] = None,
    fractionable_only: bool = False,
    shortable_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch tradable assets from Alpaca.

    Args:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        paper: Use paper trading endpoint
        exchange_filter: Filter by exchange (NYSE, NASDAQ, AMEX)
        fractionable_only: Only include fractionable stocks
        shortable_only: Only include shortable stocks

    Returns:
        List of asset dictionaries
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
    except ImportError:
        logger.error("alpaca-py not installed. Run: pip install alpaca-py")
        sys.exit(1)

    logger.info("Connecting to Alpaca API...")
    client = TradingClient(
        api_key=api_key,
        secret_key=api_secret,
        paper=paper,
    )

    # Fetch all active US equities
    request = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE,
    )

    logger.info("Fetching assets from Alpaca...")
    assets = client.get_all_assets(request)

    # Filter and convert to dict format
    result = []
    for asset in assets:
        # Skip non-tradable
        if not asset.tradable:
            continue

        # Apply filters
        if exchange_filter and str(asset.exchange) != exchange_filter:
            continue

        if fractionable_only and not asset.fractionable:
            continue

        if shortable_only and not asset.shortable:
            continue

        asset_dict = {
            "symbol": str(asset.symbol),
            "name": str(asset.name) if asset.name else "",
            "exchange": str(asset.exchange),
            "asset_class": "us_equity",
            "tradable": asset.tradable,
            "fractionable": asset.fractionable,
            "marginable": asset.marginable,
            "shortable": asset.shortable,
            "easy_to_borrow": getattr(asset, "easy_to_borrow", None),
            "maintenance_margin_requirement": getattr(
                asset, "maintenance_margin_requirement", None
            ),
        }
        result.append(asset_dict)

    logger.info(f"Found {len(result)} tradable assets")
    return result


def save_universe(
    assets: List[Dict[str, Any]],
    output_path: str,
    format_type: str = "json",
) -> None:
    """
    Save assets to file in various formats.

    Args:
        assets: List of asset dictionaries
        output_path: Output file path
        format_type: Output format ("json", "symbols", "csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format_type == "json":
        # Full JSON with metadata
        output_data = {
            "vendor": "alpaca",
            "asset_class": "us_equity",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "count": len(assets),
            "symbols": [a["symbol"] for a in assets],
            "assets": assets,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

    elif format_type == "symbols":
        # Just symbols list (compatible with existing universe format)
        symbols = sorted([a["symbol"] for a in assets])
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(symbols, f, indent=2)

    elif format_type == "csv":
        # CSV format
        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=assets[0].keys() if assets else [])
            writer.writeheader()
            writer.writerows(assets)

    logger.info(f"Saved {len(assets)} assets to {output_path}")


def create_popular_universe(
    assets: List[Dict[str, Any]],
    output_path: str,
    limit: int = 100,
) -> None:
    """
    Create a curated list of popular/liquid stocks.

    This list includes major indices components, popular tech stocks, etc.
    """
    # Popular stocks to include (S&P 500 top components + popular retail stocks)
    popular_symbols = {
        # Tech giants
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "AMD",
        "INTC",
        # Finance
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "C",
        "BRK.B",
        "V",
        "MA",
        "AXP",
        # Healthcare
        "JNJ",
        "UNH",
        "PFE",
        "ABBV",
        "MRK",
        "LLY",
        "TMO",
        "ABT",
        "BMY",
        "AMGN",
        # Consumer
        "WMT",
        "COST",
        "HD",
        "MCD",
        "NKE",
        "SBUX",
        "TGT",
        "LOW",
        "DIS",
        "NFLX",
        # Energy
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "OXY",
        "MPC",
        "VLO",
        "PSX",
        "KMI",
        # Industrial
        "CAT",
        "DE",
        "HON",
        "UNP",
        "UPS",
        "FDX",
        "BA",
        "LMT",
        "RTX",
        "GE",
        # Retail favorites
        "GME",
        "AMC",
        "PLTR",
        "SOFI",
        "RIVN",
        "LCID",
        "NIO",
        "BB",
        "NOK",
        "SNAP",
        # ETFs (major)
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "VXX",
        "ARKK",
        "XLF",
        "XLE",
    }

    popular_assets = [a for a in assets if a["symbol"] in popular_symbols]

    # Sort by symbol
    popular_assets.sort(key=lambda x: x["symbol"])

    output_path = Path(output_path)
    output_data = {
        "vendor": "alpaca",
        "asset_class": "us_equity",
        "type": "popular",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "count": len(popular_assets),
        "symbols": [a["symbol"] for a in popular_assets],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Created popular universe with {len(popular_assets)} symbols: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch Alpaca US equity universe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/universe/alpaca_symbols.json",
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["json", "symbols", "csv"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        choices=["NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"],
        help="Filter by exchange",
    )
    parser.add_argument(
        "--fractionable",
        action="store_true",
        help="Only include fractionable stocks",
    )
    parser.add_argument(
        "--shortable",
        action="store_true",
        help="Only include shortable stocks",
    )
    parser.add_argument(
        "--popular",
        action="store_true",
        help="Also create a popular stocks universe file",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ALPACA_API_KEY", ""),
        help="Alpaca API key (or set ALPACA_API_KEY env var)",
    )
    parser.add_argument(
        "--api-secret",
        type=str,
        default=os.environ.get("ALPACA_API_SECRET", ""),
        help="Alpaca API secret (or set ALPACA_API_SECRET env var)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Use paper trading endpoint (default: True)",
    )

    args = parser.parse_args()

    # Validate credentials
    if not args.api_key or not args.api_secret:
        logger.error(
            "Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_API_SECRET "
            "environment variables, or use --api-key and --api-secret arguments."
        )
        return 1

    try:
        # Fetch assets
        assets = fetch_alpaca_assets(
            api_key=args.api_key,
            api_secret=args.api_secret,
            paper=args.paper,
            exchange_filter=args.exchange,
            fractionable_only=args.fractionable,
            shortable_only=args.shortable,
        )

        if not assets:
            logger.warning("No assets found matching criteria")
            return 1

        # Save main universe
        save_universe(assets, args.output, args.format)

        # Create popular universe if requested
        if args.popular:
            popular_path = args.output.replace(".json", "_popular.json")
            create_popular_universe(assets, popular_path)

        # Summary
        exchanges = {}
        for a in assets:
            ex = a.get("exchange", "Unknown")
            exchanges[ex] = exchanges.get(ex, 0) + 1

        logger.info("Summary by exchange:")
        for ex, count in sorted(exchanges.items(), key=lambda x: -x[1]):
            logger.info(f"  {ex}: {count}")

        return 0

    except Exception as e:
        logger.error(f"Failed to fetch universe: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

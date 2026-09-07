#!/usr/bin/env python3
"""
Fetch leverage brackets from Binance Futures API.

This script retrieves leverage bracket data from Binance USDT-M Futures
and saves it to a JSON file for use by the margin calculator.

Usage:
    python scripts/fetch_leverage_brackets.py
    python scripts/fetch_leverage_brackets.py --symbols BTCUSDT,ETHUSDT
    python scripts/fetch_leverage_brackets.py --output data/futures/custom_brackets.json
    python scripts/fetch_leverage_brackets.py --testnet

Environment Variables:
    BINANCE_API_KEY - API key (required for authenticated endpoints)
    BINANCE_API_SECRET - API secret (required for authenticated endpoints)

Reference:
    https://binance-docs.github.io/apidocs/futures/en/#notional-and-leverage-brackets-user_data
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "futures" / "leverage_brackets.json"

BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
BINANCE_FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"

# Popular symbols to fetch by default
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "INJUSDT",
    "SUIUSDT",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LeverageBracketData:
    """Single leverage bracket from Binance API."""

    bracket: int
    initial_leverage: int
    notional_cap: float
    notional_floor: float
    maint_margin_ratio: float
    cum_maint: float = 0.0  # Cumulative maintenance amount

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bracket": self.bracket,
            "initial_leverage": self.initial_leverage,
            "notional_cap": self.notional_cap,
            "notional_floor": self.notional_floor,
            "maint_margin_ratio": self.maint_margin_ratio,
            "cum_maint": self.cum_maint,
        }


@dataclass
class SymbolBrackets:
    """All brackets for a single symbol."""

    symbol: str
    brackets: List[LeverageBracketData] = field(default_factory=list)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [b.to_dict() for b in self.brackets]


@dataclass
class FetchResult:
    """Result of fetching leverage brackets."""

    success: bool
    data: Dict[str, SymbolBrackets] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Binance API Client
# =============================================================================


class BinanceFuturesClient:
    """Simple client for Binance Futures API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("BINANCE_API_KEY")
        self.api_secret = api_secret or os.environ.get("BINANCE_API_SECRET")
        self.base_url = BINANCE_FUTURES_TESTNET_URL if testnet else BINANCE_FUTURES_BASE_URL
        self.timeout = timeout
        self.session = requests.Session()

        if self.api_key:
            self.session.headers["X-MBX-APIKEY"] = self.api_key

    def _sign_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sign request with HMAC-SHA256."""
        if not self.api_secret:
            raise ValueError("API secret required for signed requests")

        params = params.copy()
        params["timestamp"] = int(time.time() * 1000)

        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        params["signature"] = signature
        return params

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        """Make API request."""
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        if signed:
            params = self._sign_request(params)

        try:
            if method == "GET":
                response = self.session.get(url, params=params, timeout=self.timeout)
            else:
                response = self.session.post(url, params=params, timeout=self.timeout)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            try:
                error_data = e.response.json()
                error_msg = f"{error_data.get('code')}: {error_data.get('msg')}"
            except Exception:
                pass
            raise RuntimeError(f"Binance API error: {error_msg}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}") from e

    def get_leverage_brackets(
        self,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get leverage brackets for symbol(s).

        If API credentials are available, uses the authenticated endpoint
        which returns per-symbol brackets. Otherwise uses exchange info.

        Args:
            symbol: Optional specific symbol to fetch

        Returns:
            List of symbol bracket data
        """
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol

        # Try authenticated endpoint first (more accurate for user)
        if self.api_key and self.api_secret:
            try:
                data = self._request(
                    "GET",
                    "/fapi/v1/leverageBracket",
                    params=params,
                    signed=True,
                )
                return data if isinstance(data, list) else [data]
            except Exception as e:
                logger.warning(f"Authenticated request failed, trying public: {e}")

        # Fallback to exchange info (public)
        exchange_info = self._request("GET", "/fapi/v1/exchangeInfo")

        # Extract leverage brackets from exchange info
        brackets_data = []
        for sym_info in exchange_info.get("symbols", []):
            if symbol and sym_info.get("symbol") != symbol:
                continue

            if sym_info.get("symbol", "").endswith("USDT"):
                # Build bracket data from filters
                brackets_data.append(
                    {
                        "symbol": sym_info["symbol"],
                        "brackets": self._extract_brackets_from_filters(sym_info),
                    }
                )

        return brackets_data

    def _extract_brackets_from_filters(
        self,
        symbol_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract leverage brackets from symbol filters.

        Note: This is an approximation since the public API doesn't expose
        full bracket details. For accurate data, use authenticated endpoint.
        """
        # Get max leverage from leverage filter if available
        max_leverage = 125  # Default

        for filter_info in symbol_info.get("filters", []):
            if filter_info.get("filterType") == "MAX_NUM_ORDERS":
                # This doesn't give leverage, just order limits
                pass

        # Generate standard bracket structure based on typical patterns
        # This is approximation - real brackets may differ
        return self._generate_standard_brackets(max_leverage)

    def _generate_standard_brackets(
        self,
        max_leverage: int = 125,
    ) -> List[Dict[str, Any]]:
        """
        Generate standard bracket structure based on max leverage.

        This approximates Binance's typical bracket patterns.
        Real brackets should be fetched via authenticated endpoint.
        """
        if max_leverage >= 125:
            # BTC-like pattern
            return [
                {
                    "bracket": 1,
                    "initialLeverage": 125,
                    "notionalCap": 50000,
                    "notionalFloor": 0,
                    "maintMarginRatio": 0.004,
                    "cum": 0,
                },
                {
                    "bracket": 2,
                    "initialLeverage": 100,
                    "notionalCap": 250000,
                    "notionalFloor": 50000,
                    "maintMarginRatio": 0.005,
                    "cum": 50,
                },
                {
                    "bracket": 3,
                    "initialLeverage": 50,
                    "notionalCap": 1000000,
                    "notionalFloor": 250000,
                    "maintMarginRatio": 0.01,
                    "cum": 1300,
                },
                {
                    "bracket": 4,
                    "initialLeverage": 20,
                    "notionalCap": 5000000,
                    "notionalFloor": 1000000,
                    "maintMarginRatio": 0.025,
                    "cum": 16300,
                },
                {
                    "bracket": 5,
                    "initialLeverage": 10,
                    "notionalCap": 20000000,
                    "notionalFloor": 5000000,
                    "maintMarginRatio": 0.05,
                    "cum": 141300,
                },
                {
                    "bracket": 6,
                    "initialLeverage": 5,
                    "notionalCap": 50000000,
                    "notionalFloor": 20000000,
                    "maintMarginRatio": 0.1,
                    "cum": 1141300,
                },
                {
                    "bracket": 7,
                    "initialLeverage": 2,
                    "notionalCap": 100000000,
                    "notionalFloor": 50000000,
                    "maintMarginRatio": 0.25,
                    "cum": 8641300,
                },
                {
                    "bracket": 8,
                    "initialLeverage": 1,
                    "notionalCap": 200000000,
                    "notionalFloor": 100000000,
                    "maintMarginRatio": 0.5,
                    "cum": 33641300,
                },
            ]
        elif max_leverage >= 75:
            # Altcoin pattern
            return [
                {
                    "bracket": 1,
                    "initialLeverage": 75,
                    "notionalCap": 10000,
                    "notionalFloor": 0,
                    "maintMarginRatio": 0.0065,
                    "cum": 0,
                },
                {
                    "bracket": 2,
                    "initialLeverage": 50,
                    "notionalCap": 50000,
                    "notionalFloor": 10000,
                    "maintMarginRatio": 0.01,
                    "cum": 35,
                },
                {
                    "bracket": 3,
                    "initialLeverage": 25,
                    "notionalCap": 250000,
                    "notionalFloor": 50000,
                    "maintMarginRatio": 0.02,
                    "cum": 535,
                },
                {
                    "bracket": 4,
                    "initialLeverage": 10,
                    "notionalCap": 1000000,
                    "notionalFloor": 250000,
                    "maintMarginRatio": 0.05,
                    "cum": 8035,
                },
                {
                    "bracket": 5,
                    "initialLeverage": 5,
                    "notionalCap": 2000000,
                    "notionalFloor": 1000000,
                    "maintMarginRatio": 0.1,
                    "cum": 58035,
                },
                {
                    "bracket": 6,
                    "initialLeverage": 2,
                    "notionalCap": 5000000,
                    "notionalFloor": 2000000,
                    "maintMarginRatio": 0.25,
                    "cum": 358035,
                },
                {
                    "bracket": 7,
                    "initialLeverage": 1,
                    "notionalCap": 10000000,
                    "notionalFloor": 5000000,
                    "maintMarginRatio": 0.5,
                    "cum": 1608035,
                },
            ]
        else:
            # Low leverage pattern
            return [
                {
                    "bracket": 1,
                    "initialLeverage": 50,
                    "notionalCap": 10000,
                    "notionalFloor": 0,
                    "maintMarginRatio": 0.01,
                    "cum": 0,
                },
                {
                    "bracket": 2,
                    "initialLeverage": 25,
                    "notionalCap": 50000,
                    "notionalFloor": 10000,
                    "maintMarginRatio": 0.02,
                    "cum": 100,
                },
                {
                    "bracket": 3,
                    "initialLeverage": 10,
                    "notionalCap": 250000,
                    "notionalFloor": 50000,
                    "maintMarginRatio": 0.05,
                    "cum": 1600,
                },
                {
                    "bracket": 4,
                    "initialLeverage": 5,
                    "notionalCap": 1000000,
                    "notionalFloor": 250000,
                    "maintMarginRatio": 0.1,
                    "cum": 14100,
                },
                {
                    "bracket": 5,
                    "initialLeverage": 2,
                    "notionalCap": 5000000,
                    "notionalFloor": 1000000,
                    "maintMarginRatio": 0.25,
                    "cum": 164100,
                },
                {
                    "bracket": 6,
                    "initialLeverage": 1,
                    "notionalCap": 10000000,
                    "notionalFloor": 5000000,
                    "maintMarginRatio": 0.5,
                    "cum": 1414100,
                },
            ]

    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information."""
        return self._request("GET", "/fapi/v1/exchangeInfo")


# =============================================================================
# Bracket Fetcher
# =============================================================================


def parse_bracket_data(raw_data: Dict[str, Any]) -> Optional[LeverageBracketData]:
    """Parse raw bracket data from Binance API into our format."""
    try:
        return LeverageBracketData(
            bracket=raw_data.get("bracket", 0),
            initial_leverage=raw_data.get("initialLeverage", 0),
            notional_cap=float(raw_data.get("notionalCap", 0)),
            notional_floor=float(raw_data.get("notionalFloor", 0)),
            maint_margin_ratio=float(raw_data.get("maintMarginRatio", 0)),
            cum_maint=float(raw_data.get("cum", 0)),
        )
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse bracket data: {e}")
        return None


def fetch_leverage_brackets(
    symbols: Optional[List[str]] = None,
    testnet: bool = False,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> FetchResult:
    """
    Fetch leverage brackets from Binance Futures API.

    Args:
        symbols: List of symbols to fetch (default: popular symbols)
        testnet: Use testnet API
        api_key: API key override
        api_secret: API secret override

    Returns:
        FetchResult with bracket data
    """
    result = FetchResult(success=True)
    symbols = symbols or DEFAULT_SYMBOLS

    client = BinanceFuturesClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )

    logger.info(f"Fetching leverage brackets for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            raw_brackets = client.get_leverage_brackets(symbol)

            if not raw_brackets:
                result.errors.append(f"{symbol}: No bracket data returned")
                continue

            # Process the data
            symbol_data = raw_brackets[0] if raw_brackets else {}
            brackets_list = symbol_data.get("brackets", [])

            if not brackets_list:
                logger.warning(f"{symbol}: Empty brackets list")
                continue

            symbol_brackets = SymbolBrackets(symbol=symbol)

            for raw_bracket in brackets_list:
                bracket = parse_bracket_data(raw_bracket)
                if bracket:
                    symbol_brackets.brackets.append(bracket)

            # Sort by bracket number
            symbol_brackets.brackets.sort(key=lambda b: b.bracket)

            if symbol_brackets.brackets:
                result.data[symbol] = symbol_brackets
                logger.info(f"{symbol}: Fetched {len(symbol_brackets.brackets)} brackets")
            else:
                result.errors.append(f"{symbol}: No valid brackets parsed")

        except Exception as e:
            error_msg = f"{symbol}: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg)

    if not result.data:
        result.success = False

    return result


def save_brackets_to_json(
    result: FetchResult,
    output_path: Path,
    pretty: bool = True,
) -> None:
    """
    Save fetched brackets to JSON file.

    Args:
        result: FetchResult containing bracket data
        output_path: Path to output file
        pretty: Pretty print JSON
    """
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build output structure
    output_data = {
        "metadata": {
            "exchange": "binance",
            "market_type": "USDT_MARGINED",
            "last_updated": result.timestamp.isoformat(),
            "version": "1.0.0",
            "source": "Binance Futures API",
            "description": "Leverage brackets for USDT-margined perpetual futures",
            "symbols_count": len(result.data),
        },
        "brackets": {
            symbol: sym_brackets.to_dict() for symbol, sym_brackets in result.data.items()
        },
    }

    # Add errors if any
    if result.errors:
        output_data["metadata"]["fetch_errors"] = result.errors

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(output_data, f, ensure_ascii=False)

    logger.info(f"Saved brackets to {output_path}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch leverage brackets from Binance Futures API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch default popular symbols
    python scripts/fetch_leverage_brackets.py

    # Fetch specific symbols
    python scripts/fetch_leverage_brackets.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

    # Use testnet
    python scripts/fetch_leverage_brackets.py --testnet

    # Custom output path
    python scripts/fetch_leverage_brackets.py --output my_brackets.json

    # Verbose output
    python scripts/fetch_leverage_brackets.py -v
        """,
    )

    parser.add_argument(
        "-s",
        "--symbols",
        type=str,
        help="Comma-separated list of symbols to fetch",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT})",
    )

    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use Binance testnet API",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all available USDT-M futures symbols",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but don't save to file",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.all:
        # Fetch all symbols from exchange info
        logger.info("Fetching all available symbols...")
        client = BinanceFuturesClient(testnet=args.testnet)
        exchange_info = client.get_exchange_info()
        symbols = [
            s["symbol"]
            for s in exchange_info.get("symbols", [])
            if s.get("symbol", "").endswith("USDT")
            and s.get("status") == "TRADING"
            and s.get("contractType") == "PERPETUAL"
        ]
        logger.info(f"Found {len(symbols)} USDT-M perpetual symbols")

    # Fetch brackets
    result = fetch_leverage_brackets(
        symbols=symbols,
        testnet=args.testnet,
    )

    # Report results
    if result.success:
        logger.info(f"Successfully fetched brackets for {len(result.data)} symbols")
    else:
        logger.error("Failed to fetch any bracket data")

    if result.errors:
        logger.warning(f"Errors encountered: {len(result.errors)}")
        for error in result.errors[:10]:  # Show first 10 errors
            logger.warning(f"  - {error}")
        if len(result.errors) > 10:
            logger.warning(f"  ... and {len(result.errors) - 10} more errors")

    # Save to file
    if not args.dry_run and result.data:
        save_brackets_to_json(result, args.output)
        logger.info(f"Output written to: {args.output}")
    elif args.dry_run:
        logger.info("Dry run - no file written")
        # Print sample data
        for symbol, brackets in list(result.data.items())[:2]:
            print(f"\n{symbol}:")
            for b in brackets.brackets[:3]:
                print(
                    f"  Bracket {b.bracket}: leverage={b.initial_leverage}x, "
                    f"cap={b.notional_cap:,.0f}, mmr={b.maint_margin_ratio:.4f}"
                )
            if len(brackets.brackets) > 3:
                print(f"  ... and {len(brackets.brackets) - 3} more brackets")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

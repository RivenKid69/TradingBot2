# -*- coding: utf-8 -*-
"""
adapters/yahoo/corporate_actions.py
Yahoo Finance adapter for corporate actions (dividends, splits).

This adapter uses yfinance to fetch dividend and split history.
Data is suitable for:
- Dividend-adjusted backtesting
- Total return calculation
- Position adjustment for splits

Note on Adjustment:
    Yahoo Finance provides both unadjusted and adjusted prices.
    When using this adapter, historical prices should be adjusted
    for splits to ensure continuity. Dividend adjustments are optional
    (depends on whether you want price return or total return).

References:
    - yfinance: https://github.com/ranaroussi/yfinance
    - Yahoo Finance API limits: ~2000 requests/hour per IP
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence

from adapters.base import CorporateActionsAdapter
from adapters.models import (
    AdjustmentFactors,
    CorporateAction,
    CorporateActionType,
    Dividend,
    DividendType,
    ExchangeVendor,
    StockSplit,
)

logger = logging.getLogger(__name__)


class YahooCorporateActionsAdapter(CorporateActionsAdapter):
    """
    Yahoo Finance adapter for corporate actions.

    Fetches dividend and split history using yfinance library.
    Provides adjustment factors for historical price correction.

    Attributes:
        _cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        _ticker_cache: In-memory cache for yfinance Ticker objects
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.YAHOO,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize Yahoo corporate actions adapter.

        Args:
            vendor: Exchange vendor (should be YAHOO)
            config: Configuration options:
                - cache_ttl: Cache TTL in seconds (default: 3600)
                - max_retries: Max retries for API calls (default: 3)
        """
        super().__init__(vendor, config)

        self._cache_ttl = int(self._config.get("cache_ttl", 3600))
        self._max_retries = int(self._config.get("max_retries", 3))

        # In-memory cache for Ticker objects
        self._ticker_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}

    def _get_ticker(self, symbol: str) -> Any:
        """Get yfinance Ticker object with caching."""
        import time

        now = time.time()

        # Check cache validity
        if symbol in self._ticker_cache:
            cached_time = self._cache_timestamps.get(symbol, 0)
            if now - cached_time < self._cache_ttl:
                return self._ticker_cache[symbol]

        # Lazy import yfinance
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for Yahoo adapter. "
                "Install with: pip install yfinance>=0.2.0"
            ) from e

        ticker = yf.Ticker(symbol)
        self._ticker_cache[symbol] = ticker
        self._cache_timestamps[symbol] = now

        return ticker

    def get_dividends(
        self,
        symbol: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dividend]:
        """
        Get dividend history for a symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start_date: Start date (ISO format, inclusive)
            end_date: End date (ISO format, inclusive)

        Returns:
            List of Dividend objects, sorted by ex_date ascending
        """
        ticker = self._get_ticker(symbol)

        try:
            # Get dividend history from yfinance
            # Returns pandas Series with date index
            div_series = ticker.dividends

            if div_series is None or div_series.empty:
                logger.debug(f"No dividend data for {symbol}")
                return []

            dividends: List[Dividend] = []

            for idx, amount in div_series.items():
                ex_date = idx.strftime("%Y-%m-%d")

                # Filter by date range
                if start_date and ex_date < start_date:
                    continue
                if end_date and ex_date > end_date:
                    continue

                # Infer dividend type (regular vs special is not available from yfinance)
                div_type = DividendType.REGULAR

                dividends.append(
                    Dividend(
                        symbol=symbol.upper(),
                        ex_date=ex_date,
                        amount=Decimal(str(round(float(amount), 4))),
                        dividend_type=div_type,
                        currency="USD",  # yfinance assumes USD
                        is_adjusted=True,  # yfinance provides split-adjusted dividends
                    )
                )

            # Sort by ex_date ascending
            dividends.sort(key=lambda d: d.ex_date)

            logger.debug(f"Fetched {len(dividends)} dividends for {symbol}")
            return dividends

        except Exception as e:
            logger.error(f"Error fetching dividends for {symbol}: {e}")
            return []

    def get_splits(
        self,
        symbol: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[StockSplit]:
        """
        Get stock split history for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            List of StockSplit objects, sorted by ex_date ascending
        """
        ticker = self._get_ticker(symbol)

        try:
            # Get split history from yfinance
            # Returns pandas Series with date index, values like "4:1"
            split_series = ticker.splits

            if split_series is None or split_series.empty:
                logger.debug(f"No split data for {symbol}")
                return []

            splits: List[StockSplit] = []

            for idx, ratio_value in split_series.items():
                ex_date = idx.strftime("%Y-%m-%d")

                # Filter by date range
                if start_date and ex_date < start_date:
                    continue
                if end_date and ex_date > end_date:
                    continue

                # yfinance returns split as a float (e.g., 4.0 for 4:1 split)
                # Convert to ratio tuple (new, old)
                ratio_float = float(ratio_value)

                if ratio_float >= 1.0:
                    # Forward split: 4:1 means 4 new shares for 1 old
                    new_shares = int(ratio_float)
                    old_shares = 1
                else:
                    # Reverse split: 0.5 means 1 new share for 2 old (1:2)
                    # Convert to fraction
                    from fractions import Fraction

                    frac = Fraction(ratio_float).limit_denominator(100)
                    new_shares = frac.numerator
                    old_shares = frac.denominator

                splits.append(
                    StockSplit(
                        symbol=symbol.upper(),
                        ex_date=ex_date,
                        ratio=(new_shares, old_shares),
                        is_reverse=ratio_float < 1.0,
                    )
                )

            # Sort by ex_date ascending
            splits.sort(key=lambda s: s.ex_date)

            logger.debug(f"Fetched {len(splits)} splits for {symbol}")
            return splits

        except Exception as e:
            logger.error(f"Error fetching splits for {symbol}: {e}")
            return []

    def get_corporate_actions(
        self,
        symbol: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        action_types: Optional[Sequence[CorporateActionType]] = None,
    ) -> List[CorporateAction]:
        """
        Get all corporate actions for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            action_types: Filter by specific action types (None = all)

        Returns:
            List of CorporateAction objects, sorted by ex_date ascending
        """
        actions: List[CorporateAction] = []

        # Determine which action types to fetch
        fetch_dividends = action_types is None or CorporateActionType.DIVIDEND in action_types
        fetch_splits = action_types is None or CorporateActionType.SPLIT in action_types

        # Fetch dividends
        if fetch_dividends:
            dividends = self.get_dividends(symbol, start_date=start_date, end_date=end_date)
            for div in dividends:
                actions.append(div.to_corporate_action())

        # Fetch splits
        if fetch_splits:
            splits = self.get_splits(symbol, start_date=start_date, end_date=end_date)
            for split in splits:
                actions.append(split.to_corporate_action())

        # Sort all actions by ex_date
        actions.sort(key=lambda a: a.ex_date)

        return actions

    def get_adjustment_factors(
        self,
        symbol: str,
        date: str,
    ) -> AdjustmentFactors:
        """
        Get cumulative adjustment factors for a symbol at a date.

        Computes the factor needed to convert raw historical prices
        to split-adjusted prices as of the given date.

        Args:
            symbol: Stock symbol
            date: Reference date (ISO format)

        Returns:
            AdjustmentFactors for converting prices

        Note:
            For prices BEFORE splits, multiply by adjustment_factor
            to get the split-adjusted price.
        """
        # Fetch all splits from earliest available to target date
        splits = self.get_splits(symbol, end_date=date)

        # Compute cumulative split adjustment factor
        # For each split, multiply prices before the split by (old/new)
        cumulative_split_factor = 1.0

        for split in splits:
            cumulative_split_factor *= split.adjustment_factor

        # For now, we don't compute dividend adjustment factor
        # (would need closing prices to compute the factor)
        return AdjustmentFactors(
            symbol=symbol.upper(),
            date=date,
            split_factor=cumulative_split_factor,
            dividend_factor=1.0,
        )

    def compute_dividend_adjustment(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute dividend adjustment factors given price history.

        For dividend-adjusted prices:
        adjusted_price = raw_price * dividend_factor

        The factor is computed backwards from end_date:
        factor[t] = factor[t+1] * (1 - dividend[t+1] / close[t])

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            prices: Dict mapping date string to closing price

        Returns:
            Dict mapping date string to dividend adjustment factor
        """
        dividends = self.get_dividends(symbol, start_date=start_date, end_date=end_date)

        # Build dividend map
        div_map: Dict[str, Decimal] = {d.ex_date: d.amount for d in dividends}

        # Generate all dates between start and end
        from datetime import datetime, timedelta

        factors: Dict[str, float] = {}
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Work backwards from end_date
        current = end_dt
        factor = 1.0
        date_factors: List[tuple] = []

        while current >= start_dt:
            date_str = current.strftime("%Y-%m-%d")
            date_factors.append((date_str, factor))

            # Check for dividend on NEXT day (ex-date adjustment)
            next_day = (current + timedelta(days=1)).strftime("%Y-%m-%d")
            if next_day in div_map and date_str in prices:
                div_amount = float(div_map[next_day])
                close_price = prices[date_str]
                if close_price > 0:
                    factor = factor * (1.0 - div_amount / close_price)

            current -= timedelta(days=1)

        # Reverse to get chronological order
        for date_str, f in reversed(date_factors):
            factors[date_str] = f

        return factors

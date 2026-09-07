# -*- coding: utf-8 -*-
"""
services/sector_momentum.py
Sector Momentum Data Pipeline for Stock Features (Phase 6 L3 Enhancement).

This service provides sector momentum calculation and integration into
the stock features pipeline for observation space enhancement.

Features:
- Sector ETF data loading (XLK, XLF, XLE, etc.)
- Sector return calculation
- Symbol-to-sector mapping
- Integration with stock_features.py

Architecture:
    1. SectorDataLoader: Loads sector ETF historical data
    2. SectorMomentumCalculator: Computes sector returns
    3. SectorFeatureEnricher: Adds sector features to DataFrames

Usage:
    from services.sector_momentum import (
        SectorMomentumService,
        enrich_dataframe_with_sector_momentum,
    )

    # Create service
    service = SectorMomentumService()

    # Enrich DataFrame
    enriched_df = enrich_dataframe_with_sector_momentum(df, "AAPL")

References:
- Moskowitz & Grinblatt (1999): "Do Industries Explain Momentum?"
- GICS Sector Classification: https://www.spglobal.com/spdji/en/landing/investment-themes/gics/
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stock_features import (
    SECTOR_ETFS,
    SYMBOL_TO_SECTOR,
    calculate_sector_momentum,
    get_symbol_sector,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default lookback window for sector momentum calculation
DEFAULT_MOMENTUM_WINDOW = 20  # 20 trading days ≈ 1 month

# Minimum data points required for calculation
MIN_DATA_POINTS = 21  # At least window + 1

# Cache TTL for sector data (in seconds)
SECTOR_CACHE_TTL = 3600  # 1 hour


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SectorReturn:
    """Container for sector return data."""

    sector: str
    etf_symbol: str
    return_20d: float
    return_50d: Optional[float] = None
    volatility_20d: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class SectorMomentumResult:
    """Result of sector momentum calculation."""

    symbol: str
    sector: Optional[str]
    sector_return_20d: float
    market_return_20d: float
    sector_momentum: float
    is_valid: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SectorDataConfig:
    """Configuration for sector data loading."""

    # Data source
    data_vendor: str = "yahoo"  # yahoo, alpaca, polygon
    cache_enabled: bool = True
    cache_ttl_seconds: int = SECTOR_CACHE_TTL

    # Calculation windows
    momentum_window: int = DEFAULT_MOMENTUM_WINDOW
    volatility_window: int = 20

    # Paths
    sector_data_path: Optional[str] = None

    # Fallback values
    default_sector_momentum: float = 0.0


# =============================================================================
# Sector Data Loader
# =============================================================================


class SectorDataLoader:
    """
    Loads sector ETF data from various sources.

    Supports:
    - Yahoo Finance (default)
    - Alpaca
    - Local files

    Caches data to avoid repeated API calls.
    """

    def __init__(self, config: Optional[SectorDataConfig] = None) -> None:
        self._config = config or SectorDataConfig()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    def load_sector_etf_data(
        self,
        etf_symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load historical data for a sector ETF.

        Args:
            etf_symbol: ETF symbol (e.g., "XLK")
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{etf_symbol}_{start_date}_{end_date}"

        # Check cache
        if self._config.cache_enabled:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

        # Load from source
        df = self._load_from_source(etf_symbol, start_date, end_date)

        # Cache result
        if self._config.cache_enabled and df is not None and not df.empty:
            self._add_to_cache(cache_key, df)

        return df

    def load_all_sector_etfs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for all sector ETFs.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping ETF symbol to DataFrame
        """
        etf_data = {}
        for sector, etf_symbol in SECTOR_ETFS.items():
            try:
                df = self.load_sector_etf_data(etf_symbol, start_date, end_date)
                if df is not None and not df.empty:
                    etf_data[etf_symbol] = df
                    logger.debug(f"Loaded {len(df)} rows for {etf_symbol} ({sector})")
            except Exception as e:
                logger.warning(f"Failed to load data for {etf_symbol}: {e}")

        return etf_data

    def _load_from_source(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pd.DataFrame:
        """Load data from configured source."""
        if self._config.data_vendor == "yahoo":
            return self._load_from_yahoo(symbol, start_date, end_date)
        elif self._config.data_vendor == "alpaca":
            return self._load_from_alpaca(symbol, start_date, end_date)
        elif self._config.data_vendor == "local":
            return self._load_from_local(symbol)
        else:
            logger.error(f"Unknown data vendor: {self._config.data_vendor}")
            return pd.DataFrame()

    def _load_from_yahoo(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pd.DataFrame:
        """Load data from Yahoo Finance."""
        try:
            # Try to use the Yahoo adapter
            from adapters.yahoo.market_data import YahooMarketDataAdapter

            adapter = YahooMarketDataAdapter()
            bars = adapter.get_bars(
                symbol,
                timeframe="1d",
                start=start_date,
                end=end_date,
                limit=252,  # 1 year of trading days
            )

            if not bars:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append(
                    {
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                )

            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)

            return df

        except ImportError:
            logger.warning("Yahoo adapter not available, trying yfinance directly")
            return self._load_from_yfinance_direct(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Yahoo adapter error: {e}")
            return self._load_from_yfinance_direct(symbol, start_date, end_date)

    def _load_from_yfinance_direct(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pd.DataFrame:
        """Load directly from yfinance."""
        try:
            import yfinance as yf

            if start_date is None:
                start_date = datetime.now() - timedelta(days=365)
            if end_date is None:
                end_date = datetime.now()

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if not df.empty:
                df.columns = df.columns.str.lower()
                df = df[["open", "high", "low", "close", "volume"]]

            return df

        except ImportError:
            logger.warning("yfinance not installed, returning empty DataFrame")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"yfinance error: {e}")
            return pd.DataFrame()

    def _load_from_alpaca(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pd.DataFrame:
        """Load data from Alpaca."""
        try:
            from adapters.alpaca.market_data import AlpacaMarketDataAdapter

            adapter = AlpacaMarketDataAdapter()
            bars = adapter.get_bars(
                symbol,
                timeframe="1d",
                start=start_date,
                end=end_date,
                limit=252,
            )

            if not bars:
                return pd.DataFrame()

            data = []
            for bar in bars:
                data.append(
                    {
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                )

            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)

            return df

        except Exception as e:
            logger.error(f"Alpaca loader error: {e}")
            return pd.DataFrame()

    def _load_from_local(self, symbol: str) -> pd.DataFrame:
        """Load data from local files."""
        if not self._config.sector_data_path:
            return pd.DataFrame()

        try:
            file_path = f"{self._config.sector_data_path}/{symbol}.parquet"
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"Local file load error: {e}")
            return pd.DataFrame()

    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if not expired."""
        if key not in self._cache:
            return None

        cache_time = self._cache_timestamps.get(key)
        if cache_time is None:
            return None

        elapsed = (datetime.now() - cache_time).total_seconds()
        if elapsed > self._config.cache_ttl_seconds:
            # Cache expired
            del self._cache[key]
            del self._cache_timestamps[key]
            return None

        return self._cache[key].copy()

    def _add_to_cache(self, key: str, df: pd.DataFrame) -> None:
        """Add data to cache."""
        self._cache[key] = df.copy()
        self._cache_timestamps[key] = datetime.now()


# =============================================================================
# Sector Momentum Calculator
# =============================================================================


class SectorMomentumCalculator:
    """
    Calculates sector momentum from ETF data.

    Computes relative performance of sectors vs market benchmark.
    """

    def __init__(
        self,
        momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    ) -> None:
        self._momentum_window = momentum_window

    def calculate_sector_returns(
        self,
        etf_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, SectorReturn]:
        """
        Calculate returns for all sectors.

        Args:
            etf_data: Dict mapping ETF symbol to DataFrame

        Returns:
            Dict mapping sector name to SectorReturn
        """
        # Reverse mapping: ETF symbol -> sector name
        etf_to_sector = {v: k for k, v in SECTOR_ETFS.items()}
        sector_returns = {}

        for etf_symbol, df in etf_data.items():
            sector = etf_to_sector.get(etf_symbol)
            if sector is None:
                continue

            close_col = "close" if "close" in df.columns else "Close"
            if close_col not in df.columns or len(df) < self._momentum_window + 1:
                continue

            try:
                prices = df[close_col].values
                return_20d = self._calculate_return(prices, self._momentum_window)
                return_50d = self._calculate_return(prices, 50) if len(prices) > 50 else None
                vol_20d = self._calculate_volatility(prices, self._momentum_window)

                sector_returns[sector] = SectorReturn(
                    sector=sector,
                    etf_symbol=etf_symbol,
                    return_20d=return_20d,
                    return_50d=return_50d,
                    volatility_20d=vol_20d,
                    timestamp=datetime.now(),
                )
            except Exception as e:
                logger.warning(f"Failed to calculate returns for {sector}: {e}")

        return sector_returns

    def calculate_sector_momentum_for_symbol(
        self,
        symbol: str,
        sector_returns: Dict[str, SectorReturn],
        market_return: float,
    ) -> SectorMomentumResult:
        """
        Calculate sector momentum for a specific symbol.

        Args:
            symbol: Stock symbol
            sector_returns: Dict of sector returns
            market_return: Overall market return (e.g., SPY)

        Returns:
            SectorMomentumResult with momentum value
        """
        sector = get_symbol_sector(symbol)

        if sector is None:
            return SectorMomentumResult(
                symbol=symbol,
                sector=None,
                sector_return_20d=0.0,
                market_return_20d=market_return,
                sector_momentum=0.0,
                is_valid=False,
            )

        sector_data = sector_returns.get(sector)
        if sector_data is None:
            return SectorMomentumResult(
                symbol=symbol,
                sector=sector,
                sector_return_20d=0.0,
                market_return_20d=market_return,
                sector_momentum=0.0,
                is_valid=False,
            )

        # Calculate sector momentum (excess return vs market)
        excess_return = sector_data.return_20d - market_return

        # Normalize using tanh (same as in stock_features.py)
        # ±5% excess return → ±0.96
        normalized = math.tanh(excess_return * 10.0) if math.isfinite(excess_return) else 0.0

        return SectorMomentumResult(
            symbol=symbol,
            sector=sector,
            sector_return_20d=sector_data.return_20d,
            market_return_20d=market_return,
            sector_momentum=normalized,
            is_valid=True,
        )

    def _calculate_return(self, prices: np.ndarray, window: int) -> float:
        """Calculate return over window."""
        if len(prices) < window + 1:
            return 0.0

        price_now = float(prices[-1])
        price_past = float(prices[-(window + 1)])

        if price_past <= 0 or not math.isfinite(price_now) or not math.isfinite(price_past):
            return 0.0

        return (price_now / price_past) - 1.0

    def _calculate_volatility(self, prices: np.ndarray, window: int) -> float:
        """Calculate volatility (std of returns) over window."""
        if len(prices) < window + 1:
            return 0.0

        returns = np.diff(np.log(prices[-window - 1 :]))
        return float(np.std(returns)) * math.sqrt(252)  # Annualized


# =============================================================================
# Sector Momentum Service
# =============================================================================


class SectorMomentumService:
    """
    Main service for sector momentum calculation and integration.

    Provides a unified interface for:
    - Loading sector data
    - Calculating sector momentum
    - Enriching DataFrames with sector features
    """

    def __init__(self, config: Optional[SectorDataConfig] = None) -> None:
        self._config = config or SectorDataConfig()
        self._loader = SectorDataLoader(self._config)
        self._calculator = SectorMomentumCalculator(self._config.momentum_window)

        # Cached sector returns
        self._sector_returns: Dict[str, SectorReturn] = {}
        self._last_update: Optional[datetime] = None
        self._market_return: float = 0.0

    def update_sector_data(self) -> None:
        """
        Update sector ETF data and calculate returns.

        Call this periodically to refresh sector momentum calculations.
        """
        logger.info("Updating sector ETF data...")

        # Load all sector ETFs
        etf_data = self._loader.load_all_sector_etfs()

        if not etf_data:
            logger.warning("No sector ETF data loaded")
            return

        # Calculate sector returns
        self._sector_returns = self._calculator.calculate_sector_returns(etf_data)

        # Calculate market return (SPY as proxy)
        spy_df = self._loader.load_sector_etf_data("SPY")
        if spy_df is not None and not spy_df.empty:
            close_col = "close" if "close" in spy_df.columns else "Close"
            if close_col in spy_df.columns and len(spy_df) > self._config.momentum_window:
                prices = spy_df[close_col].values
                self._market_return = self._calculator._calculate_return(
                    prices, self._config.momentum_window
                )

        self._last_update = datetime.now()
        logger.info(
            f"Sector data updated: {len(self._sector_returns)} sectors, "
            f"market return={self._market_return:.2%}"
        )

    def get_sector_momentum(self, symbol: str) -> Tuple[float, bool]:
        """
        Get sector momentum for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            (momentum_value, is_valid) tuple
        """
        # Update data if stale
        if (
            self._last_update is None
            or (datetime.now() - self._last_update).total_seconds() > self._config.cache_ttl_seconds
        ):
            self.update_sector_data()

        # Get sector returns dict for calculate_sector_momentum
        sector_returns_dict = {sr.sector: sr.return_20d for sr in self._sector_returns.values()}

        # Use the function from stock_features.py
        momentum, is_valid = calculate_sector_momentum(
            symbol,
            sector_returns_dict,
            self._market_return,
        )

        return momentum, is_valid

    def get_all_sector_returns(self) -> Dict[str, float]:
        """
        Get current sector returns as a dict.

        Returns:
            Dict mapping sector name to return value
        """
        if self._last_update is None:
            self.update_sector_data()

        return {sr.sector: sr.return_20d for sr in self._sector_returns.values()}

    def get_market_return(self) -> float:
        """Get current market return."""
        if self._last_update is None:
            self.update_sector_data()
        return self._market_return


# =============================================================================
# DataFrame Enrichment
# =============================================================================


def enrich_dataframe_with_sector_momentum(
    df: pd.DataFrame,
    symbol: str,
    service: Optional[SectorMomentumService] = None,
) -> pd.DataFrame:
    """
    Add sector momentum column to a DataFrame.

    This function enriches a stock DataFrame with sector momentum values,
    making it ready for use in the observation space.

    Args:
        df: DataFrame with stock data
        symbol: Stock symbol
        service: SectorMomentumService instance (creates one if None)

    Returns:
        DataFrame with 'sector_momentum' column added
    """
    df = df.copy()

    if service is None:
        service = SectorMomentumService()

    # Get sector momentum
    momentum, is_valid = service.get_sector_momentum(symbol)

    # Add column
    if "sector_momentum" not in df.columns:
        df["sector_momentum"] = momentum if is_valid else 0.0
    else:
        # Update existing values
        df["sector_momentum"] = df["sector_momentum"].fillna(momentum if is_valid else 0.0)

    logger.debug(
        f"Enriched {symbol} DataFrame with sector_momentum={momentum:.4f} " f"(valid={is_valid})"
    )

    return df


def enrich_dataframe_with_all_stock_features(
    df: pd.DataFrame,
    symbol: str,
    spy_df: Optional[pd.DataFrame] = None,
    qqq_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    sector_service: Optional[SectorMomentumService] = None,
) -> pd.DataFrame:
    """
    Add all stock-specific features to a DataFrame.

    This is a convenience function that adds:
    - VIX features
    - Market regime
    - Relative strength (vs SPY, QQQ)
    - Sector momentum

    Args:
        df: DataFrame with stock data
        symbol: Stock symbol
        spy_df: SPY price DataFrame (optional)
        qqq_df: QQQ price DataFrame (optional)
        vix_df: VIX DataFrame (optional)
        sector_service: SectorMomentumService instance (optional)

    Returns:
        DataFrame with all stock features added
    """
    from stock_features import add_stock_features_to_dataframe

    # Add basic stock features
    df = add_stock_features_to_dataframe(
        df=df,
        symbol=symbol,
        spy_df=spy_df,
        qqq_df=qqq_df,
        vix_df=vix_df,
    )

    # Add sector momentum
    df = enrich_dataframe_with_sector_momentum(df, symbol, sector_service)

    return df


# =============================================================================
# Factory Functions
# =============================================================================


def create_sector_momentum_service(
    data_vendor: str = "yahoo",
    cache_enabled: bool = True,
) -> SectorMomentumService:
    """
    Create a SectorMomentumService with common defaults.

    Args:
        data_vendor: Data source ("yahoo", "alpaca", "local")
        cache_enabled: Enable data caching

    Returns:
        Configured SectorMomentumService
    """
    config = SectorDataConfig(
        data_vendor=data_vendor,
        cache_enabled=cache_enabled,
    )
    return SectorMomentumService(config)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json
    import os
    from datetime import datetime as _dt

    ap = argparse.ArgumentParser(description="Compute sector momentum and write a JSON report.")
    ap.add_argument(
        "--window",
        type=int,
        default=DEFAULT_MOMENTUM_WINDOW,
        help="Momentum lookback window (trading days).",
    )
    ap.add_argument("--vendor", default="yahoo", help="Data vendor: yahoo, alpaca, local.")
    ap.add_argument(
        "--out", default="models/momentum_report.json", help="Path to write the report JSON."
    )
    args = ap.parse_args()

    cfg = SectorDataConfig(data_vendor=args.vendor, momentum_window=int(args.window))
    service = SectorMomentumService(cfg)

    status = "ok"
    message = ""
    sectors_out: List[Dict[str, Any]] = []
    market_return = 0.0

    try:
        service.update_sector_data()
        sector_returns = service.get_all_sector_returns()
        market_return = float(service.get_market_return())

        if not sector_returns:
            status = "no_data"
            message = (
                "Не удалось загрузить данные секторных ETF "
                "(требуется доступ к рыночным данным / yfinance)."
            )
        else:
            for sector, ret in sorted(sector_returns.items(), key=lambda kv: kv[1], reverse=True):
                etf = SECTOR_ETFS.get(sector, "")
                excess = float(ret) - market_return
                sectors_out.append(
                    {
                        "sector": sector,
                        "etf": etf,
                        "return_window": float(ret),
                        "excess_vs_market": excess,
                        "rank": len(sectors_out) + 1,
                    }
                )
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        status = "error"
        message = f"Ошибка расчета momentum: {exc}"

    report = {
        "status": status,
        "message": message,
        "window": int(args.window),
        "vendor": args.vendor,
        "market_return": market_return,
        "generated_at": _dt.now().isoformat(timespec="seconds"),
        "sectors": sectors_out,
        "n_sectors": len(sectors_out),
    }

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[sector_momentum] status={status} sectors={len(sectors_out)} -> {out_path}")
    if message:
        print(f"[sector_momentum] {message}")

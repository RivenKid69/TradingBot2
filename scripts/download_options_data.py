# -*- coding: utf-8 -*-
"""
scripts/download_options_data.py
Download options historical/chain data from ThetaData or generate synthetic option chains/Greeks.
"""
from __future__ import annotations

import argparse
import sys
import os
import time
import math
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from adapters.theta_data.options import create_theta_data_adapter, THETA_DATA_AVAILABLE
except ImportError:
    THETA_DATA_AVAILABLE = False


def norm_cdf(x):
    """Cumulative normal distribution function using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    """Normal probability density function."""
    return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def is_dte_in_range(dte: int, dte_range: str) -> bool:
    if dte_range == "all":
        return True
    elif dte_range == "0-7":
        return 0 <= dte <= 7
    elif dte_range == "7-45":
        return 7 < dte <= 45
    elif dte_range == "45-90":
        return 45 < dte <= 90
    return True


def get_simulated_expirations(d: date) -> list[date]:
    exps = set()
    # 1. Weekly Fridays for the next 14 weeks (approx 100 days)
    for w in range(1, 15):
        days_ahead = (4 - d.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        exp = d + timedelta(days=days_ahead + (w - 1) * 7)
        exps.add(exp)

    # 2. Monthly third Fridays for the next 6 months
    for m_offset in range(1, 7):
        target_year = d.year
        target_month = d.month + m_offset
        while target_month > 12:
            target_month -= 12
            target_year += 1
        first_day = date(target_year, target_month, 1)
        first_friday_offset = (4 - first_day.weekday()) % 7
        third_friday = first_day + timedelta(days=first_friday_offset + 14)
        exps.add(third_friday)

    return sorted(list(exps))


def generate_synthetic_options(
    underlying: str,
    start_date: str,
    end_date: str,
    strike_range: str,
    include_greeks: bool,
    output_dir: str,
    dte_range: str = "all",
):
    print(f"Generating synthetic options data for {underlying}...")
    os.makedirs(output_dir, exist_ok=True)

    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    dates = pd.date_range(start=start_dt, end=end_dt, freq="B").date

    # Base spot price for underlying
    base_prices = {
        "AAPL": 180.0,
        "MSFT": 420.0,
        "NVDA": 950.0,
        "TSLA": 175.0,
        "GOOGL": 170.0,
        "AMZN": 180.0,
        "SPY": 510.0,
    }

    spot_base = base_prices.get(underlying.upper(), 150.0)

    # ATM strike offsets based on strike range parameter
    num_strikes = 10
    if "20" in strike_range:
        num_strikes = 20
    elif "50" in strike_range:
        num_strikes = 50
    strike_increment = 2.5 if spot_base < 300 else 5.0

    records = []

    for d in dates:
        # Simulate spot price drift
        spot = spot_base + np.random.normal(0, spot_base * 0.01)

        expirations = get_simulated_expirations(d)
        for exp in expirations:
            dte = (exp - d).days
            if dte <= 0:
                continue
            if not is_dte_in_range(dte, dte_range):
                continue
            t = max(1 / 365.0, dte / 365.0)

            # Strike prices around the spot
            atm_strike = round(spot / strike_increment) * strike_increment
            strikes = [
                atm_strike + i * strike_increment
                for i in range(-num_strikes // 2, num_strikes // 2 + 1)
            ]

            for K in strikes:
                if K <= 0:
                    continue
                for option_type in ["call", "put"]:
                    r = 0.05
                    sigma = 0.25

                    d1 = (math.log(spot / K) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
                    d2 = d1 - sigma * math.sqrt(t)

                    nd1 = norm_cdf(d1)
                    nd2 = norm_cdf(d2)
                    n_minus_d1 = norm_cdf(-d1)
                    n_minus_d2 = norm_cdf(-d2)

                    if option_type == "call":
                        price = spot * nd1 - K * math.exp(-r * t) * nd2
                        delta = nd1
                        theta = (
                            -(spot * norm_pdf(d1) * sigma) / (2 * math.sqrt(t))
                            - r * K * math.exp(-r * t) * nd2
                        )
                    else:
                        price = K * math.exp(-r * t) * n_minus_d2 - spot * n_minus_d1
                        delta = nd1 - 1.0
                        theta = (
                            -(spot * norm_pdf(d1) * sigma) / (2 * math.sqrt(t))
                            + r * K * math.exp(-r * t) * n_minus_d2
                        )

                    gamma = norm_pdf(d1) / (spot * sigma * math.sqrt(t))
                    vega = spot * math.sqrt(t) * norm_pdf(d1)

                    price = max(0.01, price)
                    spread = max(0.02, price * 0.02)
                    bid = max(0.01, price - spread / 2)
                    ask = price + spread / 2

                    vol = int(np.random.negative_binomial(10, 0.1))
                    oi = int(np.random.negative_binomial(50, 0.05))

                    date_str = exp.strftime("%y%m%d")
                    type_char = "C" if option_type == "call" else "P"
                    strike_str = f"{int(K * 1000):08d}"
                    occ_symbol = (
                        f"{underlying.upper().ljust(6)}{date_str}{type_char}{strike_str}".replace(
                            " ", ""
                        )
                    )

                    record = {
                        "date": d.strftime("%Y-%m-%d"),
                        "occ_symbol": occ_symbol,
                        "underlying": underlying.upper(),
                        "option_type": option_type,
                        "strike": float(K),
                        "expiration": exp.strftime("%Y-%m-%d"),
                        "dte": dte,
                        "spot_price": float(round(spot, 2)),
                        "bid": float(round(bid, 2)),
                        "ask": float(round(ask, 2)),
                        "mid": float(round(price, 2)),
                        "volume": vol,
                        "open_interest": oi,
                    }

                    if include_greeks:
                        record.update(
                            {
                                "implied_volatility": float(round(sigma, 4)),
                                "delta": float(round(delta, 4)),
                                "gamma": float(round(gamma, 4)),
                                "theta": float(round(theta / 365.0, 4)),
                                "vega": float(round(vega / 100.0, 4)),
                            }
                        )

                    records.append(record)

    df = pd.DataFrame(records)
    filepath = Path(output_dir) / f"{underlying.upper()}_options.parquet"
    df.to_parquet(filepath, index=False)
    print(f"Saved {len(df)} options records to {filepath}")
    return True


def download_theta_data(
    underlying: str,
    start_date: str,
    end_date: str,
    include_greeks: bool,
    output_dir: str,
    config: dict,
    dte_range: str = "all",
):
    if not THETA_DATA_AVAILABLE:
        print("ThetaData library is not available, falling back to synthetic data.")
        return False

    try:
        adapter = create_theta_data_adapter(
            username=config.get("username"),
            password=config.get("password"),
            api_key=config.get("api_key"),
            use_terminal=config.get("use_terminal", False),
        )
        adapter.connect()

        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

        # Download options data
        print(
            f"Downloading options for {underlying} from {start_date} to {end_date} using ThetaData..."
        )
        # Since ThetaData REST/Terminal fetches option chains or historical bars:
        chain = adapter.get_option_chain(underlying)
        if not chain.contracts:
            print("No contracts found, falling back to synthetic.")
            return False

        records = []
        for contract in chain.contracts:
            exp_date = (
                datetime.strptime(str(contract.expiration_date), "%Y-%m-%d").date()
                if isinstance(contract.expiration_date, str)
                else contract.expiration_date
            )
            dte = (exp_date - date.today()).days
            if not is_dte_in_range(dte, dte_range):
                continue

            records.append(
                {
                    "occ_symbol": contract.occ_symbol,
                    "underlying": contract.symbol,
                    "option_type": contract.option_type.value,
                    "strike": contract.strike_price,
                    "expiration": str(contract.expiration_date),
                    "dte": dte,
                    "bid": contract.bid,
                    "ask": contract.ask,
                    "mid": contract.mid_price,
                    "volume": contract.volume,
                    "open_interest": contract.open_interest,
                    "delta": contract.delta,
                    "gamma": contract.gamma,
                    "theta": contract.theta,
                    "vega": contract.vega,
                    "implied_volatility": contract.implied_volatility,
                }
            )

        if not records:
            print("No options contracts match the DTE filter, falling back to synthetic.")
            return False

        df = pd.DataFrame(records)
        os.makedirs(output_dir, exist_ok=True)
        filepath = Path(output_dir) / f"{underlying.upper()}_options.parquet"
        df.to_parquet(filepath, index=False)
        print(f"Successfully downloaded and saved {len(df)} options records to {filepath}")
        return True
    except Exception as e:
        print(f"ThetaData download failed due to error: {e}. Falling back to synthetic.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download or generate listed options data.")
    parser.add_argument(
        "--underlyings", nargs="+", required=True, help="Underlying symbols (e.g. AAPL MSFT)"
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--strike-range",
        default="ATM +/- 10",
        choices=["ATM +/- 10", "ATM +/- 20", "ATM +/- 50"],
        help="Strikes range around ATM",
    )
    parser.add_argument(
        "--dte-range",
        default="all",
        choices=["all", "0-7", "7-45", "45-90"],
        help="DTE range filter",
    )
    parser.add_argument(
        "--provider",
        default="theta_data",
        choices=["theta_data", "polygon", "ib"],
        help="Data provider",
    )
    parser.add_argument("--username", help="ThetaData username")
    parser.add_argument("--password", help="ThetaData password")
    parser.add_argument("--api-key", help="API Key")
    parser.add_argument(
        "--include-greeks", action="store_true", help="Calculate and include Greeks"
    )
    parser.add_argument("--output-dir", default="data/raw_options", help="Output directory")

    args = parser.parse_args()

    config = {
        "username": args.username,
        "password": args.password,
        "api_key": args.api_key,
        "use_terminal": False,
    }

    for underlying in args.underlyings:
        success = False
        if args.provider == "theta_data" and (args.username or args.api_key):
            success = download_theta_data(
                underlying,
                args.start,
                args.end,
                args.include_greeks,
                args.output_dir,
                config,
                args.dte_range,
            )

        if not success:
            generate_synthetic_options(
                underlying,
                args.start,
                args.end,
                args.strike_range,
                args.include_greeks,
                args.output_dir,
                args.dte_range,
            )


if __name__ == "__main__":
    main()

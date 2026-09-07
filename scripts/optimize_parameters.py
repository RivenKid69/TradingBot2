import os
import sys

sys.path.insert(0, os.getcwd())
import json
import argparse
import importlib.util
from typing import Any, Dict, List, Mapping
import numpy as np
import pandas as pd
from scipy.stats import norm


# Standard quant formulas
def calculate_black_scholes_delta(
    s: float,
    k: float,
    t_days: float,
    sigma: float = 0.30,
    r: float = 0.05,
    option_type: str = "call",
) -> float:
    """Calculate Black-Scholes Delta for options pricing."""
    if t_days <= 0 or s <= 0 or k <= 0 or sigma <= 0:
        return 1.0 if option_type == "call" else -1.0

    t = t_days / 365.0
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))

    if option_type.lower() == "call":
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1.0)


def calculate_black_scholes_premium(
    s: float,
    k: float,
    t_days: float,
    sigma: float = 0.30,
    r: float = 0.05,
    option_type: str = "call",
) -> float:
    """Calculate Black-Scholes option price."""
    if t_days <= 0:
        return max(0.0, s - k) if option_type == "call" else max(0.0, k - s)
    if s <= 0 or k <= 0 or sigma <= 0:
        return 0.0

    t = t_days / 365.0
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option_type.lower() == "call":
        return float(s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    else:
        return float(k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1))


def generate_grid(params_range: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """Generate cartesian product of parameters from a ranges dict."""
    import itertools

    keys = list(params_range.keys())
    lists = []
    for k in keys:
        spec = params_range[k]
        p_min = float(spec.get("min", 0.0))
        p_max = float(spec.get("max", 0.0))
        step = float(spec.get("step", 1.0))
        if step <= 0:
            step = 1.0

        values = []
        val = p_min
        # Handle float precision safety
        epsilon = step * 0.0001
        while val <= p_max + epsilon:
            # Cast to int if appropriate
            if step.is_integer() and p_min.is_integer():
                values.append(int(round(val)))
            else:
                values.append(round(val, 6))
            val += step

        if not values:
            values = [p_min]
        lists.append(values)

    grid = []
    for comb in itertools.product(*lists):
        grid.append(dict(zip(keys, comb)))
    return grid


def main():
    parser = argparse.ArgumentParser(description="Grid Search parameter optimizer for RivenQuant")
    parser.add_argument(
        "--asset",
        type=str,
        required=True,
        help="Asset class (equity, forex, futures, crypto, options)",
    )
    parser.add_argument(
        "--params_range", type=str, required=True, help="JSON string of parameter ranges"
    )
    parser.add_argument("--data_path", type=str, default=None, help="Optional data path override")
    parser.add_argument(
        "--metric",
        type=str,
        default="sharpe",
        choices=["sharpe", "max_drawdown", "profit_factor", "total_return", "win_rate"],
        help="Metric to optimize",
    )
    parser.add_argument(
        "--out", type=str, default="logs/optimization_results.json", help="Output path"
    )
    args = parser.parse_args()

    print(f"Starting parameter optimization sweep for asset: {args.asset.upper()}")
    print(f"Target optimization metric: {args.metric}")

    # 1. Resolve strategy module
    strategy_path = os.path.join("strategies", f"custom_{args.asset.lower()}.py")
    if not os.path.exists(strategy_path):
        print(f"ERROR: Strategy file not found at {strategy_path}")
        sys.exit(1)

    try:
        spec = importlib.util.spec_from_file_location(
            f"strategies.custom_{args.asset.lower()}", strategy_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"strategies.custom_{args.asset.lower()}"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"ERROR compiling strategy: {e}")
        sys.exit(1)

    # Find strategy class
    strategy_class = None
    for name, obj in vars(module).items():
        if isinstance(obj, type) and obj.__name__ not in ("BaseSignalPolicy", "BaseStrategy"):
            if hasattr(obj, "decide") and callable(getattr(obj, "decide")):
                strategy_class = obj
                break

    if not strategy_class:
        print(f"ERROR: No valid strategy class with decide() found in {strategy_path}")
        sys.exit(1)

    print(f"Loaded strategy class: {strategy_class.__name__}")

    # 2. Resolve parameter grid
    try:
        ranges = json.loads(args.params_range)
        grid = generate_grid(ranges)
    except Exception as e:
        print(f"ERROR parsing parameters range JSON: {e}")
        sys.exit(1)

    print(f"Generated parameter grid with {len(grid)} combinations.")

    # 3. Load appropriate parquet file
    data_file = args.data_path
    if not data_file:
        if args.asset.lower() == "equity":
            data_file = "data/raw_stocks/SPY.parquet"
        elif args.asset.lower() == "forex":
            data_file = "data/raw_forex/EUR_USD.parquet"
        elif args.asset.lower() == "futures":
            data_file = "data/raw_stocks/SPY.parquet"
        elif args.asset.lower() == "crypto":
            data_file = "artifacts/stocks/training_dataset.parquet"
        elif args.asset.lower() == "options":
            data_file = "data/raw_options/AAPL_options.parquet"

    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found at {data_file}")
        sys.exit(1)

    print(f"Loading dataset: {data_file}")
    try:
        df = pd.read_parquet(data_file)
        print(f"Dataset shape: {df.shape}")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    # Convert options date to timestamp
    if args.asset.lower() == "options" and "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"]).astype(np.int64) // 10**6
    elif "timestamp" not in df.columns and "ts_ms" in df.columns:
        df["timestamp"] = df["ts_ms"]

    if "timestamp" not in df.columns:
        df["timestamp"] = range(len(df))

    # Identify primary symbol
    symbol = "SPY"
    if "symbol" in df.columns and not df.empty:
        symbol = str(df["symbol"].iloc[0])
    elif "occ_symbol" in df.columns and not df.empty:
        symbol = str(df["occ_symbol"].iloc[0])

    # 4. Grid Sweep Loop
    from core_contracts import PolicyCtx

    results = []

    for i, combination in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] Running: {combination}")

        # Instantiate strategy
        try:
            strategy = strategy_class()
            if hasattr(strategy, "setup") and callable(strategy.setup):
                strategy.setup(combination)
            else:
                for k, v in combination.items():
                    setattr(strategy, k, v)
        except Exception as e:
            print(f"  Error setting up combination: {e}")
            continue

        # In-memory backtest state
        cash = 100000.0
        positions = {}  # symbol -> position quantity
        equity_curve = []
        trades = []  # list of executed trades: {entry_ts, exit_ts, pnl, etc.}
        last_prices = {}
        daily_equities = []

        # Track active orders and position trades for PNL/WinRate metric
        active_positions = {}  # symbol -> {qty, entry_price, ts}

        # Run loop over bars
        for idx, row in df.iterrows():
            ts = int(row.get("timestamp", idx))

            # Map default price
            ref_price = float(row.get("close", row.get("spot_price", row.get("ref_price", 100.0))))
            last_prices[symbol] = ref_price

            # Options-specific values
            opt_premium = 0.0
            strike_price = 0.0
            portfolio_delta = 0.0

            if args.asset.lower() == "options":
                strike_price = float(row.get("strike", ref_price))
                bid = float(row.get("bid", 0.0))
                ask = float(row.get("ask", 0.0))
                opt_premium = (
                    (bid + ask) / 2.0
                    if (bid > 0 and ask > 0)
                    else calculate_black_scholes_premium(
                        ref_price, strike_price, float(row.get("dte", 30.0))
                    )
                )
                portfolio_delta = calculate_black_scholes_delta(
                    ref_price,
                    strike_price,
                    float(row.get("dte", 30.0)),
                    option_type=str(row.get("option_type", "call")),
                )

                # Option symbol tracking
                opt_sym = str(row.get("occ_symbol", f"{symbol}_OPT"))
                last_prices[opt_sym] = opt_premium

            # Gather features
            features = dict(row)
            features["ref_price"] = ref_price
            if args.asset.lower() == "options":
                features["strike_price"] = strike_price
                features["option_premium"] = opt_premium
                features["portfolio_delta"] = portfolio_delta

            # Execute decide
            ctx = PolicyCtx(ts=ts, symbol=symbol)
            try:
                orders = strategy.decide(features, ctx)
            except Exception as e:
                # If strategy validation fails because of features we didn't mock
                orders = []

            # Execute orders immediately at bar close
            for o in orders:
                qty = float(o.quantity)
                if qty <= 0:
                    continue

                order_sym = o.symbol or symbol
                if (
                    args.asset.lower() == "options"
                    and getattr(o, "client_order_id", "") == "covered_call_write"
                ):
                    # Covered call writes call option symbol
                    order_sym = str(row.get("occ_symbol", f"{symbol}_OPT"))

                fill_price = last_prices.get(order_sym, ref_price)

                # Transaction costs: 1 bps slippage
                slippage = fill_price * 0.0001

                if o.side.value.upper() in ("BUY", "LONG"):
                    cost = qty * (fill_price + slippage)
                    cash -= cost
                    positions[order_sym] = positions.get(order_sym, 0.0) + qty

                    # Track trade entry
                    if order_sym not in active_positions or active_positions[order_sym]["qty"] <= 0:
                        active_positions[order_sym] = {"qty": qty, "price": fill_price, "ts": ts}
                else:
                    proceeds = qty * (fill_price - slippage)
                    cash += proceeds
                    positions[order_sym] = positions.get(order_sym, 0.0) - qty

                    # Track trade exit PNL
                    if order_sym in active_positions:
                        entry = active_positions[order_sym]
                        if entry["qty"] > 0:
                            pnl = (fill_price - entry["price"]) * min(qty, entry["qty"])
                            trades.append(
                                {
                                    "entry_ts": entry["ts"],
                                    "exit_ts": ts,
                                    "pnl": pnl,
                                    "pnl_pct": (
                                        pnl / (entry["price"] * entry["qty"])
                                        if entry["price"] > 0
                                        else 0
                                    ),
                                }
                            )
                            entry["qty"] -= qty
                            if entry["qty"] <= 0:
                                del active_positions[order_sym]

            # Valuation
            pos_val = sum(q * last_prices.get(sym, ref_price) for sym, q in positions.items())
            equity = cash + pos_val
            equity_curve.append(equity)
            daily_equities.append(equity)

        # 5. Compute performance statistics
        equity_series = pd.Series(equity_curve)
        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            total_return = float((equity_series.iloc[-1] - 100000.0) / 100000.0 * 100.0)

            # Sharpe Ratio
            if len(returns) > 0 and returns.std() > 0:
                sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
            else:
                sharpe = 0.0

            # Max Drawdown
            peaks = equity_series.cummax()
            drawdowns = (peaks - equity_series) / peaks
            max_drawdown = float(drawdowns.max() * 100.0)

            # Profit Factor & Win Rate
            wins = [t["pnl"] for t in trades if t["pnl"] > 0]
            losses = [abs(t["pnl"]) for t in trades if t["pnl"] < 0]

            win_rate = float(len(wins) / len(trades) * 100.0) if trades else 0.0
            profit_factor = (
                float(sum(wins) / sum(losses))
                if (losses and sum(losses) > 0)
                else (999.0 if wins else 1.0)
            )
            recovery_factor = (
                float(total_return / max_drawdown) if max_drawdown > 0 else total_return
            )
        else:
            total_return = 0.0
            sharpe = 0.0
            max_drawdown = 0.0
            win_rate = 0.0
            profit_factor = 1.0
            recovery_factor = 0.0

        results.append(
            {
                "parameters": combination,
                "metrics": {
                    "sharpe": round(sharpe, 4),
                    "max_drawdown": round(max_drawdown, 2),
                    "profit_factor": round(profit_factor, 2),
                    "total_return": round(total_return, 2),
                    "win_rate": round(win_rate, 2),
                    "recovery_factor": round(recovery_factor, 2),
                },
                "trades_count": len(trades),
            }
        )

    # Sort results by selected metric
    reverse_sort = True
    if args.metric == "max_drawdown":
        reverse_sort = False

    results.sort(key=lambda x: x["metrics"].get(args.metric, 0.0), reverse=reverse_sort)

    best_comb = results[0] if results else {}

    # Save output
    output_data = {"asset": args.asset, "best_combination": best_comb, "all_combinations": results}

    try:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"SUCCESS: Saved optimization sweep results to {args.out}")
        print(
            f"Best parameters: {best_comb.get('parameters')} -> Sharpe: {best_comb.get('metrics', {}).get('sharpe')}, MaxDD: {best_comb.get('metrics', {}).get('max_drawdown')}%"
        )
    except Exception as e:
        print(f"ERROR saving output to JSON: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

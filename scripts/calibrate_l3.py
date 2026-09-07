#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 Calibration CLI Tool.

Calibrates L3 LOB simulation parameters from historical data.

Usage:
    # From LOBSTER data
    python scripts/calibrate_l3.py --format lobster --messages data/AAPL_messages.csv --output configs/calibrated_l3.yaml

    # From trade CSV
    python scripts/calibrate_l3.py --format csv --trades data/trades.csv --symbol AAPL --output configs/calibrated_l3.yaml

    # From Parquet
    python scripts/calibrate_l3.py --format parquet --trades data/trades.parquet --output configs/calibrated_l3.yaml

    # With ADV and volatility
    python scripts/calibrate_l3.py --format csv --trades data/trades.csv --adv 10000000 --volatility 0.02 --output configs/calibrated_l3.yaml

Output:
    - YAML configuration file ready for L3ExecutionProvider
    - JSON calibration report with confidence intervals
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob.calibration_pipeline import (
    L3CalibrationPipeline,
    L3CalibrationResult,
    create_calibration_pipeline,
)
from lob.config import L3ExecutionConfig
from lob.data_adapters import (
    DataSourceType,
    LOBSTERAdapter,
    create_lob_adapter,
)
from lob.data_structures import Side

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calibrate L3 LOB simulation parameters from historical data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # LOBSTER format
    python calibrate_l3.py --format lobster --messages data/AAPL_messages.csv

    # CSV trades
    python calibrate_l3.py --format csv --trades data/trades.csv --symbol AAPL

    # With market params
    python calibrate_l3.py --format csv --trades data/trades.csv --adv 10000000 --volatility 0.02
        """,
    )

    # Input format
    parser.add_argument(
        "--format",
        choices=["lobster", "itch", "csv", "parquet", "json"],
        default="csv",
        help="Input data format (default: csv)",
    )

    # Input files
    parser.add_argument(
        "--messages",
        type=str,
        help="Path to message/order file (LOBSTER/ITCH format)",
    )
    parser.add_argument(
        "--orderbook",
        type=str,
        help="Path to orderbook snapshot file (LOBSTER format)",
    )
    parser.add_argument(
        "--trades",
        type=str,
        help="Path to trades file (CSV/Parquet format)",
    )
    parser.add_argument(
        "--orders",
        type=str,
        help="Path to orders file (CSV/Parquet format)",
    )
    parser.add_argument(
        "--latencies",
        type=str,
        help="Path to latency observations file",
    )

    # Symbol and market params
    parser.add_argument(
        "--symbol",
        type=str,
        default="",
        help="Trading symbol",
    )
    parser.add_argument(
        "--asset-class",
        choices=["equity", "crypto"],
        default="equity",
        help="Asset class (default: equity)",
    )
    parser.add_argument(
        "--adv",
        type=float,
        default=10_000_000.0,
        help="Average daily volume (default: 10,000,000)",
    )
    parser.add_argument(
        "--volatility",
        type=float,
        default=0.02,
        help="Average volatility (default: 0.02)",
    )

    # CSV column mapping
    parser.add_argument(
        "--price-col",
        type=str,
        default="price",
        help="Price column name in CSV (default: price)",
    )
    parser.add_argument(
        "--qty-col",
        type=str,
        default="qty",
        help="Quantity column name in CSV (default: qty)",
    )
    parser.add_argument(
        "--side-col",
        type=str,
        default="side",
        help="Side column name in CSV (default: side)",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column name in CSV (default: timestamp)",
    )
    parser.add_argument(
        "--pre-mid-col",
        type=str,
        default="pre_mid",
        help="Pre-trade mid price column (default: pre_mid)",
    )
    parser.add_argument(
        "--post-mid-col",
        type=str,
        default="post_mid",
        help="Post-trade mid price column (default: post_mid)",
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="configs/calibrated_l3.yaml",
        help="Output config file path (default: configs/calibrated_l3.yaml)",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Output JSON report file path",
    )

    # Options
    parser.add_argument(
        "--max-records",
        type=int,
        help="Maximum records to process",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def load_trades_csv(
    path: str,
    price_col: str,
    qty_col: str,
    side_col: str,
    timestamp_col: str,
    pre_mid_col: str,
    post_mid_col: str,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load trades from CSV file."""
    import csv

    trades = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_records and i >= max_records:
                break

            try:
                trade = {
                    "timestamp_ms": int(float(row.get(timestamp_col, 0))),
                    "price": float(row.get(price_col, 0)),
                    "qty": float(row.get(qty_col, 0)),
                    "side": int(row.get(side_col, 1)),
                }

                if pre_mid_col in row and row[pre_mid_col]:
                    trade["pre_mid"] = float(row[pre_mid_col])
                if post_mid_col in row and row[post_mid_col]:
                    trade["post_mid"] = float(row[post_mid_col])

                trades.append(trade)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid row {i}: {e}")

    return trades


def load_trades_parquet(
    path: str,
    price_col: str,
    qty_col: str,
    side_col: str,
    timestamp_col: str,
    pre_mid_col: str,
    post_mid_col: str,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load trades from Parquet file."""
    try:
        import pandas as pd

        df = pd.read_parquet(path)

        if max_records:
            df = df.head(max_records)

        trades = []
        for _, row in df.iterrows():
            trade = {
                "timestamp_ms": int(row.get(timestamp_col, 0)),
                "price": float(row.get(price_col, 0)),
                "qty": float(row.get(qty_col, 0)),
                "side": int(row.get(side_col, 1)),
            }

            if pre_mid_col in df.columns and pd.notna(row.get(pre_mid_col)):
                trade["pre_mid"] = float(row[pre_mid_col])
            if post_mid_col in df.columns and pd.notna(row.get(post_mid_col)):
                trade["post_mid"] = float(row[post_mid_col])

            trades.append(trade)

        return trades

    except ImportError:
        logger.error("pandas required for Parquet support: pip install pandas pyarrow")
        return []


def load_trades_json(
    path: str,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load trades from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    trades = data.get("trades", data) if isinstance(data, dict) else data

    if max_records:
        trades = trades[:max_records]

    return trades


def load_orders_file(
    path: str,
    price_col: str = "price",
    qty_col: str = "qty",
    side_col: str = "side",
    timestamp_col: str = "timestamp",
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load orders from file (CSV, Parquet, or JSON).

    Auto-detects format based on file extension.
    """
    path_lower = path.lower()

    if path_lower.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        orders = data.get("orders", data) if isinstance(data, dict) else data
        if max_records:
            orders = orders[:max_records]
        return orders

    elif path_lower.endswith(".parquet"):
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            if max_records:
                df = df.head(max_records)

            orders = []
            for _, row in df.iterrows():
                order = {
                    "timestamp_ns": int(row.get(timestamp_col, 0)) * 1_000_000,
                    "price": float(row.get(price_col, 0)),
                    "qty": float(row.get(qty_col, 0)),
                    "side": int(row.get(side_col, 1)),
                    "event_type": str(row.get("event_type", "ADD")),
                    "fill_qty": float(row.get("fill_qty", 0)),
                }
                orders.append(order)
            return orders

        except ImportError:
            logger.error("pandas required for Parquet: pip install pandas pyarrow")
            return []

    else:  # CSV
        import csv

        orders = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_records and i >= max_records:
                    break
                try:
                    order = {
                        "timestamp_ns": int(float(row.get(timestamp_col, 0))) * 1_000_000,
                        "price": float(row.get(price_col, 0)),
                        "qty": float(row.get(qty_col, 0)),
                        "side": int(row.get(side_col, 1)),
                        "event_type": str(row.get("event_type", "ADD")),
                        "fill_qty": float(row.get("fill_qty", 0)),
                    }
                    orders.append(order)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid order row {i}: {e}")

        return orders


def save_config_yaml(config: L3ExecutionConfig, path: str) -> None:
    """Save L3ExecutionConfig to YAML file."""
    import yaml

    # Convert config to dict
    config_dict = config.model_dump()

    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def save_report_json(
    result: L3CalibrationResult,
    config: L3ExecutionConfig,
    args: argparse.Namespace,
    path: str,
) -> None:
    """Save calibration report to JSON file."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "symbol": args.symbol,
        "asset_class": args.asset_class,
        "input_format": args.format,
        "data_summary": {
            "n_trades": result.n_trades,
            "n_orders": result.n_orders,
            "duration_sec": result.data_duration_sec,
            "calibration_quality": result.calibration_quality,
        },
        "market_params": {
            "avg_adv": args.adv,
            "avg_volatility": args.volatility,
        },
        "calibrated_parameters": {
            "impact": result.get_impact_params(),
            "fill_probability": result.get_fill_prob_params(),
        },
        "confidence_intervals": result.confidence_intervals,
    }

    # Add impact metrics if available
    if result.impact_result:
        report["impact_metrics"] = {
            "r_squared": result.impact_result.r_squared,
            "rmse": result.impact_result.rmse,
            "mae": result.impact_result.mae,
        }

    # Add latency summary if available
    if result.latency_result and result.latency_result.n_samples > 0:
        report["latency_summary"] = {
            "feed_latency_us": result.latency_result.feed_latency.mean_us,
            "order_latency_us": result.latency_result.order_latency.mean_us,
            "exchange_latency_us": result.latency_result.exchange_latency.mean_us,
            "fill_latency_us": result.latency_result.fill_latency.mean_us,
            "n_samples": result.latency_result.n_samples,
        }

    # Add queue dynamics if available
    if result.queue_dynamics:
        report["queue_dynamics"] = {
            "avg_arrival_rate": result.queue_dynamics.avg_arrival_rate,
            "avg_cancel_rate": result.queue_dynamics.avg_cancel_rate,
            "avg_time_in_queue_sec": result.queue_dynamics.avg_time_in_queue_sec,
        }

    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting L3 calibration for {args.symbol or 'unknown symbol'}")
    logger.info(f"Input format: {args.format}")

    # Create calibration pipeline
    pipeline = create_calibration_pipeline(
        symbol=args.symbol,
        asset_class=args.asset_class,
    )

    # Set market parameters
    pipeline.set_market_params(
        avg_adv=args.adv,
        avg_volatility=args.volatility,
    )

    # Load data based on format
    if args.format == "lobster":
        if not args.messages:
            logger.error("--messages required for LOBSTER format")
            return 1

        adapter = LOBSTERAdapter(symbol=args.symbol)

        logger.info(f"Loading LOBSTER messages from {args.messages}")
        for update in adapter.stream_updates(args.messages, max_messages=args.max_records):
            if update.update_type == "EXECUTE":
                pipeline.add_trade(
                    timestamp_ms=update.timestamp_ns // 1_000_000,
                    price=update.price,
                    qty=update.fill_qty or update.qty,
                    side=1 if update.side == Side.BUY else -1,
                )
            elif update.update_type in ("ADD", "DELETE", "MODIFY"):
                pipeline.add_order(
                    timestamp_ns=update.timestamp_ns,
                    price=update.price,
                    qty=update.qty,
                    side=update.side,
                    event_type=update.update_type,
                )

        logger.info(f"Processed {adapter.stats.messages_processed} messages")

    elif args.format == "csv":
        if not args.trades:
            logger.error("--trades required for CSV format")
            return 1

        logger.info(f"Loading trades from {args.trades}")
        trades = load_trades_csv(
            args.trades,
            args.price_col,
            args.qty_col,
            args.side_col,
            args.timestamp_col,
            args.pre_mid_col,
            args.post_mid_col,
            args.max_records,
        )

        for trade in trades:
            pipeline.add_trade(**trade)

        logger.info(f"Loaded {len(trades)} trades")

    elif args.format == "parquet":
        if not args.trades:
            logger.error("--trades required for Parquet format")
            return 1

        logger.info(f"Loading trades from {args.trades}")
        trades = load_trades_parquet(
            args.trades,
            args.price_col,
            args.qty_col,
            args.side_col,
            args.timestamp_col,
            args.pre_mid_col,
            args.post_mid_col,
            args.max_records,
        )

        for trade in trades:
            pipeline.add_trade(**trade)

        logger.info(f"Loaded {len(trades)} trades")

    elif args.format == "json":
        if not args.trades:
            logger.error("--trades required for JSON format")
            return 1

        logger.info(f"Loading trades from {args.trades}")
        trades = load_trades_json(args.trades, args.max_records)

        for trade in trades:
            pipeline.add_trade(
                timestamp_ms=trade.get("timestamp_ms", 0),
                price=trade.get("price", 0.0),
                qty=trade.get("qty", 0.0),
                side=trade.get("side", 1),
                pre_trade_mid=trade.get("pre_mid"),
                post_trade_mid=trade.get("post_mid"),
            )

        logger.info(f"Loaded {len(trades)} trades")

    else:
        logger.error(f"Unsupported format: {args.format}")
        return 1

    # Load additional orders if provided
    if args.orders:
        logger.info(f"Loading orders from {args.orders}")
        try:
            orders = load_orders_file(
                args.orders,
                price_col=args.price_col,
                qty_col=args.qty_col,
                side_col=args.side_col,
                timestamp_col=args.timestamp_col,
                max_records=args.max_records,
            )
            for order in orders:
                pipeline.add_order(
                    timestamp_ns=order.get(
                        "timestamp_ns", order.get("timestamp_ms", 0) * 1_000_000
                    ),
                    price=order.get("price", 0.0),
                    qty=order.get("qty", 0.0),
                    side=Side.BUY if order.get("side", 1) > 0 else Side.SELL,
                    event_type=order.get("event_type", "ADD"),
                    fill_qty=order.get("fill_qty", 0.0),
                )
            logger.info(f"Loaded {len(orders)} orders")
        except Exception as e:
            logger.warning(f"Failed to load orders: {e}")

    # Load latencies if provided
    if args.latencies:
        logger.info(f"Loading latencies from {args.latencies}")
        try:
            with open(args.latencies, "r") as f:
                latency_data = json.load(f)

            for obs in latency_data:
                pipeline.add_latency_observation(
                    timestamp_ns=obs.get("timestamp_ns", 0),
                    latency_type=obs.get("type", "order"),
                    latency_us=obs.get("latency_us", 100.0),
                )

            logger.info(f"Loaded {len(latency_data)} latency observations")
        except Exception as e:
            logger.warning(f"Failed to load latencies: {e}")

    # Check data availability
    if pipeline.n_trades == 0 and pipeline.n_orders == 0:
        logger.error("No data loaded for calibration")
        return 1

    logger.info(f"Total: {pipeline.n_trades} trades, {pipeline.n_orders} orders")

    # Run calibration
    logger.info("Running calibration...")
    config = pipeline.run_full_calibration()
    result = pipeline.get_calibration_result()

    # Log results
    logger.info(f"Calibration quality: {result.calibration_quality}")

    if result.impact_result:
        params = result.get_impact_params()
        logger.info(
            f"Impact params: eta={params.get('eta', 0):.4f}, gamma={params.get('gamma', 0):.4f}"
        )
        logger.info(f"Impact R²: {result.impact_result.r_squared:.4f}")

    if result.fill_prob_result:
        params = result.get_fill_prob_params()
        logger.info(
            f"Fill prob params: base_rate={params.get('base_rate', params.get('arrival_rate', 0)):.1f}"
        )

    # Save output
    logger.info(f"Saving config to {args.output}")
    try:
        save_config_yaml(config, args.output)
        logger.info(f"Config saved to {args.output}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return 1

    # Save report if requested
    if args.report:
        logger.info(f"Saving report to {args.report}")
        try:
            save_report_json(result, config, args, args.report)
            logger.info(f"Report saved to {args.report}")
        except Exception as e:
            logger.warning(f"Failed to save report: {e}")

    logger.info("Calibration complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

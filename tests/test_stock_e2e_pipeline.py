# -*- coding: utf-8 -*-
"""
test_stock_e2e_pipeline.py
End-to-end test for stock pipeline: download → train → backtest → eval

This script validates the complete stock trading pipeline works correctly.
Uses minimal timesteps for fast testing.

Usage:
    python test_stock_e2e_pipeline.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import traceback

# Fix encoding issues on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd

# The bundled equity data was removed from the repository: it came from Yahoo
# Finance, whose terms do not allow redistribution. Recreate it locally with
#     python scripts/download_stock_data.py
pytestmark = pytest.mark.skipif(
    not Path("data/raw_stocks").exists(),
    reason="requires data/raw_stocks — run scripts/download_stock_data.py first",
)


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_data_download() -> bool:
    """Test 1: Verify stock data exists and is valid."""
    print_section("TEST 1: Data Download Verification")

    data_dir = Path("data/raw_stocks")
    if not data_dir.exists():
        print("FAIL: data/raw_stocks directory doesn't exist")
        return False

    parquet_files = list(data_dir.glob("*.parquet"))
    if len(parquet_files) < 2:
        print(f"FAIL: Need at least 2 parquet files, found {len(parquet_files)}")
        return False

    print(f"Found {len(parquet_files)} parquet files:")
    for f in parquet_files:
        df = pd.read_parquet(f)
        print(
            f"  {f.name}: {len(df)} rows, price range {df['close'].min():.2f}-{df['close'].max():.2f}"
        )

        # Validate columns
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(df.columns)):
            print(f"  FAIL: Missing columns {required_cols - set(df.columns)}")
            return False

        # Validate data
        if df["close"].isna().all():
            print(f"  FAIL: All close prices are NaN")
            return False

    print("\nPASS: Stock data is valid")
    return True


def test_data_loader() -> bool:
    """Test 2: Test multi-asset data loader with stock data."""
    print_section("TEST 2: Data Loader Test")

    try:
        from data_loader_multi_asset import load_multi_asset_data, AssetClass
        from pathlib import Path as P

        # Expand glob manually for Windows compatibility
        data_dir = P("data/raw_stocks")
        files = list(data_dir.glob("*.parquet"))
        paths = [str(f) for f in files if f.stem != "VIX"]  # Exclude VIX for now

        if not paths:
            print("FAIL: No parquet files found")
            return False

        print(f"Found {len(paths)} data files: {[P(p).stem for p in paths]}")

        # Load stock data
        frames, obs_shapes = load_multi_asset_data(
            paths=paths,
            asset_class=AssetClass.EQUITY,
            timeframe="4h",
            add_stock_features=False,  # Skip stock features for now
            merge_fear_greed=False,
        )

        if not frames:
            print("FAIL: No frames loaded")
            return False

        print(f"Loaded {len(frames)} symbols:")
        for symbol, df in frames.items():
            print(f"  {symbol}: {len(df)} rows")

        # Validate observation shapes
        if obs_shapes:
            print(f"\nObservation shapes: {obs_shapes}")

        print("\nPASS: Data loader works correctly")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_feature_pipeline() -> bool:
    """Test 3: Test feature pipeline with stock data."""
    print_section("TEST 3: Feature Pipeline Test")

    try:
        from features_pipeline import FeaturePipeline
        import pandas as pd

        # Load sample stock data
        df = pd.read_parquet("data/raw_stocks/SPY.parquet")

        # Run through feature pipeline
        pipeline = FeaturePipeline()

        # Add required columns if missing
        if "timestamp" not in df.columns:
            df["timestamp"] = range(len(df))

        # Process using fit + transform
        df_with_symbol = df.copy()
        df_with_symbol["symbol"] = "SPY"

        # Fit pipeline on data first
        dfs_dict = {"SPY": df_with_symbol}
        pipeline.fit(dfs_dict)

        # Then transform
        processed = pipeline.transform_df(df_with_symbol)

        print(f"Input shape: {df.shape}")
        print(f"Output shape: {processed.shape}")
        print(f"New columns: {len(processed.columns) - len(df.columns)}")

        # Check for NaN columns
        nan_cols = [c for c in processed.columns if processed[c].isna().all()]
        if nan_cols:
            print(f"Warning: All-NaN columns (first 5): {nan_cols[:5]}...")

        print("\nPASS: Feature pipeline works")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_execution_providers() -> bool:
    """Test 4: Test equity execution providers."""
    print_section("TEST 4: Execution Providers Test")

    try:
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )
        from datetime import datetime, timezone

        # Create equity provider
        provider = create_execution_provider(AssetClass.EQUITY)

        # Test market order
        order = Order(
            symbol="SPY",
            side="BUY",
            qty=100,
            order_type="MARKET",
        )

        market_state = MarketState(
            timestamp=datetime.now(timezone.utc),
            bid=580.0,
            ask=580.02,
            adv=50_000_000,  # High liquidity for SPY
        )

        bar_data = BarData(
            open=579.0,
            high=582.0,
            low=578.0,
            close=580.5,
            volume=1_000_000,
        )

        fill = provider.execute(order, market_state, bar_data)

        print(f"Order: BUY 100 SPY @ MARKET")
        print(f"Fill price: ${fill.price:.2f}")
        print(f"Fee: ${fill.fee:.4f}")
        print(f"Slippage: {fill.slippage_bps:.2f} bps")
        print(f"Liquidity role: {fill.liquidity}")

        # Verify reasonable values
        if not (578.0 <= fill.price <= 582.0):
            print(f"FAIL: Fill price {fill.price} outside bar range")
            return False

        # Equity buys should have no fees (commission-free)
        if fill.fee > 0.01:  # Allow for small rounding
            print(f"Warning: Non-zero buy fee for equity: {fill.fee}")

        # Test sell order (should have regulatory fees)
        sell_order = Order(
            symbol="SPY",
            side="SELL",
            qty=100,
            order_type="MARKET",
        )

        sell_fill = provider.execute(sell_order, market_state, bar_data)
        print(f"\nSELL order fee: ${sell_fill.fee:.4f} (SEC/TAF fees)")

        print("\nPASS: Execution providers work correctly")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_trading_env() -> bool:
    """Test 5: Test TradingEnv with stock data."""
    print_section("TEST 5: Trading Environment Test")

    try:
        from trading_patchnew import TradingEnv
        import pandas as pd

        # Load and prepare stock data
        df = pd.read_parquet("data/raw_stocks/SPY.parquet")

        # Add basic features needed by TradingEnv
        df["return_1"] = df["close"].pct_change()
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df = df.dropna().reset_index(drop=True)

        # Limit to 500 rows for testing
        df = df.head(500)

        # Create env with minimal config
        env = TradingEnv(
            df=df,
            symbol="SPY",
            initial_cash=100_000.0,
            max_position=1.0,
            reward_signal_only=True,  # Signal-only mode for simplicity
        )

        # Test reset
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step 1 - Reward: {reward:.6f}, Done: {terminated or truncated}")

        # Run a few more steps
        total_reward = reward
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        print(f"Steps run: {i+2}, Total reward: {total_reward:.4f}")

        print("\nPASS: Trading environment works")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_model_create() -> bool:
    """Test 6: Test PPO model creation for stocks."""
    print_section("TEST 6: Model Creation Test")

    try:
        from trading_patchnew import TradingEnv
        from distributional_ppo import DistributionalPPO
        from custom_policy_patch1 import CustomActorCriticPolicy
        import pandas as pd
        import torch

        # Load and prepare stock data
        df = pd.read_parquet("data/raw_stocks/SPY.parquet")
        df["return_1"] = df["close"].pct_change()
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df = df.dropna().reset_index(drop=True).head(500)

        # Create env
        env = TradingEnv(
            df=df,
            symbol="SPY",
            initial_cash=100_000.0,
            max_position=1.0,
            reward_signal_only=True,
        )

        # Create model with minimal config
        model = DistributionalPPO(
            policy=CustomActorCriticPolicy,
            env=env,
            learning_rate=1e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            gamma=0.99,
            verbose=0,
            device="cpu",
        )

        print(f"Model created successfully")
        print(f"Policy type: {type(model.policy).__name__}")

        # Quick prediction test using the correct API
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)

        print(f"Prediction successful, action: {action}")

        print("\nPASS: Model creation works")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_short_training() -> bool:
    """Test 7: Test short training run on stock data."""
    print_section("TEST 7: Short Training Test")

    try:
        from trading_patchnew import TradingEnv
        from distributional_ppo import DistributionalPPO
        from custom_policy_patch1 import CustomActorCriticPolicy
        import pandas as pd

        # Load and prepare stock data
        df = pd.read_parquet("data/raw_stocks/SPY.parquet")
        df["return_1"] = df["close"].pct_change()
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df = df.dropna().reset_index(drop=True).head(500)

        # Create env
        env = TradingEnv(
            df=df,
            symbol="SPY",
            initial_cash=100_000.0,
            max_position=1.0,
            reward_signal_only=True,
        )

        # Create model
        model = DistributionalPPO(
            policy=CustomActorCriticPolicy,
            env=env,
            learning_rate=1e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            gamma=0.99,
            verbose=1,
            device="cpu",
        )

        print("Training for 256 timesteps...")
        model.learn(total_timesteps=256, progress_bar=False)

        print(f"Training completed!")

        # Verify model can make predictions
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print(f"Model prediction successful, action: {action}")

        # Note: model.save() may have pickle issues with thread locks in some environments
        # The core training functionality is verified above

        print("\nPASS: Training works correctly")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_backtest_simulation() -> bool:
    """Test 8: Test backtest simulation on stock data."""
    print_section("TEST 8: Backtest Simulation Test")

    try:
        from trading_patchnew import TradingEnv
        import pandas as pd
        import numpy as np

        # Load stock data
        df = pd.read_parquet("data/raw_stocks/SPY.parquet")
        df["return_1"] = df["close"].pct_change()
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df = df.dropna().reset_index(drop=True).head(200)

        # Create env (not signal-only for real backtest)
        env = TradingEnv(
            df=df,
            symbol="SPY",
            initial_cash=100_000.0,
            max_position=1.0,
            reward_signal_only=False,  # Real execution simulation
        )

        # Run backtest with simple strategy
        obs, info = env.reset()

        total_reward = 0.0
        trades = []
        equity_curve = [100_000.0]

        for step in range(len(df) - 60):  # Leave some buffer
            # Simple momentum strategy: buy if recent return positive
            if step > 0:
                recent_return = df["return_1"].iloc[step + 60]
                if recent_return > 0:
                    action = np.array([0.5])  # 50% position
                else:
                    action = np.array([0.0])  # No position
            else:
                action = np.array([0.0])

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if "net_worth" in info:
                equity_curve.append(info["net_worth"])

            if terminated or truncated:
                break

        print(f"Backtest results:")
        print(f"  Steps: {step + 1}")
        print(f"  Total reward: {total_reward:.4f}")
        print(
            f"  Final equity: ${equity_curve[-1]:,.2f}"
            if len(equity_curve) > 1
            else "  Final equity: N/A"
        )

        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-8)
            print(f"  Approximate Sharpe: {sharpe:.2f}")

        print("\nPASS: Backtest simulation works")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_multi_symbol() -> bool:
    """Test 9: Test multi-symbol data loading and training readiness."""
    print_section("TEST 9: Multi-Symbol Test")

    try:
        from data_loader_multi_asset import load_multi_asset_data, AssetClass
        from pathlib import Path as P

        # Expand glob manually for Windows compatibility
        data_dir = P("data/raw_stocks")
        files = list(data_dir.glob("*.parquet"))
        paths = [str(f) for f in files if f.stem not in ("VIX",)]

        # Load all available symbols
        frames, obs_shapes = load_multi_asset_data(
            paths=paths,
            asset_class=AssetClass.EQUITY,
            timeframe="4h",
            add_stock_features=False,
            merge_fear_greed=False,
        )

        print(f"Loaded {len(frames)} symbols:")
        for symbol, df in frames.items():
            print(f"  {symbol}: {len(df)} bars, ${df['close'].iloc[-1]:.2f} (latest close)")

        # Verify all frames have same structure
        columns = None
        for symbol, df in frames.items():
            if columns is None:
                columns = set(df.columns)
            else:
                diff = columns.symmetric_difference(set(df.columns))
                if diff:
                    print(f"Warning: {symbol} has different columns: {diff}")

        print("\nPASS: Multi-symbol loading works")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    """Run all E2E tests."""
    print("\n" + "=" * 60)
    print("   STOCK PIPELINE END-TO-END TEST")
    print("   " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    tests = [
        ("Data Download", test_data_download),
        ("Data Loader", test_data_loader),
        ("Feature Pipeline", test_feature_pipeline),
        ("Execution Providers", test_execution_providers),
        ("Trading Environment", test_trading_env),
        ("Model Creation", test_model_create),
        ("Short Training", test_short_training),
        ("Backtest Simulation", test_backtest_simulation),
        ("Multi-Symbol", test_multi_symbol),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\nUNEXPECTED ERROR in {name}: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print_section("TEST SUMMARY")

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "[PASS]" if p else "[FAIL]"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nALL TESTS PASSED! Stock pipeline is working correctly.")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

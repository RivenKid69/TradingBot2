import os
import subprocess
import pandas as pd
import pytest


def test_selected_features_filtering(tmp_path):
    # 1. Create a dummy price parquet file representing equity prices
    df = pd.DataFrame(
        {
            "ts_ms": [1700000000000, 1700000240000, 1700000480000],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "close": [150.0, 151.0, 152.0],
            "open": [149.0, 150.0, 151.0],
            "high": [151.0, 152.0, 153.0],
            "low": [148.0, 149.0, 150.0],
            "volume": [1000.0, 2000.0, 1500.0],
            "extra_greeks_delta": [0.5, 0.6, 0.7],  # custom feature present in raw input
        }
    )
    in_path = tmp_path / "prices.parquet"
    out_path = tmp_path / "features.parquet"
    df.to_parquet(in_path, index=False)

    # 2. Run make_features.py via subprocess with selected features
    # Note: price_col="close", open_col="open", high_col="high", low_col="low", volume_col="volume"
    cmd = [
        ".venv/bin/python",
        "make_features.py",
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--price-col",
        "close",
        "--open-col",
        "open",
        "--high-col",
        "high",
        "--low-col",
        "low",
        "--volume-col",
        "volume",
        "--lookbacks",
        "240",
        "--selected-features",
        "rsi,sma_240,extra_greeks_delta,ema_240,williams_r_240,keltner_upper_240,adx_240,obv,mfi_240,plus_di_240,minus_di_240,std_returns_240,mom_240,donchian_upper_240,ao,cmf_240,pvt",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"make_features.py failed: {res.stderr}"

    # 3. Read output and verify columns
    out_df = pd.read_parquet(out_path)
    print("Output columns:", list(out_df.columns))

    # Mandatory columns should be kept
    for col in ["ts_ms", "symbol", "ref_price", "close", "open", "high", "low"]:
        assert col in out_df.columns, f"Mandatory column {col} missing"

    # Selected columns should be present
    assert "rsi" in out_df.columns
    assert "sma_240" in out_df.columns
    assert "extra_greeks_delta" in out_df.columns
    assert "ema_240" in out_df.columns
    assert "williams_r_240" in out_df.columns
    assert "keltner_upper_240" in out_df.columns
    assert "adx_240" in out_df.columns
    assert "obv" in out_df.columns
    assert "mfi_240" in out_df.columns
    assert "plus_di_240" in out_df.columns
    assert "minus_di_240" in out_df.columns
    assert "std_returns_240" in out_df.columns
    assert "mom_240" in out_df.columns
    assert "donchian_upper_240" in out_df.columns
    assert "ao" in out_df.columns
    assert "cmf_240" in out_df.columns
    assert "pvt" in out_df.columns

    # Unselected computed columns (like ret_4h) should be filtered out
    assert "ret_4h" not in out_df.columns


def test_options_occ_grouping_and_greeks(tmp_path):
    # 1. Create a dummy options parquet file with multiple OCC symbols
    df = pd.DataFrame(
        {
            "ts_ms": [1700000000000, 1700000000000, 1700000240000, 1700000240000],
            "occ_symbol": [
                "AAPL241220C00150000",
                "AAPL241220C00160000",
                "AAPL241220C00150000",
                "AAPL241220C00160000",
            ],
            "underlying": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "mid": [2.5, 1.2, 2.8, 1.4],
            "delta": [0.65, 0.45, 0.70, 0.48],
            "gamma": [0.02, 0.03, 0.01, 0.02],
        }
    )
    in_path = tmp_path / "options.parquet"
    out_path = tmp_path / "options_features.parquet"
    df.to_parquet(in_path, index=False)

    # 2. Run make_features.py with selected features: delta, sma_480
    cmd = [
        ".venv/bin/python",
        "make_features.py",
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--price-col",
        "mid",
        "--lookbacks",
        "480",
        "--selected-features",
        "delta,sma_480",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"make_features.py failed: {res.stderr}"

    # 3. Read output and verify
    out_df = pd.read_parquet(out_path)
    print("Options Output columns:", list(out_df.columns))

    # Check option metadata columns are preserved
    for col in ["ts_ms", "symbol", "occ_symbol", "underlying", "ref_price", "mid"]:
        assert col in out_df.columns, f"Mandatory option column {col} missing"

    # Check selected features are present
    assert "delta" in out_df.columns
    assert "sma_480" in out_df.columns

    # Check unselected greeks are dropped
    assert "gamma" not in out_df.columns

    # Check that grouping worked (different contract rows did not contaminate each other's calculations)
    # The SMA calculations for each contract should be correct
    # For AAPL241220C00150000: values are 2.5, 2.8 -> sma_480 at step 2 should be (2.5 + 2.8)/2 = 2.65
    # For AAPL241220C00160000: values are 1.2, 1.4 -> sma_480 at step 2 should be (1.2 + 1.4)/2 = 1.3
    row1 = out_df[
        (out_df["occ_symbol"] == "AAPL241220C00150000") & (out_df["ts_ms"] == 1700000240000)
    ].iloc[0]
    row2 = out_df[
        (out_df["occ_symbol"] == "AAPL241220C00160000") & (out_df["ts_ms"] == 1700000240000)
    ].iloc[0]

    assert abs(row1["sma_480"] - 2.65) < 1e-5
    assert abs(row2["sma_480"] - 1.3) < 1e-5

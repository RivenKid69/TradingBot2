# -*- coding: utf-8 -*-
"""
tests/test_options_download_dte.py
Unit tests for Options Downloader DTE filtering and range logic.
"""

import os
import tempfile
import pandas as pd
from datetime import date
from scripts.download_options_data import (
    is_dte_in_range,
    get_simulated_expirations,
    generate_synthetic_options,
)

def test_is_dte_in_range():
    assert is_dte_in_range(5, "0-7") is True
    assert is_dte_in_range(8, "0-7") is False
    assert is_dte_in_range(10, "7-45") is True
    assert is_dte_in_range(6, "7-45") is False
    assert is_dte_in_range(50, "45-90") is True
    assert is_dte_in_range(40, "45-90") is False
    assert is_dte_in_range(100, "all") is True

def test_get_simulated_expirations():
    d = date(2026, 5, 20)
    expirations = get_simulated_expirations(d)
    assert len(expirations) > 0
    for exp in expirations:
        assert exp > d

def test_generate_synthetic_options_filtering():
    with tempfile.TemporaryDirectory() as tmp_dir:
        success = generate_synthetic_options(
            underlying="AAPL",
            start_date="2026-05-20",
            end_date="2026-05-22",
            strike_range="ATM +/- 10",
            include_greeks=True,
            output_dir=tmp_dir,
            dte_range="7-45",
        )
        assert success is True
        parquet_path = os.path.join(tmp_dir, "AAPL_options.parquet")
        assert os.path.exists(parquet_path)
        
        df = pd.read_parquet(parquet_path)
        assert len(df) > 0
        assert df["dte"].min() >= 8
        assert df["dte"].max() <= 45

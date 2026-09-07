#!/usr/bin/env python
"""
Lightweight smoke test that the feature layout produced by ``feature_config.make_layout``
is internally consistent and that ``obs_builder.build_observation_vector`` can fill an
observation array of the expected size.
"""
import numpy as np
import pytest
import feature_config as fc


def _get_block_offsets(layout):
    """Return a map of block name -> starting index plus the total size."""
    offsets = {}
    idx = 0
    for block in layout:
        offsets[block["name"]] = idx
        idx += block["size"]
    return offsets, idx


def test_feature_layout_and_obs_builder():
    """
    Ensure make_layout sets N_FEATURES correctly and obs_builder can populate a vector
    using the resulting layout (if the Cython extension is available).
    """
    # Build layout for a single token and 21 norm columns (legacy default)
    fc.make_layout({"max_num_tokens": 1, "ext_norm_dim": 21})

    offsets, total = _get_block_offsets(fc.FEATURES_LAYOUT)
    assert (
        total == fc.N_FEATURES
    ), f"N_FEATURES mismatch: layout sums to {total}, N_FEATURES={fc.N_FEATURES}"

    # obs_builder is optional; skip the integration part if extension is missing
    obs_builder = pytest.importorskip("obs_builder")

    # Determine external feature dimensions from the layout
    assert "external" in offsets, "Layout must include an 'external' block"
    ext_dim = next(b["size"] for b in fc.FEATURES_LAYOUT if b["name"] == "external")
    ext_start = offsets["external"]

    enable_validity_flags = any(b["name"] == "external_validity" for b in fc.FEATURES_LAYOUT)
    validity_start = offsets.get("external_validity")

    obs = np.zeros(fc.N_FEATURES, dtype=np.float32)
    norm_cols = np.linspace(0.1, 0.1 * ext_dim, ext_dim, dtype=np.float32)
    norm_validity = np.ones(ext_dim, dtype=np.uint8)

    # Call obs_builder with required parameters; values chosen to be simple and finite
    obs_builder.build_observation_vector(
        price=100.0,
        prev_price=99.0,
        log_volume_norm=0.5,
        rel_volume=1.0,
        ma5=100.5,
        ma20=100.2,
        rsi14=50.0,
        macd=0.1,
        macd_signal=0.05,
        momentum=0.2,
        atr=1.5,
        cci=0.0,
        obv=1000.0,
        bb_lower=98.0,
        bb_upper=102.0,
        is_high_importance=0.0,
        time_since_event=0.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=1.0,
        signal_pos=0.0,
        last_vol_imbalance=0.0,
        last_trade_intensity=0.0,
        last_realized_spread=0.0,
        last_agent_fill_ratio=0.0,
        token_id=0,
        max_num_tokens=1,
        num_tokens=1,
        norm_cols_values=norm_cols,
        norm_cols_validity=norm_validity,
        enable_validity_flags=enable_validity_flags,
        out_features=obs,
    )

    # Ensure norm_cols values landed in the expected slice
    ext_slice = obs[ext_start : ext_start + ext_dim]
    assert ext_slice.shape[0] == ext_dim, "External slice has unexpected shape"
    assert np.count_nonzero(ext_slice) > 0, "External slice should contain data"

    if enable_validity_flags:
        assert validity_start is not None, "Validity block offset missing"
        validity_slice = obs[validity_start : validity_start + ext_dim]
        assert np.all(validity_slice == 1.0), "Validity flags should be set to 1.0"

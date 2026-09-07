"""Integration tests for validity flags in observation space.

Tests that validity flags are correctly integrated into the observation vector
and that the observation dimension is correctly computed.
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd


def test_feature_layout_includes_validity_flags():
    """Test that feature layout includes external_validity block with 21 features."""
    from feature_config import make_layout

    layout = make_layout()

    # Find external_validity block
    validity_blocks = [b for b in layout if b["name"] == "external_validity"]
    assert len(validity_blocks) == 1, "Should have exactly one external_validity block"

    validity_block = validity_blocks[0]
    assert (
        validity_block["size"] == 21
    ), "Validity block should have 21 features (one per external feature)"
    assert validity_block["source"] == "external"

    # Verify total features = 84 (63 base + 21 validity)
    total = sum(b["size"] for b in layout)
    assert total == 84, f"Expected 84 total features (63 base + 21 validity), got {total}"


def test_observation_dim_with_validity_flags():
    """Test that observation space dimension is 84 when validity flags enabled."""
    from feature_config import make_layout

    layout = make_layout()

    # Try importing obs_builder, skip if not available
    try:
        import obs_builder

        n_features = obs_builder.compute_n_features(layout)
    except (ImportError, ModuleNotFoundError):
        # Fallback: calculate manually from layout
        n_features = sum(b["size"] for b in layout)

    assert n_features == 84, f"Expected obs_dim=84 (63 base + 21 validity), got {n_features}"


def test_observation_dim_backward_compatibility():
    """Test backward compatibility: can compute obs dim from feature layout."""
    from feature_config import make_layout, EXT_NORM_DIM, MAX_NUM_TOKENS

    # Default layout (with validity flags)
    layout_with_validity = make_layout()
    total_with_validity = sum(b["size"] for b in layout_with_validity)

    # Calculate expected: base features (63) + external_validity (21)
    # Base: 3 bar + 2 ma5 + 2 ma20 + 14 indicators + 2 derived + 6 agent + 3 micro + 2 bb + 5 meta + 21 external + 2 token_meta + 1 token
    expected_base = 3 + 2 + 2 + 14 + 2 + 6 + 3 + 2 + 5 + 21 + 2 + 1  # = 63
    expected_with_validity = expected_base + 21  # = 84

    assert (
        total_with_validity == expected_with_validity
    ), f"Expected {expected_with_validity} features, got {total_with_validity}"


@pytest.mark.skipif(
    not hasattr(__import__("sys").modules.get("obs_builder", None), "build_observation_vector"),
    reason="obs_builder not compiled (requires C++ compiler)",
)
def test_validity_flags_in_observation_vector():
    """Test that validity flags are written to observation vector at correct positions."""
    import obs_builder
    from feature_config import make_layout

    layout = make_layout()
    n_features = obs_builder.compute_n_features(layout)

    # Create observation array
    obs = np.zeros(n_features, dtype=np.float32)

    # Create norm_cols with some NaN values
    norm_cols_values = np.array(
        [
            1.0,
            2.0,
            np.nan,
            4.0,
            5.0,  # First 5: cvd_24h, cvd_7d, yang_zhang_48h (NaN), yang_zhang_7d, garch_200h
            np.nan,
            7.0,
            8.0,
            9.0,
            10.0,  # Next 5: garch_14d (NaN), ret_12h, ret_24h, ret_4h, sma_12000
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,  # Next 5
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,  # Next 5
            21.0,  # Last 1
        ],
        dtype=np.float32,
    )

    # Create validity flags
    norm_cols_validity = np.array(
        [
            True,
            True,
            False,
            True,
            True,  # yang_zhang_48h is invalid
            False,
            True,
            True,
            True,
            True,  # garch_14d is invalid
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
        dtype=np.uint8,
    )

    # Call build_observation_vector with dummy values
    try:
        obs_builder.build_observation_vector(
            1000.0,  # price
            990.0,  # prev_price
            0.5,  # log_volume_norm
            1.2,  # rel_volume
            1000.0,  # ma5
            995.0,  # ma20
            50.0,  # rsi14
            0.1,  # macd
            0.05,  # macd_signal
            0.02,  # momentum
            10.0,  # atr
            0.0,  # cci
            1000.0,  # obv
            980.0,  # bb_lower
            1020.0,  # bb_upper
            0.0,  # is_high_importance
            1.0,  # time_since_event
            50.0,  # fear_greed_value
            True,  # has_fear_greed
            False,  # risk_off_flag
            10000.0,  # cash
            0.0,  # units
            0.0,  # last_vol_imbalance
            0.0,  # last_trade_intensity
            0.0,  # last_realized_spread
            1.0,  # last_agent_fill_ratio
            0,  # token_id
            1,  # max_num_tokens
            1,  # num_tokens
            norm_cols_values,
            norm_cols_validity,
            True,  # enable_validity_flags
            obs,
        )

        # Validity flags should be at positions [63:84] (after external features at [39:60] and token metadata at [60:63])
        validity_start_idx = 63
        validity_end_idx = 84

        validity_flags = obs[validity_start_idx:validity_end_idx]

        # Check that validity flags match input
        # Index 2 (yang_zhang_48h) should be 0.0 (invalid)
        assert (
            validity_flags[2] == 0.0
        ), f"Expected validity_flags[2]=0.0 (yang_zhang_48h invalid), got {validity_flags[2]}"

        # Index 5 (garch_14d) should be 0.0 (invalid)
        assert (
            validity_flags[5] == 0.0
        ), f"Expected validity_flags[5]=0.0 (garch_14d invalid), got {validity_flags[5]}"

        # All other flags should be 1.0 (valid)
        for i in [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            assert (
                validity_flags[i] == 1.0
            ), f"Expected validity_flags[{i}]=1.0 (valid), got {validity_flags[i]}"

        # Check that all validity flags are binary (0.0 or 1.0)
        assert np.all(
            (validity_flags == 0.0) | (validity_flags == 1.0)
        ), f"Validity flags should be binary (0.0 or 1.0), got {validity_flags}"

        # Check that observation is finite (no NaN/Inf)
        assert np.all(np.isfinite(obs)), f"Observation should be all finite, got {obs}"

    except ImportError:
        pytest.skip("obs_builder not available (Cython module not compiled)")


@pytest.mark.skipif(
    not hasattr(__import__("sys").modules.get("obs_builder", None), "build_observation_vector"),
    reason="obs_builder not compiled (requires C++ compiler)",
)
def test_nan_feature_sets_validity_false():
    """Test that NaN feature results in validity=0.0 in observation."""
    import obs_builder
    from feature_config import make_layout

    layout = make_layout()
    n_features = obs_builder.compute_n_features(layout)
    obs = np.zeros(n_features, dtype=np.float32)

    # Create norm_cols with first feature as NaN
    norm_cols_values = np.zeros(21, dtype=np.float32)
    norm_cols_values[0] = np.nan  # cvd_24h is NaN

    norm_cols_validity = np.zeros(21, dtype=np.uint8)
    norm_cols_validity[0] = 0  # cvd_24h is invalid
    norm_cols_validity[1:] = 1  # All others valid

    try:
        obs_builder.build_observation_vector(
            1000.0,
            990.0,
            0.5,
            1.2,
            1000.0,
            995.0,
            50.0,
            0.1,
            0.05,
            0.02,
            10.0,
            0.0,
            1000.0,
            980.0,
            1020.0,
            0.0,
            1.0,
            50.0,
            True,
            False,
            10000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0,
            1,
            1,
            norm_cols_values,
            norm_cols_validity,
            True,
            obs,
        )

        # Validity flag at position 63 should be 0.0 (cvd_24h invalid)
        assert obs[63] == 0.0, f"Expected obs[63]=0.0 (cvd_24h validity flag), got {obs[63]}"

        # All other validity flags should be 1.0
        for i in range(64, 84):
            assert obs[i] == 1.0, f"Expected obs[{i}]=1.0 (valid), got {obs[i]}"

    except ImportError:
        pytest.skip("obs_builder not available (Cython module not compiled)")


@pytest.mark.skipif(
    not hasattr(__import__("sys").modules.get("obs_builder", None), "build_observation_vector"),
    reason="obs_builder not compiled (requires C++ compiler)",
)
def test_valid_feature_sets_validity_true():
    """Test that valid feature results in validity=1.0 in observation."""
    import obs_builder
    from feature_config import make_layout

    layout = make_layout()
    n_features = obs_builder.compute_n_features(layout)
    obs = np.zeros(n_features, dtype=np.float32)

    # Create norm_cols with all valid values
    norm_cols_values = np.arange(21, dtype=np.float32) + 1.0  # [1.0, 2.0, ..., 21.0]
    norm_cols_validity = np.ones(21, dtype=np.uint8)

    try:
        obs_builder.build_observation_vector(
            1000.0,
            990.0,
            0.5,
            1.2,
            1000.0,
            995.0,
            50.0,
            0.1,
            0.05,
            0.02,
            10.0,
            0.0,
            1000.0,
            980.0,
            1020.0,
            0.0,
            1.0,
            50.0,
            True,
            False,
            10000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0,
            1,
            1,
            norm_cols_values,
            norm_cols_validity,
            True,
            obs,
        )

        # All validity flags at positions [63:84] should be 1.0
        validity_flags = obs[63:84]
        assert np.all(
            validity_flags == 1.0
        ), f"Expected all validity flags to be 1.0, got {validity_flags}"

    except ImportError:
        pytest.skip("obs_builder not available (Cython module not compiled)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

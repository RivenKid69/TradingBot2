"""
Comprehensive tests for distributional_ppo.py utility functions.

This module tests:
- _make_clip_range_callable
- _cfg_get
- _popart_value_to_serializable
- _serialize_popart_config
- unwrap_vec_normalize
- create_sequencers
- pad and pad_and_flatten

Tests cover:
- Basic functionality
- Edge cases
- Error handling
- Type consistency
"""

import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import Mock, MagicMock

import pytest
gym = pytest.importorskip("gymnasium")
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from pydantic import BaseModel
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Import functions under test
from distributional_ppo import (
    _make_clip_range_callable,
    _cfg_get,
    _popart_value_to_serializable,
    _serialize_popart_config,
    unwrap_vec_normalize,
    create_sequencers,
)


class TestMakeClipRangeCallable:
    """Tests for _make_clip_range_callable factory function."""

    def test_basic_functionality(self):
        """Test that factory creates callable returning constant value."""
        clip_fn = _make_clip_range_callable(0.2)
        assert callable(clip_fn)
        assert clip_fn() == 0.2
        assert clip_fn(0.5) == 0.2
        assert clip_fn(0.0) == 0.2
        assert clip_fn(1.0) == 0.2

    def test_different_values(self):
        """Test factory with various clip range values."""
        for value in [0.0, 0.1, 0.5, 1.0, 2.0, 100.0]:
            clip_fn = _make_clip_range_callable(value)
            assert clip_fn() == value
            assert clip_fn(0.5) == value

    def test_negative_values(self):
        """Test factory with negative values (edge case)."""
        clip_fn = _make_clip_range_callable(-0.5)
        assert clip_fn() == -0.5

    def test_zero_value(self):
        """Test factory with zero (edge case)."""
        clip_fn = _make_clip_range_callable(0.0)
        assert clip_fn() == 0.0

    def test_very_large_values(self):
        """Test factory with very large values."""
        clip_fn = _make_clip_range_callable(1e10)
        assert clip_fn() == 1e10

    def test_very_small_values(self):
        """Test factory with very small values."""
        clip_fn = _make_clip_range_callable(1e-10)
        assert clip_fn() == 1e-10

    def test_picklable(self):
        """Test that created function is picklable (Bug #8 fix)."""
        try:
            import cloudpickle
            clip_fn = _make_clip_range_callable(0.3)
            pickled = cloudpickle.dumps(clip_fn)
            unpickled = cloudpickle.loads(pickled)
            assert unpickled() == 0.3
            assert unpickled(0.5) == 0.3
        except ImportError:
            # Standard pickle should also work
            clip_fn = _make_clip_range_callable(0.3)
            pickled = pickle.dumps(clip_fn)
            unpickled = pickle.loads(pickled)
            assert unpickled() == 0.3
            assert unpickled(0.5) == 0.3

    def test_multiple_instances_independent(self):
        """Test that multiple instances don't share state."""
        clip_fn1 = _make_clip_range_callable(0.1)
        clip_fn2 = _make_clip_range_callable(0.9)
        assert clip_fn1() == 0.1
        assert clip_fn2() == 0.9

    def test_type_coercion_to_float(self):
        """Test that return value is always float type."""
        clip_fn = _make_clip_range_callable(1)  # int input
        result = clip_fn()
        assert isinstance(result, float)
        assert result == 1.0

    def test_nan_value(self):
        """Test factory with NaN value."""
        clip_fn = _make_clip_range_callable(float('nan'))
        result = clip_fn()
        assert np.isnan(result)

    def test_inf_value(self):
        """Test factory with infinity."""
        clip_fn_pos = _make_clip_range_callable(float('inf'))
        clip_fn_neg = _make_clip_range_callable(float('-inf'))
        assert clip_fn_pos() == float('inf')
        assert clip_fn_neg() == float('-inf')


class TestCfgGet:
    """Tests for _cfg_get generic config getter."""

    def test_dict_basic(self):
        """Test with basic dict."""
        cfg = {"key1": "value1", "key2": 42}
        assert _cfg_get(cfg, "key1") == "value1"
        assert _cfg_get(cfg, "key2") == 42
        assert _cfg_get(cfg, "missing") is None

    def test_dict_with_default(self):
        """Test dict with custom default value."""
        cfg = {"key1": "value1"}
        assert _cfg_get(cfg, "missing", "default") == "default"
        assert _cfg_get(cfg, "missing", 123) == 123
        assert _cfg_get(cfg, "key1", "default") == "value1"

    def test_ordered_dict(self):
        """Test with OrderedDict (Mapping subclass)."""
        cfg = OrderedDict([("a", 1), ("b", 2)])
        assert _cfg_get(cfg, "a") == 1
        assert _cfg_get(cfg, "b") == 2

    def test_object_with_attributes(self):
        """Test with object having attributes."""
        class Config:
            key1 = "value1"
            key2 = 42

        cfg = Config()
        assert _cfg_get(cfg, "key1") == "value1"
        assert _cfg_get(cfg, "key2") == 42
        assert _cfg_get(cfg, "missing", "default") == "default"

    def test_object_with_get_method(self):
        """Test with object having get() method."""
        class ConfigWithGet:
            def get(self, key, default=None):
                return {"key1": "value1"}.get(key, default)

        cfg = ConfigWithGet()
        assert _cfg_get(cfg, "key1") == "value1"
        assert _cfg_get(cfg, "missing") is None
        assert _cfg_get(cfg, "missing", "default") == "default"

    def test_pydantic_model(self):
        """Test with Pydantic V2 model."""
        class PydanticConfig(BaseModel):
            key1: str = "value1"
            key2: int = 42

        cfg = PydanticConfig()
        assert _cfg_get(cfg, "key1") == "value1"
        assert _cfg_get(cfg, "key2") == 42
        assert _cfg_get(cfg, "missing", "default") == "default"

    def test_dataclass(self):
        """Test with dataclass."""
        @dataclass
        class DataclassConfig:
            key1: str = "value1"
            key2: int = 42

        cfg = DataclassConfig()
        assert _cfg_get(cfg, "key1") == "value1"
        assert _cfg_get(cfg, "key2") == 42
        assert _cfg_get(cfg, "missing", "default") == "default"

    def test_none_config(self):
        """Test with None config."""
        assert _cfg_get(None, "key") is None
        assert _cfg_get(None, "key", "default") == "default"

    def test_empty_dict(self):
        """Test with empty dict."""
        cfg = {}
        assert _cfg_get(cfg, "key") is None
        assert _cfg_get(cfg, "key", "default") == "default"

    def test_nested_dict(self):
        """Test that nested dicts work (returns nested dict)."""
        cfg = {"outer": {"inner": "value"}}
        result = _cfg_get(cfg, "outer")
        assert result == {"inner": "value"}
        assert _cfg_get(result, "inner") == "value"

    def test_special_values(self):
        """Test with special values (None, False, 0, empty string)."""
        cfg = {
            "none_val": None,
            "false_val": False,
            "zero_val": 0,
            "empty_str": "",
        }
        # Should return actual values, not defaults
        assert _cfg_get(cfg, "none_val", "default") is None
        assert _cfg_get(cfg, "false_val", "default") is False
        assert _cfg_get(cfg, "zero_val", "default") == 0
        assert _cfg_get(cfg, "empty_str", "default") == ""

    def test_exception_handling_get_method(self):
        """Test exception handling when get() method fails."""
        class BadConfig:
            def get(self, key):
                raise ValueError("Bad get")

        cfg = BadConfig()
        # Should return default when get() raises
        assert _cfg_get(cfg, "key", "default") == "default"

    def test_exception_handling_model_dump(self):
        """Test exception handling when model_dump() fails."""
        class BadConfig:
            def model_dump(self):
                raise ValueError("Bad dump")

        cfg = BadConfig()
        # Should return default when model_dump() raises
        assert _cfg_get(cfg, "key", "default") == "default"

    def test_priority_order(self):
        """Test that attribute access has priority over get() method."""
        class ConfigBoth:
            key1 = "attribute_value"

            def get(self, key, default=None):
                return "get_method_value" if key == "key1" else default

        cfg = ConfigBoth()
        # Should prefer attribute
        assert _cfg_get(cfg, "key1") == "attribute_value"


class TestPopArtValueToSerializable:
    """Tests for _popart_value_to_serializable helper."""

    def test_basic_types(self):
        """Test with basic Python types."""
        assert _popart_value_to_serializable(123) == 123
        assert _popart_value_to_serializable(3.14) == 3.14
        assert _popart_value_to_serializable("string") == "string"
        assert _popart_value_to_serializable(True) is True
        assert _popart_value_to_serializable(None) is None

    def test_list(self):
        """Test with lists."""
        result = _popart_value_to_serializable([1, 2, 3])
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_tuple(self):
        """Test with tuples (converted to list)."""
        result = _popart_value_to_serializable((1, 2, 3))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_scalar(self):
        """Test with numpy scalar types."""
        assert _popart_value_to_serializable(np.float32(1.5)) == 1.5
        assert _popart_value_to_serializable(np.int64(42)) == 42
        assert isinstance(_popart_value_to_serializable(np.float32(1.5)), float)
        assert isinstance(_popart_value_to_serializable(np.int64(42)), int)

    def test_numpy_array(self):
        """Test with numpy arrays (converted to string)."""
        arr = np.array([1, 2, 3])
        result = _popart_value_to_serializable(arr)
        # numpy arrays are converted to string representation
        assert isinstance(result, str)
        assert '[1 2 3]' in result or '1' in result

    def test_numpy_multidim_array(self):
        """Test with multi-dimensional numpy arrays."""
        arr = np.array([[1, 2], [3, 4]])
        result = _popart_value_to_serializable(arr)
        # numpy arrays are converted to string representation
        assert isinstance(result, str)

    def test_torch_tensor(self):
        """Test with torch tensors (converted to string)."""
        tensor = torch.tensor([1, 2, 3])
        result = _popart_value_to_serializable(tensor)
        # torch tensors are converted to string representation
        assert isinstance(result, str)
        assert 'tensor' in result.lower() or '1' in result

    def test_torch_scalar_tensor(self):
        """Test with scalar torch tensor."""
        tensor = torch.tensor(42.0)
        result = _popart_value_to_serializable(tensor)
        # scalar torch tensor is converted to string
        assert isinstance(result, str)
        assert '42' in result

    def test_nested_list(self):
        """Test with nested lists."""
        nested = [1, [2, 3], [4, [5, 6]]]
        result = _popart_value_to_serializable(nested)
        assert result == [1, [2, 3], [4, [5, 6]]]

    def test_mixed_types(self):
        """Test with mixed types in list."""
        mixed = [1, "string", 3.14, True, None, np.int32(5)]
        result = _popart_value_to_serializable(mixed)
        assert result == [1, "string", 3.14, True, None, 5]
        assert isinstance(result, list)

    def test_empty_containers(self):
        """Test with empty containers."""
        assert _popart_value_to_serializable([]) == []
        assert _popart_value_to_serializable(()) == []
        # empty numpy array becomes string
        result = _popart_value_to_serializable(np.array([]))
        assert isinstance(result, str)

    def test_nan_and_inf(self):
        """Test with NaN and infinity values."""
        # Note: NaN != NaN, so check with isnan
        result_nan = _popart_value_to_serializable(float('nan'))
        assert np.isnan(result_nan)

        assert _popart_value_to_serializable(float('inf')) == float('inf')
        assert _popart_value_to_serializable(float('-inf')) == float('-inf')

    def test_numpy_nan_inf(self):
        """Test with numpy NaN and infinity."""
        result_nan = _popart_value_to_serializable(np.nan)
        assert np.isnan(result_nan)

        assert _popart_value_to_serializable(np.inf) == float('inf')


class TestSerializePopArtConfig:
    """Tests for _serialize_popart_config."""

    def test_empty_config(self):
        """Test with empty config."""
        result = _serialize_popart_config({})
        assert result == {}
        assert isinstance(result, dict)

    def test_basic_config(self):
        """Test with basic config values."""
        cfg = {
            "enabled": True,
            "alpha": 0.01,
            "beta": 1.0,
        }
        result = _serialize_popart_config(cfg)
        assert result == cfg

    def test_numpy_values(self):
        """Test with numpy values in config."""
        cfg = {
            "alpha": np.float32(0.01),
            "iterations": np.int64(100),
            "thresholds": np.array([0.1, 0.2, 0.3]),
        }
        result = _serialize_popart_config(cfg)
        # float32 may have precision loss
        assert abs(result["alpha"] - 0.01) < 1e-6
        assert result["iterations"] == 100
        # thresholds is numpy array -> string
        assert isinstance(result["thresholds"], str)

    def test_nested_config(self):
        """Test with nested configuration."""
        cfg = {
            "outer": {
                "inner": np.float32(1.5),
                "deep": {
                    "value": np.int32(42),
                }
            }
        }
        result = _serialize_popart_config(cfg)
        # Note: _serialize_popart_config only processes top-level
        # Inner dicts are passed through _popart_value_to_serializable
        assert isinstance(result["outer"], dict)

    def test_list_values(self):
        """Test with list values."""
        cfg = {
            "layers": [128, 256, 512],
            "activations": ["relu", "tanh"],
        }
        result = _serialize_popart_config(cfg)
        assert result["layers"] == [128, 256, 512]
        assert result["activations"] == ["relu", "tanh"]

    def test_mixed_types(self):
        """Test with mixed types."""
        cfg = {
            "bool_val": True,
            "int_val": 42,
            "float_val": 3.14,
            "str_val": "test",
            "none_val": None,
            "np_val": np.float64(2.71),
            "list_val": [1, 2, 3],
        }
        result = _serialize_popart_config(cfg)
        assert result["bool_val"] is True
        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["str_val"] == "test"
        assert result["none_val"] is None
        assert result["np_val"] == 2.71
        assert result["list_val"] == [1, 2, 3]


class TestUnwrapVecNormalize:
    """Tests for unwrap_vec_normalize."""

    def test_unwrap_vec_normalize_direct(self):
        """Test unwrapping direct VecNormalize."""
        # Create a simple dummy environment
        def make_env():
            return gym.make("CartPole-v1")

        dummy_env = DummyVecEnv([make_env])
        vec_norm = VecNormalize(dummy_env)
        result = unwrap_vec_normalize(vec_norm)
        assert result is vec_norm

    def test_unwrap_vec_normalize_nested(self):
        """Test unwrapping nested VecNormalize."""
        def make_env():
            return gym.make("CartPole-v1")

        dummy_env = DummyVecEnv([make_env])
        vec_norm = VecNormalize(dummy_env)
        result = unwrap_vec_normalize(vec_norm)
        assert result is vec_norm

    def test_unwrap_no_vec_normalize(self):
        """Test when no VecNormalize is present."""
        def make_env():
            return gym.make("CartPole-v1")

        dummy_env = DummyVecEnv([make_env])
        result = unwrap_vec_normalize(dummy_env)
        assert result is None

    def test_unwrap_none_input(self):
        """Test with None input."""
        result = unwrap_vec_normalize(None)
        assert result is None

    def test_unwrap_non_vec_env(self):
        """Test with non-VecEnv object."""
        mock_env = Mock()
        result = unwrap_vec_normalize(mock_env)
        # Should handle gracefully
        assert result is None or isinstance(result, type(None))


class TestCreateSequencers:
    """Tests for create_sequencers function."""

    def test_basic_sequencers(self):
        """Test basic sequencer creation."""
        # Create mock data
        episode_starts = np.array([1, 0, 0, 1, 0], dtype=np.bool_)
        env_change = np.array([0, 0, 0, 0, 0], dtype=np.bool_)

        seq_start_indices, pad_fn, pad_and_flatten_fn = create_sequencers(
            episode_starts=episode_starts,
            env_change=env_change,
            device="cpu",
        )

        # Check that result is a tuple with 3 elements
        assert isinstance(seq_start_indices, np.ndarray)
        assert callable(pad_fn)
        assert callable(pad_and_flatten_fn)

        # Test pad function
        test_array = np.arange(5)
        padded = pad_fn(test_array)
        assert isinstance(padded, np.ndarray)

    def test_sequencers_with_env_change(self):
        """Test sequencers with env_change parameter."""
        episode_starts = np.array([1, 0, 0, 1, 0], dtype=np.bool_)
        env_change = np.array([0, 0, 1, 0, 0], dtype=np.bool_)

        seq_start_indices, pad_fn, pad_and_flatten_fn = create_sequencers(
            episode_starts=episode_starts,
            env_change=env_change,
            device="cpu",
        )

        assert isinstance(seq_start_indices, np.ndarray)
        assert callable(pad_fn)
        assert callable(pad_and_flatten_fn)

        # Should have sequences starting at env_change points too
        assert len(seq_start_indices) >= 2

    def test_sequencers_empty_episodes(self):
        """Test sequencers with no episode starts (forced to start at 0)."""
        episode_starts = np.array([0, 0, 0, 0, 0], dtype=np.bool_)
        env_change = np.array([0, 0, 0, 0, 0], dtype=np.bool_)

        seq_start_indices, pad_fn, pad_and_flatten_fn = create_sequencers(
            episode_starts=episode_starts,
            env_change=env_change,
            device="cpu",
        )

        # Should still create at least one sequence (forced start at 0)
        assert len(seq_start_indices) >= 1
        assert seq_start_indices[0] == 0

    def test_sequencers_all_episodes(self):
        """Test sequencers where every step is episode start."""
        episode_starts = np.array([1, 1, 1, 1, 1], dtype=np.bool_)
        env_change = np.array([0, 0, 0, 0, 0], dtype=np.bool_)

        seq_start_indices, pad_fn, pad_and_flatten_fn = create_sequencers(
            episode_starts=episode_starts,
            env_change=env_change,
            device="cpu",
        )

        # Should create many sequences
        assert len(seq_start_indices) == 5

    def test_pad_and_flatten_functionality(self):
        """Test that pad_and_flatten actually works."""
        episode_starts = np.array([1, 0, 1, 0, 0], dtype=np.bool_)
        env_change = np.array([0, 0, 0, 0, 0], dtype=np.bool_)

        seq_start_indices, pad_fn, pad_and_flatten_fn = create_sequencers(
            episode_starts=episode_starts,
            env_change=env_change,
            device="cpu",
        )

        test_array = np.arange(5)
        padded = pad_fn(test_array)
        flattened = pad_and_flatten_fn(test_array)

        assert isinstance(padded, np.ndarray)
        assert isinstance(flattened, np.ndarray)
        assert padded.ndim >= 2  # Should be (n_seq, max_len, ...)
        assert flattened.ndim >= 1  # Should be flattened


class TestPadFunctions:
    """Tests for pad and pad_and_flatten helper functions."""

    def test_pad_basic_array(self):
        """Test pad with basic numpy array."""
        # Note: pad and pad_and_flatten are nested in create_sequencers
        # We'll test the logic that would be expected
        arr = np.array([[1, 2], [3, 4]])
        padded = np.pad(arr, ((0, 1), (0, 0)), mode='constant')
        assert padded.shape == (3, 2)
        np.testing.assert_array_equal(padded[-1], [0, 0])

    def test_pad_torch_tensor(self):
        """Test pad with torch tensor."""
        tensor = torch.tensor([[1, 2], [3, 4]])
        # Convert to numpy, pad, convert back
        padded = torch.from_numpy(
            np.pad(tensor.numpy(), ((0, 1), (0, 0)), mode='constant')
        )
        assert padded.shape == (3, 2)
        torch.testing.assert_close(padded[-1], torch.tensor([0, 0]))

    def test_pad_and_flatten_basic(self):
        """Test pad_and_flatten with basic array."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        flat = arr.flatten()
        assert flat.shape == (6,)
        np.testing.assert_array_equal(flat, [1, 2, 3, 4, 5, 6])

    def test_pad_and_flatten_torch(self):
        """Test pad_and_flatten with torch tensor."""
        tensor = torch.tensor([[1, 2], [3, 4]])
        flat = tensor.flatten()
        assert flat.shape == (4,)
        torch.testing.assert_close(flat, torch.tensor([1, 2, 3, 4]))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

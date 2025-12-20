"""
Tests for import fallback paths in distributional_ppo.py.

These tests require careful handling of sys.modules to simulate
missing dependencies.
"""
from __future__ import annotations

import importlib
import sys
import warnings
from unittest import mock

import pytest


class TestImportFallbackPaths:
    """Test import fallback paths."""

    def test_module_imports_with_sb3_contrib(self):
        """Verify module works with sb3_contrib available."""
        import distributional_ppo

        # sb3_contrib should be the backend
        assert distributional_ppo._RECURRENT_BACKEND == "sb3_contrib"

    def test_distributional_policy_alias_exists(self):
        """Verify DistributionalPolicy alias exists."""
        import distributional_ppo

        assert "DistributionalPolicy" in distributional_ppo._DISTRIBUTIONAL_POLICY_ALIASES

    def test_module_level_constants(self):
        """Verify module level constants are set."""
        import distributional_ppo

        assert hasattr(distributional_ppo, "DEFAULT_CLIP_RANGE_VF")

    def test_popart_classes_available(self):
        """Verify PopArt classes are available."""
        from distributional_ppo import (
            PopArtController,
            PopArtHoldoutBatch,
            PopArtHoldoutEvaluation,
            PopArtCandidateMetrics,
        )

        assert PopArtController is not None
        assert PopArtHoldoutBatch is not None
        assert PopArtHoldoutEvaluation is not None
        assert PopArtCandidateMetrics is not None

    def test_safe_explained_variance_available(self):
        """Verify safe_explained_variance is available."""
        from distributional_ppo import safe_explained_variance

        import numpy as np

        result = safe_explained_variance(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 2.9]),
        )
        assert result > 0


class TestPatchRandForTests:
    """Test _patch_rand_for_tests function."""

    def test_patch_applied_in_test_env(self):
        """Verify patch is applied in test environment."""
        import torch

        # In test environment, torch.rand should be patched
        assert hasattr(torch, "_distributional_rand_patch")


class TestModuleHelperFunctions:
    """Test module-level helper functions."""

    def test_safe_ev_with_many_samples(self):
        """Test safe_explained_variance with many samples."""
        from distributional_ppo import safe_explained_variance

        import numpy as np

        y_pred = np.random.randn(100)
        y_true = y_pred + np.random.randn(100) * 0.1

        result = safe_explained_variance(y_pred, y_true)
        assert result > 0.5  # Should explain most variance

    def test_safe_ev_large_residual(self):
        """Test safe_explained_variance with large residuals."""
        from distributional_ppo import safe_explained_variance

        import numpy as np

        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([10.0, 20.0, 30.0])

        result = safe_explained_variance(y_pred, y_true)
        # Large residuals = low EV
        assert result < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

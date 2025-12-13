from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Project root is the parent of the tests/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS = PROJECT_ROOT / "tests"

# Load stdlib logging before project paths are added
sys.path = [p for p in sys.path if p not in {str(TESTS)}]
import logging  # noqa: F401

# =============================================================================
# Optional dependency detection
# =============================================================================

def _check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# Cache dependency availability checks
TORCH_AVAILABLE = _check_import("torch")
GYMNASIUM_AVAILABLE = _check_import("gymnasium")
SB3_AVAILABLE = _check_import("stable_baselines3")
PYARROW_AVAILABLE = _check_import("pyarrow")
HYPOTHESIS_AVAILABLE = _check_import("hypothesis")
SORTEDCONTAINERS_AVAILABLE = _check_import("sortedcontainers")
CLOUDPICKLE_AVAILABLE = _check_import("cloudpickle")
OPTUNA_AVAILABLE = _check_import("optuna")


# =============================================================================
# Pytest hooks for automatic test skipping
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests that require unavailable optional dependencies.

    This hook examines test file paths and module contents to determine
    which tests should be skipped based on missing dependencies.
    """
    skip_torch = pytest.mark.skip(reason="PyTorch not installed (install with: pip install torch)")
    skip_gymnasium = pytest.mark.skip(reason="gymnasium not installed (install with: pip install gymnasium)")
    skip_sb3 = pytest.mark.skip(reason="stable-baselines3 not installed (install with: pip install stable-baselines3)")
    skip_pyarrow = pytest.mark.skip(reason="pyarrow not installed (install with: pip install pyarrow)")
    skip_sortedcontainers = pytest.mark.skip(reason="sortedcontainers not installed (install with: pip install sortedcontainers)")
    skip_cloudpickle = pytest.mark.skip(reason="cloudpickle not installed (install with: pip install cloudpickle)")
    skip_optuna = pytest.mark.skip(reason="optuna not installed (install with: pip install optuna)")
    skip_hypothesis = pytest.mark.skip(reason="hypothesis not installed (install with: pip install hypothesis)")

    # Patterns indicating torch dependency
    torch_patterns = [
        "test_ppo", "test_twin_critics", "test_categorical", "test_vgs",
        "test_upgd", "test_gradient", "test_quantile", "test_popart",
        "test_lstm", "test_pbt", "test_distributional", "test_numerical",
        "test_shared_memory", "test_vf_clip", "test_vf_variance",
        "test_gae", "test_kl_direction", "test_return_scale",
        "test_state_perturbation", "test_torch", "test_ev_",
        "test_bug_fixes_2025", "test_bug8", "test_bug10",
        "test_advantage_normalization", "test_adaptive_upgd",
        "test_actual_ppo", "test_four_problems", "test_potential_issues",
        "test_unit_custom_policy", "test_unit_train_model",
    ]

    # Patterns indicating gymnasium dependency
    gymnasium_patterns = [
        "test_bug7_grouped_ev", "test_bug_fixes_final_audit",
        "test_correct_api_usage", "test_forex_improvements",
        "test_forex_training", "test_futures_training",
        "test_timing_profiles",
    ]

    # Patterns indicating LOB/sortedcontainers dependency
    lob_patterns = [
        "test_lob", "test_l3", "test_matching_engine",
        "test_hidden_liquidity", "test_queue_tracker",
        "test_cme_l3", "test_cme_risk", "test_cme_settlement",
        "test_execution_providers_l3", "test_market_impact",
        "test_fill_probability",
    ]

    # Patterns indicating stable-baselines3 dependency
    sb3_patterns = [
        "test_shared_memory_vec_env",
    ]

    for item in items:
        test_path = str(item.fspath)
        test_name = item.name

        # Check markers first
        if "requires_torch" in [m.name for m in item.iter_markers()]:
            if not TORCH_AVAILABLE:
                item.add_marker(skip_torch)
                continue

        if "requires_gymnasium" in [m.name for m in item.iter_markers()]:
            if not GYMNASIUM_AVAILABLE:
                item.add_marker(skip_gymnasium)
                continue

        if "requires_sb3" in [m.name for m in item.iter_markers()]:
            if not SB3_AVAILABLE:
                item.add_marker(skip_sb3)
                continue

        if "requires_pyarrow" in [m.name for m in item.iter_markers()]:
            if not PYARROW_AVAILABLE:
                item.add_marker(skip_pyarrow)
                continue

        # Pattern-based detection
        if not TORCH_AVAILABLE:
            for pattern in torch_patterns:
                if pattern in test_path.lower():
                    item.add_marker(skip_torch)
                    break

        if not GYMNASIUM_AVAILABLE:
            for pattern in gymnasium_patterns:
                if pattern in test_path.lower():
                    item.add_marker(skip_gymnasium)
                    break

        if not SORTEDCONTAINERS_AVAILABLE:
            for pattern in lob_patterns:
                if pattern in test_path.lower():
                    item.add_marker(skip_sortedcontainers)
                    break

        if not SB3_AVAILABLE:
            for pattern in sb3_patterns:
                if pattern in test_path.lower():
                    item.add_marker(skip_sb3)
                    break


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_torch: mark test as requiring PyTorch"
    )
    config.addinivalue_line(
        "markers", "requires_gymnasium: mark test as requiring gymnasium"
    )
    config.addinivalue_line(
        "markers", "requires_sb3: mark test as requiring stable-baselines3"
    )
    config.addinivalue_line(
        "markers", "requires_pyarrow: mark test as requiring pyarrow"
    )


# =============================================================================
# Fixtures for optional dependencies
# =============================================================================

@pytest.fixture
def requires_torch():
    """Skip test if torch is not available."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")


@pytest.fixture
def requires_gymnasium():
    """Skip test if gymnasium is not available."""
    if not GYMNASIUM_AVAILABLE:
        pytest.skip("gymnasium not installed")


@pytest.fixture
def requires_sb3():
    """Skip test if stable-baselines3 is not available."""
    if not SB3_AVAILABLE:
        pytest.skip("stable-baselines3 not installed")


@pytest.fixture
def requires_pyarrow():
    """Skip test if pyarrow is not available."""
    if not PYARROW_AVAILABLE:
        pytest.skip("pyarrow not installed")

_requests_stub = types.ModuleType("requests")


def _unavailable(*args, **kwargs):  # pragma: no cover - network calls disabled in tests
    raise RuntimeError("requests module is not available in the test environment")


_requests_stub.get = _unavailable
_requests_stub.post = _unavailable
_requests_stub.put = _unavailable
_requests_stub.delete = _unavailable
_requests_stub.request = _unavailable


class _MockSession:
    """Mock Session class for testing."""

    def __init__(self):
        self.headers = {}

    def get(self, *args, **kwargs):
        raise RuntimeError("requests.Session.get is not available in the test environment")

    def post(self, *args, **kwargs):
        raise RuntimeError("requests.Session.post is not available in the test environment")

    def close(self):
        pass


_requests_stub.Session = _MockSession

# Create stub exceptions module for testing
_requests_exceptions_stub = types.ModuleType("requests.exceptions")


class RequestException(Exception):
    """Base exception for requests."""
    pass


class HTTPError(RequestException):
    """HTTP error occurred."""
    pass


class ConnectionError(RequestException):
    """Connection error occurred."""
    pass


class Timeout(RequestException):
    """Request timed out."""
    pass


class TooManyRedirects(RequestException):
    """Too many redirects."""
    pass


# Add exception classes to both modules
_requests_exceptions_stub.RequestException = RequestException
_requests_exceptions_stub.HTTPError = HTTPError
_requests_exceptions_stub.ConnectionError = ConnectionError
_requests_exceptions_stub.Timeout = Timeout
_requests_exceptions_stub.TooManyRedirects = TooManyRedirects

_requests_stub.exceptions = _requests_exceptions_stub
_requests_stub.RequestException = RequestException
_requests_stub.HTTPError = HTTPError
_requests_stub.ConnectionError = ConnectionError
_requests_stub.Timeout = Timeout

sys.modules.setdefault("requests", _requests_stub)
sys.modules.setdefault("requests.exceptions", _requests_exceptions_stub)

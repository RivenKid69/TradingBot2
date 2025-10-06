from __future__ import annotations

import sys
import types
from pathlib import Path

# Project root contains this file; tests may live alongside sources or under tests/
PROJECT_ROOT = Path(__file__).resolve().parent
TESTS = PROJECT_ROOT / "tests"

# Load stdlib logging before project paths are added
sys.path = [p for p in sys.path if p not in {str(PROJECT_ROOT), str(TESTS)}]
import logging  # noqa: F401
sys.path.extend([str(PROJECT_ROOT), str(TESTS)])

_requests_stub = types.ModuleType("requests")


def _unavailable(*args, **kwargs):  # pragma: no cover - network calls disabled in tests
    raise RuntimeError("requests module is not available in the test environment")


_requests_stub.get = _unavailable
_requests_stub.post = _unavailable
_requests_stub.put = _unavailable
_requests_stub.delete = _unavailable
_requests_stub.request = _unavailable


sys.modules.setdefault("requests", _requests_stub)

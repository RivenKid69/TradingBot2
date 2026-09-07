# -*- coding: utf-8 -*-
"""
Pytest configuration for CCEA integration tests.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Return project root path."""
    return PROJECT_ROOT


@pytest.fixture
def sample_heartbeat():
    """Sample valid heartbeat message."""
    from datetime import datetime

    return {
        "message_type": "HEARTBEAT",
        "agent_id": "agent_test123456789012",
        "timestamp": datetime.utcnow().isoformat(),
        "state": {
            "deployment_state": "RUNNING",
            "run_state": "RUNNING",
        },
    }


@pytest.fixture
def sample_command():
    """Sample valid command message."""
    from datetime import datetime

    return {
        "command_type": "REQUEST_START_RUN",
        "idempotency_key": "key_" + "a" * 20,
        "timestamp": datetime.utcnow().isoformat(),
        "deployment_id": "deploy_test",
        "artifact_digest": "sha256:" + "a" * 64,
    }

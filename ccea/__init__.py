# -*- coding: utf-8 -*-
"""
CCEA (Cloud-Controlled Execution Architecture) Module.

This module provides:
- Guardrails for CI/CD enforcement
- Schema validation utilities
- Import boundary checking
- Protocol validation

Key Principle (non-negotiable):
    Cloud = research/build/monitoring/control plane (lifecycle requests)
    Agent = secrets + live loop + risk enforce + order creation/sending

Cloud NEVER:
    - Stores broker API keys
    - Generates or sends orders
    - Has access to trading endpoints
"""

__version__ = "1.0.0"
__author__ = "CCEA Architecture Team"

from typing import Final

# Schema version for this release
SCHEMA_VERSION: Final[str] = "1.0.0"

# Minimum supported schema version
MIN_SUPPORTED_SCHEMA_VERSION: Final[str] = "1.0.0"

# Maximum supported schema version
MAX_SUPPORTED_SCHEMA_VERSION: Final[str] = "1.99.99"

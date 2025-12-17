# -*- coding: utf-8 -*-
"""
CCEA Shared Package.

Contains safe, zone-independent components:
- Core models and contracts (OrderIntent, strategy interfaces)
- Common utilities (time, math, validation)
- Data-only adapters (market data, fees, trading hours, exchange info)

This package can be imported by BOTH Cloud and Agent zones.
It contains NO live execution code, NO order submission, NO private trading APIs.

Security Design Commitments:
- No broker API key handling
- No order creation or submission
- No private trading endpoints
- Safe for cloud deployment
"""

from typing import Final, List

__version__ = "2.0.0"

# Zone identifier
ZONE: Final[str] = "shared"

# Modules that belong to this zone (for import validation)
ZONE_MODULES: Final[List[str]] = [
    "packages.shared.*",
    "core_*",
    "impl_*",
    "execution_sim",
    "data_loader_*",
    "features_pipeline",
    "transformers",
]

# Modules that MUST NOT be imported by this zone
PROHIBITED_IMPORTS: Final[List[str]] = [
    "packages.agent.*",
    "packages.cloud.internal.*",
    "adapters.*.order_execution",
    "adapters.*.options_execution",
    "execution_providers",
    "service_signal_runner",
]

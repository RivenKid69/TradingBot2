# -*- coding: utf-8 -*-
"""
CCEA Protocol Package.

Cloud-Agent protocol handling including schema versioning and negotiation.
"""

from ccea.protocol.schema_versioning import (
    SchemaVersion,
    VersionNegotiationRequest,
    VersionNegotiationResult,
    SchemaVersionNegotiator,
    NegotiationStatus,
    VersionCompatibility,
    check_version_compatibility,
    negotiate_version,
)

__all__ = [
    "SchemaVersion",
    "VersionNegotiationRequest",
    "VersionNegotiationResult",
    "SchemaVersionNegotiator",
    "NegotiationStatus",
    "VersionCompatibility",
    "check_version_compatibility",
    "negotiate_version",
]

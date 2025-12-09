"""
GDPR compliance services.

Provides implementations for:
- Article 17: Right to Erasure
- Article 20: Right to Data Portability

References:
    - GDPR Regulation (EU) 2016/679
    - EDPB Guidelines on the Right to Erasure
    - EDPB Guidelines on Data Portability (WP242)
"""

from services.gdpr.data_deletion import (
    DataCategory,
    DeletionStatus,
    DeletionRequest,
    GDPRDeletionService,
    DeletionError,
)
from services.gdpr.data_export import (
    PortableDataPackage,
    GDPRExportService,
    ExportError,
)

__all__ = [
    # Deletion
    "DataCategory",
    "DeletionStatus",
    "DeletionRequest",
    "GDPRDeletionService",
    "DeletionError",
    # Export
    "PortableDataPackage",
    "GDPRExportService",
    "ExportError",
]

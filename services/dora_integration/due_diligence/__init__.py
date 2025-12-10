# -*- coding: utf-8 -*-
"""
Due Diligence & Audit Readiness Module.

Provides interfaces for:
- Client audit requests (Art. 30(3)(e))
- Provider information packages for ROI (Art. 28(3))
- Pooled audit coordination (Art. 30(4))
- Compliance status dashboard

DORA Context:
    Financial entities have the right to audit their ICT providers.
    We facilitate this by maintaining audit readiness and providing
    structured information packages.

Modules (to be migrated in Phase 1):
    - audit_readiness.py: Audit support and evidence management
    - provider_info_package.py: ROI data generation for clients
    - pooled_audit_support.py: Multi-client audit coordination
    - compliance_dashboard.py: Real-time compliance status

Target Exports (Phase 1):
    - DORAuditReadiness: Main audit support class
    - AuditRequest, EvidenceItem: Data structures
    - ProviderIdentification, ICTServiceType: ROI structures
    - PooledAuditSupport, PooledAuditEngagement: Pooled audit
    - DORAComplianceDashboard: Status monitoring

References:
    - DORA Article 30(3)(d): Financial entity audit rights
    - DORA Article 30(3)(e): NCA access and inspection rights
    - DORA Article 30(4): Pooled audit arrangements
    - CIR 2024/2956: ITS on Register of Information templates

Migration Status: Phase 0 - Structure only, awaiting Phase 1 migration
"""

from __future__ import annotations

__all__: list[str] = []  # Will be populated in Phase 1

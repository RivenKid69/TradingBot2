# -*- coding: utf-8 -*-
"""
Unified Reporting Layer.

Provides:
- Cross-regulatory report generation
- ITS-compliant reporting templates
- ROI (Register of Information) data generation

CRITICAL DISTINCTION:
    We generate DATA for client reports.
    We do NOT maintain client registers.
    We do NOT submit to NCAs.

ROI Data Generation:
    Clients maintain their Register of Information per Art. 28(3).
    We provide them with structured data packages to populate their ROI:
    - B_02.01: Provider identification
    - B_03.01: Service types
    - B_05.01/02: Function identification
    - B_06.01: ICT service assessment
    - B_99.01: Subcontractor chains

Modules (to be migrated in Phase 5):
    - unified_reporting.py: Cross-regulatory report aggregation
    - reporting_templates.py: ITS template generators
    - register_of_information.py: ROI data package generator

Target Exports (Phase 5):
    - UnifiedReportingManager: Report aggregation
    - DORAReportingTemplates: Template library
    - DORARegisterOfInformation: ROI data generator (NOT ROI maintainer)

References:
    - DORA Article 28(3): Register of information requirement
    - CIR 2024/2956: ITS on Register of Information templates
    - CDR 2025/301: RTS on incident reporting content and templates

Migration Status: Phase 0 - Structure only, awaiting Phase 5 migration
"""

from __future__ import annotations

__all__: list[str] = []  # Will be populated in Phase 5

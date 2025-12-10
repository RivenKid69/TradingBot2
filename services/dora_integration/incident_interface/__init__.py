# -*- coding: utf-8 -*-
"""
Incident Communication Interface.

Provides:
- Client incident notifications (Art. 30(2)(d))
- Incident classification (CDR 2024/1772)
- Incident data export for client NCA reporting
- Cyber threat notifications

CRITICAL DISTINCTION:
    We notify CLIENTS. Clients report to NCAs.
    We are ICT providers, NOT financial entities.
    We do NOT submit directly to NCAs (that's client's obligation).

Flow:
    1. DORAIncidentClassification.classify_incident()
    2. ClientNotificationService.notify_client()  # We notify client
    3. DORAIncidentReporter.generate_client_data_package()  # Client gets data
    4. Client submits to their NCA using our data package

Modules (to be migrated in Phase 2):
    - client_incident_notification.py: Client notification service
    - incident_classification.py: CDR 2024/1772 classification
    - incident_reporting.py: Export-only reporting templates
    - cyber_threat_notification.py: Art. 19(4) threat notifications
    - communication.py: Art. 14 crisis communication channels

Target Exports (Phase 2):
    - ClientNotificationService: Main notification service
    - IncidentSeverity, NotificationStatus: Enums
    - DORAIncidentClassification: Classification engine
    - DORAIncidentReporter: Report template generator
    - CyberThreatNotificationService: Threat alerts
    - DORACommunication: Communication policy management

References:
    - DORA Article 30(2)(d): Incident notification obligations
    - CDR 2024/1772: RTS on Incident Classification
    - DORA Article 19: Major incident reporting
    - DORA Article 14: Crisis communication

Migration Status: Phase 0 - Structure only, awaiting Phase 2 migration
"""

from __future__ import annotations

__all__: list[str] = []  # Will be populated in Phase 2

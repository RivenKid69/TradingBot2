# Data Processing Agreement

**Version:** 2.0.0
**Effective Date:** [DATE]
**Last Updated:** December 2024

---

## Parties

This Data Processing Agreement ("DPA") is entered into between:

**Controller:** [CLIENT COMPANY NAME]
Address: [CLIENT ADDRESS]
Contact: [CLIENT CONTACT EMAIL]
("Controller" or "Client")

**Processor:** [YOUR COMPANY NAME]
Address: [YOUR ADDRESS]
Contact: [YOUR CONTACT EMAIL]
Data Protection Officer: [DPO EMAIL]
("Processor" or "Provider")

Together referred to as the "Parties" and individually as a "Party".

---

## 1. Definitions

**1.1** "Personal Data" means any information relating to an identified or identifiable natural person as defined in GDPR Article 4(1).

**1.2** "Processing" means any operation performed on Personal Data, as defined in GDPR Article 4(2).

**1.3** "Data Subject" means an identified or identifiable natural person whose Personal Data is processed.

**1.4** "Sub-processor" means any third party engaged by the Processor to process Personal Data on behalf of the Controller.

**1.5** "GDPR" means Regulation (EU) 2016/679 of the European Parliament and of the Council.

**1.6** "Services" means the B2B research and deployment platform services provided by the Processor under the main service agreement.

---

## 2. Subject Matter and Duration

**2.1 Subject Matter**

This DPA governs the processing of Personal Data by the Processor on behalf of the Controller in connection with the provision of the Services.

**2.2 Duration**

This DPA shall remain in effect for the duration of the service agreement between the Parties, plus any period necessary for the Processor to complete deletion of Personal Data as required herein.

**2.3 Nature of Processing**

The Processor shall process Personal Data to provide:
- Research, simulation/backtesting, and artifact lifecycle services (non-orders)
- Monitoring and reliability telemetry (redacted/bucketed per configured sensitivity level)
- Analytics and operational reporting (non-performance and non-advice)
- Platform access, organization/workspace management, and support operations

**CCEA boundary note (execution and credentials):** Live order execution and broker/exchange credentials are handled in the Controller’s environment by the customer-controlled Agent. The Processor’s Cloud infrastructure does not store broker credentials and does not connect to broker/exchange trading APIs.

---

## 3. Types of Personal Data

The following categories of Personal Data may be processed:

| Category | Data Elements | Purpose |
|----------|---------------|---------|
| Account Data | Email, name, password hash | User authentication and account management |
| Research/Strategy Data | Strategies, backtests, redacted telemetry | Service provision (Cloud zone only) |
| Technical Data | IP addresses, device information, logs | Security and troubleshooting |

**IMPORTANT - CCEA Architecture Note:**

Under the Cloud-Controlled Execution Architecture (CCEA), credential handling occurs **exclusively** in the customer-managed Agent zone:

| Data Type | Cloud Zone | Agent Zone (Customer-Managed) |
|-----------|-----------|------------------------------|
| Broker API credentials | **Not stored or processed** | Encrypted local vault |
| Order execution data | **Not accessed** | Local processing only |
| Trading positions | **Aggregated/redacted only** | Full data locally |

The Processor's Cloud infrastructure:
- **Does not** store, process, or have access to broker API keys
- **Does not** execute orders or connect to broker/exchange APIs
- Only receives redacted telemetry from customer-operated Agents (if deployed)

### 3.1 Telemetry Sensitivity Levels

The Platform implements three telemetry sensitivity levels. Controller selects the level per workspace:

| Level | Data Processed | Controller Tier | Opt-in Required | Retention |
|-------|----------------|-----------------|-----------------|-----------|
| **AGGREGATED** | PnL %, drawdown, error rates, health status | All | No (default) | 90 days |
| **DETAILED_NON_SENSITIVE** | + timestamps, state transitions, queue depths | All | Yes | 30 days |
| **RAW_ORDER_EVENTS** | + order events (masked), fill events, positions | Enterprise only | Yes (explicit) | 7-30 days |

**RAW_ORDER_EVENTS Processing Requirements:**

If Controller enables `RAW_ORDER_EVENTS`:
- Controller must be on Enterprise tier
- Controller must provide explicit per-workspace opt-in
- Consent record must be created (who, what, when, scope, expiry)
- Processing is audited and access-restricted
- Alternative: Controller may select "telemetry stays local" mode (no Cloud transmission)

**Data Never Processed (at any level):**
- Broker API keys, secrets, credentials
- Environment variables
- Unmasked account identifiers
- Order-like payloads in commands (side, quantity, price)

---

## 4. Categories of Data Subjects

Personal Data may relate to the following categories of Data Subjects:
- Controller's employees and authorized users
- Controller's end users (if applicable)
- Controller's contractors with platform access

---

## 5. Processor Obligations

Pursuant to GDPR Article 28(3), the Processor shall:

**5.1 Instructions**
- Process Personal Data only on documented instructions from the Controller
- Immediately inform the Controller if any instruction infringes GDPR or other applicable law

**5.2 Confidentiality**
- Ensure that persons authorized to process Personal Data have committed to confidentiality or are under statutory confidentiality obligations

**5.3 Security Measures**
- Implement appropriate technical and organizational measures as detailed in Annex A

**5.4 Sub-processors**
- Not engage another processor without prior specific or general written authorization from the Controller
- Impose the same data protection obligations on Sub-processors as set out in this DPA
- Remain fully liable for Sub-processor compliance

**5.5 Data Subject Rights**
- Assist the Controller in responding to Data Subject requests to exercise their rights under GDPR Chapter III
- Provide such assistance within 10 business days of request

**5.6 Security and Breach Notification**
- Assist the Controller in ensuring compliance with GDPR Articles 32-36
- Notify the Controller of any Personal Data breach without undue delay, and in any event within 24 hours of becoming aware

**5.7 Deletion**
- Delete or return all Personal Data upon termination of Services, at the Controller's choice
- Delete existing copies unless EU or Member State law requires retention

**5.8 Audit**
- Make available all information necessary to demonstrate compliance
- Allow for and contribute to audits and inspections conducted by the Controller or an authorized auditor

**5.9 Support Access with Consent**

The Processor shall implement support-with-consent controls:

- Support staff access to Controller data requires **explicit consent** from an authorized Controller representative
- Consent must be recorded with: who (user_id), what (scope), when (timestamp), scope (workspace/data type), expiry (time limit)
- Default consent expiry: 72 hours (maximum: 30 days)
- Controller may revoke consent at any time with immediate effect
- All support access is logged in the governance audit trail
- Support data export is blocked without active, non-expired consent

**Consent Record Structure:**

| Field | Description |
|-------|-------------|
| consent_id | Unique identifier |
| user_id | Controller representative granting consent |
| workspace_id | Scope of access |
| granted_at | UTC timestamp |
| expires_at | Expiry timestamp |
| scope | Data types accessible (e.g., telemetry, logs, config) |
| purpose | Reason for access (linked to support ticket) |
| support_ticket_id | Associated ticket reference |
| revoked_at | Revocation timestamp (if revoked) |

---

## 6. Controller Obligations

The Controller shall:

**6.1** Ensure it has a valid legal basis for the processing of Personal Data

**6.2** Provide documented instructions to the Processor regarding processing activities

**6.3** Notify the Processor of any changes to applicable data protection laws that may affect processing

**6.4** Ensure Data Subjects are informed of the processing as required by GDPR Articles 13-14

---

## 7. Sub-processors

**7.1 Current Sub-processors**

The Controller hereby authorizes the use of the Sub-processors listed in Annex B.

**7.2 New Sub-processors**

Before engaging a new Sub-processor, the Processor shall:
- Notify the Controller in writing at least 30 days in advance
- Provide details of the processing to be undertaken
- Allow the Controller to object within 14 days

**7.3 Objection to New Sub-processors**

If the Controller objects to a new Sub-processor on reasonable grounds, the Parties shall negotiate in good faith. If no resolution is reached within 30 days, the Controller may terminate the affected Services without penalty.

---

## 8. Security Measures

**8.1** The Processor shall implement technical and organizational measures as detailed in Annex A, including but not limited to:

- Encryption of Personal Data at rest (AES-256)
- Encryption of Personal Data in transit (TLS 1.2+)
- Access controls and authentication
- Regular security testing and vulnerability assessments
- Incident response procedures
- Business continuity measures

**8.2** The Processor shall regularly test, assess, and evaluate the effectiveness of these measures.

---

## 9. Data Breach Notification

**9.1 Notification Timeline**

The Processor shall notify the Controller of any Personal Data breach:
- Within 24 hours of becoming aware of the breach
- Via email to [CONTROLLER SECURITY EMAIL] and phone to [CONTROLLER SECURITY PHONE]

**9.2 Breach Notification Content**

The notification shall include:
- Description of the nature of the breach
- Categories and approximate number of Data Subjects affected
- Categories and approximate number of Personal Data records affected
- Name and contact details of the Processor's DPO or relevant contact
- Description of likely consequences
- Description of measures taken or proposed to address the breach

**9.3 Ongoing Updates**

The Processor shall provide updates as additional information becomes available.

---

## 10. International Data Transfers

**10.1 Default: EU Storage**

Personal Data shall be stored and processed within the European Economic Area (EEA) by default.

**10.2 Transfers Outside EEA**

If transfer outside the EEA is necessary:
- The Processor shall notify the Controller in advance
- Appropriate safeguards shall be implemented (Standard Contractual Clauses per Annex C)
- Supplementary measures shall be implemented where required

**10.3 Current Data Locations**

- Primary: AWS eu-central-1 (Frankfurt, Germany)
- Backup: AWS eu-west-1 (Dublin, Ireland)

---

## 11. Audit Rights

**11.1 Information Access**

The Processor shall make available to the Controller all information necessary to demonstrate compliance with GDPR Article 28.

**11.2 On-site Audits**

The Controller may conduct audits of the Processor's facilities and records, subject to:
- 30 days prior written notice
- During normal business hours
- Reasonable scope and duration
- Confidentiality obligations regarding findings

**11.3 Third-party Audits**

The Controller may appoint a qualified third-party auditor, subject to:
- Auditor signing a confidentiality agreement
- The Processor may object to specific auditors on reasonable grounds
- Cost borne by the Controller unless audit reveals material non-compliance

**11.4 Audit Reports**

The Processor shall provide the Controller with:
- Annual SOC 2 Type II report (or equivalent), if available
- Results of penetration testing (summary)
- Evidence of relevant security audit reports/attestations, if available

---

## 12. Termination

**12.1 Data Deletion**

Upon termination of the Services:
- The Processor shall delete all Personal Data within 30 days
- Alternatively, return Personal Data in a commonly used format if requested by Controller
- Provide written confirmation of deletion upon request

**12.2 Retention Exceptions**

The Processor may retain Personal Data where required by applicable law, subject to:
- Notifying the Controller of such requirement
- Limiting processing to that required by law
- Deleting upon expiration of the retention period

**12.3 Survival**

The obligations in this DPA shall survive termination with respect to any Personal Data retained.

---

## 13. Liability

**13.1 Allocation**

Each Party shall be liable for damages caused by processing that infringes GDPR where it has not complied with its obligations.

**13.2 Limitation**

The total liability of each Party under this DPA shall not exceed the amounts payable under the main service agreement for the 12 months preceding the claim, except in cases of gross negligence or willful misconduct.

---

## 14. General Provisions

**14.1 Amendments**

This DPA may only be amended in writing, signed by both Parties.

**14.2 Conflict**

In case of conflict between this DPA and the main service agreement, this DPA shall prevail with respect to data protection matters.

**14.3 Governing Law**

This DPA shall be governed by the laws of [JURISDICTION], without regard to conflict of law principles.

**14.4 Dispute Resolution**

Disputes shall be resolved in accordance with the dispute resolution provisions of the main service agreement.

---

## Signatures

**For the Controller:**

Name: _______________________
Title: _______________________
Date: _______________________
Signature: _______________________

**For the Processor:**

Name: _______________________
Title: _______________________
Date: _______________________
Signature: _______________________

---

## Annex A: Technical and Organizational Measures

### A.1 Encryption

| Measure | Implementation |
|---------|----------------|
| Encryption at rest | AES-256-GCM for all sensitive data |
| Encryption in transit | TLS 1.2 minimum, TLS 1.3 preferred |
| Key management | AWS KMS with automatic rotation |
| Agent-local credentials | Stored and decrypted only inside the Controller-managed Agent environment; not processed by the Processor’s Cloud |

### A.2 Access Controls

| Measure | Implementation |
|---------|----------------|
| Authentication | Multi-factor authentication required for all access |
| Authorization | Role-based access control (RBAC) |
| Session management | Automatic timeout after 30 minutes inactivity |
| Password policy | Minimum 12 characters, complexity requirements |

### A.3 Network Security

| Measure | Implementation |
|---------|----------------|
| Firewall | AWS Security Groups with deny-by-default |
| DDoS protection | AWS Shield Standard |
| Intrusion detection | AWS GuardDuty |
| VPN | Required for administrative access |

### A.4 Monitoring and Logging

| Measure | Implementation |
|---------|----------------|
| Audit logging | All access to Personal Data logged |
| Log retention | 90 days online, 1 year archived |
| Alerting | Real-time alerts for security events |
| SIEM | Centralized security monitoring |

### A.5 Physical Security

| Measure | Implementation |
|---------|----------------|
| Data centers | AWS facilities (AWS-provided ISO 27001 and/or SOC 2 reports may be available depending on region/service; verification required) |
| Access control | Biometric and badge access |
| Surveillance | 24/7 CCTV monitoring |
| Environmental | Fire suppression, climate control |

### A.6 Incident Response

| Measure | Implementation |
|---------|----------------|
| Response plan | Documented incident response procedures |
| Response team | Dedicated security incident team |
| Communication | Defined escalation paths |
| Testing | Annual incident response exercises |

### A.7 Business Continuity

| Measure | Implementation |
|---------|----------------|
| Backup frequency | Daily full, hourly incremental |
| Backup encryption | AES-256 |
| Recovery testing | Quarterly restoration tests |
| RTO/RPO | 4 hours / 1 hour (design target; actual per executed agreement) |

---

## Annex B: Sub-processor List (EU-based, current configuration)

**All current sub-processors are located within the European Union.** Sub-processor changes are subject to the notification process below (Section "Sub-processor Change Notification").

| Sub-processor | Purpose | Region (EU-only) | DPA Status | Last Review |
|--------------|---------|------------------|------------|-------------|
| Amazon Web Services (AWS) | Cloud infrastructure (RDS, S3, ElastiCache, CloudWatch) | eu-central-1 (Frankfurt), eu-west-1 (Ireland) | Signed (AWS DPA) | 2025-01-15 |
| Supabase | Database hosting (PostgreSQL alternative) | EU (Germany) | Signed | 2025-01-15 |
| Stripe | Payment processing | EU (Ireland) | Signed (Stripe DPA) | 2025-01-15 |
| AWS SES / SendGrid | Transactional email | EU | Signed | 2025-01-15 |
| Sentry | Error monitoring (redacted, no PII) | EU (Germany) | Signed | 2025-01-15 |

### Sub-processor Change Notification

- **Notification period:** 30 days prior to new sub-processor engagement
- **Method:** Email to Controller's designated contact + in-app notification
- **Objection process:** Controller may object within 30 days on reasonable grounds
- **Resolution:** If objection is not resolved within 30 days, Controller may terminate affected Services without penalty

To be notified of Sub-processor changes, contact: [NOTIFICATION EMAIL]

### EU-only Residency Evidence

The current sub-processor register with EU-only evidence is maintained at:
`docs/compliance/SUBPROCESSORS_REGISTER.md`

---

## Annex C: Standard Contractual Clauses

Where required for international transfers, the Standard Contractual Clauses (EU) 2021/914 for the transfer of personal data to third countries shall apply and are incorporated by reference.

Module: Controller to Processor (Module Two)

Clause 7: Optional docking clause - Not used
Clause 9(a): Prior specific authorization required for Sub-processors
Clause 11: Independent dispute resolution body - [AUTHORITY]
Clause 17: Governing law - [MEMBER STATE] law
Clause 18: Forum - Courts of [MEMBER STATE]

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | December 2024 | Initial version | [AUTHOR] |
| 2.0.0 | December 2024 | GDPR Phase 1: Added Section 3.1 Telemetry Sensitivity Levels (AGGREGATED/DETAILED_NON_SENSITIVE/RAW_ORDER_EVENTS), Section 5.9 Support Access with Consent, EU-only sub-processor list with review timestamps | Compliance Team |

---

## References

- GDPR Regulation (EU) 2016/679
- EDPB Guidelines on Article 28 (Controller-Processor Relationships)
- Standard Contractual Clauses (EU) 2021/914
- ISO 27001:2022 Information Security Management
- SOC 2 Type II Trust Services Criteria

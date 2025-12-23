# Personal Data Breach Response Standard Operating Procedure (SOP)

**Version**: 1.0.0
**Effective Date**: 2025-12-17
**Last Review**: 2025-12-17
**Owner**: Data Protection Officer (DPO)
**Classification**: INTERNAL

## 1. Purpose

This Standard Operating Procedure establishes the process for detecting, assessing, containing, and reporting personal data breaches in accordance with GDPR Articles 33-34 and the CCEA (Cloud-Controlled Execution Architecture) platform's compliance requirements.

## 2. Scope

This SOP applies to:
- All personal data processed by the CCEA Cloud platform
- All employees, contractors, and third-party processors with access to personal data
- Breaches affecting confidentiality, integrity, or availability of personal data

**Out of Scope** (CCEA Boundary):
- Data stored exclusively in the customer-operated Agent zone
- Broker credentials, API keys, and secrets (never processed by Cloud)
- Order/trade execution data (customer-controlled)

## 3. Definitions

| Term | Definition |
|------|------------|
| **Personal Data Breach** | A breach of security leading to accidental or unlawful destruction, loss, alteration, unauthorized disclosure of, or access to, personal data (GDPR Art. 4(12)) |
| **Confidentiality Breach** | Unauthorized or accidental disclosure or access |
| **Integrity Breach** | Unauthorized or accidental alteration |
| **Availability Breach** | Accidental or unauthorized loss of access or destruction |
| **Awareness** | The point when the organization has reasonable certainty a breach occurred (starts 72h clock) |
| **Risk Score** | Calculated severity (0.0-10.0) determining notification requirements |

## 4. Notification Deadlines

| Deadline | Requirement | Reference |
|----------|-------------|-----------|
| **72 hours** | Notify supervisory authority from awareness | GDPR Art. 33(1) |
| **Without undue delay** | Notify data subjects if high risk | GDPR Art. 34(1) |
| **24 hours** | Internal tabletop target for draft notification package | Internal SLA |

## 5. Roles and Responsibilities

### 5.1 Breach Response Team

| Role | Responsibility |
|------|----------------|
| **Data Protection Officer (DPO)** | Overall breach management, authority notifications, final decisions |
| **Security Lead** | Technical investigation, containment, evidence collection |
| **Engineering Lead** | System remediation, root cause analysis |
| **Legal Counsel** | Legal assessment, regulatory guidance, subject notifications |
| **Communications Lead** | External communications, press statements (if needed) |
| **Executive Sponsor** | Escalation, resource allocation, strategic decisions |

### 5.2 On-Call Rotation *(operational requirement; verify via monitoring/on-call configuration)*

- 24/7 security monitoring (target operational posture)
- Incident responders available via PagerDuty/on-call system (design requirement)
- DPO reachable within 1 hour for breach classification (target)

## 6. Breach Response Workflow

### 6.1 Phase 1: Detection (T+0 to T+1h)

```
┌─────────────────────────────────────────────────────────────┐
│                      DETECTION                               │
├─────────────────────────────────────────────────────────────┤
│  Sources:                                                    │
│  • Security monitoring alerts                                │
│  • Employee reports                                          │
│  • Customer complaints                                       │
│  • Third-party notifications                                 │
│  • Automated abuse detection                                 │
│                                                              │
│  Actions:                                                    │
│  1. Create breach ticket in incident system                  │
│  2. Assign initial severity (P1/P2/P3)                      │
│  3. Notify Security Lead                                     │
│  4. Begin initial triage                                     │
└─────────────────────────────────────────────────────────────┘
```

**Initial Report Form:**
```
Breach Report #: [AUTO-GENERATED]
Date/Time Detected: [TIMESTAMP]
Detected By: [NAME/SYSTEM]
Initial Description: [FREE TEXT]
Affected Systems: [LIST]
Estimated Impact: [LOW/MEDIUM/HIGH/CRITICAL]
Evidence Preserved: [YES/NO]
```

### 6.2 Phase 2: Assessment (T+1h to T+4h)

```
┌─────────────────────────────────────────────────────────────┐
│                     ASSESSMENT                               │
├─────────────────────────────────────────────────────────────┤
│  Risk Factor Analysis:                                       │
│  • Data sensitivity (0.0-1.0)                               │
│  • Data volume (0.0-1.0)                                    │
│  • Identifiability (0.0-1.0)                                │
│  • Special categories present (Y/N)                         │
│  • Vulnerable individuals (Y/N)                             │
│  • Potential harm (0.0-1.0)                                 │
│  • Cross-border (Y/N)                                       │
│  • Malicious intent (Y/N)                                   │
│                                                              │
│  Risk Score Calculation:                                     │
│  Score = (sensitivity × 2.5) + (volume × 1.5) +             │
│          (identifiability × 2.0) + (harm × 2.5) +           │
│          (special_categories ? 1.0 : 0) +                   │
│          (vulnerable ? 0.5 : 0) +                           │
│          (malicious ? 1.0 : 0) + (cross_border ? 0.5 : 0)   │
└─────────────────────────────────────────────────────────────┘
```

**Risk Score Thresholds:**

| Score Range | Severity | Authority Notification | Subject Notification |
|-------------|----------|------------------------|----------------------|
| 0.0 - 2.9 | LOW | Document only | Not required |
| 3.0 - 5.9 | MEDIUM | Required (72h) | Evaluate case-by-case |
| 6.0 - 7.9 | HIGH | Required (72h) | Required |
| 8.0 - 10.0 | CRITICAL | Required (immediate) | Required (immediate) |

### 6.3 Phase 3: Notification Decision (T+4h to T+8h)

```
┌─────────────────────────────────────────────────────────────┐
│                  NOTIFICATION DECISION                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐                                       │
│  │ Risk Score ≥ 3.0 │─────Yes────▶ Authority Notification   │
│  └────────┬─────────┘             Required (72h deadline)   │
│           │No                                                │
│           ▼                                                  │
│  Document & Monitor                                          │
│                                                              │
│  ┌──────────────────┐                                       │
│  │ Risk Score ≥ 6.0 │─────Yes────▶ Subject Notification     │
│  └────────┬─────────┘             Required                  │
│           │No                                                │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ Art. 34(3)       │                                       │
│  │ Exemption        │─────Yes────▶ Document Exemption       │
│  │ Applies?         │                                       │
│  └──────────────────┘                                       │
│                                                              │
│  Exemptions (Art. 34(3)):                                   │
│  (a) Encryption: Data rendered unintelligible               │
│  (b) Subsequent measures: Risk no longer likely             │
│  (c) Disproportionate effort: Public communication          │
└─────────────────────────────────────────────────────────────┘
```

**Decision Documentation:**
```
Decision #: [AUTO-GENERATED]
Breach #: [REFERENCE]
Risk Score: [0.0-10.0]
Authority Notification: [REQUIRED/NOT REQUIRED]
Deadline: [TIMESTAMP]
Subject Notification: [REQUIRED/NOT REQUIRED/EXEMPT]
Exemption Applied: [NONE/ENCRYPTION/SUBSEQUENT_MEASURES/DISPROPORTIONATE]
Justification: [FREE TEXT]
Decided By: [NAME]
Approved By: [DPO NAME]
Date: [TIMESTAMP]
```

### 6.4 Phase 4: Authority Notification (T+8h to T+72h)

**Required Content (Art. 33(3)):**

```
SUPERVISORY AUTHORITY NOTIFICATION

To: [Authority Name, e.g., ICO, CNIL, BfDI]
From: [Organization Name]
Date: [DATE]
Reference: [BREACH-XXXX]

1. NATURE OF THE BREACH
   Description: [Confidentiality/Integrity/Availability breach description]

2. CATEGORIES OF DATA SUBJECTS
   - [e.g., Customers, Employees, Enterprise users]

3. APPROXIMATE NUMBER OF DATA SUBJECTS
   - [Number or estimate with range]

4. CATEGORIES OF PERSONAL DATA RECORDS
   - [e.g., Names, email addresses, telemetry data]

5. APPROXIMATE NUMBER OF RECORDS
   - [Number or estimate with range]

6. DPO CONTACT DETAILS
   Name: [DPO Name]
   Email: [DPO Email]
   Phone: [DPO Phone]

7. LIKELY CONSEQUENCES
   - [List potential impacts on data subjects]

8. MEASURES TAKEN TO ADDRESS THE BREACH
   - [Containment measures implemented]

9. MEASURES TO MITIGATE ADVERSE EFFECTS
   - [Remediation actions planned]

Submitted by: [Name]
Signature: [Digital/Physical]
```

### 6.5 Phase 5: Subject Notification (When Required)

**Required Content (Art. 34(2)):**

```
SUBJECT: Important Security Notice - Action May Be Required

Dear [Data Subject Name],

We are writing to inform you of a security incident that may have
affected your personal data.

WHAT HAPPENED
[Plain language description of the breach, avoiding technical jargon]

WHAT DATA WAS AFFECTED
[List of data categories in simple terms]

WHAT WE ARE DOING
[Measures taken and planned]

WHAT YOU CAN DO
[Specific, actionable recommendations]
- Monitor your accounts for unusual activity
- Change your password if you haven't recently
- Be cautious of suspicious emails or calls

CONTACT US
If you have questions, contact our Data Protection Officer:
Name: [DPO Name]
Email: [DPO Email]
Phone: [DPO Phone]

We sincerely apologize for any inconvenience this may cause.

[Organization Name]
```

### 6.6 Phase 6: Containment & Remediation (Ongoing)

**Containment Checklist:**
- [ ] Isolate affected systems
- [ ] Revoke compromised credentials
- [ ] Block malicious IPs/sources
- [ ] Preserve evidence (forensic images, logs)
- [ ] Implement emergency access controls
- [ ] Verify containment effectiveness

**Remediation Checklist:**
- [ ] Identify root cause
- [ ] Develop fix/patch
- [ ] Test remediation in staging
- [ ] Deploy remediation to production
- [ ] Verify remediation effectiveness
- [ ] Update security controls
- [ ] Document lessons learned

### 6.7 Phase 7: Resolution & Closure (Post-Incident)

**Closure Criteria:**
- [ ] All containment measures verified effective
- [ ] Root cause identified and documented
- [ ] Remediation deployed and verified
- [ ] Authority notification completed (if required)
- [ ] Subject notification completed (if required)
- [ ] Lessons learned documented
- [ ] Process improvements identified
- [ ] Evidence pack generated and archived

## 7. Timeline Summary

```
Detection        Assessment        Decision         Notification      Resolution
    │                │                │                 │                │
    T+0            T+4h             T+8h              T+72h           Ongoing
    │                │                │                 │                │
    ├────────────────┼────────────────┼─────────────────┼────────────────┤
    │                │                │                 │                │
 Detect &        Risk Score      Authority &      Submit to         Close
 Triage         Calculation     Subject Decision   Authority        Incident
```

## 8. Tabletop Exercises

### 8.1 Frequency

- **Quarterly**: Full tabletop exercise with all response team members
- **Annually**: Cross-functional exercise including executives and external counsel

### 8.2 Exercise Requirements

**Success Criteria:**
- Draft notification package produced within 24 hours
- All required content elements included
- Clear decision rationale documented
- Timeline SLA met (internal targets)

**Scenario Types:**
1. Ransomware attack on customer database
2. Unauthorized employee access
3. External data exfiltration
4. Accidental disclosure to wrong recipient
5. Third-party processor breach

### 8.3 Exercise Report Template

```
TABLETOP EXERCISE REPORT

Exercise ID: [AUTO-GENERATED]
Date Conducted: [DATE]
Scenario: [NAME]
Duration: [MINUTES]
Participants: [LIST]

SCENARIO DESCRIPTION
[Summary of the simulated breach]

TIMELINE ACHIEVED
- Detection to awareness: [MINUTES]
- Awareness to assessment: [MINUTES]
- Assessment to decision: [MINUTES]
- Decision to draft notification: [MINUTES]

DECISIONS MADE
[List of key decisions during exercise]

GAPS IDENTIFIED
[List of process gaps or issues discovered]

IMPROVEMENTS RECOMMENDED
[List of recommended improvements]

ACTION ITEMS
| Item | Owner | Due Date | Status |
|------|-------|----------|--------|
|      |       |          |        |

NEXT EXERCISE DUE: [DATE - 90 days from exercise]
```

## 9. Evidence Retention

| Evidence Type | Retention Period | Storage Location |
|---------------|------------------|------------------|
| Breach records | 7 years | Governance DB |
| Notification copies | 7 years | Document store |
| Timeline events | 7 years | Audit log |
| Root cause analysis | 7 years | Incident system |
| Tabletop reports | 7 years | Compliance store |
| Supporting evidence | 7 years | Secure archive |

## 10. Contact Information

### 10.1 Internal Contacts

| Role | Primary | Backup |
|------|---------|--------|
| DPO | [Email/Phone] | [Email/Phone] |
| Security Lead | [Email/Phone] | [Email/Phone] |
| Legal Counsel | [Email/Phone] | [Email/Phone] |
| Executive Sponsor | [Email/Phone] | [Email/Phone] |

### 10.2 Supervisory Authorities

| Authority | Jurisdiction | Contact |
|-----------|--------------|---------|
| ICO | United Kingdom | [Portal URL] |
| CNIL | France | [Portal URL] |
| BfDI | Germany | [Portal URL] |
| AEPD | Spain | [Portal URL] |
| DPA Ireland | Ireland | [Portal URL] |

## 11. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-17 | DPO | Initial version |

## 12. References

- GDPR Regulation (EU) 2016/679 - Articles 33-34
- EDPB Guidelines on personal data breach notification (WP250 rev.01)
- CCEA Design Doc - Section 15 (Security Controls)
- `packages/cloud/governance/breach_workflow.py` - Implementation
- `docs/compliance/SECURITY_PHASE7_SPEC.md` - Technical specification

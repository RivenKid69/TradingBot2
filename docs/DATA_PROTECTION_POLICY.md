# Data Protection Policy

## CustodiaCloud

**Version**: 1.0
**Effective Date**: December 2024
**Policy Owner**: Data Protection Officer
**Review Cycle**: Annual

---

## 1. Purpose

This policy establishes the principles, responsibilities, and procedures for protecting personal data processed by our organization in compliance with the General Data Protection Regulation (EU) 2016/679 (GDPR) and applicable national data protection laws.

**Architecture Foundation**: Our data protection measures are built on the **Cloud-Controlled Execution Architecture (CCEA)**, which provides strict zone separation and mandatory data protection controls.

---

## 1.1 CCEA Data Protection Model

### Zone-Based Data Separation

Our platform implements strict data separation between Cloud Zone and Agent Zone:

| Data Category | Cloud Zone | Agent Zone | Protection Mechanism |
|---------------|------------|------------|---------------------|
| **Trading Credentials** | ❌ NEVER stored | ✅ Local Vault only | Architecture enforcement |
| **Broker API Keys** | ❌ NEVER accessed | ✅ HSM/KMS/Keychain | Secrets never leave Agent |
| **User Strategies** | ✅ Research/backtest | ✅ Execution | Encryption + access control |
| **Telemetry Data** | ✅ Redacted only | Raw at source | Mandatory redaction middleware |
| **Order Data** | ❌ NEVER generated | ✅ Created locally | No order payloads in Cloud |

### Mandatory Telemetry Redaction

All telemetry sent from Agent to Cloud passes through mandatory redaction:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TELEMETRY REDACTION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Agent Zone:                                                                 │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │  Raw        │ → │  Redaction      │ → │  Sanitized Telemetry         │  │
│  │  Telemetry  │    │  Middleware     │    │  (ready for transmission)   │  │
│  │             │    │  (MANDATORY)    │    │                             │  │
│  │  - PII      │    │  Removes:       │    │  - Anonymized metrics       │  │
│  │  - Secrets  │    │  - API keys     │    │  - Performance data         │  │
│  │  - Strategy outputs │    │  - Passwords    │    │  - Error logs (safe)        │  │
│  │             │    │  - Auth tokens  │    │                             │  │
│  └─────────────┘    │  - Trading IDs  │    └─────────────────────────────┘  │
│                     └─────────────────┘              │                       │
│                                                      │                       │
└──────────────────────────────────────────────────────│───────────────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLOUD ZONE                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Telemetry Storage (receives only sanitized data)                    │    │
│  │  - CANNOT reconstruct secrets                                        │    │
│  │  - CANNOT identify trading positions                                 │    │
│  │  - CANNOT replay trading activity                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Redaction Levels

| Level | What's Redacted | When Used | Can Be Disabled |
|-------|-----------------|-----------|-----------------|
| **MANDATORY** | API keys, passwords, tokens | Always | ❌ No |
| **DEFAULT** | + PII, account IDs, order-like payloads (orders/targets/signals) | Production | ❌ No |
| **HIGHER_SENSITIVITY** | Additional fields (enterprise opt-in) | Enterprise opt-in | N/A (explicit opt-in) |

**Higher-sensitivity telemetry**: Available only via explicit opt-in and enterprise controls. Data transmitted to Cloud remains subject to redaction/telemetry controls; customers should perform their own legal review for deployments that enable additional telemetry fields.

### CCEA GDPR Alignment

| GDPR Principle | CCEA Implementation |
|----------------|---------------------|
| **Data Minimization** | Cloud never receives trading credentials |
| **Storage Limitation** | Secrets stored only in local Agent vault |
| **Integrity & Confidentiality** | Mandatory redaction prevents secret exposure |
| **Privacy by Design** | Architecture enforces data separation |
| **Accountability** | Audit trail of all data flows |

---

## 2. Scope

This policy applies to:
- All employees, contractors, and third parties processing personal data on our behalf
- All personal data processed by the organization, regardless of format or location
- All processing activities, whether automated or manual

---

## 3. Data Protection Principles

We adhere to the following principles (GDPR Article 5):

### 3.1 Lawfulness, Fairness, and Transparency
- Process personal data only with a valid legal basis
- Be transparent about how we use personal data
- Provide clear privacy notices to data subjects

### 3.2 Purpose Limitation
- Collect personal data only for specified, explicit, and legitimate purposes
- Do not process data in ways incompatible with original purposes

### 3.3 Data Minimization
- Collect only the minimum personal data necessary for each purpose
- Regularly review data holdings and delete unnecessary data

### 3.4 Accuracy
- Keep personal data accurate and up to date
- Correct or delete inaccurate data without delay

### 3.5 Storage Limitation
- Retain personal data only as long as necessary for its purpose
- Follow our data retention schedule for deletion

### 3.6 Integrity and Confidentiality
- Protect personal data with appropriate technical and organizational measures
- Prevent unauthorized access, loss, destruction, or damage

### 3.7 Accountability
- Demonstrate compliance with GDPR requirements
- Maintain records of processing activities

---

## 4. Roles and Responsibilities

### 4.1 Data Protection Officer (DPO)
- Advise on data protection obligations
- Monitor compliance with GDPR
- Serve as contact point for supervisory authorities
- Handle data subject requests

**Contact**: dpo@[company].com

### 4.2 Management
- Ensure adequate resources for data protection
- Promote a culture of data protection
- Review and approve data protection policies

### 4.3 All Employees
- Follow this policy and related procedures
- Report data protection concerns or incidents
- Complete required training
- Handle personal data securely

### 4.4 IT/Security Team
- Implement technical security measures
- Monitor for security incidents
- Support data protection by design

---

## 5. Legal Bases for Processing

We process personal data only when we have a valid legal basis:

| Legal Basis | When Used | Examples |
|-------------|-----------|----------|
| **Contract** (Art. 6(1)(b)) | Necessary for contract performance | Account management, service delivery |
| **Legal Obligation** (Art. 6(1)(c)) | Required by law | Tax records, regulatory compliance |
| **Legitimate Interest** (Art. 6(1)(f)) | Legitimate business need, balanced against data subject rights | Security monitoring, fraud prevention |
| **Consent** (Art. 6(1)(a)) | Freely given, specific, informed consent | Marketing communications |

---

## 6. Data Subject Rights

We respect and facilitate the following rights:

### 6.1 Right of Access (Art. 15)
Data subjects can request a copy of their personal data.

### 6.2 Right to Rectification (Art. 16)
Data subjects can request correction of inaccurate data.

### 6.3 Right to Erasure (Art. 17)
Data subjects can request deletion of their data in certain circumstances.

### 6.4 Right to Restriction (Art. 18)
Data subjects can request limited processing in certain circumstances.

### 6.5 Right to Data Portability (Art. 20)
Data subjects can request their data in a portable format.

### 6.6 Right to Object (Art. 21)
Data subjects can object to processing based on legitimate interests or for direct marketing.

### 6.7 Rights Related to Automated Decisions (Art. 22)
Data subjects can request human review of automated decisions.

**Request Handling**: All requests must be forwarded to dpo@[company].com and handled within 30 days.

---

## 7. Data Security

### 7.1 Technical Measures
- Encryption at rest (AES-256) and in transit (TLS 1.3)
- Access controls and authentication
- Network security (firewalls, segmentation)
- Regular security assessments

### 7.2 Organizational Measures
- Employee training and awareness
- Access limited to need-to-know basis
- Confidentiality agreements
- Regular policy reviews

### 7.3 Physical Measures
- Cloud infrastructure in data centers with relevant security certifications (provider-dependent)
- Access controls to office premises
- Secure disposal of physical media

---

## 8. Data Breach Procedures

### 8.1 Definition
A personal data breach is any security incident leading to accidental or unlawful destruction, loss, alteration, unauthorized disclosure of, or access to personal data.

### 8.2 Reporting
All suspected breaches must be reported immediately to:
- Security team: security@[company].com
- DPO: dpo@[company].com

### 8.3 Response Timeline
| Action | Timeline |
|--------|----------|
| Internal assessment | Within 24 hours |
| DPO notification | Within 24 hours |
| Supervisory authority notification (if required) | Within 72 hours |
| Data subject notification (if high risk) | Without undue delay |

---

## 9. International Data Transfers

### 9.1 Transfer Mechanisms
Personal data transferred outside the EU/EEA will only be transferred using approved mechanisms:
- Adequacy decisions
- Standard Contractual Clauses (SCCs)
- Binding Corporate Rules (where applicable)

### 9.2 Third Countries
Before transferring data to third countries, we assess the data protection laws and implement supplementary measures if needed.

---

## 10. Third-Party Processing

### 10.1 Due Diligence
Before engaging processors:
- Assess their data protection capabilities
- Verify appropriate security measures
- Review certifications (SOC 2, ISO 27001)

### 10.2 Data Processing Agreements
All processors must sign a DPA covering:
- Processing instructions
- Security obligations
- Sub-processor management
- Audit rights
- Data breach notification

---

## 11. Data Retention

### 11.1 Retention Principles
- Retain data only as long as necessary
- Follow legal retention requirements
- Securely delete data when no longer needed

### 11.2 Retention Schedule

| Category | Retention Period | Basis |
|----------|------------------|-------|
| Client account data | Service duration + 7 years | Contract, tax law |
| Transaction records | 10 years | Tax law |
| Marketing consents | Until withdrawn + 3 years | GDPR documentation |
| Security logs | 1 year | Legitimate interest |
| Job applicant data | 6 months (unsuccessful) | Legitimate interest |

---

## 12. Privacy by Design

We integrate data protection into all processing activities:
- Consider privacy from project inception
- Default to privacy-protective settings
- Minimize data collection
- Build security into systems
- Document privacy considerations

---

## 13. Training and Awareness

### 13.1 Mandatory Training
All employees must complete:
- GDPR awareness training (onboarding)
- Annual refresher training
- Role-specific training (where applicable)

### 13.2 Awareness Activities
- Regular communications on data protection topics
- Phishing simulation exercises
- Policy updates and reminders

---

## 14. Compliance Monitoring

### 14.1 Regular Activities
- Quarterly compliance reviews
- Annual policy review
- Periodic audits
- Processing activity inventory updates

### 14.2 Documentation
We maintain:
- Records of Processing Activities (ROPA)
- Data Protection Impact Assessments
- Consent records
- Data subject request logs
- Breach records

---

## 15. Policy Violations

Violations of this policy may result in:
- Disciplinary action up to termination
- Legal action for serious breaches
- Reporting to supervisory authorities

---

## 16. Contact Information

**Data Protection Officer**
Email: dpo@[company].com

**Privacy Inquiries**
Email: privacy@[company].com

**Security Incidents**
Email: security@[company].com

---

## 17. Policy Review

This policy is reviewed annually and updated as needed to reflect:
- Regulatory changes
- Business changes
- Lessons learned from incidents
- Industry best practices

---

**Document Control**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2024 | Initial version |

*This policy does not constitute legal advice. Consult with qualified legal counsel for specific data protection questions.*

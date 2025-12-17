# DORA ICT Service Contract Template
## Article 30(2) Basic Provisions - All EU Clients

**Version**: 1.0
**Date**: 2025-01-17 (DORA Application Date)
**Status**: Implementation complete
**Legal Reference**: Regulation (EU) 2022/2554 Article 30(2)(a-i)

---

## Purpose

This template is designed to help parties include contract clauses aligned with DORA Article 30(2) for ICT service agreements with EU-regulated financial entities. It includes placeholders for all nine mandatory provisions of Article 30(2), but it is not legal advice and must be reviewed/approved by qualified counsel.

---

## SCHEDULE: DORA ICT SERVICE PROVISIONS

### 1. SERVICE DESCRIPTION — Article 30(2)(a)

**1.1 ICT Services Provided**

The Provider shall deliver the following ICT services to the Client:

| Service Category | Description | Components |
|-----------------|-------------|------------|
| Quantitative Research Platform | AI/ML-powered trading strategy development | Strategy builder, backtesting engine, risk analytics |
| Market Data Services | Real-time and historical market data | Price feeds, order book data, trade history |
| Algorithmic Trading Infrastructure | Strategy execution and monitoring | Order management, execution algorithms, performance tracking |
| Risk Management Services | Portfolio risk assessment and monitoring | VaR calculations, stress testing, exposure analysis |

**1.2 Service Boundaries**

The Provider SHALL:
- Provide services as described in Schedule A (Service Specifications)
- Maintain service documentation current and accurate
- Notify Client of material service changes per Section 2.3

The Provider SHALL NOT:
- Execute trades on Client's behalf without explicit authorization
- Access Client's trading accounts except as required for service delivery
- Make investment decisions for Client

**1.3 Subcontracting**

(a) The Provider may subcontract ICT services subject to conditions in Section 10 of this Schedule.

(b) Current subcontractors are listed in Annex B (Subcontractor Register).

(c) Material subcontracting changes require advance notification per Section 10.3.

---

### 2. DATA LOCATION — Article 30(2)(b)

**2.1 Processing Locations**

| Data Type | Processing Location(s) | Storage Location(s) |
|-----------|----------------------|---------------------|
| Client Trading Data | EU (Frankfurt, DE) | EU (Frankfurt, DE) |
| Client Configuration | EU (Frankfurt, DE) | EU (Frankfurt, DE) |
| Market Data | EU (Frankfurt, DE), US (Virginia) | EU (Frankfurt, DE) |
| Backtest Results | EU (Frankfurt, DE) | EU (Frankfurt, DE) |
| ML/AI Models | EU (Frankfurt, DE) | EU (Frankfurt, DE) |
| Audit Logs | EU (Frankfurt, DE) | EU (Frankfurt, DE) |

**2.2 EU Data Residency Option**

Upon Client request, Provider SHALL ensure all Client data remains within the European Union/EEA. This option is available at [no additional cost / specified tier].

**2.3 Location Change Notification**

Provider SHALL notify Client:
- **60 days in advance** of any planned change in data processing/storage location
- **Immediately** if compelled to transfer data to a non-EU jurisdiction
- Client may object within 30 days of notification
- If objection cannot be resolved, Client may terminate per Section 8

**2.4 Data Transfer Safeguards**

Any data transfer outside the EU/EEA SHALL be subject to:
- Standard Contractual Clauses (SCCs) per Commission Decision 2021/914
- Supplementary measures as required by CJEU Schrems II ruling
- Client approval for transfers to jurisdictions without adequacy decisions

---

### 3. DATA SECURITY PROVISIONS — Article 30(2)(c)

**3.1 Availability Commitments**

| Service | Target Availability | Measurement Period | Exclusions |
|---------|-------------------|-------------------|------------|
| Trading Platform | 99.9% | Monthly | Scheduled maintenance |
| API Services | 99.9% | Monthly | Force majeure |
| Data Services | 99.5% | Monthly | Third-party outages |

**3.2 Authenticity Measures**

Provider SHALL implement:
- Multi-factor authentication for all user accounts
- API authentication via secure tokens (OAuth 2.0 / API keys)
- Mutual TLS for system-to-system communications
- Digital signatures on critical data exports

**3.3 Integrity Controls**

Provider SHALL ensure:
- All data at rest encrypted with AES-256
- All data in transit encrypted with TLS 1.3
- Cryptographic checksums (SHA-256) for data exports
- Immutable audit logs for all data modifications
- Database transaction logs with point-in-time recovery

**3.4 Confidentiality Safeguards**

Provider SHALL maintain:
- Role-based access control (RBAC) for all systems
- Logical separation of Client data from other clients
- Staff background checks for personnel with data access
- Confidentiality agreements with all employees and subcontractors
- Data classification scheme aligned with Client requirements

---

### 4. DATA ACCESS, RECOVERY AND RETURN — Article 30(2)(d)

**4.1 Data Access Rights**

Client SHALL have:
- Real-time access to own data via secure API
- Self-service data export functionality
- Read-only access to audit logs related to Client data
- Ability to request comprehensive data export at any time

**4.2 Data Recovery**

In the event of data loss or corruption:
- Provider SHALL restore from backup within RTO (see Section 3.1)
- Recovery Point Objective (RPO): Maximum 1 hour data loss
- Client SHALL be notified within 30 minutes of any data loss event
- Post-recovery verification SHALL be documented

**4.3 Data Return Upon Termination**

Upon contract termination, expiration, or Provider insolvency:

| Scenario | Export Request Response | Data Package Ready | Download Available |
|----------|------------------------|-------------------|-------------------|
| Standard Termination | 24 hours | 5 business days | 30 days |
| Urgent Termination | 4 hours | 48 hours | 14 days |
| Insolvency | Via escrow | Immediate | Per escrow terms |

**4.4 Data Export Formats**

All Client data SHALL be exportable in:
- **JSON**: Structured data (strategies, configurations)
- **CSV**: Tabular data (performance history, trades)
- **SQL**: Database dumps (PostgreSQL-compatible)
- **ONNX**: Machine learning models
- **PDF/Markdown**: Documentation and reports

**4.5 Data Scope**

Export SHALL include:
- All trading strategies and configurations
- Backtest results and performance history
- Trained ML/AI models and weights
- User preferences and settings
- Audit logs (Client-specific)
- API integration configurations

Export SHALL NOT include:
- Platform proprietary source code
- Other clients' data
- Aggregated anonymized platform metrics

**4.6 Insolvency Protection**

Provider SHALL maintain:
- Data escrow with [designated escrow provider]
- Weekly full backups to escrow
- Escrow access triggered by: insolvency filing, 30-day non-response
- Client data classified as Client property (not platform asset)
- Priority access rights in insolvency proceedings

---

### 5. SERVICE LEVEL DESCRIPTIONS — Article 30(2)(e)

**5.1 Quantitative Performance Targets**

| Metric | Target | Warning | Breach | Measurement |
|--------|--------|---------|--------|-------------|
| Availability | 99.9% | <99.7% | <99.5% | Monthly |
| API Latency (P95) | <500ms | >750ms | >1000ms | Hourly |
| Data Latency (P95) | <200ms | >400ms | >600ms | Hourly |
| Incident Response | 15 min | 30 min | 60 min | Per incident |

**5.2 Qualitative Targets**

- Support response time: 4 business hours (standard), 1 hour (critical)
- Feature request evaluation: 5 business days
- Security patch deployment: 24 hours (critical), 7 days (standard)

**5.3 SLA Reporting**

Provider SHALL provide:
- Monthly SLA performance report by 5th business day
- Real-time status page access
- Automated alerts on threshold breaches
- Quarterly business review meetings

**5.4 SLA Updates and Revisions**

- Material SLA changes require 60 days advance notice
- Client may object within 30 days
- Annual SLA review included in contract governance

---

### 6. INCIDENT ASSISTANCE — Article 30(2)(f)

**6.1 Incident Categories**

| Category | Definition | Provider Response | Client Notification |
|----------|------------|-------------------|-------------------|
| Critical | Service unavailable, data breach | 15 min response | 30 min |
| High | Degraded performance, security event | 30 min response | 1 hour |
| Medium | Limited impact issue | 4 hours response | 4 hours |
| Low | Minor issue, no impact | Next business day | Weekly summary |

**6.2 Incident Assistance Obligations**

Provider SHALL:
- Provide 24/7 incident response for Critical/High severity
- Assist Client's incident investigation at no additional cost (Critical/High)
- Provide incident reports within 24 hours of resolution
- Participate in Client's post-incident review if requested
- Preserve evidence for regulatory or legal purposes

**6.3 Cost Provisions**

- Critical/High incident assistance: Included, no additional cost
- Medium/Low incident support: Per standard support terms
- Extended investigation support: Predetermined hourly rate in Annex C

**6.4 Incident Documentation**

Provider SHALL maintain:
- Incident timeline with all actions taken
- Root cause analysis report
- Remediation actions and evidence
- Lessons learned documentation

---

### 7. AUTHORITY COOPERATION — Article 30(2)(g)

**7.1 Cooperation Scope**

Provider SHALL fully cooperate with:
- Client's competent authority (NCA)
- Client's resolution authority
- Other supervisory authorities with jurisdiction over Client

**7.2 Cooperation Obligations**

Provider SHALL:
- Respond to information requests within 5 business days
- Make personnel available for interviews as reasonably requested
- Provide documentation relevant to Client's regulatory obligations
- Support Client's regulatory examinations and inspections
- Not impede or obstruct regulatory supervision in any way

**7.3 Limitations**

Provider cooperation is limited to:
- Information and access relevant to services provided to specific Client
- Protection of other clients' confidential information
- Commercially sensitive information protected where possible

**7.4 Notice to Client**

Provider SHALL notify Client within 24 hours of:
- Any direct contact from Client's NCA regarding this arrangement
- Any regulatory inquiry that may affect services to Client
- Any enforcement action against Provider that could impact Client

---

### 8. TERMINATION RIGHTS — Article 30(2)(h)

**8.1 Termination for Convenience**

Either party may terminate upon **90 days** written notice.

**8.2 Termination for Cause**

Client may terminate immediately or upon **30 days** notice if:
- Provider commits material breach not cured within 30 days
- Provider becomes insolvent or files for bankruptcy
- Provider experiences security incident materially affecting Client
- Provider fails to maintain required certifications
- Provider commits 3+ SLA breaches within 12 months
- Provider subject to regulatory enforcement affecting service delivery

**8.3 Regulatory Termination**

Client may terminate immediately, without penalty, if:
- Client's NCA or resolution authority directs termination
- Applicable law prohibits the outsourcing arrangement
- Provider designated as CTPP and fails oversight requirements

**8.4 Transition Assistance**

Upon any termination:
- Provider SHALL provide transition assistance for minimum 90 days
- Services continue during notice period at agreed service levels
- Data export per Section 4.3 timelines
- Knowledge transfer sessions included (up to 8 hours)

**8.5 No Vendor Lock-in**

Provider SHALL NOT:
- Use proprietary data formats that impede portability
- Impose technical barriers to data migration
- Withhold documentation needed for transition

---

### 9. TRAINING PARTICIPATION — Article 30(2)(i)

**9.1 Security Awareness Programs**

Provider SHALL participate in Client's ICT security awareness programs:
- Annual security briefing session (minimum)
- Additional sessions upon reasonable request
- Remote participation preferred; on-site available

**9.2 Digital Operational Resilience Training**

Provider SHALL support Client's resilience training per Article 13(6):
- Participate in scenario-based exercises
- Support joint disaster recovery drills
- Provide platform-specific training materials

**9.3 Provider Contribution**

Provider SHALL:
- Present platform security architecture
- Explain incident response procedures
- Review shared responsibility model
- Participate in Q&A with Client security team

**9.4 Scheduling**

- Standard notice: 14 business days
- Urgent (security-related): 5 business days
- Emergency (active incident): Best effort

**9.5 Resource Commitment**

| Client Tier | Annual Hours Included | Additional Hours |
|-------------|----------------------|------------------|
| Standard | 4 hours | Standard hourly rate |
| Professional | 8 hours | Reduced rate |
| Enterprise | 16 hours | Negotiated rate |

---

### 10. SUBCONTRACTING CONDITIONS

**10.1 Current Subcontractors**

See Annex B for complete subcontractor register including:
- Subcontractor name and LEI (if available)
- Services provided
- Data processing locations
- Security certifications

**10.2 Permitted Subcontracting**

Provider may subcontract provided:
- Subcontractor meets security standards equivalent to this Agreement
- Client is notified per Section 10.3
- Subcontractor agreement includes DORA-required terms
- Provider remains fully liable for subcontractor performance

**10.3 Notification Requirements**

| Change Type | Notice Period | Client Rights |
|-------------|---------------|---------------|
| New subcontractor | 30 days advance | Objection within 15 days |
| Service scope change | 30 days advance | Review and comment |
| Location change | 60 days advance | Objection within 30 days |
| Termination | 14 days advance | Information only |

**10.4 Objection Process**

If Client objects to subcontracting change:
1. Provider SHALL engage in good faith discussion within 10 business days
2. Parties SHALL seek mutually acceptable alternative
3. If unresolved, Client may terminate affected services with 90 days notice

---

## ANNEXES

### Annex A: Service Specifications
[Detailed technical specifications for each service]

### Annex B: Subcontractor Register
[Complete list of current subcontractors with required information]

### Annex C: Fee Schedule
[Pricing for standard services, additional support, and transition assistance]

### Annex D: SLA Definitions
[Detailed measurement methodology for each SLA metric]

### Annex E: Data Processing Agreement
[GDPR-aligned DPA as required (legal review)]

---

## ACCEPTANCE

This Schedule forms an integral part of the ICT Service Agreement between Provider and Client.

**Client Acceptance:**

Name: ________________________________

Title: ________________________________

Date: ________________________________

Signature: ____________________________


**Provider Acceptance:**

Name: ________________________________

Title: ________________________________

Date: ________________________________

Signature: ____________________________

---

*Document Version Control:*
- v1.0 (2025-01-17): Initial release aligned with DORA application date
- Based on Regulation (EU) 2022/2554 Article 30(2)(a-i)
- Reviewed by: Legal, Compliance, Security, Operations

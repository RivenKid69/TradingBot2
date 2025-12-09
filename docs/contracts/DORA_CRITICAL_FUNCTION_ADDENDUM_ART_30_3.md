# DORA Critical/Important Function Addendum
## Article 30(3) Additional Provisions

**Version**: 1.0
**Date**: 2025-01-17 (DORA Application Date)
**Status**: Production Ready
**Legal Reference**: Regulation (EU) 2022/2554 Article 30(3)(a-j)

---

## Purpose

This Addendum supplements the base DORA ICT Service Provisions (Art. 30(2)) with additional requirements mandated when ICT services support Client's **critical or important functions** as defined in DORA Article 3(22).

**This Addendum applies when:**
- Client has designated Provider's services as supporting a critical/important function
- Provider's services directly impact Client's ability to comply with authorization conditions
- Provider's services support functions essential to Client's business model

---

## ADDENDUM: CRITICAL/IMPORTANT FUNCTION PROVISIONS

### Article 30(3)(a) — FULL SERVICE LEVEL AGREEMENTS

**A.1 Quantitative Performance Targets**

| Service | Metric | Target | Measurement | Breach Consequence |
|---------|--------|--------|-------------|-------------------|
| Trading Platform | Availability | 99.95% | Monthly | Service credits |
| Trading Platform | Order Latency P95 | <100ms | Per transaction | Incident escalation |
| Trading Platform | Order Latency P99 | <250ms | Per transaction | Breach notification |
| API Gateway | Availability | 99.95% | Monthly | Service credits |
| API Gateway | Response Time P95 | <200ms | Hourly | Performance review |
| Market Data | Data Freshness | <50ms | Per tick | Alert & investigation |
| Market Data | Completeness | 99.99% | Daily | Gap analysis |
| Risk Engine | Calculation Time | <5s | Per request | Capacity review |
| Backup Systems | RPO | 15 minutes | Per backup | Immediate escalation |
| Backup Systems | RTO | 4 hours | Per DR test | BCP review |

**A.2 Service Credits**

| Monthly Availability | Credit (% Monthly Fee) |
|---------------------|----------------------|
| 99.95% - 99.90% | 5% |
| 99.90% - 99.50% | 10% |
| 99.50% - 99.00% | 25% |
| < 99.00% | 50% + termination right |

**A.3 Performance Monitoring**

Provider SHALL:
- Maintain real-time monitoring dashboards accessible to Client
- Provide API endpoints for Client's own monitoring integration
- Alert Client automatically on any SLA threshold breach
- Conduct monthly SLA performance reviews with Client

---

### Article 30(3)(b) — NOTICE PERIODS AND REPORTING

**B.1 Termination Notice Periods**

| Termination Type | Provider Notice | Client Notice |
|-----------------|-----------------|---------------|
| Without Cause | 180 days | 180 days |
| For Cause | 30 days | 30 days |
| Regulatory | Immediate | Immediate |
| Non-Renewal | 180 days | 90 days |

**B.2 Material Change Notice**

| Change Type | Advance Notice | Client Response Window |
|-------------|---------------|----------------------|
| Subcontractor Change | 60 days | 30 days |
| Data Location Change | 90 days | 45 days |
| Key Personnel Change | 30 days | 14 days |
| Security Architecture Change | 60 days | 30 days |
| Pricing Change | 180 days | 90 days |
| Service Scope Change | 90 days | 45 days |

**B.3 Reporting Obligations**

| Report | Frequency | Delivery | Format |
|--------|-----------|----------|--------|
| SLA Performance | Monthly | By 5th business day | PDF + JSON |
| Security Posture | Quarterly | By 15th of quarter | PDF |
| Incident Summary | Monthly | By 5th business day | PDF + JSON |
| Audit Status | Upon request | 5 business days | PDF |
| Penetration Test Results | Annually | Upon completion | PDF (sanitized) |
| SOC2 Report | Annually | Upon issuance | PDF |
| BCP/DR Test Results | Quarterly | Within 14 days | PDF |

---

### Article 30(3)(c) — PROVIDER BUSINESS CONTINGENCY PLANS

**C.1 Required Plans**

Provider SHALL maintain and make available for Client review:

| Plan | Description | Testing Frequency |
|------|-------------|------------------|
| Business Continuity Plan (BCP) | Overall continuity strategy | Annual review |
| Disaster Recovery Plan (DRP) | Technical recovery procedures | Semi-annual test |
| Incident Response Plan (IRP) | Security incident handling | Quarterly tabletop |
| Pandemic/Crisis Response Plan | Extended disruption handling | Annual review |
| Subcontractor Failure Plan | Key supplier contingency | Annual review |

**C.2 Plan Availability**

- Summary versions available on request (5 business days)
- Full plans available for audit under NDA (10 business days)
- Test results (sanitized) provided annually

**C.3 Recovery Objectives (Critical Functions)**

| Scenario | RTO | RPO | Validation |
|----------|-----|-----|------------|
| Single Component Failure | 30 minutes | 0 (real-time replication) | Monthly |
| Data Center Failure | 4 hours | 15 minutes | Semi-annually |
| Region Failure | 8 hours | 1 hour | Annually |
| Complete Infrastructure Loss | 24 hours | 4 hours | Annually |

---

### Article 30(3)(d) — PARTICIPATION IN CLIENT'S RESILIENCE TESTING

**D.1 Testing Participation Commitment**

Provider SHALL participate in Client's resilience testing including:

| Test Type | Provider Role | Frequency | Cost |
|-----------|---------------|-----------|------|
| Failover Testing | Active coordination | Semi-annually | Included |
| DR Drills | Execute provider procedures | Annually | Included |
| Tabletop Exercises | Participate in scenarios | Quarterly | Included |
| TLPT (if designated) | Full cooperation | Per NCA schedule | Predetermined rate |
| Communication Tests | Validate escalation paths | Quarterly | Included |

**D.2 TLPT Support (Threat-Led Penetration Testing)**

If Client is required to conduct TLPT per Article 26:
- Provider SHALL cooperate fully with designated testers
- Provider SHALL provide necessary access and documentation
- Provider SHALL not obstruct or interfere with testing
- Testing costs: [Predetermined rate per Annex C]
- Advance notice: Minimum 30 days (or per TLPT framework)

**D.3 Joint Testing Coordination**

- Designated testing coordinator: [Provider contact]
- Scheduling: Minimum 30 days advance notice
- Test windows: [Specified maintenance windows]
- Results sharing: Within 14 days of test completion

---

### Article 30(3)(e) — UNRESTRICTED AUDIT RIGHTS

**E.1 Client Audit Rights**

Client (and Client's designated auditors) SHALL have **unrestricted** rights to:

| Access Type | Scope | Notice Required | Frequency |
|-------------|-------|-----------------|-----------|
| Documentation Review | All relevant policies, procedures, logs | 5 business days | Unlimited |
| On-site Inspection | Provider premises | 5 business days | Annual + for cause |
| System Access | Read-only supervised access | 5 business days | As needed |
| Personnel Interviews | Key contacts | 10 business days | As needed |
| Subcontractor Audit Reports | All relevant third-party audits | 5 business days | Annual |

**E.2 NCA/Resolution Authority Access**

Client's competent authority and resolution authority SHALL have:
- All rights granted to Client under E.1
- Right to direct access without going through Client
- Right to conduct on-site inspections
- Right to interview Provider personnel
- Right to access any information necessary for supervision

**E.3 No Impediment to Supervision**

Provider SHALL NOT:
- Impose contractual terms that obstruct effective supervision
- Limit audit frequency without cause
- Restrict scope of audits unreasonably
- Charge excessive fees for audit support
- Delay or obstruct NCA access

**E.4 Pooled Audit Option**

Per Article 30(4), Client may rely on:
- Third-party certifications (SOC2 Type II, ISO 27001)
- Pooled audit reports arranged by Provider
- Joint audits with other financial entity clients

Provider SHALL support pooled audits by:
- Organizing annual third-party audit
- Making audit reports available to participating clients
- Coordinating joint audit requests

**E.5 Audit Logistics**

- Audit support contact: [Designated contact]
- Standard audit response: 5 business days
- Emergency/incident audit: 24 hours
- Audit documentation retention: 7 years

---

### Article 30(3)(f) — EXIT STRATEGY

**F.1 Exit Planning**

Provider SHALL maintain throughout the contract term:
- Documented exit plan reviewed annually
- Data export procedures tested semi-annually
- Knowledge transfer documentation
- Succession provider coordination procedures

**F.2 Transition Period**

| Exit Scenario | Transition Period | Support Level |
|---------------|------------------|---------------|
| Standard Termination | 180 days | Full support |
| For Cause Termination | 90 days | Full support |
| Regulatory Termination | Up to 180 days | Enhanced support |
| Provider Insolvency | Via escrow | Minimal (escrow) |

**F.3 Transition Assistance**

Provider SHALL provide:
- Full data export in agreed formats (5 business days)
- Knowledge transfer sessions (minimum 40 hours)
- API documentation for successor integration
- Parallel running support (up to 60 days)
- Post-transition support (30 days read-only access)

**F.4 No Lock-in Provisions**

- All data formats documented and non-proprietary
- No technical barriers to migration
- API specifications published and version-controlled
- Trained models exported in standard format (ONNX)

**F.5 Exit Testing**

Provider SHALL:
- Conduct annual exit plan testing
- Document test results and remediation
- Update exit plan based on system changes
- Notify Client of material exit plan changes

---

### Article 30(3)(g) — SUPERVISORY OVERSIGHT PARTICIPATION

**G.1 Supervisory Cooperation**

Provider SHALL cooperate with supervisory oversight activities including:
- Providing information for Client's regulatory reporting
- Supporting on-site inspections by NCAs
- Participating in supervisory stress testing if required
- Responding to regulatory information requests via Client

**G.2 Information Provision**

Upon Client or NCA request:
- Information request response: 5 business days
- Evidence collection: 10 business days
- System access arrangements: 5 business days

**G.3 Supervisory Contact**

If NCA contacts Provider directly:
- Provider SHALL notify Client within 24 hours
- Provider SHALL coordinate response with Client
- Provider SHALL not provide information that could harm Client without prior consultation

---

### Article 30(3)(h) — BUSINESS CONTINUITY IMPLEMENTATION

**H.1 BCP Requirements**

Provider SHALL implement:
- Redundant infrastructure (multi-AZ minimum)
- Real-time data replication for critical functions
- Automated failover capabilities
- Regular BCP testing with documented results

**H.2 Service Availability Guarantees**

For critical functions:
- No single point of failure in production systems
- Geographic redundancy for data storage
- Maximum 4-hour RTO for complete service recovery
- Maximum 15-minute RPO for data recovery

**H.3 BCP Testing Evidence**

Provider SHALL provide Client:
- Annual BCP test summary report
- Quarterly tabletop exercise summaries
- Semi-annual DR test results
- Remediation plans for any identified gaps

---

### Article 30(3)(i) — SECURITY ARRANGEMENTS

**I.1 Security Standards**

Provider SHALL maintain:
- SOC2 Type II certification (annual)
- ISO 27001 certification (target)
- Regular penetration testing (annual minimum)
- Vulnerability scanning (weekly)

**I.2 Security Measures**

| Control Area | Requirement | Validation |
|--------------|-------------|------------|
| Access Control | MFA, RBAC, privileged access management | Quarterly review |
| Encryption | AES-256 at rest, TLS 1.3 in transit | Annual pen test |
| Network Security | WAF, DDoS protection, IDS/IPS | Continuous monitoring |
| Endpoint Security | EDR on all systems | Continuous monitoring |
| Logging | Centralized, immutable audit logs | Daily review |

**I.3 Security Testing**

Provider SHALL conduct and share results of:
- Annual penetration testing (third-party)
- Quarterly vulnerability assessments
- Continuous security monitoring
- Red team exercises (as appropriate)

---

### Article 30(3)(j) — SUBCONTRACTING CONDITIONS

**J.1 Prior Approval Requirement**

For services supporting critical/important functions:

| Subcontracting Change | Approval Mode | Notice Period |
|----------------------|---------------|---------------|
| New critical subcontractor | Prior written consent | 60 days |
| Material scope change | Prior notification | 45 days |
| Location change (non-EU) | Prior written consent | 90 days |
| Subcontractor replacement | Prior notification | 30 days |

**J.2 Approval Workflow**

1. Provider submits Subcontracting Change Request (SCR)
2. Client reviews within approval window
3. If approved: Provider proceeds with change
4. If objected: Parties negotiate resolution within 30 days
5. If unresolved: Client may terminate affected services

**J.3 Subcontractor Chain Monitoring**

Provider SHALL:
- Maintain complete subcontractor chain documentation
- Monitor subcontractor performance continuously
- Report subcontractor incidents within 4 hours
- Ensure subcontractors meet equivalent security standards

**J.4 Flow-Down Requirements**

All subcontractor agreements SHALL include:
- Audit rights for Client and NCA
- Data location restrictions matching this Agreement
- Security requirements equivalent to this Agreement
- Termination rights on regulatory grounds

---

## CRITICAL FUNCTION DESIGNATION

**Client Declaration:**

The Client hereby confirms that the following Provider services support the Client's critical or important function(s) and are therefore subject to this Addendum:

| Service | Critical Function Supported | Article 3(22) Classification |
|---------|---------------------------|---------------------------|
| [Trading Platform] | [Algorithmic Trading Operations] | [Critical / Important] |
| [Market Data Services] | [Price Discovery] | [Critical / Important] |
| [Risk Management] | [Risk Oversight] | [Critical / Important] |

---

## ACCEPTANCE

This Addendum supplements and forms part of the ICT Service Agreement and DORA ICT Service Provisions (Art. 30(2)) between Provider and Client.

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
- Based on Regulation (EU) 2022/2554 Article 30(3)(a-j)
- Reviewed by: Legal, Compliance, Security, Operations

# Cybersecurity Framework

## AI-Powered Quantitative Research Platform

**Version**: 1.0
**Last Updated**: December 2024
**Framework**: NIST Cybersecurity Framework 2.0

---

## 1. Executive Summary

This document defines our cybersecurity framework aligned with NIST CSF 2.0, providing comprehensive security controls for our B2B SaaS trading platform. Our security program is designed to protect client trading strategies, support platform availability, and align with regulatory requirements.

**Security Architecture Foundation**: All security controls are built on the **Cloud-Controlled Execution Architecture (CCEA)**, which provides strict zone separation between research/monitoring (Cloud Zone) and live trading execution (Agent Zone).

---

## 1.1 CCEA Security Model

### Zone-Based Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLOUD ZONE (Research & Control)                      │
│                                                                              │
│  SECURITY DESIGN COMMITMENTS (enforced at architecture level):               │
│  ✗ NO trading credentials stored (secrets never enter Cloud)                │
│  ✗ NO order generation or transmission (no trading libs in build)           │
│  ✗ NO access to exchange trading endpoints (no broker APIs)                 │
│  ✗ NO order-like payloads (side/qty/price blocked in schema)                │
│  ✓ Mandatory telemetry redaction (secrets blocked in transmission)          │
│  ✓ Signed artifacts only (supply chain security)                            │
│                                                                              │
│  COMPONENTS: Research IDE, Backtesting, Artifact Builder, Control Plane     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Lifecycle Commands Only
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT ZONE (Execution & Secrets)                     │
│                                                                              │
│  SECURITY DESIGN COMMITMENTS (enforced locally):                             │
│  ✓ Secrets stored in local vault (HSM/KMS/keychain)                         │
│  ✓ All orders created and sent locally (never via Cloud)                    │
│  ✓ Hard caps enforced by Policy Firewall (cannot be bypassed)               │
│  ✓ Artifact signature verification required (tamper detection)              │
│  ✓ Local approval required for trading-impacting commands                   │
│                                                                              │
│  COMPONENTS: Local Vault, Policy Firewall, Live Loop, Broker Connector      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CCEA Threat Model

| Threat | Attack Vector | CCEA Mitigation |
|--------|---------------|-----------------|
| **RCE in Cloud** | Exploit in web app | Cloud has no trading libs, no broker APIs - blast radius limited |
| **Key Exfiltration** | Memory dump, logs | Keys never leave Agent zone, mandatory redaction in telemetry |
| **Artifact Tampering** | MITM, supply chain | Digest pinning + signature verification required |
| **Cloud as Execution** | Rogue insider | No order-like payloads allowed in protocol schema |
| **Replay Attacks** | Message replay | Idempotency keys + timestamps in all commands |
| **Unauthorized Trading** | Compromised Cloud | Local approval required for trading-impacting commands |

### CI/CD Security Guardrails

| Guardrail | Description | Enforcement |
|-----------|-------------|-------------|
| `no-trading-libs-in-cloud` | Cloud build cannot include order_execution modules | Build-time check |
| `no-broker-clients-in-cloud` | Cloud cannot import private trading clients | Import boundary check |
| `no-order-payloads-in-schema` | Schema prohibits side/qty/price fields | Schema validation |
| `artifact-signature-required` | All artifacts must be signed before deployment | Deployment gate |
| `redaction-enabled` | Telemetry must have redaction middleware active | Runtime check |

---

## 2. Framework Overview

### 2.1 Core Functions

```
┌─────────────────────────────────────────────────────────────────┐
│                    NIST CSF 2.0 CORE FUNCTIONS                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐                                                     │
│  │ GOVERN  │────────────────────────────────────────────────────│
│  └────┬────┘                                                     │
│       │                                                          │
│  ┌────▼────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│
│  │IDENTIFY │→ │ PROTECT │→ │ DETECT  │→ │ RESPOND │→ │ RECOVER ││
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘│
│                                                                  │
│  GOVERN:    Risk strategy, policy, oversight, supply chain      │
│  IDENTIFY:  Asset management, risk assessment, improvement      │
│  PROTECT:   Access control, training, data security             │
│  DETECT:    Monitoring, detection processes, analysis           │
│  RESPOND:   Planning, communications, mitigation                │
│  RECOVER:   Planning, improvements, communications              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation Tiers

| Tier | Description | Our Target |
|------|-------------|------------|
| Tier 1 | Partial - ad hoc, reactive | - |
| Tier 2 | Risk-informed - approved but not organization-wide | - |
| **Tier 3** | **Repeatable - formal, consistent across organization** | **Current Target** |
| Tier 4 | Adaptive - continuous improvement, predictive | Future Goal |

---

## 3. GOVERN Function (GV)

### 3.1 Organizational Context (GV.OC)

**GV.OC-01: Organizational Mission Understood**
- Security supports business objectives
- Risk tolerance defined by leadership
- Security integrated into business processes

**GV.OC-02: Internal Stakeholders Understood**
- Security team identified
- Roles and responsibilities documented
- Reporting structures defined

**GV.OC-03: Legal Requirements Determined**
- GDPR compliance
- MiFID II client requirements
- Contractual obligations

### 3.2 Risk Management Strategy (GV.RM)

**GV.RM-01: Risk Management Objectives**
- Annual risk assessment
- Risk appetite statement
- Risk treatment plans

**GV.RM-02: Risk Tolerance Established**

| Risk Category | Tolerance Level |
|---------------|-----------------|
| Data breach | Zero tolerance |
| Service unavailability | <4 hours/year |
| Financial loss | <€100K/incident |
| Regulatory non-compliance | Zero tolerance |

### 3.3 Cybersecurity Supply Chain Risk (GV.SC)

**GV.SC-01: Supply Chain Risk Managed**
- Vendor security assessment process
- Contractual security requirements
- Ongoing vendor monitoring

**Key Vendors**:

| Vendor | Service | Risk Level | Controls |
|--------|---------|------------|----------|
| AWS | Cloud hosting | High | SOC 2, ISO 27001, DPA |
| MongoDB Atlas | Database | High | SOC 2, encryption |
| Stripe | Payments | Medium | PCI DSS, SOC 2 |
| SendGrid | Email | Low | SOC 2, DPA |

---

## 4. IDENTIFY Function (ID)

### 4.1 Asset Management (ID.AM)

**ID.AM-01: Physical Devices Inventoried**

| Asset Type | Count | Management |
|------------|-------|------------|
| Cloud instances | Variable | AWS inventory |
| Endpoints (dev) | 5-10 | MDM solution |
| Network devices | Cloud-managed | AWS console |

**ID.AM-02: Software Platforms Inventoried**

| Category | Components | Documentation |
|----------|------------|---------------|
| Operating Systems | Ubuntu LTS, Alpine | CMDB |
| Databases | PostgreSQL, Redis, MongoDB | Architecture docs |
| Frameworks | Python, FastAPI, React | Tech stack docs |
| Third-party | See vendor list | Vendor register |

**ID.AM-03: Data Flows Mapped**

```
┌──────────────────────────────────────────────────────────────┐
│                      DATA FLOW DIAGRAM                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Client ─────► API Gateway ─────► Application ─────► Database│
│    │              │                    │              │       │
│    │          WAF/DDoS             Encryption      Encryption │
│    │          Protection           at Transit     at Rest     │
│    │              │                    │              │       │
│    └──────────────┴────────────────────┴──────────────┘       │
│                                                               │
│  External Data:                                               │
│  Exchanges ─────► Data Pipeline ─────► Storage ─────► Analysis│
│              │                    │            │              │
│          API Auth            Encryption    Access Control     │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Risk Assessment (ID.RA)

**ID.RA-01: Vulnerabilities Identified**
- Weekly automated vulnerability scans
- Annual penetration testing
- Continuous security monitoring

**ID.RA-02: Threat Intelligence**
- Subscribe to CVE feeds
- Monitor fintech threat landscape
- Industry information sharing

**ID.RA-03: Risk Assessment Process**

| Step | Frequency | Output |
|------|-----------|--------|
| Asset inventory | Continuous | Updated CMDB |
| Vulnerability scan | Weekly | Scan reports |
| Threat assessment | Quarterly | Threat register |
| Risk calculation | Annual | Risk register |
| Treatment planning | Annual | Action plans |

### 4.3 Improvement (ID.IM)

**ID.IM-01: Lessons Learned**
- Post-incident reviews
- Security metrics analysis
- Control effectiveness testing

---

## 5. PROTECT Function (PR)

### 5.1 Identity Management (PR.AA)

**PR.AA-01: Identities Managed**

| Control | Implementation |
|---------|----------------|
| Unique identities | All users have unique accounts |
| Service accounts | Documented, minimal privileges |
| No shared accounts | Policy enforced |

**PR.AA-02: Authentication**

| Control | Implementation |
|---------|----------------|
| Multi-factor | Required for all users |
| Password policy | 12+ chars, complexity, 90-day rotation |
| Brute force protection | Account lockout after 5 failures |
| Session management | 8-hour timeout, single session |

**PR.AA-03: Access Control**

| Principle | Implementation |
|-----------|----------------|
| Least privilege | Role-based access control |
| Need to know | Data classification enforcement |
| Separation of duties | Dev/Ops/Admin separation |
| Access reviews | Quarterly automated reviews |

### 5.2 Awareness and Training (PR.AT)

**PR.AT-01: Security Training**

| Training | Audience | Frequency |
|----------|----------|-----------|
| Security fundamentals | All staff | Onboarding |
| Phishing awareness | All staff | Quarterly |
| Secure development | Developers | Onboarding + annual |
| Incident response | Response team | Semi-annual |
| GDPR awareness | All staff | Annual |

**PR.AT-02: Privileged User Training**
- Enhanced training for administrators
- Hands-on incident response exercises
- Regular tabletop exercises

### 5.3 Data Security (PR.DS)

**PR.DS-01: Data at Rest Protected**

| Data Type | Encryption | Key Management |
|-----------|------------|----------------|
| Database | AES-256 | AWS KMS |
| File storage | AES-256 | AWS KMS |
| Backups | AES-256 | AWS KMS |
| Logs | AES-256 | AWS KMS |

**PR.DS-02: Data in Transit Protected**

| Connection | Protocol | Minimum Version |
|------------|----------|-----------------|
| Client → API | TLS | 1.3 |
| Internal services | TLS | 1.2 |
| Database connections | TLS | 1.2 |
| Admin access | TLS + VPN | 1.3 |

**PR.DS-03: Data Lifecycle Management**
- Data classification scheme
- Retention policies
- Secure deletion procedures
- Backup and recovery

### 5.4 Platform Security (PR.PS)

**PR.PS-01: Configuration Management**
- Infrastructure as Code (Terraform)
- Configuration baselines
- Change management process
- Regular compliance scans

**PR.PS-02: Software Development Lifecycle**

| Phase | Security Activity |
|-------|-------------------|
| Requirements | Security requirements |
| Design | Threat modeling |
| Development | Secure coding standards |
| Testing | Security testing (SAST/DAST) |
| Deployment | Security gates |
| Operations | Vulnerability management |

**PR.PS-03: Change Management**
- Formal change request process
- Testing requirements
- Approval workflows
- Rollback procedures

### 5.5 Technology Infrastructure Resilience (PR.IR)

**PR.IR-01: Network Security**

| Control | Implementation |
|---------|----------------|
| Firewalls | AWS Security Groups, NACLs |
| Network segmentation | VPC with private subnets |
| DDoS protection | AWS Shield |
| WAF | AWS WAF with OWASP rules |
| Intrusion detection | AWS GuardDuty |

**PR.IR-02: Recovery Capabilities**
- RTO: 4 hours
- RPO: 1 hour
- Daily backups with 35-day retention
- Cross-region replication
- Annual DR testing

---

## 6. DETECT Function (DE)

### 6.1 Continuous Monitoring (DE.CM)

**DE.CM-01: Network Monitoring**

| Source | Tool | Alerting |
|--------|------|----------|
| VPC Flow Logs | AWS CloudWatch | Real-time |
| DNS logs | Route 53 Resolver | Real-time |
| WAF logs | AWS WAF | Real-time |
| ALB logs | S3 + Athena | Batch analysis |

**DE.CM-02: Security Event Monitoring**

| Event Type | Detection Method | Response Time |
|------------|------------------|---------------|
| Brute force | Account lockout | Immediate |
| Privilege escalation | CloudTrail alerts | <15 min |
| Unusual API calls | GuardDuty | <15 min |
| Data exfiltration | DLP alerts | <30 min |

**DE.CM-03: Vulnerability Monitoring**
- Weekly automated scans (Nessus/Qualys)
- Daily dependency scanning (Snyk)
- Real-time CVE monitoring

### 6.2 Adverse Event Analysis (DE.AE)

**DE.AE-01: Event Correlation**
- AWS Security Hub aggregation
- Cross-service correlation
- Anomaly detection

**DE.AE-02: Alert Prioritization**

| Severity | Examples | Response SLA |
|----------|----------|--------------|
| Critical | Active attack, data breach | 15 min |
| High | Malware, privilege escalation | 1 hour |
| Medium | Policy violation, suspicious activity | 4 hours |
| Low | Minor anomalies | 24 hours |

---

## 7. RESPOND Function (RS)

### 7.1 Incident Management (RS.MA)

**RS.MA-01: Incident Response Plan**

```
┌──────────────────────────────────────────────────────────────┐
│                 INCIDENT RESPONSE PHASES                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. PREPARATION                                               │
│     • Response team identified                                │
│     • Playbooks documented                                    │
│     • Tools and access ready                                  │
│     • Regular training conducted                              │
│                                                               │
│  2. DETECTION & ANALYSIS                                      │
│     • Alert triage                                            │
│     • Initial assessment                                      │
│     • Severity classification                                 │
│     • Stakeholder notification                                │
│                                                               │
│  3. CONTAINMENT                                               │
│     • Short-term: Limit immediate damage                      │
│     • Evidence preservation                                   │
│     • Long-term: Prevent spread                               │
│                                                               │
│  4. ERADICATION                                               │
│     • Remove threat actors                                    │
│     • Patch vulnerabilities                                   │
│     • Strengthen defenses                                     │
│                                                               │
│  5. RECOVERY                                                  │
│     • Restore systems                                         │
│     • Validate security                                       │
│     • Monitor for recurrence                                  │
│                                                               │
│  6. POST-INCIDENT                                             │
│     • Root cause analysis                                     │
│     • Lessons learned                                         │
│     • Documentation update                                    │
│     • Control improvements                                    │
└──────────────────────────────────────────────────────────────┘
```

**RS.MA-02: Incident Classification**

| Class | Description | Examples |
|-------|-------------|----------|
| Security | Confidentiality/integrity breach | Data breach, unauthorized access |
| Availability | Service disruption | Outage, degradation |
| Compliance | Regulatory violation | GDPR breach |
| Financial | Financial impact | Fraud, theft |

### 7.2 Incident Analysis (RS.AN)

**RS.AN-01: Investigation Process**
1. Preserve evidence
2. Document timeline
3. Identify root cause
4. Assess impact
5. Determine scope

**RS.AN-02: Forensic Capabilities**
- Log retention (1 year production, 7 years audit)
- Immutable audit trails
- External forensics retainer (on call)

### 7.3 Incident Response Reporting (RS.CO)

**RS.CO-01: Internal Communications**

| Severity | Notification | Escalation |
|----------|--------------|------------|
| Critical | Immediate | CEO, all stakeholders |
| High | 1 hour | Management |
| Medium | 4 hours | Team leads |
| Low | Daily summary | N/A |

**RS.CO-02: External Communications**
- Regulatory notification (72 hours for GDPR)
- Client notification (as contractually required)
- Public disclosure (if required by law)

### 7.4 Incident Mitigation (RS.MI)

**RS.MI-01: Containment Procedures**
- Account suspension
- Network isolation
- Service shutdown (if needed)
- Evidence preservation

**RS.MI-02: Eradication Procedures**
- Malware removal
- Credential rotation
- System reimaging
- Vulnerability patching

---

## 8. RECOVER Function (RC)

### 8.1 Recovery Planning (RC.RP)

**RC.RP-01: Recovery Procedures**

| System | RTO | RPO | Procedure |
|--------|-----|-----|-----------|
| Production | 4h | 1h | Failover to DR region |
| Database | 4h | 1h | Restore from replica |
| Authentication | 2h | 0 | Failover to standby |
| Monitoring | 1h | 1h | Re-deploy from IaC |

**RC.RP-02: Backup Procedures**
- Daily automated backups
- 35-day retention
- Cross-region replication
- Monthly restore testing

### 8.2 Recovery Execution (RC.EX)

**RC.EX-01: Communication During Recovery**
- Status page updates
- Client notifications
- Internal status calls

**RC.EX-02: Recovery Validation**
- Functional testing
- Security verification
- Performance baseline comparison

### 8.3 Recovery Improvements (RC.IM)

**RC.IM-01: Post-Incident Improvements**
- Blameless post-mortems
- Control enhancements
- Documentation updates
- Training adjustments

---

## 9. Security Metrics

### 9.1 Key Performance Indicators

| Metric | Target | Measurement |
|--------|--------|-------------|
| Mean time to detect (MTTD) | <15 min | Security monitoring |
| Mean time to respond (MTTR) | <4 hours | Incident tracking |
| Vulnerability remediation (critical) | <24 hours | Vuln management |
| Vulnerability remediation (high) | <7 days | Vuln management |
| Security training completion | 100% | LMS tracking |
| Phishing test failure rate | <5% | Phishing campaigns |
| Unplanned downtime | <4 hours/year | Uptime monitoring |

### 9.2 Reporting

| Report | Frequency | Audience |
|--------|-----------|----------|
| Security dashboard | Real-time | Security team |
| Monthly security report | Monthly | Management |
| Quarterly risk report | Quarterly | Board/investors |
| Annual security review | Annually | All stakeholders |

---

## 10. Framework Maintenance

### 10.1 Review Schedule

| Activity | Frequency |
|----------|-----------|
| Framework review | Annual |
| Risk assessment | Annual |
| Policy review | Annual |
| Control testing | Quarterly |
| Incident response testing | Semi-annual |
| DR testing | Annual |

### 10.2 Continuous Improvement

- Regular gap assessments
- Industry benchmark comparisons
- Adoption of emerging best practices
- Technology updates

---

## Appendix A: Related Documents

| Document | Purpose |
|----------|---------|
| Data Protection Policy | GDPR compliance details |
| Incident Response Plan | Detailed IR procedures |
| Business Continuity Plan | BCP/DR procedures |
| Access Control Policy | Detailed access procedures |
| Acceptable Use Policy | Employee guidelines |
| Vendor Management Policy | Third-party security |

---

## Appendix B: Compliance Mapping

| NIST CSF | SOC 2 TSC | ISO 27001 | GDPR |
|----------|-----------|-----------|------|
| GV | CC1, CC2 | 5.1-5.3 | Art. 24 |
| ID.AM | CC6.1 | A.8 | Art. 30 |
| ID.RA | CC3, CC4 | 6.1, 8.2 | Art. 35 |
| PR.AA | CC6.1-6.3 | A.9 | Art. 32 |
| PR.DS | CC6.7 | A.10, A.13 | Art. 32 |
| DE.CM | CC7 | A.12.4 | Art. 32 |
| RS.MA | CC7.4 | A.16 | Art. 33-34 |
| RC.RP | A1.2 | A.17 | Art. 32 |

---

**Document Owner**: Security Team
**Review Cycle**: Annual
**Next Review**: December 2025

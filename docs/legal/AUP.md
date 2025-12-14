# Acceptable Use Policy (AUP)

**AI-Powered Quantitative Research Platform**

**Effective Date:** December 2024
**Version:** 1.0.0

**Architecture:** Cloud-Controlled Execution Architecture (CCEA)

---

## 1. PURPOSE

This Acceptable Use Policy ("AUP") defines the acceptable and prohibited uses of the AI-Powered Quantitative Research Platform ("Platform"). By using the Platform, you agree to comply with this AUP.

This AUP is incorporated by reference into the Terms of Service.

---

## 2. CCEA ARCHITECTURE OVERVIEW

The Platform implements **Cloud-Controlled Execution Architecture (CCEA)**, which strictly separates:

| Zone | Function | Your Usage |
|------|----------|------------|
| **Cloud** | Research, backtesting, monitoring, lifecycle management | Compute resources for your strategies |
| **Agent** | Live execution, credential storage (user-operated) | Local execution on your hardware |

**Key Principle:** Cloud resources are for **research and monitoring only** - never for order execution or credential storage.

---

## 3. ACCEPTABLE USES

You **MAY** use the Platform for:

### 3.1 Research and Development
- Developing, testing, and refining trading strategies
- Running backtests against historical market data
- Performing quantitative research and analysis
- Training machine learning models for trading applications
- Simulating trading strategies (paper trading)

### 3.2 Monitoring and Analytics
- Monitoring performance metrics of deployed strategies
- Analyzing trading results and generating reports
- Receiving alerts and notifications about strategy performance
- Reviewing aggregated telemetry from your local Agent (if deployed)

### 3.3 Lifecycle Management
- Managing strategy deployments via the control plane
- Configuring strategy parameters (non-secret configurations)
- Scheduling backtest jobs and research tasks
- Accessing strategy artifacts and build outputs

### 3.4 Educational and Informational
- Learning about algorithmic trading concepts
- Testing trading hypotheses
- Understanding market dynamics through simulation

---

## 4. PROHIBITED USES

You **MUST NOT** use the Platform for:

### 4.1 Cloud Compute Abuse

**Resource Abuse:**
- Cryptocurrency mining or proof-of-work computations
- Distributed denial-of-service (DDoS) attacks or participation in botnets
- Running unauthorized services (proxies, VPNs, torrents)
- Hosting content unrelated to quantitative research
- Intentionally consuming excessive resources to degrade service for others

**Security Violations:**
- Attempting to bypass authentication or authorization controls
- Probing, scanning, or testing vulnerabilities of Platform systems
- Attempting to access other users' data, strategies, or resources
- Reverse engineering, decompiling, or disassembling Platform software
- Introducing malware, viruses, or malicious code

### 4.2 Protocol and Architecture Violations

**CCEA Boundary Violations:**
- Attempting to transmit broker API credentials to Cloud
- Attempting to trigger order execution from Cloud
- Circumventing telemetry redaction mechanisms
- Injecting order-like payloads (side/quantity/price) into Cloud communications
- Attempting to use Cloud for real-time trading signals intended to circumvent the Agent

**Data Integrity:**
- Manipulating or falsifying backtest results
- Submitting false or misleading telemetry data
- Corrupting shared datasets or research artifacts

### 4.3 Financial and Regulatory Violations

**Market Abuse:**
- Developing strategies designed for market manipulation (spoofing, layering, wash trading)
- Using the Platform to facilitate insider trading
- Creating strategies that violate exchange rules or regulations
- Developing tools for front-running or other unfair trading practices

**Misrepresentation:**
- Claiming the Platform provides investment advice
- Representing Platform outputs as professional financial recommendations
- Misleading others about the nature of AI-generated trading signals

### 4.4 Legal and Ethical Violations

**Illegal Activities:**
- Any activity that violates applicable laws or regulations
- Money laundering or terrorist financing
- Sanctions evasion
- Fraud or deception

**Harmful Content:**
- Storing or processing content that is illegal, defamatory, or infringes third-party rights
- Using Platform resources for harassment or abuse
- Discriminatory practices or hate-related activities

### 4.5 Network and Infrastructure Abuse

- Excessive API calls beyond documented rate limits
- Automated scraping or harvesting of Platform data
- Creating multiple accounts to circumvent limits
- Interfering with Platform availability or performance
- Exploiting Platform vulnerabilities for unauthorized access

---

## 5. CLOUD COMPUTE FAIR USE

### 5.1 Resource Limits

Cloud research resources are subject to fair use limits:

| Resource | Fair Use Limit | Overage Policy |
|----------|---------------|----------------|
| Backtest CPU-hours | Per plan allocation | Queued or throttled |
| Concurrent jobs | Per plan limit | Queued |
| Storage | Per plan allocation | Warning, then cleanup required |
| API requests | Per plan rate limits | HTTP 429 (Too Many Requests) |

### 5.2 Job Isolation

All research jobs run in isolated sandboxes with:
- No network egress except to approved data sources
- Resource quotas (CPU, memory, time limits)
- Automatic termination on quota exceeded
- Monitoring for abuse patterns

### 5.3 Abuse Detection

We monitor for abuse patterns including:
- Cryptocurrency mining signatures
- Botnet communication patterns
- Unauthorized network scanning
- Resource abuse beyond fair use

**Detection triggers immediate investigation and potential suspension.**

---

## 6. ENFORCEMENT

### 6.1 Investigation

We reserve the right to investigate suspected violations of this AUP. Investigation may include:
- Reviewing resource usage patterns
- Analyzing job outputs and behaviors
- Examining network traffic metadata
- Reviewing audit logs

### 6.2 Remedial Actions

Upon confirmed violation, we may take one or more of the following actions:

| Severity | Typical Response |
|----------|-----------------|
| **Minor** | Warning, education, request to modify behavior |
| **Moderate** | Temporary suspension, resource restriction, required remediation |
| **Severe** | Immediate suspension, account termination, legal action |
| **Critical** | Immediate termination, report to authorities, legal action |

### 6.3 Due Process

Before taking remedial action (except in emergencies), we will:
1. Notify you of the suspected violation
2. Provide opportunity to respond (typically 5 business days)
3. Consider your response in determining action
4. Notify you of the action taken

**Emergency Exception:** We may act immediately without prior notice to prevent imminent harm to the Platform, other users, or third parties.

### 6.4 Appeals

You may appeal enforcement actions by contacting legal@[company-domain].com within 30 days. Appeals will be reviewed by personnel not involved in the original decision.

---

## 7. REPORTING VIOLATIONS

### 7.1 How to Report

If you become aware of AUP violations, please report them to:
- **Email:** abuse@[company-domain].com
- **Security Issues:** security@[company-domain].com

### 7.2 Report Contents

Helpful reports include:
- Description of the suspected violation
- Evidence or indicators (if available)
- Impact (if known)
- Your contact information (optional but helpful)

### 7.3 No Retaliation

We do not retaliate against good-faith reports of AUP violations.

---

## 8. CHANGES TO THIS POLICY

### 8.1 Notification

We may update this AUP as needed. Material changes will be:
- Announced via email to registered users
- Posted on the Platform with a prominent notice
- Effective 30 days after notification (except for changes required by law)

### 8.2 Continued Use

Continued use of the Platform after changes become effective constitutes acceptance of the modified AUP.

---

## 9. CONTACT

For questions about this AUP:
- **Email:** legal@[company-domain].com
- **Abuse Reports:** abuse@[company-domain].com

---

## APPENDIX A: EXAMPLES OF PROHIBITED ACTIVITIES

### Cloud Compute Abuse Examples

| Activity | Why Prohibited |
|----------|---------------|
| Running Bitcoin mining scripts in backtest jobs | Resource abuse, unrelated to trading research |
| Setting up a proxy server on Platform infrastructure | Unauthorized service, network abuse |
| Running port scanners against external targets | Security violation, potential legal liability |
| Creating 100 accounts to bypass compute limits | Terms violation, resource abuse |

### CCEA Violation Examples

| Activity | Why Prohibited |
|----------|---------------|
| Embedding API keys in strategy code uploaded to Cloud | Credential exposure, architecture violation |
| Creating a "signal service" that sends buy/sell messages from Cloud | Order-like payload violation |
| Attempting to disable telemetry redaction | Security bypass, architecture violation |
| Using Cloud APIs to directly connect to broker endpoints | Boundary violation |

### Financial Abuse Examples

| Activity | Why Prohibited |
|----------|---------------|
| Developing spoofing strategies (placing orders with intent to cancel) | Market manipulation |
| Using Platform to coordinate pump-and-dump schemes | Securities fraud |
| Claiming backtest results are guaranteed future returns | Misrepresentation |

---

## APPENDIX B: RELATED DOCUMENTS

| Document | Purpose |
|----------|---------|
| [Terms of Service](TERMS_OF_SERVICE.md) | Full service agreement |
| [Privacy Policy](PRIVACY_POLICY.md) | Data handling practices |
| [CCEA Overview](../CCEA_OVERVIEW.md) | Architecture details |
| [Cloud Governance](../cloud/GOVERNANCE.md) | Cloud resource policies |

---

**Last Updated:** December 2024
**Document Version:** 1.0.0

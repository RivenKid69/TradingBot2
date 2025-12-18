# Acceptable Use Policy (AUP)

**CustodiaCloud**

**Effective Date:** December 2024
**Version:** 1.0.0

**Architecture:** Cloud-Controlled Execution Architecture (CCEA)

---

## 1. PURPOSE

This Acceptable Use Policy ("AUP") defines the permitted and prohibited uses of CustodiaCloud ("Platform"). This policy protects our users, systems, and the integrity of financial markets.

**By using the Platform, you agree to comply with this AUP.**

---

## 2. SCOPE

This AUP applies to:

- All users of the Cloud services (research, backtesting, monitoring)
- All users operating local Agents for live execution
- All API access to Platform services
- All strategies developed or deployed using the Platform

---

## 3. PERMITTED USES

### 3.1 Research and Development

You MAY use the Platform for:

| Use Case | Cloud | Agent |
|----------|-------|-------|
| Developing trading strategies | Yes | N/A |
| Backtesting against historical data | Yes | N/A |
| Paper trading (simulation) | Yes | Yes |
| Analyzing market data | Yes | Yes |
| Educational purposes | Yes | Yes |

### 3.2 Live Execution (Agent Required)

You MAY use the Platform for live execution when:

- You operate your own local Agent
- You use your own broker accounts and credentials
- Your trading complies with applicable laws and broker terms
- You maintain appropriate risk controls

### 3.3 Automation

You MAY automate:

| Activity | Permitted | Notes |
|----------|-----------|-------|
| Strategy execution | Yes | Via local Agent |
| Position management | Yes | Via local Agent |
| Risk monitoring | Yes | Via Agent or Cloud |
| Data collection | Yes | Within rate limits |
| Report generation | Yes | Via Cloud API |

---

## 4. PROHIBITED USES

### 4.1 Market Manipulation

**STRICTLY PROHIBITED:**

| Violation | Description | Severity |
|-----------|-------------|----------|
| Spoofing | Placing orders intended to be cancelled | Critical |
| Layering | Creating false impression of supply/demand | Critical |
| Wash trading | Trading with yourself to create false volume | Critical |
| Pump and dump | Artificially inflating then selling | Critical |
| Front-running | Trading ahead of known orders | Critical |
| Quote stuffing | Overwhelming market with orders | Critical |

**Consequences:** Immediate account termination, reporting to regulators.

### 4.2 Illegal Activities

**PROHIBITED:**

- Money laundering or terrorist financing
- Sanctions violations
- Tax evasion schemes
- Fraud of any kind
- Trading on material non-public information (insider trading)
- Operating without required licenses in your jurisdiction

### 4.3 System Abuse

**PROHIBITED:**

| Violation | Description | Rate Limits |
|-----------|-------------|-------------|
| DDoS attacks | Overwhelming our systems | Auto-blocked |
| Credential stuffing | Brute-force authentication | 10 attempts/min |
| Scraping | Unauthorized data extraction | Per API limits |
| Resource exhaustion | Consuming excessive resources | Per plan limits |
| Circumventing limits | Bypassing rate limits | Immediate ban |

### 4.4 Research Job Abuse

**PROHIBITED in Cloud research environment:**

| Violation | Description | Detection |
|-----------|-------------|-----------|
| Crypto mining | Using compute for mining | Anomaly detection |
| Malware hosting | Deploying malicious code | Code scanning |
| Network attacks | Using Cloud to attack others | Network monitoring |
| Data exfiltration | Stealing data from Cloud | Egress monitoring |
| Excessive compute | Beyond fair-use limits | Resource quotas |

### 4.5 Protocol Violations

**PROHIBITED:**

- Attempting to send order payloads through Cloud (blocked by design)
- Attempting to extract credentials from Cloud (never stored)
- Tampering with protocol signatures
- Replaying commands with modified payloads
- Bypassing local approval requirements

### 4.6 Multi-Accounting Abuse

**PROHIBITED:**

| Violation | Description |
|-----------|-------------|
| Free tier abuse | Multiple accounts for free resources |
| Ban evasion | New accounts after termination |
| Trial abuse | Repeated trials with different accounts |
| Quota circumvention | Multiple accounts to bypass limits |

---

## 5. RESOURCE LIMITS

### 5.1 Cloud Resource Quotas

| Resource | Free Tier | Standard | Professional |
|----------|-----------|----------|--------------|
| Backtests/day | 10 | 100 | Unlimited |
| Concurrent jobs | 1 | 5 | 20 |
| Data retention | 30 days | 1 year | 5 years |
| API calls/min | 60 | 300 | 1000 |
| Storage | 1 GB | 10 GB | 100 GB |

### 5.2 Fair Use

Even with "unlimited" plans:

- Resources must be used for legitimate trading research
- Sustained 100% utilization may be throttled
- Automated abuse detection applies to all tiers

### 5.3 Agent Rate Limits

| Operation | Rate Limit | Notes |
|-----------|------------|-------|
| Heartbeat | 1/30 sec | Required |
| Telemetry | 10/sec | Burst: 100 |
| Command poll | 1/sec | Long-poll preferred |
| Artifact download | 10/hour | Per deployment |

---

## 6. SECURITY REQUIREMENTS

### 6.1 Account Security

**REQUIRED:**

- Strong, unique passwords (min 12 characters)
- Two-factor authentication (required for live trading)
- Regular credential rotation (recommended: 90 days)
- Secure API key storage

### 6.2 Agent Security

**REQUIRED for Agent operators:**

| Requirement | Description |
|-------------|-------------|
| Secure hosting | Agent on secured infrastructure |
| Firewall | Restrict Agent network access |
| Updates | Keep Agent software current |
| Monitoring | Monitor for anomalies |
| Backups | Backup Agent configuration |

### 6.3 Broker Credential Security

**REQUIRED:**

- Use trade-only API keys (no withdrawal permissions)
- Never share credentials with third parties
- Rotate broker keys if compromised
- Set appropriate rate limits at broker

---

## 7. COMPLIANCE REQUIREMENTS

### 7.1 Your Regulatory Obligations

You are responsible for:

| Jurisdiction | Typical Requirements |
|--------------|---------------------|
| United States | FINRA rules, SEC regulations, state licenses |
| European Union | MiFID II compliance, national regulations |
| United Kingdom | FCA rules and authorizations |
| Other | Local financial regulations |

**We do NOT provide regulatory compliance services.** Consult with legal counsel.

### 7.2 Tax Obligations

You are responsible for:

- Reporting all trading gains and losses
- Paying applicable taxes
- Maintaining required records
- Complying with tax reporting deadlines

### 7.3 Broker Compliance

You must comply with:

- Your broker's terms of service
- API usage policies
- Position and margin requirements
- Market-specific rules

---

## 8. MONITORING AND ENFORCEMENT

### 8.1 What We Monitor

| Activity | Monitoring Method | Purpose |
|----------|------------------|---------|
| API usage | Rate counters | Abuse prevention |
| Research jobs | Resource metrics | Fair use |
| Authentication | Login patterns | Security |
| Protocol messages | Schema validation | Security |
| Telemetry | Anomaly detection | Abuse detection |

**We do NOT monitor:**
- Your Agent's local activities
- Your broker connections (we have no access)
- Your trading performance (only redacted telemetry)

### 8.2 Abuse Detection

Automated systems detect:

- Unusual resource consumption patterns
- Protocol violations
- Rate limit circumvention
- Suspicious account activity
- Known attack signatures

### 8.3 Enforcement Actions

| Severity | Actions |
|----------|---------|
| Warning | Email notification, usage guidance |
| Throttling | Temporary rate limit reduction |
| Suspension | Account suspended pending review |
| Termination | Permanent account closure |
| Legal | Referral to law enforcement or regulators |

### 8.4 Appeal Process

If you believe enforcement was in error:

1. Contact: abuse-appeals@[company-domain].com
2. Provide: Account ID, incident description, supporting evidence
3. Timeline: Review within 5 business days
4. Decision: Final determination provided in writing

---

## 9. REPORTING ABUSE

### 9.1 How to Report

If you observe AUP violations:

- **Email:** abuse@[company-domain].com
- **Security issues:** security@[company-domain].com
- **Market manipulation:** Include evidence if possible

### 9.2 Confidentiality

Reports are handled confidentially. We do not disclose reporter identity without consent or legal requirement.

### 9.3 No Retaliation

We prohibit retaliation against good-faith reports of AUP violations.

---

## 10. CHANGES TO THIS POLICY

### 10.1 Notification

Material changes will be communicated:

- Via email to registered users
- Through Platform announcements
- At least 14 days before taking effect

### 10.2 Version History

| Version | Date | Summary |
|---------|------|---------|
| 1.0.0 | December 2024 | Initial release |

---

## 11. CONTACT

### 11.1 General Inquiries

- **Email:** legal@[company-domain].com
- **Website:** [Company Website]

### 11.2 Abuse Reports

- **Email:** abuse@[company-domain].com
- **Response:** Within 24 hours for critical issues

---

## APPENDIX A: MARKET MANIPULATION DEFINITIONS

### Spoofing
Placing large orders with intent to cancel before execution to create false impression of market interest.

### Layering
Placing multiple orders at different price levels to create illusion of supply/demand, then cancelling.

### Wash Trading
Executing trades where the buyer and seller are the same party, creating artificial volume.

### Pump and Dump
Artificially inflating an asset's price through coordinated buying or misleading statements, then selling.

### Front-Running
Trading ahead of a known pending order to profit from the expected price movement.

### Quote Stuffing
Rapidly entering and withdrawing large numbers of orders to slow down other market participants.

---

## APPENDIX B: CCEA-SPECIFIC RULES

### B.1 Cloud-Only Users

If you use only Cloud services (research, backtesting):

| Rule | Description |
|------|-------------|
| No live trading | Cloud cannot execute orders |
| Resource limits apply | Per your plan |
| Data is yours | Export anytime |

### B.2 Agent Operators

If you operate a local Agent:

| Rule | Description |
|------|-------------|
| Your responsibility | Agent security, broker compliance |
| Our services | Artifact delivery, monitoring, lifecycle |
| Hard caps | Your local limits are sovereign |
| Telemetry | Redaction is mandatory, verbosity is optional |

### B.3 Enterprise Customers

If you self-host Cloud and/or Agent:

| Rule | Description |
|------|-------------|
| Your infrastructure | Your responsibility |
| License terms | Per enterprise agreement |
| Support | Per SLA |
| Auditing | Your internal policies |

---

**Last Updated:** December 2024
**Document Version:** 1.0.0

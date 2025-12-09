# Privacy Policy

**AI-Powered Quantitative Research Platform**

**Effective Date:** December 2024
**Version:** 1.0.0

**Legal Framework:** General Data Protection Regulation (EU) 2016/679 (GDPR)

---

## 1. DATA CONTROLLER

### 1.1 Identity

The data controller responsible for your personal data is:

**[Company Name]**
- **Registered Address:** [Address]
- **Registration Number:** [Number]
- **Country:** Netherlands (EU)

### 1.2 Contact Information

For privacy-related inquiries:

- **Email:** privacy@[company-domain].com
- **Data Protection Officer (DPO):** dpo@[company-domain].com
- **Postal Address:** [Company Address]

### 1.3 EU Representative

For Users outside the EU, our EU representative is:
- [Representative details if applicable]

---

## 2. DATA WE COLLECT

### 2.1 Account Data

When you create an account, we collect:

| Data Type | Purpose | Legal Basis |
|-----------|---------|-------------|
| Email address | Account identification, communication | Contract (Art. 6(1)(b)) |
| Name (optional) | Personalization | Consent (Art. 6(1)(a)) |
| Password (hashed) | Authentication | Contract (Art. 6(1)(b)) |
| Account creation date | Record keeping | Contract (Art. 6(1)(b)) |

### 2.2 Trading Data

When you use trading features, we process:

| Data Type | Purpose | Legal Basis |
|-----------|---------|-------------|
| Strategies | Service provision, backtesting | Contract (Art. 6(1)(b)) |
| Backtest results | Performance analysis | Contract (Art. 6(1)(b)) |
| Execution logs | Order tracking, audit trail | Contract + Legal obligation |
| Position data | Risk management | Contract (Art. 6(1)(b)) |

### 2.3 Broker Credentials

When you connect broker accounts:

| Data Type | Purpose | Legal Basis | Protection |
|-----------|---------|-------------|------------|
| API Key | Order execution | Contract (Art. 6(1)(b)) | AES-256-GCM encryption |
| API Secret | Authentication | Contract (Art. 6(1)(b)) | AES-256-GCM encryption |
| Passphrase (if applicable) | Additional security | Contract (Art. 6(1)(b)) | AES-256-GCM encryption |

**Security Note:** Broker credentials are encrypted at rest using AES-256-GCM encryption with per-user derived keys. They are only decrypted in memory when required for order execution.

### 2.4 Technical Data

We automatically collect:

| Data Type | Purpose | Legal Basis |
|-----------|---------|-------------|
| IP address | Security, geo-verification | Legitimate interest (Art. 6(1)(f)) |
| Device information | Service optimization | Legitimate interest (Art. 6(1)(f)) |
| Browser type | Compatibility | Legitimate interest (Art. 6(1)(f)) |
| Access timestamps | Security monitoring | Legitimate interest (Art. 6(1)(f)) |
| Error logs | Debugging, service improvement | Legitimate interest (Art. 6(1)(f)) |

### 2.5 Usage Analytics

With your consent, we may collect:

| Data Type | Purpose | Legal Basis |
|-----------|---------|-------------|
| Feature usage | Service improvement | Consent (Art. 6(1)(a)) |
| Performance metrics | Optimization | Consent (Art. 6(1)(a)) |
| User interactions | UX improvement | Consent (Art. 6(1)(a)) |

---

## 3. LEGAL BASIS FOR PROCESSING

### 3.1 Contract Performance (Article 6(1)(b))

We process data necessary to provide our services:

- Account management
- Strategy storage and execution
- Backtest computation
- Order execution via broker APIs
- Customer support

### 3.2 Legitimate Interest (Article 6(1)(f))

We process data for our legitimate business interests:

- **Security**: Protecting the platform and users from unauthorized access
- **Fraud prevention**: Detecting and preventing malicious activities
- **Service improvement**: Analyzing usage patterns to improve functionality
- **Communication**: Sending service-related notifications

**Balancing Test:** We have conducted legitimate interest assessments to ensure our interests do not override your rights and freedoms.

### 3.3 Consent (Article 6(1)(a))

For optional processing, we obtain your explicit consent:

- Marketing communications
- Analytics cookies
- Optional data collection

You can withdraw consent at any time through account settings.

### 3.4 Legal Obligation (Article 6(1)(c))

We may process data to comply with legal requirements:

- Financial regulations (audit trails)
- Tax reporting obligations
- Law enforcement requests (with valid legal process)
- Regulatory inquiries

---

## 4. DATA RETENTION

### 4.1 Retention Periods

| Data Category | Retention Period | Reason |
|---------------|------------------|--------|
| Account data | Until deletion request + 30 days | Account recovery grace period |
| Strategies | Until user deletes or account closure | Service provision |
| Backtest results | 2 years or until user deletes | Historical reference |
| Execution logs | 5 years | Financial audit trail (MiFID II Art. 25) |
| Broker credentials | Until user revokes | Service provision |
| Security logs | 2 years | Security investigation |
| Audit logs | 5 years (anonymized after deletion) | Legal compliance |

### 4.2 Deletion Process

When you delete data or your account:

1. **Immediate**: Data is marked for deletion
2. **Within 30 days**: Data is permanently deleted from primary systems
3. **Within 90 days**: Data is removed from backups
4. **Exception**: Anonymized audit logs may be retained per legal requirements

---

## 5. DATA SHARING

### 5.1 Third-Party Brokers

When you execute trades, we share with your connected broker(s):

- Order details (symbol, quantity, price, type)
- Your API credentials (for authentication)
- **We do NOT share**: Your personal information, other strategies, or analytics

### 5.2 No Sale of Data

**WE DO NOT SELL YOUR PERSONAL DATA.**

We never sell, rent, or trade your personal information to third parties for marketing or any other purposes.

### 5.3 Sub-Processors

We use the following service providers (sub-processors):

| Provider | Purpose | Location | Safeguards |
|----------|---------|----------|------------|
| AWS (Amazon Web Services) | Cloud infrastructure | EU (eu-central-1, eu-west-1) | DPA, SCCs |
| [Email Provider] | Transactional email | EU | DPA, SCCs |
| [Analytics Provider] | Usage analytics (optional) | EU | DPA, Consent |

**Updated List:** A current list of sub-processors is available at [URL].

### 5.4 Legal Disclosure

We may disclose data when required by:

- Valid court orders or subpoenas
- Regulatory requirements
- Protection of our legal rights
- Emergency situations involving safety

We will notify you of such disclosures unless legally prohibited.

---

## 6. INTERNATIONAL TRANSFERS

### 6.1 EU Storage

Your data is stored and processed in the European Union:

- **Primary Region:** AWS eu-central-1 (Frankfurt, Germany)
- **Backup Region:** AWS eu-west-1 (Dublin, Ireland)

### 6.2 Non-EU Transfers

If data transfer outside the EU is necessary, we ensure adequate protection through:

- **Standard Contractual Clauses (SCCs)** approved by the European Commission
- **Adequacy decisions** (for countries with adequate protection)
- **Binding Corporate Rules** (where applicable)

### 6.3 Broker Connections

When you connect to non-EU brokers, order data may be transmitted to broker servers outside the EU. This is:

- Necessary for contract performance
- Initiated and controlled by you
- Limited to order execution data

---

## 7. YOUR RIGHTS UNDER GDPR

### 7.1 Right of Access (Article 15)

You have the right to:

- Confirm whether we process your personal data
- Obtain a copy of your personal data
- Information about processing purposes, categories, recipients

**How to exercise:** Account Settings > Privacy > Download My Data

### 7.2 Right to Rectification (Article 16)

You have the right to:

- Correct inaccurate personal data
- Complete incomplete personal data

**How to exercise:** Account Settings > Profile, or contact support

### 7.3 Right to Erasure (Article 17)

You have the right to request deletion of your personal data when:

- Data is no longer necessary for the original purpose
- You withdraw consent (for consent-based processing)
- You object to processing (and there are no overriding legitimate grounds)
- Data has been unlawfully processed

**Exceptions:** We may retain data required for legal compliance (e.g., audit logs for 5 years).

**How to exercise:** Account Settings > Privacy > Delete Account, or contact DPO

### 7.4 Right to Restriction (Article 18)

You have the right to restrict processing when:

- You contest data accuracy (pending verification)
- Processing is unlawful but you prefer restriction over erasure
- We no longer need the data but you need it for legal claims
- You have objected to processing (pending verification)

### 7.5 Right to Data Portability (Article 20)

You have the right to:

- Receive your personal data in a structured, commonly used, machine-readable format (JSON)
- Transmit that data to another controller

**Scope:** Applies to data you provided to us, processed by automated means, based on consent or contract.

**How to exercise:** Account Settings > Privacy > Export My Data

### 7.6 Right to Object (Article 21)

You have the right to object to processing based on legitimate interests. We will cease processing unless we demonstrate compelling legitimate grounds.

### 7.7 Right Regarding Automated Decision-Making (Article 22)

You have the right not to be subject to decisions based solely on automated processing that produce legal or significant effects.

**Note:** The Platform executes trades based on YOUR strategies, not automated decisions by us.

### 7.8 Right to Withdraw Consent (Article 7(3))

Where processing is based on consent, you can withdraw consent at any time. Withdrawal does not affect the lawfulness of prior processing.

### 7.9 Right to Lodge a Complaint

You have the right to lodge a complaint with a supervisory authority:

- **Netherlands:** Autoriteit Persoonsgegevens (Dutch DPA)
  - Website: https://autoriteitpersoonsgegevens.nl
- **Your country:** Contact your local data protection authority

---

## 8. COOKIES AND TRACKING

### 8.1 Essential Cookies

We use essential cookies required for the Platform to function:

| Cookie | Purpose | Duration |
|--------|---------|----------|
| session_id | Session management | Session |
| csrf_token | Security | Session |
| auth_token | Authentication | 7 days |

**These cookies do not require consent** under ePrivacy Directive Art. 5(3).

### 8.2 Analytics Cookies (Optional)

With your consent, we may use analytics cookies:

| Cookie | Purpose | Duration |
|--------|---------|----------|
| _analytics_id | Usage analytics | 1 year |

**How to manage:** Cookie preferences are available in account settings.

### 8.3 No Third-Party Advertising

We do not use advertising cookies or allow third-party advertising trackers.

---

## 9. SECURITY MEASURES

### 9.1 Technical Measures

We implement comprehensive security controls:

- **Encryption at rest**: AES-256 for sensitive data, including broker credentials
- **Encryption in transit**: TLS 1.3 for all communications
- **Key management**: Hardware Security Modules (HSM) for master keys
- **Access controls**: Role-based access, multi-factor authentication
- **Network security**: Firewalls, intrusion detection, DDoS protection

### 9.2 Organizational Measures

- **Security training**: Regular employee security awareness training
- **Access policies**: Principle of least privilege
- **Incident response**: Documented procedures for security incidents
- **Vendor assessment**: Security evaluation of all sub-processors

### 9.3 Audit and Monitoring

- Regular security assessments and penetration testing
- Continuous monitoring for suspicious activities
- Audit logging of all sensitive data access

### 9.4 Breach Notification

In the event of a personal data breach:

- We will notify the supervisory authority within 72 hours (GDPR Art. 33)
- We will notify affected users without undue delay if high risk (GDPR Art. 34)
- We will document all breaches in our breach register

---

## 10. CHILDREN'S PRIVACY

The Platform is not intended for individuals under 18 years of age. We do not knowingly collect personal data from children.

If we become aware that we have collected data from a child, we will delete it promptly.

---

## 11. CHANGES TO THIS POLICY

### 11.1 Notification of Changes

We will notify you of material changes to this Privacy Policy:

- Via email to your registered address
- Through a prominent notice on the Platform
- At least 30 days before changes take effect

### 11.2 Version History

| Version | Date | Summary of Changes |
|---------|------|--------------------|
| 1.0.0 | December 2024 | Initial release |

### 11.3 Review

We review this Privacy Policy annually and update it as necessary to reflect changes in our practices or legal requirements.

---

## 12. CONTACT US

### 12.1 Privacy Inquiries

For questions about this Privacy Policy or your personal data:

- **Email:** privacy@[company-domain].com
- **DPO Email:** dpo@[company-domain].com

### 12.2 Data Subject Requests

To exercise your GDPR rights:

- **Self-service:** Account Settings > Privacy
- **Email:** dpo@[company-domain].com
- **Response time:** Within 30 days (extendable by 60 days for complex requests)

### 12.3 Complaints

If you are unsatisfied with our response, you may:

1. Escalate to our DPO
2. Lodge a complaint with your supervisory authority

---

## 13. LEGAL REFERENCES

| Reference | Description |
|-----------|-------------|
| GDPR Article 13 | Information for direct collection |
| GDPR Article 14 | Information for indirect collection |
| GDPR Article 15 | Right of access |
| GDPR Article 16 | Right to rectification |
| GDPR Article 17 | Right to erasure |
| GDPR Article 18 | Right to restriction |
| GDPR Article 20 | Right to data portability |
| GDPR Article 21 | Right to object |
| GDPR Article 22 | Automated decision-making |
| GDPR Article 33-34 | Breach notification |
| ePrivacy Directive Art. 5(3) | Cookie consent |
| EDPB Guidelines WP260 | Transparency |
| EDPB Guidelines WP242 | Data Portability |

---

**Last Updated:** December 2024
**Document Version:** 1.0.0

---

## APPENDIX: DATA PROCESSING ACTIVITIES REGISTER

| Activity | Data Categories | Legal Basis | Retention | Recipients |
|----------|----------------|-------------|-----------|------------|
| Account Management | Email, name, password hash | Contract | Until deletion | Internal |
| Strategy Storage | Code, parameters | Contract | Until deletion | Internal |
| Backtesting | Historical results | Contract | 2 years | Internal |
| Order Execution | Orders, positions | Contract | 5 years | Broker |
| Security Monitoring | IP, logs | Legitimate Interest | 2 years | Internal |
| Credential Storage | API keys (encrypted) | Contract | Until revoked | Broker (on use) |

# Terms of Service

**AI-Powered Quantitative Research Platform**

**Effective Date:** December 2024
**Version:** 3.0.0

**Regulatory Framework:** EU E-Commerce Directive 2000/31/EC, GDPR (EU) 2016/679, EU AI Act (EU) 2024/1689

**Architecture:** Cloud-Controlled Execution Architecture (CCEA)

---

## 1. DEFINITIONS

For the purposes of these Terms of Service, the following definitions apply:

- **"Platform"**: The AI-Powered Quantitative Research Platform software service, including all web interfaces, APIs, and related tools provided by the Company.

- **"User"**, **"You"**, **"Your"**: Any individual or entity that accesses or uses the Platform.

- **"Company"**, **"We"**, **"Us"**, **"Our"**: The legal entity operating the Platform.

- **"Strategy"**: Any trading algorithm, rule set, or automated decision-making logic created, uploaded, or executed by a User on the Platform.

- **"Broker API Keys"**: Authentication credentials (API keys, secrets, tokens) provided by third-party brokerage services that enable the Platform to execute trades on behalf of the User.

- **"Backtest"**: A simulation of a Strategy's historical performance using past market data.

- **"Live Trading"**: The execution of real orders on financial markets through connected broker accounts.

- **"Services"**: All features, tools, and functionalities provided by the Platform.

- **"Cloud"**: The Company-operated infrastructure providing research, backtesting, monitoring, and lifecycle management services.

- **"Agent"**: Optional user-operated software running locally on User's hardware that handles live trading execution, credential storage, and risk enforcement.

- **"CCEA"**: Cloud-Controlled Execution Architecture, the strict separation between Cloud (research/monitoring) and Agent (execution/secrets).

---

## 2. SERVICE DESCRIPTION

### 2.0 Architecture: Cloud-Controlled Execution Architecture (CCEA)

**IMPORTANT - ARCHITECTURAL SEPARATION:**

The Platform implements **CCEA (Cloud-Controlled Execution Architecture)**, a strict separation between Cloud and Agent:

| Component | Responsibility | Handles Credentials | Executes Orders |
|-----------|---------------|---------------------|-----------------|
| **Cloud** | Research, backtesting, monitoring, lifecycle management | **NEVER** | **NEVER** |
| **Agent** | Live execution, credential storage, risk enforcement | **YES (Local Only)** | **YES (Local Only)** |

**Security Design Commitments** (enforced at architecture level):

1. **Cloud NEVER stores your broker API keys or credentials** - All credentials are stored locally in your Agent's encrypted vault
2. **Cloud NEVER generates, transmits, or executes trading orders** - All trading operations occur exclusively in your local Agent
3. **Cloud NEVER has access to exchange trading endpoints** - Cloud cannot connect to brokers on your behalf
4. **Cloud cannot send order-like payloads** - The protocol schema prohibits side/quantity/price fields
5. **Telemetry is ALWAYS redacted** - Sensitive data is automatically removed before transmission

**Product Modes:**

| Mode | Description | Agent Required |
|------|-------------|----------------|
| Research SaaS | Cloud-based research, backtesting, simulation | No |
| Live Trading | Automated execution via local Agent | Yes (User-operated) |
| Enterprise | Self-hosted Cloud + Agent in customer infrastructure | Varies |

**Your Control:**

- You decide whether to deploy a local Agent
- You control your Agent's hard caps (which Cloud CANNOT override)
- You approve all trading-impacting changes locally
- You can disconnect from Cloud and continue trading

### 2.0.1 CCEA Privacy Design Commitments (Binding)

The following privacy design commitments are **architectural invariants** that are legally binding under these Terms:

**A. Cloud Never Receives Secrets**
- Cloud **NEVER** stores or processes your broker API keys, secrets, or credentials
- Cloud **NEVER** receives environment variables or tokens
- Violation of this commitment constitutes a material breach of these Terms

**B. No Order-like Payloads in Protocol**
- Cloud→Agent commands **NEVER** contain order-like payloads (side, quantity, price, order_id, target_position)
- This is enforced at the protocol schema level and verified by CI guardrails
- New command types require security review and auditable approval

**C. Telemetry Sensitivity Levels**
The Platform implements three telemetry levels with strict controls:

| Level | Description | Opt-in | Order Data |
|-------|-------------|--------|------------|
| **AGGREGATED** | Default for all tiers | No | Forbidden |
| **DETAILED_NON_SENSITIVE** | Technical debugging | Yes | Forbidden |
| **RAW_ORDER_EVENTS** | Enterprise-only | Enterprise + Explicit | Allowed (masked) |

- Telemetry redaction is **always mandatory** and cannot be disabled
- `RAW_ORDER_EVENTS` is available **only** to enterprise customers with explicit per-workspace opt-in

**D. EU-only Data Residency**
- All data storage, backups, logs, and processing are in EU (eu-central-1, eu-west-1)
- All sub-processors are located in EU
- EU-only residency is enforced by automated drift checks (fail-closed)

**E. DSAR Scope Boundaries**
- DSAR requests apply to Cloud-controlled data only
- Agent-zone data remains under your control and is not accessible to us
- We cannot export or delete data we never receive

### 2.1 Nature of Services

The Platform is a **software tool** that provides:

- Strategy development and coding environment
- Backtesting engine for historical simulation
- Paper trading (simulated) capabilities
- Live trading execution via User-provided broker API credentials
- Analytics, reporting, and visualization tools
- Risk management and monitoring features

### 2.2 What We Are NOT

**IMPORTANT NOTICE - PLEASE READ CAREFULLY:**

The Platform is **NOT**:

- An investment adviser or financial adviser
- A portfolio manager or asset manager
- A broker-dealer or securities firm
- A provider of personalized investment recommendations
- A provider of financial, legal, or tax advice
- **An order execution service** - Cloud NEVER executes orders; this is done locally by your Agent
- **A custodian** - We do not hold your assets or credentials in Cloud
- **A trading signal provider** - We do not transmit trading intents or signals to execute

We are a **software vendor** providing:
- Research and backtesting tools (Cloud)
- Optional local execution software (Agent)
- Monitoring and lifecycle management

**All trading decisions and execution happen in YOUR environment, under YOUR control.**

### 2.3 User Control

All trading decisions are made solely by the User. The Platform:

- Executes only what the User's Strategy instructs
- Does not modify, override, or second-guess User strategies
- Does not provide recommendations on what to trade
- Does not manage User portfolios or assets

**Reference:** ESMA Q&A MiFID II (ESMA35-43-349) - Software Tool Exclusion from Investment Services

---

## 2A. ARTIFICIAL INTELLIGENCE DISCLOSURE

### 2A.1 AI System Notice (EU AI Act Article 50)

**THIS PLATFORM USES AN ARTIFICIAL INTELLIGENCE SYSTEM.**

In accordance with Article 50 of Regulation (EU) 2024/1689 (EU AI Act), we inform you that:

1. **AI Technology**: The Platform incorporates a machine learning system based on Reinforcement Learning (Distributional PPO algorithm) for generating trading signals and portfolio recommendations.

2. **AI-Generated Outputs**: Trading signals, predictions, risk assessments, strategy recommendations, and analytical reports are generated by the AI system, not by human analysts.

3. **Limitations**: The AI system:
   - May produce inaccurate or suboptimal predictions
   - Has been trained on historical data that may not reflect future market conditions
   - Cannot guarantee profitable trading outcomes
   - May behave unexpectedly in novel market situations
   - Has performance that varies across different asset classes and market regimes

4. **Human Responsibility**: You retain full responsibility for:
   - Deciding whether to follow AI-generated signals
   - All trading decisions and their financial consequences
   - Monitoring and overriding AI outputs when appropriate
   - Setting appropriate risk limits and position sizes

### 2A.2 AI Model Information

| Attribute | Value |
|-----------|-------|
| Model Name | Distributional PPO Trading Model |
| Model Type | Reinforcement Learning |
| Architecture | LSTM + Distributional Value Network |
| Primary Use | Trading signal generation |

For detailed technical information, see our [GPAI Model Card](../compliance/GPAI_MODEL_CARD.md).

### 2A.3 Training Data Transparency

Per Article 53(1)(d) of the EU AI Act, we provide a public summary of training data used:

- **Data Types**: Historical market data (OHLCV), technical indicators, synthetic scenarios
- **Data Sources**: Licensed market data providers, public exchange APIs
- **Personal Data**: No personal data is used for model training

For the complete training data summary, see [TRAINING_DATA_SUMMARY.md](../compliance/TRAINING_DATA_SUMMARY.md).

### 2A.4 Acknowledgment Requirement

Before using live trading features, you must explicitly acknowledge that:

1. You understand you are interacting with an AI system
2. You accept the inherent limitations of AI-generated predictions
3. You retain full responsibility for all trading decisions
4. You will maintain appropriate oversight of AI outputs

This acknowledgment is recorded for compliance purposes.

### 2A.5 AI Content Marking

Content generated by the AI system is marked with:
- `[AI-GENERATED]` prefix in reports and analysis
- `X-AI-System: true` header in API responses
- Visual indicators in the user interface

**References:**
- EU AI Act Article 50(1): AI Interaction Disclosure
- EU AI Act Article 50(2): Synthetic Content Marking
- EU AI Act Article 53(1)(d): Training Data Summary
- Recital 132: Transparency for Limited-Risk AI

---

## 3. USER RESPONSIBILITIES

### 3.1 Broker Account Requirements

You must:

- **Provide your own broker account(s)**: The Platform does not provide brokerage services
- **Maintain valid API credentials**: You are responsible for obtaining and maintaining valid API keys from your chosen broker(s)
- **Ensure broker compliance**: Your use of the Platform must comply with your broker's terms of service and API usage policies
- **Monitor account status**: You are responsible for ensuring your broker account remains in good standing

### 3.2 Trading Knowledge and Experience

By using the Platform for live trading, you confirm that:

- You understand how financial markets operate
- You understand the risks associated with trading financial instruments
- You have sufficient knowledge to develop, evaluate, and deploy trading strategies
- You are capable of independently assessing the suitability of your strategies

### 3.3 Regulatory Compliance

You are responsible for:

- Compliance with all applicable laws and regulations in your jurisdiction
- Obtaining any necessary licenses or registrations for your trading activities
- Tax reporting and payment obligations arising from your trading
- Ensuring your strategies do not constitute market manipulation or other prohibited conduct

### 3.4 Age Requirement

You must be at least **18 years of age** (or the legal age of majority in your jurisdiction, whichever is higher) to use the Platform's live trading features.

---

## 4. BROKER API KEYS

### 4.1 Authorization

By submitting your Broker API Keys to the Platform, you:

- Grant the Platform permission to connect to your broker account
- Authorize the Platform to submit, modify, and cancel orders on your behalf
- Acknowledge that orders will be executed according to your Strategy's logic

### 4.2 Security Measures

Your Broker API Keys are protected by:

- **Encryption at rest**: AES-256-GCM encryption with unique keys per user
- **Encryption in transit**: TLS 1.3 for all data transmission
- **Access controls**: Keys are decrypted only when required for order execution
- **Audit logging**: All credential access is logged for security monitoring

**Reference:** NIST SP 800-57 (Key Management), OWASP Cryptographic Storage Cheat Sheet

### 4.3 Your Rights

You retain full control over your credentials:

- **Revocation**: You can revoke Platform access at any time by removing your API keys
- **Deletion**: Upon request, all stored credentials will be permanently deleted
- **Visibility**: You can view when your credentials were last accessed

### 4.4 Limitations

**CRITICAL**: The Platform is configured to NEVER have:

- Withdrawal rights from your broker account
- Ability to transfer funds out of your account
- Access to your personal banking information

You should configure your API keys with **trade-only permissions** (no withdrawal capability) for maximum security.

---

## 5. NO INVESTMENT ADVICE

### 5.1 Software Tool Disclaimer

**THE PLATFORM PROVIDES SOFTWARE TOOLS, NOT INVESTMENT ADVICE.**

In accordance with MiFID II Article 4(1)(4), the Platform:

- Does not assess your financial situation, objectives, or risk tolerance
- Does not recommend specific investments or strategies
- Does not provide personalized advice of any kind
- Does not hold itself out as providing investment services

### 5.2 Backtest Results

Backtest results displayed by the Platform:

- Are **simulations** based on historical data
- Do **NOT** represent actual trading results
- Do **NOT** guarantee or predict future performance
- May not account for all real-world factors (slippage, liquidity, fees)

**"PAST PERFORMANCE DOES NOT GUARANTEE FUTURE RESULTS"**

### 5.3 Strategy Development

Any strategies you develop using the Platform:

- Are your own intellectual property
- Are based on your own analysis and decisions
- Should be thoroughly tested before live deployment
- May result in significant financial losses

### 5.4 Educational Content

Any educational materials, tutorials, or example strategies provided by the Platform:

- Are for informational purposes only
- Do not constitute recommendations to trade
- Should not be relied upon for making investment decisions

**Reference:** ESMA Guidelines on MiFID II Product Governance Requirements

---

## 6. LIMITATION OF LIABILITY

### 6.1 Cap on Direct Damages

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW:

Our total aggregate liability for any direct damages arising from your use of the Platform shall not exceed the **total fees you have paid to us in the twelve (12) months preceding the claim**.

### 6.2 Exclusion of Certain Damages

WE SHALL NOT BE LIABLE FOR:

- **Trading losses**: Any financial losses resulting from trades executed through the Platform
- **Market movements**: Losses caused by market volatility, gaps, or adverse price movements
- **Broker failures**: Issues with your broker's systems, including order rejections or execution problems
- **Strategy performance**: Poor performance of your trading strategies
- **Indirect damages**: Lost profits, lost revenue, loss of business opportunities
- **Consequential damages**: Any indirect, incidental, special, or consequential damages

### 6.3 Force Majeure

We shall not be liable for any failure or delay in performance due to:

- Acts of God, natural disasters, or severe weather
- War, terrorism, civil unrest, or government actions
- Internet or telecommunications failures
- Exchange or market closures
- Cyber attacks or security breaches beyond our reasonable control
- Pandemics or public health emergencies

### 6.4 Basis of the Bargain

You acknowledge that the limitations of liability in this section:

- Reflect the allocation of risk between the parties
- Are an essential basis of the agreement
- Allow us to provide the Services at the current pricing

---

## 7. DISCLAIMERS

### 7.1 Risk Warnings

**TRADING INVOLVES SUBSTANTIAL RISK OF LOSS.**

You should carefully consider whether trading is appropriate for you in light of your:

- Financial condition
- Investment experience
- Risk tolerance
- Other relevant circumstances

**YOU MAY LOSE MORE THAN YOUR INITIAL INVESTMENT.**

### 7.2 No Guarantees

THE PLATFORM IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, WHETHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO:

- Merchantability
- Fitness for a particular purpose
- Non-infringement
- Accuracy or reliability of any information
- Uninterrupted or error-free operation

### 7.3 Software Limitations

You acknowledge that:

- Software may contain bugs or errors
- Network failures may affect order execution
- Market data may be delayed or inaccurate
- Execution prices may differ from expected prices
- System maintenance may cause temporary unavailability

### 7.4 Third-Party Services

The Platform integrates with third-party services (brokers, data providers). We:

- Do not control these services
- Are not responsible for their availability or accuracy
- Do not guarantee compatibility with all brokers

---

## 8. DATA & PRIVACY

### 8.1 Privacy Policy

Your use of the Platform is also governed by our [Privacy Policy](PRIVACY_POLICY.md), which describes:

- What personal data we collect
- How we use and protect your data
- Your rights under GDPR and other data protection laws

### 8.2 GDPR Rights Summary

If you are in the European Economic Area, you have the right to:

- **Access** your personal data (Article 15)
- **Rectify** inaccurate data (Article 16)
- **Erasure** ("right to be forgotten") (Article 17)
- **Data portability** (Article 20)
- **Object** to processing (Article 21)
- **Lodge a complaint** with a supervisory authority

### 8.3 Data Security

We implement appropriate technical and organizational measures to protect your data, including:

- Encryption of sensitive data at rest and in transit
- Access controls and authentication
- Regular security assessments
- Incident response procedures

---

## 9. TERMINATION

### 9.1 Your Right to Terminate

You may terminate your use of the Platform at any time by:

- Closing your account through the Platform settings
- Removing all Broker API Keys
- Contacting our support team

### 9.2 Our Right to Terminate

We may suspend or terminate your access to the Platform if:

- You breach these Terms of Service
- We are required to do so by law or regulation
- Your account shows signs of unauthorized access or misuse
- You fail to pay applicable fees

### 9.3 Effect of Termination

Upon termination:

- Your right to access the Platform ceases immediately
- Any pending orders may be cancelled (you should verify with your broker)
- Your stored data will be handled according to our Privacy Policy
- You may request deletion of your data per GDPR Article 17

### 9.4 Survival

Sections relating to limitation of liability, disclaimers, intellectual property, and governing law shall survive termination.

---

## 10. GOVERNING LAW AND DISPUTE RESOLUTION

### 10.1 Governing Law

These Terms of Service shall be governed by and construed in accordance with the laws of **the Netherlands**, without regard to its conflict of law provisions.

### 10.2 Jurisdiction

For any disputes arising from these Terms or your use of the Platform:

- Courts of Amsterdam, Netherlands shall have exclusive jurisdiction
- EU consumers may also bring actions in their country of residence

### 10.3 Alternative Dispute Resolution

For EU consumers, information about online dispute resolution is available at:
https://ec.europa.eu/consumers/odr/

### 10.4 Arbitration Option

For business users, disputes may alternatively be resolved through binding arbitration under the rules of the Netherlands Arbitration Institute (NAI).

---

## 11. CHANGES TO TERMS

### 11.1 Modifications

We may modify these Terms of Service at any time. Material changes will be communicated:

- Via email to registered users
- Through a prominent notice on the Platform
- At least 30 days before taking effect

### 11.2 Continued Use

Your continued use of the Platform after changes become effective constitutes acceptance of the modified Terms.

### 11.3 Objection

If you do not agree to modified Terms, you must stop using the Platform and may terminate your account.

---

## 12. GENERAL PROVISIONS

### 12.1 Entire Agreement

These Terms, together with the Privacy Policy and any other agreements referenced herein, constitute the entire agreement between you and the Company regarding the Platform.

### 12.2 Severability

If any provision of these Terms is found unenforceable, the remaining provisions shall continue in full force and effect.

### 12.3 Waiver

Our failure to enforce any right or provision shall not constitute a waiver of that right or provision.

### 12.4 Assignment

You may not assign your rights under these Terms without our prior written consent. We may assign our rights without restriction.

### 12.5 Contact Information

For questions about these Terms of Service:

- **Email**: legal@[company-domain].com
- **Address**: [Company Address]
- **Website**: [Company Website]

---

## 13. ACCEPTANCE

By using the Platform, you acknowledge that you have read, understood, and agree to be bound by these Terms of Service.

For live trading features, you will be asked to explicitly acknowledge specific risk warnings before activation.

---

**Last Updated:** December 2024
**Document Version:** 3.0.0

---

## APPENDIX A: REGULATORY REFERENCES

| Reference | Description |
|-----------|-------------|
| EU E-Commerce Directive 2000/31/EC Art. 5, 6 | Information requirements for service providers |
| MiFID II Directive 2014/65/EU Art. 4(1)(4) | Definition of investment advice |
| ESMA Q&A MiFID II (ESMA35-43-349) | Software vendor exclusion from investment services |
| GDPR (EU) 2016/679 | General Data Protection Regulation |
| EDPB Guidelines WP260 | Transparency requirements |
| EU AI Act (EU) 2024/1689 Art. 50 | AI System Transparency Requirements |
| EU AI Act (EU) 2024/1689 Art. 53 | GPAI Provider Obligations |
| DSM Directive (EU) 2019/790 Art. 4 | Text and Data Mining (Copyright) |

## APPENDIX B: VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | December 2024 | Initial release |
| 1.1.0 | December 2024 | Added Section 2A: AI Disclosure per EU AI Act Article 50 |
| 3.0.0 | December 2024 | GDPR Phase 1: Added Section 2.0.1 CCEA Privacy Design Commitments (secrets, order payloads, telemetry levels, EU-only residency, DSAR boundaries) |

# CCEA Terms of Service Guidelines

## Legal Framework for CCEA Platform Services

**Document Version:** 1.0
**Date:** December 2025
**Classification:** Internal - Legal Reference
**Note:** This is a guidelines document, not the actual Terms of Service. Final ToS must be reviewed by legal counsel.

---

## Purpose

This document establishes the **required provisions and language** for CCEA Platform Terms of Service. These guidelines ensure:

1. **Regulatory Compliance** - Clear positioning as software provider
2. **Liability Protection** - Appropriate limitation of liability
3. **Customer Clarity** - Clear understanding of responsibilities
4. **Architectural Alignment** - ToS reflects Cloud/Agent separation

**Reference:** See [CCEA_OVERVIEW.md](../architecture/CCEA_OVERVIEW.md) for complete architectural documentation.

---

## 1. Required ToS Provisions

### 1.1 Core Positioning Statements (MANDATORY)

The following statements MUST appear prominently in the Terms of Service:

#### Not Investment Advice

```legal
CCEA Platform does not provide investment advice, recommendations, or
suggestions regarding securities, investments, or trading strategies.
All content, including but not limited to simulation results, backtest
data, model outputs, and educational materials, is provided for
informational and educational purposes only and should not be construed
as investment advice.

Users acknowledge that any trading or investment decisions are made
solely at their own discretion and risk, without reliance on CCEA
Platform or its affiliates for guidance on the suitability, profitability,
or appropriateness of any particular trade or investment strategy.
```

#### Not a Broker or Custodian

```legal
CCEA Platform is not a broker, dealer, exchange, custodian, or any other
type of regulated financial intermediary. We do not:

- Execute trades or place orders on behalf of users
- Hold, custody, or manage user funds or assets
- Transmit orders to exchanges or brokers
- Provide access to financial markets
- Act as an intermediary between users and any financial institution

Users acknowledge that CCEA Platform is a software provider only. Any
trading activities occur through the user's own brokerage accounts and
relationships, using software tools (the "Agent") deployed and controlled
entirely within the user's own computing environment.
```

#### Execution Responsibility

```legal
All trade execution occurs exclusively within the user's own computing
environment (the "Agent") using the user's own brokerage accounts and
API credentials. The user is solely responsible for:

- Selecting and maintaining brokerage relationships
- Storing and securing API credentials
- Configuring risk management parameters
- Approving all trading-impacting changes
- Monitoring active strategies
- Complying with applicable regulations

CCEA Platform's Cloud services provide research tools, simulation
environments, training services, and strategy artifacts only. The Cloud
does NOT have access to user brokerage credentials and does NOT transmit
trading orders.
```

#### Risk Acknowledgment

```legal
Trading in financial instruments involves substantial risk of loss and
may not be suitable for all investors. Users acknowledge and agree that:

1. Past performance, including backtested or simulated results, does
   not guarantee future performance
2. Backtested results may not reflect actual trading conditions,
   including but not limited to market impact, slippage, liquidity
   constraints, and execution quality
3. Any trading strategy may result in partial or total loss of capital
4. Users are solely responsible for determining the appropriateness
   of any trading strategy for their individual circumstances
5. Market conditions may change rapidly, affecting strategy performance
6. Technical failures, including software bugs, network issues, or
   system outages, may result in losses

Users should only trade with capital they can afford to lose and should
consult with qualified financial advisors before making investment decisions.
```

---

## 2. Service Description Requirements

### 2.1 Cloud Services Description

```legal
CCEA Platform Cloud Services ("Cloud Services") include:

1. **Strategy Workspace** - Browser-based environment for strategy
   development and code editing
2. **Backtest & Simulation** - Historical simulation of trading strategies
   using L1, L2, and L3 market data models
3. **Training Service** - Machine learning model training using
   reinforcement learning and other optimization techniques
4. **Artifact Builder** - Compilation and packaging of strategies into
   deployable artifacts
5. **Artifact Registry** - Storage and version management of strategy
   artifacts
6. **Control Plane** - Communication infrastructure for sending lifecycle
   commands to user Agents

Cloud Services explicitly do NOT include:
- Live trading execution
- Order transmission to exchanges or brokers
- Access to or storage of user brokerage credentials
- Any form of investment advice or recommendations
```

### 2.2 Agent Software Description

```legal
CCEA Platform Agent Software ("Agent") is open-source software (MIT License)
that users download, deploy, and operate in their own computing environment.
The Agent provides:

1. **Strategy Runner** - Execution runtime for strategy artifacts
2. **Local Vault** - Encrypted storage for user API credentials
3. **Risk Manager** - Configurable risk controls and kill switch
4. **Broker Connectors** - Integration with user's brokerage accounts

The Agent:
- Is deployed and operated entirely by the user
- Stores all credentials locally in the user's environment
- Creates and transmits orders only upon user approval
- Provides local control including manual kill switch
- Is open-source and auditable by the user

CCEA Platform does not operate, control, or monitor user Agents except
as explicitly requested by the user through the Control Plane interface.
```

---

## 3. Liability Limitations

### 3.1 Disclaimer of Warranties

```legal
CCEA PLATFORM SERVICES ARE PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT
WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO:

- WARRANTIES OF MERCHANTABILITY
- FITNESS FOR A PARTICULAR PURPOSE
- NON-INFRINGEMENT
- ACCURACY, COMPLETENESS, OR RELIABILITY OF ANY CONTENT
- UNINTERRUPTED OR ERROR-FREE OPERATION

CCEA Platform does not warrant that:
- The services will meet user requirements
- Results obtained will be accurate or reliable
- Any errors will be corrected
- The services will be compatible with user systems
```

### 3.2 Limitation of Liability

```legal
TO THE MAXIMUM EXTENT PERMITTED BY LAW, CCEA PLATFORM AND ITS AFFILIATES,
OFFICERS, DIRECTORS, EMPLOYEES, AND AGENTS SHALL NOT BE LIABLE FOR:

1. Any indirect, incidental, special, consequential, or punitive damages
2. Any trading losses or lost profits
3. Loss of data, goodwill, or business opportunities
4. Any damages resulting from:
   - User trading decisions
   - Strategy underperformance
   - Broker execution issues
   - Market conditions or movements
   - Technical failures outside our direct control
   - Agent software operation in user environment
   - User configuration or misconfiguration

MAXIMUM LIABILITY: In no event shall CCEA Platform's total aggregate
liability exceed the fees actually paid by user in the twelve (12)
months preceding the claim.
```

### 3.3 Indemnification

```legal
User agrees to indemnify, defend, and hold harmless CCEA Platform and
its affiliates from any claims, damages, losses, or expenses arising from:

1. User's use of the services
2. User's trading activities
3. User's violation of these terms
4. User's violation of any law or regulation
5. User's infringement of third-party rights
6. User's brokerage relationships
7. Operation of the Agent software in user's environment
```

---

## 4. User Responsibilities

### 4.1 Eligibility Requirements

```legal
By using CCEA Platform services, user represents and warrants that:

1. User is at least 18 years of age (or age of majority in their
   jurisdiction)
2. User has the legal capacity to enter into binding contracts
3. User is not prohibited from using the services under applicable law
4. If using for business purposes, user has authority to bind the entity
5. User understands the risks associated with trading financial instruments
6. User will comply with all applicable laws and regulations
```

### 4.2 Compliance Obligations

```legal
User is solely responsible for:

1. Compliance with all applicable laws and regulations in their
   jurisdiction, including but not limited to:
   - Securities laws and regulations
   - Tax reporting and payment obligations
   - Anti-money laundering (AML) requirements
   - Know Your Customer (KYC) requirements
   - Exchange and broker terms of service

2. Maintaining appropriate brokerage accounts and relationships

3. Ensuring trading activities are permitted under applicable law

4. Reporting and paying any taxes on trading gains

5. Complying with exchange, broker, and market rules

CCEA Platform does not provide legal, tax, or regulatory advice.
Users should consult qualified professionals for such guidance.
```

### 4.3 Security Responsibilities

```legal
User is solely responsible for:

1. Security of their CCEA Platform account credentials
2. Security of their brokerage API credentials
3. Security of the computing environment running the Agent
4. Proper configuration of risk management parameters
5. Monitoring active strategies and Agent operation
6. Maintaining appropriate backups of configuration data

User agrees to:
- Use strong, unique passwords
- Enable two-factor authentication where available
- Keep software and systems updated
- Report suspected security breaches immediately
```

---

## 5. Data and Privacy

### 5.1 Data Collection

```legal
CCEA Platform collects and processes the following data:

1. **Account Data** - Registration information, contact details
2. **Usage Data** - Service usage patterns, feature utilization
3. **Strategy Metadata** - Strategy configurations, backtest parameters
4. **Telemetry Data** - Aggregated performance metrics (not raw order data)

CCEA Platform does NOT collect:
- Brokerage API credentials (stored locally in Agent)
- Raw order or trade data
- Actual portfolio positions or balances
- Personal financial information beyond service operation

See our Privacy Policy for complete data handling details.
```

### 5.2 Data Processing

```legal
User grants CCEA Platform a limited license to process user data for:

1. Providing the services
2. Improving platform functionality
3. Generating aggregated, anonymized insights
4. Compliance with legal obligations

User data is processed in accordance with:
- GDPR (for EU users)
- Applicable data protection laws
- Our Privacy Policy

Users retain ownership of their strategy code and configurations.
```

---

## 6. Intellectual Property

### 6.1 Platform IP

```legal
CCEA Platform retains all rights to:

1. Cloud service software and algorithms
2. Training service methods and models
3. Backtest engine and simulation technology
4. Platform documentation and content
5. Trademarks, logos, and brand assets

Users receive a limited, non-exclusive, non-transferable license to
use the services as permitted under their subscription tier.
```

### 6.2 User IP

```legal
Users retain all rights to:

1. Strategy code developed by user
2. Custom configurations and parameters
3. Data and content uploaded by user

Users grant CCEA Platform a limited license to process such content
solely for providing the services.
```

### 6.3 Agent Software (Open Source)

```legal
The CCEA Agent software is provided under the MIT License. Users may:

1. Use, copy, modify, and distribute the Agent software
2. Use the Agent for commercial purposes
3. Create derivative works based on the Agent

The MIT License text is included in the Agent repository.
```

---

## 7. Subscription and Payment

### 7.1 Subscription Terms

```legal
1. Subscriptions are billed monthly or annually as selected
2. Fees are non-refundable except as required by law
3. Automatic renewal unless cancelled before renewal date
4. Price changes effective at next renewal with 30 days notice
5. Downgrades effective at next renewal period
```

### 7.2 Usage Limits

```legal
Each subscription tier includes specific usage limits for:
- Backtest compute hours
- Training compute hours
- API calls
- Concurrent Agents (for Team+ tiers)

Overages billed at stated overage rates.
See pricing documentation for tier-specific limits.
```

---

## 8. Termination

### 8.1 User Termination

```legal
Users may terminate their account at any time by:
1. Cancelling subscription through the platform
2. Emailing termination request to [support email]

Upon termination:
- Access to Cloud services ends immediately
- Downloaded strategy artifacts remain available
- Agent software continues to function (open source)
- Data retained per our data retention policy
```

### 8.2 Platform Termination

```legal
CCEA Platform may terminate or suspend user accounts for:
1. Violation of these Terms of Service
2. Suspected fraudulent activity
3. Non-payment of fees
4. Violation of applicable law
5. Actions harmful to platform or other users

Platform may terminate service generally with 90 days notice.
```

---

## 9. Governing Law and Disputes

### 9.1 Governing Law

```legal
These Terms are governed by the laws of [Jurisdiction], without
regard to conflict of law principles.
```

### 9.2 Dispute Resolution

```legal
Any disputes shall be resolved through:

1. **Informal Resolution** - Good faith negotiation for 30 days
2. **Mediation** - If negotiation fails, non-binding mediation
3. **Arbitration/Litigation** - [Specify arbitration or court]

Class action waiver: Users agree to resolve disputes individually
and waive any right to participate in class actions.
```

---

## 10. Miscellaneous

### 10.1 Entire Agreement

```legal
These Terms, together with the Privacy Policy and any order forms,
constitute the entire agreement between user and CCEA Platform.
```

### 10.2 Severability

```legal
If any provision is held invalid, remaining provisions continue in effect.
```

### 10.3 Modifications

```legal
CCEA Platform may modify these Terms with 30 days notice. Continued
use after modification constitutes acceptance.
```

### 10.4 Assignment

```legal
Users may not assign their rights without CCEA Platform consent.
CCEA Platform may assign these Terms in connection with merger,
acquisition, or sale of assets.
```

### 10.5 No Waiver

```legal
Failure to enforce any provision does not constitute waiver.
```

---

## 11. Implementation Checklist

### Before Launching ToS

- [ ] Legal counsel review complete
- [ ] Jurisdiction-specific provisions added
- [ ] Privacy Policy aligned with ToS
- [ ] User acceptance flow implemented
- [ ] Version control established
- [ ] Notification system for updates ready
- [ ] Archive of previous versions maintained

### Jurisdiction Considerations

| Jurisdiction | Additional Requirements |
|--------------|------------------------|
| **EU/GDPR** | Data processing basis, DPO contact, EU representative |
| **UK** | Post-Brexit compliance, UK representative |
| **US** | State-specific requirements (CA, NY), CCPA |
| **Singapore** | PDPA compliance |

---

## 12. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12 | Legal Team | Initial guidelines |

**Note:** This document provides guidelines only. Actual Terms of Service must be drafted and reviewed by qualified legal counsel familiar with applicable jurisdictions.

**Review Cycle:** Quarterly
**Next Review:** Q2 2025
**Owner:** Legal Counsel

---

**Classification:** INTERNAL - Legal Reference
**Distribution:** Legal, Product, Engineering Leadership

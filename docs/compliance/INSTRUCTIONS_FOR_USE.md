# Instructions for Use

## CustodiaCloud

**Document ID**: IFU-2025-001
**Version**: 1.0
**Issue Date**: 2025-12-08
**Regulation Reference**: EU AI Act Article 13

---

## Important Notice

This document is an **Article 13 template** intended to support transparency/instructions requirements under Regulation (EU) 2024/1689 (EU AI Act).

CustodiaCloud does **not** self-classify as a “high-risk AI system” in documentation. Classification and applicable obligations depend on deployment context, roles, and jurisdiction and should be validated with qualified counsel.

**This system is intended for professional use only by qualified operators with appropriate financial market knowledge and regulatory understanding.**

---

## 1. Provider Information

### 1.1 Provider Details (Article 13(3)(a))

| Field | Value |
|-------|-------|
| **Provider Name** | [Provider Legal Name] |
| **Registered Address** | [Full Address] |
| **Contact Email** | support@provider.com |
| **Contact Phone** | +XX XXX XXX XXXX |
| **Website** | https://provider.com |

### 1.2 Technical Support

| Support Level | Contact | Hours |
|---------------|---------|-------|
| **Critical Issues** | critical-support@provider.com | Target (planned): 24/7 (capacity dependent; actual hours defined in executed service agreements) |
| **General Support** | support@provider.com | Business hours |
| **Documentation** | docs@provider.com | Business hours |

---

## 2. System Description

### 2.1 Intended Purpose (Article 13(3)(b))

CustodiaCloud is designed for:

- **Primary Function**: Quantitative research workflows (simulation/backtesting/monitoring) and packaging signed artifacts for deployment via the customer-controlled Agent (CCEA)
- **Asset Classes**: Equities-first (MVP); foundation multi-asset (options/futures/FX/digital assets as optional expansion)
- **Target Users**: Professional trading organizations and other qualified operators (B2B)
- **Deployment Context**: Client-controlled deployments in regulated financial markets (deployment context determines applicable obligations)

**CCEA boundary reminder:** Cloud does not store broker credentials and does not send live trading instructions (orders/targets/signals). Live execution (if any) occurs only via the customer-controlled Agent and the customer's own broker accounts.

### 2.2 System Architecture

| Component | Description |
|-----------|-------------|
| **Core Algorithm** | Distributional PPO with Twin Critics |
| **Policy Network** | LSTM-based recurrent architecture |
| **Value Network** | Distributional (C51-style, 21 quantiles) |
| **Risk Management** | CVaR-based risk-aware optimization |

---

## 3. Capabilities and Performance

### 3.1 System Capabilities (Article 13(3)(b)(i))

The system is capable of:

1. **Research Output Generation (equities-first)**
   - Quantitative analysis across equities for the MVP, with optional expansion to options/futures/FX/digital assets
   - Scenario analysis for strategy parameters and risk settings (operator-controlled)
   - Portfolio-level analytics (client-side, if applicable)

2. **Market Data Processing**
   - Real-time price data ingestion
   - Technical indicator computation
   - Feature engineering and normalization

3. **Risk-Aware Decision Making**
   - CVaR optimization for tail risk management
   - Distributional return prediction with uncertainty quantification
   - Dynamic risk budget allocation

4. **Adaptive Learning**
   - Continual model updates with new market data
   - Regime-aware adaptation
   - Online learning with stability controls (design goal; performance is deployment-dependent)

5. **Human Oversight Integration**
   - Real-time monitoring dashboard
   - Kill switch for immediate system halt
   - Manual override capabilities

### 3.2 Performance Characteristics

| Metric | Specification |
|--------|--------------|
| **Latency** | Deployment-dependent; measured and agreed during pilot/SLA |
| **Throughput** | Deployment-dependent |
| **System Availability** | Deployment/SLA-dependent |
| **Recovery Time** | Deployment/runbook-dependent |

### 3.3 Accuracy Metrics (Article 13(3)(b)(ii))

CustodiaCloud does not publish performance or accuracy targets in documentation. Accuracy/suitability must be evaluated by the customer in their own environment. Evidence exports and audit trails are designed to support review, governance, and validation.

---

## 4. Known Limitations

### 4.1 System Limitations (Article 13(3)(b)(iii))

The following limitations apply to this AI system:

1. **Market Condition Sensitivity**
   - Performance may degrade during extreme market volatility
   - Black swan events may exceed model assumptions
   - Regime changes may require model adaptation period

2. **Data Dependency**
   - Requires stable, low-latency data feed connectivity
   - Data quality directly impacts output quality
   - Historical data biases may affect predictions

3. **Operational Constraints**
   - Designed for professional/institutional use only
   - Requires human oversight for production deployment
   - Not intended for retail investors

4. **Model Uncertainty**
   - Predictions have inherent uncertainty
   - Confidence intervals should be considered
   - Not financial advice - human judgment required

5. **Infrastructure Requirements**
   - Minimum hardware specifications must be met
   - Network latency affects execution quality
   - Regular maintenance required

---

## 5. Known Risks and Mitigations

### 5.1 Risk Categories (Article 13(3)(b)(iv))

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Market Risk** | Potential for financial loss due to market movements | Position limits, stop-loss mechanisms, diversification |
| **Model Risk** | Risk of model underperformance or failure | Regular validation, monitoring, kill switch, conformal prediction |
| **Operational Risk** | Risk of system failures or errors | Redundancy, failover systems, monitoring alerts |
| **Data Quality Risk** | Risk from corrupted or delayed data | Data validation, multiple sources, staleness checks |
| **Cybersecurity Risk** | Risk from security threats | Input validation, encryption, access controls |
| **Liquidity Risk** | Risk of inability to execute at desired prices | Position sizing, liquidity analysis, execution algorithms |

### 5.2 Risk Mitigation Measures

1. **Real-Time Monitoring**
   - Continuous performance tracking
   - Automated anomaly detection
   - Alert system for threshold breaches

2. **Kill Switch Mechanism**
   - Immediate system halt capability
   - Multiple trigger mechanisms
   - Clear escalation procedures

3. **Position Limits**
   - Maximum position size per asset
   - Account-level risk limits (operator-configured)
   - Dynamic adjustment based on volatility

4. **Data Validation**
   - Input data quality checks
   - Staleness detection
   - Multiple data source verification

---

## 6. Human Oversight Measures

### 6.1 Oversight Requirements (Article 13(3)(b)(v))

This AI system is designed to support effective human oversight aligned with Article 14 of the EU AI Act (deployment-dependent; not a compliance claim).

### 6.2 Oversight Capabilities

| Capability | Description | Implementation |
|------------|-------------|----------------|
| **Understanding** | Operators can understand system capabilities and limitations | Training materials, documentation, dashboards |
| **Monitoring** | Real-time monitoring of system operation | Monitoring dashboard, metrics, alerts |
| **Anomaly Detection** | Ability to detect anomalies and dysfunctions | Automated detection, visual indicators |
| **Interpretation** | Ability to correctly interpret AI output | Explainability features, confidence metrics |
| **Override** | Ability to override AI decisions | Manual override controls |
| **Intervention** | Ability to intervene or stop system | Kill switch, pause function |

### 6.3 Kill Switch Operations

The system includes a kill switch per Article 14(4)(f):

| Action | Method | Effect |
|--------|--------|--------|
| **Emergency Stop** | Button / API / CLI | Immediate halt of Agent-controlled live execution |
| **Pause** | Dashboard control | Temporary suspension |
| **Position Close** | Emergency command | Close all open positions |
| **System Shutdown** | Full stop | Complete system shutdown |

### 6.4 Operator Requirements

Operators must:

- Complete system training program
- Understand AI capabilities and limitations
- Monitor system during operation
- Review significant decisions
- Be aware of automation bias risks
- Know emergency procedures

---

## 7. Technical Requirements

### 7.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 16 GB | 32-64 GB |
| **Storage** | 50 GB SSD | 100+ GB NVMe SSD |
| **GPU** | Not required | NVIDIA CUDA-capable (for training) |
| **Network** | 100 Mbps | 1 Gbps, low latency |

### 7.2 Software Requirements

| Software | Version |
|----------|---------|
| **Operating System** | Linux (Ubuntu 20.04+) or Windows 10+ |
| **Python** | 3.10 or higher |
| **PyTorch** | 2.0 or higher |
| **Docker** | 20.10+ (optional) |

### 7.3 Network Requirements

- Low-latency connection to data providers
- Stable internet connectivity
- Firewall configuration for API access
- VPN support for secure connections

---

## 8. Installation and Configuration

### 8.1 Installation Steps

1. **Clone Repository**

   ```bash
   git clone [authorized-repository-url]
   cd AI-Powered-Quantitative-Research-Platform
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize Database**

   ```bash
   python scripts/init_database.py
   ```

5. **Verify Installation**

   ```bash
   python -m pytest tests/ -v
   ```

### 8.2 Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Environment variables and API keys |
| `config.yaml` | System configuration |
| `risk_config.yaml` | Risk management parameters |
| `logging_config.yaml` | Logging configuration |

---

## 9. Operation Instructions

### 9.1 Pre-Operation Checklist

- [ ] Verify data feed connectivity
- [ ] Check system health metrics
- [ ] Review risk parameters
- [ ] Enable monitoring dashboard
- [ ] Verify kill switch functionality
- [ ] Confirm human oversight in place

### 9.2 Starting the System

1. Start monitoring dashboard
2. Initialize data connections
3. Load model and configuration
4. Begin in paper mode (simulation)
5. Validate model outputs before live execution
6. Enable live execution (via Agent) with reduced size
7. Gradually increase position limits

### 9.3 During Operation

- Monitor performance metrics continuously
- Review alerts and anomalies
- Validate significant strategy and configuration changes
- Document any interventions
- Maintain oversight presence

### 9.4 Shutdown Procedure

1. Pause new runs / disable deployment
2. Review open positions
3. Close positions if required
4. Save system state
5. Stop services gracefully
6. Document session summary

---

## 10. Logging and Audit Trail

### 10.1 Logging Capabilities (Article 13(3)(c))

The system automatically logs:

| Event Type | Data Captured | Retention |
|------------|---------------|-----------|
| **Strategy Outputs** | Output type, timestamps, confidence/metadata | Deployment-defined |
| **Predictions** | Predictions (if enabled), uncertainty metadata | Deployment-defined |
| **Orders** | Order events (local Agent); optional redacted/aggregated telemetry to Cloud | Deployment-defined |
| **Risk Events** | Threshold breaches, alerts | Deployment-defined |
| **Human Overrides** | Override actions, operator ID | Deployment-defined |
| **System Events** | Start, stop, errors, health | Deployment-defined |

### 10.2 Log Access

- Logs stored in tamper-evident format
- Chain hashing for integrity verification
- Role-based access control
- Audit trail for log access

---

## 11. Maintenance and Updates

### 11.1 Maintenance Schedule

| Frequency | Activity |
|-----------|----------|
| **Daily** | Review logs, check health metrics |
| **Weekly** | Validate model performance, data quality |
| **Monthly** | Comprehensive performance review |
| **Quarterly** | Model revalidation, stress testing |
| **Annually** | Full system audit, conformity review |

### 11.2 Update Procedures

Updates require:

1. Impact assessment
2. Testing in staging environment
3. Approval per QMS procedures
4. Controlled deployment
5. Post-update validation
6. Documentation update

### 11.3 Expected Lifetime

| Component | Expected Lifetime |
|-----------|-------------------|
| **Software Platform** | Continuous updates |
| **Model Architecture** | 1-2 years before major revision |
| **Model Weights** | Regular retraining (quarterly+) |

---

## 12. Incident Reporting

### 12.1 Reporting Requirements

Incidents must be reported if they:

- Cause or could cause serious harm
- Involve system malfunction
- Result in significant financial loss
- Violate regulatory requirements

### 12.2 Incident Reporting Procedure

1. Activate kill switch if needed
2. Document incident details
3. Notify provider immediately
4. Preserve logs and evidence
5. Complete incident report form
6. Cooperate with investigation

### 12.3 Contact for Incidents

- **Email**: incidents@provider.com
- **Phone**: +XX XXX XXX XXXX (target: 24/7 capacity-dependent)
- **Portal**: https://provider.com/incident-report

---

## 13. Regulatory Compliance

### 13.1 EU AI Act Alignment

This system is designed to align with:

- Article 9: Risk Management System
- Article 10: Data Governance
- Article 11: Technical Documentation
- Article 12: Record-Keeping
- Article 13: Transparency
- Article 14: Human Oversight
- Article 15: Accuracy, Robustness, Cybersecurity
- Article 17: Quality Management System

### 13.2 Related Regulations

Users must also comply with:

- MiFID II/MiFIR (algorithmic trading requirements)
- MAR (market abuse prevention)
- Local financial regulations
- Data protection regulations (GDPR)

---

## 14. Glossary

| Term | Definition |
|------|------------|
| **CVaR** | Conditional Value at Risk - risk measure for tail losses |
| **PPO** | Proximal Policy Optimization - reinforcement learning algorithm |
| **Kill Switch** | Emergency mechanism to halt system operation |
| **Conformal Prediction** | Statistical method for uncertainty quantification |
| **QMS** | Quality Management System |

---

## 15. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-08 | Compliance Team | Initial release |

---

## 16. Legal Disclaimer

CustodiaCloud is provided as a B2B software/ICT system for quantitative research and customer-controlled deployment via the Agent (CCEA). While markets involve substantial risk of loss, CustodiaCloud does not provide investment advice, portfolio management, or trade recommendations.

- This system does not constitute financial advice
- Past performance does not guarantee future results
- Users are responsible for their own trading decisions
- Compliance with local regulations is the user's responsibility

The provider makes no warranties regarding trading outcomes. Users should only trade with capital they can afford to lose.

---

**End of Instructions for Use**

For questions or clarifications, contact: support@provider.com

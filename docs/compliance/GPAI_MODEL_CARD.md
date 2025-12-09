# GPAI Model Card

**Model**: Distributional PPO Trading Model
**Version**: 4.0
**Provider**: AI-Powered Quantitative Research Platform
**Release Date**: 2025-12-01
**Card Version**: 1.0
**EU AI Act Reference**: Article 53(1)(b)

---

## 1. Model Overview

| Attribute | Value |
|-----------|-------|
| Architecture | LSTM + Distributional Value Network (C51-style, 21 quantiles) |
| Training Approach | Reinforcement Learning (PPO with Twin Critics, Self-Adversarial Training) |
| Parameters | 2,500,000 |
| Input Format | Normalized feature vector (OHLCV + 50 technical indicators, shape: [sequence_length, 55]) |
| Output Format | Action distribution (buy/hold/sell probs) + value distribution (21 quantiles) + uncertainty estimate |

## 2. Intended Use

### 2.1 Primary Uses

- Trading Signal Generation
- Research And Backtesting
- Risk Assessment
- Market Analysis

### 2.2 Out of Scope Uses

- Personalized investment advice for individuals
- Credit scoring or lending decisions
- Insurance pricing or underwriting
- Fully autonomous trading without human oversight
- High-frequency trading (latency requirements not met)
- Trading of illiquid or exotic assets
- Regulatory compliance decisions
- Financial advice requiring fiduciary duty

**Important**: Using this model for out-of-scope applications may result in
unreliable outputs and potential harm. Users are responsible for ensuring
appropriate use.

## 3. Performance

### 3.1 Metrics

| Metric | Value | Context | Dataset |
|--------|-------|---------|---------|
| Sharpe Ratio | 1.2  | BTC/USDT, ETH/USDT backtest | test |
| Max Drawdown | 15.0 % | Worst-case across all test periods | test |
| Win Rate | 52.0 % | Directional accuracy | test |
| Sortino Ratio | 1.8  | Downside risk-adjusted return | test |
| Calmar Ratio | 0.8  | Return/Max Drawdown | test |

### 3.2 Evaluation Methodology

Walk-forward validation with 70/15/15 train/val/test split. Test period: 2023-01 to 2024-12 (out-of-sample). Metrics computed across 10 random seeds for statistical significance. Transaction costs of 0.1% included in all calculations.

## 4. Limitations

- **Technical** (medium): Requires minimum 100ms inference latency for real-time use
  - *Mitigation*: Use batched inference or deploy on GPU for lower latency
- **Performance** (high): May underperform during unprecedented market events (black swans)
  - *Mitigation*: Implement kill switch and human oversight per Article 14
- **Performance** (medium): Trained primarily on liquid assets; illiquid assets not recommended
  - *Mitigation*: Only use for assets with daily volume > $10M
- **Data** (medium): Training data ends December 2024; may not capture recent market dynamics
  - *Mitigation*: Retrain periodically with updated data
- **Operational** (high): Requires stable data feed; missing data degrades performance
  - *Mitigation*: Implement data quality checks and fallback mechanisms

## 5. Known Biases

- **Temporal Bias**: Better performance in trending vs. ranging markets
  - Impact: May generate false signals during low-volatility periods
  - Status: Partially mitigated through regime detection
- **Asset Bias**: Optimized for major cryptocurrencies (BTC, ETH), may not generalize
  - Impact: Lower accuracy on smaller altcoins or traditional assets
  - Status: Documented; users advised to validate on target assets
- **Regime Bias**: Trained mostly on 2020-2024 data (post-COVID bull market dominated)
  - Impact: May underperform in prolonged bear markets
  - Status: Mitigated through adversarial training (SA-PPO)

## 6. Ethical Considerations

- **Financial Risk**: Model outputs should not be used as sole decision basis for trading
  - Guidance: Always combine with human judgment and risk management
- **User Understanding**: Users must understand AI limitations before live trading
  - Guidance: Implement mandatory disclosure per Article 50
- **Loss Potential**: Significant financial losses are possible
  - Guidance: Users should only trade with funds they can afford to lose
- **Vulnerable Users**: Not suitable for users without trading experience
  - Guidance: Implement user qualification checks

## 7. Requirements for Downstream Providers

Per Article 53(1)(b), downstream providers integrating this model must:

- **DR-001** (Article 14(4)(f)): Implement kill switch for immediate trading halt
  - Mandatory: Provide UI button and API endpoint for immediate halt
- **DR-002** (Article 12): Log all model outputs for audit trail
  - Mandatory: Store predictions, confidence, timestamp for minimum 5 years
- **DR-003** (Article 50): Display AI disclosure to end users
  - Mandatory: Show disclosure before first use and in all AI outputs
- **DR-004** (Article 14): Maintain human oversight capability
  - Mandatory: Human must be able to review and override all decisions
- **DR-005** (Article 9(2)): Implement position and drawdown limits
  - Mandatory: Set maximum position size and daily loss limits
- **DR-006** (Article 72): Monitor model performance in production
  - Mandatory: Track accuracy drift and alert on degradation

## 8. Human Oversight Recommendations

Per Article 14 of the EU AI Act, the following oversight measures are recommended:

- Monitor model performance daily with automated alerts
- Review all anomalous signals before execution (>2 std dev from mean)
- Set position limits per trade (recommend max 5% of portfolio)
- Set daily drawdown limit (recommend max 3% of portfolio)
- Maintain ability to halt all trading within 1 second
- Review weekly performance reports for drift detection
- Conduct monthly model performance review meetings
- Implement escalation procedures for model failures

## 9. EU AI Act Classification

| Attribute | Value |
|-----------|-------|
| Classification | General-Purpose AI Model (GPAI) |
| Relevant Articles | Article 53, Article 50, Article 52, Article 12, Article 14 |
| Compliance Deadline | August 2, 2026 |

## 10. Contact Information

| Type | Contact |
|------|---------|
| Technical Support | ai-support@platform.com |
| Compliance Inquiries | compliance@platform.com |
| General | info@platform.com |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-10 | Initial release |

---

*This model card is provided in accordance with Article 53(1)(b) of Regulation (EU) 2024/1689 (EU AI Act).*

*Last Updated: 2025-12-10*

---

## Appendix A: Technical Specifications

### A.1 Model Architecture

```
Input Layer (55 features)
    │
    ▼
LSTM Layer (256 units, 3 layers)
    │
    ├──────────────────┬──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
Policy Head        Value Head 1       Value Head 2
(Softmax)         (21 quantiles)    (21 quantiles)
    │                  │                  │
    ▼                  ▼                  ▼
Action Probs      Value Dist 1      Value Dist 2
(3 actions)       (Twin Critic)     (Twin Critic)
```

### A.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-4 (adaptive) |
| Batch Size | 64 |
| Rollout Length | 256 |
| PPO Clip | 0.2 |
| Entropy Coefficient | 0.01 |
| Value Loss Coefficient | 0.5 |
| Max Gradient Norm | 0.5 |
| GAE Lambda | 0.95 |
| Discount Factor | 0.99 |

### A.3 Input Features

| Category | Features | Count |
|----------|----------|-------|
| OHLCV | open, high, low, close, volume | 5 |
| Moving Averages | SMA(5,10,20,50,200), EMA(12,26) | 7 |
| RSI | RSI(14), RSI(7) | 2 |
| MACD | MACD, Signal, Histogram | 3 |
| Bollinger Bands | Upper, Middle, Lower, %B, Width | 5 |
| ATR | ATR(14), ATR(7) | 2 |
| Momentum | ROC, MOM, Williams %R | 3 |
| Volume | OBV, VWAP, Volume MA | 3 |
| Volatility | Historical Vol, Realized Vol | 2 |
| Other | Various normalized indicators | 23 |
| **Total** | | **55** |

---

## Appendix B: Downstream Integration Checklist

Use this checklist to verify compliance when integrating this model:

### Pre-Integration

- [ ] Read and understand this Model Card completely
- [ ] Assess suitability for intended use case
- [ ] Verify use case is not in "Out of Scope Uses"
- [ ] Review all limitations and biases
- [ ] Prepare human oversight infrastructure

### Implementation

- [ ] Implement kill switch (DR-001)
- [ ] Set up logging system (DR-002)
- [ ] Implement AI disclosure UI (DR-003)
- [ ] Enable human override capability (DR-004)
- [ ] Configure position/drawdown limits (DR-005)
- [ ] Set up performance monitoring (DR-006)

### Pre-Deployment

- [ ] Test kill switch functionality
- [ ] Verify logs are being captured correctly
- [ ] Test disclosure display in all contexts
- [ ] Test human override functionality
- [ ] Verify limits are enforced
- [ ] Validate monitoring alerts

### Post-Deployment

- [ ] Monitor daily performance metrics
- [ ] Review weekly performance reports
- [ ] Conduct monthly compliance reviews
- [ ] Track and report any incidents
- [ ] Update model card if modifications made

---

## Appendix C: Risk Matrix

| Risk ID | Category | Description | Likelihood | Impact | Mitigation | Owner |
|---------|----------|-------------|------------|--------|------------|-------|
| R-001 | Performance | Model drift | Medium | High | Monitoring + retraining | Platform |
| R-002 | Operational | Data feed failure | Low | Critical | Fallback feeds | Integrator |
| R-003 | Market | Black swan event | Low | Critical | Kill switch | Integrator |
| R-004 | Security | Adversarial inputs | Low | Medium | Input validation | Both |
| R-005 | Compliance | Disclosure failure | Medium | High | UI enforcement | Integrator |

---

*End of Document*

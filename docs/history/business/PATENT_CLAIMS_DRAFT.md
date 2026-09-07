# Patent Claims Draft

## CONFIDENTIAL - ATTORNEY-CLIENT PRIVILEGED

**Document Status:** INTERNAL DRAFT for Patent Counsel Review (NOT FILED)
**Last Updated:** 2025-12-05
**Target Filing (if pursued):** Q1 2025 (Provisional) - Subject to counsel review and FTO clearance

---

## Patent Application #1

### Title

**System and Method for Risk-Aware Reinforcement Learning in Financial Order Execution Using Distributional Value Functions and Conditional Value-at-Risk Optimization**

### Technical Field

This invention relates to machine learning systems for automated trading, specifically to reinforcement learning methods that incorporate risk measures into policy optimization for order execution in financial markets.

### Background of the Invention

#### Prior Art Analysis

**Academic Prior Art:**

1. Tamar et al. (2015) "Policy Gradient for Coherent Risk Measures" - NeurIPS
   - Proposes CVaR constraints in policy gradient
   - **Limitation:** Does not integrate CVaR into distributional value estimation

2. Dabney et al. (2018) "Distributional Reinforcement Learning with Quantile Regression" - AAAI
   - Introduces QR-DQN with quantile value functions
   - **Limitation:** No risk-aware policy optimization, focused on value estimation only

3. Chow et al. (2017) "Risk-Constrained Reinforcement Learning with Percentile Risk Criteria"
   - CVaR as constraint, not objective
   - **Limitation:** Separate constraint handling, not embedded in value function

**Commercial Prior Art:**

1. Two Sigma / Citadel execution algorithms
   - Traditional optimal execution (Almgren-Chriss)
   - **Limitation:** Not RL-based, no distributional risk modeling

2. JP Morgan LOXM
   - RL for execution but uses standard value functions
   - **Limitation:** No CVaR integration, single-point value estimates

**Patent Prior Art Search (Preliminary Internal Review):**

> **⚠️ STALE SEARCH WARNING**: The preliminary search below was conducted in **December 2024**. As of this document's date, this search is **approximately 12 months old** and likely stale. Patent databases change frequently; any reliance on this search for filing or commercial decisions requires an **updated professional FTO analysis**.

> **IMPORTANT**: This is a **preliminary internal search** using Google Patents and USPTO public search (conducted December 2024). This is **NOT** a professional Freedom-to-Operate (FTO) analysis. A comprehensive FTO search by qualified patent counsel is required before any filing or public "patent pending" claims. Per `docs/DOCUMENTATION_CANON_DESIGN.md` Section 4.5: avoid absolute claims like "no patents found" or "freedom to operate confirmed".

Sample patents reviewed (non-exhaustive):

- US10,614,520 (Goldman Sachs) - "Reinforcement learning for trading" - General RL, no CVaR integration identified
- US11,048,741 (Two Sigma) - "Execution optimization" - Not RL-based per patent abstract

**Preliminary note (non-exhaustive; not a conclusion):** The internal search was limited and should not be used to infer freedom-to-operate or the absence of relevant patents.

**Required before filing**: Professional FTO report from patent counsel covering:

- Comprehensive USPTO/EPO/WIPO database search
- Non-patent literature review (academic publications, conference papers)
- Industry-specific patent landscape analysis (fintech/algorithmic trading)
- Freedom-to-operate opinion for key claims

### Summary of the Invention

The present invention provides a novel system and method for training and deploying reinforcement learning agents for financial order execution that:

1. **Integrates CVaR directly into the value function** rather than as a constraint
2. **Uses distributional value estimation** with quantile regression for tail risk
3. **Employs dual value networks (Twin Critics)** to reduce overestimation bias in risk-sensitive settings
4. **Applies conformal prediction** to provide distribution-free uncertainty bounds on risk estimates
5. **Maintains sim-to-live parity** through continuous monitoring and adaptive policy adjustment

### Claims

#### Independent Claim 1 - System

A computer-implemented system for risk-aware automated order execution comprising:

(a) a policy network configured to output action distributions for order execution decisions;

(b) a dual value network architecture ("Twin Critics") comprising:
    - a first value network producing a first distribution of quantile values;
    - a second value network producing a second distribution of quantile values;
    wherein the system uses the minimum of the two value estimates to compute target values;

(c) a CVaR computation module configured to:
    - receive the quantile distributions from both value networks;
    - compute Conditional Value-at-Risk at a specified confidence level α;
    - integrate the CVaR estimate into policy gradient updates;

(d) a conformal prediction module configured to:
    - calibrate prediction intervals using held-out data;
    - provide coverage-guaranteed bounds on CVaR estimates;
    - adjust position sizing based on uncertainty width;

(e) a simulation environment capable of operating at multiple fidelity levels including:
    - L2 statistical execution modeling;
    - L3 limit order book simulation with queue position tracking;

(f) a sim-to-live monitoring module configured to:
    - compute parity metrics between simulated and live execution;
    - trigger policy adjustments when divergence exceeds threshold.

#### Independent Claim 2 - Method

A computer-implemented method for training a risk-aware order execution agent comprising:

(a) collecting experience tuples (s, a, r, s', d) from a simulated trading environment;

(b) computing distributional temporal difference targets using:
    - quantile regression with Huber loss;
    - the minimum value estimate from dual value networks;

(c) computing CVaR-weighted advantage estimates by:
    - identifying quantiles below the α-th percentile;
    - applying weighted averaging to tail quantiles;
    - combining CVaR advantage with standard advantage via coefficient λ;

(d) updating the policy network using a clipped surrogate objective that incorporates:
    - standard policy gradient term;
    - CVaR-weighted policy gradient term;
    - entropy regularization;

(e) periodically calibrating conformal prediction intervals on held-out data;

(f) adjusting position limits based on conformal interval width.

#### Dependent Claims

**Claim 3** (depends on 1): The system of claim 1 wherein the CVaR computation module computes CVaR using linear interpolation between adjacent quantile estimates.

**Claim 4** (depends on 1): The system of claim 1 wherein the dual value network architecture shares feature extraction layers and diverges only at output heads.

**Claim 5** (depends on 1): The system of claim 1 wherein the conformal prediction module implements Conformalized Quantile Regression (CQR).

**Claim 6** (depends on 2): The method of claim 2 wherein the CVaR confidence level α is in the range [0.01, 0.10].

**Claim 7** (depends on 2): The method of claim 2 further comprising applying Variance Gradient Scaling to normalize gradient magnitudes across network layers.

**Claim 8** (depends on 1): The system of claim 1 wherein the L3 simulation includes:
    - market impact modeling using Almgren-Chriss framework;
    - queue position estimation using market-by-price data;
    - hidden liquidity detection for iceberg orders.

**Claim 9** (depends on 2): The method of claim 2 wherein the policy network outputs are transformed using adaptive activation functions selected based on action space bounds.

**Claim 10** (depends on 1): The system of claim 1 wherein sim-to-live parity metrics include:
    - fill rate comparison;
    - slippage distribution matching;
    - adverse selection measurement.

### Detailed Description

#### Figure 1: System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Policy Network (Actor)                    │
│  [Observation] → [LSTM] → [MLP] → [Action Distribution]     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               Dual Value Networks (Twin Critics)             │
│  ┌─────────────────┐        ┌─────────────────┐             │
│  │   Critic 1      │        │   Critic 2      │             │
│  │ [Shared LSTM]   │        │ [Shared LSTM]   │             │
│  │     ↓           │        │     ↓           │             │
│  │ [MLP Head 1]    │        │ [MLP Head 2]    │             │
│  │     ↓           │        │     ↓           │             │
│  │ [Q1...Q21]      │        │ [Q1...Q21]      │             │
│  └────────┬────────┘        └────────┬────────┘             │
│           │                          │                       │
│           └──────────┬───────────────┘                       │
│                      ▼                                       │
│              min(Critic1, Critic2)                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    CVaR Computation                          │
│  Input: Quantile distribution [Q1, Q2, ..., Q21]            │
│  α = 0.05 (5th percentile)                                  │
│  CVaR = E[Q | Q ≤ Q_α] (expected shortfall)                 │
│  Output: CVaR estimate for advantage calculation            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               Conformal Prediction Bounds                    │
│  Calibration: compute conformity scores on held-out data    │
│  Inference: [CVaR_lower, CVaR_point, CVaR_upper]           │
│  Position scaling: scale ∝ 1/interval_width                 │
└─────────────────────────────────────────────────────────────┘
```

#### Implementation Details

**CVaR Computation (Novel Contribution):**

```python
def compute_cvar(quantiles: Tensor, alpha: float = 0.05) -> Tensor:
    """
    Compute CVaR from distributional value estimates.

    Novel aspect: Direct integration with distributional RL,
    using linear interpolation for non-integer quantile indices.
    """
    num_quantiles = quantiles.shape[-1]
    alpha_idx = alpha * (num_quantiles - 1)

    if alpha_idx <= 0:
        return quantiles[..., 0]

    floor_idx = int(alpha_idx)
    ceil_idx = min(floor_idx + 1, num_quantiles - 1)
    weight = alpha_idx - floor_idx

    # Weighted tail average
    tail_sum = quantiles[..., :floor_idx].sum(dim=-1)
    boundary_contribution = (1 - weight) * quantiles[..., floor_idx]
    boundary_contribution += weight * quantiles[..., ceil_idx]

    cvar = (tail_sum + boundary_contribution) / (floor_idx + 1)
    return cvar
```

**Twin Critics with Distributional Outputs (Novel Combination):**

Standard Twin Critics (Fujimoto et al., 2018) use scalar value outputs. Our novel contribution combines Twin Critics with distributional outputs:

```python
class DistributionalTwinCritics(nn.Module):
    def __init__(self, num_quantiles: int = 21):
        self.shared_features = SharedLSTMFeatures()
        self.critic_head_1 = QuantileHead(num_quantiles)
        self.critic_head_2 = QuantileHead(num_quantiles)

    def forward(self, obs) -> Tuple[Tensor, Tensor]:
        features = self.shared_features(obs)
        q1 = self.critic_head_1(features)  # Shape: [batch, num_quantiles]
        q2 = self.critic_head_2(features)  # Shape: [batch, num_quantiles]
        return q1, q2

    def get_min_quantiles(self, obs) -> Tensor:
        q1, q2 = self.forward(obs)
        return torch.min(q1, q2)  # Element-wise minimum
```

### Abstract

A system and method for training reinforcement learning agents for automated order execution in financial markets. The system integrates Conditional Value-at-Risk (CVaR) directly into distributional value estimation using a novel dual value network architecture. Twin Critics produce quantile distributions from which CVaR is computed and incorporated into policy gradient updates. Conformal prediction provides distribution-free uncertainty bounds enabling adaptive position sizing. The system operates across multiple simulation fidelity levels and maintains parity metrics for reliable sim-to-live deployment.

---

## Patent Application #2

### Title

**Unified Multi-Asset Simulation Environment with Continuous Fidelity Scaling for Algorithmic Trading System Development**

### Technical Field

This invention relates to simulation environments for developing and testing algorithmic trading systems, specifically to a unified architecture supporting multiple asset classes and execution fidelity levels.

### Summary of the Invention

The present invention provides a unified simulation environment that:

1. **Supports multiple asset classes** (crypto, equities, forex, futures) through a common interface
2. **Scales execution fidelity** from L1 (constant) through L3 (full LOB simulation)
3. **Automatically calibrates** simulation parameters from historical data
4. **Maintains consistent state representation** across all asset classes
5. **Provides parity metrics** quantifying simulation accuracy vs live execution

### Claims

#### Independent Claim 1 - System

A computer-implemented simulation environment for algorithmic trading development comprising:

(a) a unified observation space providing consistent state representation across:
    - cryptocurrency spot and perpetual markets;
    - equity markets including extended hours;
    - foreign exchange (OTC) markets;
    - regulated futures markets;

(b) a fidelity-scalable execution provider architecture comprising:
    - L1 provider using constant spread assumptions;
    - L2 provider using statistical market impact models;
    - L3 provider using full limit order book simulation;
    wherein providers share a common interface allowing seamless switching;

(c) an automatic calibration module configured to:
    - ingest historical market data including order book snapshots;
    - estimate market impact parameters per Almgren-Chriss framework;
    - calibrate queue dynamics from trade and quote data;

(d) an asset-class-specific fee and settlement module handling:
    - maker/taker fees for crypto markets;
    - regulatory fees (SEC, TAF) for US equities;
    - spread-based costs for forex;
    - SPAN margin and daily settlement for futures;

(e) a parity monitoring module computing divergence metrics between simulated and historical/live execution.

#### Dependent Claims

**Claim 2**: The system of claim 1 wherein the L3 provider includes:
    - FIFO price-time priority matching;
    - self-trade prevention with multiple modes;
    - hidden liquidity detection for iceberg orders.

**Claim 3**: The system of claim 1 wherein the calibration module estimates:
    - temporary impact coefficient η;
    - permanent impact coefficient γ;
    - queue arrival and cancellation rates.

**Claim 4**: The system of claim 1 wherein the parity monitoring module computes:
    - fill rate ratio (simulated vs actual);
    - slippage distribution KL divergence;
    - adverse selection correlation.

---

## Prior Art Differentiation Summary

| Aspect | Our Invention | Closest Prior Art | Differentiation |
|--------|---------------|-------------------|-----------------|
| CVaR in RL | Embedded in distributional value | Tamar (2015): constraint only | Direct integration, not constraint |
| Twin Critics | With distributional outputs | Fujimoto (2018): scalar only | Novel combination |
| Conformal bounds | On trading risk estimates | Romano (2019): general ML | Applied to trading RL (uncommon) |
| Multi-asset sim | Unified L1-L3 fidelity | Separate per-asset systems | Single architecture, common API |
| Auto-calibration | From LOB data | Manual parameter tuning | Automated, continuous |

---

## Next Steps for Patent Counsel

1. **Professional Prior Art Search & FTO (REQUIRED FIRST):**
   - Comprehensive USPTO/EPO/WIPO database search with claims language
   - Non-patent literature review (academic publications, NeurIPS/ICML/AAAI proceedings, quant finance journals)
   - Industry-specific patent landscape analysis (fintech, algorithmic trading, HFT patents)
   - Freedom-to-operate opinion addressing potential blocking patents
   - **Estimated cost:** €15K-30K per professional FTO report

2. **Claim Refinement:** Narrow/broaden based on FTO search results (subject to FTO clearing key claims)

3. **Commercial Assessment:** Assess patent value vs. filing costs (€20K-50K multi-jurisdiction) vs. trade secret protection

4. **Provisional Filing:** Target Q1 2025 for priority date (only if FTO clears key claims and commercial assessment positive)

5. **International Strategy:** PCT filing for EU, UK, Singapore, Japan (subject to FTO, budget, and disclosure risk analysis)

---

**Document Classification:** CONFIDENTIAL - Attorney-Client Privileged
**Prepared for:** Patent Counsel Review
**Contact:** [Legal Department]

**Document History:**

- v1.0 (2025-12-05): Initial draft for patent counsel review
- v1.1 (2025-12-19): **Due diligence audit corrections**: Replaced "No patents found" absolute claim with "Preliminary observation (subject to professional FTO validation)"; added critical disclaimer that internal search is NOT a professional FTO analysis; documented required professional FTO report deliverables and estimated costs; strengthened Next Steps to require FTO completion before filing decisions

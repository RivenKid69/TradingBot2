# Getting Started Guide

Welcome! This guide will take you from zero to running your first backtest in under 30 minutes.

> **Important**: This platform is a software tool for quantitative research and automated trading. It does not provide investment advice or guarantee any specific results. Trading involves substantial risk of loss. See [Risk Disclaimers](#risk-disclaimers) at the end of this guide.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Your First Backtest (5 minutes)](#your-first-backtest)
4. [Understanding Results](#understanding-results)
5. [Training Your First Model](#training-your-first-model)
6. [Going Live](#going-live)
7. [Next Steps](#next-steps)
8. [Troubleshooting](#troubleshooting)
9. [Risk Disclaimers](#risk-disclaimers)

---

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.10 or 3.12 (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space

### Knowledge Requirements
- Basic Python knowledge (functions, classes)
- Understanding of trading concepts (long/short, profit/loss)
- No ML experience required (we'll guide you!)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd TradingBot2
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python scripts/quickstart.py check crypto_momentum
```

**Expected output:**
```
✓ Python version: 3.12.0
✓ Dependencies installed
✓ Data directory exists
✓ Configuration valid
✓ Ready to run!
```

---

## Your First Backtest

### Option A: One-Command Quick Start

```bash
python scripts/quickstart.py run crypto_momentum
```

This runs a pre-configured momentum strategy on BTC/ETH historical data.

### Option B: Step-by-Step

#### 1. Check Available Presets

```bash
python scripts/quickstart.py list
```

**Output:**
```
Available Presets:
┌─────────────────┬─────────────┬───────────────────────┬────────────┐
│ Preset          │ Asset Class │ Strategy              │ Difficulty │
├─────────────────┼─────────────┼───────────────────────┼────────────┤
│ crypto_momentum │ Crypto Spot │ Momentum (BTC, ETH)   │ ⭐⭐       │
│ equity_swing    │ US Equity   │ Mean-Reversion        │ ⭐⭐       │
│ forex_carry     │ Forex OTC   │ Carry + Momentum      │ ⭐⭐⭐     │
│ crypto_perp     │ Futures     │ Funding Arbitrage     │ ⭐⭐⭐⭐   │
│ cme_index       │ CME Futures │ Index Momentum        │ ⭐⭐⭐⭐⭐ │
└─────────────────┴─────────────┴───────────────────────┴────────────┘
```

#### 2. Get Preset Info

```bash
python scripts/quickstart.py info crypto_momentum
```

**Output:**
```
Preset: crypto_momentum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Description:
  Simple momentum strategy on major cryptocurrencies.
  Buys when price is above 20-day SMA, sells when below.

Assets: BTC/USDT, ETH/USDT
Timeframe: 4 hours
Data Period: 2023-01-01 to 2024-01-01

Requirements:
  - No API keys needed (uses historical data)
  - ~500MB disk space for data

Note: Backtest results are hypothetical and do not guarantee future performance.
```

#### 3. Run Backtest

```bash
python scripts/quickstart.py run crypto_momentum
```

**Example output (hypothetical, for illustration purposes only):**
```
Loading data... ████████████████████ 100%
Running backtest...

=== Backtest Results ===
Period: 2023-01-01 to 2024-01-01
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Performance Metrics (Hypothetical):
  Total Return:     +42.3%
  Annual Return:    +42.3%
  Sharpe Ratio:     1.85
  Sortino Ratio:    2.41
  Max Drawdown:     -12.4%
  Win Rate:         58.2%
  Profit Factor:    1.73

Trade Statistics:
  Total Trades:     156
  Avg Trade:        +0.27%
  Best Trade:       +8.4%
  Worst Trade:      -4.2%
  Avg Duration:     18.5 hours

Results saved to: results/crypto_momentum_2024-12-04/

DISCLAIMER: These results are based on historical simulation.
Past performance does not guarantee future results.
Actual trading involves substantial risk of loss.
```

---

## Understanding Results

### Key Metrics Explained

| Metric | What It Means | Typical Range* |
|--------|---------------|----------------|
| **Total Return** | Overall profit/loss | Varies by strategy |
| **Sharpe Ratio** | Risk-adjusted return | 0.5-2.5 |
| **Max Drawdown** | Largest peak-to-trough drop | -5% to -50% |
| **Win Rate** | % of profitable trades | 40%-70% |
| **Profit Factor** | Gross profit / Gross loss | 1.0-3.0 |

*Typical ranges are for illustration only. Actual results will vary significantly based on market conditions, strategy parameters, and time period.

### Reading the Results Directory

```
results/crypto_momentum_2024-12-04/
├── summary.json          # Key metrics
├── trades.csv            # All trade details
├── equity_curve.png      # Portfolio value over time
├── drawdown.png          # Drawdown visualization
├── monthly_returns.png   # Monthly performance
└── config.yaml           # Configuration used
```

### Interpreting Backtest Results

**Important Considerations:**

1. **Backtests are hypothetical** - They use historical data and cannot predict future performance
2. **Overfitting risk** - Strong backtest results may not translate to live trading
3. **Transaction costs** - Ensure realistic fees and slippage are included
4. **Market regime** - Strategies that worked in past conditions may fail in new environments

---

## Training Your First Model

### Why Train?

Pre-built strategies use fixed rules. Training creates an ML model that:
- Adapts to changing market conditions
- Learns optimal position sizing
- Discovers patterns in data

> **Note**: ML models are tools, not guarantees. A trained model may underperform simple strategies depending on market conditions.

### Step 1: Prepare Training Data

Data is automatically downloaded for quick start presets. For custom data:

```bash
# Crypto
python scripts/fetch_binance_filters.py --universe

# Stocks
python scripts/download_stock_data.py --symbols AAPL MSFT GOOGL --start 2020-01-01
```

### Step 2: Start Training

```bash
python scripts/quickstart.py train crypto_momentum
```

**Example output (hypothetical):**
```
Starting training...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Episode 100/1000  │████░░░░░░░░░░░░░░░░│ 10%
  Avg Reward: 0.42
  Sharpe: 1.23
  Loss: 0.0234

Episode 200/1000  │████████░░░░░░░░░░░░│ 20%
  Avg Reward: 0.58
  Sharpe: 1.45
  Loss: 0.0189

... (training continues) ...

Training complete!
Model saved to: models/crypto_momentum_v1.pt

Note: Training metrics are internal measurements.
Out-of-sample validation is required before any live deployment.
```

### Step 3: Backtest Trained Model

```bash
python script_backtest.py --config configs/config_sim.yaml --model models/crypto_momentum_v1.pt
```

### Training Tips

| Tip | Explanation |
|-----|-------------|
| **Start simple** | Use default settings first |
| **Monitor overfitting** | Watch for validation divergence |
| **Save checkpoints** | Training auto-saves every 100 episodes |
| **Use GPU** | 5-10x faster with CUDA |

---

## Going Live

### ⚠️ Critical Warning

**Live trading involves real financial risk.** Before proceeding:

1. **Paper trade extensively** - Test with simulated money for weeks or months
2. **Understand the risks** - You can lose your entire investment
3. **Start very small** - Use minimum position sizes initially
4. **Never invest more than you can afford to lose**
5. **This is not investment advice** - You are solely responsible for your trading decisions

### Prerequisites

1. **Exchange Account**
   - Binance (crypto)
   - Alpaca (stocks)
   - OANDA (forex)
   - Interactive Brokers (futures/options)

2. **API Keys**
   - Generate API keys from your exchange
   - Store securely (never commit to git!)
   - Use IP whitelisting where available

### Step 1: Configure API Keys

Create `.env` file (never commit this!):

```bash
# Crypto (Binance)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# Stocks (Alpaca)
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here

# Forex (OANDA)
OANDA_API_KEY=your_key_here
OANDA_ACCOUNT_ID=your_account_id
```

### Step 2: Paper Trading (Required First Step)

**Always test with paper trading before using real money.**

**CCEA Architecture Note**: In production, live trading runs through the local Agent daemon which manages your credentials securely. The Cloud platform manages research, backtesting, and lifecycle - it **NEVER** stores your API keys or executes orders.

**Development/Testing (via script_live.py):**
```bash
# For development/testing only (not production CCEA architecture)
# Crypto (Binance Testnet)
python script_live.py --config configs/config_live.yaml --paper --dry-run

# Stocks (Alpaca Paper)
python script_live.py --config configs/config_live_alpaca.yaml --paper --dry-run
```

**Production (via local Agent daemon):**
```bash
# See docs/agent/INSTALLATION.md for full Agent setup
python -m packages.agent.daemon.agentd --config configs/agent.yaml
```

**Recommended paper trading period:** Minimum 4-8 weeks to observe strategy behavior across different market conditions.

### Step 3: Live Trading

**Only proceed after extensive paper trading and thorough risk assessment.**

**Production CCEA Architecture** (Recommended):
1. Deploy local Agent daemon (credentials stay on YOUR machine)
2. Cloud sends lifecycle commands only (start/stop/deploy)
3. See `docs/agent/INSTALLATION.md` for setup
4. See `docs/runbooks/KILL_SWITCH.md` for emergency procedures

**Development/Testing Only:**
```bash
# NOT for production - use Agent daemon instead
python script_live.py --config configs/config_live.yaml --dry-run
```

### Safety Features

| Feature | Description |
|---------|-------------|
| **Kill Switch** | Emergency stop via `services/ops_kill_switch.py` |
| **Position Limits** | Max position per symbol and total |
| **Drawdown Guard** | Auto-stop if drawdown exceeds limit |
| **Error Throttling** | Pause on repeated errors |

**Configure risk limits in `configs/risk.yaml` before going live.**

---

## Next Steps

### Level 1: Customize Strategies

1. Modify parameters in preset configs:
   ```yaml
   # configs/quickstart/crypto_momentum.yaml
   strategy:
     lookback_period: 20  # Try 10, 30, 50
     entry_threshold: 0.02  # Try 0.01, 0.03
   ```

2. Run backtest with modified config:
   ```bash
   python script_backtest.py --config configs/quickstart/crypto_momentum.yaml
   ```

### Level 2: Create Custom Strategies

1. Create strategy file:
   ```python
   # strategies/my_strategy.py
   from strategies.base import BaseStrategy

   class MyStrategy(BaseStrategy):
       def generate_signal(self, data):
           # Your logic here
           return signal
   ```

2. Register in config and run

### Level 3: Advanced Training

1. Try different asset classes
2. Experiment with PBT (Population-Based Training)
3. Add adversarial training for robustness

### Level 4: Multi-Asset Portfolio

1. Combine multiple strategies
2. Implement portfolio-level risk management
3. Cross-asset correlation analysis

---

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError"

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### "Data not found"

```bash
# Download required data
python scripts/quickstart.py check crypto_momentum
# Follow any missing data instructions
```

#### "API connection failed"

1. Check API keys in `.env`
2. Verify exchange status
3. Check network connection
4. For Binance: ensure IP is whitelisted

#### "Out of memory"

```yaml
# Reduce batch size in config
training:
  batch_size: 32  # Reduce from 64
  n_envs: 2       # Reduce from 4
```

### Getting Help

1. **Check logs**: `logs/` directory contains detailed logs
2. **Run tests**: `pytest tests/` to verify installation
3. **Documentation**: See [DOCS_INDEX.md](DOCS_INDEX.md)

---

## Quick Reference

### Essential Commands

```bash
# List presets
python scripts/quickstart.py list

# Check environment
python scripts/quickstart.py check <preset>

# Run backtest
python scripts/quickstart.py run <preset>

# Train model
python scripts/quickstart.py train <preset>

# Live trading (paper)
python script_live.py --config <config> --paper

# Run all tests
pytest tests/
```

### Directory Structure

```
TradingBot2/
├── configs/           # Configuration files
│   ├── quickstart/    # Pre-built strategy configs
│   └── *.yaml         # Main configs
├── data/              # Market data
├── models/            # Trained models
├── results/           # Backtest results
├── scripts/           # Utility scripts
├── strategies/        # Strategy implementations
├── tests/             # Automated tests
└── logs/              # Application logs
```

### Configuration Hierarchy

```
Default Config (core_config.py)
    └── Asset Class Config (asset_class_defaults.yaml)
        └── Strategy Config (quickstart/*.yaml)
            └── Command Line Arguments
```

---

## Risk Disclaimers

### Risk of Loss

**Trading in financial instruments carries a high level of risk and may not be suitable for all investors.** The high degree of leverage available in forex, futures, and crypto markets can work against you as well as for you. Before deciding to trade, you should carefully consider your investment objectives, level of experience, and risk appetite.

**There is a possibility that you could sustain a loss of some or all of your initial investment** and therefore you should not invest money that you cannot afford to lose. You should be aware of all the risks associated with trading and seek advice from an independent financial advisor if you have any doubts.

### Hypothetical Performance

**All backtest results and performance figures in this documentation are hypothetical** and are presented for illustration purposes only. They do not represent actual trading results. Hypothetical performance results have many inherent limitations, including:

1. **Hindsight bias** - Strategies are designed with the benefit of hindsight
2. **No slippage guarantee** - Actual execution prices may differ significantly
3. **No emotional factor** - Simulations don't account for psychological impacts on trading decisions
4. **Market conditions change** - Past market conditions may not repeat

### No Investment Advice

**This platform and documentation do not constitute investment advice.** The content is provided for informational and educational purposes only and should not be construed as financial, legal, or tax advice. The information does not take into account your specific circumstances and should not be relied upon in making financial decisions.

You should conduct your own research and consult with qualified professionals before making any investment decisions. The platform provider does not recommend that any specific investment, strategy, or asset is suitable for any specific person.

### Regulatory Compliance

This software is a technology tool and not a regulated financial service. Users are responsible for:

1. **Compliance with local laws** - Ensure trading is legal in your jurisdiction
2. **Tax obligations** - Report and pay taxes on trading profits
3. **Exchange requirements** - Follow the terms of service of your chosen exchanges
4. **Licensing requirements** - Obtain necessary licenses if required in your jurisdiction

### Platform Compliance & Privacy

The platform implements comprehensive regulatory compliance:

| Regulation | Status | Description |
|------------|--------|-------------|
| **MiFID II** | Toolkit Ready | Compliance tools for ICT Providers (kill switch, audit, TCA) |
| **EU AI Act** | Toolkit Ready | Tools designed to align with High-Risk AI System requirements |
| **DORA** | Toolkit Ready | Digital Operational Resilience controls implemented |
| **GDPR** | Toolkit Ready | Data protection and privacy controls implemented |

> **Note**: "Toolkit Ready" means compliance tools have been implemented and designed to help align with regulatory requirements. Independent certification or audit has not been performed. Consult qualified legal/compliance advisors for compliance assessment.

**Privacy by Design (CCEA Architecture):**
- Your broker credentials **NEVER** leave your local Agent
- Cloud **NEVER** executes trades or stores API keys
- Telemetry is **ALWAYS** redacted before transmission
- EU-only data residency enforced

For details, see:
- [Privacy Policy](../legal/PRIVACY_POLICY.md)
- [GDPR Compliance Summary](compliance/GDPR_COMPLIANCE_SUMMARY.md)
- [CCEA Privacy](../architecture/CCEA_PRIVACY.md)

### Software Disclaimer

The software is provided "as is" without warranty of any kind. The provider is not liable for any losses, damages, or costs arising from the use of this software. Users acknowledge that:

1. Software may contain bugs or errors
2. Market data may be delayed or inaccurate
3. System failures can occur
4. Past software performance does not guarantee future reliability

---

## Completion Checklist

You've completed the getting started guide. You now know how to:

- [x] Run backtests with pre-built strategies
- [x] Understand performance metrics (and their limitations)
- [x] Train ML models
- [x] Set up paper trading
- [x] Understand the risks involved

### What's Next?

| Goal | Resource |
|------|----------|
| Deep dive into architecture | [ARCHITECTURE.md](../ARCHITECTURE.md) |
| Understand execution models | [docs/bar_execution.md](bar_execution.md) |
| Advanced training options | [docs/UPGD_INTEGRATION.md](UPGD_INTEGRATION.md) |
| Full API reference | [CLAUDE.md](../CLAUDE.md) |

---

*Need help? Open an issue on GitHub or check [DOCS_INDEX.md](DOCS_INDEX.md) for more resources.*

*Last Updated: December 2025*

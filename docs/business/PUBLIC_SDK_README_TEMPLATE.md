# CCEA SDK

[![PyPI version](https://badge.fury.io/py/ccea-sdk.svg)](https://badge.fury.io/py/ccea-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.ccea.ai)

**Official Python SDK for CCEA Platform - Cloud-Controlled Execution Architecture**

## Overview

CCEA SDK provides easy access to the CCEA Cloud platform for:
- 📊 Market data streaming (crypto, equities, forex, futures)
- 🤖 Strategy development via REST/WebSocket APIs
- 📈 Backtesting and simulation in the Cloud
- 📦 Artifact management (strategy deployment)
- 🛡️ Risk monitoring and alerts

> **Note:** This SDK connects to the CCEA Cloud platform. For local execution, see the [CCEA Agent](https://github.com/ccea-platform/ccea-agent) repository. The Agent handles live order execution in your own environment.

## Installation

```bash
pip install ccea-sdk
```

## Quick Start

```python
from quantbot import CCEAClient

# Initialize client
client = CCEAClient(api_key="your-api-key")

# Get market data
bars = client.get_bars("BTCUSDT", timeframe="1h", limit=100)

# Submit backtest
backtest_id = client.submit_backtest(
    strategy="momentum",
    symbols=["BTCUSDT", "ETHUSDT"],
    start_date="2024-01-01",
    end_date="2024-12-01",
)

# Get results
results = client.get_backtest_results(backtest_id)
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

## Features

### Market Data

```python
# Real-time streaming
async for tick in client.stream_ticks(["AAPL", "MSFT"]):
    print(f"{tick.symbol}: {tick.price}")

# Historical OHLCV
df = client.get_ohlcv("ETHUSDT", "4h", start="2024-01-01")
```

### Signal Generation

```python
# Get trading signals from your deployed model
signals = client.get_signals(
    model_id="your-model-id",
    symbols=["BTCUSDT", "ETHUSDT"],
)
for signal in signals:
    print(f"{signal.symbol}: {signal.action} ({signal.confidence:.1%})")
```

### Risk Monitoring

```python
# Real-time portfolio risk
risk = client.get_portfolio_risk()
print(f"VaR (95%): ${risk.var_95:,.0f}")
print(f"CVaR (95%): ${risk.cvar_95:,.0f}")
print(f"Max Drawdown: {risk.max_drawdown:.1%}")
```

## Supported Exchanges

| Exchange | Market Data | Order Execution* |
|----------|-------------|------------------|
| Binance | ✅ | ✅ |
| Alpaca | ✅ | ✅ |
| OANDA | ✅ | ✅ |
| Interactive Brokers | ✅ | ✅ |
| Polygon.io | ✅ | N/A |

*Order execution requires Enterprise license

## Documentation

- [Getting Started Guide](https://docs.ccea.ai/getting-started)
- [API Reference](https://docs.ccea.ai/api-reference)
- [Examples](./examples/)
- [FAQ](https://docs.ccea.ai/faq)

## Examples

See the [examples/](./examples/) directory for:
- `basic_backtest.ipynb` - Running your first backtest
- `signal_generation.ipynb` - Working with trading signals
- `risk_monitoring.ipynb` - Portfolio risk analysis
- `multi_asset.ipynb` - Cross-asset strategies

## Enterprise Features

The SDK connects to CCEA AI's cloud platform. For advanced features, consider our Enterprise offering:

| Feature | SDK (Free) | Cloud Pro | Enterprise |
|---------|------------|-----------|------------|
| Market data | ✅ | ✅ | ✅ |
| Basic backtesting | ✅ | ✅ | ✅ |
| Signal generation | Limited | ✅ | ✅ |
| L3 LOB simulation | ❌ | ❌ | ✅ |
| Custom RL training | ❌ | ✅ | ✅ |
| On-premise deployment | ❌ | ❌ | ✅ |
| Dedicated support | ❌ | ✅ | ✅ |

[Contact Sales](https://ccea.ai/enterprise)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

By contributing, you agree to our [Contributor License Agreement](./CLA.md).

## License

This SDK is released under the [MIT License](./LICENSE).

**Note:** This license applies only to the SDK client code. The CCEA AI platform, including the RL execution engine and simulation infrastructure, is proprietary software available under separate license terms.

## Support

- 📧 Email: support@ccea.ai
- 💬 Discord: [CCEA Community](https://discord.gg/quantbot)
- 📚 Docs: [docs.ccea.ai](https://docs.ccea.ai)

---

**CCEA AI** - Institutional-Grade Quantitative Trading Platform

© 2025 CCEA AI. SDK released under MIT License. Platform proprietary.

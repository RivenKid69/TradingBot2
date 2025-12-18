# CustodiaCloud SDK — README Template (Internal)

**Status:** Template only (do not publish without review)  
**Canon (mandatory wording/guardrails):** `docs/DOCUMENTATION_CANON_DESIGN.md`

---

## Overview

CustodiaCloud SDK provides programmatic access to **CustodiaCloud Cloud** APIs for research, backtesting/simulation, monitoring, and lifecycle management within the **CCEA** architecture.

**CCEA boundary:** Live execution occurs only in the customer-controlled **CustodiaCloud Agent** running in the customer environment. The Cloud does not store broker credentials and does not send live trading instructions (orders/targets/signals).

---

## Installation (Placeholder)

```bash
pip install <custodiacloud-sdk-package>
```

---

## Quick Start (Placeholder)

```python
from custodiacloud_sdk import CustodiaCloudClient

client = CustodiaCloudClient(api_key="<api-key>")

# Example: submit a backtest/simulation job (equities-first example)
job_id = client.submit_backtest(
    strategy="<strategy-id>",
    symbols=["AAPL", "MSFT"],
    timeframe="1h",
)
print(job_id)
```

---

## Legal & Safety Notes (Required)

- CustodiaCloud is a B2B software/ICT product for professional trading organizations.
- CustodiaCloud does not provide investment advice, portfolio management, or trade recommendations.
- Past performance does not guarantee future results. Trading involves risk of loss.


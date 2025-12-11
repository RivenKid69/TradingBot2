# Archived MiFID II Financial Entity Modules

> ⚠️ **IMPORTANT: FOR FINANCIAL ENTITY CLIENTS ONLY**
>
> This platform is an **ICT Provider / Software Provider**, NOT an Investment Firm.
> These archived modules are provided for B2B clients who ARE Investment Firms under
> MiFID II and need compliance tools, or for building compliance products for the
> financial services market.
>
> **DO NOT** use these modules to classify or position the platform itself.

**Status:** Archived - FE modules not applicable to ICT Providers
**Version:** 1.0.0

## Platform Positioning

**What we ARE:**
- ICT Provider / Software Provider
- SaaS platform for algorithmic trading strategy development
- B2B service provider for financial institutions
- We provide software; clients trade via their own broker accounts with their own API keys

**What we are NOT:**
- Investment firm under MiFID II Art. 4(1)(1)
- Entity that executes trades on behalf of clients
- Entity that holds or manages client assets
- Entity that provides investment advice

## Why Archived?

Per MiFID II scope, ICT Providers who supply trading software but do not:
- Execute trades on behalf of clients
- Hold client assets
- Provide investment advice
- Operate a trading venue

Are **NOT Investment Firms** and these requirements do not apply to them.

## Modules

| Module | MiFID II Article | Description |
|--------|------------------|-------------|
| `config.py` | - | FE compliance configuration |
| `lei_manager.py` | - | LEI validation (ISO 17442) |
| `gleif_client.py` | - | GLEIF API integration |
| `transaction_report.py` | Art. 26 (MiFIR) | Transaction reporting (RTS 22) |
| `arm_client.py` | Art. 26 (MiFIR) | ARM submission client |
| `reporting_pipeline.py` | Art. 26 (MiFIR) | T+1 reporting pipeline |
| `self_assessment.py` | RTS 6 | Annual self-assessment |
| `governance.py` | Art. 17 | Policy document management |
| `compliance_policies.py` | Various | Policy templates |
| `nca_notification.py` | Art. 17(2) | NCA notification for algo trading |

## When to Use These Modules

Only if you are building a product **FOR Investment Firms** (your B2B clients) to manage
their own MiFID II compliance. These modules provide a reference implementation for:

- Investment firms using this platform for strategy development
- Banks with algo trading desks
- Building a MiFID II compliance SaaS product

## Usage

```python
# Import will emit a DeprecationWarning
from services.archive.mifid_financial_entity import (
    TransactionReportBuilder,
    ARMClient,
    NCANotificationManager,
)

# Direct import (no warning)
from services.archive.mifid_financial_entity.nca_notification import (
    NCANotificationManager,
    create_nca_notification_manager,
)
```

## References

- [MiFID II (Directive 2014/65/EU)](https://eur-lex.europa.eu/eli/dir/2014/65/oj)
- [MiFIR (Regulation 600/2014)](https://eur-lex.europa.eu/eli/reg/2014/600/oj)
- [RTS 6: Algo Trading Requirements](https://eur-lex.europa.eu/eli/reg_del/2017/589/oj)
- [RTS 22: Transaction Reporting](https://eur-lex.europa.eu/eli/reg_del/2017/590/oj)
- [ESMA ARM Register](https://www.esma.europa.eu/databases-library/registers-and-data/approved-reporting-mechanisms)

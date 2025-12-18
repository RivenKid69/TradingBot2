# Archive: MiFID II Financial Entity Modules

`from services.archive.mifid_financial_entity import ...`

**WARNING: These modules are NOT for ICT Providers!**

These modules implement MiFID II requirements specifically for **Investment Firms** (financial entities that execute client orders and run related reporting/controls). Importing this package emits a `DeprecationWarning`.

## Who Should Use This?

- **Investment Firms** regulated under MiFID II
- **Banks** with trading operations
- **Asset Managers** executing client orders
- **Market Makers** providing liquidity

## Who Should NOT Use This?

- **ICT Providers** (software vendors) - that's CustodiaCloud
- **Technology platforms** that don't hold client assets
- **SaaS providers** where users execute via their own broker accounts (customer-controlled)

## Module Overview

| Module | LOC | MiFID II Ref | Description |
|--------|-----|--------------|-------------|
| lei_manager | 661 | Article 26 MiFIR | LEI validation and management |
| gleif_client | 630 | Article 26 MiFIR | GLEIF API integration |
| transaction_report | 1,309 | RTS 22 | Transaction report generation |
| arm_client | 1,009 | RTS 22 | ARM submission |
| reporting_pipeline | 986 | RTS 22 | T+1 reporting pipeline |
| self_assessment | 1,405 | RTS 6 Art. 9 | Annual self-assessment |
| governance | 1,244 | RTS 6 Art. 3 | Policy document management |
| compliance_policies | 1,010 | Various | Policy templates |
| nca_notification | 1,233 | Article 17(2) | NCA notification |
| config | 500 | - | Configuration models |

## Import Warning

```python
import warnings

# This will emit a DeprecationWarning
from services.archive.mifid_financial_entity import LEIManager

# To suppress the warning (if you're sure you need these modules)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from services.archive.mifid_financial_entity import LEIManager
```

## Modules

### lei_manager

Legal Entity Identifier (LEI) management per ISO 17442.

```python
from services.archive.mifid_financial_entity.lei_manager import (
    LEIManager,
    LEIStatus,
    create_lei_manager,
)

manager = create_lei_manager(
    own_lei="549300EXAMPLE00001",
    cache_ttl_hours=24,
)

# Validate LEI
result = manager.validate_lei("549300COUNTERPARTY")
if result.valid:
    print(f"LEI valid: {result.entity_name}")
else:
    print(f"LEI invalid: {result.error}")

# Check before trade
can_trade = manager.check_before_trade("549300COUNTERPARTY")
```

### gleif_client

GLEIF (Global LEI Foundation) API integration.

```python
from services.archive.mifid_financial_entity.gleif_client import (
    GLEIFClient,
    create_gleif_client,
)

client = create_gleif_client()

# Look up LEI
response = client.lookup_lei("549300EXAMPLE00001")
if response.found:
    print(f"Entity: {response.entity.legal_name}")
    print(f"Status: {response.registration.status}")
    print(f"Next renewal: {response.registration.next_renewal_date}")
```

### transaction_report

RTS 22 transaction report generation.

```python
from services.archive.mifid_financial_entity.transaction_report import (
    TransactionReport,
    TransactionReportBuilder,
    BuySellIndicator,
    TradingCapacity,
)

# Build report using builder
report = (
    TransactionReportBuilder()
    .trading_datetime("2025-01-15T10:30:00Z")
    .trading_capacity(TradingCapacity.DEALING_ON_OWN_ACCOUNT)
    .instrument_id("ISIN", "US0378331005")
    .buy_sell(BuySellIndicator.BUY)
    .quantity(100)
    .price(150.25)
    .venue("XNAS")
    .executing_entity_lei("549300EXAMPLE00001")
    .counterparty_lei("549300COUNTERPARTY")
    .build()
)

# Validate report
errors = report.validate()
if errors:
    for error in errors:
        print(f"Validation error: {error}")
```

### arm_client

Approved Reporting Mechanism (ARM) submission.

```python
from services.archive.mifid_financial_entity.arm_client import (
    ARMClient,
    ARMProvider,
    ARMEnvironment,
    create_arm_client,
)

client = create_arm_client(
    provider=ARMProvider.BLOOMBERG_BTRL,
    environment=ARMEnvironment.PRODUCTION,
    api_key="your-api-key",
)

# Submit report
result = client.submit_report(report)
if result.success:
    print(f"Submitted: {result.submission_id}")
else:
    print(f"Failed: {result.error}")
```

### reporting_pipeline

T+1 transaction reporting pipeline.

```python
from services.archive.mifid_financial_entity.reporting_pipeline import (
    TransactionReportingPipeline,
    PipelineStatus,
    create_reporting_pipeline,
)

pipeline = create_reporting_pipeline(
    arm_client=client,
    batch_size=100,
)

pipeline.start()

# Queue report
pipeline.queue_report(report)

# Check status
metrics = pipeline.get_metrics()
print(f"Queued: {metrics.queued_count}")
print(f"Submitted: {metrics.submitted_count}")
print(f"Failed: {metrics.failed_count}")
```

### self_assessment

Annual self-assessment per RTS 6 Article 9.

```python
from services.archive.mifid_financial_entity.self_assessment import (
    AnnualSelfAssessment,
    AssessmentCategory,
    create_annual_assessment,
    get_rts6_assessment_template,
)

# Create assessment from template
assessment = create_annual_assessment(
    firm_name="Investment Firm Ltd",
    year=2025,
    template=get_rts6_assessment_template(),
)

# Answer questions
assessment.answer_question(
    category=AssessmentCategory.KILL_SWITCH,
    question_id="KS-001",
    answer="YES",
    evidence="Kill switch tested quarterly, last test 2025-01-10",
)

# Generate report
report = assessment.generate_report()
print(f"Compliance score: {report.overall_score}%")
```

### governance

Policy document management per RTS 6 Article 3.

```python
from services.archive.mifid_financial_entity.governance import (
    GovernanceFramework,
    PolicyType,
    create_governance_framework,
    create_algorithmic_trading_policy,
)

framework = create_governance_framework(
    firm_name="Investment Firm Ltd",
)

# Add policy
policy = create_algorithmic_trading_policy(
    firm_name="Investment Firm Ltd",
    approved_by="Board of Directors",
)
framework.add_policy(policy)

# Check policy status
status = framework.get_policy_status(PolicyType.ALGORITHMIC_TRADING)
print(f"Status: {status.status}")
print(f"Last review: {status.last_review_date}")
```

### nca_notification

National Competent Authority (NCA) notification per Article 17(2).

```python
from services.archive.mifid_financial_entity.nca_notification import (
    NCANotificationManager,
    NCAJurisdiction,
    NotificationType,
    create_nca_notification_manager,
)

manager = create_nca_notification_manager(
    jurisdiction=NCAJurisdiction.UK_FCA,
    firm_lei="549300EXAMPLE00001",
)

# Create notification
notification = manager.create_notification(
    notification_type=NotificationType.ALGORITHM_DEPLOYMENT,
    algorithm_id="ALGO-001",
    description="New VWAP algorithm deployment",
)

# Submit
result = manager.submit_notification(notification.id)
if result.success:
    print(f"Notification submitted: {result.reference}")
```

## Configuration

```python
from services.archive.mifid_financial_entity.config import (
    MiFIDIIComplianceConfig,
    LEIConfig,
    load_mifid_compliance_config,
)

# Load from file
config = load_mifid_compliance_config("config/mifid_compliance.yaml")

# Or create programmatically
config = MiFIDIIComplianceConfig(
    enabled=True,
    lei=LEIConfig(
        own_lei="549300EXAMPLE00001",
        gleif_api_url="https://api.gleif.org/api/v1",
        cache_ttl_hours=24,
    ),
    nca_jurisdiction="UK_FCA",
)
```

## Why Archived?

These modules are archived because:

1. **ICT Provider positioning**: Our platform is positioned as a software provider, not an Investment Firm
2. **Regulatory clarity**: MiFID II transaction reporting, LEI management, and NCA notifications apply to Investment Firms, not software vendors
3. **B2B toolkit**: These modules are now part of our B2B compliance toolkit for enterprise clients who ARE Investment Firms
4. **Separation of concerns**: Clear separation between universal risk controls and firm-specific compliance

## Migration from services.compliance

If you were using these modules from `services.compliance`, update your imports:

```python
# Old
from services.compliance import LEIManager, TransactionReport

# New
from services.archive.mifid_financial_entity import LEIManager, TransactionReport
```

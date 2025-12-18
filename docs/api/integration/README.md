# Algo Integration API (B2B Alignment/Evidence Toolkit)

`from services.algo_integration import ...`

MiFID II-related alignment/evidence tooling for enterprise financial institution clients. These modules are designed to support client assessments and internal workflows; they are not a certification claim and do not replace legal/compliance review.

## Module Overview

| Module | LOC | MiFID II Ref | Description |
|--------|-----|--------------|-------------|
| best_execution | 1,371 | Article 27 | Best execution policy and analysis |
| tca_compliance | 1,010 | Article 27 | Transaction cost analysis |
| venue_analysis | 1,092 | Article 27 | Venue performance and smart order routing |
| execution_quality_report | 1,123 | Article 27 | Execution quality reports |
| otr_monitor | 1,086 | RTS 6 | Order-to-trade ratio monitoring |
| algorithm_registry | 789 | Article 17(2) | Algorithm registration and tracking |
| conformance_testing | 1,466 | RTS 6 Art. 5 | Conformance testing framework |
| test_scenarios | 1,137 | RTS 6 Art. 5 | Standard test scenarios |
| certification | 1,080 | RTS 6 Art. 7 | Deployment attestation (internal evidence artifact) |
| config | 450 | - | Configuration models |

## Quick Import

```python
# Full import
from services.algo_integration import *

# Specific imports (recommended)
from services.algo_integration import (
    # Configuration
    AlgoIntegrationConfig,
    AlgorithmRegistryConfig,

    # Best Execution
    BestExecutionAnalyzer,
    BestExecutionPolicy,
    create_best_execution_analyzer,

    # TCA
    TCAComplianceWrapper,
    create_tca_wrapper,

    # Venue Analysis
    VenueAnalyzer,
    SmartOrderRouter,
    create_venue_analyzer,

    # OTR
    OTRMonitor,
    create_otr_monitor,

    # Registry
    AlgorithmRegistry,
    create_algorithm_registry,

    # Testing
    ConformanceTestRunner,
    create_test_runner,

    # Certification
    CertificateManager,
    create_certificate_manager,
)
```

## Modules

### best_execution

Best execution policy implementation per MiFID II Article 27.

```python
from services.algo_integration.best_execution import (
    BestExecutionAnalyzer,
    BestExecutionPolicy,
    ExecutionFactor,
    create_best_execution_analyzer,
    create_best_execution_policy,
    get_standard_eu_venues,
)

# Create policy
policy = create_best_execution_policy(
    firm_name="Investment Firm Ltd",
    primary_factors=[
        ExecutionFactor.PRICE,
        ExecutionFactor.COST,
        ExecutionFactor.SPEED,
    ],
)

# Create analyzer
analyzer = create_best_execution_analyzer(
    policy=policy,
    venues=get_standard_eu_venues(),
)

# Analyze execution
analysis = analyzer.analyze_execution(
    order_id="ORD-001",
    instrument_id="AAPL",
    executed_price=150.25,
    executed_quantity=100,
    venue_id="XNAS",
)

print(f"Quality: {analysis.quality_level}")
print(f"Price improvement: {analysis.price_improvement_bps} bps")
```

### tca_compliance

Transaction Cost Analysis for regulatory reporting.

```python
from services.algo_integration.tca_compliance import (
    TCAComplianceWrapper,
    TCABenchmark,
    create_tca_wrapper,
)

tca = create_tca_wrapper()

# Pre-trade estimate
estimate = tca.pre_trade_estimate(
    instrument_id="AAPL",
    side="BUY",
    quantity=1000,
    benchmark=TCABenchmark.ARRIVAL_PRICE,
)

print(f"Expected slippage: {estimate.expected_slippage_bps} bps")
print(f"Expected market impact: {estimate.expected_market_impact_bps} bps")

# Post-trade analysis
analysis = tca.post_trade_analysis(
    order_id="ORD-001",
    instrument_id="AAPL",
    arrival_price=150.00,
    executed_price=150.15,
    executed_quantity=1000,
)

print(f"Implementation shortfall: {analysis.implementation_shortfall_bps} bps")
```

### venue_analysis

Venue performance analysis and smart order routing.

```python
from services.algo_integration.venue_analysis import (
    VenueAnalyzer,
    SmartOrderRouter,
    create_venue_analyzer,
    create_smart_order_router,
)

# Venue analyzer
analyzer = create_venue_analyzer()

# Record execution
analyzer.record_execution(
    venue_id="XLON",
    instrument_id="VOD.L",
    executed_price=120.50,
    executed_quantity=500,
    latency_ms=15.5,
)

# Get venue metrics
metrics = analyzer.get_venue_metrics("XLON")
print(f"Fill rate: {metrics.fill_rate_pct}%")
print(f"Avg latency: {metrics.avg_latency_ms} ms")

# Smart order router
router = create_smart_order_router(analyzer=analyzer)
decision = router.route_order(
    instrument_id="VOD.L",
    side="BUY",
    quantity=1000,
)

print(f"Selected venue: {decision.selected_venue}")
print(f"Reason: {decision.selection_reason}")
```

### execution_quality_report

RTS 27/28 execution quality reports.

```python
from services.algo_integration.execution_quality_report import (
    ExecutionQualityReportGenerator,
    ReportPeriod,
    ReportFormat,
    create_report_generator,
)

generator = create_report_generator(
    firm_name="Investment Firm Ltd",
)

# Generate quarterly report
report = generator.generate_report(
    period=ReportPeriod.QUARTERLY,
    year=2025,
    quarter=1,
)

# Export to different formats
generator.export_report(report, ReportFormat.PDF, "q1_2025_report.pdf")
generator.export_report(report, ReportFormat.XML, "q1_2025_report.xml")
```

### otr_monitor

Order-to-Trade Ratio monitoring per RTS 6.

```python
from services.algo_integration.otr_monitor import (
    OTRMonitor,
    OrderEvent,
    create_otr_monitor,
)

monitor = create_otr_monitor(
    warning_ratio=3.0,
    critical_ratio=4.0,
)

monitor.start()

# Record events
monitor.record_event(OrderEvent.ORDER_SUBMITTED, algorithm_id="ALGO-001")
monitor.record_event(OrderEvent.ORDER_FILLED, algorithm_id="ALGO-001")
monitor.record_event(OrderEvent.ORDER_CANCELLED, algorithm_id="ALGO-001")

# Check OTR
metrics = monitor.get_metrics("ALGO-001")
print(f"Current OTR: {metrics.otr_ratio}")
print(f"Level: {metrics.level}")

if metrics.is_breach:
    print("WARNING: OTR breach detected!")
```

### algorithm_registry

Algorithm registration per MiFID II Article 17(2).

```python
from services.algo_integration.algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmStatus,
    create_algorithm_registry,
)

registry = create_algorithm_registry(
    firm_name="Investment Firm Ltd",
    contact_email="compliance@firm.com",
)

# Register algorithm
record = registry.register_algorithm(
    algo_id="VWAP-001",
    name="VWAP Execution Strategy",
    algo_type="VWAP",
    description="Volume Weighted Average Price execution",
    responsible_person="John Smith",
)

print(f"Registered: {record.algo_id}")
print(f"Status: {record.status}")

# Update status
registry.update_status("VWAP-001", AlgorithmStatus.PRODUCTION)

# Get all algorithms
algorithms = registry.get_all_algorithms()
for algo in algorithms:
    print(f"{algo.algo_id}: {algo.status}")
```

### conformance_testing

Conformance testing framework per RTS 6 Article 5.

```python
from services.algo_integration.conformance_testing import (
    ConformanceTestRunner,
    ConformanceTestSuite,
    TestCategory,
    create_test_runner,
    create_conformance_suite,
    get_standard_conformance_tests,
)

# Create test suite
suite = create_conformance_suite(
    algorithm_id="ALGO-001",
    tests=get_standard_conformance_tests(),
)

# Create runner
runner = create_test_runner()

# Run tests
results = runner.run_suite(suite)

print(f"Total: {results.total_tests}")
print(f"Passed: {results.passed}")
print(f"Failed: {results.failed}")

for test_result in results.test_results:
    status = "PASS" if test_result.passed else "FAIL"
    print(f"  {test_result.test_id}: {status}")
```

### test_scenarios

Standard test scenarios per RTS 6.

```python
from services.algo_integration.test_scenarios import (
    ScenarioExecutor,
    create_scenario_executor,
    get_kill_switch_scenarios,
    get_pre_trade_scenarios,
    get_stress_test_scenarios,
)

executor = create_scenario_executor()

# Get standard scenarios
scenarios = get_kill_switch_scenarios()

# Execute scenarios
for scenario in scenarios:
    result = executor.execute(scenario)
    print(f"{scenario.name}: {result.status}")
```

### certification

Deployment attestation per RTS 6 Article 7 (internal evidence artifact; not a regulatory certification claim).

```python
from services.algo_integration.certification import (
    CertificateManager,
    CertificateStatus,
    create_certificate_manager,
)

manager = create_certificate_manager(
    firm_name="Investment Firm Ltd",
)

# Create certificate
certificate = manager.create_certificate(
    algorithm_id="ALGO-001",
    test_suite_id="SUITE-001",
    environment="PRODUCTION",
)

# Approve certificate
manager.approve_certificate(
    certificate_id=certificate.id,
    approver="John Smith",
    notes="All tests passed, approved for production",
)

# Check status
cert = manager.get_certificate(certificate.id)
print(f"Status: {cert.status}")
print(f"Valid until: {cert.valid_until}")
```

## Configuration

```python
from services.algo_integration.config import (
    AlgoIntegrationConfig,
    AlgorithmRegistryConfig,
    load_algo_integration_config,
)

# Load from file
config = load_algo_integration_config("config/algo_integration.yaml")

# Or create programmatically
config = AlgoIntegrationConfig(
    enabled=True,  # Enable B2B compliance features
    algorithm_registry=AlgorithmRegistryConfig(
        registry_path="state/algo_registry.json",
        auto_register=True,
        require_responsible_person=True,
    ),
)
```

## Integration Example

Complete workflow for financial institution compliance:

```python
from services.algo_integration import (
    AlgorithmRegistry,
    ConformanceTestRunner,
    CertificateManager,
    create_algorithm_registry,
    create_test_runner,
    create_certificate_manager,
    get_standard_conformance_tests,
)

# 1. Register algorithm
registry = create_algorithm_registry(firm_name="Investment Firm Ltd")
algo = registry.register_algorithm(
    algo_id="TWAP-001",
    name="TWAP Strategy",
    algo_type="TWAP",
    responsible_person="Jane Doe",
)

# 2. Run conformance tests
runner = create_test_runner()
suite = create_conformance_suite(
    algorithm_id="TWAP-001",
    tests=get_standard_conformance_tests(),
)
results = runner.run_suite(suite)

# 3. Create internal evidence record if tests pass
if results.all_passed:
    manager = create_certificate_manager(firm_name="Investment Firm Ltd")
    cert = manager.create_certificate(
        algorithm_id="TWAP-001",
        test_suite_id=suite.id,
        environment="PRODUCTION",
    )
    manager.approve_certificate(cert.id, "Jane Doe")
    print(f"Evidence record created: {cert.id}")
else:
    print(f"Tests failed: {results.failed} failures")
```

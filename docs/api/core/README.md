# Core Risk Controls API

`from services.core.risk_controls import ...`

Universal risk management modules for all platform users. These modules implement essential trading safety features regardless of regulatory status.

## Module Overview

| Module | LOC | Description |
|--------|-----|-------------|
| audit_models | 920 | Audit record data models and builders |
| audit_storage | 1,751 | Multi-backend storage (Memory, SQLite, File) |
| audit_trail_writer | 941 | Write-once immutable audit trail |
| retention_policy | 918 | 5-7 year data retention management |
| time_sync | 647 | Clock synchronization (was compliance_clock) |
| kill_switch | 939 | Emergency stop (was enhanced_kill_switch) |
| pre_trade_controls | 1,059 | Pre-trade validation and limits |
| realtime_monitor | 1,199 | Real-time P&L and risk monitoring |
| bcp | 1,528 | Business continuity planning |
| config | 371 | Configuration models |

## Quick Import

```python
# Full import
from services.core.risk_controls import *

# Specific imports (recommended)
from services.core.risk_controls import (
    # Configuration
    RiskControlsConfig,
    TimeSyncConfig,
    PreTradeControlsConfig,

    # Audit
    AuditRecord,
    AuditRecordBuilder,
    AuditTrailWriter,
    create_audit_trail_writer,

    # Risk Controls
    EnhancedKillSwitch,
    create_enhanced_kill_switch,
    PreTradeControls,
    create_pre_trade_controls,
    RealTimeMonitor,
    create_realtime_monitor,

    # Utilities
    ComplianceClock,
    create_compliance_clock,
    BusinessContinuityPlan,
    create_business_continuity_plan,
)
```

## Modules

### audit_models

Data models for audit records. Based on MiFIR Article 25 requirements but applicable universally.

```python
from services.core.risk_controls.audit_models import (
    AuditEventType,
    AuditRecord,
    AuditRecordBuilder,
    create_order_submitted_record,
    create_order_filled_record,
)

# Create audit record using builder
record = (
    AuditRecordBuilder()
    .event_type(AuditEventType.ORDER_SUBMITTED)
    .order_id("ORD-12345")
    .algorithm_id("ALGO-001")
    .instrument_id("BTC-USDT")
    .price(50000.0)
    .quantity(1.5)
    .build()
)

# Or use factory function
record = create_order_submitted_record(
    order_id="ORD-12345",
    algorithm_id="ALGO-001",
    instrument_id="BTC-USDT",
    side="BUY",
    price=50000.0,
    quantity=1.5,
)
```

### audit_storage

Multiple storage backends for audit records.

```python
from services.core.risk_controls.audit_storage import (
    StorageBackendType,
    create_audit_storage,
)

# Memory storage (testing)
storage = create_audit_storage(StorageBackendType.MEMORY)

# SQLite storage (production)
storage = create_audit_storage(
    StorageBackendType.SQLITE,
    db_path="audit_trail.db"
)

# File storage (compliance export)
storage = create_audit_storage(
    StorageBackendType.FILE,
    file_path="audit_records.jsonl"
)
```

### audit_trail_writer

High-level audit trail writer with chain integrity.

```python
from services.core.risk_controls.audit_trail_writer import (
    AuditTrailWriter,
    create_audit_trail_writer,
    WriterMode,
)

writer = create_audit_trail_writer(
    firm_lei="549300EXAMPLE00001",
    mode=WriterMode.ASYNC,
    storage_type=StorageBackendType.SQLITE,
)

writer.start()
writer.write_order_submitted(
    order_id="ORD-001",
    instrument_id="AAPL",
    side="BUY",
    price=150.0,
    quantity=100,
)
writer.stop()
```

### kill_switch

Emergency stop functionality with multiple scopes.

```python
from services.core.risk_controls.kill_switch import (
    EnhancedKillSwitch,
    create_enhanced_kill_switch,
    KillSwitchScope,
    KillSwitchTriggerReason,
)

kill_switch = create_enhanced_kill_switch(
    algorithm_id="ALGO-001",
    firm_name="My Trading Firm",
)

kill_switch.start()

# Trigger kill switch
kill_switch.trigger(
    reason=KillSwitchTriggerReason.MANUAL,
    scope=KillSwitchScope.ALGORITHM,
    details="Manual stop requested",
)

# Check status
if kill_switch.is_triggered:
    print("Trading halted")
```

### pre_trade_controls

Pre-trade validation and risk limits.

```python
from services.core.risk_controls.pre_trade_controls import (
    PreTradeControls,
    create_pre_trade_controls,
)

controls = create_pre_trade_controls(
    max_order_value_eur=100_000,
    price_collar_pct=5.0,
    max_messages_per_second=50,
)

result = controls.check_order(
    instrument_id="BTC-USDT",
    side="BUY",
    price=50000.0,
    quantity=1.0,
    reference_price=49500.0,
)

if result.passed:
    # Execute order
    pass
else:
    print(f"Order rejected: {result.reason}")
```

### realtime_monitor

Real-time P&L and risk monitoring.

```python
from services.core.risk_controls.realtime_monitor import (
    RealTimeMonitor,
    create_realtime_monitor,
)

monitor = create_realtime_monitor(
    algorithm_id="ALGO-001",
    daily_loss_limit=-10_000,
    position_limit=100_000,
)

monitor.start()
monitor.update_pnl(pnl=-5_000)

alerts = monitor.get_active_alerts()
for alert in alerts:
    print(f"Alert: {alert.category} - {alert.message}")
```

### time_sync

Clock synchronization for RTS 25 compliance.

```python
from services.core.risk_controls.time_sync import (
    ComplianceClock,
    create_compliance_clock,
)

clock = create_compliance_clock(
    ntp_servers=["time.google.com", "pool.ntp.org"],
    max_offset_ms=100.0,
)

clock.start()
timestamp = clock.get_compliant_timestamp()
drift = clock.get_current_drift()
```

### bcp

Business continuity planning.

```python
from services.core.risk_controls.bcp import (
    BusinessContinuityPlan,
    create_business_continuity_plan,
    get_standard_bcp_scenarios,
)

bcp = create_business_continuity_plan(
    firm_name="My Trading Firm",
    scenarios=get_standard_bcp_scenarios(),
)

# Declare incident
incident = bcp.declare_incident(
    scenario_id="DATA_CENTER_FAILURE",
    description="Primary data center unreachable",
)

# Execute recovery
bcp.execute_recovery(incident.id)
```

## Configuration

```python
from services.core.risk_controls.config import (
    RiskControlsConfig,
    TimeSyncConfig,
    PreTradeControlsConfig,
    load_risk_controls_config,
)

# Load from file
config = load_risk_controls_config("config/risk_controls.yaml")

# Or create programmatically
config = RiskControlsConfig(
    enabled=True,
    time_sync=TimeSyncConfig(
        ntp_servers=["time.google.com"],
        max_offset_ms=100.0,
    ),
    pre_trade=PreTradeControlsConfig(
        price_collar_pct=5.0,
        max_order_value_eur=100_000,
    ),
)
```

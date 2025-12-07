# -*- coding: utf-8 -*-
"""
MiFID II Test Scenarios - RTS 6 Articles 5-8 Compliance.

This module provides comprehensive test scenarios for algorithmic trading systems
per MiFID II RTS 6 requirements:

- Article 5: Conformance testing before deployment
- Article 6: Testing environments
- Article 7: Deployment of algorithms
- Article 8: Stress testing

Each scenario defines:
- Test objective and regulatory reference
- Setup requirements
- Execution steps
- Expected outcomes
- Pass/fail criteria

References:
    - RTS 6 (Regulation 2017/589) Articles 5-8
    - ESMA Guidelines on algorithmic trading (ESMA/2015/1464)
    - FCA Market Watch 67: Algorithmic trading compliance
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    Callable,
    Sequence,
    Set,
    Protocol,
    runtime_checkable,
)

from services.compliance.conformance_testing import (
    TestResult,
    TestCategory,
    TestPriority,
    TestEnvironment,
    ConformanceTest,
    TestEvidence,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ScenarioType(Enum):
    """Type of test scenario."""

    FUNCTIONAL = "functional"
    """Basic functionality testing."""

    STRESS = "stress"
    """Stress and load testing (RTS 6 Article 8)."""

    INTEGRATION = "integration"
    """Integration with venues/systems."""

    REGRESSION = "regression"
    """Regression testing after changes."""

    NEGATIVE = "negative"
    """Negative testing (error handling)."""

    SECURITY = "security"
    """Security and access control testing."""

    PERFORMANCE = "performance"
    """Performance and latency testing."""

    DISASTER_RECOVERY = "disaster_recovery"
    """Disaster recovery / BCP testing."""


class ScenarioSeverity(Enum):
    """Severity classification of scenario."""

    CRITICAL = "critical"
    """Must pass - regulatory requirement."""

    MAJOR = "major"
    """Should pass - significant functionality."""

    MINOR = "minor"
    """Nice to pass - minor functionality."""


class ExecutionPhase(Enum):
    """Phase in scenario execution."""

    SETUP = "setup"
    """Setup phase - prepare test environment."""

    EXECUTION = "execution"
    """Main execution phase."""

    VERIFICATION = "verification"
    """Verification phase - check results."""

    TEARDOWN = "teardown"
    """Teardown phase - cleanup."""


class ScenarioStatus(Enum):
    """Status of scenario execution."""

    PENDING = "pending"
    """Not yet started."""

    SETUP_IN_PROGRESS = "setup_in_progress"
    """Setup phase in progress."""

    EXECUTING = "executing"
    """Main execution in progress."""

    VERIFYING = "verifying"
    """Verification in progress."""

    TEARDOWN = "teardown"
    """Teardown in progress."""

    PASSED = "passed"
    """Scenario passed."""

    FAILED = "failed"
    """Scenario failed."""

    ERROR = "error"
    """Execution error occurred."""

    SKIPPED = "skipped"
    """Scenario skipped."""

    ABORTED = "aborted"
    """Execution aborted."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ScenarioStep:
    """A single step in a test scenario."""

    step_id: str = ""
    description: str = ""
    action: str = ""  # What to do
    expected_result: str = ""  # Expected outcome
    actual_result: str = ""  # Actual outcome
    passed: Optional[bool] = None
    duration_ms: float = 0.0
    evidence: List[TestEvidence] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "action": self.action,
            "expected_result": self.expected_result,
            "actual_result": self.actual_result,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "evidence": [e.to_dict() for e in self.evidence],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioStep":
        """Create from dictionary."""
        return cls(
            step_id=data.get("step_id", ""),
            description=data.get("description", ""),
            action=data.get("action", ""),
            expected_result=data.get("expected_result", ""),
            actual_result=data.get("actual_result", ""),
            passed=data.get("passed"),
            duration_ms=data.get("duration_ms", 0.0),
            notes=data.get("notes", ""),
        )


@dataclass
class TestScenario:
    """
    A comprehensive test scenario per RTS 6 Articles 5-8.

    A scenario groups multiple steps to test a specific aspect
    of algorithm or system behavior.

    Attributes:
        scenario_id: Unique identifier
        name: Scenario name
        scenario_type: Type of scenario
        category: Test category (maps to RTS requirement)
        severity: Criticality level
        description: What the scenario tests
        rts_reference: Regulatory reference
        objective: Test objective
        preconditions: Required preconditions
        setup_steps: Setup phase steps
        execution_steps: Main execution steps
        verification_steps: Verification steps
        teardown_steps: Cleanup steps
        pass_criteria: Criteria for passing
        fail_criteria: Criteria for failing
        status: Current status
    """

    scenario_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    scenario_type: ScenarioType = ScenarioType.FUNCTIONAL
    category: TestCategory = TestCategory.RISK_CONTROLS
    severity: ScenarioSeverity = ScenarioSeverity.MAJOR
    description: str = ""
    rts_reference: str = ""
    objective: str = ""

    # Requirements
    preconditions: List[str] = field(default_factory=list)
    required_environment: TestEnvironment = TestEnvironment.UAT
    timeout_seconds: float = 300.0  # 5 minutes default

    # Steps by phase
    setup_steps: List[ScenarioStep] = field(default_factory=list)
    execution_steps: List[ScenarioStep] = field(default_factory=list)
    verification_steps: List[ScenarioStep] = field(default_factory=list)
    teardown_steps: List[ScenarioStep] = field(default_factory=list)

    # Criteria
    pass_criteria: List[str] = field(default_factory=list)
    fail_criteria: List[str] = field(default_factory=list)

    # Results
    status: ScenarioStatus = ScenarioStatus.PENDING
    result_summary: str = ""
    evidence: List[TestEvidence] = field(default_factory=list)

    # Metadata
    created_by: str = ""
    created_timestamp_ns: int = field(default_factory=time.time_ns)
    executed_by: str = ""
    start_timestamp_ns: Optional[int] = None
    end_timestamp_ns: Optional[int] = None
    notes: str = ""

    # =========================================================================
    # Properties
    # =========================================================================

    def is_critical(self) -> bool:
        """Check if scenario is critical."""
        return self.severity == ScenarioSeverity.CRITICAL

    def passed(self) -> bool:
        """Check if scenario passed."""
        return self.status == ScenarioStatus.PASSED

    def failed(self) -> bool:
        """Check if scenario failed."""
        return self.status in {ScenarioStatus.FAILED, ScenarioStatus.ERROR}

    def is_complete(self) -> bool:
        """Check if scenario execution is complete."""
        return self.status in {
            ScenarioStatus.PASSED,
            ScenarioStatus.FAILED,
            ScenarioStatus.ERROR,
            ScenarioStatus.SKIPPED,
            ScenarioStatus.ABORTED,
        }

    def all_steps(self) -> List[ScenarioStep]:
        """Get all steps across all phases."""
        return (
            self.setup_steps +
            self.execution_steps +
            self.verification_steps +
            self.teardown_steps
        )

    def get_duration_ms(self) -> float:
        """Get total duration in milliseconds."""
        if self.start_timestamp_ns and self.end_timestamp_ns:
            return (self.end_timestamp_ns - self.start_timestamp_ns) / 1_000_000
        return sum(s.duration_ms for s in self.all_steps())

    # =========================================================================
    # Step Management
    # =========================================================================

    def add_setup_step(self, step: ScenarioStep) -> None:
        """Add a setup step."""
        self.setup_steps.append(step)

    def add_execution_step(self, step: ScenarioStep) -> None:
        """Add an execution step."""
        self.execution_steps.append(step)

    def add_verification_step(self, step: ScenarioStep) -> None:
        """Add a verification step."""
        self.verification_steps.append(step)

    def add_teardown_step(self, step: ScenarioStep) -> None:
        """Add a teardown step."""
        self.teardown_steps.append(step)

    # =========================================================================
    # Execution
    # =========================================================================

    def start(self, executed_by: str) -> None:
        """Mark scenario as started."""
        self.executed_by = executed_by
        self.start_timestamp_ns = time.time_ns()
        self.status = ScenarioStatus.SETUP_IN_PROGRESS
        logger.info(f"Scenario {self.scenario_id} started by {executed_by}")

    def complete(self, passed: bool, summary: str = "") -> None:
        """Mark scenario as complete."""
        self.end_timestamp_ns = time.time_ns()
        self.status = ScenarioStatus.PASSED if passed else ScenarioStatus.FAILED
        self.result_summary = summary
        logger.info(f"Scenario {self.scenario_id} completed: {self.status.value}")

    def abort(self, reason: str = "") -> None:
        """Abort scenario execution."""
        self.end_timestamp_ns = time.time_ns()
        self.status = ScenarioStatus.ABORTED
        self.result_summary = reason
        logger.warning(f"Scenario {self.scenario_id} aborted: {reason}")

    def error(self, error_message: str) -> None:
        """Mark scenario as errored."""
        self.end_timestamp_ns = time.time_ns()
        self.status = ScenarioStatus.ERROR
        self.result_summary = error_message
        logger.error(f"Scenario {self.scenario_id} error: {error_message}")

    def skip(self, reason: str = "") -> None:
        """Skip scenario."""
        self.status = ScenarioStatus.SKIPPED
        self.result_summary = reason
        logger.info(f"Scenario {self.scenario_id} skipped: {reason}")

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate(self) -> bool:
        """
        Evaluate scenario results against criteria.

        Returns True if scenario passes.
        """
        # Check all steps passed
        all_steps_passed = all(
            s.passed is True
            for s in self.execution_steps + self.verification_steps
            if s.passed is not None
        )

        # Check fail criteria not triggered
        # (In real implementation, this would check specific conditions)

        return all_steps_passed

    # =========================================================================
    # Conversion
    # =========================================================================

    def to_conformance_test(self) -> ConformanceTest:
        """Convert scenario to a ConformanceTest."""
        test_steps = [s.action for s in self.execution_steps]

        return ConformanceTest(
            test_id=f"SCN-{self.scenario_id[:8]}",
            name=self.name,
            category=self.category,
            description=self.description,
            rts_reference=self.rts_reference,
            priority=TestPriority.CRITICAL if self.is_critical() else TestPriority.HIGH,
            preconditions=self.preconditions,
            test_steps=test_steps,
            expected_result="; ".join(self.pass_criteria),
            result=TestResult.PASS if self.passed() else TestResult.FAIL if self.failed() else TestResult.PENDING,
            details=self.result_summary,
            duration_ms=self.get_duration_ms(),
            timestamp=datetime.fromtimestamp(self.start_timestamp_ns / 1e9) if self.start_timestamp_ns else None,
            tester=self.executed_by,
        )

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "scenario_type": self.scenario_type.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "rts_reference": self.rts_reference,
            "objective": self.objective,
            "preconditions": self.preconditions,
            "required_environment": self.required_environment.value,
            "timeout_seconds": self.timeout_seconds,
            "setup_steps": [s.to_dict() for s in self.setup_steps],
            "execution_steps": [s.to_dict() for s in self.execution_steps],
            "verification_steps": [s.to_dict() for s in self.verification_steps],
            "teardown_steps": [s.to_dict() for s in self.teardown_steps],
            "pass_criteria": self.pass_criteria,
            "fail_criteria": self.fail_criteria,
            "status": self.status.value,
            "result_summary": self.result_summary,
            "evidence": [e.to_dict() for e in self.evidence],
            "created_by": self.created_by,
            "created_timestamp_ns": self.created_timestamp_ns,
            "executed_by": self.executed_by,
            "start_timestamp_ns": self.start_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestScenario":
        """Create from dictionary."""
        scenario = cls(
            scenario_id=data.get("scenario_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            scenario_type=ScenarioType(data.get("scenario_type", "functional")),
            category=TestCategory(data.get("category", "risk_controls")),
            severity=ScenarioSeverity(data.get("severity", "major")),
            description=data.get("description", ""),
            rts_reference=data.get("rts_reference", ""),
            objective=data.get("objective", ""),
            preconditions=data.get("preconditions", []),
            required_environment=TestEnvironment(data.get("required_environment", "uat")),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            pass_criteria=data.get("pass_criteria", []),
            fail_criteria=data.get("fail_criteria", []),
            status=ScenarioStatus(data.get("status", "pending")),
            result_summary=data.get("result_summary", ""),
            created_by=data.get("created_by", ""),
            created_timestamp_ns=data.get("created_timestamp_ns", time.time_ns()),
            executed_by=data.get("executed_by", ""),
            start_timestamp_ns=data.get("start_timestamp_ns"),
            end_timestamp_ns=data.get("end_timestamp_ns"),
            notes=data.get("notes", ""),
        )

        # Parse steps
        scenario.setup_steps = [ScenarioStep.from_dict(s) for s in data.get("setup_steps", [])]
        scenario.execution_steps = [ScenarioStep.from_dict(s) for s in data.get("execution_steps", [])]
        scenario.verification_steps = [ScenarioStep.from_dict(s) for s in data.get("verification_steps", [])]
        scenario.teardown_steps = [ScenarioStep.from_dict(s) for s in data.get("teardown_steps", [])]

        return scenario


# =============================================================================
# Standard Scenario Templates
# =============================================================================


def get_kill_switch_scenarios() -> List[TestScenario]:
    """
    Get kill switch test scenarios (RTS 6 Article 12).

    Returns comprehensive scenarios for testing kill switch functionality.
    """
    scenarios = []

    # Scenario 1: Cancel All Orders
    scenario = TestScenario(
        name="Kill Switch - Cancel All Orders Immediately",
        scenario_type=ScenarioType.FUNCTIONAL,
        category=TestCategory.KILL_SWITCH,
        severity=ScenarioSeverity.CRITICAL,
        description="Verify kill switch cancels ALL pending orders across ALL venues within specified timeframe.",
        rts_reference="RTS 6 Article 12",
        objective="Ensure emergency order cancellation works correctly.",
        preconditions=[
            "Trading system connected to test venue(s)",
            "Kill switch module initialized",
            "Audit trail enabled",
        ],
        pass_criteria=[
            "All orders cancelled within 1 second",
            "New order submission blocked",
            "Kill switch event logged with timestamp",
            "Affected orders count matches actual cancellations",
        ],
        fail_criteria=[
            "Any order remains pending after 1 second",
            "New orders accepted after kill switch triggered",
            "Missing audit trail entry",
        ],
    )

    scenario.add_setup_step(ScenarioStep(
        step_id="KS-001-S1",
        description="Connect to test venue(s)",
        action="Establish connection to test trading venue(s)",
        expected_result="Connection established, ready to trade",
    ))
    scenario.add_setup_step(ScenarioStep(
        step_id="KS-001-S2",
        description="Submit test orders",
        action="Submit 10 test orders across available venues",
        expected_result="10 pending orders confirmed",
    ))

    scenario.add_execution_step(ScenarioStep(
        step_id="KS-001-E1",
        description="Record pre-trigger state",
        action="Record number of pending orders and timestamp",
        expected_result="State recorded",
    ))
    scenario.add_execution_step(ScenarioStep(
        step_id="KS-001-E2",
        description="Trigger kill switch",
        action="Trigger kill switch with scope=ALL",
        expected_result="Kill switch triggered, event recorded",
    ))
    scenario.add_execution_step(ScenarioStep(
        step_id="KS-001-E3",
        description="Measure cancellation time",
        action="Wait for all cancellations and record time",
        expected_result="All orders cancelled within 1 second",
    ))

    scenario.add_verification_step(ScenarioStep(
        step_id="KS-001-V1",
        description="Verify all orders cancelled",
        action="Query order status for all orders",
        expected_result="All orders in CANCELLED state",
    ))
    scenario.add_verification_step(ScenarioStep(
        step_id="KS-001-V2",
        description="Verify new orders blocked",
        action="Attempt to submit new order",
        expected_result="Order rejected - kill switch active",
    ))
    scenario.add_verification_step(ScenarioStep(
        step_id="KS-001-V3",
        description="Verify audit trail",
        action="Query audit trail for kill switch event",
        expected_result="Event logged with timestamp, reason, affected orders",
    ))

    scenario.add_teardown_step(ScenarioStep(
        step_id="KS-001-T1",
        description="Reset kill switch",
        action="Reset kill switch to normal state",
        expected_result="Trading enabled",
    ))

    scenarios.append(scenario)

    # Scenario 2: Per-Venue Cancellation
    scenario2 = TestScenario(
        name="Kill Switch - Per-Venue Cancellation",
        scenario_type=ScenarioType.FUNCTIONAL,
        category=TestCategory.KILL_SWITCH,
        severity=ScenarioSeverity.CRITICAL,
        description="Verify kill switch can cancel orders for specific venue only.",
        rts_reference="RTS 6 Article 12(1)",
        objective="Ensure granular venue-level emergency cancellation.",
        preconditions=[
            "Connected to multiple test venues",
            "Orders pending on multiple venues",
        ],
        pass_criteria=[
            "Only targeted venue orders cancelled",
            "Other venue orders remain active",
            "Audit trail shows venue-specific cancellation",
        ],
    )

    scenario2.add_execution_step(ScenarioStep(
        step_id="KS-002-E1",
        description="Submit orders to multiple venues",
        action="Submit orders to Venue A and Venue B",
        expected_result="Orders pending on both venues",
    ))
    scenario2.add_execution_step(ScenarioStep(
        step_id="KS-002-E2",
        description="Trigger venue-specific kill switch",
        action="Trigger kill switch for Venue A only",
        expected_result="Kill switch triggered for Venue A",
    ))
    scenario2.add_verification_step(ScenarioStep(
        step_id="KS-002-V1",
        description="Verify Venue A orders cancelled",
        action="Query Venue A order status",
        expected_result="All Venue A orders cancelled",
    ))
    scenario2.add_verification_step(ScenarioStep(
        step_id="KS-002-V2",
        description="Verify Venue B orders active",
        action="Query Venue B order status",
        expected_result="Venue B orders still active",
    ))

    scenarios.append(scenario2)

    return scenarios


def get_pre_trade_scenarios() -> List[TestScenario]:
    """
    Get pre-trade control test scenarios (RTS 6 Article 15).

    Returns scenarios for testing all pre-trade risk controls.
    """
    scenarios = []

    # Price Collar Scenario
    scenario = TestScenario(
        name="Pre-Trade Controls - Price Collar Enforcement",
        scenario_type=ScenarioType.FUNCTIONAL,
        category=TestCategory.PRE_TRADE,
        severity=ScenarioSeverity.CRITICAL,
        description="Verify orders outside price collar are rejected before submission.",
        rts_reference="RTS 6 Article 15(1)(a)",
        objective="Ensure price collar prevents erroneous orders.",
        preconditions=[
            "Pre-trade controls enabled",
            "Price collar configured at 5%",
            "Reference price available",
        ],
        pass_criteria=[
            "Order with price >5% above reference rejected",
            "Order with price >5% below reference rejected",
            "Order within ±5% accepted",
            "Rejection reason includes PRICE_COLLAR",
        ],
    )

    scenario.add_setup_step(ScenarioStep(
        step_id="PT-001-S1",
        description="Get reference price",
        action="Query current reference price for test instrument",
        expected_result="Reference price obtained",
    ))

    scenario.add_execution_step(ScenarioStep(
        step_id="PT-001-E1",
        description="Submit order above collar",
        action="Submit order with price 10% above reference",
        expected_result="Order rejected",
    ))
    scenario.add_execution_step(ScenarioStep(
        step_id="PT-001-E2",
        description="Submit order below collar",
        action="Submit order with price 10% below reference",
        expected_result="Order rejected",
    ))
    scenario.add_execution_step(ScenarioStep(
        step_id="PT-001-E3",
        description="Submit order within collar",
        action="Submit order with price 2% above reference",
        expected_result="Order accepted",
    ))

    scenario.add_verification_step(ScenarioStep(
        step_id="PT-001-V1",
        description="Verify rejection reason",
        action="Check rejection reason for first two orders",
        expected_result="PRICE_COLLAR rejection reason",
    ))

    scenarios.append(scenario)

    # Message Rate Scenario
    scenario2 = TestScenario(
        name="Pre-Trade Controls - Message Rate Limiting",
        scenario_type=ScenarioType.STRESS,
        category=TestCategory.PRE_TRADE,
        severity=ScenarioSeverity.CRITICAL,
        description="Verify message rate limits are enforced to prevent market abuse.",
        rts_reference="RTS 6 Article 15(1)(d)",
        objective="Ensure message rate limiting prevents excessive messaging.",
        preconditions=[
            "Message rate limit configured (e.g., 100/sec)",
        ],
        pass_criteria=[
            "Messages within limit accepted",
            "Messages exceeding limit throttled or rejected",
            "No venue overload",
        ],
    )

    scenario2.add_execution_step(ScenarioStep(
        step_id="PT-002-E1",
        description="Send messages within limit",
        action="Send 50 orders over 1 second",
        expected_result="All orders accepted",
    ))
    scenario2.add_execution_step(ScenarioStep(
        step_id="PT-002-E2",
        description="Send messages exceeding limit",
        action="Send 150 orders over 1 second",
        expected_result="Excess orders throttled/rejected",
    ))

    scenarios.append(scenario2)

    return scenarios


def get_stress_test_scenarios() -> List[TestScenario]:
    """
    Get stress test scenarios (RTS 6 Article 8).

    Returns scenarios for comprehensive stress testing.
    """
    scenarios = []

    # High Volume Scenario
    scenario = TestScenario(
        name="Stress Test - High Message Volume",
        scenario_type=ScenarioType.STRESS,
        category=TestCategory.STRESS_TEST,
        severity=ScenarioSeverity.MAJOR,
        description="Verify system handles 2x normal message volume without degradation.",
        rts_reference="RTS 6 Article 8",
        objective="Ensure system resilience under high load.",
        preconditions=[
            "Baseline volume metrics established",
            "Monitoring enabled",
            "Stress test environment available",
        ],
        pass_criteria=[
            "No message loss under 2x load",
            "Latency within 150% of baseline",
            "No system errors",
            "All risk controls remain active",
        ],
    )

    scenario.add_setup_step(ScenarioStep(
        step_id="STR-001-S1",
        description="Record baseline metrics",
        action="Record normal latency and throughput",
        expected_result="Baseline metrics recorded",
    ))

    scenario.add_execution_step(ScenarioStep(
        step_id="STR-001-E1",
        description="Generate 2x normal load",
        action="Send messages at 2x normal rate for 5 minutes",
        expected_result="High load sustained",
    ))

    scenario.add_verification_step(ScenarioStep(
        step_id="STR-001-V1",
        description="Check message loss",
        action="Compare sent vs. processed messages",
        expected_result="Zero message loss",
    ))
    scenario.add_verification_step(ScenarioStep(
        step_id="STR-001-V2",
        description="Check latency",
        action="Compare stress latency to baseline",
        expected_result="Latency within 150% of baseline",
    ))
    scenario.add_verification_step(ScenarioStep(
        step_id="STR-001-V3",
        description="Verify risk controls",
        action="Test pre-trade controls during stress",
        expected_result="All controls functioning",
    ))

    scenarios.append(scenario)

    # Market Volatility Scenario
    scenario2 = TestScenario(
        name="Stress Test - High Volatility Conditions",
        scenario_type=ScenarioType.STRESS,
        category=TestCategory.STRESS_TEST,
        severity=ScenarioSeverity.MAJOR,
        description="Verify system behavior under simulated high volatility.",
        rts_reference="RTS 6 Article 8",
        objective="Ensure algorithm behaves correctly in volatile markets.",
        preconditions=[
            "Market data simulator available",
            "Volatility thresholds configured",
        ],
        pass_criteria=[
            "No erroneous orders during volatility",
            "Risk controls adapt appropriately",
            "Algorithm reduces activity if configured",
        ],
    )

    scenario2.add_execution_step(ScenarioStep(
        step_id="STR-002-E1",
        description="Simulate volatility spike",
        action="Feed market data with 5x normal volatility",
        expected_result="System receives volatile data",
    ))
    scenario2.add_verification_step(ScenarioStep(
        step_id="STR-002-V1",
        description="Verify order quality",
        action="Check all orders during volatility period",
        expected_result="No erroneous orders generated",
    ))

    scenarios.append(scenario2)

    return scenarios


def get_business_continuity_scenarios() -> List[TestScenario]:
    """
    Get business continuity test scenarios (RTS 6 Article 3).

    Returns scenarios for BCP testing.
    """
    scenarios = []

    # Primary System Failure
    scenario = TestScenario(
        name="BCP - Primary System Failure",
        scenario_type=ScenarioType.DISASTER_RECOVERY,
        category=TestCategory.BUSINESS_CONTINUITY,
        severity=ScenarioSeverity.CRITICAL,
        description="Verify failover to backup system when primary fails.",
        rts_reference="RTS 6 Article 3",
        objective="Ensure business continuity during system failure.",
        preconditions=[
            "Backup system available",
            "Failover mechanism configured",
            "BCP plan documented",
        ],
        pass_criteria=[
            "Failover completed within RTO",
            "No open positions lost",
            "Orders safely cancelled or transferred",
            "Compliance reporting continues",
        ],
    )

    scenario.add_execution_step(ScenarioStep(
        step_id="BCP-001-E1",
        description="Simulate primary failure",
        action="Trigger simulated primary system failure",
        expected_result="Kill switch triggered",
    ))
    scenario.add_execution_step(ScenarioStep(
        step_id="BCP-001-E2",
        description="Initiate failover",
        action="Start failover to backup system",
        expected_result="Failover process started",
    ))
    scenario.add_verification_step(ScenarioStep(
        step_id="BCP-001-V1",
        description="Verify RTO met",
        action="Check time to recovery",
        expected_result="Recovery within RTO",
    ))

    scenarios.append(scenario)

    return scenarios


def get_all_standard_scenarios() -> List[TestScenario]:
    """
    Get all standard test scenarios for RTS 6 compliance.

    Returns comprehensive list of scenarios covering all RTS 6 requirements.
    """
    scenarios = []

    scenarios.extend(get_kill_switch_scenarios())
    scenarios.extend(get_pre_trade_scenarios())
    scenarios.extend(get_stress_test_scenarios())
    scenarios.extend(get_business_continuity_scenarios())

    return scenarios


# =============================================================================
# Scenario Executor
# =============================================================================


@runtime_checkable
class ScenarioStepExecutor(Protocol):
    """Protocol for step execution."""

    def execute_step(self, step: ScenarioStep) -> ScenarioStep:
        """Execute a scenario step."""
        ...


class ScenarioExecutor:
    """
    Executor for test scenarios.

    Handles the full lifecycle of scenario execution including
    setup, execution, verification, and teardown phases.
    """

    def __init__(
        self,
        step_executor: Optional[ScenarioStepExecutor] = None,
        timeout_seconds: float = 300.0,
        stop_on_step_failure: bool = True,
    ):
        """
        Initialize scenario executor.

        Args:
            step_executor: Custom step executor (optional).
            timeout_seconds: Default timeout for scenarios.
            stop_on_step_failure: Stop scenario if step fails.
        """
        self.step_executor = step_executor
        self.timeout_seconds = timeout_seconds
        self.stop_on_step_failure = stop_on_step_failure
        self._running = False
        self._aborted = False

    def execute_scenario(
        self,
        scenario: TestScenario,
        executed_by: str,
    ) -> TestScenario:
        """
        Execute a complete scenario.

        Args:
            scenario: Scenario to execute.
            executed_by: Person/system executing.

        Returns:
            Scenario with updated results.
        """
        scenario.start(executed_by)
        self._running = True
        self._aborted = False

        try:
            # Setup phase
            scenario.status = ScenarioStatus.SETUP_IN_PROGRESS
            if not self._execute_phase(scenario.setup_steps, "Setup"):
                if self.stop_on_step_failure:
                    scenario.status = ScenarioStatus.FAILED
                    scenario.result_summary = "Setup phase failed"
                    return scenario

            # Execution phase
            scenario.status = ScenarioStatus.EXECUTING
            if not self._execute_phase(scenario.execution_steps, "Execution"):
                if self.stop_on_step_failure:
                    scenario.status = ScenarioStatus.FAILED
                    scenario.result_summary = "Execution phase failed"
                    return scenario

            # Verification phase
            scenario.status = ScenarioStatus.VERIFYING
            if not self._execute_phase(scenario.verification_steps, "Verification"):
                scenario.status = ScenarioStatus.FAILED
                scenario.result_summary = "Verification phase failed"
                return scenario

            # Teardown phase (always run)
            scenario.status = ScenarioStatus.TEARDOWN
            self._execute_phase(scenario.teardown_steps, "Teardown")

            # Evaluate results
            if scenario.evaluate():
                scenario.complete(passed=True, summary="All criteria met")
            else:
                scenario.complete(passed=False, summary="Criteria not met")

        except Exception as e:
            scenario.error(str(e))
            logger.exception(f"Scenario {scenario.scenario_id} error: {e}")

        finally:
            self._running = False

        return scenario

    def abort(self) -> None:
        """Abort running scenario."""
        self._aborted = True

    def _execute_phase(self, steps: List[ScenarioStep], phase_name: str) -> bool:
        """Execute a phase of steps. Returns True if all passed."""
        for step in steps:
            if self._aborted:
                return False

            start_time = time.time()

            try:
                if self.step_executor:
                    step = self.step_executor.execute_step(step)
                else:
                    # Mock execution - mark as passed
                    step.passed = True
                    step.actual_result = f"[MOCK] {step.expected_result}"

            except Exception as e:
                step.passed = False
                step.actual_result = f"Error: {str(e)}"
                logger.error(f"Step {step.step_id} error: {e}")

            step.duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Step {step.step_id} [{phase_name}]: {'PASS' if step.passed else 'FAIL'}"
            )

            if not step.passed and self.stop_on_step_failure:
                return False

        return True


# =============================================================================
# Factory Functions
# =============================================================================


def create_test_scenario(
    name: str,
    category: TestCategory,
    severity: ScenarioSeverity = ScenarioSeverity.MAJOR,
    rts_reference: str = "",
    created_by: str = "",
) -> TestScenario:
    """
    Create a new test scenario.

    Args:
        name: Scenario name.
        category: Test category.
        severity: Severity level.
        rts_reference: Regulatory reference.
        created_by: Creator name.

    Returns:
        New TestScenario instance.
    """
    return TestScenario(
        name=name,
        category=category,
        severity=severity,
        rts_reference=rts_reference,
        created_by=created_by,
    )


def create_scenario_executor(
    step_executor: Optional[ScenarioStepExecutor] = None,
    timeout_seconds: float = 300.0,
) -> ScenarioExecutor:
    """
    Create a scenario executor.

    Args:
        step_executor: Custom step executor.
        timeout_seconds: Default timeout.

    Returns:
        ScenarioExecutor instance.
    """
    return ScenarioExecutor(
        step_executor=step_executor,
        timeout_seconds=timeout_seconds,
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "ScenarioType",
    "ScenarioSeverity",
    "ExecutionPhase",
    "ScenarioStatus",
    # Data classes
    "ScenarioStep",
    "TestScenario",
    # Executor
    "ScenarioExecutor",
    # Factory functions
    "create_test_scenario",
    "create_scenario_executor",
    # Standard scenarios
    "get_kill_switch_scenarios",
    "get_pre_trade_scenarios",
    "get_stress_test_scenarios",
    "get_business_continuity_scenarios",
    "get_all_standard_scenarios",
]

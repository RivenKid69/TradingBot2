# -*- coding: utf-8 -*-
"""
MiFID II Conformance Testing - RTS 6 Article 5 Compliance.

This module implements conformance testing requirements for algorithmic trading
per MiFID II RTS 6 Article 5:

"Investment firms shall test the trading algorithm and trading system prior to
deployment or substantial update to ensure that:
(a) the algorithm or system is not likely to contribute to or create
    disorderly trading conditions;
(b) the algorithm or system operates in accordance with the requirements
    of this Regulation and the rules of the trading venues."

Key Requirements:
    - Pre-deployment testing of all algorithms
    - Testing after substantial updates
    - Documentation of test results
    - Conformance certification before go-live

References:
    - RTS 6 (Regulation 2017/589) Article 5: Testing conformance
    - RTS 6 Article 6: Testing environments
    - RTS 6 Article 7: Deployment and updates
    - RTS 6 Article 8: Stress testing
    - ESMA Guidelines on algorithmic trading (ESMA/2015/1464)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
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
    Type,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class TestResult(Enum):
    """Result of a conformance test."""

    PASS = "pass"
    """Test passed successfully."""

    FAIL = "fail"
    """Test failed - non-compliant behavior detected."""

    SKIP = "skip"
    """Test skipped (not applicable or precondition not met)."""

    ERROR = "error"
    """Test execution error (not a compliance failure)."""

    WARNING = "warning"
    """Test passed with warnings."""

    PENDING = "pending"
    """Test not yet executed."""


class TestCategory(Enum):
    """Category of conformance test per RTS 6."""

    RISK_CONTROLS = "risk_controls"
    """Risk control functionality (RTS 6 Articles 14-17)."""

    KILL_SWITCH = "kill_switch"
    """Kill switch / emergency measures (RTS 6 Article 12)."""

    PRE_TRADE = "pre_trade"
    """Pre-trade controls (RTS 6 Article 15)."""

    POST_TRADE = "post_trade"
    """Post-trade controls and monitoring."""

    CLOCK_SYNC = "clock_sync"
    """Clock synchronisation (RTS 25)."""

    RECORD_KEEPING = "record_keeping"
    """Audit trail and record keeping (MiFIR Article 25)."""

    TRANSACTION_REPORTING = "transaction_reporting"
    """Transaction reporting (RTS 22)."""

    BEST_EXECUTION = "best_execution"
    """Best execution (MiFID II Article 27)."""

    STRESS_TEST = "stress_test"
    """Stress testing (RTS 6 Article 8)."""

    CAPACITY = "capacity"
    """System capacity and resilience."""

    ALGORITHM_BEHAVIOR = "algorithm_behavior"
    """Algorithm-specific behavior tests."""

    INTEGRATION = "integration"
    """Integration with venues and external systems."""

    BUSINESS_CONTINUITY = "business_continuity"
    """Business continuity procedures (RTS 6 Article 3)."""


class TestPriority(Enum):
    """Priority of conformance test."""

    CRITICAL = "critical"
    """Must pass for deployment - regulatory requirement."""

    HIGH = "high"
    """Should pass - best practice."""

    MEDIUM = "medium"
    """Recommended to pass."""

    LOW = "low"
    """Nice to have."""


class TestEnvironment(Enum):
    """Testing environment per RTS 6 Article 6."""

    SANDBOX = "sandbox"
    """Isolated sandbox environment."""

    UAT = "uat"
    """User Acceptance Testing environment."""

    STAGING = "staging"
    """Pre-production staging environment."""

    PRODUCTION_REPLICA = "production_replica"
    """Production replica for realistic testing."""

    PRODUCTION = "production"
    """Live production environment (monitoring only)."""


class ConformanceSuiteStatus(Enum):
    """Status of a conformance test suite."""

    NOT_STARTED = "not_started"
    """Suite not yet started."""

    IN_PROGRESS = "in_progress"
    """Suite execution in progress."""

    COMPLETED = "completed"
    """Suite execution completed."""

    FAILED = "failed"
    """Suite failed (critical test(s) failed)."""

    PASSED = "passed"
    """Suite passed (all critical tests passed)."""

    PASSED_WITH_WARNINGS = "passed_with_warnings"
    """Suite passed but has warnings."""

    ABORTED = "aborted"
    """Suite execution aborted."""


class CertificationStatus(Enum):
    """Status of conformance certification."""

    PENDING = "pending"
    """Certification pending."""

    APPROVED = "approved"
    """Algorithm certified for deployment."""

    REJECTED = "rejected"
    """Certification rejected."""

    EXPIRED = "expired"
    """Certification expired."""

    REVOKED = "revoked"
    """Certification revoked."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TestEvidence:
    """Evidence collected during test execution."""

    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    evidence_type: str = "log"  # log, screenshot, data, metric
    description: str = ""
    content: str = ""
    file_path: Optional[str] = None
    timestamp_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "description": self.description,
            "content": self.content[:1000] if self.content else "",  # Truncate
            "file_path": self.file_path,
            "timestamp_ns": self.timestamp_ns,
        }


@dataclass
class ConformanceTest:
    """
    A single conformance test per RTS 6 Article 5.

    Each test verifies a specific regulatory requirement.

    Attributes:
        test_id: Unique identifier (e.g., "CT-001")
        name: Human-readable test name
        category: Test category
        description: What the test verifies
        rts_reference: Reference to regulation (e.g., "RTS 6 Article 12")
        priority: Test priority
        preconditions: Required preconditions
        test_steps: Steps to execute
        expected_result: Expected outcome
        result: Actual test result
        details: Result details/message
        duration_ms: Test execution duration
        evidence: Collected evidence
        timestamp: Test execution timestamp
        tester: Person/system running test
    """

    test_id: str = ""
    name: str = ""
    category: TestCategory = TestCategory.RISK_CONTROLS
    description: str = ""
    rts_reference: str = ""
    priority: TestPriority = TestPriority.HIGH

    # Test specification
    preconditions: List[str] = field(default_factory=list)
    test_steps: List[str] = field(default_factory=list)
    expected_result: str = ""

    # Execution results
    result: TestResult = TestResult.PENDING
    details: str = ""
    duration_ms: float = 0.0
    evidence: List[TestEvidence] = field(default_factory=list)

    # Metadata
    timestamp: Optional[datetime] = None
    tester: str = ""
    notes: str = ""

    def is_critical(self) -> bool:
        """Check if this is a critical test."""
        return self.priority == TestPriority.CRITICAL

    def passed(self) -> bool:
        """Check if test passed."""
        return self.result in {TestResult.PASS, TestResult.WARNING}

    def failed(self) -> bool:
        """Check if test failed."""
        return self.result in {TestResult.FAIL, TestResult.ERROR}

    def is_pending(self) -> bool:
        """Check if test is pending."""
        return self.result == TestResult.PENDING

    def add_evidence(self, evidence: TestEvidence) -> None:
        """Add evidence to test."""
        self.evidence.append(evidence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "rts_reference": self.rts_reference,
            "priority": self.priority.value,
            "preconditions": self.preconditions,
            "test_steps": self.test_steps,
            "expected_result": self.expected_result,
            "result": self.result.value,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "evidence": [e.to_dict() for e in self.evidence],
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "tester": self.tester,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConformanceTest":
        """Create from dictionary."""
        test = cls(
            test_id=data.get("test_id", ""),
            name=data.get("name", ""),
            category=TestCategory(data.get("category", "risk_controls")),
            description=data.get("description", ""),
            rts_reference=data.get("rts_reference", ""),
            priority=TestPriority(data.get("priority", "high")),
            preconditions=data.get("preconditions", []),
            test_steps=data.get("test_steps", []),
            expected_result=data.get("expected_result", ""),
            result=TestResult(data.get("result", "pending")),
            details=data.get("details", ""),
            duration_ms=data.get("duration_ms", 0.0),
            tester=data.get("tester", ""),
            notes=data.get("notes", ""),
        )

        if data.get("timestamp"):
            test.timestamp = datetime.fromisoformat(data["timestamp"])

        return test


@dataclass
class ConformanceTestSuite:
    """
    RTS 6 Article 5 Conformance Test Suite.

    A collection of conformance tests for an algorithm or system.

    Per RTS 6: "Investment firms shall test the trading algorithm and
    trading system prior to deployment or substantial update."

    Attributes:
        suite_id: Unique identifier
        name: Suite name
        description: Suite description
        algorithm_id: Algorithm being tested
        algorithm_version: Version of algorithm
        environment: Testing environment
        tests: List of tests in suite
        status: Suite status
        created_by: Person who created suite
        executed_by: Person who executed suite
    """

    suite_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Algorithm identification
    algorithm_id: str = ""
    algorithm_version: str = ""

    # Environment
    environment: TestEnvironment = TestEnvironment.UAT

    # Tests
    tests: List[ConformanceTest] = field(default_factory=list)

    # Status
    status: ConformanceSuiteStatus = ConformanceSuiteStatus.NOT_STARTED

    # Metadata
    created_by: str = ""
    created_timestamp_ns: int = field(default_factory=time.time_ns)
    executed_by: str = ""
    start_timestamp_ns: Optional[int] = None
    end_timestamp_ns: Optional[int] = None

    # =========================================================================
    # Test Management
    # =========================================================================

    def add_test(self, test: ConformanceTest) -> None:
        """Add a test to the suite."""
        self.tests.append(test)

    def get_test(self, test_id: str) -> Optional[ConformanceTest]:
        """Get test by ID."""
        for test in self.tests:
            if test.test_id == test_id:
                return test
        return None

    def get_tests_by_category(self, category: TestCategory) -> List[ConformanceTest]:
        """Get tests by category."""
        return [t for t in self.tests if t.category == category]

    def get_critical_tests(self) -> List[ConformanceTest]:
        """Get all critical tests."""
        return [t for t in self.tests if t.is_critical()]

    def get_failed_tests(self) -> List[ConformanceTest]:
        """Get all failed tests."""
        return [t for t in self.tests if t.failed()]

    def get_pending_tests(self) -> List[ConformanceTest]:
        """Get all pending tests."""
        return [t for t in self.tests if t.is_pending()]

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get suite execution metrics."""
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t.passed())
        failed = sum(1 for t in self.tests if t.failed())
        pending = sum(1 for t in self.tests if t.is_pending())
        skipped = sum(1 for t in self.tests if t.result == TestResult.SKIP)
        warnings = sum(1 for t in self.tests if t.result == TestResult.WARNING)

        critical_total = sum(1 for t in self.tests if t.is_critical())
        critical_passed = sum(1 for t in self.tests if t.is_critical() and t.passed())
        critical_failed = sum(1 for t in self.tests if t.is_critical() and t.failed())

        # Calculate pass rate
        executed = total - pending
        pass_rate = (passed / executed * 100) if executed > 0 else 0.0

        # Calculate duration
        total_duration_ms = sum(t.duration_ms for t in self.tests)

        return {
            "suite_id": self.suite_id,
            "status": self.status.value,
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pending": pending,
            "skipped": skipped,
            "warnings": warnings,
            "pass_rate_pct": pass_rate,
            "critical_total": critical_total,
            "critical_passed": critical_passed,
            "critical_failed": critical_failed,
            "total_duration_ms": total_duration_ms,
        }

    def get_category_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary by category."""
        summary: Dict[str, Dict[str, int]] = {}

        for category in TestCategory:
            tests = self.get_tests_by_category(category)
            if tests:
                summary[category.value] = {
                    "total": len(tests),
                    "passed": sum(1 for t in tests if t.passed()),
                    "failed": sum(1 for t in tests if t.failed()),
                    "pending": sum(1 for t in tests if t.is_pending()),
                }

        return summary

    # =========================================================================
    # Status Management
    # =========================================================================

    def start(self, executed_by: str) -> None:
        """Start suite execution."""
        self.status = ConformanceSuiteStatus.IN_PROGRESS
        self.executed_by = executed_by
        self.start_timestamp_ns = time.time_ns()
        logger.info(f"Conformance suite {self.suite_id} started by {executed_by}")

    def complete(self) -> None:
        """Complete suite execution and determine final status."""
        self.end_timestamp_ns = time.time_ns()

        # Check for critical failures
        critical_failed = any(
            t.is_critical() and t.failed()
            for t in self.tests
        )

        # Check for any failures
        any_failed = any(t.failed() for t in self.tests)

        # Check for warnings
        has_warnings = any(t.result == TestResult.WARNING for t in self.tests)

        if critical_failed:
            self.status = ConformanceSuiteStatus.FAILED
        elif any_failed:
            self.status = ConformanceSuiteStatus.FAILED
        elif has_warnings:
            self.status = ConformanceSuiteStatus.PASSED_WITH_WARNINGS
        else:
            self.status = ConformanceSuiteStatus.PASSED

        logger.info(
            f"Conformance suite {self.suite_id} completed with status: {self.status.value}"
        )

    def abort(self, reason: str = "") -> None:
        """Abort suite execution."""
        self.status = ConformanceSuiteStatus.ABORTED
        self.end_timestamp_ns = time.time_ns()
        logger.warning(f"Conformance suite {self.suite_id} aborted: {reason}")

    # =========================================================================
    # Certification
    # =========================================================================

    def can_certify(self) -> bool:
        """Check if suite results allow certification."""
        return self.status in {
            ConformanceSuiteStatus.PASSED,
            ConformanceSuiteStatus.PASSED_WITH_WARNINGS,
        }

    def generate_certificate_data(self) -> Optional[Dict[str, Any]]:
        """Generate data for conformance certificate."""
        if not self.can_certify():
            return None

        metrics = self.get_metrics()

        return {
            "suite_id": self.suite_id,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "environment": self.environment.value,
            "test_date": datetime.now().isoformat(),
            "status": self.status.value,
            "metrics": metrics,
            "critical_tests_passed": metrics["critical_passed"] == metrics["critical_total"],
            "pass_rate_pct": metrics["pass_rate_pct"],
            "executed_by": self.executed_by,
        }

    # =========================================================================
    # Reporting
    # =========================================================================

    def generate_summary_report(self) -> str:
        """Generate text summary report."""
        metrics = self.get_metrics()
        category_summary = self.get_category_summary()

        report = f"""
================================================================================
CONFORMANCE TEST SUITE REPORT
RTS 6 Article 5 Compliance Testing
================================================================================

SUITE INFORMATION
-----------------
Suite ID:           {self.suite_id}
Suite Name:         {self.name}
Algorithm ID:       {self.algorithm_id}
Algorithm Version:  {self.algorithm_version}
Environment:        {self.environment.value}
Status:             {self.status.value.upper()}

EXECUTION DETAILS
-----------------
Created By:         {self.created_by}
Executed By:        {self.executed_by}
Duration:           {metrics['total_duration_ms']:.0f} ms

TEST RESULTS SUMMARY
--------------------
Total Tests:        {metrics['total_tests']}
Passed:             {metrics['passed']}
Failed:             {metrics['failed']}
Pending:            {metrics['pending']}
Skipped:            {metrics['skipped']}
Warnings:           {metrics['warnings']}
Pass Rate:          {metrics['pass_rate_pct']:.1f}%

CRITICAL TESTS
--------------
Total Critical:     {metrics['critical_total']}
Passed:             {metrics['critical_passed']}
Failed:             {metrics['critical_failed']}
Critical Pass Rate: {(metrics['critical_passed']/metrics['critical_total']*100) if metrics['critical_total'] > 0 else 0:.1f}%

RESULTS BY CATEGORY
-------------------
"""

        for cat_name, cat_data in category_summary.items():
            status = "PASS" if cat_data["failed"] == 0 else "FAIL"
            report += f"  {cat_name:30s} {cat_data['passed']}/{cat_data['total']} [{status}]\n"

        # Add failed tests
        failed_tests = self.get_failed_tests()
        if failed_tests:
            report += "\nFAILED TESTS\n"
            report += "-" * 70 + "\n"
            for test in failed_tests:
                priority_marker = "[CRITICAL]" if test.is_critical() else "[" + test.priority.value.upper() + "]"
                report += f"  {priority_marker} {test.test_id}: {test.name}\n"
                report += f"      Reference: {test.rts_reference}\n"
                report += f"      Details: {test.details[:100]}...\n" if len(test.details) > 100 else f"      Details: {test.details}\n"
                report += "\n"

        # Certification status
        report += "\nCERTIFICATION STATUS\n"
        report += "-" * 70 + "\n"
        if self.can_certify():
            report += "  ELIGIBLE FOR CERTIFICATION\n"
            report += f"  All critical tests passed. Algorithm may proceed to deployment.\n"
        else:
            report += "  NOT ELIGIBLE FOR CERTIFICATION\n"
            report += f"  {metrics['critical_failed']} critical test(s) failed. Remediation required.\n"

        report += """
================================================================================
END OF REPORT
================================================================================
"""
        return report

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "environment": self.environment.value,
            "tests": [t.to_dict() for t in self.tests],
            "status": self.status.value,
            "created_by": self.created_by,
            "created_timestamp_ns": self.created_timestamp_ns,
            "executed_by": self.executed_by,
            "start_timestamp_ns": self.start_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConformanceTestSuite":
        """Create from dictionary."""
        suite = cls(
            suite_id=data.get("suite_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            algorithm_id=data.get("algorithm_id", ""),
            algorithm_version=data.get("algorithm_version", ""),
            environment=TestEnvironment(data.get("environment", "uat")),
            status=ConformanceSuiteStatus(data.get("status", "not_started")),
            created_by=data.get("created_by", ""),
            created_timestamp_ns=data.get("created_timestamp_ns", time.time_ns()),
            executed_by=data.get("executed_by", ""),
            start_timestamp_ns=data.get("start_timestamp_ns"),
            end_timestamp_ns=data.get("end_timestamp_ns"),
        )

        suite.tests = [
            ConformanceTest.from_dict(t)
            for t in data.get("tests", [])
        ]

        return suite

    @classmethod
    def from_json(cls, json_str: str) -> "ConformanceTestSuite":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Test Runner Protocol
# =============================================================================


@runtime_checkable
class TestExecutor(Protocol):
    """Protocol for test execution."""

    def execute(self, test: ConformanceTest) -> ConformanceTest:
        """Execute a single test and return result."""
        ...


@dataclass
class TestExecutorConfig:
    """Configuration for test executor."""

    timeout_seconds: float = 60.0
    retry_on_error: bool = True
    max_retries: int = 3
    collect_evidence: bool = True
    stop_on_critical_failure: bool = True
    parallel_execution: bool = False
    max_parallel_tests: int = 4


class ConformanceTestRunner:
    """
    Runner for executing conformance test suites.

    Per RTS 6 Article 5: Tests must be executed in controlled environments
    before deployment or substantial updates.
    """

    def __init__(
        self,
        config: Optional[TestExecutorConfig] = None,
        executors: Optional[Dict[TestCategory, TestExecutor]] = None,
    ):
        """
        Initialize test runner.

        Args:
            config: Execution configuration.
            executors: Category-specific test executors.
        """
        self.config = config or TestExecutorConfig()
        self.executors: Dict[TestCategory, TestExecutor] = executors or {}
        self._running = False
        self._aborted = False
        self._lock = threading.Lock()

    def register_executor(self, category: TestCategory, executor: TestExecutor) -> None:
        """Register an executor for a test category."""
        self.executors[category] = executor

    def run_suite(
        self,
        suite: ConformanceTestSuite,
        executed_by: str,
        categories: Optional[Set[TestCategory]] = None,
    ) -> ConformanceTestSuite:
        """
        Run all tests in a suite.

        Args:
            suite: Test suite to execute.
            executed_by: Person/system executing tests.
            categories: Optional filter for categories to run.

        Returns:
            Suite with updated test results.
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Runner is already executing a suite")
            self._running = True
            self._aborted = False

        try:
            suite.start(executed_by)

            tests_to_run = suite.tests
            if categories:
                tests_to_run = [t for t in tests_to_run if t.category in categories]

            for test in tests_to_run:
                if self._aborted:
                    logger.warning("Suite execution aborted")
                    break

                self._execute_test(test)

                # Check for critical failure
                if (
                    self.config.stop_on_critical_failure
                    and test.is_critical()
                    and test.failed()
                ):
                    logger.warning(
                        f"Critical test {test.test_id} failed. Stopping suite execution."
                    )
                    break

            suite.complete()
            return suite

        finally:
            with self._lock:
                self._running = False

    def run_single_test(
        self,
        test: ConformanceTest,
        tester: str,
    ) -> ConformanceTest:
        """Run a single test."""
        test.tester = tester
        return self._execute_test(test)

    def abort(self) -> None:
        """Abort running suite."""
        self._aborted = True
        logger.warning("Suite abort requested")

    def _execute_test(self, test: ConformanceTest) -> ConformanceTest:
        """Execute a single test."""
        test.timestamp = datetime.now()
        start_time = time.time()

        try:
            # Get executor for category
            executor = self.executors.get(test.category)

            if executor is None:
                # No executor - use mock execution
                test.result = TestResult.SKIP
                test.details = f"No executor registered for category: {test.category.value}"
                logger.warning(f"Test {test.test_id} skipped - no executor")
            else:
                # Execute with executor
                test = executor.execute(test)

        except Exception as e:
            test.result = TestResult.ERROR
            test.details = f"Execution error: {str(e)}"
            logger.error(f"Test {test.test_id} error: {e}")

        test.duration_ms = (time.time() - start_time) * 1000

        # Log result
        log_level = logging.INFO if test.passed() else logging.WARNING
        logger.log(
            log_level,
            f"Test {test.test_id} [{test.priority.value}]: {test.result.value} ({test.duration_ms:.0f}ms)"
        )

        return test


# =============================================================================
# Standard Test Templates
# =============================================================================


def get_standard_conformance_tests() -> List[ConformanceTest]:
    """
    Get standard RTS 6 conformance tests.

    Returns list of all standard tests required per RTS 6 Article 5.
    """
    tests = []

    # =========================================================================
    # KILL SWITCH TESTS (RTS 6 Article 12)
    # =========================================================================

    tests.extend([
        ConformanceTest(
            test_id="CT-KS-001",
            name="Kill Switch - Cancel All Orders",
            category=TestCategory.KILL_SWITCH,
            description="Verify kill switch cancels all orders immediately across all venues.",
            rts_reference="RTS 6 Article 12",
            priority=TestPriority.CRITICAL,
            preconditions=[
                "Trading system connected to test venue(s)",
                "Test orders submitted and pending",
            ],
            test_steps=[
                "Submit multiple test orders to venue(s)",
                "Trigger kill switch with scope ALL",
                "Verify all orders cancelled within 1 second",
                "Verify no new orders can be submitted",
            ],
            expected_result="All orders cancelled immediately, new orders blocked.",
        ),
        ConformanceTest(
            test_id="CT-KS-002",
            name="Kill Switch - Per-Venue Cancellation",
            category=TestCategory.KILL_SWITCH,
            description="Verify kill switch can cancel orders for specific venue only.",
            rts_reference="RTS 6 Article 12(1)",
            priority=TestPriority.CRITICAL,
            preconditions=[
                "Orders on multiple venues",
            ],
            test_steps=[
                "Submit orders to venues A and B",
                "Trigger kill switch for venue A only",
                "Verify venue A orders cancelled",
                "Verify venue B orders remain active",
            ],
            expected_result="Only venue A orders cancelled.",
        ),
        ConformanceTest(
            test_id="CT-KS-003",
            name="Kill Switch - Per-Algorithm Cancellation",
            category=TestCategory.KILL_SWITCH,
            description="Verify kill switch can cancel orders for specific algorithm.",
            rts_reference="RTS 6 Article 12(1)",
            priority=TestPriority.HIGH,
            preconditions=[
                "Multiple algorithms running",
            ],
            test_steps=[
                "Start algorithms A and B",
                "Both algorithms submit orders",
                "Trigger kill switch for algorithm A",
                "Verify only algorithm A orders cancelled",
            ],
            expected_result="Only algorithm A orders cancelled.",
        ),
        ConformanceTest(
            test_id="CT-KS-004",
            name="Kill Switch - Audit Trail",
            category=TestCategory.KILL_SWITCH,
            description="Verify kill switch events are fully logged.",
            rts_reference="RTS 6 Article 12",
            priority=TestPriority.CRITICAL,
            preconditions=[
                "Audit trail enabled",
            ],
            test_steps=[
                "Trigger kill switch",
                "Retrieve audit trail",
                "Verify event logged with timestamp, reason, operator",
                "Verify affected orders list included",
            ],
            expected_result="Complete audit trail of kill switch event.",
        ),
        ConformanceTest(
            test_id="CT-KS-005",
            name="Kill Switch - Emergency Contacts",
            category=TestCategory.KILL_SWITCH,
            description="Verify emergency contact information is accessible.",
            rts_reference="RTS 6 Article 12(2)",
            priority=TestPriority.HIGH,
            preconditions=[
                "Kill switch configured",
            ],
            test_steps=[
                "Query kill switch emergency contacts",
                "Verify primary contact exists",
                "Verify out-of-hours contact exists",
            ],
            expected_result="Emergency contacts properly configured.",
        ),
    ])

    # =========================================================================
    # PRE-TRADE CONTROLS (RTS 6 Article 15)
    # =========================================================================

    tests.extend([
        ConformanceTest(
            test_id="CT-PT-001",
            name="Pre-Trade - Price Collar",
            category=TestCategory.PRE_TRADE,
            description="Verify orders outside price collar are rejected.",
            rts_reference="RTS 6 Article 15(1)(a)",
            priority=TestPriority.CRITICAL,
            preconditions=[
                "Pre-trade controls enabled",
                "Price collar configured (e.g., 5%)",
            ],
            test_steps=[
                "Get current reference price",
                "Submit order with price 10% above reference",
                "Verify order rejected with PRICE_COLLAR reason",
                "Submit order with price within collar",
                "Verify order accepted",
            ],
            expected_result="Orders outside collar rejected, within collar accepted.",
        ),
        ConformanceTest(
            test_id="CT-PT-002",
            name="Pre-Trade - Maximum Order Value",
            category=TestCategory.PRE_TRADE,
            description="Verify orders exceeding max value are rejected.",
            rts_reference="RTS 6 Article 15(1)(b)",
            priority=TestPriority.CRITICAL,
            preconditions=[
                "Max order value configured (e.g., EUR 1M)",
            ],
            test_steps=[
                "Submit order with value exceeding max",
                "Verify order rejected with MAX_ORDER_VALUE reason",
                "Submit order below max value",
                "Verify order accepted",
            ],
            expected_result="Orders exceeding max value rejected.",
        ),
        ConformanceTest(
            test_id="CT-PT-003",
            name="Pre-Trade - Maximum Order Volume",
            category=TestCategory.PRE_TRADE,
            description="Verify orders exceeding max volume are rejected.",
            rts_reference="RTS 6 Article 15(1)(c)",
            priority=TestPriority.CRITICAL,
            preconditions=[
                "Max order volume configured",
            ],
            test_steps=[
                "Submit order with volume exceeding max",
                "Verify order rejected with MAX_ORDER_VOLUME reason",
                "Submit order below max volume",
                "Verify order accepted",
            ],
            expected_result="Orders exceeding max volume rejected.",
        ),
        ConformanceTest(
            test_id="CT-PT-004",
            name="Pre-Trade - Message Rate Limit",
            category=TestCategory.PRE_TRADE,
            description="Verify message rate limits are enforced.",
            rts_reference="RTS 6 Article 15(1)(d)",
            priority=TestPriority.HIGH,
            preconditions=[
                "Message rate limit configured",
            ],
            test_steps=[
                "Submit orders at rate below limit",
                "Verify all orders accepted",
                "Submit orders at rate exceeding limit",
                "Verify excess orders rejected/throttled",
            ],
            expected_result="Message rate limits enforced.",
        ),
        ConformanceTest(
            test_id="CT-PT-005",
            name="Pre-Trade - Trader Authorization",
            category=TestCategory.PRE_TRADE,
            description="Verify orders from unauthorized traders rejected.",
            rts_reference="RTS 6 Article 15(5)",
            priority=TestPriority.HIGH,
            preconditions=[
                "Trader authorization enabled",
            ],
            test_steps=[
                "Submit order with unauthorized trader ID",
                "Verify order rejected with UNAUTHORIZED_TRADER reason",
                "Submit order with authorized trader ID",
                "Verify order accepted",
            ],
            expected_result="Unauthorized trader orders rejected.",
        ),
        ConformanceTest(
            test_id="CT-PT-006",
            name="Pre-Trade - Fat Finger Protection",
            category=TestCategory.PRE_TRADE,
            description="Verify fat finger protection is active.",
            rts_reference="RTS 6 Article 15",
            priority=TestPriority.HIGH,
            preconditions=[
                "Fat finger protection configured",
            ],
            test_steps=[
                "Submit order with price 20% from reference",
                "Verify order rejected or flagged",
                "Submit order with normal price",
                "Verify order accepted",
            ],
            expected_result="Extreme price deviations caught.",
        ),
    ])

    # =========================================================================
    # CLOCK SYNCHRONISATION (RTS 25)
    # =========================================================================

    tests.extend([
        ConformanceTest(
            test_id="CT-CLK-001",
            name="Clock Sync - UTC Accuracy",
            category=TestCategory.CLOCK_SYNC,
            description="Verify clock synchronized to UTC within tolerance.",
            rts_reference="RTS 25",
            priority=TestPriority.CRITICAL,
            preconditions=[
                "NTP synchronization enabled",
            ],
            test_steps=[
                "Get current clock offset from NTP server",
                "Verify offset within allowed tolerance (100ms for algo trading)",
                "Record timestamp precision",
            ],
            expected_result="Clock offset within ±100ms.",
        ),
        ConformanceTest(
            test_id="CT-CLK-002",
            name="Clock Sync - Continuous Monitoring",
            category=TestCategory.CLOCK_SYNC,
            description="Verify clock drift is continuously monitored.",
            rts_reference="RTS 25",
            priority=TestPriority.HIGH,
            preconditions=[
                "Clock monitoring enabled",
            ],
            test_steps=[
                "Check clock sync status",
                "Verify last sync timestamp recent",
                "Verify drift alerts configured",
            ],
            expected_result="Clock continuously monitored.",
        ),
    ])

    # =========================================================================
    # RECORD KEEPING (MiFIR Article 25)
    # =========================================================================

    tests.extend([
        ConformanceTest(
            test_id="CT-RK-001",
            name="Audit Trail - Order Events",
            category=TestCategory.RECORD_KEEPING,
            description="Verify all order events are recorded.",
            rts_reference="MiFIR Article 25",
            priority=TestPriority.CRITICAL,
            preconditions=[
                "Audit trail enabled",
            ],
            test_steps=[
                "Submit a test order",
                "Modify the order",
                "Cancel the order",
                "Query audit trail for order ID",
                "Verify submit, modify, cancel events recorded",
            ],
            expected_result="All order lifecycle events recorded.",
        ),
        ConformanceTest(
            test_id="CT-RK-002",
            name="Audit Trail - Timestamp Precision",
            category=TestCategory.RECORD_KEEPING,
            description="Verify timestamps have required precision.",
            rts_reference="RTS 25",
            priority=TestPriority.HIGH,
            preconditions=[
                "Audit trail enabled",
            ],
            test_steps=[
                "Generate audit event",
                "Verify timestamp has nanosecond or microsecond precision",
                "Verify UTC timezone",
            ],
            expected_result="Timestamps precise and UTC.",
        ),
        ConformanceTest(
            test_id="CT-RK-003",
            name="Audit Trail - Chain Integrity",
            category=TestCategory.RECORD_KEEPING,
            description="Verify audit trail is tamper-proof.",
            rts_reference="MiFIR Article 25",
            priority=TestPriority.CRITICAL,
            preconditions=[
                "Audit trail with hash chain",
            ],
            test_steps=[
                "Write multiple audit records",
                "Verify hash chain integrity",
                "Attempt to modify record (should fail)",
            ],
            expected_result="Hash chain intact, tampering detected.",
        ),
        ConformanceTest(
            test_id="CT-RK-004",
            name="Retention - 5 Year Storage",
            category=TestCategory.RECORD_KEEPING,
            description="Verify records retained for 5 years.",
            rts_reference="MiFIR Article 25(1)",
            priority=TestPriority.HIGH,
            preconditions=[
                "Retention policy configured",
            ],
            test_steps=[
                "Query retention policy settings",
                "Verify default retention >= 5 years",
                "Verify archival mechanism exists",
            ],
            expected_result="5-year retention configured.",
        ),
    ])

    # =========================================================================
    # STRESS TESTING (RTS 6 Article 8)
    # =========================================================================

    tests.extend([
        ConformanceTest(
            test_id="CT-STR-001",
            name="Stress Test - High Volume",
            category=TestCategory.STRESS_TEST,
            description="Verify system handles high message volumes.",
            rts_reference="RTS 6 Article 8",
            priority=TestPriority.HIGH,
            preconditions=[
                "Stress test environment available",
            ],
            test_steps=[
                "Generate 2x normal message volume",
                "Monitor system performance",
                "Verify no message loss",
                "Verify latency within acceptable bounds",
            ],
            expected_result="System handles high volume gracefully.",
        ),
        ConformanceTest(
            test_id="CT-STR-002",
            name="Stress Test - Market Volatility",
            category=TestCategory.STRESS_TEST,
            description="Verify system handles volatile market conditions.",
            rts_reference="RTS 6 Article 8",
            priority=TestPriority.HIGH,
            preconditions=[
                "Market data simulator available",
            ],
            test_steps=[
                "Simulate high volatility market data",
                "Verify risk controls respond appropriately",
                "Verify no erroneous orders generated",
            ],
            expected_result="System stable under volatility.",
        ),
    ])

    # =========================================================================
    # BEST EXECUTION (MiFID II Article 27)
    # =========================================================================

    tests.extend([
        ConformanceTest(
            test_id="CT-BE-001",
            name="Best Execution - Policy Documented",
            category=TestCategory.BEST_EXECUTION,
            description="Verify best execution policy exists and is current.",
            rts_reference="MiFID II Article 27",
            priority=TestPriority.HIGH,
            preconditions=[
                "Best execution module enabled",
            ],
            test_steps=[
                "Query best execution policy",
                "Verify policy version and effective date",
                "Verify factor weights defined",
            ],
            expected_result="Best execution policy current.",
        ),
        ConformanceTest(
            test_id="CT-BE-002",
            name="Best Execution - TCA Metrics",
            category=TestCategory.BEST_EXECUTION,
            description="Verify TCA metrics are calculated.",
            rts_reference="MiFID II Article 27",
            priority=TestPriority.MEDIUM,
            preconditions=[
                "TCA enabled",
            ],
            test_steps=[
                "Execute a test trade",
                "Generate post-trade analysis",
                "Verify slippage and cost metrics calculated",
            ],
            expected_result="TCA metrics available.",
        ),
    ])

    # =========================================================================
    # TRANSACTION REPORTING (RTS 22)
    # =========================================================================

    tests.extend([
        ConformanceTest(
            test_id="CT-TR-001",
            name="Transaction Report - Field Completeness",
            category=TestCategory.TRANSACTION_REPORTING,
            description="Verify transaction reports contain all required fields.",
            rts_reference="RTS 22",
            priority=TestPriority.CRITICAL,
            preconditions=[
                "Transaction reporting enabled",
            ],
            test_steps=[
                "Generate transaction report for test trade",
                "Verify all 65 RTS 22 fields present",
                "Verify LEI format correct",
                "Verify ISIN format correct",
            ],
            expected_result="Report contains all required fields.",
        ),
        ConformanceTest(
            test_id="CT-TR-002",
            name="Transaction Report - T+1 Deadline",
            category=TestCategory.TRANSACTION_REPORTING,
            description="Verify reports submitted within T+1 deadline.",
            rts_reference="MiFIR Article 26",
            priority=TestPriority.CRITICAL,
            preconditions=[
                "ARM connectivity available",
            ],
            test_steps=[
                "Execute test trade",
                "Verify report queued immediately",
                "Verify deadline monitoring active",
            ],
            expected_result="T+1 deadline monitoring active.",
        ),
    ])

    # =========================================================================
    # BUSINESS CONTINUITY (RTS 6 Article 3)
    # =========================================================================

    tests.extend([
        ConformanceTest(
            test_id="CT-BCP-001",
            name="BCP - Plan Documented",
            category=TestCategory.BUSINESS_CONTINUITY,
            description="Verify business continuity plan exists.",
            rts_reference="RTS 6 Article 3",
            priority=TestPriority.HIGH,
            preconditions=[
                "BCP module enabled",
            ],
            test_steps=[
                "Query BCP configuration",
                "Verify scenarios defined",
                "Verify recovery procedures documented",
                "Verify emergency contacts available",
            ],
            expected_result="BCP fully documented.",
        ),
        ConformanceTest(
            test_id="CT-BCP-002",
            name="BCP - Recovery Time Objective",
            category=TestCategory.BUSINESS_CONTINUITY,
            description="Verify RTO is defined and achievable.",
            rts_reference="RTS 6 Article 3",
            priority=TestPriority.MEDIUM,
            preconditions=[
                "BCP configured",
            ],
            test_steps=[
                "Query RTO settings",
                "Verify RTO defined for critical systems",
                "Verify backup systems available",
            ],
            expected_result="RTO defined and achievable.",
        ),
    ])

    return tests


# =============================================================================
# Factory Functions
# =============================================================================


def create_conformance_suite(
    algorithm_id: str,
    algorithm_version: str,
    environment: TestEnvironment = TestEnvironment.UAT,
    created_by: str = "",
    include_standard_tests: bool = True,
    categories: Optional[Set[TestCategory]] = None,
) -> ConformanceTestSuite:
    """
    Create a conformance test suite for an algorithm.

    Args:
        algorithm_id: Algorithm identifier.
        algorithm_version: Algorithm version.
        environment: Test environment.
        created_by: Person creating the suite.
        include_standard_tests: Include standard RTS 6 tests.
        categories: Filter for specific categories.

    Returns:
        New ConformanceTestSuite instance.
    """
    suite = ConformanceTestSuite(
        name=f"Conformance Tests - {algorithm_id} v{algorithm_version}",
        description=f"RTS 6 Article 5 conformance testing for algorithm {algorithm_id}",
        algorithm_id=algorithm_id,
        algorithm_version=algorithm_version,
        environment=environment,
        created_by=created_by,
    )

    if include_standard_tests:
        tests = get_standard_conformance_tests()

        if categories:
            tests = [t for t in tests if t.category in categories]

        for test in tests:
            suite.add_test(test)

    return suite


def create_test_runner(
    config: Optional[TestExecutorConfig] = None,
) -> ConformanceTestRunner:
    """
    Create a conformance test runner.

    Args:
        config: Execution configuration.

    Returns:
        ConformanceTestRunner instance.
    """
    return ConformanceTestRunner(config=config)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "TestResult",
    "TestCategory",
    "TestPriority",
    "TestEnvironment",
    "ConformanceSuiteStatus",
    "CertificationStatus",
    # Data classes
    "TestEvidence",
    "ConformanceTest",
    "ConformanceTestSuite",
    # Runner
    "TestExecutorConfig",
    "ConformanceTestRunner",
    # Factory functions
    "create_conformance_suite",
    "create_test_runner",
    "get_standard_conformance_tests",
]

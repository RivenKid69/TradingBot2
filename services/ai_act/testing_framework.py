# -*- coding: utf-8 -*-
"""
AI Act Testing Framework (Article 9(6-7)).

EU AI Act Article 9 requires testing against prior defined metrics and
probabilistic thresholds, including real-world conditions testing.

This module provides:
1. AIActTestingFramework - Main testing orchestrator
2. TestMetric - Test metric definition and tracking
3. TestScenario - Test scenario management
4. TestExecution - Test execution and results
5. RealWorldTester - Real-world conditions testing (Article 60)

Article 9 Testing Requirements Implemented:
    - (6) Testing against prior defined metrics and probabilistic thresholds
    - (7) Testing in real-world conditions where appropriate
    - (8) Consideration of vulnerable groups

Article 60 Requirements (Sandbox Testing):
    - Real-world conditions testing
    - Controlled environment testing
    - Risk mitigation during testing

References:
    - Article 9: https://artificialintelligenceact.eu/article/9/
    - Article 60: https://artificialintelligenceact.eu/article/60/
    - ISO 29119: Software testing standards
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class TestCategory(Enum):
    """Categories of tests per Article 9."""
    FUNCTIONAL = "functional"  # Intended purpose verification
    PERFORMANCE = "performance"  # Accuracy metrics verification
    ROBUSTNESS = "robustness"  # Resilience to perturbations
    SAFETY = "safety"  # Risk scenario testing
    SECURITY = "security"  # Cybersecurity testing
    BIAS = "bias"  # Bias and discrimination testing
    INTEGRATION = "integration"  # System integration testing
    REAL_WORLD = "real_world"  # Article 60 real-world testing


class TestPriority(Enum):
    """Test priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    ERROR = "error"


class MetricType(Enum):
    """Types of test metrics."""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    COVERAGE = "coverage"
    RELIABILITY = "reliability"
    ROBUSTNESS = "robustness"
    FAIRNESS = "fairness"
    CUSTOM = "custom"


class ComparisonOperator(Enum):
    """Operators for threshold comparison."""
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "neq"
    IN_RANGE = "in_range"
    NOT_IN_RANGE = "not_in_range"


class VulnerableGroup(Enum):
    """Vulnerable groups per Article 9(9)."""
    ELDERLY = "elderly"
    DISABLED = "disabled"
    CHILDREN = "children"
    LOW_INCOME = "low_income"
    NON_NATIVE_SPEAKERS = "non_native_speakers"
    MINORITIES = "minorities"
    OTHER = "other"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TestMetric:
    """
    Test metric definition per Article 9(6).

    Represents a measurable metric with defined thresholds.
    """
    metric_id: str
    name: str
    metric_type: MetricType
    description: str

    # Threshold definition
    threshold_value: float = 0.0
    threshold_operator: ComparisonOperator = ComparisonOperator.GREATER_THAN_OR_EQUAL
    threshold_upper: Optional[float] = None  # For range checks

    # Statistical requirements
    confidence_level: float = 0.95
    min_samples: int = 30

    # Metadata
    unit: str = ""
    is_primary: bool = True
    article_reference: str = "Article 9(6)"

    # Results (populated during testing)
    actual_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: int = 0
    passed: Optional[bool] = None

    def __post_init__(self):
        if not self.metric_id:
            self.metric_id = f"METRIC-{uuid.uuid4().hex[:8].upper()}"

    def evaluate(self, value: float) -> bool:
        """Evaluate if value meets threshold."""
        if self.threshold_operator == ComparisonOperator.GREATER_THAN:
            return value > self.threshold_value
        elif self.threshold_operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
            return value >= self.threshold_value
        elif self.threshold_operator == ComparisonOperator.LESS_THAN:
            return value < self.threshold_value
        elif self.threshold_operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
            return value <= self.threshold_value
        elif self.threshold_operator == ComparisonOperator.EQUAL:
            return abs(value - self.threshold_value) < 1e-9
        elif self.threshold_operator == ComparisonOperator.NOT_EQUAL:
            return abs(value - self.threshold_value) >= 1e-9
        elif self.threshold_operator == ComparisonOperator.IN_RANGE:
            upper = self.threshold_upper or self.threshold_value
            return self.threshold_value <= value <= upper
        elif self.threshold_operator == ComparisonOperator.NOT_IN_RANGE:
            upper = self.threshold_upper or self.threshold_value
            return not (self.threshold_value <= value <= upper)
        return False


@dataclass
class TestScenario:
    """
    Test scenario definition.

    Represents a specific test case or scenario to execute.
    """
    scenario_id: str
    name: str
    description: str
    category: TestCategory

    # Test definition
    test_inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)

    # Metrics to evaluate
    metrics: List[str] = field(default_factory=list)  # metric_ids

    # Priority and classification
    priority: TestPriority = TestPriority.MEDIUM
    is_automated: bool = True
    requires_human_oversight: bool = False

    # Vulnerable groups (Article 9(9))
    vulnerable_groups_tested: List[VulnerableGroup] = field(default_factory=list)

    # Execution constraints
    max_execution_time_seconds: float = 300.0
    retry_count: int = 0

    # Status
    status: TestStatus = TestStatus.PENDING
    last_execution: Optional[str] = None

    # Metadata
    created_at: str = ""
    created_by: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.scenario_id:
            self.scenario_id = f"TEST-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TestExecution:
    """
    Test execution record.

    Documents the execution of a test scenario.
    """
    execution_id: str
    scenario_id: str
    scenario_name: str

    # Execution details
    start_time: str = ""
    end_time: Optional[str] = None
    duration_seconds: float = 0.0

    # Environment
    environment: str = "test"  # test, staging, production
    environment_details: Dict[str, Any] = field(default_factory=dict)

    # Results
    status: TestStatus = TestStatus.PENDING
    actual_outputs: Dict[str, Any] = field(default_factory=dict)
    metric_results: List[Dict[str, Any]] = field(default_factory=list)

    # Errors
    error_message: str = ""
    error_trace: str = ""

    # Evidence
    logs: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)

    # Human oversight (if required)
    human_reviewer: str = ""
    human_approval: Optional[bool] = None
    human_comments: str = ""

    # Metadata
    executed_by: str = ""
    machine_id: str = ""

    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = f"EXEC-{uuid.uuid4().hex[:8].upper()}"
        if not self.start_time:
            self.start_time = datetime.now(timezone.utc).isoformat()

    @property
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.status in (
            TestStatus.PASSED, TestStatus.FAILED,
            TestStatus.SKIPPED, TestStatus.ERROR
        )


@dataclass
class TestSuite:
    """
    Collection of test scenarios.

    Groups related tests for organized execution.
    """
    suite_id: str
    name: str
    description: str

    # Tests
    scenario_ids: List[str] = field(default_factory=list)
    category: TestCategory = TestCategory.FUNCTIONAL

    # Execution settings
    parallel_execution: bool = False
    stop_on_failure: bool = False
    max_failures: int = -1  # -1 = unlimited

    # Schedule
    schedule_cron: str = ""  # Cron expression
    last_run: Optional[str] = None
    next_run: Optional[str] = None

    # Results summary
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0

    # Metadata
    created_at: str = ""
    owner: str = ""

    def __post_init__(self):
        if not self.suite_id:
            self.suite_id = f"SUITE-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        total = self.passed_tests + self.failed_tests + self.skipped_tests
        if total == 0:
            return 0.0
        return self.passed_tests / total * 100


@dataclass
class RealWorldTestPlan:
    """
    Real-world testing plan per Article 60.

    Documents testing in real-world conditions.
    """
    plan_id: str
    name: str
    description: str

    # Scope
    test_objectives: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    # Participants
    target_population: str = ""
    sample_size: int = 0
    sampling_method: str = ""

    # Safeguards
    risk_mitigations: List[str] = field(default_factory=list)
    stopping_criteria: List[str] = field(default_factory=list)
    human_oversight_measures: List[str] = field(default_factory=list)

    # Data handling
    data_collection_methods: List[str] = field(default_factory=list)
    data_protection_measures: List[str] = field(default_factory=list)
    consent_obtained: bool = False

    # Duration
    planned_start: str = ""
    planned_end: str = ""
    actual_start: Optional[str] = None
    actual_end: Optional[str] = None

    # Status
    status: str = "planned"  # planned, approved, active, completed, terminated

    # Approval
    approved_by: str = ""
    approval_date: Optional[str] = None
    regulatory_notification: bool = False

    # Results
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = f"RWTP-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TestingConfig:
    """
    Configuration for AI Act Testing Framework.
    """
    # Test execution
    default_timeout_seconds: float = 300.0
    parallel_workers: int = 4
    retry_failed_tests: bool = True
    max_retries: int = 2

    # Thresholds
    minimum_pass_rate: float = 95.0
    minimum_coverage: float = 80.0
    confidence_level: float = 0.95

    # Statistical testing
    min_samples_per_metric: int = 30
    use_bootstrap: bool = True
    bootstrap_iterations: int = 1000

    # Real-world testing
    real_world_enabled: bool = True
    require_human_approval: bool = True

    # Paths
    log_path: str = "logs/ai_act/testing"
    artifacts_path: str = "artifacts/testing"

    # Notifications
    notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Test Executor Interface
# =============================================================================

class TestExecutor(ABC):
    """Abstract base class for test executors."""

    @abstractmethod
    def execute(
        self,
        scenario: TestScenario,
        metrics: List[TestMetric],
    ) -> TestExecution:
        """Execute a test scenario."""
        pass


class DefaultTestExecutor(TestExecutor):
    """Default test executor implementation."""

    def execute(
        self,
        scenario: TestScenario,
        metrics: List[TestMetric],
    ) -> TestExecution:
        """Execute a test scenario with default implementation."""
        execution = TestExecution(
            execution_id="",
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
        )

        try:
            execution.status = TestStatus.RUNNING

            # Simulate test execution
            # In real implementation, this would run actual tests
            for metric in metrics:
                # Simulate metric measurement
                result = {
                    "metric_id": metric.metric_id,
                    "metric_name": metric.name,
                    "threshold": metric.threshold_value,
                    "actual": metric.threshold_value * 1.1,  # Simulated
                    "passed": True,
                }
                execution.metric_results.append(result)

            execution.status = TestStatus.PASSED
            execution.end_time = datetime.now(timezone.utc).isoformat()

            start = datetime.fromisoformat(execution.start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(execution.end_time.replace("Z", "+00:00"))
            execution.duration_seconds = (end - start).total_seconds()

        except Exception as e:
            execution.status = TestStatus.ERROR
            execution.error_message = str(e)
            execution.end_time = datetime.now(timezone.utc).isoformat()

        return execution


# =============================================================================
# Main Testing Framework
# =============================================================================

class AIActTestingFramework:
    """
    AI Act compliant Testing Framework per Article 9(6-7).

    Provides comprehensive testing functionality including:
    - Metric definition and threshold management
    - Test scenario creation and execution
    - Test suite management
    - Real-world conditions testing (Article 60)
    - Statistical analysis of results
    - Vulnerable groups testing (Article 9(9))

    Usage:
        config = TestingConfig(
            default_timeout_seconds=300,
            minimum_pass_rate=95.0,
        )
        framework = AIActTestingFramework(config)

        # Define a metric
        metric = framework.define_metric(
            name="Accuracy",
            metric_type=MetricType.ACCURACY,
            threshold_value=0.95,
            threshold_operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
        )

        # Create a test scenario
        scenario = framework.create_scenario(
            name="Model Accuracy Test",
            category=TestCategory.PERFORMANCE,
            metrics=[metric.metric_id],
        )

        # Execute test
        result = framework.execute_test(scenario.scenario_id)
    """

    def __init__(
        self,
        config: Optional[TestingConfig] = None,
        executor: Optional[TestExecutor] = None,
    ):
        """Initialize testing framework."""
        self.config = config or TestingConfig()
        self._executor = executor or DefaultTestExecutor()

        # Data stores
        self._metrics: Dict[str, TestMetric] = {}
        self._scenarios: Dict[str, TestScenario] = {}
        self._suites: Dict[str, TestSuite] = {}
        self._executions: Dict[str, TestExecution] = {}
        self._real_world_plans: Dict[str, RealWorldTestPlan] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        self._artifacts_path = Path(self.config.artifacts_path)
        self._artifacts_path.mkdir(parents=True, exist_ok=True)

        logger.info("AIActTestingFramework initialized")

    # =========================================================================
    # Metric Management
    # =========================================================================

    def define_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        threshold_value: float = 0.0,
        threshold_operator: ComparisonOperator = ComparisonOperator.GREATER_THAN_OR_EQUAL,
        threshold_upper: Optional[float] = None,
        unit: str = "",
        is_primary: bool = True,
        min_samples: int = 30,
    ) -> TestMetric:
        """
        Define a test metric per Article 9(6).

        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            threshold_value: Threshold value
            threshold_operator: Comparison operator
            threshold_upper: Upper bound for range checks
            unit: Measurement unit
            is_primary: Whether this is a primary metric
            min_samples: Minimum samples required

        Returns:
            Created TestMetric
        """
        metric = TestMetric(
            metric_id="",
            name=name,
            metric_type=metric_type,
            description=description,
            threshold_value=threshold_value,
            threshold_operator=threshold_operator,
            threshold_upper=threshold_upper,
            unit=unit,
            is_primary=is_primary,
            min_samples=min_samples,
            confidence_level=self.config.confidence_level,
        )

        with self._lock:
            self._metrics[metric.metric_id] = metric

        self._log_event("metric_defined", asdict(metric))
        logger.info(f"Metric defined: {metric.metric_id} - {name}")
        return metric

    def get_metric(self, metric_id: str) -> Optional[TestMetric]:
        """Get metric by ID."""
        with self._lock:
            return self._metrics.get(metric_id)

    def get_all_metrics(self) -> List[TestMetric]:
        """Get all defined metrics."""
        with self._lock:
            return list(self._metrics.values())

    def update_metric_threshold(
        self,
        metric_id: str,
        threshold_value: float,
        threshold_operator: Optional[ComparisonOperator] = None,
    ) -> TestMetric:
        """Update metric threshold."""
        with self._lock:
            if metric_id not in self._metrics:
                raise ValueError(f"Metric {metric_id} not found")

            metric = self._metrics[metric_id]
            metric.threshold_value = threshold_value
            if threshold_operator:
                metric.threshold_operator = threshold_operator

        return metric

    # =========================================================================
    # Scenario Management
    # =========================================================================

    def create_scenario(
        self,
        name: str,
        category: TestCategory,
        description: str = "",
        test_inputs: Optional[Dict[str, Any]] = None,
        expected_outputs: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        priority: TestPriority = TestPriority.MEDIUM,
        vulnerable_groups: Optional[List[VulnerableGroup]] = None,
        tags: Optional[List[str]] = None,
    ) -> TestScenario:
        """
        Create a test scenario.

        Args:
            name: Scenario name
            category: Test category
            description: Scenario description
            test_inputs: Test input data
            expected_outputs: Expected outputs
            metrics: List of metric IDs to evaluate
            priority: Test priority
            vulnerable_groups: Vulnerable groups being tested
            tags: Tags for categorization

        Returns:
            Created TestScenario
        """
        scenario = TestScenario(
            scenario_id="",
            name=name,
            description=description,
            category=category,
            test_inputs=test_inputs or {},
            expected_outputs=expected_outputs or {},
            metrics=metrics or [],
            priority=priority,
            vulnerable_groups_tested=vulnerable_groups or [],
            tags=tags or [],
        )

        with self._lock:
            self._scenarios[scenario.scenario_id] = scenario

        self._log_event("scenario_created", {
            "scenario_id": scenario.scenario_id,
            "name": name,
            "category": category.value,
        })
        logger.info(f"Test scenario created: {scenario.scenario_id}")
        return scenario

    def get_scenario(self, scenario_id: str) -> Optional[TestScenario]:
        """Get scenario by ID."""
        with self._lock:
            return self._scenarios.get(scenario_id)

    def get_scenarios_by_category(self, category: TestCategory) -> List[TestScenario]:
        """Get all scenarios in a category."""
        with self._lock:
            return [s for s in self._scenarios.values() if s.category == category]

    def get_scenarios_for_vulnerable_group(
        self,
        group: VulnerableGroup,
    ) -> List[TestScenario]:
        """Get scenarios that test a specific vulnerable group."""
        with self._lock:
            return [s for s in self._scenarios.values()
                    if group in s.vulnerable_groups_tested]

    # =========================================================================
    # Test Suite Management
    # =========================================================================

    def create_suite(
        self,
        name: str,
        description: str = "",
        category: TestCategory = TestCategory.FUNCTIONAL,
        scenario_ids: Optional[List[str]] = None,
        parallel_execution: bool = False,
        stop_on_failure: bool = False,
    ) -> TestSuite:
        """
        Create a test suite.

        Args:
            name: Suite name
            description: Suite description
            category: Test category
            scenario_ids: List of scenario IDs to include
            parallel_execution: Whether to run tests in parallel
            stop_on_failure: Whether to stop on first failure

        Returns:
            Created TestSuite
        """
        suite = TestSuite(
            suite_id="",
            name=name,
            description=description,
            category=category,
            scenario_ids=scenario_ids or [],
            parallel_execution=parallel_execution,
            stop_on_failure=stop_on_failure,
        )
        suite.total_tests = len(suite.scenario_ids)

        with self._lock:
            self._suites[suite.suite_id] = suite

        self._log_event("suite_created", {
            "suite_id": suite.suite_id,
            "name": name,
            "test_count": len(suite.scenario_ids),
        })
        return suite

    def add_scenario_to_suite(
        self,
        suite_id: str,
        scenario_id: str,
    ) -> TestSuite:
        """Add a scenario to a test suite."""
        with self._lock:
            if suite_id not in self._suites:
                raise ValueError(f"Suite {suite_id} not found")
            if scenario_id not in self._scenarios:
                raise ValueError(f"Scenario {scenario_id} not found")

            suite = self._suites[suite_id]
            if scenario_id not in suite.scenario_ids:
                suite.scenario_ids.append(scenario_id)
                suite.total_tests = len(suite.scenario_ids)

        return suite

    def get_suite(self, suite_id: str) -> Optional[TestSuite]:
        """Get suite by ID."""
        with self._lock:
            return self._suites.get(suite_id)

    # =========================================================================
    # Test Execution
    # =========================================================================

    def execute_test(
        self,
        scenario_id: str,
        environment: str = "test",
        executed_by: str = "",
    ) -> TestExecution:
        """
        Execute a single test scenario.

        Args:
            scenario_id: ID of scenario to execute
            environment: Execution environment
            executed_by: Who initiated the test

        Returns:
            TestExecution with results
        """
        with self._lock:
            if scenario_id not in self._scenarios:
                raise ValueError(f"Scenario {scenario_id} not found")

            scenario = self._scenarios[scenario_id]
            metrics = [self._metrics[m] for m in scenario.metrics
                      if m in self._metrics]

        # Execute test
        execution = self._executor.execute(scenario, metrics)
        execution.environment = environment
        execution.executed_by = executed_by

        # Store execution
        with self._lock:
            self._executions[execution.execution_id] = execution
            scenario.status = execution.status
            scenario.last_execution = execution.execution_id

        self._log_event("test_executed", {
            "execution_id": execution.execution_id,
            "scenario_id": scenario_id,
            "status": execution.status.value,
        })

        return execution

    def execute_suite(
        self,
        suite_id: str,
        environment: str = "test",
        executed_by: str = "",
    ) -> List[TestExecution]:
        """
        Execute all tests in a suite.

        Args:
            suite_id: ID of suite to execute
            environment: Execution environment
            executed_by: Who initiated the tests

        Returns:
            List of TestExecution results
        """
        with self._lock:
            if suite_id not in self._suites:
                raise ValueError(f"Suite {suite_id} not found")

            suite = self._suites[suite_id]
            scenario_ids = list(suite.scenario_ids)

        results = []
        passed = 0
        failed = 0
        skipped = 0

        for scenario_id in scenario_ids:
            try:
                execution = self.execute_test(
                    scenario_id=scenario_id,
                    environment=environment,
                    executed_by=executed_by,
                )
                results.append(execution)

                if execution.status == TestStatus.PASSED:
                    passed += 1
                elif execution.status == TestStatus.FAILED:
                    failed += 1
                    if suite.stop_on_failure:
                        break
                else:
                    skipped += 1

            except Exception as e:
                logger.error(f"Failed to execute scenario {scenario_id}: {e}")
                skipped += 1

        # Update suite statistics
        with self._lock:
            suite.passed_tests = passed
            suite.failed_tests = failed
            suite.skipped_tests = skipped
            suite.last_run = datetime.now(timezone.utc).isoformat()

        self._log_event("suite_executed", {
            "suite_id": suite_id,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        })

        return results

    def get_execution(self, execution_id: str) -> Optional[TestExecution]:
        """Get execution by ID."""
        with self._lock:
            return self._executions.get(execution_id)

    def get_scenario_executions(self, scenario_id: str) -> List[TestExecution]:
        """Get all executions for a scenario."""
        with self._lock:
            return [e for e in self._executions.values()
                    if e.scenario_id == scenario_id]

    # =========================================================================
    # Statistical Analysis
    # =========================================================================

    def analyze_metric_results(
        self,
        metric_id: str,
        values: List[float],
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis on metric values.

        Args:
            metric_id: Metric to analyze
            values: Observed values

        Returns:
            Analysis results with confidence intervals
        """
        with self._lock:
            if metric_id not in self._metrics:
                raise ValueError(f"Metric {metric_id} not found")

            metric = self._metrics[metric_id]

        if len(values) < metric.min_samples:
            return {
                "metric_id": metric_id,
                "error": f"Insufficient samples ({len(values)} < {metric.min_samples})",
                "sample_size": len(values),
            }

        # Basic statistics
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        median = statistics.median(values)

        # Confidence interval (using t-distribution approximation)
        # For large samples, use z-score
        if len(values) >= 30:
            z = 1.96 if metric.confidence_level == 0.95 else 2.576  # 95% or 99%
            margin = z * (std_dev / (len(values) ** 0.5))
        else:
            # Use larger margin for small samples
            margin = 2.0 * (std_dev / (len(values) ** 0.5))

        ci_lower = mean - margin
        ci_upper = mean + margin

        # Bootstrap confidence interval (optional)
        bootstrap_ci = None
        if self.config.use_bootstrap and len(values) >= 10:
            bootstrap_ci = self._bootstrap_confidence_interval(
                values,
                self.config.bootstrap_iterations,
                metric.confidence_level,
            )

        # Evaluate against threshold
        passed = metric.evaluate(mean)

        # Update metric
        metric.actual_value = mean
        metric.confidence_interval = (ci_lower, ci_upper)
        metric.sample_size = len(values)
        metric.passed = passed

        return {
            "metric_id": metric_id,
            "metric_name": metric.name,
            "sample_size": len(values),
            "mean": mean,
            "std_dev": std_dev,
            "median": median,
            "min": min(values),
            "max": max(values),
            "confidence_level": metric.confidence_level,
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
            },
            "bootstrap_ci": bootstrap_ci,
            "threshold": {
                "value": metric.threshold_value,
                "operator": metric.threshold_operator.value,
            },
            "passed": passed,
        }

    def _bootstrap_confidence_interval(
        self,
        values: List[float],
        iterations: int,
        confidence_level: float,
    ) -> Dict[str, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []

        for _ in range(iterations):
            sample = random.choices(values, k=len(values))
            bootstrap_means.append(statistics.mean(sample))

        bootstrap_means.sort()

        alpha = 1 - confidence_level
        lower_idx = int(alpha / 2 * iterations)
        upper_idx = int((1 - alpha / 2) * iterations)

        return {
            "lower": bootstrap_means[lower_idx],
            "upper": bootstrap_means[min(upper_idx, len(bootstrap_means) - 1)],
        }

    # =========================================================================
    # Real-World Testing (Article 60)
    # =========================================================================

    def create_real_world_test_plan(
        self,
        name: str,
        description: str,
        test_objectives: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        target_population: str = "",
        sample_size: int = 0,
        risk_mitigations: Optional[List[str]] = None,
        stopping_criteria: Optional[List[str]] = None,
        planned_start: str = "",
        planned_end: str = "",
    ) -> RealWorldTestPlan:
        """
        Create a real-world testing plan per Article 60.

        Args:
            name: Plan name
            description: Plan description
            test_objectives: Test objectives
            success_criteria: Success criteria
            target_population: Target population description
            sample_size: Planned sample size
            risk_mitigations: Risk mitigation measures
            stopping_criteria: Criteria for stopping the test
            planned_start: Planned start date
            planned_end: Planned end date

        Returns:
            Created RealWorldTestPlan
        """
        plan = RealWorldTestPlan(
            plan_id="",
            name=name,
            description=description,
            test_objectives=test_objectives or [],
            success_criteria=success_criteria or [],
            target_population=target_population,
            sample_size=sample_size,
            risk_mitigations=risk_mitigations or [],
            stopping_criteria=stopping_criteria or [],
            planned_start=planned_start,
            planned_end=planned_end,
        )

        with self._lock:
            self._real_world_plans[plan.plan_id] = plan

        self._log_event("real_world_plan_created", {
            "plan_id": plan.plan_id,
            "name": name,
        })
        logger.info(f"Real-world test plan created: {plan.plan_id}")
        return plan

    def approve_real_world_test(
        self,
        plan_id: str,
        approved_by: str,
        human_oversight_measures: Optional[List[str]] = None,
    ) -> RealWorldTestPlan:
        """Approve a real-world test plan."""
        with self._lock:
            if plan_id not in self._real_world_plans:
                raise ValueError(f"Plan {plan_id} not found")

            plan = self._real_world_plans[plan_id]
            plan.status = "approved"
            plan.approved_by = approved_by
            plan.approval_date = datetime.now(timezone.utc).isoformat()
            if human_oversight_measures:
                plan.human_oversight_measures = human_oversight_measures

        self._log_event("real_world_plan_approved", {
            "plan_id": plan_id,
            "approved_by": approved_by,
        })
        return plan

    def start_real_world_test(self, plan_id: str) -> RealWorldTestPlan:
        """Start a real-world test."""
        with self._lock:
            if plan_id not in self._real_world_plans:
                raise ValueError(f"Plan {plan_id} not found")

            plan = self._real_world_plans[plan_id]
            if plan.status != "approved":
                raise ValueError(f"Plan {plan_id} must be approved before starting")

            plan.status = "active"
            plan.actual_start = datetime.now(timezone.utc).isoformat()

        self._log_event("real_world_test_started", {"plan_id": plan_id})
        return plan

    def complete_real_world_test(
        self,
        plan_id: str,
        findings: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
    ) -> RealWorldTestPlan:
        """Complete a real-world test."""
        with self._lock:
            if plan_id not in self._real_world_plans:
                raise ValueError(f"Plan {plan_id} not found")

            plan = self._real_world_plans[plan_id]
            plan.status = "completed"
            plan.actual_end = datetime.now(timezone.utc).isoformat()
            plan.findings = findings or []
            plan.recommendations = recommendations or []

        self._log_event("real_world_test_completed", {
            "plan_id": plan_id,
            "findings_count": len(plan.findings),
        })
        return plan

    def get_real_world_plan(self, plan_id: str) -> Optional[RealWorldTestPlan]:
        """Get real-world test plan by ID."""
        with self._lock:
            return self._real_world_plans.get(plan_id)

    # =========================================================================
    # Compliance Reporting
    # =========================================================================

    def get_compliance_summary(self) -> Dict[str, Any]:
        """
        Get testing compliance summary.

        Returns:
            Dictionary with compliance metrics
        """
        with self._lock:
            total_metrics = len(self._metrics)
            total_scenarios = len(self._scenarios)
            total_executions = len(self._executions)

            # Count passed/failed executions
            passed_execs = len([e for e in self._executions.values()
                               if e.status == TestStatus.PASSED])
            failed_execs = len([e for e in self._executions.values()
                               if e.status == TestStatus.FAILED])

            # Calculate pass rate
            completed_execs = passed_execs + failed_execs
            pass_rate = (passed_execs / completed_execs * 100
                        if completed_execs > 0 else 0.0)

            # Category coverage
            tested_categories = set()
            for s in self._scenarios.values():
                if s.status == TestStatus.PASSED:
                    tested_categories.add(s.category)

            category_coverage = len(tested_categories) / len(TestCategory) * 100

            # Vulnerable groups coverage
            groups_tested = set()
            for s in self._scenarios.values():
                groups_tested.update(s.vulnerable_groups_tested)

            # Suite statistics
            suite_stats = []
            for suite in self._suites.values():
                suite_stats.append({
                    "suite_id": suite.suite_id,
                    "name": suite.name,
                    "pass_rate": suite.pass_rate,
                    "total": suite.total_tests,
                    "passed": suite.passed_tests,
                    "failed": suite.failed_tests,
                })

        return {
            "metrics": {
                "total_defined": total_metrics,
                "with_thresholds": total_metrics,
            },
            "scenarios": {
                "total": total_scenarios,
                "by_category": self._count_by_category(),
            },
            "executions": {
                "total": total_executions,
                "passed": passed_execs,
                "failed": failed_execs,
                "pass_rate": pass_rate,
            },
            "coverage": {
                "category_coverage_percent": category_coverage,
                "tested_categories": [c.value for c in tested_categories],
                "vulnerable_groups_tested": [g.value for g in groups_tested],
            },
            "suites": suite_stats,
            "thresholds": {
                "minimum_pass_rate": self.config.minimum_pass_rate,
                "minimum_coverage": self.config.minimum_coverage,
            },
            "compliance_met": (
                pass_rate >= self.config.minimum_pass_rate and
                category_coverage >= self.config.minimum_coverage
            ),
        }

    def _count_by_category(self) -> Dict[str, int]:
        """Count scenarios by category."""
        counts: Dict[str, int] = {}
        for scenario in self._scenarios.values():
            cat = scenario.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def generate_test_report(
        self,
        include_details: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive test report.

        Args:
            include_details: Whether to include detailed execution logs

        Returns:
            Complete test report
        """
        summary = self.get_compliance_summary()

        report = {
            "report_date": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "article_references": {
                "Article 9(6)": "Testing against prior defined metrics",
                "Article 9(7)": "Real-world conditions testing",
                "Article 9(9)": "Vulnerable groups consideration",
            },
        }

        if include_details:
            with self._lock:
                report["metrics"] = [asdict(m) for m in self._metrics.values()]
                report["scenarios"] = [asdict(s) for s in self._scenarios.values()]
                report["recent_executions"] = [
                    asdict(e) for e in list(self._executions.values())[-20:]
                ]
                report["real_world_plans"] = [
                    asdict(p) for p in self._real_world_plans.values()
                ]

        return report

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a testing event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"testing_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log testing event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_testing_framework(
    config: Optional[Union[Dict[str, Any], TestingConfig]] = None,
    executor: Optional[TestExecutor] = None,
) -> AIActTestingFramework:
    """
    Create an AIActTestingFramework instance.

    Args:
        config: Optional configuration
        executor: Optional custom test executor

    Returns:
        Configured AIActTestingFramework instance
    """
    if config is None:
        testing_config = TestingConfig()
    elif isinstance(config, TestingConfig):
        testing_config = config
    else:
        testing_config = TestingConfig(
            default_timeout_seconds=config.get("default_timeout_seconds", 300.0),
            parallel_workers=config.get("parallel_workers", 4),
            retry_failed_tests=config.get("retry_failed_tests", True),
            max_retries=config.get("max_retries", 2),
            minimum_pass_rate=config.get("minimum_pass_rate", 95.0),
            minimum_coverage=config.get("minimum_coverage", 80.0),
            confidence_level=config.get("confidence_level", 0.95),
            min_samples_per_metric=config.get("min_samples_per_metric", 30),
            use_bootstrap=config.get("use_bootstrap", True),
            bootstrap_iterations=config.get("bootstrap_iterations", 1000),
            real_world_enabled=config.get("real_world_enabled", True),
            require_human_approval=config.get("require_human_approval", True),
            log_path=config.get("log_path", "logs/ai_act/testing"),
            artifacts_path=config.get("artifacts_path", "artifacts/testing"),
        )

    return AIActTestingFramework(testing_config, executor)


def get_default_test_metrics() -> List[Dict[str, Any]]:
    """
    Get default test metrics for trading AI systems.

    Returns:
        List of metric definitions
    """
    return [
        {
            "name": "Model Accuracy",
            "metric_type": MetricType.ACCURACY,
            "description": "Overall model prediction accuracy",
            "threshold_value": 0.60,
            "threshold_operator": ComparisonOperator.GREATER_THAN_OR_EQUAL,
            "unit": "ratio",
        },
        {
            "name": "Inference Latency P99",
            "metric_type": MetricType.LATENCY,
            "description": "99th percentile inference latency",
            "threshold_value": 100.0,
            "threshold_operator": ComparisonOperator.LESS_THAN_OR_EQUAL,
            "unit": "ms",
        },
        {
            "name": "Error Rate",
            "metric_type": MetricType.ERROR_RATE,
            "description": "System error rate",
            "threshold_value": 0.01,
            "threshold_operator": ComparisonOperator.LESS_THAN_OR_EQUAL,
            "unit": "ratio",
        },
        {
            "name": "System Reliability",
            "metric_type": MetricType.RELIABILITY,
            "description": "System uptime reliability",
            "threshold_value": 0.999,
            "threshold_operator": ComparisonOperator.GREATER_THAN_OR_EQUAL,
            "unit": "ratio",
        },
        {
            "name": "Robustness Score",
            "metric_type": MetricType.ROBUSTNESS,
            "description": "Model robustness to input perturbations",
            "threshold_value": 0.80,
            "threshold_operator": ComparisonOperator.GREATER_THAN_OR_EQUAL,
            "unit": "score",
        },
        {
            "name": "Fairness Score",
            "metric_type": MetricType.FAIRNESS,
            "description": "Model fairness across groups",
            "threshold_value": 0.90,
            "threshold_operator": ComparisonOperator.GREATER_THAN_OR_EQUAL,
            "unit": "score",
        },
    ]

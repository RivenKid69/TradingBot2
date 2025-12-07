# -*- coding: utf-8 -*-
"""
Tests for AI Act Testing Framework (Article 9(6-7)).

Comprehensive test coverage for testing framework functionality including:
- Metric definition and management
- Test scenario creation
- Test execution
- Statistical analysis
- Real-world testing
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

from services.ai_act.testing_framework import (
    TestCategory,
    TestPriority,
    TestStatus,
    MetricType,
    ComparisonOperator,
    VulnerableGroup,
    TestMetric,
    TestScenario,
    TestExecution,
    TestSuite,
    RealWorldTestPlan,
    TestingConfig,
    AIActTestingFramework,
    create_testing_framework,
    get_default_test_metrics,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def testing_config(temp_dir):
    """Create testing configuration."""
    return TestingConfig(
        default_timeout_seconds=60.0,
        minimum_pass_rate=90.0,
        minimum_coverage=80.0,
        confidence_level=0.95,
        log_path=f"{temp_dir}/logs",
        artifacts_path=f"{temp_dir}/artifacts",
    )


@pytest.fixture
def framework(testing_config):
    """Create testing framework instance."""
    return AIActTestingFramework(testing_config)


# =============================================================================
# Test Enums
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_test_categories(self):
        """Test test categories."""
        assert TestCategory.FUNCTIONAL.value == "functional"
        assert TestCategory.PERFORMANCE.value == "performance"
        assert TestCategory.ROBUSTNESS.value == "robustness"
        assert TestCategory.SAFETY.value == "safety"
        assert TestCategory.SECURITY.value == "security"
        assert TestCategory.BIAS.value == "bias"
        assert TestCategory.INTEGRATION.value == "integration"
        assert TestCategory.REAL_WORLD.value == "real_world"

    def test_test_priorities(self):
        """Test test priority ordering."""
        assert TestPriority.CRITICAL.value < TestPriority.HIGH.value
        assert TestPriority.HIGH.value < TestPriority.MEDIUM.value
        assert TestPriority.MEDIUM.value < TestPriority.LOW.value

    def test_test_statuses(self):
        """Test test status values."""
        assert TestStatus.PENDING.value == "pending"
        assert TestStatus.RUNNING.value == "running"
        assert TestStatus.PASSED.value == "passed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.SKIPPED.value == "skipped"
        assert TestStatus.ERROR.value == "error"

    def test_metric_types(self):
        """Test metric types."""
        assert MetricType.ACCURACY.value == "accuracy"
        assert MetricType.LATENCY.value == "latency"
        assert MetricType.THROUGHPUT.value == "throughput"
        assert MetricType.ERROR_RATE.value == "error_rate"
        assert MetricType.ROBUSTNESS.value == "robustness"
        assert MetricType.FAIRNESS.value == "fairness"

    def test_comparison_operators(self):
        """Test comparison operators."""
        assert ComparisonOperator.GREATER_THAN.value == "gt"
        assert ComparisonOperator.GREATER_THAN_OR_EQUAL.value == "gte"
        assert ComparisonOperator.LESS_THAN.value == "lt"
        assert ComparisonOperator.LESS_THAN_OR_EQUAL.value == "lte"
        assert ComparisonOperator.EQUAL.value == "eq"
        assert ComparisonOperator.IN_RANGE.value == "in_range"

    def test_vulnerable_groups(self):
        """Test vulnerable groups per Article 9(9)."""
        assert VulnerableGroup.ELDERLY.value == "elderly"
        assert VulnerableGroup.DISABLED.value == "disabled"
        assert VulnerableGroup.CHILDREN.value == "children"
        assert VulnerableGroup.LOW_INCOME.value == "low_income"


# =============================================================================
# Test Data Classes
# =============================================================================

class TestDataClasses:
    """Test data class initialization."""

    def test_test_metric_creation(self):
        """Test test metric creation."""
        metric = TestMetric(
            metric_id="",
            name="Accuracy",
            metric_type=MetricType.ACCURACY,
            description="Model accuracy metric",
            threshold_value=0.95,
            threshold_operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
        )
        assert metric.metric_id.startswith("METRIC-")
        assert metric.confidence_level == 0.95

    def test_test_metric_evaluate(self):
        """Test test metric evaluation."""
        metric = TestMetric(
            metric_id="M1",
            name="Test",
            metric_type=MetricType.ACCURACY,
            description="",
            threshold_value=0.9,
            threshold_operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
        )

        assert metric.evaluate(0.95)
        assert metric.evaluate(0.9)
        assert not metric.evaluate(0.85)

    def test_test_metric_evaluate_lt(self):
        """Test test metric evaluation with less than."""
        metric = TestMetric(
            metric_id="M1",
            name="Latency",
            metric_type=MetricType.LATENCY,
            description="",
            threshold_value=100.0,
            threshold_operator=ComparisonOperator.LESS_THAN,
        )

        assert metric.evaluate(50.0)
        assert not metric.evaluate(100.0)
        assert not metric.evaluate(150.0)

    def test_test_metric_evaluate_in_range(self):
        """Test test metric evaluation with in range."""
        metric = TestMetric(
            metric_id="M1",
            name="Range",
            metric_type=MetricType.CUSTOM,
            description="",
            threshold_value=0.4,
            threshold_upper=0.6,
            threshold_operator=ComparisonOperator.IN_RANGE,
        )

        assert metric.evaluate(0.5)
        assert metric.evaluate(0.4)
        assert metric.evaluate(0.6)
        assert not metric.evaluate(0.3)
        assert not metric.evaluate(0.7)

    def test_test_scenario_creation(self):
        """Test test scenario creation."""
        scenario = TestScenario(
            scenario_id="",
            name="Accuracy Test",
            description="Test model accuracy",
            category=TestCategory.PERFORMANCE,
        )
        assert scenario.scenario_id.startswith("TEST-")
        assert scenario.status == TestStatus.PENDING
        assert scenario.is_automated

    def test_test_execution_creation(self):
        """Test test execution creation."""
        execution = TestExecution(
            execution_id="",
            scenario_id="TEST-001",
            scenario_name="Test Scenario",
        )
        assert execution.execution_id.startswith("EXEC-")
        assert execution.status == TestStatus.PENDING
        assert execution.start_time != ""

    def test_test_execution_is_complete(self):
        """Test test execution completion check."""
        execution = TestExecution(
            execution_id="E1",
            scenario_id="T1",
            scenario_name="Test",
        )
        assert not execution.is_complete

        execution.status = TestStatus.PASSED
        assert execution.is_complete

        execution.status = TestStatus.RUNNING
        assert not execution.is_complete

    def test_test_suite_creation(self):
        """Test test suite creation."""
        suite = TestSuite(
            suite_id="",
            name="Integration Suite",
            description="Integration tests",
        )
        assert suite.suite_id.startswith("SUITE-")
        assert suite.pass_rate == 0.0

    def test_test_suite_pass_rate(self):
        """Test test suite pass rate calculation."""
        suite = TestSuite(
            suite_id="S1",
            name="Test Suite",
            description="",
            passed_tests=8,
            failed_tests=2,
            skipped_tests=0,
        )
        assert suite.pass_rate == 80.0

    def test_real_world_test_plan_creation(self):
        """Test real-world test plan creation."""
        plan = RealWorldTestPlan(
            plan_id="",
            name="Real-World Test",
            description="Testing in real conditions",
        )
        assert plan.plan_id.startswith("RWTP-")
        assert plan.status == "planned"


# =============================================================================
# Test Metric Management
# =============================================================================

class TestMetricManagement:
    """Test metric management functionality."""

    def test_define_metric(self, framework):
        """Test metric definition."""
        metric = framework.define_metric(
            name="Accuracy",
            metric_type=MetricType.ACCURACY,
            description="Model accuracy",
            threshold_value=0.95,
            threshold_operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
            unit="ratio",
            is_primary=True,
        )
        assert metric.metric_id is not None
        assert metric.threshold_value == 0.95

    def test_get_metric(self, framework):
        """Test getting metric by ID."""
        metric = framework.define_metric(
            name="Latency",
            metric_type=MetricType.LATENCY,
            threshold_value=100.0,
            threshold_operator=ComparisonOperator.LESS_THAN,
        )

        retrieved = framework.get_metric(metric.metric_id)
        assert retrieved is not None
        assert retrieved.name == "Latency"

    def test_get_all_metrics(self, framework):
        """Test getting all metrics."""
        framework.define_metric(
            name="Metric 1",
            metric_type=MetricType.ACCURACY,
            threshold_value=0.9,
        )
        framework.define_metric(
            name="Metric 2",
            metric_type=MetricType.LATENCY,
            threshold_value=50.0,
        )

        all_metrics = framework.get_all_metrics()
        assert len(all_metrics) == 2

    def test_update_metric_threshold(self, framework):
        """Test updating metric threshold."""
        metric = framework.define_metric(
            name="Test",
            metric_type=MetricType.ACCURACY,
            threshold_value=0.9,
        )

        updated = framework.update_metric_threshold(
            metric.metric_id,
            threshold_value=0.95,
        )
        assert updated.threshold_value == 0.95


# =============================================================================
# Test Scenario Management
# =============================================================================

class TestScenarioManagement:
    """Test scenario management functionality."""

    def test_create_scenario(self, framework):
        """Test scenario creation."""
        scenario = framework.create_scenario(
            name="Model Accuracy Test",
            category=TestCategory.PERFORMANCE,
            description="Test model accuracy against threshold",
            test_inputs={"data": "test_data"},
            expected_outputs={"accuracy": 0.95},
            priority=TestPriority.HIGH,
        )
        assert scenario.scenario_id is not None
        assert scenario.category == TestCategory.PERFORMANCE

    def test_create_scenario_with_vulnerable_groups(self, framework):
        """Test creating scenario for vulnerable groups."""
        scenario = framework.create_scenario(
            name="Fairness Test",
            category=TestCategory.BIAS,
            vulnerable_groups=[VulnerableGroup.ELDERLY, VulnerableGroup.DISABLED],
        )
        assert len(scenario.vulnerable_groups_tested) == 2

    def test_get_scenario(self, framework):
        """Test getting scenario by ID."""
        scenario = framework.create_scenario(
            name="Test Scenario",
            category=TestCategory.FUNCTIONAL,
        )

        retrieved = framework.get_scenario(scenario.scenario_id)
        assert retrieved is not None
        assert retrieved.name == "Test Scenario"

    def test_get_scenarios_by_category(self, framework):
        """Test getting scenarios by category."""
        framework.create_scenario(
            name="Performance 1",
            category=TestCategory.PERFORMANCE,
        )
        framework.create_scenario(
            name="Performance 2",
            category=TestCategory.PERFORMANCE,
        )
        framework.create_scenario(
            name="Safety 1",
            category=TestCategory.SAFETY,
        )

        performance = framework.get_scenarios_by_category(TestCategory.PERFORMANCE)
        assert len(performance) == 2

    def test_get_scenarios_for_vulnerable_group(self, framework):
        """Test getting scenarios for vulnerable group."""
        framework.create_scenario(
            name="Elderly Test",
            category=TestCategory.BIAS,
            vulnerable_groups=[VulnerableGroup.ELDERLY],
        )
        framework.create_scenario(
            name="General Test",
            category=TestCategory.FUNCTIONAL,
        )

        elderly_scenarios = framework.get_scenarios_for_vulnerable_group(
            VulnerableGroup.ELDERLY
        )
        assert len(elderly_scenarios) == 1


# =============================================================================
# Test Suite Management
# =============================================================================

class TestSuiteManagement:
    """Test suite management functionality."""

    def test_create_suite(self, framework):
        """Test suite creation."""
        suite = framework.create_suite(
            name="Integration Tests",
            description="Integration test suite",
            category=TestCategory.INTEGRATION,
            parallel_execution=True,
        )
        assert suite.suite_id is not None
        assert suite.parallel_execution

    def test_add_scenario_to_suite(self, framework):
        """Test adding scenarios to suite."""
        scenario = framework.create_scenario(
            name="Test",
            category=TestCategory.FUNCTIONAL,
        )
        suite = framework.create_suite(
            name="Suite",
            description="Test suite",
        )

        updated = framework.add_scenario_to_suite(
            suite.suite_id,
            scenario.scenario_id,
        )
        assert len(updated.scenario_ids) == 1
        assert updated.total_tests == 1

    def test_get_suite(self, framework):
        """Test getting suite by ID."""
        suite = framework.create_suite(
            name="Test Suite",
            description="Description",
        )

        retrieved = framework.get_suite(suite.suite_id)
        assert retrieved is not None
        assert retrieved.name == "Test Suite"


# =============================================================================
# Test Execution
# =============================================================================

class TestExecution_:
    """Test execution functionality."""

    def test_execute_test(self, framework):
        """Test executing a test."""
        metric = framework.define_metric(
            name="Accuracy",
            metric_type=MetricType.ACCURACY,
            threshold_value=0.9,
        )
        scenario = framework.create_scenario(
            name="Accuracy Test",
            category=TestCategory.PERFORMANCE,
            metrics=[metric.metric_id],
        )

        execution = framework.execute_test(
            scenario_id=scenario.scenario_id,
            environment="test",
            executed_by="Tester",
        )

        assert execution.execution_id is not None
        assert execution.status == TestStatus.PASSED
        assert execution.environment == "test"
        assert len(execution.metric_results) == 1

    def test_execute_suite(self, framework):
        """Test executing a test suite."""
        s1 = framework.create_scenario(name="Test 1", category=TestCategory.FUNCTIONAL)
        s2 = framework.create_scenario(name="Test 2", category=TestCategory.FUNCTIONAL)

        suite = framework.create_suite(
            name="Suite",
            description="Test suite",
            scenario_ids=[s1.scenario_id, s2.scenario_id],
        )

        results = framework.execute_suite(
            suite_id=suite.suite_id,
            environment="test",
        )

        assert len(results) == 2

        updated_suite = framework.get_suite(suite.suite_id)
        assert updated_suite.passed_tests == 2

    def test_get_execution(self, framework):
        """Test getting execution by ID."""
        scenario = framework.create_scenario(
            name="Test",
            category=TestCategory.FUNCTIONAL,
        )
        execution = framework.execute_test(scenario.scenario_id)

        retrieved = framework.get_execution(execution.execution_id)
        assert retrieved is not None

    def test_get_scenario_executions(self, framework):
        """Test getting all executions for a scenario."""
        scenario = framework.create_scenario(
            name="Test",
            category=TestCategory.FUNCTIONAL,
        )

        framework.execute_test(scenario.scenario_id)
        framework.execute_test(scenario.scenario_id)
        framework.execute_test(scenario.scenario_id)

        executions = framework.get_scenario_executions(scenario.scenario_id)
        assert len(executions) == 3


# =============================================================================
# Test Statistical Analysis
# =============================================================================

class TestStatisticalAnalysis:
    """Test statistical analysis functionality."""

    def test_analyze_metric_results(self, framework):
        """Test metric results analysis."""
        metric = framework.define_metric(
            name="Accuracy",
            metric_type=MetricType.ACCURACY,
            threshold_value=0.9,
            min_samples=10,
        )

        # Generate sample values
        values = [0.92, 0.93, 0.91, 0.94, 0.92, 0.93, 0.91, 0.92, 0.93, 0.94,
                  0.92, 0.93, 0.91, 0.92, 0.93, 0.94, 0.92, 0.93, 0.91, 0.92,
                  0.92, 0.93, 0.91, 0.94, 0.92, 0.93, 0.91, 0.92, 0.93, 0.94]

        analysis = framework.analyze_metric_results(metric.metric_id, values)

        assert "mean" in analysis
        assert "std_dev" in analysis
        assert "confidence_interval" in analysis
        assert "passed" in analysis
        assert analysis["passed"]
        assert analysis["sample_size"] == 30

    def test_analyze_insufficient_samples(self, framework):
        """Test analysis with insufficient samples."""
        metric = framework.define_metric(
            name="Accuracy",
            metric_type=MetricType.ACCURACY,
            threshold_value=0.9,
            min_samples=30,
        )

        values = [0.92, 0.93, 0.91]  # Only 3 samples

        analysis = framework.analyze_metric_results(metric.metric_id, values)

        assert "error" in analysis
        assert "Insufficient samples" in analysis["error"]

    def test_analyze_failing_metric(self, framework):
        """Test analysis of failing metric."""
        metric = framework.define_metric(
            name="Accuracy",
            metric_type=MetricType.ACCURACY,
            threshold_value=0.95,  # High threshold
            min_samples=10,
        )

        # Generate values below threshold
        values = [0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85,
                  0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85,
                  0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85]

        analysis = framework.analyze_metric_results(metric.metric_id, values)

        assert not analysis["passed"]


# =============================================================================
# Test Real-World Testing
# =============================================================================

class TestRealWorldTesting:
    """Test real-world testing functionality per Article 60."""

    def test_create_real_world_test_plan(self, framework):
        """Test creating real-world test plan."""
        plan = framework.create_real_world_test_plan(
            name="Production Pilot",
            description="Test in production-like conditions",
            test_objectives=["Validate accuracy", "Test robustness"],
            success_criteria=["95% accuracy", "<100ms latency"],
            target_population="Professional traders",
            sample_size=100,
            risk_mitigations=["Human oversight", "Kill switch"],
            planned_start="2025-04-01",
            planned_end="2025-05-01",
        )

        assert plan.plan_id is not None
        assert plan.status == "planned"
        assert len(plan.test_objectives) == 2
        assert len(plan.risk_mitigations) == 2

    def test_approve_real_world_test(self, framework):
        """Test approving real-world test."""
        plan = framework.create_real_world_test_plan(
            name="Test",
            description="Description",
        )

        approved = framework.approve_real_world_test(
            plan.plan_id,
            approved_by="Director",
            human_oversight_measures=["Continuous monitoring", "Daily reviews"],
        )

        assert approved.status == "approved"
        assert approved.approved_by == "Director"
        assert len(approved.human_oversight_measures) == 2

    def test_start_real_world_test(self, framework):
        """Test starting real-world test."""
        plan = framework.create_real_world_test_plan(
            name="Test",
            description="Description",
        )
        framework.approve_real_world_test(plan.plan_id, approved_by="Director")

        started = framework.start_real_world_test(plan.plan_id)

        assert started.status == "active"
        assert started.actual_start is not None

    def test_start_unapproved_test_fails(self, framework):
        """Test that starting unapproved test fails."""
        plan = framework.create_real_world_test_plan(
            name="Test",
            description="Description",
        )

        with pytest.raises(ValueError, match="must be approved"):
            framework.start_real_world_test(plan.plan_id)

    def test_complete_real_world_test(self, framework):
        """Test completing real-world test."""
        plan = framework.create_real_world_test_plan(
            name="Test",
            description="Description",
        )
        framework.approve_real_world_test(plan.plan_id, approved_by="Director")
        framework.start_real_world_test(plan.plan_id)

        completed = framework.complete_real_world_test(
            plan.plan_id,
            findings=["Finding 1", "Finding 2"],
            recommendations=["Recommendation 1"],
        )

        assert completed.status == "completed"
        assert completed.actual_end is not None
        assert len(completed.findings) == 2

    def test_get_real_world_plan(self, framework):
        """Test getting real-world plan by ID."""
        plan = framework.create_real_world_test_plan(
            name="Test Plan",
            description="Description",
        )

        retrieved = framework.get_real_world_plan(plan.plan_id)
        assert retrieved is not None
        assert retrieved.name == "Test Plan"


# =============================================================================
# Test Compliance Reporting
# =============================================================================

class TestComplianceReporting:
    """Test compliance reporting functionality."""

    def test_get_compliance_summary(self, framework):
        """Test getting compliance summary."""
        # Create and execute some tests
        s1 = framework.create_scenario(
            name="Test 1",
            category=TestCategory.PERFORMANCE,
        )
        s2 = framework.create_scenario(
            name="Test 2",
            category=TestCategory.SAFETY,
        )

        framework.execute_test(s1.scenario_id)
        framework.execute_test(s2.scenario_id)

        summary = framework.get_compliance_summary()

        assert "metrics" in summary
        assert "scenarios" in summary
        assert "executions" in summary
        assert "coverage" in summary
        assert "compliance_met" in summary

    def test_generate_test_report(self, framework):
        """Test generating test report."""
        scenario = framework.create_scenario(
            name="Test",
            category=TestCategory.FUNCTIONAL,
        )
        framework.execute_test(scenario.scenario_id)

        report = framework.generate_test_report(include_details=True)

        assert "report_date" in report
        assert "summary" in report
        assert "article_references" in report
        assert "metrics" in report
        assert "scenarios" in report
        assert "recent_executions" in report


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_testing_framework_default(self, temp_dir):
        """Test creating framework with default config."""
        framework = create_testing_framework()
        assert framework is not None
        assert isinstance(framework, AIActTestingFramework)

    def test_create_testing_framework_with_config(self, temp_dir):
        """Test creating framework with config object."""
        config = TestingConfig(
            minimum_pass_rate=95.0,
            log_path=f"{temp_dir}/logs",
            artifacts_path=f"{temp_dir}/artifacts",
        )
        framework = create_testing_framework(config)
        assert framework.config.minimum_pass_rate == 95.0

    def test_create_testing_framework_with_dict(self, temp_dir):
        """Test creating framework with dictionary config."""
        config_dict = {
            "minimum_pass_rate": 98.0,
            "confidence_level": 0.99,
            "log_path": f"{temp_dir}/logs",
            "artifacts_path": f"{temp_dir}/artifacts",
        }
        framework = create_testing_framework(config_dict)
        assert framework.config.minimum_pass_rate == 98.0
        assert framework.config.confidence_level == 0.99

    def test_get_default_test_metrics(self):
        """Test getting default test metrics."""
        metrics = get_default_test_metrics()

        assert len(metrics) >= 5

        # Check required fields
        for metric in metrics:
            assert "name" in metric
            assert "metric_type" in metric
            assert "threshold_value" in metric

        # Check specific metrics exist
        names = {m["name"] for m in metrics}
        assert "Model Accuracy" in names
        assert "Inference Latency P99" in names


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_execute_nonexistent_scenario(self, framework):
        """Test executing non-existent scenario."""
        with pytest.raises(ValueError, match="not found"):
            framework.execute_test("INVALID-ID")

    def test_analyze_nonexistent_metric(self, framework):
        """Test analyzing non-existent metric."""
        with pytest.raises(ValueError, match="not found"):
            framework.analyze_metric_results("INVALID-ID", [1, 2, 3])

    def test_add_scenario_to_nonexistent_suite(self, framework):
        """Test adding to non-existent suite."""
        scenario = framework.create_scenario(
            name="Test",
            category=TestCategory.FUNCTIONAL,
        )

        with pytest.raises(ValueError, match="not found"):
            framework.add_scenario_to_suite("INVALID-ID", scenario.scenario_id)

    def test_bootstrap_confidence_interval(self, framework):
        """Test bootstrap confidence interval calculation."""
        metric = framework.define_metric(
            name="Test",
            metric_type=MetricType.ACCURACY,
            threshold_value=0.9,
            min_samples=10,
        )

        values = [0.92] * 50  # Consistent values

        analysis = framework.analyze_metric_results(metric.metric_id, values)

        assert "bootstrap_ci" in analysis
        assert analysis["bootstrap_ci"]["lower"] <= analysis["mean"]
        assert analysis["bootstrap_ci"]["upper"] >= analysis["mean"]


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_scenario_creation(self, framework):
        """Test concurrent scenario creation."""
        import threading

        scenarios = []
        errors = []

        def create_scenario(index):
            try:
                s = framework.create_scenario(
                    name=f"Scenario {index}",
                    category=TestCategory.FUNCTIONAL,
                )
                scenarios.append(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_scenario, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(scenarios) == 10

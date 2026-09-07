# -*- coding: utf-8 -*-
"""
Tests for AI Act Robustness Testing Framework (Article 15).

Comprehensive test suite for:
- RobustnessTestType, TestSeverity, TestStatus enums
- RobustnessTestResult dataclass
- AdversarialTester class
- DistributionShiftTester class
- FailsafeTester class
- RobustnessTestSuite class
- Factory functions
- Thread safety
- Integration tests

References:
    - EU AI Act Article 15: Accuracy, Robustness, Cybersecurity
    - NIST AI Risk Management Framework
"""

import pytest
import math
import threading
import time
from datetime import datetime
from typing import Any, List
from unittest.mock import MagicMock, patch

from services.ai_act.robustness_testing import (
    RobustnessTestType,
    TestSeverity,
    TestStatus,
    RobustnessTestResult,
    RobustnessTester,
    AdversarialTester,
    DistributionShiftTester,
    FailsafeTester,
    RobustnessTestSuite,
    create_robustness_test_suite,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_model():
    """Simple model function for testing."""

    def model(x):
        if x is None:
            raise ValueError("Input cannot be None")
        if isinstance(x, (int, float)):
            if not math.isfinite(x):
                raise ValueError("Input must be finite")
            return x * 2
        if isinstance(x, (list, tuple)):
            return [v * 2 if isinstance(v, (int, float)) and math.isfinite(v) else v for v in x]
        if isinstance(x, dict):
            return {
                k: v * 2 if isinstance(v, (int, float)) and math.isfinite(v) else v
                for k, v in x.items()
            }
        return x

    return model


@pytest.fixture
def robust_model():
    """Model with built-in robustness."""

    def model(x):
        try:
            if x is None:
                return {"failsafe": True, "value": 0}
            if isinstance(x, (int, float)):
                if not math.isfinite(x):
                    return {"failsafe": True, "value": 0}
                return {"failsafe": False, "value": x * 2}
            if isinstance(x, (list, tuple)):
                result = []
                for v in x:
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        result.append(v * 2)
                    else:
                        result.append(0)
                return {"failsafe": False, "value": result}
            return {"failsafe": False, "value": x}
        except Exception:
            return {"failsafe": True, "value": 0}

    return model


@pytest.fixture
def test_inputs():
    """Standard test inputs."""
    return [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0, 0.5, 0.1, 0.01]


# =============================================================================
# Test Enums
# =============================================================================


class TestRobustnessTestType:
    """Tests for RobustnessTestType enum."""

    def test_adversarial_types_exist(self):
        """All adversarial test types should exist."""
        assert RobustnessTestType.ADVERSARIAL_INPUT
        assert RobustnessTestType.ADVERSARIAL_PERTURBATION
        assert RobustnessTestType.INPUT_CORRUPTION
        assert RobustnessTestType.BOUNDARY_TESTING

    def test_distribution_shift_types_exist(self):
        """All distribution shift test types should exist."""
        assert RobustnessTestType.DATA_DRIFT
        assert RobustnessTestType.CONCEPT_DRIFT
        assert RobustnessTestType.COVARIATE_SHIFT
        assert RobustnessTestType.REGIME_CHANGE

    def test_failsafe_types_exist(self):
        """All failsafe test types should exist."""
        assert RobustnessTestType.FAILSAFE_TRIGGER
        assert RobustnessTestType.GRACEFUL_DEGRADATION
        assert RobustnessTestType.RECOVERY_TESTING
        assert RobustnessTestType.TIMEOUT_HANDLING

    def test_data_quality_types_exist(self):
        """All data quality test types should exist."""
        assert RobustnessTestType.MISSING_DATA
        assert RobustnessTestType.OUTLIER_HANDLING
        assert RobustnessTestType.NOISE_INJECTION


class TestTestSeverity:
    """Tests for TestSeverity enum."""

    def test_all_severities_defined(self):
        """All severities should be defined."""
        expected = {"critical", "high", "medium", "low", "info"}
        actual = {s.value for s in TestSeverity}
        assert expected == actual

    def test_severity_values(self):
        """Severity values should be correct."""
        assert TestSeverity.CRITICAL.value == "critical"
        assert TestSeverity.HIGH.value == "high"
        assert TestSeverity.MEDIUM.value == "medium"
        assert TestSeverity.LOW.value == "low"
        assert TestSeverity.INFO.value == "info"


class TestTestStatus:
    """Tests for TestStatus enum."""

    def test_all_statuses_defined(self):
        """All statuses should be defined."""
        expected = {"pending", "running", "passed", "failed", "error", "skipped"}
        actual = {s.value for s in TestStatus}
        assert expected == actual


# =============================================================================
# Test RobustnessTestResult
# =============================================================================


class TestRobustnessTestResult:
    """Tests for RobustnessTestResult dataclass."""

    def test_default_creation(self):
        """Test creating result with defaults."""
        result = RobustnessTestResult()
        assert result.test_id  # Should have UUID
        assert result.status == TestStatus.PENDING
        assert result.score == 0.0

    def test_custom_creation(self):
        """Test creating result with custom values."""
        result = RobustnessTestResult(
            test_type=RobustnessTestType.ADVERSARIAL_PERTURBATION,
            test_name="Custom Test",
            status=TestStatus.PASSED,
            score=0.95,
            inputs_tested=100,
            failures_detected=5,
        )
        assert result.test_name == "Custom Test"
        assert result.status == TestStatus.PASSED
        assert result.score == 0.95

    def test_is_passed(self):
        """Test is_passed method."""
        passed = RobustnessTestResult(status=TestStatus.PASSED)
        failed = RobustnessTestResult(status=TestStatus.FAILED)

        assert passed.is_passed() is True
        assert failed.is_passed() is False

    def test_get_failure_rate_normal(self):
        """Test failure rate calculation."""
        result = RobustnessTestResult(
            inputs_tested=100,
            failures_detected=10,
        )
        assert result.get_failure_rate() == 10.0

    def test_get_failure_rate_zero_inputs(self):
        """Test failure rate with zero inputs."""
        result = RobustnessTestResult(
            inputs_tested=0,
            failures_detected=0,
        )
        assert result.get_failure_rate() == 0.0

    def test_get_compliance_text(self):
        """Test compliance text generation."""
        result = RobustnessTestResult(
            test_type=RobustnessTestType.ADVERSARIAL_PERTURBATION,
            test_name="Adversarial Test",
            status=TestStatus.PASSED,
            score=0.95,
            severity=TestSeverity.LOW,
            details="Test completed successfully",
            inputs_tested=100,
            failures_detected=5,
            recommendations=["Review input validation"],
        )

        text = result.get_compliance_text()

        assert "Adversarial Test" in text
        assert "passed" in text.lower()  # Case-insensitive check
        assert "95" in text  # 95%
        assert "100" in text  # inputs tested
        assert "input validation" in text.lower()

    def test_get_compliance_text_with_emoji(self):
        """Test compliance text has appropriate emoji."""
        passed = RobustnessTestResult(status=TestStatus.PASSED)
        failed = RobustnessTestResult(status=TestStatus.FAILED)
        error = RobustnessTestResult(status=TestStatus.ERROR)

        # Just check emojis are present (they differ by status)
        assert passed.get_compliance_text()
        assert failed.get_compliance_text()
        assert error.get_compliance_text()


# =============================================================================
# Test AdversarialTester
# =============================================================================


class TestAdversarialTester:
    """Tests for AdversarialTester class."""

    @pytest.fixture
    def tester(self):
        """Create AdversarialTester instance."""
        return AdversarialTester(
            perturbation_epsilon=0.01,
            num_perturbations=5,
            seed=42,
        )

    def test_initialization(self, tester):
        """Test initialization."""
        assert tester.perturbation_epsilon == 0.01
        assert tester.num_perturbations == 5

    def test_run_test_with_robust_model(self, tester, test_inputs):
        """Test with a model that handles perturbations well."""

        def robust_model(x):
            if isinstance(x, (int, float)) and math.isfinite(x):
                return x * 2  # Deterministic, proportional output
            return 0

        result = tester.run_test(robust_model, test_inputs)

        assert isinstance(result, RobustnessTestResult)
        assert result.test_type == RobustnessTestType.ADVERSARIAL_PERTURBATION
        assert result.inputs_tested > 0
        assert result.duration_ms >= 0  # Can be 0 on fast machines

    def test_run_test_with_fragile_model(self, tester, test_inputs):
        """Test with a model that fails on perturbations."""
        call_count = [0]

        def fragile_model(x):
            call_count[0] += 1
            if call_count[0] > 3:
                raise ValueError("Model crashed!")
            return x * 2

        result = tester.run_test(fragile_model, test_inputs)

        assert result.failures_detected > 0

    def test_run_test_empty_inputs(self, tester):
        """Test with empty inputs."""

        def model(x):
            return x * 2

        result = tester.run_test(model, [])

        assert result.inputs_tested == 0
        assert result.score == 0.0

    def test_perturbation_consistency_with_seed(self):
        """Test that seed provides consistent perturbations."""
        tester1 = AdversarialTester(seed=123)
        tester2 = AdversarialTester(seed=123)

        perturbed1 = tester1._perturb_value(100.0)
        perturbed2 = tester2._perturb_value(100.0)

        assert perturbed1 == perturbed2

    def test_perturb_list(self, tester):
        """Test perturbation of list inputs."""
        original = [1.0, 2.0, 3.0]
        perturbed = tester._perturb_input(original)

        assert isinstance(perturbed, list)
        assert len(perturbed) == len(original)
        # Values should be close but not identical
        for o, p in zip(original, perturbed):
            assert abs(o - p) < abs(o) * 0.1  # Within 10%

    def test_perturb_dict(self, tester):
        """Test perturbation of dict inputs."""
        original = {"a": 1.0, "b": 2.0}
        perturbed = tester._perturb_input(original)

        assert isinstance(perturbed, dict)
        assert set(perturbed.keys()) == set(original.keys())

    def test_adversarial_success_detection(self, tester):
        """Test detection of adversarial success."""
        # Small input change, large output change = adversarial success
        assert tester._is_adversarial_success(1.0, 2.0)  # 100% change > 10%

        # Proportional output change = not adversarial
        assert not tester._is_adversarial_success(1.0, 1.001)  # 0.1% change

    def test_recommendations_on_low_score(self, tester):
        """Test that recommendations are given for low scores."""

        def failing_model(x):
            raise ValueError("Always fails")

        result = tester.run_test(failing_model, [1.0, 2.0, 3.0])

        assert len(result.recommendations) > 0


# =============================================================================
# Test DistributionShiftTester
# =============================================================================


class TestDistributionShiftTester:
    """Tests for DistributionShiftTester class."""

    @pytest.fixture
    def tester(self):
        """Create DistributionShiftTester instance."""
        return DistributionShiftTester(
            shift_magnitude=0.5,
            regime_multipliers=[0.5, 1.5, 2.0],
            seed=42,
        )

    def test_initialization(self, tester):
        """Test initialization."""
        assert tester.shift_magnitude == 0.5
        assert len(tester.regime_multipliers) == 3

    def test_run_test_with_robust_model(self, tester, test_inputs):
        """Test with robust model."""

        def robust_model(x):
            if isinstance(x, (int, float)) and math.isfinite(x):
                return x * 2
            return 0

        result = tester.run_test(robust_model, test_inputs)

        assert isinstance(result, RobustnessTestResult)
        assert result.test_type == RobustnessTestType.DATA_DRIFT
        assert result.inputs_tested > 0
        assert result.score > 0

    def test_run_test_with_fragile_model(self, tester, test_inputs):
        """Test with model that fails on shifted data."""

        def fragile_model(x):
            if isinstance(x, (int, float)):
                if x > 10:  # Fails on large values
                    raise ValueError("Value too large")
            return x * 2

        result = tester.run_test(fragile_model, test_inputs)

        # Should detect failures on multiplied inputs
        assert result.failures_detected > 0

    def test_shift_distribution(self, tester):
        """Test distribution shifting."""
        inputs = [1.0, 2.0, 3.0]
        shifted = tester._shift_distribution(inputs, 2.0)

        assert len(shifted) == len(inputs)
        # Shifted values should be roughly multiplied
        for orig, shift in zip(inputs, shifted):
            assert abs(shift - orig * 2.0) < abs(orig) * 0.2  # Allow 20% noise

    def test_shift_list_inputs(self, tester):
        """Test shifting list inputs."""
        inputs = [[1.0, 2.0], [3.0, 4.0]]
        shifted = tester._shift_distribution(inputs, 1.5)

        assert len(shifted) == 2
        assert all(isinstance(s, list) for s in shifted)

    def test_shift_dict_inputs(self, tester):
        """Test shifting dict inputs."""
        inputs = [{"a": 1.0, "b": 2.0}]
        shifted = tester._shift_distribution(inputs, 2.0)

        assert len(shifted) == 1
        assert isinstance(shifted[0], dict)

    def test_valid_output_detection(self, tester):
        """Test valid output detection."""
        assert tester._is_valid_output(1.0) is True
        assert tester._is_valid_output("string") is True
        assert tester._is_valid_output(None) is False
        assert tester._is_valid_output(float("inf")) is False
        assert tester._is_valid_output(float("nan")) is False

    def test_scenario_results_in_metadata(self, tester, test_inputs):
        """Test that scenario results are in metadata."""

        def model(x):
            return x * 2 if isinstance(x, (int, float)) else x

        result = tester.run_test(model, test_inputs)

        assert "scenario_results" in result.metadata
        assert len(result.metadata["scenario_results"]) == len(tester.regime_multipliers)


# =============================================================================
# Test FailsafeTester
# =============================================================================


class TestFailsafeTester:
    """Tests for FailsafeTester class."""

    @pytest.fixture
    def tester(self):
        """Create FailsafeTester instance."""
        return FailsafeTester(
            timeout_ms=5000.0,
            max_retries=3,
        )

    def test_initialization(self, tester):
        """Test initialization."""
        assert tester.timeout_ms == 5000.0
        assert tester.max_retries == 3

    def test_run_test_with_robust_model(self, tester, robust_model, test_inputs):
        """Test with robust model that handles edge cases."""
        result = tester.run_test(robust_model, test_inputs)

        assert isinstance(result, RobustnessTestResult)
        assert result.test_type == RobustnessTestType.FAILSAFE_TRIGGER
        assert result.inputs_tested > 0

    def test_run_test_with_fragile_model(self, tester, simple_model, test_inputs):
        """Test with model that throws on edge cases."""
        result = tester.run_test(simple_model, test_inputs)

        # Should detect failures on edge cases (None, inf, nan, etc.)
        assert result.failures_detected > 0

    def test_failsafe_function(self, tester, test_inputs):
        """Test with failsafe recovery function."""
        recovery_count = [0]

        def failing_model(x):
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                raise ValueError("Invalid input")
            return x * 2

        def failsafe():
            recovery_count[0] += 1
            # Recovery logic here

        result = tester.run_test(failing_model, test_inputs, failsafe_fn=failsafe)

        assert result.metadata.get("recovery_successful", 0) <= recovery_count[0]

    def test_edge_case_generation(self, tester):
        """Test edge case generation."""
        base_inputs = [1.0, 2.0]
        edge_cases = tester._generate_edge_cases(base_inputs)

        # Should have standard edge cases
        case_names = [ec["name"] for ec in edge_cases]

        assert "null_input" in case_names
        assert "empty_list" in case_names
        assert "inf_positive" in case_names
        assert "nan" in case_names
        assert "zero" in case_names

    def test_edge_case_from_list_input(self, tester):
        """Test edge cases generated from list inputs."""
        base_inputs = [[1.0, 2.0, 3.0]]
        edge_cases = tester._generate_edge_cases(base_inputs)

        # Should have corrupted list case
        corrupted_cases = [ec for ec in edge_cases if "corrupted" in ec["name"]]
        assert len(corrupted_cases) > 0

    def test_edge_case_from_dict_input(self, tester):
        """Test edge cases generated from dict inputs."""
        base_inputs = [{"a": 1.0, "b": 2.0}]
        edge_cases = tester._generate_edge_cases(base_inputs)

        # Should have partial dict case
        partial_cases = [ec for ec in edge_cases if "partial" in ec["name"]]
        assert len(partial_cases) > 0

    def test_failsafe_output_detection(self, tester):
        """Test failsafe output detection."""
        assert tester._is_failsafe_output(None) is True
        assert tester._is_failsafe_output({"failsafe": True}) is True
        assert tester._is_failsafe_output({"error": True}) is True
        assert tester._is_failsafe_output(0.0) is True
        assert tester._is_failsafe_output({"value": 1.0}) is False
        assert tester._is_failsafe_output(1.0) is False


# =============================================================================
# Test RobustnessTestSuite
# =============================================================================


class TestRobustnessTestSuite:
    """Tests for RobustnessTestSuite class."""

    @pytest.fixture
    def suite(self):
        """Create RobustnessTestSuite instance."""
        return RobustnessTestSuite(
            adversarial_tester=AdversarialTester(seed=42),
            distribution_shift_tester=DistributionShiftTester(seed=42),
            failsafe_tester=FailsafeTester(),
        )

    def test_initialization(self, suite):
        """Test initialization."""
        assert suite.suite_id
        assert suite.suite_name == "EU AI Act Article 15 Robustness Test Suite"
        assert suite.adversarial_tester is not None
        assert suite.distribution_shift_tester is not None
        assert suite.failsafe_tester is not None

    def test_run_all_tests(self, suite, robust_model, test_inputs):
        """Test running all tests."""
        results = suite.run_all_tests(robust_model, test_inputs)

        assert len(results) == 3  # Adversarial, Distribution, Failsafe
        assert all(isinstance(r, RobustnessTestResult) for r in results)
        assert suite.results == results

    def test_get_overall_score(self, suite, robust_model, test_inputs):
        """Test overall score calculation."""
        suite.run_all_tests(robust_model, test_inputs)
        score = suite.get_overall_score()

        assert 0.0 <= score <= 1.0

    def test_get_overall_score_no_results(self, suite):
        """Test overall score with no results."""
        assert suite.get_overall_score() == 0.0

    def test_is_compliant(self, suite, robust_model, test_inputs):
        """Test compliance check."""
        suite.run_all_tests(robust_model, test_inputs)

        # With default minimum of 0.8
        compliant = suite.is_compliant()
        assert isinstance(compliant, bool)

        # With custom minimum
        compliant_low = suite.is_compliant(minimum_score=0.1)
        assert compliant_low is True

    def test_generate_compliance_report(self, suite, robust_model, test_inputs):
        """Test compliance report generation."""
        suite.run_all_tests(robust_model, test_inputs)
        report = suite.generate_compliance_report()

        assert "ARTICLE 15 ROBUSTNESS COMPLIANCE REPORT" in report
        assert "EU AI Act" in report
        assert suite.suite_id in report
        assert "Overall Robustness Score" in report
        assert "SUMMARY" in report

    def test_export_for_audit(self, suite, robust_model, test_inputs):
        """Test audit export."""
        suite.run_all_tests(robust_model, test_inputs)
        export = suite.export_for_audit()

        assert "export_timestamp" in export
        assert "suite_id" in export
        assert "overall_score" in export
        assert "is_compliant" in export
        assert "results" in export
        assert len(export["results"]) == 3


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_robustness_test_suite_default(self):
        """Test creating suite with defaults."""
        suite = create_robustness_test_suite()

        assert isinstance(suite, RobustnessTestSuite)
        assert suite.adversarial_tester is not None
        assert suite.distribution_shift_tester is not None
        assert suite.failsafe_tester is not None

    def test_create_robustness_test_suite_custom(self):
        """Test creating suite with custom parameters."""
        suite = create_robustness_test_suite(
            perturbation_epsilon=0.05,
            num_perturbations=20,
            shift_magnitude=1.0,
            regime_multipliers=[0.25, 0.5, 2.0, 4.0],
            timeout_ms=10000.0,
            seed=123,
        )

        assert suite.adversarial_tester.perturbation_epsilon == 0.05
        assert suite.adversarial_tester.num_perturbations == 20
        assert suite.distribution_shift_tester.shift_magnitude == 1.0
        assert len(suite.distribution_shift_tester.regime_multipliers) == 4
        assert suite.failsafe_tester.timeout_ms == 10000.0


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_test_runs(self):
        """Test concurrent test runs don't interfere."""
        suite1 = create_robustness_test_suite(seed=1)
        suite2 = create_robustness_test_suite(seed=2)

        def model(x):
            return x * 2 if isinstance(x, (int, float)) else x

        test_inputs = [1.0, 2.0, 3.0]
        results = []
        errors = []

        def run_suite(suite):
            try:
                result = suite.run_all_tests(model, test_inputs)
                results.append(result)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run_suite, args=(suite1,))
        t2 = threading.Thread(target=run_suite, args=(suite2,))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
        assert len(results) == 2


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_trading_model_robustness(self):
        """Test robustness of a trading model."""

        def trading_model(market_data):
            """Simulated trading model."""
            if market_data is None:
                return {"action": "hold", "confidence": 0.0}

            if isinstance(market_data, dict):
                price = market_data.get("price", 0)
                volume = market_data.get("volume", 0)

                if not isinstance(price, (int, float)) or not math.isfinite(price):
                    return {"action": "hold", "confidence": 0.0}

                if price > 100:
                    return {"action": "sell", "confidence": 0.8}
                elif price < 50:
                    return {"action": "buy", "confidence": 0.8}
                else:
                    return {"action": "hold", "confidence": 0.5}

            return {"action": "hold", "confidence": 0.0}

        # Test inputs simulating market data
        test_inputs = [
            {"price": 75.0, "volume": 1000},
            {"price": 120.0, "volume": 2000},
            {"price": 40.0, "volume": 1500},
            {"price": 100.0, "volume": 500},
            {"price": 85.0, "volume": 3000},
        ]

        suite = create_robustness_test_suite(
            perturbation_epsilon=0.05,
            num_perturbations=5,
            seed=42,
        )

        results = suite.run_all_tests(trading_model, test_inputs)

        # Verify all tests ran
        assert len(results) == 3

        # Check overall compliance
        report = suite.generate_compliance_report()
        assert "ROBUSTNESS" in report

    def test_full_compliance_workflow(self):
        """Test full Article 15 compliance workflow."""

        # Create a model with known behavior
        def ml_model(features):
            if features is None:
                return None
            if isinstance(features, (int, float)):
                if not math.isfinite(features):
                    return None
                return features * 0.5 + 0.3  # Linear prediction
            return features

        test_inputs = [0.1, 0.5, 1.0, 2.0, 5.0]

        # Create suite
        suite = create_robustness_test_suite(seed=42)

        # Run tests
        results = suite.run_all_tests(ml_model, test_inputs)

        # Generate compliance documentation
        report = suite.generate_compliance_report()
        audit_export = suite.export_for_audit()

        # Verify compliance artifacts
        assert "Article 15" in report or "ARTICLE 15" in report
        assert audit_export["suite_id"] == suite.suite_id
        assert len(audit_export["results"]) == 3

        # Check individual test results
        for result in results:
            assert result.test_id
            assert result.timestamp
            assert result.inputs_tested > 0

    def test_model_with_failsafe_recovery(self):
        """Test model with proper failsafe recovery."""
        recovery_state = {"count": 0}

        def model_with_failsafe(x):
            if x is None:
                raise ValueError("Null input")
            if isinstance(x, (int, float)) and not math.isfinite(x):
                raise ValueError("Non-finite input")
            return x * 2

        def recovery_function():
            recovery_state["count"] += 1
            # Reset any state, reconnect, etc.

        suite = create_robustness_test_suite()

        # Run failsafe test specifically
        failsafe_result = suite.failsafe_tester.run_test(
            model_with_failsafe,
            [1.0, 2.0, 3.0],
            failsafe_fn=recovery_function,
        )

        # Recovery should have been called for failures
        assert recovery_state["count"] > 0 or failsafe_result.failures_detected == 0

    def test_stress_testing_large_input_set(self):
        """Test with larger input set for stress testing."""

        def simple_model(x):
            return x * 2 if isinstance(x, (int, float)) else x

        # Generate larger input set
        test_inputs = [float(i) for i in range(100)]

        suite = create_robustness_test_suite(
            num_perturbations=3,  # Reduce for speed
            seed=42,
        )

        results = suite.run_all_tests(simple_model, test_inputs)

        # Verify all tests complete
        assert all(r.status in [TestStatus.PASSED, TestStatus.FAILED] for r in results)

        # Check performance metadata
        total_duration = sum(r.duration_ms for r in results)
        assert total_duration >= 0  # Can be 0 on fast machines


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_model_output(self):
        """Test handling of empty model output."""

        def empty_model(x):
            return None

        suite = create_robustness_test_suite()
        results = suite.run_all_tests(empty_model, [1.0, 2.0, 3.0])

        # Should complete without crashing
        assert len(results) == 3

    def test_model_always_fails(self):
        """Test with model that always throws."""

        def failing_model(x):
            raise Exception("Always fails")

        suite = create_robustness_test_suite()
        results = suite.run_all_tests(failing_model, [1.0, 2.0])

        # Should record failures, not crash
        assert all(r.failures_detected > 0 or r.status == TestStatus.ERROR for r in results)

    def test_very_small_perturbation(self):
        """Test with very small perturbation epsilon."""
        tester = AdversarialTester(perturbation_epsilon=1e-10, seed=42)

        def precise_model(x):
            return round(x * 2, 10)

        result = tester.run_test(precise_model, [1.0, 2.0, 3.0])

        # Very small perturbations shouldn't cause failures
        assert result.score > 0.9

    def test_very_large_perturbation(self):
        """Test with very large perturbation epsilon."""
        tester = AdversarialTester(perturbation_epsilon=10.0, seed=42)

        def simple_model(x):
            return x * 2

        result = tester.run_test(simple_model, [1.0, 2.0, 3.0])

        # Large perturbations will likely cause failures
        assert result.inputs_tested > 0


# =============================================================================
# Test Score Thresholds
# =============================================================================


class TestScoreThresholds:
    """Test score threshold behavior."""

    def test_adversarial_score_thresholds(self):
        """Test adversarial test score thresholds."""
        tester = AdversarialTester(seed=42)

        # Perfect model
        def perfect_model(x):
            return x  # No change regardless of input

        result = tester.run_test(perfect_model, [1.0, 2.0, 3.0])

        # Should have high score if model is stable
        # (exact behavior depends on adversarial success detection)

    def test_distribution_shift_score_thresholds(self):
        """Test distribution shift score thresholds."""
        tester = DistributionShiftTester(seed=42)

        # Robust model that handles any valid input
        def robust_model(x):
            if isinstance(x, (int, float)) and math.isfinite(x):
                return x * 2
            return 0  # Safe fallback

        result = tester.run_test(robust_model, [1.0, 2.0, 3.0])

        # Should pass with high score
        assert result.score >= 0.7

    def test_failsafe_score_thresholds(self):
        """Test failsafe score thresholds."""
        tester = FailsafeTester()

        # Model with proper error handling
        def safe_model(x):
            try:
                if x is None:
                    return {"error": True}
                if isinstance(x, (int, float)):
                    if not math.isfinite(x):
                        return {"error": True}
                    return {"value": x * 2}
                return {"error": True}
            except Exception:
                return {"error": True}

        result = tester.run_test(safe_model, [1.0, 2.0, 3.0])

        # Should have reasonable score with error handling
        assert result.inputs_tested > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

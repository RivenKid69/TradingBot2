# -*- coding: utf-8 -*-
"""
EU AI Act Article 15 - Robustness Testing Framework.

This module implements robustness testing for high-risk AI systems
as required by Article 15 of the EU AI Act.

Article 15 Requirements:
- High-risk AI systems shall be resilient as regards errors, faults
  or inconsistencies that may occur within the system or the
  environment in which the system operates.
- High-risk AI systems shall be resilient as regards attempts by
  unauthorised third parties to alter their use or performance.

Testing Framework (from EU_AI_ACT_INTEGRATION_PLAN.md section 1.3.2):
1. Adversarial Testing - Test resilience against adversarial inputs
2. Distribution Shift Testing - Test performance under data drift
3. Failsafe Testing - Test graceful degradation mechanisms

References:
- EU AI Act Article 15: https://artificialintelligenceact.eu/article/15/
- NIST AI Risk Management Framework
- ISO/IEC TR 24028:2020 AI trustworthiness
"""

from __future__ import annotations

import math
import random
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class RobustnessTestType(Enum):
    """Types of robustness tests per Article 15."""

    # Adversarial Testing
    ADVERSARIAL_INPUT = auto()
    ADVERSARIAL_PERTURBATION = auto()
    INPUT_CORRUPTION = auto()
    BOUNDARY_TESTING = auto()

    # Distribution Shift Testing
    DATA_DRIFT = auto()
    CONCEPT_DRIFT = auto()
    COVARIATE_SHIFT = auto()
    REGIME_CHANGE = auto()

    # Failsafe Testing
    FAILSAFE_TRIGGER = auto()
    GRACEFUL_DEGRADATION = auto()
    RECOVERY_TESTING = auto()
    TIMEOUT_HANDLING = auto()

    # Performance Degradation
    STRESS_TESTING = auto()
    LATENCY_TESTING = auto()
    THROUGHPUT_TESTING = auto()

    # Data Quality
    MISSING_DATA = auto()
    OUTLIER_HANDLING = auto()
    NOISE_INJECTION = auto()


class TestSeverity(Enum):
    """Severity level of test findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TestStatus(Enum):
    """Status of a robustness test."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class RobustnessTestResult:
    """Result of a single robustness test.

    Attributes:
        test_id: Unique identifier for this test run
        test_type: Type of robustness test performed
        test_name: Human-readable test name
        status: Pass/fail status
        severity: Severity if failed
        score: Robustness score (0.0 to 1.0)
        details: Detailed test results
        timestamp: When test was executed
        duration_ms: Test execution time in milliseconds
        inputs_tested: Number of inputs tested
        failures_detected: Number of failures detected
        recommendations: Recommendations if issues found
        metadata: Additional test metadata
    """

    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    test_type: RobustnessTestType = RobustnessTestType.ADVERSARIAL_INPUT
    test_name: str = ""
    status: TestStatus = TestStatus.PENDING
    severity: Optional[TestSeverity] = None
    score: float = 0.0
    details: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    inputs_tested: int = 0
    failures_detected: int = 0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_passed(self) -> bool:
        """Check if test passed."""
        return self.status == TestStatus.PASSED

    def get_failure_rate(self) -> float:
        """Get failure rate as percentage."""
        if self.inputs_tested == 0:
            return 0.0
        return (self.failures_detected / self.inputs_tested) * 100

    def get_compliance_text(self) -> str:
        """Generate Article 15 compliant test result description."""
        status_emoji = {
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.ERROR: "⚠️",
            TestStatus.SKIPPED: "⏭️",
            TestStatus.RUNNING: "🔄",
            TestStatus.PENDING: "⏳",
        }

        emoji = status_emoji.get(self.status, "❓")
        lines = [
            f"{emoji} Test: {self.test_name}",
            f"   Type: {self.test_type.name}",
            f"   Status: {self.status.value}",
            f"   Score: {self.score:.2%}",
            f"   Inputs Tested: {self.inputs_tested}",
            f"   Failures: {self.failures_detected} ({self.get_failure_rate():.1f}%)",
        ]

        if self.severity:
            lines.append(f"   Severity: {self.severity.value}")

        if self.details:
            lines.append(f"   Details: {self.details}")

        if self.recommendations:
            lines.append("   Recommendations:")
            for rec in self.recommendations:
                lines.append(f"     - {rec}")

        return "\n".join(lines)


class RobustnessTester(ABC):
    """Abstract base class for robustness testers.

    All robustness testers must implement the run_test method.
    """

    @abstractmethod
    def run_test(
        self,
        model_fn: Callable[[Any], Any],
        test_inputs: List[Any],
        **kwargs: Any,
    ) -> RobustnessTestResult:
        """Run robustness test.

        Args:
            model_fn: Function to test (takes input, returns output)
            test_inputs: List of test inputs
            **kwargs: Additional test parameters

        Returns:
            RobustnessTestResult with test outcome
        """
        pass


class AdversarialTester(RobustnessTester):
    """Tests resilience against adversarial inputs.

    Article 15 requires resilience against "attempts by unauthorised
    third parties to alter their use or performance."

    Tests include:
    - Input perturbations (small changes that cause different outputs)
    - Boundary attacks (extreme values)
    - Gradient-based attacks (FGSM, PGD if applicable)
    """

    def __init__(
        self,
        perturbation_epsilon: float = 0.01,
        num_perturbations: int = 10,
        seed: Optional[int] = None,
    ):
        """Initialize adversarial tester.

        Args:
            perturbation_epsilon: Maximum perturbation size
            num_perturbations: Number of perturbations per input
            seed: Random seed for reproducibility
        """
        self.perturbation_epsilon = perturbation_epsilon
        self.num_perturbations = num_perturbations
        self._rng = random.Random(seed)

    def run_test(
        self,
        model_fn: Callable[[Any], Any],
        test_inputs: List[Any],
        **kwargs: Any,
    ) -> RobustnessTestResult:
        """Run adversarial robustness test.

        Args:
            model_fn: Model function to test
            test_inputs: Original inputs to perturb
            **kwargs: Additional parameters

        Returns:
            RobustnessTestResult
        """
        start_time = datetime.utcnow()
        inputs_tested = 0
        failures_detected = 0
        total_perturbations = 0

        original_outputs = []
        perturbed_outputs = []

        try:
            for input_data in test_inputs:
                # Get original output
                try:
                    original_output = model_fn(input_data)
                    original_outputs.append(original_output)
                except Exception as e:
                    logger.warning("Model error on original input: %s", e)
                    failures_detected += 1
                    inputs_tested += 1
                    continue

                # Test perturbations
                for _ in range(self.num_perturbations):
                    perturbed_input = self._perturb_input(input_data)
                    total_perturbations += 1

                    try:
                        perturbed_output = model_fn(perturbed_input)
                        perturbed_outputs.append(perturbed_output)

                        # Check if small perturbation caused large output change
                        if self._is_adversarial_success(original_output, perturbed_output):
                            failures_detected += 1
                    except Exception as e:
                        # Model should handle perturbed inputs gracefully
                        logger.warning("Model error on perturbed input: %s", e)
                        failures_detected += 1

                inputs_tested += 1

        except Exception as e:
            logger.error("Adversarial test error: %s", e)
            return RobustnessTestResult(
                test_type=RobustnessTestType.ADVERSARIAL_PERTURBATION,
                test_name="Adversarial Perturbation Test",
                status=TestStatus.ERROR,
                details=f"Test error: {str(e)}",
                inputs_tested=inputs_tested,
                failures_detected=failures_detected,
            )

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Calculate robustness score
        if total_perturbations > 0:
            score = 1.0 - (failures_detected / total_perturbations)
        else:
            score = 0.0

        # Determine status and severity
        if score >= 0.95:
            status = TestStatus.PASSED
            severity = None
        elif score >= 0.80:
            status = TestStatus.PASSED
            severity = TestSeverity.LOW
        elif score >= 0.60:
            status = TestStatus.FAILED
            severity = TestSeverity.MEDIUM
        else:
            status = TestStatus.FAILED
            severity = TestSeverity.HIGH

        recommendations = []
        if score < 0.95:
            recommendations.append("Consider implementing input validation and sanitization")
            recommendations.append("Review model's sensitivity to small input variations")
        if score < 0.80:
            recommendations.append("Implement adversarial training or robust optimization")

        return RobustnessTestResult(
            test_type=RobustnessTestType.ADVERSARIAL_PERTURBATION,
            test_name="Adversarial Perturbation Test",
            status=status,
            severity=severity,
            score=score,
            details=(
                f"Tested {inputs_tested} inputs with {self.num_perturbations} "
                f"perturbations each (epsilon={self.perturbation_epsilon}). "
                f"Found {failures_detected} adversarial vulnerabilities."
            ),
            timestamp=start_time,
            duration_ms=duration_ms,
            inputs_tested=total_perturbations,
            failures_detected=failures_detected,
            recommendations=recommendations,
            metadata={
                "epsilon": self.perturbation_epsilon,
                "num_perturbations": self.num_perturbations,
                "original_inputs": len(test_inputs),
            },
        )

    def _perturb_input(self, input_data: Any) -> Any:
        """Perturb input data.

        Args:
            input_data: Original input

        Returns:
            Perturbed input
        """
        if isinstance(input_data, (list, tuple)):
            return type(input_data)(self._perturb_value(v) for v in input_data)
        elif isinstance(input_data, dict):
            return {k: self._perturb_value(v) for k, v in input_data.items()}
        else:
            return self._perturb_value(input_data)

    def _perturb_value(self, value: Any) -> Any:
        """Perturb a single value.

        Args:
            value: Original value

        Returns:
            Perturbed value
        """
        if isinstance(value, (int, float)):
            if math.isfinite(value) and value != 0:
                perturbation = self._rng.gauss(0, abs(value) * self.perturbation_epsilon)
                return value + perturbation
            else:
                return value + self._rng.gauss(0, self.perturbation_epsilon)
        return value

    def _is_adversarial_success(
        self,
        original: Any,
        perturbed: Any,
    ) -> bool:
        """Check if perturbation caused adversarial success.

        Args:
            original: Original output
            perturbed: Perturbed output

        Returns:
            True if perturbation was successful (large output change)
        """
        # For numerical outputs, check if change is disproportionate
        if isinstance(original, (int, float)) and isinstance(perturbed, (int, float)):
            if original == 0:
                return abs(perturbed) > 0.1  # Threshold for zero baseline
            relative_change = abs(perturbed - original) / abs(original)
            # If output changed > 10x the input perturbation, it's adversarial
            return relative_change > (self.perturbation_epsilon * 10)

        # For other types, check if outputs are different
        return original != perturbed


class DistributionShiftTester(RobustnessTester):
    """Tests resilience to distribution shift in data.

    Article 15 requires resilience to "inconsistencies that may occur
    within the environment in which the system operates."

    Tests include:
    - Covariate shift (input distribution change)
    - Concept drift (relationship change)
    - Regime change (market regime changes)
    """

    def __init__(
        self,
        shift_magnitude: float = 0.5,
        regime_multipliers: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize distribution shift tester.

        Args:
            shift_magnitude: Magnitude of distribution shift
            regime_multipliers: Multipliers for regime change testing
            seed: Random seed for reproducibility
        """
        self.shift_magnitude = shift_magnitude
        self.regime_multipliers = regime_multipliers or [0.5, 1.5, 2.0, 3.0]
        self._rng = random.Random(seed)

    def run_test(
        self,
        model_fn: Callable[[Any], Any],
        test_inputs: List[Any],
        baseline_performance: Optional[float] = None,
        **kwargs: Any,
    ) -> RobustnessTestResult:
        """Run distribution shift robustness test.

        Args:
            model_fn: Model function to test
            test_inputs: Inputs from original distribution
            baseline_performance: Optional baseline metric for comparison
            **kwargs: Additional parameters

        Returns:
            RobustnessTestResult
        """
        start_time = datetime.utcnow()
        inputs_tested = 0
        failures_detected = 0

        shift_results = []

        try:
            # Test multiple shift scenarios
            for multiplier in self.regime_multipliers:
                shifted_inputs = self._shift_distribution(test_inputs, multiplier)
                scenario_failures = 0

                for shifted_input in shifted_inputs:
                    inputs_tested += 1

                    try:
                        output = model_fn(shifted_input)

                        # Check if output is valid
                        if not self._is_valid_output(output):
                            scenario_failures += 1
                            failures_detected += 1
                    except Exception as e:
                        logger.warning(
                            "Model error on shifted input (mult=%.1f): %s", multiplier, e
                        )
                        scenario_failures += 1
                        failures_detected += 1

                shift_results.append(
                    {
                        "multiplier": multiplier,
                        "failures": scenario_failures,
                        "total": len(shifted_inputs),
                        "success_rate": 1.0 - (scenario_failures / max(1, len(shifted_inputs))),
                    }
                )

        except Exception as e:
            logger.error("Distribution shift test error: %s", e)
            return RobustnessTestResult(
                test_type=RobustnessTestType.DATA_DRIFT,
                test_name="Distribution Shift Test",
                status=TestStatus.ERROR,
                details=f"Test error: {str(e)}",
                inputs_tested=inputs_tested,
                failures_detected=failures_detected,
            )

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Calculate overall score (weighted by shift magnitude)
        if shift_results:
            total_weighted = sum(r["success_rate"] * (1.0 / r["multiplier"]) for r in shift_results)
            weight_sum = sum(1.0 / r["multiplier"] for r in shift_results)
            score = total_weighted / weight_sum if weight_sum > 0 else 0.0
        else:
            score = 0.0

        # Determine status
        if score >= 0.90:
            status = TestStatus.PASSED
            severity = None
        elif score >= 0.70:
            status = TestStatus.PASSED
            severity = TestSeverity.LOW
        elif score >= 0.50:
            status = TestStatus.FAILED
            severity = TestSeverity.MEDIUM
        else:
            status = TestStatus.FAILED
            severity = TestSeverity.HIGH

        recommendations = []
        if score < 0.90:
            recommendations.append("Implement distribution drift detection and alerting")
            recommendations.append("Consider periodic model recalibration procedures")
        if score < 0.70:
            recommendations.append("Review model's assumptions about data stationarity")
            recommendations.append("Implement regime-aware model switching or adaptation")

        return RobustnessTestResult(
            test_type=RobustnessTestType.DATA_DRIFT,
            test_name="Distribution Shift Test",
            status=status,
            severity=severity,
            score=score,
            details=(
                f"Tested {len(self.regime_multipliers)} distribution shift scenarios. "
                f"Overall robustness score: {score:.1%}."
            ),
            timestamp=start_time,
            duration_ms=duration_ms,
            inputs_tested=inputs_tested,
            failures_detected=failures_detected,
            recommendations=recommendations,
            metadata={
                "shift_magnitude": self.shift_magnitude,
                "regime_multipliers": self.regime_multipliers,
                "scenario_results": shift_results,
            },
        )

    def _shift_distribution(
        self,
        inputs: List[Any],
        multiplier: float,
    ) -> List[Any]:
        """Shift input distribution.

        Args:
            inputs: Original inputs
            multiplier: Shift multiplier

        Returns:
            Shifted inputs
        """
        shifted = []
        for inp in inputs:
            if isinstance(inp, (list, tuple)):
                shifted.append(type(inp)(self._shift_value(v, multiplier) for v in inp))
            elif isinstance(inp, dict):
                shifted.append({k: self._shift_value(v, multiplier) for k, v in inp.items()})
            else:
                shifted.append(self._shift_value(inp, multiplier))
        return shifted

    def _shift_value(self, value: Any, multiplier: float) -> Any:
        """Shift a single value.

        Args:
            value: Original value
            multiplier: Shift multiplier

        Returns:
            Shifted value
        """
        if isinstance(value, (int, float)) and math.isfinite(value):
            # Apply multiplicative shift with some noise
            noise = self._rng.gauss(0, self.shift_magnitude * 0.1)
            return value * multiplier + noise
        return value

    def _is_valid_output(self, output: Any) -> bool:
        """Check if output is valid.

        Args:
            output: Model output

        Returns:
            True if output is valid
        """
        if output is None:
            return False
        if isinstance(output, (int, float)):
            return math.isfinite(output)
        return True


class FailsafeTester(RobustnessTester):
    """Tests failsafe and graceful degradation mechanisms.

    Article 15 requires systems to be resilient to "errors, faults
    or inconsistencies."

    Tests include:
    - Failsafe trigger activation
    - Graceful degradation under stress
    - Recovery from failure states
    """

    def __init__(
        self,
        timeout_ms: float = 5000.0,
        max_retries: int = 3,
    ):
        """Initialize failsafe tester.

        Args:
            timeout_ms: Timeout for operations in milliseconds
            max_retries: Maximum retry attempts
        """
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries

    def run_test(
        self,
        model_fn: Callable[[Any], Any],
        test_inputs: List[Any],
        failsafe_fn: Optional[Callable[[], Any]] = None,
        **kwargs: Any,
    ) -> RobustnessTestResult:
        """Run failsafe robustness test.

        Args:
            model_fn: Model function to test
            test_inputs: Test inputs including edge cases
            failsafe_fn: Optional failsafe function to test
            **kwargs: Additional parameters

        Returns:
            RobustnessTestResult
        """
        start_time = datetime.utcnow()
        inputs_tested = 0
        failures_detected = 0

        edge_case_results = []
        failsafe_triggered = 0
        recovery_successful = 0

        # Generate edge case inputs
        edge_cases = self._generate_edge_cases(test_inputs)

        try:
            for edge_case in edge_cases:
                inputs_tested += 1
                case_name = edge_case.get("name", f"case_{inputs_tested}")
                case_input = edge_case.get("input")

                try:
                    # Test with potential failsafe trigger
                    output = model_fn(case_input)

                    # Check if failsafe was triggered (output might indicate this)
                    if self._is_failsafe_output(output):
                        failsafe_triggered += 1
                        edge_case_results.append(
                            {
                                "case": case_name,
                                "status": "failsafe_triggered",
                                "output": str(output)[:100],
                            }
                        )
                    else:
                        edge_case_results.append(
                            {
                                "case": case_name,
                                "status": "handled",
                                "output": str(output)[:100],
                            }
                        )

                except Exception as e:
                    # Model should handle gracefully or trigger failsafe
                    failures_detected += 1
                    edge_case_results.append(
                        {
                            "case": case_name,
                            "status": "exception",
                            "error": str(e)[:100],
                        }
                    )

                    # Test recovery if failsafe_fn provided
                    if failsafe_fn:
                        try:
                            failsafe_fn()
                            recovery_successful += 1
                        except Exception as recovery_error:
                            logger.warning(
                                "Recovery failed for case %s: %s", case_name, recovery_error
                            )

        except Exception as e:
            logger.error("Failsafe test error: %s", e)
            return RobustnessTestResult(
                test_type=RobustnessTestType.FAILSAFE_TRIGGER,
                test_name="Failsafe Mechanism Test",
                status=TestStatus.ERROR,
                details=f"Test error: {str(e)}",
                inputs_tested=inputs_tested,
                failures_detected=failures_detected,
            )

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Calculate score based on graceful handling
        handled_gracefully = inputs_tested - failures_detected
        if inputs_tested > 0:
            score = handled_gracefully / inputs_tested
        else:
            score = 0.0

        # Adjust score based on recovery success
        if failures_detected > 0 and failsafe_fn:
            recovery_rate = recovery_successful / failures_detected
            score = (score + recovery_rate) / 2

        # Determine status
        if score >= 0.95:
            status = TestStatus.PASSED
            severity = None
        elif score >= 0.80:
            status = TestStatus.PASSED
            severity = TestSeverity.LOW
        elif score >= 0.60:
            status = TestStatus.FAILED
            severity = TestSeverity.MEDIUM
        else:
            status = TestStatus.FAILED
            severity = TestSeverity.HIGH

        recommendations = []
        if failures_detected > 0:
            recommendations.append("Implement comprehensive exception handling")
        if failsafe_triggered == 0 and failures_detected > 0:
            recommendations.append("Add failsafe triggers for edge case inputs")
        if recovery_successful < failures_detected:
            recommendations.append("Improve recovery mechanisms for failure states")

        return RobustnessTestResult(
            test_type=RobustnessTestType.FAILSAFE_TRIGGER,
            test_name="Failsafe Mechanism Test",
            status=status,
            severity=severity,
            score=score,
            details=(
                f"Tested {inputs_tested} edge cases. "
                f"Failsafe triggered {failsafe_triggered} times. "
                f"Unhandled failures: {failures_detected}. "
                f"Recovery successful: {recovery_successful}/{failures_detected}."
            ),
            timestamp=start_time,
            duration_ms=duration_ms,
            inputs_tested=inputs_tested,
            failures_detected=failures_detected,
            recommendations=recommendations,
            metadata={
                "timeout_ms": self.timeout_ms,
                "max_retries": self.max_retries,
                "edge_case_results": edge_case_results,
                "failsafe_triggered": failsafe_triggered,
                "recovery_successful": recovery_successful,
            },
        )

    def _generate_edge_cases(self, base_inputs: List[Any]) -> List[Dict[str, Any]]:
        """Generate edge case inputs for testing.

        Args:
            base_inputs: Original inputs to derive edge cases from

        Returns:
            List of edge case configurations
        """
        edge_cases = []

        # Null/None cases
        edge_cases.append({"name": "null_input", "input": None})

        # Empty cases
        edge_cases.append({"name": "empty_list", "input": []})
        edge_cases.append({"name": "empty_dict", "input": {}})

        # Extreme numerical values
        edge_cases.append({"name": "inf_positive", "input": float("inf")})
        edge_cases.append({"name": "inf_negative", "input": float("-inf")})
        edge_cases.append({"name": "nan", "input": float("nan")})
        edge_cases.append({"name": "zero", "input": 0.0})
        edge_cases.append({"name": "very_large", "input": 1e308})
        edge_cases.append({"name": "very_small", "input": 1e-308})

        # Based on input structure
        for i, base in enumerate(base_inputs[:5]):  # Sample first 5
            if isinstance(base, (list, tuple)):
                # Create corrupted version
                edge_cases.append(
                    {
                        "name": f"corrupted_list_{i}",
                        "input": [float("nan")] * len(base) if base else [float("nan")],
                    }
                )
            elif isinstance(base, dict):
                # Create partial version
                edge_cases.append(
                    {
                        "name": f"partial_dict_{i}",
                        "input": {k: None for k in list(base.keys())[:1]} if base else {},
                    }
                )
            elif isinstance(base, (int, float)):
                # Create extreme versions
                edge_cases.append(
                    {
                        "name": f"extreme_negative_{i}",
                        "input": -abs(base) * 1000 if base != 0 else -1000,
                    }
                )

        return edge_cases

    def _is_failsafe_output(self, output: Any) -> bool:
        """Check if output indicates failsafe activation.

        Args:
            output: Model output

        Returns:
            True if failsafe appears to be triggered
        """
        # Check common failsafe indicators
        if output is None:
            return True
        if isinstance(output, dict):
            return output.get("failsafe", False) or output.get("error", False)
        if isinstance(output, (int, float)):
            # Zero output might indicate failsafe
            return output == 0.0
        return False


@dataclass
class RobustnessTestSuite:
    """Complete robustness test suite per Article 15.

    Combines all robustness testers into a comprehensive suite.

    Attributes:
        suite_id: Unique identifier for this suite
        suite_name: Human-readable suite name
        adversarial_tester: Adversarial robustness tester
        distribution_shift_tester: Distribution shift tester
        failsafe_tester: Failsafe mechanism tester
        results: Test results from last run
        created_at: When suite was created
    """

    suite_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    suite_name: str = "EU AI Act Article 15 Robustness Test Suite"
    adversarial_tester: AdversarialTester = field(default_factory=AdversarialTester)
    distribution_shift_tester: DistributionShiftTester = field(
        default_factory=DistributionShiftTester
    )
    failsafe_tester: FailsafeTester = field(default_factory=FailsafeTester)
    results: List[RobustnessTestResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def run_all_tests(
        self,
        model_fn: Callable[[Any], Any],
        test_inputs: List[Any],
        failsafe_fn: Optional[Callable[[], Any]] = None,
        **kwargs: Any,
    ) -> List[RobustnessTestResult]:
        """Run all robustness tests.

        Args:
            model_fn: Model function to test
            test_inputs: Test inputs
            failsafe_fn: Optional failsafe function
            **kwargs: Additional test parameters

        Returns:
            List of all test results
        """
        self.results = []

        logger.info("Running adversarial tests...")
        result = self.adversarial_tester.run_test(model_fn, test_inputs, **kwargs)
        self.results.append(result)

        logger.info("Running distribution shift tests...")
        result = self.distribution_shift_tester.run_test(model_fn, test_inputs, **kwargs)
        self.results.append(result)

        logger.info("Running failsafe tests...")
        result = self.failsafe_tester.run_test(
            model_fn, test_inputs, failsafe_fn=failsafe_fn, **kwargs
        )
        self.results.append(result)

        return self.results

    def get_overall_score(self) -> float:
        """Get overall robustness score.

        Returns:
            Weighted average of all test scores
        """
        if not self.results:
            return 0.0

        # Weight by severity of test type
        weights = {
            RobustnessTestType.ADVERSARIAL_PERTURBATION: 1.0,
            RobustnessTestType.DATA_DRIFT: 0.8,
            RobustnessTestType.FAILSAFE_TRIGGER: 0.9,
        }

        total_weighted = 0.0
        total_weight = 0.0

        for result in self.results:
            weight = weights.get(result.test_type, 0.5)
            total_weighted += result.score * weight
            total_weight += weight

        return total_weighted / total_weight if total_weight > 0 else 0.0

    def is_compliant(self, minimum_score: float = 0.8) -> bool:
        """Check if system meets minimum robustness requirements.

        Args:
            minimum_score: Minimum acceptable score

        Returns:
            True if compliant
        """
        return self.get_overall_score() >= minimum_score

    def generate_compliance_report(self) -> str:
        """Generate Article 15 compliance report.

        Returns:
            Formatted compliance report
        """
        lines = [
            "=" * 70,
            "ARTICLE 15 ROBUSTNESS COMPLIANCE REPORT",
            "EU AI Act (Regulation (EU) 2024/1689)",
            "=" * 70,
            "",
            f"Suite ID: {self.suite_id}",
            f"Suite Name: {self.suite_name}",
            f"Generated: {datetime.utcnow().isoformat()}",
            "",
            f"Overall Robustness Score: {self.get_overall_score():.1%}",
            f"Compliance Status: {'✅ COMPLIANT' if self.is_compliant() else '❌ NON-COMPLIANT'}",
            "",
            "-" * 70,
            "TEST RESULTS",
            "-" * 70,
            "",
        ]

        for result in self.results:
            lines.append(result.get_compliance_text())
            lines.append("")

        # Summary
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)

        lines.extend(
            [
                "-" * 70,
                "SUMMARY",
                "-" * 70,
                "",
                f"Tests Passed: {passed}/{len(self.results)}",
                f"Tests Failed: {failed}/{len(self.results)}",
                f"Test Errors: {errors}/{len(self.results)}",
                "",
            ]
        )

        # Aggregate recommendations
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)

        if all_recommendations:
            lines.extend(
                [
                    "-" * 70,
                    "RECOMMENDATIONS",
                    "-" * 70,
                    "",
                ]
            )
            for i, rec in enumerate(set(all_recommendations), 1):
                lines.append(f"  {i}. {rec}")

        lines.extend(
            [
                "",
                "=" * 70,
                "END OF REPORT",
                "=" * 70,
            ]
        )

        return "\n".join(lines)

    def export_for_audit(self) -> Dict[str, Any]:
        """Export test results for regulatory audit.

        Returns:
            Complete audit export dictionary
        """
        return {
            "export_timestamp": datetime.utcnow().isoformat(),
            "suite_id": self.suite_id,
            "suite_name": self.suite_name,
            "created_at": self.created_at.isoformat(),
            "overall_score": self.get_overall_score(),
            "is_compliant": self.is_compliant(),
            "results": [
                {
                    "test_id": r.test_id,
                    "test_type": r.test_type.name,
                    "test_name": r.test_name,
                    "status": r.status.value,
                    "severity": r.severity.value if r.severity else None,
                    "score": r.score,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat(),
                    "duration_ms": r.duration_ms,
                    "inputs_tested": r.inputs_tested,
                    "failures_detected": r.failures_detected,
                    "recommendations": r.recommendations,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }


def create_robustness_test_suite(
    perturbation_epsilon: float = 0.01,
    num_perturbations: int = 10,
    shift_magnitude: float = 0.5,
    regime_multipliers: Optional[List[float]] = None,
    timeout_ms: float = 5000.0,
    seed: Optional[int] = None,
) -> RobustnessTestSuite:
    """Factory function to create a RobustnessTestSuite.

    Args:
        perturbation_epsilon: Adversarial perturbation size
        num_perturbations: Number of perturbations per input
        shift_magnitude: Distribution shift magnitude
        regime_multipliers: Regime change multipliers
        timeout_ms: Failsafe timeout in milliseconds
        seed: Random seed for reproducibility

    Returns:
        Configured RobustnessTestSuite
    """
    return RobustnessTestSuite(
        adversarial_tester=AdversarialTester(
            perturbation_epsilon=perturbation_epsilon,
            num_perturbations=num_perturbations,
            seed=seed,
        ),
        distribution_shift_tester=DistributionShiftTester(
            shift_magnitude=shift_magnitude,
            regime_multipliers=regime_multipliers,
            seed=seed,
        ),
        failsafe_tester=FailsafeTester(
            timeout_ms=timeout_ms,
        ),
    )


__all__ = [
    "RobustnessTestType",
    "TestSeverity",
    "TestStatus",
    "RobustnessTestResult",
    "RobustnessTester",
    "AdversarialTester",
    "DistributionShiftTester",
    "FailsafeTester",
    "RobustnessTestSuite",
    "create_robustness_test_suite",
]

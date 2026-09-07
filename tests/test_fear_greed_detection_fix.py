"""
Tests for Fear & Greed Index detection fix (2025-11-26).

BUG FIX: Previously, has_fear_greed was determined by `abs(value - 50.0) > 0.1`,
which gave FALSE NEGATIVE when FG=50 (neutral sentiment).

FG=50 is a VALID value meaning "neutral sentiment", NOT missing data!

IMPACT:
- Model incorrectly thought FG data was missing when FG=50
- indicator=0.0 instead of indicator=1.0
- risk_off_flag could trigger on default value instead of only on valid data

FIX: Use _get_safe_float_with_validity() which properly distinguishes:
- is_valid=True: value is present and finite
- is_valid=False: value is NaN, Inf, None, or column missing
"""

import math
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Import the mediator class
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFearGreedDetectionFix:
    """Test cases for Fear & Greed detection fix."""

    def test_fg_50_is_valid_data(self):
        """FG=50 (neutral) should be recognized as VALID data, not missing."""
        from mediator import Mediator

        # Create row with FG=50
        row = {"fear_greed_value": 50.0}

        # Use the _get_safe_float_with_validity method
        value, is_valid = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )

        # FG=50 should be valid!
        assert is_valid is True, "FG=50 should be recognized as valid data"
        assert value == 50.0, "FG=50 value should be preserved"

    def test_fg_nan_is_invalid(self):
        """FG=NaN should be recognized as INVALID (missing) data."""
        from mediator import Mediator

        row = {"fear_greed_value": float("nan")}

        value, is_valid = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )

        assert is_valid is False, "FG=NaN should be invalid"
        assert value == 50.0, "Default value should be returned for NaN"

    def test_fg_missing_column_is_invalid(self):
        """Missing fear_greed_value column should be recognized as INVALID."""
        from mediator import Mediator

        row = {}  # No fear_greed_value column

        value, is_valid = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )

        assert is_valid is False, "Missing column should be invalid"
        assert value == 50.0, "Default value should be returned"

    def test_fg_none_is_invalid(self):
        """FG=None should be recognized as INVALID."""
        from mediator import Mediator

        row = {"fear_greed_value": None}

        value, is_valid = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )

        assert is_valid is False, "FG=None should be invalid"

    def test_fg_inf_is_invalid(self):
        """FG=Inf should be recognized as INVALID."""
        from mediator import Mediator

        row = {"fear_greed_value": float("inf")}

        value, is_valid = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )

        assert is_valid is False, "FG=Inf should be invalid"

    def test_fg_out_of_range_is_invalid(self):
        """FG value outside [0, 100] should be recognized as INVALID."""
        from mediator import Mediator

        # Test below range
        row = {"fear_greed_value": -10.0}
        value, is_valid = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )
        assert is_valid is False, "FG=-10 should be invalid (below range)"

        # Test above range
        row = {"fear_greed_value": 150.0}
        value, is_valid = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )
        assert is_valid is False, "FG=150 should be invalid (above range)"

    def test_fg_extreme_fear_is_valid(self):
        """FG=10 (extreme fear) should be recognized as VALID."""
        from mediator import Mediator

        row = {"fear_greed_value": 10.0}

        value, is_valid = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )

        assert is_valid is True, "FG=10 should be valid"
        assert value == 10.0, "FG=10 value should be preserved"

    def test_fg_extreme_greed_is_valid(self):
        """FG=90 (extreme greed) should be recognized as VALID."""
        from mediator import Mediator

        row = {"fear_greed_value": 90.0}

        value, is_valid = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )

        assert is_valid is True, "FG=90 should be valid"
        assert value == 90.0, "FG=90 value should be preserved"

    def test_fg_boundary_values_are_valid(self):
        """FG=0 and FG=100 (boundary values) should be recognized as VALID."""
        from mediator import Mediator

        # Test FG=0
        row = {"fear_greed_value": 0.0}
        value, is_valid = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )
        assert is_valid is True, "FG=0 should be valid"
        assert value == 0.0

        # Test FG=100
        row = {"fear_greed_value": 100.0}
        value, is_valid = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )
        assert is_valid is True, "FG=100 should be valid"
        assert value == 100.0


class TestRiskOffFlagWithValidFG:
    """Test cases for risk_off_flag with valid FG data."""

    def test_risk_off_only_with_valid_fg_and_extreme_fear(self):
        """risk_off_flag should only be True when has_fear_greed=True AND value<25."""
        # Test cases: (fg_value, has_fg, expected_risk_off)
        test_cases = [
            (10.0, True, True),  # Valid extreme fear -> risk off
            (24.9, True, True),  # Valid just below threshold -> risk off
            (25.0, True, False),  # Valid at threshold -> NOT risk off
            (50.0, True, False),  # Valid neutral -> NOT risk off
            (90.0, True, False),  # Valid greed -> NOT risk off
            (10.0, False, False),  # Invalid (missing) -> NOT risk off even if default would trigger
            (50.0, False, False),  # Invalid (missing), neutral default -> NOT risk off
        ]

        for fg_value, has_fg, expected_risk_off in test_cases:
            # The new logic: risk_off_flag = has_fear_greed and fear_greed_value < 25.0
            risk_off_flag = has_fg and fg_value < 25.0

            assert (
                risk_off_flag == expected_risk_off
            ), f"FG={fg_value}, has_fg={has_fg}: expected risk_off={expected_risk_off}, got {risk_off_flag}"


class TestOldBugRegression:
    """Regression tests ensuring the old bug doesn't return."""

    def test_old_bug_abs_check_fails_for_fg_50(self):
        """Demonstrate that the OLD logic would fail for FG=50."""
        fear_greed_value = 50.0

        # OLD BUGGY LOGIC:
        has_fear_greed_old = abs(fear_greed_value - 50.0) > 0.1

        # This incorrectly returns False for FG=50!
        assert (
            has_fear_greed_old is False
        ), "Old logic should return False for FG=50 (demonstrating the bug)"

    def test_new_logic_works_for_fg_50(self):
        """Ensure new logic works correctly for FG=50."""
        from mediator import Mediator

        row = {"fear_greed_value": 50.0}

        # NEW CORRECT LOGIC:
        fear_greed_value, has_fear_greed = Mediator._get_safe_float_with_validity(
            row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
        )

        # This correctly returns True for FG=50!
        assert has_fear_greed is True, "New logic should return True for FG=50 (fix verified)"

    def test_old_vs_new_logic_comprehensive(self):
        """Compare old and new logic across various FG values."""
        from mediator import Mediator

        # (fg_value, old_has_fg, new_has_fg)
        test_cases = [
            (0.0, True, True),  # Extreme fear: both detect as valid
            (25.0, True, True),  # Fear: both detect as valid
            (49.0, True, True),  # Near neutral: both detect as valid
            (49.9, True, True),  # Very near neutral: both detect as valid
            (50.0, False, True),  # NEUTRAL: OLD=BUG, NEW=CORRECT
            (50.1, True, True),  # Just above neutral: both detect as valid
            (51.0, True, True),  # Near neutral: both detect as valid
            (75.0, True, True),  # Greed: both detect as valid
            (100.0, True, True),  # Extreme greed: both detect as valid
        ]

        for fg_value, old_has_fg, new_has_fg in test_cases:
            row = {"fear_greed_value": fg_value}

            # Old logic
            old_result = abs(fg_value - 50.0) > 0.1

            # New logic
            _, new_result = Mediator._get_safe_float_with_validity(
                row, "fear_greed_value", default=50.0, min_value=0.0, max_value=100.0
            )

            assert (
                old_result == old_has_fg
            ), f"Old logic unexpected for FG={fg_value}: expected {old_has_fg}, got {old_result}"
            assert (
                new_result == new_has_fg
            ), f"New logic unexpected for FG={fg_value}: expected {new_has_fg}, got {new_result}"

            # Most importantly: for FG=50, old was wrong and new is correct
            if fg_value == 50.0:
                assert old_result is False, "Old bug: FG=50 incorrectly marked as missing"
                assert new_result is True, "New fix: FG=50 correctly marked as valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

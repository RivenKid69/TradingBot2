# -*- coding: utf-8 -*-
"""
Tests for CCEA Phrase Guard guardrail (WI-LEGAL-01).

Verifies that documentation does not contain phrases contradicting CCEA architecture.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ccea.guardrails.phrase_guard import (
    PhraseGuard,
    PhraseGuardResult,
    PhraseViolation,
    ProhibitedPhrase,
    PhraseCategory,
    ViolationSeverity,
    PROHIBITED_PHRASES,
)


class TestProhibitedPhrases:
    """Test prohibited phrase definitions."""

    def test_prohibited_phrases_exist(self):
        """Verify prohibited phrases registry is populated."""
        assert len(PROHIBITED_PHRASES) > 0, "No prohibited phrases defined"

    def test_all_categories_covered(self):
        """Verify all categories have at least one phrase."""
        categories = {p.category for p in PROHIBITED_PHRASES}
        expected = {
            PhraseCategory.CREDENTIAL_STORAGE,
            PhraseCategory.ORDER_EXECUTION,
            PhraseCategory.BROKERAGE_CLAIMS,
            PhraseCategory.CUSTODY_CLAIMS,
            PhraseCategory.AUTO_EXECUTION,
        }
        assert categories == expected, f"Missing categories: {expected - categories}"

    def test_all_severities_used(self):
        """Verify severity levels are used appropriately."""
        severities = {p.severity for p in PROHIBITED_PHRASES}
        # At minimum CRITICAL and HIGH should be present
        assert ViolationSeverity.CRITICAL in severities
        assert ViolationSeverity.HIGH in severities or ViolationSeverity.MEDIUM in severities

    def test_phrases_have_fix_suggestions(self):
        """Verify all phrases have fix suggestions."""
        for phrase in PROHIBITED_PHRASES:
            assert phrase.fix_suggestion, f"No fix suggestion for: {phrase.pattern}"


class TestPhraseGuardDetection:
    """Test phrase detection capabilities."""

    @pytest.fixture
    def guard(self):
        """Create a PhraseGuard instance."""
        return PhraseGuard(root_path=Path("/tmp/fake_root"))

    def test_detects_credential_storage_claim(self, guard):
        """Test detection of credential storage claims."""
        content = "We store your API keys securely in our cloud."
        violations = guard.check_content(content, "test.md")
        assert len(violations) > 0
        assert any(v.phrase.category == PhraseCategory.CREDENTIAL_STORAGE for v in violations)

    def test_detects_order_execution_claim(self, guard):
        """Test detection of order execution claims."""
        content = "We execute trades on behalf of our clients."
        violations = guard.check_content(content, "test.md")
        assert len(violations) > 0
        assert any(v.phrase.category == PhraseCategory.ORDER_EXECUTION for v in violations)

    def test_detects_cloud_execution_claim(self, guard):
        """Test detection of cloud execution claims."""
        content = "The cloud executes orders automatically."
        violations = guard.check_content(content, "test.md")
        assert len(violations) > 0
        assert any(v.phrase.category == PhraseCategory.ORDER_EXECUTION for v in violations)

    def test_detects_brokerage_claim(self, guard):
        """Test detection of brokerage claims."""
        content = "We are a registered broker in the United States."
        violations = guard.check_content(content, "test.md")
        assert len(violations) > 0
        assert any(v.phrase.category == PhraseCategory.BROKERAGE_CLAIMS for v in violations)

    def test_detects_custody_claim(self, guard):
        """Test detection of custody claims."""
        content = "We hold your assets safely."
        violations = guard.check_content(content, "test.md")
        assert len(violations) > 0
        assert any(v.phrase.category == PhraseCategory.CUSTODY_CLAIMS for v in violations)

    def test_detects_auto_execution_claim(self, guard):
        """Test detection of auto-execution claims."""
        content = "Cloud auto-executes your trading strategies."
        violations = guard.check_content(content, "test.md")
        assert len(violations) > 0
        assert any(v.phrase.category == PhraseCategory.AUTO_EXECUTION for v in violations)

    def test_no_false_positive_on_ccea_compliant(self, guard):
        """Test that CCEA-compliant language doesn't trigger violations."""
        content = """
        Your local Agent executes trades based on your strategies.
        API keys are stored locally in your Agent's encrypted vault.
        Cloud sends lifecycle commands only.
        We are a software vendor, not a broker.
        """
        violations = guard.check_content(content, "test.md")
        critical_high = [v for v in violations if v.phrase.severity in {ViolationSeverity.CRITICAL, ViolationSeverity.HIGH}]
        assert len(critical_high) == 0, f"False positives: {critical_high}"

    def test_skips_code_blocks(self, guard):
        """Test that code blocks are skipped."""
        content = """
        Normal text here.
        ```python
        # We store your API keys here (in code block - should be skipped)
        api_keys = store_keys()
        ```
        More text.
        """
        violations = guard.check_content(content, "test.md")
        # Should not detect the code block content
        code_violations = [v for v in violations if "in code block" in v.line_content]
        assert len(code_violations) == 0

    def test_case_insensitive_detection(self, guard):
        """Test case-insensitive detection."""
        content = "WE STORE YOUR API KEYS"
        violations = guard.check_content(content, "test.md")
        assert len(violations) > 0

    def test_violation_has_line_number(self, guard):
        """Test that violations include line numbers."""
        content = "Line 1\nLine 2\nWe store your API keys\nLine 4"
        violations = guard.check_content(content, "test.md")
        assert len(violations) > 0
        assert violations[0].line_number == 3


class TestPhraseGuardResult:
    """Test PhraseGuardResult functionality."""

    def test_passed_with_no_violations(self):
        """Test passed property with no violations."""
        result = PhraseGuardResult()
        assert result.passed is True

    def test_passed_with_only_warnings(self):
        """Test passed property with only warnings."""
        result = PhraseGuardResult(violations=[
            PhraseViolation(
                phrase=ProhibitedPhrase(
                    pattern="test",
                    category=PhraseCategory.CREDENTIAL_STORAGE,
                    severity=ViolationSeverity.WARNING,
                    description="Test",
                    fix_suggestion="Test",
                ),
                file_path="test.md",
                line_number=1,
                matched_text="test",
                line_content="test content",
            )
        ])
        assert result.passed is True

    def test_failed_with_critical_violation(self):
        """Test passed property with critical violation."""
        result = PhraseGuardResult(violations=[
            PhraseViolation(
                phrase=ProhibitedPhrase(
                    pattern="test",
                    category=PhraseCategory.CREDENTIAL_STORAGE,
                    severity=ViolationSeverity.CRITICAL,
                    description="Test",
                    fix_suggestion="Test",
                ),
                file_path="test.md",
                line_number=1,
                matched_text="test",
                line_content="test content",
            )
        ])
        assert result.passed is False

    def test_failed_with_high_violation(self):
        """Test passed property with high violation."""
        result = PhraseGuardResult(violations=[
            PhraseViolation(
                phrase=ProhibitedPhrase(
                    pattern="test",
                    category=PhraseCategory.CREDENTIAL_STORAGE,
                    severity=ViolationSeverity.HIGH,
                    description="Test",
                    fix_suggestion="Test",
                ),
                file_path="test.md",
                line_number=1,
                matched_text="test",
                line_content="test content",
            )
        ])
        assert result.passed is False

    def test_counts(self):
        """Test violation counts."""
        violations = [
            PhraseViolation(
                phrase=ProhibitedPhrase(
                    pattern="test",
                    category=PhraseCategory.CREDENTIAL_STORAGE,
                    severity=severity,
                    description="Test",
                    fix_suggestion="Test",
                ),
                file_path="test.md",
                line_number=1,
                matched_text="test",
                line_content="test content",
            )
            for severity in [ViolationSeverity.CRITICAL, ViolationSeverity.CRITICAL,
                           ViolationSeverity.HIGH, ViolationSeverity.MEDIUM]
        ]
        result = PhraseGuardResult(violations=violations)
        assert result.critical_count == 2
        assert result.high_count == 1
        assert result.medium_count == 1
        assert result.warning_count == 0


class TestPhraseGuardIntegration:
    """Integration tests for phrase guard on real docs."""

    @pytest.fixture
    def project_root(self):
        """Get project root path."""
        return Path(__file__).parent.parent.parent.parent

    def test_tos_compliant(self, project_root):
        """Test that Terms of Service is CCEA-compliant."""
        tos_path = project_root / "docs" / "legal" / "TERMS_OF_SERVICE.md"
        if not tos_path.exists():
            pytest.skip("TERMS_OF_SERVICE.md not found")

        guard = PhraseGuard(root_path=project_root)
        violations = guard.check_file(tos_path)
        critical_violations = [v for v in violations if v.phrase.severity == ViolationSeverity.CRITICAL]
        assert len(critical_violations) == 0, f"Critical violations in ToS: {critical_violations}"

    def test_privacy_policy_compliant(self, project_root):
        """Test that Privacy Policy is CCEA-compliant."""
        pp_path = project_root / "docs" / "legal" / "PRIVACY_POLICY.md"
        if not pp_path.exists():
            pytest.skip("PRIVACY_POLICY.md not found")

        guard = PhraseGuard(root_path=project_root)
        violations = guard.check_file(pp_path)
        critical_violations = [v for v in violations if v.phrase.severity == ViolationSeverity.CRITICAL]
        assert len(critical_violations) == 0, f"Critical violations in Privacy Policy: {critical_violations}"

    def test_dpa_template_compliant(self, project_root):
        """Test that DPA Template is CCEA-compliant."""
        dpa_path = project_root / "docs" / "legal" / "DPA_TEMPLATE.md"
        if not dpa_path.exists():
            pytest.skip("DPA_TEMPLATE.md not found")

        guard = PhraseGuard(root_path=project_root)
        violations = guard.check_file(dpa_path)
        critical_violations = [v for v in violations if v.phrase.severity == ViolationSeverity.CRITICAL]
        assert len(critical_violations) == 0, f"Critical violations in DPA: {critical_violations}"

    def test_aup_compliant(self, project_root):
        """Test that AUP is CCEA-compliant."""
        aup_path = project_root / "docs" / "legal" / "AUP.md"
        if not aup_path.exists():
            pytest.skip("AUP.md not found")

        guard = PhraseGuard(root_path=project_root)
        violations = guard.check_file(aup_path)
        critical_violations = [v for v in violations if v.phrase.severity == ViolationSeverity.CRITICAL]
        assert len(critical_violations) == 0, f"Critical violations in AUP: {critical_violations}"

    def test_readme_compliant(self, project_root):
        """Test that README is CCEA-compliant."""
        readme_path = project_root / "README.md"
        if not readme_path.exists():
            pytest.skip("README.md not found")

        guard = PhraseGuard(root_path=project_root)
        violations = guard.check_file(readme_path)
        critical_violations = [v for v in violations if v.phrase.severity == ViolationSeverity.CRITICAL]
        assert len(critical_violations) == 0, f"Critical violations in README: {critical_violations}"

    def test_full_docs_scan(self, project_root):
        """Test full documentation scan passes."""
        guard = PhraseGuard(root_path=project_root)
        result = guard.run()

        # Report stats
        print(f"\nPhrase Guard Results:")
        print(f"  Files checked: {result.files_checked}")
        print(f"  Lines checked: {result.total_lines_checked}")
        print(f"  Critical: {result.critical_count}")
        print(f"  High: {result.high_count}")
        print(f"  Medium: {result.medium_count}")
        print(f"  Warning: {result.warning_count}")

        assert result.passed, f"Phrase guard failed with {result.critical_count} critical, {result.high_count} high violations"


class TestViolationFormatting:
    """Test violation string formatting."""

    def test_violation_str_format(self):
        """Test violation string representation."""
        violation = PhraseViolation(
            phrase=ProhibitedPhrase(
                pattern="test",
                category=PhraseCategory.CREDENTIAL_STORAGE,
                severity=ViolationSeverity.CRITICAL,
                description="Claims Cloud stores API keys",
                fix_suggestion="Change to local Agent vault",
            ),
            file_path="docs/legal/TERMS.md",
            line_number=42,
            matched_text="we store your API keys",
            line_content="  we store your API keys securely  ",
        )
        s = str(violation)
        assert "CRITICAL" in s
        assert "docs/legal/TERMS.md:42" in s
        assert "credential_storage" in s
        assert "we store your API keys" in s
        assert "Change to local Agent vault" in s

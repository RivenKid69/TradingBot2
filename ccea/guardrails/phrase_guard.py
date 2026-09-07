# -*- coding: utf-8 -*-
"""
CCEA Phrase Guard - Documentation Compliance Guardrail.

Ensures legal/marketing documents do not contain phrases that contradict
the CCEA (Cloud-Controlled Execution Architecture) security boundary.

Design Doc Requirement (WI-LEGAL-01):
    "Add phrase-guard checks (ban 'we trade for you', 'cloud auto-execution',
    'keys stored [in cloud]')."

This guardrail ensures:
1. No claims that Cloud stores/handles broker API keys
2. No claims that Cloud/Platform executes trades on behalf of users
3. No claims of brokerage/custody services
4. No suggestions of automated cloud execution of orders

SECURITY: Violations create legal/regulatory risk and misrepresent the architecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Final, FrozenSet, List, Optional, Pattern, Set, Tuple
import sys


# ============================================================================
# Prohibited Phrase Definitions
# ============================================================================


class PhraseCategory(str, Enum):
    """Category of prohibited phrase."""

    CREDENTIAL_STORAGE = "credential_storage"  # Claims we store API keys
    ORDER_EXECUTION = "order_execution"  # Claims we execute orders
    BROKERAGE_CLAIMS = "brokerage_claims"  # Claims we are a broker
    CUSTODY_CLAIMS = "custody_claims"  # Claims we custody assets
    AUTO_EXECUTION = "auto_execution"  # Claims of cloud auto-trading


class ViolationSeverity(str, Enum):
    """Severity of phrase guard violation."""

    CRITICAL = "critical"  # Direct contradiction of CCEA
    HIGH = "high"  # Strong implication contradicting CCEA
    MEDIUM = "medium"  # Ambiguous phrasing that could mislead
    WARNING = "warning"  # Phrasing that should be reviewed


@dataclass
class ProhibitedPhrase:
    """A prohibited phrase with its metadata."""

    pattern: str  # Regex pattern
    category: PhraseCategory
    severity: ViolationSeverity
    description: str
    fix_suggestion: str
    case_sensitive: bool = False


# ============================================================================
# Prohibited Phrases Registry
# ============================================================================

PROHIBITED_PHRASES: Final[List[ProhibitedPhrase]] = [
    # --- Credential Storage Claims ---
    ProhibitedPhrase(
        pattern=r"(?:we|platform|cloud|system)\s+(?:store|stores|hold|holds|keep|keeps)\s+(?:your\s+)?(?:api\s+)?keys?",
        category=PhraseCategory.CREDENTIAL_STORAGE,
        severity=ViolationSeverity.CRITICAL,
        description="Claims Cloud stores API keys",
        fix_suggestion="Change to: 'API keys are stored locally in your Agent's encrypted vault'",
    ),
    ProhibitedPhrase(
        pattern=r"(?:your\s+)?(?:api\s+)?(?:keys?|credentials?)\s+(?:are\s+)?stored\s+(?:in\s+)?(?:our|the)\s+(?:cloud|platform|system|servers?)",
        category=PhraseCategory.CREDENTIAL_STORAGE,
        severity=ViolationSeverity.CRITICAL,
        description="Claims credentials stored in cloud",
        fix_suggestion="Change to: 'Credentials are stored locally in your Agent vault'",
    ),
    ProhibitedPhrase(
        pattern=r"keys?\s+stored",
        category=PhraseCategory.CREDENTIAL_STORAGE,
        severity=ViolationSeverity.MEDIUM,
        description="Ambiguous 'keys stored' phrase - may imply cloud storage",
        fix_suggestion="Clarify: 'keys stored locally in Agent vault' or 'keys stored on your hardware'",
    ),
    ProhibitedPhrase(
        pattern=r"encrypted\s+(?:api\s+)?(?:keys?|credentials?)\s+(?:in|on)\s+(?:our|the)\s+(?:cloud|servers?|platform)",
        category=PhraseCategory.CREDENTIAL_STORAGE,
        severity=ViolationSeverity.CRITICAL,
        description="Claims encrypted keys in cloud",
        fix_suggestion="Remove cloud storage claim; credentials are local-only under CCEA",
    ),
    # --- Order Execution Claims ---
    ProhibitedPhrase(
        pattern=r"we\s+(?:trade|execute|place|submit|send)\s+(?:trades?|orders?)\s+(?:for|on\s+behalf\s+of)",
        category=PhraseCategory.ORDER_EXECUTION,
        severity=ViolationSeverity.CRITICAL,
        description="Claims we execute trades for users",
        fix_suggestion="Change to: 'Your local Agent executes trades based on your strategies'",
    ),
    ProhibitedPhrase(
        pattern=r"(?:cloud|platform|system)\s+(?:executes?|places?|submits?|sends?)\s+(?:trades?|orders?)",
        category=PhraseCategory.ORDER_EXECUTION,
        severity=ViolationSeverity.CRITICAL,
        description="Claims cloud executes orders",
        fix_suggestion="Change to: 'Agent (running locally) executes orders'",
    ),
    ProhibitedPhrase(
        pattern=r"(?:execute|place|submit|send)\s+(?:trades?|orders?)\s+on\s+(?:your|client['']?s?)\s+behalf",
        category=PhraseCategory.ORDER_EXECUTION,
        severity=ViolationSeverity.CRITICAL,
        description="Claims order execution on behalf of users",
        fix_suggestion="Clarify that execution happens in user's local Agent, not our cloud",
    ),
    ProhibitedPhrase(
        pattern=r"order\s+execution\s+(?:on|for)\s+(?:your|user['']?s?|client['']?s?)\s+behalf",
        category=PhraseCategory.ORDER_EXECUTION,
        severity=ViolationSeverity.CRITICAL,
        description="Claims order execution service",
        fix_suggestion="Change to: 'Order execution happens locally in your Agent'",
    ),
    # --- Brokerage Claims ---
    ProhibitedPhrase(
        pattern=r"we\s+are\s+(?:a\s+)?(?:registered\s+)?broker",
        category=PhraseCategory.BROKERAGE_CLAIMS,
        severity=ViolationSeverity.CRITICAL,
        description="Claims brokerage registration/status",
        fix_suggestion="We are a SOFTWARE VENDOR, not a broker",
    ),
    ProhibitedPhrase(
        pattern=r"(?:our|the)\s+(?:company|platform)\s+(?:is\s+)?(?:a\s+)?broker",
        category=PhraseCategory.BROKERAGE_CLAIMS,
        severity=ViolationSeverity.CRITICAL,
        description="Claims to be a broker",
        fix_suggestion="Clarify: 'We provide software tools; we are not a broker'",
    ),
    ProhibitedPhrase(
        pattern=r"sec[\/\s]?finra\s+registered\s+broker",
        category=PhraseCategory.BROKERAGE_CLAIMS,
        severity=ViolationSeverity.CRITICAL,
        description="Claims SEC/FINRA broker registration",
        fix_suggestion="Remove - we are not a registered broker",
    ),
    # --- Custody Claims ---
    ProhibitedPhrase(
        pattern=r"we\s+(?:hold|custody|store)\s+(?:your\s+)?(?:assets?|funds?)",
        category=PhraseCategory.CUSTODY_CLAIMS,
        severity=ViolationSeverity.CRITICAL,
        description="Claims custody of assets",
        fix_suggestion="We do NOT custody assets; users trade through their own broker accounts",
    ),
    ProhibitedPhrase(
        pattern=r"(?:platform|cloud)\s+(?:holds?|custod(?:y|ies))\s+(?:assets?|funds?)",
        category=PhraseCategory.CUSTODY_CLAIMS,
        severity=ViolationSeverity.CRITICAL,
        description="Claims cloud custody",
        fix_suggestion="Remove custody claims; we are a software provider",
    ),
    # --- Auto-Execution Claims ---
    ProhibitedPhrase(
        pattern=r"cloud\s+auto(?:[-\s])?execut(?:es?|ion)",
        category=PhraseCategory.AUTO_EXECUTION,
        severity=ViolationSeverity.CRITICAL,
        description="Claims cloud auto-execution",
        fix_suggestion="Change to: 'Agent (local) auto-executes based on cloud-deployed strategies'",
    ),
    ProhibitedPhrase(
        pattern=r"automat(?:ic|ed)\s+(?:cloud\s+)?(?:trade|order)\s+execution",
        category=PhraseCategory.AUTO_EXECUTION,
        severity=ViolationSeverity.HIGH,
        description="Implies automated cloud trading",
        fix_suggestion="Clarify: 'Automated execution runs in your local Agent'",
    ),
    ProhibitedPhrase(
        pattern=r"we\s+(?:automatically|auto)\s+(?:trade|execute|place)",
        category=PhraseCategory.AUTO_EXECUTION,
        severity=ViolationSeverity.CRITICAL,
        description="Claims we automatically trade",
        fix_suggestion="Change to: 'Your Agent automatically executes based on your strategy'",
    ),
]

# Directories to scan for documentation
DOCS_DIRECTORIES: Final[List[str]] = [
    "docs/legal",
    "docs/business",
    "docs",
    ".",  # README, ARCHITECTURE, etc.
]

# File patterns to check
DOCS_FILE_PATTERNS: Final[List[str]] = [
    "*.md",
    "TERMS_OF_SERVICE*",
    "PRIVACY_POLICY*",
    "DPA*",
    "AUP*",
    "README*",
    "ARCHITECTURE*",
    "*BUSINESS*",
    "*INVESTOR*",
    "*ENTERPRISE*",
    "*DORA*",
]

# Directories to exclude from scanning
EXCLUDED_DIRS: Final[FrozenSet[str]] = frozenset(
    {
        "docs/archive",
        "docs/archive/",
        "docs/plans",  # Planning docs discuss issues to fix, not actual problems
        "docs/plans/",
        "docs/design",  # Design documents discuss issues to fix, not actual problems
        "docs/design/",
        "docs/contracts",  # Contract templates may have negation context
        "docs/contracts/",
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
    }
)

# Files to exclude (e.g., this guardrail itself, legitimate usage in code)
EXCLUDED_FILES: Final[FrozenSet[str]] = frozenset(
    {
        "phrase_guard.py",  # This file
        "test_phrase_guard.py",  # Tests
        "ccea/guardrails/phrase_guard.py",
        "tests/ccea/phase6/test_phrase_guard.py",
        "PATENT_CLAIMS_DRAFT.md",  # Patent claims are technical descriptions
        "CCEA_MASTER_REMEDIATION_PLAN.md",  # Planning doc discusses issues to fix
        "SOFTWARE_PROVIDER_COMPLIANCE_PLAN.md",  # Planning doc
    }
)


# ============================================================================
# Violation Classes
# ============================================================================


@dataclass
class PhraseViolation:
    """Represents a phrase guard violation."""

    phrase: ProhibitedPhrase
    file_path: str
    line_number: int
    matched_text: str
    line_content: str

    def __str__(self) -> str:
        return (
            f"[{self.phrase.severity.value.upper()}] {self.file_path}:{self.line_number} "
            f"({self.phrase.category.value}): {self.phrase.description}\n"
            f"  Found: '{self.matched_text}'\n"
            f"  Line: {self.line_content.strip()}\n"
            f"  Fix: {self.phrase.fix_suggestion}"
        )


@dataclass
class PhraseGuardResult:
    """Result of running phrase guard."""

    violations: List[PhraseViolation] = field(default_factory=list)
    files_checked: int = 0
    total_lines_checked: int = 0

    @property
    def passed(self) -> bool:
        """Check if guardrail passed (no critical/high violations)."""
        blocking_severities = {ViolationSeverity.CRITICAL, ViolationSeverity.HIGH}
        return not any(v.phrase.severity in blocking_severities for v in self.violations)

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.phrase.severity == ViolationSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for v in self.violations if v.phrase.severity == ViolationSeverity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for v in self.violations if v.phrase.severity == ViolationSeverity.MEDIUM)

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.phrase.severity == ViolationSeverity.WARNING)


# ============================================================================
# Main Checker Class
# ============================================================================


class PhraseGuard:
    """Checks documentation for prohibited phrases that contradict CCEA."""

    def __init__(
        self,
        root_path: Optional[Path] = None,
        extra_patterns: Optional[List[ProhibitedPhrase]] = None,
    ):
        self.root_path = root_path or Path.cwd()
        self.phrases = list(PROHIBITED_PHRASES)
        if extra_patterns:
            self.phrases.extend(extra_patterns)

        # Compile regex patterns
        self._compiled_patterns: List[Tuple[Pattern, ProhibitedPhrase]] = []
        for phrase in self.phrases:
            flags = 0 if phrase.case_sensitive else re.IGNORECASE
            try:
                compiled = re.compile(phrase.pattern, flags)
                self._compiled_patterns.append((compiled, phrase))
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{phrase.pattern}': {e}")

    def _should_check_file(self, file_path: Path) -> bool:
        """Check if file should be scanned."""
        rel_path = str(file_path.relative_to(self.root_path))

        # Check exclusions
        for excl in EXCLUDED_DIRS:
            if rel_path.startswith(excl):
                return False

        for excl in EXCLUDED_FILES:
            if rel_path.endswith(excl) or file_path.name == excl:
                return False

        return True

    def _get_files_to_check(self) -> List[Path]:
        """Get list of documentation files to check."""
        files = []

        for doc_dir in DOCS_DIRECTORIES:
            dir_path = self.root_path / doc_dir
            if not dir_path.exists():
                continue

            for pattern in DOCS_FILE_PATTERNS:
                if doc_dir == ".":
                    # Only check root-level files, not recursively
                    for f in dir_path.glob(pattern):
                        if f.is_file() and self._should_check_file(f):
                            files.append(f)
                else:
                    for f in dir_path.rglob(pattern):
                        if f.is_file() and self._should_check_file(f):
                            files.append(f)

        return list(set(files))  # Deduplicate

    def check_content(self, content: str, file_path: str = "<string>") -> List[PhraseViolation]:
        """Check content string for prohibited phrases."""
        violations = []
        lines = content.split("\n")

        in_code_block = False
        for line_num, line in enumerate(lines, start=1):
            # Track code block state
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue

            # Skip lines inside code blocks or indented code (4+ spaces)
            if in_code_block or line.startswith("    "):
                continue

            # Skip comment lines in code-like contexts
            if stripped.startswith("#") and "(" not in stripped and ")" not in stripped:
                continue

            for compiled_pattern, phrase in self._compiled_patterns:
                matches = compiled_pattern.finditer(line)
                for match in matches:
                    violations.append(
                        PhraseViolation(
                            phrase=phrase,
                            file_path=file_path,
                            line_number=line_num,
                            matched_text=match.group(),
                            line_content=line,
                        )
                    )

        return violations

    def check_file(self, file_path: Path) -> List[PhraseViolation]:
        """Check a single file for prohibited phrases."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return []

        rel_path = str(file_path.relative_to(self.root_path))
        return self.check_content(content, rel_path)

    def run(self) -> PhraseGuardResult:
        """Run phrase guard on all documentation files."""
        result = PhraseGuardResult()
        files = self._get_files_to_check()

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                result.files_checked += 1
                result.total_lines_checked += content.count("\n") + 1

                violations = self.check_file(file_path)
                result.violations.extend(violations)
            except Exception as e:
                print(f"Error checking {file_path}: {e}")

        return result


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """Run phrase guard as CLI tool."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CCEA Phrase Guard - Check documentation for prohibited phrases"
    )
    parser.add_argument(
        "--root", type=str, default=".", help="Root path to check (default: current directory)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fail-on-warning", action="store_true", help="Fail on warnings (not just critical/high)"
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    guard = PhraseGuard(root_path=root)
    result = guard.run()

    # Print results
    print(f"\nCCEA Phrase Guard Results")
    print(f"{'=' * 60}")
    print(f"Files checked: {result.files_checked}")
    print(f"Lines checked: {result.total_lines_checked}")
    print()

    if result.violations:
        print(f"Violations found: {len(result.violations)}")
        print(f"  CRITICAL: {result.critical_count}")
        print(f"  HIGH:     {result.high_count}")
        print(f"  MEDIUM:   {result.medium_count}")
        print(f"  WARNING:  {result.warning_count}")
        print()

        # Group by severity
        for severity in [
            ViolationSeverity.CRITICAL,
            ViolationSeverity.HIGH,
            ViolationSeverity.MEDIUM,
            ViolationSeverity.WARNING,
        ]:
            sev_violations = [v for v in result.violations if v.phrase.severity == severity]
            if sev_violations and (
                args.verbose or severity in {ViolationSeverity.CRITICAL, ViolationSeverity.HIGH}
            ):
                print(f"\n{severity.value.upper()} Violations:")
                print("-" * 40)
                for v in sev_violations:
                    print(str(v))
                    print()
    else:
        print("No violations found!")

    # Determine exit code
    if args.fail_on_warning:
        passed = len(result.violations) == 0
    else:
        passed = result.passed

    print()
    if passed:
        print("[OK] Phrase guard passed")
        return 0
    else:
        print("[FAIL] Phrase guard failed - CCEA boundary violations detected")
        return 1


if __name__ == "__main__":
    sys.exit(main())

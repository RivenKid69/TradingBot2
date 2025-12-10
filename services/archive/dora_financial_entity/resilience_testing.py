# -*- coding: utf-8 -*-
"""
DORA Digital Operational Resilience Testing Programme (Article 24).

Regulation (EU) 2022/2554 Article 24 defines requirements for digital
operational resilience testing:
    - Sound and comprehensive testing programme as part of ICT risk management
    - Range of assessments, tests, methodologies, practices, and tools
    - Risk-based approach proportionate to size and risk profile
    - Testing by independent parties (internal or external)
    - Annual testing of all critical ICT systems and applications

Testing Categories per Article 24(1):
    1. Vulnerability assessments and scans
    2. Open source analyses
    3. Network security assessments
    4. Gap analyses
    5. Physical security reviews
    6. Questionnaires and scanning software solutions
    7. Source code reviews (where feasible)
    8. Scenario-based tests
    9. Compatibility testing
    10. Performance testing
    11. End-to-end testing
    12. Penetration testing

References:
    - Article 24 DORA: https://www.digital-operational-resilience-act.com/Article_24.html
    - RTS on TLPT (CDR 2024/2698): https://eur-lex.europa.eu/eli/reg_del/2024/2698/oj
    - TIBER-EU Framework: https://www.ecb.europa.eu/paym/cyber-resilience/tiber-eu/html/index.en.html
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class TestCategory(Enum):
    """Test categories per DORA Article 24(1)."""
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"  # Art. 24(1)(a)
    OPEN_SOURCE_ANALYSIS = "open_source_analysis"          # Art. 24(1)(a)
    NETWORK_SECURITY = "network_security"                  # Art. 24(1)(a)
    GAP_ANALYSIS = "gap_analysis"                          # Art. 24(1)(a)
    PHYSICAL_SECURITY = "physical_security"                # Art. 24(1)(a)
    QUESTIONNAIRE = "questionnaire"                        # Art. 24(1)(a)
    SOURCE_CODE_REVIEW = "source_code_review"              # Art. 24(1)(a)
    SCENARIO_BASED = "scenario_based"                      # Art. 24(1)(a)
    COMPATIBILITY = "compatibility"                         # Art. 24(1)(a)
    PERFORMANCE = "performance"                            # Art. 24(1)(a)
    END_TO_END = "end_to_end"                              # Art. 24(1)(a)
    PENETRATION_TESTING = "penetration_testing"            # Art. 24(1)(a)


class TestFrequency(Enum):
    """Test execution frequencies."""
    CONTINUOUS = "continuous"      # Real-time/continuous monitoring
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    BIENNIAL = "biennial"          # Every 2 years
    TRIENNIAL = "triennial"        # Every 3 years (TLPT minimum)
    ON_CHANGE = "on_change"        # Triggered by system changes
    AD_HOC = "ad_hoc"              # As needed


class TestStatus(Enum):
    """Test execution status."""
    PLANNED = "planned"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TestResult(Enum):
    """Test result outcomes."""
    PASSED = "passed"
    PASSED_WITH_FINDINGS = "passed_with_findings"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"


class FindingSeverity(Enum):
    """Severity levels for test findings."""
    CRITICAL = "critical"    # Immediate action required
    HIGH = "high"            # Action within 7 days
    MEDIUM = "medium"        # Action within 30 days
    LOW = "low"              # Action within 90 days
    INFORMATIONAL = "informational"  # No action required


class FindingStatus(Enum):
    """Status of test findings."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"       # Risk accepted
    FALSE_POSITIVE = "false_positive"
    DEFERRED = "deferred"


class TesterType(Enum):
    """Type of testing party per Article 24(4)."""
    INTERNAL = "internal"        # Internal resources
    EXTERNAL = "external"        # External service providers
    MIXED = "mixed"              # Combination


class SystemCriticality(Enum):
    """System criticality levels for testing prioritization."""
    CRITICAL = "critical"        # Critical functions
    HIGH = "high"                # Important functions
    MEDIUM = "medium"            # Standard functions
    LOW = "low"                  # Support functions


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TestScope:
    """
    Test scope definition per Article 24(3).
    """
    scope_id: str = ""
    name: str = ""
    description: str = ""

    # Systems in scope
    systems: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    networks: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=list)

    # Functions in scope
    business_functions: List[str] = field(default_factory=list)
    critical_functions: List[str] = field(default_factory=list)

    # Third-party components
    third_party_providers: List[str] = field(default_factory=list)
    third_party_interfaces: List[str] = field(default_factory=list)

    # Boundaries
    included_environments: List[str] = field(default_factory=list)  # prod, staging, etc.
    excluded_systems: List[str] = field(default_factory=list)
    excluded_tests: List[str] = field(default_factory=list)

    # Risk-based prioritization
    system_criticalities: Dict[str, str] = field(default_factory=dict)

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    approved_by: str = ""

    def __post_init__(self):
        if not self.scope_id:
            self.scope_id = f"SCOPE-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TestDefinition:
    """
    Test definition per Article 24(1).
    """
    test_id: str = ""
    name: str = ""
    description: str = ""

    # Classification
    category: TestCategory = TestCategory.VULNERABILITY_ASSESSMENT
    article_reference: str = "Article 24(1)(a)"

    # Scheduling
    frequency: TestFrequency = TestFrequency.ANNUAL
    next_scheduled_date: str = ""
    last_executed_date: str = ""

    # Requirements
    minimum_frequency: TestFrequency = TestFrequency.ANNUAL
    required_for_critical_systems: bool = True

    # Execution
    methodology: str = ""
    tools: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)

    # Tester requirements
    tester_type: TesterType = TesterType.INTERNAL
    required_certifications: List[str] = field(default_factory=list)
    independence_required: bool = True

    # Duration
    estimated_duration_hours: float = 0.0

    # Prerequisites
    prerequisites: List[str] = field(default_factory=list)

    # Active status
    is_active: bool = True

    # Metadata
    created_at: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if not self.test_id:
            self.test_id = f"TEST-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TestExecution:
    """
    Test execution record per Article 24(5).
    """
    execution_id: str = ""
    test_id: str = ""
    scope_id: str = ""

    # Test details
    test_name: str = ""
    test_category: TestCategory = TestCategory.VULNERABILITY_ASSESSMENT

    # Scheduling
    scheduled_start: str = ""
    scheduled_end: str = ""
    actual_start: str = ""
    actual_end: str = ""

    # Status
    status: TestStatus = TestStatus.PLANNED
    result: TestResult = TestResult.INCONCLUSIVE

    # Execution details
    methodology_used: str = ""
    tools_used: List[str] = field(default_factory=list)
    environment: str = ""  # production, staging, etc.

    # Testers
    tester_type: TesterType = TesterType.INTERNAL
    testers: List[str] = field(default_factory=list)
    tester_organization: str = ""

    # Coverage
    systems_tested: List[str] = field(default_factory=list)
    systems_coverage_percent: float = 0.0

    # Findings summary
    findings_count: int = 0
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0
    informational_findings: int = 0

    # Results
    executive_summary: str = ""
    detailed_results: Dict[str, Any] = field(default_factory=dict)

    # Remediation
    remediation_required: bool = False
    remediation_deadline: str = ""

    # Approvals
    approved_by: str = ""
    approved_at: str = ""

    # Evidence
    evidence_references: List[str] = field(default_factory=list)
    report_path: str = ""

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = f"EXEC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TestFinding:
    """
    Finding from a test execution per Article 24(5).
    """
    finding_id: str = ""
    execution_id: str = ""

    # Finding details
    title: str = ""
    description: str = ""
    technical_details: str = ""

    # Classification
    severity: FindingSeverity = FindingSeverity.MEDIUM
    status: FindingStatus = FindingStatus.OPEN
    category: str = ""  # vulnerability type, issue category

    # Affected components
    affected_systems: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    affected_data: List[str] = field(default_factory=list)

    # Risk assessment
    likelihood: str = ""  # high, medium, low
    impact: str = ""      # high, medium, low
    risk_rating: str = ""
    cvss_score: Optional[float] = None
    cve_ids: List[str] = field(default_factory=list)

    # Evidence
    evidence: str = ""
    proof_of_concept: str = ""
    screenshots: List[str] = field(default_factory=list)

    # Remediation
    recommendation: str = ""
    remediation_steps: List[str] = field(default_factory=list)
    remediation_effort: str = ""  # hours, days, weeks
    remediation_deadline: str = ""
    remediation_owner: str = ""

    # Resolution
    resolution_details: str = ""
    resolved_at: str = ""
    resolved_by: str = ""
    verified: bool = False
    verified_at: str = ""
    verified_by: str = ""

    # Tracking
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""

    def __post_init__(self):
        if not self.finding_id:
            self.finding_id = f"FIND-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TestingProgramme:
    """
    Digital Operational Resilience Testing Programme per Article 24.
    """
    programme_id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0"

    # Entity information
    entity_name: str = ""
    entity_type: str = ""  # investment_firm, credit_institution, etc.
    entity_size: str = ""  # microenterprise, small, medium, large

    # Programme scope
    scope_id: str = ""

    # Governance
    programme_owner: str = ""
    approved_by: str = ""
    approval_date: str = ""

    # Testing strategy
    risk_based_approach: bool = True
    proportionality_applied: bool = True

    # Test definitions
    test_definitions: List[str] = field(default_factory=list)  # Test IDs

    # Annual plan
    annual_test_plan: List[Dict[str, Any]] = field(default_factory=list)

    # Schedule
    programme_start_date: str = ""
    programme_end_date: str = ""
    review_frequency: str = "annual"
    next_review_date: str = ""

    # Resources
    budget_eur: float = 0.0
    internal_resources: List[str] = field(default_factory=list)
    external_providers: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True

    # Compliance tracking
    article_24_compliance: Dict[str, bool] = field(default_factory=dict)

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.programme_id:
            self.programme_id = f"PROG-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TestingCycle:
    """
    Testing cycle (quarterly, annual) tracking.
    """
    cycle_id: str = ""
    programme_id: str = ""
    cycle_name: str = ""
    cycle_type: str = ""  # quarterly, annual

    # Period
    start_date: str = ""
    end_date: str = ""

    # Planned tests
    planned_executions: List[str] = field(default_factory=list)

    # Progress
    completed_executions: List[str] = field(default_factory=list)
    in_progress_executions: List[str] = field(default_factory=list)
    cancelled_executions: List[str] = field(default_factory=list)

    # Metrics
    planned_count: int = 0
    completed_count: int = 0
    pass_rate: float = 0.0

    # Findings summary
    total_findings: int = 0
    open_findings: int = 0
    resolved_findings: int = 0

    # Status
    status: str = "planned"  # planned, in_progress, completed

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.cycle_id:
            self.cycle_id = f"CYCLE-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ResilienceTestingConfig:
    """Configuration for Resilience Testing Programme."""
    # Programme defaults
    default_review_frequency_days: int = 365

    # Test frequencies (days)
    vulnerability_scan_frequency_days: int = 90  # Quarterly
    penetration_test_frequency_days: int = 365   # Annual
    source_code_review_on_release: bool = True

    # Findings SLA (days)
    critical_finding_sla_days: int = 7
    high_finding_sla_days: int = 14
    medium_finding_sla_days: int = 30
    low_finding_sla_days: int = 90

    # Independence requirements
    require_independent_testing: bool = True
    external_testing_frequency: int = 365  # At least annual

    # Reporting
    auto_generate_reports: bool = True
    report_retention_years: int = 5

    # Alerts
    alert_on_critical_finding: bool = True
    alert_on_overdue_test: bool = True

    # Logging
    log_path: str = "logs/dora/resilience_testing"


# =============================================================================
# Default Test Definitions per Article 24(1)
# =============================================================================

DEFAULT_TEST_DEFINITIONS = [
    {
        "name": "Vulnerability Assessment and Scanning",
        "description": "Automated and manual vulnerability assessments of ICT systems",
        "category": TestCategory.VULNERABILITY_ASSESSMENT,
        "frequency": TestFrequency.QUARTERLY,
        "minimum_frequency": TestFrequency.QUARTERLY,
        "methodology": "OWASP Testing Guide, NIST SP 800-115",
        "tools": ["Nessus", "Qualys", "OpenVAS", "Burp Suite"],
        "tester_type": TesterType.INTERNAL,
        "article_reference": "Article 24(1)(a)",
    },
    {
        "name": "Open Source Software Analysis",
        "description": "Analysis of open source components for known vulnerabilities",
        "category": TestCategory.OPEN_SOURCE_ANALYSIS,
        "frequency": TestFrequency.CONTINUOUS,
        "minimum_frequency": TestFrequency.WEEKLY,
        "methodology": "Software Composition Analysis (SCA)",
        "tools": ["OWASP Dependency-Check", "Snyk", "Black Duck", "WhiteSource"],
        "tester_type": TesterType.INTERNAL,
        "article_reference": "Article 24(1)(a)",
    },
    {
        "name": "Network Security Assessment",
        "description": "Assessment of network architecture, segmentation, and controls",
        "category": TestCategory.NETWORK_SECURITY,
        "frequency": TestFrequency.QUARTERLY,
        "minimum_frequency": TestFrequency.SEMI_ANNUAL,
        "methodology": "Network Security Assessment Framework",
        "tools": ["Nmap", "Wireshark", "Netcat", "Network Scanners"],
        "tester_type": TesterType.INTERNAL,
        "article_reference": "Article 24(1)(a)",
    },
    {
        "name": "Gap Analysis",
        "description": "Analysis of gaps between current state and desired security posture",
        "category": TestCategory.GAP_ANALYSIS,
        "frequency": TestFrequency.ANNUAL,
        "minimum_frequency": TestFrequency.ANNUAL,
        "methodology": "ISO 27001, NIST CSF Gap Analysis",
        "tools": ["Assessment Templates", "Control Mapping Tools"],
        "tester_type": TesterType.EXTERNAL,
        "article_reference": "Article 24(1)(a)",
    },
    {
        "name": "Physical Security Review",
        "description": "Review of physical security controls protecting ICT systems",
        "category": TestCategory.PHYSICAL_SECURITY,
        "frequency": TestFrequency.ANNUAL,
        "minimum_frequency": TestFrequency.ANNUAL,
        "methodology": "Physical Security Assessment",
        "tools": ["Site Assessment Checklist", "Physical Penetration Testing"],
        "tester_type": TesterType.EXTERNAL,
        "article_reference": "Article 24(1)(a)",
    },
    {
        "name": "Security Questionnaire Assessment",
        "description": "Self-assessment and third-party questionnaires",
        "category": TestCategory.QUESTIONNAIRE,
        "frequency": TestFrequency.ANNUAL,
        "minimum_frequency": TestFrequency.ANNUAL,
        "methodology": "SIG, CAIQ, Custom Questionnaires",
        "tools": ["OneTrust", "SecurityScorecard", "Custom Questionnaires"],
        "tester_type": TesterType.INTERNAL,
        "article_reference": "Article 24(1)(a)",
    },
    {
        "name": "Source Code Review",
        "description": "Security review of source code for critical applications",
        "category": TestCategory.SOURCE_CODE_REVIEW,
        "frequency": TestFrequency.ON_CHANGE,
        "minimum_frequency": TestFrequency.QUARTERLY,
        "methodology": "OWASP Code Review Guide, SAST/DAST",
        "tools": ["SonarQube", "Checkmarx", "Fortify", "Semgrep"],
        "tester_type": TesterType.INTERNAL,
        "article_reference": "Article 24(1)(a)",
    },
    {
        "name": "Scenario-Based Testing",
        "description": "Testing based on realistic threat scenarios",
        "category": TestCategory.SCENARIO_BASED,
        "frequency": TestFrequency.ANNUAL,
        "minimum_frequency": TestFrequency.ANNUAL,
        "methodology": "Threat Modeling, Attack Scenarios",
        "tools": ["MITRE ATT&CK Framework", "Atomic Red Team"],
        "tester_type": TesterType.MIXED,
        "article_reference": "Article 24(1)(a)",
    },
    {
        "name": "Compatibility Testing",
        "description": "Testing for system and software compatibility",
        "category": TestCategory.COMPATIBILITY,
        "frequency": TestFrequency.ON_CHANGE,
        "minimum_frequency": TestFrequency.QUARTERLY,
        "methodology": "Compatibility Test Planning",
        "tools": ["Test Automation Tools", "Virtual Environments"],
        "tester_type": TesterType.INTERNAL,
        "article_reference": "Article 24(1)(a)",
    },
    {
        "name": "Performance Testing",
        "description": "Load, stress, and performance testing of critical systems",
        "category": TestCategory.PERFORMANCE,
        "frequency": TestFrequency.QUARTERLY,
        "minimum_frequency": TestFrequency.SEMI_ANNUAL,
        "methodology": "Performance Testing Methodology",
        "tools": ["JMeter", "Locust", "k6", "Gatling"],
        "tester_type": TesterType.INTERNAL,
        "article_reference": "Article 24(1)(a)",
    },
    {
        "name": "End-to-End Testing",
        "description": "Full business process and workflow testing",
        "category": TestCategory.END_TO_END,
        "frequency": TestFrequency.QUARTERLY,
        "minimum_frequency": TestFrequency.SEMI_ANNUAL,
        "methodology": "E2E Test Automation",
        "tools": ["Selenium", "Cypress", "Playwright", "Custom Scripts"],
        "tester_type": TesterType.INTERNAL,
        "article_reference": "Article 24(1)(a)",
    },
    {
        "name": "Penetration Testing",
        "description": "Simulated cyber attacks to identify vulnerabilities",
        "category": TestCategory.PENETRATION_TESTING,
        "frequency": TestFrequency.ANNUAL,
        "minimum_frequency": TestFrequency.ANNUAL,
        "methodology": "PTES, OSSTMM, OWASP Testing Guide",
        "tools": ["Metasploit", "Burp Suite Pro", "Cobalt Strike", "Custom Tools"],
        "tester_type": TesterType.EXTERNAL,
        "independence_required": True,
        "article_reference": "Article 24(1)(a)",
    },
]


# =============================================================================
# Main Class Implementation
# =============================================================================

class DORAResilienceTestingProgramme:
    """
    DORA Article 24 compliant Digital Operational Resilience Testing Programme.

    Provides:
    - Testing programme definition and management
    - Test scope definition
    - Test scheduling and execution tracking
    - Findings management and remediation tracking
    - Compliance monitoring
    - Reporting

    Article 24 Requirements:
    1. Sound and comprehensive testing programme
    2. Range of assessments, tests, methodologies
    3. Risk-based approach proportionate to size and risk profile
    4. Independent testing by internal or external parties
    5. Annual testing of all ICT systems and applications
    6. Documented procedures and evidence retention

    Usage:
        config = ResilienceTestingConfig()
        programme = DORAResilienceTestingProgramme(config)

        # Create testing programme
        prog = programme.create_programme(
            name="Annual Testing Programme 2025",
            entity_name="AlgoTrade Ltd",
            entity_type="investment_firm",
        )

        # Define test scope
        scope = programme.create_scope(
            name="Production Systems",
            systems=["trading_engine", "risk_management", "market_data"],
            critical_functions=["order_execution", "position_management"],
        )

        # Schedule and execute tests
        execution = programme.schedule_test(
            test_id="TEST-001",
            scope_id=scope.scope_id,
            scheduled_start="2025-01-15T09:00:00Z",
        )
    """

    def __init__(self, config: Optional[ResilienceTestingConfig] = None):
        """Initialize Resilience Testing Programme."""
        self.config = config or ResilienceTestingConfig()

        # Data stores
        self._programmes: Dict[str, TestingProgramme] = {}
        self._scopes: Dict[str, TestScope] = {}
        self._test_definitions: Dict[str, TestDefinition] = {}
        self._executions: Dict[str, TestExecution] = {}
        self._findings: Dict[str, TestFinding] = {}
        self._cycles: Dict[str, TestingCycle] = {}

        # Initialize default test definitions
        self._init_default_tests()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAResilienceTestingProgramme initialized")

    def _init_default_tests(self) -> None:
        """Initialize default test definitions per Article 24(1)."""
        for test_def in DEFAULT_TEST_DEFINITIONS:
            definition = TestDefinition(
                name=test_def["name"],
                description=test_def["description"],
                category=test_def["category"],
                frequency=test_def["frequency"],
                minimum_frequency=test_def["minimum_frequency"],
                methodology=test_def["methodology"],
                tools=test_def["tools"],
                tester_type=test_def["tester_type"],
                article_reference=test_def["article_reference"],
            )
            if "independence_required" in test_def:
                definition.independence_required = test_def["independence_required"]

            self._test_definitions[definition.test_id] = definition

    # =========================================================================
    # Programme Management
    # =========================================================================

    def create_programme(
        self,
        name: str,
        entity_name: str,
        entity_type: str,
        entity_size: str = "medium",
        description: str = "",
        programme_owner: str = "",
        approved_by: str = "",
        budget_eur: float = 0.0,
        start_date: Optional[str] = None,
    ) -> TestingProgramme:
        """
        Create a new testing programme per Article 24.

        Args:
            name: Programme name
            entity_name: Financial entity name
            entity_type: Entity type (investment_firm, credit_institution, etc.)
            entity_size: Entity size (microenterprise, small, medium, large)
            description: Programme description
            programme_owner: Programme owner
            approved_by: Approver name
            budget_eur: Budget in EUR
            start_date: Programme start date

        Returns:
            Created TestingProgramme
        """
        programme = TestingProgramme(
            name=name,
            description=description or f"Digital Operational Resilience Testing Programme for {entity_name}",
            entity_name=entity_name,
            entity_type=entity_type,
            entity_size=entity_size,
            programme_owner=programme_owner,
            approved_by=approved_by,
            approval_date=datetime.now(timezone.utc).isoformat() if approved_by else "",
            budget_eur=budget_eur,
            programme_start_date=start_date or datetime.now(timezone.utc).isoformat(),
            test_definitions=list(self._test_definitions.keys()),
        )

        # Apply proportionality based on entity size
        if entity_size == "microenterprise":
            programme.proportionality_applied = True
            # Microenterprises have reduced testing requirements

        # Set Article 24 compliance tracking
        programme.article_24_compliance = {
            "testing_programme_established": True,
            "vulnerability_assessments_included": True,
            "network_security_included": True,
            "gap_analyses_included": True,
            "physical_security_included": True,
            "scenario_testing_included": True,
            "penetration_testing_included": True,
            "risk_based_approach": programme.risk_based_approach,
            "independent_testing": self.config.require_independent_testing,
        }

        # Calculate next review date
        programme.next_review_date = (
            datetime.now(timezone.utc) +
            timedelta(days=self.config.default_review_frequency_days)
        ).isoformat()

        self._programmes[programme.programme_id] = programme

        self._log_event("programme_created", {
            "programme_id": programme.programme_id,
            "entity_name": entity_name,
            "entity_type": entity_type,
        })

        logger.info(f"Testing programme created: {programme.programme_id}")
        return programme

    def get_programme(self, programme_id: str) -> Optional[TestingProgramme]:
        """Get programme by ID."""
        return self._programmes.get(programme_id)

    def update_programme(
        self,
        programme_id: str,
        **updates: Any,
    ) -> TestingProgramme:
        """Update programme fields."""
        if programme_id not in self._programmes:
            raise ValueError(f"Programme {programme_id} not found")

        programme = self._programmes[programme_id]
        for key, value in updates.items():
            if hasattr(programme, key):
                setattr(programme, key, value)

        programme.updated_at = datetime.now(timezone.utc).isoformat()
        return programme

    # =========================================================================
    # Scope Management
    # =========================================================================

    def create_scope(
        self,
        name: str,
        description: str = "",
        systems: Optional[List[str]] = None,
        applications: Optional[List[str]] = None,
        networks: Optional[List[str]] = None,
        business_functions: Optional[List[str]] = None,
        critical_functions: Optional[List[str]] = None,
        third_party_providers: Optional[List[str]] = None,
        included_environments: Optional[List[str]] = None,
        excluded_systems: Optional[List[str]] = None,
        system_criticalities: Optional[Dict[str, str]] = None,
        approved_by: str = "",
    ) -> TestScope:
        """
        Create a test scope per Article 24(3).

        Args:
            name: Scope name
            description: Scope description
            systems: Systems in scope
            applications: Applications in scope
            networks: Networks in scope
            business_functions: Business functions in scope
            critical_functions: Critical functions in scope
            third_party_providers: Third-party providers in scope
            included_environments: Environments in scope
            excluded_systems: Systems excluded from scope
            system_criticalities: System criticality mappings
            approved_by: Approver

        Returns:
            Created TestScope
        """
        scope = TestScope(
            name=name,
            description=description,
            systems=systems or [],
            applications=applications or [],
            networks=networks or [],
            business_functions=business_functions or [],
            critical_functions=critical_functions or [],
            third_party_providers=third_party_providers or [],
            included_environments=included_environments or ["production"],
            excluded_systems=excluded_systems or [],
            system_criticalities=system_criticalities or {},
            approved_by=approved_by,
        )

        self._scopes[scope.scope_id] = scope

        self._log_event("scope_created", {
            "scope_id": scope.scope_id,
            "name": name,
            "systems_count": len(scope.systems),
        })

        return scope

    def get_scope(self, scope_id: str) -> Optional[TestScope]:
        """Get scope by ID."""
        return self._scopes.get(scope_id)

    def update_scope(
        self,
        scope_id: str,
        **updates: Any,
    ) -> TestScope:
        """Update scope fields."""
        if scope_id not in self._scopes:
            raise ValueError(f"Scope {scope_id} not found")

        scope = self._scopes[scope_id]
        for key, value in updates.items():
            if hasattr(scope, key):
                setattr(scope, key, value)

        scope.updated_at = datetime.now(timezone.utc).isoformat()
        return scope

    # =========================================================================
    # Test Definition Management
    # =========================================================================

    def create_test_definition(
        self,
        name: str,
        description: str,
        category: TestCategory,
        frequency: TestFrequency = TestFrequency.ANNUAL,
        minimum_frequency: TestFrequency = TestFrequency.ANNUAL,
        methodology: str = "",
        tools: Optional[List[str]] = None,
        tester_type: TesterType = TesterType.INTERNAL,
        required_certifications: Optional[List[str]] = None,
        independence_required: bool = True,
        estimated_duration_hours: float = 0.0,
        prerequisites: Optional[List[str]] = None,
    ) -> TestDefinition:
        """
        Create a custom test definition.

        Args:
            name: Test name
            description: Test description
            category: Test category
            frequency: Execution frequency
            minimum_frequency: Minimum required frequency
            methodology: Testing methodology
            tools: Testing tools
            tester_type: Type of tester
            required_certifications: Required certifications
            independence_required: Whether independence is required
            estimated_duration_hours: Estimated duration
            prerequisites: Prerequisites

        Returns:
            Created TestDefinition
        """
        definition = TestDefinition(
            name=name,
            description=description,
            category=category,
            frequency=frequency,
            minimum_frequency=minimum_frequency,
            methodology=methodology,
            tools=tools or [],
            tester_type=tester_type,
            required_certifications=required_certifications or [],
            independence_required=independence_required,
            estimated_duration_hours=estimated_duration_hours,
            prerequisites=prerequisites or [],
        )

        self._test_definitions[definition.test_id] = definition

        self._log_event("test_definition_created", {
            "test_id": definition.test_id,
            "name": name,
            "category": category.value,
        })

        return definition

    def get_test_definition(self, test_id: str) -> Optional[TestDefinition]:
        """Get test definition by ID."""
        return self._test_definitions.get(test_id)

    def get_test_definitions_by_category(
        self,
        category: TestCategory,
    ) -> List[TestDefinition]:
        """Get test definitions by category."""
        return [
            t for t in self._test_definitions.values()
            if t.category == category and t.is_active
        ]

    def get_all_test_definitions(self) -> List[TestDefinition]:
        """Get all active test definitions."""
        return [t for t in self._test_definitions.values() if t.is_active]

    # =========================================================================
    # Test Scheduling and Execution
    # =========================================================================

    def schedule_test(
        self,
        test_id: str,
        scope_id: str,
        scheduled_start: str,
        scheduled_end: Optional[str] = None,
        testers: Optional[List[str]] = None,
        tester_organization: str = "",
        environment: str = "production",
    ) -> TestExecution:
        """
        Schedule a test execution.

        Args:
            test_id: Test definition ID
            scope_id: Test scope ID
            scheduled_start: Scheduled start time
            scheduled_end: Scheduled end time
            testers: Assigned testers
            tester_organization: Tester organization
            environment: Test environment

        Returns:
            Created TestExecution
        """
        if test_id not in self._test_definitions:
            raise ValueError(f"Test definition {test_id} not found")
        if scope_id not in self._scopes:
            raise ValueError(f"Scope {scope_id} not found")

        test_def = self._test_definitions[test_id]
        scope = self._scopes[scope_id]

        execution = TestExecution(
            test_id=test_id,
            scope_id=scope_id,
            test_name=test_def.name,
            test_category=test_def.category,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end or "",
            status=TestStatus.SCHEDULED,
            methodology_used=test_def.methodology,
            tools_used=test_def.tools.copy(),
            environment=environment,
            tester_type=test_def.tester_type,
            testers=testers or [],
            tester_organization=tester_organization,
            systems_tested=scope.systems.copy(),
        )

        self._executions[execution.execution_id] = execution

        # Update test definition last scheduled
        test_def.next_scheduled_date = scheduled_start

        self._log_event("test_scheduled", {
            "execution_id": execution.execution_id,
            "test_id": test_id,
            "scheduled_start": scheduled_start,
        })

        logger.info(f"Test scheduled: {execution.execution_id}")
        return execution

    def start_test_execution(
        self,
        execution_id: str,
        testers: Optional[List[str]] = None,
        notes: str = "",
    ) -> TestExecution:
        """
        Start a test execution.

        Args:
            execution_id: Execution ID
            testers: Testers (if different from scheduled)
            notes: Start notes

        Returns:
            Updated TestExecution
        """
        if execution_id not in self._executions:
            raise ValueError(f"Execution {execution_id} not found")

        execution = self._executions[execution_id]
        now = datetime.now(timezone.utc).isoformat()

        execution.status = TestStatus.IN_PROGRESS
        execution.actual_start = now
        execution.updated_at = now

        if testers:
            execution.testers = testers

        self._log_event("test_started", {
            "execution_id": execution_id,
            "started_at": now,
        })

        return execution

    def complete_test_execution(
        self,
        execution_id: str,
        result: TestResult,
        executive_summary: str = "",
        detailed_results: Optional[Dict[str, Any]] = None,
        systems_coverage_percent: float = 100.0,
        report_path: str = "",
        approved_by: str = "",
    ) -> TestExecution:
        """
        Complete a test execution.

        Args:
            execution_id: Execution ID
            result: Test result
            executive_summary: Executive summary
            detailed_results: Detailed results
            systems_coverage_percent: Coverage percentage
            report_path: Path to report file
            approved_by: Approver

        Returns:
            Updated TestExecution
        """
        if execution_id not in self._executions:
            raise ValueError(f"Execution {execution_id} not found")

        execution = self._executions[execution_id]
        now = datetime.now(timezone.utc).isoformat()

        execution.status = TestStatus.COMPLETED
        execution.result = result
        execution.actual_end = now
        execution.executive_summary = executive_summary
        execution.detailed_results = detailed_results or {}
        execution.systems_coverage_percent = systems_coverage_percent
        execution.report_path = report_path
        execution.updated_at = now

        if approved_by:
            execution.approved_by = approved_by
            execution.approved_at = now

        # Update test definition last executed
        test_def = self._test_definitions.get(execution.test_id)
        if test_def:
            test_def.last_executed_date = now

        # Update findings count
        findings = self.get_findings_for_execution(execution_id)
        execution.findings_count = len(findings)
        execution.critical_findings = sum(1 for f in findings if f.severity == FindingSeverity.CRITICAL)
        execution.high_findings = sum(1 for f in findings if f.severity == FindingSeverity.HIGH)
        execution.medium_findings = sum(1 for f in findings if f.severity == FindingSeverity.MEDIUM)
        execution.low_findings = sum(1 for f in findings if f.severity == FindingSeverity.LOW)
        execution.informational_findings = sum(1 for f in findings if f.severity == FindingSeverity.INFORMATIONAL)

        # Check if remediation required
        execution.remediation_required = execution.critical_findings > 0 or execution.high_findings > 0
        if execution.remediation_required:
            # Set remediation deadline based on most severe finding
            if execution.critical_findings > 0:
                deadline_days = self.config.critical_finding_sla_days
            else:
                deadline_days = self.config.high_finding_sla_days
            execution.remediation_deadline = (
                datetime.now(timezone.utc) + timedelta(days=deadline_days)
            ).isoformat()

        self._log_event("test_completed", {
            "execution_id": execution_id,
            "result": result.value,
            "findings_count": execution.findings_count,
        })

        logger.info(f"Test completed: {execution_id} - Result: {result.value}")
        return execution

    def cancel_test_execution(
        self,
        execution_id: str,
        reason: str,
    ) -> TestExecution:
        """Cancel a test execution."""
        if execution_id not in self._executions:
            raise ValueError(f"Execution {execution_id} not found")

        execution = self._executions[execution_id]
        execution.status = TestStatus.CANCELLED
        execution.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("test_cancelled", {
            "execution_id": execution_id,
            "reason": reason,
        })

        return execution

    def get_execution(self, execution_id: str) -> Optional[TestExecution]:
        """Get execution by ID."""
        return self._executions.get(execution_id)

    def get_executions_by_status(
        self,
        status: TestStatus,
    ) -> List[TestExecution]:
        """Get executions by status."""
        return [e for e in self._executions.values() if e.status == status]

    def get_executions_for_scope(
        self,
        scope_id: str,
    ) -> List[TestExecution]:
        """Get all executions for a scope."""
        return [e for e in self._executions.values() if e.scope_id == scope_id]

    def get_overdue_tests(self) -> List[TestDefinition]:
        """Get tests that are overdue for execution."""
        overdue = []
        now = datetime.now(timezone.utc)

        for test_def in self._test_definitions.values():
            if not test_def.is_active:
                continue

            # Calculate days since last execution
            if test_def.last_executed_date:
                last_exec = datetime.fromisoformat(
                    test_def.last_executed_date.replace("Z", "+00:00")
                )
                days_since = (now - last_exec).days
            else:
                # Never executed
                days_since = 365  # Assume overdue

            # Get frequency in days
            frequency_days = self._frequency_to_days(test_def.minimum_frequency)

            if days_since > frequency_days:
                overdue.append(test_def)

        return overdue

    def _frequency_to_days(self, frequency: TestFrequency) -> int:
        """Convert frequency enum to days."""
        mapping = {
            TestFrequency.CONTINUOUS: 1,
            TestFrequency.DAILY: 1,
            TestFrequency.WEEKLY: 7,
            TestFrequency.MONTHLY: 30,
            TestFrequency.QUARTERLY: 90,
            TestFrequency.SEMI_ANNUAL: 180,
            TestFrequency.ANNUAL: 365,
            TestFrequency.BIENNIAL: 730,
            TestFrequency.TRIENNIAL: 1095,
            TestFrequency.ON_CHANGE: 365,  # Default to annual if triggered
            TestFrequency.AD_HOC: 365,
        }
        return mapping.get(frequency, 365)

    # =========================================================================
    # Findings Management
    # =========================================================================

    def record_finding(
        self,
        execution_id: str,
        title: str,
        description: str,
        severity: FindingSeverity,
        category: str = "",
        affected_systems: Optional[List[str]] = None,
        affected_components: Optional[List[str]] = None,
        technical_details: str = "",
        evidence: str = "",
        recommendation: str = "",
        remediation_steps: Optional[List[str]] = None,
        cvss_score: Optional[float] = None,
        cve_ids: Optional[List[str]] = None,
        likelihood: str = "",
        impact: str = "",
        created_by: str = "",
    ) -> TestFinding:
        """
        Record a finding from a test execution.

        Args:
            execution_id: Execution ID
            title: Finding title
            description: Finding description
            severity: Finding severity
            category: Finding category
            affected_systems: Affected systems
            affected_components: Affected components
            technical_details: Technical details
            evidence: Evidence
            recommendation: Remediation recommendation
            remediation_steps: Remediation steps
            cvss_score: CVSS score
            cve_ids: CVE IDs
            likelihood: Likelihood
            impact: Impact
            created_by: Creator

        Returns:
            Created TestFinding
        """
        if execution_id not in self._executions:
            raise ValueError(f"Execution {execution_id} not found")

        finding = TestFinding(
            execution_id=execution_id,
            title=title,
            description=description,
            severity=severity,
            category=category,
            affected_systems=affected_systems or [],
            affected_components=affected_components or [],
            technical_details=technical_details,
            evidence=evidence,
            recommendation=recommendation,
            remediation_steps=remediation_steps or [],
            cvss_score=cvss_score,
            cve_ids=cve_ids or [],
            likelihood=likelihood,
            impact=impact,
            created_by=created_by,
        )

        # Calculate risk rating
        if likelihood and impact:
            finding.risk_rating = self._calculate_risk_rating(likelihood, impact)

        # Set remediation deadline based on severity
        sla_days = {
            FindingSeverity.CRITICAL: self.config.critical_finding_sla_days,
            FindingSeverity.HIGH: self.config.high_finding_sla_days,
            FindingSeverity.MEDIUM: self.config.medium_finding_sla_days,
            FindingSeverity.LOW: self.config.low_finding_sla_days,
            FindingSeverity.INFORMATIONAL: 0,
        }
        if severity != FindingSeverity.INFORMATIONAL:
            finding.remediation_deadline = (
                datetime.now(timezone.utc) + timedelta(days=sla_days[severity])
            ).isoformat()

        self._findings[finding.finding_id] = finding

        self._log_event("finding_recorded", {
            "finding_id": finding.finding_id,
            "execution_id": execution_id,
            "severity": severity.value,
            "title": title,
        })

        # Alert on critical findings
        if severity == FindingSeverity.CRITICAL and self.config.alert_on_critical_finding:
            logger.critical(f"CRITICAL finding recorded: {finding.finding_id} - {title}")

        return finding

    def _calculate_risk_rating(self, likelihood: str, impact: str) -> str:
        """Calculate risk rating from likelihood and impact."""
        matrix = {
            ("high", "high"): "critical",
            ("high", "medium"): "high",
            ("high", "low"): "medium",
            ("medium", "high"): "high",
            ("medium", "medium"): "medium",
            ("medium", "low"): "low",
            ("low", "high"): "medium",
            ("low", "medium"): "low",
            ("low", "low"): "low",
        }
        return matrix.get((likelihood.lower(), impact.lower()), "medium")

    def update_finding_status(
        self,
        finding_id: str,
        status: FindingStatus,
        notes: str = "",
        updated_by: str = "",
    ) -> TestFinding:
        """Update finding status."""
        if finding_id not in self._findings:
            raise ValueError(f"Finding {finding_id} not found")

        finding = self._findings[finding_id]
        finding.status = status
        finding.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("finding_status_updated", {
            "finding_id": finding_id,
            "status": status.value,
            "updated_by": updated_by,
        })

        return finding

    def resolve_finding(
        self,
        finding_id: str,
        resolution_details: str,
        resolved_by: str,
        verified: bool = False,
        verified_by: str = "",
    ) -> TestFinding:
        """Resolve a finding."""
        if finding_id not in self._findings:
            raise ValueError(f"Finding {finding_id} not found")

        finding = self._findings[finding_id]
        now = datetime.now(timezone.utc).isoformat()

        finding.status = FindingStatus.RESOLVED
        finding.resolution_details = resolution_details
        finding.resolved_at = now
        finding.resolved_by = resolved_by
        finding.updated_at = now

        if verified:
            finding.verified = True
            finding.verified_by = verified_by
            finding.verified_at = now

        self._log_event("finding_resolved", {
            "finding_id": finding_id,
            "resolved_by": resolved_by,
            "verified": verified,
        })

        return finding

    def get_finding(self, finding_id: str) -> Optional[TestFinding]:
        """Get finding by ID."""
        return self._findings.get(finding_id)

    def get_findings_for_execution(
        self,
        execution_id: str,
    ) -> List[TestFinding]:
        """Get all findings for an execution."""
        return [f for f in self._findings.values() if f.execution_id == execution_id]

    def get_open_findings(self) -> List[TestFinding]:
        """Get all open findings."""
        return [
            f for f in self._findings.values()
            if f.status in (FindingStatus.OPEN, FindingStatus.IN_PROGRESS)
        ]

    def get_findings_by_severity(
        self,
        severity: FindingSeverity,
    ) -> List[TestFinding]:
        """Get findings by severity."""
        return [f for f in self._findings.values() if f.severity == severity]

    def get_overdue_findings(self) -> List[TestFinding]:
        """Get findings past their remediation deadline."""
        now = datetime.now(timezone.utc)
        overdue = []

        for finding in self._findings.values():
            if finding.status not in (FindingStatus.OPEN, FindingStatus.IN_PROGRESS):
                continue
            if not finding.remediation_deadline:
                continue

            deadline = datetime.fromisoformat(
                finding.remediation_deadline.replace("Z", "+00:00")
            )
            if now > deadline:
                overdue.append(finding)

        return overdue

    # =========================================================================
    # Testing Cycles
    # =========================================================================

    def create_testing_cycle(
        self,
        programme_id: str,
        cycle_name: str,
        cycle_type: str,
        start_date: str,
        end_date: str,
        test_ids: Optional[List[str]] = None,
    ) -> TestingCycle:
        """
        Create a testing cycle (quarterly/annual).

        Args:
            programme_id: Programme ID
            cycle_name: Cycle name
            cycle_type: Cycle type (quarterly, annual)
            start_date: Cycle start date
            end_date: Cycle end date
            test_ids: Test IDs to include

        Returns:
            Created TestingCycle
        """
        if programme_id not in self._programmes:
            raise ValueError(f"Programme {programme_id} not found")

        cycle = TestingCycle(
            programme_id=programme_id,
            cycle_name=cycle_name,
            cycle_type=cycle_type,
            start_date=start_date,
            end_date=end_date,
        )

        # If test IDs provided, create planned executions
        if test_ids:
            cycle.planned_count = len(test_ids)

        self._cycles[cycle.cycle_id] = cycle

        self._log_event("cycle_created", {
            "cycle_id": cycle.cycle_id,
            "programme_id": programme_id,
            "cycle_type": cycle_type,
        })

        return cycle

    def get_cycle(self, cycle_id: str) -> Optional[TestingCycle]:
        """Get cycle by ID."""
        return self._cycles.get(cycle_id)

    def update_cycle_progress(
        self,
        cycle_id: str,
    ) -> TestingCycle:
        """Update cycle progress metrics."""
        if cycle_id not in self._cycles:
            raise ValueError(f"Cycle {cycle_id} not found")

        cycle = self._cycles[cycle_id]

        # Count completed executions
        completed = [
            e for e in self._executions.values()
            if e.execution_id in cycle.planned_executions
            and e.status == TestStatus.COMPLETED
        ]

        cycle.completed_count = len(completed)

        if cycle.planned_count > 0:
            cycle.pass_rate = (
                sum(1 for e in completed if e.result == TestResult.PASSED)
                / cycle.planned_count * 100
            )

        # Update findings count
        findings = []
        for exec_id in cycle.completed_executions:
            findings.extend(self.get_findings_for_execution(exec_id))

        cycle.total_findings = len(findings)
        cycle.open_findings = sum(
            1 for f in findings
            if f.status in (FindingStatus.OPEN, FindingStatus.IN_PROGRESS)
        )
        cycle.resolved_findings = sum(
            1 for f in findings
            if f.status == FindingStatus.RESOLVED
        )

        # Update status
        if cycle.completed_count >= cycle.planned_count:
            cycle.status = "completed"
        elif cycle.completed_count > 0:
            cycle.status = "in_progress"

        return cycle

    # =========================================================================
    # Reporting and Compliance
    # =========================================================================

    def get_compliance_status(
        self,
        programme_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get Article 24 compliance status.

        Args:
            programme_id: Optional programme ID

        Returns:
            Compliance status dictionary
        """
        # Get all test categories
        required_categories = [
            TestCategory.VULNERABILITY_ASSESSMENT,
            TestCategory.NETWORK_SECURITY,
            TestCategory.GAP_ANALYSIS,
            TestCategory.PHYSICAL_SECURITY,
            TestCategory.SCENARIO_BASED,
            TestCategory.PENETRATION_TESTING,
        ]

        category_status = {}
        for category in required_categories:
            tests = self.get_test_definitions_by_category(category)
            if not tests:
                category_status[category.value] = {
                    "status": "missing",
                    "last_executed": None,
                    "compliant": False,
                }
                continue

            # Find most recent execution
            last_executed = None
            for test in tests:
                if test.last_executed_date:
                    if not last_executed or test.last_executed_date > last_executed:
                        last_executed = test.last_executed_date

            # Check if within required frequency
            compliant = False
            if last_executed:
                last_dt = datetime.fromisoformat(last_executed.replace("Z", "+00:00"))
                days_since = (datetime.now(timezone.utc) - last_dt).days
                min_freq = min(t.minimum_frequency for t in tests)
                freq_days = self._frequency_to_days(min_freq)
                compliant = days_since <= freq_days

            category_status[category.value] = {
                "status": "compliant" if compliant else "overdue",
                "last_executed": last_executed,
                "compliant": compliant,
            }

        # Calculate overall compliance
        compliant_count = sum(1 for s in category_status.values() if s["compliant"])
        total_count = len(category_status)

        return {
            "article_reference": "Article 24",
            "assessment_date": datetime.now(timezone.utc).isoformat(),
            "overall_compliant": compliant_count == total_count,
            "compliance_percentage": (compliant_count / total_count * 100) if total_count > 0 else 0,
            "category_status": category_status,
            "overdue_tests": len(self.get_overdue_tests()),
            "open_critical_findings": len(self.get_findings_by_severity(FindingSeverity.CRITICAL)),
            "open_high_findings": len(self.get_findings_by_severity(FindingSeverity.HIGH)),
            "overdue_findings": len(self.get_overdue_findings()),
        }

    def generate_annual_report(
        self,
        programme_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Generate annual testing report per Article 24(5).

        Args:
            programme_id: Programme ID
            year: Report year

        Returns:
            Annual report dictionary
        """
        if programme_id not in self._programmes:
            raise ValueError(f"Programme {programme_id} not found")

        programme = self._programmes[programme_id]

        # Filter executions for the year
        start_date = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        year_executions = [
            e for e in self._executions.values()
            if e.actual_start and
            start_date <= datetime.fromisoformat(e.actual_start.replace("Z", "+00:00")) <= end_date
        ]

        # Aggregate findings
        year_findings = []
        for execution in year_executions:
            year_findings.extend(self.get_findings_for_execution(execution.execution_id))

        # Calculate metrics
        total_tests = len(year_executions)
        completed_tests = sum(1 for e in year_executions if e.status == TestStatus.COMPLETED)
        passed_tests = sum(1 for e in year_executions if e.result == TestResult.PASSED)

        findings_by_severity = {
            "critical": sum(1 for f in year_findings if f.severity == FindingSeverity.CRITICAL),
            "high": sum(1 for f in year_findings if f.severity == FindingSeverity.HIGH),
            "medium": sum(1 for f in year_findings if f.severity == FindingSeverity.MEDIUM),
            "low": sum(1 for f in year_findings if f.severity == FindingSeverity.LOW),
            "informational": sum(1 for f in year_findings if f.severity == FindingSeverity.INFORMATIONAL),
        }

        resolved_findings = sum(1 for f in year_findings if f.status == FindingStatus.RESOLVED)

        # Tests by category
        tests_by_category = {}
        for category in TestCategory:
            count = sum(1 for e in year_executions if e.test_category == category)
            if count > 0:
                tests_by_category[category.value] = count

        return {
            "report_type": "annual_testing_report",
            "article_reference": "Article 24(5)",
            "programme_id": programme_id,
            "programme_name": programme.name,
            "entity_name": programme.entity_name,
            "year": year,
            "report_date": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_tests_planned": total_tests,
                "tests_completed": completed_tests,
                "tests_passed": passed_tests,
                "pass_rate": (passed_tests / completed_tests * 100) if completed_tests > 0 else 0,
                "total_findings": len(year_findings),
                "findings_resolved": resolved_findings,
                "resolution_rate": (resolved_findings / len(year_findings) * 100) if year_findings else 100,
            },
            "findings_by_severity": findings_by_severity,
            "tests_by_category": tests_by_category,
            "compliance_status": self.get_compliance_status(programme_id),
            "recommendations": self._generate_recommendations(year_findings, year_executions),
        }

    def _generate_recommendations(
        self,
        findings: List[TestFinding],
        executions: List[TestExecution],
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check for critical findings
        critical_count = sum(1 for f in findings if f.severity == FindingSeverity.CRITICAL)
        if critical_count > 0:
            recommendations.append(
                f"Address {critical_count} critical findings as immediate priority"
            )

        # Check for overdue findings
        overdue = self.get_overdue_findings()
        if overdue:
            recommendations.append(
                f"Resolve {len(overdue)} overdue findings to meet SLA commitments"
            )

        # Check test coverage
        categories_tested = set(e.test_category for e in executions)
        missing = set(TestCategory) - categories_tested
        if missing:
            recommendations.append(
                f"Include testing for categories: {', '.join(c.value for c in missing)}"
            )

        # Check for external testing
        external_tests = sum(1 for e in executions if e.tester_type == TesterType.EXTERNAL)
        if external_tests == 0:
            recommendations.append(
                "Include external independent testing per Article 24(4)"
            )

        return recommendations

    def export_test_data(
        self,
        execution_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export test data for regulatory purposes.

        Args:
            execution_id: Optional specific execution ID

        Returns:
            Export data dictionary
        """
        if execution_id:
            if execution_id not in self._executions:
                raise ValueError(f"Execution {execution_id} not found")
            executions = [asdict(self._executions[execution_id])]
            findings = [
                asdict(f) for f in self._findings.values()
                if f.execution_id == execution_id
            ]
        else:
            executions = [asdict(e) for e in self._executions.values()]
            findings = [asdict(f) for f in self._findings.values()]

        return {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "article_reference": "Article 24",
            "executions": executions,
            "findings": findings,
            "test_definitions": [asdict(t) for t in self._test_definitions.values()],
            "scopes": [asdict(s) for s in self._scopes.values()],
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_testing_statistics(
        self,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get testing statistics for a period.

        Args:
            period_start: Start of period
            period_end: End of period

        Returns:
            Statistics dictionary
        """
        if not period_end:
            period_end = datetime.now(timezone.utc).isoformat()
        if not period_start:
            period_start = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()

        start_dt = datetime.fromisoformat(period_start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(period_end.replace("Z", "+00:00"))

        # Filter executions
        period_executions = [
            e for e in self._executions.values()
            if e.created_at and
            start_dt <= datetime.fromisoformat(e.created_at.replace("Z", "+00:00")) <= end_dt
        ]

        # Filter findings
        period_findings = [
            f for f in self._findings.values()
            if f.created_at and
            start_dt <= datetime.fromisoformat(f.created_at.replace("Z", "+00:00")) <= end_dt
        ]

        # Calculate statistics
        completed = sum(1 for e in period_executions if e.status == TestStatus.COMPLETED)
        passed = sum(1 for e in period_executions if e.result == TestResult.PASSED)

        return {
            "period_start": period_start,
            "period_end": period_end,
            "total_executions": len(period_executions),
            "completed_executions": completed,
            "completion_rate": (completed / len(period_executions) * 100) if period_executions else 0,
            "passed_tests": passed,
            "pass_rate": (passed / completed * 100) if completed > 0 else 0,
            "total_findings": len(period_findings),
            "findings_by_severity": {
                "critical": sum(1 for f in period_findings if f.severity == FindingSeverity.CRITICAL),
                "high": sum(1 for f in period_findings if f.severity == FindingSeverity.HIGH),
                "medium": sum(1 for f in period_findings if f.severity == FindingSeverity.MEDIUM),
                "low": sum(1 for f in period_findings if f.severity == FindingSeverity.LOW),
            },
            "findings_resolved": sum(1 for f in period_findings if f.status == FindingStatus.RESOLVED),
            "mean_time_to_resolve_days": self._calculate_mttr(period_findings),
        }

    def _calculate_mttr(self, findings: List[TestFinding]) -> float:
        """Calculate mean time to resolve findings."""
        resolved = [
            f for f in findings
            if f.status == FindingStatus.RESOLVED and f.resolved_at
        ]

        if not resolved:
            return 0.0

        total_days = 0.0
        for finding in resolved:
            created = datetime.fromisoformat(finding.created_at.replace("Z", "+00:00"))
            resolved_at = datetime.fromisoformat(finding.resolved_at.replace("Z", "+00:00"))
            total_days += (resolved_at - created).days

        return total_days / len(resolved)

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a resilience testing event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"resilience_testing_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_resilience_testing_programme(
    config: Optional[ResilienceTestingConfig] = None,
) -> DORAResilienceTestingProgramme:
    """
    Create a DORAResilienceTestingProgramme instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAResilienceTestingProgramme instance
    """
    return DORAResilienceTestingProgramme(config=config)


def get_test_categories() -> List[TestCategory]:
    """
    Get list of test categories per Article 24(1).

    Returns:
        List of TestCategory enums
    """
    return list(TestCategory)


def get_finding_severities() -> List[FindingSeverity]:
    """
    Get list of finding severity levels.

    Returns:
        List of FindingSeverity enums
    """
    return list(FindingSeverity)


def get_default_test_definitions() -> List[Dict[str, Any]]:
    """
    Get default test definitions per Article 24(1).

    Returns:
        List of default test definition dictionaries
    """
    return DEFAULT_TEST_DEFINITIONS.copy()

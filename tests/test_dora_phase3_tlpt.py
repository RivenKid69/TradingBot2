# -*- coding: utf-8 -*-
"""
Comprehensive Tests for DORA Phase 3 - Threat-Led Penetration Testing (Article 26).

Tests for Threat-Led Penetration Testing (TLPT) per Article 26 of DORA.

Coverage includes:
- TLPT Phase enumeration (TIBER-EU framework)
- TLPT Status management
- Threat Actor types and capabilities
- Attack Techniques (MITRE ATT&CK aligned)
- Attack Outcomes
- Finding Severity and Categories
- TLPT Scope definition
- Threat Intelligence Reports
- Red Team Scenarios
- Attack Actions
- TLPT Findings
- Purple Team Sessions (Article 26(5))
- TLPT Engagements
- TLPT Attestations (Article 26(6))

References:
- Article 26 DORA: https://www.digital-operational-resilience-act.com/Article_26.html
- RTS on TLPT (CDR 2024/2698)
- TIBER-EU Framework: https://www.ecb.europa.eu/paym/cyber-resilience/tiber-eu/html/index.en.html
"""


import pytest

pytest.skip(
    "Legacy DORA test - uses deprecated imports from services.dora.* "
    "These modules have been migrated to services.dora_integration.*. "
    "See tests/dora/ and tests/dora_integration/ for current tests.",
    allow_module_level=True,
)


import pytest
from datetime import datetime, timedelta, timezone

from services.dora.tlpt import (
    TLPTPhase,
    TLPTStatus,
    ThreatActorType,
    ThreatActorCapability,
    AttackTechnique,
    AttackOutcome,
    TLPTFindingSeverity,
    FindingCategory,
    TLPTScope,
    ThreatIntelligenceReport,
    RedTeamScenario,
    AttackAction,
    TLPTFinding,
    PurpleTeamSession,
    TLPTEngagement,
    TLPTAttestation,
    TLPTConfig,
    DORAThreadLedPenetrationTesting,
    create_tlpt,
)


# =============================================================================
# TLPT Phase Tests (TIBER-EU Framework)
# =============================================================================


class TestTLPTPhase:
    """Tests for TLPT phases per TIBER-EU framework."""

    def test_preparation_phase(self):
        """Test preparation phase value."""
        assert TLPTPhase.PREPARATION.value == "preparation"

    def test_threat_intelligence_phase(self):
        """Test threat intelligence phase value."""
        assert TLPTPhase.THREAT_INTELLIGENCE.value == "threat_intelligence"

    def test_red_team_testing_phase(self):
        """Test red team testing phase value."""
        assert TLPTPhase.RED_TEAM_TESTING.value == "red_team_testing"

    def test_closure_phase(self):
        """Test closure phase value."""
        assert TLPTPhase.CLOSURE.value == "closure"

    def test_all_four_phases_exist(self):
        """Test that all 4 TIBER-EU phases are defined."""
        assert len(TLPTPhase) == 4


# =============================================================================
# TLPT Status Tests
# =============================================================================


class TestTLPTStatus:
    """Tests for TLPT engagement status."""

    def test_draft_status(self):
        """Test draft status value."""
        assert TLPTStatus.DRAFT.value == "draft"

    def test_planning_status(self):
        """Test planning status value."""
        assert TLPTStatus.PLANNING.value == "planning"

    def test_scoping_status(self):
        """Test scoping status value."""
        assert TLPTStatus.SCOPING.value == "scoping"

    def test_approved_status(self):
        """Test approved status value."""
        assert TLPTStatus.APPROVED.value == "approved"

    def test_ti_phase_status(self):
        """Test threat intelligence phase status value."""
        assert TLPTStatus.TI_PHASE.value == "threat_intelligence_phase"

    def test_rt_phase_status(self):
        """Test red team phase status value."""
        assert TLPTStatus.RT_PHASE.value == "red_team_phase"

    def test_purple_team_status(self):
        """Test purple team status value."""
        assert TLPTStatus.PURPLE_TEAM.value == "purple_team"

    def test_completed_status(self):
        """Test completed status value."""
        assert TLPTStatus.COMPLETED.value == "completed"


# =============================================================================
# Threat Actor Type Tests
# =============================================================================


class TestThreatActorType:
    """Tests for threat actor type enumeration."""

    def test_nation_state_actor(self):
        """Test nation state actor type."""
        assert ThreatActorType.NATION_STATE.value == "nation_state"

    def test_organized_crime_actor(self):
        """Test organized crime actor type."""
        assert ThreatActorType.ORGANIZED_CRIME.value == "organized_crime"

    def test_hacktivist_actor(self):
        """Test hacktivist actor type."""
        assert ThreatActorType.HACKTIVIST.value == "hacktivist"

    def test_insider_actor(self):
        """Test insider threat actor type."""
        assert ThreatActorType.INSIDER.value == "insider"

    def test_competitor_actor(self):
        """Test competitor actor type."""
        assert ThreatActorType.COMPETITOR.value == "competitor"

    def test_cyber_terrorist_actor(self):
        """Test cyber terrorist actor type."""
        assert ThreatActorType.CYBER_TERRORIST.value == "cyber_terrorist"


# =============================================================================
# Threat Actor Capability Tests
# =============================================================================


class TestThreatActorCapability:
    """Tests for threat actor capability enumeration."""

    def test_advanced_capability(self):
        """Test advanced (APT-level) capability."""
        assert ThreatActorCapability.ADVANCED.value == "advanced"

    def test_moderate_capability(self):
        """Test moderate (organized groups) capability."""
        assert ThreatActorCapability.MODERATE.value == "moderate"

    def test_basic_capability(self):
        """Test basic (opportunistic) capability."""
        assert ThreatActorCapability.BASIC.value == "basic"

    def test_unsophisticated_capability(self):
        """Test unsophisticated capability."""
        assert ThreatActorCapability.UNSOPHISTICATED.value == "unsophisticated"


# =============================================================================
# Attack Technique Tests (MITRE ATT&CK Aligned)
# =============================================================================


class TestAttackTechnique:
    """Tests for attack technique enumeration (MITRE ATT&CK aligned)."""

    def test_initial_access_technique(self):
        """Test initial access technique."""
        assert AttackTechnique.INITIAL_ACCESS.value == "initial_access"

    def test_execution_technique(self):
        """Test execution technique."""
        assert AttackTechnique.EXECUTION.value == "execution"

    def test_persistence_technique(self):
        """Test persistence technique."""
        assert AttackTechnique.PERSISTENCE.value == "persistence"

    def test_privilege_escalation_technique(self):
        """Test privilege escalation technique."""
        assert AttackTechnique.PRIVILEGE_ESCALATION.value == "privilege_escalation"

    def test_defense_evasion_technique(self):
        """Test defense evasion technique."""
        assert AttackTechnique.DEFENSE_EVASION.value == "defense_evasion"

    def test_credential_access_technique(self):
        """Test credential access technique."""
        assert AttackTechnique.CREDENTIAL_ACCESS.value == "credential_access"

    def test_lateral_movement_technique(self):
        """Test lateral movement technique."""
        assert AttackTechnique.LATERAL_MOVEMENT.value == "lateral_movement"

    def test_exfiltration_technique(self):
        """Test exfiltration technique."""
        assert AttackTechnique.EXFILTRATION.value == "exfiltration"

    def test_impact_technique(self):
        """Test impact technique."""
        assert AttackTechnique.IMPACT.value == "impact"


# =============================================================================
# Attack Outcome Tests
# =============================================================================


class TestAttackOutcome:
    """Tests for attack outcome enumeration."""

    def test_success_outcome(self):
        """Test success outcome."""
        assert AttackOutcome.SUCCESS.value == "success"

    def test_partial_success_outcome(self):
        """Test partial success outcome."""
        assert AttackOutcome.PARTIAL_SUCCESS.value == "partial_success"

    def test_detected_but_successful_outcome(self):
        """Test detected but successful outcome."""
        assert AttackOutcome.DETECTED_BUT_SUCCESSFUL.value == "detected_but_successful"

    def test_blocked_outcome(self):
        """Test blocked outcome."""
        assert AttackOutcome.BLOCKED.value == "blocked"


# =============================================================================
# TLPT Finding Severity Tests
# =============================================================================


class TestTLPTFindingSeverity:
    """Tests for TLPT finding severity enumeration."""

    def test_critical_severity(self):
        """Test critical severity."""
        assert TLPTFindingSeverity.CRITICAL.value == "critical"

    def test_high_severity(self):
        """Test high severity."""
        assert TLPTFindingSeverity.HIGH.value == "high"

    def test_medium_severity(self):
        """Test medium severity."""
        assert TLPTFindingSeverity.MEDIUM.value == "medium"

    def test_low_severity(self):
        """Test low severity."""
        assert TLPTFindingSeverity.LOW.value == "low"

    def test_informational_severity(self):
        """Test informational severity."""
        assert TLPTFindingSeverity.INFORMATIONAL.value == "informational"


# =============================================================================
# Finding Category Tests
# =============================================================================


class TestFindingCategory:
    """Tests for finding category enumeration."""

    def test_technical_vulnerability_category(self):
        """Test technical vulnerability category."""
        assert FindingCategory.TECHNICAL_VULNERABILITY.value == "technical_vulnerability"

    def test_configuration_weakness_category(self):
        """Test configuration weakness category."""
        assert FindingCategory.CONFIGURATION_WEAKNESS.value == "configuration_weakness"

    def test_process_deficiency_category(self):
        """Test process deficiency category."""
        assert FindingCategory.PROCESS_DEFICIENCY.value == "process_deficiency"

    def test_detection_gap_category(self):
        """Test detection gap category."""
        assert FindingCategory.DETECTION_GAP.value == "detection_gap"

    def test_response_deficiency_category(self):
        """Test response deficiency category."""
        assert FindingCategory.RESPONSE_DEFICIENCY.value == "response_deficiency"


# =============================================================================
# TLPT Scope Tests (Article 26(1))
# =============================================================================


class TestTLPTScope:
    """Tests for TLPT scope definition per Article 26(1)."""

    def test_scope_creation_with_defaults(self):
        """Test scope creation with default values."""
        scope = TLPTScope()
        assert scope.scope_id != ""
        assert scope.name == ""

    def test_scope_with_name(self):
        """Test scope with specific name."""
        scope = TLPTScope(
            name="Production Trading Environment",
            description="Full TLPT scope covering critical trading systems",
        )
        assert scope.name == "Production Trading Environment"

    def test_scope_auto_generates_id(self):
        """Test that scope auto-generates a unique ID."""
        scope1 = TLPTScope()
        scope2 = TLPTScope()
        assert scope1.scope_id != scope2.scope_id


# =============================================================================
# Threat Intelligence Report Tests
# =============================================================================


class TestThreatIntelligenceReport:
    """Tests for threat intelligence report structure."""

    def test_report_creation_with_defaults(self):
        """Test report creation with default values."""
        report = ThreatIntelligenceReport()
        assert report.report_id != ""

    def test_report_with_threat_actors(self):
        """Test report with threat actor profiles."""
        report = ThreatIntelligenceReport(
            title="Financial Sector Threat Landscape",
            threat_actors=[ThreatActorType.NATION_STATE, ThreatActorType.ORGANIZED_CRIME],
        )
        assert ThreatActorType.NATION_STATE in report.threat_actors

    def test_report_auto_generates_id(self):
        """Test that report auto-generates a unique ID."""
        report1 = ThreatIntelligenceReport()
        report2 = ThreatIntelligenceReport()
        assert report1.report_id != report2.report_id


# =============================================================================
# Red Team Scenario Tests
# =============================================================================


class TestRedTeamScenario:
    """Tests for red team scenario structure."""

    def test_scenario_creation_with_defaults(self):
        """Test scenario creation with default values."""
        scenario = RedTeamScenario()
        assert scenario.scenario_id != ""

    def test_scenario_with_threat_actor(self):
        """Test scenario with specific threat actor."""
        scenario = RedTeamScenario(
            name="APT Simulation",
            threat_actor=ThreatActorType.NATION_STATE,
            capability=ThreatActorCapability.ADVANCED,
        )
        assert scenario.threat_actor == ThreatActorType.NATION_STATE

    def test_scenario_auto_generates_id(self):
        """Test that scenario auto-generates a unique ID."""
        scenario1 = RedTeamScenario()
        scenario2 = RedTeamScenario()
        assert scenario1.scenario_id != scenario2.scenario_id


# =============================================================================
# Attack Action Tests
# =============================================================================


class TestAttackAction:
    """Tests for attack action structure."""

    def test_action_creation_with_defaults(self):
        """Test action creation with default values."""
        action = AttackAction()
        assert action.action_id != ""

    def test_action_with_technique(self):
        """Test action with specific technique."""
        action = AttackAction(name="Phishing Campaign", technique=AttackTechnique.INITIAL_ACCESS)
        assert action.technique == AttackTechnique.INITIAL_ACCESS

    def test_action_with_outcome(self):
        """Test action with outcome."""
        action = AttackAction(
            name="Credential Dump",
            technique=AttackTechnique.CREDENTIAL_ACCESS,
            outcome=AttackOutcome.SUCCESS,
        )
        assert action.outcome == AttackOutcome.SUCCESS

    def test_action_auto_generates_id(self):
        """Test that action auto-generates a unique ID."""
        action1 = AttackAction()
        action2 = AttackAction()
        assert action1.action_id != action2.action_id


# =============================================================================
# TLPT Finding Tests
# =============================================================================


class TestTLPTFinding:
    """Tests for TLPT finding structure."""

    def test_finding_creation_with_defaults(self):
        """Test finding creation with default values."""
        finding = TLPTFinding()
        assert finding.finding_id != ""
        assert finding.severity == TLPTFindingSeverity.MEDIUM

    def test_finding_with_severity(self):
        """Test finding with specific severity."""
        finding = TLPTFinding(
            title="Weak Network Segmentation", severity=TLPTFindingSeverity.CRITICAL
        )
        assert finding.severity == TLPTFindingSeverity.CRITICAL

    def test_finding_with_category(self):
        """Test finding with specific category."""
        finding = TLPTFinding(title="Missing MFA", category=FindingCategory.ACCESS_CONTROL)
        assert finding.category == FindingCategory.ACCESS_CONTROL

    def test_finding_auto_generates_id(self):
        """Test that finding auto-generates a unique ID."""
        finding1 = TLPTFinding()
        finding2 = TLPTFinding()
        assert finding1.finding_id != finding2.finding_id


# =============================================================================
# Purple Team Session Tests (Article 26(5))
# =============================================================================


class TestPurpleTeamSession:
    """Tests for purple team session per Article 26(5)."""

    def test_session_creation_with_defaults(self):
        """Test session creation with default values."""
        session = PurpleTeamSession()
        assert session.session_id != ""

    def test_session_with_engagement_id(self):
        """Test session with engagement ID."""
        session = PurpleTeamSession(engagement_id="TLPT-001", name="Purple Team Exercise Q1 2025")
        assert session.engagement_id == "TLPT-001"

    def test_session_auto_generates_id(self):
        """Test that session auto-generates a unique ID."""
        session1 = PurpleTeamSession()
        session2 = PurpleTeamSession()
        assert session1.session_id != session2.session_id


# =============================================================================
# TLPT Engagement Tests
# =============================================================================


class TestTLPTEngagement:
    """Tests for TLPT engagement structure."""

    def test_engagement_creation_with_defaults(self):
        """Test engagement creation with default values."""
        engagement = TLPTEngagement()
        assert engagement.engagement_id != ""
        assert engagement.status == TLPTStatus.DRAFT

    def test_engagement_with_name(self):
        """Test engagement with specific name."""
        engagement = TLPTEngagement(
            name="Annual TLPT 2025", description="Mandatory TLPT per Article 26"
        )
        assert engagement.name == "Annual TLPT 2025"

    def test_engagement_with_status(self):
        """Test engagement with specific status."""
        engagement = TLPTEngagement(name="TLPT Q1", status=TLPTStatus.PLANNING)
        assert engagement.status == TLPTStatus.PLANNING

    def test_engagement_auto_generates_id(self):
        """Test that engagement auto-generates a unique ID."""
        eng1 = TLPTEngagement()
        eng2 = TLPTEngagement()
        assert eng1.engagement_id != eng2.engagement_id


# =============================================================================
# TLPT Attestation Tests (Article 26(6))
# =============================================================================


class TestTLPTAttestation:
    """Tests for TLPT attestation per Article 26(6)."""

    def test_attestation_creation_with_defaults(self):
        """Test attestation creation with default values."""
        attestation = TLPTAttestation()
        assert attestation.attestation_id != ""

    def test_attestation_with_engagement_id(self):
        """Test attestation with engagement ID."""
        attestation = TLPTAttestation(engagement_id="TLPT-001")
        assert attestation.engagement_id == "TLPT-001"

    def test_attestation_auto_generates_id(self):
        """Test that attestation auto-generates a unique ID."""
        att1 = TLPTAttestation()
        att2 = TLPTAttestation()
        assert att1.attestation_id != att2.attestation_id


# =============================================================================
# TLPT Config Tests
# =============================================================================


class TestTLPTConfig:
    """Tests for TLPT configuration."""

    def test_config_creation_with_defaults(self):
        """Test config creation with default values."""
        config = TLPTConfig()
        assert config is not None

    def test_config_with_entity_id(self):
        """Test config with entity ID."""
        config = TLPTConfig(entity_id="ENT-001")
        assert config.entity_id == "ENT-001"


# =============================================================================
# DORAThreadLedPenetrationTesting Creation Tests
# =============================================================================


class TestDORATLPTCreation:
    """Tests for DORAThreadLedPenetrationTesting creation."""

    def test_create_with_factory_function(self):
        """Test creation via factory function."""
        tlpt = create_tlpt()
        assert tlpt is not None
        assert isinstance(tlpt, DORAThreadLedPenetrationTesting)

    def test_create_with_config(self):
        """Test creation with custom config."""
        config = TLPTConfig(entity_id="TEST-001")
        tlpt = create_tlpt(config)
        assert tlpt is not None
        assert tlpt.config.entity_id == "TEST-001"

    def test_has_required_methods(self):
        """Test that TLPT has required methods."""
        tlpt = create_tlpt()
        assert hasattr(tlpt, "create_engagement")
        assert hasattr(tlpt, "define_scope")
        assert hasattr(tlpt, "add_threat_intelligence")


# =============================================================================
# DORAThreadLedPenetrationTesting Engagement Tests
# =============================================================================


class TestDORATLPTEngagement:
    """Tests for TLPT engagement management."""

    def test_create_engagement(self):
        """Test creating a TLPT engagement."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="TLPT 2025", description="Annual TLPT engagement")
        assert engagement is not None
        assert engagement.name == "TLPT 2025"

    def test_get_engagement(self):
        """Test retrieving an engagement."""
        tlpt = create_tlpt()
        created = tlpt.create_engagement(name="Test TLPT")
        retrieved = tlpt.get_engagement(created.engagement_id)
        assert retrieved is not None
        assert retrieved.name == "Test TLPT"

    def test_list_engagements(self):
        """Test listing all engagements."""
        tlpt = create_tlpt()
        tlpt.create_engagement(name="TLPT 1")
        tlpt.create_engagement(name="TLPT 2")
        engagements = tlpt.list_engagements()
        assert len(engagements) >= 2


# =============================================================================
# DORAThreadLedPenetrationTesting Scope Tests
# =============================================================================


class TestDORATLPTScope:
    """Tests for TLPT scope management."""

    def test_define_scope(self):
        """Test defining TLPT scope."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Scope Test TLPT")
        scope = tlpt.define_scope(
            engagement_id=engagement.engagement_id,
            name="Production Environment",
            systems=["trading_platform", "risk_engine"],
        )
        assert scope is not None
        assert "trading_platform" in scope.systems

    def test_update_scope(self):
        """Test updating TLPT scope."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Update Scope Test")
        scope = tlpt.define_scope(engagement_id=engagement.engagement_id, name="Initial Scope")
        updated = tlpt.update_scope(scope_id=scope.scope_id, systems=["new_system"])
        assert "new_system" in updated.systems


# =============================================================================
# DORAThreadLedPenetrationTesting Threat Intelligence Tests
# =============================================================================


class TestDORATLPTThreatIntelligence:
    """Tests for threat intelligence management."""

    def test_add_threat_intelligence(self):
        """Test adding threat intelligence report."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="TI Test TLPT")
        ti_report = tlpt.add_threat_intelligence(
            engagement_id=engagement.engagement_id,
            title="Financial Sector Threats Q1 2025",
            threat_actors=[ThreatActorType.ORGANIZED_CRIME],
        )
        assert ti_report is not None
        assert ThreatActorType.ORGANIZED_CRIME in ti_report.threat_actors

    def test_get_threat_intelligence(self):
        """Test retrieving threat intelligence for engagement."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="TI Retrieve Test")
        tlpt.add_threat_intelligence(engagement_id=engagement.engagement_id, title="Threat Report")
        reports = tlpt.get_threat_intelligence(engagement.engagement_id)
        assert len(reports) >= 1


# =============================================================================
# DORAThreadLedPenetrationTesting Red Team Tests
# =============================================================================


class TestDORATLPTRedTeam:
    """Tests for red team scenario management."""

    def test_add_scenario(self):
        """Test adding a red team scenario."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Scenario Test TLPT")
        scenario = tlpt.add_scenario(
            engagement_id=engagement.engagement_id,
            name="APT Simulation",
            threat_actor=ThreatActorType.NATION_STATE,
            capability=ThreatActorCapability.ADVANCED,
        )
        assert scenario is not None
        assert scenario.threat_actor == ThreatActorType.NATION_STATE

    def test_add_attack_action(self):
        """Test adding an attack action to a scenario."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Action Test TLPT")
        scenario = tlpt.add_scenario(engagement_id=engagement.engagement_id, name="Test Scenario")
        action = tlpt.add_attack_action(
            scenario_id=scenario.scenario_id,
            name="Spear Phishing",
            technique=AttackTechnique.INITIAL_ACCESS,
            outcome=AttackOutcome.SUCCESS,
        )
        assert action is not None
        assert action.technique == AttackTechnique.INITIAL_ACCESS


# =============================================================================
# DORAThreadLedPenetrationTesting Purple Team Tests (Article 26(5))
# =============================================================================


class TestDORATLPTPurpleTeam:
    """Tests for purple team activities per Article 26(5)."""

    def test_create_purple_team_session(self):
        """Test creating a purple team session."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Purple Team Test TLPT")
        session = tlpt.create_purple_team_session(
            engagement_id=engagement.engagement_id, name="Purple Team Exercise"
        )
        assert session is not None
        assert session.engagement_id == engagement.engagement_id

    def test_purple_team_is_mandatory(self):
        """Test that purple teaming is mandatory per Article 26(5)."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Mandatory Purple Team Test")
        # Purple team should be required for valid TLPT
        requirements = tlpt.get_engagement_requirements(engagement.engagement_id)
        assert "purple_team" in requirements or requirements.get("purple_team_required") == True


# =============================================================================
# DORAThreadLedPenetrationTesting Finding Tests
# =============================================================================


class TestDORATLPTFindings:
    """Tests for TLPT finding management."""

    def test_add_finding(self):
        """Test adding a finding."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Finding Test TLPT")
        finding = tlpt.add_finding(
            engagement_id=engagement.engagement_id,
            title="Weak Network Segmentation",
            severity=TLPTFindingSeverity.HIGH,
            category=FindingCategory.CONFIGURATION_WEAKNESS,
        )
        assert finding is not None
        assert finding.severity == TLPTFindingSeverity.HIGH

    def test_get_findings(self):
        """Test retrieving findings for engagement."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Get Findings Test")
        tlpt.add_finding(
            engagement_id=engagement.engagement_id,
            title="Finding 1",
            severity=TLPTFindingSeverity.MEDIUM,
        )
        findings = tlpt.get_findings(engagement.engagement_id)
        assert len(findings) >= 1


# =============================================================================
# DORAThreadLedPenetrationTesting Attestation Tests (Article 26(6))
# =============================================================================


class TestDORATLPTAttestation:
    """Tests for TLPT attestation per Article 26(6)."""

    def test_create_attestation(self):
        """Test creating an attestation."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Attestation Test TLPT")
        attestation = tlpt.create_attestation(engagement_id=engagement.engagement_id)
        assert attestation is not None
        assert attestation.engagement_id == engagement.engagement_id

    def test_submit_attestation(self):
        """Test submitting attestation to competent authority."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Submit Attestation Test")
        attestation = tlpt.create_attestation(engagement_id=engagement.engagement_id)
        result = tlpt.submit_attestation(
            attestation_id=attestation.attestation_id, authority_id="NCA-001"
        )
        assert result is not None
        assert "submitted" in str(result).lower() or result.get("status") == "submitted"


# =============================================================================
# DORAThreadLedPenetrationTesting Lifecycle Tests
# =============================================================================


class TestDORATLPTLifecycle:
    """Tests for complete TLPT engagement lifecycle."""

    def test_advance_to_threat_intelligence_phase(self):
        """Test advancing engagement to TI phase."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Lifecycle Test TLPT")
        tlpt.define_scope(engagement_id=engagement.engagement_id, name="Test Scope")
        advanced = tlpt.advance_phase(
            engagement_id=engagement.engagement_id, target_phase=TLPTPhase.THREAT_INTELLIGENCE
        )
        assert advanced.status == TLPTStatus.TI_PHASE

    def test_advance_to_red_team_phase(self):
        """Test advancing engagement to red team phase."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="RT Phase Test TLPT")
        tlpt.define_scope(engagement_id=engagement.engagement_id, name="Test Scope")
        tlpt.add_threat_intelligence(engagement_id=engagement.engagement_id, title="TI Report")
        tlpt.advance_phase(
            engagement_id=engagement.engagement_id, target_phase=TLPTPhase.THREAT_INTELLIGENCE
        )
        advanced = tlpt.advance_phase(
            engagement_id=engagement.engagement_id, target_phase=TLPTPhase.RED_TEAM_TESTING
        )
        assert advanced.status == TLPTStatus.RT_PHASE


# =============================================================================
# Three-Year Cycle Tests (Article 26(1))
# =============================================================================


class TestTLPTThreeYearCycle:
    """Tests for three-year TLPT cycle per Article 26(1)."""

    def test_tlpt_minimum_frequency(self):
        """Test TLPT minimum frequency is every 3 years."""
        tlpt = create_tlpt()
        requirements = tlpt.get_tlpt_requirements()
        assert requirements.get("minimum_cycle_years") == 3 or "3" in str(
            requirements.get("frequency", "")
        )


# =============================================================================
# Reporting Tests
# =============================================================================


class TestTLPTReporting:
    """Tests for TLPT reporting functionality."""

    def test_generate_engagement_report(self):
        """Test generating engagement report."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Report Test TLPT")
        report = tlpt.generate_engagement_report(engagement.engagement_id)
        assert report is not None
        assert "engagement_id" in report

    def test_export_findings(self):
        """Test exporting findings."""
        tlpt = create_tlpt()
        engagement = tlpt.create_engagement(name="Export Test TLPT")
        tlpt.add_finding(
            engagement_id=engagement.engagement_id,
            title="Test Finding",
            severity=TLPTFindingSeverity.HIGH,
        )
        exported = tlpt.export_findings(engagement.engagement_id)
        assert isinstance(exported, (dict, list))

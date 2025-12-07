# -*- coding: utf-8 -*-
"""
Tests for AI Act Quality Management System (Article 17).

Comprehensive test coverage for QMS functionality including:
- Procedure management
- Audit management
- Design control
- Change management
- CAPA management
- Resource management
- Accountability framework
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

from services.ai_act.qms import (
    QMSElementType,
    ProcedureStatus,
    AuditType,
    AuditStatus,
    FindingSeverity,
    ChangeType,
    ChangeImpact,
    CAPAType,
    CAPAStatus,
    QMSProcedure,
    AuditFinding,
    QMSAudit,
    DesignReview,
    ChangeRequest,
    CAPARecord,
    ResourceRecord,
    AccountabilityRecord,
    QMSConfig,
    QualityManagementSystem,
    create_qms,
    get_default_qms_procedures,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def qms_config(temp_log_dir):
    """Create QMS configuration for testing."""
    return QMSConfig(
        organization_name="Test Organization",
        organization_id="ORG-001",
        qms_version="1.0",
        procedure_review_months=12,
        audit_frequency_months=6,
        max_open_critical_findings=0,
        max_open_major_findings=5,
        log_path=temp_log_dir,
    )


@pytest.fixture
def qms(qms_config):
    """Create QMS instance for testing."""
    return QualityManagementSystem(qms_config)


# =============================================================================
# Test Enums
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_qms_element_types(self):
        """Test QMS element types match Article 17."""
        assert len(QMSElementType) == 12
        assert QMSElementType.REGULATORY_COMPLIANCE.value == "regulatory_compliance"
        assert QMSElementType.DESIGN_CONTROL.value == "design_control"
        assert QMSElementType.DEVELOPMENT_QA.value == "development_qa"
        assert QMSElementType.TESTING_VALIDATION.value == "testing_validation"
        assert QMSElementType.DATA_MANAGEMENT.value == "data_management"
        assert QMSElementType.RISK_MANAGEMENT.value == "risk_management"
        assert QMSElementType.POST_MARKET_MONITORING.value == "post_market_monitoring"
        assert QMSElementType.INCIDENT_REPORTING.value == "incident_reporting"
        assert QMSElementType.AUTHORITY_COMMUNICATION.value == "authority_communication"
        assert QMSElementType.RECORD_KEEPING.value == "record_keeping"
        assert QMSElementType.RESOURCE_MANAGEMENT.value == "resource_management"
        assert QMSElementType.ACCOUNTABILITY.value == "accountability"

    def test_procedure_status_values(self):
        """Test procedure status values."""
        assert ProcedureStatus.DRAFT.value == "draft"
        assert ProcedureStatus.UNDER_REVIEW.value == "under_review"
        assert ProcedureStatus.APPROVED.value == "approved"
        assert ProcedureStatus.EFFECTIVE.value == "effective"
        assert ProcedureStatus.SUPERSEDED.value == "superseded"
        assert ProcedureStatus.RETIRED.value == "retired"

    def test_audit_types(self):
        """Test audit types."""
        assert AuditType.INTERNAL.value == "internal"
        assert AuditType.EXTERNAL.value == "external"
        assert AuditType.REGULATORY.value == "regulatory"
        assert AuditType.SUPPLIER.value == "supplier"

    def test_finding_severity_values(self):
        """Test finding severity ordering."""
        assert FindingSeverity.OBSERVATION.value < FindingSeverity.MINOR.value
        assert FindingSeverity.MINOR.value < FindingSeverity.MAJOR.value
        assert FindingSeverity.MAJOR.value < FindingSeverity.CRITICAL.value

    def test_change_impact_levels(self):
        """Test change impact levels."""
        assert ChangeImpact.NONE.value == 0
        assert ChangeImpact.LOW.value == 1
        assert ChangeImpact.MEDIUM.value == 2
        assert ChangeImpact.HIGH.value == 3
        assert ChangeImpact.SUBSTANTIAL.value == 4

    def test_capa_types(self):
        """Test CAPA types."""
        assert CAPAType.CORRECTIVE.value == "corrective"
        assert CAPAType.PREVENTIVE.value == "preventive"
        assert CAPAType.IMPROVEMENT.value == "improvement"


# =============================================================================
# Test Data Classes
# =============================================================================

class TestDataClasses:
    """Test data class initialization and properties."""

    def test_qms_procedure_creation(self):
        """Test QMS procedure creation."""
        procedure = QMSProcedure(
            procedure_id="",
            element_type=QMSElementType.RISK_MANAGEMENT,
            title="Risk Assessment Procedure",
            description="Procedure for conducting risk assessments",
        )
        assert procedure.procedure_id.startswith("QMS-RIS-")
        assert procedure.status == ProcedureStatus.DRAFT
        assert procedure.version == "1.0"
        assert procedure.created_at != ""

    def test_audit_finding_creation(self):
        """Test audit finding creation."""
        finding = AuditFinding(
            finding_id="",
            audit_id="AUDIT-001",
            severity=FindingSeverity.MAJOR,
            title="Missing documentation",
            description="Required documentation not found",
        )
        assert finding.finding_id.startswith("FIND-")
        assert not finding.is_closed
        assert finding.created_at != ""

    def test_qms_audit_creation(self):
        """Test QMS audit creation."""
        audit = QMSAudit(
            audit_id="",
            audit_type=AuditType.INTERNAL,
            title="Q1 Internal Audit",
            scope="Risk Management System",
        )
        assert audit.audit_id.startswith("AUDIT-")
        assert audit.status == AuditStatus.PLANNED
        assert audit.findings == []

    def test_audit_findings_properties(self):
        """Test audit finding helper properties."""
        audit = QMSAudit(
            audit_id="AUDIT-001",
            audit_type=AuditType.INTERNAL,
            title="Test Audit",
            scope="Test",
        )

        # Add findings of different severities
        audit.findings.append(AuditFinding(
            finding_id="F1", audit_id="AUDIT-001",
            severity=FindingSeverity.CRITICAL, title="Critical", description="",
        ))
        audit.findings.append(AuditFinding(
            finding_id="F2", audit_id="AUDIT-001",
            severity=FindingSeverity.MAJOR, title="Major", description="",
        ))
        audit.findings.append(AuditFinding(
            finding_id="F3", audit_id="AUDIT-001",
            severity=FindingSeverity.MINOR, title="Minor", description="", is_closed=True,
        ))

        assert len(audit.critical_findings) == 1
        assert len(audit.major_findings) == 1
        assert len(audit.open_findings) == 2

    def test_design_review_creation(self):
        """Test design review creation."""
        review = DesignReview(
            review_id="",
            title="Algorithm Review",
            design_phase="detailed",
        )
        assert review.review_id.startswith("DR-")
        assert review.status == "pending"
        assert not review.approved

    def test_change_request_creation(self):
        """Test change request creation."""
        change = ChangeRequest(
            change_id="",
            change_type=ChangeType.ALGORITHM,
            title="Update model architecture",
            description="Changes to neural network layers",
        )
        assert change.change_id.startswith("CR-")
        assert change.status == "submitted"
        assert change.impact == ChangeImpact.MEDIUM

    def test_capa_record_creation(self):
        """Test CAPA record creation."""
        capa = CAPARecord(
            capa_id="",
            capa_type=CAPAType.CORRECTIVE,
            title="Fix validation issue",
            description="Address validation gap",
        )
        assert capa.capa_id.startswith("CAPA-")
        assert capa.status == CAPAStatus.INITIATED

    def test_resource_record_creation(self):
        """Test resource record creation."""
        resource = ResourceRecord(
            resource_id="",
            resource_type="infrastructure",
            name="GPU Server",
            description="Training server",
        )
        assert resource.resource_id.startswith("RES-")
        assert resource.is_available

    def test_accountability_record_creation(self):
        """Test accountability record creation."""
        role = AccountabilityRecord(
            role_id="",
            role_name="AI Safety Officer",
            role_description="Responsible for AI safety",
        )
        assert role.role_id.startswith("ROLE-")
        assert role.is_active


# =============================================================================
# Test QMS Procedure Management
# =============================================================================

class TestProcedureManagement:
    """Test procedure management functionality."""

    def test_create_procedure(self, qms):
        """Test procedure creation."""
        procedure = qms.create_procedure(
            element_type=QMSElementType.RISK_MANAGEMENT,
            title="Risk Assessment Procedure",
            description="Procedure for conducting risk assessments",
            scope="All AI system components",
            author="Test Author",
        )
        assert procedure.procedure_id is not None
        assert procedure.status == ProcedureStatus.DRAFT

    def test_submit_procedure_for_review(self, qms):
        """Test submitting procedure for review."""
        procedure = qms.create_procedure(
            element_type=QMSElementType.TESTING_VALIDATION,
            title="Testing Procedure",
            description="Testing procedures",
        )

        updated = qms.submit_procedure_for_review(
            procedure.procedure_id,
            reviewer="Quality Manager",
        )
        assert updated.status == ProcedureStatus.UNDER_REVIEW
        assert updated.reviewer == "Quality Manager"

    def test_approve_procedure(self, qms):
        """Test procedure approval."""
        procedure = qms.create_procedure(
            element_type=QMSElementType.DATA_MANAGEMENT,
            title="Data Management Procedure",
            description="Data handling procedures",
        )

        approved = qms.approve_procedure(
            procedure.procedure_id,
            approver="Director",
        )
        assert approved.status == ProcedureStatus.EFFECTIVE
        assert approved.approver == "Director"
        assert approved.approval_date is not None
        assert approved.review_date is not None

    def test_revise_procedure(self, qms):
        """Test procedure revision."""
        procedure = qms.create_procedure(
            element_type=QMSElementType.INCIDENT_REPORTING,
            title="Incident Reporting",
            description="Incident reporting procedures",
        )

        revised = qms.revise_procedure(
            procedure.procedure_id,
            changes="Added new reporting requirements",
            author="Editor",
        )
        assert revised.version == "1.1"
        assert revised.status == ProcedureStatus.DRAFT
        assert len(revised.revision_history) == 1

    def test_get_procedure(self, qms):
        """Test getting procedure by ID."""
        procedure = qms.create_procedure(
            element_type=QMSElementType.RESOURCE_MANAGEMENT,
            title="Resource Management",
            description="Resource management",
        )

        retrieved = qms.get_procedure(procedure.procedure_id)
        assert retrieved is not None
        assert retrieved.title == "Resource Management"

    def test_get_procedures_by_element(self, qms):
        """Test getting procedures by element type."""
        qms.create_procedure(
            element_type=QMSElementType.RISK_MANAGEMENT,
            title="Risk 1",
            description="First risk procedure",
        )
        qms.create_procedure(
            element_type=QMSElementType.RISK_MANAGEMENT,
            title="Risk 2",
            description="Second risk procedure",
        )
        qms.create_procedure(
            element_type=QMSElementType.TESTING_VALIDATION,
            title="Testing",
            description="Testing procedure",
        )

        risk_procedures = qms.get_procedures_by_element(QMSElementType.RISK_MANAGEMENT)
        assert len(risk_procedures) == 2

    def test_get_effective_procedures(self, qms):
        """Test getting effective procedures."""
        p1 = qms.create_procedure(
            element_type=QMSElementType.ACCOUNTABILITY,
            title="Accountability",
            description="Accountability framework",
        )
        p2 = qms.create_procedure(
            element_type=QMSElementType.AUTHORITY_COMMUNICATION,
            title="Authority Communication",
            description="Communication procedures",
        )

        qms.approve_procedure(p1.procedure_id, approver="Director")

        effective = qms.get_effective_procedures()
        assert len(effective) == 1

    def test_procedure_not_found(self, qms):
        """Test handling of non-existent procedure."""
        with pytest.raises(ValueError, match="not found"):
            qms.approve_procedure("INVALID-ID", approver="Test")


# =============================================================================
# Test Audit Management
# =============================================================================

class TestAuditManagement:
    """Test audit management functionality."""

    def test_create_audit(self, qms):
        """Test audit creation."""
        audit = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Q1 Internal Audit",
            scope="Risk Management System",
            planned_date="2025-03-01",
            lead_auditor="Lead Auditor",
        )
        assert audit.audit_id is not None
        assert audit.status == AuditStatus.PLANNED

    def test_start_audit(self, qms):
        """Test starting an audit."""
        audit = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Test Audit",
            scope="All QMS",
            planned_date="2025-02-01",
        )

        started = qms.start_audit(audit.audit_id)
        assert started.status == AuditStatus.IN_PROGRESS
        assert started.actual_date is not None

    def test_add_finding(self, qms):
        """Test adding findings to audit."""
        audit = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Test Audit",
            scope="Testing",
            planned_date="2025-02-01",
        )
        qms.start_audit(audit.audit_id)

        finding = qms.add_finding(
            audit_id=audit.audit_id,
            severity=FindingSeverity.MAJOR,
            title="Documentation Gap",
            description="Missing risk assessment documentation",
            element_type=QMSElementType.RISK_MANAGEMENT,
        )
        assert finding.finding_id is not None
        assert not finding.is_closed

        updated_audit = qms.get_audit(audit.audit_id)
        assert len(updated_audit.findings) == 1

    def test_close_finding(self, qms):
        """Test closing a finding."""
        audit = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Test Audit",
            scope="Testing",
            planned_date="2025-02-01",
        )
        qms.start_audit(audit.audit_id)

        finding = qms.add_finding(
            audit_id=audit.audit_id,
            severity=FindingSeverity.MINOR,
            title="Minor Issue",
            description="Minor documentation issue",
        )

        closed = qms.close_finding(
            audit_id=audit.audit_id,
            finding_id=finding.finding_id,
            closure_evidence="Documentation updated",
        )
        assert closed.is_closed
        assert closed.closure_date is not None

    def test_complete_audit(self, qms):
        """Test completing an audit."""
        audit = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Test Audit",
            scope="QMS",
            planned_date="2025-02-01",
        )
        qms.start_audit(audit.audit_id)

        completed = qms.complete_audit(
            audit_id=audit.audit_id,
            conclusions="Audit completed successfully",
            recommendations=["Improve documentation", "Add more tests"],
        )
        assert completed.status == AuditStatus.COMPLETED
        assert completed.completed_at is not None
        assert len(completed.recommendations) == 2

    def test_get_open_audits(self, qms):
        """Test getting open audits."""
        a1 = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Audit 1",
            scope="Scope 1",
            planned_date="2025-02-01",
        )
        a2 = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Audit 2",
            scope="Scope 2",
            planned_date="2025-03-01",
        )

        qms.start_audit(a1.audit_id)
        qms.complete_audit(a1.audit_id, conclusions="Done")

        open_audits = qms.get_open_audits()
        assert len(open_audits) == 1

    def test_follow_up_required_for_major_findings(self, qms):
        """Test follow-up requirement for major findings."""
        audit = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Test Audit",
            scope="Testing",
            planned_date="2025-02-01",
        )
        qms.start_audit(audit.audit_id)
        qms.add_finding(
            audit_id=audit.audit_id,
            severity=FindingSeverity.MAJOR,
            title="Major Issue",
            description="Significant gap",
        )

        completed = qms.complete_audit(audit.audit_id, conclusions="Issues found")
        assert completed.follow_up_required


# =============================================================================
# Test Design Control
# =============================================================================

class TestDesignControl:
    """Test design control functionality."""

    def test_create_design_review(self, qms):
        """Test design review creation."""
        review = qms.create_design_review(
            title="Algorithm Design Review",
            design_phase="detailed",
            components_reviewed=["neural_network", "feature_pipeline"],
            reviewers=["Senior Developer", "AI Architect"],
        )
        assert review.review_id is not None
        assert review.status == "pending"

    def test_complete_design_review_passed(self, qms):
        """Test completing design review with pass."""
        review = qms.create_design_review(
            title="Test Review",
            design_phase="preliminary",
        )

        completed = qms.complete_design_review(
            review_id=review.review_id,
            status="passed",
            approver="Architect",
        )
        assert completed.status == "passed"
        assert completed.approved
        assert completed.approval_date is not None

    def test_complete_design_review_with_findings(self, qms):
        """Test completing design review with findings."""
        review = qms.create_design_review(
            title="Test Review",
            design_phase="detailed",
        )

        completed = qms.complete_design_review(
            review_id=review.review_id,
            status="conditional",
            findings=["Need additional validation", "Missing error handling"],
            actions_required=["Add validation tests", "Implement error handling"],
        )
        assert completed.status == "conditional"
        assert not completed.approved
        assert len(completed.findings) == 2
        assert len(completed.actions_required) == 2


# =============================================================================
# Test Change Management
# =============================================================================

class TestChangeManagement:
    """Test change management functionality."""

    def test_create_change_request(self, qms):
        """Test change request creation."""
        change = qms.create_change_request(
            change_type=ChangeType.ALGORITHM,
            title="Update model architecture",
            description="Add new layer to neural network",
            reason="Improve performance",
            requestor="Developer",
        )
        assert change.change_id is not None
        assert change.status == "submitted"

    def test_assess_change_impact(self, qms):
        """Test change impact assessment."""
        change = qms.create_change_request(
            change_type=ChangeType.TRAINING_PROCESS,
            title="Change hyperparameters",
            description="Adjust learning rate",
        )

        assessed = qms.assess_change_impact(
            change_id=change.change_id,
            impact=ChangeImpact.MEDIUM,
            affected_components=["training", "evaluation"],
            risk_assessment="Low risk change",
        )
        assert assessed.status == "under_review"
        assert assessed.impact == ChangeImpact.MEDIUM
        assert not assessed.requires_conformity_reassessment

    def test_substantial_change_requires_reassessment(self, qms):
        """Test that substantial changes require conformity reassessment."""
        change = qms.create_change_request(
            change_type=ChangeType.ALGORITHM,
            title="Major algorithm rewrite",
            description="Complete redesign",
        )

        assessed = qms.assess_change_impact(
            change_id=change.change_id,
            impact=ChangeImpact.SUBSTANTIAL,
        )
        assert assessed.requires_conformity_reassessment

    def test_approve_change(self, qms):
        """Test change approval."""
        change = qms.create_change_request(
            change_type=ChangeType.CONFIGURATION,
            title="Update config",
            description="Config change",
        )

        approved = qms.approve_change(
            change_id=change.change_id,
            approver="Manager",
            implementation_plan="Deploy next sprint",
        )
        assert approved.status == "approved"
        assert approved.approver == "Manager"
        assert approved.approval_date is not None

    def test_reject_change(self, qms):
        """Test change rejection."""
        change = qms.create_change_request(
            change_type=ChangeType.DATA_SOURCE,
            title="New data source",
            description="Add new data",
        )

        rejected = qms.reject_change(
            change_id=change.change_id,
            rejection_reason="Data quality concerns",
        )
        assert rejected.status == "rejected"
        assert rejected.rejection_reason == "Data quality concerns"

    def test_implement_change(self, qms):
        """Test change implementation."""
        change = qms.create_change_request(
            change_type=ChangeType.DOCUMENTATION,
            title="Update docs",
            description="Documentation update",
        )
        qms.approve_change(change.change_id, approver="Manager")

        implemented = qms.implement_change(
            change_id=change.change_id,
            verification_results="Changes verified",
        )
        assert implemented.status == "implemented"
        assert implemented.implementation_date is not None

    def test_implement_unapproved_change_fails(self, qms):
        """Test that unapproved changes cannot be implemented."""
        change = qms.create_change_request(
            change_type=ChangeType.ALGORITHM,
            title="Some change",
            description="Description",
        )

        with pytest.raises(ValueError, match="not approved"):
            qms.implement_change(change.change_id)

    def test_get_pending_changes(self, qms):
        """Test getting pending changes."""
        c1 = qms.create_change_request(
            change_type=ChangeType.ALGORITHM,
            title="Change 1",
            description="Desc 1",
        )
        c2 = qms.create_change_request(
            change_type=ChangeType.DATA_SOURCE,
            title="Change 2",
            description="Desc 2",
        )
        c3 = qms.create_change_request(
            change_type=ChangeType.CONFIGURATION,
            title="Change 3",
            description="Desc 3",
        )

        qms.approve_change(c1.change_id, approver="Manager")

        pending = qms.get_pending_changes()
        assert len(pending) == 2


# =============================================================================
# Test CAPA Management
# =============================================================================

class TestCAPAManagement:
    """Test CAPA management functionality."""

    def test_create_capa(self, qms):
        """Test CAPA creation."""
        capa = qms.create_capa(
            capa_type=CAPAType.CORRECTIVE,
            title="Fix validation issue",
            description="Address validation gap",
            source="audit",
            source_reference="AUDIT-001",
            owner="Quality Lead",
        )
        assert capa.capa_id is not None
        assert capa.status == CAPAStatus.INITIATED

    def test_perform_root_cause_analysis(self, qms):
        """Test root cause analysis."""
        capa = qms.create_capa(
            capa_type=CAPAType.CORRECTIVE,
            title="Fix issue",
            description="Issue description",
        )

        analyzed = qms.perform_root_cause_analysis(
            capa_id=capa.capa_id,
            analysis_method="5-why",
            root_causes=["Cause 1", "Cause 2"],
            analysis_description="Detailed analysis",
        )
        assert analyzed.status == CAPAStatus.INVESTIGATING
        assert analyzed.analysis_method == "5-why"
        assert len(analyzed.root_causes) == 2

    def test_plan_capa_actions(self, qms):
        """Test CAPA action planning."""
        capa = qms.create_capa(
            capa_type=CAPAType.PREVENTIVE,
            title="Prevent issue",
            description="Preventive action",
        )

        planned = qms.plan_capa_actions(
            capa_id=capa.capa_id,
            actions=[
                {"description": "Action 1", "owner": "Dev 1"},
                {"description": "Action 2", "owner": "Dev 2"},
            ],
            effectiveness_criteria=["No recurrence in 3 months"],
            target_date="2025-06-01",
        )
        assert planned.status == CAPAStatus.ACTION_PLANNED
        assert len(planned.planned_actions) == 2
        assert planned.target_date == "2025-06-01"

    def test_implement_capa_action(self, qms):
        """Test CAPA action implementation."""
        capa = qms.create_capa(
            capa_type=CAPAType.CORRECTIVE,
            title="Fix issue",
            description="Description",
        )
        qms.plan_capa_actions(
            capa_id=capa.capa_id,
            actions=[
                {"description": "Action 1"},
                {"description": "Action 2"},
            ],
        )

        updated = qms.implement_capa_action(
            capa_id=capa.capa_id,
            action_index=0,
            implementation_details="Action 1 completed",
        )
        assert updated.status == CAPAStatus.IMPLEMENTING
        assert len(updated.implemented_actions) == 1

    def test_verify_capa(self, qms):
        """Test CAPA verification."""
        capa = qms.create_capa(
            capa_type=CAPAType.CORRECTIVE,
            title="Fix issue",
            description="Description",
        )
        qms.plan_capa_actions(
            capa_id=capa.capa_id,
            actions=[{"description": "Action"}],
        )
        qms.implement_capa_action(capa.capa_id, 0)

        verified = qms.verify_capa(
            capa_id=capa.capa_id,
            verification_method="Review",
            verification_results="Issue resolved",
            is_effective=True,
        )
        assert verified.status == CAPAStatus.CLOSED
        assert verified.is_effective
        assert verified.completion_date is not None

    def test_get_open_capas(self, qms):
        """Test getting open CAPAs."""
        c1 = qms.create_capa(
            capa_type=CAPAType.CORRECTIVE,
            title="CAPA 1",
            description="Desc 1",
        )
        c2 = qms.create_capa(
            capa_type=CAPAType.PREVENTIVE,
            title="CAPA 2",
            description="Desc 2",
        )

        qms.plan_capa_actions(c1.capa_id, actions=[{"description": "Action"}])
        qms.implement_capa_action(c1.capa_id, 0)
        qms.verify_capa(c1.capa_id, "Review", "Done", True)

        open_capas = qms.get_open_capas()
        assert len(open_capas) == 1


# =============================================================================
# Test Resource Management
# =============================================================================

class TestResourceManagement:
    """Test resource management functionality."""

    def test_register_resource(self, qms):
        """Test resource registration."""
        resource = qms.register_resource(
            resource_type="infrastructure",
            name="GPU Cluster",
            description="Training infrastructure",
            specifications={"gpus": 8, "memory": "256GB"},
            quantity=1.0,
            unit="cluster",
        )
        assert resource.resource_id is not None
        assert resource.is_available

    def test_qualify_supplier(self, qms):
        """Test supplier qualification."""
        resource = qms.register_resource(
            resource_type="software",
            name="ML Framework",
            description="Machine learning framework",
        )

        qualified = qms.qualify_supplier(
            resource_id=resource.resource_id,
            supplier_name="Framework Vendor",
            supplier_id="VENDOR-001",
        )
        assert qualified.supplier_qualified
        assert qualified.supplier_name == "Framework Vendor"


# =============================================================================
# Test Accountability Framework
# =============================================================================

class TestAccountabilityFramework:
    """Test accountability framework functionality."""

    def test_define_role(self, qms):
        """Test role definition."""
        role = qms.define_role(
            role_name="AI Safety Officer",
            role_description="Responsible for AI safety",
            responsibilities=["Monitor safety", "Review incidents"],
            authorities=["Approve deployments", "Stop operations"],
            department="AI Safety",
        )
        assert role.role_id is not None
        assert len(role.responsibilities) == 2

    def test_assign_role(self, qms):
        """Test role assignment."""
        role = qms.define_role(
            role_name="Quality Manager",
            role_description="Manages quality",
        )

        assigned = qms.assign_role(
            role_id=role.role_id,
            assigned_to="John Smith",
        )
        assert assigned.assigned_to == "John Smith"
        assert assigned.is_active
        assert assigned.effective_date != ""

    def test_get_all_roles(self, qms):
        """Test getting all roles."""
        qms.define_role(role_name="Role 1", role_description="Desc 1")
        qms.define_role(role_name="Role 2", role_description="Desc 2")
        qms.define_role(role_name="Role 3", role_description="Desc 3")

        roles = qms.get_all_roles()
        assert len(roles) == 3


# =============================================================================
# Test Compliance Status
# =============================================================================

class TestComplianceStatus:
    """Test compliance status functionality."""

    def test_get_compliance_status(self, qms):
        """Test getting compliance status."""
        # Create some procedures
        p1 = qms.create_procedure(
            element_type=QMSElementType.RISK_MANAGEMENT,
            title="Risk Procedure",
            description="Risk management",
        )
        qms.approve_procedure(p1.procedure_id, approver="Director")

        # Create an audit with findings
        audit = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Test Audit",
            scope="All",
            planned_date="2025-02-01",
        )
        qms.start_audit(audit.audit_id)
        qms.add_finding(
            audit_id=audit.audit_id,
            severity=FindingSeverity.MAJOR,
            title="Finding",
            description="Description",
        )
        qms.complete_audit(audit.audit_id, conclusions="Done")

        status = qms.get_compliance_status()

        assert "compliance_score" in status
        assert "element_coverage" in status
        assert "procedures" in status
        assert "audits" in status
        assert "findings" in status
        assert "capas" in status
        assert "changes" in status
        assert status["procedures"]["effective"] == 1

    def test_compliance_score_calculation(self, qms):
        """Test compliance score calculation."""
        # Perfect score scenario
        for element in [QMSElementType.RISK_MANAGEMENT, QMSElementType.TESTING_VALIDATION]:
            p = qms.create_procedure(
                element_type=element,
                title=f"{element.value} Procedure",
                description="Description",
            )
            qms.approve_procedure(p.procedure_id, approver="Director")

        status = qms.get_compliance_status()
        assert status["compliance_score"] > 0

    def test_compliance_issues_identified(self, qms):
        """Test compliance issues identification."""
        # Create audit with critical finding
        audit = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Test Audit",
            scope="All",
            planned_date="2025-02-01",
        )
        qms.start_audit(audit.audit_id)
        qms.add_finding(
            audit_id=audit.audit_id,
            severity=FindingSeverity.CRITICAL,
            title="Critical Issue",
            description="Critical finding",
        )
        qms.complete_audit(audit.audit_id, conclusions="Issues found")

        status = qms.get_compliance_status()
        assert len(status["compliance_issues"]) > 0


# =============================================================================
# Test Export and Reporting
# =============================================================================

class TestExportAndReporting:
    """Test export and reporting functionality."""

    def test_export_qms_summary(self, qms):
        """Test QMS summary export."""
        # Create some data
        qms.create_procedure(
            element_type=QMSElementType.RISK_MANAGEMENT,
            title="Test Procedure",
            description="Description",
        )
        qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Test Audit",
            scope="All",
            planned_date="2025-02-01",
        )

        export = qms.export_qms_summary()

        assert "export_date" in export
        assert "config" in export
        assert "compliance_status" in export
        assert "procedures" in export
        assert "audits" in export
        assert len(export["procedures"]) == 1
        assert len(export["audits"]) == 1

    def test_generate_management_review_report(self, qms):
        """Test management review report generation."""
        # Create audit and CAPA
        audit = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Q1 Audit",
            scope="All",
            planned_date="2025-02-01",
        )
        qms.start_audit(audit.audit_id)
        qms.complete_audit(audit.audit_id, conclusions="Done")

        capa = qms.create_capa(
            capa_type=CAPAType.CORRECTIVE,
            title="CAPA",
            description="Description",
        )
        qms.plan_capa_actions(capa.capa_id, actions=[{"desc": "Action"}])
        qms.implement_capa_action(capa.capa_id, 0)
        qms.verify_capa(capa.capa_id, "Review", "Done", True)

        report = qms.generate_management_review_report()

        assert "report_date" in report
        assert "period_covered" in report
        assert "overall_compliance" in report
        assert "recent_audits" in report
        assert "capa_effectiveness_percent" in report
        assert "recommendations" in report


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_qms_default(self, temp_log_dir):
        """Test creating QMS with default config."""
        qms = create_qms()
        assert qms is not None
        assert isinstance(qms, QualityManagementSystem)

    def test_create_qms_with_config(self, temp_log_dir):
        """Test creating QMS with config object."""
        config = QMSConfig(
            organization_name="Test Org",
            log_path=temp_log_dir,
        )
        qms = create_qms(config)
        assert qms.config.organization_name == "Test Org"

    def test_create_qms_with_dict(self, temp_log_dir):
        """Test creating QMS with dictionary config."""
        config_dict = {
            "organization_name": "Dict Org",
            "qms_version": "2.0",
            "log_path": temp_log_dir,
        }
        qms = create_qms(config_dict)
        assert qms.config.organization_name == "Dict Org"
        assert qms.config.qms_version == "2.0"

    def test_get_default_qms_procedures(self):
        """Test getting default QMS procedures."""
        procedures = get_default_qms_procedures()

        assert len(procedures) == 12  # 12 elements per Article 17

        # Check all elements are covered
        element_types = {p["element_type"] for p in procedures}
        assert len(element_types) == 12

        # Check each has required fields
        for proc in procedures:
            assert "element_type" in proc
            assert "title" in proc
            assert "description" in proc
            assert "article_reference" in proc


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Test thread safety of QMS operations."""

    def test_concurrent_procedure_creation(self, qms):
        """Test concurrent procedure creation."""
        import threading

        procedures = []
        errors = []

        def create_proc(index):
            try:
                p = qms.create_procedure(
                    element_type=QMSElementType.RISK_MANAGEMENT,
                    title=f"Procedure {index}",
                    description=f"Description {index}",
                )
                procedures.append(p)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_proc, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(procedures) == 10

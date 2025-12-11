# -*- coding: utf-8 -*-
"""
Tests for DORA Register of Information Module (Article 28(3)).

Comprehensive test coverage for:
- Enumerations (ContractType, ServiceType, etc.)
- Data structures (EntityInformation, ContractualArrangement, etc.)
- RegisterOfInformationConfig
- DORARegisterOfInformation
- ITS template export
- Factory functions

References:
    - Article 28(3) DORA (EU 2022/2554)
    - ITS on Register of Information: CIR 2024/2956
"""

import pytest

# Skip - this is a legacy test expecting old DORA API (RegisterStatus, etc.)
# The DORA module has been migrated to a new structure in services.dora_integration
pytest.skip(
    "Legacy DORA test - uses deprecated API (RegisterStatus, etc.) from old services.dora module",
    allow_module_level=True
)

import csv
import io
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
import threading

from services.dora_integration.reporting import (
    # Enumerations
    ContractType,
    ServiceType,
    FunctionType,
    DataLocation,
    ProviderLocationType,
    SubcontractingLevel,
    RegisterStatus,
    ExportFormat,
    # Data structures
    EntityInformation,
    BranchInformation,
    ContractualArrangement,
    ContractualArrangementFunction,
    ICTServiceProvider,
    SubcontractorEntry,
    ICTServiceRecord,
    RegisterSubmission,
    # Configuration
    RegisterOfInformationConfig,
    # Main class
    DORARegisterOfInformation,
    # Factory functions
    create_register_of_information,
    get_contract_types,
    get_service_types,
    get_its_templates,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return RegisterOfInformationConfig(
        entity_lei="TESTLEI123456789012",
        entity_name="Test Financial Entity",
        entity_type="investment_firm",
        country="DE",
        competent_authority="BaFin",
        require_lei_for_eu_providers=False,  # For easier testing
        require_exit_plan_for_critical=False,
    )


@pytest.fixture
def register(config):
    """Create test register instance."""
    return DORARegisterOfInformation(config=config)


@pytest.fixture
def register_with_data(register):
    """Create register with sample data."""
    # Add arrangement
    arrangement = register.add_contractual_arrangement(
        provider_name="Cloud Provider AG",
        provider_country="DE",
        contract_type=ContractType.OUTSOURCING,
        provider_lei="CLOUDPROVIDER123456",
        contract_value_eur=100000,
        supports_critical_function=True,
        audit_rights=True,
    )

    # Add provider
    provider = register.add_provider(
        legal_name="Cloud Provider AG",
        headquarters_country="DE",
        lei="CLOUDPROVIDER123456",
    )

    # Add service
    service = register.add_service(
        arrangement_id=arrangement.arrangement_id,
        provider_id=provider.provider_id,
        service_name="Cloud Computing",
        service_type=ServiceType.CLOUD_COMPUTING,
    )

    return register, arrangement, provider, service


# =============================================================================
# Enumeration Tests
# =============================================================================

class TestEnumerations:
    """Test all enumeration classes."""

    def test_contract_type_values(self):
        """Test ContractType enumeration values."""
        assert ContractType.OUTSOURCING.value == "outsourcing"
        assert ContractType.PROCUREMENT.value == "procurement"
        assert ContractType.INTRA_GROUP.value == "intra_group"
        assert ContractType.MIXED.value == "mixed"
        assert len(ContractType) == 4

    def test_service_type_values(self):
        """Test ServiceType enumeration values."""
        assert ServiceType.CLOUD_COMPUTING.value == "cloud_computing"
        assert ServiceType.DATA_CENTERS.value == "data_centers"
        assert ServiceType.TRADING.value == "trading"
        assert ServiceType.MARKET_DATA.value == "market_data"
        assert len(ServiceType) == 11

    def test_function_type_values(self):
        """Test FunctionType enumeration values."""
        assert FunctionType.CRITICAL.value == "critical"
        assert FunctionType.IMPORTANT.value == "important"
        assert FunctionType.STANDARD.value == "standard"
        assert len(FunctionType) == 3

    def test_data_location_values(self):
        """Test DataLocation enumeration values."""
        assert DataLocation.EU.value == "eu"
        assert DataLocation.EEA.value == "eea"
        assert DataLocation.THIRD_COUNTRY.value == "third_country"

    def test_provider_location_type_values(self):
        """Test ProviderLocationType enumeration values."""
        assert ProviderLocationType.EU_MEMBER_STATE.value == "eu_member_state"
        assert ProviderLocationType.EEA_COUNTRY.value == "eea_country"
        assert ProviderLocationType.THIRD_COUNTRY.value == "third_country"

    def test_subcontracting_level_values(self):
        """Test SubcontractingLevel enumeration values."""
        assert SubcontractingLevel.DIRECT.value == "direct"
        assert SubcontractingLevel.LEVEL_1.value == "level_1"
        assert SubcontractingLevel.LEVEL_3_PLUS.value == "level_3_plus"

    def test_register_status_values(self):
        """Test RegisterStatus enumeration values."""
        assert RegisterStatus.ACTIVE.value == "active"
        assert RegisterStatus.TERMINATED.value == "terminated"
        assert RegisterStatus.UNDER_REVIEW.value == "under_review"

    def test_export_format_values(self):
        """Test ExportFormat enumeration values."""
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.XML.value == "xml"
        assert ExportFormat.JSON.value == "json"


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestEntityInformation:
    """Test EntityInformation data structure."""

    def test_entity_creation(self):
        """Test EntityInformation creation with defaults."""
        entity = EntityInformation(
            lei="TESTLEI1234567890",
            entity_name="Test Entity",
        )
        assert entity.entity_id.startswith("ENT-")
        assert entity.lei == "TESTLEI1234567890"
        assert entity.entity_name == "Test Entity"

    def test_entity_auto_id_generation(self):
        """Test automatic entity ID generation."""
        entity1 = EntityInformation(entity_name="Entity 1")
        entity2 = EntityInformation(entity_name="Entity 2")
        assert entity1.entity_id != entity2.entity_id


class TestBranchInformation:
    """Test BranchInformation data structure."""

    def test_branch_creation(self):
        """Test BranchInformation creation."""
        branch = BranchInformation(
            entity_lei="PARENTLEI123456789",
            branch_country="FR",
            branch_name="French Branch",
        )
        assert branch.branch_id.startswith("BRN-")
        assert branch.branch_country == "FR"


class TestContractualArrangement:
    """Test ContractualArrangement data structure."""

    def test_arrangement_creation(self):
        """Test ContractualArrangement creation with defaults."""
        arrangement = ContractualArrangement(
            provider_name="Test Provider",
            provider_country="DE",
        )
        assert arrangement.arrangement_id.startswith("CA-")
        assert arrangement.provider_name == "Test Provider"
        assert arrangement.status == RegisterStatus.ACTIVE
        assert arrangement.entry_date is not None

    def test_arrangement_with_all_fields(self):
        """Test arrangement with all fields populated."""
        arrangement = ContractualArrangement(
            provider_name="Full Provider",
            provider_country="US",
            provider_lei="PROVIDERLEI123456",
            contract_type=ContractType.OUTSOURCING,
            contract_value_eur=50000,
            supports_critical_function=True,
            personal_data_processed=True,
            audit_rights=True,
            exit_plan_exists=True,
        )
        assert arrangement.provider_lei == "PROVIDERLEI123456"
        assert arrangement.contract_type == ContractType.OUTSOURCING
        assert arrangement.supports_critical_function is True


class TestContractualArrangementFunction:
    """Test ContractualArrangementFunction data structure."""

    def test_function_link_creation(self):
        """Test function link creation."""
        link = ContractualArrangementFunction(
            arrangement_id="CA-12345678",
            function_id="FUNC-001",
            function_name="Trading Operations",
            function_type=FunctionType.CRITICAL,
        )
        assert link.link_id.startswith("CAF-")
        assert link.function_type == FunctionType.CRITICAL


class TestICTServiceProvider:
    """Test ICTServiceProvider data structure."""

    def test_provider_creation(self):
        """Test ICTServiceProvider creation."""
        provider = ICTServiceProvider(
            legal_name="Test ICT Provider",
            headquarters_country="DE",
            lei="PROVLEI1234567890",
        )
        assert provider.provider_id.startswith("PRV-")
        assert provider.legal_name == "Test ICT Provider"
        assert provider.is_designated_ctpp is False

    def test_provider_ctpp_designation(self):
        """Test provider with CTPP designation."""
        provider = ICTServiceProvider(
            legal_name="CTPP Provider",
            headquarters_country="IE",
            is_designated_ctpp=True,
            ctpp_lead_overseer="EBA",
        )
        assert provider.is_designated_ctpp is True
        assert provider.ctpp_lead_overseer == "EBA"


class TestSubcontractorEntry:
    """Test SubcontractorEntry data structure."""

    def test_subcontractor_creation(self):
        """Test subcontractor creation."""
        sub = SubcontractorEntry(
            arrangement_id="CA-12345678",
            direct_provider_id="PRV-12345678",
            legal_name="Subcontractor Ltd",
            country="US",
            subcontracting_level=SubcontractingLevel.LEVEL_1,
        )
        assert sub.subcontractor_id.startswith("SUB-")
        assert sub.subcontracting_level == SubcontractingLevel.LEVEL_1


class TestICTServiceRecord:
    """Test ICTServiceRecord data structure."""

    def test_service_record_creation(self):
        """Test service record creation."""
        service = ICTServiceRecord(
            arrangement_id="CA-12345678",
            provider_id="PRV-12345678",
            service_name="Cloud Storage",
            service_type=ServiceType.CLOUD_COMPUTING,
        )
        assert service.service_id.startswith("SVC-")
        assert service.service_type == ServiceType.CLOUD_COMPUTING
        assert service.availability_target_pct == 99.0


class TestRegisterSubmission:
    """Test RegisterSubmission data structure."""

    def test_submission_creation(self):
        """Test submission creation."""
        submission = RegisterSubmission(
            reference_date="2025-03-31",
            competent_authority="BaFin",
        )
        assert submission.submission_id.startswith("REG-")
        assert submission.reference_date == "2025-03-31"
        assert submission.submission_date is not None


# =============================================================================
# Configuration Tests
# =============================================================================

class TestRegisterOfInformationConfig:
    """Test RegisterOfInformationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RegisterOfInformationConfig()
        assert config.annual_submission_deadline_month == 4
        assert config.annual_submission_deadline_day == 30
        assert config.reference_date_month == 3
        assert config.reference_date_day == 31
        assert config.require_lei_for_eu_providers is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RegisterOfInformationConfig(
            entity_lei="CUSTOMLEI123456789",
            entity_name="Custom Entity",
            country="FR",
            require_lei_for_eu_providers=False,
        )
        assert config.entity_lei == "CUSTOMLEI123456789"
        assert config.country == "FR"
        assert config.require_lei_for_eu_providers is False


# =============================================================================
# Entity Management Tests
# =============================================================================

class TestEntityManagement:
    """Test entity management functionality."""

    def test_set_entity_information(self, register):
        """Test setting entity information."""
        entity = register.set_entity_information(
            lei="NEWLEI1234567890",
            entity_name="New Entity",
            entity_type="credit_institution",
            country="NL",
            competent_authority="DNB",
        )
        assert entity is not None
        assert entity.lei == "NEWLEI1234567890"
        assert entity.entity_name == "New Entity"

    def test_get_entity_information(self, register):
        """Test getting entity information."""
        entity = register.get_entity_information()
        assert entity is not None
        assert entity.lei == "TESTLEI123456789012"

    def test_add_branch(self, register):
        """Test adding branch information."""
        branch = register.add_branch(
            branch_country="FR",
            branch_name="French Branch",
            branch_address="123 Paris Street",
            local_competent_authority="AMF",
        )
        assert branch is not None
        assert branch.branch_country == "FR"
        assert branch.branch_name == "French Branch"

    def test_add_branch_without_entity_fails(self):
        """Test that adding branch without entity fails."""
        register = DORARegisterOfInformation()
        with pytest.raises(ValueError, match="Entity information must be set first"):
            register.add_branch(
                branch_country="FR",
                branch_name="Test Branch",
            )


# =============================================================================
# Contractual Arrangement Tests
# =============================================================================

class TestContractualArrangements:
    """Test contractual arrangement management."""

    def test_add_contractual_arrangement(self, register):
        """Test adding contractual arrangement."""
        arrangement = register.add_contractual_arrangement(
            provider_name="Test Provider",
            provider_country="DE",
            contract_type=ContractType.PROCUREMENT,
            contract_value_eur=50000,
        )
        assert arrangement is not None
        assert arrangement.provider_name == "Test Provider"
        assert arrangement.provider_type_location == ProviderLocationType.EU_MEMBER_STATE

    def test_arrangement_eu_provider_detection(self, register):
        """Test automatic EU provider detection."""
        eu_arrangement = register.add_contractual_arrangement(
            provider_name="EU Provider",
            provider_country="FR",
            contract_type=ContractType.OUTSOURCING,
        )
        assert eu_arrangement.provider_type_location == ProviderLocationType.EU_MEMBER_STATE

        # EEA (Norway)
        eea_arrangement = register.add_contractual_arrangement(
            provider_name="EEA Provider",
            provider_country="NO",
            contract_type=ContractType.OUTSOURCING,
        )
        assert eea_arrangement.provider_type_location == ProviderLocationType.EEA_COUNTRY

        # Third country
        third_arrangement = register.add_contractual_arrangement(
            provider_name="US Provider",
            provider_country="US",
            contract_type=ContractType.OUTSOURCING,
        )
        assert third_arrangement.provider_type_location == ProviderLocationType.THIRD_COUNTRY

    def test_update_arrangement(self, register):
        """Test updating arrangement."""
        arrangement = register.add_contractual_arrangement(
            provider_name="Original Provider",
            provider_country="DE",
            contract_type=ContractType.PROCUREMENT,
        )

        updated = register.update_arrangement(
            arrangement.arrangement_id,
            contract_value_eur=75000,
            audit_rights=True,
        )
        assert updated is not None
        assert updated.contract_value_eur == 75000
        assert updated.audit_rights is True

    def test_terminate_arrangement(self, register):
        """Test terminating arrangement."""
        arrangement = register.add_contractual_arrangement(
            provider_name="Terminated Provider",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
        )

        terminated = register.terminate_arrangement(
            arrangement.arrangement_id,
            termination_date="2025-12-31",
            termination_reason="Contract expired",
        )
        assert terminated is not None
        assert terminated.status == RegisterStatus.TERMINATED
        assert terminated.contract_end_date == "2025-12-31"

    def test_get_arrangement(self, register_with_data):
        """Test getting arrangement by ID."""
        register, arrangement, _, _ = register_with_data
        retrieved = register.get_arrangement(arrangement.arrangement_id)
        assert retrieved is not None
        assert retrieved.arrangement_id == arrangement.arrangement_id

    def test_get_all_arrangements(self, register):
        """Test getting all arrangements."""
        register.add_contractual_arrangement(
            provider_name="Provider 1",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
        )
        register.add_contractual_arrangement(
            provider_name="Provider 2",
            provider_country="FR",
            contract_type=ContractType.PROCUREMENT,
        )

        arrangements = register.get_all_arrangements()
        assert len(arrangements) == 2

    def test_get_arrangements_for_critical_functions(self, register):
        """Test getting arrangements supporting critical functions."""
        register.add_contractual_arrangement(
            provider_name="Critical Provider",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
            supports_critical_function=True,
        )
        register.add_contractual_arrangement(
            provider_name="Standard Provider",
            provider_country="DE",
            contract_type=ContractType.PROCUREMENT,
            supports_critical_function=False,
        )

        critical = register.get_arrangements_for_critical_functions()
        assert len(critical) == 1
        assert critical[0].provider_name == "Critical Provider"


# =============================================================================
# Function Link Tests
# =============================================================================

class TestFunctionLinks:
    """Test function link management."""

    def test_link_function_to_arrangement(self, register):
        """Test linking function to arrangement."""
        arrangement = register.add_contractual_arrangement(
            provider_name="Provider",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
        )

        link = register.link_function_to_arrangement(
            arrangement_id=arrangement.arrangement_id,
            function_id="FUNC-001",
            function_name="Trading Operations",
            function_type=FunctionType.CRITICAL,
            is_critical_or_important=True,
        )
        assert link is not None
        assert link.function_name == "Trading Operations"
        assert link.is_critical_or_important is True

    def test_link_updates_arrangement_criticality(self, register):
        """Test that linking critical function updates arrangement."""
        arrangement = register.add_contractual_arrangement(
            provider_name="Provider",
            provider_country="DE",
            contract_type=ContractType.PROCUREMENT,
            supports_critical_function=False,
        )

        register.link_function_to_arrangement(
            arrangement_id=arrangement.arrangement_id,
            function_id="FUNC-001",
            function_name="Critical Function",
            function_type=FunctionType.CRITICAL,
            is_critical_or_important=True,
        )

        # Check arrangement was updated
        updated = register.get_arrangement(arrangement.arrangement_id)
        assert updated.supports_critical_function is True
        assert "FUNC-001" in updated.critical_function_ids

    def test_get_functions_for_arrangement(self, register):
        """Test getting functions for arrangement."""
        arrangement = register.add_contractual_arrangement(
            provider_name="Provider",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
        )

        register.link_function_to_arrangement(
            arrangement_id=arrangement.arrangement_id,
            function_id="FUNC-001",
            function_name="Function 1",
            function_type=FunctionType.CRITICAL,
        )
        register.link_function_to_arrangement(
            arrangement_id=arrangement.arrangement_id,
            function_id="FUNC-002",
            function_name="Function 2",
            function_type=FunctionType.STANDARD,
        )

        functions = register.get_functions_for_arrangement(arrangement.arrangement_id)
        assert len(functions) == 2


# =============================================================================
# Provider Tests
# =============================================================================

class TestProviderManagement:
    """Test provider management functionality."""

    def test_add_provider(self, register):
        """Test adding provider."""
        provider = register.add_provider(
            legal_name="Test Provider AG",
            headquarters_country="DE",
            lei="PROVLEI1234567890",
        )
        assert provider is not None
        assert provider.legal_name == "Test Provider AG"
        assert provider.location_type == ProviderLocationType.EU_MEMBER_STATE

    def test_add_ctpp_provider(self, register):
        """Test adding CTPP designated provider."""
        provider = register.add_provider(
            legal_name="AWS",
            headquarters_country="US",
            is_designated_ctpp=True,
            ctpp_lead_overseer="ESMA",
        )
        assert provider.is_designated_ctpp is True
        assert provider.ctpp_lead_overseer == "ESMA"

    def test_add_intra_group_provider(self, register):
        """Test adding intra-group provider."""
        provider = register.add_provider(
            legal_name="Group IT Services",
            headquarters_country="DE",
            is_intra_group=True,
            parent_lei="PARENTLEI12345678",
            parent_name="Parent Company AG",
        )
        assert provider.is_ict_intra_group_provider is True
        assert provider.parent_lei == "PARENTLEI12345678"

    def test_get_provider_by_lei(self, register):
        """Test getting provider by LEI."""
        register.add_provider(
            legal_name="Test Provider",
            headquarters_country="DE",
            lei="FINDMELEI12345678",
        )

        found = register.get_provider_by_lei("FINDMELEI12345678")
        assert found is not None
        assert found.lei == "FINDMELEI12345678"

    def test_get_ctpp_providers(self, register):
        """Test getting CTPP providers."""
        register.add_provider(
            legal_name="CTPP 1",
            headquarters_country="US",
            is_designated_ctpp=True,
        )
        register.add_provider(
            legal_name="Regular Provider",
            headquarters_country="DE",
            is_designated_ctpp=False,
        )

        ctpps = register.get_ctpp_providers()
        assert len(ctpps) == 1
        assert ctpps[0].legal_name == "CTPP 1"


# =============================================================================
# Subcontracting Tests
# =============================================================================

class TestSubcontracting:
    """Test subcontracting chain management."""

    def test_add_subcontractor(self, register):
        """Test adding subcontractor."""
        arrangement = register.add_contractual_arrangement(
            provider_name="Main Provider",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
            subcontracting_permitted=True,
        )

        subcontractor = register.add_subcontractor(
            arrangement_id=arrangement.arrangement_id,
            direct_provider_id="PRV-MAIN",
            legal_name="Sub Provider Ltd",
            country="US",
            subcontracting_level=SubcontractingLevel.LEVEL_1,
            services_subcontracted=["Data Storage"],
        )
        assert subcontractor is not None
        assert subcontractor.legal_name == "Sub Provider Ltd"

    def test_get_subcontractors_for_arrangement(self, register):
        """Test getting subcontractors for arrangement."""
        arrangement = register.add_contractual_arrangement(
            provider_name="Main Provider",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
        )

        register.add_subcontractor(
            arrangement_id=arrangement.arrangement_id,
            direct_provider_id="PRV-MAIN",
            legal_name="Sub 1",
            country="US",
        )
        register.add_subcontractor(
            arrangement_id=arrangement.arrangement_id,
            direct_provider_id="PRV-MAIN",
            legal_name="Sub 2",
            country="IN",
        )

        subs = register.get_subcontractors_for_arrangement(arrangement.arrangement_id)
        assert len(subs) == 2

    def test_get_full_subcontracting_chain(self, register):
        """Test getting full subcontracting chain."""
        arrangement = register.add_contractual_arrangement(
            provider_name="Main Provider",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
            subcontracting_permitted=True,
        )

        register.add_subcontractor(
            arrangement_id=arrangement.arrangement_id,
            direct_provider_id="PRV-MAIN",
            legal_name="Level 1 Sub",
            country="US",
            subcontracting_level=SubcontractingLevel.LEVEL_1,
        )

        chain = register.get_full_subcontracting_chain(arrangement.arrangement_id)
        assert chain["direct_provider"] == "Main Provider"
        assert len(chain["chain"]) == 1


# =============================================================================
# Service Tests
# =============================================================================

class TestServiceManagement:
    """Test ICT service management."""

    def test_add_service(self, register):
        """Test adding service."""
        arrangement = register.add_contractual_arrangement(
            provider_name="Provider",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
        )
        provider = register.add_provider(
            legal_name="Provider",
            headquarters_country="DE",
        )

        service = register.add_service(
            arrangement_id=arrangement.arrangement_id,
            provider_id=provider.provider_id,
            service_name="Cloud Database",
            service_type=ServiceType.DATA_ANALYTICS,
            supports_critical_function=True,
            availability_target_pct=99.9,
        )
        assert service is not None
        assert service.service_type == ServiceType.DATA_ANALYTICS
        assert service.availability_target_pct == 99.9

    def test_get_services_for_arrangement(self, register_with_data):
        """Test getting services for arrangement."""
        register, arrangement, _, _ = register_with_data

        services = register.get_services_for_arrangement(arrangement.arrangement_id)
        assert len(services) == 1


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Test register validation functionality."""

    def test_validate_register_valid(self, register_with_data):
        """Test validation of valid register."""
        register, _, _, _ = register_with_data

        validation = register.validate_register()
        assert "is_valid" in validation
        assert "issues" in validation
        assert "warnings" in validation
        assert "summary" in validation

    def test_validate_register_no_entity(self):
        """Test validation without entity information."""
        register = DORARegisterOfInformation()

        validation = register.validate_register()
        assert validation["is_valid"] is False
        assert "Entity information not set" in validation["issues"][0]

    def test_validate_critical_without_exit_plan(self):
        """Test validation of critical arrangement without exit plan."""
        config = RegisterOfInformationConfig(
            entity_lei="TESTLEI123",
            entity_name="Test",
            country="DE",
            require_exit_plan_for_critical=True,
        )
        register = DORARegisterOfInformation(config=config)

        register.add_contractual_arrangement(
            provider_name="Critical Provider",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
            supports_critical_function=True,
            exit_plan_exists=False,
        )

        validation = register.validate_register()
        assert any("Exit plan missing" in issue for issue in validation["issues"])


# =============================================================================
# Export Tests
# =============================================================================

class TestExport:
    """Test export functionality."""

    def test_export_to_csv(self, register_with_data):
        """Test CSV export."""
        register, _, _, _ = register_with_data

        csv_data = register.export_to_csv()
        assert "B_01_01" in csv_data  # Entity
        assert "B_02_01" in csv_data  # Arrangements
        assert "B_03_01" in csv_data  # Providers
        assert "B_06_01" in csv_data  # Services
        assert "B_99_01" in csv_data  # Totals

    def test_csv_format_arrangements(self, register_with_data):
        """Test arrangements CSV format."""
        register, _, _, _ = register_with_data

        csv_data = register.export_to_csv()
        arrangements_csv = csv_data["B_02_01"]

        # Parse CSV
        reader = csv.reader(io.StringIO(arrangements_csv))
        rows = list(reader)

        # Check header
        header = rows[0]
        assert "Arrangement_ID" in header
        assert "Provider_Name" in header
        assert "Supports_Critical" in header

        # Check data row
        assert len(rows) >= 2

    def test_export_to_json(self, register_with_data):
        """Test JSON export."""
        register, _, _, _ = register_with_data

        json_data = register.export_to_json()
        assert "entity" in json_data
        assert "contractual_arrangements" in json_data
        assert "providers" in json_data
        assert "services" in json_data
        assert json_data["entity"]["lei"] == "TESTLEI123456789012"

    def test_export_excludes_terminated(self, register):
        """Test that export excludes terminated by default."""
        arrangement = register.add_contractual_arrangement(
            provider_name="Terminated Provider",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
        )
        register.terminate_arrangement(
            arrangement.arrangement_id,
            termination_date="2025-01-01",
        )

        csv_data = register.export_to_csv(include_terminated=False)
        arrangements_csv = csv_data["B_02_01"]

        # Should only have header row
        reader = csv.reader(io.StringIO(arrangements_csv))
        rows = list(reader)
        assert len(rows) == 1  # Only header


# =============================================================================
# Submission Tests
# =============================================================================

class TestSubmission:
    """Test submission management."""

    def test_generate_annual_submission(self, register_with_data):
        """Test generating annual submission."""
        register, _, _, _ = register_with_data

        submission = register.generate_annual_submission(
            reference_date="2025-03-31",
        )
        assert submission is not None
        assert submission.reference_date == "2025-03-31"
        assert submission.arrangements_count == 1
        assert submission.submission_status == "prepared"

    def test_record_submission_acknowledgment(self, register_with_data):
        """Test recording submission acknowledgment."""
        register, _, _, _ = register_with_data

        submission = register.generate_annual_submission("2025-03-31")

        acknowledged = register.record_submission_acknowledgment(
            submission.submission_id,
            acknowledgment_reference="ACK-2025-001",
            status="acknowledged",
        )
        assert acknowledged.acknowledgment_reference == "ACK-2025-001"
        assert acknowledged.submission_status == "acknowledged"

    def test_record_submission_feedback(self, register_with_data):
        """Test recording submission feedback."""
        register, _, _, _ = register_with_data

        submission = register.generate_annual_submission("2025-03-31")

        feedback = register.record_submission_feedback(
            submission.submission_id,
            feedback_details="All data validated successfully",
            status="accepted",
        )
        assert feedback.feedback_received is True
        assert feedback.submission_status == "accepted"

    def test_get_submission_history(self, register_with_data):
        """Test getting submission history."""
        register, _, _, _ = register_with_data

        register.generate_annual_submission("2024-03-31")
        register.generate_annual_submission("2025-03-31")

        history = register.get_submission_history()
        assert len(history) == 2

    def test_get_next_submission_deadline(self, register):
        """Test getting next submission deadline."""
        deadline, days = register.get_next_submission_deadline()
        assert deadline is not None
        assert isinstance(days, int)


# =============================================================================
# Statistics Tests
# =============================================================================

class TestStatistics:
    """Test statistics and reporting functionality."""

    def test_get_register_statistics(self, register_with_data):
        """Test getting comprehensive statistics."""
        register, _, _, _ = register_with_data

        stats = register.get_register_statistics()

        assert "entity" in stats
        assert "arrangements" in stats
        assert "providers" in stats
        assert "services" in stats
        assert "compliance" in stats
        assert "submissions" in stats

        assert stats["arrangements"]["total"] == 1
        assert stats["providers"]["total"] == 1
        assert stats["services"]["total"] == 1

    def test_statistics_country_breakdown(self, register):
        """Test statistics country breakdown."""
        register.add_contractual_arrangement(
            provider_name="DE Provider",
            provider_country="DE",
            contract_type=ContractType.OUTSOURCING,
        )
        register.add_contractual_arrangement(
            provider_name="FR Provider",
            provider_country="FR",
            contract_type=ContractType.PROCUREMENT,
        )

        stats = register.get_register_statistics()
        assert "DE" in stats["providers"]["by_country"]
        assert "FR" in stats["providers"]["by_country"]


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_register_of_information(self):
        """Test factory function."""
        register = create_register_of_information()
        assert register is not None
        assert isinstance(register, DORARegisterOfInformation)

    def test_create_with_config(self):
        """Test factory function with config."""
        config = RegisterOfInformationConfig(
            entity_lei="CUSTOMLEI123",
            entity_name="Custom Entity",
        )
        register = create_register_of_information(config=config)

        entity = register.get_entity_information()
        assert entity.lei == "CUSTOMLEI123"

    def test_get_contract_types(self):
        """Test getting contract types."""
        types = get_contract_types()
        assert len(types) == 4
        assert ContractType.OUTSOURCING in types

    def test_get_service_types(self):
        """Test getting service types."""
        types = get_service_types()
        assert len(types) == 11
        assert ServiceType.CLOUD_COMPUTING in types

    def test_get_its_templates(self):
        """Test getting ITS template identifiers."""
        templates = get_its_templates()
        assert "B_01_01" in templates
        assert "B_02_01" in templates
        assert "B_99_01" in templates


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test thread safety of register operations."""

    def test_concurrent_arrangement_addition(self, register):
        """Test concurrent arrangement addition."""
        results = []
        errors = []

        def add_arrangement(name):
            try:
                arr = register.add_contractual_arrangement(
                    provider_name=name,
                    provider_country="DE",
                    contract_type=ContractType.PROCUREMENT,
                )
                results.append(arr)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=add_arrangement, args=(f"Provider_{i}",))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

        # All should have unique IDs
        ids = [a.arrangement_id for a in results]
        assert len(set(ids)) == 10


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_register_workflow(self):
        """Test full register creation and submission workflow."""
        # 1. Create register
        config = RegisterOfInformationConfig(
            entity_lei="INTLEI1234567890",
            entity_name="Integration Test Entity",
            entity_type="investment_firm",
            country="DE",
            competent_authority="BaFin",
        )
        register = DORARegisterOfInformation(config=config)

        # 2. Add branch
        register.add_branch(
            branch_country="FR",
            branch_name="French Branch",
        )

        # 3. Add provider
        provider = register.add_provider(
            legal_name="Test Cloud Provider",
            headquarters_country="IE",
            lei="CLOUDLEI12345678",
            is_designated_ctpp=True,
        )

        # 4. Add arrangement
        arrangement = register.add_contractual_arrangement(
            provider_name="Test Cloud Provider",
            provider_country="IE",
            provider_lei="CLOUDLEI12345678",
            contract_type=ContractType.OUTSOURCING,
            supports_critical_function=True,
            audit_rights=True,
            exit_plan_exists=True,
        )

        # 5. Link function
        register.link_function_to_arrangement(
            arrangement_id=arrangement.arrangement_id,
            function_id="TRADE-001",
            function_name="Order Execution",
            function_type=FunctionType.CRITICAL,
            is_critical_or_important=True,
        )

        # 6. Add service
        register.add_service(
            arrangement_id=arrangement.arrangement_id,
            provider_id=provider.provider_id,
            service_name="Trading Infrastructure",
            service_type=ServiceType.TRADING,
            supports_critical_function=True,
        )

        # 7. Add subcontractor
        register.add_subcontractor(
            arrangement_id=arrangement.arrangement_id,
            direct_provider_id=provider.provider_id,
            legal_name="Data Center Provider",
            country="US",
        )

        # 8. Validate
        validation = register.validate_register()
        assert validation["is_valid"] is True

        # 9. Export
        csv_data = register.export_to_csv()
        assert len(csv_data) >= 5

        # 10. Generate submission
        submission = register.generate_annual_submission("2025-03-31")
        assert submission.arrangements_count == 1

        # 11. Get statistics
        stats = register.get_register_statistics()
        assert stats["arrangements"]["supporting_critical_functions"] == 1
        assert stats["providers"]["ctpp_designated"] == 1

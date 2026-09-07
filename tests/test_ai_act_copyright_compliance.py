# -*- coding: utf-8 -*-
"""
Tests for Article 53(1)(c) EU AI Act - Copyright Compliance.

This module provides comprehensive tests for the copyright compliance
functionality required by Article 53(1)(c) of the EU AI Act and
DSM Directive 2019/790 Article 4.

Coverage includes:
- DataSourceRecord management
- CopyrightComplianceManager operations
- Opt-out checking mechanisms
- Policy document generation
- Compliance status verification
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from services.ai_act.copyright_compliance import (
    # Enums
    DataSourceType,
    CopyrightStatus,
    OptOutMechanism,
    # Data structures
    DataSourceRecord,
    OptOutCheck,
    RightsHolderRequest,
    # Constants
    DEFAULT_DATA_SOURCES,
    # Main class
    CopyrightComplianceManager,
    # Factory functions
    create_copyright_manager,
    get_default_data_sources,
    validate_source_record,
)


class TestDataSourceType:
    """Test DataSourceType enum."""

    def test_all_types_defined(self):
        """Test all data source types are defined."""
        types = [
            DataSourceType.PUBLIC_MARKET_DATA,
            DataSourceType.LICENSED_DATA,
            DataSourceType.OPEN_DATA,
            DataSourceType.PROPRIETARY,
            DataSourceType.SYNTHETIC,
            DataSourceType.RESEARCH_DATA,
            DataSourceType.GOVERNMENT_DATA,
        ]
        for t in types:
            assert t is not None

    def test_type_values_are_strings(self):
        """Test type values are strings."""
        for t in DataSourceType:
            assert isinstance(t.value, str)


class TestCopyrightStatus:
    """Test CopyrightStatus enum."""

    def test_all_statuses_defined(self):
        """Test all copyright statuses are defined."""
        statuses = [
            CopyrightStatus.PUBLIC_DOMAIN,
            CopyrightStatus.LICENSED,
            CopyrightStatus.FAIR_USE,
            CopyrightStatus.TDM_EXCEPTION,
            CopyrightStatus.OPT_OUT_RESPECTED,
            CopyrightStatus.NOT_APPLICABLE,
            CopyrightStatus.PENDING_REVIEW,
        ]
        for s in statuses:
            assert s is not None


class TestOptOutMechanism:
    """Test OptOutMechanism enum."""

    def test_all_mechanisms_defined(self):
        """Test all opt-out mechanisms are defined."""
        mechanisms = [
            OptOutMechanism.ROBOTS_TXT,
            OptOutMechanism.TDMREP_HEADER,
            OptOutMechanism.TDMREP_META,
            OptOutMechanism.AI_TXT,
            OptOutMechanism.DIRECT_NOTICE,
            OptOutMechanism.LICENSE_TERMS,
        ]
        for m in mechanisms:
            assert m is not None


class TestDataSourceRecord:
    """Test DataSourceRecord dataclass."""

    def test_create_market_data_source(self):
        """Test creating market data source."""
        source = DataSourceRecord(
            source_id="test",
            source_name="Test Data",
            source_type=DataSourceType.PUBLIC_MARKET_DATA,
            copyright_status=CopyrightStatus.NOT_APPLICABLE,
            provider="Test Provider",
        )
        assert source.source_id == "test"
        assert source.source_type == DataSourceType.PUBLIC_MARKET_DATA
        assert source.copyright_status == CopyrightStatus.NOT_APPLICABLE

    def test_create_licensed_source(self):
        """Test creating licensed data source."""
        source = DataSourceRecord(
            source_id="licensed",
            source_name="Licensed Data",
            source_type=DataSourceType.LICENSED_DATA,
            copyright_status=CopyrightStatus.LICENSED,
            provider="Provider",
            license_type="Commercial",
            license_url="https://example.com/license",
        )
        assert source.license_type == "Commercial"
        assert source.license_url is not None

    def test_default_values(self):
        """Test default values are set correctly."""
        source = DataSourceRecord(
            source_id="test",
            source_name="Test",
            source_type=DataSourceType.SYNTHETIC,
            copyright_status=CopyrightStatus.NOT_APPLICABLE,
            provider="Internal",
        )
        assert source.opt_out_checked is False
        assert source.data_category == "market_data"
        assert source.geographic_scope == "global"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        source = DataSourceRecord(
            source_id="test",
            source_name="Test Data",
            source_type=DataSourceType.PUBLIC_MARKET_DATA,
            copyright_status=CopyrightStatus.NOT_APPLICABLE,
            provider="Provider",
        )
        data = source.to_dict()
        assert data["source_id"] == "test"
        assert data["source_type"] == "public_market_data"
        assert data["copyright_status"] == "not_applicable"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        original = DataSourceRecord(
            source_id="test",
            source_name="Test Data",
            source_type=DataSourceType.PUBLIC_MARKET_DATA,
            copyright_status=CopyrightStatus.NOT_APPLICABLE,
            provider="Provider",
        )
        data = original.to_dict()
        restored = DataSourceRecord.from_dict(data)

        assert restored.source_id == original.source_id
        assert restored.source_type == original.source_type
        assert restored.copyright_status == original.copyright_status


class TestOptOutCheck:
    """Test OptOutCheck dataclass."""

    def test_create_opt_out_check(self):
        """Test creating opt-out check record."""
        check = OptOutCheck(
            check_id="check1",
            source_id="source1",
            check_date=datetime.utcnow(),
            mechanism_checked="robots.txt",
            opt_out_found=False,
            action_taken="proceeded",
        )
        assert check.check_id == "check1"
        assert check.opt_out_found is False
        assert check.action_taken == "proceeded"

    def test_check_with_evidence(self):
        """Test check with evidence hash."""
        check = OptOutCheck(
            check_id="check1",
            source_id="source1",
            check_date=datetime.utcnow(),
            mechanism_checked="robots.txt",
            opt_out_found=False,
            action_taken="proceeded",
            evidence_hash="abc123",
        )
        assert check.evidence_hash == "abc123"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        check = OptOutCheck(
            check_id="check1",
            source_id="source1",
            check_date=datetime.utcnow(),
            mechanism_checked="robots.txt",
            opt_out_found=False,
            action_taken="proceeded",
        )
        data = check.to_dict()
        assert "check_id" in data
        assert "mechanism_checked" in data


class TestCopyrightComplianceManager:
    """Test CopyrightComplianceManager."""

    @pytest.fixture
    def manager(self) -> CopyrightComplianceManager:
        """Create manager instance."""
        return create_copyright_manager()

    def test_default_sources_initialized(self, manager):
        """Test default data sources are initialized."""
        assert len(manager.data_sources) > 0
        assert "binance_ohlcv" in manager.data_sources

    def test_register_new_source(self, manager):
        """Test registering new data source."""
        source = DataSourceRecord(
            source_id="new_source",
            source_name="New Data",
            source_type=DataSourceType.OPEN_DATA,
            copyright_status=CopyrightStatus.PUBLIC_DOMAIN,
            provider="Open Data Provider",
        )
        source_id = manager.register_data_source(source)
        assert source_id == "new_source"
        assert "new_source" in manager.data_sources

    def test_update_data_source(self, manager):
        """Test updating existing data source."""
        result = manager.update_data_source("binance_ohlcv", {"description": "Updated description"})
        assert result is True
        assert manager.data_sources["binance_ohlcv"].description == "Updated description"

    def test_update_nonexistent_source(self, manager):
        """Test updating nonexistent source fails."""
        result = manager.update_data_source("nonexistent", {"description": "Test"})
        assert result is False

    def test_remove_data_source(self, manager):
        """Test removing data source."""
        source = DataSourceRecord(
            source_id="to_remove",
            source_name="Remove Me",
            source_type=DataSourceType.SYNTHETIC,
            copyright_status=CopyrightStatus.NOT_APPLICABLE,
            provider="Internal",
        )
        manager.register_data_source(source)
        assert "to_remove" in manager.data_sources

        result = manager.remove_data_source("to_remove")
        assert result is True
        assert "to_remove" not in manager.data_sources

    def test_remove_nonexistent_source(self, manager):
        """Test removing nonexistent source."""
        result = manager.remove_data_source("nonexistent")
        assert result is False

    def test_check_opt_out(self, manager):
        """Test opt-out check recording."""
        check = manager.check_opt_out(source_id="binance_ohlcv", mechanism="robots.txt")
        assert check is not None
        assert check.mechanism_checked == "robots.txt"
        assert manager.data_sources["binance_ohlcv"].opt_out_checked

    def test_check_opt_out_with_evidence(self, manager):
        """Test opt-out check with evidence hash."""
        check = manager.check_opt_out(
            source_id="binance_ohlcv", mechanism="robots.txt", content_hash="abc123"
        )
        assert check.evidence_hash == "abc123"

    def test_check_opt_out_multiple_mechanisms(self, manager):
        """Test checking multiple opt-out mechanisms."""
        mechanisms = ["robots.txt", "TDMRep", "meta_tag", "ai.txt"]
        for mechanism in mechanisms:
            check = manager.check_opt_out("binance_ohlcv", mechanism)
            assert check.mechanism_checked == mechanism

    def test_compliance_status(self, manager):
        """Test compliance status calculation."""
        status = manager.get_compliance_status()
        assert "total_sources" in status
        assert "compliance_percentage" in status
        assert "opt_out_checked" in status
        assert status["compliance_percentage"] >= 0

    def test_training_data_sources_list(self, manager):
        """Test getting training data sources list."""
        sources = manager.get_training_data_sources()
        assert isinstance(sources, list)
        assert len(sources) > 0
        assert "name" in sources[0]
        assert "copyright_status" in sources[0]

    def test_get_sources_by_status(self, manager):
        """Test filtering sources by copyright status."""
        sources = manager.get_sources_by_status(CopyrightStatus.NOT_APPLICABLE)
        assert all(s.copyright_status == CopyrightStatus.NOT_APPLICABLE for s in sources)

    def test_get_sources_by_type(self, manager):
        """Test filtering sources by type."""
        sources = manager.get_sources_by_type(DataSourceType.PUBLIC_MARKET_DATA)
        assert all(s.source_type == DataSourceType.PUBLIC_MARKET_DATA for s in sources)

    def test_get_opt_out_checks(self, manager):
        """Test getting opt-out check records."""
        manager.check_opt_out("binance_ohlcv", "robots.txt")
        checks = manager.get_opt_out_checks()
        assert len(checks) > 0

    def test_get_opt_out_checks_by_source(self, manager):
        """Test filtering opt-out checks by source."""
        manager.check_opt_out("binance_ohlcv", "robots.txt")
        manager.check_opt_out("polygon_stocks", "robots.txt")

        checks = manager.get_opt_out_checks(source_id="binance_ohlcv")
        assert all(c["source_id"] == "binance_ohlcv" for c in checks)


class TestRightsHolderRequests:
    """Test rights holder request handling."""

    @pytest.fixture
    def manager(self) -> CopyrightComplianceManager:
        """Create manager instance."""
        return create_copyright_manager()

    def test_record_request(self, manager):
        """Test recording rights holder request."""
        request = manager.record_rights_holder_request(
            requester_name="John Doe",
            requester_email="john@example.com",
            request_type="information",
            content_description="Market data usage inquiry",
        )
        assert request is not None
        assert request.requester_name == "John Doe"
        assert request.status == "received"

    def test_request_types(self, manager):
        """Test different request types."""
        types = ["information", "opt_out", "removal", "inquiry"]
        for req_type in types:
            request = manager.record_rights_holder_request(
                requester_name="Test",
                requester_email="test@example.com",
                request_type=req_type,
                content_description="Test request",
            )
            assert request.request_type == req_type


class TestPolicyDocument:
    """Test policy document generation."""

    @pytest.fixture
    def manager(self) -> CopyrightComplianceManager:
        """Create manager instance."""
        return create_copyright_manager()

    def test_generate_policy(self, manager):
        """Test policy document generation."""
        doc = manager.generate_policy_document()
        assert "Copyright Compliance Policy" in doc
        assert "Article 53" in doc
        assert "opt-out" in doc.lower()

    def test_policy_includes_sources(self, manager):
        """Test policy includes data sources."""
        doc = manager.generate_policy_document()
        assert "Binance" in doc or "binance" in doc

    def test_policy_includes_dsm_reference(self, manager):
        """Test policy references DSM Directive."""
        doc = manager.generate_policy_document()
        assert "2019/790" in doc or "DSM" in doc


class TestSourceCompliance:
    """Test individual source compliance verification."""

    @pytest.fixture
    def manager(self) -> CopyrightComplianceManager:
        """Create manager instance."""
        return create_copyright_manager()

    def test_verify_existing_source(self, manager):
        """Test verifying compliance of existing source."""
        result = manager.verify_source_compliance("binance_ohlcv")
        assert result["found"] is True
        assert "compliant" in result

    def test_verify_nonexistent_source(self, manager):
        """Test verifying nonexistent source."""
        result = manager.verify_source_compliance("nonexistent")
        assert result["found"] is False
        assert result["compliant"] is False

    def test_verify_licensed_source(self, manager):
        """Test verifying licensed source has license info."""
        result = manager.verify_source_compliance("polygon_stocks")
        assert result["found"] is True
        assert result["checks"]["license_documented"] is True


class TestFactoryFunctions:
    """Test factory and utility functions."""

    def test_create_copyright_manager(self):
        """Test factory function."""
        manager = create_copyright_manager()
        assert isinstance(manager, CopyrightComplianceManager)

    def test_get_default_data_sources(self):
        """Test getting default data sources."""
        sources = get_default_data_sources()
        assert len(sources) > 0
        assert all(isinstance(s, DataSourceRecord) for s in sources)


class TestValidateSourceRecord:
    """Test source record validation."""

    def test_validate_complete_record(self):
        """Test validating complete record."""
        source = DataSourceRecord(
            source_id="test",
            source_name="Test",
            source_type=DataSourceType.PUBLIC_MARKET_DATA,
            copyright_status=CopyrightStatus.NOT_APPLICABLE,
            provider="Provider",
        )
        result = validate_source_record(source)
        assert result["all_valid"] is True

    def test_validate_licensed_source_without_license(self):
        """Test licensed source without license info."""
        source = DataSourceRecord(
            source_id="test",
            source_name="Test",
            source_type=DataSourceType.LICENSED_DATA,
            copyright_status=CopyrightStatus.LICENSED,
            provider="Provider",
            license_type=None,  # Missing!
        )
        result = validate_source_record(source)
        assert result["has_license_type"] is False
        assert result["all_valid"] is False

    def test_validate_licensed_source_with_license(self):
        """Test licensed source with license info."""
        source = DataSourceRecord(
            source_id="test",
            source_name="Test",
            source_type=DataSourceType.LICENSED_DATA,
            copyright_status=CopyrightStatus.LICENSED,
            provider="Provider",
            license_type="Commercial",
        )
        result = validate_source_record(source)
        assert result["has_license_type"] is True


class TestArticle53cCompliance:
    """Integration tests for Article 53(1)(c) compliance."""

    @pytest.fixture
    def manager(self) -> CopyrightComplianceManager:
        """Create manager instance."""
        return create_copyright_manager()

    def test_all_sources_have_copyright_status(self, manager):
        """Test all sources have copyright status defined."""
        for source in manager.data_sources.values():
            assert source.copyright_status is not None

    def test_licensed_sources_have_license_info(self, manager):
        """Test licensed sources have license information."""
        for source in manager.data_sources.values():
            if source.copyright_status == CopyrightStatus.LICENSED:
                assert source.license_type is not None

    def test_compliance_percentage_calculation(self, manager):
        """Test compliance percentage is calculated correctly."""
        status = manager.get_compliance_status()

        # All default sources should be pre-checked
        assert status["compliance_percentage"] >= 80

    def test_full_compliance_workflow(self, manager):
        """Test full compliance workflow."""
        # 1. Register new source
        source = DataSourceRecord(
            source_id="new_test",
            source_name="New Test Source",
            source_type=DataSourceType.OPEN_DATA,
            copyright_status=CopyrightStatus.PUBLIC_DOMAIN,
            provider="Test Provider",
            description="Test data source",
        )
        manager.register_data_source(source)

        # 2. Check opt-out
        check = manager.check_opt_out("new_test", "robots.txt")
        assert check is not None

        # 3. Verify compliance
        result = manager.verify_source_compliance("new_test")
        assert result["compliant"] is True

        # 4. Get updated status
        status = manager.get_compliance_status()
        assert status["total_sources"] > len(DEFAULT_DATA_SOURCES)


class TestOptOutMechanisms:
    """Test DSM Directive Article 4(3) opt-out compliance."""

    @pytest.fixture
    def manager(self) -> CopyrightComplianceManager:
        """Create manager instance."""
        return create_copyright_manager()

    def test_opt_out_check_records_date(self, manager):
        """Test opt-out check records timestamp."""
        check = manager.check_opt_out("binance_ohlcv", "robots.txt")
        assert check.check_date is not None
        assert isinstance(check.check_date, datetime)

    def test_opt_out_updates_source(self, manager):
        """Test opt-out check updates source record."""
        manager.check_opt_out("binance_ohlcv", "robots.txt")
        source = manager.data_sources["binance_ohlcv"]
        assert source.opt_out_checked is True
        assert source.opt_out_check_date is not None

    def test_supported_mechanisms(self, manager):
        """Test all supported mechanisms."""
        mechanisms = [m.value for m in OptOutMechanism]
        for mech in mechanisms:
            check = manager.check_opt_out("binance_ohlcv", mech)
            assert check.mechanism_checked == mech


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def manager(self) -> CopyrightComplianceManager:
        """Create manager instance."""
        return create_copyright_manager()

    def test_empty_source_id(self, manager):
        """Test with empty source ID."""
        source = DataSourceRecord(
            source_id="",
            source_name="Test",
            source_type=DataSourceType.SYNTHETIC,
            copyright_status=CopyrightStatus.NOT_APPLICABLE,
            provider="Internal",
        )
        manager.register_data_source(source)
        assert "" in manager.data_sources

    def test_special_characters_in_source_id(self, manager):
        """Test source ID with special characters."""
        source = DataSourceRecord(
            source_id="test/source:v1",
            source_name="Test",
            source_type=DataSourceType.SYNTHETIC,
            copyright_status=CopyrightStatus.NOT_APPLICABLE,
            provider="Internal",
        )
        manager.register_data_source(source)
        assert "test/source:v1" in manager.data_sources

    def test_compliance_with_no_sources(self):
        """Test compliance status with no sources."""
        manager = CopyrightComplianceManager()
        manager.data_sources.clear()
        status = manager.get_compliance_status()
        assert status["compliance_percentage"] == 100  # Division by zero handling

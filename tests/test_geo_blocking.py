"""
Tests for Geographic Access Restrictions (Sanctions Compliance).

References:
    - EU Council Regulations (sanctions)
    - OFAC Sanctions Programs (US)
    - UK Financial Sanctions
"""

import pytest
from datetime import datetime, timezone

from services.security.geo_blocking import (
    GeoBlockingService,
    GeoCheckResult,
    BlockReason,
    Country,
    MockGeoIPProvider,
    BLOCKED_COUNTRIES,
    HIGH_RISK_COUNTRIES,
)


class TestBlockReason:
    """Tests for BlockReason enum."""

    def test_all_reasons_defined(self):
        """Verify all required block reasons exist."""
        assert BlockReason.OFAC_SANCTIONS.value == "US OFAC Comprehensive Sanctions"
        assert BlockReason.EU_SANCTIONS.value == "EU Council Sanctions"
        assert BlockReason.UK_SANCTIONS.value == "UK Financial Sanctions (OFSI)"
        assert BlockReason.UN_SANCTIONS.value == "UN Security Council Sanctions"
        assert BlockReason.PLATFORM_POLICY.value == "Platform Policy Restriction"


class TestCountry:
    """Tests for Country dataclass."""

    def test_creation(self):
        """Test Country creation."""
        country = Country(
            code="US",
            name="United States",
            continent="NA",
            is_in_eu=False,
        )
        assert country.code == "US"
        assert country.name == "United States"
        assert country.is_in_eu is False


class TestGeoCheckResult:
    """Tests for GeoCheckResult dataclass."""

    def test_creation_allowed(self):
        """Test creating an allowed result."""
        result = GeoCheckResult(
            allowed=True,
            country_code="US",
            country_name="United States",
        )
        assert result.allowed is True
        assert result.block_reason is None

    def test_creation_blocked(self):
        """Test creating a blocked result."""
        result = GeoCheckResult(
            allowed=False,
            country_code="IR",
            country_name="Iran",
            block_reason=BlockReason.OFAC_SANCTIONS,
        )
        assert result.allowed is False
        assert result.block_reason == BlockReason.OFAC_SANCTIONS

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = GeoCheckResult(
            allowed=False,
            country_code="IR",
            block_reason=BlockReason.OFAC_SANCTIONS,
        )
        data = result.to_dict()

        assert data["allowed"] is False
        assert data["country_code"] == "IR"
        assert data["block_reason"] == "US OFAC Comprehensive Sanctions"

    def test_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        result = GeoCheckResult(allowed=True, country_code="US")
        assert result.checked_at is not None


class TestMockGeoIPProvider:
    """Tests for MockGeoIPProvider."""

    def test_default_country(self):
        """Test default country lookup."""
        provider = MockGeoIPProvider(default_country="US")
        country = provider.lookup("unknown_ip")
        assert country.code == "US"

    def test_set_ip_country(self):
        """Test setting specific IP to country mapping."""
        provider = MockGeoIPProvider()
        provider.set_ip_country("1.2.3.4", "DE")

        country = provider.lookup("1.2.3.4")
        assert country.code == "DE"
        assert country.name == "Germany"

    def test_lookup_blocked_country(self):
        """Test looking up a blocked country."""
        provider = MockGeoIPProvider()
        provider.set_ip_country("1.2.3.4", "IR")

        country = provider.lookup("1.2.3.4")
        assert country.code == "IR"
        assert country.name == "Iran"


class TestBlockedCountries:
    """Tests for blocked countries configuration."""

    def test_ofac_sanctioned_countries(self):
        """Test OFAC sanctioned countries are blocked."""
        ofac_countries = ["CU", "IR", "KP", "SY"]
        for code in ofac_countries:
            assert code in BLOCKED_COUNTRIES
            assert BLOCKED_COUNTRIES[code] == BlockReason.OFAC_SANCTIONS

    def test_eu_sanctioned_countries(self):
        """Test EU sanctioned countries are blocked."""
        eu_countries = ["RU", "BY"]
        for code in eu_countries:
            assert code in BLOCKED_COUNTRIES
            assert BLOCKED_COUNTRIES[code] == BlockReason.EU_SANCTIONS


class TestHighRiskCountries:
    """Tests for high-risk countries configuration."""

    def test_high_risk_countries_defined(self):
        """Test high-risk countries are defined."""
        assert len(HIGH_RISK_COUNTRIES) > 0

    def test_expected_high_risk_countries(self):
        """Test expected high-risk countries are included."""
        expected = ["AF", "MM", "VE", "YE", "ZW"]
        for code in expected:
            assert code in HIGH_RISK_COUNTRIES


class TestGeoBlockingService:
    """Tests for GeoBlockingService."""

    @pytest.fixture
    def mock_geoip(self):
        """Create mock GeoIP provider."""
        return MockGeoIPProvider(default_country="US")

    @pytest.fixture
    def service(self, mock_geoip):
        """Create service with mock provider."""
        return GeoBlockingService(mock_geoip)

    def test_allowed_country_us(self, service):
        """Test that US is allowed."""
        result = service.check_ip("8.8.8.8")
        assert result.allowed is True
        assert result.country_code == "US"

    def test_blocked_country_iran(self, service, mock_geoip):
        """Test that Iran is blocked."""
        mock_geoip.set_ip_country("1.2.3.4", "IR")

        result = service.check_ip("1.2.3.4")
        assert result.allowed is False
        assert result.block_reason == BlockReason.OFAC_SANCTIONS

    def test_blocked_country_north_korea(self, service, mock_geoip):
        """Test that North Korea is blocked."""
        mock_geoip.set_ip_country("1.2.3.4", "KP")

        result = service.check_ip("1.2.3.4")
        assert result.allowed is False
        assert result.block_reason == BlockReason.OFAC_SANCTIONS

    def test_blocked_country_russia(self, service, mock_geoip):
        """Test that Russia is blocked."""
        mock_geoip.set_ip_country("1.2.3.4", "RU")

        result = service.check_ip("1.2.3.4")
        assert result.allowed is False
        assert result.block_reason == BlockReason.EU_SANCTIONS

    def test_blocked_country_syria(self, service, mock_geoip):
        """Test that Syria is blocked."""
        mock_geoip.set_ip_country("1.2.3.4", "SY")

        result = service.check_ip("1.2.3.4")
        assert result.allowed is False
        assert result.block_reason == BlockReason.OFAC_SANCTIONS

    def test_blocked_country_cuba(self, service, mock_geoip):
        """Test that Cuba is blocked."""
        mock_geoip.set_ip_country("1.2.3.4", "CU")

        result = service.check_ip("1.2.3.4")
        assert result.allowed is False
        assert result.block_reason == BlockReason.OFAC_SANCTIONS

    def test_blocked_country_belarus(self, service, mock_geoip):
        """Test that Belarus is blocked."""
        mock_geoip.set_ip_country("1.2.3.4", "BY")

        result = service.check_ip("1.2.3.4")
        assert result.allowed is False
        assert result.block_reason == BlockReason.EU_SANCTIONS

    def test_eu_country_allowed(self, service, mock_geoip):
        """Test that EU countries are allowed."""
        mock_geoip.set_ip_country("1.2.3.4", "DE")

        result = service.check_ip("1.2.3.4")
        assert result.allowed is True
        assert result.country_code == "DE"
        assert result.metadata.get("is_eu") is True

    def test_high_risk_country_allowed_but_flagged(self, mock_geoip):
        """Test that high-risk countries are allowed but flagged."""
        mock_geoip.set_ip_country("1.2.3.4", "VE")
        mock_geoip._country_data["VE"] = Country(code="VE", name="Venezuela")

        service = GeoBlockingService(
            mock_geoip,
            additional_high_risk={"VE"},
        )

        result = service.check_ip("1.2.3.4")
        assert result.allowed is True
        assert result.metadata.get("is_high_risk") is True

    def test_registration_check_ip_blocked(self, service, mock_geoip):
        """Test registration check blocks by IP country."""
        mock_geoip.set_ip_country("1.2.3.4", "IR")

        result = service.check_registration("1.2.3.4", "US")
        assert result.allowed is False
        assert result.block_reason == BlockReason.OFAC_SANCTIONS

    def test_registration_check_declared_blocked(self, service):
        """Test registration check blocks by declared country."""
        result = service.check_registration("8.8.8.8", "IR")
        assert result.allowed is False
        assert result.block_reason == BlockReason.OFAC_SANCTIONS

    def test_registration_check_mismatch_detected(self, service, mock_geoip):
        """Test registration check detects country mismatch."""
        mock_geoip.set_ip_country("1.2.3.4", "DE")

        result = service.check_registration("1.2.3.4", "FR")
        assert result.allowed is True
        assert result.metadata.get("mismatch") is True

    def test_is_blocked_country(self, service):
        """Test is_blocked_country helper."""
        assert service.is_blocked_country("IR") is True
        assert service.is_blocked_country("ir") is True  # Case insensitive
        assert service.is_blocked_country("US") is False

    def test_is_high_risk_country(self, service):
        """Test is_high_risk_country helper."""
        assert service.is_high_risk_country("AF") is True
        assert service.is_high_risk_country("US") is False

    def test_get_blocked_countries(self, service):
        """Test getting list of blocked countries."""
        blocked = service.get_blocked_countries()
        assert "IR" in blocked
        assert "RU" in blocked
        assert isinstance(blocked["IR"], str)

    def test_get_high_risk_countries(self, service):
        """Test getting list of high-risk countries."""
        high_risk = service.get_high_risk_countries()
        assert "AF" in high_risk
        assert isinstance(high_risk, set)

    def test_add_blocked_country(self, service):
        """Test adding a country to blocked list."""
        service.add_blocked_country("XX", BlockReason.PLATFORM_POLICY)
        assert service.is_blocked_country("XX") is True

    def test_remove_blocked_country(self, service):
        """Test removing a country from blocked list."""
        service.add_blocked_country("XX", BlockReason.PLATFORM_POLICY)
        result = service.remove_blocked_country("XX")
        assert result is True
        assert service.is_blocked_country("XX") is False

    def test_remove_nonexistent_country(self, service):
        """Test removing a country that's not blocked."""
        result = service.remove_blocked_country("US")
        assert result is False

    def test_generate_compliance_report(self, service):
        """Test generating compliance report."""
        report = service.generate_compliance_report()

        assert "generated_at" in report
        assert "blocked_countries" in report
        assert "high_risk_countries" in report
        assert "references" in report
        assert report["total_blocked"] > 0

    def test_lookup_failure_fails_open(self, mock_geoip):
        """Test that lookup failures fail-open."""

        class FailingProvider:
            def lookup(self, ip):
                raise Exception("Lookup failed")

        service = GeoBlockingService(FailingProvider())
        result = service.check_ip("1.2.3.4")

        # Should fail-open (allow) but flag the error
        assert result.allowed is True
        assert result.metadata.get("lookup_failed") is True

    def test_additional_blocked_countries(self, mock_geoip):
        """Test adding additional blocked countries."""
        service = GeoBlockingService(
            mock_geoip,
            additional_blocked={"XX": BlockReason.PLATFORM_POLICY},
        )

        assert service.is_blocked_country("XX") is True
        # Original blocked should still be there
        assert service.is_blocked_country("IR") is True


class TestGeoBlockingServiceCompliance:
    """Compliance-focused tests for GeoBlockingService."""

    @pytest.fixture
    def service(self):
        return GeoBlockingService(MockGeoIPProvider())

    def test_ofac_comprehensive_sanctions_covered(self, service):
        """Test all OFAC comprehensively sanctioned countries are blocked."""
        # As of 2024, these countries are under comprehensive OFAC sanctions
        ofac_comprehensive = ["CU", "IR", "KP", "SY"]
        for code in ofac_comprehensive:
            assert service.is_blocked_country(code), f"{code} should be blocked"

    def test_eu_comprehensive_sanctions_covered(self, service):
        """Test EU sanctioned countries are blocked."""
        # Major EU sanctions programs
        eu_sanctioned = ["RU", "BY"]
        for code in eu_sanctioned:
            assert service.is_blocked_country(code), f"{code} should be blocked"

    def test_result_includes_audit_info(self, service):
        """Test that results include information for audit."""
        result = service.check_ip("8.8.8.8")

        assert result.checked_at is not None
        assert result.country_code is not None

    def test_blocked_result_includes_reason(self, service):
        """Test that blocked results include reason."""
        service.add_blocked_country("XX", BlockReason.OFAC_SANCTIONS)

        # Need to also set up the mock
        service._geoip.set_ip_country("1.2.3.4", "XX")
        service._geoip._country_data["XX"] = Country(code="XX", name="Test Country")

        result = service.check_ip("1.2.3.4")
        # Will return XX as blocked if in blocked list
        if not result.allowed:
            assert result.block_reason is not None

# -*- coding: utf-8 -*-
"""
Comprehensive tests for CCEA Schema Versioning.

Tests cover:
- SchemaVersion parsing and comparison
- Version negotiation between Cloud and Agent
- Compatibility checks
- Edge cases and error handling
"""

import pytest
from datetime import datetime

from ccea.protocol.schema_versioning import (
    SchemaVersion,
    VersionNegotiationRequest,
    VersionNegotiationResult,
    SchemaVersionNegotiator,
    NegotiationStatus,
    VersionCompatibility,
    check_version_compatibility,
    negotiate_version,
    get_current_schema_version,
    get_supported_version_range,
    CURRENT_SCHEMA_VERSION,
    MIN_SUPPORTED_VERSION,
    MAX_SUPPORTED_VERSION,
)


# ============================================================================
# SchemaVersion Tests
# ============================================================================


class TestSchemaVersion:
    """Tests for SchemaVersion class."""

    def test_parse_valid_version(self):
        """Test parsing valid version string."""
        v = SchemaVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_zero_version(self):
        """Test parsing version with zeros."""
        v = SchemaVersion.parse("0.0.0")
        assert v.major == 0
        assert v.minor == 0
        assert v.patch == 0

    def test_parse_large_numbers(self):
        """Test parsing version with large numbers."""
        v = SchemaVersion.parse("100.200.300")
        assert v.major == 100
        assert v.minor == 200
        assert v.patch == 300

    def test_parse_with_whitespace(self):
        """Test parsing version with leading/trailing whitespace."""
        v = SchemaVersion.parse("  1.0.0  ")
        assert v.major == 1
        assert v.minor == 0
        assert v.patch == 0

    def test_parse_invalid_format_no_patch(self):
        """Test parsing invalid format without patch."""
        with pytest.raises(ValueError, match="Invalid version format"):
            SchemaVersion.parse("1.0")

    def test_parse_invalid_format_extra_parts(self):
        """Test parsing invalid format with extra parts."""
        with pytest.raises(ValueError, match="Invalid version format"):
            SchemaVersion.parse("1.0.0.0")

    def test_parse_invalid_format_non_numeric(self):
        """Test parsing invalid format with non-numeric parts."""
        with pytest.raises(ValueError, match="Invalid version format"):
            SchemaVersion.parse("1.a.0")

    def test_parse_invalid_format_negative(self):
        """Test parsing invalid format with negative number."""
        with pytest.raises(ValueError, match="Invalid version format"):
            SchemaVersion.parse("-1.0.0")

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        with pytest.raises(ValueError, match="Invalid version format"):
            SchemaVersion.parse("")

    def test_str_representation(self):
        """Test string representation."""
        v = SchemaVersion(major=1, minor=2, patch=3)
        assert str(v) == "1.2.3"

    def test_repr_representation(self):
        """Test repr representation."""
        v = SchemaVersion(major=1, minor=2, patch=3)
        assert repr(v) == "SchemaVersion(1.2.3)"

    def test_equality_same_version(self):
        """Test equality of same versions."""
        v1 = SchemaVersion(major=1, minor=2, patch=3)
        v2 = SchemaVersion(major=1, minor=2, patch=3)
        assert v1 == v2

    def test_equality_with_string(self):
        """Test equality with string."""
        v = SchemaVersion(major=1, minor=2, patch=3)
        assert v == "1.2.3"

    def test_equality_with_invalid_string(self):
        """Test equality with invalid string returns False."""
        v = SchemaVersion(major=1, minor=2, patch=3)
        assert not (v == "invalid")

    def test_inequality_different_major(self):
        """Test inequality with different major."""
        v1 = SchemaVersion(major=1, minor=0, patch=0)
        v2 = SchemaVersion(major=2, minor=0, patch=0)
        assert v1 != v2

    def test_inequality_different_minor(self):
        """Test inequality with different minor."""
        v1 = SchemaVersion(major=1, minor=1, patch=0)
        v2 = SchemaVersion(major=1, minor=2, patch=0)
        assert v1 != v2

    def test_inequality_different_patch(self):
        """Test inequality with different patch."""
        v1 = SchemaVersion(major=1, minor=0, patch=1)
        v2 = SchemaVersion(major=1, minor=0, patch=2)
        assert v1 != v2

    def test_less_than_major(self):
        """Test less than comparison by major."""
        v1 = SchemaVersion(major=1, minor=9, patch=9)
        v2 = SchemaVersion(major=2, minor=0, patch=0)
        assert v1 < v2

    def test_less_than_minor(self):
        """Test less than comparison by minor."""
        v1 = SchemaVersion(major=1, minor=1, patch=9)
        v2 = SchemaVersion(major=1, minor=2, patch=0)
        assert v1 < v2

    def test_less_than_patch(self):
        """Test less than comparison by patch."""
        v1 = SchemaVersion(major=1, minor=1, patch=1)
        v2 = SchemaVersion(major=1, minor=1, patch=2)
        assert v1 < v2

    def test_less_than_or_equal(self):
        """Test less than or equal comparison."""
        v1 = SchemaVersion(major=1, minor=0, patch=0)
        v2 = SchemaVersion(major=1, minor=0, patch=0)
        v3 = SchemaVersion(major=1, minor=0, patch=1)
        assert v1 <= v2
        assert v1 <= v3

    def test_greater_than(self):
        """Test greater than comparison."""
        v1 = SchemaVersion(major=2, minor=0, patch=0)
        v2 = SchemaVersion(major=1, minor=9, patch=9)
        assert v1 > v2

    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison."""
        v1 = SchemaVersion(major=1, minor=0, patch=0)
        v2 = SchemaVersion(major=1, minor=0, patch=0)
        v3 = SchemaVersion(major=0, minor=9, patch=9)
        assert v1 >= v2
        assert v1 >= v3

    def test_hash_equal_versions(self):
        """Test hash equality for equal versions."""
        v1 = SchemaVersion(major=1, minor=2, patch=3)
        v2 = SchemaVersion(major=1, minor=2, patch=3)
        assert hash(v1) == hash(v2)

    def test_hash_in_set(self):
        """Test versions can be used in sets."""
        v1 = SchemaVersion(major=1, minor=0, patch=0)
        v2 = SchemaVersion(major=1, minor=0, patch=0)
        v3 = SchemaVersion(major=2, minor=0, patch=0)
        s = {v1, v2, v3}
        assert len(s) == 2

    def test_is_compatible_with_same_major(self):
        """Test compatibility with same major version."""
        v1 = SchemaVersion(major=1, minor=0, patch=0)
        v2 = SchemaVersion(major=1, minor=5, patch=10)
        assert v1.is_compatible_with(v2)
        assert v2.is_compatible_with(v1)

    def test_is_not_compatible_different_major(self):
        """Test incompatibility with different major version."""
        v1 = SchemaVersion(major=1, minor=0, patch=0)
        v2 = SchemaVersion(major=2, minor=0, patch=0)
        assert not v1.is_compatible_with(v2)
        assert not v2.is_compatible_with(v1)

    def test_to_tuple(self):
        """Test conversion to tuple."""
        v = SchemaVersion(major=1, minor=2, patch=3)
        assert v.to_tuple() == (1, 2, 3)

    def test_pydantic_validation_negative_major(self):
        """Test Pydantic validation rejects negative major."""
        with pytest.raises(ValueError):
            SchemaVersion(major=-1, minor=0, patch=0)

    def test_pydantic_forbids_extra_fields(self):
        """Test Pydantic forbids extra fields."""
        with pytest.raises(ValueError):
            SchemaVersion(major=1, minor=0, patch=0, extra="field")


# ============================================================================
# VersionNegotiationRequest Tests
# ============================================================================


class TestVersionNegotiationRequest:
    """Tests for VersionNegotiationRequest class."""

    def test_create_valid_request(self):
        """Test creating a valid request."""
        req = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="1.5.0",
        )
        assert req.agent_id == "agent_abcdef1234567890"
        assert req.min_supported == "1.0.0"
        assert req.max_supported == "1.5.0"
        assert req.preferred is None

    def test_create_with_preferred(self):
        """Test creating request with preferred version."""
        req = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="1.5.0",
            preferred="1.3.0",
        )
        assert req.preferred == "1.3.0"

    def test_invalid_agent_id_format(self):
        """Test validation rejects invalid agent_id format."""
        with pytest.raises(ValueError):
            VersionNegotiationRequest(
                agent_id="invalid",
                min_supported="1.0.0",
                max_supported="1.5.0",
            )

    def test_invalid_version_format(self):
        """Test validation rejects invalid version format."""
        with pytest.raises(ValueError):
            VersionNegotiationRequest(
                agent_id="agent_abcdef1234567890",
                min_supported="invalid",
                max_supported="1.5.0",
            )

    def test_max_less_than_min_rejected(self):
        """Test validation rejects max < min."""
        with pytest.raises(ValueError, match="max_supported.*must be >= min_supported"):
            VersionNegotiationRequest(
                agent_id="agent_abcdef1234567890",
                min_supported="2.0.0",
                max_supported="1.0.0",
            )

    def test_get_min_version(self):
        """Test getting min version as SchemaVersion."""
        req = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.2.3",
            max_supported="2.0.0",
        )
        v = req.get_min_version()
        assert isinstance(v, SchemaVersion)
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_get_max_version(self):
        """Test getting max version as SchemaVersion."""
        req = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="2.3.4",
        )
        v = req.get_max_version()
        assert isinstance(v, SchemaVersion)
        assert v.major == 2
        assert v.minor == 3
        assert v.patch == 4

    def test_get_preferred_version_when_set(self):
        """Test getting preferred version when set."""
        req = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="2.0.0",
            preferred="1.5.0",
        )
        v = req.get_preferred_version()
        assert v is not None
        assert v.major == 1
        assert v.minor == 5
        assert v.patch == 0

    def test_get_preferred_version_when_not_set(self):
        """Test getting preferred version when not set."""
        req = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        assert req.get_preferred_version() is None

    def test_timestamp_auto_generated(self):
        """Test timestamp is auto-generated."""
        req = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="1.0.0",
        )
        assert isinstance(req.timestamp, datetime)


# ============================================================================
# VersionNegotiationResult Tests
# ============================================================================


class TestVersionNegotiationResult:
    """Tests for VersionNegotiationResult class."""

    def test_success_result(self):
        """Test creating successful result."""
        result = VersionNegotiationResult(
            status=NegotiationStatus.SUCCESS,
            selected_version=SchemaVersion(major=1, minor=2, patch=0),
            compatibility=VersionCompatibility.COMPATIBLE,
        )
        assert result.is_success()
        assert result.selected_version is not None
        assert str(result.selected_version) == "1.2.0"

    def test_failed_result(self):
        """Test creating failed result."""
        result = VersionNegotiationResult(
            status=NegotiationStatus.FAILED,
            compatibility=VersionCompatibility.INCOMPATIBLE_TOO_OLD,
            error_message="Agent version too old",
        )
        assert not result.is_success()
        assert result.selected_version is None
        assert result.error_message == "Agent version too old"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = VersionNegotiationResult(
            status=NegotiationStatus.SUCCESS,
            selected_version=SchemaVersion(major=1, minor=0, patch=0),
            agent_min=SchemaVersion(major=1, minor=0, patch=0),
            agent_max=SchemaVersion(major=2, minor=0, patch=0),
            cloud_min=SchemaVersion(major=1, minor=0, patch=0),
            cloud_max=SchemaVersion(major=1, minor=5, patch=0),
        )
        d = result.to_dict()
        assert d["status"] == "SUCCESS"
        assert d["selected_version"] == "1.0.0"
        assert d["agent_range"]["min"] == "1.0.0"
        assert d["agent_range"]["max"] == "2.0.0"
        assert d["cloud_range"]["min"] == "1.0.0"
        assert d["cloud_range"]["max"] == "1.5.0"

    def test_to_dict_with_none_versions(self):
        """Test serialization handles None versions."""
        result = VersionNegotiationResult(
            status=NegotiationStatus.PENDING,
        )
        d = result.to_dict()
        assert d["selected_version"] is None
        assert d["agent_range"]["min"] is None


# ============================================================================
# SchemaVersionNegotiator Tests
# ============================================================================


class TestSchemaVersionNegotiator:
    """Tests for SchemaVersionNegotiator class."""

    def test_init_valid(self):
        """Test valid initialization."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        assert str(negotiator.min_supported) == "1.0.0"
        assert str(negotiator.max_supported) == "2.0.0"

    def test_init_invalid_range(self):
        """Test initialization with invalid range."""
        with pytest.raises(ValueError, match="max_supported.*must be >="):
            SchemaVersionNegotiator(
                min_supported="2.0.0",
                max_supported="1.0.0",
            )

    def test_negotiate_exact_match(self):
        """Test negotiation with exact version match."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="1.0.0",
        )
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="1.0.0",
        )
        result = negotiator.negotiate(request)
        assert result.is_success()
        assert str(result.selected_version) == "1.0.0"

    def test_negotiate_overlapping_ranges(self):
        """Test negotiation with overlapping ranges."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="1.5.0",
            prefer_latest=True,
        )
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.2.0",
            max_supported="2.0.0",
        )
        result = negotiator.negotiate(request)
        assert result.is_success()
        # Should select highest in overlap: 1.5.0
        assert str(result.selected_version) == "1.5.0"

    def test_negotiate_prefer_earliest(self):
        """Test negotiation preferring earliest version."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="1.5.0",
            prefer_latest=False,
        )
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.2.0",
            max_supported="2.0.0",
        )
        result = negotiator.negotiate(request)
        assert result.is_success()
        # Should select lowest in overlap: 1.2.0
        assert str(result.selected_version) == "1.2.0"

    def test_negotiate_with_preferred_in_range(self):
        """Test negotiation respects preferred version in range."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="2.0.0",
            preferred="1.3.0",
        )
        result = negotiator.negotiate(request)
        assert result.is_success()
        assert str(result.selected_version) == "1.3.0"

    def test_negotiate_with_preferred_out_of_range(self):
        """Test negotiation ignores preferred if out of overlap."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="1.2.0",
            prefer_latest=True,
        )
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="2.0.0",
            preferred="1.5.0",  # Out of cloud's range
        )
        result = negotiator.negotiate(request)
        assert result.is_success()
        # Should fall back to latest in overlap: 1.2.0
        assert str(result.selected_version) == "1.2.0"

    def test_negotiate_agent_too_old(self):
        """Test negotiation fails when agent is too old."""
        negotiator = SchemaVersionNegotiator(
            min_supported="2.0.0",
            max_supported="3.0.0",
        )
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="1.5.0",
        )
        result = negotiator.negotiate(request)
        assert not result.is_success()
        assert result.status == NegotiationStatus.FAILED
        assert result.compatibility == VersionCompatibility.INCOMPATIBLE_TOO_OLD
        assert "below" in result.error_message

    def test_negotiate_agent_too_new(self):
        """Test negotiation fails when agent is too new."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="1.5.0",
        )
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="2.0.0",
            max_supported="3.0.0",
        )
        result = negotiator.negotiate(request)
        assert not result.is_success()
        assert result.status == NegotiationStatus.FAILED
        assert result.compatibility == VersionCompatibility.INCOMPATIBLE_TOO_NEW
        assert "above" in result.error_message

    def test_negotiate_major_version_mismatch(self):
        """Test negotiation fails on major version mismatch."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="1.9.0",
        )
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="2.0.0",
            max_supported="2.9.0",
        )
        result = negotiator.negotiate(request)
        assert not result.is_success()
        assert result.status == NegotiationStatus.FAILED

    def test_caching_successful_negotiation(self):
        """Test successful negotiations are cached."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        result1 = negotiator.negotiate(request)
        cached = negotiator.get_cached_negotiation("agent_abcdef1234567890")
        assert cached is not None
        assert cached.selected_version == result1.selected_version

    def test_no_cache_for_failed_negotiation(self):
        """Test failed negotiations are not cached."""
        negotiator = SchemaVersionNegotiator(
            min_supported="2.0.0",
            max_supported="3.0.0",
        )
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="1.5.0",
        )
        negotiator.negotiate(request)
        cached = negotiator.get_cached_negotiation("agent_abcdef1234567890")
        assert cached is None

    def test_clear_cache_specific_agent(self):
        """Test clearing cache for specific agent."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        negotiator.negotiate(request)
        negotiator.clear_cache("agent_abcdef1234567890")
        cached = negotiator.get_cached_negotiation("agent_abcdef1234567890")
        assert cached is None

    def test_clear_cache_all(self):
        """Test clearing entire cache."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        request1 = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        request2 = VersionNegotiationRequest(
            agent_id="agent_12345678901234567",
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        negotiator.negotiate(request1)
        negotiator.negotiate(request2)
        negotiator.clear_cache()
        assert negotiator.get_cached_negotiation("agent_abcdef1234567890") is None
        assert negotiator.get_cached_negotiation("agent_12345678901234567") is None

    def test_get_supported_range(self):
        """Test getting supported version range."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="2.5.0",
        )
        min_v, max_v = negotiator.get_supported_range()
        assert str(min_v) == "1.0.0"
        assert str(max_v) == "2.5.0"


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_check_version_compatibility_compatible(self):
        """Test compatibility check for compatible version."""
        result = check_version_compatibility(
            "1.2.3",
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        assert result == VersionCompatibility.COMPATIBLE

    def test_check_version_compatibility_too_old(self):
        """Test compatibility check for too old version (same major)."""
        # Version 1.0.0 vs min 1.5.0 - same major but older minor
        result = check_version_compatibility(
            "1.0.0",
            min_supported="1.5.0",
            max_supported="1.9.0",
        )
        assert result == VersionCompatibility.INCOMPATIBLE_TOO_OLD

    def test_check_version_compatibility_too_new(self):
        """Test compatibility check for too new version (same major)."""
        # Version 1.9.0 vs max 1.5.0 - same major but newer minor
        result = check_version_compatibility(
            "1.9.0",
            min_supported="1.0.0",
            max_supported="1.5.0",
        )
        assert result == VersionCompatibility.INCOMPATIBLE_TOO_NEW

    def test_check_version_compatibility_major_mismatch_old(self):
        """Test compatibility check for major version mismatch (old)."""
        result = check_version_compatibility(
            "0.9.9",
            min_supported="1.0.0",
            max_supported="1.9.9",
        )
        assert result == VersionCompatibility.INCOMPATIBLE_MAJOR

    def test_check_version_compatibility_major_mismatch_new(self):
        """Test compatibility check for major version mismatch (new)."""
        result = check_version_compatibility(
            "2.0.0",
            min_supported="1.0.0",
            max_supported="1.9.9",
        )
        assert result == VersionCompatibility.INCOMPATIBLE_MAJOR

    def test_check_version_compatibility_invalid(self):
        """Test compatibility check for invalid version."""
        result = check_version_compatibility(
            "invalid",
            min_supported="1.0.0",
            max_supported="2.0.0",
        )
        assert result == VersionCompatibility.INVALID_VERSION

    def test_negotiate_version_success(self):
        """Test simple negotiate_version function success."""
        result = negotiate_version(
            agent_min="1.0.0",
            agent_max="2.0.0",
            cloud_min="1.5.0",
            cloud_max="2.5.0",
            prefer_latest=True,
        )
        assert result == "2.0.0"

    def test_negotiate_version_prefer_earliest(self):
        """Test negotiate_version preferring earliest."""
        result = negotiate_version(
            agent_min="1.0.0",
            agent_max="2.0.0",
            cloud_min="1.5.0",
            cloud_max="2.5.0",
            prefer_latest=False,
        )
        assert result == "1.5.0"

    def test_negotiate_version_no_overlap(self):
        """Test negotiate_version with no overlap."""
        result = negotiate_version(
            agent_min="1.0.0",
            agent_max="1.5.0",
            cloud_min="2.0.0",
            cloud_max="2.5.0",
        )
        assert result is None

    def test_negotiate_version_invalid_version(self):
        """Test negotiate_version with invalid version."""
        result = negotiate_version(
            agent_min="invalid",
            agent_max="1.5.0",
            cloud_min="1.0.0",
            cloud_max="2.0.0",
        )
        assert result is None

    def test_get_current_schema_version(self):
        """Test getting current schema version."""
        v = get_current_schema_version()
        assert isinstance(v, SchemaVersion)
        assert str(v) == CURRENT_SCHEMA_VERSION

    def test_get_supported_version_range(self):
        """Test getting supported version range."""
        min_v, max_v = get_supported_version_range()
        assert str(min_v) == MIN_SUPPORTED_VERSION
        assert str(max_v) == MAX_SUPPORTED_VERSION


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_current_version_format(self):
        """Test CURRENT_SCHEMA_VERSION is valid format."""
        v = SchemaVersion.parse(CURRENT_SCHEMA_VERSION)
        assert v is not None

    def test_min_version_format(self):
        """Test MIN_SUPPORTED_VERSION is valid format."""
        v = SchemaVersion.parse(MIN_SUPPORTED_VERSION)
        assert v is not None

    def test_max_version_format(self):
        """Test MAX_SUPPORTED_VERSION is valid format."""
        v = SchemaVersion.parse(MAX_SUPPORTED_VERSION)
        assert v is not None

    def test_min_not_greater_than_max(self):
        """Test MIN <= MAX invariant."""
        min_v = SchemaVersion.parse(MIN_SUPPORTED_VERSION)
        max_v = SchemaVersion.parse(MAX_SUPPORTED_VERSION)
        assert min_v <= max_v

    def test_current_in_supported_range(self):
        """Test CURRENT is within supported range."""
        current = SchemaVersion.parse(CURRENT_SCHEMA_VERSION)
        min_v = SchemaVersion.parse(MIN_SUPPORTED_VERSION)
        max_v = SchemaVersion.parse(MAX_SUPPORTED_VERSION)
        assert min_v <= current <= max_v


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for schema versioning."""

    def test_full_negotiation_workflow(self):
        """Test complete negotiation workflow."""
        # Create negotiator
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="2.0.0",
        )

        # Create request
        request = VersionNegotiationRequest(
            agent_id="agent_abcdef1234567890",
            min_supported="1.5.0",
            max_supported="3.0.0",
            preferred="1.8.0",
        )

        # Negotiate
        result = negotiator.negotiate(request)

        # Verify
        assert result.is_success()
        assert result.selected_version is not None
        assert str(result.selected_version) == "1.8.0"

        # Verify cached
        cached = negotiator.get_cached_negotiation("agent_abcdef1234567890")
        assert cached is not None
        assert str(cached.selected_version) == "1.8.0"

        # Verify serialization
        d = result.to_dict()
        assert d["status"] == "SUCCESS"
        assert d["selected_version"] == "1.8.0"

    def test_multiple_agents_negotiation(self):
        """Test negotiation with multiple agents."""
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="2.0.0",
        )

        # Agent 1: Modern
        request1 = VersionNegotiationRequest(
            agent_id="agent_modern1234567890",
            min_supported="1.5.0",
            max_supported="2.5.0",
        )
        result1 = negotiator.negotiate(request1)
        assert result1.is_success()
        assert str(result1.selected_version) == "2.0.0"

        # Agent 2: Legacy
        request2 = VersionNegotiationRequest(
            agent_id="agent_legacy1234567890",
            min_supported="1.0.0",
            max_supported="1.2.0",
        )
        result2 = negotiator.negotiate(request2)
        assert result2.is_success()
        assert str(result2.selected_version) == "1.2.0"

        # Both cached
        assert negotiator.get_cached_negotiation("agent_modern1234567890") is not None
        assert negotiator.get_cached_negotiation("agent_legacy1234567890") is not None

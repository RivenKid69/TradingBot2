# -*- coding: utf-8 -*-
"""
Tests for MiFID II GLEIF Client Module.

Comprehensive tests for GLEIF API integration for LEI verification.
"""

import pytest
from datetime import date, datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json

from services.compliance.gleif_client import (
    GLEIFErrorCode,
    GLEIFError,
    GLEIFEntity,
    GLEIFRegistration,
    GLEIFResponse,
    GLEIFClient,
    create_gleif_client,
)


class TestGLEIFErrorCode:
    """Tests for GLEIFErrorCode enum."""

    def test_not_found_value(self):
        assert GLEIFErrorCode.NOT_FOUND.value == "NOT_FOUND"

    def test_invalid_lei_value(self):
        assert GLEIFErrorCode.INVALID_LEI.value == "INVALID_LEI"

    def test_rate_limited_value(self):
        assert GLEIFErrorCode.RATE_LIMITED.value == "RATE_LIMITED"

    def test_api_error_value(self):
        assert GLEIFErrorCode.API_ERROR.value == "API_ERROR"

    def test_network_error_value(self):
        assert GLEIFErrorCode.NETWORK_ERROR.value == "NETWORK_ERROR"


class TestGLEIFError:
    """Tests for GLEIFError dataclass."""

    def test_create_error(self):
        error = GLEIFError(
            code=GLEIFErrorCode.NOT_FOUND,
            message="LEI not found",
            lei="5493001KJTIIGC8Y1R12",
        )
        assert error.code == GLEIFErrorCode.NOT_FOUND
        assert error.message == "LEI not found"
        assert error.lei == "5493001KJTIIGC8Y1R12"

    def test_error_str(self):
        error = GLEIFError(
            code=GLEIFErrorCode.NOT_FOUND,
            message="LEI not found",
        )
        assert "NOT_FOUND" in str(error)
        assert "LEI not found" in str(error)

    def test_error_with_http_status(self):
        error = GLEIFError(
            code=GLEIFErrorCode.API_ERROR,
            message="Server error",
            http_status=500,
        )
        assert error.http_status == 500


class TestGLEIFEntity:
    """Tests for GLEIFEntity dataclass."""

    def test_create_entity(self):
        entity = GLEIFEntity(
            lei="5493001KJTIIGC8Y1R12",
            legal_name="Test Company Ltd",
            legal_address_country="GB",
        )
        assert entity.lei == "5493001KJTIIGC8Y1R12"
        assert entity.legal_name == "Test Company Ltd"
        assert entity.legal_address_country == "GB"

    def test_entity_default_values(self):
        entity = GLEIFEntity(lei="5493001KJTIIGC8Y1R12", legal_name="Test")
        assert entity.legal_address_city == ""
        assert entity.headquarters_city == ""
        assert entity.jurisdiction == ""


class TestGLEIFRegistration:
    """Tests for GLEIFRegistration dataclass."""

    def test_create_registration(self):
        reg = GLEIFRegistration(
            initial_registration_date=date(2020, 1, 1),
            next_renewal_date=date(2025, 1, 1),
            status="ISSUED",
            managing_lou="EVK05KS7XY1DEII3R011",
        )
        assert reg.status == "ISSUED"
        assert reg.managing_lou == "EVK05KS7XY1DEII3R011"

    def test_registration_default_values(self):
        reg = GLEIFRegistration()
        assert reg.status == "ISSUED"
        assert reg.managing_lou == ""


class TestGLEIFResponse:
    """Tests for GLEIFResponse dataclass."""

    def test_create_success_response(self):
        entity = GLEIFEntity(lei="5493001KJTIIGC8Y1R12", legal_name="Test")
        registration = GLEIFRegistration(status="ISSUED")
        response = GLEIFResponse(
            lei="5493001KJTIIGC8Y1R12",
            is_success=True,
            entity=entity,
            registration=registration,
        )
        assert response.is_success is True
        assert response.entity.legal_name == "Test"

    def test_create_error_response(self):
        error = GLEIFError(code=GLEIFErrorCode.NOT_FOUND, message="Not found")
        response = GLEIFResponse(
            lei="5493001KJTIIGC8Y1R12",
            is_success=False,
            error=error,
        )
        assert response.is_success is False
        assert response.error is not None

    def test_to_lei_record_success(self):
        entity = GLEIFEntity(
            lei="5493001KJTIIGC8Y1R12",
            legal_name="Test Company",
            legal_address_country="GB",
        )
        registration = GLEIFRegistration(
            status="ISSUED",
            initial_registration_date=date(2020, 1, 1),
            next_renewal_date=date(2025, 1, 1),
        )
        response = GLEIFResponse(
            lei="5493001KJTIIGC8Y1R12",
            is_success=True,
            entity=entity,
            registration=registration,
        )
        record = response.to_lei_record()
        assert record is not None
        assert record.lei == "5493001KJTIIGC8Y1R12"
        assert record.legal_name == "Test Company"
        assert record.country == "GB"

    def test_to_lei_record_failure(self):
        response = GLEIFResponse(
            lei="5493001KJTIIGC8Y1R12",
            is_success=False,
        )
        record = response.to_lei_record()
        assert record is None


class TestGLEIFClient:
    """Tests for GLEIFClient class."""

    def test_init_default(self):
        client = GLEIFClient()
        assert client._base_url == "https://api.gleif.org/api/v1"
        assert client._enable_rate_limiting is True

    def test_init_custom_url(self):
        client = GLEIFClient(base_url="https://custom.api.com/v1/")
        assert client._base_url == "https://custom.api.com/v1"

    def test_init_disable_rate_limiting(self):
        client = GLEIFClient(enable_rate_limiting=False)
        assert client._enable_rate_limiting is False

    def test_check_rate_limit_no_wait(self):
        client = GLEIFClient()
        wait_time = client._check_rate_limit()
        assert wait_time == 0.0

    def test_check_rate_limit_disabled(self):
        client = GLEIFClient(enable_rate_limiting=False)
        # Even with many requests, should return 0
        for _ in range(100):
            client._record_request()
        wait_time = client._check_rate_limit()
        assert wait_time == 0.0

    def test_record_request(self):
        client = GLEIFClient()
        initial = client._stats["requests_total"]
        client._record_request()
        assert client._stats["requests_total"] == initial + 1

    def test_parse_date_valid(self):
        client = GLEIFClient()
        result = client._parse_date("2020-01-15")
        assert result == date(2020, 1, 15)

    def test_parse_date_none(self):
        client = GLEIFClient()
        result = client._parse_date(None)
        assert result is None

    def test_parse_date_invalid(self):
        client = GLEIFClient()
        result = client._parse_date("invalid")
        assert result is None

    def test_parse_datetime_valid(self):
        client = GLEIFClient()
        result = client._parse_datetime("2020-01-15T10:30:00Z")
        assert result is not None
        assert result.year == 2020
        assert result.month == 1

    def test_parse_datetime_none(self):
        client = GLEIFClient()
        result = client._parse_datetime(None)
        assert result is None

    def test_parse_response_empty_data(self):
        client = GLEIFClient()
        result = client._parse_response({"data": []})
        assert result.is_success is False
        assert result.error.code == GLEIFErrorCode.NOT_FOUND

    def test_parse_response_valid_data(self):
        client = GLEIFClient()
        data = {
            "data": [{
                "attributes": {
                    "lei": "5493001KJTIIGC8Y1R12",
                    "entity": {
                        "legalName": {"name": "Test Company"},
                        "legalAddress": {"country": "GB", "city": "London"},
                        "headquartersAddress": {"country": "GB"},
                        "jurisdiction": "GB",
                    },
                    "registration": {
                        "status": "ISSUED",
                        "initialRegistrationDate": "2020-01-01",
                        "nextRenewalDate": "2025-01-01",
                        "managingLou": "EVK05KS7XY1DEII3R011",
                    },
                },
            }],
        }
        result = client._parse_response(data)
        assert result.is_success is True
        assert result.lei == "5493001KJTIIGC8Y1R12"
        assert result.entity.legal_name == "Test Company"

    def test_get_statistics(self):
        client = GLEIFClient()
        client._record_request()
        stats = client.get_statistics()
        assert "requests_total" in stats
        assert "requests_success" in stats
        assert "requests_failed" in stats
        assert "current_rate_limit_queue" in stats

    @pytest.mark.asyncio
    async def test_get_lei_not_found(self):
        """Test LEI lookup when LEI doesn't exist."""
        client = GLEIFClient()

        # Mock httpx response
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.text = "Not found"

            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            response = await client.get_lei("INVALID_LEI_12345678")
            assert response.is_success is False
            assert response.error.code == GLEIFErrorCode.NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_lei_rate_limited(self):
        """Test LEI lookup when rate limited."""
        client = GLEIFClient()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 429

            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            response = await client.get_lei("5493001KJTIIGC8Y1R12")
            assert response.is_success is False
            assert response.error.code == GLEIFErrorCode.RATE_LIMITED

    def test_get_lei_sync_network_error(self):
        """Test sync LEI lookup with network error."""
        client = GLEIFClient()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Network error")

            response = client.get_lei_sync("5493001KJTIIGC8Y1R12")
            assert response.is_success is False
            assert response.error.code == GLEIFErrorCode.NETWORK_ERROR

    def test_get_lei_sync_http_error_404(self):
        """Test sync LEI lookup with 404 error."""
        client = GLEIFClient()

        import urllib.error
        with patch("urllib.request.urlopen") as mock_urlopen:
            error = urllib.error.HTTPError(
                url="test", code=404, msg="Not Found", hdrs={}, fp=None
            )
            mock_urlopen.side_effect = error

            response = client.get_lei_sync("INVALID_LEI_12345678")
            assert response.is_success is False
            assert response.error.code == GLEIFErrorCode.NOT_FOUND


class TestCreateGLEIFClient:
    """Tests for create_gleif_client factory function."""

    def test_create_default(self):
        client = create_gleif_client()
        assert isinstance(client, GLEIFClient)
        assert client._base_url == GLEIFClient.DEFAULT_BASE_URL

    def test_create_custom_url(self):
        client = create_gleif_client(base_url="https://custom.api.com")
        assert client._base_url == "https://custom.api.com"

    def test_create_custom_timeout(self):
        client = create_gleif_client(timeout=60.0)
        assert client._timeout == 60.0

    def test_create_disable_rate_limiting(self):
        client = create_gleif_client(enable_rate_limiting=False)
        assert client._enable_rate_limiting is False

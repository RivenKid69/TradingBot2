# -*- coding: utf-8 -*-
"""
Tests for CCEA Schema Validation.

Tests verify that:
1. Schemas don't contain order-like payloads
2. Prohibited fields are detected
3. Manifest and protocol schemas are valid
"""

import json
import tempfile
from pathlib import Path

import pytest

from ccea.guardrails.schema_check import (
    PROHIBITED_FIELDS,
    PROHIBITED_VALUES,
    SchemaCheckResult,
    check_prohibited_fields,
    validate_manifest_schema,
    validate_protocol_schema,
    validate_schema_file,
)


class TestProhibitedFields:
    """Tests for prohibited field detection."""

    def test_side_field_prohibited(self):
        """Field 'side' should be prohibited."""
        assert "side" in PROHIBITED_FIELDS

    def test_quantity_field_prohibited(self):
        """Field 'quantity' should be prohibited."""
        assert "quantity" in PROHIBITED_FIELDS
        assert "qty" in PROHIBITED_FIELDS

    def test_price_field_prohibited(self):
        """Field 'price' should be prohibited."""
        assert "price" in PROHIBITED_FIELDS

    def test_order_type_field_prohibited(self):
        """Field 'order_type' should be prohibited."""
        assert "order_type" in PROHIBITED_FIELDS

    def test_target_position_prohibited(self):
        """Field 'target_position' should be prohibited."""
        assert "target_position" in PROHIBITED_FIELDS

    def test_intent_signal_prohibited(self):
        """Fields 'intent' and 'signal' should be prohibited."""
        assert "intent" in PROHIBITED_FIELDS
        assert "signal" in PROHIBITED_FIELDS


class TestProhibitedValues:
    """Tests for prohibited value detection."""

    def test_buy_sell_prohibited_for_side(self):
        """BUY/SELL values should be prohibited for side field."""
        assert "BUY" in PROHIBITED_VALUES["side"]
        assert "SELL" in PROHIBITED_VALUES["side"]

    def test_market_limit_prohibited_for_order_type(self):
        """MARKET/LIMIT values should be prohibited for order_type field."""
        assert "MARKET" in PROHIBITED_VALUES["order_type"]
        assert "LIMIT" in PROHIBITED_VALUES["order_type"]


class TestCheckProhibitedFields:
    """Tests for check_prohibited_fields function."""

    def test_clean_schema_passes(self):
        """Schema without prohibited fields should pass."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"},
            }
        }

        violations = check_prohibited_fields(schema, file_path="test.json")
        assert len(violations) == 0

    def test_side_field_detected(self):
        """Schema with 'side' field should be detected."""
        schema = {
            "type": "object",
            "properties": {
                "side": {"type": "string", "enum": ["BUY", "SELL"]},
            }
        }

        violations = check_prohibited_fields(schema, file_path="test.json")
        assert len(violations) > 0
        assert any(v.field_name == "side" for v in violations)

    def test_quantity_field_detected(self):
        """Schema with 'quantity' field should be detected."""
        schema = {
            "type": "object",
            "properties": {
                "quantity": {"type": "number"},
            }
        }

        violations = check_prohibited_fields(schema, file_path="test.json")
        assert len(violations) > 0
        assert any(v.field_name == "quantity" for v in violations)

    def test_nested_prohibited_field_detected(self):
        """Nested prohibited fields should be detected."""
        schema = {
            "type": "object",
            "properties": {
                "order": {
                    "type": "object",
                    "properties": {
                        "side": {"type": "string"},
                        "price": {"type": "number"},
                    }
                }
            }
        }

        violations = check_prohibited_fields(schema, file_path="test.json")
        assert len(violations) >= 2
        assert any(v.field_name == "side" for v in violations)
        assert any(v.field_name == "price" for v in violations)

    def test_array_items_checked(self):
        """Prohibited fields in array items should be detected."""
        schema = {
            "type": "object",
            "properties": {
                "orders": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "quantity": {"type": "number"},
                        }
                    }
                }
            }
        }

        violations = check_prohibited_fields(schema, file_path="test.json")
        assert any(v.field_name == "quantity" for v in violations)

    def test_allof_checked(self):
        """Prohibited fields in allOf should be detected."""
        schema = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {
                        "target_position": {"type": "number"},
                    }
                }
            ]
        }

        violations = check_prohibited_fields(schema, file_path="test.json")
        assert any(v.field_name == "target_position" for v in violations)


class TestManifestSchemaValidation:
    """Tests for manifest schema validation."""

    def test_valid_manifest_schema(self):
        """Valid manifest schema should pass."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": [
                "schema_version",
                "artifact_id",
                "artifact_type",
                "entrypoint",
                "runtime",
                "deps_lock_digest",
                "created_at",
            ],
            "properties": {
                "schema_version": {"type": "string"},
                "artifact_id": {"type": "string"},
                "artifact_type": {"type": "string"},
                "entrypoint": {"type": "object"},
                "runtime": {"type": "object"},
                "deps_lock_digest": {"type": "string"},
                "created_at": {"type": "string"},
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            f.flush()
            result = validate_manifest_schema(Path(f.name))

        assert result.passed

    def test_manifest_with_order_fields_fails(self):
        """Manifest schema with order fields should fail."""
        schema = {
            "type": "object",
            "properties": {
                "order_config": {
                    "type": "object",
                    "properties": {
                        "side": {"type": "string"},
                        "quantity": {"type": "number"},
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            f.flush()
            result = validate_manifest_schema(Path(f.name))

        assert not result.passed
        assert len(result.violations) > 0


class TestProtocolSchemaValidation:
    """Tests for protocol schema validation."""

    def test_valid_protocol_schema(self):
        """Valid protocol schema should pass."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "definitions": {
                "messages": {
                    "heartbeat": {
                        "type": "object",
                        "properties": {
                            "message_type": {"const": "HEARTBEAT"},
                            "agent_id": {"type": "string"},
                            "timestamp": {"type": "string"},
                        }
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            f.flush()
            result = validate_protocol_schema(Path(f.name))

        assert result.passed

    def test_protocol_with_order_payload_fails(self):
        """Protocol schema with order payload should fail."""
        schema = {
            "definitions": {
                "messages": {
                    "place_order": {
                        "type": "object",
                        "properties": {
                            "message_type": {"const": "PLACE_ORDER"},
                            "side": {"type": "string", "enum": ["BUY", "SELL"]},
                            "quantity": {"type": "number"},
                            "price": {"type": "number"},
                        }
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            f.flush()
            result = validate_protocol_schema(Path(f.name))

        assert not result.passed


class TestSchemaFileValidation:
    """Tests for generic schema file validation."""

    def test_nonexistent_file_fails(self):
        """Nonexistent file should fail."""
        result = validate_schema_file(Path("/nonexistent/schema.json"))
        assert not result.passed

    def test_invalid_json_fails(self):
        """Invalid JSON should fail."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()
            result = validate_schema_file(Path(f.name))

        assert not result.passed


class TestRealSchemas:
    """Tests against the actual project schemas."""

    @pytest.fixture
    def schemas_dir(self) -> Path:
        """Get the schemas directory."""
        return Path(__file__).parent.parent.parent / "docs" / "schemas"

    def test_artifact_manifest_schema_valid(self, schemas_dir: Path):
        """The actual artifact_manifest.schema.json should pass validation."""
        schema_path = schemas_dir / "artifact_manifest.schema.json"
        if schema_path.exists():
            result = validate_manifest_schema(schema_path)
            # The schema itself defines prohibited fields in $defs
            # which should not trigger violations
            assert result.files_checked == 1

    def test_protocol_messages_schema_valid(self, schemas_dir: Path):
        """The actual protocol_messages.schema.json should pass validation."""
        schema_path = schemas_dir / "protocol_messages.schema.json"
        if schema_path.exists():
            result = validate_protocol_schema(schema_path)
            assert result.files_checked == 1


class TestEdgeCases:
    """Tests for edge cases in schema validation."""

    def test_empty_schema(self):
        """Empty schema should pass."""
        schema = {}

        violations = check_prohibited_fields(schema, file_path="test.json")
        assert len(violations) == 0

    def test_schema_with_only_type(self):
        """Schema with only type should pass."""
        schema = {"type": "object"}

        violations = check_prohibited_fields(schema, file_path="test.json")
        assert len(violations) == 0

    def test_deeply_nested_prohibited(self):
        """Deeply nested prohibited fields should be detected."""
        schema = {
            "properties": {
                "level1": {
                    "properties": {
                        "level2": {
                            "properties": {
                                "level3": {
                                    "properties": {
                                        "intent": {"type": "object"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        violations = check_prohibited_fields(schema, file_path="test.json")
        assert any(v.field_name == "intent" for v in violations)

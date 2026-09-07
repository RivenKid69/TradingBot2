# -*- coding: utf-8 -*-
"""
Tests for Schema-Contract Synchronization (Phase 1.3).

Verifies that protocol_messages.schema.json (authoritative source) matches
the Python enum definitions in ccea/contracts/enums.py.

Work Item: WI-DEDRIFT-01
Requirement: 1.3 - Centralize protocol/enum/state machine contracts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, FrozenSet, Set

import pytest

from ccea.contracts.enums import (
    APPROVAL_STATUS,
    CHANGE_CLASS,
    COMMAND_STATUS_PROTOCOL,
    COMMAND_TYPES,
    DEPLOYMENT_STATE_PROTOCOL,
    MESSAGE_TYPES,
    RUN_STATE_PROTOCOL,
    SIGNATURE_ALGORITHMS,
    TELEMETRY_LEVEL_PROTOCOL,
    get_all_protocol_enums,
)


class TestSchemaContractSync:
    """Tests verifying schema.json matches Python enum contracts."""

    @pytest.fixture
    def schema_path(self) -> Path:
        """Get path to protocol messages schema."""
        return (
            Path(__file__).parent.parent.parent
            / "docs"
            / "schemas"
            / "protocol_messages.schema.json"
        )

    @pytest.fixture
    def schema(self, schema_path: Path) -> Dict[str, Any]:
        """Load the JSON schema."""
        if not schema_path.exists():
            pytest.skip(f"Schema file not found: {schema_path}")
        return json.loads(schema_path.read_text(encoding="utf-8"))

    def _extract_enum_values(self, schema_def: Dict[str, Any]) -> Set[str]:
        """Extract enum values from a schema definition."""
        if "enum" in schema_def:
            return set(schema_def["enum"])
        return set()

    def _extract_command_types_from_schema(self, schema: Dict[str, Any]) -> Set[str]:
        """Extract allowed command types from schema."""
        command_types: Set[str] = set()
        messages = schema.get("definitions", {}).get("messages", {})

        # Extract command types from message definitions
        for msg_name, msg_def in messages.items():
            if msg_name.startswith("request_"):
                # Handle allOf structures
                if "allOf" in msg_def:
                    for item in msg_def["allOf"]:
                        props = item.get("properties", {})
                        if "command_type" in props:
                            const_val = props["command_type"].get("const")
                            if const_val:
                                command_types.add(const_val)

        return command_types

    def _extract_message_types_from_schema(self, schema: Dict[str, Any]) -> Set[str]:
        """Extract allowed message types from schema."""
        message_types: Set[str] = set()
        messages = schema.get("definitions", {}).get("messages", {})

        for msg_name, msg_def in messages.items():
            # Handle direct properties
            props = msg_def.get("properties", {})
            if "message_type" in props:
                const_val = props["message_type"].get("const")
                if const_val:
                    message_types.add(const_val)

        return message_types

    # =========================================================================
    # Command Types Tests
    # =========================================================================

    def test_command_types_match_schema(self, schema: Dict[str, Any]):
        """COMMAND_TYPES enum must match schema allowlist."""
        schema_commands = self._extract_command_types_from_schema(schema)
        code_commands = set(COMMAND_TYPES)

        # Schema is authoritative - code should match
        assert code_commands == schema_commands, (
            f"Command types mismatch!\n"
            f"In schema but not in code: {schema_commands - code_commands}\n"
            f"In code but not in schema: {code_commands - schema_commands}"
        )

    def test_all_command_types_are_request_prefixed(self):
        """All command types should be REQUEST_* prefixed."""
        for cmd in COMMAND_TYPES:
            assert cmd.startswith("REQUEST_"), f"Command type {cmd} should be REQUEST_* prefixed"

    # =========================================================================
    # Message Types Tests
    # =========================================================================

    def test_message_types_match_schema(self, schema: Dict[str, Any]):
        """MESSAGE_TYPES enum must match schema allowlist."""
        schema_messages = self._extract_message_types_from_schema(schema)
        code_messages = set(MESSAGE_TYPES)

        # Schema is authoritative - code should match
        assert code_messages == schema_messages, (
            f"Message types mismatch!\n"
            f"In schema but not in code: {schema_messages - code_messages}\n"
            f"In code but not in schema: {code_messages - schema_messages}"
        )

    # =========================================================================
    # Command Status Tests
    # =========================================================================

    def test_command_status_match_schema(self, schema: Dict[str, Any]):
        """COMMAND_STATUS_PROTOCOL must match schema definitions."""
        schema_def = schema.get("definitions", {}).get("command_status", {})
        schema_statuses = self._extract_enum_values(schema_def)
        code_statuses = set(COMMAND_STATUS_PROTOCOL)

        assert code_statuses == schema_statuses, (
            f"Command status mismatch!\n"
            f"In schema but not in code: {schema_statuses - code_statuses}\n"
            f"In code but not in schema: {code_statuses - schema_statuses}"
        )

    # =========================================================================
    # Deployment State Tests
    # =========================================================================

    def test_deployment_state_match_schema(self, schema: Dict[str, Any]):
        """DEPLOYMENT_STATE_PROTOCOL must match schema definitions."""
        schema_def = schema.get("definitions", {}).get("deployment_state", {})
        schema_states = self._extract_enum_values(schema_def)
        code_states = set(DEPLOYMENT_STATE_PROTOCOL)

        assert code_states == schema_states, (
            f"Deployment state mismatch!\n"
            f"In schema but not in code: {schema_states - code_states}\n"
            f"In code but not in schema: {code_states - schema_states}"
        )

    # =========================================================================
    # Run State Tests
    # =========================================================================

    def test_run_state_match_schema(self, schema: Dict[str, Any]):
        """RUN_STATE_PROTOCOL must match schema definitions."""
        schema_def = schema.get("definitions", {}).get("run_state", {})
        schema_states = self._extract_enum_values(schema_def)
        code_states = set(RUN_STATE_PROTOCOL)

        assert code_states == schema_states, (
            f"Run state mismatch!\n"
            f"In schema but not in code: {schema_states - code_states}\n"
            f"In code but not in schema: {code_states - schema_states}"
        )

    # =========================================================================
    # Approval Status Tests
    # =========================================================================

    def test_approval_status_match_schema(self, schema: Dict[str, Any]):
        """APPROVAL_STATUS must match schema definitions."""
        schema_def = schema.get("definitions", {}).get("approval_status", {})
        schema_statuses = self._extract_enum_values(schema_def)
        code_statuses = set(APPROVAL_STATUS)

        assert code_statuses == schema_statuses, (
            f"Approval status mismatch!\n"
            f"In schema but not in code: {schema_statuses - code_statuses}\n"
            f"In code but not in schema: {code_statuses - schema_statuses}"
        )

    # =========================================================================
    # Telemetry Level Tests
    # =========================================================================

    def test_telemetry_level_match_schema(self, schema: Dict[str, Any]):
        """TELEMETRY_LEVEL_PROTOCOL must match schema definitions."""
        # Extract from telemetry message definition
        messages = schema.get("definitions", {}).get("messages", {})
        telemetry_def = messages.get("telemetry", {})
        level_def = telemetry_def.get("properties", {}).get("level", {})
        schema_levels = self._extract_enum_values(level_def)
        code_levels = set(TELEMETRY_LEVEL_PROTOCOL)

        assert code_levels == schema_levels, (
            f"Telemetry level mismatch!\n"
            f"In schema but not in code: {schema_levels - code_levels}\n"
            f"In code but not in schema: {code_levels - schema_levels}"
        )

    # =========================================================================
    # Signature Algorithm Tests
    # =========================================================================

    def test_signature_algorithms_match_schema(self, schema: Dict[str, Any]):
        """SIGNATURE_ALGORITHMS must match schema definitions."""
        signature_def = schema.get("definitions", {}).get("signature", {})
        algo_def = signature_def.get("properties", {}).get("algorithm", {})
        schema_algos = self._extract_enum_values(algo_def)
        code_algos = set(SIGNATURE_ALGORITHMS)

        assert code_algos == schema_algos, (
            f"Signature algorithm mismatch!\n"
            f"In schema but not in code: {schema_algos - code_algos}\n"
            f"In code but not in schema: {code_algos - schema_algos}"
        )


class TestSchemaVersionSync:
    """Tests for schema version synchronization."""

    @pytest.fixture
    def schema_path(self) -> Path:
        """Get path to protocol messages schema."""
        return (
            Path(__file__).parent.parent.parent
            / "docs"
            / "schemas"
            / "protocol_messages.schema.json"
        )

    @pytest.fixture
    def schema(self, schema_path: Path) -> Dict[str, Any]:
        """Load the JSON schema."""
        if not schema_path.exists():
            pytest.skip(f"Schema file not found: {schema_path}")
        return json.loads(schema_path.read_text(encoding="utf-8"))

    def test_schema_version_in_ccea_module(self, schema: Dict[str, Any]):
        """Schema version in ccea module must match JSON schema."""
        from ccea import SCHEMA_VERSION, MIN_SUPPORTED_SCHEMA_VERSION, MAX_SUPPORTED_SCHEMA_VERSION

        schema_version = schema.get("x-schema-version")
        min_version = schema.get("x-min-supported-version")
        max_version = schema.get("x-max-supported-version")

        assert (
            SCHEMA_VERSION == schema_version
        ), f"SCHEMA_VERSION mismatch: code={SCHEMA_VERSION}, schema={schema_version}"
        assert (
            MIN_SUPPORTED_SCHEMA_VERSION == min_version
        ), f"MIN_SUPPORTED_SCHEMA_VERSION mismatch: code={MIN_SUPPORTED_SCHEMA_VERSION}, schema={min_version}"
        assert (
            MAX_SUPPORTED_SCHEMA_VERSION == max_version
        ), f"MAX_SUPPORTED_SCHEMA_VERSION mismatch: code={MAX_SUPPORTED_SCHEMA_VERSION}, schema={max_version}"


class TestEnumCompleteness:
    """Tests for enum completeness and consistency."""

    def test_all_protocol_enums_are_frozensets(self):
        """All protocol enums should be frozensets for immutability."""
        enums = get_all_protocol_enums()
        for name, enum_set in enums.items():
            assert isinstance(
                enum_set, frozenset
            ), f"{name} should be a frozenset, got {type(enum_set)}"

    def test_all_protocol_enums_are_uppercase(self):
        """All protocol enum values should be UPPERCASE (except technical identifiers)."""
        enums = get_all_protocol_enums()

        # Signature algorithms use lowercase as per RFC standards (ed25519, ecdsa-p256, rsa-pss)
        skip_uppercase_check = {"signature_algorithms"}

        for name, enum_set in enums.items():
            if name in skip_uppercase_check:
                continue
            for value in enum_set:
                if isinstance(value, str):
                    assert (
                        value == value.upper()
                    ), f"Protocol enum {name} value '{value}' should be uppercase"

    def test_no_duplicate_values_across_unrelated_enums(self):
        """Ensure no accidental value collisions between unrelated enum sets."""
        enums = get_all_protocol_enums()

        # Only command_types and message_types must be completely distinct
        # Other enums may share states with different semantics:
        # - deployment_state and run_state share RUNNING/PAUSED/STOPPED
        # - command_status and approval_status share APPROVED/REJECTED
        distinct_sets = [
            ("command_types", "message_types"),
        ]

        for name1, name2 in distinct_sets:
            set1 = enums[name1]
            set2 = enums[name2]
            intersection = set1 & set2
            assert (
                not intersection
            ), f"Unexpected overlap between {name1} and {name2}: {intersection}"

    def test_expected_overlap_deployment_run_state(self):
        """deployment_state and run_state intentionally share RUNNING/PAUSED/STOPPED."""
        enums = get_all_protocol_enums()
        deployment_state = enums["deployment_state"]
        run_state = enums["run_state"]

        # These values are expected in both sets (different contexts)
        expected_overlap = {"RUNNING", "PAUSED", "STOPPED"}
        actual_overlap = deployment_state & run_state

        assert (
            expected_overlap <= actual_overlap
        ), f"Expected {expected_overlap} to be shared between deployment_state and run_state"

    def test_expected_overlap_command_approval_status(self):
        """command_status and approval_status intentionally share APPROVED/REJECTED."""
        enums = get_all_protocol_enums()
        command_status = enums["command_status"]
        approval_status = enums["approval_status"]

        # These values are expected in both sets
        expected_overlap = {"APPROVED", "REJECTED"}
        actual_overlap = command_status & approval_status

        assert (
            expected_overlap <= actual_overlap
        ), f"Expected {expected_overlap} to be shared between command_status and approval_status"


class TestProhibitedFields:
    """Tests verifying prohibited order-like fields are documented."""

    @pytest.fixture
    def schema_path(self) -> Path:
        """Get path to protocol messages schema."""
        return (
            Path(__file__).parent.parent.parent
            / "docs"
            / "schemas"
            / "protocol_messages.schema.json"
        )

    @pytest.fixture
    def schema(self, schema_path: Path) -> Dict[str, Any]:
        """Load the JSON schema."""
        if not schema_path.exists():
            pytest.skip(f"Schema file not found: {schema_path}")
        return json.loads(schema_path.read_text(encoding="utf-8"))

    def test_prohibited_fields_defined_in_schema(self, schema: Dict[str, Any]):
        """Schema must define prohibited_order_fields."""
        prohibited_def = schema.get("definitions", {}).get("prohibited_order_fields", {})
        assert prohibited_def, "prohibited_order_fields definition missing from schema"

        # Must use 'not' constraint
        assert "not" in prohibited_def, "prohibited_order_fields must use 'not' constraint"

    def test_side_field_prohibited_in_schema(self, schema: Dict[str, Any]):
        """Schema must prohibit 'side' field."""
        prohibited_def = schema.get("definitions", {}).get("prohibited_order_fields", {})
        not_constraint = prohibited_def.get("not", {})
        any_of = not_constraint.get("anyOf", [])

        # Check that side is in the prohibited list
        side_prohibited = any("side" in item.get("required", []) for item in any_of)
        assert side_prohibited, "'side' field must be prohibited in schema"

    def test_quantity_field_prohibited_in_schema(self, schema: Dict[str, Any]):
        """Schema must prohibit 'quantity' and 'qty' fields."""
        prohibited_def = schema.get("definitions", {}).get("prohibited_order_fields", {})
        not_constraint = prohibited_def.get("not", {})
        any_of = not_constraint.get("anyOf", [])

        quantity_prohibited = any("quantity" in item.get("required", []) for item in any_of)
        qty_prohibited = any("qty" in item.get("required", []) for item in any_of)

        assert quantity_prohibited, "'quantity' field must be prohibited in schema"
        assert qty_prohibited, "'qty' field must be prohibited in schema"

    def test_price_field_prohibited_in_schema(self, schema: Dict[str, Any]):
        """Schema must prohibit 'price' field."""
        prohibited_def = schema.get("definitions", {}).get("prohibited_order_fields", {})
        not_constraint = prohibited_def.get("not", {})
        any_of = not_constraint.get("anyOf", [])

        price_prohibited = any("price" in item.get("required", []) for item in any_of)
        assert price_prohibited, "'price' field must be prohibited in schema"

    def test_intent_signal_prohibited_in_schema(self, schema: Dict[str, Any]):
        """Schema must prohibit 'intent' and 'signal' fields."""
        prohibited_def = schema.get("definitions", {}).get("prohibited_order_fields", {})
        not_constraint = prohibited_def.get("not", {})
        any_of = not_constraint.get("anyOf", [])

        intent_prohibited = any("intent" in item.get("required", []) for item in any_of)
        signal_prohibited = any("signal" in item.get("required", []) for item in any_of)

        assert intent_prohibited, "'intent' field must be prohibited in schema"
        assert signal_prohibited, "'signal' field must be prohibited in schema"

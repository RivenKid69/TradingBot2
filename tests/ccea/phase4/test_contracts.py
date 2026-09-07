# -*- coding: utf-8 -*-
"""
Tests for CCEA Phase 4: Protocol/Contracts Consistency.

Tests for:
- WI-PROTOCOL-01: Schema version negotiation consistency
- WI-PROTOCOL-02: Command type allowlist consistency
- WI-CONTRACTS-01: Enum/state-machine drift detection

These tests ensure schema, protocol models, and DB models stay in sync.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def project_root() -> Path:
    """Get project root path."""
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd
    if (cwd.parent / "pyproject.toml").exists():
        return cwd.parent
    # Try relative from test file
    test_file = Path(__file__).resolve()
    for parent in test_file.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


@pytest.fixture
def schema_path(project_root: Path) -> Path:
    """Get protocol schema path."""
    return project_root / "docs" / "schemas" / "protocol_messages.schema.json"


@pytest.fixture
def schema(schema_path: Path) -> dict:
    """Load protocol schema."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# WI-PROTOCOL-01: Schema Version Negotiation Tests
# ============================================================================


class TestSchemaVersionNegotiation:
    """Tests for schema version negotiation mechanism."""

    def test_schema_has_version_metadata(self, schema: dict):
        """Test that schema has version metadata."""
        assert "x-schema-version" in schema
        assert "x-min-supported-version" in schema
        assert "x-max-supported-version" in schema

    def test_schema_version_format(self, schema: dict):
        """Test that schema versions are valid semver."""
        import re

        semver_pattern = re.compile(r"^\d+\.\d+\.\d+$")

        assert semver_pattern.match(schema["x-schema-version"])
        assert semver_pattern.match(schema["x-min-supported-version"])
        assert semver_pattern.match(schema["x-max-supported-version"])

    def test_ccea_init_versions_match_schema(self, schema: dict):
        """Test that ccea/__init__.py versions match schema."""
        import ccea

        assert ccea.SCHEMA_VERSION == schema["x-schema-version"]
        assert ccea.MIN_SUPPORTED_SCHEMA_VERSION == schema["x-min-supported-version"]
        assert ccea.MAX_SUPPORTED_SCHEMA_VERSION == schema["x-max-supported-version"]

    def test_schema_versioning_module_matches(self):
        """Test that schema_versioning.py matches ccea/__init__.py."""
        import ccea
        from ccea.protocol.schema_versioning import (
            CURRENT_SCHEMA_VERSION,
            MIN_SUPPORTED_VERSION,
            MAX_SUPPORTED_VERSION,
        )

        assert CURRENT_SCHEMA_VERSION == ccea.SCHEMA_VERSION
        assert MIN_SUPPORTED_VERSION == ccea.MIN_SUPPORTED_SCHEMA_VERSION
        assert MAX_SUPPORTED_VERSION == ccea.MAX_SUPPORTED_SCHEMA_VERSION

    def test_poll_commands_has_version_negotiation(self, schema: dict):
        """Test POLL_COMMANDS uses version_negotiation, not supported_schema_versions."""
        messages = schema.get("definitions", {}).get("messages", {})
        poll_commands = messages.get("poll_commands", {})
        properties = poll_commands.get("properties", {})

        # Should have version_negotiation
        assert "version_negotiation" in properties

        # Should NOT have supported_schema_versions
        assert "supported_schema_versions" not in properties

    def test_version_negotiation_model_exists(self):
        """Test VersionNegotiation model exists in protocol.py."""
        from ccea.models.protocol import VersionNegotiation

        # Test model validation
        vn = VersionNegotiation(min_supported="1.0.0", max_supported="1.0.0")
        assert vn.min_supported == "1.0.0"
        assert vn.max_supported == "1.0.0"

    def test_version_negotiation_validates_range(self):
        """Test VersionNegotiation validates max >= min."""
        from ccea.models.protocol import VersionNegotiation
        from pydantic import ValidationError

        # Valid range
        vn = VersionNegotiation(min_supported="1.0.0", max_supported="2.0.0")
        assert vn.max_supported == "2.0.0"

        # Invalid range (max < min) should fail
        with pytest.raises(ValidationError):
            VersionNegotiation(min_supported="2.0.0", max_supported="1.0.0")

    def test_poll_commands_message_uses_version_negotiation(self):
        """Test PollCommandsMessage has version_negotiation field."""
        from ccea.models.protocol import PollCommandsMessage, VersionNegotiation

        msg = PollCommandsMessage(
            agent_id="agent_test1234567890123456",
            version_negotiation=VersionNegotiation(
                min_supported="1.0.0",
                max_supported="1.0.0",
            ),
        )
        assert msg.version_negotiation is not None
        assert msg.version_negotiation.min_supported == "1.0.0"


# ============================================================================
# WI-PROTOCOL-02: Command Type Allowlist Tests
# ============================================================================


class TestCommandTypeAllowlist:
    """Tests for command type allowlist consistency."""

    def test_no_request_resume_run_in_schema(self, schema: dict):
        """Test REQUEST_RESUME_RUN is NOT in schema."""
        messages = schema.get("definitions", {}).get("messages", {})

        # Check message names
        assert "request_resume_run" not in messages

        # Check command_type consts
        for msg_name, msg_spec in messages.items():
            if isinstance(msg_spec, dict) and "allOf" in msg_spec:
                for item in msg_spec["allOf"]:
                    if isinstance(item, dict) and "properties" in item:
                        props = item["properties"]
                        if "command_type" in props:
                            cmd_spec = props["command_type"]
                            if "const" in cmd_spec:
                                assert cmd_spec["const"] != "REQUEST_RESUME_RUN"

    def test_no_request_resume_run_in_protocol_models(self):
        """Test REQUEST_RESUME_RUN is NOT in protocol models."""
        from ccea.models.protocol import CommandType

        command_values = {e.value for e in CommandType}
        assert "REQUEST_RESUME_RUN" not in command_values

    def test_no_request_resume_run_in_cloud_commands(self):
        """Test REQUEST_RESUME_RUN is NOT in cloud commands.py."""
        from packages.cloud.control_plane.commands import CommandType

        command_values = {e.value for e in CommandType}
        assert "REQUEST_RESUME_RUN" not in command_values

    def test_no_request_resume_run_in_boundary(self):
        """Test REQUEST_RESUME_RUN is NOT in boundary.py."""
        from packages.cloud.control_plane.boundary import LIFECYCLE_COMMANDS

        assert "REQUEST_RESUME_RUN" not in LIFECYCLE_COMMANDS

    def test_command_types_consistent_across_modules(self):
        """Test command types are consistent across all modules."""
        from ccea.models.protocol import CommandType as ProtocolCommandType
        from packages.cloud.control_plane.commands import CommandType as CloudCommandType
        from packages.cloud.control_plane.boundary import LIFECYCLE_COMMANDS

        protocol_commands = {e.value for e in ProtocolCommandType}
        cloud_commands = {e.value for e in CloudCommandType}

        # Protocol and cloud should have same commands
        assert protocol_commands == cloud_commands, (
            f"Protocol vs Cloud drift: "
            f"only in protocol={protocol_commands - cloud_commands}, "
            f"only in cloud={cloud_commands - protocol_commands}"
        )

        # Boundary should have same commands
        assert protocol_commands == LIFECYCLE_COMMANDS, (
            f"Protocol vs Boundary drift: "
            f"only in protocol={protocol_commands - LIFECYCLE_COMMANDS}, "
            f"only in boundary={LIFECYCLE_COMMANDS - protocol_commands}"
        )


# ============================================================================
# WI-CONTRACTS-01: Enum/State-Machine Drift Tests
# ============================================================================


class TestContractEnums:
    """Tests for enum consistency between schema, models, and DB."""

    def test_command_status_schema_matches_protocol(self, schema: dict):
        """Test command_status enum matches between schema and protocol models."""
        from ccea.models.protocol import CommandStatus

        # Extract from schema
        schema_status = set(schema.get("definitions", {}).get("command_status", {}).get("enum", []))

        # Extract from protocol models
        model_status = {e.value for e in CommandStatus}

        assert schema_status == model_status, (
            f"command_status drift: "
            f"only in schema={schema_status - model_status}, "
            f"only in model={model_status - schema_status}"
        )

    def test_approval_status_schema_matches_protocol(self, schema: dict):
        """Test approval_status enum matches between schema and protocol models."""
        from ccea.models.protocol import ApprovalStatus

        schema_status = set(
            schema.get("definitions", {}).get("approval_status", {}).get("enum", [])
        )
        model_status = {e.value for e in ApprovalStatus}

        assert schema_status == model_status

    def test_run_state_schema_matches_protocol(self, schema: dict):
        """Test run_state enum matches between schema and protocol models."""
        from ccea.models.protocol import RunState

        schema_state = set(schema.get("definitions", {}).get("run_state", {}).get("enum", []))
        model_state = {e.value for e in RunState}

        assert schema_state == model_state

    def test_deployment_state_schema_matches_protocol(self, schema: dict):
        """Test deployment_state enum matches between schema and protocol models."""
        from ccea.models.protocol import DeploymentState

        schema_state = set(
            schema.get("definitions", {}).get("deployment_state", {}).get("enum", [])
        )
        model_state = {e.value for e in DeploymentState}

        assert schema_state == model_state

    def test_signature_algorithms_match(self, schema: dict):
        """Test signature algorithms match between schema and protocol models."""
        from ccea.models.protocol import SignatureAlgorithm

        schema_algs = set(
            schema.get("definitions", {})
            .get("signature", {})
            .get("properties", {})
            .get("algorithm", {})
            .get("enum", [])
        )
        model_algs = {e.value for e in SignatureAlgorithm}

        assert schema_algs == model_algs

    def test_contracts_module_enums_match_schema(self, schema: dict):
        """Test contracts module enums match schema."""
        from ccea.contracts.enums import (
            COMMAND_STATUS_PROTOCOL,
            APPROVAL_STATUS,
            RUN_STATE_PROTOCOL,
            DEPLOYMENT_STATE_PROTOCOL,
        )

        schema_command_status = set(
            schema.get("definitions", {}).get("command_status", {}).get("enum", [])
        )
        schema_approval_status = set(
            schema.get("definitions", {}).get("approval_status", {}).get("enum", [])
        )
        schema_run_state = set(schema.get("definitions", {}).get("run_state", {}).get("enum", []))
        schema_deployment_state = set(
            schema.get("definitions", {}).get("deployment_state", {}).get("enum", [])
        )

        assert COMMAND_STATUS_PROTOCOL == schema_command_status
        assert APPROVAL_STATUS == schema_approval_status
        assert RUN_STATE_PROTOCOL == schema_run_state
        assert DEPLOYMENT_STATE_PROTOCOL == schema_deployment_state


class TestContractStatusMappings:
    """Tests for status mapping between protocol and DB representations."""

    def test_command_status_mapping_complete(self):
        """Test all protocol statuses have DB mappings."""
        from ccea.contracts.enums import (
            COMMAND_STATUS_PROTOCOL,
            CommandStatusMapping,
        )

        for status in COMMAND_STATUS_PROTOCOL:
            db_status = CommandStatusMapping.to_db(status)
            assert db_status is not None, f"No DB mapping for protocol status: {status}"

    def test_command_status_reverse_mapping_complete(self):
        """Test all DB statuses have protocol mappings."""
        from ccea.contracts.enums import (
            COMMAND_STATUS_DB,
            CommandStatusMapping,
        )

        for status in COMMAND_STATUS_DB:
            protocol_status = CommandStatusMapping.to_protocol(status)
            assert protocol_status is not None, f"No protocol mapping for DB status: {status}"

    def test_deployment_state_mapping_complete(self):
        """Test all protocol deployment states have DB mappings."""
        from ccea.contracts.enums import (
            DEPLOYMENT_STATE_PROTOCOL,
            DeploymentStateMapping,
        )

        for state in DEPLOYMENT_STATE_PROTOCOL:
            db_state = DeploymentStateMapping.to_db(state)
            assert db_state is not None, f"No DB mapping for protocol state: {state}"

    def test_run_state_mapping_complete(self):
        """Test all protocol run states have DB mappings."""
        from ccea.contracts.enums import (
            RUN_STATE_PROTOCOL,
            RunStateMapping,
        )

        for state in RUN_STATE_PROTOCOL:
            db_state = RunStateMapping.to_db(state)
            assert db_state is not None, f"No DB mapping for protocol state: {state}"


class TestContractValidation:
    """Tests for contract validation module."""

    def test_validate_contract_consistency(self, project_root: Path):
        """Test contract validation passes."""
        from ccea.contracts.validation import validate_contract_consistency

        result = validate_contract_consistency(project_root=project_root)

        # Print any errors for debugging
        if not result.passed:
            for error in result.errors:
                print(f"Error: {error}")
            for drift in result.drifts:
                print(f"Drift: {drift}")

        assert result.passed, f"Contract validation failed with {len(result.drifts)} drifts"

    def test_extract_schema_enums(self, schema_path: Path):
        """Test schema enum extraction."""
        from ccea.contracts.validation import extract_schema_enums

        enums = extract_schema_enums(schema_path)

        assert "command_status" in enums
        assert "approval_status" in enums
        assert "run_state" in enums
        assert "deployment_state" in enums
        assert len(enums["command_status"]) > 0


# ============================================================================
# Version Check Guardrail Tests
# ============================================================================


class TestVersionCheckGuardrail:
    """Tests for version_check.py guardrail."""

    def test_version_check_passes(self, project_root: Path):
        """Test version check guardrail passes."""
        from ccea.guardrails.version_check import check_version_consistency

        result = check_version_consistency(project_root=project_root)

        if not result.passed:
            for error in result.errors:
                print(f"Error: {error}")

        assert result.passed, f"Version check failed: {result.errors}"

    def test_schema_versioning_module_check(self):
        """Test schema_versioning module consistency check."""
        from ccea.guardrails.version_check import check_schema_versioning_module

        result = check_schema_versioning_module()

        assert result.passed, f"Schema versioning module check failed: {result.errors}"

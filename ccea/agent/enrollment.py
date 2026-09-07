# -*- coding: utf-8 -*-
"""
CCEA Agent Enrollment.

Handles agent enrollment with cloud:
1. Generate device keypair locally
2. Request enrollment with token
3. Store agent credentials securely

Security:
- Private key NEVER leaves the agent
- Keypair stored in local vault
- Only public key sent to cloud
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from ccea.crypto.keys import (
    KeyPair,
    KeyAlgorithm,
    generate_keypair,
    serialize_private_key,
    serialize_public_key,
    load_private_key,
    load_public_key,
)
from ccea.models.enrollment import (
    EnrollmentRequest,
    EnrollmentResponse,
    AgentCapability,
)


logger = logging.getLogger(__name__)


class EnrollmentState(str, Enum):
    """Agent enrollment state."""

    NOT_ENROLLED = "NOT_ENROLLED"
    ENROLLING = "ENROLLING"
    ENROLLED = "ENROLLED"
    ENROLLMENT_FAILED = "ENROLLMENT_FAILED"
    REVOKED = "REVOKED"


@dataclass
class AgentCredentials:
    """
    Agent credentials stored locally.

    Contains keypair and enrollment info.
    Private key NEVER leaves local storage.
    """

    agent_id: str
    workspace_id: str
    private_key_pem: str  # Encrypted on disk
    public_key_pem: str
    key_algorithm: str
    enrolled_at: datetime
    heartbeat_endpoint: str
    commands_endpoint: str
    telemetry_endpoint: str
    supported_schema_versions: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "workspace_id": self.workspace_id,
            "private_key_pem": self.private_key_pem,
            "public_key_pem": self.public_key_pem,
            "key_algorithm": self.key_algorithm,
            "enrolled_at": self.enrolled_at.isoformat(),
            "heartbeat_endpoint": self.heartbeat_endpoint,
            "commands_endpoint": self.commands_endpoint,
            "telemetry_endpoint": self.telemetry_endpoint,
            "supported_schema_versions": self.supported_schema_versions,
            "capabilities": self.capabilities,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCredentials":
        """Deserialize from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            workspace_id=data["workspace_id"],
            private_key_pem=data["private_key_pem"],
            public_key_pem=data["public_key_pem"],
            key_algorithm=data["key_algorithm"],
            enrolled_at=datetime.fromisoformat(data["enrolled_at"]),
            heartbeat_endpoint=data["heartbeat_endpoint"],
            commands_endpoint=data["commands_endpoint"],
            telemetry_endpoint=data["telemetry_endpoint"],
            supported_schema_versions=data.get("supported_schema_versions", []),
            capabilities=data.get("capabilities", []),
        )


class AgentEnrollment:
    """
    Agent enrollment manager.

    Handles:
    - Keypair generation
    - Enrollment request
    - Credential storage
    """

    def __init__(
        self,
        data_dir: Path,
        cloud_base_url: str = "http://localhost:8000",
        agent_version: str = "1.0.0",
    ):
        """
        Initialize enrollment manager.

        Args:
            data_dir: Directory for agent data
            cloud_base_url: Cloud control plane URL
            agent_version: Agent software version
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cloud_base_url = cloud_base_url.rstrip("/")
        self.agent_version = agent_version

        self._credentials_path = self.data_dir / "credentials.json"
        self._credentials: Optional[AgentCredentials] = None
        self._keypair: Optional[KeyPair] = None
        self._state = EnrollmentState.NOT_ENROLLED

        # Load existing credentials if present
        self._load_credentials()

    @property
    def state(self) -> EnrollmentState:
        """Get enrollment state."""
        return self._state

    @property
    def credentials(self) -> Optional[AgentCredentials]:
        """Get credentials if enrolled."""
        return self._credentials

    @property
    def agent_id(self) -> Optional[str]:
        """Get agent ID if enrolled."""
        return self._credentials.agent_id if self._credentials else None

    def is_enrolled(self) -> bool:
        """Check if agent is enrolled."""
        return self._state == EnrollmentState.ENROLLED and self._credentials is not None

    def enroll(
        self,
        token: str,
        capabilities: Optional[List[AgentCapability]] = None,
        hostname: Optional[str] = None,
    ) -> EnrollmentResponse:
        """
        Enroll agent with cloud.

        Args:
            token: Enrollment token from cloud
            capabilities: Requested capabilities
            hostname: Optional hostname

        Returns:
            EnrollmentResponse from cloud

        Raises:
            RuntimeError: If already enrolled
        """
        if self.is_enrolled():
            raise RuntimeError(f"Agent already enrolled: {self._credentials.agent_id}")

        self._state = EnrollmentState.ENROLLING

        try:
            # Generate keypair
            logger.info("Generating device keypair...")
            keypair = generate_keypair(KeyAlgorithm.ED25519)

            # Create enrollment request
            request = EnrollmentRequest(
                token=token,
                public_key=keypair.get_public_key_pem(),
                public_key_algorithm=keypair.algorithm.value,
                agent_version=self.agent_version,
                hostname=hostname,
                capabilities=capabilities or [AgentCapability.PAPER_TRADING],
            )

            # Send enrollment request
            logger.info("Sending enrollment request...")
            response = self._send_enrollment_request(request)

            if not response.success:
                self._state = EnrollmentState.ENROLLMENT_FAILED
                logger.error(
                    "Enrollment failed",
                    extra={
                        "error_code": response.error_code,
                        "error_message": response.error_message,
                    },
                )
                return response

            # Store credentials
            self._credentials = AgentCredentials(
                agent_id=response.agent_id,
                workspace_id=response.workspace_id,
                private_key_pem=keypair.get_private_key_pem(),
                public_key_pem=keypair.get_public_key_pem(),
                key_algorithm=keypair.algorithm.value,
                enrolled_at=response.enrolled_at or datetime.utcnow(),
                heartbeat_endpoint=response.heartbeat_endpoint,
                commands_endpoint=response.commands_endpoint,
                telemetry_endpoint=response.telemetry_endpoint,
                supported_schema_versions=response.supported_schema_versions,
                capabilities=[c.value for c in (capabilities or [])],
            )

            self._keypair = keypair
            self._save_credentials()
            self._state = EnrollmentState.ENROLLED

            logger.info(
                "Enrollment successful",
                extra={
                    "agent_id": response.agent_id,
                    "workspace_id": response.workspace_id,
                },
            )

            return response

        except Exception as e:
            self._state = EnrollmentState.ENROLLMENT_FAILED
            logger.exception("Enrollment failed with exception")
            return EnrollmentResponse.error_response(
                "ENROLLMENT_ERROR",
                str(e),
            )

    def get_private_key(self) -> Optional[Any]:
        """
        Get private key for signing.

        Returns:
            Private key object or None if not enrolled
        """
        if not self._credentials:
            return None

        if self._keypair is None:
            self._keypair = KeyPair(
                algorithm=KeyAlgorithm(self._credentials.key_algorithm),
                private_key=load_private_key(self._credentials.private_key_pem),
                public_key=load_public_key(self._credentials.public_key_pem),
            )

        return self._keypair.private_key

    def get_public_key(self) -> Optional[Any]:
        """Get public key."""
        if not self._credentials:
            return None

        if self._keypair is None:
            self._keypair = KeyPair(
                algorithm=KeyAlgorithm(self._credentials.key_algorithm),
                private_key=load_private_key(self._credentials.private_key_pem),
                public_key=load_public_key(self._credentials.public_key_pem),
            )

        return self._keypair.public_key

    def _send_enrollment_request(
        self,
        request: EnrollmentRequest,
    ) -> EnrollmentResponse:
        """Send enrollment request to cloud."""
        url = f"{self.cloud_base_url}/api/v1/enroll"

        try:
            resp = requests.post(
                url,
                json=request.model_dump(mode="json"),
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                return EnrollmentResponse(**data)
            else:
                return EnrollmentResponse.error_response(
                    f"HTTP_{resp.status_code}",
                    resp.text,
                )
        except requests.RequestException as e:
            return EnrollmentResponse.error_response(
                "NETWORK_ERROR",
                str(e),
            )

    def _save_credentials(self) -> None:
        """Save credentials to disk."""
        if self._credentials:
            with open(self._credentials_path, "w") as f:
                json.dump(self._credentials.to_dict(), f, indent=2)
            # Secure permissions
            self._credentials_path.chmod(0o600)

    def _load_credentials(self) -> None:
        """Load credentials from disk."""
        if self._credentials_path.exists():
            try:
                with open(self._credentials_path, "r") as f:
                    data = json.load(f)
                self._credentials = AgentCredentials.from_dict(data)
                self._state = EnrollmentState.ENROLLED
                logger.info(f"Loaded credentials for agent: {self._credentials.agent_id}")
            except Exception as e:
                logger.error(f"Failed to load credentials: {e}")
                self._state = EnrollmentState.NOT_ENROLLED

    def unenroll(self) -> None:
        """
        Unenroll agent (local only).

        Note: This only clears local credentials.
        Cloud-side revocation should be done separately.
        """
        if self._credentials_path.exists():
            self._credentials_path.unlink()

        self._credentials = None
        self._keypair = None
        self._state = EnrollmentState.NOT_ENROLLED

        logger.info("Agent unenrolled (local credentials cleared)")

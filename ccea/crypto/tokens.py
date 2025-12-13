# -*- coding: utf-8 -*-
"""
CCEA Secure Token Generation.

Provides cryptographically secure token generation for:
- Enrollment tokens
- Idempotency keys
- Agent IDs
- Command IDs

Security:
- Uses secrets module for cryptographic randomness
- Tokens follow defined patterns from JSON schema
"""

from __future__ import annotations

import secrets
import string
from datetime import datetime


# Character sets
ALPHANUMERIC = string.ascii_letters + string.digits
ALPHANUMERIC_SAFE = string.ascii_letters + string.digits + "_-"


def generate_enrollment_token(length: int = 48) -> str:
    """
    Generate a secure enrollment token.

    Args:
        length: Token length (default: 48)

    Returns:
        Token string matching pattern: enroll_[a-zA-Z0-9]{32,64}
    """
    if length < 32 or length > 64:
        raise ValueError("Token length must be between 32 and 64")

    random_part = "".join(secrets.choice(ALPHANUMERIC) for _ in range(length))
    return f"enroll_{random_part}"


def generate_idempotency_key(length: int = 32) -> str:
    """
    Generate a unique idempotency key.

    Args:
        length: Key length (default: 32)

    Returns:
        Key string matching pattern: [a-zA-Z0-9_-]{16,64}
    """
    if length < 16 or length > 64:
        raise ValueError("Key length must be between 16 and 64")

    return "".join(secrets.choice(ALPHANUMERIC_SAFE) for _ in range(length))


def generate_agent_id(length: int = 24) -> str:
    """
    Generate a unique agent ID.

    Args:
        length: ID length (default: 24)

    Returns:
        ID string matching pattern: agent_[a-zA-Z0-9]{16,32}
    """
    if length < 16 or length > 32:
        raise ValueError("Agent ID length must be between 16 and 32")

    random_part = "".join(secrets.choice(ALPHANUMERIC) for _ in range(length))
    return f"agent_{random_part}"


def generate_command_id() -> str:
    """
    Generate a unique command ID.

    Returns:
        Command ID string with timestamp prefix
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    random_part = "".join(secrets.choice(ALPHANUMERIC) for _ in range(16))
    return f"cmd_{timestamp}_{random_part}"


def generate_deployment_id() -> str:
    """
    Generate a unique deployment ID.

    Returns:
        Deployment ID string
    """
    random_part = "".join(secrets.choice(ALPHANUMERIC) for _ in range(20))
    return f"deploy_{random_part}"


def generate_workspace_id() -> str:
    """
    Generate a unique workspace ID.

    Returns:
        Workspace ID string
    """
    random_part = "".join(secrets.choice(ALPHANUMERIC) for _ in range(16))
    return f"ws_{random_part}"


def generate_blob_id(digest: str) -> str:
    """
    Generate a blob ID from digest.

    Args:
        digest: SHA256 digest string

    Returns:
        Blob ID (truncated digest)
    """
    if digest.startswith("sha256:"):
        hash_part = digest[7:19]  # First 12 hex chars
    else:
        hash_part = digest[:12]
    return f"blob_{hash_part}"


def is_valid_enrollment_token(token: str) -> bool:
    """Check if string is a valid enrollment token."""
    if not token.startswith("enroll_"):
        return False
    random_part = token[7:]
    if len(random_part) < 32 or len(random_part) > 64:
        return False
    return all(c in ALPHANUMERIC for c in random_part)


def is_valid_agent_id(agent_id: str) -> bool:
    """Check if string is a valid agent ID."""
    if not agent_id.startswith("agent_"):
        return False
    random_part = agent_id[6:]
    if len(random_part) < 16 or len(random_part) > 32:
        return False
    return all(c in ALPHANUMERIC for c in random_part)


def is_valid_idempotency_key(key: str) -> bool:
    """Check if string is a valid idempotency key."""
    if len(key) < 16 or len(key) > 64:
        return False
    return all(c in ALPHANUMERIC_SAFE for c in key)

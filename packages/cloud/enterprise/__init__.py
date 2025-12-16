# -*- coding: utf-8 -*-
"""
CCEA Cloud Enterprise Package.

Phase 9/10 Implementation: Enterprise/on-prem pack + signed agent updates.

This package provides enterprise features for CCEA Cloud:
- Cryptographic Operations: Ed25519 signing and verification
- Registry Mirror: Local artifact mirroring for air-gapped deployments
- Evidence Pack Exporter: Audit trail and compliance documentation
- Agent Update Manager: Signed updates with staged rollout
- Version Pinning: Enterprise version control and change windows
- TUF Repository: Secure update framework protection

Design Doc Reference:
    - Phase 9: Enterprise/on-prem pack + signed agent updates
    - Phase 10: Enterprise signing and key management
    - Design Doc 16.1: Evidence pack exporter
    - Design Doc 15.2/5.2: Agent updates
"""

# Cryptographic operations (Phase 10)
try:
    from packages.cloud.enterprise.crypto import (
        Ed25519Signer,
        SigningKey,
        Signature,
        CryptoError,
        SignatureError,
        VerificationError,
        create_signer,
        sign_data,
        verify_data,
        CRYPTO_AVAILABLE,
    )
except ImportError:
    Ed25519Signer = None
    SigningKey = None
    Signature = None
    CryptoError = Exception
    SignatureError = Exception
    VerificationError = Exception
    create_signer = None
    sign_data = None
    verify_data = None
    CRYPTO_AVAILABLE = False

from packages.cloud.enterprise.registry_mirror import (
    RegistryMirror,
    RegistryMirrorConfig,
    MirrorStatus,
    MirrorSyncResult,
)
from packages.cloud.enterprise.evidence_pack import (
    EvidencePackExporter,
    EvidencePackConfig,
    EvidencePack,
    EvidenceType,
)
from packages.cloud.enterprise.agent_updates import (
    AgentUpdateManager,
    AgentUpdateConfig,
    UpdateChannel,
    UpdateState,
    AgentUpdate,
)
from packages.cloud.enterprise.staged_rollout import (
    StagedRolloutManager,
    RolloutConfig,
    RolloutStage,
    RolloutState,
    RolloutProgress,
)
from packages.cloud.enterprise.version_pinning import (
    VersionPinningManager,
    VersionPinConfig,
    ChangeWindow,
    VersionConstraint,
)
from packages.cloud.enterprise.tuf_repository import (
    TUFRepository,
    TUFConfig,
    TUFRole,
    TUFMetadata,
    SignedMetadata,
)
from packages.cloud.enterprise.multi_region import (
    MultiRegionReplicator,
    Region,
    RegionStatus,
    ReplicationConfig,
    ReplicationMode,
    ReplicationItem,
    ReplicationStats,
    ConflictResolution,
    create_default_regions,
)

__all__ = [
    # Cryptographic Operations (Phase 10)
    "Ed25519Signer",
    "SigningKey",
    "Signature",
    "CryptoError",
    "SignatureError",
    "VerificationError",
    "create_signer",
    "sign_data",
    "verify_data",
    "CRYPTO_AVAILABLE",
    # Registry Mirror
    "RegistryMirror",
    "RegistryMirrorConfig",
    "MirrorStatus",
    "MirrorSyncResult",
    # Evidence Pack
    "EvidencePackExporter",
    "EvidencePackConfig",
    "EvidencePack",
    "EvidenceType",
    # Agent Updates
    "AgentUpdateManager",
    "AgentUpdateConfig",
    "UpdateChannel",
    "UpdateState",
    "AgentUpdate",
    # Staged Rollout
    "StagedRolloutManager",
    "RolloutConfig",
    "RolloutStage",
    "RolloutState",
    "RolloutProgress",
    # Version Pinning
    "VersionPinningManager",
    "VersionPinConfig",
    "ChangeWindow",
    "VersionConstraint",
    # TUF Repository
    "TUFRepository",
    "TUFConfig",
    "TUFRole",
    "TUFMetadata",
    "SignedMetadata",
    # Multi-Region Replication
    "MultiRegionReplicator",
    "Region",
    "RegionStatus",
    "ReplicationConfig",
    "ReplicationMode",
    "ReplicationItem",
    "ReplicationStats",
    "ConflictResolution",
    "create_default_regions",
]

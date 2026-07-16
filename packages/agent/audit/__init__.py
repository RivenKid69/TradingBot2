# -*- coding: utf-8 -*-
"""Agent-zone tamper-evident audit primitives (keyed hash chains)."""

from packages.agent.audit.hash_chain import (
    HashChain,
    ChainRecord,
    chain_hash,
    verify_chain,
    GENESIS_HASH,
)

__all__ = ["HashChain", "ChainRecord", "chain_hash", "verify_chain", "GENESIS_HASH"]

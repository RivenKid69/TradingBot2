# -*- coding: utf-8 -*-
"""Ed25519 model-signature enforcement for the agent artifact→run path (P0-E).

AGENT ZONE. Closes the residual half of §4.7 / P0-E: the RL *inference* loader
(``service_rl_inference``) already gates model checkpoints through
``services.model_signature_gate``, but the **daemon's own** artifact-activation
path (``RunController.initialize`` → ``_init_live_runner``) only verified the
SHA-256 *digest* (integrity), never the Ed25519 *signature* (authenticity). An
SB3 ``.zip`` checkpoint is pickle — deserializing an unsigned/tampered artifact
is arbitrary code execution in the process that holds broker keys. The CCEA
design doc requires "Artifact Signature Verification: REQUIRED".

This module locates model checkpoints inside an extracted artifact and runs each
through the SAME gate the RL loader uses, so **every** live model-load path in
the daemon is covered. Fail-closed for LIVE (enforce): an unsigned/unregistered/
tampered checkpoint raises ``ModelSignatureError`` and the run never starts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Model checkpoint formats that carry executable/pickled payloads. SB3 ships
# ``.zip``; raw torch/pickle/onnx are covered for defence-in-depth.
MODEL_EXTENSIONS = (".zip", ".pt", ".pth", ".pkl", ".ckpt", ".onnx", ".safetensors")


def find_model_files(root: Any) -> List[Path]:
    """Return model-checkpoint files under ``root`` (file or directory)."""
    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in MODEL_EXTENSIONS else []
    return [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS
    ]


def verify_artifact_models(
    extracted_path: Any,
    *,
    live: bool,
    policy: Optional[str] = None,
    registry: Any = None,
    context: str = "agent-run",
) -> List[Any]:
    """Verify every model checkpoint in an extracted artifact.

    Delegates each file to ``services.model_signature_gate.verify_model_artifact``
    — the exact gate the RL inference loader uses. In ``enforce`` policy (the
    default for ``live=True``) the first failure raises ``ModelSignatureError``
    BEFORE any pickle is read, so the caller's run initialization fails closed.

    Returns the list of per-file ``SignatureVerdict``. An empty list means no
    model checkpoint was found (a code-only strategy) — the manifest digest and
    sandbox controls still apply; this is logged, not silently ignored.
    """
    from services.model_signature_gate import resolve_policy, verify_model_artifact

    eff_policy = resolve_policy(policy, live=live)
    files = find_model_files(extracted_path)
    if not files:
        logger.info(
            "model-gate[%s]: no model checkpoint under %s (code-only strategy?) — "
            "manifest digest + sandbox controls apply, signature gate N/A",
            context, extracted_path,
        )
        return []

    verdicts = []
    for f in files:
        # Raises ModelSignatureError in enforce on any failure (fail-closed).
        verdicts.append(
            verify_model_artifact(
                str(f), policy=eff_policy, live=live, registry=registry, context=context
            )
        )
    logger.info(
        "model-gate[%s]: %d checkpoint(s) verified (policy=%s, live=%s)",
        context, len(verdicts), eff_policy, live,
    )
    return verdicts


__all__ = ["MODEL_EXTENSIONS", "find_model_files", "verify_artifact_models"]

# -*- coding: utf-8 -*-
"""
convert_legacy_models.py
---------------------------------------------------------------
Converts legacy PyTorch models to secure format (state_dict only).

Security Context:
- Legacy models saved with torch.save(model) can contain arbitrary Python objects
- These objects are deserialized via pickle, enabling arbitrary code execution
- Secure format uses weights_only=True compatible state_dict
- See: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models

Usage:
    python tools/convert_legacy_models.py [--models-dir models/] [--backup] [--dry-run]

This script:
1. Scans for .pt/.pth files in models directory
2. Attempts secure loading (weights_only=True)
3. If secure loading fails, loads unsafely and re-saves as state_dict
4. Verifies converted model can be loaded securely
"""
import argparse
import hashlib
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_model_security(path: Path) -> Tuple[bool, Optional[str]]:
    """Check if model can be loaded securely.

    Returns:
        (is_secure, error_message)
    """
    import torch

    try:
        torch.load(path, map_location="cpu", weights_only=True)
        return True, None
    except Exception as e:
        return False, str(e)


def convert_model(path: Path, backup: bool = True, dry_run: bool = False) -> Tuple[bool, str]:
    """Convert legacy model to secure format.

    Returns:
        (success, message)
    """
    import torch

    # Check if already secure
    is_secure, error = check_model_security(path)
    if is_secure:
        return True, f"Already secure: {path.name}"

    logger.warning(f"Model {path.name} requires conversion: {error}")

    if dry_run:
        return True, f"[DRY-RUN] Would convert: {path.name}"

    # Create backup if requested
    if backup:
        backup_dir = path.parent / "legacy_backup"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{path.stem}_{timestamp}{path.suffix}"
        shutil.copy2(path, backup_path)
        logger.info(f"Created backup: {backup_path}")

    # Load model unsafely (controlled conversion context)
    # SECURITY: This unsafe loading is ONLY used for one-time conversion of known legacy
    # models. This risk is documented and controlled per docs/security/THREAT_MODEL_MODEL_LOADING.md
    # Controls: C3 (conversion utility), backup creation, immediate re-save as secure format
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"Failed to load model: {e}"

    # Extract state dict
    if hasattr(model, "state_dict"):
        state_dict = model.state_dict()
        model_class_name = model.__class__.__name__
    elif isinstance(model, dict):
        # Already a state dict or similar
        state_dict = model
        model_class_name = "dict"
    else:
        return False, f"Cannot extract state_dict from {type(model)}"

    # Save securely
    original_hash = compute_file_hash(path)
    try:
        # Save only the state dict
        torch.save(state_dict, path)
    except Exception as e:
        return False, f"Failed to save converted model: {e}"

    new_hash = compute_file_hash(path)

    # Verify it can now be loaded securely
    is_secure, error = check_model_security(path)
    if not is_secure:
        return False, f"Conversion failed - still insecure: {error}"

    return True, (
        f"Converted {path.name} ({model_class_name}): "
        f"hash {original_hash[:12]}... -> {new_hash[:12]}..."
    )


def main():
    parser = argparse.ArgumentParser(description="Convert legacy PyTorch models to secure format")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing models (default: models/)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backups before conversion (default: True)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Skip creating backups",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check models without converting",
    )
    args = parser.parse_args()

    if not args.models_dir.exists():
        logger.error(f"Models directory not found: {args.models_dir}")
        sys.exit(1)

    # Find all PyTorch model files
    pt_files = list(args.models_dir.glob("*.pt")) + list(args.models_dir.glob("*.pth"))

    if not pt_files:
        logger.info(f"No .pt/.pth files found in {args.models_dir}")
        sys.exit(0)

    logger.info(f"Found {len(pt_files)} model file(s) in {args.models_dir}")

    results = {"secure": 0, "converted": 0, "failed": 0}

    for path in sorted(pt_files):
        is_secure, _ = check_model_security(path)

        if is_secure:
            logger.info(f"[SECURE] {path.name}")
            results["secure"] += 1
            continue

        success, message = convert_model(path, backup=args.backup, dry_run=args.dry_run)

        if success:
            if args.dry_run or "Already secure" in message:
                logger.info(f"[OK] {message}")
                results["secure"] += 1
            else:
                logger.info(f"[CONVERTED] {message}")
                results["converted"] += 1
        else:
            logger.error(f"[FAILED] {path.name}: {message}")
            results["failed"] += 1

    # Summary
    logger.info("=" * 60)
    logger.info(
        f"Summary: {results['secure']} secure, "
        f"{results['converted']} converted, "
        f"{results['failed']} failed"
    )

    if results["failed"] > 0:
        logger.error("Some models failed conversion - manual intervention required")
        sys.exit(1)

    if results["converted"] > 0 and not args.dry_run:
        logger.info("Conversion complete. Verify models work correctly before deploying.")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error("PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    main()

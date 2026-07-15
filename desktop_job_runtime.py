"""Frozen-desktop Python job dispatcher.

PyInstaller sets ``sys.executable`` to the sidecar executable.  Launching
``[sys.executable, "make_features.py", ...]`` therefore starts the HTTP server
entry point again instead of the requested Python program.  This module keeps
the normal command shape in source mode and translates it to an explicit,
allow-listed worker mode in a frozen build.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence


WORKER_MODULES = frozenset(
    {
        "apply_no_trade_mask",
        "apply_calibrator",
        "build_training_table",
        "compare_slippage_curve",
        "cot_data_loader",
        "diag_val_split",
        "drift",
        "ingest_orchestrator",
        "make_costaware_targets",
        "make_features",
        "make_walkforward_splits",
        "run_conformal_calibration",
        "script_calibrate_slippage",
        "script_calibrate_tcost",
        "script_futures_live",
        "script_live",
        "service_train",
        "train_model_multi_patch",
        "train_calibrator",
        "training_pbt_adversarial_integration",
        "tune_threshold",
        "research.advanced_features",
        "research.cv_overfitting",
        "research.dataset_versioning",
        "research.eda_profiler",
        "research.feature_analytics",
        "research.target_diagnostics",
        "scripts.build_hourly_seasonality",
        "scripts.download_economic_calendar",
        "scripts.download_edgar_fundamentals",
        "scripts.download_forex_data",
        "scripts.download_interest_rates",
        "scripts.download_options_data",
        "scripts.download_stock_data",
        "scripts.download_swap_rates",
        "scripts.fetch_binance_filters",
        "scripts.optimize_parameters",
        "scripts.refresh_universe",
        "services.corporate_actions",
        "services.sector_momentum",
        "tools.check_feature_parity",
        "tools.xs_crypto_real_sweep",
        "tools.xs_equity_real_report",
    }
)

_WORKER_SCRIPT = "--riven-worker-script"
_WORKER_MODULE = "--riven-worker-module"
_WORKER_CODE = "--riven-worker-code"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def code_root() -> str:
    """Repository root in source mode; sys._MEIPASS parent in a frozen build.

    Source-mode subprocesses must resolve *code* from here even when the
    server's working directory is a separate RIVEN_DATA_DIR (audit L2-016).
    Data paths stay relative to the working directory (the data root).
    """
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def worker_environment(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for worker subprocesses: PYTHONPATH always includes the
    code root so ``import app`` / project modules resolve regardless of CWD."""
    env = dict(os.environ if base is None else base)
    root = code_root()
    parts = [root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != root]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Worker logs are written/parsed as UTF-8; without this a legacy console
    # code page (cp1251) crashes any worker that prints non-ANSI characters.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _script_to_module(script: str) -> str:
    normalized = str(script).replace("\\", "/")
    path = PurePosixPath(normalized)
    if ".." in path.parts or path.suffix.lower() != ".py":
        raise ValueError(f"unsafe worker script path: {script!r}")
    # app.py keeps a few repository-root entrypoints as absolute paths.  In a
    # frozen bundle those modules live in the PYZ, so resolve by basename only;
    # the allow-list check below still prevents arbitrary module execution.
    # NOTE: PurePosixPath("C:/...").is_absolute() is False — a Windows drive
    # path must be detected via os.path.isabs on the original string too.
    if path.is_absolute() or os.path.isabs(str(script)):
        return path.stem
    return ".".join(path.with_suffix("").parts)


def prepare_python_command(command: Sequence[str]) -> List[str]:
    """Translate a Python subprocess argv for the PyInstaller sidecar.

    Source/development commands are returned unchanged.  Only commands whose
    executable is the current Python/sidecar executable are translated.
    """

    cmd = [str(part) for part in command]
    if not is_frozen():
        # Source mode with a separate data directory: the CWD holds data, the
        # repository holds code. Absolutize the worker script path against the
        # code root when it does not exist relative to the CWD (audit L2-016).
        if (
            len(cmd) >= 2
            and cmd[0] == sys.executable
            and cmd[1].endswith(".py")
            and not os.path.isabs(cmd[1])
            and not os.path.exists(cmd[1])
        ):
            candidate = os.path.join(code_root(), cmd[1])
            if os.path.exists(candidate):
                cmd[1] = candidate
        return cmd
    if len(cmd) < 2 or cmd[0] != sys.executable:
        return cmd

    arg0 = cmd[1]
    if arg0 in {_WORKER_SCRIPT, _WORKER_MODULE, _WORKER_CODE}:
        return cmd
    if arg0 == "-c" and len(cmd) >= 3:
        return [sys.executable, _WORKER_CODE, cmd[2], *cmd[3:]]
    if arg0 == "-m" and len(cmd) >= 3:
        module = cmd[2]
        if module not in WORKER_MODULES:
            raise ValueError(f"worker module is not packaged/allowed: {module}")
        return [sys.executable, _WORKER_MODULE, module, *cmd[3:]]

    module = _script_to_module(arg0)
    if module not in WORKER_MODULES:
        raise ValueError(f"worker script is not packaged/allowed: {arg0}")
    return [sys.executable, _WORKER_SCRIPT, arg0, *cmd[2:]]


def dispatch_worker(argv: Optional[Sequence[str]] = None) -> Optional[int]:
    """Execute a frozen worker invocation, or return ``None`` for server mode."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {_WORKER_SCRIPT, _WORKER_MODULE, _WORKER_CODE}:
        return None
    if len(args) < 2:
        raise SystemExit("worker target is required")

    mode, target, *job_args = args
    if mode == _WORKER_CODE:
        sys.argv = ["-c", *job_args]
        namespace = {"__name__": "__main__", "__file__": "<riven-worker>"}
        exec(compile(target, "<riven-worker>", "exec"), namespace, namespace)
        return 0

    module = target if mode == _WORKER_MODULE else _script_to_module(target)
    if module not in WORKER_MODULES:
        raise SystemExit(f"worker target is not packaged/allowed: {module}")
    sys.argv = [target, *job_args]
    runpy.run_module(module, run_name="__main__", alter_sys=False)
    return 0


__all__ = [
    "WORKER_MODULES",
    "code_root",
    "dispatch_worker",
    "is_frozen",
    "prepare_python_command",
    "worker_environment",
]

"""Runtime security guards for live trading.

- StrategyFSGuard: blocks write operations from strategy code unless paths are allowlisted.
- AdapterNetworkGuard: restricts outbound network hosts for adapters.
"""

from __future__ import annotations

import builtins
import inspect
import ipaddress
import logging
import socket
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

logger = logging.getLogger(__name__)


def _resolve_paths(paths: Iterable[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    for item in paths:
        try:
            candidate = Path(item)
        except TypeError:
            continue
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved.append(candidate.resolve())
    return resolved


class StrategyFSGuard:
    """Guard that enforces read-only filesystem access for strategy code."""

    def __init__(
        self,
        mode: str = "read_only",
        allow_write_paths: Sequence[str | Path] | None = None,
        strategy_frame_predicate: Callable[[inspect.FrameInfo], bool] | None = None,
    ) -> None:
        self.mode = (mode or "").lower()
        self.allow_write_paths = _resolve_paths(allow_write_paths or [])
        self._predicate = strategy_frame_predicate or self._default_predicate
        self._installed = False
        self._orig_open = builtins.open
        self._orig_path_open = Path.open

    @property
    def installed(self) -> bool:
        return self._installed

    def _default_predicate(self, frame_info: inspect.FrameInfo) -> bool:
        module = frame_info.frame.f_globals.get("__name__", "")
        filename = (frame_info.filename or "").replace("\\", "/")
        return module.startswith("strategies") or "/strategies/" in filename

    def _is_strategy_frame(self) -> bool:
        for frame_info in inspect.stack(context=0)[:10]:
            try:
                if self._predicate(frame_info):
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _is_write_mode(mode: str | None) -> bool:
        if not mode:
            return False
        lowered = str(mode)
        return any(token in lowered for token in ("w", "a", "+", "x"))

    def _is_allowed_path(self, path: Path) -> bool:
        for allowed in self.allow_write_paths:
            try:
                if path.is_relative_to(allowed) or path == allowed:
                    return True
            except ValueError:
                continue
        return False

    def _patched_open(self, file, mode="r", *args, **kwargs):  # type: ignore[override]
        if self.mode == "read_only" and self._is_write_mode(mode) and self._is_strategy_frame():
            try:
                target = Path(file).resolve()
            except Exception:
                return self._orig_open(file, mode, *args, **kwargs)
            if not self._is_allowed_path(target):
                raise PermissionError(f"Strategy write blocked by read-only guard: {target}")
        return self._orig_open(file, mode, *args, **kwargs)

    def _patched_path_open(self, path, mode="r", *args, **kwargs):  # type: ignore[override]
        return self._patched_open(path, mode, *args, **kwargs)

    def install(self) -> None:
        if self._installed or self.mode != "read_only":
            return
        builtins.open = self._patched_open  # type: ignore[assignment]
        Path.open = self._patched_path_open  # type: ignore[assignment]
        self._installed = True
        logger.info(
            "Strategy filesystem guard enabled (allow_write_paths=%s)",
            [str(p) for p in self.allow_write_paths],
        )

    def uninstall(self) -> None:
        if not self._installed:
            return
        builtins.open = self._orig_open  # type: ignore[assignment]
        Path.open = self._orig_path_open  # type: ignore[assignment]
        self._installed = False
        logger.info("Strategy filesystem guard disabled")


class AdapterNetworkGuard:
    """Guard outbound network connections with a host allowlist."""

    def __init__(
        self,
        allowed_hosts: Sequence[str] | None = None,
        *,
        allowed_cidrs: Sequence[str] | None = None,
        allow_subdomains: bool = True,
    ) -> None:
        self.allowed_hosts = {str(h).lower() for h in (allowed_hosts or []) if h}
        self.allowed_cidrs = []
        for cidr in allowed_cidrs or []:
            try:
                self.allowed_cidrs.append(ipaddress.ip_network(str(cidr), strict=False))
            except ValueError:
                continue
        self.allow_subdomains = allow_subdomains
        self._orig_create_connection: Callable | None = None
        self._installed = False

    @property
    def installed(self) -> bool:
        return self._installed

    def validate_host(self, host: str) -> None:
        if not host:
            return
        normalized = str(host).split(":")[0].strip().lower()
        if normalized in {"localhost", "127.0.0.1", "::1"}:
            return
        try:
            ip_addr = ipaddress.ip_address(normalized)
        except ValueError:
            ip_addr = None
        if ip_addr is not None:
            for cidr in self.allowed_cidrs:
                if ip_addr in cidr:
                    return
            raise PermissionError(f"Outbound IP not allowlisted: {normalized}")

        if normalized in self.allowed_hosts:
            return
        if self.allow_subdomains:
            for host_suffix in self.allowed_hosts:
                if normalized.endswith(f".{host_suffix}"):
                    return
        raise PermissionError(f"Outbound host not allowlisted: {normalized}")

    def _guarded_create_connection(self, address, *args, **kwargs):  # type: ignore[override]
        host = address[0] if isinstance(address, (tuple, list)) else address
        self.validate_host(host)
        if self._orig_create_connection is None:
            raise RuntimeError("Adapter network guard is not initialized")
        return self._orig_create_connection(address, *args, **kwargs)

    def install(self) -> None:
        if self._installed or not (self.allowed_hosts or self.allowed_cidrs):
            return
        self._orig_create_connection = socket.create_connection
        socket.create_connection = self._guarded_create_connection  # type: ignore[assignment]
        self._installed = True
        logger.info(
            "Adapter network guard enabled (hosts=%s, cidrs=%s)",
            sorted(self.allowed_hosts),
            [str(c) for c in self.allowed_cidrs],
        )

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._orig_create_connection is not None:
            socket.create_connection = self._orig_create_connection  # type: ignore[assignment]
        self._installed = False
        logger.info("Adapter network guard disabled")


def configure_runtime_security(
    *,
    mode: str,
    config: Mapping[str, object] | None,
) -> MutableMapping[str, object]:
    """Install runtime security guards when running live."""

    mode_norm = (mode or "").lower()
    if mode_norm != "live" or not isinstance(config, Mapping):
        return {}

    status: MutableMapping[str, object] = {}

    fs_cfg = config.get("strategy_fs") if isinstance(config, Mapping) else None
    if isinstance(fs_cfg, Mapping):
        guard = StrategyFSGuard(
            mode=str(fs_cfg.get("mode", "read_only")),
            allow_write_paths=fs_cfg.get("allow_write_paths") or [],
        )
        guard.install()
        status["strategy_fs"] = {"active": guard.installed, "mode": guard.mode}

    net_cfg = config.get("network") if isinstance(config, Mapping) else None
    if isinstance(net_cfg, Mapping) and net_cfg.get("enforce"):
        guard = AdapterNetworkGuard(
            allowed_hosts=net_cfg.get("allowed_hosts") or [],
            allowed_cidrs=net_cfg.get("allowed_cidrs") or [],
            allow_subdomains=bool(net_cfg.get("allow_subdomains", True)),
        )
        guard.install()
        status["network"] = {"active": guard.installed}

    return status

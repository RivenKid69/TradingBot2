# -*- coding: utf-8 -*-
"""Gas Guard: реальный on-chain gas-oracle + порог блокировки (закрытие «NOT
IMPLEMENTED»).

Раньше слайдер Gas Guard был disabled с баннером «в Lite нет DEX execution-
контура, порог не сохраняется и не применяется». Теперь это настоящий контрол:

1. **Реальный gas oracle** — читает текущую цену газа с публичного JSON-RPC
   узла (`eth_gasPrice`, EIP-1559-совместимо) для выбранной EVM-сети. Ключи не
   нужны (публичные узлы publicnode.com). Backend делает вызов — CSP/offline
   браузера не мешает.
2. **Долговечный порог** — `GasGuardConfig` (enabled + threshold_gwei per chain)
   сохраняется в `state/web3_gas_guard.json`.
3. **Применение** — `evaluate(chain)` возвращает вердикт ARMED/BREACHED против
   живого газа; `preflight(chain)` — это pre-trade gate: любая on-chain
   транзакция, которую инициирует приложение (или Agent-side execution),
   ОБЯЗАНА пройти его. Blocked=True → транзакцию слать нельзя.

Честная граница: в Lite нет авто-DEX-исполнения, поэтому guard применяется как
**pre-flight перед ручной/агентской транзакцией**, а не глушит несуществующий
поток ордеров. Но порог реально сохраняется и реально оценивается против
живого газа — это и есть закрытие гэпа.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join("state", "web3_gas_guard.json")

# Публичные JSON-RPC узлы (без ключей). Переопределяются через
# GasGuardConfig.rpc_overrides или env RIVEN_RPC_<CHAIN>.
DEFAULT_RPC: Dict[str, str] = {
    "ethereum": "https://ethereum-rpc.publicnode.com",
    "arbitrum": "https://arbitrum-one-rpc.publicnode.com",
    "base": "https://base-rpc.publicnode.com",
    "optimism": "https://optimism-rpc.publicnode.com",
    "polygon": "https://polygon-bor-rpc.publicnode.com",
}

DEFAULT_THRESHOLD_GWEI = 35.0


@dataclass
class GasGuardConfig:
    enabled: bool = False
    threshold_gwei: float = DEFAULT_THRESHOLD_GWEI
    chain: str = "ethereum"
    rpc_overrides: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "threshold_gwei": self.threshold_gwei,
            "chain": self.chain,
            "rpc_overrides": self.rpc_overrides,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GasGuardConfig":
        d = d or {}
        return GasGuardConfig(
            enabled=bool(d.get("enabled", False)),
            threshold_gwei=float(d.get("threshold_gwei", DEFAULT_THRESHOLD_GWEI)),
            chain=str(d.get("chain", "ethereum")),
            rpc_overrides={str(k): str(v) for k, v in (d.get("rpc_overrides", {}) or {}).items()},
        )


def load_config(path: str = CONFIG_PATH) -> GasGuardConfig:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return GasGuardConfig.from_dict(json.load(f))
    except Exception:
        logger.warning("gas-guard: не удалось прочитать %s", path, exc_info=True)
    return GasGuardConfig()


def save_config(cfg: GasGuardConfig, path: str = CONFIG_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _rpc_url(chain: str, cfg: Optional[GasGuardConfig] = None) -> Optional[str]:
    chain = (chain or "ethereum").lower()
    if cfg and chain in cfg.rpc_overrides:
        return cfg.rpc_overrides[chain]
    env = os.environ.get(f"RIVEN_RPC_{chain.upper()}")
    if env:
        return env
    return DEFAULT_RPC.get(chain)


def get_gas_price_gwei(chain: str = "ethereum", *, cfg: Optional[GasGuardConfig] = None,
                       timeout: int = 12, fetch_fn: Any = None) -> Dict[str, Any]:
    """Прочитать текущую цену газа (gwei) с публичного RPC.

    ``fetch_fn(url, payload)->dict`` инжектируется в тестах; в проде — requests.
    Возвращает {ok, chain, gas_gwei, rpc, error}.
    """
    url = _rpc_url(chain, cfg)
    if not url:
        return {"ok": False, "chain": chain, "gas_gwei": None, "rpc": None,
                "error": f"нет RPC-узла для сети {chain!r}"}

    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_gasPrice", "params": []}

    if fetch_fn is None:
        import requests

        def fetch_fn(u, p):  # type: ignore[misc]
            r = requests.post(u, json=p, timeout=timeout,
                              headers={"Content-Type": "application/json"})
            r.raise_for_status()
            return r.json()

    try:
        data = fetch_fn(url, payload)
        if not isinstance(data, dict) or "result" not in data:
            return {"ok": False, "chain": chain, "gas_gwei": None, "rpc": url,
                    "error": f"RPC вернул неожиданный ответ: {str(data)[:120]}"}
        wei = int(str(data["result"]), 16)
        gwei = wei / 1e9
        return {"ok": True, "chain": chain, "gas_gwei": round(gwei, 3), "rpc": url,
                "at": _now()}
    except Exception as exc:
        logger.warning("gas-guard: RPC %s недоступен: %s", url, exc)
        return {"ok": False, "chain": chain, "gas_gwei": None, "rpc": url,
                "error": str(exc)}


def evaluate(chain: Optional[str] = None, *, cfg: Optional[GasGuardConfig] = None,
             fetch_fn: Any = None) -> Dict[str, Any]:
    """Живой вердикт guard'а: армирован ли, пробит ли порог текущим газом."""
    cfg = cfg or load_config()
    chain = chain or cfg.chain
    gas = get_gas_price_gwei(chain, cfg=cfg, fetch_fn=fetch_fn)
    out: Dict[str, Any] = {
        "enabled": cfg.enabled,
        "chain": chain,
        "threshold_gwei": cfg.threshold_gwei,
        "gas_gwei": gas.get("gas_gwei"),
        "rpc": gas.get("rpc"),
        "gas_ok": gas.get("ok", False),
    }
    if not gas.get("ok"):
        out.update({"status": "gas_unavailable", "blocked": False,
                    "reason": gas.get("error", "цена газа недоступна"),
                    "usage_pct": None})
        return out

    over = gas["gas_gwei"] > cfg.threshold_gwei
    usage = (gas["gas_gwei"] / cfg.threshold_gwei * 100.0) if cfg.threshold_gwei > 0 else None
    if not cfg.enabled:
        status, blocked, reason = "disabled", False, "guard выключен (порог не применяется)"
    elif over:
        status, blocked = "breached", True
        reason = (f"цена газа {gas['gas_gwei']:.1f} Gwei > порога "
                  f"{cfg.threshold_gwei:.0f} Gwei — on-chain транзакции блокируются")
    else:
        status, blocked = "armed", False
        reason = (f"цена газа {gas['gas_gwei']:.1f} Gwei ≤ порога "
                  f"{cfg.threshold_gwei:.0f} Gwei")
    out.update({"status": status, "blocked": blocked, "reason": reason,
                "usage_pct": round(usage, 1) if usage is not None else None})
    return out


def preflight(chain: Optional[str] = None, *, cfg: Optional[GasGuardConfig] = None,
              fetch_fn: Any = None) -> Dict[str, Any]:
    """Pre-trade gate: любая on-chain транзакция обязана его пройти.

    ``allow=False`` → транзакцию слать нельзя (газ дороже порога при включённом
    guard). Fail-open только когда guard выключен; при включённом и недоступном
    газе — fail-closed (не знаем цену → не шлём).
    """
    v = evaluate(chain, cfg=cfg, fetch_fn=fetch_fn)
    if not v["enabled"]:
        return {"allow": True, **v}
    if not v["gas_ok"]:
        # enabled + цену не знаем → fail-closed (безопасно не отправлять)
        return {"allow": False, **v,
                "reason": "guard включён, но цена газа недоступна — fail-closed"}
    return {"allow": not v["blocked"], **v}


def _now() -> float:
    return time.time()


__all__ = [
    "CONFIG_PATH", "DEFAULT_RPC", "DEFAULT_THRESHOLD_GWEI",
    "GasGuardConfig", "evaluate", "get_gas_price_gwei", "load_config",
    "preflight", "save_config",
]

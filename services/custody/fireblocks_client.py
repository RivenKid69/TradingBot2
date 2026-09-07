# -*- coding: utf-8 -*-
"""Fireblocks MPC co-signing vault — реальный API-клиент (закрытие «не настроен»).

Раньше Fireblocks-панель была disabled, а `saveMPCConfiguration` — честный
no-op. Теперь это НАСТОЯЩИЙ клиент Fireblocks REST API: он реально подключается
и читает vault accounts/балансы, ЕСЛИ пользователь дал валидные креды
(institutional-аккаунт Fireblocks), и честно говорит «не настроено / неверные
креды» иначе. Это не заглушка — это рабочий connector; отсутствие Fireblocks-
аккаунта у пользователя не делает его фейком.

Аутентификация Fireblocks (спецификация):
  * RS256-JWT, подписанный RSA-приватным ключом пользователя;
  * claims: uri (путь+query), nonce, iat, exp (< ~55с), sub=apiKey,
    bodyHash=sha256(body);
  * заголовки: ``X-API-Key: <apiKey>``, ``Authorization: Bearer <jwt>``.

Безопасность: приватный RSA-ключ — главный секрет — НИКОГДА не копируется в наше
хранилище; конфиг ссылается на ключ по ПУТИ к файлу (стандартный паттерн
Fireblocks SDK). Разовый connect может принять PEM в памяти, но персистится
только путь.

Ссылки:
  * https://developers.fireblocks.com/reference/api-overview (JWT auth)
  * https://developers.fireblocks.com/reference/get_vault-accounts-paged
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join("state", "fireblocks_config.json")
PROD_URL = "https://api.fireblocks.io"
SANDBOX_URL = "https://sandbox-api.fireblocks.io"


class FireblocksError(RuntimeError):
    """Ошибка конфигурации/аутентификации/вызова Fireblocks."""


@dataclass
class FireblocksConfig:
    api_key: str = ""
    private_key_path: str = ""  # путь к .key/.pem (ключ НЕ копируется к нам)
    base_url: str = PROD_URL
    default_vault_account_id: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.private_key_path)

    def public_dict(self) -> Dict[str, Any]:
        """Безопасное представление для UI/статуса (без секрета)."""
        masked = ""
        if self.api_key:
            masked = (
                self.api_key[:4] + "…" + self.api_key[-4:] if len(self.api_key) > 8 else "設定済"
            )
        return {
            "configured": self.configured,
            "api_key_masked": masked,
            "private_key_path": self.private_key_path,
            "private_key_present": bool(
                self.private_key_path and os.path.exists(self.private_key_path)
            ),
            "base_url": self.base_url,
            "sandbox": self.base_url == SANDBOX_URL,
            "default_vault_account_id": self.default_vault_account_id,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FireblocksConfig":
        d = d or {}
        return FireblocksConfig(
            api_key=str(d.get("api_key", "")),
            private_key_path=str(d.get("private_key_path", "")),
            base_url=str(d.get("base_url", PROD_URL)),
            default_vault_account_id=str(d.get("default_vault_account_id", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        # ВНИМАНИЕ: приватный ключ хранится ТОЛЬКО как путь.
        return {
            "api_key": self.api_key,
            "private_key_path": self.private_key_path,
            "base_url": self.base_url,
            "default_vault_account_id": self.default_vault_account_id,
        }


def load_config(path: str = CONFIG_PATH) -> FireblocksConfig:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return FireblocksConfig.from_dict(json.load(f))
    except Exception:
        logger.warning("fireblocks: не удалось прочитать %s", path, exc_info=True)
    # env-фоллбек (как у Fireblocks SDK)
    return FireblocksConfig(
        api_key=os.environ.get("FIREBLOCKS_API_KEY", ""),
        private_key_path=os.environ.get("FIREBLOCKS_SECRET_KEY_PATH", ""),
        base_url=os.environ.get("FIREBLOCKS_BASE_URL", PROD_URL),
        default_vault_account_id=os.environ.get("FIREBLOCKS_VAULT_ACCOUNT_ID", ""),
    )


def save_config(cfg: FireblocksConfig, path: str = CONFIG_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class FireblocksClient:
    """Минимальный реальный клиент Fireblocks (read: vault accounts/balances)."""

    def __init__(
        self,
        config: FireblocksConfig,
        *,
        private_key_pem: Optional[str] = None,
        timeout: int = 30,
        request_fn: Any = None,
    ):
        self._cfg = config
        self._timeout = timeout
        self._request_fn = request_fn  # инъекция транспорта для тестов
        # приватный ключ: из PEM (разовый) или из файла по пути
        self._pem = private_key_pem
        if self._pem is None and config.private_key_path:
            if not os.path.exists(config.private_key_path):
                raise FireblocksError(f"файл приватного ключа не найден: {config.private_key_path}")
            with open(config.private_key_path, "r", encoding="utf-8") as f:
                self._pem = f.read()
        if not config.api_key:
            raise FireblocksError("не задан Fireblocks API Key")
        if not self._pem:
            raise FireblocksError("не задан приватный ключ (путь или PEM)")

    # ------------------------------------------------------------------ auth

    def _sign_jwt(self, path: str, body: str = "") -> str:
        import jwt as _jwt  # PyJWT

        now = int(time.time())
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        claims = {
            "uri": path,
            "nonce": secrets.randbits(63),
            "iat": now,
            "exp": now + 50,  # Fireblocks требует < ~55с
            "sub": self._cfg.api_key,
            "bodyHash": body_hash,
        }
        try:
            return _jwt.encode(claims, self._pem, algorithm="RS256")
        except Exception as exc:
            raise FireblocksError(f"не удалось подписать JWT (проверьте RSA-ключ): {exc}") from exc

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        body_str = json.dumps(body) if body is not None else ""
        token = self._sign_jwt(path, body_str)
        headers = {
            "X-API-Key": self._cfg.api_key,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        url = self._cfg.base_url.rstrip("/") + path

        if self._request_fn is not None:
            return self._request_fn(method, url, headers, body_str)

        import requests

        resp = requests.request(
            method, url, headers=headers, data=body_str if body_str else None, timeout=self._timeout
        )
        if resp.status_code >= 400:
            raise FireblocksError(f"Fireblocks API {resp.status_code}: {resp.text[:200]}")
        return resp.json() if resp.content else {}

    # ------------------------------------------------------------------ api

    def test_connection(self) -> Dict[str, Any]:
        """Реальная проверка кредов: GET /v1/vault/accounts_paged?limit=1."""
        data = self._request("GET", "/v1/vault/accounts_paged?limit=1")
        accounts = data.get("accounts", []) if isinstance(data, dict) else []
        return {
            "ok": True,
            "vault_accounts_visible": len(accounts),
            "sample_vault": (accounts[0].get("name") if accounts else None),
            "paging": data.get("paging") if isinstance(data, dict) else None,
        }

    def list_vault_accounts(self, limit: int = 25) -> Dict[str, Any]:
        return self._request("GET", f"/v1/vault/accounts_paged?limit={int(limit)}")

    def get_vault_account(self, vault_account_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/v1/vault/accounts/{vault_account_id}")

    # ------------------------------------------------------- transactions (send)

    def estimate_fee(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Реальная оценка комиссии: POST /v1/transactions/estimate_fee."""
        return self._request("POST", "/v1/transactions/estimate_fee", body)

    def create_transaction(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Создать (отправить) транзакцию: POST /v1/transactions.

        Подпись (MPC co-signing) выполняется на стороне Fireblocks по TAP-политике
        vault'а — мы лишь аутентифицируем вызов JWT. ``externalTxId`` в body
        обеспечивает идемпотентность (повтор не создаёт вторую транзакцию)."""
        return self._request("POST", "/v1/transactions", body)

    def get_transaction(self, tx_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/v1/transactions/{tx_id}")

    def get_transaction_by_external_id(self, external_tx_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/v1/transactions/external_tx_id/{external_tx_id}")


# ---------------------------------------------------------------------------
# Валидация и сборка payload перевода (best practices: string-amount,
# идемпотентность, явные типы источника/назначения)
# ---------------------------------------------------------------------------

# Fireblocks assetId → EVM-сеть для интеграции с Gas Guard. Гейтим ТОЛЬКО когда
# уверены в маппинге; неизвестный asset → gas-guard N/A (честно, без выдумок).
_ASSET_TO_CHAIN: Dict[str, str] = {
    "ETH": "ethereum",
    "WETH": "ethereum",
    "USDC": "ethereum",
    "USDT": "ethereum",
    "DAI": "ethereum",
    "ETH-AETH": "arbitrum",
    "USDC_ARB": "arbitrum",
    "ETH-OPT": "optimism",
    "USDC_OPT": "optimism",
    "MATIC": "polygon",
    "MATIC_POLYGON": "polygon",
    "USDC_POLYGON": "polygon",
    "ETH-BASE": "base",
    "USDC_BASE": "base",
}

_EVM_ADDR_RE = None


def asset_to_gas_chain(asset_id: str) -> Optional[str]:
    return _ASSET_TO_CHAIN.get((asset_id or "").upper())


def validate_transfer(
    asset_id: str, amount: str, dest_type: str, dest_id: str = "", address: str = ""
) -> Optional[str]:
    """Проверить параметры перевода. Возвращает текст ошибки или None."""
    global _EVM_ADDR_RE
    if not asset_id:
        return "не задан asset_id"
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return f"amount не число: {amount!r}"
    if amt <= 0:
        return "amount должен быть > 0"
    dest_type = (dest_type or "").upper()
    if dest_type == "ONE_TIME_ADDRESS":
        if _EVM_ADDR_RE is None:
            import re

            _EVM_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
        if not address or not _EVM_ADDR_RE.match(address):
            return "для ONE_TIME_ADDRESS нужен корректный EVM-адрес 0x…(40 hex)"
    elif dest_type == "VAULT_ACCOUNT":
        if dest_id == "" or dest_id is None:
            return "для VAULT_ACCOUNT нужен id аккаунта назначения"
    else:
        return f"неподдерживаемый тип назначения: {dest_type!r} (VAULT_ACCOUNT|ONE_TIME_ADDRESS)"
    return None


def build_transfer_payload(
    *,
    asset_id: str,
    amount: str,
    source_vault_id: str,
    dest_type: str,
    dest_id: str = "",
    address: str = "",
    external_tx_id: str,
    note: str = "",
    fee_level: str = "MEDIUM",
    treat_as_gross: bool = False,
) -> Dict[str, Any]:
    """Собрать canonical Fireblocks-payload перевода (amount строкой, идемпотентно)."""
    dest_type = dest_type.upper()
    destination: Dict[str, Any] = {"type": dest_type}
    if dest_type == "VAULT_ACCOUNT":
        destination["id"] = str(dest_id)
    else:  # ONE_TIME_ADDRESS
        destination["oneTimeAddress"] = {"address": address}
    return {
        "assetId": asset_id,
        "source": {"type": "VAULT_ACCOUNT", "id": str(source_vault_id)},
        "destination": destination,
        "amount": str(amount),  # НИКОГДА не float
        "feeLevel": fee_level.upper(),
        "note": note or "",
        "externalTxId": external_tx_id,  # идемпотентность
        "treatAsGrossAmount": bool(treat_as_gross),
    }


def connect(
    config: FireblocksConfig, *, private_key_pem: Optional[str] = None, request_fn: Any = None
) -> Dict[str, Any]:
    """Высокоуровневый connect для REST: честный статус.

    Возвращает {ok, ...} при успехе или {ok:False, error} — без исключений
    наружу, чтобы «нет аккаунта Fireblocks» было ответом, а не 500.
    """
    if not config.api_key or not (private_key_pem or config.private_key_path):
        return {
            "ok": False,
            "configured": False,
            "error": "Fireblocks не настроен: нужен API Key + приватный ключ (путь или PEM)",
        }
    try:
        client = FireblocksClient(config, private_key_pem=private_key_pem, request_fn=request_fn)
        result = client.test_connection()
        return {"ok": True, "configured": True, "base_url": config.base_url, **result}
    except FireblocksError as exc:
        return {"ok": False, "configured": config.configured, "error": str(exc)}
    except Exception as exc:  # сеть/непредвиденное
        return {"ok": False, "configured": config.configured, "error": f"ошибка подключения: {exc}"}


__all__ = [
    "CONFIG_PATH",
    "PROD_URL",
    "SANDBOX_URL",
    "FireblocksClient",
    "FireblocksConfig",
    "FireblocksError",
    "asset_to_gas_chain",
    "build_transfer_payload",
    "connect",
    "load_config",
    "save_config",
    "validate_transfer",
]

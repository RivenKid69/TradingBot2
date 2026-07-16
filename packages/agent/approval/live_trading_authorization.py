"""Локальная авторизация авто-торговли на LIVE-брокере (Agent-зона).

Закрывает последний пункт §4.9 из PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md:
регулярный XS-ребаланс работал только на paper (`paper_only`, fail-closed на
live). Авто-отправка ордеров на реальный счёт — самый чувствительный контур
платформы, поэтому она открывается ТОЛЬКО через явную локальную авторизацию
оператора, спроектированную по образцу реального algo-governance.

Модель (соответствует MiFID II RTS 6: «material change to an algorithm requires
re-authorisation» и практике prime-brokerage pre-trade mandates):

* **Human-in-the-loop, локально.** Авторизацию выдаёт только оператор в Agent-
  зоне (CCEA-принцип: Cloud НИКОГДА не может её выдать). Двухшаговая церемония
  (подтверждающий токен) реализуется на уровне REST/UI.
* **Привязка к хешу конфига стратегии.** Авторизация действительна ТОЛЬКО для
  того конфига, который оператор видел (sha256 канонизированного YAML). Любое
  изменение конфига → авторизация невалидна → снова fail-closed. Это ключевая
  защита: «одобрили одну стратегию — торгует другая» невозможно.
* **Привязка к брокеру.** Авторизация для ``binance`` не открывает ``oanda``.
* **Потолок лимитов (limit ceiling).** Оператор задаёт максимумы (turnover,
  notional/ребаланс, orders/ребаланс). Рантайм-лимиты ребаланса могут быть
  строже, но НИКОГДА не слабее (проверяется при использовании).
* **TTL.** Авторизации истекают (торговые мандаты не вечны).
* **Бюджет (опционально).** Максимум суммарного нотионала и/или числа
  ребалансов; при исчерпании — авторизация закрывается.
* **Revoke в любой момент** (kill switch для мандата).
* **Долговечность + tamper-evidence.** Состояние — JSON на диске (atomic);
  каждое событие (GRANT/CONSUME/REVOKE/EXPIRE/REJECT) пишется в keyed
  hash-chain аудит Agent-зоны. Переживает рестарт.

Ничего в этом модуле не отправляет ордера — он только РАЗРЕШАЕТ или ЗАПРЕЩАЕТ.
Исполнение остаётся за OMS Agent'а.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STATUS_ACTIVE = "active"
STATUS_REVOKED = "revoked"
STATUS_EXPIRED = "expired"
STATUS_EXHAUSTED = "exhausted"

# Жёсткие верхние границы, которые оператор не может превысить даже вручную
# (defense-in-depth: даже опечатка в токене не откроет неограниченный мандат).
HARD_MAX_TTL_SEC = 7 * 24 * 3600          # мандат живёт максимум неделю
HARD_MAX_NOTIONAL_PER_REBALANCE = 5_000_000.0
HARD_MAX_TURNOVER = 1.0


def canonical_config_hash(config: Any) -> str:
    """sha256 канонизированного конфига (стабильная сериализация ключей)."""
    if isinstance(config, str):
        # Уже готовый hex-хеш — вернуть как есть (позволяет вызывающему передать
        # заранее посчитанный digest файла).
        if len(config) == 64 and all(c in "0123456789abcdef" for c in config.lower()):
            return config.lower()
        payload = config.encode("utf-8")
    else:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class LimitCeiling:
    """Максимумы, заданные оператором при выдаче мандата."""

    max_turnover: float = 0.10
    max_notional_per_rebalance: float = 100_000.0
    max_orders_per_rebalance: int = 25

    def clamp_to_hard(self) -> "LimitCeiling":
        return LimitCeiling(
            max_turnover=min(float(self.max_turnover), HARD_MAX_TURNOVER),
            max_notional_per_rebalance=min(
                float(self.max_notional_per_rebalance), HARD_MAX_NOTIONAL_PER_REBALANCE),
            max_orders_per_rebalance=int(self.max_orders_per_rebalance),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiveTradingAuthorization:
    """Один долговечный мандат авто-торговли на live-брокере."""

    auth_id: str
    strategy_id: str
    config_hash: str
    broker: str
    limit_ceiling: LimitCeiling
    granted_at: float
    expires_at: float
    granted_by: str = "local_operator"
    status: str = STATUS_ACTIVE
    # Бюджет (None = без лимита по этой оси).
    max_total_notional: Optional[float] = None
    max_rebalances: Optional[int] = None
    # Потребление.
    consumed_notional: float = 0.0
    consumed_rebalances: int = 0
    revoked_at: Optional[float] = None
    revoke_reason: Optional[str] = None
    note: str = ""

    def is_active(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return self.status == STATUS_ACTIVE and now < self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["limit_ceiling"] = self.limit_ceiling.to_dict()
        return d

    def public_view(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now if now is not None else time.time()
        d = self.to_dict()
        d["active"] = self.is_active(now)
        d["seconds_remaining"] = max(0, int(self.expires_at - now))
        d["config_hash_short"] = self.config_hash[:12]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LiveTradingAuthorization":
        lc = d.get("limit_ceiling") or {}
        return cls(
            auth_id=d["auth_id"], strategy_id=d["strategy_id"],
            config_hash=d["config_hash"], broker=d["broker"],
            limit_ceiling=LimitCeiling(**lc),
            granted_at=float(d["granted_at"]), expires_at=float(d["expires_at"]),
            granted_by=d.get("granted_by", "local_operator"),
            status=d.get("status", STATUS_ACTIVE),
            max_total_notional=d.get("max_total_notional"),
            max_rebalances=d.get("max_rebalances"),
            consumed_notional=float(d.get("consumed_notional", 0.0)),
            consumed_rebalances=int(d.get("consumed_rebalances", 0)),
            revoked_at=d.get("revoked_at"), revoke_reason=d.get("revoke_reason"),
            note=d.get("note", ""),
        )


@dataclass
class AuthCheck:
    """Результат проверки права на один live-ребаланс."""

    allowed: bool
    reason: str
    auth_id: Optional[str] = None
    effective_ceiling: Optional[LimitCeiling] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"allowed": self.allowed, "reason": self.reason, "auth_id": self.auth_id}
        if self.effective_ceiling is not None:
            d["effective_ceiling"] = self.effective_ceiling.to_dict()
        return d


class LiveTradingAuthorizationStore:
    """Долговечное хранилище мандатов + keyed hash-chain аудит (Agent-зона).

    ``audit_key`` — тот же HMAC-ключ, что у tamper-evident журналов Agent'а
    (например vault master key), чтобы аудит нельзя было подделать без ключа.
    """

    def __init__(
        self,
        state_path: str,
        *,
        audit_path: Optional[str] = None,
        audit_key: Optional[bytes] = None,
        time_fn=time.time,
    ) -> None:
        self.state_path = state_path
        self.audit_path = audit_path or (state_path + ".audit.jsonl")
        self._time = time_fn
        self._lock = threading.RLock()
        self._auths: Dict[str, LiveTradingAuthorization] = {}
        self._chain = None  # type: ignore[assignment]
        self._audit_key = audit_key
        self._init_chain(audit_key)
        self._load()

    # --------------------------------------------------------------- аудит

    def _init_chain(self, audit_key: Optional[bytes]) -> None:
        try:
            from packages.agent.audit.hash_chain import HashChain
            self._chain = HashChain(key=audit_key)
        except Exception:  # pragma: no cover - hash_chain всегда доступен
            self._chain = None

    def _audit(self, event: str, payload: Dict[str, Any]) -> None:
        entry = {
            "event": event,
            "at": self._time(),
            "at_iso": _iso(self._time()),
            **payload,
        }
        if self._chain is not None:
            rec = self._chain.append(entry)
            entry = {**entry, "seq": rec.seq, "entry_hash": rec.entry_hash,
                     "prev_hash": rec.prev_hash}
        try:
            os.makedirs(os.path.dirname(self.audit_path) or ".", exist_ok=True)
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("live-auth: не удалось записать аудит")

    def verify_audit(self) -> Dict[str, Any]:
        if self._chain is None:
            return {"valid": None, "reason": "chain unavailable"}
        return self._chain.verify()

    # --------------------------------------------------------------- state

    def _load(self) -> None:
        with self._lock:
            if not os.path.exists(self.state_path):
                return
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                for d in data.get("auths", []):
                    a = LiveTradingAuthorization.from_dict(d)
                    self._auths[a.auth_id] = a
                # Rebuild in-memory audit chain so head_hash продолжает цепочку.
                if self._chain is not None and os.path.exists(self.audit_path):
                    from packages.agent.audit.hash_chain import HashChain
                    chain = HashChain(key=self._audit_key)
                    with open(self.audit_path, "r", encoding="utf-8") as af:
                        for line in af:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                chain.append({k: v for k, v in rec.items()
                                              if k not in ("seq", "entry_hash", "prev_hash")})
                            except Exception:
                                continue
                    self._chain = chain
            except Exception:
                logger.exception("live-auth: не удалось загрузить состояние")

    def _save(self) -> None:
        with self._lock:
            from services.utils_app import atomic_write_json
            atomic_write_json(self.state_path, {
                "updated_at": _iso(self._time()),
                "auths": [a.to_dict() for a in self._auths.values()],
            })

    # ------------------------------------------------------------- grant

    def grant(
        self,
        *,
        strategy_id: str,
        config: Any,
        broker: str,
        limit_ceiling: LimitCeiling,
        ttl_sec: int,
        confirmation_token: str,
        expected_token: str,
        max_total_notional: Optional[float] = None,
        max_rebalances: Optional[int] = None,
        granted_by: str = "local_operator",
        note: str = "",
    ) -> Dict[str, Any]:
        """Выдать мандат. Двухшаговая церемония: ``confirmation_token`` должен
        совпасть с ``expected_token`` (оператор явно подтверждает намерение).

        live-брокер ``sim_paper`` отклоняется — мандат имеет смысл только для
        настоящего брокера.
        """
        if str(broker).strip().lower() == "sim_paper":
            return {"ok": False, "error": "sim_paper не требует авторизации live-торговли"}
        if not secrets.compare_digest(str(confirmation_token), str(expected_token)):
            self._audit("REJECT_GRANT", {"strategy_id": strategy_id, "broker": broker,
                                         "reason": "confirmation token mismatch"})
            return {"ok": False, "error": "подтверждающий токен не совпал — мандат не выдан"}
        ttl = max(60, min(int(ttl_sec), HARD_MAX_TTL_SEC))
        ceiling = limit_ceiling.clamp_to_hard()
        now = self._time()
        cfg_hash = canonical_config_hash(config)
        auth = LiveTradingAuthorization(
            auth_id=uuid.uuid4().hex,
            strategy_id=str(strategy_id),
            config_hash=cfg_hash,
            broker=str(broker).strip().lower(),
            limit_ceiling=ceiling,
            granted_at=now,
            expires_at=now + ttl,
            granted_by=granted_by,
            max_total_notional=(float(max_total_notional) if max_total_notional else None),
            max_rebalances=(int(max_rebalances) if max_rebalances else None),
            note=note,
        )
        with self._lock:
            # Один активный мандат на (strategy_id, broker): новый вытесняет старый.
            for old in self._auths.values():
                if (old.strategy_id == auth.strategy_id and old.broker == auth.broker
                        and old.status == STATUS_ACTIVE):
                    old.status = STATUS_REVOKED
                    old.revoked_at = now
                    old.revoke_reason = "superseded by new grant"
                    self._audit("SUPERSEDE", {"auth_id": old.auth_id})
            self._auths[auth.auth_id] = auth
            self._save()
        self._audit("GRANT", {
            "auth_id": auth.auth_id, "strategy_id": auth.strategy_id,
            "broker": auth.broker, "config_hash": auth.config_hash,
            "expires_at": auth.expires_at, "ceiling": ceiling.to_dict(),
            "max_total_notional": auth.max_total_notional,
            "max_rebalances": auth.max_rebalances, "granted_by": granted_by,
        })
        logger.warning("live-auth: GRANTED %s strategy=%s broker=%s ttl=%ds",
                       auth.auth_id, auth.strategy_id, auth.broker, ttl)
        return {"ok": True, "authorization": auth.public_view(now)}

    # ------------------------------------------------------------- revoke

    def revoke(self, auth_id: str, *, reason: str = "operator revoke") -> Dict[str, Any]:
        with self._lock:
            auth = self._auths.get(auth_id)
            if auth is None:
                return {"ok": False, "error": f"мандат {auth_id} не найден"}
            if auth.status != STATUS_ACTIVE:
                return {"ok": True, "already": auth.status, "auth_id": auth_id}
            auth.status = STATUS_REVOKED
            auth.revoked_at = self._time()
            auth.revoke_reason = reason
            self._save()
        self._audit("REVOKE", {"auth_id": auth_id, "reason": reason})
        logger.warning("live-auth: REVOKED %s (%s)", auth_id, reason)
        return {"ok": True, "auth_id": auth_id, "status": STATUS_REVOKED}

    def revoke_all(self, *, reason: str = "revoke all") -> Dict[str, Any]:
        n = 0
        with self._lock:
            for auth in self._auths.values():
                if auth.status == STATUS_ACTIVE:
                    auth.status = STATUS_REVOKED
                    auth.revoked_at = self._time()
                    auth.revoke_reason = reason
                    n += 1
            if n:
                self._save()
        if n:
            self._audit("REVOKE_ALL", {"count": n, "reason": reason})
        return {"ok": True, "revoked": n}

    # -------------------------------------------------------------- check

    def check(
        self,
        *,
        strategy_id: str,
        config: Any,
        broker: str,
        turnover: float,
        notional: float,
        n_orders: int,
    ) -> AuthCheck:
        """Проверить право на ОДИН live-ребаланс с конкретными параметрами.

        Не мутирует состояние (это делает :meth:`consume` после успешной
        отправки). Возвращает эффективный потолок для этого ребаланса.
        """
        now = self._time()
        cfg_hash = canonical_config_hash(config)
        broker = str(broker).strip().lower()
        with self._lock:
            auth = self._active_for(strategy_id, broker, now)
            if auth is None:
                self._audit("REJECT", {"strategy_id": strategy_id, "broker": broker,
                                       "reason": "no active authorization"})
                return AuthCheck(False, "нет активной авторизации live-торговли для этой стратегии/брокера")
            if auth.config_hash != cfg_hash:
                self._audit("REJECT", {"auth_id": auth.auth_id, "reason": "config hash mismatch",
                                       "expected": auth.config_hash, "got": cfg_hash})
                return AuthCheck(False,
                                 "конфиг стратегии изменился с момента авторизации — требуется повторная авторизация",
                                 auth.auth_id)
            c = auth.limit_ceiling
            if turnover > c.max_turnover + 1e-9:
                return self._reject_limit(auth, "turnover", turnover, c.max_turnover)
            if notional > c.max_notional_per_rebalance + 1e-6:
                return self._reject_limit(auth, "notional", notional, c.max_notional_per_rebalance)
            if n_orders > c.max_orders_per_rebalance:
                return self._reject_limit(auth, "orders", n_orders, c.max_orders_per_rebalance)
            if auth.max_rebalances is not None and auth.consumed_rebalances >= auth.max_rebalances:
                return AuthCheck(False, "исчерпан бюджет ребалансов мандата", auth.auth_id)
            if (auth.max_total_notional is not None
                    and auth.consumed_notional + notional > auth.max_total_notional + 1e-6):
                return AuthCheck(False, "исчерпан нотиональный бюджет мандата", auth.auth_id)
            return AuthCheck(True, "авторизовано", auth.auth_id, effective_ceiling=c)

    def consume(self, auth_id: str, *, notional: float, n_orders: int) -> Dict[str, Any]:
        """Зафиксировать использование мандата ПОСЛЕ фактической отправки."""
        now = self._time()
        with self._lock:
            auth = self._auths.get(auth_id)
            if auth is None or not auth.is_active(now):
                return {"ok": False, "error": "мандат неактивен на момент consume"}
            auth.consumed_notional += float(max(0.0, notional))
            auth.consumed_rebalances += 1
            exhausted = False
            if auth.max_rebalances is not None and auth.consumed_rebalances >= auth.max_rebalances:
                exhausted = True
            if (auth.max_total_notional is not None
                    and auth.consumed_notional >= auth.max_total_notional):
                exhausted = True
            if exhausted:
                auth.status = STATUS_EXHAUSTED
            self._save()
        self._audit("CONSUME", {"auth_id": auth_id, "notional": round(float(notional), 2),
                                "n_orders": int(n_orders),
                                "consumed_notional": round(auth.consumed_notional, 2),
                                "consumed_rebalances": auth.consumed_rebalances,
                                "exhausted": exhausted})
        return {"ok": True, "exhausted": exhausted,
                "consumed_notional": round(auth.consumed_notional, 2),
                "consumed_rebalances": auth.consumed_rebalances}

    # ----------------------------------------------------------- helpers

    def _active_for(self, strategy_id: str, broker: str,
                    now: float) -> Optional[LiveTradingAuthorization]:
        expired_any = False
        result = None
        for auth in self._auths.values():
            if auth.strategy_id != strategy_id or auth.broker != broker:
                continue
            if auth.status == STATUS_ACTIVE and now >= auth.expires_at:
                auth.status = STATUS_EXPIRED
                expired_any = True
                self._audit("EXPIRE", {"auth_id": auth.auth_id})
                continue
            if auth.is_active(now):
                result = auth
        if expired_any:
            self._save()
        return result

    def _reject_limit(self, auth: LiveTradingAuthorization, kind: str,
                      value: float, ceiling: float) -> AuthCheck:
        self._audit("REJECT", {"auth_id": auth.auth_id, "reason": f"{kind} exceeds ceiling",
                               "value": value, "ceiling": ceiling})
        return AuthCheck(False,
                         f"{kind}={value:.4g} превышает потолок мандата {ceiling:.4g}",
                         auth.auth_id)

    def status(self) -> Dict[str, Any]:
        now = self._time()
        with self._lock:
            # Пометить истёкшие при чтении статуса (ленивый sweep).
            changed = False
            for auth in self._auths.values():
                if auth.status == STATUS_ACTIVE and now >= auth.expires_at:
                    auth.status = STATUS_EXPIRED
                    changed = True
                    self._audit("EXPIRE", {"auth_id": auth.auth_id})
            if changed:
                self._save()
            active = [a.public_view(now) for a in self._auths.values() if a.is_active(now)]
            recent = sorted((a.public_view(now) for a in self._auths.values()),
                            key=lambda d: d["granted_at"], reverse=True)[:20]
        audit = self.verify_audit()
        return {
            "active": active,
            "recent": recent,
            "audit_valid": audit.get("valid"),
            "audit_records": audit.get("n"),
        }


def _iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "AuthCheck",
    "HARD_MAX_NOTIONAL_PER_REBALANCE",
    "HARD_MAX_TTL_SEC",
    "HARD_MAX_TURNOVER",
    "LimitCeiling",
    "LiveTradingAuthorization",
    "LiveTradingAuthorizationStore",
    "canonical_config_hash",
]

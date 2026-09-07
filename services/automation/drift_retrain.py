# -*- coding: utf-8 -*-
"""
services/automation/drift_retrain.py
====================================

Scheduled-ретрейн по дрейфу данных (P2): мониторит PSI (Population Stability Index,
из ``drift.py`` / ``models/drift_report.json``) и **автоматически инициирует ретрейн**
при превышении порога, с cooldown (не чаще, чем раз в N). Закрывает разрыв «дрейф детектится,
но ретрейн ручной».

DI-время (``time_fn``) → детерминированные тесты. Ретрейн — через callback (``retrain_fn``);
сам ретрейн (тяжёлый) не трогаем — только триггерим. Слой services.
"""

from __future__ import annotations

import json
import logging
import os
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Стандартные интерпретации PSI (industry): <0.1 stable, 0.1-0.25 moderate, >0.25 significant.
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: str
    max_psi: float
    triggering_features: List[str] = field(default_factory=list)
    on_cooldown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_retrain": self.should_retrain,
            "reason": self.reason,
            "max_psi": self.max_psi,
            "triggering_features": list(self.triggering_features),
            "on_cooldown": self.on_cooldown,
        }


def psi_from_report(report: Any) -> Dict[str, float]:
    """Извлечь {feature: psi} из drift-отчёта (dict / вложенный / JSON-файл)."""
    if isinstance(report, str) and os.path.exists(report):
        with open(report, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    out: Dict[str, float] = {}
    if isinstance(report, dict):
        # формат {feature: {psi: x}} или {feature: x} или {"features": {...}}
        src = report.get("features", report)
        if isinstance(src, dict):
            for k, v in src.items():
                if isinstance(v, dict) and "psi" in v:
                    out[str(k)] = float(v["psi"])
                elif isinstance(v, (int, float)):
                    out[str(k)] = float(v)
    return out


class DriftRetrainScheduler:
    def __init__(
        self,
        *,
        psi_threshold: float = PSI_SIGNIFICANT,
        cooldown_sec: float = 86_400.0,
        min_features: int = 1,
        time_fn: Callable[[], float] = _time.time,
    ) -> None:
        self.psi_threshold = float(psi_threshold)
        self.cooldown_sec = float(cooldown_sec)
        self.min_features = int(min_features)
        self._time = time_fn
        self._last_retrain_ts: Optional[float] = None

    def check(self, report: Any) -> RetrainDecision:
        """Решение о ретрейне по drift-отчёту (без выполнения)."""
        psi = psi_from_report(report)
        if not psi:
            return RetrainDecision(False, "no PSI data", 0.0)
        trig = sorted([f for f, v in psi.items() if v >= self.psi_threshold], key=lambda f: -psi[f])
        max_psi = max(psi.values())
        if len(trig) < self.min_features:
            return RetrainDecision(
                False, f"max PSI {max_psi:.3f} < threshold {self.psi_threshold}", max_psi
            )
        # cooldown
        if (
            self._last_retrain_ts is not None
            and (self._time() - self._last_retrain_ts) < self.cooldown_sec
        ):
            return RetrainDecision(False, "retrain on cooldown", max_psi, trig, on_cooldown=True)
        return RetrainDecision(
            True, f"{len(trig)} feature(s) drift PSI>={self.psi_threshold}", max_psi, trig
        )

    def run(
        self, report: Any, retrain_fn: Optional[Callable[[RetrainDecision], Any]] = None
    ) -> RetrainDecision:
        """Проверить и при необходимости запустить ретрейн (callback). Фиксирует cooldown."""
        decision = self.check(report)
        if decision.should_retrain:
            self._last_retrain_ts = self._time()
            logger.warning(
                "DriftRetrain: triggering retrain — %s (features=%s)",
                decision.reason,
                decision.triggering_features,
            )
            if retrain_fn is not None:
                try:
                    retrain_fn(decision)
                except Exception as exc:  # pragma: no cover
                    logger.error("retrain_fn failed: %s", exc)
        return decision

    def run_closed_loop(
        self,
        report: Any,
        retrain_fn: Callable[[RetrainDecision], Any],
        *,
        register_fn: Optional[Callable[[Any, RetrainDecision], Any]] = None,
        verify_fn: Optional[Callable[[Any], bool]] = None,
        promote_fn: Optional[Callable[[Any], Any]] = None,
    ) -> Dict[str, Any]:
        """Closed-loop retrain (P2 #24): check → retrain → register → verify → promote.

        Unlike ``run`` (a fire-and-forget trigger), this completes the loop:
          1. ``retrain_fn(decision)`` -> new artifact;
          2. ``register_fn(artifact, decision)`` -> registry entry (optional);
          3. ``verify_fn(artifact)`` -> only promote if it actually improves (gate);
          4. ``promote_fn(artifact)`` -> stage to production (optional).
        Returns a structured outcome dict so the caller has full provenance.
        """
        decision = self.check(report)
        out: Dict[str, Any] = {
            "decision": decision.to_dict(),
            "retrained": False,
            "registered": False,
            "verified": None,
            "promoted": False,
            "artifact": None,
            "error": None,
        }
        if not decision.should_retrain:
            return out
        self._last_retrain_ts = self._time()
        try:
            artifact = retrain_fn(decision)
            out["retrained"] = True
            out["artifact"] = getattr(artifact, "name", str(artifact))
            if register_fn is not None:
                register_fn(artifact, decision)
                out["registered"] = True
            verified = True
            if verify_fn is not None:
                verified = bool(verify_fn(artifact))
                out["verified"] = verified
            if verified and promote_fn is not None:
                promote_fn(artifact)
                out["promoted"] = True
            logger.warning(
                "DriftRetrain closed-loop: retrained=%s registered=%s verified=%s promoted=%s",
                out["retrained"],
                out["registered"],
                out["verified"],
                out["promoted"],
            )
        except Exception as exc:  # pragma: no cover - never crash the loop
            out["error"] = str(exc)
            logger.error("closed-loop retrain failed: %s", exc)
        return out

    @property
    def last_retrain_ts(self) -> Optional[float]:
        return self._last_retrain_ts


__all__ = [
    "RetrainDecision",
    "DriftRetrainScheduler",
    "psi_from_report",
    "PSI_MODERATE",
    "PSI_SIGNIFICANT",
]

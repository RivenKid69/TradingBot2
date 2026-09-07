# -*- coding: utf-8 -*-
"""
User AI Acknowledgment System.

This module implements explicit user acknowledgment functionality per
Article 50 of the EU AI Act. Users must acknowledge they understand
they are interacting with an AI system before accessing AI-powered features.

Key Requirements:
- Article 50(1): Users must be informed they are interacting with AI
- Article 50: Acknowledgment must be explicit and recorded
- GPAI Code of Practice: Clear user consent and understanding

The acknowledgment system:
1. Tracks required acknowledgments per feature
2. Records explicit user consent
3. Maintains audit trail for compliance
4. Supports multiple acknowledgment types
5. Prevents access to features without proper acknowledgments

References:
    - EU AI Act Article 50: https://artificialintelligenceact.eu/article/50/
    - Recital 132: Transparency for limited-risk AI
    - GDPR Article 7: Conditions for consent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple, Set
import hashlib
import json


class AcknowledgmentType(Enum):
    """
    Types of acknowledgments required.

    Each type corresponds to specific user understanding requirements.
    """

    AI_SYSTEM_AWARENESS = "ai_system_awareness"
    RISK_UNDERSTANDING = "risk_understanding"
    LIMITATION_ACCEPTANCE = "limitation_acceptance"
    LIVE_TRADING_CONSENT = "live_trading_consent"
    DATA_PROCESSING_CONSENT = "data_processing_consent"
    PERFORMANCE_DISCLAIMER = "performance_disclaimer"


class AcknowledgmentStatus(Enum):
    """Status of an acknowledgment."""

    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    EXPIRED = "expired"
    REVOKED = "revoked"


class FeatureCategory(Enum):
    """Categories of platform features requiring acknowledgments."""

    REGISTRATION = "registration"
    STRATEGY_CREATION = "strategy_creation"
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    API_ACCESS = "api_access"


@dataclass
class UserAcknowledgment:
    """
    Record of user acknowledgment.

    Captures explicit user consent per Article 50 requirements.

    Attributes:
        acknowledgment_id: Unique identifier
        user_id: User who acknowledged
        acknowledgment_type: Type of acknowledgment
        timestamp: When acknowledged
        status: Current status
        ip_address: IP address at time of acknowledgment
        user_agent: Browser/client information
        version: Version of acknowledgment text
        text_hash: Hash of acknowledged text for verification
        metadata: Additional metadata
    """

    acknowledgment_id: str
    user_id: str
    acknowledgment_type: AcknowledgmentType
    timestamp: datetime
    status: AcknowledgmentStatus = AcknowledgmentStatus.ACKNOWLEDGED
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    version: str = "1.0"
    text_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        user_id: str,
        ack_type: AcknowledgmentType,
        text_content: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "UserAcknowledgment":
        """
        Create new acknowledgment record.

        Args:
            user_id: User identifier
            ack_type: Type of acknowledgment
            text_content: Text that was acknowledged
            ip_address: Optional IP address
            user_agent: Optional user agent
            metadata: Optional additional metadata

        Returns:
            New UserAcknowledgment instance
        """
        ack_id = hashlib.sha256(
            f"{user_id}:{ack_type.value}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        text_hash = hashlib.sha256(text_content.encode()).hexdigest()

        return cls(
            acknowledgment_id=ack_id,
            user_id=user_id,
            acknowledgment_type=ack_type,
            timestamp=datetime.utcnow(),
            status=AcknowledgmentStatus.ACKNOWLEDGED,
            ip_address=ip_address,
            user_agent=user_agent,
            text_hash=text_hash,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "acknowledgment_id": self.acknowledgment_id,
            "user_id": self.user_id,
            "acknowledgment_type": self.acknowledgment_type.value,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "version": self.version,
            "text_hash": self.text_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserAcknowledgment":
        """Create from dictionary."""
        return cls(
            acknowledgment_id=data["acknowledgment_id"],
            user_id=data["user_id"],
            acknowledgment_type=AcknowledgmentType(data["acknowledgment_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=AcknowledgmentStatus(data.get("status", "acknowledged")),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            version=data.get("version", "1.0"),
            text_hash=data.get("text_hash", ""),
            metadata=data.get("metadata", {}),
        )

    def is_valid(self) -> bool:
        """Check if acknowledgment is currently valid."""
        return self.status == AcknowledgmentStatus.ACKNOWLEDGED


@dataclass
class AcknowledgmentAuditRecord:
    """Audit record for acknowledgment actions."""

    record_id: str
    acknowledgment_id: str
    user_id: str
    action: str  # created, revoked, expired
    timestamp: datetime
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Acknowledgment texts per Article 50
ACKNOWLEDGMENT_TEXTS: Dict[AcknowledgmentType, Dict[str, str]] = {
    AcknowledgmentType.AI_SYSTEM_AWARENESS: {
        "en": """
I acknowledge that:

1. This platform uses an Artificial Intelligence (AI) system based on
   machine learning technology (Reinforcement Learning).

2. Trading signals and predictions are generated by the AI model,
   not by human analysts.

3. I am interacting with an AI system when using the trading features
   of this platform.

4. Per Article 50 of the EU AI Act, I have been informed about the
   AI nature of this system.
""",
        "ru": """
Я подтверждаю, что:

1. Данная платформа использует систему искусственного интеллекта (ИИ)
   на основе технологии машинного обучения (Reinforcement Learning).

2. Торговые сигналы и прогнозы генерируются ИИ-моделью,
   а не аналитиками-людьми.

3. Я взаимодействую с ИИ-системой при использовании торговых функций
   данной платформы.

4. В соответствии со Статьёй 50 EU AI Act, я был(а) проинформирован(а)
   о природе ИИ данной системы.
""",
    },
    AcknowledgmentType.RISK_UNDERSTANDING: {
        "en": """
I understand and accept that:

1. AI predictions may be inaccurate or suboptimal.

2. Past performance does not guarantee future results.

3. I may experience significant financial losses when using this platform.

4. The AI system has limitations and may behave unexpectedly in certain
   market conditions.

5. Trading in financial markets involves substantial risk of loss.

6. I should not trade with funds I cannot afford to lose.
""",
        "ru": """
Я понимаю и принимаю, что:

1. Прогнозы ИИ могут быть неточными или неоптимальными.

2. Прошлые результаты не гарантируют будущую доходность.

3. Я могу понести значительные финансовые потери при использовании
   данной платформы.

4. ИИ-система имеет ограничения и может вести себя неожиданно
   в определённых рыночных условиях.

5. Торговля на финансовых рынках связана со значительным риском убытков.

6. Мне не следует торговать средствами, потерю которых я не могу себе позволить.
""",
    },
    AcknowledgmentType.LIMITATION_ACCEPTANCE: {
        "en": """
I acknowledge the following AI system limitations:

1. The model was trained on historical data that may not reflect
   future market conditions.

2. Performance may degrade during unprecedented market events
   (e.g., flash crashes, black swan events).

3. The system requires human oversight for safe operation.

4. I am responsible for monitoring AI outputs and making final
   trading decisions.

5. The system may not perform equally well across all market conditions
   or asset classes.

6. Technical failures may occur, requiring manual intervention.
""",
        "ru": """
Я признаю следующие ограничения ИИ-системы:

1. Модель была обучена на исторических данных, которые могут не отражать
   будущие рыночные условия.

2. Производительность может снизиться во время беспрецедентных рыночных
   событий (например, flash crash, события типа "чёрный лебедь").

3. Система требует человеческого контроля для безопасной работы.

4. Я несу ответственность за мониторинг выходных данных ИИ и принятие
   окончательных торговых решений.

5. Система может работать по-разному в различных рыночных условиях
   или классах активов.

6. Возможны технические сбои, требующие ручного вмешательства.
""",
    },
    AcknowledgmentType.LIVE_TRADING_CONSENT: {
        "en": """
Before enabling live trading, I confirm that:

1. I understand I am using an AI-powered trading system.

2. I accept full responsibility for all trading decisions and their
   financial consequences.

3. I will maintain appropriate oversight of the AI system as required
   by Article 14 of the EU AI Act.

4. I understand that substantial losses are possible and likely.

5. I am trading with funds I can afford to lose completely.

6. I have read and understood all risk disclosures.

7. I will not hold the platform liable for losses caused by AI
   predictions or system failures.

8. I understand how to use the kill switch to immediately halt
   trading if needed.
""",
        "ru": """
Перед активацией реальной торговли я подтверждаю, что:

1. Я понимаю, что использую торговую систему на основе ИИ.

2. Я принимаю полную ответственность за все торговые решения и их
   финансовые последствия.

3. Я буду поддерживать надлежащий контроль над ИИ-системой согласно
   требованиям Статьи 14 EU AI Act.

4. Я понимаю, что значительные убытки возможны и вероятны.

5. Я торгую средствами, полную потерю которых могу себе позволить.

6. Я прочитал(а) и понял(а) все предупреждения о рисках.

7. Я не буду возлагать ответственность на платформу за убытки,
   вызванные прогнозами ИИ или сбоями системы.

8. Я понимаю, как использовать аварийную остановку для немедленного
   прекращения торговли при необходимости.
""",
    },
    AcknowledgmentType.DATA_PROCESSING_CONSENT: {
        "en": """
I consent to the processing of my data as follows:

1. My trading data will be used to improve AI model performance.

2. Anonymous, aggregated data may be used for research purposes.

3. My data will be stored securely per GDPR requirements.

4. I can request data deletion per GDPR Article 17.

5. I can withdraw this consent at any time.
""",
        "ru": """
Я даю согласие на обработку моих данных следующим образом:

1. Мои торговые данные будут использоваться для улучшения
   производительности ИИ-модели.

2. Анонимные агрегированные данные могут использоваться в
   исследовательских целях.

3. Мои данные будут храниться в безопасности согласно требованиям GDPR.

4. Я могу запросить удаление данных согласно Статье 17 GDPR.

5. Я могу отозвать это согласие в любое время.
""",
    },
    AcknowledgmentType.PERFORMANCE_DISCLAIMER: {
        "en": """
I acknowledge the following regarding performance:

1. Past performance is not indicative of future results.

2. Backtested results may not reflect real trading conditions.

3. Slippage, fees, and market impact may reduce actual returns.

4. The Sharpe ratio and other metrics are estimates, not guarantees.

5. Results may vary significantly in live trading conditions.
""",
        "ru": """
Я признаю следующее относительно производительности:

1. Прошлые результаты не являются показателем будущих результатов.

2. Результаты бэктестинга могут не отражать реальные торговые условия.

3. Проскальзывание, комиссии и влияние на рынок могут снизить
   фактическую доходность.

4. Коэффициент Шарпа и другие метрики являются оценками, а не гарантиями.

5. Результаты могут значительно отличаться в условиях реальной торговли.
""",
    },
}

# Feature to required acknowledgments mapping
FEATURE_REQUIREMENTS: Dict[FeatureCategory, List[AcknowledgmentType]] = {
    FeatureCategory.REGISTRATION: [
        AcknowledgmentType.AI_SYSTEM_AWARENESS,
    ],
    FeatureCategory.STRATEGY_CREATION: [
        AcknowledgmentType.AI_SYSTEM_AWARENESS,
        AcknowledgmentType.LIMITATION_ACCEPTANCE,
    ],
    FeatureCategory.BACKTESTING: [
        AcknowledgmentType.AI_SYSTEM_AWARENESS,
        AcknowledgmentType.LIMITATION_ACCEPTANCE,
        AcknowledgmentType.PERFORMANCE_DISCLAIMER,
    ],
    FeatureCategory.PAPER_TRADING: [
        AcknowledgmentType.AI_SYSTEM_AWARENESS,
        AcknowledgmentType.LIMITATION_ACCEPTANCE,
        AcknowledgmentType.RISK_UNDERSTANDING,
    ],
    FeatureCategory.LIVE_TRADING: [
        AcknowledgmentType.AI_SYSTEM_AWARENESS,
        AcknowledgmentType.RISK_UNDERSTANDING,
        AcknowledgmentType.LIMITATION_ACCEPTANCE,
        AcknowledgmentType.LIVE_TRADING_CONSENT,
    ],
    FeatureCategory.API_ACCESS: [
        AcknowledgmentType.AI_SYSTEM_AWARENESS,
        AcknowledgmentType.LIMITATION_ACCEPTANCE,
        AcknowledgmentType.DATA_PROCESSING_CONSENT,
    ],
}


class UserAcknowledgmentManager:
    """
    Manages user acknowledgments for AI features.

    Responsibilities:
    1. Track required acknowledgments per feature
    2. Record explicit user consent
    3. Verify feature access permissions
    4. Maintain audit trail for compliance
    5. Support acknowledgment revocation

    Example:
        >>> manager = create_acknowledgment_manager()
        >>> required = manager.get_required_acknowledgments("user1", "live_trading")
        >>> can_access, missing = manager.check_feature_access("user1", "live_trading")
        >>> if missing:
        ...     for ack_type in missing:
        ...         text = manager.get_acknowledgment_text(ack_type)
        ...         # Show text to user, get consent
        ...         manager.record_acknowledgment("user1", ack_type)
    """

    def __init__(self):
        """Initialize the acknowledgment manager."""
        self.acknowledgments: Dict[str, List[UserAcknowledgment]] = {}
        self.audit_log: List[AcknowledgmentAuditRecord] = []
        self._feature_requirements = FEATURE_REQUIREMENTS.copy()

    def get_required_acknowledgments(self, user_id: str, feature: str) -> List[AcknowledgmentType]:
        """
        Get acknowledgments required for a feature that user hasn't completed.

        Args:
            user_id: User identifier
            feature: Feature name (e.g., "registration", "live_trading")

        Returns:
            List of acknowledgment types still required
        """
        # Get feature category
        try:
            category = FeatureCategory(feature)
        except ValueError:
            # Unknown feature, require AI awareness at minimum
            return [AcknowledgmentType.AI_SYSTEM_AWARENESS]

        required = self._feature_requirements.get(category, [])
        acknowledged = self._get_user_acknowledged_types(user_id)

        return [r for r in required if r not in acknowledged]

    def record_acknowledgment(
        self,
        user_id: str,
        ack_type: AcknowledgmentType,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UserAcknowledgment:
        """
        Record user acknowledgment.

        Args:
            user_id: User identifier
            ack_type: Type of acknowledgment
            ip_address: Optional IP for audit trail
            user_agent: Optional browser/client info
            language: Language of acknowledged text
            metadata: Optional additional metadata

        Returns:
            Created UserAcknowledgment instance
        """
        text = self.get_acknowledgment_text(ack_type, language)
        ack = UserAcknowledgment.create(
            user_id=user_id,
            ack_type=ack_type,
            text_content=text,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
        )

        if user_id not in self.acknowledgments:
            self.acknowledgments[user_id] = []
        self.acknowledgments[user_id].append(ack)

        # Audit log
        self._log_action(
            ack.acknowledgment_id, user_id, "created", f"Acknowledged {ack_type.value}"
        )

        return ack

    def check_feature_access(
        self, user_id: str, feature: str
    ) -> Tuple[bool, List[AcknowledgmentType]]:
        """
        Check if user can access feature (has all required acknowledgments).

        Args:
            user_id: User identifier
            feature: Feature name

        Returns:
            Tuple of (can_access: bool, missing_acknowledgments: list)
        """
        missing = self.get_required_acknowledgments(user_id, feature)
        return (len(missing) == 0, missing)

    def get_acknowledgment_text(self, ack_type: AcknowledgmentType, language: str = "en") -> str:
        """
        Get acknowledgment text for display.

        Args:
            ack_type: Type of acknowledgment
            language: Language code

        Returns:
            Acknowledgment text string
        """
        texts = ACKNOWLEDGMENT_TEXTS.get(ack_type, {})
        return texts.get(language, texts.get("en", ""))

    def get_all_acknowledgment_texts(self, language: str = "en") -> Dict[str, str]:
        """
        Get all acknowledgment texts.

        Args:
            language: Language code

        Returns:
            Dictionary of acknowledgment type to text
        """
        return {
            ack_type.value: self.get_acknowledgment_text(ack_type, language)
            for ack_type in AcknowledgmentType
        }

    def get_user_acknowledgments(
        self, user_id: str, ack_type: Optional[AcknowledgmentType] = None
    ) -> List[UserAcknowledgment]:
        """
        Get all acknowledgments for a user.

        Args:
            user_id: User identifier
            ack_type: Optional filter by type

        Returns:
            List of UserAcknowledgment instances
        """
        acks = self.acknowledgments.get(user_id, [])

        if ack_type:
            acks = [a for a in acks if a.acknowledgment_type == ack_type]

        return acks

    def revoke_acknowledgment(
        self, user_id: str, ack_type: AcknowledgmentType, reason: str
    ) -> bool:
        """
        Revoke a user's acknowledgment.

        Args:
            user_id: User identifier
            ack_type: Type to revoke
            reason: Reason for revocation

        Returns:
            True if revoked successfully
        """
        if user_id not in self.acknowledgments:
            return False

        revoked = False
        for ack in self.acknowledgments[user_id]:
            if (
                ack.acknowledgment_type == ack_type
                and ack.status == AcknowledgmentStatus.ACKNOWLEDGED
            ):
                ack.status = AcknowledgmentStatus.REVOKED
                self._log_action(ack.acknowledgment_id, user_id, "revoked", reason)
                revoked = True

        return revoked

    def verify_compliance(self, user_id: str) -> Dict[str, Any]:
        """
        Verify user's acknowledgment compliance status.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with compliance status per feature
        """
        result = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "features": {},
            "total_acknowledgments": len(self.acknowledgments.get(user_id, [])),
        }

        for feature in FeatureCategory:
            can_access, missing = self.check_feature_access(user_id, feature.value)
            result["features"][feature.value] = {
                "can_access": can_access,
                "missing_acknowledgments": [m.value for m in missing],
            }

        return result

    def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get audit trail for acknowledgments.

        Args:
            user_id: Optional filter by user
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            List of audit records
        """
        records = self.audit_log

        if user_id:
            records = [r for r in records if r.user_id == user_id]
        if start_date:
            records = [r for r in records if r.timestamp >= start_date]
        if end_date:
            records = [r for r in records if r.timestamp <= end_date]

        return [
            {
                "record_id": r.record_id,
                "acknowledgment_id": r.acknowledgment_id,
                "user_id": r.user_id,
                "action": r.action,
                "timestamp": r.timestamp.isoformat(),
                "reason": r.reason,
                "metadata": r.metadata,
            }
            for r in records
        ]

    def get_feature_requirements(self, feature: str) -> List[AcknowledgmentType]:
        """
        Get all acknowledgments required for a feature.

        Args:
            feature: Feature name

        Returns:
            List of required acknowledgment types
        """
        try:
            category = FeatureCategory(feature)
            return self._feature_requirements.get(category, []).copy()
        except ValueError:
            return [AcknowledgmentType.AI_SYSTEM_AWARENESS]

    def _get_user_acknowledged_types(self, user_id: str) -> Set[AcknowledgmentType]:
        """Get types user has already acknowledged and are still valid."""
        if user_id not in self.acknowledgments:
            return set()
        return {a.acknowledgment_type for a in self.acknowledgments[user_id] if a.is_valid()}

    def _log_action(
        self, acknowledgment_id: str, user_id: str, action: str, reason: Optional[str] = None
    ) -> None:
        """Log action to audit trail."""
        record = AcknowledgmentAuditRecord(
            record_id=hashlib.sha256(
                f"{acknowledgment_id}:{action}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:16],
            acknowledgment_id=acknowledgment_id,
            user_id=user_id,
            action=action,
            timestamp=datetime.utcnow(),
            reason=reason,
        )
        self.audit_log.append(record)


def create_acknowledgment_manager() -> UserAcknowledgmentManager:
    """
    Factory function to create UserAcknowledgmentManager.

    Returns:
        Configured UserAcknowledgmentManager instance
    """
    return UserAcknowledgmentManager()


def get_acknowledgment_texts() -> Dict[AcknowledgmentType, Dict[str, str]]:
    """
    Get all acknowledgment texts.

    Returns:
        Dictionary of acknowledgment type to language to text mapping
    """
    return ACKNOWLEDGMENT_TEXTS.copy()


def get_feature_requirements() -> Dict[FeatureCategory, List[AcknowledgmentType]]:
    """
    Get all feature requirements.

    Returns:
        Dictionary of feature to required acknowledgments
    """
    return FEATURE_REQUIREMENTS.copy()


def validate_acknowledgment(ack: UserAcknowledgment) -> Dict[str, bool]:
    """
    Validate an acknowledgment record.

    Args:
        ack: Acknowledgment to validate

    Returns:
        Dictionary with validation results
    """
    return {
        "has_id": bool(ack.acknowledgment_id),
        "has_user_id": bool(ack.user_id),
        "has_type": ack.acknowledgment_type is not None,
        "has_timestamp": ack.timestamp is not None,
        "has_text_hash": bool(ack.text_hash),
        "is_valid_status": ack.status in AcknowledgmentStatus,
    }


def get_acknowledgment_summary(user_id: str, manager: UserAcknowledgmentManager) -> Dict[str, Any]:
    """
    Get summary of user's acknowledgment status.

    Args:
        user_id: User identifier
        manager: Acknowledgment manager instance

    Returns:
        Summary dictionary
    """
    acks = manager.get_user_acknowledgments(user_id)
    acknowledged_types = {a.acknowledgment_type for a in acks if a.is_valid()}

    return {
        "user_id": user_id,
        "total_acknowledged": len(acknowledged_types),
        "total_required": len(AcknowledgmentType),
        "acknowledged_types": [t.value for t in acknowledged_types],
        "pending_types": [t.value for t in AcknowledgmentType if t not in acknowledged_types],
        "can_access_live_trading": manager.check_feature_access(user_id, "live_trading")[0],
    }

# -*- coding: utf-8 -*-
"""
Tests for User AI Acknowledgment System.

This module provides comprehensive tests for the user acknowledgment
functionality required by Article 50 of the EU AI Act.

Coverage includes:
- UserAcknowledgment dataclass
- UserAcknowledgmentManager functionality
- Enums and data structures
- Feature access control
- Audit trail functionality
- Multi-language support
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List

from services.ai_act.user_acknowledgment import (
    # Enums
    AcknowledgmentType,
    AcknowledgmentStatus,
    FeatureCategory,
    # Data structures
    UserAcknowledgment,
    AcknowledgmentAuditRecord,
    # Constants
    ACKNOWLEDGMENT_TEXTS,
    FEATURE_REQUIREMENTS,
    # Main class
    UserAcknowledgmentManager,
    # Factory functions
    create_acknowledgment_manager,
    get_acknowledgment_texts,
    get_feature_requirements,
    validate_acknowledgment,
    get_acknowledgment_summary,
)


class TestAcknowledgmentType:
    """Test AcknowledgmentType enum."""

    def test_all_types_defined(self):
        """Test all acknowledgment types are defined."""
        types = [
            AcknowledgmentType.AI_SYSTEM_AWARENESS,
            AcknowledgmentType.RISK_UNDERSTANDING,
            AcknowledgmentType.LIMITATION_ACCEPTANCE,
            AcknowledgmentType.LIVE_TRADING_CONSENT,
            AcknowledgmentType.DATA_PROCESSING_CONSENT,
            AcknowledgmentType.PERFORMANCE_DISCLAIMER,
        ]
        assert len(types) == 6
        for t in types:
            assert t is not None

    def test_type_values(self):
        """Test acknowledgment type values."""
        assert AcknowledgmentType.AI_SYSTEM_AWARENESS.value == "ai_system_awareness"
        assert AcknowledgmentType.LIVE_TRADING_CONSENT.value == "live_trading_consent"


class TestAcknowledgmentStatus:
    """Test AcknowledgmentStatus enum."""

    def test_all_statuses_defined(self):
        """Test all statuses are defined."""
        statuses = [
            AcknowledgmentStatus.PENDING,
            AcknowledgmentStatus.ACKNOWLEDGED,
            AcknowledgmentStatus.EXPIRED,
            AcknowledgmentStatus.REVOKED,
        ]
        assert len(statuses) == 4

    def test_status_values(self):
        """Test status values."""
        assert AcknowledgmentStatus.ACKNOWLEDGED.value == "acknowledged"
        assert AcknowledgmentStatus.REVOKED.value == "revoked"


class TestFeatureCategory:
    """Test FeatureCategory enum."""

    def test_all_categories_defined(self):
        """Test all feature categories are defined."""
        categories = [
            FeatureCategory.REGISTRATION,
            FeatureCategory.STRATEGY_CREATION,
            FeatureCategory.BACKTESTING,
            FeatureCategory.PAPER_TRADING,
            FeatureCategory.LIVE_TRADING,
            FeatureCategory.API_ACCESS,
        ]
        assert len(categories) == 6


class TestUserAcknowledgment:
    """Test UserAcknowledgment dataclass."""

    def test_create_acknowledgment(self):
        """Test creating acknowledgment with factory method."""
        ack = UserAcknowledgment.create(
            user_id="user1",
            ack_type=AcknowledgmentType.AI_SYSTEM_AWARENESS,
            text_content="Test text"
        )
        assert ack.user_id == "user1"
        assert ack.acknowledgment_type == AcknowledgmentType.AI_SYSTEM_AWARENESS
        assert ack.status == AcknowledgmentStatus.ACKNOWLEDGED
        assert ack.text_hash != ""
        assert len(ack.acknowledgment_id) > 0

    def test_acknowledgment_with_metadata(self):
        """Test acknowledgment with IP and user agent."""
        ack = UserAcknowledgment.create(
            user_id="user1",
            ack_type=AcknowledgmentType.AI_SYSTEM_AWARENESS,
            text_content="Test",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        assert ack.ip_address == "192.168.1.1"
        assert ack.user_agent == "Mozilla/5.0"

    def test_acknowledgment_to_dict(self):
        """Test acknowledgment serialization."""
        ack = UserAcknowledgment.create(
            user_id="user1",
            ack_type=AcknowledgmentType.RISK_UNDERSTANDING,
            text_content="Test"
        )
        data = ack.to_dict()
        assert data["user_id"] == "user1"
        assert data["acknowledgment_type"] == "risk_understanding"
        assert data["status"] == "acknowledged"
        assert "timestamp" in data

    def test_acknowledgment_from_dict(self):
        """Test acknowledgment deserialization."""
        ack = UserAcknowledgment.create(
            user_id="user1",
            ack_type=AcknowledgmentType.AI_SYSTEM_AWARENESS,
            text_content="Test"
        )
        data = ack.to_dict()
        restored = UserAcknowledgment.from_dict(data)
        assert restored.user_id == ack.user_id
        assert restored.acknowledgment_type == ack.acknowledgment_type
        assert restored.text_hash == ack.text_hash

    def test_acknowledgment_is_valid(self):
        """Test is_valid method."""
        ack = UserAcknowledgment.create(
            user_id="user1",
            ack_type=AcknowledgmentType.AI_SYSTEM_AWARENESS,
            text_content="Test"
        )
        assert ack.is_valid() is True

        ack.status = AcknowledgmentStatus.REVOKED
        assert ack.is_valid() is False


class TestAcknowledgmentTexts:
    """Test acknowledgment texts."""

    def test_all_types_have_texts(self):
        """Test all acknowledgment types have texts."""
        for ack_type in AcknowledgmentType:
            assert ack_type in ACKNOWLEDGMENT_TEXTS

    def test_texts_have_english(self):
        """Test all texts have English version."""
        for ack_type, texts in ACKNOWLEDGMENT_TEXTS.items():
            assert "en" in texts
            assert len(texts["en"]) > 0

    def test_texts_have_russian(self):
        """Test all texts have Russian version."""
        for ack_type, texts in ACKNOWLEDGMENT_TEXTS.items():
            assert "ru" in texts
            assert len(texts["ru"]) > 0

    def test_ai_awareness_text_mentions_ai(self):
        """Test AI awareness text mentions AI."""
        text = ACKNOWLEDGMENT_TEXTS[AcknowledgmentType.AI_SYSTEM_AWARENESS]["en"]
        assert "AI" in text or "Artificial Intelligence" in text

    def test_risk_text_mentions_losses(self):
        """Test risk understanding text mentions losses."""
        text = ACKNOWLEDGMENT_TEXTS[AcknowledgmentType.RISK_UNDERSTANDING]["en"]
        assert "loss" in text.lower()

    def test_live_trading_text_comprehensive(self):
        """Test live trading text is comprehensive."""
        text = ACKNOWLEDGMENT_TEXTS[AcknowledgmentType.LIVE_TRADING_CONSENT]["en"]
        assert "responsibility" in text.lower()
        assert "AI" in text or "AI-powered" in text
        assert "loss" in text.lower()


class TestFeatureRequirements:
    """Test feature requirements mapping."""

    def test_registration_requires_ai_awareness(self):
        """Test registration requires AI awareness."""
        reqs = FEATURE_REQUIREMENTS[FeatureCategory.REGISTRATION]
        assert AcknowledgmentType.AI_SYSTEM_AWARENESS in reqs

    def test_live_trading_requires_all_key_acknowledgments(self):
        """Test live trading requires comprehensive acknowledgments."""
        reqs = FEATURE_REQUIREMENTS[FeatureCategory.LIVE_TRADING]
        assert AcknowledgmentType.AI_SYSTEM_AWARENESS in reqs
        assert AcknowledgmentType.RISK_UNDERSTANDING in reqs
        assert AcknowledgmentType.LIVE_TRADING_CONSENT in reqs
        assert AcknowledgmentType.LIMITATION_ACCEPTANCE in reqs

    def test_api_access_requires_data_consent(self):
        """Test API access requires data processing consent."""
        reqs = FEATURE_REQUIREMENTS[FeatureCategory.API_ACCESS]
        assert AcknowledgmentType.DATA_PROCESSING_CONSENT in reqs

    def test_all_features_have_ai_awareness(self):
        """Test all features require AI awareness."""
        for category, reqs in FEATURE_REQUIREMENTS.items():
            assert AcknowledgmentType.AI_SYSTEM_AWARENESS in reqs


class TestUserAcknowledgmentManager:
    """Test UserAcknowledgmentManager."""

    @pytest.fixture
    def manager(self) -> UserAcknowledgmentManager:
        """Create a manager instance."""
        return create_acknowledgment_manager()

    def test_create_manager(self, manager):
        """Test manager creation."""
        assert isinstance(manager, UserAcknowledgmentManager)

    def test_get_required_acknowledgments_new_user(self, manager):
        """Test getting requirements for new user."""
        required = manager.get_required_acknowledgments("new_user", "registration")
        assert AcknowledgmentType.AI_SYSTEM_AWARENESS in required

    def test_get_required_acknowledgments_live_trading(self, manager):
        """Test getting requirements for live trading."""
        required = manager.get_required_acknowledgments("user1", "live_trading")
        assert len(required) == 4
        assert AcknowledgmentType.LIVE_TRADING_CONSENT in required

    def test_record_acknowledgment(self, manager):
        """Test recording acknowledgment."""
        ack = manager.record_acknowledgment(
            user_id="user1",
            ack_type=AcknowledgmentType.AI_SYSTEM_AWARENESS
        )
        assert ack is not None
        assert ack.user_id == "user1"
        assert ack.acknowledgment_type == AcknowledgmentType.AI_SYSTEM_AWARENESS

    def test_record_acknowledgment_with_metadata(self, manager):
        """Test recording with IP and user agent."""
        ack = manager.record_acknowledgment(
            user_id="user1",
            ack_type=AcknowledgmentType.AI_SYSTEM_AWARENESS,
            ip_address="10.0.0.1",
            user_agent="TestClient/1.0"
        )
        assert ack.ip_address == "10.0.0.1"
        assert ack.user_agent == "TestClient/1.0"

    def test_acknowledged_not_required_again(self, manager):
        """Test acknowledged type not required again."""
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        required = manager.get_required_acknowledgments("user1", "registration")
        assert AcknowledgmentType.AI_SYSTEM_AWARENESS not in required

    def test_check_feature_access_denied(self, manager):
        """Test feature access denied without acknowledgments."""
        can_access, missing = manager.check_feature_access("user1", "live_trading")
        assert can_access is False
        assert len(missing) > 0

    def test_check_feature_access_granted(self, manager):
        """Test feature access granted with all acknowledgments."""
        for ack_type in FEATURE_REQUIREMENTS[FeatureCategory.LIVE_TRADING]:
            manager.record_acknowledgment("user1", ack_type)

        can_access, missing = manager.check_feature_access("user1", "live_trading")
        assert can_access is True
        assert len(missing) == 0

    def test_check_registration_access(self, manager):
        """Test registration access control."""
        can_access, missing = manager.check_feature_access("user1", "registration")
        assert can_access is False

        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        can_access, missing = manager.check_feature_access("user1", "registration")
        assert can_access is True

    def test_get_acknowledgment_text(self, manager):
        """Test getting acknowledgment text."""
        text = manager.get_acknowledgment_text(AcknowledgmentType.AI_SYSTEM_AWARENESS, "en")
        assert len(text) > 0
        assert "AI" in text

    def test_get_acknowledgment_text_russian(self, manager):
        """Test getting Russian acknowledgment text."""
        text = manager.get_acknowledgment_text(AcknowledgmentType.AI_SYSTEM_AWARENESS, "ru")
        assert len(text) > 0
        # Check for Russian content
        assert "ИИ" in text or "искусственного интеллекта" in text

    def test_get_all_acknowledgment_texts(self, manager):
        """Test getting all acknowledgment texts."""
        texts = manager.get_all_acknowledgment_texts("en")
        assert len(texts) == len(AcknowledgmentType)
        for ack_type in AcknowledgmentType:
            assert ack_type.value in texts

    def test_get_user_acknowledgments(self, manager):
        """Test getting user acknowledgments."""
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        manager.record_acknowledgment("user1", AcknowledgmentType.RISK_UNDERSTANDING)

        acks = manager.get_user_acknowledgments("user1")
        assert len(acks) == 2

    def test_get_user_acknowledgments_filtered(self, manager):
        """Test filtering user acknowledgments by type."""
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        manager.record_acknowledgment("user1", AcknowledgmentType.RISK_UNDERSTANDING)

        acks = manager.get_user_acknowledgments(
            "user1",
            ack_type=AcknowledgmentType.AI_SYSTEM_AWARENESS
        )
        assert len(acks) == 1
        assert acks[0].acknowledgment_type == AcknowledgmentType.AI_SYSTEM_AWARENESS

    def test_revoke_acknowledgment(self, manager):
        """Test revoking acknowledgment."""
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)

        result = manager.revoke_acknowledgment(
            "user1",
            AcknowledgmentType.AI_SYSTEM_AWARENESS,
            "User requested revocation"
        )
        assert result is True

        # Check it's now required again
        required = manager.get_required_acknowledgments("user1", "registration")
        assert AcknowledgmentType.AI_SYSTEM_AWARENESS in required

    def test_revoke_nonexistent(self, manager):
        """Test revoking nonexistent acknowledgment."""
        result = manager.revoke_acknowledgment(
            "user1",
            AcknowledgmentType.AI_SYSTEM_AWARENESS,
            "Test"
        )
        assert result is False

    def test_verify_compliance(self, manager):
        """Test compliance verification."""
        result = manager.verify_compliance("new_user")
        assert "user_id" in result
        assert "features" in result
        assert result["features"]["registration"]["can_access"] is False

    def test_verify_compliance_with_acknowledgments(self, manager):
        """Test compliance after acknowledgments."""
        for ack_type in AcknowledgmentType:
            manager.record_acknowledgment("user1", ack_type)

        result = manager.verify_compliance("user1")
        assert result["features"]["live_trading"]["can_access"] is True
        assert result["total_acknowledgments"] == len(AcknowledgmentType)

    def test_get_feature_requirements(self, manager):
        """Test getting feature requirements."""
        reqs = manager.get_feature_requirements("live_trading")
        assert len(reqs) == 4
        assert AcknowledgmentType.LIVE_TRADING_CONSENT in reqs

    def test_unknown_feature_requires_ai_awareness(self, manager):
        """Test unknown feature defaults to AI awareness."""
        reqs = manager.get_required_acknowledgments("user1", "unknown_feature")
        assert AcknowledgmentType.AI_SYSTEM_AWARENESS in reqs


class TestAuditTrail:
    """Test audit trail functionality."""

    @pytest.fixture
    def manager(self) -> UserAcknowledgmentManager:
        """Create a manager instance."""
        return create_acknowledgment_manager()

    def test_audit_trail_on_create(self, manager):
        """Test audit trail records creation."""
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        trail = manager.get_audit_trail(user_id="user1")
        assert len(trail) == 1
        assert trail[0]["action"] == "created"

    def test_audit_trail_on_revoke(self, manager):
        """Test audit trail records revocation."""
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        manager.revoke_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS, "Test")

        trail = manager.get_audit_trail(user_id="user1")
        actions = [r["action"] for r in trail]
        assert "created" in actions
        assert "revoked" in actions

    def test_audit_trail_filtering_by_user(self, manager):
        """Test audit trail filtering by user."""
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        manager.record_acknowledgment("user2", AcknowledgmentType.AI_SYSTEM_AWARENESS)

        trail = manager.get_audit_trail(user_id="user1")
        assert all(r["user_id"] == "user1" for r in trail)

    def test_audit_trail_has_timestamp(self, manager):
        """Test audit records have timestamps."""
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        trail = manager.get_audit_trail(user_id="user1")
        assert "timestamp" in trail[0]


class TestFactoryFunctions:
    """Test factory and utility functions."""

    def test_create_acknowledgment_manager(self):
        """Test factory function."""
        manager = create_acknowledgment_manager()
        assert isinstance(manager, UserAcknowledgmentManager)

    def test_get_acknowledgment_texts(self):
        """Test getting acknowledgment texts."""
        texts = get_acknowledgment_texts()
        assert len(texts) == len(AcknowledgmentType)

    def test_get_feature_requirements_func(self):
        """Test getting feature requirements."""
        reqs = get_feature_requirements()
        assert len(reqs) == len(FeatureCategory)

    def test_validate_acknowledgment(self):
        """Test acknowledgment validation."""
        ack = UserAcknowledgment.create(
            user_id="user1",
            ack_type=AcknowledgmentType.AI_SYSTEM_AWARENESS,
            text_content="Test"
        )
        result = validate_acknowledgment(ack)
        assert result["has_id"] is True
        assert result["has_user_id"] is True
        assert result["has_type"] is True
        assert result["has_text_hash"] is True

    def test_get_acknowledgment_summary(self):
        """Test getting acknowledgment summary."""
        manager = create_acknowledgment_manager()
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)

        summary = get_acknowledgment_summary("user1", manager)
        assert summary["user_id"] == "user1"
        assert summary["total_acknowledged"] == 1
        assert "ai_system_awareness" in summary["acknowledged_types"]


class TestMultipleUsers:
    """Test multiple user scenarios."""

    @pytest.fixture
    def manager(self) -> UserAcknowledgmentManager:
        """Create a manager instance."""
        return create_acknowledgment_manager()

    def test_separate_user_acknowledgments(self, manager):
        """Test users have separate acknowledgments."""
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        manager.record_acknowledgment("user2", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        manager.record_acknowledgment("user2", AcknowledgmentType.RISK_UNDERSTANDING)

        user1_acks = manager.get_user_acknowledgments("user1")
        user2_acks = manager.get_user_acknowledgments("user2")

        assert len(user1_acks) == 1
        assert len(user2_acks) == 2

    def test_user_compliance_independent(self, manager):
        """Test user compliance is independent."""
        # User 1 gets all acknowledgments
        for ack_type in AcknowledgmentType:
            manager.record_acknowledgment("user1", ack_type)

        # User 2 has none
        user1_can, _ = manager.check_feature_access("user1", "live_trading")
        user2_can, _ = manager.check_feature_access("user2", "live_trading")

        assert user1_can is True
        assert user2_can is False


class TestArticle50Compliance:
    """Integration tests for Article 50 compliance."""

    @pytest.fixture
    def manager(self) -> UserAcknowledgmentManager:
        """Create a manager instance."""
        return create_acknowledgment_manager()

    def test_full_acknowledgment_flow(self, manager):
        """Test complete acknowledgment flow."""
        # Step 1: Check what's required
        required = manager.get_required_acknowledgments("user1", "live_trading")
        assert len(required) > 0

        # Step 2: Show texts and record acknowledgments
        for ack_type in required:
            text = manager.get_acknowledgment_text(ack_type)
            assert len(text) > 0
            manager.record_acknowledgment("user1", ack_type)

        # Step 3: Verify access
        can_access, missing = manager.check_feature_access("user1", "live_trading")
        assert can_access is True
        assert len(missing) == 0

    def test_progressive_access_levels(self, manager):
        """Test progressive feature access."""
        # Registration only needs AI awareness
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        reg_access, _ = manager.check_feature_access("user1", "registration")
        assert reg_access is True

        # Live trading still blocked
        live_access, _ = manager.check_feature_access("user1", "live_trading")
        assert live_access is False

        # Add remaining acknowledgments
        manager.record_acknowledgment("user1", AcknowledgmentType.RISK_UNDERSTANDING)
        manager.record_acknowledgment("user1", AcknowledgmentType.LIMITATION_ACCEPTANCE)
        manager.record_acknowledgment("user1", AcknowledgmentType.LIVE_TRADING_CONSENT)

        live_access, _ = manager.check_feature_access("user1", "live_trading")
        assert live_access is True

    def test_acknowledgment_mentions_article_50(self):
        """Test that relevant acknowledgments reference Article 50."""
        text = ACKNOWLEDGMENT_TEXTS[AcknowledgmentType.AI_SYSTEM_AWARENESS]["en"]
        assert "Article 50" in text

    def test_audit_trail_completeness(self, manager):
        """Test audit trail captures all actions."""
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        manager.record_acknowledgment("user1", AcknowledgmentType.RISK_UNDERSTANDING)

        trail = manager.get_audit_trail(user_id="user1")
        assert len(trail) == 2

        for record in trail:
            assert "record_id" in record
            assert "timestamp" in record
            assert "action" in record


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def manager(self) -> UserAcknowledgmentManager:
        """Create a manager instance."""
        return create_acknowledgment_manager()

    def test_empty_user_id(self, manager):
        """Test with empty user ID."""
        ack = manager.record_acknowledgment("", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        assert ack.user_id == ""

    def test_special_characters_in_user_id(self, manager):
        """Test with special characters in user ID."""
        user_id = "user@domain.com"
        ack = manager.record_acknowledgment(user_id, AcknowledgmentType.AI_SYSTEM_AWARENESS)
        assert ack.user_id == user_id

    def test_duplicate_acknowledgment(self, manager):
        """Test recording same acknowledgment twice."""
        ack1 = manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        ack2 = manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)

        # Both should be recorded
        acks = manager.get_user_acknowledgments("user1")
        assert len(acks) == 2

        # But feature check should pass with just one
        can_access, _ = manager.check_feature_access("user1", "registration")
        assert can_access is True

    def test_get_acknowledgments_nonexistent_user(self, manager):
        """Test getting acknowledgments for nonexistent user."""
        acks = manager.get_user_acknowledgments("nonexistent")
        assert len(acks) == 0

    def test_verify_compliance_nonexistent_user(self, manager):
        """Test compliance verification for nonexistent user."""
        result = manager.verify_compliance("nonexistent")
        assert result["total_acknowledgments"] == 0

    def test_revoke_wrong_type(self, manager):
        """Test revoking wrong acknowledgment type."""
        manager.record_acknowledgment("user1", AcknowledgmentType.AI_SYSTEM_AWARENESS)
        result = manager.revoke_acknowledgment(
            "user1",
            AcknowledgmentType.RISK_UNDERSTANDING,  # Not recorded
            "Test"
        )
        assert result is False


class TestLanguageSupport:
    """Test multi-language support."""

    @pytest.fixture
    def manager(self) -> UserAcknowledgmentManager:
        """Create a manager instance."""
        return create_acknowledgment_manager()

    def test_english_texts(self, manager):
        """Test English texts are available."""
        for ack_type in AcknowledgmentType:
            text = manager.get_acknowledgment_text(ack_type, "en")
            assert len(text) > 0

    def test_russian_texts(self, manager):
        """Test Russian texts are available."""
        for ack_type in AcknowledgmentType:
            text = manager.get_acknowledgment_text(ack_type, "ru")
            assert len(text) > 0

    def test_record_acknowledgment_with_language(self, manager):
        """Test recording acknowledgment specifies language."""
        ack = manager.record_acknowledgment(
            "user1",
            AcknowledgmentType.AI_SYSTEM_AWARENESS,
            language="ru"
        )
        # Text hash should be based on Russian text
        assert ack.text_hash != ""

# -*- coding: utf-8 -*-
"""
Tests for Article 50 EU AI Act - Transparency Disclosure System.

This module provides comprehensive tests for the transparency disclosure
functionality required by Article 50 of the EU AI Act.

Coverage includes:
- AIDisclosure dataclass
- TransparencyDisclosureManager
- SyntheticContentMarker
- Disclosure requirements validation
- Multi-language support
- Audit trail functionality
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from services.ai_act.transparency_disclosure import (
    # Enums
    DisclosureType,
    DisclosureContext,
    DisclosureLanguage,
    # Data structures
    AIDisclosure,
    DisclosureRequirement,
    DisclosureAuditRecord,
    # Constants
    DISCLOSURE_REQUIREMENTS,
    # Main classes
    TransparencyDisclosureManager,
    SyntheticContentMarker,
    # Factory functions
    create_transparency_manager,
    get_disclosure_requirements,
    validate_disclosure_text,
)


class TestDisclosureType:
    """Test DisclosureType enum."""

    def test_all_types_defined(self):
        """Test all required disclosure types are defined."""
        assert DisclosureType.AI_INTERACTION is not None
        assert DisclosureType.SYNTHETIC_CONTENT is not None
        assert DisclosureType.EMOTION_RECOGNITION is not None
        assert DisclosureType.DEEP_FAKE is not None

    def test_type_values(self):
        """Test disclosure type values."""
        assert DisclosureType.AI_INTERACTION.value == "ai_interaction"
        assert DisclosureType.SYNTHETIC_CONTENT.value == "synthetic_content"


class TestDisclosureContext:
    """Test DisclosureContext enum."""

    def test_all_contexts_defined(self):
        """Test all contexts are defined."""
        contexts = [
            DisclosureContext.REGISTRATION,
            DisclosureContext.STRATEGY_CREATION,
            DisclosureContext.LIVE_TRADING_ACTIVATION,
            DisclosureContext.MODEL_OUTPUT,
            DisclosureContext.API_RESPONSE,
            DisclosureContext.DASHBOARD,
            DisclosureContext.REPORT_GENERATION,
        ]
        for context in contexts:
            assert context is not None

    def test_context_values(self):
        """Test context values are strings."""
        for context in DisclosureContext:
            assert isinstance(context.value, str)


class TestAIDisclosure:
    """Test AIDisclosure dataclass."""

    @pytest.fixture
    def sample_disclosure(self) -> AIDisclosure:
        """Create a sample disclosure for testing."""
        return AIDisclosure(
            disclosure_id="test123",
            disclosure_type=DisclosureType.AI_INTERACTION,
            context=DisclosureContext.REGISTRATION,
            timestamp=datetime.utcnow(),
            user_id="user1",
        )

    def test_disclosure_creation(self, sample_disclosure):
        """Test basic disclosure creation."""
        assert sample_disclosure.disclosure_id == "test123"
        assert sample_disclosure.disclosure_type == DisclosureType.AI_INTERACTION
        assert sample_disclosure.context == DisclosureContext.REGISTRATION
        assert sample_disclosure.user_id == "user1"
        assert sample_disclosure.acknowledged is False
        assert sample_disclosure.acknowledgment_timestamp is None

    def test_disclosure_default_values(self, sample_disclosure):
        """Test default values are set correctly."""
        assert sample_disclosure.ai_system_name == "Distributional PPO Trading Model"
        assert sample_disclosure.ai_system_type == "Reinforcement Learning"
        assert "Trading" in sample_disclosure.ai_purpose
        assert sample_disclosure.version == "1.0"

    def test_disclosure_text_english(self, sample_disclosure):
        """Test English disclosure text generation."""
        text = sample_disclosure.generate_disclosure_text("en")
        assert "AI SYSTEM DISCLOSURE" in text["header"]
        assert "Artificial Intelligence" in text["body"]
        assert "machine learning" in text["body"]
        assert "Reinforcement Learning" in text["body"]
        assert (
            "understand" in text["acknowledge"].lower()
            or "acknowledge" in text["acknowledge"].lower()
        )

    def test_disclosure_text_russian(self, sample_disclosure):
        """Test Russian disclosure text generation."""
        text = sample_disclosure.generate_disclosure_text("ru")
        assert "ИИ-СИСТЕМЕ" in text["header"]
        assert "искусственного интеллекта" in text["body"]
        assert "машинного обучения" in text["body"]

    def test_disclosure_text_german(self, sample_disclosure):
        """Test German disclosure text generation."""
        text = sample_disclosure.generate_disclosure_text("de")
        assert "KI-SYSTEM" in text["header"]
        assert "Künstlichen Intelligenz" in text["body"]

    def test_disclosure_text_french(self, sample_disclosure):
        """Test French disclosure text generation."""
        text = sample_disclosure.generate_disclosure_text("fr")
        assert "SYSTEME" in text["header"]
        assert "Intelligence Artificielle" in text["body"]

    def test_disclosure_text_dutch(self, sample_disclosure):
        """Test Dutch disclosure text generation."""
        text = sample_disclosure.generate_disclosure_text("nl")
        assert "AI-SYSTEEM" in text["header"]
        assert "Kunstmatige Intelligentie" in text["body"]

    def test_disclosure_text_fallback_to_english(self, sample_disclosure):
        """Test unknown language falls back to English."""
        text = sample_disclosure.generate_disclosure_text("unknown")
        assert "AI SYSTEM DISCLOSURE" in text["header"]

    def test_disclosure_to_dict(self, sample_disclosure):
        """Test serialization to dictionary."""
        data = sample_disclosure.to_dict()
        assert data["disclosure_id"] == "test123"
        assert data["disclosure_type"] == "ai_interaction"
        assert data["context"] == "registration"
        assert data["user_id"] == "user1"
        assert data["acknowledged"] is False

    def test_disclosure_from_dict(self, sample_disclosure):
        """Test deserialization from dictionary."""
        data = sample_disclosure.to_dict()
        restored = AIDisclosure.from_dict(data)
        assert restored.disclosure_id == sample_disclosure.disclosure_id
        assert restored.disclosure_type == sample_disclosure.disclosure_type
        assert restored.context == sample_disclosure.context
        assert restored.user_id == sample_disclosure.user_id


class TestDisclosureRequirement:
    """Test DisclosureRequirement dataclass."""

    def test_requirement_creation(self):
        """Test creating a disclosure requirement."""
        req = DisclosureRequirement(
            context=DisclosureContext.REGISTRATION,
            required=True,
            reason="User begins interaction",
            article_reference="Article 50(1)",
        )
        assert req.context == DisclosureContext.REGISTRATION
        assert req.required is True
        assert "50" in req.article_reference

    def test_blocking_default(self):
        """Test blocking default is True."""
        req = DisclosureRequirement(
            context=DisclosureContext.REGISTRATION,
            required=True,
            reason="Test",
            article_reference="Article 50",
        )
        assert req.blocking is True


class TestDisclosureRequirements:
    """Test disclosure requirements configuration."""

    def test_registration_requires_disclosure(self):
        """Test registration requires AI disclosure."""
        reg_req = next(
            (r for r in DISCLOSURE_REQUIREMENTS if r.context == DisclosureContext.REGISTRATION),
            None,
        )
        assert reg_req is not None
        assert reg_req.required is True
        assert "Article 50" in reg_req.article_reference

    def test_live_trading_requires_disclosure(self):
        """Test live trading requires disclosure."""
        lt_req = next(
            (
                r
                for r in DISCLOSURE_REQUIREMENTS
                if r.context == DisclosureContext.LIVE_TRADING_ACTIVATION
            ),
            None,
        )
        assert lt_req is not None
        assert lt_req.required is True
        assert lt_req.blocking is True

    def test_api_response_requires_disclosure(self):
        """Test API response requires disclosure."""
        api_req = next(
            (r for r in DISCLOSURE_REQUIREMENTS if r.context == DisclosureContext.API_RESPONSE),
            None,
        )
        assert api_req is not None
        assert api_req.required is True

    def test_all_requirements_have_references(self):
        """Test all requirements reference Article 50."""
        for req in DISCLOSURE_REQUIREMENTS:
            assert "Article 50" in req.article_reference

    def test_all_requirements_have_reasons(self):
        """Test all requirements have reasons."""
        for req in DISCLOSURE_REQUIREMENTS:
            assert len(req.reason) > 0


class TestTransparencyDisclosureManager:
    """Test TransparencyDisclosureManager."""

    @pytest.fixture
    def manager(self) -> TransparencyDisclosureManager:
        """Create a manager instance."""
        return create_transparency_manager()

    def test_create_disclosure(self, manager):
        """Test disclosure creation."""
        disclosure = manager.create_disclosure(
            user_id="user1", context=DisclosureContext.REGISTRATION
        )
        assert disclosure is not None
        assert disclosure.user_id == "user1"
        assert disclosure.context == DisclosureContext.REGISTRATION
        assert disclosure.acknowledged is False

    def test_create_disclosure_with_metadata(self, manager):
        """Test disclosure creation with IP and user agent."""
        disclosure = manager.create_disclosure(
            user_id="user1",
            context=DisclosureContext.REGISTRATION,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )
        assert disclosure.ip_address == "192.168.1.1"
        assert disclosure.user_agent == "Mozilla/5.0"

    def test_record_acknowledgment(self, manager):
        """Test acknowledgment recording."""
        disclosure = manager.create_disclosure(
            user_id="user1", context=DisclosureContext.REGISTRATION
        )
        result = manager.record_acknowledgment(disclosure.disclosure_id, "user1")
        assert result is True
        assert disclosure.acknowledged is True
        assert disclosure.acknowledgment_timestamp is not None

    def test_acknowledgment_wrong_user(self, manager):
        """Test acknowledgment by wrong user fails."""
        disclosure = manager.create_disclosure(
            user_id="user1", context=DisclosureContext.REGISTRATION
        )
        result = manager.record_acknowledgment(disclosure.disclosure_id, "user2")
        assert result is False
        assert disclosure.acknowledged is False

    def test_acknowledgment_invalid_disclosure(self, manager):
        """Test acknowledgment with invalid disclosure ID."""
        result = manager.record_acknowledgment("invalid_id", "user1")
        assert result is False

    def test_check_disclosure_required_new_user(self, manager):
        """Test disclosure required for new user."""
        required = manager.check_disclosure_required("new_user", DisclosureContext.REGISTRATION)
        assert required is True

    def test_check_disclosure_required_after_creation(self, manager):
        """Test disclosure required after creation but before ack."""
        manager.create_disclosure("user1", DisclosureContext.REGISTRATION)
        required = manager.check_disclosure_required("user1", DisclosureContext.REGISTRATION)
        assert required is True

    def test_check_disclosure_not_required_after_ack(self, manager):
        """Test disclosure not required after acknowledgment."""
        disclosure = manager.create_disclosure(
            user_id="user1", context=DisclosureContext.REGISTRATION
        )
        manager.record_acknowledgment(disclosure.disclosure_id, "user1")
        required = manager.check_disclosure_required("user1", DisclosureContext.REGISTRATION)
        assert required is False

    def test_get_user_disclosures(self, manager):
        """Test getting user disclosures."""
        manager.create_disclosure("user1", DisclosureContext.REGISTRATION)
        manager.create_disclosure("user1", DisclosureContext.LIVE_TRADING_ACTIVATION)

        disclosures = manager.get_user_disclosures("user1")
        assert len(disclosures) == 2

    def test_get_user_disclosures_by_context(self, manager):
        """Test filtering disclosures by context."""
        manager.create_disclosure("user1", DisclosureContext.REGISTRATION)
        manager.create_disclosure("user1", DisclosureContext.LIVE_TRADING_ACTIVATION)

        disclosures = manager.get_user_disclosures("user1", context=DisclosureContext.REGISTRATION)
        assert len(disclosures) == 1
        assert disclosures[0].context == DisclosureContext.REGISTRATION

    def test_api_headers(self, manager):
        """Test API headers contain AI disclosure."""
        headers = manager.get_api_headers()
        assert headers["X-AI-System"] == "true"
        assert "AI" in headers["X-AI-Disclosure"]
        assert "X-AI-Model" in headers
        assert "X-AI-Act-Compliance" in headers

    def test_api_response_metadata(self, manager):
        """Test API response metadata."""
        metadata = manager.get_api_response_metadata()
        assert "_ai_disclosure" in metadata
        assert metadata["_ai_disclosure"]["is_ai_generated"] is True
        assert "EU AI Act" in metadata["_ai_disclosure"]["compliance"]

    def test_verify_compliance_new_user(self, manager):
        """Test compliance verification for new user."""
        result = manager.verify_compliance("new_user")
        assert result["article_50_compliant"] is False

    def test_verify_compliance_after_all_acks(self, manager):
        """Test compliance after all acknowledgments."""
        for req in DISCLOSURE_REQUIREMENTS:
            if req.required:
                disclosure = manager.create_disclosure("user1", req.context)
                manager.record_acknowledgment(disclosure.disclosure_id, "user1")

        result = manager.verify_compliance("user1")
        assert result["article_50_compliant"] is True

    def test_audit_trail(self, manager):
        """Test audit trail is maintained."""
        disclosure = manager.create_disclosure("user1", DisclosureContext.REGISTRATION)
        manager.record_acknowledgment(disclosure.disclosure_id, "user1")

        trail = manager.get_audit_trail(user_id="user1")
        assert len(trail) >= 2  # created and acknowledged

    def test_audit_trail_filtering(self, manager):
        """Test audit trail filtering."""
        manager.create_disclosure("user1", DisclosureContext.REGISTRATION)
        manager.create_disclosure("user2", DisclosureContext.REGISTRATION)

        trail = manager.get_audit_trail(user_id="user1")
        assert all(r["user_id"] == "user1" for r in trail)


class TestSyntheticContentMarker:
    """Test SyntheticContentMarker."""

    def test_mark_content(self):
        """Test marking content as AI-generated."""
        content = "This is some analysis."
        marked = SyntheticContentMarker.mark_content(content)
        assert marked.startswith("[AI-GENERATED]")
        assert "This is some analysis" in marked

    def test_mark_content_with_footer(self):
        """Test content marking includes footer by default."""
        content = "Analysis text."
        marked = SyntheticContentMarker.mark_content(content, include_footer=True)
        assert "Article 50(2)" in marked

    def test_mark_content_without_footer(self):
        """Test content marking without footer."""
        content = "Analysis text."
        marked = SyntheticContentMarker.mark_content(content, include_footer=False)
        assert "Article 50(2)" not in marked

    def test_add_metadata(self):
        """Test adding metadata to dictionary."""
        data = {"key": "value"}
        result = SyntheticContentMarker.add_metadata(data)
        assert "_synthetic_content" in result
        assert result["_synthetic_content"]["is_ai_generated"] is True
        assert "Article 50(2)" in result["_synthetic_content"]["compliance"]

    def test_is_marked_true(self):
        """Test detecting marked content."""
        content = "[AI-GENERATED]\nSome content"
        assert SyntheticContentMarker.is_marked(content) is True

    def test_is_marked_false(self):
        """Test detecting unmarked content."""
        content = "Some content without marker"
        assert SyntheticContentMarker.is_marked(content) is False


class TestFactoryFunctions:
    """Test factory and utility functions."""

    def test_create_transparency_manager(self):
        """Test factory function creates manager."""
        manager = create_transparency_manager()
        assert isinstance(manager, TransparencyDisclosureManager)

    def test_create_transparency_manager_with_storage(self):
        """Test factory with custom storage."""
        storage = {"custom": "storage"}
        manager = create_transparency_manager(storage_backend=storage)
        assert manager.storage == storage

    def test_get_disclosure_requirements(self):
        """Test getting disclosure requirements."""
        requirements = get_disclosure_requirements()
        assert len(requirements) > 0
        assert all(isinstance(r, DisclosureRequirement) for r in requirements)


class TestValidateDisclosureText:
    """Test disclosure text validation."""

    @pytest.fixture
    def disclosure(self) -> AIDisclosure:
        """Create disclosure for testing."""
        return AIDisclosure(
            disclosure_id="test",
            disclosure_type=DisclosureType.AI_INTERACTION,
            context=DisclosureContext.REGISTRATION,
            timestamp=datetime.utcnow(),
            user_id="user1",
        )

    def test_validate_english_text(self, disclosure):
        """Test validation of English text."""
        result = validate_disclosure_text(disclosure, "en")
        assert result["has_header"] is True
        assert result["has_body"] is True
        assert result["has_acknowledge"] is True
        assert result["mentions_ai"] is True
        assert result["mentions_ml"] is True
        assert result["all_valid"] is True

    def test_validate_russian_text(self, disclosure):
        """Test validation of Russian text."""
        result = validate_disclosure_text(disclosure, "ru")
        assert result["has_header"] is True
        assert result["has_body"] is True
        # Russian uses "ИИ" instead of "AI", validation checks for both
        assert result["has_acknowledge"] is True

    def test_validate_all_languages(self, disclosure):
        """Test validation passes for all supported languages."""
        languages = ["en", "ru", "de", "fr", "nl"]
        for lang in languages:
            result = validate_disclosure_text(disclosure, lang)
            assert result["has_header"] is True, f"Failed for {lang}"
            assert result["has_body"] is True, f"Failed for {lang}"


class TestArticle50Compliance:
    """Integration tests for Article 50 compliance."""

    @pytest.fixture
    def manager(self) -> TransparencyDisclosureManager:
        """Create manager for testing."""
        return create_transparency_manager()

    def test_full_disclosure_flow(self, manager):
        """Test complete disclosure flow: create -> show -> acknowledge."""
        # Step 1: Check disclosure required
        assert manager.check_disclosure_required("user1", DisclosureContext.REGISTRATION)

        # Step 2: Create disclosure
        disclosure = manager.create_disclosure("user1", DisclosureContext.REGISTRATION)

        # Step 3: Get text to show user
        text = disclosure.generate_disclosure_text("en")
        assert "AI" in text["body"]

        # Step 4: User acknowledges
        manager.record_acknowledgment(disclosure.disclosure_id, "user1")

        # Step 5: Disclosure no longer required
        assert not manager.check_disclosure_required("user1", DisclosureContext.REGISTRATION)

    def test_multiple_contexts_independent(self, manager):
        """Test disclosures for different contexts are independent."""
        # Acknowledge registration
        d1 = manager.create_disclosure("user1", DisclosureContext.REGISTRATION)
        manager.record_acknowledgment(d1.disclosure_id, "user1")

        # Live trading still requires disclosure
        assert manager.check_disclosure_required("user1", DisclosureContext.LIVE_TRADING_ACTIVATION)

    def test_live_trading_requires_all_prior_disclosures(self, manager):
        """Test live trading flow requires prior disclosures."""
        # Registration first
        reg = manager.create_disclosure("user1", DisclosureContext.REGISTRATION)
        manager.record_acknowledgment(reg.disclosure_id, "user1")

        # Then live trading
        lt = manager.create_disclosure("user1", DisclosureContext.LIVE_TRADING_ACTIVATION)
        manager.record_acknowledgment(lt.disclosure_id, "user1")

        # Both should be acknowledged
        assert not manager.check_disclosure_required("user1", DisclosureContext.REGISTRATION)
        assert not manager.check_disclosure_required(
            "user1", DisclosureContext.LIVE_TRADING_ACTIVATION
        )

    def test_api_integration_headers(self, manager):
        """Test API headers for Article 50 compliance."""
        headers = manager.get_api_headers()

        # Required headers per Article 50
        assert "X-AI-System" in headers
        assert "X-AI-Disclosure" in headers

        # Values should be informative
        assert headers["X-AI-System"] == "true"
        assert len(headers["X-AI-Disclosure"]) > 10

    def test_audit_trail_completeness(self, manager):
        """Test audit trail captures all actions."""
        # Perform actions
        disclosure = manager.create_disclosure("user1", DisclosureContext.REGISTRATION)
        manager.record_acknowledgment(disclosure.disclosure_id, "user1")

        # Get audit trail
        trail = manager.get_audit_trail(user_id="user1")

        # Should have both create and acknowledge
        actions = [r["action"] for r in trail]
        assert "created" in actions
        assert "acknowledged" in actions


class TestMultipleUsers:
    """Test multiple user scenarios."""

    @pytest.fixture
    def manager(self) -> TransparencyDisclosureManager:
        """Create manager for testing."""
        return create_transparency_manager()

    def test_separate_user_disclosures(self, manager):
        """Test users have separate disclosures."""
        manager.create_disclosure("user1", DisclosureContext.REGISTRATION)
        manager.create_disclosure("user2", DisclosureContext.REGISTRATION)

        user1_disclosures = manager.get_user_disclosures("user1")
        user2_disclosures = manager.get_user_disclosures("user2")

        assert len(user1_disclosures) == 1
        assert len(user2_disclosures) == 1
        assert user1_disclosures[0].user_id == "user1"
        assert user2_disclosures[0].user_id == "user2"

    def test_user_cannot_acknowledge_other_user(self, manager):
        """Test user cannot acknowledge another user's disclosure."""
        disclosure = manager.create_disclosure("user1", DisclosureContext.REGISTRATION)

        # User2 tries to acknowledge user1's disclosure
        result = manager.record_acknowledgment(disclosure.disclosure_id, "user2")
        assert result is False

        # Original disclosure unchanged
        assert disclosure.acknowledged is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def manager(self) -> TransparencyDisclosureManager:
        """Create manager for testing."""
        return create_transparency_manager()

    def test_empty_user_id(self, manager):
        """Test with empty user ID."""
        disclosure = manager.create_disclosure("", DisclosureContext.REGISTRATION)
        assert disclosure.user_id == ""

    def test_special_characters_in_user_id(self, manager):
        """Test with special characters in user ID."""
        user_id = "user@domain.com"
        disclosure = manager.create_disclosure(user_id, DisclosureContext.REGISTRATION)
        assert disclosure.user_id == user_id

    def test_duplicate_disclosure_same_context(self, manager):
        """Test creating duplicate disclosure for same context."""
        d1 = manager.create_disclosure("user1", DisclosureContext.REGISTRATION)
        d2 = manager.create_disclosure("user1", DisclosureContext.REGISTRATION)

        # Should have same ID (deterministic)
        assert d1.disclosure_id == d2.disclosure_id

    def test_verify_compliance_empty_user(self, manager):
        """Test compliance verification for user with no disclosures."""
        result = manager.verify_compliance("nonexistent_user")
        assert result["article_50_compliant"] is False
        assert "contexts" in result

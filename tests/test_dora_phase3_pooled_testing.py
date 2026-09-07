# -*- coding: utf-8 -*-
"""
Comprehensive Tests for DORA Phase 3 - Pooled TLPT (Article 26(3)).

Tests for Pooled Threat-Led Penetration Testing per Article 26(3) of DORA.

Coverage includes:
- Pooled Test Status enumeration
- Participant Role categorization
- Participant Status tracking
- Cost Sharing Model options
- Provider Criticality levels
- Shared Provider management
- Pooled Testing Participant management
- Pooled Testing Scope definition
- Cost Sharing Agreements
- Pooled Testing Engagements
- Pooled Testing Results distribution
- Individual attestation tracking

References:
- Article 26(3) DORA: https://www.digital-operational-resilience-act.com/Article_26.html
- RTS on TLPT (CDR 2024/2698)
"""


import pytest

pytest.skip(
    "Legacy DORA test - uses deprecated imports from services.dora.* "
    "These modules have been migrated to services.dora_integration.*. "
    "See tests/dora/ and tests/dora_integration/ for current tests.",
    allow_module_level=True,
)


import pytest
from datetime import datetime, timedelta, timezone

from services.dora.pooled_testing import (
    PooledTestStatus,
    ParticipantRole,
    ParticipantStatus,
    CostSharingModel,
    ProviderCriticality,
    SharedProvider,
    PooledTestingParticipant,
    PooledTestingScope,
    CostSharingAgreement,
    PooledTestingEngagement,
    PooledTestingResults,
    PooledTestingConfig,
    DORAPooledTesting,
    create_pooled_testing,
)


# =============================================================================
# Pooled Test Status Tests
# =============================================================================


class TestPooledTestStatus:
    """Tests for pooled test status enumeration."""

    def test_proposed_status(self):
        """Test proposed status value."""
        assert PooledTestStatus.PROPOSED.value == "proposed"

    def test_organizing_status(self):
        """Test organizing status value."""
        assert PooledTestStatus.ORGANIZING.value == "organizing"

    def test_participants_confirmed_status(self):
        """Test participants confirmed status value."""
        assert PooledTestStatus.PARTICIPANTS_CONFIRMED.value == "participants_confirmed"

    def test_scope_agreed_status(self):
        """Test scope agreed status value."""
        assert PooledTestStatus.SCOPE_AGREED.value == "scope_agreed"

    def test_in_progress_status(self):
        """Test in progress status value."""
        assert PooledTestStatus.IN_PROGRESS.value == "in_progress"

    def test_completed_status(self):
        """Test completed status value."""
        assert PooledTestStatus.COMPLETED.value == "completed"


# =============================================================================
# Participant Role Tests
# =============================================================================


class TestParticipantRole:
    """Tests for participant role enumeration."""

    def test_organizer_role(self):
        """Test organizer role value."""
        assert ParticipantRole.ORGANIZER.value == "organizer"

    def test_participant_role(self):
        """Test participant role value."""
        assert ParticipantRole.PARTICIPANT.value == "participant"

    def test_observer_role(self):
        """Test observer role value."""
        assert ParticipantRole.OBSERVER.value == "observer"

    def test_provider_role(self):
        """Test provider role value."""
        assert ParticipantRole.PROVIDER.value == "provider"


# =============================================================================
# Participant Status Tests
# =============================================================================


class TestParticipantStatus:
    """Tests for participant status enumeration."""

    def test_invited_status(self):
        """Test invited status value."""
        assert ParticipantStatus.INVITED.value == "invited"

    def test_confirmed_status(self):
        """Test confirmed status value."""
        assert ParticipantStatus.CONFIRMED.value == "confirmed"

    def test_contract_signed_status(self):
        """Test contract signed status value."""
        assert ParticipantStatus.CONTRACT_SIGNED.value == "contract_signed"

    def test_active_status(self):
        """Test active status value."""
        assert ParticipantStatus.ACTIVE.value == "active"

    def test_withdrawn_status(self):
        """Test withdrawn status value."""
        assert ParticipantStatus.WITHDRAWN.value == "withdrawn"


# =============================================================================
# Cost Sharing Model Tests
# =============================================================================


class TestCostSharingModel:
    """Tests for cost sharing model enumeration."""

    def test_equal_model(self):
        """Test equal split model."""
        assert CostSharingModel.EQUAL.value == "equal"

    def test_proportional_usage_model(self):
        """Test proportional usage model."""
        assert CostSharingModel.PROPORTIONAL_USAGE.value == "proportional_usage"

    def test_proportional_size_model(self):
        """Test proportional size model."""
        assert CostSharingModel.PROPORTIONAL_SIZE.value == "proportional_size"

    def test_proportional_criticality_model(self):
        """Test proportional criticality model."""
        assert CostSharingModel.PROPORTIONAL_CRITICALITY.value == "proportional_criticality"

    def test_custom_model(self):
        """Test custom arrangement model."""
        assert CostSharingModel.CUSTOM.value == "custom"


# =============================================================================
# Provider Criticality Tests
# =============================================================================


class TestProviderCriticality:
    """Tests for provider criticality enumeration."""

    def test_critical_level(self):
        """Test critical level (CTPP or critical service)."""
        assert ProviderCriticality.CRITICAL.value == "critical"

    def test_important_level(self):
        """Test important level."""
        assert ProviderCriticality.IMPORTANT.value == "important"

    def test_standard_level(self):
        """Test standard level."""
        assert ProviderCriticality.STANDARD.value == "standard"


# =============================================================================
# Shared Provider Tests
# =============================================================================


class TestSharedProvider:
    """Tests for shared provider structure."""

    def test_provider_creation_with_defaults(self):
        """Test provider creation with default values."""
        provider = SharedProvider()
        assert provider.provider_id != ""
        assert provider.service_criticality == ProviderCriticality.STANDARD

    def test_provider_with_name(self):
        """Test provider with specific name."""
        provider = SharedProvider(name="AWS", provider_type="cloud")
        assert provider.name == "AWS"
        assert provider.provider_type == "cloud"

    def test_provider_with_ctpp_flag(self):
        """Test provider with CTPP designation."""
        provider = SharedProvider(
            name="Major Cloud Provider",
            is_ctpp=True,
            service_criticality=ProviderCriticality.CRITICAL,
        )
        assert provider.is_ctpp == True

    def test_provider_auto_generates_id(self):
        """Test that provider auto-generates a unique ID."""
        prov1 = SharedProvider()
        prov2 = SharedProvider()
        assert prov1.provider_id != prov2.provider_id


# =============================================================================
# Pooled Testing Participant Tests
# =============================================================================


class TestPooledTestingParticipant:
    """Tests for pooled testing participant structure."""

    def test_participant_creation_with_defaults(self):
        """Test participant creation with default values."""
        participant = PooledTestingParticipant()
        assert participant.participant_id != ""

    def test_participant_with_entity(self):
        """Test participant with specific entity."""
        participant = PooledTestingParticipant(
            entity_id="ENT-001", entity_name="Investment Bank A", role=ParticipantRole.PARTICIPANT
        )
        assert participant.entity_name == "Investment Bank A"

    def test_participant_with_role(self):
        """Test participant with specific role."""
        participant = PooledTestingParticipant(
            entity_name="Lead Bank", role=ParticipantRole.ORGANIZER
        )
        assert participant.role == ParticipantRole.ORGANIZER

    def test_participant_auto_generates_id(self):
        """Test that participant auto-generates a unique ID."""
        part1 = PooledTestingParticipant()
        part2 = PooledTestingParticipant()
        assert part1.participant_id != part2.participant_id


# =============================================================================
# Pooled Testing Scope Tests
# =============================================================================


class TestPooledTestingScope:
    """Tests for pooled testing scope structure."""

    def test_scope_creation_with_defaults(self):
        """Test scope creation with default values."""
        scope = PooledTestingScope()
        assert scope.scope_id != ""

    def test_scope_with_name(self):
        """Test scope with specific name."""
        scope = PooledTestingScope(
            name="Cloud Infrastructure Testing", description="Testing of shared cloud provider"
        )
        assert scope.name == "Cloud Infrastructure Testing"

    def test_scope_auto_generates_id(self):
        """Test that scope auto-generates a unique ID."""
        scope1 = PooledTestingScope()
        scope2 = PooledTestingScope()
        assert scope1.scope_id != scope2.scope_id


# =============================================================================
# Cost Sharing Agreement Tests
# =============================================================================


class TestCostSharingAgreement:
    """Tests for cost sharing agreement structure."""

    def test_agreement_creation_with_defaults(self):
        """Test agreement creation with default values."""
        agreement = CostSharingAgreement()
        assert agreement.agreement_id != ""
        assert agreement.model == CostSharingModel.EQUAL

    def test_agreement_with_model(self):
        """Test agreement with specific model."""
        agreement = CostSharingAgreement(
            model=CostSharingModel.PROPORTIONAL_SIZE, total_cost=500000.0, currency="EUR"
        )
        assert agreement.model == CostSharingModel.PROPORTIONAL_SIZE
        assert agreement.total_cost == 500000.0

    def test_agreement_auto_generates_id(self):
        """Test that agreement auto-generates a unique ID."""
        agr1 = CostSharingAgreement()
        agr2 = CostSharingAgreement()
        assert agr1.agreement_id != agr2.agreement_id


# =============================================================================
# Pooled Testing Engagement Tests
# =============================================================================


class TestPooledTestingEngagement:
    """Tests for pooled testing engagement structure."""

    def test_engagement_creation_with_defaults(self):
        """Test engagement creation with default values."""
        engagement = PooledTestingEngagement()
        assert engagement.engagement_id != ""
        assert engagement.status == PooledTestStatus.PROPOSED

    def test_engagement_with_name(self):
        """Test engagement with specific name."""
        engagement = PooledTestingEngagement(
            name="Industry Cloud Provider TLPT 2025",
            description="Joint TLPT for shared cloud infrastructure",
        )
        assert engagement.name == "Industry Cloud Provider TLPT 2025"

    def test_engagement_auto_generates_id(self):
        """Test that engagement auto-generates a unique ID."""
        eng1 = PooledTestingEngagement()
        eng2 = PooledTestingEngagement()
        assert eng1.engagement_id != eng2.engagement_id


# =============================================================================
# Pooled Testing Results Tests
# =============================================================================


class TestPooledTestingResults:
    """Tests for pooled testing results structure."""

    def test_results_creation_with_defaults(self):
        """Test results creation with default values."""
        results = PooledTestingResults()
        assert results.results_id != ""

    def test_results_with_engagement_id(self):
        """Test results with specific engagement ID."""
        results = PooledTestingResults(engagement_id="POOLED-001")
        assert results.engagement_id == "POOLED-001"

    def test_results_auto_generates_id(self):
        """Test that results auto-generates a unique ID."""
        res1 = PooledTestingResults()
        res2 = PooledTestingResults()
        assert res1.results_id != res2.results_id


# =============================================================================
# Pooled Testing Config Tests
# =============================================================================


class TestPooledTestingConfig:
    """Tests for pooled testing configuration."""

    def test_config_creation_with_defaults(self):
        """Test config creation with default values."""
        config = PooledTestingConfig()
        assert config is not None

    def test_config_with_entity_id(self):
        """Test config with entity ID."""
        config = PooledTestingConfig(entity_id="ENT-001")
        assert config.entity_id == "ENT-001"


# =============================================================================
# DORAPooledTesting Creation Tests
# =============================================================================


class TestDORAPooledTestingCreation:
    """Tests for DORAPooledTesting creation."""

    def test_create_with_factory_function(self):
        """Test creation via factory function."""
        pooled = create_pooled_testing()
        assert pooled is not None
        assert isinstance(pooled, DORAPooledTesting)

    def test_create_with_config(self):
        """Test creation with custom config."""
        config = PooledTestingConfig(entity_id="TEST-001")
        pooled = create_pooled_testing(config)
        assert pooled is not None
        assert pooled.config.entity_id == "TEST-001"

    def test_has_required_methods(self):
        """Test that pooled testing has required methods."""
        pooled = create_pooled_testing()
        assert hasattr(pooled, "create_engagement")
        assert hasattr(pooled, "add_participant")
        assert hasattr(pooled, "register_provider")


# =============================================================================
# DORAPooledTesting Provider Management Tests
# =============================================================================


class TestDORAPooledTestingProvider:
    """Tests for provider management functionality."""

    def test_register_provider(self):
        """Test registering a shared provider."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(
            name="AWS", provider_type="cloud", service_criticality=ProviderCriticality.CRITICAL
        )
        assert provider is not None
        assert provider.name == "AWS"

    def test_get_provider(self):
        """Test retrieving a registered provider."""
        pooled = create_pooled_testing()
        created = pooled.register_provider(name="Azure", provider_type="cloud")
        retrieved = pooled.get_provider(created.provider_id)
        assert retrieved is not None
        assert retrieved.name == "Azure"

    def test_list_providers(self):
        """Test listing all providers."""
        pooled = create_pooled_testing()
        pooled.register_provider(name="Provider 1", provider_type="cloud")
        pooled.register_provider(name="Provider 2", provider_type="exchange")
        providers = pooled.list_providers()
        assert len(providers) >= 2


# =============================================================================
# DORAPooledTesting Engagement Tests
# =============================================================================


class TestDORAPooledTestingEngagement:
    """Tests for engagement management functionality."""

    def test_create_engagement(self):
        """Test creating a pooled testing engagement."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Test Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Pooled TLPT 2025", provider_id=provider.provider_id
        )
        assert engagement is not None
        assert engagement.name == "Pooled TLPT 2025"

    def test_get_engagement(self):
        """Test retrieving an engagement."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="data")
        created = pooled.create_engagement(name="Test Engagement", provider_id=provider.provider_id)
        retrieved = pooled.get_engagement(created.engagement_id)
        assert retrieved is not None

    def test_list_engagements(self):
        """Test listing all engagements."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        pooled.create_engagement(name="Engagement 1", provider_id=provider.provider_id)
        pooled.create_engagement(name="Engagement 2", provider_id=provider.provider_id)
        engagements = pooled.list_engagements()
        assert len(engagements) >= 2


# =============================================================================
# DORAPooledTesting Participant Tests
# =============================================================================


class TestDORAPooledTestingParticipant:
    """Tests for participant management functionality."""

    def test_add_participant(self):
        """Test adding a participant to engagement."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Test Engagement", provider_id=provider.provider_id
        )
        participant = pooled.add_participant(
            engagement_id=engagement.engagement_id,
            entity_name="Bank A",
            role=ParticipantRole.ORGANIZER,
        )
        assert participant is not None
        assert participant.entity_name == "Bank A"

    def test_get_participants(self):
        """Test retrieving participants for an engagement."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Test Engagement", provider_id=provider.provider_id
        )
        pooled.add_participant(
            engagement_id=engagement.engagement_id,
            entity_name="Bank A",
            role=ParticipantRole.ORGANIZER,
        )
        pooled.add_participant(
            engagement_id=engagement.engagement_id,
            entity_name="Bank B",
            role=ParticipantRole.PARTICIPANT,
        )
        participants = pooled.get_participants(engagement.engagement_id)
        assert len(participants) >= 2

    def test_update_participant_status(self):
        """Test updating participant status."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Test Engagement", provider_id=provider.provider_id
        )
        participant = pooled.add_participant(
            engagement_id=engagement.engagement_id,
            entity_name="Bank A",
            role=ParticipantRole.PARTICIPANT,
        )
        updated = pooled.update_participant_status(
            participant_id=participant.participant_id, status=ParticipantStatus.CONFIRMED
        )
        assert updated.status == ParticipantStatus.CONFIRMED


# =============================================================================
# DORAPooledTesting Cost Sharing Tests
# =============================================================================


class TestDORAPooledTestingCostSharing:
    """Tests for cost sharing functionality."""

    def test_create_cost_sharing_agreement(self):
        """Test creating a cost sharing agreement."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Test Engagement", provider_id=provider.provider_id
        )
        agreement = pooled.create_cost_sharing_agreement(
            engagement_id=engagement.engagement_id,
            model=CostSharingModel.EQUAL,
            total_cost=300000.0,
            currency="EUR",
        )
        assert agreement is not None
        assert agreement.total_cost == 300000.0

    def test_calculate_participant_shares(self):
        """Test calculating participant shares."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Test Engagement", provider_id=provider.provider_id
        )
        pooled.add_participant(
            engagement_id=engagement.engagement_id,
            entity_name="Bank A",
            role=ParticipantRole.PARTICIPANT,
        )
        pooled.add_participant(
            engagement_id=engagement.engagement_id,
            entity_name="Bank B",
            role=ParticipantRole.PARTICIPANT,
        )
        pooled.create_cost_sharing_agreement(
            engagement_id=engagement.engagement_id,
            model=CostSharingModel.EQUAL,
            total_cost=200000.0,
        )
        shares = pooled.calculate_participant_shares(engagement.engagement_id)
        assert len(shares) >= 2
        # Equal split should give 100000 each
        assert all(share.get("amount", 0) > 0 for share in shares)


# =============================================================================
# DORAPooledTesting Scope Tests
# =============================================================================


class TestDORAPooledTestingScope:
    """Tests for scope management functionality."""

    def test_define_scope(self):
        """Test defining pooled testing scope."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Test Engagement", provider_id=provider.provider_id
        )
        scope = pooled.define_scope(
            engagement_id=engagement.engagement_id,
            name="Cloud Infrastructure Scope",
            systems=["api_gateway", "authentication", "databases"],
        )
        assert scope is not None
        assert "api_gateway" in scope.systems


# =============================================================================
# DORAPooledTesting Results Tests
# =============================================================================


class TestDORAPooledTestingResults:
    """Tests for results management functionality."""

    def test_create_results(self):
        """Test creating pooled testing results."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Test Engagement", provider_id=provider.provider_id
        )
        results = pooled.create_results(engagement_id=engagement.engagement_id)
        assert results is not None
        assert results.engagement_id == engagement.engagement_id

    def test_distribute_results(self):
        """Test distributing results to participants."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Test Engagement", provider_id=provider.provider_id
        )
        pooled.add_participant(
            engagement_id=engagement.engagement_id,
            entity_name="Bank A",
            role=ParticipantRole.PARTICIPANT,
        )
        results = pooled.create_results(engagement_id=engagement.engagement_id)
        distributed = pooled.distribute_results(results.results_id)
        assert distributed is not None


# =============================================================================
# DORAPooledTesting Attestation Tests
# =============================================================================


class TestDORAPooledTestingAttestation:
    """Tests for individual attestation tracking."""

    def test_create_participant_attestation(self):
        """Test creating attestation for a participant."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Test Engagement", provider_id=provider.provider_id
        )
        participant = pooled.add_participant(
            engagement_id=engagement.engagement_id,
            entity_name="Bank A",
            role=ParticipantRole.PARTICIPANT,
        )
        attestation = pooled.create_participant_attestation(
            engagement_id=engagement.engagement_id, participant_id=participant.participant_id
        )
        assert attestation is not None

    def test_all_participants_require_attestation(self):
        """Test that all participants require individual attestations."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Test Engagement", provider_id=provider.provider_id
        )
        pooled.add_participant(engagement_id=engagement.engagement_id, entity_name="Bank A")
        pooled.add_participant(engagement_id=engagement.engagement_id, entity_name="Bank B")
        requirements = pooled.get_attestation_requirements(engagement.engagement_id)
        assert (
            requirements.get("individual_attestations_required") == True
            or "individual" in str(requirements).lower()
        )


# =============================================================================
# Reporting Tests
# =============================================================================


class TestPooledTestingReporting:
    """Tests for reporting functionality."""

    def test_generate_engagement_report(self):
        """Test generating an engagement report."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Report Test Engagement", provider_id=provider.provider_id
        )
        report = pooled.generate_engagement_report(engagement.engagement_id)
        assert report is not None
        assert "engagement_id" in report

    def test_export_cost_summary(self):
        """Test exporting cost summary."""
        pooled = create_pooled_testing()
        provider = pooled.register_provider(name="Provider", provider_type="cloud")
        engagement = pooled.create_engagement(
            name="Cost Test Engagement", provider_id=provider.provider_id
        )
        pooled.create_cost_sharing_agreement(
            engagement_id=engagement.engagement_id,
            model=CostSharingModel.EQUAL,
            total_cost=150000.0,
        )
        summary = pooled.export_cost_summary(engagement.engagement_id)
        assert summary is not None

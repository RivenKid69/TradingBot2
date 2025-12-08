# -*- coding: utf-8 -*-
"""
DORA Phase 0: Proportionality Assessment Tests.

Comprehensive test suite for DORA Phase 0 implementation:
- Scope Verification (Article 2)
- Function Classification (Article 3(22))
- Proportionality Assessment (Articles 4, 16)

Test Coverage Target: 100%
"""

import pytest
from datetime import datetime

# Import DORA Phase 0 modules
from services.dora.scope_verification import (
    DORAEntityType,
    DORAScopeResult,
    ScopeVerification,
    EntityAuthorization,
    DORAScope,
    create_scope_verifier,
    get_entity_type_description,
    ENTITY_TYPE_INFO,
)

from services.dora.function_classification import (
    FunctionCriticality,
    ImpairmentType,
    FunctionClassification,
    ICTService,
    ThirdPartyProvider,
    FunctionClassifier,
    create_function_classifier,
    get_platform_functions,
    get_ict_providers,
)

from services.dora.proportionality import (
    DORARegime,
    ExemptionType,
    EntityClassification,
    ProportionalityAssessment,
    RegimeExemption,
    ProportionalityAssessor,
    create_proportionality_assessor,
    assess_entity_proportionality,
    MICROENTERPRISE_EXEMPTIONS,
)


# =============================================================================
# SCOPE VERIFICATION TESTS (Article 2)
# =============================================================================

class TestDORAScopeVerification:
    """Tests for DORA Article 2 scope verification."""

    def test_create_scope_verifier(self):
        """Test factory function creates DORAScope instance."""
        verifier = create_scope_verifier()
        assert isinstance(verifier, DORAScope)

    def test_investment_firm_in_scope(self):
        """Test investment firm is correctly identified as in scope."""
        verifier = DORAScope()
        result = verifier.verify_scope(
            entity_type=DORAEntityType.INVESTMENT_FIRM,
        )

        assert result.result == DORAScopeResult.IN_SCOPE
        assert result.dora_article_reference == "Article 2(1)(e)"
        assert result.confidence_level == "medium"  # No authorization provided
        assert len(result.applicable_articles) > 0
        assert len(result.key_requirements) > 0

    def test_investment_firm_with_authorization(self):
        """Test investment firm with authorization has high confidence."""
        verifier = DORAScope()
        authorization = EntityAuthorization(
            legal_name="Test Trading Ltd",
            lei="549300TESTLEI123456",
            authorization_type="MiFID II",
            authorizing_nca="BaFin",
            member_state="DE",
        )

        result = verifier.verify_authorization(authorization)

        assert result.result == DORAScopeResult.IN_SCOPE
        assert result.confidence_level == "high"
        assert result.entity_type == DORAEntityType.INVESTMENT_FIRM

    def test_crypto_asset_service_provider_in_scope(self):
        """Test crypto-asset service provider is in scope."""
        verifier = DORAScope()
        result = verifier.verify_scope(
            entity_type=DORAEntityType.CRYPTO_ASSET_SERVICE_PROVIDER,
        )

        assert result.result == DORAScopeResult.IN_SCOPE
        assert result.dora_article_reference == "Article 2(1)(f)"

    def test_not_financial_entity_out_of_scope(self):
        """Test non-financial entity is out of scope."""
        verifier = DORAScope()
        result = verifier.verify_scope(
            entity_type=DORAEntityType.NOT_A_FINANCIAL_ENTITY,
        )

        assert result.result == DORAScopeResult.OUT_OF_SCOPE
        assert len(result.warnings) > 0  # Should warn about ICT provider possibility

    def test_unclear_entity_type_requires_legal_review(self):
        """Test unclear entity type requires legal review."""
        verifier = DORAScope()
        result = verifier.verify_scope(
            entity_type=DORAEntityType.UNCLEAR,
        )

        assert result.result == DORAScopeResult.UNCLEAR
        assert result.legal_review_required is True
        assert result.confidence_level == "low"

    def test_ict_third_party_provider_has_special_warnings(self):
        """Test ICT third-party service provider has oversight warnings."""
        verifier = DORAScope()
        result = verifier.verify_scope(
            entity_type=DORAEntityType.ICT_THIRD_PARTY_SERVICE_PROVIDER,
        )

        assert result.result == DORAScopeResult.IN_SCOPE
        assert any("oversight" in w.lower() or "ctpp" in w.lower() for w in result.warnings)

    def test_get_entity_type_for_algo_trading(self):
        """Test default entity type for algo trading is investment firm."""
        verifier = DORAScope()
        entity_type = verifier.get_entity_type_for_algo_trading()

        assert entity_type == DORAEntityType.INVESTMENT_FIRM

    def test_all_entity_types_have_info(self):
        """Test all entity types (except special) have reference info."""
        for entity_type in DORAEntityType:
            if entity_type not in (DORAEntityType.NOT_A_FINANCIAL_ENTITY, DORAEntityType.UNCLEAR):
                assert entity_type in ENTITY_TYPE_INFO
                info = ENTITY_TYPE_INFO[entity_type]
                assert "article" in info
                assert "description" in info

    def test_get_entity_type_description(self):
        """Test getting human-readable entity type description."""
        description = get_entity_type_description(DORAEntityType.INVESTMENT_FIRM)
        assert "investment" in description.lower()

    def test_scope_verification_has_unique_id(self):
        """Test scope verification generates unique ID."""
        verifier = DORAScope()
        result1 = verifier.verify_scope(entity_type=DORAEntityType.INVESTMENT_FIRM)
        result2 = verifier.verify_scope(entity_type=DORAEntityType.INVESTMENT_FIRM)

        assert result1.verification_id != result2.verification_id
        assert result1.verification_id.startswith("SCOPE-")


# =============================================================================
# FUNCTION CLASSIFICATION TESTS (Article 3(22))
# =============================================================================

class TestFunctionClassification:
    """Tests for DORA Article 3(22) function classification."""

    def test_create_function_classifier(self):
        """Test factory function creates FunctionClassifier instance."""
        classifier = create_function_classifier()
        assert isinstance(classifier, FunctionClassifier)

    def test_classify_critical_function(self):
        """Test function with multiple impairments is classified as critical."""
        classification = FunctionClassification(
            function_name="Order Execution",
            function_description="Trading order execution",
            impairs_financial_performance=True,
            impairs_service_soundness=True,
            impairs_regulatory_compliance=True,
        )

        assert classification.is_critical_or_important is True
        assert classification.criticality == FunctionCriticality.CRITICAL

    def test_classify_important_function(self):
        """Test function with single impairment is classified as important."""
        classification = FunctionClassification(
            function_name="Regulatory Reporting",
            function_description="Transaction reporting",
            impairs_financial_performance=False,
            impairs_service_soundness=False,
            impairs_regulatory_compliance=True,  # Only regulatory impact
        )

        assert classification.is_critical_or_important is True
        assert classification.criticality == FunctionCriticality.IMPORTANT

    def test_classify_standard_function(self):
        """Test function with no impairments is classified as standard."""
        classification = FunctionClassification(
            function_name="Backtesting",
            function_description="Strategy backtesting",
            impairs_financial_performance=False,
            impairs_service_soundness=False,
            impairs_regulatory_compliance=False,
        )

        assert classification.is_critical_or_important is False
        assert classification.criticality == FunctionCriticality.STANDARD

    def test_impairment_types_property(self):
        """Test impairment types property returns correct list."""
        classification = FunctionClassification(
            function_name="Test",
            function_description="Test function",
            impairs_financial_performance=True,
            impairs_service_soundness=True,
            impairs_regulatory_compliance=False,
        )

        impairment_types = classification.impairment_types
        assert ImpairmentType.FINANCIAL_PERFORMANCE in impairment_types
        assert ImpairmentType.SERVICE_SOUNDNESS in impairment_types
        assert ImpairmentType.REGULATORY_COMPLIANCE not in impairment_types

    def test_get_platform_functions_returns_defaults(self):
        """Test default platform functions are available."""
        functions = get_platform_functions()

        assert "order_execution" in functions
        assert "market_data" in functions
        assert "risk_monitoring" in functions
        assert "backtesting" in functions

        # Order execution should be critical
        assert functions["order_execution"].criticality == FunctionCriticality.CRITICAL
        # Backtesting should be standard
        assert functions["backtesting"].criticality == FunctionCriticality.STANDARD

    def test_get_ict_providers_returns_defaults(self):
        """Test default ICT providers are available."""
        providers = get_ict_providers()

        assert "binance" in providers
        assert "alpaca" in providers
        assert "polygon" in providers

        # Check provider structure
        binance = providers["binance"]
        assert binance.provider_name == "Binance"
        assert binance.supports_critical_function is True
        assert len(binance.services) > 0

    def test_classifier_classify_function(self):
        """Test classifier can classify functions."""
        classifier = FunctionClassifier()
        func = classifier.classify_function(
            function_name="Test Function",
            function_description="A test function",
            impairs_financial_performance=True,
            impairs_service_soundness=False,
            impairs_regulatory_compliance=False,
        )

        assert func.function_name == "Test Function"
        assert func.is_critical_or_important is True

    def test_classifier_get_critical_functions(self):
        """Test classifier can retrieve critical functions."""
        classifier = FunctionClassifier()
        classifier.load_platform_defaults()

        critical = classifier.get_critical_functions()
        assert len(critical) > 0
        for func in critical:
            assert func.criticality == FunctionCriticality.CRITICAL

    def test_classifier_get_critical_or_important_functions(self):
        """Test classifier can retrieve critical or important functions."""
        classifier = FunctionClassifier()
        classifier.load_platform_defaults()

        critical_or_important = classifier.get_critical_or_important_functions()
        assert len(critical_or_important) > 0
        for func in critical_or_important:
            assert func.is_critical_or_important is True

    def test_classifier_get_providers_for_critical_functions(self):
        """Test classifier can identify providers for critical functions."""
        classifier = FunctionClassifier()
        classifier.load_platform_defaults()

        providers = classifier.get_providers_for_critical_functions()
        assert len(providers) > 0
        # Should include major providers
        assert any("binance" in p.lower() for p in providers)

    def test_classification_summary(self):
        """Test classification summary is generated correctly."""
        classifier = FunctionClassifier()
        classifier.load_platform_defaults()

        summary = classifier.get_classification_summary()

        assert "total_functions" in summary
        assert "critical_count" in summary
        assert "important_count" in summary
        assert "standard_count" in summary
        assert summary["total_functions"] > 0

    def test_third_party_provider_dataclass(self):
        """Test ThirdPartyProvider dataclass."""
        provider = ThirdPartyProvider(
            provider_name="Test Provider",
            provider_type="exchange",
            services=["market_data", "execution"],
            supports_critical_function=True,
        )

        assert provider.provider_id.startswith("PROV-")
        assert provider.provider_name == "Test Provider"
        assert len(provider.services) == 2

    def test_ict_service_dataclass(self):
        """Test ICTService dataclass."""
        service = ICTService(
            service_name="Market Data Feed",
            service_type="market_data",
            is_critical=True,
            rto_hours=0.5,
        )

        assert service.service_id.startswith("SVC-")
        assert service.is_critical is True


# =============================================================================
# PROPORTIONALITY ASSESSMENT TESTS (Articles 4, 16)
# =============================================================================

class TestProportionalityAssessment:
    """Tests for DORA Articles 4, 16 proportionality assessment."""

    def test_create_proportionality_assessor(self):
        """Test factory function creates ProportionalityAssessor instance."""
        assessor = create_proportionality_assessor()
        assert isinstance(assessor, ProportionalityAssessor)

    def test_microenterprise_classification_correct_logic(self):
        """Test microenterprise uses OR for turnover/balance sheet, not AND."""
        # This is a critical test per plan v3.0 correction
        entity = EntityClassification(
            legal_name="Micro Test Ltd",
            employee_count=8,  # < 10
            annual_turnover_eur=3_000_000,  # > €2M
            balance_sheet_eur=1_500_000,  # < €2M
        )

        # Should qualify because: <10 employees AND (<€2M turnover OR <€2M balance)
        # Balance sheet < €2M, so should be microenterprise
        assert entity.is_microenterprise is True

    def test_microenterprise_classification_not_qualified(self):
        """Test entity that doesn't qualify as microenterprise."""
        entity = EntityClassification(
            legal_name="Not Micro Ltd",
            employee_count=8,  # < 10
            annual_turnover_eur=3_000_000,  # > €2M
            balance_sheet_eur=3_000_000,  # > €2M
        )

        # Should NOT qualify: has <10 employees but both turnover AND balance > €2M
        assert entity.is_microenterprise is False

    def test_microenterprise_too_many_employees(self):
        """Test entity with too many employees is not microenterprise."""
        entity = EntityClassification(
            legal_name="Not Micro Ltd",
            employee_count=15,  # >= 10
            annual_turnover_eur=1_000_000,  # < €2M
            balance_sheet_eur=500_000,  # < €2M
        )

        assert entity.is_microenterprise is False

    def test_small_enterprise_classification(self):
        """Test small enterprise classification."""
        entity = EntityClassification(
            employee_count=30,
            annual_turnover_eur=5_000_000,
            balance_sheet_eur=4_000_000,
        )

        assert entity.is_microenterprise is False
        assert entity.is_small_enterprise is True

    def test_medium_enterprise_classification(self):
        """Test medium enterprise classification."""
        entity = EntityClassification(
            employee_count=150,
            annual_turnover_eur=40_000_000,
            balance_sheet_eur=35_000_000,
        )

        assert entity.is_microenterprise is False
        assert entity.is_small_enterprise is False
        assert entity.is_medium_enterprise is True

    def test_full_regime_for_large_entity(self):
        """Test large entity gets full DORA regime."""
        entity = EntityClassification(
            legal_name="Large Trading Corp",
            employee_count=300,
            annual_turnover_eur=100_000_000,
            balance_sheet_eur=80_000_000,
        )

        assert entity.applicable_regime == DORARegime.FULL

    def test_microenterprise_regime_for_qualifying_entity(self):
        """Test microenterprise gets microenterprise regime."""
        entity = EntityClassification(
            legal_name="Tiny Trading Ltd",
            employee_count=5,
            annual_turnover_eur=1_000_000,
            balance_sheet_eur=800_000,
        )

        assert entity.applicable_regime == DORARegime.MICROENTERPRISE

    def test_simplified_regime_for_article_16_entity(self):
        """Test Article 16 entity gets simplified regime."""
        entity = EntityClassification(
            legal_name="Small Investment Firm",
            employee_count=50,
            annual_turnover_eur=20_000_000,
            balance_sheet_eur=15_000_000,
            is_small_non_interconnected=True,  # Article 16(1)(a)
        )

        assert entity.qualifies_for_simplified_framework is True
        assert entity.applicable_regime == DORARegime.SIMPLIFIED

    def test_assessor_full_assessment(self):
        """Test full proportionality assessment."""
        assessor = ProportionalityAssessor()
        entity = EntityClassification(
            legal_name="Test Trading Ltd",
            employee_count=8,
            annual_turnover_eur=1_500_000,
            balance_sheet_eur=1_000_000,
            entity_type="investment_firm",
        )

        assessment = assessor.assess(entity)

        assert assessment.applicable_regime == DORARegime.MICROENTERPRISE
        assert len(assessment.exemptions) > 0
        assert len(assessment.recommended_actions) > 0
        assert "proportionality_factors" in dir(assessment)

    def test_microenterprise_exemptions_present(self):
        """Test microenterprise exemptions are defined."""
        assert len(MICROENTERPRISE_EXEMPTIONS) >= 4

        # Check specific exemptions exist
        exemption_types = [e.exemption_type for e in MICROENTERPRISE_EXEMPTIONS]
        assert ExemptionType.MICROENTERPRISE_NO_TPP_STRATEGY in exemption_types
        assert ExemptionType.MICROENTERPRISE_SIMPLIFIED_RISK in exemption_types
        assert ExemptionType.MICROENTERPRISE_NO_RECURRING_INCIDENTS in exemption_types

    def test_convenience_function_assess_entity(self):
        """Test convenience function for quick assessment."""
        assessment = assess_entity_proportionality(
            legal_name="Quick Test Ltd",
            employee_count=5,
            annual_turnover_eur=500_000,
            balance_sheet_eur=300_000,
        )

        assert assessment.applicable_regime == DORARegime.MICROENTERPRISE
        assert assessment.entity_classification.legal_name == "Quick Test Ltd"

    def test_check_microenterprise_status_utility(self):
        """Test utility function for microenterprise check."""
        assessor = ProportionalityAssessor()

        # Should be microenterprise
        assert assessor.check_microenterprise_status(
            employee_count=8,
            annual_turnover_eur=1_500_000,
            balance_sheet_eur=1_000_000,
        ) is True

        # Should NOT be microenterprise
        assert assessor.check_microenterprise_status(
            employee_count=15,
            annual_turnover_eur=1_000_000,
            balance_sheet_eur=1_000_000,
        ) is False

    def test_assessment_has_unique_id(self):
        """Test assessment generates unique ID."""
        assessor = ProportionalityAssessor()
        entity = EntityClassification(
            legal_name="Test",
            employee_count=5,
            annual_turnover_eur=1_000_000,
            balance_sheet_eur=800_000,
        )

        assessment1 = assessor.assess(entity)
        assessment2 = assessor.assess(entity)

        assert assessment1.assessment_id != assessment2.assessment_id
        assert assessment1.assessment_id.startswith("PROP-")

    def test_proportionality_factors_included(self):
        """Test proportionality factors are included in assessment."""
        assessor = ProportionalityAssessor()
        entity = EntityClassification(
            legal_name="Test",
            employee_count=5,
            annual_turnover_eur=1_000_000,
            balance_sheet_eur=800_000,
            risk_profile="low",
            operational_complexity="low",
        )

        assessment = assessor.assess(entity, include_proportionality_factors=True)

        factors = assessment.proportionality_factors
        assert "size" in factors
        assert "risk_profile" in factors
        assert "complexity" in factors

    def test_requirements_categorization(self):
        """Test requirements are properly categorized by regime."""
        assessor = ProportionalityAssessor()

        # Microenterprise
        micro_entity = EntityClassification(
            legal_name="Micro",
            employee_count=5,
            annual_turnover_eur=1_000_000,
            balance_sheet_eur=800_000,
        )
        micro_assessment = assessor.assess(micro_entity)

        assert len(micro_assessment.exempted_requirements) > 0
        assert any("28(2)" in req for req in micro_assessment.exempted_requirements)

        # Full regime
        large_entity = EntityClassification(
            legal_name="Large",
            employee_count=300,
            annual_turnover_eur=100_000_000,
            balance_sheet_eur=80_000_000,
        )
        large_assessment = assessor.assess(large_entity)

        assert len(large_assessment.full_requirements) > 0
        assert len(large_assessment.exempted_requirements) == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDORAPhase0Integration:
    """Integration tests for DORA Phase 0 modules."""

    def test_complete_phase0_workflow(self):
        """Test complete Phase 0 workflow from scope to assessment."""
        # Step 1: Verify scope
        scope_verifier = create_scope_verifier()
        authorization = EntityAuthorization(
            legal_name="Integration Test Trading Ltd",
            lei="549300TESTLEI123456",
            authorization_type="MiFID II",
            authorizing_nca="BaFin",
            member_state="DE",
        )
        scope_result = scope_verifier.verify_authorization(authorization)

        assert scope_result.result == DORAScopeResult.IN_SCOPE

        # Step 2: Classify functions
        classifier = create_function_classifier()
        classifier.load_platform_defaults()

        critical_functions = classifier.get_critical_or_important_functions()
        assert len(critical_functions) > 0

        critical_providers = classifier.get_providers_for_critical_functions()
        assert len(critical_providers) > 0

        # Step 3: Assess proportionality
        entity = EntityClassification(
            legal_name="Integration Test Trading Ltd",
            lei="549300TESTLEI123456",
            employee_count=8,
            annual_turnover_eur=1_500_000,
            balance_sheet_eur=1_000_000,
            entity_type="investment_firm",
        )

        assessor = create_proportionality_assessor()
        assessment = assessor.assess(entity)

        assert assessment.applicable_regime == DORARegime.MICROENTERPRISE
        assert len(assessment.exemptions) > 0

    def test_module_imports_work(self):
        """Test all Phase 0 module imports work correctly."""
        # This tests the __init__.py exports
        from services.dora import (
            # Scope verification
            DORAEntityType,
            DORAScopeResult,
            DORAScope,
            create_scope_verifier,
            # Function classification
            FunctionCriticality,
            FunctionClassification,
            FunctionClassifier,
            create_function_classifier,
            get_platform_functions,
            # Proportionality
            DORARegime,
            EntityClassification,
            ProportionalityAssessor,
            create_proportionality_assessor,
            assess_entity_proportionality,
        )

        # Verify all imports are usable
        assert DORAEntityType.INVESTMENT_FIRM is not None
        assert DORAScopeResult.IN_SCOPE is not None
        assert FunctionCriticality.CRITICAL is not None
        assert DORARegime.FULL is not None

    def test_dora_module_version(self):
        """Test DORA module version is set."""
        from services.dora import __version__, __dora_compliance_phase__

        assert __version__ == "0.2.0"
        assert __dora_compliance_phase__ == 1


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""

    def test_entity_classification_with_zero_values(self):
        """Test entity classification handles zero values."""
        entity = EntityClassification(
            legal_name="Zero Test",
            employee_count=0,
            annual_turnover_eur=0.0,
            balance_sheet_eur=0.0,
        )

        # Should be microenterprise (0 < 10 and 0 < 2M)
        assert entity.is_microenterprise is True

    def test_entity_classification_boundary_values(self):
        """Test classification at boundary values."""
        # Exactly at microenterprise boundary
        entity = EntityClassification(
            employee_count=10,  # NOT < 10
            annual_turnover_eur=1_000_000,
            balance_sheet_eur=1_000_000,
        )

        assert entity.is_microenterprise is False

    def test_function_classification_with_empty_lists(self):
        """Test function classification with empty provider lists."""
        classification = FunctionClassification(
            function_name="Empty Test",
            function_description="Test",
            impairs_financial_performance=True,
            ict_services_supporting=[],
            third_party_providers=[],
        )

        assert classification.is_critical_or_important is True
        assert len(classification.ict_services_supporting) == 0

    def test_scope_verification_with_activities(self):
        """Test scope verification with activities list."""
        verifier = DORAScope()
        result = verifier.verify_scope(
            entity_type=DORAEntityType.INVESTMENT_FIRM,
            activities=["algorithmic_trading", "portfolio_management"],
        )

        assert result.result == DORAScopeResult.IN_SCOPE
        assert len(result.in_scope_activities) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

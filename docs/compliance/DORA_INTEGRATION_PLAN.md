# DORA Integration Plan
# Digital Operational Resilience Act (EU Regulation 2022/2554)
# План интеграции в AI-Powered Quantitative Research Platform

**Версия документа**: 4.1.0
**Дата создания**: 2025-12-08
**Последнее обновление**: 2025-12-12
**Целевое соответствие**: Regulation (EU) 2022/2554 (DORA)
**Дата вступления в силу**: 17 января 2025
**Статус проекта**: COMPLIANCE REMEDIATION (DORA уже применяется)

---

> ### ICT Provider Positioning Update (v4.1)
>
> **ВАЖНО:** Этот документ содержит как исторический контекст планирования для Financial Entities,
> так и актуальные требования для ICT Providers. Мы позиционируемся как **ICT Third-Party Service Provider (Art. 30)**.
>
> **Текущая архитектура:**
> - `services/dora_integration/` — активные модули для ICT Provider обязательств
> - `services/archive/dora_financial_entity/` — заархивированные FE-специфичные модули
> - `configs/dora/` — активные конфиги (digital_resilience_strategy, third_party_management, information_sharing)
>
> **Заархивированные FE конфиги** (перемещены в `services/archive/dora_financial_entity/configs/`):
> - `entity_classification.yaml` — классификация FE по Art. 2
> - `nca_identification.yaml` — идентификация NCA для FE
> - `proportionality_assessment.yaml` — определение режима (Art. 4, 16)
>
> Примеры кода с путями `services/dora/` в этом документе относятся к **историческим планам**.
> Актуальное расположение модулей см. в `services/archive/dora_financial_entity/README.md`.

---

## CRITICAL REVIEW v4.0 — Независимый аудит и финальные исправления

> **ВАЖНО**: Проведён независимый аудит плана v3.0 с проверкой по официальным источникам EUR-Lex и ESAs.

### Исправления v4.0 (Audit-based)

| # | Проблема | Severity | Исправление |
|---|----------|----------|-------------|
| 1 | Article 2(1)(b) для investment firm | **CRITICAL** | Исправлено на **Article 2(1)(e)** |
| 2 | "ESAs have NOT YET designated CTPPs" | **CRITICAL** | Обновлено: **19 CTPPs designated 19 Nov 2025** (AWS, Google, Microsoft, etc.) |
| 3 | JC 2024-33 для incident reporting | HIGH | Исправлено на **CDR 2025/301** + **CIR 2025/302** |
| 4 | Template prefixes RT_xx.xx | HIGH | Исправлено на **B_xx.xx** (DPM 4.0) |
| 5 | Client threshold 10,000 | HIGH | Исправлено на **100,000** (per RTS Art.9) |
| 6 | Country deadlines неточные | HIGH | Исправлено: Germany **11 Apr**, France **15 Apr** |
| 7 | Reference date отсутствует | HIGH | Добавлено: **31 March 2025** |
| 8 | Test counts несогласованы | MEDIUM | Унифицировано: **~1015 tests** |
| 9 | Weekend/Holiday extension | MEDIUM | Добавлено per CDR 2025/301 Art.4 |
| 10 | Incident upgrade procedure | MEDIUM | Добавлено per CDR 2025/301 |

---

## CRITICAL REVIEW v3.0 — Второй раунд критического анализа

> **ВАЖНО**: Выявлены дополнительные критические проблемы в плане v2.0.

### Новые критические проблемы (v3.0)

| # | Проблема | Severity | Статус v2.0 | Исправление v3.0 |
|---|----------|----------|-------------|------------------|
| 11 | **Scope verification не выполнена** | CRITICAL | ❌ Отсутствует | Добавлена проверка Article 2 scope |
| 12 | **Critical/Important function не определена** | CRITICAL | ❌ Отсутствует | Добавлена Article 3(22) classification |
| 13 | **Inconsistent test counts** | MEDIUM | ❌ Phase 1: 180 vs 250 | Унифицировано |
| 14 | **RTS JC 2023 86 controls не mapped** | HIGH | ❌ Не детализировано | Добавлен control mapping |
| 15 | **NCA identification missing** | CRITICAL | ❌ "YOUR_NCA" placeholder | Добавлена процедура идентификации |
| 16 | **Incident timeline interpretation error** | CRITICAL | ❌ Неверная логика | Исправлено: 4h OR 24h (раньше) |
| 17 | **ITS submission format not specified** | HIGH | ❌ Только dataclasses | Добавлены CSV/XML requirements |
| 18 | **LEI handling for non-EU providers** | HIGH | ❌ Не рассмотрено | Добавлена альтернативная идентификация |
| 19 | **Microenterprise definition error** | MEDIUM | ❌ AND вместо OR | Исправлено на OR |
| 20 | **Submission deadlines by country** | HIGH | ❌ Не указаны | Добавлены по странам |

---

### v3.0 Critical Fix #1: DORA Scope Verification (NEW)

**ПРОБЛЕМА**: План предполагает, что DORA применяется, но это не проверено.

**РЕШЕНИЕ**: Добавить Phase -1 (Scope Verification) до Phase 0.

Согласно [DORA Article 2](https://www.digital-operational-resilience-act.com/Article_2.html), регулирование применяется к 21 типу financial entities:

```python
# services/dora/scope_verification.py

class DORAScope:
    """
    DORA Article 2 - Scope verification.

    CRITICAL: Verify if DORA applies BEFORE any implementation.
    """

    # Article 2(1) - Entities in scope
    ENTITIES_IN_SCOPE = {
        "a": "credit_institutions",
        "b": "payment_institutions",
        "c": "account_information_service_providers",
        "d": "electronic_money_institutions",
        "e": "investment_firms",  # <-- Most likely for algo trading
        "f": "crypto_asset_service_providers",  # MiCA
        "g": "central_securities_depositories",
        "h": "central_counterparties",
        "i": "trading_venues",  # <-- If we operate a trading venue
        "j": "trade_repositories",
        "k": "managers_of_alternative_investment_funds",
        "l": "management_companies",
        "m": "data_reporting_service_providers",
        "n": "insurance_and_reinsurance_undertakings",
        "o": "insurance_intermediaries",
        "p": "institutions_for_occupational_retirement_provision",
        "q": "credit_rating_agencies",
        "r": "administrators_of_critical_benchmarks",
        "s": "crowdfunding_service_providers",
        "t": "securitisation_repositories",
        "u": "ict_third_party_service_providers",  # If we provide ICT to FIs
    }

    @classmethod
    def verify_scope(cls, entity_type: str, authorization: str) -> ScopeResult:
        """
        Verify if entity is subject to DORA.

        Returns:
        - IN_SCOPE: Full DORA requirements apply
        - OUT_OF_SCOPE: DORA does not apply
        - UNCLEAR: Needs legal clarification
        """

    @classmethod
    def get_applicable_entity_type(cls) -> str:
        """
        For algorithmic trading platform:

        Most likely:
        - "investment_firm" (Article 2(1)(e)) if MiFID authorized
        - "crypto_asset_service_provider" (Article 2(1)(f)) if MiCA authorized

        If NOT authorized as financial entity:
        - DORA may NOT apply
        - Consult legal counsel
        """
```

**КРИТИЧЕСКИ ВАЖНО**: Если платформа не является лицензированной финансовой организацией, DORA может НЕ применяться!

> **УТОЧНЕНИЕ (v4.1)**: Эта платформа позиционируется как **ICT Provider / Software Provider** (Art. 30).
> Мы НЕ являемся Investment Firm или иной Financial Entity. Наши B2B клиенты (банки, инвестфирмы)
> могут быть Financial Entities, и для них предоставляется compliance toolkit в
> `services/archive/dora_financial_entity/`.

---

### v3.0 Critical Fix #2: Critical/Important Function Definition (NEW)

**ПРОБЛЕМА**: План не определяет, какие функции являются "critical or important" — это фундаментально для всего DORA.

Согласно [DORA Article 3(22)](https://www.dora-info.eu/dora/article-3/):

> "Critical or important function" means a function, the disruption of which would **materially impair**:
> 1. The financial performance of a financial entity, OR
> 2. The soundness or continuity of its services and activities, OR
> 3. The continuing compliance with authorization conditions or regulatory obligations

**ДОБАВИТЬ В Phase 0**:

```python
# services/dora/function_classification.py

@dataclass
class FunctionClassification:
    """
    Article 3(22) - Critical or Important Function classification.

    MUST be done BEFORE third-party risk assessment.
    """
    function_name: str
    function_description: str

    # Assessment criteria (Article 3(22))
    impairs_financial_performance: bool
    impairs_service_soundness: bool
    impairs_regulatory_compliance: bool

    # Supporting ICT services
    ict_services_supporting: List[str]
    third_party_providers: List[str]

    @property
    def is_critical_or_important(self) -> bool:
        """Function is critical/important if ANY criterion is True."""
        return any([
            self.impairs_financial_performance,
            self.impairs_service_soundness,
            self.impairs_regulatory_compliance,
        ])

# Example classification for our platform
PLATFORM_FUNCTIONS = {
    "order_execution": FunctionClassification(
        function_name="Order Execution",
        function_description="Placing and executing trading orders",
        impairs_financial_performance=True,  # Direct revenue impact
        impairs_service_soundness=True,      # Core service
        impairs_regulatory_compliance=True,  # Best execution obligation
        ict_services_supporting=["exchange_api", "order_management"],
        third_party_providers=["binance", "alpaca", "oanda", "ib"],
    ),  # → CRITICAL

    "market_data": FunctionClassification(
        function_name="Market Data",
        function_description="Receiving real-time market prices",
        impairs_financial_performance=True,
        impairs_service_soundness=True,
        impairs_regulatory_compliance=False,
        ict_services_supporting=["data_feeds", "websockets"],
        third_party_providers=["binance", "polygon", "alpaca"],
    ),  # → CRITICAL

    "risk_monitoring": FunctionClassification(
        function_name="Risk Monitoring",
        function_description="Monitoring positions and risk limits",
        impairs_financial_performance=True,
        impairs_service_soundness=True,
        impairs_regulatory_compliance=True,  # MiFID II risk requirements
        ict_services_supporting=["risk_engine", "position_tracking"],
        third_party_providers=[],  # Internal
    ),  # → CRITICAL

    "reporting": FunctionClassification(
        function_name="Regulatory Reporting",
        function_description="Submitting regulatory reports",
        impairs_financial_performance=False,
        impairs_service_soundness=False,
        impairs_regulatory_compliance=True,  # Direct compliance impact
        ict_services_supporting=["reporting_system"],
        third_party_providers=[],
    ),  # → IMPORTANT (regulatory only)

    "backtesting": FunctionClassification(
        function_name="Strategy Backtesting",
        function_description="Testing strategies on historical data",
        impairs_financial_performance=False,
        impairs_service_soundness=False,
        impairs_regulatory_compliance=False,
        ict_services_supporting=["backtest_engine"],
        third_party_providers=["polygon"],  # Historical data
    ),  # → NOT Critical/Important
}
```

---

### v3.0 Critical Fix #3: Incident Reporting Timeline (CORRECTED)

**ПРОБЛЕМА v2.0**: Неверная интерпретация timeline.

**НЕВЕРНО в v2.0**:
> "4 hours after classification, 24 hours after detection"

**ПРАВИЛЬНО** (согласно [Article 19](https://www.digital-operational-resilience-act.com/Article_19.html)):

```python
# Correct interpretation
INCIDENT_REPORTING_DEADLINES = {
    "initial_notification": {
        # TWO conditions, whichever comes FIRST:
        "condition_1": "4 hours after classification as MAJOR",
        "condition_2": "24 hours after initial detection",
        "logic": "WHICHEVER IS EARLIER",  # <-- Critical difference!
    },
    "intermediate_report": "72 hours after initial notification",
    "final_report": "1 month after resolution",
}

# Example timeline
"""
T+0h:   Incident detected (timer starts for 24h)
T+2h:   Incident classified as MAJOR (timer starts for 4h)
T+6h:   DEADLINE = T+2h + 4h = T+6h  ✓ (classification-based)
        BUT ALSO check: T+0h + 24h = T+24h
        Result: T+6h is earlier, so deadline is T+6h

Alternative scenario:
T+0h:   Incident detected
T+22h:  Incident classified as MAJOR
T+24h:  DEADLINE = T+0h + 24h = T+24h  ✓ (detection-based)
        (because T+22h + 4h = T+26h would be later)

UPGRADE SCENARIO (per CDR 2025/301):
T+0h:   Incident detected (not initially major)
T+30h:  Incident upgraded to MAJOR
T+34h:  DEADLINE = T+30h + 4h = T+34h  ✓ (upgrade triggers new 4h window)
        The 24h from detection no longer applies after upgrade
"""

# IMPORTANT ADDITIONS per CDR 2025/301:

# 1. Weekend/Holiday Extension
DEADLINE_EXTENSIONS = {
    "weekend_or_public_holiday": {
        "applies_to": ["initial_notification"],
        "extension": "until noon of next working day",
        "entities": "certain financial entities per CDR 2025/301 Art. 4",
    }
}

# 2. Incident Upgrade Procedure
INCIDENT_UPGRADE = {
    "trigger": "Incident initially not major, later becomes major",
    "deadline": "4 hours from upgrade classification",
    "note": "24h from detection no longer constraining after upgrade",
}
```

---

### v3.0 Critical Fix #4: NCA Identification (NEW)

**ПРОБЛЕМА**: План говорит "YOUR_NCA" без указания как определить правильный NCA.

```yaml
# config/dora/nca_identification.yaml
nca_identification:
  # Step 1: Determine Member State of authorization
  authorization_member_state: null  # e.g., "DE", "FR", "NL"

  # Step 2: Identify NCA for your entity type
  # Different NCAs for different entity types!
  nca_by_entity_type:
    investment_firm:
      DE: "BaFin"
      FR: "AMF"
      NL: "AFM"
      IE: "Central Bank of Ireland"
      LU: "CSSF"
      # ... etc

    credit_institution:
      # Significant = report to NCA + ECB forwards
      # Less significant = report to NCA only
      significant: "ECB (via NCA)"
      less_significant: "National NCA"

  # Step 3: For incident reporting, may need single NCA if multiple supervisors
  # Per Article 19(1): Member States designate single NCA if multiple
  designated_incident_nca: null

  # Step 4: Reporting channels
  reporting_channels:
    register_of_information:
      platform: null  # e.g., "CSSF eDesk", "BaFin MVP Portal"
      format: "CSV per ITS"
    incident_reports:
      platform: null
      format: "As per NCA specification"
      backup_method: "email"  # If technical impossibility
```

**Дедлайны по странам** для Register of Information:

**ВАЖНО**: Reference date для первого submission = **31 March 2025**
(Register должен содержать все arrangements до этой даты)

| Member State | NCA | Deadline | Platform | Source |
|--------------|-----|----------|----------|--------|
| Ireland | CBI | 1-4 April 2025 | ONR | CBI guidance |
| Germany | BaFin | **11 April 2025** | MVP Portal | BaFin announcement |
| France | AMF/ACPR | **15 April 2025** | TBD | ACPR guidance |
| Luxembourg | CSSF | 1-15 April 2025 | eDesk | [CSSF](https://www.cssf.lu/en/2025/04/dora-submission-timeframe-for-register-of-information-edesk-portal-open-as-of-1-april-2025/) |
| Italy | Banca d'Italia | 30 April 2025 | TBD | BoI guidance |
| Netherlands | AFM/DNB | 30 April 2025 | TBD | AFM/DNB guidance |
| **ESA deadline** | All NCAs | **30 April 2025** | ESA portal | NCAs submit to ESAs |

---

### v3.0 Critical Fix #5: ITS Submission Format (NEW)

**ПРОБЛЕМА**: Plan только описывает dataclasses, но real submission требует specific formats.

Согласно [ITS JC 2023 85](https://www.esma.europa.eu/sites/default/files/2024-01/JC_2023_85_-_Final_report_on_draft_ITS_on_Register_of_Information.pdf):

```python
# services/dora/its_export.py

class ITSExporter:
    """
    Export Register of Information in ITS-compliant format.

    Format requirements:
    - Plain CSV format (not Excel)
    - UTF-8 encoding
    - Specific column order per template
    - Validation rules per DPM 4.0
    """

    # Column definitions per ITS Annex
    RT_02_01_COLUMNS = [
        "contractual_arrangement_reference_number",
        "lei_of_entity_making_use_of_ict_services",
        "lei_of_ict_third_party_service_provider",
        # ... 40+ columns per template
    ]

    def export_to_csv(
        self,
        register: DORARegisterOfInformation,
        template: str  # "B_02.01", "B_03.01", etc. (DPM 4.0 naming)
    ) -> bytes:
        """Export to plain CSV format."""

    def validate_against_dpm(
        self,
        data: pd.DataFrame,
        template: str
    ) -> ValidationResult:
        """
        Validate against Data Point Model (DPM 4.0).

        Checks:
        - Required fields present
        - Data types correct
        - Cross-field validation rules
        - Referential integrity between templates
        """
```

---

### v3.0 Critical Fix #6: LEI for Non-EU Providers (NEW)

**ПРОБЛЕМА**: Alpaca, Polygon — US companies без EU LEI. Binance структура сложная.

```yaml
# config/dora/provider_identification.yaml
provider_identification:
  # Providers WITH LEI
  with_lei:
    interactive_brokers:
      lei: "549300GYZ1LOSP5FNQ37"
      legal_name: "Interactive Brokers LLC"
      country: "US"

  # Providers WITHOUT LEI - use alternative identifier
  without_lei:
    binance:
      # Binance has multiple entities - identify the correct one
      alternatives:
        - identifier_type: "registration_number"
          identifier: "Binance Holdings Limited (Malta)"
          country: "MT"
          registration_authority: "Malta Business Registry"
        - identifier_type: "trade_name"
          identifier: "Binance"
          notes: "Report as 'non-LEI entity' per ITS guidance"

    alpaca:
      alternatives:
        - identifier_type: "sec_crd"
          identifier: "Alpaca Securities LLC - CRD# 288202"
          country: "US"
          registration_authority: "SEC/FINRA"

    polygon:
      alternatives:
        - identifier_type: "registration_number"
          identifier: "Polygon.io Inc"
          country: "US"
          notes: "Delaware corporation"

  # ITS handling for non-LEI entities
  non_lei_handling:
    # Per ITS, if no LEI:
    # 1. Use alternative identifier
    # 2. Mark as "non-LEI" in appropriate field
    # 3. Provide explanation in free text field
    procedure: |
      For ICT third-party service providers without LEI:
      1. Use national registration number
      2. Set LEI field to placeholder per NCA guidance
      3. Document in B_99.01 Definitions
```

---

### v3.0 Critical Fix #7: Microenterprise Definition (CORRECTED)

**НЕВЕРНО в v2.0**:
```python
# WRONG
return self.employee_count < 10 and self.annual_turnover_eur < 2_000_000
```

**ПРАВИЛЬНО** (EU Recommendation 2003/361):
```python
# CORRECT - OR not AND for turnover/balance sheet
@property
def is_microenterprise(self) -> bool:
    """
    EU Recommendation 2003/361 definition.

    Microenterprise = <10 employees AND (<€2M turnover OR <€2M balance sheet)
    """
    return (
        self.employee_count < 10
        and (
            self.annual_turnover_eur < 2_000_000
            or self.balance_sheet_eur < 2_000_000  # <-- OR, not AND!
        )
    )
```

---

### v3.0 Critical Fix #8: RTS Control Mapping (NEW)

**ПРОБЛЕМА**: Plan не mappит конкретные контроли из [RTS JC 2023 86](https://www.esma.europa.eu/sites/default/files/2024-01/JC_2023_86_-_Final_report_on_draft_RTS_on_ICT_Risk_Management_Framework_and_on_simplified_ICT_Risk_Management_Framework.pdf).

**RTS Control Categories** (Commission Delegated Regulation 2024/1774):

| RTS Chapter | Article | Control Area | Implementation File |
|-------------|---------|--------------|---------------------|
| **I. ICT Security** | Art. 2-5 | Security policies, procedures | `ict_security_policies.py` |
| | Art. 6-8 | ICT asset management | `ict_asset_management.py` |
| | Art. 9-10 | Encryption & cryptography | `encryption.py` |
| | Art. 11-13 | ICT operations security | `ict_operations.py` |
| | Art. 14-15 | Network security | `network_security.py` |
| | Art. 16-18 | ICT project/change mgmt | `change_management.py` |
| **II. HR Policy** | Art. 19 | Human resources security | `hr_security.py` |
| **III. Access Control** | Art. 20-22 | Identity & access mgmt | `access_control.py` |
| **IV. Detection** | Art. 23-24 | Incident detection | `incident_detection.py` |
| **V. Response** | Art. 25-26 | Incident response | `incident_response.py` |
| **VI. BCP** | Art. 27-30 | Business continuity | `business_continuity.py` |
| **VII. Review** | Art. 31-33 | ICT risk review | `ict_risk_review.py` |

**Каждый control должен быть mapped**:

```python
# services/dora/rts_compliance.py

RTS_CONTROLS = {
    "Art.6_ICT_Asset_Management": {
        "requirement": "Maintain updated inventory of ICT assets",
        "implementation": "services/dora/ict_asset_management.py",
        "existing_module": None,  # New implementation needed
        "tests": ["test_asset_inventory", "test_asset_classification"],
    },
    "Art.9_Encryption": {
        "requirement": "Encryption for data at rest and in transit",
        "implementation": "services/dora/encryption.py",
        "existing_module": "services/ai_act/cybersecurity.py",  # Extend
        "tests": ["test_encryption_at_rest", "test_encryption_in_transit"],
    },
    "Art.19_HR_Security": {
        "requirement": "HR security throughout employment lifecycle",
        "implementation": "services/dora/hr_security.py",
        "existing_module": None,  # New
        "tests": ["test_pre_employment", "test_termination_procedures"],
    },
    "Art.20_Access_Control": {
        "requirement": "Least privilege, access reviews",
        "implementation": "services/dora/access_control.py",
        "existing_module": "services/ai_act/cybersecurity.py",  # Extend
        "tests": ["test_least_privilege", "test_access_review"],
    },
    # ... all 33 RTS articles mapped
}
```

---

## CRITICAL REVIEW & CORRECTIONS (v2.0)

> **ВАЖНО**: Данный раздел содержит критический анализ первоначального плана и обязательные исправления.

### Выявленные критические проблемы

| # | Проблема | Severity | Исправление |
|---|----------|----------|-------------|
| 1 | **Пропущены Articles 15-16** | CRITICAL | Добавлены в Phase 1 |
| 2 | **Register of Information — неполная структура ITS** | CRITICAL | Полная структура 15 templates |
| 3 | **Incident thresholds — отсутствуют количественные критерии** | HIGH | Добавлены из Regulation 2024/1772 |
| 4 | **Proportionality — полностью игнорируется** | HIGH | Добавлена секция Article 16 simplified framework |
| 5 | **TLPT — нереалистичные требования** | HIGH | Уточнены критерии применимости |
| 6 | **Third-party contracts — невозможные ожидания** | HIGH | Реалистичный gap analysis |
| 7 | **Articles 31-44 — неверное понимание scope** | MEDIUM | Исправлено: это ESA oversight, не наши требования |
| 8 | **Временные рамки — DORA уже active** | CRITICAL | Переход от planning к remediation |
| 9 | **Exit strategies — отсутствует feasibility analysis** | HIGH | Добавлен реальный анализ |
| 10 | **Test estimates — произвольные цифры** | MEDIUM | Привязка к конкретным requirements |

### Ключевые исправления v2.0

#### 1. Proportionality Assessment (NEW)

**Определение применимого режима**:

Согласно [Article 16 DORA](https://www.digital-operational-resilience-act.com/Article_16.html), упрощенный ICT Risk Management Framework применяется к:

| Entity Type | Simplified Framework? | Reference |
|-------------|----------------------|-----------|
| Small, non-interconnected investment firms | ✅ YES | Art. 16(1)(a) |
| Payment institutions exempted per PSD2 | ✅ YES | Art. 16(1)(b) |
| Institutions exempted per CRD | ✅ YES | Art. 16(1)(c) |
| Electronic money institutions exempted | ✅ YES | Art. 16(1)(d) |
| Small IORPs | ✅ YES | Art. 16(1)(e) |
| **Microenterprises** (any type) | ✅ Частичные исключения | Art. 6(6), 28(2) |

**Для нашей платформы**:
- Если квалифицируемся как **microenterprise** (<10 сотрудников, <€2M оборот): применяется упрощенный режим
- Исключения для microenterprises:
  - НЕ требуется third-party ICT risk strategy (Art. 28(2))
  - Упрощенные требования к ICT risk management (Art. 6(6))
  - НЕ применяется recurring incidents assessment (Reg. 2024/1772 Art. 11(3))

```yaml
# config/dora/proportionality_assessment.yaml
entity_classification:
  # ОПРЕДЕЛИТЬ ДО НАЧАЛА РЕАЛИЗАЦИИ
  is_microenterprise: null  # true/false - <10 employees AND <€2M turnover
  is_small_enterprise: null  # <50 employees AND <€10M turnover
  applicable_regime: null  # "full" | "simplified" | "microenterprise_exemptions"

  assessment_date: null
  assessed_by: null
  nca_confirmation: false
```

#### 2. Incident Classification — Quantitative Thresholds (CORRECTED)

Согласно [Commission Delegated Regulation 2024/1772](https://eur-lex.europa.eu/eli/reg_del/2024/1772/oj/eng):

```yaml
# config/dora/incident_classification_thresholds.yaml
# ТОЧНЫЕ ПОРОГИ из CDR 2024/1772 Article 9

major_incident_determination:
  # Major = critical services affected + (malicious OR 2+ thresholds met)

  materiality_thresholds:
    # Article 9(1) - Clients/Counterparties/Transactions
    # CORRECTED: RTS Article 9 specifies 100,000 OR 10%, NOT different values per client type
    clients_affected:
      condition: "OR"  # Either absolute OR relative threshold
      absolute_threshold: 100000  # 100,000 clients affected
      relative_threshold_percent: 10  # OR 10% of all clients using affected service
      notes: |
        Per CDR 2024/1772 Article 9:
        - Absolute threshold increased from 50,000 to 100,000 after consultation
        - Relative threshold ensures proportionality for smaller entities
        - If actual number unknown, estimate based on comparable reference periods

    transactions_affected:
      # Количество или объем транзакций
      relative_threshold_percent: 10  # 10% of daily average value
      notes: "10% of daily average value of transactions related to affected service"

    # Article 9(2) - Reputational Impact
    reputational_impact:
      condition: "ANY"
      triggers:
        - "incident_visible_in_media"
        - "repeated_client_complaints"
        - "regulatory_action_expected"
        - "likely_loss_of_clients"

    # Article 9(3) - Duration/Service Downtime
    duration:
      critical_services_downtime_hours: 2  # >2 часов для critical
      important_services_downtime_hours: 4  # >4 часов для important
      recovery_time_exceeds_rto: true

    # Article 9(4) - Geographic Spread
    geographic_spread:
      member_states_affected: 2  # 2+ Member States

    # Article 9(5) - Data Losses (AUTOMATIC MAJOR if data breach)
    data_losses:
      condition: "ANY"
      automatic_major_triggers:
        - "availability_breach"  # Data unavailable
        - "authenticity_breach"  # Data authenticity compromised
        - "integrity_breach"     # Data modified
        - "confidentiality_breach"  # Unauthorized access
      data_types:
        - "personal_data"        # GDPR intersection
        - "payment_data"         # PSD2 intersection
        - "trade_secrets"

    # Article 9(6) - Economic Impact
    economic_impact:
      direct_costs_eur: 100000  # >€100,000 direct costs/losses
      indirect_costs_eur: 500000  # >€500,000 total impact
      includes:
        - "recovery_costs"
        - "legal_costs"
        - "regulatory_fines"
        - "lost_revenue"
        - "compensation_to_clients"

  # Article 11 - Recurring Incidents
  recurring_incidents:
    # НЕ применяется к microenterprises (Art. 11(3))
    applies_to_microenterprises: false
    threshold:
      count: 2  # 2+ incidents
      period_months: 6
      same_root_cause: true
    assessment_frequency: "monthly"
```

#### 3. Register of Information — Full ITS Structure (CORRECTED)

Согласно [CIR 2024/2956](https://eur-lex.europa.eu/eli/reg_impl/2024/2956/oj/eng) (final ITS):

**ВАЖНО**: Template IDs изменены с "RT_xx.xx" на "B_xx.xx" в соответствии с DPM 4.0.
Содержание осталось тем же, изменились только идентификаторы.

**15 обязательных templates** (DPM 4.0 naming):

| Template | Name | Description | Our Data Source |
|----------|------|-------------|-----------------|
| **B_01.01** | Entity Identification | Reporting entity info | Company registration |
| **B_02.01** | Contractual Arrangement | Basic contract info | Adapter configs |
| **B_02.02** | Entities Using ICT Services | Entities in scope | Group structure |
| **B_02.03** | Intra-group Linkages | Intra-group connections | N/A (single entity) |
| **B_03.01** | ICT Service Provider ID | Provider identification | Adapter metadata |
| **B_03.02** | Direct Providers | Direct ICT providers | Binance, Alpaca, etc. |
| **B_03.03** | Intra-group Providers | Group ICT providers | N/A |
| **B_04.01** | Entity Making Use | Entity using services | Our platform |
| **B_05.01** | ICT Services | Service descriptions | API services |
| **B_05.02** | ICT Service Chain | Sub-contractors | Provider dependencies |
| **B_05.03** | Service Chain Details | Chain details | Unknown (provider side) |
| **B_06.01** | Functions | Business functions | Trading, data, risk |
| **B_07.01** | Assessments | Risk assessments | Our risk analysis |
| **B_08.01** | Costs | ICT service costs | Fee schedules |
| **B_99.01** | Definitions | Entity-specific definitions | Custom definitions |

**Relational Keys** (связи между templates):
- `contractual_arrangement_reference_number` — связывает B_02.* с B_03-07
- `lei_entity_using_ict_services` — идентификация нашей entity
- `ict_service_provider_identifier` — идентификация провайдера

#### 4. Third-Party Contracts — Realistic Assessment (CORRECTED)

**Реальность контрактов с exchanges**:

| Provider | Contract Type | DORA Art.30(3) Compliance | Gaps | Mitigation |
|----------|--------------|---------------------------|------|------------|
| **Binance** | Standard Terms | ❌ ~20% | No audit rights, No SLA, No exit support | Document gaps, internal monitoring |
| **Alpaca** | Standard Terms | ❌ ~25% | Limited SLA, No NCA access | US-regulated, use as-is |
| **Polygon** | SaaS Agreement | ❌ ~30% | No incident notification, No audit | Alternative: IEX, Alpha Vantage |
| **OANDA** | Standard Terms | ⚠️ ~40% | Limited BCP info | FCA-regulated entity |
| **IB** | Client Agreement | ⚠️ ~50% | Some SLA terms exist | Better than crypto exchanges |

**Реалистичная стратегия**:

```python
# Вместо "contract amendments" которые невозможны:

class RealisticThirdPartyCompliance:
    """
    DORA compliance strategy for standard-terms providers.

    Reality: Binance/Alpaca will NOT modify their terms for us.
    Strategy: Document gaps + implement compensating controls.
    """

    COMPENSATING_CONTROLS = {
        "no_audit_rights": [
            "Monitor public security audits/SOC2 reports",
            "Track provider's regulatory status",
            "Subscribe to provider's status page",
            "Document reliance on provider's own compliance",
        ],
        "no_sla": [
            "Implement internal SLA monitoring",
            "Track actual availability metrics",
            "Document historical uptime",
            "Set internal alert thresholds",
        ],
        "no_incident_notification": [
            "Monitor provider status pages (automated)",
            "Subscribe to provider announcements",
            "Implement health check endpoints",
            "Detect incidents via API errors",
        ],
        "no_exit_support": [
            "Maintain parallel adapters (already have)",
            "Document data export procedures",
            "Test failover quarterly",
            "Keep alternative provider accounts active",
        ],
    }
```

#### 5. TLPT Applicability (CORRECTED)

**Проверка применимости Article 26**:

TLPT обязателен ТОЛЬКО для entities designated by competent authorities на основе:
- Systemic importance
- ICT risk profile
- Size and complexity

```yaml
# config/dora/tlpt_applicability.yaml
tlpt_assessment:
  # БОЛЬШИНСТВО алгоритмических трейдеров НЕ попадают под TLPT

  designation_criteria:
    # NCA designates based on:
    systemic_importance: false  # Мы не системно значимы
    critical_ict_functions_for_sector: false
    cross_border_significant: false

  our_assessment:
    likely_designated: false
    reasoning: |
      Small/medium algorithmic trading platform.
      No systemic importance to EU financial sector.
      Limited cross-border operations.
      NCA designation unlikely.

  # Если НЕ designated:
  alternative_testing:
    # Article 24 (general testing) still applies:
    vulnerability_assessments: "quarterly"
    penetration_testing: "yearly"  # Standard, not TLPT
    scenario_based_testing: "yearly"
    source_code_reviews: "per_major_release"

  # Если designated (unlikely):
  tlpt_requirements:
    frequency: "every_3_years"
    external_provider_required: "1_of_3_engagements"
    estimated_cost_eur: "100000-500000"
    preparation_time_months: 6
```

#### 6. Articles 31-44 Scope (CORRECTED)

**НЕВЕРНО в v1.0**: Мы должны "implement" Articles 31-44.

**ПРАВИЛЬНО**: Articles 31-44 описывают **oversight framework ESAs над Critical Third-Party Providers (CTPPs)**.

Это НЕ наши requirements. Это как ESAs будут надзирать за designated CTPPs (AWS, Microsoft, Google).

**Что это значит для нас**:
1. Мы НЕ реализуем Articles 31-44
2. Мы проверяем, используем ли мы designated CTPPs
3. Если да — нам нужно учитывать ESA recommendations для наших contracts

```python
# Simplified approach
class CTPPConsiderations:
    """What we actually need to do regarding Articles 31-44."""

    # UPDATED 2025-11-19: ESAs designated 19 CTPPs
    # Source: https://www.esma.europa.eu/press-news/esma-news/european-supervisory-authorities-designate-critical-ict-third-party-providers
    DESIGNATED_CTPPS = [
        "Amazon Web Services (AWS)",
        "Google Cloud Platform",
        "Microsoft Azure",
        "Oracle",
        "SAP",
        "Deutsche Telekom",
        # ... and 13 more providers
    ]

    def check_ctpp_usage(self) -> List[str]:
        """
        Check if we use any designated CTPPs.

        IMPORTANT: As of 19 November 2025, ESAs HAVE designated 19 CTPPs.
        This includes major cloud providers: AWS, Google Cloud, Microsoft Azure,
        as well as Oracle, SAP, Deutsche Telekom and others.

        If we use any of these providers, we should:
        1. Monitor ESA oversight findings
        2. Consider implications for our risk assessments
        3. Be aware of potential contract review requirements
        """
        # Check our infrastructure against designated list
        our_providers = self._get_infrastructure_providers()
        return [p for p in our_providers if p in self.DESIGNATED_CTPPS]

    def implications_if_using_ctpp(self):
        """
        If using designated CTPP (AWS, GCP, Azure, etc.):
        - CTPPs are now under direct ESA oversight (Joint Examination Teams)
        - CTPPs must cooperate with ESA inspections
        - Annual ESA risk analyses published - monitor findings
        - Consider ESA findings in our risk assessment
        - May need to review contracts if ESA finds issues
        - Indirect cost implications (CTPPs pay oversight fees)

        We do NOT implement Articles 31-44 ourselves.
        These articles describe ESA oversight framework, not our obligations.
        """
```

---

## Executive Summary

Digital Operational Resilience Act (DORA) — регулирование ЕС, устанавливающее единые требования к цифровой операционной устойчивости для финансового сектора. Данный план описывает интеграцию DORA в платформу алгоритмической торговли с учетом:

- Существующей архитектуры EU AI Act compliance (1007+ тестов)
- Существующей инфраструктуры MiFID II compliance (7 фаз)
- 5 ключевых направлений DORA
- Технических стандартов ESAs (RTS/ITS)

### Scope of Application

Платформа попадает под действие DORA как:
- **Инвестиционная фирма** (Article 2(1)(e)) — использование алгоритмической торговли
- **Пользователь ICT-сервисов третьих сторон** — интеграции с Binance, Alpaca, Polygon, OANDA, Interactive Brokers, Deribit

### Synergy with Existing Compliance

| Existing Module | DORA Reuse Potential | Gap Analysis |
|-----------------|---------------------|--------------|
| `services/ai_act/risk_management.py` | High — расширить для ICT risk | Добавить ICT-специфичные категории |
| `services/ai_act/post_market_monitoring.py` | High — incident tracking | Расширить для DORA incident classification |
| `services/ai_act/logging_system.py` | High — audit trail | Добавить ICT event logging |
| `services/ai_act/cybersecurity.py` | High — security measures | Расширить threat detection |
| `configs/compliance/mifid_compliance.yaml` | Medium — BCP, kill switch | Расширить для DORA requirements |
| `adapters/*` | Critical — third-party providers | Создать Register of Information |

---

## DORA Requirements Overview

### 5 Pillars of DORA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DORA - 5 Key Pillars                                 │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────┤
│   PILLAR 1      │   PILLAR 2      │   PILLAR 3      │   PILLAR 4          │
│   ICT Risk      │   Incident      │   Digital       │   Third-Party       │
│   Management    │   Reporting     │   Resilience    │   Risk Management   │
│   (Art. 5-16)   │   (Art. 17-23)  │   Testing       │   (Art. 28-44)      │
│                 │                 │   (Art. 24-27)  │                     │
├─────────────────┴─────────────────┴─────────────────┴─────────────────────┤
│                           PILLAR 5                                         │
│              Information Sharing (Article 45)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Implementation Plan (CORRECTED v2.0)

### Phase Overview

| Phase | Название | Articles | Tests | Ключевые Deliverables | Priority |
|-------|----------|----------|-------|----------------------|----------|
| **Phase 0** | Proportionality Assessment | Art. 4, 16 | ~15 | Entity classification, applicable regime | **P0 - IMMEDIATE** |
| **Phase 1** | ICT Risk Management Framework | Art. 5-16 (ALL) | ~250 | ICT Risk Framework, Governance, **Simplified if Art.16** | P0 |
| **Phase 2** | ICT Incident Management & Reporting | Art. 17-23 | ~200 | Incident Classification (CDR 2024/1772), ITS Templates | P0 |
| **Phase 3** | Digital Resilience Testing | Art. 24-25 (+26-27 if designated) | ~180 | Vulnerability Testing, Pentest, **TLPT only if required** | P1 |
| **Phase 4** | Third-Party ICT Risk Management | Art. 28-30 (NOT 31-44) | ~220 | Register of Information (15 ITS templates), Gap Analysis | **P0 - DEADLINE Apr** |
| **Phase 5** | Information Sharing & Integration | Art. 45 + Final | ~150 | Threat Intelligence, Cross-Regulation | P2 |
| **TOTAL** | | | **~1015** | Proportionate DORA Compliance | |

### CORRECTED: Phase Dependencies

```
Phase 0 (Proportionality) ──┬──→ Phase 1 (Risk Management)
                            │         │
                            │         ↓
                            │    Phase 2 (Incidents)
                            │         │
                            │         ↓
                            └──→ Phase 4 (Third-Party) ──→ Phase 3 (Testing)
                                                                │
                                                                ↓
                                                          Phase 5 (Integration)
```

### Критические даты

| Deadline | Requirement | Status |
|----------|-------------|--------|
| **17 Jan 2025** | DORA application date | ⚠️ PASSED - remediation mode |
| **30 Apr 2025** | Register of Information submission | 🔴 4.5 months remaining |
| **Ongoing** | Major incident reporting (4h/24h/72h) | Must be ready NOW |
| **Yearly** | Annual ICT risk assessment review | First by Jan 2026 |

---

# Phase 0: Proportionality Assessment (NEW)
## Articles 4, 16 — ОБЯЗАТЕЛЬНО ПЕРВЫМ

**Приоритет**: P0 - IMMEDIATE (определяет scope всех остальных фаз)
**Срок**: До начала любой другой работы

### 0.1 Entity Classification

**Цель**: Определить, какой режим DORA применяется к нашей entity.

**Файл**: `services/dora/proportionality.py`

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class DORARegime(Enum):
    """Applicable DORA regime based on entity classification."""
    FULL = "full"                        # Standard requirements
    SIMPLIFIED = "simplified"            # Article 16 simplified framework
    MICROENTERPRISE = "microenterprise"  # Partial exemptions

@dataclass
class EntityClassification:
    """
    Classification per DORA Article 4 (Proportionality) and Article 16.
    """
    # Size criteria
    employee_count: int
    annual_turnover_eur: float
    balance_sheet_eur: float

    # Entity type
    entity_type: str  # "investment_firm", "payment_institution", etc.

    # Exemption checks
    is_small_non_interconnected: bool = False  # Art. 16(1)(a)
    is_psd2_exempted: bool = False             # Art. 16(1)(b)
    is_crd_exempted: bool = False              # Art. 16(1)(c)
    is_emd_exempted: bool = False              # Art. 16(1)(d)
    is_small_iorp: bool = False                # Art. 16(1)(e)

    @property
    def is_microenterprise(self) -> bool:
        """EU Recommendation 2003/361 definition."""
        return self.employee_count < 10 and self.annual_turnover_eur < 2_000_000

    @property
    def is_small_enterprise(self) -> bool:
        return self.employee_count < 50 and self.annual_turnover_eur < 10_000_000

    @property
    def applicable_regime(self) -> DORARegime:
        """Determine which DORA regime applies."""
        # Check for Article 16 simplified framework
        if any([
            self.is_small_non_interconnected,
            self.is_psd2_exempted,
            self.is_crd_exempted,
            self.is_emd_exempted,
            self.is_small_iorp,
        ]):
            return DORARegime.SIMPLIFIED

        if self.is_microenterprise:
            return DORARegime.MICROENTERPRISE

        return DORARegime.FULL

    @property
    def exemptions(self) -> list[str]:
        """List of specific exemptions that apply."""
        exemptions = []

        if self.applicable_regime == DORARegime.SIMPLIFIED:
            exemptions.extend([
                "Articles 5-15 simplified (Article 16)",
                "Reduced documentation requirements",
                "Simplified testing requirements",
            ])

        if self.is_microenterprise:
            exemptions.extend([
                "No third-party ICT risk strategy required (Art. 28(2))",
                "Simplified ICT risk management (Art. 6(6))",
                "No recurring incidents assessment (Reg. 2024/1772 Art. 11(3))",
                "No management body training requirements (Art. 5(4))",
            ])

        return exemptions
```

**Конфигурация**: `services/archive/dora_financial_entity/configs/entity_classification.yaml` *(archived - FE-specific)*

```yaml
# ЗАПОЛНИТЬ ПЕРЕД НАЧАЛОМ РЕАЛИЗАЦИИ
entity_classification:
  legal_name: ""
  lei: ""  # Legal Entity Identifier

  # Size metrics (as of assessment date)
  assessment_date: "2025-01-01"
  employee_count: null      # Заполнить
  annual_turnover_eur: null # Заполнить
  balance_sheet_eur: null   # Заполнить

  # Entity type per MiFID II
  entity_type: "investment_firm"  # or appropriate type
  mifid_authorization: ""

  # Article 16 exemption checks
  exemption_checks:
    small_non_interconnected_investment_firm:
      applicable: false
      documentation: ""
    psd2_exempted:
      applicable: false
      exemption_reference: ""
    crd_exempted:
      applicable: false
      member_state_decision: ""
    emd_exempted:
      applicable: false
      exemption_reference: ""
    small_iorp:
      applicable: false
      member_state: ""

  # Result (filled by assessment)
  determined_regime: null  # "full" | "simplified" | "microenterprise"
  nca_confirmed: false
  confirmation_date: null
```

### 0.2 Phase 0 Deliverables

> **Note (v4.1):** Phase 0 modules were for Financial Entity scope. Now archived.

| Deliverable | Current Location | Status |
|-------------|------------------|--------|
| Entity Classification | `services/archive/dora_financial_entity/proportionality.py` | Archived |
| Classification Config | `services/archive/dora_financial_entity/configs/entity_classification.yaml` | Archived |
| Assessment Report | `docs/compliance/dora/proportionality_assessment.md` | Reference only |

### 0.3 Phase 0 Acceptance Criteria

- [ ] Entity size metrics documented
- [ ] All Article 16 exemption criteria checked
- [ ] Applicable regime determined
- [ ] If simplified/microenterprise: exemptions documented
- [ ] Assessment reviewed by legal/compliance
- [ ] Regime documented for NCA if requested

---

# Phase 1: ICT Risk Management Framework
## Articles 5-16 Implementation (CORRECTED: включает 15-16)

**Приоритет**: P0 (Critical Path)
**Зависимости**: Phase 0, существующий `services/ai_act/risk_management.py`
**Scope**: Зависит от результата Phase 0 (full vs simplified)

### 1.1 Governance and Control Framework (Article 5)

#### 1.1.1 Management Body Responsibilities

**Требования Article 5(2)**:
- Ultimate responsibility for ICT risk management lies with management body
- Define, approve, oversee implementation of ICT risk management framework
- Approve digital operational resilience strategy

**Реализация**:

**Файл**: `services/dora/governance.py`

```
DORAGovernanceFramework:
├── ManagementBodyOversight
│   ├── approve_ict_risk_framework()
│   ├── review_digital_resilience_strategy()
│   ├── approve_ict_budget()
│   └── oversee_arrangements_with_ict_providers()
├── RoleAssignment
│   ├── ICTRiskOfficer (control function)
│   ├── segregation_of_duties()
│   └── three_lines_of_defence()
├── TrainingRequirements
│   ├── mandatory_ict_training()
│   ├── track_training_completion()
│   └── assess_knowledge_skills()
└── AuditIntegration
    ├── internal_audit_schedule()
    └── audit_findings_tracking()
```

**Ключевые аспекты для реализации**:
- [ ] Создать структуру для управленческого контроля ICT рисков
- [ ] Интеграция с существующим `QualityManagementSystem`
- [ ] Определить роли и ответственность по DORA
- [ ] Программа обязательного обучения ICT рискам (Article 5(4))

#### 1.1.2 Digital Operational Resilience Strategy

**Требования Article 6(8)**:
- Стратегия должна описывать реализацию ICT risk management framework
- Устанавливать risk tolerance level
- Определять clear information security objectives
- Включать KPI и KRM (Key Risk Metrics)

**Файл**: `config/dora/digital_resilience_strategy.yaml`

**Элементы стратегии** (per Article 6(8)):
1. Risk tolerance level для ICT risk
2. Information security objectives с KPIs
3. ICT reference architecture explanation
4. Dependencies on ICT third-party providers
5. Full ICT business continuity policy
6. Communication plan for ICT-related incidents

### 1.2 ICT Risk Management Framework (Article 6)

#### 1.2.1 Framework Structure

**Требования Article 6(1-7)**:
- Documented framework with strategies, policies, procedures
- Minimize impact of ICT risk
- Subject to internal audit

**Интеграция с существующим кодом**:

```python
# Расширение services/ai_act/risk_management.py

class DORAICTRiskCategory(Enum):
    """ICT-specific risk categories per DORA."""
    ICT_AVAILABILITY = "ict_availability"
    ICT_AUTHENTICITY = "ict_authenticity"
    ICT_INTEGRITY = "ict_integrity"
    ICT_CONFIDENTIALITY = "ict_confidentiality"
    ICT_THIRD_PARTY = "ict_third_party"
    ICT_CONCENTRATION = "ict_concentration"
    ICT_INFRASTRUCTURE = "ict_infrastructure"
    ICT_CHANGE_MANAGEMENT = "ict_change_management"
    ICT_DISASTER_RECOVERY = "ict_disaster_recovery"
    CYBER_THREAT = "cyber_threat"

class DORAICTRiskFramework:
    """
    DORA Article 6 compliant ICT Risk Management Framework.

    Extends AIActRiskManager with ICT-specific requirements.
    """
    def __init__(self, ai_act_risk_manager: AIActRiskManager):
        self.ai_act_rm = ai_act_risk_manager
        self.ict_risk_registry = DORAICTRiskRegistry()
        self.control_framework = ControlFramework()
```

#### 1.2.2 ICT Systems, Protocols and Tools (Article 7)

**Требования**:
- Use and maintain updated ICT systems
- Design ICT systems ensuring resilience, continuity, availability
- Proportionate to business needs

**Файл**: `services/dora/ict_systems.py`

**Ключевые компоненты**:
```
ICTSystemsManagement:
├── ICTAssetInventory
│   ├── register_all_ict_assets()
│   ├── classify_criticality()
│   ├── document_dependencies()
│   └── version_tracking()
├── ICTSecurityPolicy
│   ├── network_security_rules()
│   ├── access_control_policies()
│   ├── encryption_requirements()
│   └── patch_management()
├── ICTCapacityManagement
│   ├── performance_monitoring()
│   ├── capacity_planning()
│   └── scalability_provisions()
└── ICTChangeManagement
    ├── change_approval_workflow()
    ├── testing_before_deployment()
    └── rollback_procedures()
```

### 1.3 Identification (Article 8)

**Требования Article 8(1-6)**:
- Identify all sources of ICT risk
- Identify all ICT-supported business functions
- Identify all assets and their criticality
- Map ICT assets to business functions
- Assess cyber threats and vulnerabilities

**Файл**: `services/dora/ict_identification.py`

**Компоненты идентификации**:

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **ICT Asset Register** | All ICT assets with criticality | `ICTAssetRegister` class |
| **Business Function Mapping** | ICT → Business function links | `BusinessFunctionMapper` |
| **ICT Provider Dependencies** | Third-party provider mapping | `ICTProviderDependencyMap` |
| **Vulnerability Assessment** | Regular vulnerability scanning | `VulnerabilityScanner` |
| **Threat Landscape** | Current threats identification | `ThreatLandscapeAnalyzer` |

**Интеграция с адаптерами** (критично):

```python
# Mapping existing adapters to ICT assets
ICT_THIRD_PARTY_PROVIDERS = {
    "binance": {
        "provider_type": "exchange",
        "services": ["market_data", "order_execution", "account_management"],
        "criticality": "CRITICAL",
        "adapter_path": "adapters/binance/",
        "contracts": ["spot", "futures"],
    },
    "alpaca": {
        "provider_type": "broker",
        "services": ["market_data", "order_execution"],
        "criticality": "CRITICAL",
        "adapter_path": "adapters/alpaca/",
    },
    # ... all other adapters
}
```

### 1.4 Protection and Prevention (Article 9)

**Требования**:
- Continuous monitoring and control of ICT systems
- Implementation of ICT security policies
- Mechanisms against intrusion and data misuse

**Расширение**: `services/ai_act/cybersecurity.py`

**Новые компоненты для DORA**:

```
DORAProtectionMeasures:
├── ContinuousMonitoring (extends existing monitoring.py)
│   ├── network_traffic_monitoring()
│   ├── security_event_monitoring()
│   ├── anomaly_detection()
│   └── real_time_alerting()
├── ICTSecurityPolicies
│   ├── access_control_implementation()
│   ├── strong_authentication()
│   ├── data_classification()
│   └── encryption_in_transit_at_rest()
├── PatchManagement
│   ├── vulnerability_tracking()
│   ├── patch_prioritization()
│   ├── deployment_automation()
│   └── verification_testing()
└── NetworkSecurity
    ├── segmentation()
    ├── firewall_rules()
    ├── intrusion_detection()
    └── secure_configuration()
```

### 1.5 Detection (Article 10)

**Требования**:
- Mechanisms to detect anomalous activities
- Multiple layers of control
- Detection of single points of failure

**Файл**: `services/dora/detection.py`

**Ключевые детекторы**:

| Detector | Function | Integration |
|----------|----------|-------------|
| `AnomalyDetector` | Unusual ICT activity | Extend `services/ai_act/human_oversight.py` |
| `PerformanceDegradationDetector` | Performance issues | Extend `services/monitoring.py` |
| `SecurityIncidentDetector` | Security breaches | New component |
| `SinglePointOfFailureDetector` | SPOF identification | New component |
| `ICTIncidentDetector` | ICT incidents | Extend `post_market_monitoring.py` |

### 1.6 Response and Recovery (Article 11)

**Требования**:
- ICT business continuity policy
- ICT response and recovery plans
- Crisis communication plans
- Testing of plans at least yearly

**Интеграция с MiFID II BCP**:

Существующий `configs/compliance/mifid_compliance.yaml` уже содержит:
- Business continuity configuration
- RTO/RPO targets
- BCP scenarios
- Drill requirements

**Расширения для DORA**:

**Файл**: `services/dora/response_recovery.py`

```python
class DORAResponseRecovery:
    """
    Article 11 compliant response and recovery system.

    Extends MiFID II BCP with DORA-specific requirements.
    """

    # Article 11(3) - Dedicated response/recovery plans for each ICT scenario
    ICT_SCENARIOS = [
        "cyber_attack",
        "system_failure",
        "data_corruption",
        "third_party_failure",
        "infrastructure_failure",
        "natural_disaster",
        "power_outage",
        "communication_failure",
    ]

    # Article 11(4) - Crisis management function
    def activate_crisis_management(self, incident: ICTIncident):
        """Activate crisis management with clear procedures."""

    # Article 11(5) - Regular testing yearly
    def execute_annual_bcp_test(self):
        """Execute comprehensive BCP testing."""

    # Article 11(6) - Cyber attack scenarios
    def test_cyber_attack_scenario(self, scenario: CyberAttackScenario):
        """Test response to simulated cyber attacks."""
```

### 1.7 Backup Policies and Recovery (Article 12)

**Требования**:
- Backup policies and procedures
- Restoration and recovery methods
- Backup systems physically and logically segregated
- Redundant ICT capacities

**Интеграция**:

Существующий `services/state_storage.py` и `configs/state.yaml`:
- Snapshot interval
- Backup retention
- Atomic writes

**Расширения для DORA**:

**Файл**: `services/dora/backup_recovery.py`

```
DORABackupSystem:
├── BackupPolicyEngine
│   ├── define_backup_scope()        # Article 12(1)(a)
│   ├── set_backup_frequency()       # Based on data criticality
│   ├── manage_retention()           # Compliance with retention requirements
│   └── verify_backup_integrity()
├── RestorationProcedures
│   ├── define_recovery_procedures() # Article 12(1)(b)
│   ├── segregated_recovery_systems() # Article 12(3)
│   ├── recovery_time_objectives()
│   └── recovery_point_objectives()
├── RedundantCapacities
│   ├── infrastructure_redundancy()  # Article 12(5)
│   ├── geographic_separation()
│   ├── failover_automation()
│   └── capacity_testing()
└── DataIntegrityVerification
    ├── post_recovery_checks()       # Article 12(7)
    ├── reconciliation_procedures()
    └── consistency_validation()
```

### 1.8 Learning and Evolving (Article 13)

**Требования**:
- Gather information on vulnerabilities and threats
- Assess impact of severe disruption
- Post-incident reviews
- Regular testing and updating

**Файл**: `services/dora/learning.py`

```python
class DORALearningSystem:
    """
    Article 13 - Continuous learning and evolution.
    """

    def collect_threat_intelligence(self):
        """Gather information on vulnerabilities and cyber threats."""

    def conduct_post_incident_review(self, incident: DORAIncident):
        """
        Post-incident review per Article 13(2).

        Analyze:
        - Root cause
        - Effectiveness of response
        - Lessons learned
        - Process improvements
        """

    def update_risk_assessment(self, new_threats: List[Threat]):
        """Update ICT risk assessment based on new information."""

    def incorporate_testing_results(self, test_results: TestResults):
        """Incorporate results from resilience testing."""
```

### 1.9 Communication (Article 14)

**Требования**:
- Crisis communication plans
- Internal and external communication procedures
- Disclosure obligations for ICT incidents

**Файл**: `services/dora/communication.py`

```
DORACommuncationPlan:
├── InternalCommunication
│   ├── escalation_procedures()
│   ├── staff_notification()
│   └── management_reporting()
├── ExternalCommunication
│   ├── client_notification()
│   ├── counterparty_notification()
│   └── service_provider_notification()
├── RegulatoryReporting
│   ├── competent_authority_notification()
│   ├── incident_reporting()
│   └── annual_reporting()
└── PublicDisclosure
    ├── public_statement_procedures()
    └── media_handling()
```

### 1.10 ICT Business Continuity Management (Article 15) — ДОБАВЛЕНО v2.0

**Требования Article 15** (ранее пропущен):
- ICT business continuity plans derived from BIA
- Testing of ICT business continuity plans
- Review after major changes or incidents
- Communication plans for clients and counterparties

**Файл**: `services/dora/ict_business_continuity.py`

```python
class DORABusinessContinuity:
    """
    Article 15 - ICT business continuity management.

    Note: Overlaps significantly with MiFID II BCP (already implemented).
    This module extends existing MiFID II BCP with DORA-specific requirements.
    """

    def conduct_business_impact_analysis(self) -> BIAResult:
        """
        Article 15(1) - Business Impact Analysis.

        Identify critical business functions and their ICT dependencies.
        Determine RTO/RPO for each function.
        """

    def develop_ict_bcp(
        self,
        bia: BIAResult
    ) -> ICTBusinessContinuityPlan:
        """
        Article 15(2) - ICT Business Continuity Plans.

        Plans must address:
        - All ICT scenarios from Article 11
        - Alternative solutions for critical functions
        - Transition to recovery sites
        - Minimum service levels during disruption
        """

    def test_ict_bcp(
        self,
        plan: ICTBusinessContinuityPlan,
        test_type: str  # "tabletop", "walkthrough", "simulation", "full"
    ) -> BCPTestResult:
        """
        Article 15(3) - Testing of ICT BCP.

        Must test at least yearly or after significant changes.
        Document results and remediation actions.
        """

    def review_after_incident(
        self,
        incident: DORAIncident,
        plan: ICTBusinessContinuityPlan
    ) -> BCPReview:
        """
        Article 15(4) - Review after incidents.

        Review and update BCP after:
        - Major ICT incidents
        - Material changes to ICT systems
        - Changes to business functions
        """
```

**Интеграция с MiFID II**:

Существующий `configs/compliance/mifid_compliance.yaml` уже содержит:
- `business_continuity` section
- RTO/RPO targets
- BCP scenarios
- Drill requirements

**Gap с DORA**:
1. BIA не формализован → нужен `services/dora/bia.py`
2. ICT-specific scenarios → расширить список
3. Review triggers → автоматизировать post-incident review

### 1.11 Simplified ICT Risk Management Framework (Article 16) — ДОБАВЛЕНО v2.0

**Применимость**: Определяется в Phase 0.

**Если применяется Article 16** (simplified framework):

```python
class SimplifiedICTRiskManagement:
    """
    Article 16 - Simplified framework for qualifying entities.

    Applies to:
    - Small, non-interconnected investment firms (Art. 16(1)(a))
    - Exempted payment institutions (Art. 16(1)(b))
    - Exempted credit institutions (Art. 16(1)(c))
    - Exempted e-money institutions (Art. 16(1)(d))
    - Small IORPs (Art. 16(1)(e))
    """

    # Article 16(2) - Simplified requirements
    SIMPLIFIED_REQUIREMENTS = {
        "ict_risk_management_framework": {
            "full_article_6": False,
            "simplified_version": True,
            "documentation": "Basic framework document",
        },
        "ict_systems": {
            "full_article_7": False,
            "requirements": [
                "Sound and documented framework",
                "Continuous monitoring",
                "Minimize ICT risk impact",
                "Allow identification of risk sources",
            ],
        },
        "governance": {
            "simplified": True,
            "management_training": False,  # Not required for micro
        },
        "testing": {
            "full_testing_programme": False,
            "basic_testing": True,  # Still required
        },
    }

    def generate_simplified_framework(self) -> SimplifiedFramework:
        """Generate Article 16 compliant simplified framework."""

    def validate_eligibility(
        self,
        entity: EntityClassification
    ) -> EligibilityResult:
        """Validate entity qualifies for simplified framework."""
```

**Конфигурация для simplified regime**:

```yaml
# config/dora/simplified_framework.yaml
# Applies ONLY if Phase 0 determines simplified regime

simplified_framework:
  enabled: false  # Set to true if Article 16 applies

  documentation:
    # Reduced documentation requirements
    framework_document: "docs/compliance/dora/simplified_framework.md"
    review_frequency: "yearly"

  controls:
    # Minimum required controls
    ict_risk_identification: true
    basic_monitoring: true
    incident_response: true  # Still required
    continuity_arrangements: true

  exemptions:
    # What is NOT required under simplified framework
    full_governance_structure: true
    detailed_testing_programme: true
    complex_documentation: true
```

### Phase 1 Deliverables Summary (CORRECTED)

| Deliverable | File/Path | Tests |
|-------------|-----------|-------|
| DORA Governance Framework | `services/dora/governance.py` | ~40 |
| Digital Resilience Strategy Config | `config/dora/digital_resilience_strategy.yaml` | ~10 |
| ICT Risk Management Framework | `services/dora/ict_risk_framework.py` | ~50 |
| ICT Systems Management | `services/dora/ict_systems.py` | ~30 |
| ICT Identification | `services/dora/ict_identification.py` | ~30 |
| Detection System | `services/dora/detection.py` | ~25 |
| Response and Recovery | `services/dora/response_recovery.py` | ~30 |
| Backup System | `services/dora/backup_recovery.py` | ~20 |
| Learning System | `services/dora/learning.py` | ~15 |
| Communication Plan | `services/dora/communication.py` | ~20 |
| **TOTAL** | | **~250** |

### Phase 1 Test Requirements

```bash
# New test files
tests/
├── dora/
│   ├── test_dora_governance.py
│   ├── test_dora_ict_risk_framework.py
│   ├── test_dora_ict_systems.py
│   ├── test_dora_identification.py
│   ├── test_dora_detection.py
│   ├── test_dora_response_recovery.py
│   ├── test_dora_backup.py
│   ├── test_dora_learning.py
│   └── test_dora_communication.py
```

**Критерии завершения Phase 1**:
- [ ] 100% тестовое покрытие всех новых модулей
- [ ] Интеграционные тесты с существующими AI Act модулями
- [ ] Документация обновлена
- [ ] Все 250+ тестов проходят

---

# Phase 2: ICT Incident Management & Reporting
## Articles 17-23 Implementation

**Приоритет**: P0 (Critical - Regulatory Reporting)
**Зависимости**: Phase 1, существующий `post_market_monitoring.py`

### 2.1 ICT-Related Incident Management Process (Article 17)

**Требования**:
- Process to detect, manage, and notify ICT-related incidents
- Early warning indicators
- Allocation of roles and responsibilities

**Расширение**: `services/ai_act/post_market_monitoring.py`

**Файл**: `services/dora/incident_management.py`

```python
class DORAIncidentManagement:
    """
    Article 17 compliant ICT incident management.

    Extends AI Act IncidentTracker with DORA-specific requirements.
    """

    # Incident management process
    INCIDENT_PHASES = [
        "DETECTION",      # Early warning, automated detection
        "RECORDING",      # Record and log incident
        "CLASSIFICATION", # Classify per Article 18 criteria
        "ESCALATION",     # Internal escalation
        "NOTIFICATION",   # Regulatory notification if major
        "INVESTIGATION",  # Root cause analysis
        "RESOLUTION",     # Corrective actions
        "CLOSURE",        # Post-incident review
    ]

    def detect_ict_incident(self, event: ICTEvent) -> Optional[DORAIncident]:
        """Early detection using indicators per Article 17(3)(a)."""

    def classify_incident(self, incident: DORAIncident) -> IncidentClassification:
        """Classify using Article 18 criteria."""

    def initiate_notification_workflow(self, incident: DORAIncident):
        """Initiate reporting workflow for major incidents."""
```

### 2.2 Classification of ICT-Related Incidents (Article 18)

**Критерии классификации** (Commission Delegated Regulation 2024/1772):

| Criterion | Description | Threshold for Major |
|-----------|-------------|---------------------|
| **Clients affected** | Number/relevance of affected clients | Substantial impact |
| **Duration** | Duration of incident | Material duration |
| **Geographic spread** | Geographic scope | Multiple jurisdictions |
| **Data losses** | Data integrity/confidentiality | Any data loss |
| **Criticality of services** | Critical services affected | Critical services impacted |
| **Economic impact** | Direct/indirect costs | Material financial impact |
| **Reputational impact** | Potential reputational damage | Significant damage |

**Файл**: `services/dora/incident_classification.py`

```python
class DORAIncidentClassification:
    """
    Article 18 compliant incident classification.

    Uses RTS criteria from Commission Delegated Regulation 2024/1772.
    """

    @dataclass
    class ClassificationCriteria:
        clients_affected: int
        client_relevance: str  # retail, professional, counterparty
        duration_hours: float
        geographic_spread: List[str]  # country codes
        data_losses: bool
        data_type: Optional[str]  # personal, confidential, public
        critical_services_affected: bool
        critical_service_names: List[str]
        economic_impact_eur: float
        reputational_impact: str  # low, medium, high

    def classify_incident(
        self,
        incident: DORAIncident,
        criteria: ClassificationCriteria
    ) -> str:  # "MAJOR" or "SIGNIFICANT" or "MINOR"
        """
        Classify incident using Article 18 criteria.

        Major Incident Conditions (any 2 or more):
        1. Critical services affected + malicious access = MAJOR
        2. Material impact on 2+ criteria = MAJOR
        3. Recurring incidents (3+ in 3 months, same cause) = MAJOR
        """
```

**Порогі для Major Incidents** (per RTS):

```yaml
# config/dora/incident_thresholds.yaml
incident_classification:
  major_incident_thresholds:
    # If ANY of these are met, classify as MAJOR
    critical_service_breach:
      affected: true
      malicious_access: true

    # If 2+ of these criteria have material impact
    criteria_thresholds:
      clients_affected:
        retail_count: 5000
        professional_count: 100
        counterparty_count: 10

      duration:
        hours: 4

      geographic_spread:
        countries: 2

      data_losses:
        any: true

      economic_impact:
        eur: 100000

      reputational:
        level: "high"

  # Recurring incidents threshold
  recurring_incident:
    count: 3
    period_months: 3
    same_cause: true
```

### 2.3 Reporting of Major ICT-Related Incidents (Article 19)

**Временные рамки репортинга** (per RTS):

| Report Type | Deadline | Content |
|-------------|----------|---------|
| **Initial Notification** | 4 hours after classification, 24 hours after detection | Basic incident info |
| **Intermediate Report** | 72 hours after initial | Detailed analysis, status |
| **Final Report** | 1 month after incident | Root cause, lessons learned |

**Файл**: `services/dora/incident_reporting.py`

```python
class DORAIncidentReporter:
    """
    Article 19 compliant incident reporting to competent authorities.

    Implements RTS/ITS reporting requirements.
    """

    # Report templates per CDR 2025/301 (RTS) and CIR 2025/302 (ITS)
    # Published OJ 20.02.2025, in force 12.03.2025
    REPORT_TEMPLATES = {
        "initial": "ITS_INITIAL_NOTIFICATION",
        "intermediate": "ITS_INTERMEDIATE_REPORT",
        "final": "ITS_FINAL_REPORT",
    }

    # Deadlines in hours
    DEADLINES = {
        "initial_from_classification": 4,
        "initial_from_detection": 24,
        "intermediate": 72,
        "final_days": 30,
    }

    def generate_initial_notification(
        self,
        incident: DORAIncident,
        classification: IncidentClassification
    ) -> InitialNotification:
        """
        Generate initial notification per ITS Annex I.

        Limited fields to avoid burden during active incident:
        - Incident reference
        - Date/time of detection
        - Date/time of classification
        - Member state of occurrence
        - Brief description
        - Services affected
        - Estimated impact
        """

    def generate_intermediate_report(
        self,
        incident: DORAIncident
    ) -> IntermediateReport:
        """
        Generate intermediate report per ITS Annex II.

        Additional fields:
        - Detailed incident description
        - Root cause analysis (preliminary)
        - Actions taken
        - Ongoing activities
        - Updated impact assessment
        """

    def generate_final_report(
        self,
        incident: DORAIncident
    ) -> FinalReport:
        """
        Generate final report per ITS Annex III.

        Complete information:
        - Full root cause analysis
        - Timeline of events
        - Effectiveness of response
        - Lessons learned
        - Remediation measures
        - Prevention measures
        """

    def submit_to_competent_authority(
        self,
        report: Union[InitialNotification, IntermediateReport, FinalReport],
        authority: CompetentAuthority
    ):
        """Submit report to NCA."""
```

### 2.4 Notification of Significant Cyber Threats (Article 19(4))

**Требования**:
- Voluntary notification of significant cyber threats
- Threats relevant to financial system, users, or clients

**Файл**: `services/dora/cyber_threat_notification.py`

```python
class CyberThreatNotification:
    """
    Article 19(4) - Voluntary cyber threat notification.

    Financial entities may notify significant cyber threats
    when deemed relevant to financial system.
    """

    def assess_threat_significance(self, threat: CyberThreat) -> bool:
        """
        Assess if threat is significant per DORA criteria.

        Significant if:
        - Impact on critical/important business functions
        - Impact on other financial institutions
        - Impact on third parties or clients
        """

    def notify_significant_threat(
        self,
        threat: CyberThreat,
        assessment: ThreatAssessment
    ):
        """Submit voluntary threat notification to NCA."""
```

### 2.5 Harmonised Reporting Templates (Article 20)

**Final Technical Standards** (Published OJ 20.02.2025):
- **CDR 2025/301** - RTS on content and time limits for incident reports
- **CIR 2025/302** - ITS on standard forms, templates, and procedures
- Entry into force: 12 March 2025

**Файл**: `services/dora/reporting_templates.py`

```python
# Implementation of ITS Annex templates

@dataclass
class ITSInitialNotificationTemplate:
    """ITS Annex I - Initial Notification Template."""
    # Mandatory fields
    incident_reference: str
    reporting_entity_lei: str
    detection_datetime: datetime
    classification_datetime: datetime
    member_states_affected: List[str]
    incident_type: str  # cyber_attack, system_failure, etc.
    brief_description: str
    critical_services_affected: List[str]
    estimated_impact: str
    is_recurring: bool

@dataclass
class ITSIntermediateReportTemplate:
    """ITS Annex II - Intermediate Report Template."""
    # All initial fields plus:
    detailed_description: str
    affected_ict_services: List[str]
    affected_clients_count: int
    geographic_spread: List[str]
    data_compromised: bool
    preliminary_root_cause: str
    actions_taken: List[str]
    ongoing_actions: List[str]
    external_support: bool
    estimated_resolution_time: Optional[datetime]

@dataclass
class ITSFinalReportTemplate:
    """ITS Annex III - Final Report Template."""
    # All intermediate fields plus:
    incident_resolved: bool
    resolution_datetime: datetime
    final_root_cause: str
    root_cause_category: str
    full_timeline: List[TimelineEvent]
    total_duration_hours: float
    total_clients_affected: int
    total_economic_impact_eur: float
    data_loss_details: Optional[str]
    response_effectiveness: str
    lessons_learned: List[str]
    remediation_measures: List[str]
    preventive_measures: List[str]
    follow_up_actions: List[str]
```

### 2.6 Centralised Reporting Hub (Article 21)

**Примечание**: ESAs могут создать централизованный EU hub для репортинга. Пока используется direct NCA reporting.

### 2.7 Supervisory Feedback (Article 22)

**Файл**: `services/dora/supervisory_feedback.py`

```python
class SupervisoryFeedbackHandler:
    """
    Handle feedback from competent authorities on incident reports.
    """

    def receive_feedback(self, feedback: SupervisoryFeedback):
        """Process feedback from NCA."""

    def implement_guidance(self, guidance: NCAguidance):
        """Implement NCA guidance on incident handling."""
```

### 2.8 Operational or Security Incidents at Third-Party Providers (Article 23)

**Файл**: `services/dora/third_party_incidents.py`

```python
class ThirdPartyIncidentHandler:
    """
    Article 23 - Handle incidents at ICT third-party providers.

    Critical for our platform as we depend on multiple exchanges.
    """

    def register_provider_incident(
        self,
        provider: str,  # binance, alpaca, etc.
        incident: ProviderIncident
    ):
        """Record incident at third-party provider."""

    def assess_impact_on_operations(
        self,
        provider: str,
        incident: ProviderIncident
    ) -> ImpactAssessment:
        """Assess how provider incident affects our operations."""

    def activate_contingency(
        self,
        provider: str,
        contingency_plan: ContingencyPlan
    ):
        """Activate contingency plan for provider failure."""
```

### Phase 2 Deliverables Summary

| Deliverable | File/Path | Tests |
|-------------|-----------|-------|
| ICT Incident Management | `services/dora/incident_management.py` | ~50 |
| Incident Classification | `services/dora/incident_classification.py` | ~40 |
| Incident Reporting | `services/dora/incident_reporting.py` | ~40 |
| Cyber Threat Notification | `services/dora/cyber_threat_notification.py` | ~20 |
| Reporting Templates (ITS) | `services/dora/reporting_templates.py` | ~30 |
| Supervisory Feedback | `services/dora/supervisory_feedback.py` | ~10 |
| Third-Party Incidents | `services/dora/third_party_incidents.py` | ~20 |
| **TOTAL** | | **~200** |

### Phase 2 Test Requirements

```bash
tests/dora/
├── test_dora_incident_management.py
├── test_dora_incident_classification.py
├── test_dora_incident_reporting.py
├── test_dora_cyber_threat_notification.py
├── test_dora_reporting_templates.py
├── test_dora_supervisory_feedback.py
└── test_dora_third_party_incidents.py
```

**Критерии завершения Phase 2**:
- [ ] All incident classification scenarios covered
- [ ] ITS templates fully implemented
- [ ] Integration with existing IncidentTracker
- [ ] Mock NCA submission workflow tested
- [ ] All 200+ тестов проходят

---

# Phase 3: Digital Resilience Testing
## Articles 24-27 Implementation

**Приоритет**: P1 (High)
**Зависимости**: Phase 1, Phase 2

### 3.1 General Requirements for Testing (Article 24)

**Требования**:
- Sound and comprehensive digital operational resilience testing programme
- Range of assessments, tests, methodologies
- Proportionate to size and risk profile

**Файл**: `services/dora/resilience_testing.py`

```python
class DORAResilienceTestingProgramme:
    """
    Article 24 compliant digital operational resilience testing.

    Testing programme includes:
    1. Vulnerability assessments and scans
    2. Open source analyses
    3. Network security assessments
    4. Gap analyses
    5. Physical security reviews
    6. Questionnaires and scanning software
    7. Source code reviews (where feasible)
    8. Scenario-based tests
    9. Compatibility testing
    10. Performance testing
    11. End-to-end testing
    12. Penetration testing
    """

    TEST_CATEGORIES = {
        "vulnerability_assessment": {
            "frequency": "quarterly",
            "scope": "all_systems",
            "article": "24(1)(a)",
        },
        "network_security": {
            "frequency": "quarterly",
            "scope": "network_infrastructure",
            "article": "24(1)(a)",
        },
        "penetration_testing": {
            "frequency": "yearly",
            "scope": "critical_systems",
            "article": "24(1)(a)",
        },
        "scenario_based": {
            "frequency": "yearly",
            "scope": "business_continuity",
            "article": "24(1)(a)",
        },
        "source_code_review": {
            "frequency": "per_release",
            "scope": "critical_code",
            "article": "24(1)(a)",
        },
    }

    def create_testing_programme(
        self,
        entity_profile: EntityProfile
    ) -> TestingProgramme:
        """Create risk-based testing programme."""

    def execute_testing_cycle(
        self,
        programme: TestingProgramme,
        cycle: str  # quarterly, yearly
    ) -> TestingReport:
        """Execute scheduled testing cycle."""
```

### 3.2 Testing of ICT Tools and Systems (Article 25)

**Требования**:
- Apply testing programme to all ICT systems
- Risk-based approach
- Independent parties for testing

**Файл**: `services/dora/ict_testing.py`

```python
class ICTSystemTesting:
    """
    Article 25 - Testing of all critical ICT systems.
    """

    def identify_systems_for_testing(self) -> List[ICTSystem]:
        """
        Identify all ICT systems requiring testing.

        Priority based on:
        - Criticality of business function
        - Risk profile
        - Recent changes
        """

    def execute_vulnerability_scan(
        self,
        system: ICTSystem
    ) -> VulnerabilityScanResult:
        """Execute automated vulnerability scanning."""

    def execute_penetration_test(
        self,
        system: ICTSystem,
        scope: PentestScope
    ) -> PenetrationTestResult:
        """Execute penetration testing (yearly minimum)."""

    def validate_third_party_interfaces(
        self,
        provider: str
    ) -> InterfaceTestResult:
        """Test interfaces with third-party ICT providers."""
```

### 3.3 Threat-Led Penetration Testing (TLPT) (Article 26)

**Требования**:
- Advanced testing mimicking real threat actors
- At least every 3 years for significant entities
- Live production systems testing
- Based on TIBER-EU framework

**Применимость к нашей платформе**:
- Проверить с NCA, требуется ли TLPT (зависит от размера/системности)
- Для алгоритмической торговли вероятно требуется

**Файл**: `services/dora/tlpt.py`

```python
class ThreatLedPenetrationTesting:
    """
    Article 26 - TLPT (Threat-Led Penetration Testing).

    Based on TIBER-EU framework.

    TLPT Scope per Article 26:
    1. Cover critical or important functions
    2. Performed on live production systems
    3. Every 3 years (or more if NCA requires)
    4. Must include ICT third-party providers (Article 26(4))
    """

    TLPT_PHASES = {
        "preparation": {
            "activities": [
                "scope_definition",
                "threat_intelligence_gathering",
                "scenario_development",
            ],
            "duration_weeks": 4,
        },
        "threat_intelligence": {
            "activities": [
                "targeted_threat_intelligence",
                "scenario_refinement",
                "attack_plan_development",
            ],
            "duration_weeks": 4,
        },
        "red_team_testing": {
            "activities": [
                "attack_simulation",
                "exploitation_attempts",
                "lateral_movement",
                "objective_achievement",
            ],
            "duration_weeks": 8,
        },
        "closure": {
            "activities": [
                "purple_teaming",  # Required by DORA
                "reporting",
                "remediation_planning",
            ],
            "duration_weeks": 4,
        },
    }

    def plan_tlpt_engagement(
        self,
        scope: TLPTScope,
        threat_intelligence: ThreatIntelligence
    ) -> TLPTEngagementPlan:
        """Plan TLPT engagement per TIBER-EU."""

    def validate_testers(
        self,
        testers: List[TLPTTester]
    ) -> TesterValidationResult:
        """
        Validate testers meet Article 27 requirements:
        - Highest suitability and reputability
        - Technical and organisational capabilities
        - Expertise in threat intelligence
        - Expertise in penetration testing
        - Expertise in red team testing
        - Certified by accreditation body
        """

    def conduct_purple_teaming(
        self,
        red_team_results: RedTeamResults,
        blue_team: BlueTeam
    ) -> PurpleTeamReport:
        """
        Mandatory purple teaming per DORA Article 26(5).

        Blue team (defenders) work with red team to:
        - Review attack techniques
        - Understand vulnerabilities exploited
        - Improve detection capabilities
        - Strengthen defenses
        """

    def generate_tlpt_report(
        self,
        engagement: TLPTEngagement
    ) -> TLPTReport:
        """Generate comprehensive TLPT report."""

    def submit_tlpt_attestation(
        self,
        report: TLPTReport,
        authority: CompetentAuthority
    ):
        """Submit TLPT attestation to NCA per Article 26(6)."""
```

### 3.4 Requirements for Testers (Article 27)

**Требования к тестировщикам TLPT**:

| Requirement | Description |
|-------------|-------------|
| **Suitability** | Highest suitability and reputability |
| **Capabilities** | Technical and organisational |
| **Expertise** | Threat intelligence, penetration testing, red team |
| **Certification** | Certified by accreditation body (CREST, OSCP, etc.) |
| **Insurance** | Professional indemnity insurance |
| **Independence** | No conflicts of interest |

**Internal vs External Testers**:
- Internal: Allowed 2 out of 3 TLPTs
- External: Required for 1 out of 3 TLPTs
- Threat intelligence: Always from external party

**Файл**: `services/dora/tester_management.py`

```python
class TLPTTesterManagement:
    """
    Article 27 - Management of TLPT testers.
    """

    REQUIRED_CERTIFICATIONS = [
        "CREST_CRT",
        "CREST_CCT",
        "OSCP",
        "OSCE",
        "GPEN",
        "GWAPT",
    ]

    def validate_tester_qualifications(
        self,
        tester: TLPTTester
    ) -> QualificationValidation:
        """Validate tester meets Article 27 requirements."""

    def check_conflict_of_interest(
        self,
        tester: TLPTTester
    ) -> ConflictCheck:
        """Check for conflicts of interest."""

    def verify_internal_tester_conditions(
        self,
        internal_tester: InternalTester
    ) -> InternalTesterApproval:
        """
        Verify conditions for using internal testers:
        - Approved by NCA
        - No conflict of interest
        - External threat intelligence provider
        """
```

### 3.5 Pooled Testing (Article 26(3))

**Для third-party providers** (важно для наших адаптеров):

```python
class PooledTLPT:
    """
    Article 26(3) - Pooled TLPT for shared ICT services.

    Allows multiple financial entities to jointly test
    shared third-party ICT service providers.
    """

    def organize_pooled_tlpt(
        self,
        participants: List[FinancialEntity],
        shared_provider: ICTProvider
    ) -> PooledTLPTEngagement:
        """Organize pooled TLPT for shared provider."""
```

### Phase 3 Deliverables Summary

| Deliverable | File/Path | Tests |
|-------------|-----------|-------|
| Resilience Testing Programme | `services/dora/resilience_testing.py` | ~50 |
| ICT System Testing | `services/dora/ict_testing.py` | ~40 |
| TLPT Framework | `services/dora/tlpt.py` | ~50 |
| Tester Management | `services/dora/tester_management.py` | ~20 |
| Pooled Testing | `services/dora/pooled_testing.py` | ~20 |
| **TOTAL** | | **~180** |

### Phase 3 Test Requirements

```bash
tests/dora/
├── test_dora_resilience_testing.py
├── test_dora_ict_testing.py
├── test_dora_tlpt.py
├── test_dora_tester_management.py
└── test_dora_pooled_testing.py
```

**Критерии завершения Phase 3**:
- [ ] Testing programme fully documented
- [ ] Vulnerability scanning integrated
- [ ] TLPT framework ready (engagement planning)
- [ ] Tester qualification checks implemented
- [ ] All 180+ тестов проходят

---

# Phase 4: Third-Party ICT Risk Management
## Articles 28-44 Implementation

**Приоритет**: P0 (Critical - Core Platform Dependency)
**Зависимости**: Phase 1

### 4.1 General Principles (Article 28)

**Требования**:
- Manage ICT third-party risk as integral part of ICT risk framework
- Full responsibility remains with financial entity
- Proportionate approach based on nature and criticality

**Критическая важность для платформы**:
Наша платформа зависит от множества ICT third-party providers:

| Provider | Services | Criticality |
|----------|----------|-------------|
| Binance | Market data, Order execution | **CRITICAL** |
| Alpaca | Market data, Order execution | **CRITICAL** |
| Polygon.io | Market data | HIGH |
| OANDA | Forex trading | **CRITICAL** |
| Interactive Brokers | Futures trading | **CRITICAL** |
| Deribit | Crypto options | HIGH |
| Dukascopy | Forex data | MEDIUM |

**Файл**: `services/dora/third_party_risk.py`

```python
class DORAThirdPartyRiskManagement:
    """
    Article 28 - ICT Third-Party Risk Management.

    Manage risk from all ICT service providers.
    """

    def assess_third_party_risk(
        self,
        provider: ICTProvider
    ) -> ThirdPartyRiskAssessment:
        """
        Comprehensive risk assessment including:
        - Criticality of services
        - Concentration risk
        - Substitutability
        - Provider's resilience
        - Geographic location risks
        """

    def maintain_control(
        self,
        provider: ICTProvider
    ) -> ControlAssessment:
        """
        Ensure we maintain control per Article 28(1)(b):
        - Full responsibility remains with us
        - Adequate oversight capabilities
        - No impediment to supervision
        """
```

### 4.2 Register of Information (Article 28(3))

**Критическое требование DORA** - Реестр всех договорных отношений

**Файл**: `services/dora/register_of_information.py`

**ITS Templates структура** (per JC 2023 85):

```python
@dataclass
class RegisterOfInformationEntry:
    """
    Single entry in Register of Information per ITS.

    Template B_02.01 - Contractual arrangement level (DPM 4.0).
    """
    # Contractual arrangement identification
    contractual_arrangement_ref: str
    lei_counterparty: str
    counterparty_name: str

    # Contract details
    contract_type: str  # outsourcing, procurement, intra_group
    start_date: date
    end_date: Optional[date]
    termination_notice_period_days: int

    # Services provided
    services_provided: List[str]
    functions_supported: List[str]
    is_supporting_critical_function: bool

    # Location information
    data_processing_locations: List[str]
    data_storage_locations: List[str]

    # Sub-contracting
    permits_subcontracting: bool
    subcontractors: List[str]

    # Audit rights
    audit_rights_granted: bool
    last_audit_date: Optional[date]

    # Exit strategy
    exit_strategy_documented: bool
    transition_plan_available: bool

class DORARegisterOfInformation:
    """
    Article 28(3) - Register of Information for ICT third-party providers.

    Must be maintained and updated at entity level.
    Report to competent authority yearly (by 30 April per ESA Decision).
    """

    def __init__(self):
        self.entries: Dict[str, RegisterOfInformationEntry] = {}
        self.last_submission_date: Optional[datetime] = None

    def register_provider(
        self,
        provider: ICTProvider,
        contract: ICTContract
    ) -> RegisterOfInformationEntry:
        """Register new ICT third-party provider."""

    def update_entry(
        self,
        entry_id: str,
        updates: Dict[str, Any]
    ):
        """Update existing entry."""

    def classify_criticality(
        self,
        entry_id: str
    ) -> str:  # CRITICAL, IMPORTANT, STANDARD
        """Classify criticality of ICT services."""

    def generate_annual_report(self) -> RegisterReport:
        """
        Generate annual report for NCA.

        Due: 30 April each year (ESA Decision).
        """

    def export_to_its_template(self) -> ITSRegisterTemplate:
        """Export in ITS-compliant format."""
```

**Реализация для наших адаптеров**:

```python
# config/dora/register_of_information.yaml
register_of_information:
  entities:
    - entity_lei: "YOUR_LEI_HERE"
      entity_name: "Your Company Name"

  contractual_arrangements:
    - ref_number: "CA-2025-001"
      provider:
        lei: "BINANCE_LEI"  # Or equivalent identifier
        name: "Binance Holdings Limited"
        country: "MT"  # Malta (European entity)
      contract:
        type: "ICT_SERVICE"
        start_date: "2024-01-01"
        end_date: null  # Ongoing
        notice_period_days: 30
      services:
        - code: "MARKET_DATA"
          description: "Real-time and historical market data"
          critical_function: true
        - code: "ORDER_EXECUTION"
          description: "Order placement and execution"
          critical_function: true
        - code: "ACCOUNT_MANAGEMENT"
          description: "Account balance and position queries"
          critical_function: true
      data:
        processing_locations: ["MT", "SG"]
        storage_locations: ["MT", "SG"]
      subcontracting:
        permitted: true
        subcontractors: []  # Unknown/not disclosed
      audit:
        rights_granted: false  # Per standard terms
        last_audit: null
      exit:
        strategy_documented: true
        transition_plan: true

    - ref_number: "CA-2025-002"
      provider:
        lei: "ALPACA_LEI"
        name: "Alpaca Securities LLC"
        country: "US"
      # ... similar structure
```

### 4.3 Contractual Arrangements (Articles 30)

**Требования Article 30(2)** (basic ICT services):
- Clear description of services
- Locations of data processing
- Service level descriptions
- Assistance in case of incidents
- Termination rights

**Требования Article 30(3)** (critical/important functions):
- All basic requirements plus:
- Full service level agreements (SLAs)
- Notice periods and reporting obligations
- Audit rights for entity and NCA
- Exit strategies and transition support
- Performance targets with remedial actions

**Файл**: `services/dora/contractual_requirements.py`

```python
class DORAContractualRequirements:
    """
    Article 30 - Contractual arrangements with ICT providers.
    """

    # Basic requirements per Article 30(2)
    BASIC_REQUIREMENTS = [
        "clear_service_description",
        "data_processing_locations",
        "data_storage_locations",
        "service_level_descriptions",
        "incident_assistance_obligations",
        "cooperation_with_authorities",
        "termination_rights",
    ]

    # Additional for critical functions per Article 30(3)
    CRITICAL_FUNCTION_REQUIREMENTS = [
        "full_sla_with_targets",
        "notice_periods",
        "reporting_obligations",
        "entity_audit_rights",
        "nca_audit_access",
        "exit_strategy",
        "transition_assistance",
        "performance_remediation",
        "business_continuity",
        "security_measures",
    ]

    def assess_contract_compliance(
        self,
        contract: ICTContract,
        is_critical: bool
    ) -> ContractComplianceReport:
        """Assess if contract meets DORA requirements."""

    def generate_contract_gap_analysis(
        self,
        provider: str
    ) -> ContractGapAnalysis:
        """
        Analyze gaps in existing contract with provider.

        For our exchanges (Binance, Alpaca, etc.):
        - Most use standard terms
        - Audit rights typically limited
        - Exit strategies may need development
        """

    def create_contract_amendment_request(
        self,
        provider: str,
        gaps: List[ContractGap]
    ) -> AmendmentRequest:
        """Create request for contract amendments."""
```

### 4.4 Exit Strategies (Article 28(8))

**Файл**: `services/dora/exit_strategies.py`

```python
class DORAExitStrategy:
    """
    Article 28(8) - Exit strategies for ICT third-party providers.

    Critical for our platform - need exit plans for each exchange.
    """

    def create_exit_plan(
        self,
        provider: str
    ) -> ExitPlan:
        """
        Create exit plan including:
        - Alternative providers identified
        - Data migration procedures
        - Service transition timeline
        - Impact on operations
        - Cost estimates
        """

    def identify_alternatives(
        self,
        provider: str,
        services: List[str]
    ) -> List[AlternativeProvider]:
        """
        Identify alternative providers.

        For example:
        - Binance → Kraken, Coinbase, Bybit
        - Alpaca → Interactive Brokers, TD Ameritrade
        - Polygon → Alpha Vantage, IEX Cloud
        """

    def validate_transition_plan(
        self,
        exit_plan: ExitPlan
    ) -> TransitionValidation:
        """Validate exit/transition plan is feasible."""
```

### 4.5 Concentration Risk (Article 29)

**Файл**: `services/dora/concentration_risk.py`

```python
class DORAConcentrationRisk:
    """
    Article 29 - Preliminary assessment of ICT concentration risk.

    Assess concentration risk at:
    - Entity level
    - Sub-consolidated level
    - Consolidated level
    """

    def assess_concentration_risk(self) -> ConcentrationRiskReport:
        """
        Assess concentration risk across all providers.

        Key questions:
        1. How many critical functions depend on single provider?
        2. What if Binance goes down?
        3. What if API connectivity to all crypto exchanges fails?
        4. Geographic concentration (all providers in same region)?
        """

    def calculate_dependency_metrics(self) -> DependencyMetrics:
        """
        Calculate dependency metrics:
        - % of trades via each provider
        - % of market data from each source
        - Provider redundancy ratio
        """

    def develop_mitigation_measures(
        self,
        concentration_risks: List[ConcentrationRisk]
    ) -> List[MitigationMeasure]:
        """Develop measures to reduce concentration risk."""
```

### 4.6 Oversight Framework for Critical ICT Providers (Articles 31-44)

**Примечание**: ESAs designate Critical Third-Party Providers (CTPPs). Our direct exchanges may not be designated, but cloud providers (AWS, GCP) might be.

**Файл**: `services/dora/ctpp_oversight.py`

```python
class CTPPOversight:
    """
    Handle requirements if using designated Critical Third-Party Providers.

    Check ESA list: https://www.esma.europa.eu/dora
    """

    def check_ctpp_designation(
        self,
        provider: str
    ) -> bool:
        """Check if provider is designated CTPP."""

    def implement_ctpp_requirements(
        self,
        ctpp: CriticalProvider
    ):
        """Implement additional requirements for CTPPs."""
```

### Phase 4 Deliverables Summary

| Deliverable | File/Path | Tests |
|-------------|-----------|-------|
| Third-Party Risk Management | `services/dora/third_party_risk.py` | ~40 |
| Register of Information | `services/dora/register_of_information.py` | ~50 |
| Contractual Requirements | `services/dora/contractual_requirements.py` | ~40 |
| Exit Strategies | `services/dora/exit_strategies.py` | ~30 |
| Concentration Risk | `services/dora/concentration_risk.py` | ~30 |
| CTPP Oversight | `services/dora/ctpp_oversight.py` | ~30 |
| **TOTAL** | | **~220** |

### Phase 4 Configuration

```yaml
# config/dora/third_party_management.yaml
third_party_management:
  register_of_information:
    storage_path: "state/dora/register_of_information"
    annual_reporting:
      due_date_month: 4
      due_date_day: 30
      authority: "YOUR_NCA"

  exit_strategies:
    review_frequency_months: 12
    test_frequency_months: 24

  concentration_risk:
    max_single_provider_critical_functions_pct: 30
    geographic_concentration_limit: 2  # countries

  contract_review:
    review_frequency_months: 12
    critical_contracts_frequency_months: 6
```

### Phase 4 Test Requirements

```bash
tests/dora/
├── test_dora_third_party_risk.py
├── test_dora_register_of_information.py
├── test_dora_contractual_requirements.py
├── test_dora_exit_strategies.py
├── test_dora_concentration_risk.py
└── test_dora_ctpp_oversight.py
```

**Критерии завершения Phase 4**:
- [ ] Register of Information fully populated
- [ ] All provider contracts analyzed
- [ ] Exit strategies documented for all critical providers
- [ ] Concentration risk assessed
- [ ] All 220+ тестов проходят

---

# Phase 5: Information Sharing & Final Integration
## Article 45 + Cross-Regulation Integration

**Приоритет**: P1
**Зависимости**: Phases 1-4

### 5.1 Information Sharing Arrangements (Article 45)

**Требования**:
- Share cyber threat information among trusted communities
- Protect sensitive nature of shared information
- Comply with data protection (GDPR)
- Comply with competition law

**Файл**: `services/dora/information_sharing.py`

```python
class DORAInformationSharing:
    """
    Article 45 - Cyber threat information sharing.

    Participate in trusted information sharing communities.
    """

    # Types of information that can be shared
    SHAREABLE_INFORMATION = [
        "indicators_of_compromise",  # IOCs
        "tactics_techniques_procedures",  # TTPs
        "cybersecurity_alerts",
        "configuration_tools",
    ]

    def join_sharing_community(
        self,
        community: SharingCommunity
    ):
        """
        Join information sharing community.

        Examples: FS-ISAC, CERT-EU, National CSIRTs
        """

    def share_threat_intelligence(
        self,
        threat: CyberThreat,
        community: SharingCommunity
    ):
        """
        Share threat intelligence with community.

        Ensure:
        - Anonymization where needed
        - No business confidential data
        - GDPR compliance for personal data
        """

    def receive_threat_intelligence(
        self,
        intelligence: ThreatIntelligence
    ):
        """Process received threat intelligence."""

    def notify_nca_of_participation(
        self,
        community: SharingCommunity
    ):
        """
        Notify competent authority of community participation.

        Required per Article 45(3).
        """
```

### 5.2 Cross-Regulation Integration

**Интеграция DORA с существующими compliance frameworks**:

| Regulation | Overlap Areas | Integration Points |
|------------|---------------|-------------------|
| **EU AI Act** | Risk management, Logging, Incident handling | Extend existing modules |
| **MiFID II** | BCP, Kill switch, Audit trail | Leverage MiFID II config |
| **GDPR** | Data protection, Breach notification | Data governance alignment |
| **NIS2** | Cybersecurity, Incident reporting | Timing alignment |

**Файл**: `services/dora/cross_regulation.py`

```python
class DORARegulationIntegration:
    """
    Integrate DORA with EU AI Act and MiFID II.
    """

    def align_incident_reporting(self):
        """
        Align DORA incident reporting with:
        - AI Act Article 73 (serious incidents)
        - NIS2 requirements

        Note: DORA 24h/72h timeline aligns with NIS2.
        """

    def integrate_risk_frameworks(self):
        """
        Integrate ICT risk with AI Act risk management:
        - Share risk registry
        - Unified risk assessment
        - Combined reporting
        """

    def align_logging_systems(self):
        """
        Extend AI Act logging for DORA:
        - ICT events
        - Security events
        - Incident logs
        """
```

### 5.3 Final Integration and Orchestration

**Файл**: `services/dora/__init__.py`

```python
"""
DORA Compliance Module for AI-Powered Quantitative Research Platform.

Digital Operational Resilience Act (EU Regulation 2022/2554)

This package provides comprehensive DORA compliance:

Phase 1 - ICT Risk Management Framework (Articles 5-16):
    - governance: Management body oversight, roles
    - ict_risk_framework: ICT risk management
    - ict_systems: ICT asset management
    - ict_identification: Risk identification
    - detection: Anomaly and incident detection
    - response_recovery: BCP and recovery
    - backup_recovery: Backup systems
    - learning: Continuous improvement
    - communication: Crisis communication

Phase 2 - ICT Incident Management (Articles 17-23):
    - incident_management: Incident handling
    - incident_classification: Major incident classification
    - incident_reporting: NCA reporting
    - cyber_threat_notification: Threat notification
    - reporting_templates: ITS templates
    - third_party_incidents: Provider incidents

Phase 3 - Digital Resilience Testing (Articles 24-27):
    - resilience_testing: Testing programme
    - ict_testing: System testing
    - tlpt: Threat-Led Penetration Testing
    - tester_management: TLPT tester requirements

Phase 4 - Third-Party ICT Risk (Articles 28-44):
    - third_party_risk: Provider risk management
    - register_of_information: Article 28(3) register
    - contractual_requirements: Contract compliance
    - exit_strategies: Exit plans
    - concentration_risk: Concentration assessment

Phase 5 - Information Sharing (Article 45):
    - information_sharing: Threat intelligence sharing
    - cross_regulation: EU AI Act, MiFID II integration

Application Date: 17 January 2025
"""

__version__ = "1.0.0"
__dora_compliance_phase__ = 0  # Will increment as phases complete
```

### 5.4 Unified Dashboard and Reporting

**Файл**: `services/dora/compliance_dashboard.py`

```python
class DORAComplianceDashboard:
    """
    Unified DORA compliance monitoring dashboard.
    """

    def get_compliance_status(self) -> ComplianceStatus:
        """Get overall DORA compliance status."""

    def generate_compliance_report(
        self,
        period: str
    ) -> DORAComplianceReport:
        """Generate periodic compliance report."""

    def get_upcoming_deadlines(self) -> List[Deadline]:
        """Get upcoming compliance deadlines."""

    def get_open_issues(self) -> List[ComplianceIssue]:
        """Get open compliance issues."""
```

### Phase 5 Deliverables Summary

| Deliverable | File/Path | Tests |
|-------------|-----------|-------|
| Information Sharing | `services/dora/information_sharing.py` | ~40 |
| Cross-Regulation Integration | `services/dora/cross_regulation.py` | ~30 |
| Compliance Dashboard | `services/dora/compliance_dashboard.py` | ~30 |
| Unified Reporting | `services/dora/unified_reporting.py` | ~25 |
| DORA Module Init | `services/dora/__init__.py` | ~25 |
| **TOTAL** | | **~150** |

---

## Project Directory Structure

```
AI-Powered-Quantitative-Research-Platform/
├── services/
│   ├── ai_act/                    # Existing EU AI Act (1007 tests)
│   │   ├── risk_management.py
│   │   ├── post_market_monitoring.py
│   │   ├── logging_system.py
│   │   ├── cybersecurity.py
│   │   └── ... (15 modules)
│   │
│   └── dora/                      # NEW: DORA compliance (~1015 tests)
│       ├── __init__.py
│       │
│       ├── # Phase 1: ICT Risk Management
│       ├── governance.py
│       ├── ict_risk_framework.py
│       ├── ict_systems.py
│       ├── ict_identification.py
│       ├── detection.py
│       ├── response_recovery.py
│       ├── backup_recovery.py
│       ├── learning.py
│       ├── communication.py
│       │
│       ├── # Phase 2: Incident Management
│       ├── incident_management.py
│       ├── incident_classification.py
│       ├── incident_reporting.py
│       ├── cyber_threat_notification.py
│       ├── reporting_templates.py
│       ├── supervisory_feedback.py
│       ├── third_party_incidents.py
│       │
│       ├── # Phase 3: Resilience Testing
│       ├── resilience_testing.py
│       ├── ict_testing.py
│       ├── tlpt.py
│       ├── tester_management.py
│       ├── pooled_testing.py
│       │
│       ├── # Phase 4: Third-Party Risk
│       ├── third_party_risk.py
│       ├── register_of_information.py
│       ├── contractual_requirements.py
│       ├── exit_strategies.py
│       ├── concentration_risk.py
│       ├── ctpp_oversight.py
│       │
│       └── # Phase 5: Information Sharing
│       ├── information_sharing.py
│       ├── cross_regulation.py
│       ├── compliance_dashboard.py
│       └── unified_reporting.py
│
├── configs/
│   ├── compliance/
│   │   ├── mifid_compliance.yaml  # Existing
│   │   └── ...
│   │
│   └── dora/                      # NEW: DORA configs
│       ├── digital_resilience_strategy.yaml
│       ├── ict_risk_thresholds.yaml
│       ├── incident_thresholds.yaml
│       ├── register_of_information.yaml
│       ├── third_party_management.yaml
│       ├── testing_programme.yaml
│       └── information_sharing.yaml
│
├── docs/
│   └── compliance/
│       ├── EU_AI_ACT_INTEGRATION_PLAN.md   # Existing
│       ├── DORA_INTEGRATION_PLAN.md        # THIS DOCUMENT
│       ├── dora/                            # NEW
│       │   ├── digital_resilience_strategy.md
│       │   ├── ict_business_continuity_policy.md
│       │   ├── exit_strategies/
│       │   ├── register_of_information/
│       │   └── incident_reports/
│       └── ...
│
├── tests/
│   ├── test_ai_act_*.py           # Existing (14 files)
│   │
│   └── dora/                      # NEW: DORA tests (~1015 tests)
│       ├── # Phase 1
│       ├── test_dora_governance.py
│       ├── test_dora_ict_risk_framework.py
│       ├── test_dora_ict_systems.py
│       ├── test_dora_identification.py
│       ├── test_dora_detection.py
│       ├── test_dora_response_recovery.py
│       ├── test_dora_backup.py
│       ├── test_dora_learning.py
│       ├── test_dora_communication.py
│       │
│       ├── # Phase 2
│       ├── test_dora_incident_management.py
│       ├── test_dora_incident_classification.py
│       ├── test_dora_incident_reporting.py
│       ├── test_dora_cyber_threat_notification.py
│       ├── test_dora_reporting_templates.py
│       ├── test_dora_supervisory_feedback.py
│       ├── test_dora_third_party_incidents.py
│       │
│       ├── # Phase 3
│       ├── test_dora_resilience_testing.py
│       ├── test_dora_ict_testing.py
│       ├── test_dora_tlpt.py
│       ├── test_dora_tester_management.py
│       ├── test_dora_pooled_testing.py
│       │
│       ├── # Phase 4
│       ├── test_dora_third_party_risk.py
│       ├── test_dora_register_of_information.py
│       ├── test_dora_contractual_requirements.py
│       ├── test_dora_exit_strategies.py
│       ├── test_dora_concentration_risk.py
│       ├── test_dora_ctpp_oversight.py
│       │
│       └── # Phase 5
│       ├── test_dora_information_sharing.py
│       ├── test_dora_cross_regulation.py
│       ├── test_dora_compliance_dashboard.py
│       └── test_dora_unified_reporting.py
│
└── state/
    └── dora/                      # NEW: DORA runtime state
        ├── register_of_information/
        ├── incidents/
        ├── testing/
        └── reports/
```

---

## Implementation Considerations

### 1. Reuse Strategy

| Existing Component | DORA Reuse | Modification Required |
|-------------------|------------|----------------------|
| `AIActRiskManager` | Extend for ICT risks | Add `DORAICTRiskCategory` |
| `IncidentTracker` | Extend for DORA incidents | Add DORA classification |
| `AIActLogger` | Extend for ICT events | Add ICT event types |
| `AIActCybersecurity` | Extend for DORA | Add continuous monitoring |
| MiFID II BCP config | Leverage directly | Minor extensions |
| Adapters | Document as ICT providers | Create Register of Information |

### 2. Priority Order

```
Phase 4 (Third-Party Risk) → Can start immediately (Register of Information)
Phase 1 (ICT Risk Management) → Core framework
Phase 2 (Incident Reporting) → Regulatory deadline sensitive
Phase 3 (Resilience Testing) → Requires framework first
Phase 5 (Integration) → Final integration
```

**Рекомендация**: Начать с Phase 4 параллельно с Phase 1, так как Register of Information является независимым требованием и имеет deadline (30 April 2025).

### 3. RTS/ITS Compliance

| Standard | Status | Implementation |
|----------|--------|----------------|
| RTS on ICT Risk Management | Final (CDR 2024/1774) | Phase 1 |
| RTS on Incident Classification | Final (CDR 2024/1772) | Phase 2 |
| RTS on Incident Reporting | Final (CDR 2025/301) | Phase 2 |
| ITS on Incident Reporting Templates | Final (CIR 2025/302) | Phase 2 |
| RTS on Third-Party Policy | Final (CDR 2024/1773) | Phase 4 |
| ITS on Register of Information | Final (CIR 2024/2956, based on JC 2023 85) | Phase 4 |
| RTS on TLPT | Final (OJ L 2025/1190) | Phase 3 |
| RTS on Subcontracting | Final (CDR 2025/532) | Phase 4 |

### 4. Testing Strategy

Для каждой фазы:
1. Unit tests для каждого модуля (>95% coverage)
2. Integration tests с существующими модулями
3. End-to-end compliance validation tests
4. Mock NCA submission tests (для incident reporting)

---

## References & Sources

### Official EU Sources
- [DORA Full Text (EUR-Lex)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R2554)
- [ESMA DORA Page](https://www.esma.europa.eu/esmas-activities/digital-finance-and-innovation/digital-operational-resilience-act-dora)
- [EBA DORA Technical Standards](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/operational-resilience)
- [EIOPA DORA Page](https://www.eiopa.europa.eu/digital-operational-resilience-act-dora_en)

### Technical Standards (RTS/ITS) — Final Regulations
- [CDR 2024/1774 - RTS on ICT Risk Management Framework](https://eur-lex.europa.eu/eli/reg_del/2024/1774/oj/eng)
- [CDR 2024/1772 - RTS on Incident Classification](https://eur-lex.europa.eu/eli/reg_del/2024/1772/oj/eng)
- [CDR 2024/1773 - RTS on Third-Party Policy](https://eur-lex.europa.eu/eli/reg_del/2024/1773/oj/eng)
- [CDR 2025/301 - RTS on Incident Reporting Content](https://eur-lex.europa.eu/eli/reg_del/2025/301/oj/eng)
- [CIR 2025/302 - ITS on Incident Reporting Templates](https://eur-lex.europa.eu/eli/reg_impl/2025/302/oj/eng)
- [ITS on Register of Information (CIR 2024/2956)](https://eur-lex.europa.eu/eli/reg_impl/2024/2956/oj/eng)
- [CDR 2025/532 - RTS on Subcontracting](https://eur-lex.europa.eu/eli/reg_del/2025/532/oj/eng)
- [RTS on TLPT (OJ L 2025/1190)](https://eur-lex.europa.eu/eli/reg_del/2025/1190/oj/eng)

### DORA Articles Reference
- [Article 5-11: ICT Risk Management](https://www.digital-operational-resilience-act.com/DORA_Articles.html)
- [Article 11: Response and Recovery](https://www.digital-operational-resilience-act.com/Article_11.html)
- [Article 12: Backup Policies](https://www.digital-operational-resilience-act.com/Article_12.html)
- [Article 17-23: Incident Reporting](https://www.digital-operational-resilience-act.com/DORA_Articles.html)
- [Article 24-27: Digital Resilience Testing](https://www.digital-operational-resilience-act.com/DORA_Articles.html)
- [Article 28: Third-Party Risk](https://www.digital-operational-resilience-act.com/Article_28.html)
- [Article 45: Information Sharing](https://www.digital-operational-resilience-act.com/Article_45.html)

### Implementation Guides
- [FS-ISAC DORA Implementation Guidance](https://www.fsisac.com/hubfs/Knowledge/DORA/FSISAC_DORA-ImplementationGuidance.pdf)
- [IBM DORA Overview](https://www.ibm.com/think/topics/digital-operational-resilience-act)
- [IT Governance DORA Guide](https://www.itgovernanceusa.com/eu-digital-operational-resilience-act)

---

## Summary (CORRECTED v3.0)

| Metric | v1.0 | v2.0 | v3.0 |
|--------|------|------|------|
| **Total Phases** | 5 | 6 | **7** (added Phase -1: Scope) |
| **Total New Modules** | ~30 | ~25 | **~30** (added RTS controls) |
| **Total New Tests** | ~1000 | 685 | **~1015** (unified with deliverables) |
| **Articles Covered** | Incomplete | Partially | **Complete + RTS 2024/1774** |
| **Scope Verification** | None | None | **Article 2 check required** |
| **Critical Functions** | None | None | **Article 3(22) classification** |
| **NCA Identification** | Placeholder | Placeholder | **By country + platform** |
| **ITS Format** | None | Mentioned | **CSV + DPM 4.0 validation** |
| **LEI Handling** | Assumed | Assumed | **Non-LEI alternatives** |
| **RTS Controls** | None | None | **33 articles mapped** |

### Corrected Risk Assessment

| Risk | Original Assessment | Corrected Assessment |
|------|---------------------|----------------------|
| Contract amendments | "Need amendments" | ❌ Impossible — use compensating controls |
| TLPT | "Required" | ⚠️ Probably NOT required — verify with NCA |
| Art. 31-44 | "Must implement" | ✅ Not our requirement — ESA oversight framework |
| Register deadline | Noted | 🔴 **Critical: 30 April 2025** |
| Proportionality | Not considered | Must assess FIRST — may reduce scope significantly |

### Corrected Next Steps

**IMMEDIATE (Phase 0)**:
1. ⏰ Determine entity classification (microenterprise/small/standard)
2. ⏰ Document applicable DORA regime
3. ⏰ If simplified framework applies — reduce implementation scope

**SHORT-TERM (Phases 4 + 1 parallel)**:
1. 🔴 Register of Information — complete by **March 2025** for April submission
2. Populate all 15 ITS templates
3. Document gaps in third-party contracts (don't expect amendments)
4. ICT Risk Management Framework (scope per Phase 0 result)

**MEDIUM-TERM (Phases 2 + 3)**:
1. Incident classification with 2024/1772 thresholds
2. ITS reporting templates ready
3. Basic testing programme (TLPT only if designated)

**Sources**:
- [DORA Full Text](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R2554)
- [Article 16 Simplified Framework](https://www.digital-operational-resilience-act.com/Article_16.html)
- [Commission Delegated Regulation 2024/1772](https://eur-lex.europa.eu/eli/reg_del/2024/1772/oj/eng)
- [ITS Register of Information JC 2023 85](https://www.esma.europa.eu/sites/default/files/2024-01/JC_2023_85_-_Final_report_on_draft_ITS_on_Register_of_Information.pdf)
- [RTS on ICT Risk Management](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/operational-resilience)

---

**Document Version History**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-08 | Claude | Initial comprehensive plan |
| 2.0.0 | 2025-12-08 | Claude | **Critical corrections**: Added Phase 0 (Proportionality), Added Articles 15-16, Fixed incident thresholds (Reg. 2024/1772), Fixed Register of Information ITS structure (15 templates), Corrected Articles 31-44 scope, Added realistic third-party contract assessment, Fixed TLPT applicability, Reduced test count to mapped estimates |
| 3.0.0 | 2025-12-08 | Claude | **Second critical review**: Added Phase -1 (DORA Scope Verification per Article 2), Added Critical/Important Function classification (Article 3(22)), Fixed incident timeline (4h OR 24h whichever EARLIER), Added NCA identification by country + submission platforms, Added ITS export format requirements (CSV + DPM 4.0), Added LEI handling for non-EU providers (alternative identifiers), Fixed microenterprise definition (OR not AND), Added full RTS JC 2023 86 control mapping (33 articles), Added country-specific submission deadlines |
| 4.0.0 | 2025-12-08 | Claude | **Independent audit corrections**: (1) Fixed Article 2(1)(b)→2(1)(e) for investment firms, (2) Updated CTPP section with 19 designated CTPPs (AWS, Google, Microsoft, etc. per 19 Nov 2025 ESA decision), (3) Updated incident reporting refs JC 2024-33→CDR 2025/301 + CIR 2025/302, (4) Fixed ITS template prefixes RT→B per DPM 4.0, (5) Fixed client threshold 10K→100K per RTS Art.9, (6) Fixed country deadlines (Germany: 11 Apr, France: 15 Apr), (7) Added reference date 31 March 2025, (8) Unified test counts to ~1015, (9) Added weekend/holiday extension and incident upgrade procedures per CDR 2025/301, (10) Updated all RTS/ITS references to final regulations |

---

## Приложение A: Фундаментальные вопросы перед началом (v3.0 NEW)

> **СТОП**: Ответьте на эти вопросы ПЕРЕД любой реализацией.

### A.1 DORA Scope Check

| Вопрос | Ответ | Implications |
|--------|-------|--------------|
| Является ли entity лицензированной финансовой организацией? | ❓ | Если НЕТ → DORA может не применяться |
| Какой тип entity по Article 2(1)? | ❓ | Определяет NCA и requirements |
| В какой Member State авторизация? | ❓ | Определяет NCA |
| Есть ли LEI? | ❓ | Если НЕТ → нужно получить |

### A.2 Entity Classification

| Вопрос | Ответ | Implications |
|--------|-------|--------------|
| Количество сотрудников? | ❓ | <10 → microenterprise exemptions |
| Годовой оборот (EUR)? | ❓ | <€2M → microenterprise |
| Баланс (EUR)? | ❓ | <€2M → microenterprise |
| Small non-interconnected investment firm? | ❓ | → Article 16 simplified |

### A.3 Critical Functions

| Функция | Critical? | Providers | Article 30(3) applies? |
|---------|-----------|-----------|------------------------|
| Order Execution | ❓ | ❓ | ❓ |
| Market Data | ❓ | ❓ | ❓ |
| Risk Monitoring | ❓ | ❓ | ❓ |
| Reporting | ❓ | ❓ | ❓ |

**Без ответов на эти вопросы дальнейшая реализация бессмысленна.**


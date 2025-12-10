# -*- coding: utf-8 -*-
"""
MiFID II Compliance Policy Templates.

This module provides complete policy document templates for MiFID II compliance:

- Best Execution Policy (Article 27)
- Order Handling Policy (Article 28)
- Conflicts of Interest Policy (Article 23)
- Kill Switch Procedures (RTS 6 Article 12)
- Transaction Reporting Policy (RTS 22)

These templates are designed to be used with the governance.py module
for centralized policy management.

References:
    - MiFID II (Directive 2014/65/EU)
    - MiFIR (Regulation 600/2014)
    - RTS 6 (Regulation 2017/589)
    - RTS 22 (Regulation 2017/590)
    - ESMA Best Execution Guidelines (ESMA35-43-620)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, List

from .governance import (
    PolicyDocument,
    PolicySection,
    PolicyType,
    PolicyStatus,
    ApprovalLevel,
    ReviewFrequency,
)


# =============================================================================
# Best Execution Policy (MiFID II Article 27)
# =============================================================================


def create_best_execution_policy(
    firm_lei: str,
    firm_name: str,
    owner: str,
    factor_weights: Optional[dict] = None,
) -> PolicyDocument:
    """
    Create Best Execution Policy per MiFID II Article 27.

    This policy documents how the firm achieves best execution
    for order execution.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        owner: Policy owner.
        factor_weights: Optional custom factor weights.

    Returns:
        PolicyDocument for best execution.
    """
    if factor_weights is None:
        factor_weights = {
            "price": 35,
            "cost": 25,
            "speed": 15,
            "likelihood": 15,
            "settlement": 5,
            "size": 3,
            "nature": 2,
        }

    policy = PolicyDocument(
        policy_type=PolicyType.BEST_EXECUTION,
        title="Best Execution Policy",
        description=(
            "This policy establishes the framework for achieving best execution "
            f"at {firm_name} in compliance with MiFID II Article 27. The policy "
            "defines how execution quality is measured, monitored, and reported."
        ),
        owner=owner,
        firm_lei=firm_lei,
        firm_name=firm_name,
        approval_level=ApprovalLevel.SENIOR_MANAGEMENT,
        regulatory_references=[
            "MiFID II Article 27: Obligation to execute orders on terms most favourable to clients",
            "ESMA Best Execution Guidelines (ESMA35-43-620)",
            "RTS 27 (suspended) - Execution venue quality data",
            "RTS 28 (repealed) - Annual execution quality reports",
        ],
    )

    # Section 1: Introduction and Scope
    policy.add_section(PolicySection(
        section_id="1",
        title="Introduction and Scope",
        content=(
            f"This Best Execution Policy applies to all order execution activities "
            f"conducted by {firm_name}.\n\n"
            "Per MiFID II Article 27, the firm must take all sufficient steps to obtain "
            "the best possible result for clients taking into account price, costs, speed, "
            "likelihood of execution and settlement, size, nature or any other consideration "
            "relevant to the execution of the order.\n\n"
            "For proprietary trading, this policy ensures execution quality is systematically "
            "monitored and optimized in line with the firm's trading objectives."
        ),
        regulatory_reference="MiFID II Article 27(1)",
    ))

    # Section 2: Execution Factors
    policy.add_section(PolicySection(
        section_id="2",
        title="Execution Factors and Weighting",
        content=(
            "The following factors are considered when executing orders, with the indicated "
            "relative importance (weights must sum to 100%):\n\n"
            f"1. PRICE ({factor_weights['price']}%)\n"
            "   The actual execution price achieved relative to the market.\n\n"
            f"2. COST ({factor_weights['cost']}%)\n"
            "   Total cost of execution including explicit costs (commission, fees) "
            "and implicit costs (spread, market impact).\n\n"
            f"3. SPEED ({factor_weights['speed']}%)\n"
            "   The time taken to execute the order from submission to fill.\n\n"
            f"4. LIKELIHOOD ({factor_weights['likelihood']}%)\n"
            "   The probability of achieving full execution.\n\n"
            f"5. SETTLEMENT ({factor_weights['settlement']}%)\n"
            "   The certainty and timing of settlement.\n\n"
            f"6. SIZE ({factor_weights['size']}%)\n"
            "   The ability to execute the full order size without significant "
            "market impact.\n\n"
            f"7. NATURE ({factor_weights['nature']}%)\n"
            "   Considerations specific to the order type and trading strategy."
        ),
        regulatory_reference="MiFID II Article 27(1)",
    ))

    # Section 3: Execution Venues
    policy.add_section(PolicySection(
        section_id="3",
        title="Execution Venues",
        content=(
            "The firm maintains a list of approved execution venues ranked by "
            "execution quality. Venues are assessed on:\n\n"
            "- Average spread\n"
            "- Execution latency\n"
            "- Fill rate\n"
            "- Price improvement achieved\n"
            "- Total cost of execution\n"
            "- Market share and liquidity\n\n"
            "Venue rankings are reviewed at least annually and updated based on "
            "execution quality metrics. Significant changes in venue performance "
            "trigger ad-hoc reviews.\n\n"
            "Venue Types:\n"
            "- Regulated Markets (RM)\n"
            "- Multilateral Trading Facilities (MTF)\n"
            "- Organised Trading Facilities (OTF)\n"
            "- Systematic Internalisers (SI)\n"
            "- Market Makers"
        ),
        regulatory_reference="MiFID II Article 27(2)",
    ))

    # Section 4: Monitoring
    policy.add_section(PolicySection(
        section_id="4",
        title="Execution Quality Monitoring",
        content=(
            "Execution quality is monitored continuously through:\n\n"
            "1. Real-time monitoring of execution metrics\n"
            "2. Daily execution quality reports\n"
            "3. Monthly aggregate analysis\n"
            "4. Quarterly venue performance reviews\n"
            "5. Annual policy review\n\n"
            "Key Metrics Monitored:\n"
            "- Price improvement vs. reference price\n"
            "- Total execution cost (bps)\n"
            "- Execution latency (ms)\n"
            "- Fill rate (%)\n"
            "- Market impact (bps)\n\n"
            "Alerts are generated when execution quality falls below thresholds."
        ),
        regulatory_reference="MiFID II Article 27(7)",
    ))

    # Section 5: Review
    policy.add_section(PolicySection(
        section_id="5",
        title="Policy Review",
        content=(
            "This policy is reviewed at least annually, or more frequently if:\n\n"
            "- Material changes in market structure occur\n"
            "- Significant changes in execution quality are observed\n"
            "- New execution venues become available\n"
            "- Regulatory requirements change\n\n"
            "Each review assesses:\n"
            "- Effectiveness of current execution arrangements\n"
            "- Appropriateness of factor weightings\n"
            "- Accuracy of venue rankings\n"
            "- Compliance with regulatory requirements\n\n"
            "Review findings are documented and presented to senior management."
        ),
        regulatory_reference="MiFID II Article 27(7)",
    ))

    return policy


# =============================================================================
# Order Handling Policy (MiFID II Article 28)
# =============================================================================


def create_order_handling_policy(
    firm_lei: str,
    firm_name: str,
    owner: str,
) -> PolicyDocument:
    """
    Create Order Handling Policy per MiFID II Article 28.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        owner: Policy owner.

    Returns:
        PolicyDocument for order handling.
    """
    policy = PolicyDocument(
        policy_type=PolicyType.ORDER_HANDLING,
        title="Order Handling Policy",
        description=(
            "This policy establishes procedures for handling orders at "
            f"{firm_name} in compliance with MiFID II Article 28 client order "
            "handling rules."
        ),
        owner=owner,
        firm_lei=firm_lei,
        firm_name=firm_name,
        approval_level=ApprovalLevel.HEAD_OF_TRADING,
        regulatory_references=[
            "MiFID II Article 28: Client order handling rules",
            "MiFID II Article 27: Best execution obligations",
        ],
    )

    policy.add_section(PolicySection(
        section_id="1",
        title="Order Execution Sequence",
        content=(
            "Orders are executed in a fair and expedient manner:\n\n"
            "1. Orders are time-stamped upon receipt to nanosecond precision\n"
            "2. Comparable orders are executed sequentially unless characteristics "
            "   of the order or market conditions make this impractical\n"
            "3. Limit orders that cannot be immediately executed are registered "
            "   unless the client instructs otherwise\n"
            "4. The firm maintains systems to ensure orderly execution"
        ),
        regulatory_reference="MiFID II Article 28(1)",
    ))

    policy.add_section(PolicySection(
        section_id="2",
        title="Order Aggregation",
        content=(
            "The firm may aggregate orders under the following conditions:\n\n"
            "1. Aggregation is unlikely to work to the disadvantage of any order\n"
            "2. Fair allocation policies are in place and followed\n"
            "3. Each order and its allocation is documented\n"
            "4. If aggregated order is only partially filled, allocation is "
            "   proportional unless specific rules apply\n"
            "5. Proprietary orders aggregated with client orders are allocated "
            "   on terms no more favorable to the firm"
        ),
        regulatory_reference="MiFID II Article 28(2)",
    ))

    policy.add_section(PolicySection(
        section_id="3",
        title="Limit Order Handling",
        content=(
            "For limit orders in shares traded on regulated markets:\n\n"
            "1. Orders not immediately executable are registered\n"
            "2. Orders are published or made known to the market unless:\n"
            "   - Client instructs otherwise, or\n"
            "   - Order size is large relative to normal market size\n"
            "3. Large orders may be held in reserve to minimize market impact\n"
            "4. All handling decisions are documented in the audit trail"
        ),
        regulatory_reference="MiFID II Article 28(2)",
    ))

    policy.add_section(PolicySection(
        section_id="4",
        title="Fair Allocation",
        content=(
            "When orders are aggregated, the following allocation rules apply:\n\n"
            "1. Pre-determined allocation rules are documented\n"
            "2. Partial fills are allocated proportionally\n"
            "3. No preferential treatment for proprietary orders\n"
            "4. Allocation decisions are recorded with rationale\n"
            "5. Clients can request allocation documentation"
        ),
        regulatory_reference="MiFID II Article 28",
    ))

    return policy


# =============================================================================
# Conflicts of Interest Policy (MiFID II Article 23)
# =============================================================================


def create_conflicts_of_interest_policy(
    firm_lei: str,
    firm_name: str,
    owner: str,
) -> PolicyDocument:
    """
    Create Conflicts of Interest Policy per MiFID II Article 23.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        owner: Policy owner.

    Returns:
        PolicyDocument for conflicts of interest.
    """
    policy = PolicyDocument(
        policy_type=PolicyType.CONFLICTS_OF_INTEREST,
        title="Conflicts of Interest Policy",
        description=(
            "This policy establishes procedures for identifying, preventing, and "
            f"managing conflicts of interest at {firm_name} in compliance with "
            "MiFID II Article 23."
        ),
        owner=owner,
        firm_lei=firm_lei,
        firm_name=firm_name,
        approval_level=ApprovalLevel.SENIOR_MANAGEMENT,
        regulatory_references=[
            "MiFID II Article 23: Conflicts of interest",
            "MiFID II Delegated Regulation Articles 33-35",
        ],
    )

    policy.add_section(PolicySection(
        section_id="1",
        title="Identification of Conflicts",
        content=(
            "The firm maintains procedures to identify conflicts of interest:\n\n"
            "Conflicts may arise when the firm or a connected person:\n"
            "- Is likely to make a financial gain at the client's expense\n"
            "- Has an interest in the outcome different from the client's\n"
            "- Has incentive to favor one client over another\n"
            "- Carries on the same business as the client\n"
            "- Receives inducement from a third party\n\n"
            "For algorithmic trading, specific conflicts include:\n"
            "- Proprietary trading vs. client order execution\n"
            "- Venue selection incentives\n"
            "- Information advantages from order flow analysis"
        ),
        regulatory_reference="MiFID II Article 23(1)",
    ))

    policy.add_section(PolicySection(
        section_id="2",
        title="Prevention Measures",
        content=(
            "The following measures prevent conflicts:\n\n"
            "1. Information barriers between trading desks\n"
            "2. Separation of remuneration from conflicted activities\n"
            "3. Independent oversight of conflicted activities\n"
            "4. Physical separation of personnel where appropriate\n"
            "5. Best execution obligations take precedence\n"
            "6. Venue selection based on objective criteria only"
        ),
        regulatory_reference="MiFID II Article 23(2)",
    ))

    policy.add_section(PolicySection(
        section_id="3",
        title="Management of Residual Conflicts",
        content=(
            "Where conflicts cannot be fully prevented:\n\n"
            "1. The nature and source of conflict is disclosed\n"
            "2. Disclosure is made before undertaking activity\n"
            "3. Disclosure is sufficiently specific to enable informed decision\n"
            "4. Disclosure is made in a durable medium\n"
            "5. Alternative arrangements are offered where possible"
        ),
        regulatory_reference="MiFID II Article 23(3)",
    ))

    policy.add_section(PolicySection(
        section_id="4",
        title="Conflicts Register",
        content=(
            "A Conflicts of Interest Register is maintained containing:\n\n"
            "1. All identified conflicts of interest\n"
            "2. Assessment of potential impact\n"
            "3. Mitigating controls in place\n"
            "4. Any disclosures made\n"
            "5. Review dates and outcomes\n\n"
            "The register is reviewed at least annually by Compliance."
        ),
        regulatory_reference="MiFID II Article 23",
    ))

    return policy


# =============================================================================
# Kill Switch Procedures (RTS 6 Article 12)
# =============================================================================


def create_kill_switch_policy(
    firm_lei: str,
    firm_name: str,
    owner: str,
) -> PolicyDocument:
    """
    Create Kill Switch Procedures per RTS 6 Article 12.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        owner: Policy owner.

    Returns:
        PolicyDocument for kill switch procedures.
    """
    policy = PolicyDocument(
        policy_type=PolicyType.KILL_SWITCH,
        title="Kill Switch Procedures",
        description=(
            "This document establishes procedures for the kill switch mechanism at "
            f"{firm_name} in compliance with RTS 6 Article 12 requirements for "
            "emergency trading cessation."
        ),
        owner=owner,
        firm_lei=firm_lei,
        firm_name=firm_name,
        approval_level=ApprovalLevel.HEAD_OF_TRADING,
        regulatory_references=[
            "RTS 6 Article 12: Kill functionality",
            "MiFID II Article 17: Algorithmic trading",
        ],
    )

    policy.add_section(PolicySection(
        section_id="1",
        title="Kill Switch Capability",
        content=(
            "The kill switch provides the ability to:\n\n"
            "1. Cancel immediately, as an emergency measure, any or all "
            "   unexecuted orders submitted by algorithms\n"
            "2. Halt the submission of new orders\n"
            "3. Apply to specific algorithms, instruments, or venues\n"
            "4. Apply globally to all algorithmic trading activity\n\n"
            "The kill switch must be operable 24/7 by authorized personnel."
        ),
        regulatory_reference="RTS 6 Article 12(1)",
    ))

    policy.add_section(PolicySection(
        section_id="2",
        title="Activation Criteria",
        content=(
            "The kill switch SHALL be activated when:\n\n"
            "AUTOMATIC TRIGGERS:\n"
            "- System failure affecting order management\n"
            "- Loss of market data for extended period\n"
            "- Position limits exceeded\n"
            "- Daily loss limits exceeded\n"
            "- Order-to-trade ratio exceeds emergency threshold\n"
            "- Algorithm exhibits erroneous behavior\n\n"
            "MANUAL TRIGGERS:\n"
            "- Compliance or Risk instruction\n"
            "- Exchange request or trading halt\n"
            "- Cybersecurity incident\n"
            "- Any situation posing market integrity risk"
        ),
        regulatory_reference="RTS 6 Article 12",
    ))

    policy.add_section(PolicySection(
        section_id="3",
        title="Authorized Personnel",
        content=(
            "The following roles are authorized to activate the kill switch:\n\n"
            "1. Head of Trading\n"
            "2. Trading Desk Manager (on-call)\n"
            "3. Chief Risk Officer\n"
            "4. Compliance Officer\n"
            "5. Designated IT Operations staff\n\n"
            "A 24/7 contact list is maintained and tested monthly.\n"
            "Compliance staff can contact any of these individuals to request "
            "kill switch activation per RTS 6 Article 12(2)."
        ),
        regulatory_reference="RTS 6 Article 12(2)",
    ))

    policy.add_section(PolicySection(
        section_id="4",
        title="Activation Procedure",
        content=(
            "STEP 1: IDENTIFY ISSUE\n"
            "- Confirm nature and scope of the problem\n"
            "- Determine if kill switch is appropriate response\n\n"
            "STEP 2: ACTIVATE KILL SWITCH\n"
            "- Use kill switch interface or emergency command\n"
            "- Specify scope: algorithm, venue, instrument, or global\n"
            "- Confirm cancellation orders sent to venues\n\n"
            "STEP 3: VERIFY\n"
            "- Confirm all target orders cancelled\n"
            "- Verify no new orders being submitted\n"
            "- Document current positions\n\n"
            "STEP 4: NOTIFY\n"
            "- Notify compliance immediately\n"
            "- Notify senior management\n"
            "- Notify venues if required\n"
            "- Log all details in incident record"
        ),
        regulatory_reference="RTS 6 Article 12",
    ))

    policy.add_section(PolicySection(
        section_id="5",
        title="Recovery Procedure",
        content=(
            "Before resuming trading after kill switch activation:\n\n"
            "1. Root cause analysis completed\n"
            "2. Issue resolved or adequate workaround in place\n"
            "3. System stability verified\n"
            "4. Positions reconciled\n"
            "5. Risk limits reviewed and reset if needed\n"
            "6. Compliance approval obtained\n"
            "7. Recovery documented in incident log\n\n"
            "Trading resumes with enhanced monitoring for initial period."
        ),
        regulatory_reference="RTS 6 Article 12",
    ))

    policy.add_section(PolicySection(
        section_id="6",
        title="Testing Requirements",
        content=(
            "The kill switch must be tested:\n\n"
            "1. REGULAR TESTING: At least quarterly\n"
            "2. AFTER CHANGES: Following any system modifications\n"
            "3. NEW ALGORITHMS: Before any new algorithm goes live\n"
            "4. NEW VENUES: Before connecting to new venues\n\n"
            "Tests verify:\n"
            "- All order types can be cancelled\n"
            "- Cancellation reaches all venues\n"
            "- Response time meets requirements\n"
            "- Logging and notification function correctly"
        ),
        regulatory_reference="RTS 6 Article 12",
    ))

    return policy


# =============================================================================
# Transaction Reporting Policy (RTS 22)
# =============================================================================


def create_transaction_reporting_policy(
    firm_lei: str,
    firm_name: str,
    owner: str,
) -> PolicyDocument:
    """
    Create Transaction Reporting Policy per RTS 22.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        owner: Policy owner.

    Returns:
        PolicyDocument for transaction reporting.
    """
    policy = PolicyDocument(
        policy_type=PolicyType.TRANSACTION_REPORTING,
        title="Transaction Reporting Policy",
        description=(
            "This policy establishes procedures for transaction reporting at "
            f"{firm_name} in compliance with MiFIR Article 26 and RTS 22 "
            "requirements."
        ),
        owner=owner,
        firm_lei=firm_lei,
        firm_name=firm_name,
        approval_level=ApprovalLevel.COMPLIANCE_OFFICER,
        regulatory_references=[
            "MiFIR Article 26: Transaction reporting",
            "RTS 22 (Regulation 2017/590): Transaction reporting content",
            "ESMA Transaction Reporting Guidelines",
        ],
    )

    policy.add_section(PolicySection(
        section_id="1",
        title="Reporting Obligation",
        content=(
            "The firm must report all transactions in financial instruments to "
            "the National Competent Authority (NCA).\n\n"
            "Reportable transactions include:\n"
            "- Purchase or sale of financial instruments\n"
            "- Acquisitions and disposals\n"
            "- Transactions executed on trading venues\n"
            "- OTC transactions in instruments admitted to trading\n\n"
            "Reports must be submitted within T+1 (by close of business on "
            "the working day following execution)."
        ),
        regulatory_reference="MiFIR Article 26(1)",
    ))

    policy.add_section(PolicySection(
        section_id="2",
        title="Report Content",
        content=(
            "Transaction reports must include 65 data fields per RTS 22:\n\n"
            "KEY FIELDS:\n"
            "- Transaction reference number\n"
            "- Trading venue (MIC code)\n"
            "- Instrument identification (ISIN)\n"
            "- Buyer/Seller LEI\n"
            "- Trading date and time (UTC)\n"
            "- Quantity and price\n"
            "- Trading capacity (DEAL, MTCH, AOTC)\n"
            "- Short selling indicator\n"
            "- Algorithm ID (if applicable)\n\n"
            "All mandatory fields must be completed accurately."
        ),
        regulatory_reference="RTS 22 Annex I",
    ))

    policy.add_section(PolicySection(
        section_id="3",
        title="LEI Requirements",
        content=(
            "'No LEI, No Trade' - the firm will not execute transactions without "
            "valid LEI identification.\n\n"
            "Requirements:\n"
            "1. Firm's own LEI must be valid and renewed annually\n"
            "2. Counterparty LEI verified before execution (where required)\n"
            "3. LEI status checked via GLEIF before reporting\n"
            "4. LEI renewal reminders set 30 days before expiry"
        ),
        regulatory_reference="RTS 22 Article 4",
    ))

    policy.add_section(PolicySection(
        section_id="4",
        title="Approved Reporting Mechanism (ARM)",
        content=(
            "Reports are submitted via an Approved Reporting Mechanism (ARM).\n\n"
            "ARM Requirements:\n"
            "1. ARM must be authorized by competent authority\n"
            "2. Connection maintained and monitored\n"
            "3. Failed submissions logged and retried\n"
            "4. Confirmation of successful submission obtained\n"
            "5. Backup procedures for ARM unavailability\n\n"
            "Report submission is monitored for T+1 compliance."
        ),
        regulatory_reference="MiFIR Article 26(7)",
    ))

    policy.add_section(PolicySection(
        section_id="5",
        title="Error Handling",
        content=(
            "Procedures for handling reporting errors:\n\n"
            "1. FAILED SUBMISSIONS\n"
            "   - Automatic retry up to 3 times\n"
            "   - Manual review if still failing\n"
            "   - Escalation to compliance if deadline at risk\n\n"
            "2. REJECTED REPORTS\n"
            "   - Analyze rejection reason\n"
            "   - Correct data and resubmit\n"
            "   - Document correction in audit trail\n\n"
            "3. CANCELLATIONS/AMENDMENTS\n"
            "   - Submit CANC/AMND transaction as required\n"
            "   - Reference original transaction\n"
            "   - Document reason for correction"
        ),
        regulatory_reference="RTS 22",
    ))

    policy.add_section(PolicySection(
        section_id="6",
        title="Monitoring and Controls",
        content=(
            "Transaction reporting is monitored through:\n\n"
            "1. Daily reconciliation of trades vs. reports\n"
            "2. T+1 compliance rate tracking\n"
            "3. Rejection rate monitoring\n"
            "4. Field-level validation before submission\n"
            "5. Monthly compliance reporting to management\n\n"
            "KPIs tracked:\n"
            "- T+1 compliance rate (target: 100%)\n"
            "- Rejection rate (target: <1%)\n"
            "- Amendment rate (target: <5%)"
        ),
        regulatory_reference="MiFIR Article 26",
    ))

    return policy


# =============================================================================
# Market Abuse Prevention Policy (MAR)
# =============================================================================


def create_market_abuse_prevention_policy(
    firm_lei: str,
    firm_name: str,
    owner: str,
) -> PolicyDocument:
    """
    Create Market Abuse Prevention Policy per MAR.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        owner: Policy owner.

    Returns:
        PolicyDocument for market abuse prevention.
    """
    policy = PolicyDocument(
        policy_type=PolicyType.MARKET_ABUSE_PREVENTION,
        title="Market Abuse Prevention Policy",
        description=(
            "This policy establishes procedures for preventing market abuse at "
            f"{firm_name} in compliance with the Market Abuse Regulation (MAR) "
            "and MiFID II requirements."
        ),
        owner=owner,
        firm_lei=firm_lei,
        firm_name=firm_name,
        approval_level=ApprovalLevel.SENIOR_MANAGEMENT,
        regulatory_references=[
            "Market Abuse Regulation (EU) 596/2014",
            "MAR Delegated Regulations",
            "MiFID II Article 17(6): Market manipulation by algorithms",
        ],
    )

    policy.add_section(PolicySection(
        section_id="1",
        title="Prohibited Conduct",
        content=(
            "The following conduct is prohibited:\n\n"
            "INSIDER DEALING (MAR Article 14)\n"
            "- Trading on inside information\n"
            "- Disclosing inside information improperly\n"
            "- Recommending trades based on inside information\n\n"
            "MARKET MANIPULATION (MAR Article 15)\n"
            "- Giving false or misleading signals\n"
            "- Securing prices at abnormal levels\n"
            "- Disseminating misleading information\n"
            "- Benchmark manipulation\n\n"
            "ALGORITHMIC MANIPULATION\n"
            "- Spoofing/layering\n"
            "- Quote stuffing\n"
            "- Momentum ignition\n"
            "- Ping orders"
        ),
        regulatory_reference="MAR Articles 14-15",
    ))

    policy.add_section(PolicySection(
        section_id="2",
        title="Algorithm Controls",
        content=(
            "Algorithmic trading must not engage in market manipulation.\n\n"
            "CONTROLS:\n"
            "1. Algorithm design reviewed for manipulation risk\n"
            "2. Order patterns monitored for suspicious behavior\n"
            "3. Order-to-trade ratio monitoring\n"
            "4. Cancellation rate monitoring\n"
            "5. Spoofing detection algorithms\n\n"
            "PROHIBITED STRATEGIES:\n"
            "- Layering orders with intent to cancel\n"
            "- Placing orders to create false momentum\n"
            "- Submitting excessive orders to slow systems\n"
            "- Pinging for hidden liquidity detection"
        ),
        regulatory_reference="MiFID II Article 17(6)",
    ))

    policy.add_section(PolicySection(
        section_id="3",
        title="Suspicious Transaction Reporting",
        content=(
            "Staff must report suspicious orders and transactions.\n\n"
            "STORS PROCESS:\n"
            "1. Employee identifies suspicious activity\n"
            "2. Report to Compliance immediately\n"
            "3. Compliance assesses within 24 hours\n"
            "4. If suspicious, submit STOR to NCA without delay\n"
            "5. Maintain confidentiality of reporting\n\n"
            "SUSPICIOUS INDICATORS:\n"
            "- Unusual order patterns\n"
            "- Trading ahead of news/announcements\n"
            "- Concentrated trading at close\n"
            "- Orders with no economic rationale"
        ),
        regulatory_reference="MAR Article 16",
    ))

    policy.add_section(PolicySection(
        section_id="4",
        title="Training and Awareness",
        content=(
            "All staff involved in trading must receive MAR training:\n\n"
            "1. Initial training on joining\n"
            "2. Annual refresher training\n"
            "3. Training on policy updates\n"
            "4. Training records maintained\n\n"
            "Training covers:\n"
            "- Definition of market abuse\n"
            "- Recognition of suspicious activity\n"
            "- Reporting procedures\n"
            "- Consequences of non-compliance"
        ),
        regulatory_reference="MAR",
    ))

    return policy


# =============================================================================
# Business Continuity Policy (RTS 6 Article 3)
# =============================================================================


def create_business_continuity_policy(
    firm_lei: str,
    firm_name: str,
    owner: str,
) -> PolicyDocument:
    """
    Create Business Continuity Policy per RTS 6 Article 3.

    Note: This is the policy document; the detailed BCP is in bcp.py.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        owner: Policy owner.

    Returns:
        PolicyDocument for business continuity.
    """
    policy = PolicyDocument(
        policy_type=PolicyType.BUSINESS_CONTINUITY,
        title="Business Continuity Policy",
        description=(
            "This policy establishes the business continuity framework for "
            f"algorithmic trading at {firm_name} in compliance with RTS 6 "
            "Article 3."
        ),
        owner=owner,
        firm_lei=firm_lei,
        firm_name=firm_name,
        approval_level=ApprovalLevel.SENIOR_MANAGEMENT,
        regulatory_references=[
            "RTS 6 Article 3: Business continuity arrangements",
            "MiFID II Article 17: Algorithmic trading",
            "ISO 22301: Business continuity management",
        ],
    )

    policy.add_section(PolicySection(
        section_id="1",
        title="Scope and Objectives",
        content=(
            "This policy ensures effective business continuity arrangements "
            "for algorithmic trading systems.\n\n"
            "OBJECTIVES:\n"
            "1. Minimize disruption to trading operations\n"
            "2. Protect market integrity\n"
            "3. Ensure regulatory compliance during incidents\n"
            "4. Enable rapid recovery to normal operations\n\n"
            "SCOPE:\n"
            "- All algorithmic trading systems\n"
            "- Supporting infrastructure\n"
            "- Market data systems\n"
            "- Order management systems"
        ),
        regulatory_reference="RTS 6 Article 3",
    ))

    policy.add_section(PolicySection(
        section_id="2",
        title="Disruption Thresholds",
        content=(
            "Per RTS 6, the firm maintains predetermined thresholds for "
            "trading cessation.\n\n"
            "TRADING HALT TRIGGERS:\n"
            "- Multiple system failures within 24 hours\n"
            "- Complete loss of market data\n"
            "- Network connectivity failure\n"
            "- Cybersecurity incident\n"
            "- Kill switch activation for systemic issue\n\n"
            "The specific threshold for trading halt is defined in the "
            "detailed Business Continuity Plan."
        ),
        regulatory_reference="RTS 6 Article 3(1)",
    ))

    policy.add_section(PolicySection(
        section_id="3",
        title="Testing Requirements",
        content=(
            "Business continuity arrangements are tested:\n\n"
            "1. ANNUAL FULL TEST\n"
            "   - Complete failover to backup systems\n"
            "   - End-to-end scenario testing\n\n"
            "2. QUARTERLY COMPONENT TESTS\n"
            "   - Individual system recovery tests\n"
            "   - Backup restoration tests\n\n"
            "3. MONTHLY COMMUNICATION TESTS\n"
            "   - Emergency contact verification\n"
            "   - Escalation procedure tests\n\n"
            "All tests are documented with findings and remediation."
        ),
        regulatory_reference="RTS 6 Article 3(2)",
    ))

    policy.add_section(PolicySection(
        section_id="4",
        title="Documentation",
        content=(
            "The following documentation is maintained:\n\n"
            "1. Detailed Business Continuity Plan (BCP)\n"
            "2. Disaster Recovery procedures\n"
            "3. Emergency contact lists\n"
            "4. System recovery procedures\n"
            "5. Incident response playbooks\n"
            "6. Test results and remediation records\n\n"
            "All documentation is reviewed at least annually."
        ),
        regulatory_reference="RTS 6 Article 3",
    ))

    return policy


# =============================================================================
# Convenience Functions
# =============================================================================


def create_all_standard_policies(
    firm_lei: str,
    firm_name: str,
    owner: str,
) -> List[PolicyDocument]:
    """
    Create all standard MiFID II policy documents.

    Args:
        firm_lei: Firm's LEI.
        firm_name: Firm's legal name.
        owner: Policy owner.

    Returns:
        List of all standard policy documents.
    """
    return [
        create_best_execution_policy(firm_lei, firm_name, owner),
        create_order_handling_policy(firm_lei, firm_name, owner),
        create_conflicts_of_interest_policy(firm_lei, firm_name, owner),
        create_kill_switch_policy(firm_lei, firm_name, owner),
        create_transaction_reporting_policy(firm_lei, firm_name, owner),
        create_market_abuse_prevention_policy(firm_lei, firm_name, owner),
        create_business_continuity_policy(firm_lei, firm_name, owner),
    ]


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "create_best_execution_policy",
    "create_order_handling_policy",
    "create_conflicts_of_interest_policy",
    "create_kill_switch_policy",
    "create_transaction_reporting_policy",
    "create_market_abuse_prevention_policy",
    "create_business_continuity_policy",
    "create_all_standard_policies",
]

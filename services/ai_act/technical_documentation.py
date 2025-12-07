# -*- coding: utf-8 -*-
"""
EU AI Act Technical Documentation System (Article 11, Annex IV).

EU AI Act Article 11 requires high-risk AI systems to be accompanied by
technical documentation that demonstrates compliance with the regulation.

This module provides:
1. TechnicalDocumentationGenerator - Generates Annex IV compliant documentation
2. DocumentationSection - Structured documentation sections
3. ComplianceEvidence - Evidence collection for audit trails
4. DocumentationExporter - Export to various formats (Markdown, HTML, PDF metadata)

Annex IV Required Elements:
    1. General description of the AI system
    2. Detailed description of elements and development process
    3. Monitoring, functioning, and control of the AI system
    4. Description of appropriateness of performance metrics
    5. Detailed description of the risk management system
    6. Description of any changes made to the system

References:
    - Article 11: https://artificialintelligenceact.eu/article/11/
    - Annex IV: https://artificialintelligenceact.eu/annex/4/
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class DocumentationSectionType(Enum):
    """
    Annex IV section types.

    Maps to the required sections in Annex IV of the EU AI Act.
    """
    GENERAL_DESCRIPTION = "general_description"
    ALGORITHM_AND_DATA = "algorithm_and_data"
    MONITORING_AND_CONTROL = "monitoring_and_control"
    PERFORMANCE_METRICS = "performance_metrics"
    RISK_MANAGEMENT = "risk_management"
    CHANGE_LOG = "change_log"
    COMPLIANCE_EVIDENCE = "compliance_evidence"
    HUMAN_OVERSIGHT = "human_oversight"
    DATA_GOVERNANCE = "data_governance"
    TESTING_VALIDATION = "testing_validation"


class ComplianceStatus(Enum):
    """Compliance status for documentation sections."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"


class ExportFormat(Enum):
    """Export formats for documentation."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    PDF_METADATA = "pdf_metadata"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DocumentationMetadata:
    """
    Metadata for technical documentation.

    Per Annex IV, documentation must include identification information.
    """
    document_id: str = field(default_factory=lambda: f"DOC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    author: str = ""
    reviewer: str = ""
    approval_status: str = "draft"

    # System identification (Annex IV.1(a))
    system_name: str = ""
    system_version: str = ""
    provider_name: str = ""
    provider_address: str = ""

    # Classification
    risk_classification: str = "high_risk"  # Default for trading AI
    intended_purpose: str = ""

    # References
    eu_ai_act_reference: str = "Regulation (EU) 2024/1689"
    annex_iv_compliance: bool = True


@dataclass
class ComplianceEvidence:
    """
    Evidence supporting compliance claims.

    Per Article 11(2), documentation must demonstrate compliance.
    """
    evidence_id: str = field(default_factory=lambda: f"EVD-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    article_reference: str = ""  # e.g., "Article 9" or "Annex IV.2"
    requirement_text: str = ""
    evidence_description: str = ""
    evidence_type: str = ""  # e.g., "test_result", "code_reference", "process_document"
    evidence_location: str = ""  # File path or URL
    verification_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    verified_by: str = ""
    compliance_status: str = ComplianceStatus.UNDER_REVIEW.value


@dataclass
class DocumentationSection:
    """
    A section of technical documentation.

    Represents one of the required sections per Annex IV.
    """
    section_id: str = field(default_factory=lambda: f"SEC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}")
    section_type: str = DocumentationSectionType.GENERAL_DESCRIPTION.value
    title: str = ""
    content: str = ""
    subsections: List[Dict[str, Any]] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)  # Evidence IDs
    compliance_status: str = ComplianceStatus.UNDER_REVIEW.value
    last_reviewed: str = ""
    reviewer_notes: str = ""

    def to_markdown(self) -> str:
        """Convert section to Markdown format."""
        md = f"# {self.title}\n\n"
        md += f"{self.content}\n\n"

        for subsection in self.subsections:
            level = subsection.get("level", 2)
            md += f"{'#' * level} {subsection.get('title', '')}\n\n"
            md += f"{subsection.get('content', '')}\n\n"

        if self.evidence_refs:
            md += "## Supporting Evidence\n\n"
            for ref in self.evidence_refs:
                md += f"- Evidence: {ref}\n"

        return md


@dataclass
class ChangeRecord:
    """
    Record of changes to the AI system.

    Per Annex IV.6, all changes must be documented.
    """
    change_id: str = field(default_factory=lambda: f"CHG-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    change_type: str = ""  # "algorithm", "data", "config", "deployment"
    description: str = ""
    rationale: str = ""
    impact_assessment: str = ""
    is_substantial_modification: bool = False  # Per Article 6(3)
    requires_reassessment: bool = False
    approved_by: str = ""
    verification_reference: str = ""


@dataclass
class TechnicalDocumentationConfig:
    """Configuration for documentation generation."""
    output_path: str = "docs/compliance/technical_documentation"
    template_path: str = "templates/ai_act"
    auto_generate_from_code: bool = True
    include_code_references: bool = True
    include_git_history: bool = True
    include_test_results: bool = True
    max_change_history: int = 100
    generate_compliance_matrix: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["markdown", "json"])


# =============================================================================
# Technical Documentation Generator
# =============================================================================


class TechnicalDocumentationGenerator:
    """
    Generator for EU AI Act Annex IV compliant technical documentation.

    This class automatically generates and maintains technical documentation
    required by Article 11 and Annex IV of the EU AI Act.

    Usage:
        generator = TechnicalDocumentationGenerator(config)

        # Generate full documentation
        generator.generate_all()

        # Generate specific section
        generator.generate_section(DocumentationSectionType.ALGORITHM_AND_DATA)

        # Add evidence
        generator.add_evidence(ComplianceEvidence(...))

        # Export documentation
        generator.export(ExportFormat.MARKDOWN)
    """

    def __init__(self, config: Optional[TechnicalDocumentationConfig] = None):
        """Initialize the documentation generator."""
        self.config = config or TechnicalDocumentationConfig()
        self._output_path = Path(self.config.output_path)
        self._output_path.mkdir(parents=True, exist_ok=True)

        # Documentation storage
        self.metadata = DocumentationMetadata()
        self.sections: Dict[str, DocumentationSection] = {}
        self.evidence: Dict[str, ComplianceEvidence] = {}
        self.change_log: List[ChangeRecord] = []

        # Code introspection cache
        self._code_cache: Dict[str, str] = {}

        logger.info(f"TechnicalDocumentationGenerator initialized. Output: {self._output_path}")

    # =========================================================================
    # Section Generation (Annex IV Sections)
    # =========================================================================

    def generate_general_description(
        self,
        system_name: str = "AI-Powered Quantitative Research Platform",
        system_version: str = "2.0.0",
        provider_name: str = "",
        intended_purpose: str = "",
        target_users: str = "",
        deployment_contexts: List[str] = None,
        hardware_requirements: Dict[str, str] = None,
        software_dependencies: List[str] = None,
    ) -> DocumentationSection:
        """
        Generate Annex IV Section 1: General Description.

        Per Annex IV.1, this must include:
        (a) Trade name, provider details
        (b) Version information
        (c) Intended purpose
        (d) Interaction with other systems
        (e) Hardware/software requirements
        """
        # Update metadata
        self.metadata.system_name = system_name
        self.metadata.system_version = system_version
        self.metadata.provider_name = provider_name
        self.metadata.intended_purpose = intended_purpose

        # Build content
        content = f"""
This document provides the general description of {system_name} in accordance with
Annex IV.1 of the EU AI Act (Regulation (EU) 2024/1689).

## System Identification

| Field | Value |
|-------|-------|
| System Name | {system_name} |
| Version | {system_version} |
| Provider | {provider_name} |
| Risk Classification | High-Risk (Algorithmic Trading) |

## Intended Purpose

{intended_purpose if intended_purpose else "AI-powered algorithmic trading system using reinforcement learning for generating trading signals across multiple asset classes."}

## Target Users

{target_users if target_users else "Professional traders, quantitative researchers, and financial institutions operating in regulated markets."}

## Deployment Contexts

{self._format_list(deployment_contexts or ["EU regulated financial markets", "US regulated exchanges", "Professional trading environments"])}

## Hardware Requirements

{self._format_dict(hardware_requirements or {"GPU": "NVIDIA GPU with CUDA support (recommended)", "RAM": "32GB minimum", "Storage": "100GB SSD minimum"})}

## Software Dependencies

{self._format_list(software_dependencies or ["Python 3.10+", "PyTorch 2.0+", "NumPy", "Pandas"])}
"""

        section = DocumentationSection(
            section_type=DocumentationSectionType.GENERAL_DESCRIPTION.value,
            title="1. General Description of the AI System",
            content=content.strip(),
        )

        self.sections[section.section_id] = section
        return section

    def generate_algorithm_description(
        self,
        algorithm_name: str = "Distributional PPO with Twin Critics",
        algorithm_type: str = "Reinforcement Learning",
        core_components: Dict[str, str] = None,
        training_process: Dict[str, Any] = None,
        data_sources: List[Dict[str, str]] = None,
        hyperparameters: Dict[str, Any] = None,
    ) -> DocumentationSection:
        """
        Generate Annex IV Section 2: Algorithm and Data Description.

        Per Annex IV.2, this must include:
        (a) Development methods and techniques
        (b) Design specifications
        (c) System architecture
        (d) Computational resources
        (e) Data requirements
        """
        core_components = core_components or {
            "Policy Network": "LSTM-based recurrent policy for temporal dependencies",
            "Value Network": "Distributional (C51-style with N=21 atoms)",
            "Optimizer": "AdaptiveUPGD for continual learning",
            "Gradient Scaling": "VGS v3.2 for variance reduction",
        }

        content = f"""
This section describes the algorithm architecture and data processing in accordance
with Annex IV.2 of the EU AI Act.

## Algorithm Overview

| Aspect | Description |
|--------|-------------|
| Algorithm Name | {algorithm_name} |
| Algorithm Type | {algorithm_type} |
| Base Architecture | Proximal Policy Optimization (Schulman et al., 2017) |

## Core Components

{self._format_dict(core_components)}

## Algorithm Architecture

### Policy Network
- Architecture: LSTM with attention mechanism
- Input: Market observations, portfolio state
- Output: Action distribution (position sizing, direction)

### Value Network
- Architecture: Distributional value head
- Output: Return distribution (21 quantiles)
- Risk metric: CVaR-based risk-aware learning

## Training Process

{self._format_dict(training_process or {"Method": "Online reinforcement learning", "Update frequency": "Per episode", "Replay buffer": "Prioritized Experience Replay"})}

## Hyperparameters

{self._format_dict(hyperparameters or {"Learning rate": "3e-4", "Discount factor": "0.99", "GAE lambda": "0.95", "Clip range": "0.2"})}

## Data Sources

{self._format_data_sources(data_sources)}

## Data Quality Measures

- **Winsorization**: [1%, 99%] percentiles to handle outliers
- **Missing values**: Forward fill, then backward fill
- **Outlier detection**: 3-sigma rule + domain-specific checks
- **Data validation**: Schema validation, temporal consistency checks
"""

        section = DocumentationSection(
            section_type=DocumentationSectionType.ALGORITHM_AND_DATA.value,
            title="2. Algorithm and Data Description",
            content=content.strip(),
        )

        self.sections[section.section_id] = section
        return section

    def generate_monitoring_description(
        self,
        monitoring_metrics: List[Dict[str, str]] = None,
        alert_thresholds: Dict[str, Any] = None,
        human_oversight_mechanisms: List[str] = None,
        logging_details: Dict[str, str] = None,
    ) -> DocumentationSection:
        """
        Generate Annex IV Section 3: Monitoring, Functioning, and Control.

        Per Annex IV.3, this must include:
        (a) Capabilities and limitations
        (b) Degree of accuracy
        (c) Foreseeable unintended outcomes
        (d) Human oversight measures
        """
        content = f"""
This section describes the monitoring, functioning, and control mechanisms in
accordance with Annex IV.3 of the EU AI Act.

## Real-time Monitoring

### Metrics Tracked

{self._format_monitoring_metrics(monitoring_metrics)}

### Alert Thresholds

{self._format_dict(alert_thresholds or {"Max drawdown": "10%", "Position size limit": "5% of portfolio", "Loss limit (daily)": "2%"})}

## Human Oversight Controls

Per Article 14 of the EU AI Act, the following human oversight mechanisms are implemented:

{self._format_list(human_oversight_mechanisms or [
    "Kill switch: Immediate halt of all trading activity",
    "Manual override: Operator can override any AI decision",
    "Alert escalation: Critical alerts sent to human operators",
    "Approval workflow: Large trades require human approval",
    "Session management: Operators can pause/resume sessions",
])}

## Logging and Audit Trail

Per Article 12, comprehensive logging is maintained:

{self._format_dict(logging_details or {
    "Event types logged": "All trading decisions, predictions, orders, risk events",
    "Log format": "Structured JSON with integrity hashes",
    "Retention period": "Minimum 6 months (per Article 19)",
    "Storage": "Tamper-evident storage with chain hashing",
})}

## System Capabilities and Limitations

### Capabilities
- Multi-asset trading across crypto, equity, forex, futures
- Real-time market data processing
- Risk-aware decision making with CVaR optimization
- Adaptive learning with continual updates

### Known Limitations
- Performance may degrade in extreme market conditions
- Requires stable data feed connectivity
- Model predictions have inherent uncertainty
- Historical performance does not guarantee future results
"""

        section = DocumentationSection(
            section_type=DocumentationSectionType.MONITORING_AND_CONTROL.value,
            title="3. Monitoring, Functioning, and Control",
            content=content.strip(),
        )

        self.sections[section.section_id] = section
        return section

    def generate_performance_metrics_description(
        self,
        accuracy_metrics: Dict[str, Any] = None,
        robustness_metrics: Dict[str, Any] = None,
        fairness_metrics: Dict[str, Any] = None,
        validation_results: Dict[str, Any] = None,
    ) -> DocumentationSection:
        """
        Generate Annex IV Section 4: Performance Metrics.

        Per Annex IV.4, this must include:
        (a) Performance metrics
        (b) Testing procedures
        (c) Validation methods
        """
        content = f"""
This section describes the performance metrics and validation procedures in
accordance with Annex IV.4 of the EU AI Act.

## Accuracy Metrics

{self._format_dict(accuracy_metrics or {
    "Sharpe Ratio": "Risk-adjusted return metric",
    "Sortino Ratio": "Downside risk-adjusted return",
    "Maximum Drawdown": "Peak-to-trough decline",
    "Win Rate": "Percentage of profitable trades",
    "Profit Factor": "Gross profit / Gross loss",
})}

## Robustness Testing

Per Article 15, robustness testing includes:

{self._format_dict(robustness_metrics or {
    "Distribution shift testing": "Performance under market regime changes",
    "Adversarial testing": "Response to adversarial market conditions",
    "Stress testing": "Behavior during extreme volatility",
    "Input perturbation": "Sensitivity to noisy data",
})}

## Fairness Assessment

{self._format_dict(fairness_metrics or {
    "Temporal fairness": "Consistent performance across time periods",
    "Asset fairness": "Balanced treatment of different assets",
    "Regime fairness": "Performance across market regimes",
})}

## Validation Results

{self._format_dict(validation_results or {
    "Backtesting period": "2019-2024 (5 years)",
    "Out-of-sample validation": "Walk-forward analysis",
    "Cross-validation": "Time-series cross-validation",
    "Paper trading": "Live market simulation",
})}

## Continuous Performance Monitoring

- Daily performance reports
- Weekly model health checks
- Monthly comprehensive reviews
- Quarterly model revalidation
"""

        section = DocumentationSection(
            section_type=DocumentationSectionType.PERFORMANCE_METRICS.value,
            title="4. Performance Metrics and Validation",
            content=content.strip(),
        )

        self.sections[section.section_id] = section
        return section

    def generate_risk_management_description(
        self,
        risk_categories: List[Dict[str, str]] = None,
        mitigation_measures: Dict[str, str] = None,
        residual_risks: List[Dict[str, str]] = None,
    ) -> DocumentationSection:
        """
        Generate Annex IV Section 5: Risk Management System.

        Per Annex IV.5 and Article 9, this must describe:
        (a) Risk management system
        (b) Risk identification and assessment
        (c) Risk mitigation measures
        """
        content = f"""
This section describes the risk management system in accordance with
Annex IV.5 and Article 9 of the EU AI Act.

## Risk Management Framework

The AI system implements a continuous, iterative risk management process
throughout its lifecycle as required by Article 9(1).

## Risk Categories

{self._format_risk_categories(risk_categories)}

## Risk Mitigation Measures

Per Article 9(4), the following mitigation measures are implemented:

{self._format_dict(mitigation_measures or {
    "Model robustness": "VGS gradient scaling, conformal prediction",
    "Market stability": "Kill switch, position limits, circuit breakers",
    "Data quality": "Validation pipeline, anomaly detection",
    "Human oversight": "Manual override, approval workflows",
    "Cybersecurity": "Input validation, sandboxing, encryption",
})}

## Residual Risks

After mitigation, the following residual risks remain:

{self._format_residual_risks(residual_risks)}

## Risk Monitoring

- Real-time risk metric tracking
- Automated alerts for threshold breaches
- Daily risk reports
- Incident response procedures

## Risk Review Process

- Weekly risk review meetings
- Monthly risk assessment updates
- Quarterly comprehensive risk audits
- Annual risk management system review
"""

        section = DocumentationSection(
            section_type=DocumentationSectionType.RISK_MANAGEMENT.value,
            title="5. Risk Management System",
            content=content.strip(),
        )

        self.sections[section.section_id] = section
        return section

    def generate_change_log_section(self) -> DocumentationSection:
        """
        Generate Annex IV Section 6: Change Log.

        Per Annex IV.6, all changes to the system must be documented.
        """
        content = f"""
This section documents all changes made to the AI system in accordance with
Annex IV.6 of the EU AI Act.

## Change Management Process

All changes to the AI system follow a structured process:

1. Change request submission
2. Impact assessment
3. Review and approval
4. Implementation
5. Verification
6. Documentation update

## Substantial Modifications

Per Article 6(3), the following changes are considered substantial modifications
that may require reassessment:

- Changes to the algorithm architecture
- Changes to training data sources
- Changes to deployment context
- Changes affecting the intended purpose

## Version History

{self._format_change_log()}

## Change Request Procedure

For any proposed changes:

1. Submit change request with rationale
2. Conduct impact assessment
3. Obtain required approvals
4. Implement in staging environment
5. Validate against test suite
6. Deploy to production
7. Update documentation
8. Monitor post-deployment
"""

        section = DocumentationSection(
            section_type=DocumentationSectionType.CHANGE_LOG.value,
            title="6. Change Log and Version History",
            content=content.strip(),
        )

        self.sections[section.section_id] = section
        return section

    # =========================================================================
    # Evidence Management
    # =========================================================================

    def add_evidence(self, evidence: ComplianceEvidence) -> str:
        """Add compliance evidence to the documentation."""
        self.evidence[evidence.evidence_id] = evidence
        logger.info(f"Added evidence: {evidence.evidence_id} for {evidence.article_reference}")
        return evidence.evidence_id

    def link_evidence_to_section(
        self,
        section_id: str,
        evidence_id: str,
    ) -> bool:
        """Link evidence to a documentation section."""
        if section_id not in self.sections:
            logger.warning(f"Section {section_id} not found")
            return False

        if evidence_id not in self.evidence:
            logger.warning(f"Evidence {evidence_id} not found")
            return False

        if evidence_id not in self.sections[section_id].evidence_refs:
            self.sections[section_id].evidence_refs.append(evidence_id)

        return True

    def generate_compliance_matrix(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a compliance matrix mapping requirements to evidence.

        Returns a structured matrix showing:
        - Each Annex IV requirement
        - Associated documentation sections
        - Supporting evidence
        - Compliance status
        """
        matrix = {
            "Annex IV.1 - General Description": [],
            "Annex IV.2 - Algorithm and Data": [],
            "Annex IV.3 - Monitoring and Control": [],
            "Annex IV.4 - Performance Metrics": [],
            "Annex IV.5 - Risk Management": [],
            "Annex IV.6 - Change Log": [],
        }

        # Map sections to requirements
        section_mapping = {
            DocumentationSectionType.GENERAL_DESCRIPTION.value: "Annex IV.1 - General Description",
            DocumentationSectionType.ALGORITHM_AND_DATA.value: "Annex IV.2 - Algorithm and Data",
            DocumentationSectionType.MONITORING_AND_CONTROL.value: "Annex IV.3 - Monitoring and Control",
            DocumentationSectionType.PERFORMANCE_METRICS.value: "Annex IV.4 - Performance Metrics",
            DocumentationSectionType.RISK_MANAGEMENT.value: "Annex IV.5 - Risk Management",
            DocumentationSectionType.CHANGE_LOG.value: "Annex IV.6 - Change Log",
        }

        for section_id, section in self.sections.items():
            requirement = section_mapping.get(section.section_type)
            if requirement and requirement in matrix:
                matrix[requirement].append({
                    "section_id": section_id,
                    "title": section.title,
                    "compliance_status": section.compliance_status,
                    "evidence_count": len(section.evidence_refs),
                })

        return matrix

    # =========================================================================
    # Change Management
    # =========================================================================

    def record_change(self, change: ChangeRecord) -> str:
        """Record a change to the system."""
        self.change_log.append(change)

        # Limit change log size
        if len(self.change_log) > self.config.max_change_history:
            self.change_log = self.change_log[-self.config.max_change_history:]

        logger.info(f"Recorded change: {change.change_id} - {change.description}")
        return change.change_id

    def get_substantial_modifications(self) -> List[ChangeRecord]:
        """Get all substantial modifications that may require reassessment."""
        return [c for c in self.change_log if c.is_substantial_modification]

    # =========================================================================
    # Generation and Export
    # =========================================================================

    def generate_all(
        self,
        system_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DocumentationSection]:
        """
        Generate all documentation sections.

        This is the main entry point for generating complete documentation.
        """
        system_info = system_info or {}

        logger.info("Generating complete technical documentation...")

        # Generate all sections
        self.generate_general_description(
            system_name=system_info.get("system_name", "AI-Powered Quantitative Research Platform"),
            system_version=system_info.get("version", "2.0.0"),
            provider_name=system_info.get("provider", ""),
            intended_purpose=system_info.get("purpose", ""),
        )

        self.generate_algorithm_description()
        self.generate_monitoring_description()
        self.generate_performance_metrics_description()
        self.generate_risk_management_description()
        self.generate_change_log_section()

        logger.info(f"Generated {len(self.sections)} documentation sections")

        return self.sections

    def export(
        self,
        format: Union[ExportFormat, str] = ExportFormat.MARKDOWN,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export documentation to specified format.

        Returns the path to the exported documentation.
        """
        if isinstance(format, str):
            format = ExportFormat(format)

        output_path = Path(output_path or self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if format == ExportFormat.MARKDOWN:
            return self._export_markdown(output_path)
        elif format == ExportFormat.JSON:
            return self._export_json(output_path)
        elif format == ExportFormat.HTML:
            return self._export_html(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_markdown(self, output_path: Path) -> str:
        """Export documentation to Markdown format."""
        # Create index file
        index_content = f"""# Technical Documentation

**{self.metadata.system_name}**

Version: {self.metadata.system_version}
Generated: {datetime.now(timezone.utc).isoformat()}

## Table of Contents

"""
        # Export each section
        for section_id, section in self.sections.items():
            filename = f"{section.section_type}.md"
            filepath = output_path / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(section.to_markdown())

            index_content += f"- [{section.title}]({filename})\n"

        # Write index
        index_path = output_path / "README.md"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_content)

        logger.info(f"Exported Markdown documentation to {output_path}")
        return str(output_path)

    def _export_json(self, output_path: Path) -> str:
        """Export documentation to JSON format."""
        export_data = {
            "metadata": asdict(self.metadata),
            "sections": {sid: {
                "section_id": s.section_id,
                "section_type": s.section_type,
                "title": s.title,
                "content": s.content,
                "compliance_status": s.compliance_status,
                "evidence_refs": s.evidence_refs,
            } for sid, s in self.sections.items()},
            "evidence": {eid: asdict(e) for eid, e in self.evidence.items()},
            "change_log": [asdict(c) for c in self.change_log],
            "compliance_matrix": self.generate_compliance_matrix(),
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        filepath = output_path / "technical_documentation.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported JSON documentation to {filepath}")
        return str(filepath)

    def _export_html(self, output_path: Path) -> str:
        """Export documentation to HTML format."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Technical Documentation - {self.metadata.system_name}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #eee; border-radius: 5px; }}
        .metadata {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Technical Documentation</h1>
    <div class="metadata">
        <p><strong>System:</strong> {self.metadata.system_name}</p>
        <p><strong>Version:</strong> {self.metadata.system_version}</p>
        <p><strong>Generated:</strong> {datetime.now(timezone.utc).isoformat()}</p>
    </div>
"""

        for section in self.sections.values():
            html_content += f"""
    <div class="section">
        {self._markdown_to_html(section.to_markdown())}
    </div>
"""

        html_content += """
</body>
</html>
"""

        filepath = output_path / "technical_documentation.html"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Exported HTML documentation to {filepath}")
        return str(filepath)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _format_list(self, items: List[str]) -> str:
        """Format a list for Markdown."""
        return "\n".join(f"- {item}" for item in items)

    def _format_dict(self, data: Dict[str, Any]) -> str:
        """Format a dictionary as a Markdown table."""
        if not data:
            return "*No data available*"

        lines = ["| Item | Value |", "|------|-------|"]
        for key, value in data.items():
            lines.append(f"| {key} | {value} |")
        return "\n".join(lines)

    def _format_data_sources(self, sources: Optional[List[Dict[str, str]]]) -> str:
        """Format data sources as a table."""
        if not sources:
            sources = [
                {"Source": "Binance", "Type": "Crypto OHLCV", "Period": "2019-present"},
                {"Source": "Alpaca", "Type": "US Equity", "Period": "2020-present"},
            ]

        lines = ["| Source | Type | Period |", "|--------|------|--------|"]
        for source in sources:
            lines.append(f"| {source.get('Source', '')} | {source.get('Type', '')} | {source.get('Period', '')} |")
        return "\n".join(lines)

    def _format_monitoring_metrics(self, metrics: Optional[List[Dict[str, str]]]) -> str:
        """Format monitoring metrics."""
        if not metrics:
            metrics = [
                {"Metric": "PnL", "Description": "Real-time profit and loss"},
                {"Metric": "Position Size", "Description": "Current position exposure"},
                {"Metric": "Drawdown", "Description": "Current drawdown from peak"},
                {"Metric": "Sharpe Ratio", "Description": "Rolling risk-adjusted return"},
            ]

        lines = ["| Metric | Description |", "|--------|-------------|"]
        for metric in metrics:
            lines.append(f"| {metric.get('Metric', '')} | {metric.get('Description', '')} |")
        return "\n".join(lines)

    def _format_risk_categories(self, categories: Optional[List[Dict[str, str]]]) -> str:
        """Format risk categories."""
        if not categories:
            categories = [
                {"Category": "Model Robustness", "Description": "Risks from model failures", "Severity": "High"},
                {"Category": "Market Stability", "Description": "Risks to market stability", "Severity": "Critical"},
                {"Category": "Data Quality", "Description": "Risks from data issues", "Severity": "High"},
                {"Category": "Human Oversight", "Description": "Risks from oversight failures", "Severity": "Critical"},
                {"Category": "Cybersecurity", "Description": "Security-related risks", "Severity": "High"},
            ]

        lines = ["| Category | Description | Severity |", "|----------|-------------|----------|"]
        for cat in categories:
            lines.append(f"| {cat.get('Category', '')} | {cat.get('Description', '')} | {cat.get('Severity', '')} |")
        return "\n".join(lines)

    def _format_residual_risks(self, risks: Optional[List[Dict[str, str]]]) -> str:
        """Format residual risks."""
        if not risks:
            risks = [
                {"Risk": "Unprecedented market events", "Likelihood": "Low", "Impact": "High"},
                {"Risk": "Model degradation over time", "Likelihood": "Medium", "Impact": "Medium"},
                {"Risk": "Data feed failures", "Likelihood": "Low", "Impact": "Medium"},
            ]

        lines = ["| Residual Risk | Likelihood | Impact |", "|---------------|------------|--------|"]
        for risk in risks:
            lines.append(f"| {risk.get('Risk', '')} | {risk.get('Likelihood', '')} | {risk.get('Impact', '')} |")
        return "\n".join(lines)

    def _format_change_log(self) -> str:
        """Format the change log."""
        if not self.change_log:
            return "*No changes recorded*"

        lines = ["| Version | Date | Changes | Impact |", "|---------|------|---------|--------|"]
        for change in self.change_log[-10:]:  # Last 10 changes
            date = change.timestamp[:10] if change.timestamp else ""
            lines.append(f"| {change.change_id} | {date} | {change.description[:50]} | {change.impact_assessment[:30]} |")
        return "\n".join(lines)

    def _markdown_to_html(self, markdown: str) -> str:
        """Simple Markdown to HTML conversion."""
        import re

        html = markdown

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Lists
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        # Tables (basic)
        html = re.sub(r'\|(.+)\|', r'<tr><td>\1</td></tr>', html)

        # Paragraphs
        html = re.sub(r'\n\n', r'</p><p>', html)

        return f"<p>{html}</p>"


# =============================================================================
# Factory Functions
# =============================================================================


def create_technical_documentation_generator(
    config: Optional[Union[TechnicalDocumentationConfig, Dict[str, Any]]] = None,
) -> TechnicalDocumentationGenerator:
    """
    Factory function to create a TechnicalDocumentationGenerator.

    Args:
        config: Configuration for the generator (optional)

    Returns:
        Configured TechnicalDocumentationGenerator instance
    """
    if config is None:
        config = TechnicalDocumentationConfig()
    elif isinstance(config, dict):
        config = TechnicalDocumentationConfig(**config)

    return TechnicalDocumentationGenerator(config)


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Enums
    "DocumentationSectionType",
    "ComplianceStatus",
    "ExportFormat",
    # Data structures
    "DocumentationMetadata",
    "ComplianceEvidence",
    "DocumentationSection",
    "ChangeRecord",
    "TechnicalDocumentationConfig",
    # Main class
    "TechnicalDocumentationGenerator",
    # Factory
    "create_technical_documentation_generator",
]

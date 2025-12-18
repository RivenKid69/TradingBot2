# Conformity Assessment (Article 43)

This directory contains documents for the EU AI Act conformity assessment process.

---

## Overview

This directory contains **templates and tooling notes** to support an EU AI Act assessment workflow (e.g., Article 43 / Annex VI) for providers and customers.

**Important:** CustodiaCloud does not self-classify as a high-risk AI system in documentation. Classification and required procedures are jurisdiction- and deployment-dependent and should be validated with qualified counsel (and, where applicable, a notified body).

---

## Assessment Documents

### Checklist
- [checklist.md](checklist.md) - Complete conformity assessment checklist per Annex VI requirements

### Supporting Documents
- [EU_DECLARATION_OF_CONFORMITY.md](../EU_DECLARATION_OF_CONFORMITY.md) - EU Declaration of Conformity (Article 47)
- [INSTRUCTIONS_FOR_USE.md](../INSTRUCTIONS_FOR_USE.md) - Instructions for Use (Article 13)

---

## Assessment Process

### 1. Pre-Assessment
1. Verify quality management system (Article 17)
2. Examine technical documentation (Article 11)
3. Complete checklist.md

### 2. Conformity Verification
1. Verify conformity with Articles 8-15
2. Review test results against defined metrics
3. Validate risk management measures

### 3. Documentation
1. If applicable, prepare the required provider documentation set
2. If applicable, draw up an EU declaration of conformity (Article 47)
3. If applicable, register in an EU database (Article 49)

---

## Automated Assessment Tool

Run the conformity self-assessment tool (internal alignment tooling):

```python
from services.ai_act import ConformityAssessmentTool

tool = ConformityAssessmentTool()
result = tool.run_full_assessment()
print(f"Compliance Score: {result.compliance_percentage}%")
print(f"Status: {result.assessment_decision}")
```

---

## Phase 4 Implementation

The conformity assessment module was implemented in Phase 4:
- [EU_AI_ACT_PHASE4_COMPLETION_REPORT.md](../EU_AI_ACT_PHASE4_COMPLETION_REPORT.md)
- Implementation: `services/ai_act/conformity_assessment.py`
- Tests: 81/81 passed

---

**Last Updated**: 2025-12-08

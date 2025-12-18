# Technical Documentation (Annex IV)

This directory contains references to the technical documentation required by Annex IV of the EU AI Act.

---

## Documentation Structure

Per Annex IV requirements, technical documentation is generated dynamically using the `TechnicalDocumentationGenerator` class and maintained across the following source files:

### Section 1: General Description
- **Source**: Configuration files and metadata
- **Generator**: `services/ai_act/technical_documentation.py`
- **References**:
  - [DOCUMENTATION_CANON_DESIGN.md](../../DOCUMENTATION_CANON_DESIGN.md)
  - [ARCHITECTURE.md](../../../ARCHITECTURE.md)
  - [README.md](../../../README.md)

### Section 2: Algorithm and Data
- **Source**: Core algorithm implementations
- **Generator**: `services/ai_act/technical_documentation.py`
- **References**:
  - [claude.md](../../../claude.md) - Complete technical reference
  - [docs/twin_critics.md](../../twin_critics.md) - Twin critics architecture
  - [docs/UPGD_INTEGRATION.md](../../UPGD_INTEGRATION.md) - UPGD optimizer
  - [docs/pipeline.md](../../pipeline.md) - Decision pipeline

### Section 3: Monitoring and Control
- **Source**: Service implementations
- **Generator**: `services/ai_act/technical_documentation.py`
- **References**:
  - [docs/OPERATIONS_RUNBOOK.md](../../OPERATIONS_RUNBOOK.md)
  - [INSTRUCTIONS_FOR_USE.md](../INSTRUCTIONS_FOR_USE.md)
  - Human oversight: `services/ai_act/human_oversight.py`

### Section 4: Performance Metrics
- **Source**: Accuracy and evaluation modules
- **Generator**: `services/ai_act/technical_documentation.py`
- **References**:
  - [docs/eval.md](../../eval.md)
  - Accuracy metrics: `services/ai_act/accuracy_metrics.py`
  - Robustness testing: `services/ai_act/robustness_testing.py`

### Section 5: Risk Management
- **Source**: Risk management system
- **Generator**: `services/ai_act/technical_documentation.py`
- **References**:
  - Risk management: `services/ai_act/risk_management.py`
  - Risk registry: `services/ai_act/risk_registry.py`
  - [EU_AI_ACT_PHASE1_COMPLETION_REPORT.md](../EU_AI_ACT_PHASE1_COMPLETION_REPORT.md)

### Section 6: Change Log
- **Source**: Git history and version control
- **Generator**: `services/ai_act/technical_documentation.py`
- **References**:
  - [CHANGELOG.md](../../../CHANGELOG.md)
  - Git version control history

---

## Generating Technical Documentation

To generate complete Annex IV-aligned technical documentation:

```python
from services.ai_act import create_technical_documentation_generator

generator = create_technical_documentation_generator()
full_doc = generator.generate_full_documentation()

# Export to different formats
generator.export_to_markdown("technical_documentation.md")
generator.export_to_json("technical_documentation.json")
generator.export_to_html("technical_documentation.html")
```

---

## Related Documents

- [EU_AI_ACT_INTEGRATION_PLAN.md](../EU_AI_ACT_INTEGRATION_PLAN.md) - Master integration plan
- [EU_AI_ACT_PHASE2_COMPLETION_REPORT.md](../EU_AI_ACT_PHASE2_COMPLETION_REPORT.md) - Technical documentation implementation
- [EU_DECLARATION_OF_CONFORMITY.md](../EU_DECLARATION_OF_CONFORMITY.md) - EU Declaration
- [INSTRUCTIONS_FOR_USE.md](../INSTRUCTIONS_FOR_USE.md) - User instructions

---

**Last Updated**: 2025-12-08

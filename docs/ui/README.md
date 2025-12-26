# UI Guidelines & Guardrails

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16

## Overview

This directory contains UI guidelines, compliance requirements, and implementation guides for the Platform's user interface.

## Documents

| Document | Description |
|----------|-------------|
| [ONBOARDING_GUARDRAILS.md](./ONBOARDING_GUARDRAILS.md) | Disclaimers, acknowledgments, and warnings |

## Key Principles

### 1. Transparency First

Users must always know:
- Whether they're in paper or live trading mode
- When content is AI-generated
- What data stays local vs. goes to Cloud
- What their risk controls are

### 2. Informed Consent

All critical actions require explicit acknowledgment:
- Registration: ToS, Privacy, Not-Advice
- Live Trading: Risk warning, responsibility
- Agent Setup: Architecture understanding
- Deployment: Strategy responsibility

### 3. Clear Architecture Communication

CCEA architecture must be clearly explained:
- Credentials stay local (Agent)
- Cloud is designed not to execute orders
- Hard caps cannot be overridden
- User maintains control

### 4. Regulatory Compliance

UI must comply with:
- EU AI Act Article 50 (AI disclosure)
- MiFID II information requirements
- GDPR consent requirements
- General consumer protection

## Quick Reference

### Mandatory UI Elements

```
┌─────────────────────────────────────────────────────────┐
│ Header                                    [User Menu]   │
├─────────────────────────────────────────────────────────┤
│ [🔴 LIVE] or [📝 PAPER] mode indicator                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Main Content Area                                      │
│                                                         │
│  [AI-GENERATED] prefix on all AI outputs               │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ Risk Controls: Daily Loss: -$X / $Y limit              │
├─────────────────────────────────────────────────────────┤
│ Footer: [Terms] [Privacy] [Not Investment Advice]      │
└─────────────────────────────────────────────────────────┘
```

### Color Coding

| Color | Meaning | Usage |
|-------|---------|-------|
| 🔴 Red | Live/Critical | Live trading mode, kill switch, errors |
| 🟠 Orange | Warning | Degraded mode, approaching limits |
| 🟡 Yellow | Attention | Approval required, pending |
| 🔵 Blue | Info | Paper trading, informational |
| 🟢 Green | Success | Order executed, healthy status |

## Implementation

See [ONBOARDING_GUARDRAILS.md](./ONBOARDING_GUARDRAILS.md) for detailed implementation requirements including:
- HTML/form examples
- Acknowledgment flows
- Warning banner specifications
- Compliance checklists

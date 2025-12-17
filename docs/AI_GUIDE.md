# AI Agent Guide for TradingBot2-1

> **Target Audience**: AI Assistants (Claude, GPT, Gemini, etc.) working on this codebase.
> **Purpose**: Provide high-level context, architectural constraints, and "source of truth" pointers to avoid hallucinations and regression.

---

## 🚀 Project Overview

**TradingBot2** is a production-oriented quantitative trading platform supporting **5 asset classes**: Crypto (Binance), US Equities (Alpaca), Forex (OANDA), CME Futures (IB), and Options (Deribit/IB). It features a sophisticated Reinforcement Learning (RL) engine based on **Distributional PPO** with several advanced modifications.

**Key Metrics**: 14,000+ tests | MiFID II Compliance-Ready | EU AI Act Compliance-Ready | DORA Compliance-Ready | Designed for Production

### Key Technologies
- **RL Algorithm**: Distributional PPO (Quantile/Categorical) with Twin Critics.
- **Optimizer**: AdaptiveUPGD (Utility-Regularized Primal-Dual) - *Custom implementation*.
- **Scheduler**: Population-Based Training (PBT) with adversarial exploitation.
- **Scaling**: Variance Gradient Scaler (VGS) v3.2 - *Custom implementation*.
- **Simulation**: Full Order Book (LOB) simulation, not just OHLCV.
- **Compliance**: MiFID II (7/7 phases implemented), EU AI Act (4/4 phases implemented), DORA (5/5 phases implemented) - *Designed to align with requirements*.

---

## 🗺️ Navigation & Source of Truth

| Context | Primary File | Description |
|---------|--------------|-------------|
| **General** | `CLAUDE.md` | **The absolute source of truth.** Read this first. |
| **Status** | `DOCS_INDEX.md` | Current project status and verified features. |
| **Architecture** | `ARCHITECTURE.md` | Dependency injection, layer separation rules. |
| **Fixes** | `README.md` | List of "Latest Critical Fixes" (verified). |

### ⚠️ Critical "Do Not Touch" Areas
*Unless explicitly instructed and fully understood:*

1.  **`distributional_ppo.py` Core Logic**:
    *   **Twin Critics**: Uses `min(Q1, Q2)` for GAE targets. **DO NOT REMOVE**.
    *   **VF Clipping**: Independent clipping for each critic. **DO NOT MERGE**.
    *   **LSTM Reset**: `_reset_lstm_states_for_done_envs` is critical for preventing temporal leakage.

2.  **`optimizers/adaptive_upgd.py`**:
    *   **Learning Rate**: Must use `-1.0 * lr` (verified fix). Do not revert to `-2.0`.
    *   **Utility Normalization**: Uses Min-Max.

3.  **`variance_gradient_scaler.py`**:
    *   **Lagging Statistics**: This is a FEATURE, not a bug.
    *   **Stochastic Variance**: Uses `E[g^2]`, not `(E[g])^2`.

---

## 🏗️ Architecture & Patterns

### Layered Architecture
Strict dependency flow: `core_` -> `impl_` -> `service_` -> `strategies` -> `script_`.
*   **Core**: Interfaces, data models.
*   **Impl**: Concrete implementations (Exchange adapters, Fees).
*   **Service**: Business logic (Training, Backtest).
*   **Script**: CLI entry points (Argument parsing ONLY).

### Dependency Injection
The project uses a custom DI container (`Container`).
*   **Do not hardcode classes**. Use `container.resolve(Interface)`.
*   **Config**: Loaded via `core_config` and passed down.

---

## 🧪 Verification & Testing

*   **Tests are mandatory** for any logic change.
*   **Critical Tests**:
    *   `tests/test_twin_critics_vf_clipping_correctness.py`
    *   `tests/test_upgd_integration.py`
    *   `tests/test_lstm_episode_boundary_reset.py`

---

## 📝 Documentation Standards

*   **Language**: Russian (Primary for docs), English (Code comments/Commits).
*   **Format**: Markdown.
*   **Update Rule**: If you change code, you **MUST** update the corresponding documentation in `docs/` and `CLAUDE.md`.

---

**Last Updated**: 2025-12-08
**Version**: 1.1

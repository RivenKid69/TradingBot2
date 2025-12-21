# CustodiaCloud — Бизнес‑обзор (B2B, equities-first)

**Canonical positioning / naming / legally safe language**: `docs/DOCUMENTATION_CANON_DESIGN.md`.

**CCEA technical reference (Cloud/Agent boundary details)**: `archive/root_files/Design Doc CCEA Cloud.txt`.

**Non‑legal notice**: этот документ предназначен для описания продукта и планирования. Это **не** юридическая, инвестиционная или налоговая консультация.

---

## Кратко

CustodiaCloud — **B2B** risk‑first платформа для **квант‑исследований и деплоя** (research & deployment) для профессиональных систематических команд с **equities‑first** go‑to‑market.

Мы измеряем успех через onboarding/операционные KPI (например, time‑to‑first‑backtest, time‑to‑first‑live‑run, стабильность, качество evidence exports), а не обещаниями торговой доходности.

---

## Что именно мы продаём (committee‑friendly)

- Повторяемые workflows для research/backtesting/simulation
- Инструменты деплоя и управления жизненным циклом запусков (lifecycle control plane)
- Risk‑first контроль и governance‑пакет: evidence exports, журналы изменений, контроль доступов, телеметрия с редактированием

---

## Архитектура: CCEA (Cloud‑Controlled Execution Architecture)

CCEA — ключевой «procurement‑friendly» принцип: строгая граница Cloud/Agent.

| Компонент | Роль | Секреты | Live‑исполнение |
|---|---|---|---|
| CustodiaCloud Cloud | research/simulation/monitoring + artifact builder/registry + lifecycle control plane | **нет** | **нет** |
| CustodiaCloud Agent | секреты, риск‑enforcement, локальные approvals, формирование и отправка ордеров | **локально** | **да (у клиента)** |

**Hard rule (использовать последовательно в документах):** Cloud **спроектирован так, чтобы не хранить** broker credentials и **не отправлять** live trading instructions (orders/targets/signals). Это архитектурное ограничение и обеспечивается валидацией схемы, CI guardrails и allowlist-ами протокола. Любое live‑исполнение возможно только через customer‑controlled Agent и аккаунты клиента у брокера.

---

## Asset coverage (5 типов) — корректная рамка

Два слоя скоупа:

- **Foundation (multi‑asset by design):** equities, options, futures, FX и **optional digital assets** (как расширение, зависящее от юрисдикции/клиента).
- **MVP/beachhead:** **equities‑first**.

---

## Пилот и проверка спроса (реалистичный план)

**Pilot program (customer validation):**
- Формат: **3‑месячный** пилот‑кохорт **3–5 компаний** (planned; not yet launched)
- Условие: регулярные onboarding‑сессии и недельный feedback
- Пилот‑цена: ~**€500/мес** (discounted; illustrative)
- **Current status (internal, as of 2025-12-19)**: Pilot program not yet launched; no signed participants

**Цель пилота**: доказать repeatable onboarding и procurement‑friendly posture (CCEA + governance/evidence exports + risk‑first), а не «показать доходность».

---

## Коммерческая модель (B2B)

- Основной диапазон (illustrative): **€2,000–€5,000/мес** для beachhead‑команд; enterprise tier — по запросу.
- Deployment: BYO host для Agent (VPS/on‑prem/VPC клиента); Cloud — research/monitoring + lifecycle control (non‑orders).

---

## Финансирование и runway (committee/investor safe)

- Funding ask (illustrative): **€500K–€750K**
- Runway target: **18–24 months**
- Use of proceeds (illustrative): **40% GTM / 35% engineering / 15% operations / 10% reserve**

---

## ЕС‑план: Estonia‑first, но мульти‑страна

- Primary path: Estonia‑first (формат и требования зависят от программы и кейса).
- Параллельно сохраняем возможность подачи в другие страны ЕС (по fit программы и наличию facilitator/incubator).
- Пилоты и customer validation возможны EU/UK‑wide при операционной базе в стране учреждения.

---

## Что важно не заявлять (guardrails)

- Не использовать red-flag фрейминг (бот/«managed money»/советы/копирование/исполнение «за клиента») — см. канон: `docs/DOCUMENTATION_CANON_DESIGN.md`.
- Не делать утверждений о “compliance/certification” по MiFID II / DORA / EU AI Act и не делать self‑classification по EU AI Act без юр. проверки.
- Допустимые формулировки: “designed to support”, “evidence exports”, “privacy‑by‑design”, “customer‑controlled execution via Agent”.

---

*Updated: 2025‑12‑19*

# Contributing

Thank you for considering contributing to the CustodiaCloud codebase (`AI-Powered-Quantitative-Research-Platform` repository).

> **CCEA Architecture**: Этот проект использует [Cloud-Controlled Execution Architecture](docs/CCEA_OVERVIEW.md). Перед внесением изменений ознакомьтесь с зональными ограничениями.

---

## CCEA Zone Requirements (ОБЯЗАТЕЛЬНО)

Проект разделён на три зоны с **строгими границами импортов**:

| Зона | Пакеты | Secrets | Orders | Импорты |
|------|--------|---------|--------|---------|
| **SHARED** | `packages/shared/`, `core_*`, `impl_*`, simulation, features | No | No | Без ограничений |
| **AGENT** | `packages/agent/*`: vault, policy, execution, daemon | Yes | Yes | Может импортировать SHARED |
| **CLOUD** | `packages/cloud/*`: control_plane, builder, governance | No | No | Может импортировать SHARED, **НЕ МОЖЕТ** импортировать AGENT |

### Что ЗАПРЕЩЕНО в Cloud zone

1. **Broker/Exchange trading clients** — никаких `order_execution`, `broker_connector` модулей
2. **Secret storage** — никакого хранения API keys в Cloud БД/логах
3. **Order-like payloads** — никаких `side`, `quantity`, `price`, `order_type` полей в командах
4. **Live trading instructions** — Cloud отправляет только lifecycle requests (REQUEST_START, REQUEST_STOP)

### CI Guardrails

Перед merge ваш код проверяется автоматически:

| Check | Что проверяет | Блокирует |
|-------|---------------|-----------|
| `no-trading-libs-in-cloud` | Cloud build без order_execution | Build |
| `no-order-payloads-in-schema` | JSON schema без side/qty/price | Merge |
| `import-boundary-check` | Agent imports в Cloud | Build |
| `artifact-signature-required` | Артефакт подписан | Publish |

### Как проверить локально

```bash
# Проверить границы импортов
python -m ccea.guardrails.import_check

# Проверить JSON schema
python -m ccea.guardrails.schema_check

# Запустить все CCEA тесты
pytest tests/ccea/ -v
```

---

## Seasonality review requirements

Before submitting changes, evaluate how seasonality could impact the affected strategies and describe your findings in the pull request.
Contributions lacking a seasonality review may be delayed until this analysis is provided.

---

## Issue tracking and testing

Open an issue using the `Task` template for every piece of work. The template
requires explicit completion criteria along with links to code reviews and test
evidence. Provide this information before requesting a merge.

---

## Dataset artefact windows

Offline dataset splits are versioned contracts. When updating `configs/offline*.yml`
definitions or regenerating ADV, seasonality, or fee artefacts, ensure the
recorded data window does not extend beyond each split's end timestamp.

1. Regenerate the artefact metadata by running the relevant builder (for
   example `scripts/build_adv.py`) with the `--split` flag so the metadata block
   contains the actual window used during the refresh.
2. Execute `pytest tests/test_offline_split_windows.py` to confirm the
   regenerated metadata and offline configuration stay within the declared
   split boundaries. **Note**: This test should be implemented to validate
   that artefact windows do not extend beyond split boundaries.
3. If the check fails, adjust the artefact input window or split definition so
   the `data_window.actual.end` value never exceeds the split's `end`
   timestamp before submitting your changes.

---

## Code style and testing

- Follow existing code style (Black, isort, flake8)
- Add tests for new functionality
- Update documentation for API changes
- Run `pytest tests/` before submitting

---

## Legal posture reminder

Этот проект позиционируется как **Software Provider / ICT Provider**, а не Investment Adviser или Broker-Dealer. При добавлении новых функций:

- **НЕ** добавляйте функционал, который может быть истолкован как инвестиционные рекомендации
- **НЕ** добавляйте Cloud-исполнение ордеров
- **ВСЕГДА** сохраняйте разделение Cloud/Agent

Подробнее: [CCEA Legal Posture](docs/CCEA_OVERVIEW.md#6-legal-posture-design-doc-18)

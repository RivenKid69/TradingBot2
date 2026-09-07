# Contributing

This is a personal project, released under Apache-2.0 and not commercially developed.
Issues and pull requests are welcome; there is one maintainer, so reviews are best-effort.

By contributing you agree that your contribution is licensed under Apache-2.0, and you
are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security problems go to
the address in [SECURITY.md](SECURITY.md), not to a public issue.

## Getting set up

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -U pip wheel
pip install -r requirements-build.txt
pip install -r requirements-cpu.lock.txt
pip install -r requirements-dev.lock.txt
python setup.py build_ext --inplace               # needs a C++17 toolchain
python scripts/doctor.py --skip-network
```

Install the hooks once with `pre-commit install`; they run the same checks CI does.

## Workflow

1. Branch off `main` — `fix/...`, `feat/...`, `docs/...`.
2. Make the change. Keep the diff to one concern.
3. Run the checks below.
4. Open a pull request describing what changed and how you verified it. If the change
   touches strategy behaviour, say what you expect it to do to results and why.
5. CI must be green before merge.

## Checks

```bash
ruff check .            # lint (config in pyproject.toml)
black --check .         # formatting, line length 100
pytest -q               # full suite
python tools/check_markdown_links.py
```

Tests that need market data or credentials skip themselves with a reason that names the
command to fix it. Do not turn a skip into a failure by hard-coding a data path.

## The Cloud/Agent boundary

The codebase is split into three zones, and the split is enforced, not just documented.

| Zone | Packages | Holds secrets | Creates orders |
|---|---|---|---|
| Shared | `packages/shared/`, `core_*`, `impl_*`, simulation, features | no | no |
| Agent | `packages/agent/*` — vault, policy, execution, daemon | yes | yes |
| Cloud | `packages/cloud/*` — control plane, builder, governance | no | no |

Cloud may import Shared. Cloud may not import Agent. In Cloud code, none of the
following belongs: broker or exchange trading clients, credential storage, order-shaped
payload fields (`side`, `quantity`, `price`, `order_type`), or anything that amounts to a
live trading instruction. Cloud sends lifecycle requests; the Agent decides what to send
to a broker.

Verify locally before pushing:

```bash
python -m ccea.guardrails.import_check --target cloud --directory packages/cloud
python -m ccea.guardrails.intent_prohibition --cloud-path packages/cloud
python -m ccea.guardrails.cloud_allowlist --cloud-dir packages/cloud
python -m ccea.guardrails.schema_check docs/schemas/
python -m ccea.guardrails.protocol_check --schema docs/schemas/protocol_messages.schema.json
python -m ccea.guardrails.design_doc_check
lint-imports --config importlinter.ini
pytest tests/ccea/ -q
```

## Things that will get a change sent back

- Moving or renaming a root-level module. Imports there are by bare module name and the
  flat layout is load-bearing; new code should go into a package instead.
- Changing a config schema or a public function signature without a migration note.
- Reformatting unrelated files in the same commit.
- Widening the Cloud zone's capabilities.

## Seasonality

Liquidity seasonality multipliers affect spreads, latency and fill probability. If your
change touches execution, simulation or feature timing, say in the pull request what you
expect it to do to seasonality behaviour and how you checked. See
[docs/seasonality.md](docs/seasonality.md).

## Dataset artefact windows

Offline dataset splits are versioned contracts. When you regenerate ADV, seasonality or
fee artefacts, or edit `configs/offline*.yml`, keep the recorded data window inside the
split it belongs to:

1. Rebuild with the `--split` flag (for example `python scripts/build_adv.py --split ...`)
   so the metadata records the window actually used.
2. Run `pytest tests/test_offline_split_windows.py`.
3. If it fails, narrow the artefact window rather than widening the split.

## Positioning

The project is research and execution software, not an investment adviser and not a
broker. Do not add features that produce investment recommendations, and do not move
order execution into the cloud. See [docs/CCEA_OVERVIEW.md](docs/CCEA_OVERVIEW.md).

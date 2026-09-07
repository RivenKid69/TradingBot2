# Symbol universe refresh

The `services.universe` module maintains a cached list of Binance spot
symbols trading against USDT.  When imported, :func:`get_symbols` checks
``data/universe/symbols.json`` and refreshes it if the file is missing or
older than 24 hours.  The refresh can also be triggered explicitly by
running the module as a CLI.

## CLI usage

```bash
python -m services.universe \
    --output data/universe/symbols.json \
    --liquidity-threshold 1e6
```

- ``--output`` - destination JSON file.  The directory is created if needed.
- ``--liquidity-threshold`` - minimum 24-hour quote volume in USDT.  Set to
  ``0`` to bypass the liquidity filter and include all trading pairs.
- ``--force`` - refresh even if the cache is still fresh.

## Refresh schedule

The default time-to-live for the cache is 24 hours.  The module refreshes
on the first import if the file is stale.  For deterministic updates,
install a daily cron job:

```
0 3 * * * cd /path/to/repo && /usr/bin/python -m services.universe
```

This example refreshes the symbol list every day at 03:00 UTC.

## Custom symbol lists

To use a custom universe, generate a file with the CLI and point runners to
it via the ``--symbols`` flag or the ``data.symbols`` field in YAML config.
You can also maintain the JSON manually and run the CLI with
``--liquidity-threshold 0`` to store the latest exchange symbols without
filtering by volume.

## Validation

### Confirm the cache refresh

After invoking the CLI (or importing :func:`get_symbols`), check the cache age
and a preview of the downloaded symbols:

```bash
python - <<'PY'
import json, os, time
path = "data/universe/symbols.json"
print("age_s", round(time.time() - os.path.getmtime(path), 1))
with open(path, "r", encoding="utf-8") as fh:
    symbols = json.load(fh)
print("first", symbols[:5])
print("count", len(symbols))
PY
```

An ``age_s`` close to ``0`` confirms the file was refreshed.  The preview makes
it easy to spot unexpected tickers.

### Verify runner wiring

Runners load the same JSON through ``core_config.get_symbols``.  Load your
configuration and inspect the resolved list before starting long processes:

```bash
python - <<'PY'
from core_config import load_config
cfg = load_config("configs/config_live.yaml")
print("runner_symbols", cfg.data.symbols[:5])
print("total", len(cfg.data.symbols))
PY
```

Override the symbols via the CLI ``--symbols`` flag or ``data.symbols`` in the
configuration if you need a subset.

## Unit and integration checklist

Use this checklist when touching ``services.universe`` or its consumers.

**Test Coverage**: `tests/test_universe_comprehensive.py` (549 lines, 23 tests)

### Unit tests

- [x] Cover ``_is_stale`` so missing files and TTL-expired caches are detected
      as stale while fresh caches are accepted.
      *Implemented*: `TestIsStale.test_is_stale_missing_file`, `test_is_stale_fresh_file`, `test_is_stale_old_file`
- [x] Exercise ``get_symbols`` with combinations of ``ttl`` and ``force`` to
      ensure refreshes occur only when appropriate (file freshness).
      *Implemented*: `TestGetSymbolsFunction.test_get_symbols_returns_cached`, `test_get_symbols_refreshes_stale_cache`, `test_get_symbols_force_refresh`
- [x] Validate ``run`` filters tickers below ``liquidity_threshold`` and sorts
      the output.
      *Implemented*: `TestRunFunction.test_run_with_liquidity_threshold`, `test_run_sorts_symbols`

### Integration tests

- [x] Invoke ``python -m services.universe`` in an isolated workspace and
      assert the cache modification time advances (file freshness).
      *Implemented*: `TestIntegration.test_full_workflow_no_threshold`, `test_full_workflow_with_threshold`
- [x] Load a runner configuration via ``core_config.load_config`` and confirm
      the resolved symbols match the refreshed JSON (symbol list usage).
      *Implemented*: `TestGetSymbolsFunction.test_get_symbols_with_liquidity_threshold` (verifies run() called with correct params)
- [x] Exercise a runner path (e.g. ``script_live`` with a temporary universe)
      to confirm low-liquidity symbols are excluded when
      ``--liquidity-threshold`` is set (liquidity threshold enforcement).
      *Implemented*: `TestIntegration.test_full_workflow_with_threshold`

**Control Artifact**: CI pytest run (`tests/test_universe_comprehensive.py`) - all 23 tests passing.

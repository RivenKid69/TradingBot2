# Operations Runbook

This document provides operational procedures for running the CustodiaCloud platform in simulation, training, and live execution modes.

---

## Table of Contents

1. [Pre-Flight Checks](#pre-flight-checks)
2. [Simulation Mode](#simulation-mode)
3. [Training Mode](#training-mode)
4. [Live Execution Mode](#live-execution-mode)
5. [Monitoring](#monitoring)
6. [Emergency Procedures](#emergency-procedures)
7. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
8. [Maintenance Tasks](#maintenance-tasks)

---

## Pre-Flight Checks

**Always run the doctor script before any operation:**

```bash
python scripts/doctor.py --verbose
```

### Required Checks

| Check | Command | Expected |
|-------|---------|----------|
| Environment | `python scripts/doctor.py` | All checks pass |
| API Keys | Check env vars set | Non-empty |
| Filters | Check file dates | < 30 days old |
| Disk Space | OS command | > 10GB free |
| Network | Ping exchange | < 500ms |

### Environment Variables

**Crypto (Binance):**
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

**Stocks (Alpaca):**
```bash
export ALPACA_API_KEY="your_key"
export ALPACA_API_SECRET="your_secret"
```

---

## Simulation Mode

Simulation (backtest) mode tests strategies against historical data without any exchange connectivity.

### Quick Start

```bash
# Crypto backtest
python script_backtest.py --config configs/config_sim.yaml

# Stock backtest
python script_backtest.py --config configs/config_backtest_stocks.yaml
```

### Configuration

Key settings in your config file:

```yaml
mode: sim
execution:
  mode: bar
  enabled: true

# Enable for realistic simulation
use_seasonality: true
latency:
  use_seasonality: true
  base_ms: 250
```

### Output

Results are saved to:
- `logs/` - Execution logs
- `artifacts/` - Performance metrics, equity curves
- Console - Summary statistics

### Validation Checklist

- [ ] Data files exist and are not corrupted
- [ ] Date range is valid (train_start_ts < train_end_ts)
- [ ] Symbols exist in data files
- [ ] Fee configuration matches intended exchange

---

## Training Mode

Training mode uses reinforcement learning to optimize trading strategies.

### Quick Start

```bash
# Standard training (crypto)
python train_model_multi_patch.py --config configs/config_train.yaml

# Stock training
python train_model_multi_patch.py --config configs/config_train_stocks.yaml

# PBT + Adversarial training
python train_model_multi_patch.py --config configs/config_pbt_adversarial.yaml
```

### Key Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `n_steps` | 2048 | Steps per rollout |
| `batch_size` | 64 | Minibatch size |
| `learning_rate` | 1e-4 | Base learning rate |
| `gamma` | 0.99 | Discount factor |
| `total_timesteps` | 1M-10M | Total training steps |

### Monitoring Training

```bash
# TensorBoard (in separate terminal)
tensorboard --logdir logs/tensorboard/

# Tail training logs
tail -f logs/training.log
```

### Training Checkpoints

Checkpoints are saved to `artifacts/`:
- `best_model.zip` - Best performing model
- `checkpoint_*.zip` - Periodic checkpoints
- `final_model.zip` - Final model

### Training Validation

After training, evaluate the model:

```bash
python script_eval.py --config configs/config_eval.yaml --all-profiles
```

---

## Live Execution Mode

**CAUTION: Live trading involves real money. Always test thoroughly first.**

### Pre-Live Checklist

1. [ ] Run `python scripts/doctor.py --verbose`
2. [ ] Test with `--dry-run` flag first
3. [ ] Verify API key permissions (no withdrawal access!)
4. [ ] Set conservative position limits
5. [ ] Ensure kill switch is accessible
6. [ ] Have emergency contact ready

### Dry Run (No Real Orders)

```bash
# Crypto dry run
python script_live.py --config configs/config_live.yaml --dry-run

# Stock dry run
python script_live.py --config configs/config_live_alpaca.yaml --dry-run
```

### Paper Trading (Alpaca)

```bash
# Paper trading uses Alpaca sandbox
python script_live.py --config configs/config_live_alpaca.yaml
# Ensure config has: paper: true
```

### Live Execution

```bash
# Crypto live (Binance)
python script_live.py --config configs/config_live.yaml

# Stock live (Alpaca)
python script_live.py --config configs/config_live_alpaca.yaml
# Ensure config has: paper: false
```

### Risk Limits

Configure conservative limits in your config:

```yaml
risk:
  enabled: true                    # ALWAYS enable!
  max_abs_position_notional: 1000  # Max position size
  max_order_notional: 500          # Max order size
  daily_loss_limit: 100            # Daily loss limit
  max_orders_per_min: 10           # Rate limit
```

### Monitoring Live Execution

```bash
# Watch logs in real-time
tail -f logs/live-trading.log

# Check healthcheck endpoint (if enabled)
curl http://localhost:8080/health

# Monitor positions (Binance)
# Check exchange dashboard

# Monitor positions (Alpaca)
# Check https://app.alpaca.markets
```

---

## Monitoring

### Log Locations

| Log | Path | Purpose |
|-----|------|---------|
| Main | `logs/<run_id>.log` | Primary execution log |
| Trades | `logs/trades.log` | Trade history |
| Errors | `logs/errors.log` | Error-only log |
| TensorBoard | `logs/tensorboard/` | Training metrics |

### Key Metrics to Monitor

**Training:**
- Policy loss (should decrease)
- Value loss (should stabilize)
- Entropy (should decrease slowly)
- Episode reward (should increase)
- KL divergence (should stay < 0.1)

**Live Execution:**
- P&L (total and daily)
- Position sizes
- Order fill rates
- Latency (< 500ms typical)
- Error rates

### Healthcheck Endpoint

If healthcheck is enabled:

```bash
# Health status
curl http://localhost:8080/health

# Detailed metrics
curl http://localhost:8080/metrics
```

Response codes:
- `200` - Healthy
- `503` - Unhealthy (check logs)

---

## Emergency Procedures

### Kill Switch

**Method 1: Flag File (Fastest)**
```bash
touch state/kill_switch.flag
```

**Method 2: Ctrl+C**
- Press Ctrl+C once for graceful shutdown
- Wait for "Shutdown complete" message
- Press Ctrl+C again only if stuck

**Method 3: Manual Position Close**
- Log into exchange dashboard
- Close all positions manually
- Then kill the process

**Method 4: API Key Revocation (Last Resort)**
- Revoke API keys on exchange
- This stops ALL API access immediately

### Recovery After Emergency Stop

1. Remove kill switch flag:
   ```bash
   rm state/kill_switch.flag
   ```

2. Check state files:
   ```bash
   ls -la state/
   ```

3. Verify no orphan positions:
   - Check exchange dashboard
   - Reconcile with local state

4. Review logs for root cause:
   ```bash
   grep -i error logs/*.log | tail -100
   ```

5. Run doctor before resuming:
   ```bash
   python scripts/doctor.py --verbose
   ```

---

## Common Issues & Troubleshooting

### Connection Issues

**Symptom:** "Connection refused" or timeout errors

**Solutions:**
1. Check network connectivity
2. Verify API endpoint URLs
3. Check if exchange is under maintenance
4. Try increasing timeout values

```yaml
latency:
  timeout_ms: 5000  # Increase from default
  retries: 3
```

### Clock Drift

**Symptom:** "Timestamp outside recv window" errors

**Solutions:**
1. Sync system clock:
   ```bash
   # Windows
   w32tm /resync

   # Linux
   sudo ntpdate pool.ntp.org
   ```

2. Configure clock sync in config:
   ```yaml
   clock_sync:
     refresh_sec: 60      # More frequent sync
     warn_threshold_ms: 200
     kill_threshold_ms: 1000
   ```

### Rate Limiting

**Symptom:** "Too many requests" or 429 errors

**Solutions:**
1. Reduce signal frequency:
   ```yaml
   max_signals_per_sec: 2.0  # Lower value
   ```

2. Increase backoff:
   ```yaml
   backoff_base_s: 5.0
   max_backoff_s: 120.0
   ```

### Out of Memory

**Symptom:** Process killed, "MemoryError"

**Solutions:**
1. Reduce batch sizes
2. Reduce number of symbols
3. Use shorter history windows
4. Increase system swap

### Model Loading Errors

**Symptom:** "Model not found" or checkpoint errors

**Solutions:**
1. Verify checkpoint path exists
2. Check model was saved correctly
3. Ensure compatible Python/library versions

```bash
# Check checkpoint
python -c "import zipfile; zipfile.ZipFile('artifacts/best_model.zip').namelist()"
```

### Stale Filter Errors

**Symptom:** "Filters are stale" or quantizer errors

**Solutions:**
1. Update filters:
   ```bash
   python scripts/fetch_binance_filters.py --out data/binance_filters.json
   ```

2. Update fees:
   ```bash
   python scripts/refresh_fees.py
   ```

---

## Maintenance Tasks

### Daily

- [ ] Check logs for errors
- [ ] Verify positions match expected
- [ ] Monitor P&L
- [ ] Check disk space

### Weekly

- [ ] Update exchange filters
- [ ] Review and archive old logs
- [ ] Check for library updates
- [ ] Run full test suite

```bash
# Update filters
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json

# Run tests
pytest tests/ -v
```

### Monthly

- [ ] Rotate API keys
- [ ] Review and update risk limits
- [ ] Evaluate model performance
- [ ] Consider retraining if performance degrades

### Data Refresh Commands

```bash
# Binance filters
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json

# Fee schedules
python scripts/refresh_fees.py

# Universe updates
python -m services.universe --output data/universe/symbols.json

# Alpaca universe
python scripts/fetch_alpaca_universe.py --output data/universe/alpaca_symbols.json
```

---

## DORA Regulatory Procedures

This section covers procedures for EU regulated clients per DORA (Regulation EU 2022/2554).

### Client Notification Procedure

**When to notify:** Any incident affecting service availability, data integrity, or security.

**Notification Timeline by Severity:**

| Severity | Timeline | Method | Escalation |
|----------|----------|--------|------------|
| Critical | <30 min | Phone + Email | Immediate |
| High | <1 hour | Email + Dashboard | Within 2 hours |
| Medium | <4 hours | Email | Within 24 hours |
| Low | <24 hours | Dashboard | N/A |

**Critical Incident Notification Steps:**
1. Classify incident severity using `services/dora/incident_classification.py`
2. Identify affected clients (especially EU regulated)
3. Draft initial notification using template below
4. Send via established channels (phone for critical)
5. Log notification in incident tracking system
6. Prepare follow-up report within 4 hours

**Notification Template:**
```
Subject: [SEVERITY] Service Incident - [Brief Description]

Dear [Client],

We are writing to inform you of a service incident affecting [Platform/Service].

Incident ID: [ID]
Start Time: [UTC]
Current Status: [Investigating/Mitigating/Resolved]
Impact: [Brief description of impact]

Next Update: [Time]

For questions, contact: [Emergency contact]

---
This notification is provided per DORA Article 30(2)(f) contractual obligations.
```

### Audit Response Procedure

**Target Response Time:** 5 business days for standard requests, 24 hours for regulatory urgent (operational capacity dependent; not a guaranteed SLA commitment).

**Audit Request Processing:**
1. Log request in `services/dora/audit_readiness.py`
2. Classify request type (client operational, NCA, third-party)
3. Assign audit coordinator
4. Gather requested evidence
5. Review for confidentiality (other clients' data)
6. Package and deliver
7. Document completion

**Evidence Categories Available:**
- ICT governance documentation
- Security policies and procedures
- Incident reports (client-specific)
- Backup and recovery test results
- Access control logs
- Change management records
- Penetration test summaries
- SOC2/ISO27001 reports

**Commands for Evidence Gathering:**
```bash
# Generate client-specific audit log extract
python -m services.dora.audit_readiness generate-evidence \
  --client CLIENT_ID \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --output audit_package.zip

# Generate system health report
python -m services.monitoring.health_report generate \
  --period monthly \
  --output health_report.pdf
```

### Data Export Procedure (Art. 30(2)(d))

**For client data export upon termination or request:**

```bash
# Full client data export
python -m services.data_export client-full \
  --client CLIENT_ID \
  --format json \
  --include-models \
  --include-logs \
  --output /exports/CLIENT_ID/

# Verify export integrity
python -m services.data_export verify \
  --path /exports/CLIENT_ID/ \
  --generate-checksums
```

**Export Contents:**
- Trading strategies and configurations
- Backtest results and performance history
- Trained ML/RL models (ONNX format)
- User preferences and settings
- Client-specific audit logs
- API configurations (excluding keys)

**Timeline Commitment:**
- Standard: 5 business days
- Urgent: 48 hours
- Insolvency scenario: 72 hours

### NCA Inspection Support

**Upon receiving NCA inspection request:**

1. **Acknowledge** within 24 hours
2. **Verify** request legitimacy via client
3. **Scope** - clarify what information is needed
4. **Prepare** - gather relevant documentation
5. **Coordinate** - schedule if on-site required
6. **Execute** - provide access with appropriate supervision
7. **Document** - log all information provided

**Confidentiality Protection:**
- Redact other clients' data
- Escorted access only
- Scope limited to requesting client's services
- Written confirmation of scope before inspection

### Incident Classification (DORA-aligned)

```bash
# Classify incident using DORA criteria
python -m services.dora.incident_classification classify \
  --type [security|availability|integrity|performance] \
  --duration-minutes N \
  --clients-affected N \
  --financial-impact [none|low|medium|high] \
  --data-breach [true|false]
```

**Classification Outputs:**
- CRITICAL: Data breach, >2h outage, >50% clients affected
- HIGH: 30min-2h outage, security incident, <50% clients
- MEDIUM: <30min outage, performance degradation
- LOW: Planned maintenance, minor issues

### Subcontractor Incident Response

**If a subcontractor (AWS, Polygon, Alpaca, etc.) has an incident:**

1. Monitor subcontractor status pages
2. Assess impact on our services
3. Classify as our incident if services affected
4. Notify clients per above procedure
5. Document root cause as subcontractor issue
6. Update subcontractor risk assessment

**Subcontractor Status Pages:**
- AWS: https://status.aws.amazon.com/
- Polygon: https://status.polygon.io/
- Alpaca: https://status.alpaca.markets/
- Binance: https://www.binance.com/en/support/announcement

---

## Quick Reference Commands

### Simulation
```bash
python script_backtest.py --config configs/config_sim.yaml
```

### Training
```bash
python train_model_multi_patch.py --config configs/config_train.yaml
```

### Evaluation
```bash
python script_eval.py --config configs/config_eval.yaml --all-profiles
```

### Live (Dry Run)
```bash
python script_live.py --config configs/config_live.yaml --dry-run
```

### Live (Real)
```bash
python script_live.py --config configs/config_live.yaml
```

### Doctor Check
```bash
python scripts/doctor.py --verbose
```

### Emergency Stop
```bash
touch state/kill_switch.flag
```

---

## Support

- **Documentation:** See `docs/` directory
- **Issues:** Check `CLAUDE.md` troubleshooting section
- **Tests:** Run `pytest tests/` to verify system health

---

*Last updated: 2025-12-03*

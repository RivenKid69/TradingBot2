# Recovery Procedures

This document describes recovery procedures for common failure scenarios in CustodiaCloud.

---

## Table of Contents

1. [Network Disconnection](#1-network-disconnection)
2. [Clock Drift](#2-clock-drift)
3. [Kill Switch Triggered](#3-kill-switch-triggered)
4. [Partial Fills](#4-partial-fills)
5. [Exchange Maintenance](#5-exchange-maintenance)
6. [Process Crash](#6-process-crash)
7. [Data Corruption](#7-data-corruption)
8. [API Key Issues](#8-api-key-issues)
9. [Position Mismatch](#9-position-mismatch)
10. [Memory Exhaustion](#10-memory-exhaustion)

---

## DORA Notification Checkpoints

**For EU Regulated Clients**: All recovery procedures must include DORA notification checkpoints. Apply this framework to each scenario:

### Notification Decision Framework

```
┌─────────────────────────────────────────────────────────────┐
│               DORA NOTIFICATION CHECKPOINT                   │
├─────────────────────────────────────────────────────────────┤
│ 1. DURATION: Has outage exceeded 15 minutes?                │
│    YES → Proceed to step 2                                  │
│    NO  → Continue recovery, re-check at 15 min mark         │
│                                                             │
│ 2. IMPACT: Are EU regulated clients affected?               │
│    YES → Proceed to step 3                                  │
│    NO  → Continue recovery, document incident               │
│                                                             │
│ 3. SEVERITY:                                                │
│    - Data breach/security? → CRITICAL (notify <30 min)      │
│    - Trading unavailable?  → HIGH (notify <1 hour)          │
│    - Degraded performance? → MEDIUM (notify <4 hours)       │
│    - Minor/planned?        → LOW (notify within 24 hours)   │
│                                                             │
│ 4. NOTIFY:                                                  │
│    a) Identify affected clients (esp. critical functions)   │
│    b) Use template from OPERATIONS_RUNBOOK.md               │
│    c) Log notification in incident tracker                  │
│    d) Schedule follow-up updates                            │
└─────────────────────────────────────────────────────────────┘
```

### Quick Reference: When to Notify

| Scenario | Typical Severity | Notification Timeline |
|----------|------------------|----------------------|
| Network Disconnection (>15min) | HIGH | Within 1 hour |
| Clock Drift (kill switch triggered) | MEDIUM | Within 4 hours |
| Kill Switch (risk breach) | HIGH | Within 1 hour |
| Partial Fills | LOW | Within 24 hours |
| Exchange Maintenance (planned) | LOW | Pre-notify if possible |
| Process Crash (>15min recovery) | HIGH | Within 1 hour |
| Data Corruption | CRITICAL | Within 30 minutes |
| API Key Issues | HIGH | Within 1 hour |
| Position Mismatch | HIGH | Within 1 hour |
| Memory Exhaustion | MEDIUM | Within 4 hours |

### Evidence Preservation

During any recovery, preserve evidence for potential audits:
```bash
# Snapshot logs before any recovery action
cp -r logs/ logs_incident_$(date +%Y%m%d_%H%M%S)/

# Record state files
cp -r state/ state_incident_$(date +%Y%m%d_%H%M%S)/

# Document timeline
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) - [ACTION]" >> incident_timeline.log
```

---

## 1. Network Disconnection

**DORA Checkpoint**: If disconnection >15 minutes AND affects EU clients → Notify per severity table above.

### Symptoms
- "Connection refused" errors in logs
- WebSocket disconnect messages
- Timeout errors on API calls
- Stale market data

### Automatic Recovery
The system has built-in retry logic:
```yaml
latency:
  timeout_ms: 2500
  retries: 1
```

### Manual Recovery

**Step 1: Verify network connectivity**
```bash
# Test DNS resolution
nslookup api.binance.com

# Test connectivity
ping -c 3 api.binance.com

# Test API endpoint
curl -I https://api.binance.com/api/v3/ping
```

**Step 2: Check if exchange is reachable**
- Visit exchange status page
- Check social media for outage reports

**Step 3: If local network issue**
1. Restart network adapter
2. Check firewall rules
3. Verify proxy settings if used

**Step 4: Resume trading**
```bash
# Run doctor check first
python scripts/doctor.py --verbose

# Resume with dry-run to verify
python script_live.py --config your_config.yaml --dry-run

# If OK, resume live
python script_live.py --config your_config.yaml
```

### Prevention
- Use redundant network connections
- Configure appropriate timeouts
- Enable WebSocket heartbeats

---

## 2. Clock Drift

### Symptoms
- "Timestamp for this request is outside of the recvWindow" errors
- Orders rejected with timing errors
- Kill switch triggered by clock drift

### Detection
The system monitors clock drift:
```yaml
clock_sync:
  warn_threshold_ms: 500    # Warning at 500ms
  kill_threshold_ms: 2000   # Stop at 2000ms
```

### Recovery

**Step 1: Sync system clock**

*Windows:*
```cmd
w32tm /resync /force
```

*Linux:*
```bash
sudo ntpdate -s pool.ntp.org
# or
sudo systemctl restart systemd-timesyncd
```

*macOS:*
```bash
sudo sntp -sS pool.ntp.org
```

**Step 2: Verify sync**
```bash
# Check current offset
python -c "import time; import requests; r=requests.get('https://api.binance.com/api/v3/time'); print(f'Offset: {int(time.time()*1000) - r.json()[\"serverTime\"]}ms')"
```

**Step 3: Resume trading**
```bash
# Clear kill switch if it was triggered
rm state/kill_switch.flag

# Run doctor
python scripts/doctor.py --verbose

# Resume
python script_live.py --config your_config.yaml
```

### Prevention
- Use NTP daemon for continuous sync
- Configure stricter sync intervals:
  ```yaml
  clock_sync:
    refresh_sec: 60  # Sync every minute
  ```

---

## 3. Kill Switch Triggered

### Symptoms
- Process stops accepting new events/commands
- "Kill switch triggered" in logs
- `state/kill_switch.flag` file exists

### Causes
- Manual trigger (intentional)
- Clock drift exceeded threshold
- Risk limit breach
- Too many consecutive errors

### Recovery

**Step 1: Identify cause**
```bash
# Check kill switch state
cat state/kill_switch_state.json

# Check recent logs
grep -i "kill\|error\|trigger" logs/*.log | tail -50
```

**Step 2: Resolve root cause**
- If clock drift: sync clock (see section 2)
- If risk breach: review and adjust limits
- If errors: investigate and fix

**Step 3: Clear kill switch**
```bash
rm state/kill_switch.flag
rm state/kill_switch_state.json  # Optional: reset counters
```

**Step 4: Verify positions**
- Check exchange dashboard for open positions
- Compare with local state files

**Step 5: Resume**
```bash
python scripts/doctor.py --verbose
python script_live.py --config your_config.yaml --dry-run
# If OK:
python script_live.py --config your_config.yaml
```

### Prevention
- Set appropriate thresholds
- Monitor error rates
- Configure alerts for kill switch activation

---

## 4. Partial Fills

### Symptoms
- Order shows "PARTIALLY_FILLED" status
- Position size doesn't match expected
- Remainder of order still open

### Recovery

**Step 1: Check order status**
```python
# Via exchange API or dashboard
# Get list of open orders for the symbol
```

**Step 2: Decide action**

*Option A: Cancel remainder*
- If you don't want more fills
- Cancel via exchange dashboard or API

*Option B: Let it fill*
- If you want the full position
- Adjust limit price if needed

*Option C: Adjust position*
- Submit new order for remaining quantity

**Step 3: Sync local state**
- The system should auto-sync on next cycle
- If mismatch persists, restart the service

### Prevention
- Use market orders for immediate fills
- Configure appropriate TTL for orders:
  ```yaml
  execution_params:
    ttl_steps: 1  # Cancel if not filled in 1 bar
  ```

---

## 5. Exchange Maintenance

### Symptoms
- "Service unavailable" errors
- API returns 503 status
- WebSocket disconnects and won't reconnect

### Detection
- Check exchange status page
- Check official social media channels
- Look for scheduled maintenance announcements

### Recovery

**Step 1: Wait for maintenance to complete**
- Do NOT repeatedly retry during maintenance
- Set calendar reminders for scheduled maintenance

**Step 2: Verify exchange is back**
```bash
curl https://api.binance.com/api/v3/ping
# Should return: {}
```

**Step 3: Check for any changes**
- Review exchange announcements
- Check if any rules/limits changed
- Update filters if needed:
  ```bash
  python scripts/fetch_binance_filters.py --out data/binance_filters.json
  ```

**Step 4: Resume trading**
```bash
python scripts/doctor.py --verbose
python script_live.py --config your_config.yaml
```

### Prevention
- Subscribe to exchange status updates
- Configure graceful handling of 503 errors
- Have maintenance windows in no-trade config

---

## 6. Process Crash

### Symptoms
- Process terminates unexpectedly
- No "Shutdown complete" message
- Orphan positions may exist

### Recovery

**Step 1: Check what happened**
```bash
# Check system logs (Linux)
journalctl -u ccea-agent --since "1 hour ago"

# Check application logs
tail -100 logs/*.log

# Check for core dumps
ls -la /var/crash/ 2>/dev/null || ls -la core.* 2>/dev/null
```

**Step 2: Check for orphan positions**
- Log into exchange dashboard
- List all open positions
- Compare with expected state

**Step 3: Handle orphan positions**

*Option A: Close all positions*
- If unsure of state, close everything
- Start fresh

*Option B: Reconcile*
- Update local state to match exchange
- Resume trading

**Step 4: Investigate root cause**
- Memory exhaustion? (check `dmesg | grep -i memory`)
- Unhandled exception? (check logs)
- System issue? (check system logs)

**Step 5: Resume**
```bash
python scripts/doctor.py --verbose
python script_live.py --config your_config.yaml
```

### Prevention
- Use process supervisor (systemd, supervisord)
- Configure automatic restart
- Set up monitoring/alerts
- Ensure adequate system resources

Example systemd service:
```ini
[Unit]
Description=CustodiaCloud Live Execution
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/path/to/AI-Powered-Quantitative-Research-Platform
ExecStart=/usr/bin/python script_live.py --config configs/config_live.yaml
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

---

## 7. Data Corruption

### Symptoms
- "Invalid data" or parsing errors
- NaN values in features
- Model produces garbage predictions

### Recovery

**Step 1: Identify corrupted files**
```bash
# Check data files
python -c "import pandas as pd; df=pd.read_parquet('data/file.parquet'); print(df.isna().sum())"

# Check state files
python -c "import json; json.load(open('state/file.json'))"
```

**Step 2: Restore from backup**
```bash
# If you have backups
cp backup/data/file.parquet data/file.parquet
```

**Step 3: Regenerate if possible**
```bash
# Re-fetch filters
python scripts/fetch_binance_filters.py --out data/binance_filters.json

# Re-fetch fees
python scripts/refresh_fees.py

# Re-download data
python scripts/download_stock_data.py --symbols AAPL --start 2020-01-01
```

**Step 4: Clear state files**
```bash
# If state is corrupted, remove it
rm state/*.json

# Note: This resets position tracking!
# Manually verify and set correct positions
```

### Prevention
- Regular backups of data and state
- Validate data on load
- Use atomic writes for state files

---

## 8. API Key Issues

### Symptoms
- "Invalid API key" errors
- "Signature verification failed"
- 401/403 HTTP errors

### Recovery

**Step 1: Verify key is set**
```bash
# Check environment variable (don't print the actual value!)
echo "BINANCE_API_KEY is set: ${BINANCE_API_KEY:+yes}"
```

**Step 2: Verify key works**
```bash
# Test API connectivity
python -c "
from binance.client import Client
import os
c = Client(os.environ['BINANCE_API_KEY'], os.environ['BINANCE_API_SECRET'])
print(c.get_account_status())
"
```

**Step 3: If key is invalid**
1. Log into exchange account
2. Check if key was revoked or expired
3. Generate new API key
4. Update environment variables
5. Restart trading service

**Step 4: Check permissions**
- Ensure key has trading permission
- Ensure key does NOT have withdrawal permission
- Check IP restrictions if configured

### Prevention
- Rotate keys periodically (monthly)
- Use separate keys for different purposes
- Monitor for unauthorized access

---

## 9. Position Mismatch

### Symptoms
- Local position tracking differs from exchange
- Unexpected P&L calculations
- Risk limits triggered incorrectly

### Recovery

**Step 1: Get actual positions from exchange**
```python
# Binance
from binance.client import Client
client = Client(api_key, api_secret)
positions = client.futures_position_information()  # For futures
# or
account = client.get_account()  # For spot

# Alpaca
from alpaca.trading.client import TradingClient
client = TradingClient(api_key, secret_key)
positions = client.get_all_positions()
```

**Step 2: Compare with local state**
```bash
cat state/positions.json
```

**Step 3: Reconcile**

*Option A: Trust exchange (recommended)*
- Update local state to match exchange
- This is the source of truth

*Option B: Close and restart*
- Close all positions on exchange
- Clear local state
- Start fresh

**Step 4: Investigate cause**
- Check for missed fill notifications
- Check for network issues during order execution
- Review order history for discrepancies

### Prevention
- Periodic position reconciliation (built into system)
- Log all order events
- Use order IDs for tracking

---

## 10. Memory Exhaustion

### Symptoms
- Process killed by OOM killer
- System becomes unresponsive
- "MemoryError" in Python

### Recovery

**Step 1: Free memory**
```bash
# Check memory usage
free -h

# Find memory hogs
ps aux --sort=-%mem | head -20

# Clear system cache (Linux)
sudo sync && sudo sysctl -w vm.drop_caches=3
```

**Step 2: Restart with limits**
```bash
# Limit Python memory (Linux)
ulimit -v 8000000  # 8GB limit
python script_live.py --config your_config.yaml
```

**Step 3: Optimize configuration**
```yaml
# Reduce batch sizes
model:
  params:
    batch_size: 32      # Reduced from 64
    n_steps: 1024       # Reduced from 2048

# Reduce history
data:
  lookback_bars: 500    # Reduced from 1000
```

### Prevention
- Monitor memory usage
- Set up alerts for high memory
- Use memory profiling during development
- Consider using swap space as backup

---

## Recovery Checklist Template

Use this checklist for any recovery scenario:

```
[ ] 1. Identify the problem
    - Check logs: grep -i error logs/*.log | tail -50
    - Check system: dmesg | tail -20
    - Check exchange: visit status page

[ ] 2. Stop the bleeding
    - Trigger kill switch if needed: touch state/kill_switch.flag
    - Close positions manually if required

[ ] 3. Assess impact
    - Check current positions
    - Calculate any realized losses
    - Document the incident

[ ] 4. Fix root cause
    - Apply appropriate fix from sections above
    - Test the fix

[ ] 5. Verify system health
    - Run: python scripts/doctor.py --verbose
    - All checks should pass

[ ] 6. Resume operations
    - Start with dry-run: --dry-run flag
    - Monitor closely for first hour
    - Resume normal operations

[ ] 7. Post-incident
    - Document what happened
    - Identify prevention measures
    - Update runbook if needed
```

---

## Contact & Escalation

For issues beyond this document:

1. Check `CLAUDE.md` troubleshooting section
2. Review closed issues in project repository
3. Check exchange documentation and status pages

---

*Last updated: 2025-12-03*

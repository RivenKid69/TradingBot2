import os, sys
os.environ.setdefault("SEASONALITY_API_TOKEN", "mock_token")
sys.path.insert(0, os.getcwd())
import app
from fastapi.testclient import TestClient
c = TestClient(app.api, headers={"X-API-Key": "mock_token"})
fails = 0
def check(name, cond):
    global fails
    print(("PASS" if cond else "FAIL"), name); fails += (0 if cond else 1)

# /api/trades -> dict with simulated flag (demo fallback, no real logs)
r = c.get("/api/trades"); j = r.json()
check("trades is dict", isinstance(j, dict) and "trades" in j)
check("trades simulated flagged", j.get("simulated") is True and j.get("data_source")=="demo_mock")

# /api/risk/summary -> no hardcoded SAFE
r = c.get("/api/risk/summary"); j = r.json()
check("risk leak status not SAFE", j["leak_guard"]["status"] in ("ACTIVE","NOT_CONFIGURED"))
check("risk clock honest", j["compliance_clock"].get("data_source") in ("live","unavailable"))

# /api/ai-act/explain/<missing> -> 404 (no synthesis)
r = c.get("/api/ai-act/explain/NONEXISTENT_TX_123")
check("ai-act explain 404 on missing", r.status_code == 404)

# /api/compliance/best-execution -> demo flag
r = c.get("/api/compliance/best-execution/report"); j = r.json()
check("best-exec demo flagged", j.get("demo") is True and "disclaimer" in j)

# /api/dora/concentration-risk -> demo flag
r = c.get("/api/dora/concentration-risk"); j = r.json()
check("dora concentration demo flagged", j.get("demo") is True)

# /api/portfolio/holdings -> simulated flag (no creds)
r = c.get("/api/portfolio/holdings"); j = r.json()
check("holdings simulated flagged", j.get("simulated") is True)

# /api/copilot -> rule_based label
r = c.post("/api/copilot", json={"message":"status"}); j = r.json()
check("copilot labeled rule-based", j.get("engine")=="rule_based_advisory")

print("\nALL HONESTY CHECKS PASS" if fails==0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)

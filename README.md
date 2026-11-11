# AI Network Operations Agent

An AI-powered network troubleshooting and automated remediation system that monitors a real virtual network, detects routing failures, diagnoses root causes using a LangGraph agent (GPT-4o-mini), and safely restores connectivity.

---

## Problem Statement

Production networks fail in complex ways — interface flaps, BGP session drops, routing misconfigurations. Operators must correlate telemetry across multiple devices, identify root causes, and execute recovery steps under time pressure. This project automates that workflow using real routing software (FRRouting), real BGP/OSPF sessions, and an AI agent that reasons over live telemetry to diagnose and remediate failures.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     macOS host (Docker Desktop)                 │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   router1    │  │   router2    │  │   router3    │          │
│  │  FRRouting   │  │  FRRouting   │  │  FRRouting   │          │
│  │  AS 65001    │  │  AS 65002    │  │  OSPF only   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│      eBGP (net12)      eBGP+OSPF        OSPF (net23)            │
│                                                                 │
│  monitor.py ──► HealthState ──► incidents                       │
│                                      │                          │
│                               agent.py (LangGraph)              │
│                               6 read-only vtysh tools           │
│                                      │                          │
│                               remediation.py                    │
│                               allowlist + approval gate         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Network Topology

```
router1 ──────────[10.0.12.0/29]──────────── router2 ──────────[10.0.23.0/29]────────── router3
AS65001  10.0.12.2            10.0.12.3      AS65002  10.0.23.2            10.0.23.3
lo: 10.1.1.1/32                              lo: 10.2.2.2/32               lo: 10.3.3.3/32

eBGP:  router1 (AS65001) ↔ router2 (AS65002)
OSPF:  router2 ↔ router3, area 0
Redistribution: router2 redistributes BGP↔OSPF for end-to-end reachability
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Virtual routing | FRRouting (FRR) 8.4 in Docker |
| Routing protocols | eBGP (RFC 4271), OSPFv2 (RFC 2328) |
| Containerization | Docker Compose v2 |
| AI agent | LangGraph + GPT-4o-mini (OpenAI) |
| Telemetry | `docker exec vtysh` — real FRR CLI |
| Topology config | `topology.yml` (YAML) |
| Logging | Structured JSON |
| Testing | pytest — unit + live integration |
| Language | Python 3.11+ |

---

## BGP and OSPF Configuration

**BGP (router1 ↔ router2)**
- Type: eBGP (AS65001 ↔ AS65002)
- Timers: keepalive=3s, holdtime=9s for fast test convergence
- router2 runs `redistribute ospf` into BGP and `redistribute bgp` into OSPF

**OSPF (router2 ↔ router3)**
- Area: backbone area 0
- router2 becomes DR (higher router-id wins)
- End-to-end: router3 learns 10.1.1.1/32 via OSPF redistribution; router1 learns 10.3.3.3/32 via BGP

---

## AI Agent Workflow

```
START → agent_node ──has tool_calls?──YES──► tools (ToolNode) ──► agent_node (loop)
              │
              NO
              ▼
        finalize_node (extract JSON)
              │
             END
```

**6 read-only tools available to the LLM:**

| Tool | vtysh command |
|---|---|
| `tool_get_interfaces` | `show interface brief` |
| `tool_get_bgp_summary` | `show bgp ipv4 unicast summary` |
| `tool_get_bgp_config` | `show running-config bgpd` |
| `tool_get_ospf_neighbors` | `show ip ospf neighbor` |
| `tool_get_routing_table` | `show ip route` |
| `tool_check_connectivity` | `ping -c 3` |

The LLM never receives the fault label and has no shell access. Diagnosis output:

```json
{
  "suspected_root_cause": "...",
  "supporting_evidence": ["..."],
  "confidence": 0.0,
  "affected_devices": ["router1"],
  "recommended_action": "..."
}
```

---

## Fault Injection Scenarios

**Scenario A — Interface Shutdown**
1. Healthy baseline verified
2. `shutdown` on router1 eth0 via vtysh
3. BGP hold timer (9s) expires → session drops → end-to-end connectivity fails
4. Agent diagnoses: interface down on router1 (confidence 90%)
5. Remediation: `no shutdown` on eth0
6. BGP re-establishes; routes converge; connectivity restored

**Scenario B — BGP AS Mismatch**
1. Healthy baseline verified
2. router2: `neighbor 10.0.12.2 remote-as 99999` (was 65001)
3. OPEN message rejected → session drops
4. Agent calls `tool_get_bgp_config` on both peers, finds `remote-as 99999` vs expected 65001
5. Diagnosis: *"BGP session failure due to mismatched remote AS configuration"* (confidence 90%)
6. Remediation: restore `neighbor 10.0.12.2 remote-as 65001`
7. Full connectivity restored

---

## Remediation Safety

- **Allowlist:** 6 specific actions permitted; anything else is hard-rejected
- **Lab isolation:** router names validated against `topology.yml` containers; production names rejected
- **Human approval:** every action requires `y` confirmation before execution
- **Audit log:** all proposed/approved/rejected/executed actions written to `remediation.log` as JSON
- **Recovery verification:** BGP + OSPF + connectivity verified after every remediation

---

## Setup

**Prerequisites:** Docker Desktop, Python 3.11+, OpenAI API key

```bash
git clone https://github.com/<you>/ai-network-ops-agent
cd ai-network-ops-agent

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# add your OPENAI_API_KEY to .env

docker compose up -d
# wait ~15s for FRR to converge

docker exec router1 vtysh -c "show bgp summary"
docker exec router2 vtysh -c "show ip ospf neighbor"
```

---

## Demo

```bash
# Full demo — both scenarios, real AI, manual approval
.venv/bin/python demo.py

# Auto-approve (no prompts), mock AI — fastest
.venv/bin/python demo.py --no-ai --auto-approve

# Single scenario
.venv/bin/python demo.py --scenario A --auto-approve
.venv/bin/python demo.py --scenario B --auto-approve

# Run all tests
.venv/bin/python -m pytest tests/ -v

# Continuous monitor (Ctrl+C to stop)
.venv/bin/python -m src.monitor

# Manual inspection
docker exec router2 vtysh -c "show ip route"
docker exec router3 vtysh -c "show ip ospf neighbor detail"

# Teardown
docker compose down
```

---

## Results

All metrics measured on Apple Silicon macOS with Docker Desktop. Source: `monitor.log`, `remediation.log`.

| Scenario | Detection latency | Recovery time | Success rate | n |
|---|---|---|---|---|
| A — Interface shutdown | ~22.5s | ~15.7s | 100% | 2 |
| B — BGP AS mismatch | ~27.3s | ~21.7s | 100% | 4 |

> Detection latency includes an intentional wait period (15s for Scenario A, 20s for Scenario B) to allow protocol timers to expire. Net telemetry collection time is ~7s per poll.

**AI diagnosis accuracy (GPT-4o-mini, live network):**
- Scenario A: 1/1 correctly identified interface shutdown
- Scenario B: 1/1 correctly identified exact AS number mismatch after `tool_get_bgp_config` added

**Tests:** 34/34 passing — 30 unit (mocked), 4 live integration against real Docker containers.

---

## Limitations

- FRR image is `linux/amd64` only; runs via Rosetta emulation on Apple Silicon
- Polling-based monitoring (default 10s); sub-second detection would require gNMI streaming
- BGP timers set to 3s/9s for lab speed; production uses 60s/180s
- No persistent state store across monitor restarts

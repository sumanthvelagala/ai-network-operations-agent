"""
End-to-end demo script for the AI Network Operations Agent.

Runs both failure scenarios:
  Scenario A: Interface shutdown on router1
  Scenario B: BGP AS mismatch on router2

Usage:
    .venv/bin/python demo.py [--scenario A|B] [--no-ai] [--auto-approve]

Requires:
    - Lab running: docker compose up -d
    - OPENAI_API_KEY set in .env or environment (unless --no-ai)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from src.network_tools import collect_snapshot
from src.monitor import HealthState, _json_log
from src.remediation import execute_remediation, verify_recovery

BANNER = "=" * 65


def banner(msg: str):
    print(f"\n{BANNER}")
    print(f"  {msg}")
    print(BANNER)


def step(n: int, msg: str):
    print(f"\n[Step {n}] {msg}")


def show_health(snap) -> dict:
    state = HealthState()
    state.update(snap)
    h = state.summary()

    print(f"  Network healthy: {h['healthy']}")
    if h["bgp_down"]:
        print(f"  BGP DOWN:        {h['bgp_down']}")
    if h["ospf_not_full"]:
        print(f"  OSPF NOT FULL:   {h['ospf_not_full']}")
    if h["interfaces_down"]:
        print(f"  INTERFACES DOWN: {h['interfaces_down']}")
    if h["connectivity_failed"]:
        print(f"  CONNECTIVITY:    {h['connectivity_failed']}")

    # Connectivity detail
    for k, v in snap.connectivity.items():
        status = "OK" if v["success"] else "FAIL"
        rtt = f"{v['rtt_avg_ms']}ms" if v["rtt_avg_ms"] else "N/A"
        print(f"  {k}: {status} (loss={v['packet_loss_pct']}% rtt={rtt})")

    return h


def run_ai_diagnosis(incident: str, snap) -> dict:
    """Run the LangGraph agent. Falls back to mock if no API key."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("  [!] OPENAI_API_KEY not set — using mock diagnosis")
        from src.agent import run_diagnosis_with_mock
        return run_diagnosis_with_mock(incident)

    from src.agent import run_diagnosis
    summary_lines = []
    for router, bgp in snap.bgp_summaries.items():
        for nbr in bgp.get("neighbors", []):
            summary_lines.append(
                f"{router} BGP neighbor {nbr['neighbor']} AS{nbr['remote_as']}: "
                f"established={nbr['established']} pfx_rcvd={nbr['state_or_prefixes_rcvd']}"
            )
    for router, nbrs in snap.ospf_neighbors.items():
        for nbr in nbrs:
            summary_lines.append(
                f"{router} OSPF neighbor {nbr['neighbor_id']}: state={nbr['state']}"
            )
    for k, v in snap.connectivity.items():
        summary_lines.append(f"Connectivity {k}: success={v['success']} loss={v['packet_loss_pct']}%")

    snapshot_summary = "\n".join(summary_lines)
    return run_diagnosis(incident, snapshot_summary)


def print_diagnosis(diag: dict):
    print(f"\n  Root cause:   {diag.get('suspected_root_cause', '?')}")
    print(f"  Confidence:   {diag.get('confidence', 0):.0%}")
    print(f"  Devices:      {diag.get('affected_devices', [])}")
    print(f"  Action:       {diag.get('recommended_action', '?')}")
    print(f"  Evidence:")
    for e in diag.get("supporting_evidence", []):
        print(f"    - {e}")


# ── Scenario A: Interface shutdown ────────────────────────────────────────────

def scenario_a(auto_approve: bool, no_ai: bool):
    banner("SCENARIO A: Interface Shutdown on router1")

    t_start = time.time()

    step(1, "Confirm healthy baseline")
    snap0 = collect_snapshot()
    h0 = show_health(snap0)
    if not h0["healthy"]:
        print("  [!] Network is not healthy at start. Aborting scenario.")
        return

    step(2, "Inject fault: shut down eth0 on router1")
    result = execute_remediation(
        "disable_interface",
        {"router": "router1", "interface": "eth0"},
        require_approval=not auto_approve,
    )
    if not result["executed"]:
        print("  Fault injection cancelled.")
        return

    t_inject = time.time()
    print("  Waiting 15s for failure to propagate...")
    time.sleep(15)

    step(3, "Collect telemetry after fault")
    snap1 = collect_snapshot()
    h1 = show_health(snap1)
    t_detect = time.time()
    detection_latency = t_detect - t_inject
    print(f"\n  Detection latency: {detection_latency:.1f}s")

    step(4, "AI diagnosis")
    incident = (
        "Network anomaly detected. "
        f"Issues: BGP down={h1['bgp_down']}, "
        f"interfaces down={h1['interfaces_down']}, "
        f"connectivity failed={h1['connectivity_failed']}. "
        "Determine root cause."
    )
    if not no_ai:
        diag = run_ai_diagnosis(incident, snap1)
    else:
        from src.agent import run_diagnosis_with_mock
        diag = run_diagnosis_with_mock(incident)
    print_diagnosis(diag)

    step(5, "Automated remediation")
    t_remedy_start = time.time()
    result2 = execute_remediation(
        "enable_interface",
        {"router": "router1", "interface": "eth0"},
        require_approval=not auto_approve,
    )
    if not result2["executed"]:
        print("  Remediation cancelled.")
        return

    step(6, "Verify recovery")
    print("  Waiting up to 30s for convergence...")
    recovery = verify_recovery(timeout=30)
    t_recovered = time.time()
    recovery_time = t_recovered - t_remedy_start

    print(f"\n  Recovery: {recovery}")
    print(f"  Recovery time: {recovery_time:.1f}s")
    print(f"  Total time (inject→recovered): {t_recovered - t_inject:.1f}s")

    _json_log("INFO", "scenario_a_complete",
              detection_latency_s=round(detection_latency, 2),
              recovery_time_s=round(recovery_time, 2),
              recovered=recovery["recovered"])


# ── Scenario B: BGP AS mismatch ───────────────────────────────────────────────

def scenario_b(auto_approve: bool, no_ai: bool):
    banner("SCENARIO B: BGP AS Mismatch on router2")

    t_start = time.time()

    step(1, "Confirm healthy baseline")
    snap0 = collect_snapshot()
    h0 = show_health(snap0)
    if not h0["healthy"]:
        print("  [!] Network is not healthy at start. Aborting scenario.")
        return

    step(2, "Inject fault: introduce BGP AS mismatch on router2 (neighbor router1)")
    result = execute_remediation(
        "introduce_bgp_as_mismatch",
        {"router": "router2", "neighbor_ip": "10.0.12.2", "wrong_as": "99999"},
        require_approval=not auto_approve,
    )
    if not result["executed"]:
        print("  Fault injection cancelled.")
        return

    t_inject = time.time()
    print("  Waiting 20s for BGP session to drop (BGP hold timer = 9s)...")
    time.sleep(20)

    step(3, "Collect telemetry after fault")
    snap1 = collect_snapshot()
    h1 = show_health(snap1)
    t_detect = time.time()
    detection_latency = t_detect - t_inject
    print(f"\n  Detection latency: {detection_latency:.1f}s")

    step(4, "AI diagnosis")
    incident = (
        "Network anomaly detected. "
        f"Issues: BGP down={h1['bgp_down']}, "
        f"connectivity failed={h1['connectivity_failed']}. "
        "The BGP session has dropped. Determine root cause."
    )
    if not no_ai:
        diag = run_ai_diagnosis(incident, snap1)
    else:
        from src.agent import run_diagnosis_with_mock
        diag = run_diagnosis_with_mock(incident)
    print_diagnosis(diag)

    step(5, "Automated remediation: fix BGP AS mismatch")
    t_remedy_start = time.time()
    result2 = execute_remediation(
        "fix_bgp_as_mismatch",
        {"router": "router2", "neighbor_ip": "10.0.12.2", "correct_as": "65001"},
        require_approval=not auto_approve,
    )
    if not result2["executed"]:
        print("  Remediation cancelled.")
        return

    step(6, "Verify recovery")
    print("  Waiting up to 30s for BGP re-establishment...")
    recovery = verify_recovery(timeout=30)
    t_recovered = time.time()
    recovery_time = t_recovered - t_remedy_start

    print(f"\n  Recovery: {recovery}")
    print(f"  Recovery time: {recovery_time:.1f}s")
    print(f"  Total time (inject→recovered): {t_recovered - t_inject:.1f}s")

    _json_log("INFO", "scenario_b_complete",
              detection_latency_s=round(detection_latency, 2),
              recovery_time_s=round(recovery_time, 2),
              recovered=recovery["recovered"])


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Network Operations Agent demo")
    parser.add_argument("--scenario", choices=["A", "B", "both"], default="both",
                        help="Which scenario to run (default: both)")
    parser.add_argument("--no-ai", action="store_true",
                        help="Use mock diagnosis instead of real LLM")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Skip approval prompts (useful for testing)")
    args = parser.parse_args()

    banner("AI Network Operations Agent — Demo")
    print(f"  Scenario: {args.scenario}")
    print(f"  AI:       {'mock' if args.no_ai else 'real (GPT-4o-mini)'}")
    print(f"  Approve:  {'auto' if args.auto_approve else 'manual'}")

    if args.scenario in ("A", "both"):
        scenario_a(auto_approve=args.auto_approve, no_ai=args.no_ai)

    if args.scenario in ("B", "both"):
        # Ensure we start from a clean state
        time.sleep(5)
        scenario_b(auto_approve=args.auto_approve, no_ai=args.no_ai)

    banner("Demo complete — check monitor.log and remediation.log for full event history")


if __name__ == "__main__":
    main()

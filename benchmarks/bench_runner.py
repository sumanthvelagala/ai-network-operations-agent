"""
Performance benchmarking for the AI Network Operations Agent.

Instrumented events (all within one process, perf_counter-based):
  1. Fault injection     — vtysh command execution time
  2. Fault detection     — telemetry snapshot collected, fault visible in state
  3. Diagnosis           — LangGraph agent start → structured JSON output
  4. Remediation         — fix command execution time
  5. Convergence         — fix applied → BGP/OSPF/connectivity verified via FRRouting

Timing note:
  detection_latency_s includes an intentional sleep to let BGP hold timers expire
  (15 s for Scenario A, 20 s for Scenario B). detection_net_s subtracts that sleep
  and reflects only telemetry collection overhead.

Approval wait:
  Benchmarks always use auto-approve (require_approval=False) so approval_wait_s
  is 0.0 for all runs. It is tracked as a field so automated_recovery_latency_s
  = total_recovery_latency_s - approval_wait_s remains meaningful if the script
  is ever adapted for manual-approval runs.

Recovery verification:
  Uses verify_recovery() which polls actual FRRouting state:
    - BGP session established (show bgp ipv4 unicast summary)
    - OSPF neighbor Full (show ip ospf neighbor)
    - End-to-end ping (router1 loopback → router3 loopback)
  A run is only marked recovered=True when all three pass.

Usage:
    .venv/bin/python -m benchmarks.bench_runner [options]

    --scenario  A | B | both  (default: both)
    --mode      mock | live | both  (default: both)
    --runs      N  (default: 5, minimum enforced: 1)
    --label     string prefix for output file names

    mock  — run_diagnosis_with_mock(), no API key required
    live  — run_diagnosis() via GPT-4o-mini, requires OPENAI_API_KEY
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network_tools import collect_snapshot
from src.monitor import HealthState
from src.remediation import execute_remediation, verify_recovery
from src.agent import run_diagnosis, run_diagnosis_with_mock

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Intentional wait per scenario (seconds) for protocol timers to expire.
# BGP hold timer = 9 s; these margins ensure the session has dropped before polling.
SCENARIO_WAIT: dict[str, int] = {"A": 15, "B": 20}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkRun:
    run_id: int
    scenario: str                    # "A" or "B"
    mode: str                        # "mock" or "live"
    timestamp_utc: str

    # timing (seconds, perf_counter deltas)
    fault_injection_latency_s: float = 0.0   # vtysh inject command
    wait_for_propagation_s: float = 0.0      # intentional sleep (constant)
    detection_latency_s: float = 0.0         # inject → detect (includes sleep)
    detection_net_s: float = 0.0             # detection_latency - sleep
    diagnosis_latency_s: float = 0.0         # agent start → JSON output
    approval_wait_s: float = 0.0             # always 0.0 (auto-approve)
    remediation_latency_s: float = 0.0       # fix command execution
    convergence_latency_s: float = 0.0       # fix applied → recovery verified
    total_recovery_latency_s: float = 0.0    # inject → recovery verified
    automated_recovery_latency_s: float = 0.0  # total - approval_wait

    # outcomes
    fault_injected: bool = False
    fault_detected: bool = False
    diagnosis_correct: bool = False
    diagnosis_confidence: float = 0.0
    remediation_executed: bool = False
    recovery_verified: bool = False
    bgp_ok: bool = False
    ospf_ok: bool = False
    connectivity_ok: bool = False

    # diagnosis detail
    suspected_root_cause: str = ""
    affected_devices: list = field(default_factory=list)
    recommended_action: str = ""

    failure_reason: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_correct_diagnosis(scenario: str, diag: dict) -> bool:
    """True if the diagnosis correctly identifies the injected fault type."""
    cause = (diag.get("suspected_root_cause") or "").lower()
    if scenario == "A":
        # Must mention interface AND some indicator of shutdown / down state
        return "interface" in cause and any(
            kw in cause for kw in ["shutdown", "down", "disabled", "admin"]
        )
    if scenario == "B":
        # Must identify an AS-level misconfiguration
        return any(
            kw in cause for kw in [
                "mismatch", "misconfigur", "wrong as", "incorrect as",
                "remote-as", "remote_as", "as number", "as mismatch",
                "autonomous system", "99999",
            ]
        )
    return False


def _snapshot_healthy() -> bool:
    """Single-shot health check: collect snapshot and return is_healthy()."""
    snap = collect_snapshot()
    state = HealthState()
    state.update(snap)
    return state.is_healthy()


def _wait_for_healthy(timeout: int = 60) -> bool:
    """Block until network is healthy or timeout expires. Returns True if healthy."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if _snapshot_healthy():
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _build_snapshot_summary(snap) -> str:
    lines = []
    for router, bgp in snap.bgp_summaries.items():
        for nbr in bgp.get("neighbors", []):
            lines.append(
                f"{router} BGP neighbor {nbr['neighbor']} AS{nbr['remote_as']}: "
                f"established={nbr['established']}"
            )
    for router, nbrs in snap.ospf_neighbors.items():
        for nbr in nbrs:
            lines.append(f"{router} OSPF neighbor {nbr['neighbor_id']}: state={nbr['state']}")
    for k, v in snap.connectivity.items():
        lines.append(f"Connectivity {k}: success={v['success']} loss={v['packet_loss_pct']}%")
    return "\n".join(lines)


def _force_reset_a():
    execute_remediation(
        "enable_interface",
        {"router": "router1", "interface": "eth0"},
        require_approval=False,
    )


def _force_reset_b():
    execute_remediation(
        "fix_bgp_as_mismatch",
        {"router": "router2", "neighbor_ip": "10.0.12.2", "correct_as": 65001},
        require_approval=False,
    )


# ── Scenario runners ──────────────────────────────────────────────────────────

def run_scenario_a(run_id: int, mode: str) -> BenchmarkRun:
    run = BenchmarkRun(
        run_id=run_id,
        scenario="A",
        mode=mode,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        wait_for_propagation_s=float(SCENARIO_WAIT["A"]),
        approval_wait_s=0.0,
    )

    print(f"    [A/{mode} #{run_id}] Verifying baseline...", end=" ", flush=True)
    if not _wait_for_healthy(timeout=45):
        run.failure_reason = "Network not healthy at run start"
        print("FAIL (unhealthy)")
        return run
    print("OK")

    # ── 1. Fault injection ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    inject = execute_remediation(
        "disable_interface",
        {"router": "router1", "interface": "eth0"},
        require_approval=False,
    )
    t_injected = time.perf_counter()
    run.fault_injection_latency_s = round(t_injected - t0, 3)

    if not inject.get("executed"):
        run.failure_reason = "Fault injection not executed"
        return run
    run.fault_injected = True

    # ── 2. Wait for BGP hold timer ────────────────────────────────────────────
    print(f"    [A/{mode} #{run_id}] Injected. Waiting {SCENARIO_WAIT['A']}s for hold timer...", end=" ", flush=True)
    time.sleep(SCENARIO_WAIT["A"])
    print("done")

    # ── 3. Telemetry collection ───────────────────────────────────────────────
    snap = collect_snapshot()
    t_detected = time.perf_counter()
    run.detection_latency_s = round(t_detected - t_injected, 3)
    run.detection_net_s = round(run.detection_latency_s - run.wait_for_propagation_s, 3)

    health = HealthState()
    health.update(snap)
    summary = health.summary()
    run.fault_detected = not summary["healthy"]

    if not run.fault_detected:
        run.failure_reason = "Fault not visible in telemetry after wait"
        _force_reset_a()
        return run

    # ── 4. AI diagnosis ───────────────────────────────────────────────────────
    incident = (
        "Network anomaly detected. "
        f"BGP down: {summary['bgp_down']}, "
        f"interfaces down: {summary['interfaces_down']}, "
        f"connectivity failed: {summary['connectivity_failed']}. "
        "Determine root cause."
    )

    t_diag_start = time.perf_counter()
    if mode == "live":
        diag = run_diagnosis(incident, _build_snapshot_summary(snap))
    else:
        diag = run_diagnosis_with_mock(incident)
    t_diag_end = time.perf_counter()

    run.diagnosis_latency_s = round(t_diag_end - t_diag_start, 3)
    run.diagnosis_correct = _is_correct_diagnosis("A", diag)
    run.diagnosis_confidence = float(diag.get("confidence", 0.0))
    run.suspected_root_cause = diag.get("suspected_root_cause", "")
    run.affected_devices = diag.get("affected_devices", [])
    run.recommended_action = diag.get("recommended_action", "")

    # ── 5. Remediation ────────────────────────────────────────────────────────
    t_rem_start = time.perf_counter()
    remedy = execute_remediation(
        "enable_interface",
        {"router": "router1", "interface": "eth0"},
        require_approval=False,
    )
    t_rem_end = time.perf_counter()
    run.remediation_latency_s = round(t_rem_end - t_rem_start, 3)
    run.remediation_executed = remedy.get("executed", False)

    if not run.remediation_executed:
        run.failure_reason = "Remediation not executed"
        return run

    # ── 6. Recovery verification (real BGP/OSPF/connectivity state) ───────────
    recovery = verify_recovery(timeout=45)
    t_recovered = time.perf_counter()

    run.convergence_latency_s = round(t_recovered - t_rem_end, 3)
    run.total_recovery_latency_s = round(t_recovered - t_injected, 3)
    run.automated_recovery_latency_s = round(run.total_recovery_latency_s - run.approval_wait_s, 3)
    run.recovery_verified = recovery.get("recovered", False)
    run.bgp_ok = recovery.get("bgp_ok", False)
    run.ospf_ok = recovery.get("ospf_ok", False)
    run.connectivity_ok = recovery.get("connectivity_ok", False)

    if not run.recovery_verified:
        run.failure_reason = (
            f"Recovery not confirmed within timeout "
            f"(bgp={run.bgp_ok} ospf={run.ospf_ok} conn={run.connectivity_ok})"
        )

    print(
        f"    [A/{mode} #{run_id}] "
        f"detect={run.detection_latency_s:.1f}s "
        f"diag={run.diagnosis_latency_s:.1f}s "
        f"rem={run.remediation_latency_s:.1f}s "
        f"conv={run.convergence_latency_s:.1f}s "
        f"total={run.total_recovery_latency_s:.1f}s "
        f"correct={run.diagnosis_correct} recovered={run.recovery_verified}"
    )
    return run


def run_scenario_b(run_id: int, mode: str) -> BenchmarkRun:
    run = BenchmarkRun(
        run_id=run_id,
        scenario="B",
        mode=mode,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        wait_for_propagation_s=float(SCENARIO_WAIT["B"]),
        approval_wait_s=0.0,
    )

    print(f"    [B/{mode} #{run_id}] Verifying baseline...", end=" ", flush=True)
    if not _wait_for_healthy(timeout=45):
        run.failure_reason = "Network not healthy at run start"
        print("FAIL (unhealthy)")
        return run
    print("OK")

    # ── 1. Fault injection ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    inject = execute_remediation(
        "introduce_bgp_as_mismatch",
        {"router": "router2", "neighbor_ip": "10.0.12.2", "wrong_as": 99999},
        require_approval=False,
    )
    t_injected = time.perf_counter()
    run.fault_injection_latency_s = round(t_injected - t0, 3)

    if not inject.get("executed"):
        run.failure_reason = "Fault injection not executed"
        return run
    run.fault_injected = True

    # ── 2. Wait for BGP hold timer ────────────────────────────────────────────
    print(f"    [B/{mode} #{run_id}] Injected. Waiting {SCENARIO_WAIT['B']}s for hold timer...", end=" ", flush=True)
    time.sleep(SCENARIO_WAIT["B"])
    print("done")

    # ── 3. Telemetry collection ───────────────────────────────────────────────
    snap = collect_snapshot()
    t_detected = time.perf_counter()
    run.detection_latency_s = round(t_detected - t_injected, 3)
    run.detection_net_s = round(run.detection_latency_s - run.wait_for_propagation_s, 3)

    health = HealthState()
    health.update(snap)
    summary = health.summary()
    run.fault_detected = not summary["healthy"]

    if not run.fault_detected:
        run.failure_reason = "Fault not visible in telemetry after wait"
        _force_reset_b()
        return run

    # ── 4. AI diagnosis ───────────────────────────────────────────────────────
    incident = (
        "Network anomaly detected. "
        f"BGP down: {summary['bgp_down']}, "
        f"connectivity failed: {summary['connectivity_failed']}. "
        "The BGP session has dropped. Determine root cause, "
        "including checking BGP configuration on both peers."
    )

    t_diag_start = time.perf_counter()
    if mode == "live":
        diag = run_diagnosis(incident, _build_snapshot_summary(snap))
    else:
        diag = run_diagnosis_with_mock(incident)
    t_diag_end = time.perf_counter()

    run.diagnosis_latency_s = round(t_diag_end - t_diag_start, 3)
    run.diagnosis_correct = _is_correct_diagnosis("B", diag)
    run.diagnosis_confidence = float(diag.get("confidence", 0.0))
    run.suspected_root_cause = diag.get("suspected_root_cause", "")
    run.affected_devices = diag.get("affected_devices", [])
    run.recommended_action = diag.get("recommended_action", "")

    # ── 5. Remediation ────────────────────────────────────────────────────────
    t_rem_start = time.perf_counter()
    remedy = execute_remediation(
        "fix_bgp_as_mismatch",
        {"router": "router2", "neighbor_ip": "10.0.12.2", "correct_as": 65001},
        require_approval=False,
    )
    t_rem_end = time.perf_counter()
    run.remediation_latency_s = round(t_rem_end - t_rem_start, 3)
    run.remediation_executed = remedy.get("executed", False)

    if not run.remediation_executed:
        run.failure_reason = "Remediation not executed"
        return run

    # ── 6. Recovery verification (real BGP/OSPF/connectivity state) ───────────
    recovery = verify_recovery(timeout=45)
    t_recovered = time.perf_counter()

    run.convergence_latency_s = round(t_recovered - t_rem_end, 3)
    run.total_recovery_latency_s = round(t_recovered - t_injected, 3)
    run.automated_recovery_latency_s = round(run.total_recovery_latency_s - run.approval_wait_s, 3)
    run.recovery_verified = recovery.get("recovered", False)
    run.bgp_ok = recovery.get("bgp_ok", False)
    run.ospf_ok = recovery.get("ospf_ok", False)
    run.connectivity_ok = recovery.get("connectivity_ok", False)

    if not run.recovery_verified:
        run.failure_reason = (
            f"Recovery not confirmed within timeout "
            f"(bgp={run.bgp_ok} ospf={run.ospf_ok} conn={run.connectivity_ok})"
        )

    print(
        f"    [B/{mode} #{run_id}] "
        f"detect={run.detection_latency_s:.1f}s "
        f"diag={run.diagnosis_latency_s:.1f}s "
        f"rem={run.remediation_latency_s:.1f}s "
        f"conv={run.convergence_latency_s:.1f}s "
        f"total={run.total_recovery_latency_s:.1f}s "
        f"correct={run.diagnosis_correct} recovered={run.recovery_verified}"
    )
    return run


# ── Statistics and output ─────────────────────────────────────────────────────

def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "n": len(values),
    }


def compute_summary(runs: list[BenchmarkRun]) -> dict:
    groups: dict[tuple, list[BenchmarkRun]] = {}
    for run in runs:
        groups.setdefault((run.scenario, run.mode), []).append(run)

    summary = {}
    for (scenario, mode), grp in sorted(groups.items()):
        total = len(grp)
        injected = [r for r in grp if r.fault_injected]
        detected = [r for r in grp if r.fault_detected]
        remediated = [r for r in grp if r.remediation_executed]
        recovered = [r for r in grp if r.recovery_verified]

        summary[f"scenario_{scenario}_{mode}"] = {
            "scenario": scenario,
            "mode": mode,
            "total_runs": total,
            "fault_injected_count": len(injected),
            "fault_detected_count": len(detected),
            "remediation_executed_count": len(remediated),
            "recovery_success_count": len(recovered),
            "recovery_success_rate": round(len(recovered) / total, 3) if total else 0,
            "diagnosis_correct_count": sum(1 for r in grp if r.diagnosis_correct),
            "diagnosis_accuracy": round(
                sum(1 for r in grp if r.diagnosis_correct) / total, 3
            ) if total else 0,
            "latency": {
                "fault_injection_s":    _stats([r.fault_injection_latency_s for r in injected]),
                "detection_s":          _stats([r.detection_latency_s for r in detected]),
                "detection_net_s":      _stats([r.detection_net_s for r in detected]),
                "diagnosis_s":          _stats([r.diagnosis_latency_s for r in detected]),
                "remediation_s":        _stats([r.remediation_latency_s for r in remediated]),
                "convergence_s":        _stats([r.convergence_latency_s for r in recovered]),
                "total_recovery_s":     _stats([r.total_recovery_latency_s for r in recovered]),
                "automated_recovery_s": _stats([r.automated_recovery_latency_s for r in recovered]),
                "approval_wait_s":      _stats([r.approval_wait_s for r in grp]),
            },
            "notes": {
                "detection_latency_includes_propagation_wait_s": SCENARIO_WAIT[scenario],
                "detection_net_s_is_telemetry_collection_overhead_only": True,
                "approval_wait_always_zero_automated_benchmark": True,
                "recovery_verified_via": (
                    "BGP session established (show bgp ipv4 unicast summary) + "
                    "OSPF neighbor Full (show ip ospf neighbor) + "
                    "end-to-end ping (router1 loopback → router3 loopback)"
                ),
            },
            "failures": [
                {"run_id": r.run_id, "reason": r.failure_reason}
                for r in grp if r.failure_reason
            ],
        }
    return summary


def save_results(runs: list[BenchmarkRun], summary: dict, label: str = "") -> tuple[Path, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    prefix = f"{label}_" if label else ""

    for run in runs:
        fname = RESULTS_DIR / f"{prefix}run_{run.scenario}_{run.mode}_{run.run_id:03d}.json"
        with open(fname, "w") as f:
            json.dump(asdict(run), f, indent=2)

    summary_path = RESULTS_DIR / f"{prefix}summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "generated_utc": ts,
            "total_runs": len(runs),
            "summary": summary,
        }, f, indent=2)

    csv_path = RESULTS_DIR / f"{prefix}runs_{ts}.csv"
    if runs:
        fieldnames = list(asdict(runs[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for run in runs:
                row = asdict(run)
                row["affected_devices"] = json.dumps(row["affected_devices"])
                writer.writerow(row)

    return summary_path, csv_path


def print_summary_table(summary: dict) -> None:
    def _fmt(d: dict) -> str:
        if d["n"] == 0:
            return "n/a"
        return f"{d['mean']:.2f} / {d['median']:.2f} / {d['min']:.2f} / {d['max']:.2f}"

    print("\n" + "=" * 72)
    print("  BENCHMARK SUMMARY  (mean / median / min / max  [seconds])")
    print("=" * 72)

    for s in summary.values():
        lat = s["latency"]
        wait = s["notes"]["detection_latency_includes_propagation_wait_s"]
        print(f"\n  Scenario {s['scenario']}  |  {s['mode'].upper()} agent  |  {s['total_runs']} runs")
        print(f"  {'─' * 60}")
        print(f"  Recovery success rate : {s['recovery_success_rate']:.0%}  "
              f"({s['recovery_success_count']}/{s['total_runs']})")
        print(f"  Diagnosis accuracy    : {s['diagnosis_accuracy']:.0%}  "
              f"({s['diagnosis_correct_count']}/{s['total_runs']})")
        print(f"  Latency breakdown:")
        print(f"    fault injection     : {_fmt(lat['fault_injection_s'])}")
        print(f"    detection (total)   : {_fmt(lat['detection_s'])}  "
              f"[incl. {wait}s protocol wait]")
        print(f"    detection (net)     : {_fmt(lat['detection_net_s'])}  "
              f"[telemetry overhead only]")
        print(f"    diagnosis           : {_fmt(lat['diagnosis_s'])}")
        print(f"    remediation         : {_fmt(lat['remediation_s'])}")
        print(f"    convergence         : {_fmt(lat['convergence_s'])}")
        print(f"    total recovery      : {_fmt(lat['total_recovery_s'])}")
        print(f"    automated only      : {_fmt(lat['automated_recovery_s'])}  "
              f"[approval_wait={_fmt(lat['approval_wait_s'])}s excluded]")
        if s["failures"]:
            print(f"  Failures:")
            for f_ in s["failures"]:
                print(f"    run {f_['run_id']}: {f_['reason']}")

    print("\n" + "=" * 72)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the AI Network Operations Agent"
    )
    parser.add_argument(
        "--scenario", choices=["A", "B", "both"], default="both",
        help="Scenario to run (default: both)"
    )
    parser.add_argument(
        "--mode", choices=["mock", "live", "both"], default="both",
        help="mock=no API key, live=GPT-4o-mini (default: both)"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Runs per (scenario × mode) combination (default: 5)"
    )
    parser.add_argument(
        "--label", default="",
        help="Optional prefix for output file names"
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    modes = ["mock", "live"] if args.mode == "both" else [args.mode]
    scenarios = ["A", "B"] if args.scenario == "both" else [args.scenario]

    if "live" in modes and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Required for --mode live/both.")
        print("       Add it to .env or set in environment, or use --mode mock.")
        sys.exit(1)

    total_runs = len(scenarios) * len(modes) * args.runs
    print("=" * 65)
    print("  AI Network Operations Agent — Performance Benchmark")
    print("=" * 65)
    print(f"  Scenarios : {scenarios}")
    print(f"  Modes     : {modes}")
    print(f"  Runs each : {args.runs}")
    print(f"  Total     : {total_runs}")
    print(f"  Output    : {RESULTS_DIR}/")
    print()

    all_runs: list[BenchmarkRun] = []
    runner_map = {"A": run_scenario_a, "B": run_scenario_b}

    for scenario in scenarios:
        for mode in modes:
            print(f"\n  ── Scenario {scenario} / {mode.upper()} ──")
            runner = runner_map[scenario]

            for run_id in range(1, args.runs + 1):
                run = runner(run_id=run_id, mode=mode)
                all_runs.append(run)

                if run_id < args.runs:
                    print(f"    Waiting for healthy network before run {run_id + 1}...", end=" ", flush=True)
                    if not _wait_for_healthy(timeout=60):
                        print("WARNING: still unhealthy — force-resetting")
                        if scenario == "A":
                            _force_reset_a()
                        else:
                            _force_reset_b()
                        time.sleep(10)
                    else:
                        print("OK")

    summary = compute_summary(all_runs)
    print_summary_table(summary)
    summary_path, csv_path = save_results(all_runs, summary, label=args.label)

    print(f"\n  Saved {len(all_runs)} per-run JSON files to {RESULTS_DIR}/")
    print(f"  Summary : {summary_path.name}")
    print(f"  CSV     : {csv_path.name}")


if __name__ == "__main__":
    main()

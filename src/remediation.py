import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .network_tools import get_bgp_summary, collect_snapshot
from .topology import get_topology

REMEDIATION_LOG = Path("remediation.log")

ALLOWED_ACTIONS = {
    "enable_interface": {
        "description": "Re-enable an administratively shut-down interface",
        "params": ["router", "interface"],
        "risk": "low",
    },
    "disable_interface": {
        "description": "Administratively shut down an interface",
        "params": ["router", "interface"],
        "risk": "medium",
    },
    "clear_bgp_session": {
        "description": "Soft-clear a BGP session to force route refresh",
        "params": ["router", "neighbor_ip"],
        "risk": "low",
    },
    "restore_bgp_config": {
        "description": "Restore BGP configuration from the known-good config file",
        "params": ["router"],
        "risk": "medium",
    },
    "introduce_bgp_as_mismatch": {
        "description": "Change a BGP neighbor remote-as to wrong value (failure injection)",
        "params": ["router", "neighbor_ip", "wrong_as"],
        "risk": "medium",
    },
    "fix_bgp_as_mismatch": {
        "description": "Correct a BGP neighbor remote-as to the right value",
        "params": ["router", "neighbor_ip", "correct_as"],
        "risk": "low",
    },
}


def _log(level: str, event: str, **kwargs):
    record = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "event": event, **kwargs}
    line = json.dumps(record)
    print(line, flush=True)
    with REMEDIATION_LOG.open("a") as f:
        f.write(line + "\n")


def _vtysh(container: str, *commands: str) -> tuple[str, int]:
    args = ["docker", "exec", container, "vtysh"]
    for cmd in commands:
        args += ["-c", cmd]
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    out = "\n".join(l for l in (result.stdout + result.stderr).splitlines() if "vtysh.conf" not in l)
    return out.strip(), result.returncode


def validate_action(action: str, params: dict) -> tuple[bool, str]:
    if action not in ALLOWED_ACTIONS:
        return False, f"Action '{action}' is not in the allowlist. Allowed: {list(ALLOWED_ACTIONS)}"

    for required_param in ALLOWED_ACTIONS[action]["params"]:
        if required_param not in params:
            return False, f"Missing required parameter '{required_param}' for action '{action}'"

    lab_containers = get_topology().containers
    if "router" in params and params["router"] not in lab_containers:
        return False, f"Router '{params['router']}' is not in the lab. Lab routers: {lab_containers}"

    if action in ("enable_interface", "disable_interface") and not params.get("interface"):
        return False, "interface parameter cannot be empty"

    return True, "OK"


def enable_interface(router: str, interface: str) -> dict:
    out, rc = _vtysh(router, "conf t", f"interface {interface}", "no shutdown", "end", "write memory")
    return {"router": router, "interface": interface, "output": out, "success": rc == 0}


def disable_interface(router: str, interface: str) -> dict:
    out, rc = _vtysh(router, "conf t", f"interface {interface}", "shutdown", "end", "write memory")
    return {"router": router, "interface": interface, "output": out, "success": rc == 0}


def clear_bgp_session(router: str, neighbor_ip: str) -> dict:
    out, rc = _vtysh(router, f"clear bgp ipv4 unicast {neighbor_ip} soft")
    return {"router": router, "neighbor_ip": neighbor_ip, "output": out, "success": rc == 0}


def introduce_bgp_as_mismatch(router: str, neighbor_ip: str, wrong_as: int) -> dict:
    summary = get_bgp_summary(router)
    correct_as = next((n["remote_as"] for n in summary.get("neighbors", []) if n["neighbor"] == neighbor_ip), None)
    if correct_as is None:
        return {"success": False, "error": f"Neighbor {neighbor_ip} not found on {router}"}
    out, rc = _vtysh(
        router, "conf t", f"router bgp {summary['local_as']}",
        f"no neighbor {neighbor_ip} remote-as {correct_as}",
        f"neighbor {neighbor_ip} remote-as {wrong_as}",
        "end", "write memory",
    )
    return {"router": router, "neighbor_ip": neighbor_ip, "original_as": correct_as, "wrong_as": wrong_as, "output": out, "success": rc == 0}


def fix_bgp_as_mismatch(router: str, neighbor_ip: str, correct_as: int) -> dict:
    summary = get_bgp_summary(router)
    local_as = summary.get("local_as")
    if local_as is None:
        return {"success": False, "error": f"Could not determine local AS on {router}"}
    current_as = next((n["remote_as"] for n in summary.get("neighbors", []) if n["neighbor"] == neighbor_ip), None)
    cmds = ["conf t", f"router bgp {local_as}"]
    if current_as and current_as != correct_as:
        cmds += [f"no neighbor {neighbor_ip} remote-as {current_as}"]
    cmds += [f"neighbor {neighbor_ip} remote-as {correct_as}", "end", "write memory"]
    out, rc = _vtysh(router, *cmds)
    return {"router": router, "neighbor_ip": neighbor_ip, "correct_as": correct_as, "output": out, "success": rc == 0}


def restore_bgp_config(router: str) -> dict:
    _vtysh(router, "conf t", "end")
    _vtysh(router, "write memory")
    out = subprocess.run(["docker", "exec", router, "vtysh", "-b"], capture_output=True, text=True, timeout=15).stdout
    return {"router": router, "success": True, "output": out}


def verify_recovery(timeout: int = 30) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = collect_snapshot()
        bgp_ok = all(n["established"] for r, s in snap.bgp_summaries.items() for n in s.get("neighbors", []))
        ospf_ok = all(n["full"] for r, nbrs in snap.ospf_neighbors.items() for n in nbrs)
        conn_ok = snap.connectivity.get("r1_to_r3", {}).get("success", False)
        if bgp_ok and ospf_ok and conn_ok:
            return {"recovered": True, "bgp_ok": bgp_ok, "ospf_ok": ospf_ok, "connectivity_ok": conn_ok,
                    "time_to_recover_s": timeout - (deadline - time.time())}
        time.sleep(3)

    snap = collect_snapshot()
    return {
        "recovered": False,
        "bgp_ok": all(n["established"] for r, s in snap.bgp_summaries.items() for n in s.get("neighbors", [])),
        "ospf_ok": all(n["full"] for r, nbrs in snap.ospf_neighbors.items() for n in nbrs),
        "connectivity_ok": snap.connectivity.get("r1_to_r3", {}).get("success", False),
    }


def execute_remediation(action: str, params: dict, require_approval: bool = True) -> dict:
    ok, reason = validate_action(action, params)
    if not ok:
        _log("ERROR", "validation_failed", action=action, params=params, reason=reason)
        return {"approved": False, "executed": False, "success": False, "reason": reason}

    spec = ALLOWED_ACTIONS[action]
    _log("INFO", "action_proposed", action=action, params=params, description=spec["description"], risk=spec["risk"])

    if require_approval:
        print(f"\n{'='*60}")
        print(f"  Action:      {action}")
        print(f"  Description: {spec['description']}")
        print(f"  Risk:        {spec['risk']}")
        print(f"  Parameters:  {json.dumps(params, indent=4)}")
        print(f"{'='*60}")
        if input("Approve? [y/N]: ").strip().lower() != "y":
            _log("INFO", "action_rejected_by_user", action=action, params=params)
            return {"approved": False, "executed": False, "success": False, "reason": "Rejected by user"}

    _log("INFO", "action_approved", action=action, params=params)

    try:
        if action == "enable_interface":
            result = enable_interface(params["router"], params["interface"])
        elif action == "disable_interface":
            result = disable_interface(params["router"], params["interface"])
        elif action == "clear_bgp_session":
            result = clear_bgp_session(params["router"], params["neighbor_ip"])
        elif action == "restore_bgp_config":
            result = restore_bgp_config(params["router"])
        elif action == "introduce_bgp_as_mismatch":
            result = introduce_bgp_as_mismatch(params["router"], params["neighbor_ip"], int(params["wrong_as"]))
        elif action == "fix_bgp_as_mismatch":
            result = fix_bgp_as_mismatch(params["router"], params["neighbor_ip"], int(params["correct_as"]))
        else:
            result = {"success": False, "error": f"No executor for action '{action}'"}
    except Exception as exc:
        result = {"success": False, "error": str(exc)}

    _log(
        "INFO" if result.get("success") else "ERROR",
        "action_executed", action=action, params=params,
        success=result.get("success"), output=result.get("output", "")[:300],
    )
    return {"approved": True, "executed": True, "success": result.get("success", False), "result": result}

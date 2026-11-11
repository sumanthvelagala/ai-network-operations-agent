import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .network_tools import NetworkSnapshot, collect_snapshot
from .topology import get_topology

LOG_FILE = Path("monitor.log")
_running = True


def _json_log(level: str, event: str, **kwargs):
    record = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "event": event, **kwargs}
    line = json.dumps(record)
    print(line, flush=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def _bgp_established(snap: NetworkSnapshot) -> list[str]:
    return [
        f"{router}->{nbr['neighbor']}"
        for router, summary in snap.bgp_summaries.items()
        for nbr in summary.get("neighbors", [])
        if not nbr["established"]
    ]


def _ospf_full(snap: NetworkSnapshot) -> list[str]:
    return [
        f"{router}->ospf-{nbr['neighbor_id']}"
        for router, neighbors in snap.ospf_neighbors.items()
        for nbr in neighbors
        if not nbr["full"]
    ]


def _interfaces_down(snap: NetworkSnapshot) -> list[str]:
    return [
        f"{router}:{iface['interface']}"
        for router, ifaces in snap.interfaces.items()
        for iface in ifaces
        if iface["admin_status"] == "up" and iface["line_status"] != "up"
    ]


def _connectivity_failed(snap: NetworkSnapshot) -> list[str]:
    return [key for key, result in snap.connectivity.items() if not result["success"]]


class HealthState:
    def __init__(self):
        self.bgp_down: set = set()
        self.ospf_not_full: set = set()
        self.interfaces_down: set = set()
        self.connectivity_failed: set = set()
        self.initialized = False

    def update(self, snap: NetworkSnapshot) -> list[dict]:
        new_bgp_down = set(_bgp_established(snap))
        new_ospf_not_full = set(_ospf_full(snap))
        new_ifaces_down = set(_interfaces_down(snap))
        new_conn_failed = set(_connectivity_failed(snap))

        incidents = []
        if self.initialized:
            for item in new_bgp_down - self.bgp_down:
                incidents.append({"type": "BGP_SESSION_DOWN", "detail": item})
            for item in new_ospf_not_full - self.ospf_not_full:
                incidents.append({"type": "OSPF_NEIGHBOR_NOT_FULL", "detail": item})
            for item in new_ifaces_down - self.interfaces_down:
                incidents.append({"type": "INTERFACE_DOWN", "detail": item})
            for item in new_conn_failed - self.connectivity_failed:
                incidents.append({"type": "CONNECTIVITY_LOSS", "detail": item})
            for item in self.bgp_down - new_bgp_down:
                incidents.append({"type": "BGP_SESSION_RESTORED", "detail": item})
            for item in self.ospf_not_full - new_ospf_not_full:
                incidents.append({"type": "OSPF_NEIGHBOR_RESTORED", "detail": item})
            for item in self.interfaces_down - new_ifaces_down:
                incidents.append({"type": "INTERFACE_RESTORED", "detail": item})
            for item in self.connectivity_failed - new_conn_failed:
                incidents.append({"type": "CONNECTIVITY_RESTORED", "detail": item})

        self.bgp_down = new_bgp_down
        self.ospf_not_full = new_ospf_not_full
        self.interfaces_down = new_ifaces_down
        self.connectivity_failed = new_conn_failed
        self.initialized = True
        return incidents

    def is_healthy(self) -> bool:
        return not any([self.bgp_down, self.ospf_not_full, self.interfaces_down, self.connectivity_failed])

    def summary(self) -> dict:
        return {
            "healthy": self.is_healthy(),
            "bgp_down": list(self.bgp_down),
            "ospf_not_full": list(self.ospf_not_full),
            "interfaces_down": list(self.interfaces_down),
            "connectivity_failed": list(self.connectivity_failed),
        }


def run_monitor(interval: float = 10.0, once: bool = False) -> Optional[dict]:
    global _running
    state = HealthState()

    def _handle_signal(sig, frame):
        global _running
        _running = False
        _json_log("INFO", "monitor_stopping", signal=sig)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _json_log("INFO", "monitor_started", interval=interval, routers=get_topology().containers)

    while _running:
        t0 = time.time()
        try:
            snap = collect_snapshot()
            incidents = state.update(snap)
            health = state.summary()
            _json_log("INFO", "poll", **health, poll_duration_ms=round((time.time() - t0) * 1000))
            for incident in incidents:
                level = "INFO" if "RESTORED" in incident["type"] else "ERROR"
                _json_log(level, "incident", **incident)
        except Exception as exc:
            _json_log("ERROR", "poll_error", error=str(exc))

        if once:
            return state.summary()

        time.sleep(max(0.0, interval - (time.time() - t0)))

    _json_log("INFO", "monitor_stopped")
    return None


if __name__ == "__main__":
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    run_monitor(interval=interval)

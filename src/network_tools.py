import subprocess
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from .topology import get_topology


def run_vtysh(container: str, command: str, timeout: int = 10) -> tuple[str, str]:
    result = subprocess.run(
        ["docker", "exec", container, "vtysh", "-c", command],
        capture_output=True, text=True, timeout=timeout,
    )
    stdout = result.stdout.strip()
    stderr = "\n".join(
        line for line in result.stderr.splitlines()
        if "vtysh.conf" not in line
    ).strip()
    return stdout, stderr


def get_interfaces(container: str) -> list[dict]:
    out, _ = run_vtysh(container, "show interface brief")
    interfaces = []
    for line in out.splitlines():
        if line.startswith("Interface") or line.startswith("-") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        interfaces.append({
            "interface": parts[0],
            "admin_status": parts[1],
            "line_status": parts[1],
            "addresses": [p for p in parts[3:] if re.match(r'\d+\.\d+\.\d+\.\d+/', p)] if len(parts) > 3 else [],
        })
    return interfaces


def get_bgp_summary(container: str) -> dict:
    out, _ = run_vtysh(container, "show bgp ipv4 unicast summary")
    result: dict = {"router_id": None, "local_as": None, "neighbors": []}

    m = re.search(r'BGP router identifier (\S+), local AS number (\d+)', out)
    if m:
        result["router_id"] = m.group(1)
        result["local_as"] = int(m.group(2))

    for line in out.splitlines():
        m = re.match(
            r'^(\d+\.\d+\.\d+\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)',
            line.strip()
        )
        if m:
            pfx_rcvd = m.group(7)
            result["neighbors"].append({
                "neighbor": m.group(1),
                "remote_as": int(m.group(3)),
                "msg_rcvd": int(m.group(4)),
                "msg_sent": int(m.group(5)),
                "up_down": m.group(6),
                "state_or_prefixes_rcvd": pfx_rcvd,
                "prefixes_sent": m.group(8),
                "established": pfx_rcvd.isdigit(),
            })
    return result


def get_ospf_neighbors(container: str) -> list[dict]:
    out, _ = run_vtysh(container, "show ip ospf neighbor")
    neighbors = []
    for line in out.splitlines():
        m = re.match(
            r'^(\d+\.\d+\.\d+\.\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+\.\d+\.\d+\.\d+)\s+(\S+)',
            line.strip()
        )
        if m:
            state = m.group(3)
            neighbors.append({
                "neighbor_id": m.group(1),
                "priority": int(m.group(2)),
                "state": state,
                "up_time": m.group(4),
                "dead_time": m.group(5),
                "address": m.group(6),
                "interface": m.group(7),
                "full": "Full" in state,
            })
    return neighbors


def get_routing_table(container: str) -> list[dict]:
    out, _ = run_vtysh(container, "show ip route")
    routes = []
    for line in out.splitlines():
        m = re.match(
            r'^([BCOSK>*]+)\s+(\d+\.\d+\.\d+\.\d+/\d+)\s+\[(\d+)/(\d+)\]\s+via\s+(\d+\.\d+\.\d+\.\d+)',
            line.strip()
        )
        if m:
            flags = m.group(1)
            routes.append({
                "flags": flags,
                "prefix": m.group(2),
                "admin_distance": int(m.group(3)),
                "metric": int(m.group(4)),
                "next_hop": m.group(5),
                "protocol": _flag_to_protocol(flags),
                "selected": ">" in flags,
                "fib": "*" in flags,
            })
        m2 = re.match(
            r'^([CS>*]+)\s+(\d+\.\d+\.\d+\.\d+/\d+)\s+is directly connected',
            line.strip()
        )
        if m2:
            flags = m2.group(1)
            routes.append({
                "flags": flags,
                "prefix": m2.group(2),
                "admin_distance": 0,
                "metric": 0,
                "next_hop": "directly connected",
                "protocol": "connected",
                "selected": ">" in flags,
                "fib": "*" in flags,
            })
    return routes


def _flag_to_protocol(flags: str) -> str:
    if "B" in flags: return "bgp"
    if "O" in flags: return "ospf"
    if "K" in flags: return "kernel"
    if "C" in flags: return "connected"
    if "S" in flags: return "static"
    return "unknown"


def get_bgp_config(container: str) -> str:
    out, _ = run_vtysh(container, "show running-config bgpd")
    return out


def check_connectivity(src_container: str, dst_ip: str, src_ip: Optional[str] = None) -> dict:
    cmd = ["docker", "exec", src_container, "ping", "-c", "3", "-W", "2"]
    if src_ip:
        cmd += ["-I", src_ip]
    cmd.append(dst_ip)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    out = result.stdout

    packet_loss = 100
    rtt_avg = None

    m = re.search(r'(\d+)% packet loss', out)
    if m:
        packet_loss = int(m.group(1))
    m = re.search(r'min/avg/max = [\d.]+/([\d.]+)/', out)
    if m:
        rtt_avg = float(m.group(1))

    return {
        "src": src_container,
        "dst": dst_ip,
        "success": result.returncode == 0,
        "packet_loss_pct": packet_loss,
        "rtt_avg_ms": rtt_avg,
    }


@dataclass
class NetworkSnapshot:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    interfaces: dict = field(default_factory=dict)
    bgp_summaries: dict = field(default_factory=dict)
    ospf_neighbors: dict = field(default_factory=dict)
    routing_tables: dict = field(default_factory=dict)
    connectivity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "interfaces": self.interfaces,
            "bgp_summaries": self.bgp_summaries,
            "ospf_neighbors": self.ospf_neighbors,
            "routing_tables": self.routing_tables,
            "connectivity": self.connectivity,
        }


def collect_snapshot() -> NetworkSnapshot:
    topo = get_topology()
    snap = NetworkSnapshot()
    for container in topo.containers:
        snap.interfaces[container] = get_interfaces(container)
        snap.bgp_summaries[container] = get_bgp_summary(container)
        snap.ospf_neighbors[container] = get_ospf_neighbors(container)
        snap.routing_tables[container] = get_routing_table(container)
    for check in topo.connectivity_checks:
        snap.connectivity[check.name] = check_connectivity(
            check.src_router, check.dst_ip, src_ip=check.src_ip
        )
    return snap

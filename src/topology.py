"""
Topology loader — reads topology.yml and exposes typed helpers.
All other modules import from here instead of hardcoding router names.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

_DEFAULT_PATH = Path(__file__).parent.parent / "topology.yml"


@dataclass
class RouterInfo:
    name: str
    container: str
    asn: Optional[int]
    loopback: str
    role: str


@dataclass
class LinkEndpoint:
    router: str
    ip: str          # e.g. "10.0.12.2/29"


@dataclass
class Link:
    name: str
    endpoints: list[LinkEndpoint]
    protocol: str    # "BGP", "OSPF", etc.


@dataclass
class ConnCheck:
    name: str
    src_router: str
    src_ip: str
    dst_ip: str


@dataclass
class Topology:
    routers: dict[str, RouterInfo]      # keyed by logical name
    links: list[Link]
    connectivity_checks: list[ConnCheck]

    # ── convenience accessors ────────────────────────────────────────────────

    @property
    def router_names(self) -> list[str]:
        return list(self.routers.keys())

    @property
    def containers(self) -> list[str]:
        return [r.container for r in self.routers.values()]

    def container_for(self, router_name: str) -> str:
        return self.routers[router_name].container

    def bgp_links(self) -> list[Link]:
        return [l for l in self.links if "BGP" in l.protocol.upper()]

    def ospf_links(self) -> list[Link]:
        return [l for l in self.links if "OSPF" in l.protocol.upper()]

    def system_prompt_description(self) -> str:
        """Build a topology paragraph for the LLM system prompt."""
        lines = ["Network topology:"]
        for name, r in self.routers.items():
            asn_str = f"AS{r.asn}" if r.asn else "no BGP AS"
            lines.append(f"- {name} ({asn_str}): {r.role}. Loopback {r.loopback}.")
        lines.append("")
        lines.append("Links:")
        for link in self.links:
            eps = ", ".join(f"{e.router}={e.ip}" for e in link.endpoints)
            lines.append(f"- {link.name} [{link.protocol}]: {eps}")
        return "\n".join(lines)


def load_topology(path: Path = _DEFAULT_PATH) -> Topology:
    """Load and parse topology.yml. Raises FileNotFoundError if missing."""
    with open(path) as f:
        data = yaml.safe_load(f)

    routers = {
        name: RouterInfo(
            name=name,
            container=cfg["container"],
            asn=cfg.get("asn"),
            loopback=cfg["loopback"],
            role=cfg.get("role", ""),
        )
        for name, cfg in data["routers"].items()
    }

    links = [
        Link(
            name=lnk["name"],
            endpoints=[LinkEndpoint(**ep) for ep in lnk["endpoints"]],
            protocol=lnk.get("protocol", ""),
        )
        for lnk in data.get("links", [])
    ]

    checks = [
        ConnCheck(**c) for c in data.get("connectivity_checks", [])
    ]

    return Topology(routers=routers, links=links, connectivity_checks=checks)


# Module-level singleton — loaded once on import
_topo: Optional[Topology] = None


def get_topology() -> Topology:
    global _topo
    if _topo is None:
        _topo = load_topology()
    return _topo

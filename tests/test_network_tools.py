"""
Unit and integration tests for network telemetry parsing.

Unit tests use mock subprocess output (no Docker required).
Integration tests require the lab to be running.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network_tools import (
    get_bgp_summary,
    get_interfaces,
    get_ospf_neighbors,
    get_routing_table,
    check_connectivity,
)
from src.topology import get_topology


# ── Mock vtysh output samples ─────────────────────────────────────────────────

MOCK_INTERFACE_BRIEF = """Interface       Status  VRF             Addresses
---------       ------  ---             ---------
eth0            up      default         10.0.12.2/29
lo              up      default         10.1.1.1/32
"""

MOCK_BGP_SUMMARY_ESTABLISHED = """
IPv4 Unicast Summary (VRF default):
BGP router identifier 10.1.1.1, local AS number 65001 vrf-id 0
BGP table version 3
RIB entries 5, using 960 bytes of memory
Peers 1, using 717 KiB of memory

Neighbor        V         AS   MsgRcvd   MsgSent   TblVer  InQ OutQ  Up/Down State/PfxRcd   PfxSnt Desc
10.0.12.3       4      65002       290       294        0    0    0 00:13:54            2        3 R2-eBGP
"""

MOCK_BGP_SUMMARY_DOWN = """
IPv4 Unicast Summary (VRF default):
BGP router identifier 10.1.1.1, local AS number 65001 vrf-id 0
BGP table version 1
RIB entries 1, using 192 bytes of memory
Peers 1, using 717 KiB of memory

Neighbor        V         AS   MsgRcvd   MsgSent   TblVer  InQ OutQ  Up/Down State/PfxRcd   PfxSnt Desc
10.0.12.3       4      65002         0         0        0    0    0 never       Active              0 R2-eBGP
"""

MOCK_OSPF_NEIGHBOR_FULL = """Neighbor ID     Pri State           Up Time         Dead Time Address         Interface                        RXmtL RqstL DBsmL
10.3.3.3          1 Full/DR         11m28s            36.584s 10.0.23.3       eth1:10.0.23.2                       0     0     0
"""

MOCK_OSPF_NEIGHBOR_EMPTY = ""

MOCK_ROUTING_TABLE = """Codes: K - kernel route, C - connected, S - static, R - RIP,
       O - OSPF, I - IS-IS, B - BGP, E - EIGRP, N - NHRP,
       T - Table, v - VNC, V - VNC-Direct, A - Babel, F - PBR,
       f - OpenFabric,
       > - selected route, * - FIB route, q - queued, r - rejected, b - backup

K>* 0.0.0.0/0 [0/0] via 10.0.12.1, eth0, 00:14:19
C>* 10.0.12.0/29 is directly connected, eth0, 00:14:19
C>* 10.1.1.1/32 is directly connected, lo, 00:14:19
B>* 10.2.2.2/32 [20/0] via 10.0.12.3, eth0, weight 1, 00:00:43
B>* 10.3.3.3/32 [20/10] via 10.0.12.3, eth0, weight 1, 00:00:43
"""

MOCK_PING_SUCCESS = """PING 10.3.3.3 (10.3.3.3) from 10.1.1.1: 56 data bytes
64 bytes from 10.3.3.3: seq=0 ttl=63 time=0.354 ms
64 bytes from 10.3.3.3: seq=1 ttl=63 time=0.625 ms
64 bytes from 10.3.3.3: seq=2 ttl=63 time=0.309 ms

--- 10.3.3.3 ping statistics ---
3 packets transmitted, 3 packets received, 0% packet loss
round-trip min/avg/max = 0.309/0.429/0.625 ms
"""

MOCK_PING_FAILURE = """PING 10.3.3.3 (10.3.3.3) from 10.1.1.1: 56 data bytes

--- 10.3.3.3 ping statistics ---
3 packets transmitted, 0 packets received, 100% packet loss
"""


def _mock_run_vtysh(container, command, timeout=10):
    """Return mock vtysh output based on command."""
    if "interface brief" in command:
        return MOCK_INTERFACE_BRIEF, ""
    if "bgp ipv4 unicast summary" in command or "bgp summary" in command:
        if container == "router_down":
            return MOCK_BGP_SUMMARY_DOWN, ""
        return MOCK_BGP_SUMMARY_ESTABLISHED, ""
    if "ospf neighbor" in command:
        if container == "router3":
            return MOCK_OSPF_NEIGHBOR_FULL, ""
        return MOCK_OSPF_NEIGHBOR_EMPTY, ""
    if "ip route" in command:
        return MOCK_ROUTING_TABLE, ""
    return "", ""


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestInterfaceParsing(unittest.TestCase):

    @patch("src.network_tools.run_vtysh", side_effect=_mock_run_vtysh)
    def test_interfaces_parsed(self, _mock):
        ifaces = get_interfaces("router1")
        names = [i["interface"] for i in ifaces]
        self.assertIn("eth0", names)
        self.assertIn("lo", names)

    @patch("src.network_tools.run_vtysh", side_effect=_mock_run_vtysh)
    def test_interface_has_address(self, _mock):
        ifaces = get_interfaces("router1")
        eth0 = next(i for i in ifaces if i["interface"] == "eth0")
        self.assertIn("10.0.12.2/29", eth0["addresses"])

    @patch("src.network_tools.run_vtysh", side_effect=_mock_run_vtysh)
    def test_interface_status_up(self, _mock):
        ifaces = get_interfaces("router1")
        eth0 = next(i for i in ifaces if i["interface"] == "eth0")
        self.assertEqual(eth0["admin_status"], "up")


class TestBGPParsing(unittest.TestCase):

    @patch("src.network_tools.run_vtysh", side_effect=_mock_run_vtysh)
    def test_bgp_established(self, _mock):
        summary = get_bgp_summary("router1")
        self.assertEqual(summary["local_as"], 65001)
        self.assertEqual(len(summary["neighbors"]), 1)
        nbr = summary["neighbors"][0]
        self.assertTrue(nbr["established"])
        self.assertEqual(nbr["remote_as"], 65002)

    @patch("src.network_tools.run_vtysh", side_effect=_mock_run_vtysh)
    def test_bgp_down_detected(self, _mock):
        summary = get_bgp_summary("router_down")
        # Should parse without crashing even if neighbor is in Active state
        self.assertIsInstance(summary, dict)

    @patch("src.network_tools.run_vtysh", side_effect=_mock_run_vtysh)
    def test_bgp_router_id_parsed(self, _mock):
        summary = get_bgp_summary("router1")
        self.assertEqual(summary["router_id"], "10.1.1.1")


class TestOSPFParsing(unittest.TestCase):

    @patch("src.network_tools.run_vtysh", side_effect=_mock_run_vtysh)
    def test_ospf_full_state(self, _mock):
        nbrs = get_ospf_neighbors("router3")
        self.assertEqual(len(nbrs), 1)
        self.assertEqual(nbrs[0]["state"], "Full/DR")
        self.assertTrue(nbrs[0]["full"])

    @patch("src.network_tools.run_vtysh", side_effect=_mock_run_vtysh)
    def test_ospf_empty_on_bgp_only_router(self, _mock):
        nbrs = get_ospf_neighbors("router1")  # router1 has no OSPF
        self.assertEqual(nbrs, [])


class TestRoutingTableParsing(unittest.TestCase):

    @patch("src.network_tools.run_vtysh", side_effect=_mock_run_vtysh)
    def test_routes_parsed(self, _mock):
        routes = get_routing_table("router1")
        prefixes = [r["prefix"] for r in routes]
        self.assertIn("10.3.3.3/32", prefixes)
        self.assertIn("10.1.1.1/32", prefixes)

    @patch("src.network_tools.run_vtysh", side_effect=_mock_run_vtysh)
    def test_bgp_route_protocol(self, _mock):
        routes = get_routing_table("router1")
        r3_route = next(r for r in routes if r["prefix"] == "10.3.3.3/32")
        self.assertEqual(r3_route["protocol"], "bgp")

    @patch("src.network_tools.run_vtysh", side_effect=_mock_run_vtysh)
    def test_connected_route_protocol(self, _mock):
        routes = get_routing_table("router1")
        lo_route = next(r for r in routes if r["prefix"] == "10.1.1.1/32")
        self.assertEqual(lo_route["protocol"], "connected")


class TestConnectivityParsing(unittest.TestCase):

    @patch("subprocess.run")
    def test_ping_success_parsed(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=MOCK_PING_SUCCESS, stderr=""
        )
        result = check_connectivity("router1", "10.3.3.3", src_ip="10.1.1.1")
        self.assertTrue(result["success"])
        self.assertEqual(result["packet_loss_pct"], 0)
        self.assertIsNotNone(result["rtt_avg_ms"])

    @patch("subprocess.run")
    def test_ping_failure_parsed(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout=MOCK_PING_FAILURE, stderr=""
        )
        result = check_connectivity("router1", "10.3.3.3", src_ip="10.1.1.1")
        self.assertFalse(result["success"])
        self.assertEqual(result["packet_loss_pct"], 100)


# ── Integration tests (require live Docker lab) ───────────────────────────────

_first_container = get_topology().containers[0]
@unittest.skipUnless(
    subprocess.run(["docker", "inspect", _first_container], capture_output=True).returncode == 0,
    f"Lab not running (docker inspect {_first_container} failed)"
)
class TestLiveNetwork(unittest.TestCase):

    def test_live_bgp_established(self):
        summary = get_bgp_summary("router1")
        self.assertTrue(len(summary["neighbors"]) > 0)
        nbr = summary["neighbors"][0]
        self.assertTrue(nbr["established"], f"BGP not established: {nbr}")

    def test_live_ospf_full(self):
        nbrs = get_ospf_neighbors("router2")
        self.assertTrue(len(nbrs) > 0)
        self.assertTrue(nbrs[0]["full"], f"OSPF not Full: {nbrs[0]}")

    def test_live_r1_to_r3_connectivity(self):
        result = check_connectivity("router1", "10.3.3.3", src_ip="10.1.1.1")
        self.assertTrue(result["success"], f"Ping failed: {result}")
        self.assertEqual(result["packet_loss_pct"], 0)

    def test_live_r3_has_route_to_r1(self):
        routes = get_routing_table("router3")
        r1_route = next((r for r in routes if r["prefix"] == "10.1.1.1/32"), None)
        self.assertIsNotNone(r1_route, "R3 has no route to R1 loopback")


if __name__ == "__main__":
    unittest.main(verbosity=2)

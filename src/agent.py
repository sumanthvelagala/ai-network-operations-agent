import json
import os
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .network_tools import (
    get_interfaces, get_bgp_summary, get_bgp_config,
    get_ospf_neighbors, get_routing_table, check_connectivity,
)
from .topology import get_topology

load_dotenv()


def _build_system_prompt() -> str:
    topo = get_topology()
    return f"""You are a senior network operations engineer performing root-cause analysis on a virtual network.

{topo.system_prompt_description()}

Use your tools systematically following this diagnostic ladder:

STEP 1 — Check interfaces on all affected routers (tool_get_interfaces).
STEP 2 — Check BGP session states (tool_get_bgp_summary).
STEP 3 — If interfaces are UP but BGP is NOT established, ALWAYS call tool_get_bgp_config
          on both BGP peers to compare their configured remote-as values.
STEP 4 — Check OSPF neighbors (tool_get_ospf_neighbors) on transit routers.
STEP 5 — Check routing tables (tool_get_routing_table) to see which prefixes are missing.
STEP 6 — Run connectivity checks (tool_check_connectivity) to confirm impact.

Your final answer MUST be valid JSON matching this schema exactly:
{{
  "suspected_root_cause": "<brief description>",
  "supporting_evidence": ["<evidence item 1>", ...],
  "confidence": 0.0,
  "affected_devices": ["<router name>", ...],
  "recommended_action": "<specific action to take>"
}}
"""


@tool
def tool_get_interfaces(router: str) -> str:
    """Get interface status and IP addresses for a router (router1, router2, or router3)."""
    if router not in get_topology().containers:
        return f"Error: unknown router '{router}'. Valid: {get_topology().containers}"
    return json.dumps(get_interfaces(router), indent=2)


@tool
def tool_get_bgp_summary(router: str) -> str:
    """Get BGP neighbor summary (session state, prefix counts) for a router."""
    if router not in get_topology().containers:
        return f"Error: unknown router '{router}'. Valid: {get_topology().containers}"
    return json.dumps(get_bgp_summary(router), indent=2)


@tool
def tool_get_bgp_config(router: str) -> str:
    """Get the BGP running configuration for a router. Shows neighbor remote-as values — use this to detect AS misconfigurations."""
    if router not in get_topology().containers:
        return f"Error: unknown router '{router}'. Valid: {get_topology().containers}"
    return get_bgp_config(router)


@tool
def tool_get_ospf_neighbors(router: str) -> str:
    """Get OSPF neighbor states for a router."""
    if router not in get_topology().containers:
        return f"Error: unknown router '{router}'. Valid: {get_topology().containers}"
    return json.dumps(get_ospf_neighbors(router), indent=2)


@tool
def tool_get_routing_table(router: str) -> str:
    """Get the IP routing table for a router."""
    if router not in get_topology().containers:
        return f"Error: unknown router '{router}'. Valid: {get_topology().containers}"
    return json.dumps(get_routing_table(router), indent=2)


@tool
def tool_check_connectivity(src_router: str, dst_ip: str, src_ip: str = "") -> str:
    """Ping dst_ip from src_router. Optionally bind to src_ip. Returns packet loss and RTT."""
    if src_router not in get_topology().containers:
        return f"Error: unknown router '{src_router}'. Valid: {get_topology().containers}"
    return json.dumps(check_connectivity(src_router, dst_ip, src_ip=src_ip or None), indent=2)


TOOLS = [
    tool_get_interfaces, tool_get_bgp_summary, tool_get_bgp_config,
    tool_get_ospf_neighbors, tool_get_routing_table, tool_check_connectivity,
]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    diagnosis: dict


def _make_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=4096).bind_tools(TOOLS)


def agent_node(state: AgentState) -> AgentState:
    return {"messages": [_make_llm().invoke(state["messages"])]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "finalize"


def finalize_node(state: AgentState) -> AgentState:
    content = state["messages"][-1].content
    try:
        m = re.search(r'\{[^{}]*"suspected_root_cause"[^{}]*\}', content, re.DOTALL)
        diagnosis = json.loads(m.group(0)) if m else json.loads(content)
    except (json.JSONDecodeError, ValueError, AttributeError):
        diagnosis = {
            "suspected_root_cause": "Unable to parse structured output",
            "supporting_evidence": [content[:500]],
            "confidence": 0.0,
            "affected_devices": [],
            "recommended_action": "Manual investigation required",
        }
    return {"diagnosis": diagnosis}


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("finalize", finalize_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "finalize": "finalize"})
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)
    return builder.compile()


def run_diagnosis(incident_description: str, snapshot_summary: str = "") -> dict:
    user_message = f"A network anomaly has been detected. Investigate and diagnose the root cause.\n\nINCIDENT:\n{incident_description}"
    if snapshot_summary:
        user_message += f"\n\nPRE-COLLECTED TELEMETRY:\n{snapshot_summary}"
    user_message += "\n\nUse your tools to gather evidence, then provide your diagnosis as JSON."

    final_state = build_graph().invoke(
        {
            "messages": [SystemMessage(content=_build_system_prompt()), HumanMessage(content=user_message)],
            "diagnosis": {},
        },
        config={"recursion_limit": 25},
    )
    return final_state.get("diagnosis", {})


def run_diagnosis_with_mock(incident_description: str) -> dict:
    return {
        "suspected_root_cause": "MOCK: Interface administrative shutdown on router1 eth0",
        "supporting_evidence": [
            "MOCK: Interface eth0 on router1 shows admin=down",
            "MOCK: BGP session router1->router2 is not established",
            "MOCK: Connectivity from router1 to router3 fails",
        ],
        "confidence": 0.90,
        "affected_devices": ["router1"],
        "recommended_action": "Re-enable interface eth0 on router1",
    }

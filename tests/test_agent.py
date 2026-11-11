"""
Tests for the AI diagnosis agent output validation.
Uses mock LLM responses (no API key required).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import run_diagnosis_with_mock, _validate_diagnosis


REQUIRED_KEYS = {
    "suspected_root_cause",
    "supporting_evidence",
    "confidence",
    "affected_devices",
    "recommended_action",
}


class TestDiagnosisOutputSchema:

    def test_mock_diagnosis_has_required_keys(self):
        result = run_diagnosis_with_mock("BGP session dropped")
        assert REQUIRED_KEYS.issubset(result.keys()), f"Missing keys: {REQUIRED_KEYS - result.keys()}"

    def test_confidence_is_float_between_0_and_1(self):
        result = run_diagnosis_with_mock("Interface down")
        confidence = result["confidence"]
        assert isinstance(confidence, (int, float)), "confidence must be numeric"
        assert 0.0 <= confidence <= 1.0, f"confidence out of range: {confidence}"

    def test_supporting_evidence_is_list(self):
        result = run_diagnosis_with_mock("OSPF adjacency lost")
        assert isinstance(result["supporting_evidence"], list)
        assert len(result["supporting_evidence"]) > 0

    def test_affected_devices_is_list(self):
        result = run_diagnosis_with_mock("Connectivity failure")
        assert isinstance(result["affected_devices"], list)

    def test_root_cause_is_nonempty_string(self):
        result = run_diagnosis_with_mock("Routing loop detected")
        assert isinstance(result["suspected_root_cause"], str)
        assert len(result["suspected_root_cause"]) > 0

    def test_recommended_action_is_nonempty_string(self):
        result = run_diagnosis_with_mock("BGP flap")
        assert isinstance(result["recommended_action"], str)
        assert len(result["recommended_action"]) > 0


class TestDiagnosisValidation:
    """Tests for the programmatic completeness gate in validate_node."""

    _FULL_TOOLS = {
        "tool_get_interfaces", "tool_get_bgp_summary",
        "tool_get_bgp_config", "tool_get_ospf_neighbors",
        "tool_get_routing_table", "tool_check_connectivity",
    }

    def _good_diag(self, cause: str = "Administrative interface shutdown on router1 eth0") -> dict:
        return {
            "suspected_root_cause": cause,
            "supporting_evidence": ["eth0 is admin-down on router1", "BGP session not established"],
            "confidence": 0.9,
            "affected_devices": ["router1"],
            "recommended_action": "Enable eth0 on router1",
        }

    def test_missing_interface_check_returns_error(self):
        errors = _validate_diagnosis(self._good_diag(), tools_called={"tool_get_bgp_summary"})
        assert any("interface" in e.lower() for e in errors), errors

    def test_missing_bgp_summary_returns_error(self):
        errors = _validate_diagnosis(self._good_diag(), tools_called={"tool_get_interfaces"})
        assert any("bgp" in e.lower() for e in errors), errors

    def test_complete_diagnosis_passes(self):
        errors = _validate_diagnosis(self._good_diag(), tools_called=self._FULL_TOOLS)
        assert errors == [], errors

    def test_insufficient_evidence_returns_error(self):
        diag = self._good_diag()
        diag["supporting_evidence"] = ["only one item"]
        errors = _validate_diagnosis(diag, tools_called=self._FULL_TOOLS)
        assert any("evidence" in e.lower() for e in errors), errors

    def test_bgp_root_cause_without_config_check_returns_error(self):
        diag = self._good_diag(cause="BGP session failure caused connectivity loss")
        errors = _validate_diagnosis(
            diag,
            tools_called={"tool_get_interfaces", "tool_get_bgp_summary"},
        )
        assert any("bgp_config" in e.lower() or "tool_get_bgp_config" in e for e in errors), errors

    def test_bgp_root_cause_with_interface_explained_passes(self):
        diag = self._good_diag(
            cause="BGP session failure due to interface eth0 administrative shutdown"
        )
        errors = _validate_diagnosis(
            diag,
            tools_called={"tool_get_interfaces", "tool_get_bgp_summary"},
        )
        assert not any("bgp_config" in e.lower() or "tool_get_bgp_config" in e for e in errors)

    def test_empty_tools_returns_multiple_errors(self):
        errors = _validate_diagnosis(self._good_diag(), tools_called=set())
        assert len(errors) >= 2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

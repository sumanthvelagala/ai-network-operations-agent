"""
Tests for the AI diagnosis agent output validation.
Uses mock LLM responses (no API key required).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import run_diagnosis_with_mock


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


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

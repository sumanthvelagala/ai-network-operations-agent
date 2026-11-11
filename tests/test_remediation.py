"""
Tests for the remediation controller policy enforcement.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.remediation import validate_action, ALLOWED_ACTIONS


class TestValidation:

    def test_allowed_action_valid_params(self):
        ok, reason = validate_action("enable_interface", {"router": "router1", "interface": "eth0"})
        assert ok, reason

    def test_unknown_action_rejected(self):
        ok, reason = validate_action("rm_rf", {"router": "router1"})
        assert not ok
        assert "not in the allowlist" in reason

    def test_missing_param_rejected(self):
        ok, reason = validate_action("enable_interface", {"router": "router1"})
        assert not ok
        assert "interface" in reason

    def test_unknown_router_rejected(self):
        ok, reason = validate_action("enable_interface", {"router": "prod-router-99", "interface": "eth0"})
        assert not ok
        assert "not in the lab" in reason

    def test_disable_interface_allowed(self):
        ok, reason = validate_action("disable_interface", {"router": "router2", "interface": "eth1"})
        assert ok, reason

    def test_bgp_as_mismatch_allowed(self):
        ok, reason = validate_action(
            "introduce_bgp_as_mismatch",
            {"router": "router2", "neighbor_ip": "10.0.12.2", "wrong_as": "99999"}
        )
        assert ok, reason

    def test_fix_bgp_as_allowed(self):
        ok, reason = validate_action(
            "fix_bgp_as_mismatch",
            {"router": "router2", "neighbor_ip": "10.0.12.2", "correct_as": "65001"}
        )
        assert ok, reason

    def test_all_allowlist_actions_have_description(self):
        for action, spec in ALLOWED_ACTIONS.items():
            assert "description" in spec, f"Action '{action}' missing description"
            assert "params" in spec, f"Action '{action}' missing params"
            assert "risk" in spec, f"Action '{action}' missing risk"

    def test_empty_interface_rejected(self):
        ok, reason = validate_action("enable_interface", {"router": "router1", "interface": ""})
        assert not ok


class TestExecuteWithoutApproval:
    """Test that execute_remediation respects the approval gate."""

    def test_user_rejection_stops_execution(self):
        from src.remediation import execute_remediation
        with patch("builtins.input", return_value="n"):
            result = execute_remediation(
                "enable_interface",
                {"router": "router1", "interface": "eth0"},
                require_approval=True,
            )
        assert not result["approved"]
        assert not result["executed"]

    def test_invalid_action_never_executes(self):
        from src.remediation import execute_remediation
        result = execute_remediation(
            "arbitrary_shell_command",
            {"router": "router1"},
            require_approval=False,
        )
        assert not result["executed"]
        assert not result["success"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

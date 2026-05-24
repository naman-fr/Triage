"""
Tool Call Validation Module.
Validates agent tool calls against the internal_tools.json schema.
"""

import json
from pathlib import Path
from typing import Optional

from config import INTERNAL_TOOLS_PATH


# Load tool schemas at module level
def _load_tool_schemas() -> dict:
    """Load and index tool schemas from internal_tools.json."""
    with open(INTERNAL_TOOLS_PATH, "r", encoding="utf-8") as f:
        tools = json.load(f)

    schemas = {}
    for tool in tools:
        name = tool["name"]
        schemas[name] = {
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        }
    return schemas


TOOL_SCHEMAS = _load_tool_schemas()

# Valid tool names
VALID_TOOL_NAMES = set(TOOL_SCHEMAS.keys())

# Tools that require identity verification first
DESTRUCTIVE_TOOLS = {"issue_refund", "lock_account", "modify_subscription", "reset_password"}


def validate_tool_calls(
    actions: list[dict],
    identity_verified: bool = False,
) -> tuple[list[dict], list[str]]:
    """
    Validate a list of tool calls against the internal_tools.json schema.

    Args:
        actions: List of tool call dicts [{"action": "...", "parameters": {...}}]
        identity_verified: Whether identity has been verified in conversation context

    Returns:
        (validated_actions, list_of_warnings)
    """
    validated = []
    warnings = []

    # Track if verify_identity appears in the chain
    has_verify_identity = any(
        a.get("action") == "verify_identity" for a in actions
    )

    for i, action in enumerate(actions):
        action_name = action.get("action", "")
        params = action.get("parameters", {})

        # Check tool name is valid
        if action_name not in VALID_TOOL_NAMES:
            warnings.append(f"Action {i}: Unknown tool '{action_name}' — removed")
            continue

        # Get schema
        schema = TOOL_SCHEMAS[action_name]
        param_schema = schema.get("parameters", {})
        properties = param_schema.get("properties", {})
        required = param_schema.get("required", [])

        # Validate required parameters
        missing = [r for r in required if r not in params]
        if missing:
            warnings.append(
                f"Action {i} ({action_name}): Missing required params {missing} — removed"
            )
            continue

        # Validate parameter types
        valid_params = {}
        for key, value in params.items():
            if key in properties:
                expected_type = properties[key].get("type", "string")
                if expected_type == "number" and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        warnings.append(
                            f"Action {i} ({action_name}): Param '{key}' should be number — skipped"
                        )
                        continue
                valid_params[key] = value
            # Allow extra params silently

        # Check prerequisite: destructive tools need identity verification
        if action_name in DESTRUCTIVE_TOOLS:
            if not identity_verified and not has_verify_identity:
                # Insert verify_identity before this action
                warnings.append(
                    f"Action {i} ({action_name}): Requires identity verification — "
                    f"adding verify_identity step"
                )
                validated.append({
                    "action": "verify_identity",
                    "parameters": {
                        "method": "email_otp",
                        "target": params.get("user_email", params.get("user_identifier", "user"))
                    }
                })
                has_verify_identity = True

        # Validate specific tool constraints
        if action_name == "issue_refund":
            amount = params.get("amount", 0)
            if isinstance(amount, (int, float)) and amount > 500:
                warnings.append(
                    f"Action {i} (issue_refund): Amount ${amount} exceeds $500 limit — "
                    f"requires supervisor approval, escalating instead"
                )
                # Replace with escalation
                validated.append({
                    "action": "escalate_to_human",
                    "parameters": {
                        "priority": "high",
                        "department": "billing",
                        "summary": f"Refund of ${amount} exceeds $500 limit — requires supervisor approval"
                    }
                })
                continue

            reason = params.get("reason", "")
            valid_reasons = {"duplicate", "fraud", "customer_request", "service_failure"}
            if reason and reason not in valid_reasons:
                params["reason"] = "customer_request"  # safe default

        if action_name == "escalate_to_human":
            priority = params.get("priority", "normal")
            valid_priorities = {"low", "normal", "high", "urgent"}
            if priority not in valid_priorities:
                params["priority"] = "normal"

            department = params.get("department", "general")
            valid_departments = {"billing", "technical", "security", "legal", "general"}
            if department not in valid_departments:
                params["department"] = "general"

        if action_name == "modify_subscription":
            action_val = params.get("action", "")
            valid_actions = {"upgrade", "downgrade", "cancel", "pause"}
            if action_val not in valid_actions:
                warnings.append(
                    f"Action {i} (modify_subscription): Invalid action '{action_val}' — removed"
                )
                continue

        if action_name == "lock_account":
            reason = params.get("lock_reason", "")
            valid_reasons = {"suspected_fraud", "user_requested", "compliance_violation"}
            if reason not in valid_reasons:
                params["lock_reason"] = "suspected_fraud"

        if action_name == "verify_identity":
            method = params.get("method", "")
            valid_methods = {"email_otp", "sms_otp", "security_questions"}
            if method not in valid_methods:
                params["method"] = "email_otp"

        validated.append({"action": action_name, "parameters": valid_params})

    return validated, warnings


def format_actions_json(actions: list[dict]) -> str:
    """Format validated actions as a JSON string for CSV output."""
    if not actions:
        return "[]"
    return json.dumps(actions, ensure_ascii=False)


def get_tool_descriptions() -> str:
    """Get human-readable tool descriptions for the LLM prompt."""
    lines = []
    for name, schema in TOOL_SCHEMAS.items():
        desc = schema["description"]
        params = schema.get("parameters", {}).get("properties", {})
        required = schema.get("parameters", {}).get("required", [])

        param_lines = []
        for pname, pinfo in params.items():
            req_mark = " (REQUIRED)" if pname in required else ""
            param_lines.append(
                f"      - {pname}: {pinfo.get('type', 'string')} — {pinfo.get('description', '')}{req_mark}"
            )

        lines.append(f"  - {name}: {desc}")
        if param_lines:
            lines.extend(param_lines)

    return "\n".join(lines)

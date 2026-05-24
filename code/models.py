"""
Data models for the MLE Hiring Challenge Agent.
Uses Pydantic for strict schema validation.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import json


# ─── Input Models ────────────────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    """A single message in a support ticket conversation."""
    role: str = Field(description="'user' or 'agent'")
    content: str = Field(description="Message content")


class SupportTicket(BaseModel):
    """A single support ticket from the input CSV."""
    issue: str = Field(description="Raw JSON string of conversation history")
    subject: str = Field(default="", description="Ticket subject (may be misleading)")
    company: str = Field(default="None", description="Company field (may be wrong)")

    # Parsed fields (computed from raw input)
    conversation: list[ConversationMessage] = Field(
        default_factory=list,
        description="Parsed conversation history"
    )
    raw_text: str = Field(
        default="",
        description="Concatenated text from all messages for analysis"
    )

    def parse_conversation(self) -> None:
        """Parse the JSON issue field into conversation messages."""
        try:
            if not self.issue or self.issue.strip() == "":
                self.conversation = []
                self.raw_text = ""
                return

            parsed = json.loads(self.issue)
            if isinstance(parsed, list):
                self.conversation = [
                    ConversationMessage(**msg) for msg in parsed
                    if isinstance(msg, dict) and "role" in msg and "content" in msg
                ]
            else:
                self.conversation = []

            self.raw_text = " ".join(
                msg.content for msg in self.conversation
            ).strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            self.conversation = []
            self.raw_text = self.issue if self.issue else ""


# ─── Tool Call Models ────────────────────────────────────────────────────────

class ToolCall(BaseModel):
    """A single tool/API call the agent intends to make."""
    action: str = Field(description="Tool name from internal_tools.json")
    parameters: dict = Field(
        default_factory=dict,
        description="Tool parameters matching the schema"
    )


# ─── Output Models ───────────────────────────────────────────────────────────

class AgentOutput(BaseModel):
    """The agent's complete output for a single ticket."""
    # Echoed from input
    issue: str = ""
    subject: str = ""
    company: str = ""

    # Agent-generated fields
    response: str = Field(default="", description="User-facing answer")
    product_area: str = Field(default="", description="Support category")
    status: str = Field(default="replied", description="'replied' or 'escalated'")
    request_type: str = Field(
        default="product_issue",
        description="'product_issue', 'feature_request', 'bug', or 'invalid'"
    )
    justification: str = Field(default="", description="Reasoning for decision")
    confidence_score: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Calibrated confidence (0.0 to 1.0)"
    )
    source_documents: str = Field(
        default="",
        description="Pipe-separated corpus file paths"
    )
    risk_level: str = Field(default="low", description="low/medium/high/critical")
    pii_detected: str = Field(default="false", description="'true' or 'false'")
    language: str = Field(default="en", description="ISO 639-1 language code")
    actions_taken: str = Field(
        default="[]",
        description="JSON array of tool calls"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"replied", "escalated"}:
            return "replied"
        return v

    @field_validator("request_type")
    @classmethod
    def validate_request_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"product_issue", "feature_request", "bug", "invalid"}:
            return "product_issue"
        return v

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"low", "medium", "high", "critical"}:
            return "low"
        return v

    @field_validator("pii_detected")
    @classmethod
    def validate_pii_detected(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"true", "false"}:
            return "false"
        return v

    @field_validator("actions_taken")
    @classmethod
    def validate_actions_taken(cls, v: str) -> str:
        """Ensure actions_taken is valid JSON array."""
        if not v or not v.strip():
            return "[]"
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, list):
                return "[]"
            return json.dumps(parsed)
        except (json.JSONDecodeError, TypeError):
            return "[]"

    def to_csv_dict(self) -> dict:
        """Convert to a dictionary matching the output CSV schema."""
        return {
            "issue": self.issue,
            "subject": self.subject,
            "company": self.company,
            "response": self.response,
            "product_area": self.product_area,
            "status": self.status,
            "request_type": self.request_type,
            "justification": self.justification,
            "confidence_score": str(self.confidence_score),
            "source_documents": self.source_documents,
            "risk_level": self.risk_level,
            "pii_detected": self.pii_detected,
            "language": self.language,
            "actions_taken": self.actions_taken,
        }


# ─── Retrieval Models ────────────────────────────────────────────────────────

class RetrievedDocument(BaseModel):
    """A document chunk retrieved from the corpus."""
    file_path: str = Field(description="Relative path from repo root")
    content: str = Field(description="Document text content")
    score: float = Field(default=0.0, description="Relevance score")
    source: str = Field(default="", description="'bm25' or 'faiss'")

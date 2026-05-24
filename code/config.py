"""
Configuration module for the MLE Hiring Challenge Agent.
All secrets are read from environment variables only.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

# ─── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT = _REPO_ROOT
DATA_DIR = REPO_ROOT / "data"
CORPUS_DIRS = {
    "devplatform": DATA_DIR / "devplatform",
    "claude": DATA_DIR / "claude",
    "visa": DATA_DIR / "visa",
}
API_SPECS_DIR = DATA_DIR / "api_specs"
INTERNAL_TOOLS_PATH = API_SPECS_DIR / "internal_tools.json"
SUPPORT_TICKETS_PATH = REPO_ROOT / "support_tickets" / "support_tickets.csv"
SAMPLE_TICKETS_PATH = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"
OUTPUT_PATH = REPO_ROOT / "support_tickets" / "output.csv"

# ─── LLM API Keys (read from env vars only) ─────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ─── LLM Settings (deterministic) ───────────────────────────────────────────
LLM_TEMPERATURE = 0.0
LLM_SEED = 42
LLM_MAX_TOKENS = 4096

# Auto-detect available LLM provider (priority order)
def get_llm_provider() -> str:
    """Returns the first available LLM provider based on API key availability."""
    if GOOGLE_API_KEY:
        return "google"
    if OPENAI_API_KEY:
        return "openai"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if GROQ_API_KEY:
        return "groq"
    raise ValueError(
        "No LLM API key found. Set one of: GOOGLE_API_KEY, OPENAI_API_KEY, "
        "ANTHROPIC_API_KEY, or GROQ_API_KEY in your .env file."
    )

# ─── LLM Model Names ────────────────────────────────────────────────────────
LLM_MODELS = {
    "google": "gemini-2.5-flash",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "groq": "llama-3.3-70b-versatile",
}

# ─── Retrieval Settings ─────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512  # tokens per chunk
CHUNK_OVERLAP = 100  # overlap tokens between chunks
BM25_TOP_K = 10
FAISS_TOP_K = 10
FINAL_TOP_K = 5  # after re-ranking / merging

# ─── Output Schema ──────────────────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "issue", "subject", "company", "response", "product_area",
    "status", "request_type", "justification", "confidence_score",
    "source_documents", "risk_level", "pii_detected", "language",
    "actions_taken",
]

VALID_STATUS = {"replied", "escalated"}
VALID_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}
VALID_RISK_LEVEL = {"low", "medium", "high", "critical"}
VALID_PII_DETECTED = {"true", "false"}

# ─── Random Seed (for full determinism) ──────────────────────────────────────
RANDOM_SEED = 42

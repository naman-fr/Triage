# Support Triage Agent — Setup & Run Guide

## Prerequisites

- Python 3.11+
- An API key for at least one LLM provider (Google Gemini, OpenAI, Anthropic, or Groq)

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd MLE-hiring

# 2. Install dependencies
pip install -r code/requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and add at least one API key:
#   GOOGLE_API_KEY=your_key
#   OPENAI_API_KEY=your_key
#   ANTHROPIC_API_KEY=your_key
#   GROQ_API_KEY=your_key
```

## Run

```bash
# Process all tickets and generate output.csv
python code/main.py

# Validate output format
python code/validate_output.py
```

## Output

The agent writes results to `support_tickets/output.csv` with all 14 required columns:
`issue`, `subject`, `company`, `response`, `product_area`, `status`, `request_type`,
`justification`, `confidence_score`, `source_documents`, `risk_level`, `pii_detected`,
`language`, `actions_taken`

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full agent design documentation.

## Determinism

- LLM temperature = 0.0
- Random seed = 42
- All retrieval operations are sorted deterministically
- Running twice on the same input produces identical output

## Dependencies

All dependencies are pinned in `requirements.txt`. Key libraries:
- `google-generativeai` / `openai` / `anthropic` / `groq` — LLM providers
- `sentence-transformers` + `faiss-cpu` — Semantic retrieval
- `rank-bm25` — Keyword retrieval
- `pydantic` — Data validation
- `python-dotenv` — Environment configuration

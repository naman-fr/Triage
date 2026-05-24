"""
FastAPI Server for the Triage Agent Dashboard.
Exposes results, triggers agent processing runs, and provides a sandbox.
"""

import os
import csv
import json
import subprocess
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Add parent directory of dashboard to path to import code modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_PATH, SUPPORT_TICKETS_PATH
from models import SupportTicket, AgentOutput
from agent import TriageAgent

app = FastAPI(title="Triage Agent Dashboard")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class SandboxQuery(BaseModel):
    company: str
    subject: str
    issue: str


@app.get("/", response_class=HTMLResponse)
def read_root():
    """Render main dashboard template."""
    template_path = TEMPLATES_DIR / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard template not found")
    return template_path.read_text(encoding="utf-8")


@app.get("/api/results")
def get_results():
    """Load agent outputs from the output CSV."""
    if not OUTPUT_PATH.exists():
        return []

    results = []
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Format to JSON compatible structures where needed
                results.append({
                    "company": row.get("company", "None"),
                    "subject": row.get("subject", ""),
                    "issue": row.get("issue", "[]"),
                    "status": row.get("status", "replied"),
                    "response": row.get("response", ""),
                    "product_area": row.get("product_area", "general_support"),
                    "request_type": row.get("request_type", "product_issue"),
                    "justification": row.get("justification", ""),
                    "confidence_score": float(row.get("confidence_score", "0.5")),
                    "source_documents": row.get("source_documents", ""),
                    "risk_level": row.get("risk_level", "low"),
                    "pii_detected": row.get("pii_detected", "false"),
                    "language": row.get("language", "en"),
                    "actions_taken": row.get("actions_taken", "[]")
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {e}")

    return results


@app.post("/api/run")
def run_pipeline():
    """Trigger the main pipeline execution in a background subprocess."""
    main_py_path = Path(__file__).resolve().parent.parent / "main.py"
    try:
        res = subprocess.run(
            [sys.executable, str(main_py_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        if res.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Agent run failed: {res.stderr}"
            )
        return {"status": "success", "stdout": res.stdout}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Agent run timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sandbox")
def run_sandbox(query: SandboxQuery):
    """Run the agent live on a custom query from the sandbox."""
    try:
        agent = TriageAgent()
        agent.initialize()

        # Parse company
        co = query.company
        if co.lower() == "none":
            co = "None"

        # Format conversation issue format
        # If user text is not valid JSON, treat it as a single message
        issue_text = query.issue.strip()
        if not (issue_text.startswith("[") and issue_text.endswith("]")):
            # Wrap as standard message format
            issue_text = json.dumps([{"role": "user", "content": issue_text}])

        ticket = SupportTicket(
            issue=issue_text,
            subject=query.subject,
            company=co
        )
        
        output: AgentOutput = agent.process_ticket(ticket)
        
        # Return detail format
        return {
            "company": output.company,
            "subject": output.subject,
            "issue": output.issue,
            "status": output.status,
            "response": output.response,
            "product_area": output.product_area,
            "request_type": output.request_type,
            "justification": output.justification,
            "confidence_score": output.confidence_score,
            "source_documents": output.source_documents,
            "risk_level": output.risk_level,
            "pii_detected": output.pii_detected,
            "language": output.language,
            "actions_taken": output.actions_taken
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("[Dashboard] Launching server on http://localhost:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

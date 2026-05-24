#!/usr/bin/env python3
"""
MLE Hiring Challenge — Main Entry Point

Usage:
    python code/main.py

Processes all tickets in support_tickets/support_tickets.csv and writes
results to support_tickets/output.csv.
"""

import csv
import sys
import time
import random
import numpy as np
from pathlib import Path

# Ensure reproducibility
random.seed(42)
np.random.seed(42)

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import SUPPORT_TICKETS_PATH, OUTPUT_PATH, RANDOM_SEED
from models import SupportTicket, AgentOutput
from agent import TriageAgent
from output_writer import write_outputs


def load_tickets(path: Path) -> list[SupportTicket]:
    """Load support tickets from CSV."""
    tickets = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticket = SupportTicket(
                issue=row.get("Issue", row.get("issue", "")),
                subject=row.get("Subject", row.get("subject", "")),
                company=row.get("Company", row.get("company", "None")),
            )
            tickets.append(ticket)

    print(f"[Main] Loaded {len(tickets)} tickets from {path}")
    return tickets


def main():
    """Main entry point — process all tickets and write output."""
    print("=" * 60)
    print("MLE Hiring Challenge — Support Triage Agent")
    print("=" * 60)

    start_time = time.time()

    # Load tickets
    tickets = load_tickets(SUPPORT_TICKETS_PATH)
    if not tickets:
        print("[Main] ERROR: No tickets found!")
        sys.exit(1)

    # Initialize agent (indexes corpus)
    print("\n[Main] Initializing agent...")
    agent = TriageAgent()
    agent.initialize()

    init_time = time.time() - start_time
    print(f"[Main] Initialization complete in {init_time:.1f}s")

    # Process each ticket
    print(f"\n[Main] Processing {len(tickets)} tickets...")
    outputs: list[AgentOutput] = []

    for i, ticket in enumerate(tickets, 1):
        ticket_start = time.time()

        try:
            output = agent.process_ticket(ticket)
        except Exception as e:
            print(f"[Main] ERROR on ticket {i}: {e}")
            # Safe fallback — never crash
            output = AgentOutput(
                issue=ticket.issue,
                subject=ticket.subject,
                company=ticket.company,
                response="I apologize, but I'm unable to process your request. Escalating to a human agent.",
                product_area="general_support",
                status="escalated",
                request_type="product_issue",
                justification=f"Processing error: {str(e)[:200]}",
                confidence_score=0.3,
                source_documents="",
                risk_level="medium",
                pii_detected="false",
                language="en",
                actions_taken="[]",
            )

        outputs.append(output)

        ticket_time = time.time() - ticket_start
        status_icon = "✅" if output.status == "replied" else "⬆️"
        print(
            f"  [{i}/{len(tickets)}] {status_icon} {output.status:>9} | "
            f"risk={output.risk_level:>8} | conf={output.confidence_score:.2f} | "
            f"type={output.request_type:>15} | "
            f"pii={output.pii_detected:>5} | "
            f"{ticket_time:.1f}s"
        )

    # Write output
    print(f"\n[Main] Writing output to {OUTPUT_PATH}...")
    write_outputs(outputs)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(outputs)} tickets processed in {total_time:.1f}s")
    print(f"  Replied:   {sum(1 for o in outputs if o.status == 'replied')}")
    print(f"  Escalated: {sum(1 for o in outputs if o.status == 'escalated')}")
    print(f"  PII found: {sum(1 for o in outputs if o.pii_detected == 'true')}")
    print(f"  Injections: {sum(1 for o in outputs if 'ADVERSARIAL' in o.justification)}")
    print(f"{'=' * 60}")

    # Run validation
    print("\n[Main] Running output validation...")
    try:
        from validate_output import validate
        validate()
    except Exception as e:
        print(f"[Main] Validation error: {e}")


if __name__ == "__main__":
    main()

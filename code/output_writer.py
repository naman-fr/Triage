"""
Output Writer Module — Formats and writes agent output to CSV.
"""

import csv
from pathlib import Path
from typing import Optional

from models import AgentOutput
from config import OUTPUT_PATH, OUTPUT_COLUMNS


def write_outputs(
    outputs: list[AgentOutput],
    output_path: Optional[Path] = None,
) -> None:
    """
    Write agent outputs to the output CSV file.

    Args:
        outputs: List of AgentOutput objects
        output_path: Optional custom path (defaults to support_tickets/output.csv)
    """
    path = output_path or OUTPUT_PATH

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for output in outputs:
            row = output.to_csv_dict()
            writer.writerow(row)

    print(f"[Output] Wrote {len(outputs)} rows to {path}")

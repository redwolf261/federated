"""Inspect saved experiment report artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    report_dir = workspace_root / "outputs" / "reports"

    if not report_dir.exists():
        print("No report directory found.")
        return

    report_files = sorted(report_dir.glob("*.json"))
    if not report_files:
        print("No report JSON files found.")
        return

    for report_file in report_files:
        payload = json.loads(report_file.read_text(encoding="utf-8"))
        print(f"\nReport: {report_file.name}")
        if isinstance(payload, dict) and "final_metrics" in payload:
            print(f"  final_metrics: {payload['final_metrics']}")
            print(f"  communication: {payload.get('communication', {})}")
        else:
            print(f"  keys: {list(payload.keys())[:8]}")


if __name__ == "__main__":
    main()

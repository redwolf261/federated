from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _is_finite_number(value: object) -> bool:
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def validate_run_schema(run_path: Path, expected_clients: int | None) -> list[str]:
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    status = str(payload.get("status", "")).upper()
    rounds_expected = int(payload.get("rounds_expected", -1))
    rounds_completed = int(payload.get("rounds_completed", -1))

    if rounds_completed != rounds_expected:
        errors.append(
            f"round mismatch: completed={rounds_completed} expected={rounds_expected}"
        )

    communication = payload.get("communication", {})
    total_bytes = float(communication.get("total_bytes", 0.0))
    if total_bytes <= 0.0:
        errors.append(f"communication.total_bytes must be > 0, got {total_bytes}")

    metrics = payload.get("metrics", {})
    per_round = metrics.get("per_round", [])
    final = metrics.get("final", {})

    if len(per_round) != rounds_expected:
        errors.append(
            f"metrics.per_round length must equal rounds_expected, got {len(per_round)} vs {rounds_expected}"
        )

    if not isinstance(final, dict) or not final:
        errors.append("metrics.final missing or empty")

    for round_row in per_round:
        gm = round_row.get("global_metrics", {})
        for key, value in gm.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if not _is_finite_number(sub_val):
                        errors.append(f"non-finite metric {key}.{sub_key}: {sub_val}")
            elif not _is_finite_number(value):
                errors.append(f"non-finite metric {key}: {value}")

        if expected_clients is not None:
            client_acc = gm.get("client_accuracies", {})
            if isinstance(client_acc, dict) and len(client_acc) != expected_clients:
                errors.append(
                    f"client count mismatch in round {round_row.get('round')}: "
                    f"{len(client_acc)} != expected {expected_clients}"
                )

    if status == "SUCCESS" and errors:
        errors.append("status is SUCCESS but invariants failed")

    if status == "FAIL" and rounds_completed == rounds_expected and not payload.get("error"):
        errors.append("status is FAIL but no error and full rounds completed")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate single run schema artifact")
    parser.add_argument("--run", required=True, help="Path to run schema JSON")
    parser.add_argument("--expected-clients", type=int, default=None)
    args = parser.parse_args()

    run_path = Path(args.run)
    failures = validate_run_schema(run_path=run_path, expected_clients=args.expected_clients)

    if failures:
        print("VALIDATION_ERROR")
        for row in failures:
            print(f"- {row}")
        raise SystemExit(1)

    print("VALIDATION_OK")


if __name__ == "__main__":
    main()

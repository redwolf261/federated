from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

WORKSPACE = Path(__file__).resolve().parents[1]
RUNS_DIR = WORKSPACE / "artifacts" / "runs"
PLOTS_DIR = WORKSPACE / "artifacts" / "plots"


def load_success_runs() -> list[dict]:
    runs: list[dict] = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("status")) == "SUCCESS":
            runs.append(payload)
    return runs


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_success_runs()
    if not runs:
        raise RuntimeError("No successful runs found in artifacts/runs")

    grouped: dict[str, list[dict]] = {}
    for r in runs:
        key = f"{r['method']}_a{r['alpha']}"
        grouped.setdefault(key, []).append(r)

    plt.figure(figsize=(10, 5))
    for key, rows in grouped.items():
        rows.sort(key=lambda x: x["seed"])
        rounds = rows[0]["metrics"]["per_round"]
        mean_curve = [p["global_metrics"]["mean_client_accuracy"] for p in rounds]
        plt.plot(range(1, len(mean_curve) + 1), mean_curve, label=key)
    plt.xlabel("Round")
    plt.ylabel("Mean Client Accuracy")
    plt.title("Accuracy vs Rounds")
    plt.legend(fontsize=7)
    plt.tight_layout()
    acc_path = PLOTS_DIR / "accuracy_curve.png"
    plt.savefig(acc_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    for key, rows in grouped.items():
        rows.sort(key=lambda x: x["seed"])
        rounds = rows[0]["metrics"]["per_round"]
        worst_curve = [p["global_metrics"]["worst_client_accuracy"] for p in rounds]
        plt.plot(range(1, len(worst_curve) + 1), worst_curve, label=key)
    plt.xlabel("Round")
    plt.ylabel("Worst Client Accuracy")
    plt.title("Worst-Client Accuracy vs Rounds")
    plt.legend(fontsize=7)
    plt.tight_layout()
    worst_path = PLOTS_DIR / "worst_client_curve.png"
    plt.savefig(worst_path, dpi=150)
    plt.close()

    print(f"Wrote {acc_path}")
    print(f"Wrote {worst_path}")


if __name__ == "__main__":
    main()

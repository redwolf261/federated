import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2_q1_validation import FEMNIST_NUM_CLASSES, run_scaffold


def main() -> None:
    result = run_scaffold(
        dataset_name="femnist",
        num_classes=FEMNIST_NUM_CLASSES,
        num_clients=2,
        rounds=10,
        local_epochs=10,
        seed=42,
        alpha=1.0,
        lr=0.003,
        batch_size=64,
        max_samples=20000,
        zero_control=False,
        control_strength=1.0,
        apply_control=True,
        use_control_scaling=False,
        optimizer_name="sgd",
        control_in_parameter_space=False,
        return_trace=True,
    )

    out = {
        "final_accuracy": result["mean_accuracy"],
        "rounds_1_5": [
            round(row["global_metrics"]["mean_client_accuracy"], 4)
            for row in result["per_round"][:5]
        ],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

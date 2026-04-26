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
        rounds=5,
        local_epochs=1,
        seed=42,
        alpha=1.0,
        lr=0.003,
        batch_size=64,
        max_samples=20000,
        return_trace=True,
    )

    rows = result.get("round_debug", [])

    print("mean_accuracy ||c_i|| ||c|| ||gradient||")
    for row in rows:
        print(
            f"r{row['round']}: "
            f"{row['mean_accuracy']:.6f} "
            f"{row['c_i_norm']:.6f} "
            f"{row['c_norm']:.6f} "
            f"{row['gradient_norm']:.6f}"
        )

    out_path = Path("outputs") / "scaffold_micro_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"round_debug": rows}, f, indent=2)

    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

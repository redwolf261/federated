"""Stage 0/1 FEMNIST validation runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from flex_persona.validation.phase2_reference import (
    run_centralized_reference,
    run_iid_fedavg_validation,
)


def train_centralized(seed: int) -> float:
    """Locked Stage 0 centralized reference requested in PHASE 2."""
    from flex_persona.validation.phase2_reference import train_centralized as _train_centralized

    return _train_centralized(seed=seed, workspace_root=WORKSPACE_ROOT)


def main() -> None:
    seeds = [11, 17, 23]
    centralized = run_centralized_reference(workspace_root=WORKSPACE_ROOT, seeds=seeds)
    fedavg = run_iid_fedavg_validation(workspace_root=WORKSPACE_ROOT, seed=seeds[0], num_clients=2)

    print("Stage 0: Centralized reference")
    print(json.dumps(centralized, indent=2))
    print("\nStage 1: IID FedAvg sanity")
    print(json.dumps(
        {
            "seed": fedavg.seed,
            "rounds": fedavg.rounds,
            "final_accuracy": fedavg.final_accuracy,
            "round_accuracies": fedavg.round_accuracies,
            "round_parameter_norms": fedavg.round_parameter_norms,
            "communication": fedavg.communication,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

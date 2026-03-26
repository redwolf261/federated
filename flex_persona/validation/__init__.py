"""Reusable validation entrypoints for research readiness checks."""

from .phase2_reference import (
    CentralizedReferenceResult,
    FedAvgValidationResult,
    run_centralized_reference,
    run_iid_fedavg_validation,
    train_centralized,
)

__all__ = [
    "CentralizedReferenceResult",
    "FedAvgValidationResult",
    "run_centralized_reference",
    "run_iid_fedavg_validation",
    "train_centralized",
]

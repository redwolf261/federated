"""Federated simulation loop implementing FLEX-Persona round flow."""

from __future__ import annotations

import math
import os
import platform
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


from ..config.experiment_config import ExperimentConfig
from ..data.client_data_manager import ClientDataManager
from ..evaluation.communication_tracker import CommunicationTracker
from ..evaluation.convergence_logger import ConvergenceLogger
from ..evaluation.group_metrics import GroupMetrics
from ..evaluation.metrics import Evaluator
from ..evaluation.report_builder import ReportBuilder
from ..models.model_factory import ModelFactory
from ..utils.seed import set_global_seed
from .client import Client
from .round_state import RoundState
from .server import Server
from ..prototypes.prototype_distribution import PrototypeDistribution



class FederatedSimulator:
    """Coordinates client and server interactions across rounds."""

    def __init__(self, workspace_root: str | Path, config: ExperimentConfig) -> None:
        self.workspace_root = Path(workspace_root)
        self.config = config
        self.config.validate()
        set_global_seed(int(self.config.random_seed), deterministic=True)

        self.data_manager = ClientDataManager(self.workspace_root, self.config)
        self._client_bundles = []
        self.clients = self._build_clients()
        self._global_state: dict[str, torch.Tensor] | None = None
        if self.config.training.aggregation_mode in {"fedavg", "fedprox"}:
            self._validate_fedavg_compatibility()
            self._synchronize_fedavg_initial_state()
        self.server = Server(
            num_clusters=self.config.clustering.num_clusters,
            sigma=self.config.similarity.sigma,
            random_state=self.config.clustering.random_state,
            mode=self.config.training.ablation_mode,
        )

        self.evaluator = Evaluator()
        self.group_metrics = GroupMetrics()
        self.communication_tracker = CommunicationTracker()
        self.convergence_logger = ConvergenceLogger()
        self.report_builder = ReportBuilder()
        self._last_run_summary: dict[str, object] = {
            "run_status": "SUCCESS",
            "rounds_configured": int(self.config.training.rounds),
            "rounds_executed": 0,
            "stopped_early": False,
            "termination_reason": "completed_configured_rounds",
            "error_message": None,
            "early_stopping_metric": "mean_client_accuracy",
            "best_metric": 0.0,
            "best_round": None,
            "rounds_without_improvement": 0,
        }

    @staticmethod
    def _assert_finite_metrics(metrics: Mapping[object, float], context: str) -> None:
        for key, value in metrics.items():
            if not math.isfinite(float(value)):
                raise FloatingPointError(f"Non-finite metric detected at {context}.{key}: {value}")

    def _build_clients(self) -> list[Client]:
        bundles = self.data_manager.build_client_bundles()
        self._client_bundles = bundles
        clients: list[Client] = []
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible is not None and cuda_visible.strip() in {"", "-1"}:
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        for bundle in bundles:
            model = ModelFactory.build_client_model(
                client_id=bundle.client_id,
                model_config=self.config.model,
                dataset_name=self.config.dataset_name,
            )
            clients.append(
                Client(
                    client_id=bundle.client_id,
                    model=model,
                    train_loader=bundle.train_loader,
                    eval_loader=bundle.eval_loader,
                    num_classes=self.config.model.num_classes,
                    device=device,
                )
            )

        return clients

    @staticmethod
    def _state_float32_bytes(state: dict[str, torch.Tensor]) -> int:
        total_params = 0
        for tensor in state.values():
            total_params += int(tensor.numel())
        return int(total_params * 4)

    def _validate_fedavg_compatibility(self) -> None:
        if not self.clients:
            raise ValueError("Cannot run FedAvg with zero clients")

        reference = self.clients[0].model.state_dict()
        reference_keys = tuple(reference.keys())

        for client in self.clients[1:]:
            current = client.model.state_dict()
            current_keys = tuple(current.keys())
            if current_keys != reference_keys:
                raise ValueError(
                    "FedAvg requires homogeneous model architectures with matching state_dict keys. "
                    "Set model.client_backbones to a single repeated backbone for all clients."
                )

            for key in reference_keys:
                ref_tensor = reference[key]
                cur_tensor = current[key]
                if ref_tensor.shape != cur_tensor.shape or ref_tensor.dtype != cur_tensor.dtype:
                    raise ValueError(
                        "FedAvg requires homogeneous model architectures with matching tensor shapes/dtypes. "
                        f"Mismatch found at key '{key}'."
                    )

    def _synchronize_fedavg_initial_state(self) -> None:
        if not self.clients:
            return
        global_state = self._serialized_model_state(self.clients[0].model)
        self._global_state = {k: v.clone() for k, v in global_state.items()}
        for client in self.clients[1:]:
            client.model.load_state_dict(global_state, strict=True)

    def _apply_self_only(self, messages: list) -> list:
        """Replace each client's cluster prototype with their own original prototype."""
        client_dists = self.server.get_client_distributions()
        new_messages = []
        for msg in messages:
            client_id = msg.client_id
            own_dist = client_dists[client_id]
            support_dict = {
                int(label.item()): own_dist.support_points[idx]
                for idx, label in enumerate(own_dist.support_labels)
            }
            from .messages import ServerToClientMessage
            new_msg = ServerToClientMessage(
                client_id=client_id,
                round_idx=msg.round_idx,
                cluster_id=msg.cluster_id,
                cluster_prototype_distribution=own_dist,
                cluster_prototype_dict=support_dict,
                similarity_weights=msg.similarity_weights,
                metadata=msg.metadata,
            )
            new_messages.append(new_msg)
        return new_messages

    def _apply_shuffled_prototypes(self, messages: list, round_idx: int) -> list:
        """Shuffle cluster prototype assignments across clients."""
        rng = np.random.default_rng(self.config.random_seed + round_idx)
        cluster_dists = [msg.cluster_prototype_distribution for msg in messages]
        indices = list(range(len(cluster_dists)))
        rng.shuffle(indices)
        shuffled_dists = [cluster_dists[i] for i in indices]
        new_messages = []
        for msg, dist in zip(messages, shuffled_dists):
            support_dict = {
                int(label.item()): dist.support_points[idx]
                for idx, label in enumerate(dist.support_labels)
            }
            from .messages import ServerToClientMessage
            new_msg = ServerToClientMessage(
                client_id=msg.client_id,
                round_idx=msg.round_idx,
                cluster_id=msg.cluster_id,
                cluster_prototype_distribution=dist,
                cluster_prototype_dict=support_dict,
                similarity_weights=msg.similarity_weights,
                metadata=msg.metadata,
            )
            new_messages.append(new_msg)
        return new_messages

    def _apply_noise_prototypes(self, messages: list) -> list:
        """Replace cluster prototypes with random noise of same shape and scale."""
        new_messages = []
        for msg in messages:
            orig_dist = msg.cluster_prototype_distribution
            support_points = orig_dist.support_points
            support_labels = orig_dist.support_labels
            orig_mean = float(support_points.mean())
            orig_std = float(support_points.std())
            noise = torch.randn_like(support_points) * orig_std + orig_mean
            noise_dist = PrototypeDistribution(
                support_points=noise,
                support_labels=support_labels,
                num_support=orig_dist.num_support,
            )
            support_dict = {
                int(label.item()): noise[idx]
                for idx, label in enumerate(support_labels)
            }
            from .messages import ServerToClientMessage
            new_msg = ServerToClientMessage(
                client_id=msg.client_id,
                round_idx=msg.round_idx,
                cluster_id=msg.cluster_id,
                cluster_prototype_distribution=noise_dist,
                cluster_prototype_dict=support_dict,
                similarity_weights=msg.similarity_weights,
                metadata=msg.metadata,
            )
            new_messages.append(new_msg)
        return new_messages

    @staticmethod
    def _serialized_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:

        state = model.state_dict()
        return {key: tensor.detach().cpu().clone() for key, tensor in state.items()}

    @staticmethod
    def _state_l2_norm(state: dict[str, torch.Tensor]) -> float:
        total = 0.0
        for tensor in state.values():
            if torch.is_floating_point(tensor):
                total += float(torch.sum(tensor.to(dtype=torch.float32) ** 2).item())
        return float(total ** 0.5)

    @staticmethod
    def _aggregate_fedavg_states(
        client_states: list[dict[str, torch.Tensor]],
        sample_counts: list[int],
    ) -> dict[str, torch.Tensor]:
        if not client_states:
            raise ValueError("FedAvg aggregation requires at least one client state")

        total_samples = float(sum(sample_counts))
        if total_samples <= 0:
            weights = [1.0 / len(client_states)] * len(client_states)
        else:
            weights = [count / total_samples for count in sample_counts]

        aggregated: dict[str, torch.Tensor] = {}
        keys = client_states[0].keys()
        for key in keys:
            template = client_states[0][key]

            if torch.is_floating_point(template):
                accumulator = torch.zeros_like(template, dtype=torch.float32)
                for weight, state in zip(weights, client_states):
                    accumulator += state[key].to(dtype=torch.float32) * float(weight)
                aggregated[key] = accumulator.to(dtype=template.dtype)
            else:
                aggregated[key] = template.clone()

        return aggregated

    def run_round(self, round_idx: int) -> RoundState:
        state = RoundState(round_idx=round_idx, client_ids=[c.client_id for c in self.clients])

        reference_state: dict[str, torch.Tensor] | None = None
        if self.config.training.aggregation_mode == "fedprox":
            reference_state = self._global_state or self._serialized_model_state(self.clients[0].model)

        # Local training
        for client in self.clients:
            local_metrics = client.train_local(
                local_epochs=self.config.training.local_epochs,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                fedprox_mu=float(self.config.training.fedprox_mu)
                if self.config.training.aggregation_mode == "fedprox"
                else 0.0,
                reference_state=reference_state,
            )
            self._assert_finite_metrics(local_metrics, context=f"client_{client.client_id}.local")
            state.local_metrics[client.client_id] = local_metrics

        if self.config.training.ablation_mode == "no_prototype_sharing":
            # Skip prototype exchange entirely - local training only
            c2s_round_bytes = 0
            s2c_round_bytes = 0
            for client in self.clients:
                model_state = self._serialized_model_state(client.model)
                model_bytes = self._state_float32_bytes(model_state)
                c2s_round_bytes += int(model_bytes)
                s2c_round_bytes += int(model_bytes)
            self.communication_tracker.log_round(round_idx, c2s_bytes=c2s_round_bytes, s2c_bytes=s2c_round_bytes)
            state.metadata["aggregation_mode"] = "prototype"
            state.metadata["structured_alignment"] = {
                "ablation_mode": "no_prototype_sharing",
                "note": "No prototype exchange performed",
            }
        elif self.config.training.aggregation_mode in {"fedavg", "fedprox"}:

            client_states: list[dict[str, torch.Tensor]] = []
            sample_counts: list[int] = []
            c2s_round_bytes = 0

            for client in self.clients:
                num_samples = int(len(client.train_loader.dataset))
                local_state = self._serialized_model_state(client.model)
                model_bytes = self._state_float32_bytes(local_state)
                c2s_round_bytes += model_bytes
                client_states.append(local_state)
                sample_counts.append(max(num_samples, 1))

            global_state = self._aggregate_fedavg_states(client_states=client_states, sample_counts=sample_counts)
            self._global_state = {k: v.clone() for k, v in global_state.items()}
            client_norms = {
                str(client.client_id): self._state_l2_norm(local_state)
                for client, local_state in zip(self.clients, client_states, strict=True)
            }
            global_norm = self._state_l2_norm(global_state)

            s2c_round_bytes = 0
            for client in self.clients:
                client.model.load_state_dict(global_state, strict=True)
                model_bytes = self._state_float32_bytes(global_state)
                s2c_round_bytes += model_bytes

            self.communication_tracker.client_to_server_bytes += int(c2s_round_bytes)
            self.communication_tracker.server_to_client_bytes += int(s2c_round_bytes)

            self.communication_tracker.log_round(round_idx, c2s_bytes=c2s_round_bytes, s2c_bytes=s2c_round_bytes)
            state.metadata["aggregation_mode"] = self.config.training.aggregation_mode
            state.metadata["fedavg"] = {
                "num_uploaded_models": len(client_states),
                "mean_samples_per_client": float(sum(sample_counts) / max(len(sample_counts), 1)),
                "client_parameter_norms": client_norms,
                "global_parameter_norm": float(global_norm),
            }
            if self.config.training.aggregation_mode == "fedprox":
                state.metadata["fedprox"] = {"mu": float(self.config.training.fedprox_mu)}
        else:
            # FRepAlign: Extract feature means from all clients.
            from flex_persona.training.feature_mean_utils import get_client_feature_mean

            client_feature_means: dict[int, torch.Tensor] = {}
            for client in self.clients:
                mu = get_client_feature_mean(client.model, client.train_loader, client.device)
                if mu is None:
                    raise RuntimeError(f"Client {client.client_id} has no feature mean.")
                client_feature_means[client.client_id] = mu

            # Send feature means to server and compute similarity.
            self.server.receive_client_feature_means(client_feature_means)
            state.similarity_matrix = self.server.compute_feature_mean_similarity_matrix()
            state.similarity_matrix, state.adjacency_matrix = self.server.build_similarity_and_adjacency(
                state.similarity_matrix
            )
            state.distance_matrix = 1.0 - state.similarity_matrix
            state.cluster_assignments = self.server.cluster_clients(
                state.similarity_matrix,
                round_idx=round_idx,
                num_clients=len(state.client_ids),
            )


            # Compute cluster centers from feature means and build per-client center lookup.
            cluster_member_means: dict[int, list[torch.Tensor]] = {}
            client_to_cluster: dict[int, int] = {}
            for idx, client_id in enumerate(state.client_ids):
                cluster_id = int(state.cluster_assignments[idx].item())
                client_to_cluster[client_id] = cluster_id
                cluster_member_means.setdefault(cluster_id, []).append(client_feature_means[client_id])

            cluster_centers = {
                cluster_id: torch.stack(member_means, dim=0).mean(dim=0)
                for cluster_id, member_means in cluster_member_means.items()
            }

            lambda_cluster_center = float(self.config.training.lambda_cluster_center)
            warmup_rounds = max(int(self.config.training.cluster_center_warmup_rounds), 1)
            cluster_center_warmup_scale = min(1.0, float(round_idx + 1) / float(warmup_rounds))

            upload_messages = [client.build_upload_message(round_idx) for client in self.clients]
            c2s_structured_bytes = sum(self.communication_tracker.bytes_client_to_server(msg) for msg in upload_messages)
            self.server.receive_client_messages(upload_messages)

            state.client_distributions = self.server.get_client_distributions()
            state.cluster_distributions = self.server.compute_cluster_distributions(state.cluster_assignments)

            broadcast_messages = self.server.build_broadcast_messages(
                round_idx=round_idx,
                cluster_assignments=state.cluster_assignments,
                cluster_distributions=state.cluster_distributions,
                affinity_matrix=state.similarity_matrix,
            )

            # Apply Block G prototype corruption modes
            if self.config.training.ablation_mode == "self_only":
                broadcast_messages = self._apply_self_only(broadcast_messages)
            elif self.config.training.ablation_mode == "shuffled_prototypes":
                broadcast_messages = self._apply_shuffled_prototypes(broadcast_messages, round_idx)
            elif self.config.training.ablation_mode == "noise_prototypes":
                broadcast_messages = self._apply_noise_prototypes(broadcast_messages)

            s2c_structured_bytes = sum(self.communication_tracker.bytes_server_to_client(msg) for msg in broadcast_messages)


            # Use normalized model-size accounting for all methods.
            c2s_round_bytes = 0
            s2c_round_bytes = 0
            for client in self.clients:
                model_state = self._serialized_model_state(client.model)
                model_bytes = self._state_float32_bytes(model_state)
                c2s_round_bytes += int(model_bytes)
                s2c_round_bytes += int(model_bytes)

            self.communication_tracker.log_round(round_idx, c2s_bytes=c2s_round_bytes, s2c_bytes=s2c_round_bytes)

            message_by_client = {msg.client_id: msg for msg in broadcast_messages}
            for client in self.clients:
                cluster_id = client_to_cluster[client.client_id]
                metrics = client.apply_cluster_guidance(
                    message=message_by_client[client.client_id],
                    cluster_aware_epochs=self.config.training.cluster_aware_epochs,
                    learning_rate=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                    lambda_cluster=self.config.training.lambda_cluster,
                    cluster_feature_mean=cluster_centers[cluster_id],
                    lambda_cluster_center=lambda_cluster_center,
                    cluster_center_warmup_scale=cluster_center_warmup_scale,
                    alignment_mode=self.config.training.alignment_mode,
                )

                state.local_metrics[client.client_id].update(metrics)

            state.metadata["aggregation_mode"] = "prototype"
            state.metadata["structured_alignment"] = {
                "lambda_cluster_center": lambda_cluster_center,
                "cluster_center_warmup_scale": cluster_center_warmup_scale,
                "cluster_center_warmup_rounds": warmup_rounds,
                "num_clusters_active": int(len(cluster_centers)),
                "structured_message_bytes": {
                    "client_to_server": int(c2s_structured_bytes),
                    "server_to_client": int(s2c_structured_bytes),
                },
            }

        client_accuracy = {client.client_id: client.evaluate_accuracy() for client in self.clients}
        self._assert_finite_metrics(client_accuracy, context=f"round_{round_idx}.client_accuracy")
        mean_accuracy = self.evaluator.mean_client_accuracy(client_accuracy)
        worst_client_accuracy = self.evaluator.worst_client_accuracy(client_accuracy)
        p10_client_accuracy = self.evaluator.p10_client_accuracy(client_accuracy)
        bottom3_client_accuracy = self.evaluator.bottom_k_client_accuracy(client_accuracy, k=3)

        cluster_to_acc_values: dict[str, list[float]] = {}
        if state.cluster_assignments is not None:
            for idx, client_id in enumerate(state.client_ids):
                cluster_id = int(state.cluster_assignments[idx].item())
                key = f"cluster_{cluster_id}"
                cluster_to_acc_values.setdefault(key, []).append(client_accuracy[client_id])
        else:
            cluster_to_acc_values["cluster_all"] = [client_accuracy[cid] for cid in state.client_ids]

        group_accuracy = {
            group_id: float(sum(values) / len(values))
            for group_id, values in cluster_to_acc_values.items()
        }
        worst_group_accuracy = self.group_metrics.worst_group_accuracy(group_accuracy)

        state.metadata["evaluation"] = {
            "mean_client_accuracy": mean_accuracy,
            "worst_client_accuracy": worst_client_accuracy,
            "p10_client_accuracy": p10_client_accuracy,
            "bottom3_client_accuracy": bottom3_client_accuracy,
            "worst_group_accuracy": worst_group_accuracy,
            "client_accuracies": {str(k): float(v) for k, v in client_accuracy.items()},
        }
        state.metadata["communication"] = {
            "round_client_to_server_bytes": int(c2s_round_bytes),
            "round_server_to_client_bytes": int(s2c_round_bytes),
            "round_total_bytes": int(c2s_round_bytes + s2c_round_bytes),
        }

        self.convergence_logger.log("mean_client_accuracy", mean_accuracy)
        self.convergence_logger.log("worst_client_accuracy", worst_client_accuracy)
        self.convergence_logger.log("p10_client_accuracy", p10_client_accuracy)
        self.convergence_logger.log("bottom3_client_accuracy", bottom3_client_accuracy)
        self.convergence_logger.log("worst_group_accuracy", worst_group_accuracy)

        return state

    def run_experiment(self) -> list[RoundState]:
        history: list[RoundState] = []

        early_stop_enabled = bool(self.config.training.early_stopping_enabled)
        patience = int(self.config.training.early_stopping_patience)
        min_delta = float(self.config.training.early_stopping_min_delta)

        best_metric = float("-inf")
        best_round: int | None = None
        rounds_without_improvement = 0
        stopped_early = False

        unlimited_rounds = self.config.training.rounds == -1
        configured_rounds = int(self.config.training.rounds)
        max_unlimited_rounds = int(self.config.training.max_unlimited_rounds)
        termination_reason = "completed_configured_rounds"
        error_message: str | None = None

        round_idx = 1
        while True:
            try:
                state = self.run_round(round_idx)
            except Exception as exc:
                stopped_early = True
                termination_reason = "exception"
                error_message = f"{type(exc).__name__}: {exc}"
                break
            history.append(state)
            evaluation = state.metadata.get("evaluation", {})
            communication = state.metadata.get("communication", {})
            mean_acc = float(evaluation.get("mean_client_accuracy", 0.0))
            worst_acc = float(evaluation.get("worst_client_accuracy", 0.0))
            round_total_bytes = int(communication.get("round_total_bytes", 0))
            if unlimited_rounds:
                progress_suffix = "?"
            else:
                progress_suffix = str(configured_rounds)
            print(
                f"[ROUND] exp={self.config.experiment_name} round={round_idx}/{progress_suffix} "
                f"mean={mean_acc:.4f} worst={worst_acc:.4f} bytes={round_total_bytes}",
                flush=True,
            )

            if early_stop_enabled:
                current_metric = float(evaluation.get("mean_client_accuracy", 0.0))

                if current_metric > (best_metric + min_delta):
                    best_metric = current_metric
                    best_round = round_idx
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if rounds_without_improvement >= patience:
                    stopped_early = True
                    termination_reason = "early_stopping"
                    break

            if unlimited_rounds:
                if len(history) >= max_unlimited_rounds:
                    termination_reason = "max_unlimited_rounds_reached"
                    break
                round_idx += 1
                continue

            if round_idx >= configured_rounds:
                termination_reason = "completed_configured_rounds"
                break

            round_idx += 1

        if best_metric == float("-inf"):
            best_metric = 0.0

        run_status = "SUCCESS"
        if termination_reason == "exception":
            run_status = "FAIL"
        elif not unlimited_rounds and len(history) < configured_rounds:
            run_status = "FAIL"

        self._last_run_summary = {
            "run_status": run_status,
            "rounds_configured": int(self.config.training.rounds),
            "rounds_executed": len(history),
            "stopped_early": bool(stopped_early),
            "termination_reason": termination_reason,
            "error_message": error_message,
            "early_stopping_metric": "mean_client_accuracy",
            "early_stopping_enabled": early_stop_enabled,
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
            "max_unlimited_rounds": max_unlimited_rounds,
            "best_metric": float(best_metric),
            "best_round": best_round,
            "rounds_without_improvement": int(rounds_without_improvement),
        }

        return history

    def build_report(self, history: list[RoundState]) -> dict[str, object]:
        if history:
            final_eval = history[-1].metadata.get("evaluation", {})
        else:
            final_eval = {}
        scalar_final_eval = {
            str(k): float(v)
            for k, v in final_eval.items()
            if isinstance(v, (int, float))
        }
        report = self.report_builder.build(
            final_round_metrics=scalar_final_eval,
            communication_summary=self.communication_tracker.summarize(),
            convergence_traces=self.convergence_logger.as_dict(),
        )
        report["run_summary"] = self._last_run_summary
        return report

    def build_run_schema(self, history: list[RoundState]) -> dict[str, object]:
        rounds: list[dict[str, object]] = []
        for state in history:
            evaluation = state.metadata.get("evaluation", {})
            communication = state.metadata.get("communication", {})
            rounds.append(
                {
                    "round": int(state.round_idx),
                    "client_metrics": {
                        str(client_id): {k: float(v) for k, v in metrics.items()}
                        for client_id, metrics in state.local_metrics.items()
                    },
                    "global_metrics": {k: v for k, v in evaluation.items()},
                    "communication": {k: int(v) for k, v in communication.items()},
                }
            )

        partition_manifest = {
            str(bundle.client_id): {
                "num_samples": int(bundle.num_samples),
                "class_distribution": {str(k): int(v) for k, v in bundle.class_histogram.items()},
                "source_indices": [int(i) for i in bundle.source_indices],
            }
            for bundle in sorted(self._client_bundles, key=lambda b: b.client_id)
        }

        return {
            "config": {
                "experiment_name": self.config.experiment_name,
                "dataset_name": self.config.dataset_name,
                "num_clients": int(self.config.num_clients),
                "partition_mode": self.config.partition_mode,
                "dirichlet_alpha": float(self.config.dirichlet_alpha),
                "aggregation_mode": self.config.training.aggregation_mode,
                "rounds": int(self.config.training.rounds),
                "local_epochs": int(self.config.training.local_epochs),
                "cluster_aware_epochs": int(self.config.training.cluster_aware_epochs),
                "learning_rate": float(self.config.training.learning_rate),
                "batch_size": int(self.config.training.batch_size),
                "lambda_cluster": float(self.config.training.lambda_cluster),
                "lambda_cluster_center": float(self.config.training.lambda_cluster_center),
            },
            "seed": int(self.config.random_seed),
            "environment": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": int(torch.backends.cudnn.version() or 0),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            },
            "partition_fingerprint": self.data_manager.partition_fingerprint(self._client_bundles),
            "partition_manifest": partition_manifest,
            "rounds": rounds,
            "final_summary": self.build_report(history),
        }

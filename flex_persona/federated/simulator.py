"""Federated simulation loop implementing FLEX-Persona round flow."""

from __future__ import annotations

from pathlib import Path

from ..config.experiment_config import ExperimentConfig
from ..data.client_data_manager import ClientDataManager
from ..evaluation.communication_tracker import CommunicationTracker
from ..evaluation.convergence_logger import ConvergenceLogger
from ..evaluation.group_metrics import GroupMetrics
from ..evaluation.metrics import Evaluator
from ..evaluation.report_builder import ReportBuilder
from ..models.model_factory import ModelFactory
from .client import Client
from .round_state import RoundState
from .server import Server


class FederatedSimulator:
    """Coordinates client and server interactions across rounds."""

    def __init__(self, workspace_root: str | Path, config: ExperimentConfig) -> None:
        self.workspace_root = Path(workspace_root)
        self.config = config
        self.config.validate()

        self.data_manager = ClientDataManager(self.workspace_root, self.config)
        self.clients = self._build_clients()
        self.server = Server(
            num_clusters=self.config.clustering.num_clusters,
            sigma=self.config.similarity.sigma,
            random_state=self.config.clustering.random_state,
        )
        self.evaluator = Evaluator()
        self.group_metrics = GroupMetrics()
        self.communication_tracker = CommunicationTracker()
        self.convergence_logger = ConvergenceLogger()
        self.report_builder = ReportBuilder()

    def _build_clients(self) -> list[Client]:
        bundles = self.data_manager.build_client_bundles()
        clients: list[Client] = []

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
                    device="cpu",
                )
            )

        return clients

    def run_round(self, round_idx: int) -> RoundState:
        state = RoundState(round_idx=round_idx, client_ids=[c.client_id for c in self.clients])

        for client in self.clients:
            local_metrics = client.train_local(
                local_epochs=self.config.training.local_epochs,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
            state.local_metrics[client.client_id] = local_metrics

        upload_messages = [client.build_upload_message(round_idx) for client in self.clients]
        c2s_round_bytes = sum(self.communication_tracker.bytes_client_to_server(msg) for msg in upload_messages)
        self.server.receive_client_messages(upload_messages)

        state.client_distributions = self.server.get_client_distributions()
        state.distance_matrix = self.server.compute_wasserstein_matrix()
        state.similarity_matrix, state.adjacency_matrix = self.server.build_similarity_and_adjacency(
            state.distance_matrix
        )
        state.cluster_assignments = self.server.cluster_clients(state.similarity_matrix)
        state.cluster_distributions = self.server.compute_cluster_distributions(state.cluster_assignments)

        broadcast_messages = self.server.build_broadcast_messages(
            round_idx=round_idx,
            cluster_assignments=state.cluster_assignments,
            cluster_distributions=state.cluster_distributions,
            affinity_matrix=state.similarity_matrix,
        )
        s2c_round_bytes = sum(self.communication_tracker.bytes_server_to_client(msg) for msg in broadcast_messages)
        self.communication_tracker.log_round(round_idx, c2s_round_bytes=c2s_round_bytes, s2c_bytes=s2c_round_bytes)

        message_by_client = {msg.client_id: msg for msg in broadcast_messages}

        for client in self.clients:
            metrics = client.apply_cluster_guidance(
                message=message_by_client[client.client_id],
                cluster_aware_epochs=self.config.training.cluster_aware_epochs,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                lambda_cluster=self.config.training.lambda_cluster,
            )
            state.local_metrics[client.client_id].update(metrics)

        client_accuracy = {client.client_id: client.evaluate_accuracy() for client in self.clients}
        mean_accuracy = self.evaluator.mean_client_accuracy(client_accuracy)
        worst_client_accuracy = self.evaluator.worst_client_accuracy(client_accuracy)

        cluster_to_acc_values: dict[str, list[float]] = {}
        for idx, client_id in enumerate(state.client_ids):
            cluster_id = int(state.cluster_assignments[idx].item()) if state.cluster_assignments is not None else -1
            key = f"cluster_{cluster_id}"
            cluster_to_acc_values.setdefault(key, []).append(client_accuracy[client_id])
        group_accuracy = {
            group_id: float(sum(values) / len(values))
            for group_id, values in cluster_to_acc_values.items()
        }
        worst_group_accuracy = self.group_metrics.worst_group_accuracy(group_accuracy)

        state.metadata["evaluation"] = {
            "mean_client_accuracy": mean_accuracy,
            "worst_client_accuracy": worst_client_accuracy,
            "worst_group_accuracy": worst_group_accuracy,
        }
        state.metadata["communication"] = {
            "round_client_to_server_bytes": int(c2s_round_bytes),
            "round_server_to_client_bytes": int(s2c_round_bytes),
            "round_total_bytes": int(c2s_round_bytes + s2c_round_bytes),
        }

        self.convergence_logger.log("mean_client_accuracy", mean_accuracy)
        self.convergence_logger.log("worst_client_accuracy", worst_client_accuracy)
        self.convergence_logger.log("worst_group_accuracy", worst_group_accuracy)

        return state

    def run_experiment(self) -> list[RoundState]:
        history: list[RoundState] = []
        for round_idx in range(1, self.config.training.rounds + 1):
            history.append(self.run_round(round_idx))
        return history

    def build_report(self, history: list[RoundState]) -> dict[str, object]:
        if history:
            final_eval = history[-1].metadata.get("evaluation", {})
        else:
            final_eval = {}
        return self.report_builder.build(
            final_round_metrics={str(k): float(v) for k, v in final_eval.items()},
            communication_summary=self.communication_tracker.summarize(),
            convergence_traces=self.convergence_logger.as_dict(),
        )

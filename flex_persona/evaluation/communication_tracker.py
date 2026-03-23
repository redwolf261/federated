"""Communication overhead tracking for client-server messages."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

from ..federated.messages import ClientToServerMessage, ServerToClientMessage


@dataclass
class CommunicationTracker:
    """Tracks transferred bytes for each communication direction."""

    client_to_server_bytes: int = 0
    server_to_client_bytes: int = 0
    per_round: dict[int, dict[str, int]] = field(default_factory=dict)

    @staticmethod
    def _estimate_bytes(payload: object) -> int:
        return len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    def bytes_client_to_server_payload(self, payload: object) -> int:
        size = self._estimate_bytes(payload)
        self.client_to_server_bytes += size
        return size

    def bytes_server_to_client_payload(self, payload: object) -> int:
        size = self._estimate_bytes(payload)
        self.server_to_client_bytes += size
        return size

    def bytes_client_to_server(self, message: ClientToServerMessage) -> int:
        return self.bytes_client_to_server_payload(message)

    def bytes_server_to_client(self, message: ServerToClientMessage) -> int:
        return self.bytes_server_to_client_payload(message)

    def log_round(
        self,
        round_idx: int,
        c2s_bytes: int,
        s2c_bytes: int,
    ) -> None:
        self.per_round[round_idx] = {
            "client_to_server": int(c2s_bytes),
            "server_to_client": int(s2c_bytes),
            "total": int(c2s_bytes + s2c_bytes),
        }

    def summarize(self) -> dict[str, int]:
        total = self.client_to_server_bytes + self.server_to_client_bytes
        return {
            "client_to_server_bytes": int(self.client_to_server_bytes),
            "server_to_client_bytes": int(self.server_to_client_bytes),
            "total_bytes": int(total),
        }

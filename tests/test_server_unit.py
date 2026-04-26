"""Unit tests for Server clustering and similarity computation."""

import torch

from flex_persona.federated.server import Server


def test_server_initialization() -> None:
    server = Server(num_clusters=3, sigma=1.5, random_state=42)
    assert server.num_clusters == 3
    assert server.sigma == 1.5


def test_feature_mean_similarity() -> None:
    server = Server(num_clusters=2, sigma=1.0, random_state=42)
    feature_means = {
        0: torch.tensor([1.0, 0.0, 0.0]),
        1: torch.tensor([0.0, 1.0, 0.0]),
        2: torch.tensor([0.99, 0.01, 0.0]),  # Very similar to client 0
    }
    server.receive_client_feature_means(feature_means)
    similarity = server.compute_feature_mean_similarity_matrix()
    
    assert similarity.shape == (3, 3)
    assert torch.allclose(similarity.diag(), torch.ones(3), atol=1e-6)
    # Clients 0 and 2 should be more similar than 0 and 1
    assert similarity[0, 2] > similarity[0, 1]


def test_cluster_clients_groups_similar_clients() -> None:
    server = Server(num_clusters=2, sigma=1.0, random_state=42)
    # Create an affinity matrix where clients 0,2 are similar and 1,3 are similar
    affinity = torch.tensor([
        [1.0, 0.1, 0.9, 0.1],
        [0.1, 1.0, 0.1, 0.9],
        [0.9, 0.1, 1.0, 0.1],
        [0.1, 0.9, 0.1, 1.0],
    ])
    assignments = server.cluster_clients(affinity)
    assert len(assignments) == 4
    assert assignments[0] == assignments[2]
    assert assignments[1] == assignments[3]


def test_build_similarity_and_adjacency() -> None:
    server = Server(num_clusters=2, sigma=1.0, random_state=42)
    similarity = torch.tensor([
        [1.0, 0.5, 0.2, 0.3, 0.1],
        [0.5, 1.0, 0.6, 0.4, 0.2],
        [0.2, 0.6, 1.0, 0.5, 0.3],
        [0.3, 0.4, 0.5, 1.0, 0.7],
        [0.1, 0.2, 0.3, 0.7, 1.0],
    ])
    sim, adj = server.build_similarity_and_adjacency(similarity)
    assert sim.shape == (5, 5)
    assert adj.shape == (5, 5)
    assert adj.dtype == torch.bool
    # Each row should have at least k entries (k=min(num_clusters+1, N)=3)
    for i in range(5):
        assert adj[i].sum() >= 3

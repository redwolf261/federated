import torch

from flex_persona.clustering.spectral_clusterer import SpectralClusterer
from flex_persona.prototypes.prototype_distribution import PrototypeDistribution
from flex_persona.similarity.wasserstein_distance import WassersteinDistanceCalculator


def _dist(client_id: int, offset: float) -> PrototypeDistribution:
    support_points = torch.tensor([[0.0 + offset, 0.0], [1.0 + offset, 1.0]], dtype=torch.float32)
    support_labels = torch.tensor([0, 1], dtype=torch.long)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
    return PrototypeDistribution(client_id, support_points, support_labels, weights, num_classes=2).normalized()


def test_wasserstein_and_spectral_smoke() -> None:
    dcalc = WassersteinDistanceCalculator(prefer_pot=False)
    dists = {0: _dist(0, 0.0), 1: _dist(1, 0.1), 2: _dist(2, 3.0)}
    matrix = dcalc.pairwise_wasserstein_matrix(dists)

    affinity = torch.exp(-(matrix / 1.0))
    affinity.fill_diagonal_(1.0)

    labels = SpectralClusterer(num_clusters=2, random_state=42).fit_predict(affinity)
    assert labels.shape[0] == 3

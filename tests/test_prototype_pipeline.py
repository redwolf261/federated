import torch

from flex_persona.prototypes.distribution_builder import PrototypeDistributionBuilder
from flex_persona.prototypes.prototype_extractor import PrototypeExtractor


def test_prototype_extraction_and_distribution_smoke() -> None:
    features = torch.randn(20, 8)
    labels = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2, 1, 0, 2, 1])

    prototype_dict, class_counts = PrototypeExtractor.compute_class_prototypes(
        shared_features=features,
        labels=labels,
        num_classes=3,
    )
    dist = PrototypeDistributionBuilder.build_distribution(
        client_id=0,
        prototype_dict=prototype_dict,
        class_counts=class_counts,
        num_classes=3,
    )

    assert dist.num_support > 0
    assert abs(float(dist.weights.sum().item()) - 1.0) < 1e-6

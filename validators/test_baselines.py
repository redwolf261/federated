from __future__ import annotations

import torch


def moon_contrastive_loss(z_local: torch.Tensor, z_global: torch.Tensor, z_prev: torch.Tensor, temperature: float) -> torch.Tensor:
    cos = torch.nn.CosineSimilarity(dim=1)
    pos = cos(z_local, z_global).unsqueeze(1)
    neg = cos(z_local, z_prev).unsqueeze(1)
    logits = torch.cat([pos, neg], dim=1) / temperature
    labels = torch.zeros(z_local.size(0), dtype=torch.long, device=z_local.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def scaffold_correction(grad: torch.Tensor, c_global: torch.Tensor, c_local: torch.Tensor) -> torch.Tensor:
    return grad - c_local + c_global


def test_moon_loss_prefers_global_similarity() -> None:
    batch = 8
    dim = 16
    z_local = torch.randn(batch, dim)
    z_global = z_local + 0.01 * torch.randn(batch, dim)
    z_prev = -z_local + 0.2 * torch.randn(batch, dim)

    good = moon_contrastive_loss(z_local, z_global, z_prev, temperature=0.5)
    bad = moon_contrastive_loss(z_local, z_prev, z_global, temperature=0.5)

    assert float(good.item()) < float(bad.item())


def test_scaffold_correction_matches_formula() -> None:
    grad = torch.tensor([1.0, -2.0, 0.5])
    c_global = torch.tensor([0.1, 0.1, 0.1])
    c_local = torch.tensor([0.2, -0.3, 0.0])

    corrected = scaffold_correction(grad, c_global, c_local)
    expected = torch.tensor([0.9, -1.6, 0.6])

    assert torch.allclose(corrected, expected)

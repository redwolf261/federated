#!/usr/bin/env python3
"""Direct comparison of backbone implementations to find the specific issue."""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.models.backbones import SmallCNNBackbone

def compare_backbone_implementations():
    """Compare FLEX SmallCNNBackbone vs working simple CNN backbone."""
    print("="*70)
    print("BACKBONE IMPLEMENTATION COMPARISON")
    print("="*70)

    # Create test input
    test_input = torch.randn(4, 1, 28, 28)  # Small batch of FEMNIST images
    print(f"Test input shape: {test_input.shape}")
    print(f"Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
    print()

    # 1. Working simple CNN backbone (from SimpleWorkingCNN)
    class SimpleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 16, 5)  # 28->24
            self.pool = nn.MaxPool2d(2, 2)   # 24->12
            self.flatten = nn.Flatten()
            self.output_dim = 16 * 12 * 12  # 2304

        def forward(self, x):
            x = self.conv(x)            # (1, 28, 28) -> (16, 24, 24)
            x = torch.relu(x)
            x = self.pool(x)            # (16, 24, 24) -> (16, 12, 12)
            x = self.flatten(x)         # -> (2304,)
            return x

    # 2. FLEX SmallCNNBackbone
    flex_backbone = SmallCNNBackbone(in_channels=1)
    simple_backbone = SimpleBackbone()

    print("ARCHITECTURE COMPARISON:")
    print("-" * 30)
    print("Simple Backbone:")
    print("  Conv2d(1, 16, 5) -> ReLU -> MaxPool2d(2) -> Flatten")
    print(f"  Output dim: {simple_backbone.output_dim}")
    print()

    print("FLEX SmallCNNBackbone:")
    for name, module in flex_backbone.named_modules():
        if name:  # Skip the root module
            print(f"  {name}: {module}")
    print(f"  Output dim: {flex_backbone.output_dim}")
    print()

    # Test forward passes
    models = {
        'simple': simple_backbone,
        'flex': flex_backbone
    }

    for name, model in models.items():
        print(f"{name.upper()} FORWARD PASS:")
        print("-" * 30)

        with torch.no_grad():
            output = model(test_input)
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Output std: {output.std():.3f}")
            print(f"Output mean: {output.mean():.3f}")

            # Check for issues
            zero_outputs = (output == 0).sum().item()
            total_outputs = output.numel()
            zero_pct = 100 * zero_outputs / total_outputs

            print(f"Zero outputs: {zero_pct:.1f}% ({zero_outputs}/{total_outputs})")

            if torch.isnan(output).any():
                print("WARNING: NaN values found!")
            if torch.isinf(output).any():
                print("WARNING: Infinite values found!")
            if output.std() < 0.01:
                print("WARNING: Very low variance - may not be learning features")
            if zero_pct > 50:
                print("WARNING: >50% zero outputs - possible dead neurons")

        print()

    # Compare parameter counts and initialization
    print("PARAMETER COMPARISON:")
    print("-" * 30)

    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name.upper()}: {total_params:,} parameters")

        # Check parameter statistics
        all_weights = torch.cat([p.flatten() for p in model.parameters()])
        print(f"  Weight range: [{all_weights.min():.4f}, {all_weights.max():.4f}]")
        print(f"  Weight std: {all_weights.std():.4f}")
        print(f"  Weight mean: {all_weights.mean():.4f}")

        zero_weights = (all_weights.abs() < 1e-6).sum().item()
        zero_weight_pct = 100 * zero_weights / len(all_weights)
        print(f"  Zero weights: {zero_weight_pct:.1f}%")
        print()

    # Test gradient flow
    print("GRADIENT FLOW TEST:")
    print("-" * 30)

    criterion = nn.MSELoss()
    target = torch.randn(4, simple_backbone.output_dim)  # Random target for gradient test

    for name, model in models.items():
        model.zero_grad()

        if name == 'flex':
            # Adapt target to FLEX output size
            flex_target = torch.randn(4, flex_backbone.output_dim)
            output = model(test_input)
            loss = criterion(output, flex_target)
        else:
            output = model(test_input)
            loss = criterion(output, target)

        loss.backward()

        # Check gradients
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        if grad_norms:
            total_grad_norm = sum(grad_norms)
            avg_grad_norm = total_grad_norm / len(grad_norms)
            max_grad_norm = max(grad_norms)

            print(f"{name.upper()}:")
            print(f"  Total gradient norm: {total_grad_norm:.4f}")
            print(f"  Average gradient norm: {avg_grad_norm:.4f}")
            print(f"  Max gradient norm: {max_grad_norm:.4f}")

            if total_grad_norm < 1e-4:
                print("  WARNING: Very small gradients - vanishing gradient problem")
            elif total_grad_norm > 100:
                print("  WARNING: Very large gradients - exploding gradient problem")
        else:
            print(f"{name.upper()}: No gradients computed!")
        print()

if __name__ == "__main__":
    compare_backbone_implementations()
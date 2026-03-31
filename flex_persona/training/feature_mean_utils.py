import torch
from torch.utils.data import DataLoader

def get_client_feature_mean(model, loader, device):
    """Extract mean feature embedding from penultimate layer for a client."""
    model.eval()
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            # Use forward_shared to get shared latent features
            f = model.forward_shared(x)
            feats.append(f.detach().cpu())
    if feats:
        return torch.cat(feats, dim=0).mean(dim=0)
    else:
        return None

# Example usage (to be integrated into client training loop):
# model: ClientModel
# loader: DataLoader
# device: 'cuda' or 'cpu'
# mu_i = get_client_feature_mean(model, loader, device)

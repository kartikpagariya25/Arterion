import torch
import torch.nn as nn
import torchvision


class DominanceNet(nn.Module):
    """Shared ResNet-18 frame encoder -> per-video feature (+ view embedding) -> study logits.

    frames: (B, V, F, 3, H, W); views: (B, V) with 0 = RCA, 1 = LCA; mask: (B, V) bool.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        enc = torchvision.models.resnet18(weights=weights)
        enc.fc = nn.Identity()
        self.encoder = enc
        self.view_emb = nn.Embedding(2, 32)
        self.head = nn.Sequential(nn.Linear(512 + 32, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, frames, views, mask=None):
        b, v, f, c, h, w = frames.shape
        feats = self.encoder(frames.reshape(b * v * f, c, h, w)).reshape(b, v, f, -1).mean(2)
        feats = torch.cat([feats, self.view_emb(views)], dim=-1)
        logits = self.head(feats)  # (B, V, 2) per-video logits
        if mask is None:
            return logits.mean(1)
        m = mask.unsqueeze(-1).float()
        return (logits * m).sum(1) / m.sum(1).clamp(min=1)

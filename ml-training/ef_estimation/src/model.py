import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class EFRegressor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

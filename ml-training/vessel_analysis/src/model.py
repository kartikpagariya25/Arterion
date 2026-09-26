import torch.nn as nn
import segmentation_models_pytorch as smp


class VesselUNet(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, x):
        return self.net(x)

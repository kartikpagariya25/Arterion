import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CardioNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResBlock1D(32, 32, stride=1)
        self.block2 = ResBlock1D(32, 64, stride=2)
        self.block3 = ResBlock1D(64, 128, stride=2)
        self.block4 = ResBlock1D(128, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.rr_branch = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        self.gradcam_activation = None

    def forward(self, beat: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        x = self.stem(beat)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        self.gradcam_activation = x

        x = self.pool(x).flatten(1)
        r = self.rr_branch(rr)

        combined = torch.cat([x, r], dim=1)
        return self.head(combined)

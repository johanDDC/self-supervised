import torch
import torch.nn as nn


class Encoder1D(nn.Module):
    """
        Inspired by PANNs: Large-Scale Pretrained Audio Neural Networks for
                        Audio Pattern Recognition
    """
    def __init__(self):
        super().__init__()
        self.in_conv = nn.Conv1d(1, 512, kernel_size=11, stride=5, bias=False)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(32, 512),
            nn.GELU(),
        ) for _ in range(3)])
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        x = self.in_conv(x)
        for i in range(3):
            x = self.blocks[i](x) + x
            x = self.pooling(x)
        return self.avg_pool(x).squeeze()


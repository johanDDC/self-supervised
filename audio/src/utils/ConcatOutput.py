import torch
import torch.nn as nn


class ConcatOutput(nn.Module):
    def forward(self, x):
        a, b = x
        return torch.hstack([a, b])

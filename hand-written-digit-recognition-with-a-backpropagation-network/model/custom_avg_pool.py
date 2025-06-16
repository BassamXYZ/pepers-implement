from torch import nn
import torch


class CustomAvgPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(2, stride=2)
        # Learnable scalar per channel
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        # Learnable bias per channel
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        x = self.pool(x)
        return x * self.scale + self.bias

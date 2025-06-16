from torch import nn
import torch


class CustomConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 5, bias=False)
        self.conv2 = nn.Conv2d(1, channels, 5, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        x1 = x[:, :1, :, :]
        x2 = x[:, 1:, :, :]
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        return x1 + x2 + self.bias

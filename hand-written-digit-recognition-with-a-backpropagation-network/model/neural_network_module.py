import torch
from torch import nn
from .custom_conv import CustomConv
from .custom_avg_pool import CustomAvgPool


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Conv2d(1, 4, 5)
        self.h2 = CustomAvgPool(4)
        self.h3_group1 = nn.Conv2d(4, 4, 5)
        self.h3_group2 = CustomConv(4)
        self.h3_group3 = CustomConv(4)
        self.h4 = CustomAvgPool(12)
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12*4*4, 10),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.tanh(self.h1(x))
        x = torch.tanh(self.h2(x))
        h2_group1 = x[:, :2, :, :]
        h2_group2 = x[:, 2:, :, :]
        h3_1 = torch.tanh(self.h3_group1(x))
        h3_2 = torch.tanh(self.h3_group2(h2_group1))
        h3_3 = torch.tanh(self.h3_group2(h2_group2))
        x = torch.cat([h3_1, h3_2, h3_3], dim=1)
        x = torch.tanh(self.h4(x))
        return self.out(x)

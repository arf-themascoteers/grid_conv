import torch.nn as nn
import torch


class PearsonCorrelation(nn.Module):
    def __init__(self):
        super(PearsonCorrelation, self).__init__()

    def forward(self, x, y):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        cov_xy = torch.sum((x - mean_x) * (y - mean_y))

        std_x = torch.sqrt(torch.sum((x - mean_x)**2))
        std_y = torch.sqrt(torch.sum((y - mean_y)**2))

        correlation_coefficient = cov_xy / (std_x * std_y)

        return correlation_coefficient

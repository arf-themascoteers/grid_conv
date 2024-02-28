import torch
import torch.nn as nn
from ann_savi_base import ANNSAVIBase


class ANNSAVILearnable(ANNSAVIBase):
    def __init__(self, device, train_x, train_y, test_x, test_y, validation_x, validation_y):
        super().__init__(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
        self.L = nn.Parameter(torch.tensor(0.5))
        self.linear = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

    def get_L(self, x):
        return self.L

    def forward(self, x):
        savi = self.si(x)
        savi = savi.reshape(savi.shape[0], 1)
        x = self.linear(savi)
        return x


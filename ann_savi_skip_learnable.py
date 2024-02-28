import torch
import torch.nn as nn
from ann_savi_base import ANNSAVIBase


class ANNSAVISkipLearnable(ANNSAVIBase):
    def __init__(self, device, train_x, train_y, test_x, test_y, validation_x, validation_y):
        super().__init__(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
        self.L = nn.Parameter(torch.tensor(0.5).to(device))
        self.linear = nn.Sequential(
            nn.Linear(4, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

    def get_L(self, x):
        return self.L

    def forward(self, x):
        savi = self.si(x)
        savi = savi.reshape(savi.shape[0],1)
        band_8 = x[:, 10]
        band_4 = x[:, 3]
        band_8 = band_8.reshape(band_8.shape[0], 1)
        band_4 = band_4.reshape(band_4.shape[0], 1)
        x = torch.hstack((savi, band_8, band_4, self.L.repeat(x.shape[0], 1)))
        x = self.linear(x)
        return x


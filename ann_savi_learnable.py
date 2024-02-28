import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from soil_dataset import SoilDataset
import utils
from pc import PearsonCorrelation


class ANNSAVILearnable(nn.Module):
    def __init__(self, device, train_x, train_y, test_x, test_y, validation_x, validation_y):
        super().__init__()
        self.verbose = True
        self.TEST = False
        self.device = device
        self.train_ds = SoilDataset(train_x, train_y)
        self.test_ds = SoilDataset(test_x, test_y)
        self.validation_ds = SoilDataset(validation_x, validation_y)
        self.num_epochs = 5000
        self.batch_size = 3000
        self.lr = 0.001
        self.L = nn.Parameter(torch.tensor(0.5))
        self.linear = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )
        self.pc = PearsonCorrelation()

    def forward(self, x):
        savi = self.si(x)
        savi = savi.reshape(savi.shape[0],1)
        x = self.linear(savi)
        return x, savi

    def si(self, x):
        band_8 = x[:,10]
        band_4 = x[:,3]
        savi = ((band_8-band_4)/(band_8+band_4+self.L))*(1+self.L)
        return savi

    def train_model(self):
        if self.TEST:
            return
        self.train()
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        criterion = torch.nn.MSELoss(reduction='mean')
        dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        total_batch = len(dataloader)
        for epoch in range(self.num_epochs):
            for batch_number, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat, savi = self(x)
                y_hat = y_hat.reshape(-1)
                loss = criterion(y_hat, y)
                pc = self.pc(savi.reshape(-1), y)
                pc_loss = -torch.abs(pc)/100
                loss_total = loss + pc_loss

                if self.verbose:
                    r2_test = r2_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
                    y_all, y_hat_all = self.evaluate(self.validation_ds)
                    r2_validation = r2_score(y_all, y_hat_all)
                    s2,y2 = self.evaluate_si(self.validation_ds)
                    pc_scipy = utils.calculate_pc(s2,y2)
                    s = self.si(x)
                    pc_scipy2 = utils.calculate_pc(s, y)
                    print(f'Epoch:{epoch} (of {self.num_epochs}), Batch: {batch_number+1} of {total_batch}, '
                          f'Loss:{loss.item():.6f}, PC_Loss:{pc_loss.item():.6f}, PC:{pc.item():.3f}, '
                          f'LossTotal:{loss_total.item():.3f}, R2_TRAIN: {r2_test:.3f}, R2_Validation: {r2_validation:.3f}, '
                          f'PC_scipy_val: {pc_scipy:.3f}, PC_scipy_train: {pc_scipy2:.3f}, L: {self.L.item():.3f}')

                loss_total.backward()
                optimizer.step()
                optimizer.zero_grad()

    def evaluate_si(self, ds):
        batch_size = 30000
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        for (x, y) in dataloader:
            x = x.to(self.device)
            s = self.si(x)
            return s,y

    def evaluate(self, ds):
        batch_size = 30000
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        y_all = np.zeros(0)
        y_hat_all = np.zeros(0)

        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat, savi = self(x)
            y_hat = y_hat.reshape(-1)
            y = y.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()

            y_all = np.concatenate((y_all, y))
            y_hat_all = np.concatenate((y_hat_all, y_hat))

        return y_all, y_hat_all

    def test(self):
        self.eval()
        self.to(self.device)
        y_all, y_hat_all = self.evaluate(self.test_ds)
        return y_hat_all


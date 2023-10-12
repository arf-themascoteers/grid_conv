import torch
from torch.utils.data import Dataset
import numpy as np


class SoilDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = torch.tensor(y, dtype=torch.float32)
        self.path = "data/processed/8e09234d1e1696d5c65e715b39d56b55/nbs"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        scene = int(x['scene'])
        row = int(x['row'])
        column = int(x['column'])
        file_name = f"{scene}_{row}_{column}.npy"
        file_path = f"{self.path}/{file_name}"
        data = np.load(file_path)
        return data, self.y[idx]

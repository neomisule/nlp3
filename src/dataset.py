import numpy as np
import torch
from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.X[idx])
        y = torch.FloatTensor([self.y[idx]])
        return x, y
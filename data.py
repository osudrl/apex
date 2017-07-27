from torch.utils.data import Dataset
import numpy as np

# extends torch Dataset to support validation splits
class SplitDataset(Dataset):
    def __init__(self, X, y, percent_val=0.0):
        self.X = X
        self.y = y

        val_size = int(percent_val * X.shape[0])
        val_idx = np.random.choice(X.shape[0], val_size, replace=False)

        self.validation_set = (self.X[val_idx], self.y[val_idx])

        self.X = np.delete(self.X, val_idx, axis=0)
        self.y = np.delete(self.y, val_idx, axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

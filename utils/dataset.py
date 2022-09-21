from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.dim = X.shape[0]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.dim
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y
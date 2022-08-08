from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, X, Y, transform, device):
        self.data = X[device] # [1, 64, 431]
        self.labels = Y[device]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = self.transform(self.data[idx])
        return self.data[idx], self.labels[idx]

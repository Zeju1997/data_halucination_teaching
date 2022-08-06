from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, X, Y, transform):
        self.domain = "A"
        self.data = X[self.domain] # [1, 64, 431]
        self.labels = Y[self.domain]
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = self.transform(self.data[idx])
        return self.data[idx], self.labels[idx]

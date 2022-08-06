from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, X, Y):
        self.domain = "A"
        self.data = X[self.domain]
        self.labels = Y[self.domain]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

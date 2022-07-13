from torch.utils.data import Dataset


class MoonDataset(Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.labels = Y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

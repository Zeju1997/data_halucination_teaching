from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap




class HalfMoon(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, points, labels):
        'Initialization'
        self.labels = labels
        self.points = points

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.points[index]
        y = self.labels[index]

        return X, y

def create_halfmoon_dataset(n_samples=1000, noise=0.1, random_state=0, test_ratio=0.2, to_plot=False):

    datasets = make_moons(n_samples=n_samples, noise=noise, random_state=random_state, shuffle=True)

    X, y = datasets
    X = StandardScaler().fit_transform(X)
    X = X.astype('float32')

    if to_plot:
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        fig, ax = plt.subplots()
        ax.set_title("Input data")

        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                    edgecolors='k')


        plt.tight_layout()
        plt.show()

    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_ratio, random_state=random_state)

    moon_train = HalfMoon(X_train, y_train)
    moon_test = HalfMoon(X_test, y_test)

    return moon_train, moon_test




def load_data(dataset_name, batch_size):

    if dataset_name == "HalfMoon":
        train_dataset, test_dataset = create_halfmoon_dataset(n_samples=1000, noise=0.1, random_state=0, test_ratio=0.2, to_plot=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


# load_data('HalfMoon', batch_size=256)
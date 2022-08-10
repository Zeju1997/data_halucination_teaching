import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        """
        Constructeur classifieur linéaire simple
        Classification binaire (une seule sortie)
        :param n_in: nombre de features
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, bias=False)
        self.conv2 = nn.Conv2d(64, 96, kernel_size=3, stride=1, bias=False)
        self.conv3 = nn.Conv2d(96, 128, kernel_size=3, stride=1, bias=False)
        self.max_pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.lin1 = nn.Linear(512, 256, bias=False)
        self.lin2 = nn.Linear(256, num_classes, bias=False)
        self.act = nn.ReLU()
        self.output_act = nn.Softmax()

    def forward(self, x):
        """
        Méthode forward du modèle
        :param x: la donnée de size = (batch_size, nb_features) ou (nb_features)
        :return: la sortie du réseau à simple couche
        """
        x = self.max_pool(self.act(self.conv1(x)))
        x = self.max_pool(self.act(self.conv2(x)))
        x = self.max_pool(self.act(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        return x
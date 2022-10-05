import torch.nn as nn


class LinearClassifier_normal(nn.Module):
    def __init__(self, n_in):
        """
        Constructeur classifieur linéaire simple
        Classification binaire (une seule sortie)
        :param n_in: nombre de features
        """
        super(LinearClassifier, self).__init__()
        self.lin = nn.Linear(n_in, 1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Méthode forward du modèle
        :param x: la donnée de size = (batch_size, nb_features) ou (nb_features)
        :return: la sortie du réseau à simple couche
        """
        out = self.lin(x)
        return self.sig(out)


    # TODO: Whitebox optimized label
class LinearClassifier(nn.Module):
    def __init__(self, n_in):
        """
        Constructeur classifieur linéaire simple
        Classification binaire (une seule sortie)
        :param n_in: nombre de features
        """
        super(LinearClassifier, self).__init__()
        self.lin = nn.Linear(n_in, 2, bias=False)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Méthode forward du modèle
        :param x: la donnée de size = (batch_size, nb_features) ou (nb_features)
        :return: la sortie du réseau à simple couche
        """
        out = self.lin(x)
        return out


class FullLayer(nn.Module):
    '''explicitly define the full connected layer'''

    def __init__(self, feature_dim, n_classes):
        super(FullLayer, self).__init__()
        self.class_num = n_classes
        self.lin = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        x = self.lin(x)
        return x

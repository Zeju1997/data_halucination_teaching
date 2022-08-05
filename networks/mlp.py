import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_in=784):
        """
        Constructeur classifieur linéaire simple
        Classification binaire (une seule sortie)
        :param n_in: nombre de features
        """
        super(MLP, self).__init__()
        n_in = 784
        self.lin1 = nn.Linear(n_in, 128, bias=False)
        self.lin2 = nn.Linear(128, 1, bias=False)
        self.act = nn.ReLU()
        self.output_act = nn.Softmax()

    def forward(self, x):
        """
        Méthode forward du modèle
        :param x: la donnée de size = (batch_size, nb_features) ou (nb_features)
        :return: la sortie du réseau à simple couche
        """
        x = self.act(self.lin1(x))
        x = self.output_act(self.lin2(x))
        return x

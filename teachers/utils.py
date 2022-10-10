import networks.conv as conv
import networks.linear as linear
import torch
import torch.nn as nn


class BaseLinear(linear.LinearClassifier):
    """
    Modèle linéaire de base.
    Contient le modèle (lui-même), la fonction de perte et l'optimiseur
    """
    def __init__(self, n_in):
        super(BaseLinear, self).__init__(n_in)
        self.loss_fn = nn.BCELoss()
        # self.loss_fn = nn.CrossEntropyLoss() # TODO: CrossEntropy only used for whitebox optimized
        self.cuda()
        self.eta = 1e-3
        self.optim = torch.optim.SGD(self.parameters(), lr=self.eta)

    def update(self, X, y):
        """
        Méthode d'apprentissage
        :param X: La données / le batch de données
        :param y: Le label / le batch de labels
        :return: Rien (procedure)
        """
        self.train()
        self.optim.zero_grad()
        out = self(X)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optim.step()

        # grad = torch.autograd.grad(loss, self.lin.weight, create_graph=True)
        # new_weight = self.lin.weight - self.eta * grad[0]
        # self.lin.weight = torch.nn.Parameter(new_weight.cuda())


class BaseLinear1(torch.nn.Module):
    def __init__(self, n_in):
        super(BaseLinear, self).__init__()
        self.lin = nn.Linear(n_in, 1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.lin(x)
        return self.sig(out)


class BaseConv(conv.ConvModel):
    """
    Modèle à cnvolution de base.
    Contient le modèle (lui-même), la fonction de perte, et l'optimiseur.
    """
    def __init__(self, eta):
        super(BaseConv, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn.cuda()
        self.cuda()
        self.eta = eta
        self.optim = torch.optim.SGD(self.parameters(), lr=self.eta)

    def update(self, X, y):
        """
        Méthode d'apprentissage
        :param X: Les données d'apprentissage
        :param y: Les labels
        :return: Rien (procedure)
        """
        self.train()
        self.optim.zero_grad()
        out = self(X)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optim.step()

        # b = list(self.parameters())[0].clone()
        # print(torch.equal(a.data, b.data))

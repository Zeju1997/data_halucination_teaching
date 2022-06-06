import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# shape pour cifar10 /!\
'''
class ConvModel(nn.Module):
    def __init__(self):
        """
        Constructeur modèle convolutionnel.
        Dimension réglées pour CIFAR-10
        """
        img_size = 32
        super(ConvModel, self).__init__()

        # conv1 : img 3 * 32 * 32 -> img 20 * 28 * 28
        # maxpool1 : img 20 * 28 * 28 -> img 20 * 14 * 14
        # conv2 : img 20 * 14 * 14 -> img 50 * 10 * 10
        # maxpool2 : img 50 * 10 * 10 -> 50 * 5 * 5

        self.seq = nn.Sequential(nn.Conv2d(3, 20, (5,5)),
                                 nn.MaxPool2d((2, 2), stride=(2, 2)),
                                 nn.ReLU(),
                                 nn.Conv2d(20, 50, (5, 5)),
                                 nn.MaxPool2d((2, 2), stride=(2, 2)),
                                 nn.ReLU())

        self.linear1_dim = int((((img_size - 4) / 2 - 4) / 2) ** 2 * 50)
        self.lin = nn.Linear(self.linear1_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Méthode forward
        :param x: image de size = (nb_batch, 3, 32, 32)
        :return: La sortie du réseau de size = (nb_batch, 1)
        """
        # pour rajouter une dimension pour le batch
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
            one_data = True
        else:
            one_data = False

        out = self.seq(x).view(-1, self.linear1_dim)
        out = self.lin(out)
        out = self.sig(out)
        return out.squeeze(0) if one_data else out
'''


class ConvBlock(nn.Module):
    """
    The residual block used by ResNet.

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        stride: Stride size of the first convolution, used for downsampling
    """

    def __init__(self, in_channels, out_channels, stride=1, rep=2):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        self.conv_rep = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        self.max_pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.rep = rep - 1

    def forward(self, input):
        x = self.conv_in(input)
        for _ in range(self.rep):
            x = self.conv_rep(x)
        x = self.max_pool(x)
        return F.relu(x)


class ConvStack(nn.Module):
    """
    A stack of residual blocks.

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first layer
        stride: Stride size of the first layer, used for downsampling
        num_blocks: Number of residual blocks
    """

    def __init__(self, rep=2):
        super().__init__()

        blocks = [ConvBlock(in_channels=3, out_channels=16, rep=rep),
                  ConvBlock(in_channels=16, out_channels=32, rep=rep),
                  ConvBlock(in_channels=32, out_channels=64, rep=rep)]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        x = input
        for block in self.blocks:
            x = block(x)
        return x


class ConvModel(nn.Module):
    def __init__(self, rep=2):
        """
        Constructeur modèle convolutionnel.
        Dimension réglées pour CIFAR-10
        """
        img_size = 32
        super(ConvModel, self).__init__()

        self.seq = ConvStack(rep=rep)
        self.linear1_dim = int((((img_size / 2) / 2) / 2) ** 2 * 64)
        self.lin = nn.Linear(self.linear1_dim, 10)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Méthode forward
        :param x: image de size = (nb_batch, 3, 32, 32)
        :return: La sortie du réseau de size = (nb_batch, 1)
        """
        # pour rajouter une dimension pour le batch
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
            one_data = True
        else:
            one_data = False

        # out = self.seq(x)
        out = self.seq(x).view(-1, self.linear1_dim)
        out = self.lin(out)
        out = self.sig(out)
        return out.squeeze(0) if one_data else out

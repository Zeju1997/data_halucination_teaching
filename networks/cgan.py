import torch.nn as nn
import torch
import numpy as np


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.opt = opt
        self.label_emb = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.opt.dim + self.opt.label_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        self.model = nn.Sequential(
            nn.Linear(opt.label_dim + int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            # nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class Generator1(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.opt = opt

        self.label_emb = nn.Embedding(self.opt.n_classes, self.opt.n_classes*5)

        self.opt = opt
        in_channels = self.opt.dim + self.opt.n_classes*5
        hidden_dim = self.opt.hidden_dim

        self.input_fc = nn.Linear(in_channels, hidden_dim*4, bias=False)
        self.hidden_fc = nn.Linear(hidden_dim*4, hidden_dim*2, bias=False)
        self.output_fc = nn.Linear(hidden_dim*2, self.opt.dim, bias=False)
        self.activation = nn.LeakyReLU(0.1)
        # self.activation = nn.Tanh()

    def forward(self, z, labels):
        x = torch.cat((z, self.label_emb(labels)), -1)
        x = self.activation(self.input_fc(x))
        x = self.activation(self.hidden_fc(x))
        return self.output_fc(x)


class Discriminator1(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        # self.opt.n_classes = 10
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.n_classes*5)
        hidden_dim = self.opt.hidden_dim
        data_dim = self.opt.dim

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes*5 + data_dim, hidden_dim*4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim*4, hidden_dim*4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim*4, hidden_dim*2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim*2, 1, bias=False),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

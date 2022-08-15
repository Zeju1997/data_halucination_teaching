import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class Discriminator_MNIST(nn.Module):
  """ D(x) """
  def __init__(self):
    # initalize super module
    super(Discriminator_MNIST, self).__init__()

    # creating layer for image input , input size : (batch_size, 1, 28, 28)
    self.layer_x = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32,
                                           kernel_size=4, stride=2, padding=1, bias=False),
                                 # out size : (batch_size, 32, 14, 14)
                                 nn.LeakyReLU(0.2, inplace=True),
                                 # out size : (batch_size, 32, 14, 14)
                                )

    # creating layer for label input, input size : (batch_size, 10, 28, 28)
    self.layer_y = nn.Sequential(nn.Conv2d(in_channels=10, out_channels=32,
                                           kernel_size=4, stride=2, padding=1, bias=False),
                                 # out size : (batch_size, 32, 14, 14)
                                 nn.LeakyReLU(0.2, inplace=True),
                                 # out size : (batch_size, 32, 14, 14)
                                 )

    # layer for concat of image layer and label layer, input size : (batch_size, 64, 14, 14)
    self.layer_xy = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128,
                                            kernel_size=4, stride=2, padding=1, bias=False),
                               # out size : (batch_size, 128, 7, 7)
                               nn.BatchNorm2d(128),
                               # out size : (batch_size, 128, 7, 7)
                               nn.LeakyReLU(0.2, inplace=True),
                               # out size : (batch_size, 128, 7, 7)
                               nn.Conv2d(in_channels=128, out_channels=256,
                                         kernel_size=3, stride=2, padding=0, bias=False),
                               # out size : (batch_size, 256, 3, 3)
                               nn.BatchNorm2d(256),
                               # out size : (batch_size, 256, 3, 3)
                               nn.LeakyReLU(0.2, inplace=True),
                               # out size : (batch_size, 256, 3, 3)
                               # Notice in below layer, we are using out channels as 1, we don't need to use Linear layer
                               # Same is recommended in DCGAN paper also
                               nn.Conv2d(in_channels=256, out_channels=1,
                                         kernel_size=3, stride=1, padding=0, bias=False),
                               # out size : (batch_size, 1, 1, 1)
                               # sigmoid layer to convert in [0,1] range
                               nn.Sigmoid()
                               )

  def forward(self, x, y):
    # size of x : (batch_size, 1, 28, 28)
    x = self.layer_x(x)
    # size of x : (batch_size, 32, 14, 14)

    # size of y : (batch_size, 10, 28, 28)
    y = self.layer_y(y)
    # size of y : (batch_size, 32, 14, 14)

    # concat image layer and label layer output
    xy = torch.cat([x,y], dim=1)
    # size of xy : (batch_size, 64, 14, 14)
    xy = self.layer_xy(xy)
    # size of xy : (batch_size, 1, 1, 1)
    xy = xy.view(xy.shape[0], -1)
    # size of xy : (batch_size, 1)
    return xy


class Discriminator_CIFAR100(nn.Module):
    def __init__(self, ngpu=1, nc=64, ndf=64):
        super(Discriminator_CIFAR100, self).__init__()
        self.ngpu = ngpu

        # creating layer for image input , input size : (batch_size, 1, 28, 28)
        self.layer_x = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32,
                                              kernel_size=3, stride=1, padding=1, bias=False),
                                    # out size : (batch_size, 32, 14, 14)
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # out size : (batch_size, 32, 14, 14)
                                    )

        # creating layer for label input, input size : (batch_size, 10, 28, 28)
        self.layer_y = nn.Sequential(nn.Conv2d(in_channels=100, out_channels=32,
                                              kernel_size=3, stride=1, padding=1, bias=False),
                                    # out size : (batch_size, 32, 14, 14)
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # out size : (batch_size, 32, 14, 14)
                                    )

        self.layer_xy = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # size of x : (batch_size, 1, 28, 28)
        x = self.layer_x(x)
        # size of x : (batch_size, 32, 14, 14)

        # size of y : (batch_size, 10, 28, 28)
        y = self.layer_y(y)
        # size of y : (batch_size, 32, 14, 14)

        # concat image layer and label layer output
        xy = torch.cat([x,y], dim=1)
        # size of xy : (batch_size, 64, 14, 14)
        xy = self.layer_xy(xy)
        # size of xy : (batch_size, 1, 1, 1)
        xy = xy.view(xy.shape[0], -1)
        # size of xy : (batch_size, 1)
        return xy


class Discriminator_CIFAR10(nn.Module):
    def __init__(self, ngpu=1, nc=64, ndf=64):
        super(Discriminator_CIFAR10, self).__init__()
        self.ngpu = ngpu

        # creating layer for image input , input size : (batch_size, 1, 28, 28)
        self.layer_x = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32,
                                              kernel_size=3, stride=1, padding=1, bias=False),
                                    # out size : (batch_size, 32, 14, 14)
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # out size : (batch_size, 32, 14, 14)
                                    )

        # creating layer for label input, input size : (batch_size, 10, 28, 28)
        self.layer_y = nn.Sequential(nn.Conv2d(in_channels=10, out_channels=32,
                                              kernel_size=3, stride=1, padding=1, bias=False),
                                    # out size : (batch_size, 32, 14, 14)
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # out size : (batch_size, 32, 14, 14)
                                    )

        self.layer_xy = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # size of x : (batch_size, 1, 28, 28)
        x = self.layer_x(x)
        # size of x : (batch_size, 32, 14, 14)

        # size of y : (batch_size, 10, 28, 28)
        y = self.layer_y(y)
        # size of y : (batch_size, 32, 14, 14)

        # concat image layer and label layer output
        xy = torch.cat([x,y], dim=1)
        # size of xy : (batch_size, 64, 14, 14)
        xy = self.layer_xy(xy)
        # size of xy : (batch_size, 1, 1, 1)
        xy = xy.view(xy.shape[0], -1)
        # size of xy : (batch_size, 1)
        return xy


class Generator_MNIST(nn.Module):
  """ G(z) """
  def __init__(self, input_size=100):
    # initalize super module
    super(Generator_MNIST, self).__init__()

    # noise z input layer : (batch_size, 100, 1, 1)
    self.layer_x = nn.Sequential(nn.ConvTranspose2d(in_channels=100, out_channels=128, kernel_size=3,
                                                  stride=1, padding=0, bias=False),
                                 # out size : (batch_size, 128, 3, 3)
                                 nn.BatchNorm2d(128),
                                 # out size : (batch_size, 128, 3, 3)
                                 nn.ReLU(),
                                 # out size : (batch_size, 128, 3, 3)
                                )
    
    # label input layer : (batch_size, 10, 1, 1)
    self.layer_y = nn.Sequential(nn.ConvTranspose2d(in_channels=10, out_channels=128, kernel_size=3,
                                                  stride=1, padding=0, bias=False),
                                 # out size : (batch_size, 128, 3, 3)
                                 nn.BatchNorm2d(128),
                                 # out size : (batch_size, 128, 3, 3)
                                 nn.ReLU(),
                                 # out size : (batch_size, 128, 3, 3)
                                )
    
    # noise z and label concat input layer : (batch_size, 256, 3, 3)
    self.layer_xy = nn.Sequential(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3,
                                                  stride=2, padding=0, bias=False),
                               # out size : (batch_size, 128, 7, 7)
                               nn.BatchNorm2d(128),
                               # out size : (batch_size, 128, 7, 7)
                               nn.ReLU(),
                               # out size : (batch_size, 128, 7, 7)
                               nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
                                                  stride=2, padding=1, bias=False),
                               # out size : (batch_size, 64, 14, 14)
                               nn.BatchNorm2d(64),
                               # out size : (batch_size, 64, 14, 14)
                               nn.ReLU(),
                               # out size : (batch_size, 64, 14, 14)
                               nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4,
                                                  stride=2, padding=1, bias=False),
                               # out size : (batch_size, 1, 28, 28)
                               nn.Tanh())
                               # out size : (batch_size, 1, 28, 28)
    
  def forward(self, x, y):
    # x size : (batch_size, 100)
    x = x.view(x.shape[0], x.shape[1], 1, 1)
    # x size : (batch_size, 100, 1, 1)
    x = self.layer_x(x)
    # x size : (batch_size, 128, 3, 3)
    
    # y size : (batch_size, 10)
    y = y.view(y.shape[0], y.shape[1], 1, 1)
    # y size : (batch_size, 100, 1, 1)
    y = self.layer_y(y)
    # y size : (batch_size, 128, 3, 3)

    # concat x and y 
    xy = torch.cat([x,y], dim=1)
    # xy size : (batch_size, 256, 3, 3)
    xy = self.layer_xy(xy)
    # xy size : (batch_size, 1, 28, 28)
    return xy

class Generator_CIFAR100(nn.Module):
    def __init__(self, ngpu=1, nc=3, nz=256, ngf=64):
        super(Generator_CIFAR100, self).__init__()
        self.ngpu = ngpu

        # noise z input layer : (batch_size, 100, 1, 1)
        self.layer_x = nn.Sequential(nn.ConvTranspose2d(in_channels=100, out_channels=128, kernel_size=3,
                                                      stride=1, padding=1, bias=False),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.BatchNorm2d(128),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.ReLU(),
                                    # out size : (batch_size, 128, 3, 3)
                                    )
        
        # label input layer : (batch_size, 10, 1, 1)
        self.layer_y = nn.Sequential(nn.ConvTranspose2d(in_channels=100, out_channels=128, kernel_size=3,
                                                      stride=1, padding=1, bias=False),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.BatchNorm2d(128),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.ReLU(),
                                    # out size : (batch_size, 128, 3, 3)
                                    )

        self.layer_xy = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x, y):
        # x size : (batch_size, 100)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        # x size : (batch_size, 100, 1, 1)
        x = self.layer_x(x)
        # x size : (batch_size, 128, 3, 3)
        
        # y size : (batch_size, 10)
        y = y.view(y.shape[0], y.shape[1], 1, 1)
        # y size : (batch_size, 100, 1, 1)
        y = self.layer_y(y)
        # y size : (batch_size, 128, 3, 3)
        
        # concat x and y 
        xy = torch.cat([x,y], dim=1)
        # xy size : (batch_size, 256, 3, 3)
        xy = self.layer_xy(xy)
        # xy size : (batch_size, 1, 28, 28)
        return xy

class Generator_CIFAR10(nn.Module):
    def __init__(self, ngpu=1, nc=3, nz=256, ngf=64):
        super(Generator_CIFAR10, self).__init__()
        self.ngpu = ngpu

        # noise z input layer : (batch_size, 100, 1, 1)
        self.layer_x = nn.Sequential(nn.ConvTranspose2d(in_channels=100, out_channels=128, kernel_size=3,
                                                      stride=1, padding=1, bias=False),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.BatchNorm2d(128),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.ReLU(),
                                    # out size : (batch_size, 128, 3, 3)
                                    )
        
        # label input layer : (batch_size, 10, 1, 1)
        self.layer_y = nn.Sequential(nn.ConvTranspose2d(in_channels=10, out_channels=128, kernel_size=3,
                                                      stride=1, padding=1, bias=False),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.BatchNorm2d(128),
                                    # out size : (batch_size, 128, 3, 3)
                                    nn.ReLU(),
                                    # out size : (batch_size, 128, 3, 3)
                                    )

        self.layer_xy = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x, y):
        # x size : (batch_size, 100)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        # x size : (batch_size, 100, 1, 1)
        x = self.layer_x(x)
        # x size : (batch_size, 128, 3, 3)
        
        # y size : (batch_size, 10)
        y = y.view(y.shape[0], y.shape[1], 1, 1)
        # y size : (batch_size, 100, 1, 1)
        y = self.layer_y(y)
        # y size : (batch_size, 128, 3, 3)
        
        # concat x and y 
        xy = torch.cat([x,y], dim=1)
        # xy size : (batch_size, 256, 3, 3)
        xy = self.layer_xy(xy)
        # xy size : (batch_size, 1, 28, 28)
        return xy

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, model=18, in_channels=1, num_classes=10, pretrained=False):
        super(ResNet, self).__init__()
        if model == 19:
            self.model = models.resnet18(pretrained=pretrained)
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(512, num_classes)
        else:
            self.model = models.resnet50(pretrained=pretrained)
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


def ResNet18():
    return ResNet(model=18)

def ResNet34():
    return ResNet(model=34)

def ResNet50():
    return ResNet(model=50)

def ResNet101():
    return ResNet(model=101)

def ResNet152():
    return ResNet(model=152)


def test():
    net = ResNet18()
    y = net(torch.randn(2,1,256,256))
    print(y)
    print(y.size())
    print(net)

# test()

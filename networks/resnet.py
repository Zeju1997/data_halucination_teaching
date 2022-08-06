import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, model=18, in_channels=1, num_classes=10, pretrained=False):
        super(ResNet, self).__init__()
        if model == 18:
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


def ResNet18(in_channels=3):
    return ResNet(model=18, in_channels=in_channels)


def ResNet34(in_channels=3):
    return ResNet(model=34, in_channels=in_channels)


def ResNet50(in_channels=3):
    return ResNet(model=50, in_channels=in_channels)


def ResNet101(in_channels=3):
    return ResNet(model=101, in_channels=in_channels)


def ResNet152(in_channels=3):
    return ResNet(model=152, in_channels=in_channels)


def test():
    net = ResNet18()
    y = net(torch.randn(2,1,256,256))
    print(y)
    print(y.size())
    print(net)

# test()

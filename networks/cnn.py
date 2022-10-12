import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
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
        x = self.max_pool(self.act(self.conv1(x)))
        x = self.max_pool(self.act(self.conv2(x)))
        x = self.max_pool(self.act(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.act(self.lin1(x))
        x = self.lin2(x)
        return x



class CNN2(nn.Module):
    def __init__(self, name, in_channels=3, num_classes=100):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.lin = nn.Identity()

        self.feature_num = 400

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.lin(x)
        return x


cfg = {
    'CNN3': [64, 'M', 128, 'M', 256, 'M'],
    'CNN6': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'CNN9': [64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M'],
    'CNN15': [64, 64, 64, 64, 64, 'M', 128, 128, 128, 128, 128, 'M', 256, 256, 256, 256, 256, 'M'],
}

class CNN(nn.Module):
    def __init__(self, cnn_name, in_channels=3, num_classes=10, feature_extractor=True):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.feature_num = 256
        self.features = self._make_layers(cfg[cnn_name])
        if feature_extractor:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(self.feature_num, num_classes)
        self.in_channels = in_channels

    def forward(self, x):
        out = self.features(x)
        # out = out.view(out.size(0), -1)
        out = torch.flatten(out, 1) # flatten all dimensions except batch
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                           # nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.Conv2d(256, 256, kernel_size=4, padding=0),
                   nn.BatchNorm2d(256),
                   nn.ReLU()]
        # layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)


class NET(nn.Module):
    def __init__(self, n_in=784, in_channels=1, num_classes=10):
        super(NET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(p=0.2)
        if n_in == 784:
            self.lin1 = nn.Linear(16 * 4 * 4, 120)
        else:
            self.lin1 = nn.Linear(16 * 5 * 5, 120)
        self.lin1 = nn.Linear(32 * 2 * 2, 120)
        self.lin2 = nn.Linear(120, num_classes)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.feature_num = 120
        self.output_act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.output_act(x)
        return x

class NET1(nn.Module):
    def __init__(self, n_in=784, in_channels=3, num_classes=10):
        super(NET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if n_in == 784:
            self.lin1 = nn.Linear(16 * 4 * 4, 120)
        else:
            self.lin1 = nn.Linear(16 * 5 * 5, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

from torch.autograd import Variable

# net = CNN('CNN6')
# x = torch.randn(2, 3, 32, 32)
# print(net(Variable(x)).size())
# print(net)

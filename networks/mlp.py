import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels=784, num_classes=2):
        super(MLP, self).__init__()
        n_in = 784
        self.feature_num = 128
        self.features = nn.Linear(n_in, self.feature_num, bias=False)
        # self.classifier = nn.Linear(self.feature_num, 1, bias=False)
        self.classifier = nn.Identity()
        self.act = nn.ReLU()
        self.output_act = nn.Softmax()

    def forward(self, x):
        out = self.act(self.features(x))
        out = self.classifier(out)
        return out

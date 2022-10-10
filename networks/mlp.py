import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, n_in=784, num_classes=2):
        super(MLP, self).__init__()
        self.feature_num = 128
        self.features = nn.Linear(n_in, self.feature_num, bias=False)
        # self.classifier = nn.Linear(self.feature_num, 1, bias=False)
        self.classifier = nn.Identity()
        self.act = nn.ReLU()
        self.output_act = nn.Softmax()

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        out = self.act(self.features(x))
        out = self.classifier(out)
        return out

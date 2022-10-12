import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, n_in=784, num_classes=2, features_extractor=True):
        super(MLP, self).__init__()
        self.feature_num = 128
        self.features = nn.Linear(n_in, self.feature_num, bias=False)
        if features_extractor:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(self.feature_num, num_classes, bias=False)

        self.act = nn.ReLU()
        self.output_act = nn.Softmax()

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        out = self.act(self.features(x))
        out = self.classifier(out)
        return out

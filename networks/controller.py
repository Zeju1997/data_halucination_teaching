import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):
    def __init__(self, opt, input_size=11, hidden_size=64, num_steps=4, device=''):
        super(Agent, self).__init__()

        # Could add an embedding layer
        # embedding_size = 100
        # self.embedding = nn.Embedding(input_size, embedding_size)
        # dropout layer
        #self.drop = nn.Dropout(dropout)
        self.DEVICE = device
        self.num_alpha_options = 11

        self.lstm1 = nn.LSTMCell(input_size, hidden_size).cuda()
        # May be could just use different decoder if these two numbers are the same, not sure
        self.decoder = nn.Linear(hidden_size, self.num_alpha_options).cuda()
        #self.decoder2 = nn.Linear(hidden_size, self.filter_size_option)

        # num_steps = max_layer * 2 # two conv layer * 2 h-parameters (kernel size and number of kernels)
        self.num_steps = opt.n_epochs
        self.nhid = hidden_size
        self.hidden = self.init_hidden()

    def forward(self, input):
        outputs = []
        h_t, c_t = self.hidden

        for i in range(self.num_steps):
            # input_data = self.embedding(step_data)

            h_t, c_t = self.lstm1(input, (h_t, c_t))
            # Add drop out
            # h_t = self.drop(h_t)
            output = self.decoder(h_t)
            input = output
            outputs += [output]

        outputs = torch.stack(outputs).squeeze(1)

        return outputs

    def init_hidden(self):
        h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)

        return (h_t, c_t)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(3, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 11)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.opt = opt
        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.label_dim)
        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        self.offset = torch.tensor([-0.1, 0, 0.1]).cuda()

        in_channels = self.opt.label_dim + int(np.prod(self.img_shape))

        self.model = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.opt.n_classes),
            # nn.Sigmoid()
        )
        # feat_dim = torch.combinations(torch.arange(self.opt.n_query_classes))
        feat_dim = self.opt.n_query_classes

        self.fc1 = nn.Linear(self.opt.n_classes + feat_dim, 16)
        self.fc2 = nn.Linear(16, 10)
        self.fc3 = nn.Linear(10 + 4, 3)

        # self.act = nn.Sigmoid()
        self.act = nn.Softmax(dim=1)

    def forward(self, img, label, feat_model, feat_sim, lam):

        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((img1.view(img1.size(0), -1), (img2.view(img2.size(0), -1), self.label_embedding(label1), self.label_embedding(label2)), -1))
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(label)), -1)
        x = self.model(d_in)

        feat = feat_sim.unsqueeze(0).repeat(img.shape[0], 1)
        x = torch.cat((x, feat), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        feat_model = feat_model.unsqueeze(0).repeat(x.shape[0], 1)
        lam_repeat = lam.unsqueeze(0).repeat(x.shape[0], 1)
        x = torch.cat((x, feat_model, lam_repeat), dim=1)
        # actions = self.fc3(x)
        x = self.act(self.fc3(x))

        offset = x @ self.offset

        print("offset", offset.mean())

        lam = lam + offset.mean()
        val_lam = torch.clamp(lam, min=0, max=1)

        return val_lam

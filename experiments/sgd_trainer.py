import torch
import numpy as np
import torch.nn as nn
import os
import csv
import sys
from tqdm import tqdm


def plot_classifier(model, max, min):
    w = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            w = layer.state_dict()['weight'].cpu().numpy()

    slope = (-w[0, 0]/w[0, 1] - 1) / (1 + w[0, 1]/w[0, 0])

    x = np.linspace(min, max, 100)
    y = slope * x
    return x, y


class SGDTrainer(nn.Module):
    def __init__(self, opt, X_train, Y_train, X_test, Y_test, data=None):
        super(SGDTrainer, self).__init__()

        self.opt = opt

        self.opt.experiment = "SGD"

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.data = data

    def train(self, model, w_star):
        self.opt.experiment = "SGD"
        print("Start training {} ...".format(self.opt.experiment))
        logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
        if not os.path.exists(logname):
            with open(logname, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['iter', 'test acc', 'w diff'])

        res_sgd = []
        a_example = []
        b_example = []
        w_diff_sgd = []

        nb_batch = int(self.opt.nb_train / self.opt.batch_size)

        for idx in tqdm(range(self.opt.n_iter)):
            if idx != 0:
                i = torch.randint(0, nb_batch, size=(1,)).item()
                sample, label = self.data_sampler(self.X_train, self.Y_train, i)

                if self.data is not None:
                    random_sample, random_label = self.data[i]
                    if idx == 1:
                        random_samples = random_sample.unsqueeze(0).cpu().detach().numpy()  # [np.newaxis, :]
                        random_labels = random_label.unsqueeze(0).unsqueeze(1).cpu().detach().numpy()  # [np.newaxis, :]
                    else:
                        random_samples = np.concatenate((random_samples, random_sample.unsqueeze(0).cpu().detach().numpy()), axis=0)
                        random_labels = np.concatenate((random_labels, random_label.unsqueeze(0).unsqueeze(1).cpu().detach().numpy()), axis=0)
                else:
                    if idx == 1:
                        random_samples = sample.cpu().detach().numpy()  # [np.newaxis, :]
                        random_labels = label.unsqueeze(1).cpu().detach().numpy()  # [np.newaxis, :]
                    else:
                        random_samples = np.concatenate((random_samples, sample.cpu().detach().numpy()), axis=0)
                        random_labels = np.concatenate((random_labels, label.unsqueeze(1).cpu().detach().numpy()), axis=0)

                model.update(sample, label.unsqueeze(1))

            model.eval()
            test = model(self.X_test.cuda()).cpu()

            # a, b = plot_classifier(model, self.X_train.max(axis=0), self.X_train.min(axis=0))
            # a_example.append(a)
            # b_example.append(b)

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "covid":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == self.Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == self.Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()

            acc = nb_correct / self.X_test.size(0)
            res_sgd.append(acc)

            w = model.lin.weight
            w = w / torch.norm(w)
            diff = torch.linalg.norm(w_star - w, ord=2) ** 2
            w_diff_sgd.append(diff.detach().clone().cpu())

            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([idx, acc, diff.item()])

        return random_samples, random_labels

    def data_sampler(self, X, Y, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = X[i_min:i_max].cuda()
        y = Y[i_min:i_max].cuda()

        return x, y

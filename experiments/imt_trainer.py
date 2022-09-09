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


class IMTTrainer(nn.Module):
    def __init__(self, opt, X_train, Y_train, X_test, Y_test):
        super(IMTTrainer, self).__init__()

        self.opt = opt

        self.experiment = "SGD"

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def train(self, model, teacher, w_star):
        self.experiment = "IMT_Baseline"
        print("Start training {} ...".format(self.experiment))
        logname = os.path.join(self.opt.log_path, 'results' + '_' + self.experiment + '_' + str(self.opt.seed) + '.csv')
        if not os.path.exists(logname):
            with open(logname, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['iter', 'test acc', 'w diff'])

        res_baseline = []
        a_baseline = []
        b_baseline = []
        w_diff_baseline = []

        for t in tqdm(range(self.opt.n_iter)):
            if t != 0:
                i = teacher.select_example(model, self.X_train.cuda(), self.Y_train.cuda(), self.opt.batch_size)
                # i = torch.randint(0, nb_batch, size=(1,)).item()

                best_data, best_label = self.data_sampler(self.X_train, self.Y_train, i)

                selected_data = best_data.detach().clone().cpu().numpy()
                selected_label = best_label.detach().clone().cpu().numpy()
                if t == 1:
                    selected_samples = selected_data # [np.newaxis, :]
                    selected_labels = selected_label # [np.newaxis, :]
                else:
                    selected_samples = np.concatenate((selected_samples, selected_data), axis=0)
                    selected_labels = np.concatenate((selected_labels, selected_label), axis=0)

                model.update(best_data, best_label)

            model.eval()
            test = model(self.X_test.cuda()).cpu()

            # a, b = plot_classifier(model, X.max(axis=0), X.min(axis=0))
            # a_baseline.append(a)
            # b_baseline.append(b)

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == self.Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == self.Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc_base = nb_correct / self.X_test.size(0)
            res_baseline.append(acc_base)

            w = model.lin.weight
            w = w / torch.norm(w)
            diff = torch.linalg.norm(w_star - w, ord=2) ** 2
            w_diff_baseline.append(diff.detach().clone().cpu())

            print("acc", acc_base)

            # sys.stdout.write("\r" + str(t) + "/" + str(self.opt.n_iter) + ", idx=" + str(i) + " " * 100)
            # sys.stdout.flush()

            print("Base line trained\n")

            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([t, acc_base, diff.item()])

    def data_sampler(self, X, Y, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = X[i_min:i_max].cuda()
        y = Y[i_min:i_max].cuda()

        return x, y
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
    def __init__(self, opt, X_train, Y_train, X_test, Y_test, data=None):
        super(IMTTrainer, self).__init__()

        self.opt = opt

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.data = data

    def train(self, model, teacher, w_star):
        print("Start training {} ...".format(self.opt.experiment))
        logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
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
                if self.opt.experiment == "IMT_Baseline":
                    i = teacher.select_example(model, self.X_train.cuda(), self.Y_train.cuda(), self.opt.batch_size)
                    # i = torch.randint(0, 1000, size=(1,)).item()
                else:
                    i = teacher.select_example_random_label(model, self.X_train.cuda(), self.Y_train.cuda(), self.opt.batch_size)

                best_sample, best_label = self.data_sampler(self.X_train, self.Y_train, i)

                if self.data is not None:
                    selected_sample, selected_label = self.data[i]
                    if t == 1:
                        selected_samples = selected_sample.unsqueeze(0).cpu().detach().numpy()  # [np.newaxis, :]
                        selected_labels = selected_label.unsqueeze(0).unsqueeze(1).cpu().detach().numpy()  # [np.newaxis, :]
                    else:
                        selected_samples = np.concatenate((selected_samples, selected_sample.unsqueeze(0).cpu().detach().numpy()), axis=0)
                        selected_labels = np.concatenate((selected_labels, selected_label.unsqueeze(0).unsqueeze(1).cpu().detach().numpy()), axis=0)
                else:
                    if t == 1:
                        selected_samples = best_sample.cpu().detach().numpy()  # [np.newaxis, :]
                        selected_labels = best_label.unsqueeze(1).cpu().detach().numpy()  # [np.newaxis, :]
                    else:
                        selected_samples = np.concatenate((selected_samples, best_sample.cpu().detach().numpy()), axis=0)
                        selected_labels = np.concatenate((selected_labels, best_label.unsqueeze(1).cpu().detach().numpy()), axis=0)

                model.update(best_sample, best_label.unsqueeze(1))

            model.eval()
            test = model(self.X_test.cuda()).cpu()

            # a, b = plot_classifier(model, X.max(axis=0), X.min(axis=0))
            # a_baseline.append(a)
            # b_baseline.append(b)

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable" or self.opt.data_mode == "covid":
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

        return selected_samples, selected_labels

    def data_sampler(self, X, Y, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = X[i_min:i_max].cuda()
        y = Y[i_min:i_max].cuda()

        return x, y

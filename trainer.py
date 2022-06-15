from __future__ import absolute_import, division, print_function

import numpy as np
import time
import json

import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn as nn
import json
import os
import networks
import data
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from train_utils import *
from eval import EvalMetrics
import teachers.omniscient_teacher_nn as omniscient
import teachers.surrogate_teacher as surrogate
import teachers.imitation_teacher as imitation
import teachers.utils as utils
import matplotlib.pyplot as plt
import data.dataset_loader as data_loader

from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split

sys.path.append('..') #Hack add ROOT DIR
from baseconfig import CONF


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def plot_classifier1(model, max, min):
    w = 0
    b = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            w = layer.state_dict()['weight'].cpu().numpy()
            b = layer.state_dict()['bias'].cpu().numpy()

    slope = -(b/w[0, 1])/(b/w[0, 0])
    intercept = b/w[0, 1]

    x = np.linspace(min, max, 100)
    y = slope * x + intercept
    return x, y


def plot_classifier(model, max, min):
    w = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            w = layer.state_dict()['weight'].cpu().numpy()

    slope = (-w[0, 0]/w[0, 1] - 1) / (1 + w[0, 1]/w[0, 0])

    x = np.linspace(min, max, 100)
    y = slope * x
    return x, y


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.visualize = True

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        if self.opt.data_mode == "cifar10":
            self.train_dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=True, download=True, transform=ToTensor())
            self.test_dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=False, download=True, transform=ToTensor())
            # self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size)
            self.train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset))
            # self.test_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size)
            self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))
        elif self.opt.data_mode == "mnist":
            self.train_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=True, download=True, transform=ToTensor())
            self.test_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=False, download=True, transform=ToTensor())
        elif self.opt.data_mode == "gaussian":
            print("Generating Gaussian data ...")
        elif self.opt.data_mode == "moon":
            print("Generating moon data ...")
        elif self.opt.data_mode == "linearly_seperable":
            print("Generating linearly seperable data ...")
        else:
            print("Unrecognized data!")
            sys.exit()

        self.get_teacher_student()

    def get_teacher_student(self):
        if self.opt.teaching_mode == "omniscient":
            if self.opt.data_mode == "cifar10":
                self.teacher = omniscient.OmniscientConvTeacher(self.opt.eta)
                self.student = omniscient.OmniscientConvStudent(self.opt.eta)
            else: # mnist / gaussian / moon
                self.teacher = omniscient.OmniscientLinearTeacher(self.opt.dim)
                self.student = omniscient.OmniscientLinearStudent(self.opt.dim)

                # self.teacher = omniscient.TeacherClassifier(self.opt.dim)
                # self.student = omniscient.StudentClassifier(self.opt.dim)
        elif self.opt.teaching_mode == "surrogate":
            if self.opt.data_mode == "cifar10":
                if self.opt.same_feat_space:
                    self.teacher = surrogate.SurrogateConvTeacher(self.opt.eta)
                    self.student = surrogate.SurrogateConvStudent(self.opt.eta)
                else:
                    self.teacher = surrogate.SurrogateConvTeacher(self.opt.eta)
                    self.student = surrogate.SurrogateConvStudent(self.opt.eta)
            elif self.opt.data_mode == "mnist":
                if self.opt.same_feat_space:
                    self.teacher = surrogate.SurrogateLinearTeacher(self.opt.dim)
                    self.student = surrogate.SurrogateLinearStudent(self.opt.dim)
                else:
                    self.teacher = surrogate.SurrogateDiffLinearTeacher(self.opt.dim, 24, normal_dist=True)
                    self.student = surrogate.SurrogateLinearStudent(self.opt.dim)
        elif self.opt.teaching_mode == "imitation":
            if self.opt.data_mode == "cifar10":
                if self.opt.same_feat_space:
                    tmp = next(iter(self.train_loader))[0]
                    fst_x = torch.Tensor(tmp[torch.randint(0, tmp.shape[0], (1,)).item()]).unsqueeze(0).cuda()
                    self.teacher = imitation.ImmitationConvTeacher(self.opt.eta, fst_x)
                    self.student = utils.BaseConv(self.opt.eta)
                else:
                    fst_x = torch.Tensor(data[torch.randint(0, data.shape[0], (1,)).item()]).unsqueeze(0).cuda()
                    self.teacher = imitation.ImmitationConvTeacher(self.opt.eta, fst_x)
                    self.student = utils.BaseConv(self.opt.eta)
            elif self.opt.data_mode == "mnist":
                if self.opt.same_feat_space:
                    fst_x = torch.Tensor(self.opt.dim).cuda()
                    self.teacher = imitation.ImitationLinearTeacher(self.opt.dim, fst_x)
                    self.student = utils.BaseLinear(self.opt.dim)
                else:
                    fst_x = torch.Tensor(self.opt.dim).cuda()
                    self.teacher = imitation.ImitationDiffLinearTeacher(self.opt.eta, fst_x)
                    self.student = utils.BaseConv(self.opt.eta)
        else:
            print("Unrecognized teacher!")
            sys.exit()
        self.student.load_state_dict(self.teacher.state_dict())

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def init_data(self, dim, nb_data_per_class):
        """
        Création des données gaussien
        :param dim: la dimension des données
        :param nb_data_per_class: le nombre d'exemple par classe
        :return: un tuple (données, labels)
        """
        X1 = np.random.multivariate_normal([0.5] * dim, np.identity(dim), nb_data_per_class)
        y1 = np.ones((nb_data_per_class,))

        X2 = np.random.multivariate_normal([-0.5] * dim, np.identity(dim), nb_data_per_class)
        y2 = np.zeros((nb_data_per_class,))

        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        indices = np.indices((nb_data_per_class * 2,))
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]
        return X.squeeze(0), y.squeeze(0)

    def train(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        # self.set_train()

        # X_train = self.train_dataset.data
        # Y_train = self.train_dataset.targets

        if self.opt.data_mode == "cifar10":
            train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset))
            X_train = next(iter(train_loader))[0].numpy()
            Y_train = next(iter(train_loader))[1].numpy()
            (N, W, H, C) = self.train_dataset.data.shape
            dim = W*H*C
            example = utils.BaseConv(self.opt.eta)

            X = X_train
            y = Y_train

        elif self.opt.data_mode == "mnist":
            train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset))
            X_train = next(iter(train_loader))[0].numpy()
            Y_train = next(iter(train_loader))[1].numpy()
            (N, W, H) = self.train_dataset.data.shape
            dim = W*H
            example = utils.BaseLinear(dim)
            # X_train = np.asarray(self.train_dataset.data.reshape((N, dim)))
            X_train = X_train.reshape((N, dim))
            # Y_train = np.asarray(self.train_dataset.targets)
            # Y_train = np.asarray(self.train_dataset.targets)

            # create new data set with class 1 as 0 and class 2 as 1
            f = (Y_train == self.opt.class_1) | (Y_train == self.opt.class_2)
            X = X_train[f]
            y = Y_train[f]
            y = np.where(y == self.opt.class_1, 0, 1)

        elif self.opt.data_mode == "gaussian":
            dim__diff = 7
            nb_data_per_class = 1000

            X, y = self.init_data(self.opt.dim, nb_data_per_class)

            example = utils.BaseLinear(self.opt.dim)
            baseline = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=y)
                plt.title('Gaussian Data')
                plt.show()

        elif self.opt.data_mode == "moon":
            np.random.seed(0)
            noise_val = 0.2

            X, y = make_moons(self.opt.nb_train+self.opt.nb_test, noise=noise_val)

            example = utils.BaseLinear(self.opt.dim)
            baseline = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=y)
                plt.title('Moon Data')
                plt.show()

        elif self.opt.data_mode == "linearly_seperable":
            X, y = make_classification(
                n_samples=self.opt.nb_train+self.opt.nb_test, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
            )
            rng = np.random.RandomState(2)
            X += 2 * rng.uniform(size=X.shape)

            example = utils.BaseLinear(self.opt.dim)
            baseline = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=y)
                plt.title('Linearly Seperable Data')
                plt.show()

        else:
            print("Unrecognized data mode!")
            sys.exit()

        example.load_state_dict(self.teacher.state_dict())
        baseline.load_state_dict(self.teacher.state_dict())

        # X_train = np.asarray(self.train_dataset.data.reshape((N, dim)))
        # X_train = np.asarray(X_train)
        # Y_train = np.asarray(self.train_dataset.targets)
        # Y_train = np.asarray(Y_train)

        # Shuffle datasets
        randomize = np.arange(X.shape[0])
        np.random.shuffle(randomize)
        X = X[randomize]
        y = y[randomize]

        X = X[:self.opt.nb_train + self.opt.nb_test]
        y = y[:self.opt.nb_train + self.opt.nb_test]

        nb_batch = int(self.opt.nb_train / self.opt.batch_size)

        if self.opt.data_mode == "cifar10":
            X_train = torch.tensor(X[:self.opt.nb_train])
            y_train = torch.tensor(y[:self.opt.nb_train], dtype=torch.long)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test])
            y_test = torch.tensor(y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.long)
        else:
            X_train = torch.tensor(X[:self.opt.nb_train], dtype=torch.float)
            y_train = torch.tensor(y[:self.opt.nb_train], dtype=torch.float)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)
            y_test = torch.tensor(y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)

        # train teacher
        accuracies = []
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.teacher.optim, milestones=[80, 160], gamma=0.1)
        for n in tqdm(range(self.opt.n_teacher_runs)):
            if n != 0:
                for i in range(nb_batch):
                    i_min = i * self.opt.batch_size
                    i_max = (i + 1) * self.opt.batch_size
                    self.teacher.update(X_train[i_min:i_max].cuda(), y_train[i_min:i_max].cuda())
            self.teacher.eval()
            test = self.teacher(X_test.cuda()).cpu()

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)
            accuracies.append(acc)
            print("Accuracy:", acc)
            self.scheduler.step()

        if self.visualize == True:
            fig = plt.figure()
            plt.plot(accuracies, c="b", label="Teacher (CNN)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

            fig = plt.figure(figsize=(8, 5))
            a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            plt.plot(a, b, '-r', label='y=wx+b')
            plt.scatter(X[:, 0], X[:, 1], c=y)
            plt.title('Linearly Seperable Data')
            plt.show()

        # train example
        res_example = []
        for n in tqdm(range(self.opt.n_iter)):
            if n != 0:
                i = torch.randint(0, nb_batch, size=(1,)).item()
                i_min = i * self.opt.batch_size
                i_max = (i + 1) * self.opt.batch_size

                data = X_train[i_min:i_max].cuda()
                label = y_train[i_min:i_max].cuda()

                example.update(data, label)

            example.eval()
            test = example(X_test.cuda()).cpu()

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)
            res_example.append(acc)

        if self.visualize == True:
            fig = plt.figure()
            plt.plot(res_example, c="b", label="Example (CNN)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

        print("Base line trained\n")

        # train student
        res_student = []
        res_baseline = []
        generated_samples = np.zeros(2)
        score_statistics = np.zeros(self.opt.n_iter)
        for t in tqdm(range(self.opt.n_iter)):
            if t != 0:
                # labels = torch.randint(0, 1, (self.opt.batch_size,), dtype=torch.float).cuda()
                best_data, best_label, data, labels, better = self.teacher.generate_example(self.student, X_train.cuda(), y_train.cuda(), self.opt.batch_size, self.opt.lr_factor, self.opt.gd_n)

                if better:
                    score_statistics[t] = 1

                # data = self.teacher.generate_example(self.student, X_train.cuda(), labels, self.opt.batch_size)

                # i = self.teacher.select_example(self.student, X_train.cuda(), y_train.cuda(), self.opt.batch_size)
                # i = torch.randint(0, nb_batch, size=(1,)).item()
                # i_min = i * self.opt.batch_size
                # i_max = (i + 1) * self.opt.batch_size

                # x_t = X_train[i_min:i_max].cuda()
                # y_t = y_train[i_min:i_max].cuda()

                # self.student.update(x_t, y_t)

                self.student.update(torch.cuda.FloatTensor(data), labels)

                baseline.update(best_data, best_label)

                data = data.cpu().detach().numpy()
                if t == 1:
                    generated_samples = data[np.newaxis, :]
                else:
                    generated_samples = np.concatenate((generated_samples, data[np.newaxis, :]), axis=0)
                print("iter", len(generated_samples))

            self.student.eval()
            test = self.student(X_test.cuda()).cpu()

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)
            res_student.append(acc)

            baseline.eval()
            test = baseline(X_test.cuda()).cpu()

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices0
                nb_correct = torch.where(tmp == y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc_base = nb_correct / X_test.size(0)
            res_baseline.append(acc_base)

            print("acc", acc, "acc baseline", acc_base)

            sys.stdout.write("\r" + str(t) + "/" + str(self.opt.n_iter) + ", idx=" + str(i) + " " * 100)
            sys.stdout.flush()

        if self.visualize == True:
            fig, axs = plt.subplots(3, 2, figsize=(10, 16))

            axs[0, 0].plot(res_example, c='b', label="linear classifier")
            axs[0, 0].plot(res_baseline, c='g', label="%s & baseline" % self.opt.teaching_mode)
            axs[0, 0].plot(res_student, c='r', label="%s & linear classifier" % self.opt.teaching_mode)
            # axs[0, 0].set_title(str(self.opt.data_mode) + "Model (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
            axs[0, 0].set_title("Accuracy Comparison")
            axs[0, 0].set_xlabel("Iteration")
            axs[0, 0].set_ylabel("Accuracy")
            axs[0, 0].legend()
            # plt.show()
            # plt.savefig("{}.jpg".format(res_example[0]))

            # plot
            x = 0.5 + np.arange(self.opt.n_iter)
            axs[0, 1].stem(x, score_statistics)
            # plt.show()

            axs[1, 0].scatter(X[:, 0], X[:, 1], c=y)
            # axs[1, 0].scatter(generated_samples[:, 0], generated_samples[:, 1], color='k')
            axs[1, 0].set_title('Updated Data')
            # axs[1, 0].show()

            a_gt, b_gt = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            a, b = plot_classifier(self.student, X.max(axis=0), X.min(axis=0))
            axs[1, 1].plot(a_gt, a_gt, '-g', label='y=wx+b')
            axs[1, 1].plot(a, b, '-r', label='y=wx+b')
            axs[1, 1].scatter(X[:, 0], X[:, 1], c=y)
            axs[1, 1].set_title('Student Classifier')
            # plt.show()

            a_gt, b_gt = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            a, b = plot_classifier(example, X.max(axis=0), X.min(axis=0))
            axs[2, 0].plot(a_gt, a_gt, '-g', label='y=wx+b')
            axs[2, 0].plot(a, b, '-r', label='y=wx+b')
            axs[2, 0].scatter(X[:, 0], X[:, 1], c=y)
            axs[2, 0].set_title('SGD Classifier')
            # plt.show()

            a_gt, b_gt = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            a, b = plot_classifier(baseline, X.max(axis=0), X.min(axis=0))
            axs[2, 1].plot(a_gt, a_gt, '-g', label='y=wx+b')
            axs[2, 1].plot(a, b, '-r', label='y=wx+b')
            axs[2, 1].scatter(X[:, 0], X[:, 1], c=y)
            axs[2, 1].set_title('Baseline IMT Classifier')
            plt.show()

    def main(self):
        X_test = next(iter(self.test_loader))[0].numpy()
        y_test = next(iter(self.test_loader))[1].numpy()

        accuracies = []
        for epoch in tqdm(range(100)):
            print('\nEpoch: %d' % epoch)
            self.teacher.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.teacher.update(inputs, targets)

                outputs = self.teacher(inputs.cuda())
                predicted = torch.max(outputs, dim=1).indices

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # print('Acc: %.3f%% (%d/%d)'% (100.*correct/total, correct, total))

            self.teacher.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.teacher(inputs.cuda())
                    predicted = torch.max(outputs, dim=1).indices

                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            # Save checkpoint.
            acc = 100.*correct/total
            accuracies.append(acc)

            print("Epoch", epoch, "Acc", acc)
        plt.plot(accuracies, c="b", label="Teacher (CNN)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        '''
            test = self.teacher(X_test.cuda()).cpu()
            tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
            nb_correct = torch.where(tmp.view(-1) == y_test, torch.ones(1), torch.zeros(1)).sum().item()
            accuracies.append(nb_correct / X_test.size(0))

        plt.plot(accuracies, c="b", label="Teacher (CNN)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        '''

    def process_batch(self, inputs):
        #for key, ipt in inputs.items():
        #    if key != 'case_name':
        #        inputs[key] = ipt.to(self.device)

        outputs = {}

        # features = self.models["encoder"](inputs["image"])
        # preds = self.models["decoder"](features)
        preds = self.models["unet"](inputs["image"].to(self.device))

        outputs["pred"] = preds
        outputs["pred_idx"] = torch.argmax(preds, dim=1, keepdim=True)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.compute_accuracy(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)

            del inputs, outputs, losses

        self.set_train()

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        pred = outputs['pred']
        target = inputs['label']

        #to_optimise = self.criterion(output=pred,
        #                             target=target)

        to_optimise = self.criterion(pred, target.type(torch.LongTensor).cuda())

        total_loss += to_optimise
        losses["loss"] = total_loss
        return losses

    def compute_accuracy(self, inputs, outputs, losses):
        with torch.no_grad():
            # acc_dice = 0
            # acc_iou = 0
            for cls in range(1, self.opt.n_classes):
                fluid = self.cls_to_fluid[cls]
                losses["accuracy/dice_{}".format(fluid)] = self.eval_metric.compute_coef(outputs["pred_idx"].cpu().data,
                                                                                         inputs['label'].cpu().data,
                                                                                         mode='dice',
                                                                                         cls=cls)
                # acc_dice += losses["accuracy/dice_{}".format(fluid)]

                losses["accuracy/iou_{}".format(fluid)] = self.eval_metric.compute_coef(outputs["pred_idx"].cpu().data,
                                                                                        inputs['label'].cpu().data,
                                                                                        mode='iou',
                                                                                        cls=cls)
                # acc_iou += losses["accuracy/iou_{}".format(fluid)]

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size, inputs["image"].shape[0])):  # write a maxmimum of four images
            writer.add_image("inputs/{}".format(j), normalize_image(inputs["image"][j].data), self.step)
            writer.add_image("labels/{}".format(j), normalize_target(inputs["label"][j].unsqueeze(0).data), self.step)
            writer.add_image("predictions/{}".format(j), normalize_target(outputs["pred_idx"][j].data), self.step)
            # writer.add_image("positive_region/{}".format(j), outputs["mask"][j].data, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        if self.epoch >= self.opt.start_saving_optimizer:
            save_path = os.path.join(save_folder, "{}.pth".format("adam"))
            torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

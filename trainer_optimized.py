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
import teachers.omniscient_teacher_optimizer as omniscient
import teachers.utils as utils
import matplotlib.pyplot as plt
import data.dataset_loader as data_loader

from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split

import subprocess
import glob

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
        self.opt.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.visualize = True

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.get_teacher_student()

    def get_teacher_student(self):
        if self.opt.data_mode == "cifar10":
            self.teacher = omniscient.OmniscientConvTeacher(self.opt.eta)
            self.student = omniscient.OmniscientConvStudent(self.opt.eta)
        else: # mnist / gaussian / moon
            self.teacher = omniscient.OmniscientLinearTeacher(self.opt.dim)
            self.student = omniscient.OmniscientLinearStudent(self.opt.dim)
            self.baseline = omniscient.OmniscientLinearStudent(self.opt.dim)

            # self.teacher = omniscient.TeacherClassifier(self.opt.dim)
            # self.student = omniscient.StudentClassifier(self.opt.dim)

        self.student.load_state_dict(self.teacher.state_dict())
        self.baseline.load_state_dict(self.teacher.state_dict())

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

    def main(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        # self.set_train()

        torch.manual_seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.cuda.manual_seed(self.opt.seed)
        # torch.cuda.set_device(args.gpu)
        # cudnn.benchmark = True
        # cudnn.enabled=True

        if self.opt.data_mode == "cifar10":
            print("Loading CIFAR10 data ...")

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616)),
            ])

            train_dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.CIFAR10(root=CONF.PATH.DATA, train=False, download=True, transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

            X = next(iter(train_loader))[0].numpy()
            y = next(iter(train_loader))[1].numpy()
            (N, W, H, C) = train_dataset.data.shape
            dim = W*H*C
            sgd_example = utils.BaseConv(self.opt.eta)
            # baseline = utils.BaseConv(self.opt.eta)

        elif self.opt.data_mode == "mnist":
            print("Loading MNIST data ...")

            # MNIST normalizing
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            train_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(root=CONF.PATH.DATA, train=False, download=True, transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
            X = next(iter(train_loader))[0].numpy()
            Y = next(iter(train_loader))[1].numpy()
            (N, W, H) = train_dataset.data.shape
            dim = W*H
            sgd_example = utils.BaseLinear(self.opt.dim)
            # baseline = utils.BaseLinear(self.opt.dim)
            # X_train = np.asarray(self.train_dataset.data.reshape((N, dim)))
            X = X.reshape((N, dim))
            # Y_train = np.asarray(self.train_dataset.targets)
            # Y_train = np.asarray(self.train_dataset.targets)

            # create new data set with class 1 as 0 and class 2 as 1
            f = (Y == self.opt.class_1) | (Y == self.opt.class_2)
            X = X[f]
            y = Y[f]
            y = np.where(y == self.opt.class_1, 0, 1)

        elif self.opt.data_mode == "gaussian":
            print("Generating Gaussian data ...")

            dim__diff = 7
            nb_data_per_class = 1000

            X, Y = self.init_data(self.opt.dim, nb_data_per_class)

            sgd_example = utils.BaseLinear(self.opt.dim)
            # baseline = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=y)
                plt.title('Gaussian Data')
                #plt.show()
                plt.close()

        elif self.opt.data_mode == "moon":
            print("Generating moon data ...")

            np.random.seed(0)
            noise_val = 0.2

            X, Y = make_moons(self.opt.nb_train+self.opt.nb_test, noise=noise_val)

            sgd_example = utils.BaseLinear(self.opt.dim)
            # baseline = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=y)
                plt.title('Moon Data')
                #plt.show()
                plt.close()

        elif self.opt.data_mode == "linearly_seperable":
            print("Generating linearly seperable data ...")

            X, Y = make_classification(
                n_samples=self.opt.nb_train+self.opt.nb_test, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
            )
            rng = np.random.RandomState(2)
            X += 2 * rng.uniform(size=X.shape)

            sgd_example = utils.BaseLinear(self.opt.dim)
            # baseline = utils.BaseLinear(self.opt.dim)

            if self.visualize:
                fig = plt.figure(figsize=(8, 5))
                a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
                plt.plot(a, b, '-r', label='y=wx+b')
                plt.scatter(X[:, 0], X[:, 1], c=y)
                plt.title('Linearly Seperable Data')
                # plt.show()
                plt.close()
        else:
            print("Unrecognized data!")
            sys.exit()

        sgd_example.load_state_dict(self.teacher.state_dict())
        # baseline.load_state_dict(self.teacher.state_dict())

        # Shuffle datasets
        randomize = np.arange(X.shape[0])
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]

        nb_batch = int(self.opt.nb_train / self.opt.batch_size)

        if self.opt.data_mode == "cifar10":
            X_train = torch.tensor(X[:self.opt.nb_train])
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.long)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test])
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.long)
        elif self.opt.data_mode == "mnist":
            X_train = torch.tensor(X[:self.opt.nb_train], dtype=torch.float)
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.float)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)

            proj_matrix = torch.empty(X.shape[1], self.opt.dim).normal_(mean=0, std=0.1)
            X_train = X_train @ proj_matrix
            X_test = X_test @ proj_matrix
        else:
            X_train = torch.tensor(X[:self.opt.nb_train], dtype=torch.float)
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.float)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)

        # ---------------------
        #  Train Teacher
        # ---------------------

        accuracies = []
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.teacher.optim, milestones=[80, 160], gamma=0.1)
        for n in tqdm(range(self.opt.n_teacher_runs)):
            if n != 0:
                for i in range(nb_batch):
                    i_min = i * self.opt.batch_size
                    i_max = (i + 1) * self.opt.batch_size
                    self.teacher.update(X_train[i_min:i_max].cuda(), Y_train[i_min:i_max].cuda())
            self.teacher.eval()
            test = self.teacher(X_test.cuda()).cpu()

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)
            accuracies.append(acc)
            print("Accuracy:", acc)

            if acc > 0.6 and n == 0:
                sys.exit()

        w_star = self.teacher.lin.weight

        if self.visualize == True:
            fig = plt.figure()
            plt.plot(accuracies, c="b", label="Teacher (CNN)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.close()

            fig = plt.figure(figsize=(8, 5))
            a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            plt.plot(a, b, '-r', label='y=wx+b')
            plt.scatter(X[:, 0], X[:, 1], c=y)
            plt.title('Initial Classifer Weight')
            plt.close()

        teacher_acc = accuracies[-1]

        # ---------------------
        #  Train Student
        # ---------------------

        res_i = [1] * 200

        res_student = []
        a_student = []
        b_student = []
        generated_samples = np.zeros(2)
        w_diff_student = []
        for t in tqdm(range(self.opt.n_iter)):
            if t != 0:
                # labels = torch.randint(0, 1, (self.opt.batch_size,), dtype=torch.float).cuda()
                new_data, new_labels = self.teacher.generate_example(self.student, X_train.cuda(), Y_train.cuda(), self.opt.batch_size, self.opt.lr_factor, self.opt.gd_n, res_i[t-1], self.opt.optim)

                generated_data = new_data.detach().clone().cpu().numpy()
                generated_label = new_labels.detach().clone().cpu().numpy()
                if t == 1:
                    generated_samples = generated_data # [np.newaxis, :]
                    generated_labels = generated_label # [np.newaxis, :]
                else:
                    generated_samples = np.concatenate((generated_samples, generated_data), axis=0)
                    generated_labels = np.concatenate((generated_labels, generated_label), axis=0)

                self.student.update(torch.cuda.FloatTensor(new_data), new_labels)
            self.student.eval()
            test = self.student(X_test.cuda()).cpu()

            a, b = plot_classifier(self.student, X.max(axis=0), X.min(axis=0))
            a_student.append(a)
            b_student.append(b)

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)
            res_student.append(acc)

            diff = torch.linalg.norm(w_star - self.student.lin.weight, ord=2) ** 2
            w_diff_student.append(diff.detach().clone().cpu())

            print("iter", t, "acc student", acc)

            sys.stdout.write("\r" + str(t) + "/" + str(self.opt.n_iter) + ", idx=" + str(i) + " " * 100)
            sys.stdout.flush()

        # ---------------------
        #  Train SGD
        # ---------------------

        res_sgd = []
        res_i = []
        w_diff_sgd = []
        for n in tqdm(range(self.opt.n_iter)):
            if n != 0:
                i = torch.randint(0, nb_batch, size=(1,)).item()
                i_min = i * self.opt.batch_size
                i_max = (i + 1) * self.opt.batch_size

                data = X_train[i_min:i_max].cuda()
                label = Y_train[i_min:i_max].cuda()

                sgd_example.update(data, label)

                res_i.append(i)

            sgd_example.eval()
            test = sgd_example(X_test.cuda()).cpu()

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc = nb_correct / X_test.size(0)
            res_sgd.append(acc)

            diff = torch.linalg.norm(w_star - sgd_example.lin.weight, ord=2) ** 2
            w_diff_sgd.append(diff.detach().clone().cpu())

        print("IMT baseline trained\n")

        # ---------------------
        #  Train IMT
        # ---------------------

        res_baseline = []
        a_baseline = []
        b_baseline = []
        w_diff_baseline = []
        for t in tqdm(range(self.opt.n_iter)):
            if t != 0:
                # labels = torch.randint(0, 1, (self.opt.batch_size,), dtype=torch.float).cuda()
                best_data, best_label = self.teacher.select_example(self.baseline, X_train.cuda(), Y_train.cuda(), self.opt.batch_size)

                selected_data = best_data.detach().clone().cpu().numpy()
                selected_label = best_label.detach().clone().cpu().numpy()
                if t == 1:
                    selected_samples = selected_data # [np.newaxis, :]
                    selected_labels = selected_label # [np.newaxis, :]
                else:
                    selected_samples = np.concatenate((selected_samples, selected_data), axis=0)
                    selected_labels = np.concatenate((selected_labels, selected_label), axis=0)

                self.baseline.update(best_data, best_label)

            self.baseline.eval()
            test = self.baseline(X_test.cuda()).cpu()

            a, b = plot_classifier(self.baseline, X.max(axis=0), X.min(axis=0))
            a_baseline.append(a)
            b_baseline.append(b)

            if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            elif self.opt.data_mode == "cifar10":
                tmp = torch.max(test, dim=1).indices
                nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
            else:
                sys.exit()
            acc_base = nb_correct / X_test.size(0)
            res_baseline.append(acc_base)

            diff = torch.linalg.norm(w_star - self.baseline.lin.weight, ord=2) ** 2
            w_diff_baseline.append(diff.detach().clone().cpu())

            print("iter", t, "acc baseline", acc_base)

            sys.stdout.write("\r" + str(t) + "/" + str(self.opt.n_iter) + ", idx=" + str(i) + " " * 100)
            sys.stdout.flush()

        if self.visualize == True:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(12, 6)
            ax1.plot(res_sgd, c='g', label="SGD %s" % self.opt.data_mode)
            ax1.plot(res_baseline, c='b', label="IMT %s" % self.opt.data_mode)
            ax1.plot(res_student, c='r', label="Student %s" % self.opt.data_mode)
            # ax1.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
            ax1.set_title("Test accuracy " + str(self.opt.data_mode) + " (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Accuracy")
            ax1.legend(loc="lower right")

            ax2.plot(w_diff_sgd, 'go', label="SGD %s" % self.opt.data_mode)
            ax2.plot(w_diff_baseline, 'bo', label="IMT %s" % self.opt.data_mode, alpha=0.5)
            ax2.plot(w_diff_student, 'ro', label="Student %s" % self.opt.data_mode, alpha=0.5)
            ax2.legend(loc="lower left")
            ax2.set_title("w diff " + str(self.opt.data_mode) + " (class : " + str(self.opt.class_1) + ", " + str(self.opt.class_2) + ")")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Distance between $w^t$ and $w^*$")
            #ax2.set_aspect('equal')

            # plt.savefig('results_mnist_final.jpg')
            # plt.close()
            plt.show()

        if self.visualize == False:
            a, b = plot_classifier(self.teacher, X.max(axis=0), X.min(axis=0))
            for i in range(len(res_student)):
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                fig.set_size_inches(20, 5.8)
                #ax1.plot(a, b, '-k', label='Teacher Classifier')
                ax1.plot(a_student[i], b_student[i], '-r', label='Optimizer Classifier')
                ax1.scatter(X[:, 0], X[:, 1], c=y)
                ax1.scatter(generated_samples[:i+1, 0], generated_samples[:i+1, 1], c=generated_labels[:i+1], marker='x')
                ax1.legend(loc="upper right")
                ax1.set_title("Data Generation (Optimizer)")
                ax1.set_xlim([X.min()-0.5, X.max()+0.5])
                ax1.set_ylim([X.min()-0.5, X.max()+0.5])

                #ax2.plot(a, b, '-k', label='Teacher Classifier')
                ax2.plot(a_baseline[i], b_baseline[i], '-g', label='IMT Classifier')
                ax2.scatter(X[:, 0], X[:, 1], c=y)
                ax2.scatter(selected_samples[:i+1, 0], selected_samples[:i+1, 1], c=selected_labels[:i+1, 0], marker='x')
                ax2.legend(loc="upper right")
                ax2.set_title("Data Selection (IMT)")
                ax2.set_xlim([X.min()-0.5, X.max()+0.5])
                ax2.set_xlim([X.min()-0.5, X.max()+0.5])

                ax3.plot(res_sgd, 'bo', label="linear classifier", alpha=0.5)
                ax3.plot(res_baseline[:i+1], 'go', label="%s & baseline" % self.opt.teaching_mode, alpha=0.5)
                ax3.plot(res_student[:i+1], 'ro', label="%s & linear classifier" % self.opt.teaching_mode, alpha=0.5)
                ax3.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
                ax3.legend(loc="upper right")
                ax3.set_title("Test Set Accuracy")
                #ax3.set_aspect('equal')

                plt.savefig(CONF.PATH.OUTPUT + "/file%02d.png" % i)

                plt.close()

            os.chdir(CONF.PATH.OUTPUT)
            subprocess.call([
                'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
                'video_name.mp4'
            ])
            for file_name in glob.glob("*.png"):
                os.remove(file_name)

    def main1(self):
        X_test = next(iter(self.test_loader))[0].numpy()
        Y_test = next(iter(self.test_loader))[1].numpy()

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
        #plt.show()
        '''
            test = self.teacher(X_test.cuda()).cpu()
            tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
            nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
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
        models_dir = os.path.join(self.opt.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.opt.log_path, "models", "weights_{}".format(self.epoch))
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

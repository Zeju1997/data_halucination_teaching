# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import sys

import torch

import torch.nn as nn
import json
import os
import csv

from tqdm import tqdm

from train_utils import *

import teachers.omniscient_teacher_optimizer as omniscient
import teachers.utils as utils
import matplotlib.pyplot as plt

from utils.visualize import make_results_video, make_results_video_2d, make_results_img, make_results_img_2d, make_results
from utils.data import init_data, load_experiment_result, plot_graphs_optimized
from utils.network import initialize_weights

from experiments import SGDTrainer, IMTTrainer, WSTARTrainer

import torch.nn.functional as F

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

        self.opt.model_name = "omniscient_optimized_" + self.opt.data_mode

        self.opt.log_path = os.path.join(CONF.PATH.LOG, self.opt.model_name)
        if not os.path.exists(self.opt.log_path):
            os.makedirs(self.opt.log_path)

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
            self.teacher.apply(initialize_weights)
            self.student = omniscient.OmniscientLinearStudent(self.opt.dim)
            self.baseline = omniscient.OmniscientLinearStudent(self.opt.dim)
            self.student_label = omniscient.OmniscientLinearStudent(self.opt.dim)
            self.label = omniscient.OmniscientLinearStudent(self.opt.dim)
            self.imt_label = omniscient.OmniscientLinearStudent(self.opt.dim)
            torch.save(self.teacher.state_dict(), 'teacher_w0.pth')

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

        if self.opt.init_data:
            init_data(self.opt)

        X = torch.load('X.pt')
        Y = torch.load('Y.pt')

        # Shuffle datasets
        # randomize = np.arange(X.shape[0])
        # np.random.shuffle(randomize)
        # X = X[randomize]
        # Y = Y[randomize]

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

            X_train = X_train.reshape((self.opt.nb_train, self.opt.img_size**2))
            X_test = X_test.reshape((self.opt.nb_test, self.opt.img_size**2))

            proj_matrix = torch.empty(self.opt.img_size**2, self.opt.dim).normal_(mean=0, std=0.1)
            # proj_matrix = torch.load('proj_matrix.pt')
            X_train = torch.matmul(X_train, proj_matrix)
            X_test = torch.matmul(X_test, proj_matrix)

        else:
            X_train = torch.tensor(X[:self.opt.nb_train], dtype=torch.float)
            Y_train = torch.tensor(Y[:self.opt.nb_train], dtype=torch.float)
            X_test = torch.tensor(X[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)
            Y_test = torch.tensor(Y[self.opt.nb_train:self.opt.nb_train + self.opt.nb_test], dtype=torch.float)

        # ---------------------
        #  Train Teacher
        # ---------------------

        self.opt.experiment = "WSTAR"
        if self.opt.train_wstar == False:
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
            if os.path.exists(logname):
                os.remove(logname)
            if not os.path.exists(logname):
                with open(logname, 'w') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow(['epoch', 'test acc'])

            nb_batch = int(self.opt.nb_train / self.opt.batch_size)

            accuracies = []
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.teacher.optim, milestones=[25], gamma=0.1)
            for n in tqdm(range(self.opt.n_teacher_runs)):
                if n != 0:
                    for i in range(nb_batch):
                        i_min = i * self.opt.batch_size
                        i_max = (i + 1) * self.opt.batch_size
                        x = X_train[i_min:i_max].cuda()
                        y = Y_train[i_min:i_max].cuda()

                        y = F.one_hot(y.long(), num_classes=2).type(torch.cuda.FloatTensor)

                        self.teacher.update(x, y)

                self.teacher.eval()
                test = self.teacher(X_test.cuda()).cpu()

                if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable" or self.opt.data_mode == "covid":
                    # tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                elif self.opt.data_mode == "cifar10":
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                else:
                    sys.exit()
                acc = nb_correct / X_test.size(0)
                accuracies.append(acc)
                print("Accuracy:", acc)
                self.scheduler.step()

                if acc > 0.6 and n == 0:
                    sys.exit()

                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([n, acc])

            torch.save(self.teacher.state_dict(), 'teacher_wstar.pth')

        self.teacher.load_state_dict(torch.load('teacher_wstar.pth'))
        w_star = self.teacher.lin.weight
        w_star = w_star / torch.norm(w_star)

        # ---------------------
        #  Train SGD
        # ---------------------

        self.opt.experiment = "SGD"
        if self.opt.train_sgd == False:
            sgd_example = utils.BaseLinear(self.opt.dim)
            sgd_example.load_state_dict(torch.load('teacher_w0.pth'))

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
                    sample, label = self.data_sampler(X_train, Y_train, i)

                    label = F.one_hot(label.long(), num_classes=2).type(torch.cuda.FloatTensor)

                    sgd_example.update(sample, label)

                sgd_example.eval()
                test = sgd_example(X_test.cuda()).cpu()

                # a, b = plot_classifier(model, X_train.max(axis=0), X_train.min(axis=0))
                # a_example.append(a)
                # b_example.append(b)

                if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "covid":
                    tmp = torch.max(test, dim=1).indices
                    # tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                    nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                elif self.opt.data_mode == "cifar10":
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                else:
                    sys.exit()

                acc = nb_correct / X_test.size(0)
                res_sgd.append(acc)

                w = sgd_example.lin.weight
                w = w / torch.norm(w)
                diff = torch.linalg.norm(w_star - w, ord=2) ** 2
                w_diff_sgd.append(diff.detach().clone().cpu())

                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([idx, acc, diff.item()])

        res_sgd, w_diff_sgd = load_experiment_result(self.opt)

        # ---------------------
        #  Train IMT Baseline
        # ---------------------

        # self.opt.experiment = "IMT_Baseline_random_label"
        self.opt.experiment = "IMT_Baseline"
        if self.opt.train_baseline == False:
            self.baseline.load_state_dict(torch.load('teacher_w0.pth'))

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
                        i = self.teacher.select_example(self.baseline, self.opt, X_train.cuda(), Y_train.cuda())
                        # i = torch.randint(0, 1000, size=(1,)).item()
                    else:
                        i = self.teacher.select_example_random_label(self.baseline, X_train.cuda(), Y_train.cuda(), self.opt.batch_size)

                    best_sample, best_label = self.data_sampler(X_train, Y_train, i)
                    best_label = F.one_hot(best_label.long(), num_classes=2).type(torch.cuda.FloatTensor)

                    self.baseline.update(best_sample, best_label)

                self.baseline.eval()
                test = self.baseline(X_test.cuda()).cpu()

                # a, b = plot_classifier(model, X.max(axis=0), X.min(axis=0))
                # a_baseline.append(a)
                # b_baseline.append(b)

                if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable" or self.opt.data_mode == "covid":
                    tmp = torch.max(test, dim=1).indices
                    # tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                    nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                elif self.opt.data_mode == "cifar10":
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                else:
                    sys.exit()
                acc_base = nb_correct / X_test.size(0)
                res_baseline.append(acc_base)

                w = self.baseline.lin.weight
                w = w / torch.norm(w)
                diff = torch.linalg.norm(w_star - w, ord=2) ** 2
                w_diff_baseline.append(diff.detach().clone().cpu())

                print("iter", t, "acc IMT", acc_base)

                # sys.stdout.write("\r" + str(t) + "/" + str(self.opt.n_iter) + ", idx=" + str(i) + " " * 100)
                # sys.stdout.flush()

                print("Base line trained\n")

                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([t, acc_base, diff.item()])

        res_baseline, w_diff_baseline = load_experiment_result(self.opt)

        # ---------------------
        #  Train Student
        # ---------------------

        self.opt.experiment = "Student"
        if self.opt.train_student == False:
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
            if not os.path.exists(logname):
                with open(logname, 'w') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow(['iter', 'test acc', 'w diff'])
            res_student = []
            a_student = []
            b_student = []
            generated_samples = np.zeros(2)
            w_diff_student = []
            self.student.load_state_dict(torch.load('teacher_w0.pth'))
            for t in tqdm(range(self.opt.n_iter)):
                if t != 0:
                    # labels = torch.randint(0, 1, (self.opt.batch_size,), dtype=torch.float).cuda()
                    new_data, new_labels = self.teacher.generate_example(self.opt, self.student, X_train.cuda(), Y_train.cuda(), optimize_label=False)

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
                    # tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                elif self.opt.data_mode == "cifar10":
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                else:
                    sys.exit()
                acc = nb_correct / X_test.size(0)
                res_student.append(acc)

                w = self.student.lin.weight
                w = w / torch.norm(w)
                diff = torch.linalg.norm(w_star - w, ord=2) ** 2
                w_diff_student.append(diff.detach().clone().cpu())

                print("iter", t, "acc student", acc)

                # sys.stdout.write("\r" + str(t) + "/" + str(self.opt.n_iter) + ", idx=" + str(i) + " " * 100)
                # sys.stdout.flush()

                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([t, acc, diff.item()])

        res_student, w_diff_student = load_experiment_result(self.opt)

        # ---------------------
        #  Train Student with Label
        # ---------------------
        self.opt.experiment = "Student_with_Label"
        if self.opt.train_student == False:
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
            if not os.path.exists(logname):
                with open(logname, 'w') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow(['iter', 'test acc', 'w diff'])
            res_student_label = []
            a_student_label = []
            b_student_label = []
            generated_samples = np.zeros(2)
            w_diff_student_label = []
            self.student_label.load_state_dict(torch.load('teacher_w0.pth'))
            for t in tqdm(range(self.opt.n_iter)):
                if t != 0:
                    # labels = torch.randint(0, 1, (self.opt.batch_size,), dtype=torch.float).cuda()
                    new_data, new_labels = self.teacher.generate_example(self.opt, self.student_label, X_train.cuda(), Y_train.cuda(), optimize_label=True)

                    self.student_label.update(torch.cuda.FloatTensor(new_data), new_labels)
                self.student_label.eval()
                test = self.student_label(X_test.cuda()).cpu()

                a, b = plot_classifier(self.student_label, X.max(axis=0), X.min(axis=0))
                a_student_label.append(a)
                b_student_label.append(b)

                if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                    # tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                elif self.opt.data_mode == "cifar10":
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                else:
                    sys.exit()
                acc = nb_correct / X_test.size(0)
                res_student_label.append(acc)

                w = self.student_label.lin.weight
                w = w / torch.norm(w)
                diff = torch.linalg.norm(w_star - w, ord=2) ** 2
                w_diff_student_label.append(diff.detach().clone().cpu())

                print("iter", t, "acc student with label", acc)

                # sys.stdout.write("\r" + str(t) + "/" + str(self.opt.n_iter) + ", idx=" + str(i) + " " * 100)
                # sys.stdout.flush()

                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([t, acc, diff.item()])

        res_student_label, w_diff_student_label = load_experiment_result(self.opt)

        # ---------------------
        #  Train Label
        # ---------------------
        self.opt.experiment = "Label"
        if self.opt.train_student == False:
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
            if not os.path.exists(logname):
                with open(logname, 'w') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow(['iter', 'test acc', 'w diff'])
            res_label = []
            a_label = []
            b_label = []
            generated_samples = np.zeros(2)
            w_diff_label = []
            self.label.load_state_dict(torch.load('teacher_w0.pth'))
            for t in tqdm(range(self.opt.n_iter)):
                if t != 0:
                    # labels = torch.randint(0, 1, (self.opt.batch_size,), dtype=torch.float).cuda()
                    new_data, new_labels = self.teacher.generate_label(self.opt, self.label, X_train.cuda(), Y_train.cuda())

                    self.label.update(torch.cuda.FloatTensor(new_data), new_labels)
                self.label.eval()
                test = self.label(X_test.cuda()).cpu()

                a, b = plot_classifier(self.label, X.max(axis=0), X.min(axis=0))
                a_label.append(a)
                b_label.append(b)

                if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                    # tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                elif self.opt.data_mode == "cifar10":
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                else:
                    sys.exit()
                acc = nb_correct / X_test.size(0)
                res_label.append(acc)

                w = self.label.lin.weight
                w = w / torch.norm(w)
                diff = torch.linalg.norm(w_star - w, ord=2) ** 2
                w_diff_label.append(diff.detach().clone().cpu())

                print("iter", t, "acc label", acc)

                # sys.stdout.write("\r" + str(t) + "/" + str(self.opt.n_iter) + ", idx=" + str(i) + " " * 100)
                # sys.stdout.flush()

                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([t, acc, diff.item()])

        res_label, w_diff_label = load_experiment_result(self.opt)

        # ---------------------
        #  Train IMT + Label
        # ---------------------
        self.opt.experiment = "IMT_Label"
        if self.opt.train_baseline == False:
            print("Start training {} ...".format(self.opt.experiment))
            logname = os.path.join(self.opt.log_path, 'results' + '_' + self.opt.experiment + '_' + str(self.opt.seed) + '.csv')
            if not os.path.exists(logname):
                with open(logname, 'w') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow(['iter', 'test acc', 'w diff'])
            res_imt_label = []
            a_imt_label = []
            b_imt_label = []
            generated_samples = np.zeros(2)
            w_diff_imt_label = []
            self.imt_label.load_state_dict(torch.load('teacher_w0.pth'))
            for t in tqdm(range(self.opt.n_iter)):
                if t != 0:
                    # labels = torch.randint(0, 1, (self.opt.batch_size,), dtype=torch.float).cuda()
                    new_data, new_labels = self.teacher.select_example(self.imt_label, self.opt, X_train.cuda(), Y_train.cuda(), optimize_label=True)

                    self.imt_label.update(torch.cuda.FloatTensor(new_data), new_labels)
                self.imt_label.eval()
                test = self.imt_label(X_test.cuda()).cpu()

                a, b = plot_classifier(self.imt_label, X.max(axis=0), X.min(axis=0))
                a_imt_label.append(a)
                b_imt_label.append(b)

                if self.opt.data_mode == "mnist" or self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon" or self.opt.data_mode == "linearly_seperable":
                    # tmp = torch.where(test > 0.5, torch.ones(1), torch.zeros(1))
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp.view(-1) == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                elif self.opt.data_mode == "cifar10":
                    tmp = torch.max(test, dim=1).indices
                    nb_correct = torch.where(tmp == Y_test, torch.ones(1), torch.zeros(1)).sum().item()
                else:
                    sys.exit()
                acc = nb_correct / X_test.size(0)
                res_imt_label.append(acc)

                w = self.imt_label.lin.weight
                w = w / torch.norm(w)
                diff = torch.linalg.norm(w_star - w, ord=2) ** 2
                w_diff_imt_label.append(diff.detach().clone().cpu())

                print("iter", t, "acc student", acc)

                # sys.stdout.write("\r" + str(t) + "/" + str(self.opt.n_iter) + ", idx=" + str(i) + " " * 100)
                # sys.stdout.flush()

                with open(logname, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([t, acc, diff.item()])

        res_imt_label, w_diff_imt_label = load_experiment_result(self.opt)

        if self.opt.data_mode == "gaussian" or self.opt.data_mode == "moon":
            make_results(self.opt, res_sgd, res_baseline, res_student, res_student_label, res_label, res_imt_label, w_diff_sgd, w_diff_baseline, w_diff_student, w_diff_student_label, w_diff_label, w_diff_imt_label, 0, self.opt.seed)
            # make_results_img_2d(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, 0, self.opt.seed)
            # make_results_video_2d(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, 0, self.opt.seed res_student_label, w_diff_student_label)
        else:
            make_results(self.opt, res_sgd, res_baseline, res_student, res_student_label, res_label, res_imt_label, w_diff_sgd, w_diff_baseline, w_diff_student, w_diff_student_label, w_diff_label, w_diff_imt_label, 0, self.opt.seed)
            # make_results_img(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, 0, self.opt.seed, proj_matrix)
            # make_results_video(self.opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, 0, self.opt.seed, proj_matrix)


    def data_sampler(self, X, Y, i):
        i_min = i * self.opt.batch_size
        i_max = (i + 1) * self.opt.batch_size

        x = X[i_min:i_max].cuda()
        y = Y[i_min:i_max].cuda()

        return x, y

